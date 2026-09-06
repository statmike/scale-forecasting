"""Resolve a run into its execution DAG: one job per model family, plus the ensemble node.

A run's ``cfg.models`` spans up to four families — ``statistical`` / ``ml`` / ``deep_learning`` /
``native``. Each family that is present becomes an **independent job** under one shared ``run_id``:
the three Python families each run on their own resolved runtime (Spark *xor* Ray, chosen per
family via `config.RunConfig.resolve_family_compute`), while ``native`` runs in BigQuery. When
ensembling is enabled a downstream ensemble node depends on all of them. This module owns the pure,
offline plan; `main.run` executes it and `launch_plan.plan_run` / ``stage_run`` emit its commands.

The families run in parallel, so a run's wall-clock is the slowest family's job, not the sum —
adding a BigQuery-native model to a Spark run costs ``max(spark, bq)``, not ``spark + bq``. The DAG
is a pure function of the config (no clocks, no GCP), so the same config always plans the same DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import ConfigError, get_logger
from .models import get_model, list_models
from .registry.ids import make_run_id

if TYPE_CHECKING:
    from .config import ResolvedFamilyCompute, RunConfig

_log = get_logger(__name__)

# The order families are listed in the DAG: the Python families first, ``native`` last, so logs and
# manifests read consistently. Purely cosmetic — the jobs execute in parallel, not in this order.
_FAMILY_ORDER: tuple[str, ...] = ("statistical", "ml", "deep_learning", "native")


@dataclass(frozen=True)
class FamilyJob:
    """One family's job in the run DAG: which models it runs, on what resolved compute.

    ``compute`` is the fully-resolved per-family compute (`config.RunConfig.resolve_family_compute`)
    for a Python family, or ``None`` for ``native`` — native models always run in BigQuery and take
    no runtime choice. ``models`` are the config's models of this family, in config order.
    """

    family: str
    models: tuple[str, ...]
    compute: ResolvedFamilyCompute | None

    @property
    def runtime(self) -> str:
        """Where this job runs: ``"bigquery"`` for native, else the resolved family runtime."""
        return "bigquery" if self.compute is None else self.compute.runtime


@dataclass(frozen=True)
class RunDag:
    """A run resolved to its DAG: the shared ``run_id`` and one job per present family.

    ``jobs`` holds one `FamilyJob` per family present in the config, ordered by `_FAMILY_ORDER`.
    ``ensemble_enabled`` records whether the downstream ensemble node runs after the base jobs.
    Pure product of the config (`plan_dag`) — no GCP, no clocks.
    """

    run_id: str
    jobs: tuple[FamilyJob, ...]
    ensemble_enabled: bool

    @property
    def families(self) -> list[str]:
        """The families with a job in this run, in DAG order."""
        return [job.family for job in self.jobs]

    @property
    def python_jobs(self) -> list[FamilyJob]:
        """The jobs that run on a Python runtime (everything but ``native``)."""
        return [job for job in self.jobs if job.family != "native"]

    @property
    def native_job(self) -> FamilyJob | None:
        """The BigQuery-native job, if the run has native models — else ``None``."""
        return next((job for job in self.jobs if job.family == "native"), None)


@dataclass(frozen=True)
class DagNode:
    """One node in a run's execution DAG — a family job or the ensemble — with its identity,
    resolved placement, and upstream dependencies.

    ``job_key`` is the node's canonical cross-system id (`registry.ids.make_job_key`, attempt 1 —
    the *planned* identity): the one name the platform stamps as its own job id and a trace keys on.
    ``runtime`` is where it runs (``spark`` / ``ray`` / ``bigquery``); ``hardware`` / ``gpu_type`` /
    ``spark_mode`` are the resolved compute, ``None`` for the ``native`` and ``ensemble`` nodes
    (both run in BigQuery, taking no runtime choice). ``depends_on`` lists the ``job_key`` of every
    node that must finish first: the ensemble depends on all family jobs; a family job depends on
    nothing.
    """

    job_key: str
    family: str
    runtime: str
    models: tuple[str, ...]
    hardware: str | None
    gpu_type: str | None
    spark_mode: str | None
    depends_on: tuple[str, ...]


def dag_nodes(run_dag: RunDag) -> tuple[DagNode, ...]:
    """Resolve a run's DAG into its nodes with deterministic ids and dependencies (pure, offline).

    One node per family job (`RunDag.jobs`) carrying its resolved runtime/hardware and its planned
    ``job_key`` (`registry.ids.make_job_key`, attempt 1), plus — when ensembling is enabled — a
    downstream ensemble node that depends on every family job. This is the offline "given a config,
    which jobs will run and under what ids" surface: the same ``job_key``\\ s the executor stamps
    onto each platform job and its ``run_jobs`` row, so a run's cross-system trace can be derived
    from the config alone, before anything runs.
    """
    from .registry.ids import make_job_key

    nodes = [
        DagNode(
            job_key=make_job_key(run_dag.run_id, job.family, 1),
            family=job.family,
            runtime=job.runtime,
            models=job.models,
            hardware=None if job.compute is None else job.compute.hardware,
            gpu_type=None if job.compute is None else job.compute.gpu_type,
            spark_mode=None if job.compute is None else job.compute.spark_mode,
            depends_on=(),
        )
        for job in run_dag.jobs
    ]
    if run_dag.ensemble_enabled:
        nodes.append(
            DagNode(
                job_key=make_job_key(run_dag.run_id, "ensemble", 1),
                family="ensemble",
                runtime="bigquery",
                models=(),
                hardware=None,
                gpu_type=None,
                spark_mode=None,
                depends_on=tuple(n.job_key for n in nodes),
            )
        )
    return tuple(nodes)


def group_models_by_family(cfg: RunConfig) -> dict[str, list[str]]:
    """Group ``cfg.models`` by each model's registered ``family`` (pure; deterministic order).

    Families are returned in `_FAMILY_ORDER` and models keep their config order within a family.
    Unknown model names raise `errors.ModelError` (via `models.get_model`) rather than being
    silently dropped — the same validation the engines rely on, surfaced once up front.
    """
    grouped: dict[str, list[str]] = {}
    for name in cfg.models:
        grouped.setdefault(get_model(name).family, []).append(name)
    return {family: grouped[family] for family in _FAMILY_ORDER if family in grouped}


def check_model_params(cfg: RunConfig) -> None:
    """Validate ``cfg.model_params`` against the model registry. Raises `errors.ConfigError`.

    Two checks, both of which need the registry loaded and therefore cannot live in ``config.py``
    (eager model-stack imports on the submit path have broken a live run before):

    * a block keyed by a name no model is registered under is a typo, and a typo here is silent —
      the params simply never reach a model;
    * each selected model gets to refuse a block it cannot honour, through
      `models.base_model.BaseModel.validate_params`. The model is told the longest horizon the run
      will ask for, which is the forward horizon or the backtest horizon, whichever is larger.

    Deliberately **not** called from `plan_dag`, which stays total so pure inspection — the SDK's
    ``dag``, a dry run, a test — never raises. Called from the paths that are about to spend.
    """
    known = list_models()
    unknown = sorted(set(cfg.model_params) - set(known))
    if unknown:
        raise ConfigError(
            f"model_params names {unknown}, which are not registered models. Registered: {known}."
        )
    unselected = sorted(set(cfg.model_params) - set(cfg.models))
    if unselected:
        _log.warning(
            "model_params carries entries for %s, which this run does not select in models; "
            "they will have no effect.",
            unselected,
        )
    max_horizon = max(cfg.data.horizon, cfg.backtest.horizon if cfg.backtest.enabled else 0)
    for name in cfg.models:
        authored = dict(cfg.model_params.get(name, {}))
        get_model(name).validate_params(authored, max_horizon=max_horizon)


def plan_dag(cfg: RunConfig) -> RunDag:
    """Resolve a config into its execution DAG (pure, offline — the single planner main.run uses).

    Computes the shared ``run_id`` (`registry.ids.make_run_id`), groups the models by family
    (`group_models_by_family`), and resolves each Python family's compute
    (`config.RunConfig.resolve_family_compute`) — ``native`` carries no compute (it runs in
    BigQuery). The result is the full set of parallel jobs the run schedules under one ``run_id``,
    plus whether the downstream ensemble node runs. No shape is rejected here: every family that has
    models gets a job, and per-family compute was already validated at config load.
    """
    grouped = group_models_by_family(cfg)
    jobs = tuple(
        FamilyJob(
            family=family,
            models=tuple(models),
            compute=None if family == "native" else cfg.resolve_family_compute(family),
        )
        for family, models in grouped.items()
    )
    return RunDag(run_id=make_run_id(cfg), jobs=jobs, ensemble_enabled=cfg.ensemble.enabled)
