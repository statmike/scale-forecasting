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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import ConfigError, get_logger
from .models import get_model, list_models
from .registry.ids import base_family, make_run_id, repair_family

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
        """The jobs that run on a Python runtime (everything but ``native``).

        Asks `registry.ids.base_family`, so a repair DAG's ``native_repair`` job routes to BigQuery
        like the family it repairs. Splitting them here — the one place the two launchers are
        chosen — is what keeps the repair token an identity concern and nothing more.
        """
        return [job for job in self.jobs if base_family(job.family) != "native"]

    @property
    def native_job(self) -> FamilyJob | None:
        """The BigQuery-native job, if the run has native models — else ``None``."""
        return next((job for job in self.jobs if base_family(job.family) == "native"), None)


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

    **A retry node is never one of these, and cannot be.** An emitted Airflow DAG may carry one
    (`airflow_emit.emit_airflow_dag` with ``with_retry``), but a repair's identity is not knowable
    from the config: which families it touches depends on what failed, and its attempt number is
    whatever `registry.jobs.next_job_attempt` hands out at the time — while every key here is
    attempt 1 by construction, because this function is pure and a planned identity has no history
    to count. The repair resolves its own ``job_key`` inside `airflow_tasks.retry_families`, at the
    moment it has the run in front of it.
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
    for name in cfg.models:
        authored = dict(cfg.model_params.get(name, {}))
        get_model(name).validate_params(authored, max_horizon=cfg.max_horizon)


def check_hardware_coherence(cfg: RunConfig, jobs: tuple[FamilyJob, ...]) -> None:
    """Refuse a plan that would buy a device nothing will route to. Raises `errors.ConfigError`.

    **This is expected to be silent, and that is the point.** Provisioning and routing were made to
    read one function (`config.RunConfig.resolve_family_compute`, via
    `engines.ray_io.resolve_job_gpu`) precisely so they cannot disagree, and `ComputeConfig` already
    refuses ``hardware: "gpu"`` on a family that is not ``deep_learning``. Every incoherence we know
    about is therefore closed upstream. What this adds is a **standing tripwire**: it recomputes the
    routing answer down the engine's own path and compares it against the hardware the DAG planned
    — two call paths, one expected answer — so the day an edit reintroduces a flat-field read the
    plan refuses instead of quietly paying for idle accelerators for a whole fleet-hour. That defect
    shipped once and nothing reported it; the cost of the check is a pure function call.

    No config field and no escape hatch. Incoherence has exactly one correct answer and the remedy
    is always a config edit, so an override would only let a run buy hardware it cannot use.

    Called from the spend paths only (via `preflight`) — `main.run`, `launch_plan.stage_run`,
    `airflow_tasks.begin_run` — never from `plan_dag`, which stays total.
    """
    from .engines.ray_io import resolve_job_gpu, split_gpu_cpu_models

    routed_gpu, routed_type = resolve_job_gpu(cfg)
    for job in jobs:
        if job.compute is None or job.compute.hardware != "gpu":
            continue
        # A GPU job must have at least one model the engine will actually send to the GPU pool.
        # `split_gpu_cpu_models` is the function that does the sending, so ask it rather than
        # restating its rule here — a divergence between the two is exactly what this catches.
        gpu_models, _ = split_gpu_cpu_models(cfg, list(job.models), use_gpu=True)
        if not gpu_models:
            raise ConfigError(
                f"family '{job.family}' resolves to GPU hardware but none of its models "
                f"{list(job.models)} route to the GPU pool, so the run would provision "
                f"accelerators that nothing schedules onto. Set "
                f"compute.families.{job.family}.hardware to 'cpu', or select a model the pool "
                f"can use."
            )
        if not routed_gpu:
            raise ConfigError(
                f"family '{job.family}' is planned onto GPU hardware but the engine's own routing "
                f"resolves this run to CPU. Provisioning and routing must read one answer; they "
                f"have diverged, which would leave the devices idle for the whole run."
            )
        if job.compute.gpu_type != routed_type:
            raise ConfigError(
                f"family '{job.family}' is planned onto {job.compute.gpu_type} but the engine "
                f"routes to {routed_type}. The device type sets the memory denominator the GPU "
                f"fraction is derived from, so a mismatch mis-sizes the pool."
            )


def gpu_usefulness_report(cfg: RunConfig, jobs: tuple[FamilyJob, ...]) -> list[str]:
    """Warnings about a device that will be paid for and barely used. Never refuses.

    Returns human-readable lines for the caller to log; an empty list means nothing to say. The
    separation from `check_hardware_coherence` is deliberate and is the difference between "this
    plan is wrong" and "this plan is probably wasteful": a GPU that no model can use is a mistake
    with one correct answer, while a GPU a model *can* use but will barely touch is a judgement
    call about cost, and the only configuration that would satisfy a hard check
    (``model_params.neuralprophet.n_lags > 0``) has never been run at scale here. Refusing on
    usefulness would break the shipped smokes and demo configs on the day it landed, in favour of a
    setting with no green run behind it. So: warn, loudly, with both remedies and which one is
    proven.

    Surfaced wherever a plan is shown or submitted — a dry run, the quota preflight, the SDK's
    ``dag``, and the submit log.
    """
    lines: list[str] = []
    gpu_jobs = [j for j in jobs if j.compute is not None and j.compute.hardware == "gpu"]

    for job in gpu_jobs:
        capable = [m for m in job.models if get_model(m).gpu_capable]
        incapable = [m for m in job.models if m not in capable]
        if incapable:
            lines.append(
                f"family '{job.family}' has a {job.compute.gpu_type} attached, but "
                f"{incapable} cannot use a device at all — those cells will run on the host CPU "
                f"while the accelerator is billed."
            )
        idle = [m for m in capable if not get_model(m).gpu_useful(cfg.model_params.get(m, {}))]
        if idle:
            lines.append(
                f"family '{job.family}' has a {job.compute.gpu_type} attached and {idle} can use "
                f"it, but not at the hyperparameters this config authors. Measured over 31,356 "
                f"fits on live T4s: peak device memory 50-78 KB against a 17 GB card, and "
                f"cpu_seconds/fit_seconds 0.93-0.996 — the device is engaged and idle, so this is "
                f"a CPU run being billed as a GPU run. Two remedies: set "
                f"compute.families.{job.family}.hardware to 'cpu', which is the PROVEN one and "
                f"costs nothing in accuracy; or turn on autoregression with "
                f"model_params.neuralprophet.n_lags, which is what would make the device earn its "
                f"cost but has NO green run behind it yet — no accuracy A/B and no live smoke."
            )

    # The inverse: `use_gpu` is set and nothing lands on a device. Two different causes, and naming
    # the wrong one sends a reader hunting for a config bug that is not there. The A/B's CPU arm is
    # the second case exactly — it selects neuralprophet and routes it off the accelerator — and it
    # was told the first, that no deep-learning model was selected.
    if not gpu_jobs and cfg.compute.use_gpu:
        routed_off = sorted(
            {
                m
                for job in jobs
                if job.compute is not None and job.compute.hardware != "gpu"
                for m in job.models
                if get_model(m).gpu_capable
            }
        )
        if routed_off:
            lines.append(
                f"compute.use_gpu is true but {routed_off} resolves to CPU hardware, so no job "
                f"resolves to GPU and the flag buys nothing. A families.<family>.hardware override "
                f"wins over the flat flag by design, so this is not an error — it is worth saying "
                f"only because a config that reads as a GPU run and is not one is easy to mistake "
                f"for one that is."
            )
        else:
            lines.append(
                "compute.use_gpu is true but no GPU-capable model is selected, so no job resolves "
                "to GPU hardware and the flag has no effect. Select a GPU-capable model or drop "
                "the flag; leaving it set makes the config read as a GPU run when it is not one."
            )
    return lines


def narrow_to_models(run_dag: RunDag, models: Iterable[str]) -> RunDag:
    """The same DAG with every job narrowed to ``models``, dropping jobs left with nothing (pure).

    This is how a repair submits less than a run: `job_launch.submit_retry` walks the narrowed DAG
    and each surviving job carries a shorter `FamilyJob.models`, which reaches the driver as
    ``--models`` (`commands.build_driver_args`) — the subset seam that already existed for
    per-family execution, reused rather than reinvented. Resolved compute rides along untouched, so
    a repaired family lands on the same runtime and hardware the original attempt chose.

    Each surviving job is re-stamped with its **repair family token** (`registry.ids.repair_family`
    — ``statistical`` becomes ``statistical_repair``), which is how the repair gets a ``run_jobs``
    row of its own instead of overwriting the row of the attempt it repairs. `v_run_jobs` keeps the
    highest attempt per (run_id, family), so a forty-cell repair filed under ``statistical`` would
    become the only ``statistical`` row the registry shows and report the whole family COMPLETED.
    Nothing downstream has to know: every routing decision asks `registry.ids.base_family`, so the
    job runs on the runtime, hardware, and launcher the original attempt chose.

    ``ensemble_enabled`` is always ``False`` on the result. A repair re-runs base models; whether
    the ensemble is recomputed afterwards is a question about the *run*, answered by the node
    ordering in `airflow_emit` rather than by a narrowed job list, and a DAG that advertised an
    ensemble node nobody was going to run would be a lie in the one structure the trace reads from.

    Raises `errors.ConfigError` if ``models`` names anything the DAG does not plan — a repair can
    only re-ask a question this run already asked, and quietly ignoring the name would submit a
    smaller job than the caller believes they asked for.
    """
    keep = set(models)
    planned = {m for job in run_dag.jobs for m in job.models}
    unknown = keep - planned
    if unknown:
        raise ConfigError(
            f"cannot narrow run {run_dag.run_id} to {sorted(unknown)}: not planned by "
            f"this run (it runs {sorted(planned)})"
        )
    jobs = tuple(
        FamilyJob(
            family=repair_family(job.family),
            models=tuple(m for m in job.models if m in keep),
            compute=job.compute,
        )
        for job in run_dag.jobs
        if any(m in keep for m in job.models)
    )
    return RunDag(run_id=run_dag.run_id, jobs=jobs, ensemble_enabled=False)


def preflight(cfg: RunConfig) -> RunDag:
    """Everything a run is refused or warned about before it provisions anything. Returns the DAG.

    One call so the spend paths cannot drift apart on which checks they run: an authored
    ``model_params`` block no model can honour (`check_model_params`), a plan that would buy a
    device nothing routes to (`check_hardware_coherence`), and — logged, never fatal — a device that
    will be billed and barely used (`gpu_usefulness_report`).

    Separate from `plan_dag` on purpose. Planning is pure inspection and must stay total: the SDK's
    ``dag``, a notebook, and several hundred tests call it and none of them are spending anything.
    Refusing is a different act, so it is a different function, and only `main.run`,
    `launch_plan.stage_run` and `airflow_tasks.begin_run` perform it.
    """
    check_model_params(cfg)
    run_dag = plan_dag(cfg)
    check_hardware_coherence(cfg, run_dag.jobs)
    for line in gpu_usefulness_report(cfg, run_dag.jobs):
        _log.warning("%s", line)
    return run_dag


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
