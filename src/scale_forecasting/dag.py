"""Resolve a run into its execution DAG: one job per model family, plus the ensemble node.

A run's ``cfg.models`` spans up to four families — ``statistical`` / ``ml`` / ``deep_learning`` /
``native``. Each family that is present becomes an **independent job** under one shared ``run_id``:
the three Python families each run on their own resolved runtime (Spark *xor* Ray, chosen per
family via `config.RunConfig.resolve_family_compute`), while ``native`` runs in BigQuery. When
ensembling is enabled a downstream ensemble node depends on all of them. This module owns the pure,
offline plan; `main.run` executes it and `main.plan_run` / `main.stage_run` emit its commands.

The families run in parallel, so a run's wall-clock is the slowest family's job, not the sum —
adding a BigQuery-native model to a Spark run costs ``max(spark, bq)``, not ``spark + bq``. The DAG
is a pure function of the config (no clocks, no GCP), so the same config always plans the same DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import get_model
from .registry.ids import make_run_id

if TYPE_CHECKING:
    from .config import ResolvedFamilyCompute, RunConfig

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
