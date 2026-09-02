"""The easy path: a thin, well-typed ``Forecaster`` facade over `main.run`.

This is the "simple SDK on top" — one class you point at a config and call. It adds **no**
forecasting logic; every method delegates to the same code the CLI and Composer run
(`scale_forecasting.main.run`, the run registry, the config layer), so a run driven from the
SDK is byte-for-byte the run driven from ``python -m scale_forecasting.main``. Users who need
to drive Spark or Ray themselves skip this class entirely and call the direct surface
(`run_group`, ``make_group_runner``,
``make_chunk_runner``, ``run_cell``) — both paths reuse the identical model machinery. See
``docs/using_the_sdk.md``.

Import cost: this module is cheap to import. The heavy model modules load only when a method that
runs models is called (``main`` is imported lazily inside ``run``/``dry_run``), which preserves the
near-instant ``import scale_forecasting`` contract enforced by ``test_sdk.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import Fanout, RunConfig, estimate_fanout, load_config
from .registry.ids import make_run_id
from .registry.views import VIEW_NAMES
from .router import split_by_runtime

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from .dag import DagNode
    from .settings import Settings

__all__ = [
    "Forecaster",
    "Registry",
    "DryRunResult",
    "RunResult",
    "ModelResult",
    "JobTrace",
    "build_trace_frame",
    "plot_trace",
]

# Columns of the long-form trace frame `build_trace_frame` returns — one row per timed span (a job
# or a cell), stacked so a single frame drives both the per-job overview and the per-cell detail.
_TRACE_COLUMNS: tuple[str, ...] = (
    "kind",  # "job" | "cell"
    "lane",  # the y-axis track: family (job) or worker_id (cell)
    "label",  # human label for the span
    "start",  # absolute wall-clock start
    "end",  # absolute wall-clock end
    "duration_s",  # span length in seconds (None when it can't be derived)
    "status",  # job status ("COMPLETED"/…); None for cells
    "runtime",  # job runtime ("spark"/…) or cell compute_engine
    "model_type",  # cell only; None for jobs
    "ts_id",  # cell only; None for jobs
)

# Registry statuses that mean a run has stopped changing — what `Forecaster.wait` blocks for.
# CANCELLED is terminal too (a stopped run is settled), so `wait`/`monitor` don't hang on one.
_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "PARTIAL", "CANCELLED"})


@dataclass(frozen=True)
class DryRunResult:
    """What a run *would* do, computed offline — no GCP calls, no compute launched.

    ``run_id`` is the deterministic config hash (identical to what an actual run lands under);
    ``fanout`` is the estimated cell count; ``python_models``/``bq_models`` are the runtime split.
    """

    run_id: str
    fanout: Fanout
    python_models: list[str]
    bq_models: list[str]


@dataclass(frozen=True)
class RunResult:
    """A pointer to where a run's results live — the run_id plus how to query them.

    ``dataset_ref`` is ``project.dataset`` (``None`` when the GCP identity can't be resolved, e.g.
    an offline `Forecaster.review`); ``views`` are the registry view names to query under it
    (e.g. ``v_run_summary``, ``v_model_leaderboard``). Filter any of them by ``run_id``.
    """

    run_id: str
    dataset_ref: str | None
    views: tuple[str, ...]


@dataclass(frozen=True)
class ModelResult:
    """One model's outcome on a run, read from ``v_model_leaderboard``.

    ``ensemble_id`` is ``None`` for a base model and the ensemble-config digest for an ensemble
    pseudo-model. ``mean_wape`` / ``mean_mae`` are the mean decision metrics where a backtest scored
    them (``None`` otherwise); ``no_artifact_rate`` near ``1.0`` flags a model that failed most
    cells. Rows come back best-first (lowest ``mean_wape``).
    """

    model_type: str
    ensemble_id: str | None
    compute_engine: str | None
    n_cells: int
    no_artifact_rate: float | None
    median_fit_seconds: float | None
    mean_wape: float | None
    mean_mae: float | None


@dataclass(frozen=True)
class JobTrace:
    """One family's job on a run, read from ``v_run_jobs`` — the cross-system trace row.

    ``job_key`` is the job's canonical id (``run_jobs.job_id``); ``system_job_id`` is the platform's
    own id for the same job (a Dataproc batch/job id, a Ray ``submission_id``, or a BigQuery parent
    job id), so a status query can go straight to that platform. ``runtime`` says which platform to
    query (``spark`` / ``ray`` / ``bigquery``); ``hardware`` / ``gpu_type`` / ``spark_mode`` record
    the resolved placement. ``status`` is the job's registry/terminal status and ``runtime_seconds``
    its wall-clock; ``attempt`` distinguishes a forced re-run's job from the original under one
    ``run_id``.
    """

    family: str
    job_key: str
    system_job_id: str | None
    runtime: str
    status: str | None
    attempt: int | None
    hardware: str | None
    gpu_type: str | None
    spark_mode: str | None
    runtime_seconds: float | None


class Forecaster:
    """The easy path: point it at a config, then `dry_run`, `run`, `status`, `wait`, or `results`.

    A run driven here is identical to the CLI/Composer run — this class only wraps
    `scale_forecasting.main.run`. Construct from an in-memory `RunConfig`, or use
    `from_file` / `from_dict`. An optional ``settings`` injects the GCP infra identity;
    ``None`` resolves it from the ``SF_*`` environment at run time (the default deployments use).

    The lifecycle closes the loop from one object: `dry_run` (offline plan), `dag` (the planned
    per-job DAG), `run` (execute), `status`/`wait` (track a submission), `results` (read the
    per-model leaderboard), `jobs` (the per-job cross-system trace), and `trace` (the per-job +
    per-cell execution timeline) — all keyed by the config's deterministic ``run_id``, so
    `status`/`results`/`jobs`/`trace` work as a reattach path even in a fresh process.
    """

    def __init__(self, config: RunConfig, *, settings: Settings | None = None) -> None:
        self._config = config
        self._settings = settings

    @classmethod
    def from_file(cls, path: str | Path, *, settings: Settings | None = None) -> Forecaster:
        """Build from a JSON config file (delegates to
        `load_config`, which raises
        `ConfigError` on a bad file)."""
        return cls(load_config(path), settings=settings)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, settings: Settings | None = None) -> Forecaster:
        """Build from an already-parsed config dict (validates via `RunConfig`, raising
        `ConfigError` on a schema violation)."""
        try:
            cfg = RunConfig.model_validate(data)
        except Exception as exc:  # pydantic ValidationError → the package's ConfigError
            from .errors import ConfigError

            raise ConfigError(f"invalid config: {exc}") from exc
        return cls(cfg, settings=settings)

    @property
    def config(self) -> RunConfig:
        """The validated run configuration this forecaster will execute."""
        return self._config

    @property
    def run_id(self) -> str:
        """The deterministic run_id for this config (pure hash; no GCP call)."""
        return make_run_id(self._config)

    def dry_run(self) -> DryRunResult:
        """Validate the config and report the planned fan-out + runtime split — touches no GCP.

        Delegates the run_id to `main.run` (``dry_run=True``) so it is the single source of
        truth, then adds the estimated fanout and the python/BigQuery model split.
        """
        from . import main

        run_id = main.run(self._config, dry_run=True)
        python_models, bq_models = split_by_runtime(self._config)
        return DryRunResult(
            run_id=run_id,
            fanout=estimate_fanout(self._config),
            python_models=python_models,
            bq_models=bq_models,
        )

    def run(
        self,
        *,
        spark: object | None = None,
        n_series: int | None = None,
        max_executors: int | None = None,
    ) -> RunResult:
        """Execute the run (Spark/Ray ∥ BigQuery under one run_id) and return where to query it.

        Delegates to `main.run`, threading this forecaster's ``settings`` (so an injected
        identity is honored). ``spark`` optionally injects a `SparkSession` /
        ``DataprocSparkSession`` for the in-process Spark path (notebook / Connect demo).
        ``n_series`` overrides ``data.series_limit`` (the scale knob — it changes the ``run_id``, so
        each scale is its own run); ``max_executors`` caps the remote Spark batch's executor
        ceiling. Returns a `RunResult` pointing at the registry views under the resolved dataset.

        Note the returned ``run_id`` reflects any ``n_series`` override, so it may differ from this
        forecaster's base `run_id`; use it (or `status`/`results` with it) to track this run.
        """
        from . import main

        run_id = main.run(
            self._config,
            spark=spark,
            settings=self._settings,
            n_series=n_series,
            max_executors=max_executors,
        )
        return RunResult(run_id=run_id, dataset_ref=self._resolved_dataset_ref(), views=VIEW_NAMES)

    def review(self) -> RunResult:
        """Return a pointer to this config's results **without** running anything (offline).

        The run_id is the deterministic config hash, so this resolves the same location a completed
        run landed under. ``dataset_ref`` is ``None`` when the GCP identity can't be resolved (no
        ``SF_*`` env and no injected ``settings``) — the run_id + view names are still returned so a
        caller can construct the query once infra is known.
        """
        return RunResult(
            run_id=self.run_id, dataset_ref=self._resolved_dataset_ref(), views=VIEW_NAMES
        )

    def monitor(self, run_id: str | None = None, *, probe: bool = False) -> Any:
        """Live progress of a run: per-family job state + series done vs. expected (`RunProgress`).

        Delegates to `review.monitor_run` for ``run_id`` (default: this config's id) — header
        status, each family's runner and status, how many cells have landed vs. the expected total,
        and how long each family has been quiet. Poll it while a run is in flight; pair with
        `review.plot_progress` for the bar.

        ``probe=True`` also escalates the non-terminal jobs to their runtime and attaches the
        reconciled `probes.reconcile.ProbeReport` — use it when a bar has stopped moving and you
        need to know whether the job is still alive. It is off by default because a poll loop must
        stay registry-only; `probe` (the drill-down) is the deliberate version of the same read.
        """
        from .review import monitor_run

        return monitor_run(run_id or self.run_id, probe=probe, settings=self._settings)

    def review_run(self, run_id: str | None = None) -> Any:
        """Data-science review of a finished run: bests, metric panel, ensemble lift (`RunReview`).

        Delegates to `review.review_run` for ``run_id`` (default: this config's id) — best model per
        family and overall, the full metric panel aggregated across series, and each ensemble's lift
        over the best base model. The reading counterpart to the offline `review` pointer; pair with
        `review.plot_leaderboard` / `review.plot_metric_distribution`.
        """
        from .review import review_run as _review_run

        return _review_run(run_id or self.run_id, settings=self._settings)

    def status(self, run_id: str | None = None) -> str | None:
        """The current registry status of a run (``RUNNING``/``COMPLETED``/``FAILED``/``PARTIAL``).

        Reads the run's header (`registry.header.header_status`); returns ``None`` when this config
        has never run. ``run_id`` defaults to this config's deterministic id, so
        ``forecaster.status()`` answers "did my config's run finish?" — the reattach path for a
        ``wait=False`` submit.
        """
        from .registry.header import header_status

        return header_status(run_id or self.run_id, settings=self._settings)

    def wait(
        self, run_id: str | None = None, *, timeout: float = 3600.0, poll_seconds: float = 15.0
    ) -> str:
        """Block until a run reaches a terminal status and return it; raise on timeout/not-found.

        Polls `status` every ``poll_seconds`` until the status is terminal
        (``COMPLETED``/``FAILED``/``PARTIAL``). Raises `TimeoutError` if ``timeout`` elapses first,
        or `ConfigError` if the run has no header yet (never submitted). Each poll re-reads the
        header through a fresh client, so a long wait re-authenticates rather than reusing a stale
        token.
        """
        from .errors import ConfigError

        rid = run_id or self.run_id
        deadline = time.monotonic() + timeout
        while True:
            status = self.status(rid)
            if status is None:
                raise ConfigError(f"no run found for run_id {rid}: nothing to wait on")
            if status in _TERMINAL_STATUSES:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(f"run {rid} still {status} after {timeout:.0f}s")
            time.sleep(poll_seconds)

    def results(self, run_id: str | None = None) -> list[ModelResult]:
        """The per-model leaderboard for a run — one `ModelResult` each, best (lowest WAPE) first.

        Reads ``v_model_leaderboard`` (`registry.reads.read_leaderboard`) for ``run_id`` (default:
        this config's id). Returns ``[]`` when the run has produced no scored rows yet. This is the
        first-class "which model won?" read that `RunResult` only pointed at before.
        """
        from .registry.reads import read_leaderboard

        rows = read_leaderboard(run_id or self.run_id, settings=self._settings)
        return [
            ModelResult(
                model_type=r["model_type"],
                ensemble_id=r.get("ensemble_id"),
                compute_engine=r.get("compute_engine"),
                n_cells=r["n_cells"],
                no_artifact_rate=r.get("no_artifact_rate"),
                median_fit_seconds=r.get("median_fit_seconds"),
                mean_wape=r.get("mean_wape"),
                mean_mae=r.get("mean_mae"),
            )
            for r in rows
        ]

    def dag(self) -> tuple[DagNode, ...]:
        """The planned execution DAG for this config — pure, offline, no GCP.

        Resolves the config into its nodes (`dag.dag_nodes`): one per model family plus the ensemble
        (when enabled), each with the deterministic ``job_key`` it will run under, its resolved
        runtime/hardware, and its upstream dependencies. Because ``job_key``\\ s are derived from
        the config alone, this gives every job's identity *before* the run — the offline counterpart
        to `jobs`, so a caller can line up planned nodes against executed rows by ``job_key``.
        """
        from .dag import dag_nodes, plan_dag

        return dag_nodes(plan_dag(self._config))

    def emit_airflow(self, config_uri: str | None = None, *, dag_id: str | None = None) -> str:
        """Render this config's Airflow DAG as a ``dag_<run_id>.py`` source string — pure, offline.

        The Composer counterpart to `dag`: instead of the node list, it returns a standalone Airflow
        DAG file (`airflow_emit.emit_airflow_dag`) whose tasks reproduce this exact run — one job
        per family on its resolved runtime, the ensemble node, and (when the run has several
        ephemeral Ray/Dataproc-cluster families) the shared-cluster bracket — all under the same
        deterministic ``run_id``. ``config_uri`` is embedded verbatim as the DAG's ``CONFIG_URI``
        (what each task loads the config from); pass the ``gs://`` URI of a staged config for a
        Composer-runnable DAG, or omit it to reference this config's own `run_id` file under the
        convention (``runs/<run_id>.json``). ``dag_id`` overrides the default
        ``scale_forecasting_<run_id>``. Touches no GCP — write the returned string to the Airflow
        DAGs folder (or stage it with `staging.stage_dag`).
        """
        from .airflow_emit import emit_airflow_dag

        uri = config_uri if config_uri is not None else f"runs/{self.run_id}.json"
        return emit_airflow_dag(self._config, uri, dag_id=dag_id)

    def jobs(self, run_id: str | None = None) -> list[JobTrace]:
        """The per-job cross-system trace for a run — one `JobTrace` per family, plus the ensemble.

        Reads ``v_run_jobs`` (`registry.jobs.read_run_jobs`) for ``run_id`` (default: this config's
        id): the authoritative map from each family's canonical ``job_key`` to the platform job that
        actually ran it (its ``system_job_id`` on Spark/Ray/BigQuery) and how it fared. Returns
        ``[]`` when the run has no jobs yet. This is the "where did each family run, and can I go
        look at that platform's job?" read — the executed counterpart to the offline `dag`.
        """
        from .registry.jobs import read_run_jobs

        rows = read_run_jobs(run_id or self.run_id, settings=self._settings)
        return [
            JobTrace(
                family=r["family"],
                job_key=r["job_id"],
                system_job_id=r.get("system_job_id"),
                runtime=r["runtime"],
                status=r.get("status"),
                attempt=r.get("attempt"),
                hardware=r.get("hardware"),
                gpu_type=r.get("gpu_type"),
                spark_mode=r.get("spark_mode"),
                runtime_seconds=r.get("runtime_seconds"),
            )
            for r in rows
        ]

    def probe(self, run_id: str | None = None, job: str | None = None) -> Any:
        """Reconcile a run against its runtime: per-family verdict + a disagreement flag.

        Delegates to `probes.reconcile.probe_run` for ``run_id`` (default: this config's id): the
        registry-first drill-down that fuses the header + landed artifacts with, **only for the
        incomplete or stale jobs**, their live native state (Spark / Ray / BigQuery) — flagging each
        row where the runtime contradicts the registry. ``job`` narrows the escalation to one family
        (``statistical``/``ml``/``deep_learning``/``native``/``ensemble``). A run that is already
        terminal short-circuits and touches no runtime; for the fleet-wide view use `monitor`.
        """
        from .probes.reconcile import probe_run

        return probe_run(run_id or self.run_id, job=job, settings=self._settings)

    def cancel(
        self,
        run_id: str | None = None,
        job: str | None = None,
        *,
        confirm: bool = False,
        reason: str = "",
    ) -> Any:
        """Cancel a run (or one family) — a **preview by default**; stop for real only on confirm.

        Delegates to `probes.cancel.cancel_run` for ``run_id`` (default: this config's id): it
        probes the run to reconcile live state, then returns a `CancelReport`. Without ``confirm``
        the report is a blast-radius **preview** — it touches no runtime and no registry, so you see
        exactly which jobs would stop and what partial data is retained. With ``confirm=True`` it
        stops each non-terminal family's runtime job, finalizes its registry row to ``CANCELLED``
        (retaining — never deleting — landed partial results), records ``reason`` + the ADC actor
        for audit, and rolls the run header up. ``job`` narrows to one family
        (``statistical``/``ml``/``deep_learning``/``native``/``ensemble``); an already-terminal job
        is a no-op.
        """
        from .probes.cancel import cancel_run

        return cancel_run(
            run_id or self.run_id,
            job=job,
            confirm=confirm,
            reason=reason,
            settings=self._settings,
        )

    def trace(self, run_id: str | None = None, *, cell_limit: int = 5000) -> pd.DataFrame:
        """The run's execution timeline as a long-form frame — per-job spans + per-cell spans.

        Reads the per-job trace (``v_run_jobs`` via `jobs`' reader) and the per-cell wall-clock
        brackets (``forecast_metadata`` via `registry.reads.read_cell_timing`, capped at
        ``cell_limit``) for ``run_id`` (default: this config's id), then stacks them into one frame
        (columns ``kind``/``lane``/``label``/``start``/``end``/``duration_s``/``status``/
        ``runtime``/``model_type``/``ts_id``) via `build_trace_frame`. Feed it to `plot_trace` for a
        Gantt/waterfall, or slice it directly (``frame[frame.kind == "cell"]``). Returns an empty
        frame (with the columns) when the run has no timed rows — e.g. an older run written before
        the trace columns existed. This is the "how did this run unfold over wall-clock time, and
        which worker ran what?" read that sits under the per-job `jobs` summary.
        """
        from .registry.jobs import read_run_jobs
        from .registry.reads import read_cell_timing

        rid = run_id or self.run_id
        job_rows = read_run_jobs(rid, settings=self._settings)
        cell_rows = read_cell_timing(rid, limit=cell_limit, settings=self._settings)
        return build_trace_frame(job_rows, cell_rows)

    def _resolved_dataset_ref(self) -> str | None:
        """``project.dataset`` from the injected/resolved `Settings`, or ``None`` if
        unresolvable (missing ``SF_*`` env) — keeps `review` graceful offline.

        The **registry** dataset: what a `RunResult` points at is the results + views, not the
        source panel."""
        settings = self._settings
        if settings is None:
            from .errors import ConfigError
            from .settings import Settings

            try:
                settings = Settings.resolve()
            except ConfigError:
                return None
        return settings.registry_dataset_ref

    def registry(self) -> Registry:
        """A `Registry` handle for the registry this config's runs land in.

        The lifecycle verbs (`Registry.doctor`, `Registry.drop_run`, …) are keyed on the *registry*,
        not on a config, so they live on their own object — but reaching them from the `Forecaster`
        you already have is the common case (`f.registry().drop_run(f.run_id)`).
        """
        return Registry(settings=self._settings)


class Registry:
    """The easy path for **managing** a registry — the operator counterpart to `Forecaster`.

    `Forecaster` produces runs; this reads and prunes them. It is keyed on the registry the ``SF_*``
    environment resolves to (`Settings.registry_dataset_ref`), not on a config, so it needs no
    config at all — which is exactly right for "what is in this registry, and clean up run X".

    Like `Forecaster`, it adds no logic: every method delegates to `registry.ops`, the same code
    ``python -m scale_forecasting.registry.ops`` runs, so the notebook, the SDK, and the CLI can
    never diverge. The destructive verbs preview by default and execute only on ``yes=True``.

        >>> from scale_forecasting import Registry
        >>> reg = Registry()
        >>> print(reg.doctor())                 # read-only: counts, stuck runs, orphans
        >>> reg.drop_run("abc123")              # preview — deletes nothing
        >>> reg.drop_run("abc123", yes=True)    # artifacts, then models, then rows

    There is deliberately no wipe-everything method: a full teardown is ``bq rm -r -f <dataset>``.
    """

    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings

    def __repr__(self) -> str:
        return f"Registry({self.dataset_ref or 'unresolved'})"

    @property
    def dataset_ref(self) -> str | None:
        """``project.dataset`` for this registry, or ``None`` when ``SF_*`` can't be resolved."""
        settings = self._settings
        if settings is None:
            from .errors import ConfigError
            from .settings import Settings

            try:
                settings = Settings.resolve()
            except ConfigError:
                return None
        return settings.registry_dataset_ref

    def init(self, *, create_dataset: bool = False) -> str:
        """Create this registry's tables + views (idempotent). See `registry.ops.init`."""
        from .registry import ops

        return ops.init(settings=self._settings, create_dataset=create_dataset)

    def doctor(self) -> Any:
        """A read-only `registry.ops.DoctorReport` — row counts, stuck runs, orphaned artifacts."""
        from .registry import ops

        return ops.doctor(settings=self._settings)

    def close_runs(self, *run_ids: str, yes: bool = False) -> Any:
        """Finalize abandoned ``RUNNING`` headers to what their job rows already imply.

        No arguments means every stuck header in this registry. Preview unless ``yes``. Deletes
        nothing and skips any run with a job row that is not yet terminal — probe those first with
        ``monitor(probe=True)``. See `registry.ops.close_runs`.
        """
        from .registry import ops

        return ops.close_runs(list(run_ids) or None, settings=self._settings, yes=yes)

    def drop_run(self, *run_ids: str, yes: bool = False, force: bool = False) -> Any:
        """Delete named run(s) — artifacts, then BQML models, then rows. Preview unless ``yes``.

        See `registry.ops.drop_run` for the ordering rule and the in-flight refusal ``force``
        overrides.
        """
        from .registry import ops

        return ops.drop_run(list(run_ids), settings=self._settings, yes=yes, force=force)

    def sweep_orphans(self, *, yes: bool = False) -> Any:
        """Delete artifacts under this registry with no ``run_registry`` row.

        Preview unless ``yes``. See `registry.ops.sweep_orphans`.
        """
        from .registry import ops

        return ops.sweep_orphans(settings=self._settings, yes=yes)

    def snapshot(
        self, suffix: str, *, into: str | None = None, expiration_days: int | None = None
    ) -> dict[str, str]:
        """Table-snapshot every registry table; returns ``{table: snapshot ref}``."""
        from .registry import ops

        return ops.snapshot(
            suffix, settings=self._settings, into=into, expiration_days=expiration_days
        )

    def export(self, destination_root: str, *, fmt: str = "PARQUET") -> dict[str, str]:
        """Export every registry table to GCS; returns ``{table: destination prefix}``."""
        from .registry import ops

        return ops.export(destination_root, settings=self._settings, fmt=fmt)


def _duration_s(start: Any, end: Any, fallback: Any) -> float | None:
    """Seconds between two timestamps, or ``fallback`` when the pair can't yield a span."""
    if start is not None and end is not None:
        return (end - start).total_seconds()
    return float(fallback) if fallback is not None else None


def build_trace_frame(
    job_rows: list[dict[str, Any]], cell_rows: list[dict[str, Any]]
) -> pd.DataFrame:
    """Stack per-job and per-cell timing rows into one long-form trace frame — pure, no I/O.

    ``job_rows`` are ``v_run_jobs`` rows (see `registry.jobs.read_run_jobs`) and ``cell_rows`` are
    per-cell timing rows (see `registry.reads.read_cell_timing`); this is the pure assembly step
    `Forecaster.trace` calls after reading them, split out so it unit-tests without GCP. Each input
    row becomes one span record with the shared `_TRACE_COLUMNS` schema: a job lands on its
    ``family`` lane, a cell on its ``worker_id`` lane, so a plot can show the DAG over the workers
    that ran it. Rows without a start stamp are dropped (nothing to place on a timeline); a row with
    a start but no end (a job that never recorded completion) keeps a zero-width span at its start.
    Returns an empty frame carrying the columns when both inputs are empty.
    """
    import pandas as pd

    records: list[dict[str, Any]] = []
    for r in job_rows:
        start = r.get("started_at")
        if start is None:
            continue
        end = r.get("ended_at")
        records.append(
            {
                "kind": "job",
                "lane": r.get("family") or "",
                "label": r.get("family") or r.get("job_id") or "",
                "start": start,
                "end": end if end is not None else start,
                "duration_s": _duration_s(start, end, r.get("runtime_seconds")),
                "status": r.get("status"),
                "runtime": r.get("runtime"),
                "model_type": None,
                "ts_id": None,
            }
        )
    for r in cell_rows:
        start = r.get("cell_started_at")
        if start is None:
            continue
        end = r.get("cell_ended_at")
        model_type = r.get("model_type")
        ts_id = r.get("ts_id")
        records.append(
            {
                "kind": "cell",
                "lane": r.get("worker_id") or "",
                "label": f"{model_type}:{ts_id}",
                "start": start,
                "end": end if end is not None else start,
                "duration_s": _duration_s(start, end, None),
                "status": None,
                "runtime": r.get("compute_engine"),
                "model_type": model_type,
                "ts_id": ts_id,
            }
        )
    return pd.DataFrame.from_records(records, columns=list(_TRACE_COLUMNS))


def plot_trace(frame: pd.DataFrame, *, ax: Any = None, title: str = "run trace") -> Any:
    """Render a `build_trace_frame` frame as a Gantt/waterfall and return the matplotlib ``Axes``.

    A convenience over the frame (the frame is the real deliverable): one horizontal bar per span,
    lanes stacked on the y-axis (jobs and cells kept apart so a ``family`` job and a ``worker_id``
    cell never share a track), colored by ``kind`` (job vs cell). Pass an existing ``ax`` to compose
    into a larger figure, or let it create one. matplotlib imports lazily here so it never touches
    the near-instant ``import scale_forecasting`` path. An empty frame renders an empty titled axes
    rather than raising.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(title if not frame.empty else f"{title} (no timed rows)")
    if frame.empty:
        return ax

    lane_keys = list(dict.fromkeys(zip(frame["kind"], frame["lane"], strict=True)))
    colors = {"job": "#4C72B0", "cell": "#DD8452"}
    for y, (kind, lane) in enumerate(lane_keys):
        grp = frame[(frame["kind"] == kind) & (frame["lane"] == lane)]
        spans = [
            (mdates.date2num(row.start), mdates.date2num(row.end) - mdates.date2num(row.start))
            for row in grp.itertuples(index=False)
        ]
        ax.broken_barh(spans, (y - 0.4, 0.8), facecolors=colors.get(kind, "#999999"))
    ax.set_yticks(range(len(lane_keys)))
    ax.set_yticklabels([f"{kind}:{lane}" for kind, lane in lane_keys])
    ax.xaxis_date()
    ax.set_xlabel("wall-clock time")
    return ax
