"""Pure row assembly — a `CellResult` and a `RunConfig` in, the exact table rows out.

No BigQuery client, no network: column mapping, stamping ``run_id``/``ts_id``/``model_type``/
``compute_engine``, JSON serialization, and the per-cell idempotency key. Everything here is
tested offline; the modules that carry these rows to BigQuery (`registry.cells`,
`registry.header`, `registry.jobs`) are the ones that need a client.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any, get_args

from ..config import DecisionMetric

if TYPE_CHECKING:
    from ..config import RunConfig
    from ..worker import CellResult


# The full metric panel, in table-column order — derived from the config's DecisionMetric
# literal so there is exactly one source of truth.
METRIC_COLUMNS: tuple[str, ...] = get_args(DecisionMetric)


def cell_dedup_key(result: CellResult) -> dict[str, str]:
    """The run-scoped identity anchor for a cell's rows.

    Idempotency is **append-only + dedupe-on-read**, anchored on ``run_id``. `write_cells`
    never DELETEs — a DELETE that matches rows still in the Storage Write API streaming buffer
    is rejected for the whole buffer window (~90 min), so a clear-then-append is not viable
    against the default stream. Instead we rely on
    ``run_id`` being a pure function of the config (``make_run_id``): the same ``run_id`` implies
    the same config implies byte-identical rows, so a re-run's "duplicates" are exact copies.
    Serving views dedupe with ``DISTINCT``/``GROUP BY`` on ``run_id`` (+ cell keys); no write-time
    delete is needed. ``model_hash`` uniquely identifies the cell on ``forecast_metadata`` for
    lineage.
    """
    return {"run_id": result.run_id}


def assemble_prediction_rows(result: CellResult) -> list[dict[str, Any]]:
    """Canonical prediction frame → ``forecast_predictions`` rows.

    Stamps run/series/model/engine onto each row and maps ``ds`` → ``forecast_date``.
    ``quantiles`` is serialized to a JSON string (or None).
    """
    rows: list[dict[str, Any]] = []
    for rec in result.predictions.to_dict("records"):
        rows.append(
            {
                "run_id": result.run_id,
                "ts_id": result.ts_id,
                "model_type": result.model_type,
                "compute_engine": result.compute_engine,
                "forecast_date": _as_date(rec["ds"]),
                "yhat": _as_float(rec.get("yhat")),
                # The two arms `yhat` was chosen between. `yhat_raw` is the model's own output;
                # `yhat_adjusted` adds the out-of-fold bias correction. Written side by side so
                # that switching `output.point_forecast` on a delivered run is a query, not a
                # re-run — and so the comparison between them is a fact in the registry rather
                # than a claim in a docstring.
                "yhat_raw": _as_float(rec.get("yhat_raw")),
                "yhat_adjusted": _as_float(rec.get("yhat_adjusted")),
                "yhat_lower": _as_float(rec.get("yhat_lower")),
                "yhat_upper": _as_float(rec.get("yhat_upper")),
                "quantiles": _as_json(rec.get("quantiles")),
            }
        )
    return rows


def assemble_oof_rows(result: CellResult) -> list[dict[str, Any]]:
    """Canonical OOF frame (`backtest.OOF_COLUMNS`) → ``backtest_oof`` rows. Empty if no backtest.

    Four of these columns were declared in the schema and written by nobody. They are all things
    the fold loop had in hand and discarded: the bounds it now scores coverage on, the origin date
    the fold forecast from, and the step within the fold's horizon. Read with ``.get`` so a frame
    assembled by something other than `backtest_cell` still produces a valid row.
    """
    if result.oof is None:
        return []
    rows: list[dict[str, Any]] = []
    for rec in result.oof.to_dict("records"):
        rows.append(
            {
                "run_id": result.run_id,
                "ts_id": result.ts_id,
                "model_type": result.model_type,
                "fold_id": int(rec["fold_id"]),
                "forecast_date": _as_date(rec["ds"]),
                "y_true": _as_float(rec.get("y_true")),
                "yhat": _as_float(rec.get("yhat")),
                # These are the residual source. `calibration.calibrate_from_oof` learns the
                # correction from `y_true - yhat_raw`, so the column it learns from has to be in
                # the registry for the number to be auditable after the fact.
                "yhat_raw": _as_float(rec.get("yhat_raw")),
                "yhat_adjusted": _as_float(rec.get("yhat_adjusted")),
                "yhat_lower": _as_float(rec.get("yhat_lower")),
                "yhat_upper": _as_float(rec.get("yhat_upper")),
                # The fold's training cutoff — its identity across a ragged panel, where the same
                # `fold_id` covers different dates for different series (`ensembler._pivot_oof`).
                "cutoff_date": _as_date(rec.get("cutoff_date")),
                "horizon_step": _as_int(rec.get("horizon_step")),
            }
        )
    return rows


def assemble_metadata_row(
    result: CellResult, created_at: datetime, model_artifact: str | None = None
) -> dict[str, Any]:
    """One full-fit ``forecast_metadata`` row: metrics panel + artifact link.

    ``fold_id`` is None (this is the full-fit summary row). ``model_artifact`` is the
    ObjectRef/URI filled in by the writer after the artifact upload. ``worker_id`` and the
    ``cell_started_at``/``cell_ended_at`` wall-clock bracket come off the cell (the Python worker
    stamps them); they are None for cells produced outside `run_cell` (native SQL / ensemble).
    """
    row: dict[str, Any] = {
        "run_id": result.run_id,
        "ts_id": result.ts_id,
        "model_type": result.model_type,
        "compute_engine": result.compute_engine,
        "model_hash": result.model_hash,
        "fold_id": None,
        "fit_seconds": _as_float(result.fit_seconds),
        "best_params": _as_json(result.best_params),
        "model_artifact": model_artifact,
        "created_at": created_at,
        "worker_id": result.worker_id,
        "cell_started_at": result.cell_started_at,
        "cell_ended_at": result.cell_ended_at,
        # Harvested compute measurement (compute.profile.measure). All None when measurement is
        # off, which is also how rows written before these columns existed read back — so
        # `profiling.cost.harvest_profile` needs no version check, only a NULL check.
        "cpu_seconds": _as_float(result.cpu_seconds),
        "process_rss_bytes": result.process_rss_bytes,
        "peak_gpu_bytes": result.peak_gpu_bytes,
        "intraop_threads": result.intraop_threads,
        "n_obs": result.n_obs,
        # Device evidence (the GPU contract's Layer 4), recorded rather than inferred. Unlike the
        # harvest columns above these are NOT gated on profiling: whether a device the run paid for
        # was ever visible is not a profiling question, and gating it left 8 of 21 historical GPU
        # jobs with no evidence either way.
        "device_requested": result.device_requested,
        "device_available": result.device_available,
        "device_used": result.device_used,
        "device_name": result.device_name,
        # How the *scoring* went, which is not how the cell went. All three NULL means backtesting
        # was never asked for; a NULL metric panel alone cannot say that, because it is also what a
        # series too short to score looks like. `n_folds_achieved` is the column that makes a
        # leaderboard readable across a ragged panel — two series with the same WAPE are not
        # comparable if one was scored on five folds and the other on one.
        "backtest_status": result.backtest_status,
        "backtest_note": result.backtest_note,
        "n_folds_achieved": result.n_folds_achieved,
        # Where this cell's prediction bounds came from — and therefore what its `coverage`,
        # `pinball` and `interval_score` are evidence *about*. A model with native intervals is
        # reporting its own uncertainty; a model without one is being scored on the empirical
        # spread of its in-sample residuals, which is a different and generally more optimistic
        # claim. Ranking the two on coverage without this column compares two different things.
        #
        # It describes what the *model* produced. `interval_calibration` below describes what
        # happened to it afterwards, and the two are independent: when a backtest ran, the shipped
        # band is re-estimated per horizon step from out-of-fold residuals regardless of which of
        # the two the model started with. Folding that into `interval_source` would have made a
        # native-interval model and a residual-interval model indistinguishable after calibration,
        # which is the one comparison the column exists to support.
        "interval_source": result.interval_source,
        # Which arm `yhat` is (`raw` / `median` / `mean`), how that arm was decided
        # (`configured` or one of the `auto-*` outcomes), how its band was calibrated
        # (`oof-per-step` / `oof-flat` / `in-sample`), and by how much the corrected arm beat the
        # raw one on this cell's own out-of-fold folds, in the run's `decision_metric`. The margin
        # is signed and always measured corrected-minus-raw, whichever arm the cell selected, so
        # it stays comparable across cells that chose differently. A fleet-wide GROUP BY on these
        # four is the diagnostic — whether the correction is earning its place is a question
        # about this run's data, and nothing but this run's data can answer it.
        "point_forecast_source": result.point_forecast_source,
        "point_forecast_decision": result.point_forecast_decision,
        "interval_calibration": result.interval_calibration,
        "point_forecast_margin": result.point_forecast_margin,
        # Whether the hyperparameter search behind `best_params` reserved the newest fold, or had
        # no inner fold left and scored on all of them. NULL on the majority of cells, where no
        # search ran at all. It is here rather than derivable because the answer depends on this
        # series' own length under per-series tuning, and a reader comparing two rows of the same
        # run cannot recover that from the config.
        "hpo_scoring": result.hpo_scoring,
        # How the *cell* went. `run_cell` has always computed this and thrown it away at the table
        # boundary: an error cell was written as a row of NULL metrics with `fit_seconds = 0`, and
        # telling it apart from a successful cell that simply was not scored meant knowing that
        # convention. Now it says so. `error_class` is the fixed vocabulary you can GROUP BY
        # (`worker.ERROR_CLASSES`); `error_detail` is the raw text, truncated, for reading one row.
        "cell_status": result.status,
        "error_class": result.error_class,
        "error_detail": _truncate(result.error),
    }
    for name in METRIC_COLUMNS:
        row[name] = _as_float(result.metrics.get(name))
    return row


def assemble_header_row(
    cfg: RunConfig,
    run_id: str,
    created_at: datetime,
    *,
    snapshot_millis: int | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Build the ``run_registry`` header row from a config.

    ``raw_config`` is the validated config as a **dict** — the config *is* the record.
    ``run_registry.raw_config`` is a native ``JSON`` column, and the client's JSON query
    parameter serializes the value itself (``json.dumps``), so the row must carry the dict, not
    a pre-serialized string (a string would be double-encoded). ``bq_models`` is left empty here
    and filled by the router once model runtimes are known; status starts RUNNING.

    ``user_id`` is the principal that launched the run (the ADC identity resolved by `write_header`
    via `identity.resolve_principal` — a runner SA under Composer/CI, a user's email on a laptop),
    stamped so *launch* is attributable in the audit trail. ``None`` leaves it NULL (the pre-audit
    behavior, and when the principal couldn't be resolved cheaply).

    ``snapshot_millis`` is the input-data snapshot the run pins every read to (epoch millis on the
    BigQuery clock, resolved once by `resolve_snapshot_millis`): stored on the header so every
    family job — whichever runtime — can look it up by ``run_id`` (`snapshot_millis_for`) and read
    the *identical* source state. It is deliberately **not** part of the config (it would perturb
    the config-derived ``run_id``), so it is passed in here, not derived. ``None`` leaves it NULL —
    the reads fall back to unpinned (the pre-snapshot behavior).
    """
    return {
        "run_id": run_id,
        "created_at": created_at,
        "snapshot_millis": snapshot_millis,
        "user_id": user_id,
        "git_sha": None,
        "python_runtime": cfg.python_runtime,
        "bq_models": [],
        "backtest_on": cfg.backtest.enabled,
        "decision_metric": cfg.backtest.decision_metric,
        "ensemble_strategies": list(cfg.ensemble.strategies) if cfg.ensemble.enabled else [],
        "raw_config": cfg.model_dump(mode="json"),
        "status": "RUNNING",
        "n_series": cfg.data.series_limit,
        "n_models": len(cfg.models),
        "runtime_seconds": None,
        # Dataproc-level job telemetry (executor sizing, wall/startup split, DCU usage): a native
        # JSON column filled in after the batch finishes by the submitter (extract_job_telemetry →
        # update_header) as a **dict** (the JSON query param serializes it). NULL here at RUNNING
        # and for any run whose telemetry couldn't be read (best-effort — never blocks a run).
        # See the run_registry DDL.
        "job_telemetry": None,
    }


def assemble_job_row(
    run_id: str,
    family: str,
    attempt: int,
    created_at: datetime,
    *,
    runtime: str | None = None,
    spark_mode: str | None = None,
    hardware: str | None = None,
    gpu_type: str | None = None,
    system_job_id: str | None = None,
    status: str = "RUNNING",
    started_at: datetime | None = None,
    probe_handle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``run_jobs`` row for one family's job under a run.

    The ``job_id`` is derived here (`registry.ids.make_job_key`) from ``(run_id, family, attempt)``
    so the row's identity always matches the id a submitter hands the platform — the two can't
    drift. The resolved compute fields (``runtime``/``spark_mode``/``hardware``/``gpu_type``) are
    passed in, not re-derived, so this stays a pure mapping: the orchestrator resolves them
    (`config.RunConfig.resolve_family_compute` for a model family, the ensemble node's own config
    for ``ensemble``) and hands them over. ``status`` starts RUNNING; ``runtime_seconds``,
    ``failure_reason`` and ``job_telemetry`` are NULL until the job finishes and the submitter
    updates the row.

    ``started_at`` is the job's execution start (defaults to ``created_at`` when not given); the
    matching ``ended_at`` is NULL here and stamped by `run_job` at exit — together they give the
    trace an absolute wall-clock lane per job, alongside the measured ``runtime_seconds``.
    """
    from .ids import make_job_key

    return {
        "job_id": make_job_key(run_id, family, attempt),
        "run_id": run_id,
        "family": family,
        "attempt": attempt,
        "runtime": runtime,
        "spark_mode": spark_mode,
        "hardware": hardware,
        "gpu_type": gpu_type,
        "system_job_id": system_job_id,
        "status": status,
        "created_at": created_at,
        "started_at": started_at if started_at is not None else created_at,
        "ended_at": None,
        "runtime_seconds": None,
        # NULL until something fails with a reason worth naming (`capacity.CAPACITY_EXHAUSTED` is
        # the first). Present-and-NULL rather than absent so the row's keys stay exactly the
        # writable column set — the tripwire below this module's tests enforce.
        "failure_reason": None,
        # The probe handle (runtime coordinates for reconciliation) is stamped at RUNNING entry so a
        # reader can check a live job; NULL when no handle was captured (a pre-feature run).
        "job_telemetry": {"probe_handle": probe_handle} if probe_handle is not None else None,
    }


# --- small pure coercers -------------------------------------------------------

# How much of an error message reaches the table. `repr(exc)` is usually one line, but a library
# that puts a DataFrame or an entire SQL statement in its message turns one bad cell into a
# multi-megabyte column, and at fleet scale that is the row that fails an `append_rows` batch and
# takes its neighbours with it. Diagnosis lives in the first characters; the rest is padding.
_MAX_ERROR_DETAIL = 2000


def _truncate(text: str | None, limit: int = _MAX_ERROR_DETAIL) -> str | None:
    """Cap a free-text column, marking the cut so a reader is not misled by a clean-looking end."""
    if text is None or len(text) <= limit:
        return text
    return f"{text[:limit]}… [truncated, {len(text)} chars]"


def _as_float(value: Any) -> float | None:
    """Coerce to float, mapping missing/non-finite to None (BQ NULL).

    The BigQuery Storage Write API rejects NaN and ±Inf for a FLOAT64 column, and a single
    rejected row fails the whole ``append_rows`` request — which, in a Spark/Ray worker, kills
    the task and cascades to the entire run. A non-finite forecast is a per-series pathology
    (e.g. ``log1p``'s ``expm1`` inverse overflowing to ``+Inf`` on a runaway series), so it must
    not take the fleet down: coerce it to NULL here, at the one boundary every engine's rows flow
    through, so the bad cell lands as a missing value and the run completes.
    """
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None  # NaN and ±Inf → NULL


def _as_json(value: Any) -> str | None:
    """Serialize a dict (or None/empty) to a JSON string, or None.

    Non-finite values (NaN/±Inf) are dropped from a dict before serializing: Python's
    ``json.dumps`` emits the bare literals ``NaN``/``Infinity`` by default, which are invalid
    JSON — and BigQuery's ``JSON`` column parser rejects them ("syntax error while parsing value
    - invalid literal"), failing the whole Storage Write API append. A quantile dict on a runaway
    series (``log1p``'s ``expm1`` overflow) can carry such values; dropping the offending keys —
    parity with `_as_float`'s scalar NULL — keeps the row writable. An all-non-finite dict
    collapses to NULL.
    """
    if value is None or (isinstance(value, dict) and not value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        clean = {
            k: v for k, v in value.items() if not (isinstance(v, float) and not math.isfinite(v))
        }
        if not clean:
            return None
        return json.dumps(clean, sort_keys=True)
    return json.dumps(value, sort_keys=True)


def _as_date(value: Any) -> Any:
    """Normalize a timestamp-ish value to a ``date`` for BQ DATE columns."""
    if hasattr(value, "date"):
        return value.date()
    return value


def _as_int(value: Any) -> int | None:
    """Coerce to int, mapping missing/NaN to None — the INT64 counterpart of `_as_float`.

    A pandas column that ever held a NaN comes back as float, so ``int(rec[...])`` on a value that
    round-tripped through a frame is not safe on its own.
    """
    if value is None or value != value:  # None or NaN
        return None
    return int(value)
