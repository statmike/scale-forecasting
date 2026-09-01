"""The cell runner — the unit of work that runs identically local / Spark / Ray.

``run_cell`` fits, optionally backtests, and predicts ONE
``(ts_id, model)`` cell and returns a `CellResult`. Engines differ only in how
they *call* it and *collect* its results — that symmetry is what makes "same code
everywhere" real.

`CellResult` is defined here because it is the worker's output type; the registry
writers (``registry/``) consume it. It carries plain data (frames + scalars), no
behavior, so it is the clean seam between compute and lineage.
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .backtest import backtest_cell
from .errors import ConfigError, get_logger
from .features import (
    build_features,
    build_future_features,
    fit_transform_lambda,
    holiday_frame,
)
from .metrics import METRIC_NAMES
from .models import get_model
from .models.base_model import PREDICTION_COLUMNS, BaseModel, ModelContext
from .registry.ids import make_model_hash, make_run_id

if TYPE_CHECKING:
    from .config import RunConfig

_log = get_logger(__name__)


@dataclass(frozen=True)
class CellResult:
    """The result of one ``(ts_id, model)`` cell.

    A failing cell sets ``status="error"`` with ``error`` populated and empty
    ``predictions`` — it never raises out of ``run_cell``, so one bad
    cell can't sink a 100k-series batch.
    """

    run_id: str
    ts_id: str
    model_type: str
    compute_engine: str  # "spark" | "ray" | "bigquery"
    model_hash: str
    status: str  # "ok" | "error"
    error: str | None
    predictions: pd.DataFrame  # canonical prediction frame
    oof: pd.DataFrame | None  # canonical OOF frame, or None if backtest off
    metrics: dict[str, float]  # full-fit metrics
    best_params: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0
    # Per-cell wall-clock bracket + worker identity for the run trace (SDK trace()). fit_seconds is
    # the precise (monotonic) fit duration; these are absolute wall-clock stamps that position the
    # cell on a Gantt/waterfall lane, and worker_id (hostname:pid) attributes it to a worker.
    worker_id: str | None = None
    cell_started_at: datetime | None = None
    cell_ended_at: datetime | None = None
    # Serialized fitted model (from BaseModel.serialize), or None when persistence is off / the
    # model opts out. Carried as bytes rather than a temp-file path so it crosses the executor
    # boundary as plain data with no local-fs lifecycle; the registry writer uploads it to GCS and
    # stamps the ObjectRef onto forecast_metadata.model_artifact for model-artifact lineage.
    artifact_bytes: bytes | None = None
    # --- harvested compute measurement (compute.profile.measure) -------------------------------
    # What this cell cost, recorded so a completed run can size a later one. All None/0 when
    # measurement is off, which is also how a row written before these columns existed reads.
    # `fit_seconds` above is the wall-clock half of the same measurement, so it is not repeated.
    cpu_seconds: float | None = None  # time.process_time delta — sums across threads
    # The worker process's ABSOLUTE RSS high-water, not this cell's increment. Deliberate: a slot
    # must hold the interpreter, the libraries and the fit together, and the increment swings 17x
    # on the order cells happened to run in (see `profiling.measure.MeasuredFit`). Monotone within
    # a worker, so MAX across a family's cells is exactly the slot size that family needs.
    process_rss_bytes: int | None = None
    peak_gpu_bytes: int | None = None  # torch.cuda high-water; None == NOT MEASURED, never zero
    # The native-thread cap in force while this cell ran (OMP_NUM_THREADS). Without it
    # cpu_seconds/fit_seconds is uninterpretable: under a cap the ratio reports the cap back.
    intraop_threads: int | None = None
    n_obs: int | None = None  # rows fed to the fit — the data signature a later run matches on


def _worker_id() -> str:
    """A runtime-agnostic worker identity (``hostname:pid``).

    The same call on the driver, a Spark executor, or a Ray worker, so the trace attributes each
    cell to the physical worker that ran it without any engine-specific API — keeping ``run_cell``
    identical everywhere.
    """
    return f"{socket.gethostname()}:{os.getpid()}"


def _intraop_threads() -> int | None:
    """The native-thread cap this process is running under, or None when nothing caps it.

    Read from the environment rather than inferred, because that is where the fleet actually
    sets it: `resources` exports ``OMP_NUM_THREADS`` (and its four siblings) to
    ``spark.task.cpus`` on every Spark job, and Ray exports it to a task's ``num_cpus``. A cell
    that records the cap it ran under is a cell whose `effective_cores` can be read honestly
    later; one that does not is a number that silently repeats the pin back to you.
    """
    raw = os.environ.get("OMP_NUM_THREADS")
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _process_rss_bytes() -> int | None:
    """This worker process's absolute RSS high-water in bytes, or None where unmeasurable.

    Delegates to the probe `profiling` already owns, imported lazily because `profiling` imports
    *this* module (``measure_fit`` drives ``run_cell``) — a module-level import would be a cycle.
    The lazy call is a ``sys.modules`` hit after the first cell.
    """
    from .profiling.measure import _rss_bytes

    return _rss_bytes()


_gpu_probe_useful: bool | None = None  # None = not yet asked; False = no accelerator here


def _peak_gpu_bytes() -> int | None:
    """Peak CUDA bytes this process has allocated, or None when NOT MEASURED.

    ``None`` on every no-accelerator path, never ``0`` — a consumer that read a missing device
    as zero would compute a minimum GPU fraction and pack ten tasks onto a device that fits two.

    The "is there a GPU" half of the answer is cached per process because the probe's cheap case
    is not cheap at cell scale: a *failed* ``import torch`` is not memoized in ``sys.modules``, so
    on a CPU-only Spark worker every one of a hundred thousand cells would re-walk ``sys.path``.
    Neither torch's presence nor a device's appears mid-process, so one ask settles it; only the
    high-water *value* is re-read, and only where a device actually exists.
    """
    global _gpu_probe_useful
    if _gpu_probe_useful is False:
        return None
    from .profiling.measure import _peak_gpu_bytes as probe

    peak = probe()
    _gpu_probe_useful = peak is not None
    return peak


def _compute_engine(model_cls: type[BaseModel], cfg: RunConfig) -> str:
    """The engine that will execute this cell: the Python runtime, or BigQuery for native
    models (which run as SQL regardless of the run's Python runtime)."""
    return "bigquery" if model_cls.runtime == "bigquery" else cfg.python_runtime


def _model_context(cfg: RunConfig, transform_lambda: float | None = None) -> ModelContext:
    """Build the per-cell `ModelContext` from the run config.

    ``transform_lambda`` is the cell's fitted Box-Cox λ (None for stateless transforms), fit
    once in `run_cell` and shared by the backtest folds and the final fit.
    """
    holidays = holiday_frame(cfg) if cfg.features.holidays else None
    return ModelContext(
        freq=cfg.data.freq,
        horizon=cfg.data.horizon,
        seed=0,
        holidays=holidays,
        transform=cfg.features.transform,
        transform_lambda=transform_lambda,
    )


def _resolve_params(
    series: pd.DataFrame,
    model_name: str,
    cfg: RunConfig,
    ctx: ModelContext,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the hyperparameters this cell builds its model with (see `run_cell`).

    Pre-resolved ``params`` (the fleetwide driver pre-pass) win outright. Otherwise, per-series HPO
    tunes on *this* series when enabled at that granularity; failing that, the ``{}`` default. Kept
    tiny and separate so the resolution policy is one readable place and the HPO import stays lazy
    (Optuna loads only when a run actually tunes).
    """
    if params is not None:
        return params
    if cfg.hpo.enabled and cfg.hpo.granularity == "per_series":
        from .hpo import tune_model

        return tune_model(model_name, [series], cfg, ctx)
    return {}


def _rollup_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average the per-fold metric panels into one panel.

    NaNs are ignored per metric (a metric undefined on one fold shouldn't sink the mean);
    a metric NaN on every fold stays NaN. Always returns the full panel.
    """
    out: dict[str, float] = {}
    for name in METRIC_NAMES:
        vals = np.array([fm.get(name, np.nan) for fm in fold_metrics], dtype=float)
        finite = vals[~np.isnan(vals)]
        out[name] = float(finite.mean()) if finite.size else float("nan")
    return out


def _empty_predictions() -> pd.DataFrame:
    """An empty canonical prediction frame (for error cells)."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in PREDICTION_COLUMNS})


def run_cell(
    series: pd.DataFrame,
    model_name: str,
    cfg: RunConfig,
    params: dict[str, Any] | None = None,
) -> CellResult:
    """Fit + (optional backtest) + predict ONE ``(ts_id, model)`` cell.

    Pure-ish and deterministic: reads nothing global, writes nothing (no BQ), returns a
    `CellResult` carrying plain data. A failing cell returns ``status="error"`` and
    never raises, so one bad series can't sink a batch.

    ``params`` are the hyperparameters this cell's model is built with. Resolution order:

    * ``params`` given (not None) → use them. This is the **fleetwide** path: the driver tuned the
      model once on a sample (`resolve_fleetwide`) and threads the
      winning params here — never through ``cfg`` (the config is the run_id identity key).
    * else if ``cfg.hpo.enabled`` and ``granularity == "per_series"`` → tune on *this* series now.
    * else → ``{}`` (the default: today's untuned behavior).

    The resolved params drive **both** the backtest folds and the final fit, so
    ``best_params = model.get_params()`` reflects what actually ran.
    """
    ts_id = _ts_id(series, cfg)
    run_id = make_run_id(cfg)
    model_hash = make_model_hash(run_id, ts_id, model_name, cfg)
    # Wall-clock lane + worker identity for the trace, captured for every return path (ok or error).
    cell_started_at = datetime.now(UTC)
    worker_id = _worker_id()

    def _error(msg: str, engine: str) -> CellResult:
        return CellResult(
            run_id=run_id,
            ts_id=ts_id,
            model_type=model_name,
            compute_engine=engine,
            model_hash=model_hash,
            status="error",
            error=msg,
            predictions=_empty_predictions(),
            oof=None,
            metrics={name: float("nan") for name in METRIC_NAMES},
            worker_id=worker_id,
            cell_started_at=cell_started_at,
            cell_ended_at=datetime.now(UTC),
        )

    try:
        model_cls = get_model(model_name)
    except Exception as e:  # unknown model name → error cell, engine unknown
        return _error(repr(e), cfg.python_runtime)

    engine = _compute_engine(model_cls, cfg)
    # Harvest: record what this fit costs so a later run can be sized from it. Three cheap probes
    # around work the run was doing anyway — no sample, no pre-pass. Neither RSS nor the CUDA
    # high-water is reset first: the absolute peak is the number that sizes a slot (see
    # `CellResult.process_rss_bytes`), and resetting would also perturb whatever else shares this
    # worker. `intraop_threads` is captured before the fit because that is when it is in force.
    measuring = cfg.compute.profile.records_measurements
    intraop_threads = _intraop_threads() if measuring else None
    cpu_started = time.process_time()
    started = time.perf_counter()
    try:
        # Fit the transform's stateful λ once per cell (None for none/log1p), on the raw target.
        # It lives on ctx so the backtest folds and the final fit share one λ — never refit at
        # predict (the whole point of carrying it on the cell).
        lam = fit_transform_lambda(_target(series, cfg), cfg.features.transform)
        ctx = _model_context(cfg, transform_lambda=lam)
        resolved = _resolve_params(series, model_name, cfg, ctx, params)

        # Optional backtest first (fresh model per fold) → OOF frame + rolled-up metrics.
        oof: pd.DataFrame | None = None
        metrics = {name: float("nan") for name in METRIC_NAMES}
        if cfg.backtest.enabled:
            oof, fold_metrics = backtest_cell(series, lambda: model_cls(resolved, ctx), cfg, lam)
            metrics = _rollup_metrics(fold_metrics)

        # Final fit on the full history, then forecast the horizon.
        y, X = build_features(series, cfg, lam)
        model = model_cls(resolved, ctx)
        model.fit(y, X)
        # The design frame for the horizon, indexed by the *future* dates: holiday flags and
        # Fourier phase are recomputed there (exact — they are functions of the date), the
        # level-shift step is carried forward, and only user-supplied exog falls back to a
        # recency stand-in because it is genuinely unknown. Tree models ignore any lag_*
        # columns here (see _lag_forecaster).
        future_exog = build_future_features(y, X, cfg)
        predictions = model.predict(cfg.data.horizon, future_exog)

        # Persist the fitted model as an artifact only when the run opts in (model-artifact
        # lineage). A serialize failure must not sink an otherwise-good forecast, so it degrades to
        # no artifact rather than turning the cell into an error.
        artifact_bytes: bytes | None = None
        if cfg.compute.persist_models:
            try:
                artifact_bytes = model.serialize()
            except Exception as e:  # noqa: BLE001 - persistence is best-effort, never fatal
                _log.warning("serialize failed for %s/%s: %r", ts_id, model_name, e)

        fit_seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        return CellResult(
            run_id=run_id,
            ts_id=ts_id,
            model_type=model_name,
            compute_engine=engine,
            model_hash=model_hash,
            status="ok",
            error=None,
            predictions=predictions,
            oof=oof,
            metrics=metrics,
            best_params=model.get_params(),
            fit_seconds=fit_seconds,
            worker_id=worker_id,
            cell_started_at=cell_started_at,
            cell_ended_at=datetime.now(UTC),
            artifact_bytes=artifact_bytes,
            cpu_seconds=cpu_seconds if measuring else None,
            process_rss_bytes=_process_rss_bytes() if measuring else None,
            peak_gpu_bytes=_peak_gpu_bytes() if measuring else None,
            intraop_threads=intraop_threads,
            n_obs=len(series) if measuring else None,
        )
    except Exception as e:  # any failure → error cell, batch survives
        return _error(repr(e), engine)


def _ts_id(series: pd.DataFrame, cfg: RunConfig) -> str:
    """The series id for this cell: the ``ts_id_col`` value, or ``"unknown"`` if absent."""
    col = cfg.data.ts_id_col
    if col in series.columns and len(series):
        return str(series[col].iloc[0])
    return "unknown"


def _target(series: pd.DataFrame, cfg: RunConfig) -> pd.Series:
    """The raw target column as a float Series, for fitting the transform λ (before features).

    Raises ``ConfigError`` if the target column is absent — the same failure ``build_features``
    would raise a moment later, surfaced here so the λ fit names it.
    """
    col = cfg.data.target_col
    if col not in series:
        raise ConfigError(
            f"series missing required target column '{col}'; has {list(series.columns)}"
        )
    return series[col].astype(float)
