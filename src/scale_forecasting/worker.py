"""The cell runner — the G1 unit of work that runs identically local / Spark / Ray.

``run_cell`` (BUILD step 2.6) fits, optionally backtests, and predicts ONE
``(ts_id, model)`` cell and returns a :class:`CellResult`. Engines differ only in how
they *call* it and *collect* its results — that symmetry is what makes "same code
everywhere" (G1) real.

:class:`CellResult` is defined here because it is the worker's output type; the registry
writers (``registry/bq.py``) consume it. It carries plain data (frames + scalars), no
behavior, so it is the clean seam between compute and lineage (CONTRACTS §3.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .backtest import backtest_cell
from .errors import get_logger
from .features import build_features, holiday_frame
from .metrics import METRIC_NAMES
from .models import get_model
from .models.base_model import PREDICTION_COLUMNS, BaseModel, ModelContext
from .registry.ids import make_model_hash, make_run_id

if TYPE_CHECKING:
    from .config import RunConfig

_log = get_logger(__name__)


@dataclass(frozen=True)
class CellResult:
    """The result of one ``(ts_id, model)`` cell (CONTRACTS §3.2).

    A failing cell sets ``status="error"`` with ``error`` populated and empty
    ``predictions`` — it never raises out of ``run_cell`` (CONTRACTS §3.3), so one bad
    cell can't sink a 100k-series batch.
    """

    run_id: str
    ts_id: str
    model_type: str
    compute_engine: str  # "spark" | "ray" | "bigquery"
    model_hash: str
    status: str  # "ok" | "error"
    error: str | None
    predictions: pd.DataFrame  # canonical §2.1
    oof: pd.DataFrame | None  # canonical §2.2, or None if backtest off
    metrics: dict[str, float]  # §2.3 (full-fit metrics)
    best_params: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0
    # Serialized fitted model (from BaseModel.serialize), or None when persistence is off / the
    # model opts out. Carried as bytes rather than a temp-file path so it crosses the executor
    # boundary as plain data with no local-fs lifecycle; the registry writer uploads it to GCS and
    # stamps the ObjectRef onto forecast_metadata.model_artifact (CONTRACTS §3.4, G3).
    artifact_bytes: bytes | None = None


def _compute_engine(model_cls: type[BaseModel], cfg: RunConfig) -> str:
    """The engine that will execute this cell: the Python runtime, or BigQuery for native
    models (which run as SQL regardless of the run's Python runtime)."""
    return "bigquery" if model_cls.runtime == "bigquery" else cfg.python_runtime


def _model_context(cfg: RunConfig) -> ModelContext:
    """Build the per-cell :class:`ModelContext` from the run config (CONTRACTS §1)."""
    holidays = holiday_frame(cfg) if cfg.features.holidays else None
    return ModelContext(
        freq=cfg.data.freq,
        horizon=cfg.data.horizon,
        seed=0,
        holidays=holidays,
        transform=cfg.features.transform,
    )


def _rollup_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average the per-fold metric panels into one panel (CONTRACTS §3.2).

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
    """An empty canonical §2.1 prediction frame (for error cells)."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in PREDICTION_COLUMNS})


def run_cell(series: pd.DataFrame, model_name: str, cfg: RunConfig) -> CellResult:
    """Fit + (optional backtest) + predict ONE ``(ts_id, model)`` cell (CONTRACTS §3.1).

    Pure-ish and deterministic: reads nothing global, writes nothing (no BQ), returns a
    :class:`CellResult` carrying plain data. A failing cell returns ``status="error"`` and
    never raises (CONTRACTS §3.3), so one bad series can't sink a batch.
    """
    ts_id = _ts_id(series, cfg)
    run_id = make_run_id(cfg)
    model_hash = make_model_hash(run_id, ts_id, model_name, cfg)

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
        )

    try:
        model_cls = get_model(model_name)
    except Exception as e:  # unknown model name → error cell, engine unknown
        return _error(repr(e), cfg.python_runtime)

    engine = _compute_engine(model_cls, cfg)
    started = time.perf_counter()
    try:
        ctx = _model_context(cfg)

        # Optional backtest first (fresh model per fold) → OOF frame + rolled-up metrics.
        oof: pd.DataFrame | None = None
        metrics = {name: float("nan") for name in METRIC_NAMES}
        if cfg.backtest.enabled:
            oof, fold_metrics = backtest_cell(series, lambda: model_cls({}, ctx), cfg)
            metrics = _rollup_metrics(fold_metrics)

        # Final fit on the full history, then forecast the horizon.
        y, X = build_features(series, cfg)
        model = model_cls({}, ctx)
        model.fit(y, X)
        # Offline has no *true* future exog (that arrives with a real run, Arc B). As a
        # stand-in we hand exog-aware models the first `horizon` rows of the design matrix
        # so shapes line up; values are historical, so exog-driven forecasts are indicative
        # only. Tree models ignore any lag_* columns here (see _lag_forecaster).
        future_exog = X.iloc[: cfg.data.horizon] if X is not None else None
        predictions = model.predict(cfg.data.horizon, future_exog)

        # Persist the fitted model as an artifact only when the run opts in (G3 lineage). A
        # serialize failure must not sink an otherwise-good forecast, so it degrades to no
        # artifact (CONTRACTS §3.3) rather than turning the cell into an error.
        artifact_bytes: bytes | None = None
        if cfg.compute.persist_models:
            try:
                artifact_bytes = model.serialize()
            except Exception as e:  # noqa: BLE001 - persistence is best-effort, never fatal
                _log.warning("serialize failed for %s/%s: %r", ts_id, model_name, e)

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
            fit_seconds=time.perf_counter() - started,
            artifact_bytes=artifact_bytes,
        )
    except Exception as e:  # any failure → error cell, batch survives (CONTRACTS §3.3)
        return _error(repr(e), engine)


def _ts_id(series: pd.DataFrame, cfg: RunConfig) -> str:
    """The series id for this cell: the ``ts_id_col`` value, or ``"unknown"`` if absent."""
    col = cfg.data.ts_id_col
    if col in series.columns and len(series):
        return str(series[col].iloc[0])
    return "unknown"
