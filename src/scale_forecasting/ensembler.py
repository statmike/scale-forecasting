"""Consensus across base models — calculated + learned — pure logic, all in pandas.

The outer loop combines every base model's forecast per ``ts_id`` into a
consensus. Two families, both **pure** here (no GCP calls):

* **Calculated** (``mean``, ``median``, ``inverse_error``) — heuristics that need no training,
  so they work even when backtesting is off. `combine_calculated` blends the base
  ``forecast_predictions`` **in pandas** and returns ``ensemble_<s>`` prediction rows the caller
  appends via the Storage Write API — the same append path the learned strategies use (every
  append-only cell-table write goes through the Write API, no ``INSERT…SELECT`` DML).
* **Learned** (``nnls``, ``ridge``, ``xgb``) — meta-learners that train on the backtest OOF to
  learn per-model trust weights. `fit_learned` fits them; because it only ever sees
  ``backtest_oof`` (never in-sample fits) leakage is structurally impossible, and it refuses to
  run when backtesting is off.

A run may request several strategies at once (``ensemble.strategies`` is a list); each yields
its own weights/rows so calculated and learned consensuses sit side-by-side in one run.

Public surface: `combine_calculated`, `combine_oof`, `fit_learned` (plus the
pure combine helpers `mean_combine`, `median_combine`, `inverse_error_weights`).
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .config import CALCULATED_STRATEGIES, LEARNED_STRATEGIES
from .errors import ConfigError, get_logger

if TYPE_CHECKING:
    from .config import RunConfig

_log = get_logger(__name__)

_RIDGE_ALPHA = 1.0  # L2 strength for the ridge meta-learner


# --- pure combine helpers (calculated) -----------------------------------------


def mean_combine(yhats: np.ndarray) -> np.ndarray:
    """Row-wise mean across models. ``yhats`` is ``(n_models, horizon)``."""
    return np.asarray(yhats, dtype=float).mean(axis=0)


def median_combine(yhats: np.ndarray) -> np.ndarray:
    """Row-wise median across models — robust to a single wild base forecast."""
    return np.median(np.asarray(yhats, dtype=float), axis=0)


def inverse_error_weights(errors: np.ndarray) -> np.ndarray:
    """Weights ∝ 1/error, normalized to sum to 1.

    Lower-error models get more weight. A zero-error model would divide by zero, so zeros are
    treated as "perfectly trusted" and share the weight equally among themselves. Falls back to
    uniform weights when no error is finite and positive.
    """
    err = np.asarray(errors, dtype=float)
    if err.ndim != 1 or err.size == 0:
        raise ValueError("errors must be a non-empty 1-D array")

    zero = err == 0.0
    if zero.any():
        w = zero.astype(float)
        return w / w.sum()

    finite = np.isfinite(err) & (err > 0.0)
    if not finite.any():
        return np.full(err.size, 1.0 / err.size)

    inv = np.where(finite, 1.0 / err, 0.0)
    return inv / inv.sum()


# --- OOF-space consensus (scoring) ---------------------------------------------

# The columns of the OOF-blend frame combine_oof returns — the shape the scorer consumes.
_OOF_BLEND_COLS = ("ts_id", "model_type", "fold_id", "forecast_date", "y_true", "yhat")


def _weighted_blend(vals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Row-wise weighted mean over the *present* (non-NaN) models, weights renormalized per row.

    ``vals`` is ``(n_rows, n_models)``; ``weights`` is ``(n_models,)``. A row where no weighted
    model is present (all NaN, or the present weights sum to zero) yields NaN — the caller drops it.
    Same renormalize-over-present rule as `ensemble_run._apply_weights`, so the OOF-scored
    consensus applies the identical blend the future prediction does (for the shared weights).
    """
    present = ~np.isnan(vals)
    wrow = present * weights
    denom = wrow.sum(axis=1)
    num = np.nansum(np.where(present, vals, 0.0) * weights, axis=1)
    return np.where(denom > 0.0, num / np.where(denom > 0.0, denom, 1.0), np.nan)


def combine_oof(
    oof_df: pd.DataFrame,
    cfg: RunConfig,
    learned_weights: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Blend base-model OOF forecasts into ensemble OOF, per requested strategy (pure).

    The scoring counterpart to the future-prediction consensus: because the base
    ``forecast_predictions`` are a true beyond-data forecast (no actuals to join), each ensemble is
    scored on the **backtest OOF window** — exactly the window the base models are scored on. This
    applies the same consensus rules in OOF space (where ``y_true`` lives) and returns long-format
    ``(ts_id, model_type='ensemble_<s>', fold_id, forecast_date, y_true, yhat)`` for the caller to
    score with `metrics.compute_metrics`.

    Blends over whichever base models are present per ``(ts_id, fold_id, forecast_date)`` key:
    ``mean``/``median`` are unweighted; ``inverse_error`` weights each model per ``ts_id`` by
    ``1/WAPE`` over that model's OOF (self-contained — the same signal ``forecast_metadata`` would
    carry); learned strategies apply ``learned_weights[strategy]`` (skipped if absent). Bounds are
    not carried (OOF has none), so ensemble coverage/pinball are NaN — consistent with the Spark
    base models, whose fold metrics also omit intervals. Returns an empty frame when the OOF is
    empty or no strategy produces a blend.
    """
    learned_weights = learned_weights or {}
    empty = pd.DataFrame(columns=list(_OOF_BLEND_COLS))
    if oof_df.empty:
        return empty
    models = list(cfg.models)
    keys = ["ts_id", "fold_id", "forecast_date"]
    wide = oof_df.pivot_table(index=keys, columns="model_type", values="yhat", aggfunc="first")
    present_models = [m for m in models if m in wide.columns]
    if not present_models:
        return empty
    vals = wide.reindex(columns=present_models).to_numpy(dtype=float)
    truth = oof_df.drop_duplicates(keys).set_index(keys)["y_true"].reindex(wide.index).to_numpy()
    ts_ids = wide.index.get_level_values("ts_id").to_numpy()

    parts: list[pd.DataFrame] = []
    for strategy in cfg.ensemble.strategies:
        yhat = _blend_oof_strategy(strategy, vals, present_models, ts_ids, truth, learned_weights)
        if yhat is None:
            continue
        part = pd.DataFrame(
            {
                "ts_id": wide.index.get_level_values("ts_id"),
                "model_type": f"ensemble_{strategy}",
                "fold_id": wide.index.get_level_values("fold_id"),
                "forecast_date": wide.index.get_level_values("forecast_date"),
                "y_true": truth,
                "yhat": yhat,
            }
        )
        parts.append(part[~part["yhat"].isna()].reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)[list(_OOF_BLEND_COLS)] if parts else empty


def _blend_oof_strategy(
    strategy: str,
    vals: np.ndarray,
    models: list[str],
    ts_ids: np.ndarray,
    truth: np.ndarray,
    learned_weights: dict[str, dict[str, float]],
) -> np.ndarray | None:
    """The blended yhat for one strategy over the OOF value matrix, or ``None`` to skip it.

    ``vals`` is ``(n_rows, n_models)`` of base OOF yhats aligned to ``models``; ``ts_ids`` and
    ``truth`` are per-row. Skips (returns ``None``) a learned strategy with no fitted weights.
    """
    if strategy == "mean":
        return _weighted_blend(vals, np.ones(len(models)))
    if strategy == "median":
        return np.nanmedian(vals, axis=1)
    if strategy == "inverse_error":
        return _inverse_error_blend(vals, ts_ids, truth)
    wmap = learned_weights.get(strategy)
    if not wmap:  # learned strategy not fitted (e.g. backtest off) → nothing to score
        return None
    weights = np.array([wmap.get(m, 0.0) for m in models], dtype=float)
    return _weighted_blend(vals, weights)


def _inverse_error_blend(vals: np.ndarray, ts_ids: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Inverse-WAPE-weighted blend, weights computed **per ts_id** from the OOF itself.

    For each series, every base model's WAPE over its OOF rows sets its weight
    (`inverse_error_weights`); the blend then renormalizes over the models present per row.
    Self-contained — computed straight from the OOF rather than read back from
    ``forecast_metadata`` — so scoring needs no registry round-trip. (WAPE is the natural error for
    this weighting; it need not equal ``backtest.decision_metric``, which drives the *future*
    ``inverse_error`` blend — this is the scored counterpart, not a byte-for-byte replay.)
    """
    out = np.full(vals.shape[0], np.nan)
    for tid in np.unique(ts_ids):
        rows = ts_ids == tid
        block = vals[rows]
        denom = np.nansum(np.abs(truth[rows]))
        # Per-model WAPE over this series' OOF (NaN where the model never forecast the series).
        abs_err = np.abs(block - truth[rows][:, None])
        wape = np.nansum(abs_err, axis=0) / denom if denom > 0 else np.nansum(abs_err, axis=0)
        weights = inverse_error_weights(wape)
        out[rows] = _weighted_blend(block, weights)
    return out


# --- learned meta-learners -----------------------------------------------------


def _pivot_oof(oof_df: pd.DataFrame, models: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Reshape long OOF rows into ``(X, y)`` for stacking.

    ``X`` is ``(n_samples, n_models)`` of base-model OOF forecasts, ``y`` the aligned truth.
    Samples are ``(ts_id, fold_id, forecast_date)`` keys; rows missing any base model are
    dropped so every column is comparable. Column order follows ``models`` (stable weights).
    """
    date_col = "forecast_date" if "forecast_date" in oof_df.columns else "ds"
    wide = oof_df.pivot_table(
        index=["ts_id", "fold_id", date_col],
        columns="model_type",
        values="yhat",
        aggfunc="first",
    )
    missing = [m for m in models if m not in wide.columns]
    if missing:
        raise ConfigError(f"OOF is missing base models {missing}; cannot fit learned ensemble")
    wide = wide[models].dropna()

    truth = (
        oof_df.drop_duplicates(["ts_id", "fold_id", date_col])
        .set_index(["ts_id", "fold_id", date_col])["y_true"]
        .reindex(wide.index)
    )
    return wide.to_numpy(dtype=float), truth.to_numpy(dtype=float)


def _fit_nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares: weights ≥ 0 (no model enters with a negative weight)."""
    weights, _ = nnls(X, y)
    return np.asarray(weights, dtype=float)


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = _RIDGE_ALPHA) -> np.ndarray:
    """Closed-form ridge regression weights: ``(XᵀX + αI)⁻¹ Xᵀy`` (no intercept)."""
    xtx = X.T @ X
    reg = xtx + alpha * np.eye(xtx.shape[0])
    return np.asarray(np.linalg.solve(reg, X.T @ y), dtype=float)


def _fit_xgb(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, object]:
    """Light XGBoost meta-learner; "weights" are its normalized feature importances."""
    try:
        from xgboost import XGBRegressor
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ConfigError("xgb ensemble needs xgboost; install the 'models' extra") from e
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    model.fit(X, y)
    importances = np.asarray(model.feature_importances_, dtype=float)
    total = importances.sum()
    weights = importances / total if total > 0 else np.full(X.shape[1], 1.0 / X.shape[1])
    return weights, model


def fit_learned(
    oof_df: pd.DataFrame, cfg: RunConfig
) -> tuple[dict[str, dict[str, float]], dict[str, bytes]]:
    """Fit every learned strategy in ``cfg`` on the backtest OOF.

    Returns ``(weights, artifacts)``:
      * ``weights[strategy]`` maps base-model name → learned weight,
      * ``artifacts[strategy]`` is the pickled, uploadable meta-learner payload.

    Refuses to run when backtesting is off — a learned ensemble has no legitimate,
    leakage-free data to train on (the guard that keeps stacking honest). Base models come
    from ``cfg.models`` in order; native vs python doesn't matter — only their OOF forecasts do.
    """
    learned = [s for s in cfg.ensemble.strategies if s in LEARNED_STRATEGIES]
    if not learned:
        return {}, {}
    if not cfg.backtest.enabled:
        raise ConfigError(
            "learned ensemble strategies require backtest.enabled=true (they train on OOF)"
        )
    if oof_df.empty:
        raise ConfigError("cannot fit a learned ensemble: backtest_oof is empty")

    models = list(cfg.models)
    X, y = _pivot_oof(oof_df, models)
    if X.shape[0] == 0:
        raise ConfigError("no complete OOF rows across all base models to train on")

    weights: dict[str, dict[str, float]] = {}
    artifacts: dict[str, bytes] = {}
    for strategy in learned:
        if strategy == "nnls":
            w = _fit_nnls(X, y)
            payload: object = {"strategy": strategy, "models": models, "weights": w.tolist()}
        elif strategy == "ridge":
            w = _fit_ridge(X, y)
            payload = {"strategy": strategy, "models": models, "weights": w.tolist()}
        else:  # xgb
            w, payload = _fit_xgb(X, y, seed=0)
        weights[strategy] = {m: float(wi) for m, wi in zip(models, w, strict=True)}
        artifacts[strategy] = pickle.dumps(payload)
        _log.info("fit %s ensemble on %d OOF rows", strategy, X.shape[0])

    return weights, artifacts


# --- calculated ensembles in pandas (Write-API path) ---------------------------

# The prediction-row columns combine_calculated emits (ensemble_id + run_id filled by the caller).
_PRED_OUT_COLS = ("ts_id", "model_type", "forecast_date", "yhat", "yhat_lower", "yhat_upper")


def _pruned_models(cfg: RunConfig, metric_df: pd.DataFrame | None) -> list[str]:
    """The base models a calculated ensemble should blend, after optional pruning (pure).

    Pruning (``ensemble.prune_threshold`` > 0) drops any model whose mean backtest decision metric
    is worse than the threshold, so a bad base model can't drag the consensus down — equivalent to
    a ``model_type NOT IN (… metric > threshold)`` filter, but applied fleet-wide (a model pruned
    on its run-level metric is dropped from every series' blend). Threshold 0.0, or no metric
    frame, means no pruning. Order follows ``cfg.models`` (stable).
    """
    models = list(cfg.models)
    threshold = cfg.ensemble.prune_threshold
    if threshold <= 0.0 or metric_df is None or metric_df.empty:
        return models
    metric = cfg.backtest.decision_metric
    if metric not in metric_df.columns:
        return models
    mean_by_model = metric_df.groupby("model_type")[metric].mean()
    return [m for m in models if not (mean_by_model.get(m, float("nan")) > threshold)]


def _calc_blend(strategy: str, vals: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Blend an ``(n_rows, n_models)`` value matrix by one calculated strategy (pure).

    ``mean``/``median`` are unweighted (``median`` robust to a wild base forecast);
    ``inverse_error`` passes the per-model weights (∝ 1/decision_metric, renormalized over the
    present models per row). NaNs (a model absent for a row) are ignored — the same
    renormalize-over-present rule as the learned blend, so a date only some base models forecast
    still yields a value rather than NaN.
    """
    if strategy == "median":
        return np.nanmedian(vals, axis=1)
    w = np.ones(vals.shape[1]) if weights is None else weights
    return _weighted_blend(vals, w)


def combine_calculated(
    base_df: pd.DataFrame,
    cfg: RunConfig,
    metric_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Blend base predictions into ``ensemble_<s>`` rows for every calculated strategy (pure).

    Builds the ``ensemble_<s>`` rows in pandas rather than via SQL ``INSERT…SELECT``: reads the base
    ``forecast_predictions`` (long-format ``ts_id, model_type, forecast_date, yhat, yhat_lower,
    yhat_upper``) and returns prediction-row dicts the caller stamps with ``run_id``/``ensemble_id``
    and appends via the Storage Write API — the same path `ensemble_run._apply_weights` uses
    for the learned strategies. ``inverse_error`` weights each model by ``1/mean(decision_metric)``
    from ``metric_df`` (``forecast_metadata`` for this run); ``mean``/``median`` need no metrics.
    Optional pruning drops weak base models first (`_pruned_models`). Blends over whichever
    (pruned) base models are present per ``(ts_id, forecast_date)``, renormalizing weights over that
    present subset. Returns an empty list when no calculated strategy is requested or the base is
    empty. ``yhat``/bounds are floats or ``None`` (NaN → NULL); ``run_id``/``ensemble_id`` are added
    by the orchestrator so this stays a pure, config-only function.
    """
    calculated = [s for s in cfg.ensemble.strategies if s in CALCULATED_STRATEGIES]
    if not calculated or base_df.empty:
        return []
    models = _pruned_models(cfg, metric_df)
    present = [m for m in models if m in set(base_df["model_type"].unique())]
    if not present:
        return []

    # inverse_error weights: 1/mean(metric) per model, renormalized; uniform when no metric frame.
    ie_weights: np.ndarray | None = None
    if "inverse_error" in calculated:
        ie_weights = _inverse_error_run_weights(present, cfg, metric_df)

    # One wide frame per value column, aligned to the present base models, reused across strategies.
    keys = ["ts_id", "forecast_date"]
    wide = {
        col: base_df.pivot_table(index=keys, columns="model_type", values=col).reindex(
            columns=present
        )
        for col in ("yhat", "yhat_lower", "yhat_upper")
    }
    index = wide["yhat"].index

    rows: list[dict[str, Any]] = []
    for strategy in calculated:
        weights = ie_weights if strategy == "inverse_error" else None
        blended = {
            col: _calc_blend(strategy, wide[col].to_numpy(dtype=float), weights)
            for col in ("yhat", "yhat_lower", "yhat_upper")
        }
        model_type = f"ensemble_{strategy}"
        for i, (ts_id, forecast_date) in enumerate(index):
            yh = blended["yhat"][i]
            if np.isnan(yh):
                continue
            lo, up = blended["yhat_lower"][i], blended["yhat_upper"][i]
            rows.append(
                {
                    "ts_id": ts_id,
                    "model_type": model_type,
                    "forecast_date": forecast_date,
                    "yhat": float(yh),
                    "yhat_lower": None if np.isnan(lo) else float(lo),
                    "yhat_upper": None if np.isnan(up) else float(up),
                }
            )
    return rows


def _inverse_error_run_weights(
    models: list[str], cfg: RunConfig, metric_df: pd.DataFrame | None
) -> np.ndarray:
    """Per-model inverse-error weights from the run's ``forecast_metadata`` (pure).

    Weight ∝ ``1/mean(decision_metric)`` over the model's metadata rows, via
    `inverse_error_weights` (zeros dominate, non-finite → uniform). Falls back to uniform when
    no metric frame / column is available — degrading ``inverse_error`` to ``mean`` rather than
    failing (the ``SAFE_DIVIDE(1, NULLIF(AVG(metric), 0))`` NULL-tolerant behavior).
    """
    n = len(models)
    metric = cfg.backtest.decision_metric
    if metric_df is None or metric_df.empty or metric not in metric_df.columns:
        return np.full(n, 1.0 / n)
    mean_by_model = metric_df.groupby("model_type")[metric].mean()
    errors = np.array([mean_by_model.get(m, np.nan) for m in models], dtype=float)
    return inverse_error_weights(errors)
