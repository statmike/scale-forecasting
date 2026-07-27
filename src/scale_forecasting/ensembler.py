"""Consensus across base models — calculated + learned — pure logic + SQL builders.

The outer loop (DESIGN §5.2) combines every base model's forecast per ``ts_id`` into a
consensus. Two families, both **pure** here (no GCP calls — CONTRACTS §0, §6):

* **Calculated** (``mean``, ``median``, ``inverse_error``) — heuristics that need no training,
  so they work even when backtesting is off. These are emitted as **BigQuery SQL** by
  :func:`build_ensemble_sql`; BigQuery runs the SQL and writes each as a pseudo-model
  (``ensemble_mean`` …) into ``forecast_predictions``, comparable to the base models.
* **Learned** (``nnls``, ``ridge``, ``xgb``) — meta-learners that train on the backtest OOF to
  learn per-model trust weights. :func:`fit_learned` fits them; because it only ever sees
  ``backtest_oof`` (never in-sample fits) leakage is structurally impossible, and it refuses to
  run when backtesting is off.

A run may request several strategies at once (``ensemble.strategies`` is a list); each yields
its own weights/rows so calculated and learned consensuses sit side-by-side in one run.

Public surface: :func:`build_ensemble_sql`, :func:`fit_learned` (plus the pure combine
helpers :func:`mean_combine`, :func:`median_combine`, :func:`inverse_error_weights`).
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

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
    """Weights ∝ 1/error, normalized to sum to 1 (DESIGN §5.2).

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
    """Fit every learned strategy in ``cfg`` on the backtest OOF (CONTRACTS §6, DESIGN §5.2).

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


# --- SQL builders (calculated) -------------------------------------------------

# Column list shared by every ensemble INSERT into forecast_predictions.
_INSERT_COLS = (
    "  (run_id, ts_id, model_type, compute_engine, forecast_date,\n"
    "   yhat, yhat_lower, yhat_upper, quantiles)"
)


def _base_pred_cte(cfg: RunConfig, dataset: str) -> str:
    """``base_pred`` CTE: base-model prediction rows for this run, with optional pruning.

    Pruning (``ensemble.prune_threshold`` > 0) drops models whose backtest decision metric is
    worse than the threshold (via ``forecast_metadata``) so a bad base model can't drag the
    consensus down. Threshold 0.0 means no pruning.
    """
    models = ", ".join(f"'{m}'" for m in cfg.models)
    lines = [
        "base_pred AS (",
        "  SELECT run_id, ts_id, model_type, forecast_date, yhat, yhat_lower, yhat_upper",
        f"  FROM `{dataset}.forecast_predictions`",
        "  WHERE run_id = @run_id",
        f"    AND model_type IN ({models})",
    ]
    if cfg.ensemble.prune_threshold > 0.0:
        metric = cfg.backtest.decision_metric
        lines += [
            "    AND model_type NOT IN (",
            f"      SELECT model_type FROM `{dataset}.forecast_metadata`",
            f"      WHERE run_id = @run_id AND {metric} > {cfg.ensemble.prune_threshold}",
            "    )",
        ]
    lines.append(")")
    return "\n".join(lines)


def _mean_or_median_stmt(strategy: str, cfg: RunConfig, dataset: str) -> str:
    """A ``mean``/``median`` ensemble: aggregate base_pred, write an ``ensemble_<s>`` row set."""
    if strategy == "mean":
        yhat, lower, upper = "AVG(yhat)", "AVG(yhat_lower)", "AVG(yhat_upper)"
    else:  # median
        yhat = "APPROX_QUANTILES(yhat, 2)[OFFSET(1)]"
        lower = "APPROX_QUANTILES(yhat_lower, 2)[OFFSET(1)]"
        upper = "APPROX_QUANTILES(yhat_upper, 2)[OFFSET(1)]"
    # BigQuery: the WITH clause belongs to the query that follows INSERT INTO (cols),
    # so INSERT comes first, then WITH, then SELECT (not WITH ... INSERT).
    return (
        f"INSERT INTO `{dataset}.forecast_predictions`\n"
        f"{_INSERT_COLS}\n"
        f"WITH {_base_pred_cte(cfg, dataset)}\n"
        f"SELECT run_id, ts_id, 'ensemble_{strategy}' AS model_type,\n"
        "       'ensemble' AS compute_engine, forecast_date,\n"
        f"       {yhat} AS yhat, {lower} AS yhat_lower, {upper} AS yhat_upper,\n"
        "       NULL AS quantiles\n"
        "FROM base_pred\n"
        "GROUP BY run_id, ts_id, forecast_date;"
    )


def _inverse_error_stmt(cfg: RunConfig, dataset: str) -> str:
    """Inverse-error weighting: weight ∝ 1/decision_metric per (ts_id, model), normalized."""
    metric = cfg.backtest.decision_metric
    models = ", ".join(f"'{m}'" for m in cfg.models)
    return (
        f"INSERT INTO `{dataset}.forecast_predictions`\n"
        f"{_INSERT_COLS}\n"
        f"WITH {_base_pred_cte(cfg, dataset)},\n"
        "model_weight AS (\n"
        "  SELECT ts_id, model_type,\n"
        f"         SAFE_DIVIDE(1, NULLIF(AVG({metric}), 0)) AS w\n"
        f"  FROM `{dataset}.forecast_metadata`\n"
        f"  WHERE run_id = @run_id AND model_type IN ({models})\n"
        "  GROUP BY ts_id, model_type\n"
        ")\n"
        "SELECT p.run_id, p.ts_id, 'ensemble_inverse_error' AS model_type,\n"
        "       'ensemble' AS compute_engine, p.forecast_date,\n"
        "       SAFE_DIVIDE(SUM(p.yhat * w.w), SUM(w.w)) AS yhat,\n"
        "       SAFE_DIVIDE(SUM(p.yhat_lower * w.w), SUM(w.w)) AS yhat_lower,\n"
        "       SAFE_DIVIDE(SUM(p.yhat_upper * w.w), SUM(w.w)) AS yhat_upper,\n"
        "       NULL AS quantiles\n"
        "FROM base_pred p\n"
        "JOIN model_weight w USING (ts_id, model_type)\n"
        "WHERE w.w IS NOT NULL\n"
        "GROUP BY p.run_id, p.ts_id, p.forecast_date;"
    )


def build_ensemble_sql(cfg: RunConfig, dataset: str = "{dataset}") -> str:
    """Render the BigQuery SQL for the run's **calculated** ensembles (CONTRACTS §6).

    Emits one self-contained statement per requested calculated strategy, each writing an
    ``ensemble_<strategy>`` pseudo-model into ``forecast_predictions`` (comparable to the base
    models). Learned strategies are handled by :func:`fit_learned` — their weights aren't known
    until trained — so they're skipped here. Returns an empty string when no calculated
    strategy is requested.

    ``dataset`` defaults to a ``{dataset}`` template token so a config-only call renders; the
    engine substitutes the real ``project.dataset`` at run time.
    """
    calculated = [s for s in cfg.ensemble.strategies if s in CALCULATED_STRATEGIES]
    if not calculated:
        return ""

    parts: list[str] = []
    for strategy in calculated:
        if strategy == "inverse_error":
            parts.append(_inverse_error_stmt(cfg, dataset))
        else:
            parts.append(_mean_or_median_stmt(strategy, cfg, dataset))
    return "\n\n".join(parts)
