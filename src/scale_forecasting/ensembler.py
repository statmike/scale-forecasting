"""Consensus across base models — calculated + learned — pure logic, all in pandas.

The outer loop combines every base model's forecast per ``ts_id`` into a
consensus. Two families, both **pure** here (no GCP calls):

* **Calculated** (``mean``, ``median``, ``inverse_error``, ``champion``, ``horizon_specialist``) —
  heuristics and per-series selectors that need no cross-series meta-learner training.
  `combine_calculated` blends the base ``forecast_predictions`` **in pandas** and returns
  ``ensemble_<s>`` prediction rows the caller appends via the Storage Write API — the same append
  path the learned strategies use (every append-only cell-table write goes through the Write API,
  no ``INSERT…SELECT`` DML).
* **Learned** (``nnls``, ``ridge``, ``xgb``) — meta-learners that train on the backtest OOF to
  learn per-model trust weights. `fit_learned` fits them; because it only ever sees
  ``backtest_oof`` (never in-sample fits) leakage is structurally impossible, and it refuses to
  run when backtesting is off.

A run may request several strategies at once (``ensemble.strategies`` is a list); each yields
its own weights/rows so calculated and learned consensuses sit side-by-side in one run.

Public surface: `combine_calculated`, `combine_oof`, `fit_learned`,
`calibrate_ensemble_predictions`, `ensemble_oof_bounds` (plus the pure combine helpers
`mean_combine`, `median_combine`, `inverse_error_weights`).
"""

from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .backtest import holdout_fold_id
from .calibration import (
    apply_output_constraints,
    apply_quantile_and_cost_rules,
    calibrate_from_oof,
    has_output_constraints,
    rearrange_monotone_quantiles,
)
from .config import CALCULATED_STRATEGIES, LEARNED_STRATEGIES
from .errors import ConfigError, get_logger
from .metrics import loss_of
from .models import base_model_name

if TYPE_CHECKING:
    from .config import RunConfig

_log = get_logger(__name__)

_RIDGE_ALPHA = 1.0  # L2 strength for the ridge meta-learner

# Stand-ins for "the inner-fold fit was not attempted", so the fallback below is one
# `X.shape[0] == 0` check rather than two branches that have to stay in step.
_EMPTY_X = np.empty((0, 0))
_EMPTY_Y = np.empty(0)

_BASELINE_MODELS: frozenset[str] = frozenset(
    {
        "naive",
        "naive_last",
        "naive_mean",
        "naive_seasonal",
        "naive_drift",
        "naive_moving_average",
        "snaive",
        "moving_average",
        "croston",
        "sba",
        "tsb",
        "adida",
        "imapa",
    }
)
_STATISTICAL_MODELS: frozenset[str] = frozenset(
    {
        "theta",
        "auto_theta",
        "holtwinters",
        "ets",
        "autoets",
        "auto_arima",
        "auto_ces",
        "sarimax",
        "stl_bagging",
        "tbats",
        "ces",
        "mstl",
        "fft",
        "kalman",
        "ucm",
        "arima_plus",
    }
)
_ML_MODELS: frozenset[str] = frozenset({"ridge_lags", "xgboost", "lightgbm", "catboost", "prophet"})


def _model_simplicity_rank(model_name: str, order_idx: int) -> tuple[int, int]:
    """Return ``(complexity_tier, config_order_idx)`` for Occam's-razor parsimony tie-breaking."""
    root = base_model_name(model_name)
    base = root.split("_")[0] if root not in _BASELINE_MODELS else root
    if root in _BASELINE_MODELS or base in _BASELINE_MODELS:
        return (0, order_idx)
    if root in _STATISTICAL_MODELS:
        return (1, order_idx)
    if root in _ML_MODELS:
        return (2, order_idx)
    return (3, order_idx)


def filter_variants_for_ensemble(
    models: list[str],
    cfg: RunConfig,
    *,
    oof_df: pd.DataFrame | None = None,
    metric_df: pd.DataFrame | None = None,
) -> list[str]:
    """Select the ensemble candidate pool across `<base_model>@<tag>` variants (pure).

    Controlled by ``cfg.ensemble.variant_policy``:
    - ``"all"``: keep all configured model variants.
    - ``"primary_only"``: keep at most one variant per ``base_model_name(m)`` — preferring the
      untagged base model if present, else the first variant of that base model in config order.
    - ``"best_per_base_model"`` (default): when multiple variants of the same ``base_model`` exist,
      keep the variant with the lowest inner-fold OOF loss (or mean ``decision_metric`` loss in
      ``metric_df`` when ``oof_df`` is absent), preventing collinear multi-variant dilution.
    """
    policy = cfg.ensemble.variant_policy
    if policy == "all" or len(models) <= 1:
        return list(models)

    groups: dict[str, list[str]] = {}
    for m in models:
        groups.setdefault(base_model_name(m), []).append(m)
    if all(len(g) == 1 for g in groups.values()):
        return list(models)

    if policy == "primary_only":
        chosen_primary = {(base if base in g else g[0]) for base, g in groups.items()}
        return [m for m in models if m in chosen_primary]

    # policy == "best_per_base_model"
    losses: dict[str, float] = {}
    metric = cfg.backtest.decision_metric
    if (
        oof_df is not None
        and not oof_df.empty
        and {"model_type", "y_true", "yhat"}.issubset(oof_df.columns)
    ):
        inner, _ = inner_fold_mask(oof_df, cfg)
        ref_oof = oof_df[inner] if inner.any() else oof_df
        for m in models:
            sub = ref_oof[ref_oof["model_type"] == m]
            if sub.empty:
                losses[m] = float("nan")
                continue
            yt = sub["y_true"].to_numpy(dtype=float)
            yh = sub["yhat"].to_numpy(dtype=float)
            valid = np.isfinite(yt) & np.isfinite(yh)
            if not valid.any():
                losses[m] = float("nan")
                continue
            denom = float(np.sum(np.abs(yt[valid])))
            err = float(np.sum(np.abs(yh[valid] - yt[valid])))
            losses[m] = err / denom if denom > 0.0 else err
    elif metric_df is not None and not metric_df.empty and metric in metric_df.columns:
        mean_by_model = metric_df.groupby("model_type")[metric].mean()
        for m in models:
            val = float(mean_by_model.get(m, float("nan")))
            losses[m] = loss_of(metric, val)

    order_map = {m: idx for idx, m in enumerate(models)}
    chosen_best: set[str] = set()
    for base, g in groups.items():
        if len(g) == 1:
            chosen_best.add(g[0])
            continue
        finite_candidates = [m for m in g if m in losses and np.isfinite(losses[m])]
        if not finite_candidates:
            chosen_best.add(base if base in g else g[0])
        else:
            winner = min(
                finite_candidates,
                key=lambda m: (
                    losses[m],
                    0 if "@" in m and "univariate" in m else (0 if "cov_none" in m else 1),
                    order_map[m],
                ),
            )
            chosen_best.add(winner)
    return [m for m in models if m in chosen_best]


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

    ``errors`` must be **losses** — non-negative, zero is perfect — not raw metric values. That is
    what `metrics.loss_of` produces, and every caller here routes through it. Handing this function
    a metric where larger is better inverts the weighting silently, which is exactly what used to
    happen with ``decision_metric="coverage"``.
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


def _select_or_weight_models(
    errors: np.ndarray,
    models: list[str],
    *,
    top_k: int | None = None,
    parsimony_margin: float = 0.0,
    champion_only: bool = False,
) -> np.ndarray:
    """Compute per-model weights from losses ``errors`` under ``champion``, ``top_k``, and
    ``parsimony_margin`` rules (pure).
    """
    err = np.asarray(errors, dtype=float)
    n = err.size
    if n == 0:
        return np.empty(0, dtype=float)
    finite_idx = [i for i in range(n) if np.isfinite(err[i]) and err[i] >= 0.0]
    if not finite_idx:
        if champion_only:
            best_i = min(range(n), key=lambda i: _model_simplicity_rank(models[i], i))
            w = np.zeros(n, dtype=float)
            w[best_i] = 1.0
            return w
        return np.full(n, 1.0 / n, dtype=float)

    min_err = min(float(err[i]) for i in finite_idx)
    tol = min_err * (1.0 + parsimony_margin) if min_err > 0.0 else parsimony_margin

    if champion_only:
        if parsimony_margin > 0.0:
            elig = [i for i in finite_idx if float(err[i]) <= tol + 1e-12]
        else:
            elig = [i for i in finite_idx if abs(float(err[i]) - min_err) <= 1e-12]
        winner = min(elig, key=lambda i: _model_simplicity_rank(models[i], i))
        w = np.zeros(n, dtype=float)
        w[winner] = 1.0
        return w

    filtered = err.copy()
    if parsimony_margin > 0.0:
        # Drop any higher-complexity model that fails to beat the best simpler model by at least
        # `parsimony_margin`.
        for i in finite_idx:
            rank_i = _model_simplicity_rank(models[i], i)[0]
            simpler_losses = [
                float(err[j])
                for j in finite_idx
                if _model_simplicity_rank(models[j], j)[0] < rank_i
            ]
            if simpler_losses:
                best_simple = min(simpler_losses)
                req = best_simple / (1.0 + parsimony_margin) if best_simple > 0.0 else 0.0
                if float(err[i]) > req + 1e-12:
                    filtered[i] = np.nan

    surviving = [i for i in range(n) if np.isfinite(filtered[i]) and filtered[i] >= 0.0]
    if not surviving:
        surviving = finite_idx
        filtered = err.copy()

    if top_k is not None and top_k < len(surviving):
        ranked = sorted(
            surviving,
            key=lambda i: (float(filtered[i]), _model_simplicity_rank(models[i], i)),
        )
        keep_set = set(ranked[:top_k])
        for i in range(n):
            if i not in keep_set:
                filtered[i] = np.nan

    return inverse_error_weights(filtered)


# --- OOF-space consensus (scoring) ---------------------------------------------

# The columns of the OOF-blend frame combine_oof returns — the shape the scorer consumes.
_OOF_BLEND_COLS = (
    "ts_id",
    "model_type",
    "fold_id",
    # Carried, not just joined on: the scorer needs it to cut each series' history at the fold's
    # last training date before handing it to `compute_metrics` as the MASE/RMSSE denominator. An
    # ensemble scored against the full history — the window it is being judged on included — is not
    # comparable to the base models it sits beside on the leaderboard.
    "cutoff_date",
    "forecast_date",
    # Carried for the same reason `fold_id` is: the blended rows are written into the same
    # `backtest_oof` table as the base rows, and a per-horizon read (`reads.read_coverage_by_step`)
    # that finds this NULL on the ensembles can only report on the base models. NaN when the base
    # OOF did not carry it — an older run, or an engine that never filled it.
    "horizon_step",
    "y_true",
    "yhat",
)


def _fold_key(oof_df: pd.DataFrame) -> list[str]:
    """The columns that identify one fold's one forecast date, for joining models to each other.

    `cutoff_date` — the fold's last training date — rather than `fold_id`, because `fold_id` is an
    ordinal within one series' own plan and the two engines number from different anchors. The
    Python path anchors each series on its own last observation; the BigQuery-native path anchors
    every series on a single global ``MAX(ds)``. On a panel where series end on different dates
    those disagree, and the disagreement is silent: rows pair up on a matching ordinal that stands
    for different training windows, and the rows that *should* have paired fall out of the join.
    Two models blend only when they saw the same history and forecast the same date, which is what
    the cutoff says and the ordinal does not.

    Falls back to `fold_id` when `cutoff_date` is absent or not fully populated, which is the state
    of any OOF written before it was projected on both engines. Blending on the ordinal is what the
    product did for its whole history — the fallback is that behaviour, not a new risk — and it
    beats dropping every row of an older run on the floor.
    """
    date_col = "forecast_date" if "forecast_date" in oof_df.columns else "ds"
    if "cutoff_date" in oof_df.columns and oof_df["cutoff_date"].notna().all():
        return ["ts_id", "cutoff_date", date_col]
    return ["ts_id", "fold_id", date_col]


def inner_fold_mask(oof_df: pd.DataFrame, cfg: RunConfig) -> tuple[np.ndarray, str]:
    """Which OOF rows a fit is allowed to see, and what that says about the resulting numbers.

    Returns ``(mask, basis)``. ``mask`` is True on the rows belonging to a fold other than
    `backtest.holdout_fold_id` — the folds a stacker, an ``inverse_error`` weight or a
    hyperparameter search may learn from. ``basis`` is ``"holdout"`` when a real split exists and
    ``"in_sample"`` when it does not, in which case the mask is all-True and the caller fits on
    everything it will later be scored on.

    Two ways the split can fail to exist, and both are ordinary rather than exceptional: the run
    asked for ``n_folds: 1``, or a short series in a ragged panel achieved only the newest fold.
    Falling back is the right behaviour — a stacker with no rows is worse than an optimistic one --
    but it has to be *said*, because an in-sample number is indistinguishable from an honest one
    once it is sitting in a metric column. That is what ``basis`` is written to the registry for.
    """
    holdout = holdout_fold_id(cfg)
    if "fold_id" not in oof_df.columns:
        return np.ones(len(oof_df), dtype=bool), "in_sample"
    mask = (oof_df["fold_id"] != holdout).to_numpy(dtype=bool)
    if not mask.any():
        return np.ones(len(oof_df), dtype=bool), "in_sample"
    return mask, "holdout"


def _carried(aligned: pd.DataFrame, keys: list[str], name: str, dtype: str) -> pd.Series:
    """A column to write on each ensemble row, read from wherever the join key left it.

    `_fold_key` decides what the ensemble joins on; ``fold_id`` and ``cutoff_date`` are both written
    out either way. ``fold_id`` is the ordinal a reader recognises and the ensemble rows sit in the
    same table as the base rows; ``cutoff_date`` is what the scorer cuts history at. Whichever one
    the key used *is* an index level, and the other is still an ordinary column. Reading either from
    the wrong side is how it silently comes out all-NaN — which for the cutoff means every ensemble
    quietly reverting to a whole-history MASE denominator.
    """
    if name in keys:
        return pd.Series(aligned.index.get_level_values(name), index=aligned.index)
    if name in aligned.columns:
        return aligned[name]
    return pd.Series(index=aligned.index, dtype=dtype)


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
    ``(ts_id, model_type='ensemble_<s>', fold_id, cutoff_date, forecast_date, y_true, yhat)`` for
    the caller to score with `metrics.compute_metrics`. The cutoff is carried out as well as joined
    on, because the caller cuts each series' history there to get the fold's MASE denominator.

    Blends over whichever base models are present per ``(ts_id, cutoff_date, forecast_date)`` key
    (see `_fold_key` for why the cutoff and not the fold ordinal):
    ``mean``/``median`` are unweighted; ``inverse_error`` weights each model per ``ts_id`` by
    ``1/WAPE`` over that model's OOF (self-contained — the same signal ``forecast_metadata`` would
    carry); learned strategies apply ``learned_weights[strategy]`` (skipped if absent). Returns an
    empty frame when the OOF is empty or no strategy produces a blend.

    **Bounds are not blended, so an ensemble's ``coverage``/``pinball``/``interval_score`` are
    NaN.** The base models' are not, any more — the OOF frame now carries ``yhat_lower`` and
    ``yhat_upper`` per fold, and the base cells score them. Blending them is a separate question
    with a real answer to choose (averaging two models' 80% intervals does not give an 80%
    interval), so the columns stay NaN rather than being filled with something plausible and
    wrong. An ensemble is compared to its base models on the point metrics.
    """
    learned_weights = learned_weights or {}
    empty = pd.DataFrame(columns=list(_OOF_BLEND_COLS))
    if oof_df.empty:
        return empty
    models = filter_variants_for_ensemble(list(cfg.models), cfg, oof_df=oof_df)
    keys = _fold_key(oof_df)
    wide = oof_df.pivot_table(index=keys, columns="model_type", values="yhat", aggfunc="first")
    present_models = [m for m in models if m in wide.columns]
    if not present_models:
        return empty
    vals = wide.reindex(columns=present_models).to_numpy(dtype=float)
    aligned = oof_df.drop_duplicates(keys).set_index(keys).reindex(wide.index)
    truth = aligned["y_true"].to_numpy()
    # `fold_id` is still written out — it is the ordinal a reader recognises and the ensemble rows
    # have to line up with the base rows in the same table. It is carried, not joined on, so it is
    # read off whichever side of `aligned` the key put it on: the index when the fallback key is in
    # use, a column when the join went on the cutoff.
    fold_ids = _carried(aligned, keys, "fold_id", "float64")
    cutoffs = _carried(aligned, keys, "cutoff_date", "object")
    steps = _carried(aligned, keys, "horizon_step", "float64")
    ts_ids = wide.index.get_level_values("ts_id").to_numpy()
    # `inverse_error` is the one calculated strategy that *fits* something — a per-series weight
    # off these very rows — so it is held to the same rule as the meta-learners and only sees the
    # inner folds. Every strategy is then scored on all of them: the holdout is reserved from the
    # fit, not from the score. `mean` and `median` fit nothing, so the mask does not reach them.
    fit_rows = (fold_ids != holdout_fold_id(cfg)).to_numpy(dtype=bool)

    parts: list[pd.DataFrame] = []
    steps_arr = steps.to_numpy(dtype=float)
    for strategy in cfg.ensemble.strategies:
        yhat = _blend_oof_strategy(
            strategy,
            vals,
            present_models,
            ts_ids,
            steps_arr,
            truth,
            learned_weights,
            fit_rows,
            cfg,
        )
        if yhat is None:
            continue
        part = pd.DataFrame(
            {
                "ts_id": wide.index.get_level_values("ts_id"),
                "model_type": f"ensemble_{strategy}",
                "fold_id": fold_ids.to_numpy(),
                "cutoff_date": cutoffs.to_numpy(),
                "forecast_date": wide.index.get_level_values(keys[-1]),
                "horizon_step": steps.to_numpy(),
                "y_true": truth,
                "yhat": yhat,
            }
        )
        part = part[~part["yhat"].isna()].reset_index(drop=True)
        if has_output_constraints(cfg.output) and not part.empty:
            constrained_parts: list[pd.DataFrame] = []
            for (_tid, _fid), fg in part.groupby(["ts_id", "fold_id"], sort=False, dropna=False):
                fg_sorted = fg.sort_values("forecast_date").copy()
                fg_c = apply_output_constraints(fg_sorted[["yhat"]].copy(), cfg.output)
                fg_sorted["yhat"] = fg_c["yhat"].to_numpy(dtype=float)
                constrained_parts.append(fg_sorted)
            part = pd.concat(constrained_parts, ignore_index=True)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)[list(_OOF_BLEND_COLS)] if parts else empty


def _blend_oof_strategy(
    strategy: str,
    vals: np.ndarray,
    models: list[str],
    ts_ids: np.ndarray,
    steps: np.ndarray,
    truth: np.ndarray,
    learned_weights: dict[str, dict[str, float]],
    fit_rows: np.ndarray,
    cfg: RunConfig,
) -> np.ndarray | None:
    """The blended yhat for one strategy over the OOF value matrix, or ``None`` to skip it.

    ``vals`` is ``(n_rows, n_models)`` of base OOF yhats aligned to ``models``; ``ts_ids``,
    ``steps``, ``truth`` and ``fit_rows`` are per-row, the last marking the folds a weight may be
    estimated from. Skips (returns ``None``) a learned strategy with no fitted weights.
    """
    top_k = cfg.ensemble.top_k
    margin = cfg.ensemble.parsimony_margin
    if strategy in ("mean", "median") and top_k is None and margin == 0.0:
        if strategy == "mean":
            return _weighted_blend(vals, np.ones(len(models)))
        return np.nanmedian(vals, axis=1)
    if strategy in ("mean", "median"):
        return _filtered_unweighted_oof_blend(
            strategy,
            vals,
            models,
            ts_ids,
            truth,
            fit_rows,
            top_k=top_k,
            parsimony_margin=margin,
        )
    if strategy in ("inverse_error", "champion"):
        return _inverse_error_blend(
            vals,
            models,
            ts_ids,
            truth,
            fit_rows,
            top_k=top_k,
            parsimony_margin=margin,
            champion_only=(strategy == "champion"),
        )
    if strategy == "horizon_specialist":
        return _horizon_specialist_oof_blend(
            vals,
            models,
            ts_ids,
            steps,
            truth,
            fit_rows,
            top_k=top_k,
            parsimony_margin=margin,
        )
    wmap = learned_weights.get(strategy)
    if not wmap:  # learned strategy not fitted (e.g. backtest off) → nothing to score
        return None
    weights = np.array([wmap.get(m, 0.0) for m in models], dtype=float)
    return _weighted_blend(vals, weights)


def _series_oof_wape(wblock: np.ndarray, wtruth: np.ndarray) -> np.ndarray:
    """Per-model WAPE over ``wblock`` (``(n_rows, n_models)``) vs ``wtruth`` (``(n_rows,)``).

    Returns ``np.nan`` for any model column that is entirely NaN on ``wblock``.
    """
    denom = float(np.nansum(np.abs(wtruth)))
    abs_err = np.abs(wblock - wtruth[:, None])
    has_obs = np.any(~np.isnan(abs_err), axis=0)
    raw = np.nansum(abs_err, axis=0) / denom if denom > 0.0 else np.nansum(abs_err, axis=0)
    return np.where(has_obs, raw, np.nan)


def _filtered_unweighted_oof_blend(
    strategy: str,
    vals: np.ndarray,
    models: list[str],
    ts_ids: np.ndarray,
    truth: np.ndarray,
    fit_rows: np.ndarray,
    *,
    top_k: int | None,
    parsimony_margin: float,
) -> np.ndarray:
    """``mean`` or ``median`` after per-series ``top_k`` and ``parsimony_margin`` pruning."""
    out = np.full(vals.shape[0], np.nan)
    for tid in np.unique(ts_ids):
        rows = ts_ids == tid
        block = vals[rows].copy()
        weight_rows = rows & fit_rows
        if not weight_rows.any():
            weight_rows = rows
        wape = _series_oof_wape(vals[weight_rows], truth[weight_rows])
        w = _select_or_weight_models(
            wape,
            models,
            top_k=top_k,
            parsimony_margin=parsimony_margin,
            champion_only=False,
        )
        active = w > 0.0
        if strategy == "median":
            masked = np.where(active[None, :], block, np.nan)
            out[rows] = np.nanmedian(masked, axis=1)
        else:
            uniform_active = np.where(active, 1.0, 0.0)
            out[rows] = _weighted_blend(block, uniform_active)
    return out


def _inverse_error_blend(
    vals: np.ndarray,
    models: list[str],
    ts_ids: np.ndarray,
    truth: np.ndarray,
    fit_rows: np.ndarray,
    *,
    top_k: int | None = None,
    parsimony_margin: float = 0.0,
    champion_only: bool = False,
) -> np.ndarray:
    """Inverse-WAPE-weighted (or single-winner ``champion``) blend per ``ts_id`` from OOF."""
    out = np.full(vals.shape[0], np.nan)
    for tid in np.unique(ts_ids):
        rows = ts_ids == tid
        block = vals[rows]
        weight_rows = rows & fit_rows
        if not weight_rows.any():
            weight_rows = rows
        wape = _series_oof_wape(vals[weight_rows], truth[weight_rows])
        weights = _select_or_weight_models(
            wape,
            models,
            top_k=top_k,
            parsimony_margin=parsimony_margin,
            champion_only=champion_only,
        )
        out[rows] = _weighted_blend(block, weights)
    return out


def _horizon_specialist_oof_blend(
    vals: np.ndarray,
    models: list[str],
    ts_ids: np.ndarray,
    steps: np.ndarray,
    truth: np.ndarray,
    fit_rows: np.ndarray,
    *,
    top_k: int | None = None,
    parsimony_margin: float = 0.0,
) -> np.ndarray:
    """Per-``(ts_id, horizon_step)`` inverse-WAPE blend over inner folds."""
    out = np.full(vals.shape[0], np.nan)
    for tid in np.unique(ts_ids):
        s_rows = ts_ids == tid
        s_fit = s_rows & fit_rows
        if not s_fit.any():
            s_fit = s_rows
        fallback_wape = _series_oof_wape(vals[s_fit], truth[s_fit])
        fallback_w = _select_or_weight_models(
            fallback_wape,
            models,
            top_k=top_k,
            parsimony_margin=parsimony_margin,
            champion_only=False,
        )
        s_steps = steps[s_rows]
        if np.all(np.isnan(s_steps)):
            out[s_rows] = _weighted_blend(vals[s_rows], fallback_w)
            continue
        for step_val in np.unique(s_steps[~np.isnan(s_steps)]):
            h_rows = s_rows & (steps == step_val)
            h_fit = h_rows & fit_rows
            if not h_fit.any():
                h_fit = h_rows
            wape = _series_oof_wape(vals[h_fit], truth[h_fit])
            if np.all(np.isnan(wape)):
                weights = fallback_w
            else:
                weights = _select_or_weight_models(
                    wape,
                    models,
                    top_k=top_k,
                    parsimony_margin=parsimony_margin,
                    champion_only=False,
                )
            out[h_rows] = _weighted_blend(vals[h_rows], weights)
        nan_step_rows = s_rows & np.isnan(steps)
        if nan_step_rows.any():
            out[nan_step_rows] = _weighted_blend(vals[nan_step_rows], fallback_w)
    return out


# --- learned meta-learners -----------------------------------------------------


def _pivot_oof(oof_df: pd.DataFrame, models: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Reshape long OOF rows into ``(X, y)`` for stacking.

    ``X`` is ``(n_samples, n_models)`` of base-model OOF forecasts, ``y`` the aligned truth.
    Samples are `_fold_key` keys — ``(ts_id, cutoff_date, forecast_date)`` — and rows missing any
    base model are dropped so every column is comparable. That `dropna` is why the key matters
    here more than anywhere else: a key that fails to align two engines does not merely blend the
    wrong things, it silently deletes the training rows a meta-learner would have fitted on.
    Column order follows ``models`` (stable weights).
    """
    keys = _fold_key(oof_df)
    wide = oof_df.pivot_table(
        index=keys,
        columns="model_type",
        values="yhat",
        aggfunc="first",
    )
    missing = [m for m in models if m not in wide.columns]
    if missing:
        raise ConfigError(f"OOF is missing base models {missing}; cannot fit learned ensemble")
    wide = wide[models].dropna()

    truth = oof_df.drop_duplicates(keys).set_index(keys)["y_true"].reindex(wide.index)
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
) -> tuple[dict[str, dict[str, float]], dict[str, bytes], str]:
    """Fit every learned strategy in ``cfg`` on the backtest OOF's **inner** folds.

    Returns ``(weights, artifacts, scoring_basis)``:
      * ``weights[strategy]`` maps base-model name → learned weight,
      * ``artifacts[strategy]`` is the pickled, uploadable meta-learner payload,
      * ``scoring_basis`` is ``"holdout"`` when the newest fold was successfully reserved from the
        fit and ``"in_sample"`` when it could not be — the caller writes it to the registry beside
        the metrics it qualifies.

    **The newest fold is not in ``X``.** A meta-learner that trains and is scored on the same folds
    reports a fit statistic, and it lands in the same leaderboard column as the base models'
    genuinely out-of-fold numbers, where nothing distinguishes the two. Holding the newest fold out
    of the fit is what makes ``ensemble_nnls`` comparable to ``theta`` — see `inner_fold_mask`.

    Refuses to run when backtesting is off — a learned ensemble has no legitimate,
    leakage-free data to train on (the guard that keeps stacking honest). Base models come
    from ``cfg.models`` in order; native vs python doesn't matter — only their OOF forecasts do.
    """
    learned = [s for s in cfg.ensemble.strategies if s in LEARNED_STRATEGIES]
    if not learned:
        return {}, {}, "holdout"
    if not cfg.backtest.enabled:
        raise ConfigError(
            "learned ensemble strategies require backtest.enabled=true (they train on OOF)"
        )
    if oof_df.empty:
        raise ConfigError("cannot fit a learned ensemble: backtest_oof is empty")

    models = filter_variants_for_ensemble(list(cfg.models), cfg, oof_df=oof_df)
    inner, basis = inner_fold_mask(oof_df, cfg)
    X, y = _pivot_oof(oof_df[inner], models) if basis == "holdout" else (_EMPTY_X, _EMPTY_Y)
    if X.shape[0] == 0:
        # The inner folds exist but no *complete* row survives across every base model there --
        # one model only forecast the newest window, say. Refusing would cost the run its whole
        # learned ensemble over a fold-coverage gap; fitting on everything and saying so does not.
        basis = "in_sample"
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
        _log.info("fit %s ensemble on %d %s OOF rows", strategy, X.shape[0], basis)

    return weights, artifacts, basis


# --- calculated ensembles in pandas (Write-API path) ---------------------------

# The prediction-row columns combine_calculated emits (ensemble_id + run_id filled by the caller).
_PRED_OUT_COLS = ("ts_id", "model_type", "forecast_date", "yhat", "yhat_lower", "yhat_upper")


def _pruned_models(
    cfg: RunConfig,
    metric_df: pd.DataFrame | None,
    oof_df: pd.DataFrame | None = None,
) -> list[str]:
    """The base models a calculated ensemble should blend, after optional pruning (pure).

    Pruning (``ensemble.prune_threshold`` > 0) drops any model whose mean backtest decision metric
    is worse than the threshold, so a bad base model can't drag the consensus down — equivalent to
    a ``model_type NOT IN (… metric > threshold)`` filter, but applied fleet-wide (a model pruned
    on its run-level metric is dropped from every series' blend). Threshold 0.0, or no metric
    frame, means no pruning. Order follows ``cfg.models`` (stable).
    """
    models = filter_variants_for_ensemble(list(cfg.models), cfg, oof_df=oof_df, metric_df=metric_df)
    threshold = cfg.ensemble.prune_threshold
    if threshold <= 0.0 or metric_df is None or metric_df.empty:
        return models
    metric = cfg.backtest.decision_metric
    if metric not in metric_df.columns:
        return models
    mean_by_model = metric_df.groupby("model_type")[metric].mean()

    def keep(name: str) -> bool:
        mean = mean_by_model.get(name, float("nan"))
        if mean != mean:  # no scored rows for this model — no evidence, so keep it (as before)
            return True
        return not (loss_of(metric, float(mean)) > threshold)

    return [m for m in models if keep(m)]


def _calc_blend(strategy: str, vals: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Blend an ``(n_rows, n_models)`` value matrix by one calculated strategy (pure).

    ``mean``/``median`` are unweighted when ``weights`` is ``None``; when ``top_k`` or
    ``parsimony_margin`` is active on ``median``, zero-weight columns are masked to NaN before
    taking the row median. Weighted strategies pass their per-row weight matrix.
    """
    if strategy == "median":
        if weights is not None:
            masked = np.where(weights > 0.0, vals, np.nan)
            return np.nanmedian(masked, axis=1)
        return np.nanmedian(vals, axis=1)
    w = np.ones(vals.shape[1]) if weights is None else weights
    return _weighted_blend(vals, w)


def combine_calculated(
    base_df: pd.DataFrame,
    cfg: RunConfig,
    metric_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Blend base predictions into ``ensemble_<s>`` rows for every calculated strategy (pure).

    Builds the ``ensemble_<s>`` rows in pandas rather than via SQL ``INSERT…SELECT``: reads the base
    ``forecast_predictions`` (long-format ``ts_id, model_type, forecast_date, yhat, yhat_lower,
    yhat_upper``) and returns prediction-row dicts the caller stamps with ``run_id``/``ensemble_id``
    and appends via the Storage Write API. Supports ``mean``, ``median``, ``inverse_error``,
    ``champion`` (per-series single winner), and ``horizon_specialist`` (per-series,
    per-horizon-step inverse-error blend when ``oof_df`` is supplied, falling back to series-level
    ``inverse_error`` otherwise), as well as ``top_k`` and ``parsimony_margin`` pruning.
    """
    calculated = [s for s in cfg.ensemble.strategies if s in CALCULATED_STRATEGIES]
    if not calculated or base_df.empty:
        return []
    models = _pruned_models(cfg, metric_df, oof_df=oof_df)
    present = [m for m in models if m in set(base_df["model_type"].unique())]
    if not present:
        return []

    # One wide frame per value column, aligned to the present base models, reused across strategies.
    keys = ["ts_id", "forecast_date"]
    wide = {
        col: base_df.pivot_table(index=keys, columns="model_type", values=col).reindex(
            columns=present
        )
        for col in ("yhat", "yhat_lower", "yhat_upper")
    }
    index = wide["yhat"].index
    ts_ids_arr = np.asarray(index.get_level_values("ts_id"))
    has_selection_filter = cfg.ensemble.top_k is not None or cfg.ensemble.parsimony_margin > 0.0

    ie_weights: np.ndarray | None = None
    if "inverse_error" in calculated or (
        has_selection_filter and ("mean" in calculated or "median" in calculated)
    ):
        ie_weights = _inverse_error_weight_matrix(
            present, cfg, metric_df, ts_ids_arr, champion_only=False
        )

    champ_weights: np.ndarray | None = None
    if "champion" in calculated:
        champ_weights = _inverse_error_weight_matrix(
            present, cfg, metric_df, ts_ids_arr, champion_only=True
        )

    hs_weights: np.ndarray | None = None
    if "horizon_specialist" in calculated:
        hs_weights = _horizon_specialist_weight_matrix(present, cfg, metric_df, oof_df, index)

    rows: list[dict[str, Any]] = []
    for strategy in calculated:
        weights: np.ndarray | None = None
        if strategy == "inverse_error":
            weights = ie_weights
        elif strategy == "champion":
            weights = champ_weights
        elif strategy == "horizon_specialist":
            weights = hs_weights
        elif has_selection_filter and ie_weights is not None:
            weights = np.where(ie_weights > 0.0, 1.0, 0.0)
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


def _inverse_error_weight_matrix(
    models: list[str],
    cfg: RunConfig,
    metric_df: pd.DataFrame | None,
    ts_ids: np.ndarray,
    *,
    champion_only: bool = False,
) -> np.ndarray:
    """Inverse-error (or ``champion``) weights per prediction row, estimated **per series**."""
    n = len(models)
    metric = cfg.backtest.decision_metric
    top_k = cfg.ensemble.top_k
    margin = cfg.ensemble.parsimony_margin
    uniform = np.full(n, 1.0 / n)
    if metric_df is None or metric_df.empty or metric not in metric_df.columns:
        if champion_only:
            fallback = np.zeros(n, dtype=float)
            fallback[0] = 1.0
            return np.tile(fallback, (len(ts_ids), 1))
        return np.tile(uniform, (len(ts_ids), 1))
    if "ts_id" not in metric_df.columns:
        return np.tile(
            _inverse_error_run_weights(models, cfg, metric_df, champion_only=champion_only),
            (len(ts_ids), 1),
        )

    mean_by_series = (
        metric_df.groupby(["ts_id", "model_type"])[metric]
        .mean()
        .unstack("model_type")
        .reindex(columns=models)
    )
    per_series = {
        str(ts_id): _select_or_weight_models(
            np.array([loss_of(metric, float(v)) for v in row], dtype=float),
            models,
            top_k=top_k,
            parsimony_margin=margin,
            champion_only=champion_only,
        )
        for ts_id, row in zip(mean_by_series.index, mean_by_series.to_numpy(), strict=True)
    }
    return np.array([per_series.get(str(t), uniform) for t in ts_ids], dtype=float)


def _inverse_error_run_weights(
    models: list[str],
    cfg: RunConfig,
    metric_df: pd.DataFrame | None,
    *,
    champion_only: bool = False,
) -> np.ndarray:
    """Per-model inverse-error weights pooled over the whole metric frame (pure)."""
    n = len(models)
    metric = cfg.backtest.decision_metric
    if metric_df is None or metric_df.empty or metric not in metric_df.columns:
        return np.full(n, 1.0 / n)
    mean_by_model = metric_df.groupby("model_type")[metric].mean()
    errors = np.array(
        [loss_of(metric, float(mean_by_model.get(m, np.nan))) for m in models], dtype=float
    )
    return _select_or_weight_models(
        errors,
        models,
        top_k=cfg.ensemble.top_k,
        parsimony_margin=cfg.ensemble.parsimony_margin,
        champion_only=champion_only,
    )


def _horizon_specialist_weight_matrix(
    models: list[str],
    cfg: RunConfig,
    metric_df: pd.DataFrame | None,
    oof_df: pd.DataFrame | None,
    index: pd.Index,
) -> np.ndarray:
    """Per-``(ts_id, horizon_step)`` weight matrix ``(n_rows, n_models)`` for future predictions.

    Assigns each future prediction row its 1-based ordinal ``horizon_step`` within its ``ts_id``
    (sorted by ``forecast_date``) and looks up the inner-fold OOF WAPE for ``(ts_id, step)``.
    Falls back to series-level `_inverse_error_weight_matrix` when ``oof_df`` is absent or empty.
    """
    ts_ids = np.asarray(index.get_level_values("ts_id"))
    fallback = _inverse_error_weight_matrix(models, cfg, metric_df, ts_ids, champion_only=False)
    if oof_df is None or oof_df.empty or "horizon_step" not in oof_df.columns:
        return fallback

    keys = _fold_key(oof_df)
    wide = oof_df.pivot_table(index=keys, columns="model_type", values="yhat", aggfunc="first")
    present_in_oof = [m for m in models if m in wide.columns]
    if not present_in_oof:
        return fallback

    vals = wide.reindex(columns=models).to_numpy(dtype=float)
    aligned = oof_df.drop_duplicates(keys).set_index(keys).reindex(wide.index)
    truth = aligned["y_true"].to_numpy(dtype=float)
    fold_ids = _carried(aligned, keys, "fold_id", "float64")
    steps = _carried(aligned, keys, "horizon_step", "float64").to_numpy(dtype=float)
    oof_ts = wide.index.get_level_values("ts_id").to_numpy()
    fit_rows = (fold_ids != holdout_fold_id(cfg)).to_numpy(dtype=bool)

    top_k = cfg.ensemble.top_k
    margin = cfg.ensemble.parsimony_margin
    step_weights: dict[tuple[str, int], np.ndarray] = {}
    for tid in np.unique(oof_ts):
        s_rows = oof_ts == tid
        s_steps = steps[s_rows]
        if np.all(np.isnan(s_steps)):
            continue
        for step_val in np.unique(s_steps[~np.isnan(s_steps)]):
            h_rows = s_rows & (steps == step_val)
            h_fit = h_rows & fit_rows
            if not h_fit.any():
                h_fit = h_rows
            wape = _series_oof_wape(vals[h_fit], truth[h_fit])
            if not np.all(np.isnan(wape)):
                step_weights[(str(tid), round(float(step_val)))] = _select_or_weight_models(
                    wape,
                    models,
                    top_k=top_k,
                    parsimony_margin=margin,
                    champion_only=False,
                )

    if not step_weights:
        return fallback

    out = fallback.copy()
    # Compute 1-based horizon_step for each row in `index` (already sorted by (ts_id, forecast_date)
    # from pivot_table).
    step_counter: dict[str, int] = {}
    for i, tid_raw in enumerate(ts_ids):
        tid_str = str(tid_raw)
        step_idx = step_counter.get(tid_str, 0) + 1
        step_counter[tid_str] = step_idx
        w = step_weights.get((tid_str, step_idx))
        if w is not None:
            out[i] = w
    return out


# --- ensemble empirical interval calibration & constraint enforcement ----------


def ensemble_oof_bounds(
    ens_oof_group: pd.DataFrame,
    fold_id: object,
    cfg: RunConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Empirical 80% prediction intervals for one fold of an ensemble's OOF rows (pure).

    Uses leave-one-fold-out OOF residuals (``y_true - yhat`` from all other folds in
    ``ens_oof_group``, falling back to all folds when only a single fold exists) so every
    ``ensemble_*`` pseudo-model earns non-NaN interval metrics (``coverage_80``, ``pinball_loss``,
    ``winkler_score``, ``interval_width``, ``wis``) on the same footing as base models. Applies
    `calibration.apply_output_constraints` when configured.
    """

    fg = ens_oof_group[ens_oof_group["fold_id"] == fold_id].sort_values("forecast_date")
    if fg.empty:
        return None, None
    other = ens_oof_group[ens_oof_group["fold_id"] != fold_id]
    ref = other if not other.empty else ens_oof_group
    residuals = (ref["y_true"] - ref["yhat"]).dropna().to_numpy(dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return None, None
    if len(residuals) == 1:
        half = max(abs(float(residuals[0])), 1e-6)
        q10, q90 = -half, half
    else:
        q10, q90 = np.percentile(residuals, [10.0, 90.0])
        if q10 == q90:
            half = max(abs(float(q10)) * 0.1, 1e-6)
            q10, q90 = float(q10) - half, float(q90) + half

    yhat = fg["yhat"].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "yhat": yhat,
            "yhat_lower": yhat + float(q10),
            "yhat_upper": yhat + float(q90),
        }
    )
    if has_output_constraints(cfg.output):
        frame = apply_output_constraints(frame, cfg.output)
    return frame["yhat_lower"].to_numpy(dtype=float), frame["yhat_upper"].to_numpy(dtype=float)


def calibrate_ensemble_predictions(
    pred_rows: list[dict[str, Any]],
    ens_oof: pd.DataFrame,
    cfg: RunConfig,
) -> list[dict[str, Any]]:
    """Calibrate ensemble future prediction intervals from ``ens_oof`` & apply output constraints.

    When ``cfg.output.calibrate_intervals`` is True and ``ens_oof`` has >= 5 residual rows for a
    ``(ts_id, model_type)`` cell, replaces the linearly blended bounds with empirical 10th/90th
    residual quantiles from `calibration.calibrate_from_oof`. Always enforces physical output
    constraints (`calibration.apply_output_constraints`) when configured.
    """

    if not pred_rows:
        return pred_rows
    need_constraints = has_output_constraints(cfg.output)
    need_q_or_cost = (
        cfg.output.quantiles is not None or cfg.output.effective_target_quantile is not None
    )
    need_cal = not ens_oof.empty
    if not need_constraints and not need_cal and not need_q_or_cost:
        return pred_rows

    df = pd.DataFrame(pred_rows)
    had_quantiles_col = "quantiles" in df.columns
    target_qs = cfg.output.effective_quantiles
    lo_q, hi_q = target_qs[0], target_qs[-1]
    oof_by_cell: dict[tuple[str, str], pd.DataFrame] = {}
    if need_cal:
        for (tid, mtype), g in ens_oof.groupby(["ts_id", "model_type"]):
            oof_by_cell[(str(tid), str(mtype))] = g

    emit_quantiles = cfg.output.quantiles is not None or had_quantiles_col
    out_parts: list[pd.DataFrame] = []
    for (tid, mtype), group in df.groupby(["ts_id", "model_type"], sort=False):
        cell_df = group.copy()
        if (need_constraints or need_q_or_cost) and "yhat_raw" not in cell_df.columns:
            cell_df["yhat_raw"] = cell_df["yhat"].to_numpy(dtype=float)
        calibrated_this_cell = False
        if need_cal:
            oof_slice = oof_by_cell.get((str(tid), str(mtype)))
            if oof_slice is not None and not oof_slice.empty:
                cal = calibrate_from_oof(oof_slice, target_qs, residual_column="yhat")
                if cal is not None:
                    calibrated_this_cell = True
                    yhat_vals = cell_df["yhat"].to_numpy(dtype=float)
                    qlo_arr = np.array(
                        [cal.for_step(s).quantiles[lo_q] for s in range(1, len(cell_df) + 1)],
                        dtype=float,
                    )
                    qhi_arr = np.array(
                        [cal.for_step(s).quantiles[hi_q] for s in range(1, len(cell_df) + 1)],
                        dtype=float,
                    )
                    cell_df["yhat_lower"] = yhat_vals + qlo_arr
                    cell_df["yhat_upper"] = yhat_vals + qhi_arr
                    if emit_quantiles:
                        qjson_rows: list[str] = []
                        for idx in range(len(cell_df)):
                            step_q = cal.for_step(idx + 1).quantiles
                            raw_qmap = {
                                q: float(yhat_vals[idx] + step_q[q])
                                for q in target_qs
                                if q in step_q
                            }
                            mono_qmap = rearrange_monotone_quantiles(raw_qmap)
                            qjson_rows.append(
                                json.dumps(
                                    {str(q): v for q, v in mono_qmap.items()}, sort_keys=True
                                )
                            )
                        cell_df["quantiles"] = qjson_rows
                    if cfg.output.effective_target_quantile is not None:
                        q_star = cfg.output.effective_target_quantile
                        qstar_arr = np.array(
                            [cal.for_step(s).quantiles[q_star] for s in range(1, len(cell_df) + 1)],
                            dtype=float,
                        )
                        cell_df["yhat"] = yhat_vals + qstar_arr
                else:
                    resid = (oof_slice["y_true"] - oof_slice["yhat"]).dropna().to_numpy(dtype=float)
                    resid = resid[np.isfinite(resid)]
                    if len(resid) >= 2:
                        calibrated_this_cell = True
                        qlo, qhi = np.percentile(resid, [lo_q * 100.0, hi_q * 100.0])
                        yhat_vals = cell_df["yhat"].to_numpy(dtype=float)
                        cell_df["yhat_lower"] = yhat_vals + float(qlo)
                        cell_df["yhat_upper"] = yhat_vals + float(qhi)
                        if emit_quantiles:
                            q_pcts = np.percentile(resid, [q * 100.0 for q in target_qs])
                            qjson_rows = []
                            for idx in range(len(cell_df)):
                                raw_qmap = {
                                    q: float(yhat_vals[idx] + q_pcts[j])
                                    for j, q in enumerate(target_qs)
                                }
                                mono_qmap = rearrange_monotone_quantiles(raw_qmap)
                                qjson_rows.append(
                                    json.dumps(
                                        {str(q): v for q, v in mono_qmap.items()}, sort_keys=True
                                    )
                                )
                            cell_df["quantiles"] = qjson_rows
                        if cfg.output.effective_target_quantile is not None:
                            qstar = float(
                                np.percentile(resid, cfg.output.effective_target_quantile * 100.0)
                            )
                            cell_df["yhat"] = yhat_vals + qstar
        if not calibrated_this_cell and need_q_or_cost:
            cell_df, _ = apply_quantile_and_cost_rules(cell_df, cfg.output)
            if not emit_quantiles and "quantiles" in cell_df.columns:
                cell_df = cell_df.drop(columns=["quantiles"])
        if need_constraints:
            cell_df = apply_output_constraints(cell_df, cfg.output)
        out_parts.append(cell_df)

    merged = pd.concat(out_parts, ignore_index=True)
    # Restore None for any NaN bounds so BigQuery Write API receives NULL rather than NaN.
    result: list[dict[str, Any]] = []
    for raw_rec in merged.to_dict(orient="records"):
        rec: dict[str, Any] = {str(k): v for k, v in raw_rec.items()}
        for bcol in ("yhat_lower", "yhat_upper"):
            val = rec.get(bcol)
            if val is not None and isinstance(val, float) and np.isnan(val):
                rec[bcol] = None
        result.append(rec)
    return result
