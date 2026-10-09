"""Econometric & operational forecast diagnostics — pure DataFrame analytics (zero GCP imports).

Provides the post-backtest and post-run diagnostic suite for evaluating model value, lead-time
accuracy decay, directional bias, and temporal/lead-time aggregation:

* **Forecast Value Added (FVA) & Statistical Significance (`compute_fva`, `diebold_mariano_test`)**
  — step-by-step accuracy waterfall across modeling tiers (`Baseline` -> `Statistical` -> `ML` ->
  `Deep Learning` -> `AutoML` -> `Ensemble`) with **Harvey, Leybourne & Newbold (1997) modified
  Diebold–Mariano** and **paired Wilcoxon signed-rank** $p$-values on out-of-fold loss
  differentials.
* **Lead-Time Horizon Decay (`compute_horizon_decay`)** — accuracy and empirical interval coverage
  at each forecast step $h = 1 \\dots H$, plus the crossover horizon step $h^*$ where errors exceed
  the baseline.
* **Directional Bias & Tracking Signal Audit (`compute_bias_report`)** — per-`(ts_id, model_type)`
  audit of signed percentage bias, **Trigg / APICS Tracking Signal** ($\\text{TS} = \\sum e_t /
  \\text{MAD}$), and consecutive over/under-forecasting fold streaks.
* **Autocorrelation-Aware Temporal & Lead-Time Rollup (`rollup_time`)** — aggregates point
  forecasts and actuals to Weekly (`"W"`), Monthly (`"MS"`/`"M"`), Quarterly (`"QS"`/`"Q"`), or
  fixed $L$-step supplier lead-time windows (`freq="7D"` or `window=7`) while computing
  period-level prediction intervals from empirical multi-step error covariance rather than naively
  summing daily interval bounds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import compute_metrics, loss_of

__all__ = [
    "BASELINE_CANDIDATES",
    "compute_bias_report",
    "compute_fva",
    "compute_horizon_decay",
    "diebold_mariano_test",
    "plot_fva",
    "plot_horizon_decay",
    "rollup_time",
    "wilcoxon_pvalue",
]

# Preferred order when automatically picking a naive benchmark in `compute_fva` /
# `compute_horizon_decay`.
BASELINE_CANDIDATES: tuple[str, ...] = (
    "naive_seasonal",
    "naive_last",
    "naive_moving_average",
    "naive_mean",
    "naive_drift",
    "snaive",
)

_STAGE_ORDER: dict[str, tuple[int, str]] = {
    "baseline": (1, "Baseline"),
    "statistical": (2, "Statistical"),
    "native": (2, "Statistical / SQL"),
    "ml": (3, "Machine Learning"),
    "deep_learning": (4, "Deep Learning"),
    "automl": (5, "AutoML"),
    "ensemble": (6, "Ensemble"),
}


def _classify_model_stage(model_type: str, baseline_model: str) -> tuple[int, str]:
    """Map ``model_type`` to ``(stage_rank, stage_label)`` on the FVA ladder."""
    if model_type == baseline_model or model_type.startswith("naive_"):
        return _STAGE_ORDER["baseline"]
    if model_type.startswith("ensemble_"):
        return _STAGE_ORDER["ensemble"]
    try:
        from .models import get_model

        fam = get_model(model_type).family
        return _STAGE_ORDER.get(fam, (2, "Statistical"))
    except Exception:  # noqa: BLE001 - unknown/custom model names default to Statistical tier
        return _STAGE_ORDER["statistical"]


def _resolve_baseline_model(models: Sequence[str], baseline: str | None) -> str:
    """Choose the benchmark model name from ``models``."""
    unique = list(dict.fromkeys(str(m) for m in models))
    if not unique:
        return baseline or "naive_seasonal"
    if baseline is not None and baseline in unique:
        return baseline
    for cand in BASELINE_CANDIDATES:
        if cand in unique:
            return cand
    non_ens = [m for m in unique if not m.startswith("ensemble_")]
    return non_ens[0] if non_ens else unique[0]


def diebold_mariano_test(
    loss_model: Sequence[float] | np.ndarray,
    loss_baseline: Sequence[float] | np.ndarray,
    *,
    h: int = 1,
) -> tuple[float, float]:
    """Harvey, Leybourne & Newbold (1997) modified Diebold–Mariano test on paired losses.

    Tests $H_0: \\mathbb{E}[L(\\text{model}) - L(\\text{baseline})] = 0$ against the two-sided
    alternative using Newey–West / Bartlett-style autocovariance up to lag $h - 1$ and the HLN
    small-sample Student-$t_{n-1}$ adjustment:

    $$\\kappa_{\\text{HLN}} = \\sqrt{\\frac{n + 1 - 2h + h(h - 1)/n}{n}}$$

    Returns ``(dm_stat, p_value)``. Negative ``dm_stat`` means ``loss_model < loss_baseline``
    (the candidate model improved on the baseline).
    """
    lm = np.asarray(loss_model, dtype=float)
    lb = np.asarray(loss_baseline, dtype=float)
    mask = np.isfinite(lm) & np.isfinite(lb)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")

    d = lm[mask] - lb[mask]
    n = int(d.size)
    d_bar = float(np.mean(d))
    if np.allclose(d, 0.0):
        return 0.0, 1.0

    centered = d - d_bar
    gamma_0 = float(np.dot(centered, centered) / n)
    if gamma_0 <= 1e-18:
        return (float("-inf") if d_bar < 0 else float("inf")), 0.0

    eff_h = max(1, min(int(h), n // 2))
    var_d = gamma_0
    for lag in range(1, eff_h):
        weight = 1.0 - lag / eff_h
        gamma_k = float(np.dot(centered[lag:], centered[:-lag]) / n)
        var_d += 2.0 * weight * gamma_k

    # Guard against negative finite-sample spectral density estimates.
    var_d_bar = max(var_d, gamma_0 * 0.1) / n
    raw_dm = d_bar / math.sqrt(var_d_bar)

    hln_num = n + 1.0 - 2.0 * eff_h + (eff_h * (eff_h - 1.0)) / n
    hln_factor = math.sqrt(max(hln_num / n, 1.0 / n))
    dm_stat = float(raw_dm * hln_factor)
    p_val = float(2.0 * stats.t.sf(abs(dm_stat), df=max(1, n - 1)))
    return dm_stat, p_val


def wilcoxon_pvalue(
    loss_model: Sequence[float] | np.ndarray,
    loss_baseline: Sequence[float] | np.ndarray,
) -> float:
    """Two-sided paired Wilcoxon signed-rank $p$-value on finite non-tied loss differentials."""
    lm = np.asarray(loss_model, dtype=float)
    lb = np.asarray(loss_baseline, dtype=float)
    mask = np.isfinite(lm) & np.isfinite(lb)
    if int(mask.sum()) < 3:
        return float("nan")
    diff = lm[mask] - lb[mask]
    non_zero = diff[np.abs(diff) > 1e-12]
    if non_zero.size < 3:
        return 1.0 if diff.size >= 3 else float("nan")
    try:
        res = stats.wilcoxon(non_zero, zero_method="wilcox", alternative="two-sided")
        return float(res.pvalue)
    except Exception:  # noqa: BLE001 - degenerate rank distributions return NaN gracefully
        return float("nan")


def _ensure_oof_columns(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date and step columns on an OOF DataFrame without mutating the caller's frame."""
    df = oof_df.copy()
    if "forecast_date" not in df.columns and "ds" in df.columns:
        df["forecast_date"] = df["ds"]
    if "ds" not in df.columns and "forecast_date" in df.columns:
        df["ds"] = df["forecast_date"]
    if "ts_id" not in df.columns:
        df["ts_id"] = "series_0"
    if "fold_id" not in df.columns:
        df["fold_id"] = 0
    if "horizon_step" not in df.columns or df["horizon_step"].isna().all():
        df = df.sort_values(["ts_id", "model_type", "fold_id", "forecast_date"])
        df["horizon_step"] = (
            df.groupby(["ts_id", "model_type", "fold_id"], sort=False).cumcount() + 1
        )
    return df


def _cell_metric_loss(group: pd.DataFrame, metric: str) -> float:
    """Compute scalar loss for ``metric`` on one ``(ts_id, model_type)`` or fold slice."""
    y_true = pd.to_numeric(group["y_true"], errors="coerce").to_numpy(dtype=float)
    yhat = pd.to_numeric(group["yhat"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(yhat)
    if not keep.any():
        return float("nan")
    lower = (
        pd.to_numeric(group["yhat_lower"], errors="coerce").to_numpy(dtype=float)[keep]
        if "yhat_lower" in group.columns
        else None
    )
    upper = (
        pd.to_numeric(group["yhat_upper"], errors="coerce").to_numpy(dtype=float)[keep]
        if "yhat_upper" in group.columns
        else None
    )
    m_dict = compute_metrics(
        y_true[keep],
        yhat[keep],
        lower=lower if lower is not None and np.isfinite(lower).any() else None,
        upper=upper if upper is not None and np.isfinite(upper).any() else None,
    )
    return float(loss_of(metric, m_dict.get(metric, float("nan"))))


def compute_fva(
    oof_df: pd.DataFrame,
    *,
    baseline: str | None = None,
    metric: str = "wape",
    series_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the Forecast Value Added (FVA) Waterfall & Statistical Significance table.

    Evaluates each ``model_type`` in ``oof_df`` against ``baseline`` (defaulting to
    ``"naive_seasonal"`` or the simplest benchmark present) and reports:

    - ``fva_stage_rank`` / ``fva_stage``: position on the FVA ladder (`Baseline`, `Statistical`,
      `Machine Learning`, `Deep Learning`, `AutoML`, `Ensemble`).
    - ``metric``: the decision metric name (e.g. ``"wape"``).
    - ``mean_metric`` / ``pooled_wape`` / ``mean_mae``: accuracy across all scored series.
    - ``baseline_model`` / ``baseline_metric``: benchmark reference loss.
    - ``fva_abs`` ($L_{\\text{baseline}} - L_{\\text{model}}$) and ``fva_pct``
      ($(L_{\\text{baseline}} - L_{\\text{model}}) / L_{\\text{baseline}}$): positive values mean
      the model added value over the baseline; negative values mean it destroyed value.
    - ``win_rate_vs_baseline``: fraction of series where ``model_type`` achieved strictly lower
      loss than ``baseline``.
    - ``dm_stat``, ``p_value_dm``, ``p_value_wilcoxon``, ``significant_at_05``: Harvey–Leybourne–
      Newbold (1997) modified Diebold–Mariano statistic/p-value and paired Wilcoxon signed-rank
      p-value testing whether the accuracy gain over ``baseline`` is statistically significant.
    """
    cols = [
        "fva_stage_rank",
        "fva_stage",
        "model_type",
        "demand_class",
        "n_series",
        "n_points",
        "metric",
        "mean_metric",
        "pooled_wape",
        "mean_mae",
        "baseline_model",
        "baseline_metric",
        "fva_abs",
        "fva_pct",
        "win_rate_vs_baseline",
        "dm_stat",
        "p_value_dm",
        "p_value_wilcoxon",
        "significant_at_05",
    ]
    if oof_df is None or oof_df.empty:
        return pd.DataFrame(columns=cols)

    df = _ensure_oof_columns(oof_df)
    if series_df is not None and not series_df.empty and "demand_class" not in df.columns:
        from .features import compute_demand_profile

        ts_col = "ts_id" if "ts_id" in series_df.columns else None
        y_col = "y" if "y" in series_df.columns else None
        if ts_col and y_col:
            regime_map = {
                str(tid): str(compute_demand_profile(grp[y_col])["demand_class"])
                for tid, grp in series_df.groupby(ts_col, sort=False)
            }
            df["demand_class"] = df["ts_id"].astype(str).map(regime_map).fillna("all")
    if "demand_class" not in df.columns:
        df["demand_class"] = "all"

    all_models = list(dict.fromkeys(df["model_type"].astype(str)))
    base_model = _resolve_baseline_model(all_models, baseline)

    # Compute per-(ts_id, model_type) loss and per-(ts_id, fold_id, forecast_date, model_type)
    # pointwise absolute errors so DM tests have power even on single-series panels.
    series_losses: dict[tuple[str, str], float] = {}
    fold_losses: dict[tuple[str, int, str], float] = {}
    for (tid, mtype), grp in df.groupby(["ts_id", "model_type"], sort=False):
        series_losses[(str(tid), str(mtype))] = _cell_metric_loss(grp, metric)
    for (tid, fid, mtype), grp in df.groupby(["ts_id", "fold_id", "model_type"], sort=False):
        fold_losses[(str(tid), int(str(fid)), str(mtype))] = _cell_metric_loss(grp, metric)

    df["_abs_err"] = np.abs(
        pd.to_numeric(df["y_true"], errors="coerce") - pd.to_numeric(df["yhat"], errors="coerce")
    )
    point_pivot = df.pivot_table(
        index=["ts_id", "fold_id", "forecast_date"],
        columns="model_type",
        values="_abs_err",
        aggfunc="mean",
    )

    all_ts = sorted({str(t) for t in df["ts_id"].unique()})
    all_folds = sorted({(str(t), int(f)) for t, f in zip(df["ts_id"], df["fold_id"], strict=False)})

    base_series_vec = np.array(
        [series_losses.get((tid, base_model), float("nan")) for tid in all_ts], dtype=float
    )
    base_mean_metric = (
        float(np.nanmean(base_series_vec)) if np.isfinite(base_series_vec).any() else float("nan")
    )

    records: list[dict[str, Any]] = []
    for mtype in all_models:
        stage_rank, stage_label = _classify_model_stage(mtype, base_model)
        m_df = df[df["model_type"] == mtype]
        y_true = pd.to_numeric(m_df["y_true"], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(m_df["yhat"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y_true) & np.isfinite(yhat)
        n_points = int(keep.sum())
        n_series = int(m_df["ts_id"].nunique())

        denom = float(np.sum(np.abs(y_true[keep]))) if n_points else 0.0
        pooled_wape = (
            float(np.sum(np.abs(y_true[keep] - yhat[keep])) / denom) if denom > 0 else float("nan")
        )
        mean_mae = float(np.mean(np.abs(y_true[keep] - yhat[keep]))) if n_points else float("nan")

        m_series_vec = np.array(
            [series_losses.get((tid, mtype), float("nan")) for tid in all_ts], dtype=float
        )
        mean_metric = (
            float(np.nanmean(m_series_vec)) if np.isfinite(m_series_vec).any() else float("nan")
        )

        both_series = np.isfinite(m_series_vec) & np.isfinite(base_series_vec)
        if mtype == base_model:
            win_rate = 0.5
            dm_stat, p_dm, p_wilcox = 0.0, 1.0, 1.0
        else:
            win_rate = (
                float(np.mean(m_series_vec[both_series] < base_series_vec[both_series]))
                if both_series.any()
                else float("nan")
            )
            # Choose the finest-grained paired sample with >= 5 observations:
            # 1) per-series losses when panel has >= 5 series;
            # 2) per-(series, fold) losses when >= 5 folds;
            # 3) pointwise absolute errors across (ts_id, fold_id, forecast_date).
            if int(both_series.sum()) >= 5:
                dm_stat, p_dm = diebold_mariano_test(m_series_vec, base_series_vec, h=1)
                p_wilcox = wilcoxon_pvalue(m_series_vec, base_series_vec)
            else:
                m_fold_vec = np.array(
                    [fold_losses.get((tid, fid, mtype), float("nan")) for tid, fid in all_folds],
                    dtype=float,
                )
                b_fold_vec = np.array(
                    [
                        fold_losses.get((tid, fid, base_model), float("nan"))
                        for tid, fid in all_folds
                    ],
                    dtype=float,
                )
                both_folds = np.isfinite(m_fold_vec) & np.isfinite(b_fold_vec)
                if int(both_folds.sum()) >= 5:
                    dm_stat, p_dm = diebold_mariano_test(m_fold_vec, b_fold_vec, h=1)
                    p_wilcox = wilcoxon_pvalue(m_fold_vec, b_fold_vec)
                elif mtype in point_pivot.columns and base_model in point_pivot.columns:
                    max_h = int(pd.to_numeric(m_df["horizon_step"], errors="coerce").max() or 1)
                    mp = point_pivot[mtype].to_numpy(dtype=float)
                    bp = point_pivot[base_model].to_numpy(dtype=float)
                    dm_stat, p_dm = diebold_mariano_test(mp, bp, h=max(1, min(max_h, 7)))
                    p_wilcox = wilcoxon_pvalue(mp, bp)
                else:
                    dm_stat, p_dm, p_wilcox = float("nan"), float("nan"), float("nan")

        fva_abs = (
            float(base_mean_metric - mean_metric)
            if math.isfinite(base_mean_metric) and math.isfinite(mean_metric)
            else float("nan")
        )
        fva_pct = (
            float(fva_abs / base_mean_metric)
            if math.isfinite(fva_abs) and base_mean_metric > 0
            else float("nan")
        )
        sig_05 = bool(
            mtype != base_model
            and math.isfinite(fva_abs)
            and fva_abs > 0.0
            and math.isfinite(p_dm)
            and p_dm < 0.05
        )
        dom_regime = (
            str(m_df["demand_class"].mode().iloc[0])
            if not m_df["demand_class"].dropna().empty
            else "all"
        )
        records.append(
            {
                "fva_stage_rank": stage_rank,
                "fva_stage": stage_label,
                "model_type": mtype,
                "demand_class": dom_regime,
                "n_series": n_series,
                "n_points": n_points,
                "metric": metric,
                "mean_metric": mean_metric,
                "pooled_wape": pooled_wape,
                "mean_mae": mean_mae,
                "baseline_model": base_model,
                "baseline_metric": base_mean_metric,
                "fva_abs": fva_abs,
                "fva_pct": fva_pct,
                "win_rate_vs_baseline": win_rate,
                "dm_stat": dm_stat,
                "p_value_dm": p_dm,
                "p_value_wilcoxon": p_wilcox,
                "significant_at_05": sig_05,
            }
        )

    out = pd.DataFrame.from_records(records, columns=cols)
    return out.sort_values(
        ["fva_stage_rank", "mean_metric"], ascending=[True, True], ignore_index=True
    )


def compute_horizon_decay(
    oof_df: pd.DataFrame,
    *,
    metric: str = "wape",
    baseline: str | None = None,
) -> pd.DataFrame:
    """Evaluate forecast accuracy, bias, and interval coverage by lead-time step ($h = 1 \\dots H$).

    Also identifies ``crossover_step`` ($h^*$) for each model: the first horizon step where the
    model's loss exceeds the ``baseline`` model's loss (or ``None`` if it beats the baseline across
    the entire horizon).
    """
    cols = [
        "model_type",
        "horizon_step",
        "n_obs",
        "metric",
        "metric_value",
        "mae",
        "rmse",
        "bias",
        "coverage_80",
        "baseline_model",
        "baseline_metric_value",
        "fva_vs_baseline_pct",
        "crossover_step",
    ]
    if oof_df is None or oof_df.empty:
        return pd.DataFrame(columns=cols)

    df = _ensure_oof_columns(oof_df)
    all_models = list(dict.fromkeys(df["model_type"].astype(str)))
    base_model = _resolve_baseline_model(all_models, baseline)

    step_loss: dict[tuple[str, int], float] = {}
    partial_rows: list[dict[str, Any]] = []
    for (mtype, step_raw), grp in df.groupby(["model_type", "horizon_step"], sort=True):
        step = int(str(step_raw))
        y_true = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(grp["yhat"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y_true) & np.isfinite(yhat)
        n_obs = int(keep.sum())
        if n_obs == 0:
            continue
        yt, yh = y_true[keep], yhat[keep]
        m_val = _cell_metric_loss(grp, metric)
        step_loss[(str(mtype), step)] = m_val
        err = yh - yt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        bias = float(np.mean(err))

        cov80 = float("nan")
        if "yhat_lower" in grp.columns and "yhat_upper" in grp.columns:
            lo = pd.to_numeric(grp["yhat_lower"], errors="coerce").to_numpy(dtype=float)[keep]
            hi = pd.to_numeric(grp["yhat_upper"], errors="coerce").to_numpy(dtype=float)[keep]
            b_keep = np.isfinite(lo) & np.isfinite(hi)
            if b_keep.any():
                cov80 = float(np.mean((yt[b_keep] >= lo[b_keep]) & (yt[b_keep] <= hi[b_keep])))

        partial_rows.append(
            {
                "model_type": str(mtype),
                "horizon_step": step,
                "n_obs": n_obs,
                "metric": metric,
                "metric_value": m_val,
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "coverage_80": cov80,
                "baseline_model": base_model,
            }
        )

    # Determine crossover step h* per model relative to base_model.
    crossover_by_model: dict[str, int | None] = {}
    for mtype in all_models:
        if mtype == base_model:
            crossover_by_model[mtype] = None
            continue
        steps_for_m = sorted(s for (m, s) in step_loss if m == mtype)
        cross: int | None = None
        for s in steps_for_m:
            ml = step_loss.get((mtype, s), float("nan"))
            bl = step_loss.get((base_model, s), float("nan"))
            if math.isfinite(ml) and math.isfinite(bl) and ml > bl:
                cross = s
                break
        crossover_by_model[mtype] = cross

    for row in partial_rows:
        mtype = row["model_type"]
        step = row["horizon_step"]
        ml = row["metric_value"]
        bl = step_loss.get((base_model, step), float("nan"))
        row["baseline_metric_value"] = bl
        row["fva_vs_baseline_pct"] = (
            float((bl - ml) / bl)
            if math.isfinite(ml) and math.isfinite(bl) and bl > 0
            else float("nan")
        )
        row["crossover_step"] = crossover_by_model.get(mtype)

    out = pd.DataFrame.from_records(partial_rows, columns=cols)
    return out.sort_values(["model_type", "horizon_step"], ignore_index=True)


def compute_bias_report(
    oof_df: pd.DataFrame,
    *,
    threshold_pct: float = 0.05,
    tracking_signal_limit: float = 4.0,
    min_consecutive_folds: int = 2,
) -> pd.DataFrame:
    """Audit every ``(ts_id, model_type)`` cell for chronic over- or under-forecasting bias.

    Computes signed error $\\hat{y}_t - y_t$, percentage bias $\\sum(\\hat{y}_t - y_t)/\\sum|y_t|$,
    the **Trigg / APICS Tracking Signal** $\\text{TS} = \\sum(\\hat{y}_t - y_t) / \\text{MAD}$, and
    the longest streak of consecutive folds with positive (over-forecasting) or negative
    (under-forecasting) fold-mean error.
    """
    cols = [
        "ts_id",
        "model_type",
        "n_obs",
        "n_folds",
        "mean_y_true",
        "mean_yhat",
        "mean_signed_error",
        "bias_pct",
        "mad",
        "tracking_signal",
        "max_consecutive_over_folds",
        "max_consecutive_under_folds",
        "bias_status",
    ]
    if oof_df is None or oof_df.empty:
        return pd.DataFrame(columns=cols)

    df = _ensure_oof_columns(oof_df)
    records: list[dict[str, Any]] = []
    for (tid, mtype), grp in df.groupby(["ts_id", "model_type"], sort=True):
        y_true = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(grp["yhat"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y_true) & np.isfinite(yhat)
        if not keep.any():
            continue
        yt, yh = y_true[keep], yhat[keep]
        signed_err = yh - yt
        n_obs = int(keep.sum())
        mean_yt = float(np.mean(yt))
        mean_yh = float(np.mean(yh))
        mean_se = float(np.mean(signed_err))
        abs_sum = float(np.sum(np.abs(yt)))
        bias_pct = float(np.sum(signed_err) / abs_sum) if abs_sum > 0 else 0.0
        mad = float(np.mean(np.abs(signed_err)))
        tracking_sig = float(np.sum(signed_err) / mad) if mad > 1e-12 else 0.0

        # Consecutive fold streaks.
        sub_valid = grp.loc[keep].copy()
        sub_valid["_se"] = signed_err
        fold_means = sub_valid.groupby("fold_id", sort=True)["_se"].mean().to_numpy(dtype=float)
        max_over, max_under, cur_over, cur_under = 0, 0, 0, 0
        for fm in fold_means:
            if fm > 0:
                cur_over += 1
                cur_under = 0
                max_over = max(max_over, cur_over)
            elif fm < 0:
                cur_under += 1
                cur_over = 0
                max_under = max(max_under, cur_under)
            else:
                cur_over, cur_under = 0, 0

        exceeds_mag = abs(bias_pct) >= threshold_pct
        exceeds_ts = abs(tracking_sig) >= tracking_signal_limit
        if exceeds_mag and (exceeds_ts or max_over >= min_consecutive_folds) and bias_pct > 0:
            status = "over_forecasting"
        elif exceeds_mag and (exceeds_ts or max_under >= min_consecutive_folds) and bias_pct < 0:
            status = "under_forecasting"
        else:
            status = "unbiased"

        records.append(
            {
                "ts_id": str(tid),
                "model_type": str(mtype),
                "n_obs": n_obs,
                "n_folds": len(fold_means),
                "mean_y_true": mean_yt,
                "mean_yhat": mean_yh,
                "mean_signed_error": mean_se,
                "bias_pct": bias_pct,
                "mad": mad,
                "tracking_signal": tracking_sig,
                "max_consecutive_over_folds": max_over,
                "max_consecutive_under_folds": max_under,
                "bias_status": status,
            }
        )

    out = pd.DataFrame.from_records(records, columns=cols)
    return out.sort_values(
        "tracking_signal", key=lambda s: s.abs(), ascending=False, ignore_index=True
    )


def _ar1_sum_scale(sigmas: np.ndarray, rho: float) -> float:
    """Standard deviation of $\\sum_{i=1}^k e_i$ under AR(1) correlation $\\rho$."""
    k = len(sigmas)
    if k == 0:
        return 0.0
    if k == 1:
        return float(sigmas[0])
    r = float(np.clip(rho, -0.95, 0.98))
    var = float(np.sum(sigmas**2))
    for i in range(k):
        for j in range(i + 1, k):
            var += 2.0 * (r ** (j - i)) * float(sigmas[i]) * float(sigmas[j])
    return math.sqrt(max(var, 0.0))


def _estimate_oof_error_rho(oof_df: pd.DataFrame | None) -> dict[tuple[str, str], float]:
    """Estimate lag-1 residual autocorrelation $\\hat{\\rho}_1$ per ``(ts_id, model_type)``."""
    if oof_df is None or oof_df.empty:
        return {}
    df = _ensure_oof_columns(oof_df)
    rho_map: dict[tuple[str, str], float] = {}
    for (tid, mtype), grp in df.groupby(["ts_id", "model_type"], sort=False):
        ordered = grp.sort_values(["fold_id", "forecast_date"])
        err = (
            (
                pd.to_numeric(ordered["y_true"], errors="coerce")
                - pd.to_numeric(ordered["yhat"], errors="coerce")
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        err = err[np.isfinite(err)]
        if err.size >= 4:
            e0, e1 = err[:-1] - np.mean(err[:-1]), err[1:] - np.mean(err[1:])
            denom = float(np.sqrt(np.sum(e0**2) * np.sum(e1**2)))
            if denom > 1e-12:
                rho_map[(str(tid), str(mtype))] = float(
                    np.clip(np.sum(e0 * e1) / denom, -0.5, 0.95)
                )
    return rho_map


def rollup_time(
    predictions_df: pd.DataFrame,
    *,
    freq: str = "W",
    oof_df: pd.DataFrame | None = None,
    level: float = 0.80,
) -> pd.DataFrame:
    """Aggregate forecasts (and optional actuals) to coarser calendar buckets or lead-time windows
    with autocorrelation-aware prediction intervals.

    Why this exists: daily point forecasts (`yhat`) sum linearly into Weekly (`"W"`) or Monthly
    (`"MS"` / `"M"`) totals, **but daily `yhat_lower` and `yhat_upper` cannot be summed linearly**
    (summing daily bounds assumes perfect $\\rho = 1.0$ error correlation across days, inflating a
    7-day interval by up to $2.6\\times$). `rollup_time` estimates the lag-1 residual
    autocorrelation $\\hat{\\rho}_1$ from ``oof_df`` (defaulting to a conservative $\\rho = 0.35$
    when ``oof_df`` is omitted) and computes period-level uncertainty via the AR(1) sum variance
    $\\text{Var}(\\sum_{i=1}^k e_i) = \\sum_i \\sigma_i^2 + 2\\sum_{i<j} \\hat{\\rho}_1^{j-i}
    \\sigma_i\\sigma_j$, reporting both the calibrated bounds (``yhat_lower``, ``yhat_upper``) and
    the naive linear sum (``naive_sum_lower``, ``naive_sum_upper``) for side-by-side comparison.
    """
    cols = [
        "ts_id",
        "model_type",
        "period_start",
        "period_end",
        "n_steps",
        "y_true",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "naive_sum_lower",
        "naive_sum_upper",
        "error_autocorr_rho",
    ]
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=cols)

    df = predictions_df.copy()
    if "segment" in df.columns and (df["segment"] == "forecast").any():
        df = df[df["segment"] == "forecast"].copy()
    date_col = (
        "ds" if "ds" in df.columns else ("forecast_date" if "forecast_date" in df.columns else None)
    )
    if date_col is None:
        raise ValueError("predictions_df must contain a 'ds' or 'forecast_date' column")
    if "ts_id" not in df.columns:
        df["ts_id"] = "series_0"
    if "model_type" not in df.columns:
        df["model_type"] = "model"

    df["_dt"] = pd.to_datetime(df[date_col])
    df = df.sort_values(["ts_id", "model_type", "_dt"])

    norm_freq = {"M": "ME", "Q": "QE"}.get(freq, freq)
    df["_period"] = df["_dt"].dt.to_period(
        norm_freq if norm_freq not in ("MS", "QS", "ME", "QE") else norm_freq[0]
    )

    rho_map = _estimate_oof_error_rho(oof_df)
    alpha_tail = 0.5 + 0.5 * float(level)
    z_val = float(stats.norm.ppf(alpha_tail))
    z_80 = float(stats.norm.ppf(0.90))

    records: list[dict[str, Any]] = []
    for (tid, mtype, _per), grp in df.groupby(["ts_id", "model_type", "_period"], sort=True):
        yh = pd.to_numeric(grp["yhat"], errors="coerce").to_numpy(dtype=float)
        yt = (
            pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
            if "y_true" in grp.columns
            else np.full(len(grp), np.nan)
        )
        yl = (
            pd.to_numeric(grp["yhat_lower"], errors="coerce").to_numpy(dtype=float)
            if "yhat_lower" in grp.columns
            else yh.copy()
        )
        yu = (
            pd.to_numeric(grp["yhat_upper"], errors="coerce").to_numpy(dtype=float)
            if "yhat_upper" in grp.columns
            else yh.copy()
        )

        sum_yh = float(np.nansum(yh))
        sum_yt = float(np.nansum(yt)) if np.isfinite(yt).any() else float("nan")
        naive_lo = float(np.nansum(yl))
        naive_hi = float(np.nansum(yu))

        # Step-level sigma_i from (yhat_upper - yhat_lower) / (2 * z_80), preserving asymmetry.
        half_lo = np.maximum(yh - yl, 0.0)
        half_hi = np.maximum(yu - yh, 0.0)
        sigma_lo = np.where(np.isfinite(half_lo), half_lo / z_80, 0.0)
        sigma_hi = np.where(np.isfinite(half_hi), half_hi / z_80, 0.0)

        rho = rho_map.get((str(tid), str(mtype)), 0.35)
        scale_lo = _ar1_sum_scale(sigma_lo, rho)
        scale_hi = _ar1_sum_scale(sigma_hi, rho)

        records.append(
            {
                "ts_id": str(tid),
                "model_type": str(mtype),
                "period_start": grp["_dt"].min(),
                "period_end": grp["_dt"].max(),
                "n_steps": len(grp),
                "y_true": sum_yt,
                "yhat": sum_yh,
                "yhat_lower": float(sum_yh - z_val * scale_lo),
                "yhat_upper": float(sum_yh + z_val * scale_hi),
                "naive_sum_lower": naive_lo,
                "naive_sum_upper": naive_hi,
                "error_autocorr_rho": rho,
            }
        )

    return pd.DataFrame.from_records(records, columns=cols)


def plot_fva(
    fva_df: pd.DataFrame,
    *,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot the Forecast Value Added (FVA) accuracy waterfall by model and modeling tier."""
    from .errors import require_extra

    require_extra("notebook", purpose="Plotting FVA waterfall")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if fva_df is None or fva_df.empty:
        ax.set_title(title or "Forecast Value Added (no backtest OOF rows)")
        return ax

    ordered = fva_df.sort_values(["fva_stage_rank", "mean_metric"], ascending=[True, False])
    labels = [
        f"{r.model_type} ({r.fva_stage})" + (" *" if r.significant_at_05 else "")
        for r in ordered.itertuples(index=False)
    ]
    vals = ordered["fva_pct"].fillna(0.0).to_numpy(dtype=float) * 100.0
    colors = ["#2E7D32" if v > 0 else ("#757575" if v == 0 else "#C62828") for v in vals]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color=colors, alpha=0.88)
    ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"FVA % Improvement vs. {ordered['baseline_model'].iloc[0]} (* p < 0.05)")
    ax.set_title(title or "Forecast Value Added (FVA) Ladder")
    ax.grid(axis="x", alpha=0.25)
    return ax


def plot_horizon_decay(
    decay_df: pd.DataFrame,
    *,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot forecast error decay across lead-time steps $h = 1 \\dots H$ per model."""
    from .errors import require_extra

    require_extra("notebook", purpose="Plotting horizon decay")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if decay_df is None or decay_df.empty:
        ax.set_title(title or "Horizon Error Decay (no backtest OOF rows)")
        return ax

    metric_name = str(decay_df["metric"].iloc[0]) if "metric" in decay_df.columns else "loss"
    for mtype, grp in decay_df.groupby("model_type", sort=False):
        ordered = grp.sort_values("horizon_step")
        ax.plot(
            ordered["horizon_step"],
            ordered["metric_value"],
            marker="o",
            markersize=4,
            label=str(mtype),
        )
    ax.set_xlabel("Forecast Horizon Step (h)")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(title or f"Lead-Time Horizon Accuracy Decay ({metric_name.upper()})")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.25)
    return ax
