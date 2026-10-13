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

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import compute_metrics, loss_of

__all__ = [
    "BASELINE_CANDIDATES",
    "compute_bias_report",
    "compute_exceptions_queue",
    "compute_fva",
    "compute_horizon_decay",
    "compute_realized_accuracy",
    "compute_revision_stability",
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
    from .models import base_model_name, get_model

    root = base_model_name(model_type)
    if model_type == baseline_model or root.startswith("naive_"):
        return _STAGE_ORDER["baseline"]
    if model_type.startswith("ensemble_"):
        return _STAGE_ORDER["ensemble"]
    try:
        fam = get_model(model_type).family
        return _STAGE_ORDER.get(fam, (2, "Statistical"))
    except Exception:  # noqa: BLE001 - unknown/custom model names default to Statistical tier
        return _STAGE_ORDER["statistical"]


def _match_sibling_baselines(
    model_type: str, all_models: Sequence[str]
) -> tuple[str | None, str | None]:
    """Find matched same-model univariate and local baselines for ``model_type`` in ``all_models``.

    Returns ``(univariate_baseline, local_baseline)``.
    """
    from .models import base_model_name

    if model_type.startswith("ensemble_"):
        return None, None
    bname = base_model_name(model_type)
    siblings = [m for m in all_models if base_model_name(m) == bname and m != model_type]
    if not siblings:
        return None, None

    tag = model_type.split("@", 1)[1] if "@" in model_type else ""
    is_univariate = (
        tag in ("univariate", "cov_none")
        or tag.endswith(("_univariate", "_cov_none"))
        or (not tag and f"{bname}@xreg" in siblings)
    )
    has_global_uni_sibling = any(
        s in siblings for s in (f"{bname}@global_univariate", f"{bname}@global_cov_none")
    )
    has_local_override_sibling = any(s.startswith(f"{bname}@local") for s in siblings)
    inherits_primary_scope = not tag or not tag.startswith(("local", "global", "hybrid"))
    inferred_global_primary = (
        inherits_primary_scope
        and (has_global_uni_sibling or has_local_override_sibling)
        and f"{bname}@global_cov_all" not in siblings
        and f"{bname}@global" not in siblings
    )
    is_global = tag == "global" or tag.startswith("global_") or inferred_global_primary
    is_hybrid = tag == "hybrid" or tag.startswith("hybrid_")

    uni_base: str | None = None
    if not is_univariate:
        if is_global:
            cands = [
                f"{bname}@global_univariate",
                f"{bname}@global_cov_none",
                f"{bname}@univariate",
                f"{bname}@cov_none",
                f"{bname}@local_univariate",
                f"{bname}@local_cov_none",
            ]
        elif is_hybrid:
            cands = [
                f"{bname}@hybrid_univariate",
                f"{bname}@hybrid_cov_none",
                f"{bname}@univariate",
                f"{bname}@cov_none",
                f"{bname}@local_univariate",
                f"{bname}@local_cov_none",
            ]
        else:
            cands = [
                f"{bname}@local_univariate",
                f"{bname}@local_cov_none",
                f"{bname}@univariate",
                f"{bname}@cov_none",
            ]
            if f"{bname}@xreg" in all_models:
                cands.append(bname)
        uni_base = next((c for c in cands if c in siblings), None)

    local_base: str | None = None
    if is_global or is_hybrid:
        prefix = "global_" if is_global else "hybrid_"
        if tag.startswith(prefix):
            cov_suffix = tag[len(prefix) :]
        elif tag in ("univariate", "cov_none"):
            cov_suffix = "univariate"
        elif tag.startswith("cov_"):
            cov_suffix = tag
        else:
            cov_suffix = "cov_all"
        cands_local = [f"{bname}@local_{cov_suffix}", f"{bname}@{cov_suffix}"]
        if cov_suffix in ("cov_none", "univariate"):
            cands_local.extend(
                [
                    f"{bname}@local_univariate",
                    f"{bname}@local_cov_none",
                    f"{bname}@univariate",
                    f"{bname}@cov_none",
                ]
            )
        elif cov_suffix == "cov_all":
            cands_local.extend(
                [
                    bname,
                    f"{bname}@local",
                    f"{bname}@local_univariate",
                    f"{bname}@local_cov_none",
                ]
            )
        local_base = next((c for c in cands_local if c in siblings and c != model_type), None)

    return uni_base, local_base


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


def _cell_metric_loss(
    group: pd.DataFrame,
    metric: str,
    *,
    y_train: np.ndarray | None = None,
) -> float:
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
    eff_y_train = (
        y_train
        if (y_train is not None and len(y_train) >= 2)
        else (y_true[keep] if len(y_true[keep]) >= 2 else None)
    )
    m_dict = compute_metrics(
        y_true[keep],
        yhat[keep],
        y_train=eff_y_train,
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
    from .models import base_model_name

    cols = [
        "fva_stage_rank",
        "fva_stage",
        "model_type",
        "base_model",
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
        "univariate_baseline_model",
        "covariate_fva_abs",
        "covariate_fva_pct",
        "covariate_p_value_dm",
        "covariate_significant_at_05",
        "local_baseline_model",
        "pooling_fva_abs",
        "pooling_fva_pct",
        "pooling_p_value_dm",
        "pooling_significant_at_05",
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

    def _compare_pair(
        cand_model: str,
        ref_model: str | None,
        cand_series_vec: np.ndarray,
        cand_mean: float,
        max_h: int,
    ) -> tuple[float, float, float, float, float, float, float, bool]:
        if ref_model is None:
            return (
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                False,
            )
        ref_series_vec = np.array(
            [series_losses.get((tid, ref_model), float("nan")) for tid in all_ts], dtype=float
        )
        ref_mean = (
            float(np.nanmean(ref_series_vec)) if np.isfinite(ref_series_vec).any() else float("nan")
        )
        both_series = np.isfinite(cand_series_vec) & np.isfinite(ref_series_vec)
        if cand_model == ref_model:
            win_rate = 0.5
            dm_stat, p_dm, p_wilcox = 0.0, 1.0, 1.0
        else:
            win_rate = (
                float(np.mean(cand_series_vec[both_series] < ref_series_vec[both_series]))
                if both_series.any()
                else float("nan")
            )
            if int(both_series.sum()) >= 5:
                dm_stat, p_dm = diebold_mariano_test(cand_series_vec, ref_series_vec, h=1)
                p_wilcox = wilcoxon_pvalue(cand_series_vec, ref_series_vec)
            else:
                m_fold_vec = np.array(
                    [
                        fold_losses.get((tid, fid, cand_model), float("nan"))
                        for tid, fid in all_folds
                    ],
                    dtype=float,
                )
                b_fold_vec = np.array(
                    [
                        fold_losses.get((tid, fid, ref_model), float("nan"))
                        for tid, fid in all_folds
                    ],
                    dtype=float,
                )
                both_folds = np.isfinite(m_fold_vec) & np.isfinite(b_fold_vec)
                if int(both_folds.sum()) >= 5:
                    dm_stat, p_dm = diebold_mariano_test(m_fold_vec, b_fold_vec, h=1)
                    p_wilcox = wilcoxon_pvalue(m_fold_vec, b_fold_vec)
                elif cand_model in point_pivot.columns and ref_model in point_pivot.columns:
                    mp = point_pivot[cand_model].to_numpy(dtype=float)
                    bp = point_pivot[ref_model].to_numpy(dtype=float)
                    dm_stat, p_dm = diebold_mariano_test(mp, bp, h=max(1, min(max_h, 7)))
                    p_wilcox = wilcoxon_pvalue(mp, bp)
                else:
                    dm_stat, p_dm, p_wilcox = float("nan"), float("nan"), float("nan")
        fva_abs = (
            float(ref_mean - cand_mean)
            if math.isfinite(ref_mean) and math.isfinite(cand_mean)
            else float("nan")
        )
        fva_pct = (
            float(fva_abs / ref_mean) if math.isfinite(fva_abs) and ref_mean > 0 else float("nan")
        )
        sig_05 = bool(
            cand_model != ref_model
            and math.isfinite(fva_abs)
            and fva_abs > 0.0
            and math.isfinite(p_dm)
            and p_dm < 0.05
        )
        return ref_mean, fva_abs, fva_pct, win_rate, dm_stat, p_dm, p_wilcox, sig_05

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
        max_h = int(pd.to_numeric(m_df["horizon_step"], errors="coerce").max() or 1)

        (
            base_mean_metric,
            fva_abs,
            fva_pct,
            win_rate,
            dm_stat,
            p_dm,
            p_wilcox,
            sig_05,
        ) = _compare_pair(mtype, base_model, m_series_vec, mean_metric, max_h)

        uni_base, local_base = _match_sibling_baselines(mtype, all_models)
        (
            _,
            cov_fva_abs,
            cov_fva_pct,
            _,
            _,
            cov_p_dm,
            _,
            cov_sig_05,
        ) = _compare_pair(mtype, uni_base, m_series_vec, mean_metric, max_h)
        (
            _,
            pool_fva_abs,
            pool_fva_pct,
            _,
            _,
            pool_p_dm,
            _,
            pool_sig_05,
        ) = _compare_pair(mtype, local_base, m_series_vec, mean_metric, max_h)

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
                "base_model": base_model_name(mtype),
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
                "univariate_baseline_model": uni_base,
                "covariate_fva_abs": cov_fva_abs,
                "covariate_fva_pct": cov_fva_pct,
                "covariate_p_value_dm": cov_p_dm,
                "covariate_significant_at_05": cov_sig_05,
                "local_baseline_model": local_base,
                "pooling_fva_abs": pool_fva_abs,
                "pooling_fva_pct": pool_fva_pct,
                "pooling_p_value_dm": pool_p_dm,
                "pooling_significant_at_05": pool_sig_05,
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


def compute_realized_accuracy(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    *,
    oof_df: pd.DataFrame | None = None,
    metric: str = "wape",
    drift_threshold: float = 1.5,
) -> pd.DataFrame:
    """Join published forward forecasts with newly arrived actuals on ``(ts_id, date)`` and score
    realized accuracy, interval coverage, and accuracy drift relative to backtest expectations.

    Args:
        predictions_df: Forward predictions frame containing ``ts_id``, ``model_type``,
            ``forecast_date`` (or ``ds``), ``yhat`` (or ``forecast``), and optional ``yhat_lower``
            / ``yhat_upper`` (or ``lower_bound`` / ``upper_bound``).
        actuals_df: Newly arrived actuals containing ``ts_id``, ``ds`` (or ``forecast_date``), and
            ``y`` (or ``y_true`` / ``actual``).
        oof_df: Optional backtest OOF frame from the same run to compare realized error against
            expected backtest error (`drift_ratio = realized_metric / backtest_metric`).
        metric: Primary accuracy metric (default ``"wape"``).
        drift_threshold: Ratio threshold above which ``drift_flagged`` is set True (default 1.5).
    """
    cols = [
        "ts_id",
        "model_type",
        "n_realized",
        "metric",
        "realized_metric",
        "realized_mae",
        "realized_rmse",
        "realized_bias",
        "realized_coverage_80",
        "backtest_metric",
        "drift_ratio",
        "drift_flagged",
    ]
    if predictions_df is None or predictions_df.empty or actuals_df is None or actuals_df.empty:
        return pd.DataFrame(columns=cols)

    pred = predictions_df.copy()
    if "segment" in pred.columns and (pred["segment"] == "forecast").any():
        pred = pred[pred["segment"] == "forecast"].copy()
    p_date = "forecast_date" if "forecast_date" in pred.columns else "ds"
    p_yhat = "yhat" if "yhat" in pred.columns else "forecast"
    p_lo = (
        "yhat_lower"
        if "yhat_lower" in pred.columns
        else ("lower_bound" if "lower_bound" in pred.columns else None)
    )
    p_hi = (
        "yhat_upper"
        if "yhat_upper" in pred.columns
        else ("upper_bound" if "upper_bound" in pred.columns else None)
    )
    if "ts_id" not in pred.columns:
        pred["ts_id"] = "series_0"
    if "model_type" not in pred.columns:
        pred["model_type"] = "model"

    act = actuals_df.copy()
    a_date = "ds" if "ds" in act.columns else "forecast_date"
    a_y = "y" if "y" in act.columns else ("y_true" if "y_true" in act.columns else "actual")
    if "ts_id" not in act.columns:
        act["ts_id"] = "series_0"

    pred["_dt"] = pd.to_datetime(pred[p_date])
    act["_dt"] = pd.to_datetime(act[a_date])
    act_dedup = (
        act[["ts_id", "_dt", a_y]]
        .drop_duplicates(subset=["ts_id", "_dt"], keep="last")
        .rename(columns={a_y: "_y_actual"})
    )
    merged = pred.merge(act_dedup, on=["ts_id", "_dt"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=cols)

    bt_loss_map: dict[tuple[str, str], float] = {}
    if oof_df is not None and not oof_df.empty:
        oof_norm = _ensure_oof_columns(oof_df)
        for (tid, mtype), grp in oof_norm.groupby(["ts_id", "model_type"], sort=False):
            bt_loss_map[(str(tid), str(mtype))] = _cell_metric_loss(grp, metric)

    records: list[dict[str, Any]] = []
    for (tid, mtype), grp in merged.groupby(["ts_id", "model_type"], sort=True):
        yt = pd.to_numeric(grp["_y_actual"], errors="coerce").to_numpy(dtype=float)
        yh = pd.to_numeric(grp[p_yhat], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(yt) & np.isfinite(yh)
        n_real = int(keep.sum())
        if n_real == 0:
            continue
        yt_k, yh_k = yt[keep], yh[keep]
        lo_k = (
            pd.to_numeric(grp[p_lo], errors="coerce").to_numpy(dtype=float)[keep]
            if p_lo is not None
            else None
        )
        hi_k = (
            pd.to_numeric(grp[p_hi], errors="coerce").to_numpy(dtype=float)[keep]
            if p_hi is not None
            else None
        )
        m_panel = compute_metrics(
            yt_k,
            yh_k,
            lower=lo_k if lo_k is not None and np.isfinite(lo_k).any() else None,
            upper=hi_k if hi_k is not None and np.isfinite(hi_k).any() else None,
        )
        real_loss = float(loss_of(metric, m_panel.get(metric, float("nan"))))
        bt_loss = bt_loss_map.get((str(tid), str(mtype)), float("nan"))
        drift = (
            float(real_loss / bt_loss)
            if math.isfinite(real_loss) and math.isfinite(bt_loss) and bt_loss > 1e-12
            else float("nan")
        )
        flagged = bool(math.isfinite(drift) and drift >= float(drift_threshold))
        records.append(
            {
                "ts_id": str(tid),
                "model_type": str(mtype),
                "n_realized": n_real,
                "metric": metric,
                "realized_metric": real_loss,
                "realized_mae": float(m_panel.get("mae", float("nan"))),
                "realized_rmse": float(m_panel.get("rmse", float("nan"))),
                "realized_bias": float(m_panel.get("bias", float("nan"))),
                "realized_coverage_80": float(m_panel.get("coverage", float("nan"))),
                "backtest_metric": bt_loss,
                "drift_ratio": drift,
                "drift_flagged": flagged,
            }
        )

    out = pd.DataFrame.from_records(records, columns=cols)
    return out.sort_values(
        ["drift_flagged", "realized_metric"], ascending=[False, False], ignore_index=True
    )


def compute_revision_stability(
    predictions_by_run: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    jump_threshold_pct: float = 0.30,
) -> pd.DataFrame:
    """Measure cross-run forecast revision stability ("whipsaw") across overlapping target dates.

    Accepts either a mapping ``{run_id: predictions_df}`` (ordered oldest-to-newest) or a single
    concatenated DataFrame containing a ``run_id`` column. Computes per ``(ts_id, model_type)``:

    - ``n_runs`` / ``n_overlapping_dates``
    - ``mean_forecast``: average forecast magnitude across overlapping dates
    - ``mar``: Mean Absolute Revision $|\\hat{y}^{(r)}_t - \\hat{y}^{(r-1)}_t|$
    - ``revision_cov``: Coefficient of Variation of revisions ($\\text{std}(\\Delta \\hat{y}) /
      |\\overline{\\hat{y}}|$)
    - ``max_single_step_jump_pct``: largest single-date percentage jump between consecutive runs
    - ``whipsaw_flagged``: True when ``max_single_step_jump_pct >= jump_threshold_pct``
    """
    cols = [
        "ts_id",
        "model_type",
        "n_runs",
        "n_overlapping_dates",
        "mean_forecast",
        "mar",
        "revision_cov",
        "max_single_step_jump_pct",
        "whipsaw_flagged",
    ]
    frames: list[pd.DataFrame] = []
    run_order: list[str] = []
    if isinstance(predictions_by_run, Mapping):
        for idx, (rid, f_df) in enumerate(predictions_by_run.items()):
            if f_df is None or f_df.empty:
                continue
            sub = f_df.copy()
            sub["_run_order"] = idx
            sub["run_id"] = str(rid)
            frames.append(sub)
            run_order.append(str(rid))
    elif isinstance(predictions_by_run, pd.DataFrame) and not predictions_by_run.empty:
        raw_df = predictions_by_run.copy()
        if "run_id" not in raw_df.columns:
            return pd.DataFrame(columns=cols)
        for idx, rid in enumerate(dict.fromkeys(raw_df["run_id"].astype(str))):
            sub = raw_df[raw_df["run_id"].astype(str) == rid].copy()
            sub["_run_order"] = idx
            frames.append(sub)
            run_order.append(rid)

    if len(frames) < 2:
        return pd.DataFrame(columns=cols)

    combined = pd.concat(frames, ignore_index=True)
    if "segment" in combined.columns and (combined["segment"] == "forecast").any():
        combined = combined[combined["segment"] == "forecast"].copy()
    date_col = "forecast_date" if "forecast_date" in combined.columns else "ds"
    yhat_col = "yhat" if "yhat" in combined.columns else "forecast"
    if "ts_id" not in combined.columns:
        combined["ts_id"] = "series_0"
    if "model_type" not in combined.columns:
        combined["model_type"] = "model"

    combined["_dt"] = pd.to_datetime(combined[date_col])
    combined["_yh"] = pd.to_numeric(combined[yhat_col], errors="coerce")

    records: list[dict[str, Any]] = []
    for (tid, mtype), grp in combined.groupby(["ts_id", "model_type"], sort=True):
        piv = grp.pivot_table(index="_dt", columns="_run_order", values="_yh", aggfunc="mean")
        if piv.shape[1] < 2:
            continue
        diffs: list[float] = []
        pct_jumps: list[float] = []
        overlap_dates: set[Any] = set()
        run_cols = sorted(piv.columns)
        for c_prev, c_curr in itertools.pairwise(run_cols):
            pair = piv[[c_prev, c_curr]].dropna()
            if pair.empty:
                continue
            overlap_dates.update(pair.index.tolist())
            prev_v = pair[c_prev].to_numpy(dtype=float)
            curr_v = pair[c_curr].to_numpy(dtype=float)
            delta = curr_v - prev_v
            diffs.extend(np.abs(delta).tolist())
            denom = np.maximum(np.abs(prev_v), 1e-6)
            pct_jumps.extend((np.abs(delta) / denom).tolist())

        if not diffs:
            continue
        mean_fc = float(np.nanmean(piv.to_numpy(dtype=float)))
        mar = float(np.mean(diffs))
        rev_std = float(np.std(diffs))
        rev_cov = rev_std / max(abs(mean_fc), 1e-6)
        max_jump = float(np.max(pct_jumps))
        flagged = bool(max_jump >= float(jump_threshold_pct))
        records.append(
            {
                "ts_id": str(tid),
                "model_type": str(mtype),
                "n_runs": int(piv.shape[1]),
                "n_overlapping_dates": len(overlap_dates),
                "mean_forecast": mean_fc,
                "mar": mar,
                "revision_cov": rev_cov,
                "max_single_step_jump_pct": max_jump,
                "whipsaw_flagged": flagged,
            }
        )

    out = pd.DataFrame.from_records(records, columns=cols)
    return out.sort_values(
        ["whipsaw_flagged", "max_single_step_jump_pct"],
        ascending=[False, False],
        ignore_index=True,
    )


def compute_exceptions_queue(
    *,
    oof_df: pd.DataFrame | None = None,
    predictions_df: pd.DataFrame | None = None,
    realized_df: pd.DataFrame | None = None,
    stability_df: pd.DataFrame | None = None,
    floor: float | None = None,
    cap: float | None = None,
    metric: str = "wape",
    baseline: str | None = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """Build the Top-$N$ Exception Triage Queue across 5 operational forecasting rules.

    Evaluates each ``(ts_id, model_type)`` cell and assigns a composite ``severity_score`` and
    plain-English ``exception_reasons`` across:

    1. ``worse_than_naive``: Model's held-out backtest loss exceeds the naive benchmark on this
       series.
    2. ``chronic_bias``: Trigg Tracking Signal $|\\text{TS}| \\ge 4.0$ or $\\ge 2$ consecutive
       biased folds (chronic over- or under-forecasting).
    3. ``realized_drift``: Realized live error exceeds $1.5\\times$ expected backtest error.
    4. ``whipsaw_jump``: Cross-run forecast revision jump $\\ge 30\\%$ on an overlapping date.
    5. ``capacity_ceiling_hit``: Forward forecast hits or clips against physical ``floor`` / ``cap``
       bounds on $\\ge 20\\%$ of horizon steps.
    """
    cols = [
        "ts_id",
        "model_type",
        "severity_score",
        "n_flags",
        "exception_flags",
        "exception_reasons",
        "metric_loss",
        "baseline_loss",
        "bias_pct",
        "tracking_signal",
        "drift_ratio",
        "max_jump_pct",
    ]
    cell_flags: dict[tuple[str, str], list[str]] = {}
    cell_reasons: dict[tuple[str, str], list[str]] = {}
    cell_sev: dict[tuple[str, str], float] = {}
    cell_meta: dict[tuple[str, str], dict[str, float]] = {}

    def _touch(key: tuple[str, str]) -> None:
        if key not in cell_flags:
            cell_flags[key] = []
            cell_reasons[key] = []
            cell_sev[key] = 0.0
            cell_meta[key] = {
                "metric_loss": float("nan"),
                "baseline_loss": float("nan"),
                "bias_pct": float("nan"),
                "tracking_signal": float("nan"),
                "drift_ratio": float("nan"),
                "max_jump_pct": float("nan"),
            }

    # 1 & 2. Backtest OOF checks: worse_than_naive + chronic_bias
    if oof_df is not None and not oof_df.empty:
        oof_norm = _ensure_oof_columns(oof_df)
        all_models = list(dict.fromkeys(oof_norm["model_type"].astype(str)))
        base_m = _resolve_baseline_model(all_models, baseline)

        losses: dict[tuple[str, str], float] = {}
        for (tid, mtype), grp in oof_norm.groupby(["ts_id", "model_type"], sort=False):
            losses[(str(tid), str(mtype))] = _cell_metric_loss(grp, metric)

        for (tid_s, mtype_s), m_loss in losses.items():
            b_loss = losses.get((tid_s, base_m), float("nan"))
            key = (tid_s, mtype_s)
            _touch(key)
            cell_meta[key]["metric_loss"] = m_loss
            cell_meta[key]["baseline_loss"] = b_loss
            if (
                mtype_s != base_m
                and math.isfinite(m_loss)
                and math.isfinite(b_loss)
                and m_loss > b_loss * 1.02
            ):
                ratio = m_loss / max(b_loss, 1e-6)
                cell_flags[key].append("worse_than_naive")
                cell_reasons[key].append(
                    f"Backtest {metric.upper()} ({m_loss:.3f}) is worse than {base_m} "
                    f"({b_loss:.3f}, {ratio:.2f}x)."
                )
                cell_sev[key] += min(40.0, 15.0 * ratio)

        bias_df = compute_bias_report(oof_norm)
        for b_row in bias_df.to_dict(orient="records"):
            key = (str(b_row["ts_id"]), str(b_row["model_type"]))
            _touch(key)
            b_pct = float(b_row["bias_pct"])
            t_sig = float(b_row["tracking_signal"])
            cell_meta[key]["bias_pct"] = b_pct
            cell_meta[key]["tracking_signal"] = t_sig
            if str(b_row["bias_status"]) != "unbiased":
                direction = (
                    "over-forecasting"
                    if b_row["bias_status"] == "over_forecasting"
                    else "under-forecasting"
                )
                cell_flags[key].append("chronic_bias")
                cell_reasons[key].append(
                    f"Chronic {direction}: bias={b_pct * 100:+.1f}%, Trigg TS={t_sig:+.1f}."
                )
                cell_sev[key] += min(35.0, 5.0 * abs(t_sig))

    # 3. Realized drift check
    if realized_df is not None and not realized_df.empty:
        for row in realized_df.itertuples(index=False):
            key = (str(row.ts_id), str(row.model_type))
            _touch(key)
            d_ratio = float(getattr(row, "drift_ratio", float("nan")))
            cell_meta[key]["drift_ratio"] = d_ratio
            if bool(getattr(row, "drift_flagged", False)):
                cell_flags[key].append("realized_drift")
                cell_reasons[key].append(
                    f"Live accuracy drift: realized loss is {d_ratio:.2f}x backtest expectation."
                )
                cell_sev[key] += min(45.0, 20.0 * d_ratio)

    # 4. Whipsaw revision stability check
    if stability_df is not None and not stability_df.empty:
        for row in stability_df.itertuples(index=False):
            key = (str(row.ts_id), str(row.model_type))
            _touch(key)
            jump = float(getattr(row, "max_single_step_jump_pct", float("nan")))
            cell_meta[key]["max_jump_pct"] = jump
            if bool(getattr(row, "whipsaw_flagged", False)):
                cell_flags[key].append("whipsaw_jump")
                cell_reasons[key].append(
                    f"Forecast whipsaw: {jump * 100:.1f}% single-step jump across consecutive runs."
                )
                cell_sev[key] += min(30.0, 40.0 * jump)

    # 5. Physical capacity ceiling / floor hit check
    if (
        predictions_df is not None
        and not predictions_df.empty
        and (floor is not None or cap is not None)
    ):
        pred = predictions_df.copy()
        if "segment" in pred.columns and (pred["segment"] == "forecast").any():
            pred = pred[pred["segment"] == "forecast"].copy()
        yhat_col = "yhat" if "yhat" in pred.columns else "forecast"
        if "ts_id" not in pred.columns:
            pred["ts_id"] = "series_0"
        if "model_type" not in pred.columns:
            pred["model_type"] = "model"
        for (tid, mtype), grp in pred.groupby(["ts_id", "model_type"], sort=False):
            yh = pd.to_numeric(grp[yhat_col], errors="coerce").dropna().to_numpy(dtype=float)
            if yh.size == 0:
                continue
            hit_mask = np.zeros(yh.size, dtype=bool)
            if cap is not None:
                hit_mask |= yh >= (float(cap) - 1e-6)
            if floor is not None and float(floor) > 0.0:
                hit_mask |= yh <= (float(floor) + 1e-6)
            hit_share = float(np.mean(hit_mask))
            if hit_share >= 0.20:
                key = (str(tid), str(mtype))
                _touch(key)
                cell_flags[key].append("capacity_ceiling_hit")
                cell_reasons[key].append(
                    f"Capacity bound reached on {hit_share * 100:.0f}% of horizon steps."
                )
                cell_sev[key] += 25.0 * hit_share

    records: list[dict[str, Any]] = []
    for key, flags in cell_flags.items():
        if not flags:
            continue
        tid, mtype = key
        meta = cell_meta[key]
        records.append(
            {
                "ts_id": tid,
                "model_type": mtype,
                "severity_score": round(cell_sev[key], 2),
                "n_flags": len(flags),
                "exception_flags": ",".join(flags),
                "exception_reasons": " ".join(cell_reasons[key]),
                "metric_loss": meta["metric_loss"],
                "baseline_loss": meta["baseline_loss"],
                "bias_pct": meta["bias_pct"],
                "tracking_signal": meta["tracking_signal"],
                "drift_ratio": meta["drift_ratio"],
                "max_jump_pct": meta["max_jump_pct"],
            }
        )

    if not records:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame.from_records(records, columns=cols)
    out = out.sort_values(
        ["severity_score", "n_flags"], ascending=[False, False], ignore_index=True
    )
    return out.head(max(1, int(top_n)))
