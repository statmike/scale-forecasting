"""Lifecycle markdown & perishable clearance MILP, shelf-space reclamation, and vendor contract
volume optimization.

Provides three coupled prescriptive optimization capabilities built on `scipy.optimize.milp`
(`HiGHS`) and `Track P` digital-twin similarity priors:

1. **Lifecycle Markdown & Perishable Expiry Optimization (:func:`optimize_clearance`,
   :func:`optimize_clearance_df`):** Solves for a monotone markdown discount ladder across the
   forecast horizon ($d_{i,t} \\le d_{i,t+1}$) subject to finite starting on-hand inventory
   $I_{i,0}$, per-SKU shelf-life / end-of-season expiry deadlines $T_{\\text{expiry},i}$, terminal
   salvage value or spoilage disposal cost, cross-SKU cannibalization penalties, and digital-twin
   deep-discount elasticity borrowing when a SKU has never been marked down before.
2. **Warehouse & Shelf Space Reclamation (`target_space_freed`, `unit_volume`, `reclaim_by_step`):**
   Adds a linear cubic-volume / shelf-facing reclamation constraint
   $\\sum_i v_i \\sum_{t=1}^{T_{\\text{reclaim}}} q_{i,t}(d_{i,t}) \\ge V_{\\text{target}}$ so the
   solver finds the minimum-margin-erosion clearance schedule that frees target capacity for
   incoming high-velocity SKUs.
3. **Vendor Contract Volume & Tiered Rebate / Take-or-Pay Optimization
   (:func:`optimize_contract_volume`, :func:`optimize_contract_volume_df`):** Allocates discounts
   across a multi-SKU vendor portfolio to cross cumulative all-units rebate tiers $(\\tau_m, R_m)$
   or satisfy Take-or-Pay minimum volume commitments $Q_{\\min}$ while accounting for Copula
   $\\text{CVaR}_{\\alpha}$ demand shortfall risk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

__all__ = [
    "ASSORTMENT_OPTIMIZER_COLUMNS",
    "ASSORTMENT_TRANSFERENCE_COLUMNS",
    "CLEARANCE_SCHEDULE_COLUMNS",
    "CLEARANCE_SUMMARY_COLUMNS",
    "CONTRACT_VOLUME_COLUMNS",
    "CONTRACT_VOLUME_SUMMARY_COLUMNS",
    "DEFAULT_DISCOUNT_LADDER",
    "AssortmentOptimizationResult",
    "ClearanceOptimizationResult",
    "ContractVolumeOptimizationResult",
    "optimize_assortment",
    "optimize_assortment_df",
    "optimize_clearance",
    "optimize_clearance_df",
    "optimize_contract_volume",
    "optimize_contract_volume_df",
    "plot_clearance_schedule",
]

DEFAULT_DISCOUNT_LADDER: tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.50)

CLEARANCE_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "ts_id",
    "ds",
    "horizon_step",
    "baseline_price",
    "optimal_discount_pct",
    "optimal_price",
    "baseline_units",
    "expected_units",
    "incremental_units",
    "cumulative_units",
    "remaining_inventory",
    "unit_cost",
    "salvage_value",
    "unit_volume",
    "space_freed_step",
    "cumulative_space_freed",
    "baseline_revenue",
    "optimal_revenue",
    "baseline_margin",
    "optimal_margin",
    "cannibalization_loss",
    "net_profit_contribution",
    "is_expired_step",
    "price_elasticity",
    "elasticity_source",
)

CLEARANCE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "ts_id",
    "initial_inventory",
    "expiry_step",
    "baseline_sell_through_units",
    "optimal_sell_through_units",
    "sell_through_pct",
    "unsold_terminal_units",
    "salvage_recovery",
    "spoilage_cost_incurred",
    "total_space_freed",
    "max_discount_pct",
    "first_markdown_step",
    "baseline_gross_profit",
    "optimal_gross_profit",
    "net_profit_lift",
    "price_elasticity",
    "elasticity_source",
    "status",
)

CONTRACT_VOLUME_COLUMNS: tuple[str, ...] = (
    "ts_id",
    "ds",
    "horizon_step",
    "baseline_price",
    "optimal_discount_pct",
    "optimal_price",
    "baseline_units",
    "expected_units",
    "incremental_units",
    "cumulative_portfolio_units",
    "unit_cost",
    "baseline_margin",
    "optimal_margin",
)

CONTRACT_VOLUME_SUMMARY_COLUMNS: tuple[str, ...] = (
    "ts_id",
    "baseline_units",
    "optimal_units",
    "incremental_units",
    "mean_discount_pct",
    "max_discount_pct",
    "baseline_margin",
    "optimal_margin",
    "rebate_allocated",
    "shortfall_penalty_allocated",
    "net_contract_value",
    "price_elasticity",
    "elasticity_source",
)

_MIN_ELASTICITY = -6.0
_MAX_ELASTICITY = -0.05
_MAX_DEMAND_MULTIPLIER = 8.0


@dataclass(frozen=True)
class ClearanceOptimizationResult:
    """Structured output of :func:`optimize_clearance`.

    Attributes:
        schedule_df: Step-by-step markdown schedule with columns
            :data:`CLEARANCE_SCHEDULE_COLUMNS`.
        summary_df: Per-SKU clearance & space-reclamation summary with columns
            :data:`CLEARANCE_SUMMARY_COLUMNS`.
        status: Solver status (`"optimal"`, `"feasible"`, `"infeasible"`, or `"empty"`).
        is_feasible: ``True`` when all sell-through and warehouse space targets were met without
            relaxation.
        total_baseline_profit: Portfolio net profit under a $0\\%$ discount policy (including
            terminal salvage minus spoilage).
        total_optimal_profit: Portfolio net profit under the optimized markdown schedule.
        total_profit_lift: ``total_optimal_profit - total_baseline_profit``.
        total_space_freed: Total warehouse/shelf volume freed by ``reclaim_by_step``.
        target_space_freed: Requested warehouse/shelf volume target (or ``None``).
        overall_sell_through_pct: Portfolio-wide fraction of initial inventory sold before expiry.
        relaxation_hint: Diagnostic message explaining which constraint was unreachable when
            ``is_feasible`` is ``False``.
    """

    schedule_df: pd.DataFrame
    summary_df: pd.DataFrame
    status: str
    is_feasible: bool
    total_baseline_profit: float
    total_optimal_profit: float
    total_profit_lift: float
    total_space_freed: float
    target_space_freed: float | None
    overall_sell_through_pct: float
    relaxation_hint: str | None = None
    hierarchy_rollup_df: pd.DataFrame | None = None


@dataclass(frozen=True)
class ContractVolumeOptimizationResult:
    """Structured output of :func:`optimize_contract_volume`.

    Attributes:
        schedule_df: Step-by-step promotional discount schedule with columns
            :data:`CONTRACT_VOLUME_COLUMNS`.
        summary_df: Per-SKU volume, margin, and rebate allocation summary with columns
            :data:`CONTRACT_VOLUME_SUMMARY_COLUMNS`.
        status: Solver status (`"optimal"`, `"feasible"`, `"infeasible"`, or `"empty"`).
        is_feasible: ``True`` when the contract optimization solved cleanly.
        baseline_portfolio_units: Cumulative portfolio volume at $0\\%$ discount.
        optimal_portfolio_units: Cumulative portfolio volume under the optimal discount plan.
        target_contract_units: Highest active target volume threshold ($\\tau_m$ or $Q_{\\min}$).
        achieved_tier_units: Volume threshold of the rebate tier unlocked by the optimal plan.
        rebate_earned: Lump-sum vendor rebate unlocked ($R_m$).
        shortfall_units: Remaining volume shortfall below ``min_contract_volume``.
        shortfall_penalty: Total Take-or-Pay shortfall penalty incurred.
        cvar_shortfall_units: Expected tail shortfall at ``cvar_alpha`` across demand paths.
        baseline_net_value: Baseline gross margin plus baseline rebate minus baseline penalty.
        optimal_net_value: Optimal gross margin plus earned rebate minus shortfall penalty.
        net_value_lift: ``optimal_net_value - baseline_net_value``.
        relaxation_hint: Optional diagnostic hint if ``min_contract_volume`` exceeds maximum
            physical demand.
    """

    schedule_df: pd.DataFrame
    summary_df: pd.DataFrame
    status: str
    is_feasible: bool
    baseline_portfolio_units: float
    optimal_portfolio_units: float
    target_contract_units: float
    achieved_tier_units: float
    rebate_earned: float
    shortfall_units: float
    shortfall_penalty: float
    cvar_shortfall_units: float
    baseline_net_value: float
    optimal_net_value: float
    net_value_lift: float
    relaxation_hint: str | None = None
    hierarchy_rollup_df: pd.DataFrame | None = None


ASSORTMENT_OPTIMIZER_COLUMNS: tuple[str, ...] = (
    "ts_id",
    "action",
    "is_retained",
    "baseline_units",
    "walk_rate",
    "transferable_share",
    "transferred_out_units",
    "transferred_in_units",
    "lost_walk_units",
    "post_assortment_units",
    "unit_price",
    "unit_cost",
    "unit_margin",
    "fixed_slot_cost",
    "shelf_space",
    "baseline_profit",
    "post_assortment_profit",
    "net_profit_contribution",
    "primary_substitute_ts_id",
    "max_peer_similarity",
    "status",
)

ASSORTMENT_TRANSFERENCE_COLUMNS: tuple[str, ...] = (
    "delisted_ts_id",
    "recipient_ts_id",
    "similarity",
    "transfer_share",
    "transferred_units",
    "transferred_margin",
)


@dataclass(frozen=True)
class AssortmentOptimizationResult:
    """Structured output of :func:`optimize_assortment`.

    Attributes:
        assortment_df: Per-SKU assortment decision (`"KEEP"` vs. `"DELIST"`), walk rate,
            transferred demand, and profit contribution (`ASSORTMENT_OPTIMIZER_COLUMNS`).
        transference_df: Pairwise demand transference flows from delisted SKUs to retained
            substitute SKUs (`ASSORTMENT_TRANSFERENCE_COLUMNS`).
        status: Solver status (`"optimal"`, `"feasible"`, or `"infeasible"`).
        is_feasible: True when all SKU count, shelf space, and category retention constraints
            are satisfied.
        total_candidate_skus: Total candidate SKUs evaluated.
        retained_skus: Count of SKUs kept (`action == "KEEP"`).
        delisted_skus: Count of SKUs delisted (`action == "DELIST"`).
        baseline_category_units: Total baseline demand across all candidate SKUs.
        post_assortment_units: Total retained + transferred demand after assortment rationalization.
        transferred_units: Total demand transferred from delisted SKUs to retained substitutes.
        lost_walk_units: Total category demand lost to shopper walk-away (`walk_rate`).
        category_retention_pct: Fraction of baseline category demand retained (`[0, 1]`).
        mean_walk_rate: Mean intrinsic walk rate across candidate SKUs.
        baseline_net_profit: Net profit of keeping all candidate SKUs (margin minus slot costs).
        optimal_net_profit: Net profit of the optimized assortment including demand transference.
        net_profit_lift: ``optimal_net_profit - baseline_net_profit``.
        shelf_space_used: Total shelf space / facings consumed by retained SKUs.
        relaxation_hint: Actionable diagnostic message when constraints are infeasible.
        hierarchy_rollup_df: Optional bottom-up hierarchy rollup of assortment metrics.
    """

    assortment_df: pd.DataFrame
    transference_df: pd.DataFrame
    status: str
    is_feasible: bool
    total_candidate_skus: int
    retained_skus: int
    delisted_skus: int
    baseline_category_units: float
    post_assortment_units: float
    transferred_units: float
    lost_walk_units: float
    category_retention_pct: float
    mean_walk_rate: float
    baseline_net_profit: float
    optimal_net_profit: float
    net_profit_lift: float
    shelf_space_used: float
    relaxation_hint: str | None = None
    hierarchy_rollup_df: pd.DataFrame | None = None


def _resolve_sku_param(
    spec: float | int | Mapping[str, float | int] | str | None,
    ts_ids: Sequence[str],
    default: float,
    *,
    df: pd.DataFrame | None = None,
    ts_id_col: str = "ts_id",
) -> dict[str, float]:
    """Resolve a scalar, per-SKU mapping, or DataFrame column name into a ``{ts_id: float}`` map."""
    if spec is None:
        return {ts: float(default) for ts in ts_ids}
    if isinstance(spec, str):
        if df is not None and spec in df.columns and ts_id_col in df.columns:
            grouped = (
                df.groupby(df[ts_id_col].astype(str))[spec]
                .apply(lambda s: float(pd.to_numeric(s, errors="coerce").dropna().median()))
                .to_dict()
            )
            return {
                ts: (
                    float(grouped[ts])
                    if ts in grouped and np.isfinite(grouped[ts])
                    else float(default)
                )
                for ts in ts_ids
            }
        return {ts: float(default) for ts in ts_ids}
    if isinstance(spec, Mapping):
        str_map = {str(k): float(v) for k, v in spec.items() if v is not None and np.isfinite(v)}
        return {ts: float(str_map.get(ts, default)) for ts in ts_ids}
    val = float(spec) if np.isfinite(float(spec)) else float(default)
    return dict.fromkeys(ts_ids, val)


_AGGREGATE_TS_IDS = frozenset({"__total__", "Total", "TOTAL", "ALL", "__all__"})
_RECONCILIATION_PRIORITY: tuple[str, ...] = (
    "cross_temporal_mint",
    "mint_nonneg",
    "mint_shrink",
    "wls_struct",
    "wls_var",
    "ols",
    "bottom_up",
    "top_down",
)
_ENSEMBLE_PRIORITY: tuple[str, ...] = (
    "ensemble_champion",
    "champion",
    "ensemble_nnls",
    "ensemble_ridge",
    "ensemble_xgb",
    "ensemble_inverse_error",
    "ensemble_mean",
    "ensemble_median",
)


def _filter_aggregate_nodes(
    df: pd.DataFrame,
    *,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Filter out aggregate hierarchy rollup nodes so leaf SKUs are never double-counted."""
    if df.empty or "ts_id" not in df.columns:
        return df

    out = df[~df["ts_id"].isin(_AGGREGATE_TS_IDS)].copy()
    if "is_aggregate" in out.columns:
        non_agg = out[~out["is_aggregate"].fillna(False).astype(bool)]
        if not non_agg.empty:
            out = non_agg.copy()
    if "hierarchy_level" in out.columns:
        leaf_mask = (
            out["hierarchy_level"]
            .fillna("leaf")
            .astype(str)
            .str.lower()
            .isin({"leaf", "bottom", ""})
        )
        if leaf_mask.any():
            out = out[leaf_mask].copy()

    # If history_df is provided, restrict to leaf ts_ids present in history_df when they overlap.
    if history_df is not None and not history_df.empty and "ts_id" in history_df.columns:
        hist_ids = {
            str(x)
            for x in history_df["ts_id"].dropna().unique()
            if str(x) not in _AGGREGATE_TS_IDS
            and not str(x).startswith("level_")
            and "=" not in str(x)
        }
        if hist_ids:
            matched = out[out["ts_id"].isin(hist_ids)]
            if not matched.empty:
                out = matched.copy()
    else:
        # Exclude synthetic rollup IDs like "region=East" or "East::Beverages" when plain leaves
        # exist.
        plain_mask = (
            ~out["ts_id"].str.startswith("level_")
            & ~out["ts_id"].str.contains("=", regex=False)
            & ~out["ts_id"].str.contains("::", regex=False)
        )
        if plain_mask.any() and not plain_mask.all():
            out = out[plain_mask].copy()

    return out


def _select_best_model_and_reconciliation(
    df: pd.DataFrame,
    *,
    model_type: str | None = None,
    reconciliation_method: str | None = None,
    prefer_reconciled: bool = True,
    oof_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select a single coherent model/reconciliation slice and deduplicate `(ts_id, ds)`."""
    if df.empty:
        return df

    out = df.copy()
    # 1. Exclude backtest fold rows if final forecast horizon rows are present alongside them.
    if "is_backtest" in out.columns:
        final_mask = ~out["is_backtest"].fillna(False).astype(bool)
        if final_mask.any():
            out = out[final_mask].copy()
    elif "fold" in out.columns:
        na_mask = out["fold"].isna() | (pd.to_numeric(out["fold"], errors="coerce") < 0)
        if na_mask.any():
            out = out[na_mask].copy()

    # 2. Filter by reconciliation_method / is_reconciled when present.
    if "reconciliation_method" in out.columns:
        rec_series = out["reconciliation_method"].fillna("").astype(str)
        if reconciliation_method is not None:
            exact_rec = out[rec_series == str(reconciliation_method)]
            if not exact_rec.empty:
                out = exact_rec.copy()
        elif prefer_reconciled:
            for cand_rec in _RECONCILIATION_PRIORITY:
                cand_rows = out[rec_series == cand_rec]
                if not cand_rows.empty:
                    out = cand_rows.copy()
                    break
    elif prefer_reconciled and "is_reconciled" in out.columns and model_type is None:
        rec_mask = out["is_reconciled"].fillna(False).astype(bool)
        if rec_mask.any():
            out = out[rec_mask].copy()

    # 3. Filter or select model_type across base models, @<tag> variants, ensembles, and OOF
    # champions.
    if "model_type" in out.columns and out["model_type"].dropna().nunique() > 0:
        m_col = out["model_type"].astype(str)
        if model_type is not None:
            req = str(model_type)
            exact = out[m_col == req]
            if not exact.empty:
                out = exact.copy()
            else:
                prefix_match = out[
                    m_col.str.startswith(req + "@") | m_col.str.startswith(req + "_")
                ]
                if not prefix_match.empty:
                    first_m = str(prefix_match["model_type"].dropna().iloc[0])
                    out = prefix_match[prefix_match["model_type"].astype(str) == first_m].copy()
        elif out["model_type"].dropna().nunique() > 1:
            avail_models = list(dict.fromkeys(out["model_type"].dropna().astype(str)))
            chosen_global: str | None = None
            # a) Prefer reconciled model_type suffix if prefer_reconciled is True.
            if prefer_reconciled:
                for cand_rec in _RECONCILIATION_PRIORITY:
                    rec_m = next((m for m in avail_models if cand_rec in m), None)
                    if rec_m is not None:
                        chosen_global = rec_m
                        break
            # b) Prefer ensemble strategies in priority order.
            if chosen_global is None:
                for ens_m in _ENSEMBLE_PRIORITY:
                    if ens_m in avail_models:
                        chosen_global = ens_m
                        break
            # c) If OOF predictions are available, select per-SKU champion (lowest OOF WAPE/MAE).
            if (
                chosen_global is None
                and oof_df is not None
                and not oof_df.empty
                and {"ts_id", "model_type", "y", "yhat"}.issubset(oof_df.columns)
            ):
                oof_sub = oof_df[oof_df["model_type"].astype(str).isin(avail_models)].copy()
                if not oof_sub.empty:
                    oof_sub["abs_err"] = (
                        pd.to_numeric(oof_sub["y"], errors="coerce")
                        - pd.to_numeric(oof_sub["yhat"], errors="coerce")
                    ).abs()
                    sku_errs = (
                        oof_sub.groupby(
                            [oof_sub["ts_id"].astype(str), oof_sub["model_type"].astype(str)]
                        )["abs_err"]
                        .mean()
                        .reset_index()
                    )
                    best_per_sku = (
                        sku_errs.sort_values(["ts_id", "abs_err"])
                        .drop_duplicates(subset=["ts_id"], keep="first")
                        .set_index("ts_id")["model_type"]
                        .to_dict()
                    )
                    if best_per_sku:
                        default_m = str(sku_errs.groupby("model_type")["abs_err"].mean().idxmin())
                        mapped_m = out["ts_id"].map(lambda t: best_per_sku.get(str(t), default_m))
                        filtered_oof = out[out["model_type"].astype(str) == mapped_m]
                        if not filtered_oof.empty:
                            out = filtered_oof.copy()
                            chosen_global = "__per_sku_oof_champion__"
            # d) Prefer specialized @<tag> variants (@monotone, @similarity_*, @transform_*) or
            # first model.
            if chosen_global is None:
                tagged = [m for m in avail_models if "@" in m]
                chosen_global = tagged[0] if tagged else avail_models[0]
            if chosen_global != "__per_sku_oof_champion__":
                out = out[out["model_type"].astype(str) == chosen_global].copy()

    return out


def _prepare_forecast_frame(
    predictions_or_panel_df: pd.DataFrame,
    *,
    history_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    model_type: str | None = None,
    reconciliation_method: str | None = None,
    prefer_reconciled: bool = True,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    horizon: int = 14,
    freq: str = "D",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Normalize input into a clean `(forecasts_df, history_df)` pair."""
    if predictions_or_panel_df is None or predictions_or_panel_df.empty:
        return pd.DataFrame(), history_df

    df = predictions_or_panel_df.copy()
    if ts_id_col != "ts_id" and ts_id_col in df.columns and "ts_id" not in df.columns:
        df = df.rename(columns={ts_id_col: "ts_id"})
    if "ts_id" not in df.columns:
        df["ts_id"] = "series_0"
    df["ts_id"] = df["ts_id"].astype(str)

    if "forecast_date" in df.columns and "ds" not in df.columns:
        df = df.rename(columns={"forecast_date": "ds"})
    elif date_col != "ds" and date_col in df.columns and "ds" not in df.columns:
        df = df.rename(columns={date_col: "ds"})

    if "segment" in df.columns and (df["segment"] == "forecast").any():
        if history_df is None and (df["segment"] == "history").any():
            history_df = df[df["segment"] == "history"].copy()
        df = df[df["segment"] == "forecast"].copy()

    if history_df is not None and not history_df.empty:
        h_copy = history_df.copy()
        if ts_id_col != "ts_id" and ts_id_col in h_copy.columns and "ts_id" not in h_copy.columns:
            h_copy = h_copy.rename(columns={ts_id_col: "ts_id"})
        if "ts_id" in h_copy.columns:
            h_copy["ts_id"] = h_copy["ts_id"].astype(str)
            h_copy = _filter_aggregate_nodes(h_copy)
        history_df = h_copy

    df = _filter_aggregate_nodes(df, history_df=history_df)

    if "yhat" in df.columns:
        df = _select_best_model_and_reconciliation(
            df,
            model_type=model_type,
            reconciliation_method=reconciliation_method,
            prefer_reconciled=prefer_reconciled,
            oof_df=oof_df,
        )
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df = (
            df.dropna(subset=["ds"])
            .sort_values(["ts_id", "ds"])
            .drop_duplicates(subset=["ts_id", "ds"], keep="last")
            .reset_index(drop=True)
        )
        return df, history_df

    # Input is a historical panel without 'yhat'; synthesize baseline horizon forecasts.
    eff_hist = history_df if history_df is not None else df.copy()
    val_col = target_col if target_col in df.columns else ("y" if "y" in df.columns else None)
    if val_col is None:
        return pd.DataFrame(), eff_hist

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = (
        df.dropna(subset=["ds", val_col])
        .sort_values(["ts_id", "ds"])
        .drop_duplicates(subset=["ts_id", "ds"], keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        return pd.DataFrame(), eff_hist

    eff_h = max(1, int(horizon))
    rows: list[dict[str, Any]] = []
    for ts_id, grp in df.groupby("ts_id", sort=True):
        y_vals = grp[val_col].to_numpy(dtype=float)
        y_vals = y_vals[np.isfinite(y_vals)]
        base_level = float(np.nanmean(y_vals[-min(len(y_vals), 14) :])) if len(y_vals) else 10.0
        base_level = max(0.0, base_level)
        last_ds = pd.Timestamp(grp["ds"].max())
        try:
            future_dates = pd.date_range(last_ds, periods=eff_h + 1, freq=freq)[1:]
        except ValueError:
            future_dates = pd.date_range(last_ds, periods=eff_h + 1, freq="D")[1:]
        for step_idx, f_ds in enumerate(future_dates):
            seasonal_val = (
                float(max(0.0, y_vals[-(7 - (step_idx % 7))])) if len(y_vals) >= 7 else base_level
            )
            yhat_val = 0.7 * base_level + 0.3 * seasonal_val
            rows.append({"ts_id": str(ts_id), "ds": f_ds, "yhat": float(max(0.0, yhat_val))})
    return pd.DataFrame(rows), eff_hist


def _resolve_price_elasticities_and_similarity(
    ts_ids: Sequence[str],
    *,
    history_df: pd.DataFrame | None,
    price_elasticity: float | Mapping[str, float] | None,
    price_col: str | None,
    covariate_cols: Sequence[str] | None = None,
    elasticity_df: pd.DataFrame | None = None,
    use_digital_twin_elasticity: bool,
    default_elasticity: float,
    similarity_df: pd.DataFrame | None,
    similarity_method: str,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
) -> tuple[dict[str, float], dict[str, str], dict[str, float]]:
    """Resolve per-SKU own-price elasticity, elasticity provenance, and peer similarity weights.

    Returns:
        Tuple of ``(elasticity_map, source_map, mean_peer_similarity_map)``.
    """
    fallback_elast = float(
        np.clip(-abs(float(default_elasticity)), _MIN_ELASTICITY, _MAX_ELASTICITY)
    )
    elast_map: dict[str, float] = {}
    source_map: dict[str, str] = {}

    if isinstance(price_elasticity, (int, float)) and np.isfinite(float(price_elasticity)):
        val = float(np.clip(-abs(float(price_elasticity)), _MIN_ELASTICITY, _MAX_ELASTICITY))
        for ts in ts_ids:
            elast_map[ts] = val
            source_map[ts] = "own_history"
    elif isinstance(price_elasticity, Mapping):
        for k, v in price_elasticity.items():
            if v is not None and np.isfinite(float(v)):
                elast_map[str(k)] = float(np.clip(-abs(float(v)), _MIN_ELASTICITY, _MAX_ELASTICITY))
                source_map[str(k)] = "own_history"

    # Ingest precomputed Double-ML elasticity frame (e.g. from forecaster.promo_elasticity_df()).
    if elasticity_df is not None and not elasticity_df.empty and "ts_id" in elasticity_df.columns:
        for cand_ecol in (
            "price_elasticity",
            "own_price_elasticity",
            "net_causal_elasticity",
            "elasticity",
        ):
            if cand_ecol in elasticity_df.columns:
                grouped_e = (
                    elasticity_df.groupby(elasticity_df["ts_id"].astype(str))[cand_ecol]
                    .apply(lambda s: float(pd.to_numeric(s, errors="coerce").dropna().median()))
                    .to_dict()
                )
                for ts_k, e_val in grouped_e.items():
                    if ts_k not in elast_map and np.isfinite(e_val) and float(e_val) != 0.0:
                        elast_map[str(ts_k)] = float(
                            np.clip(-abs(float(e_val)), _MIN_ELASTICITY, _MAX_ELASTICITY)
                        )
                        source_map[str(ts_k)] = "own_history"
                break

    # Estimate own-history price elasticity from history_df when not explicitly provided.
    hist_norm: pd.DataFrame | None = None
    if history_df is not None and not history_df.empty:
        hist_norm = history_df.copy()
        if (
            ts_id_col != "ts_id"
            and ts_id_col in hist_norm.columns
            and "ts_id" not in hist_norm.columns
        ):
            hist_norm = hist_norm.rename(columns={ts_id_col: "ts_id"})
        if date_col != "ds" and date_col in hist_norm.columns and "ds" not in hist_norm.columns:
            hist_norm = hist_norm.rename(columns={date_col: "ds"})
        if target_col != "y" and target_col in hist_norm.columns and "y" not in hist_norm.columns:
            hist_norm = hist_norm.rename(columns={target_col: "y"})
        if "ts_id" in hist_norm.columns:
            hist_norm["ts_id"] = hist_norm["ts_id"].astype(str)

        eff_pcol = price_col
        if eff_pcol is None:
            for cand in ("price", "unit_price", "price_index", "base_price"):
                if cand in hist_norm.columns:
                    eff_pcol = cand
                    break

        if eff_pcol is not None and eff_pcol in hist_norm.columns and "y" in hist_norm.columns:
            # Auto-detect exogenous control covariates (promotions, holidays, weather degree-days)
            # so simultaneous promo/weather spikes never confound own-price elasticity.
            ctrl_candidates: list[str] = []
            if covariate_cols:
                ctrl_candidates.extend(
                    str(c)
                    for c in covariate_cols
                    if str(c) in hist_norm.columns and str(c) not in {eff_pcol, "y", "ts_id", "ds"}
                )
            for col_name in hist_norm.columns:
                c_str = str(col_name)
                c_low = c_str.lower()
                if (
                    c_str not in {eff_pcol, "y", "ts_id", "ds"}
                    and c_str not in ctrl_candidates
                    and any(
                        tok in c_low
                        for tok in (
                            "promo",
                            "holiday",
                            "hdd",
                            "cdd",
                            "adstock",
                            "feature",
                            "marketing",
                            "temp",
                            "event",
                            "is_weekend",
                        )
                    )
                ):
                    ctrl_candidates.append(c_str)

            for ts_key, grp in hist_norm.sort_values("ds").groupby("ts_id"):
                ts_str = str(ts_key)
                if ts_str in elast_map:
                    continue
                use_cols = [eff_pcol, "y", *[c for c in ctrl_candidates if c in grp.columns]]
                sub_raw = grp[use_cols].apply(pd.to_numeric, errors="coerce")
                valid_mask = sub_raw[eff_pcol].notna() & sub_raw["y"].notna()
                sub = sub_raw.loc[valid_mask].fillna(0.0)
                if len(sub) < 6:
                    continue
                p_vals = sub[eff_pcol].to_numpy(dtype=float)
                y_vals = sub["y"].to_numpy(dtype=float)
                p_med = float(np.median(p_vals))
                p_min = float(np.min(p_vals))
                p_std = float(np.std(p_vals))
                has_markdown_history = (
                    p_med > 0.0 and p_std > 1e-4 and (p_med - p_min) / p_med >= 0.03
                )
                if not has_markdown_history or np.all(y_vals <= 0.0):
                    continue

                # Frisch-Waugh-Lovell log-log own-price elasticity partialling out trend,
                # weekly harmonics, and any exogenous promotional/holiday/weather covariates.
                n_obs = len(p_vals)
                t_grid = np.arange(n_obs, dtype=float)
                nuisance_cols = [
                    np.ones(n_obs, dtype=float),
                    t_grid / float(max(n_obs, 1)),
                    np.sin(2.0 * np.pi * t_grid / 7.0),
                    np.cos(2.0 * np.pi * t_grid / 7.0),
                    np.sin(4.0 * np.pi * t_grid / 7.0),
                    np.cos(4.0 * np.pi * t_grid / 7.0),
                ]
                for c_col in ctrl_candidates:
                    if c_col in sub.columns:
                        c_vec = sub[c_col].to_numpy(dtype=float)
                        c_std = float(np.std(c_vec))
                        if c_std > 1e-8:
                            nuisance_cols.append((c_vec - float(np.mean(c_vec))) / c_std)
                w_mat = np.column_stack(nuisance_cols)
                log_p = np.log(np.maximum(p_vals, 1e-6))
                log_y = np.log(np.maximum(y_vals, 1e-6))
                gram = w_mat.T @ w_mat + 1e-4 * np.eye(w_mat.shape[1])
                p_tilde = log_p - w_mat @ np.linalg.solve(gram, w_mat.T @ log_p)
                y_tilde = log_y - w_mat @ np.linalg.solve(gram, w_mat.T @ log_y)
                denom = float(np.dot(p_tilde, p_tilde))
                if denom > 1e-8:
                    est_val = float(np.dot(p_tilde, y_tilde) / denom)
                    if np.isfinite(est_val) and est_val < -0.05:
                        elast_map[ts_str] = float(
                            np.clip(est_val, _MIN_ELASTICITY, _MAX_ELASTICITY)
                        )
                        source_map[ts_str] = "own_history"

    # Compute or normalize pairwise similarity for digital-twin borrowing & cannibalization.
    pairs_df = similarity_df
    need_twin_borrowing = (
        use_digital_twin_elasticity
        and bool(elast_map)
        and any(ts not in elast_map for ts in ts_ids)
    )
    if (
        pairs_df is None
        and need_twin_borrowing
        and hist_norm is not None
        and "ts_id" in hist_norm.columns
        and "y" in hist_norm.columns
        and hist_norm["ts_id"].nunique() >= 2
    ):
        from .similarity import compute_similarity_matrix

        eff_sim_method = "spectral_phase" if similarity_method == "hybrid" else similarity_method
        sim_res = compute_similarity_matrix(
            hist_norm,
            method=eff_sim_method,
            ts_id_col="ts_id",
            date_col="ds",
            target_col="y",
        )
        pairs_df = sim_res.pairs_df

    peer_sim_map: dict[str, float] = {ts: (0.25 if len(ts_ids) > 1 else 0.0) for ts in ts_ids}
    donor_weights: dict[str, list[tuple[str, float]]] = {ts: [] for ts in ts_ids}

    if pairs_df is not None and not pairs_df.empty:
        col_a = "ts_id" if "ts_id" in pairs_df.columns else pairs_df.columns[0]
        col_b = (
            "neighbor_ts_id"
            if "neighbor_ts_id" in pairs_df.columns
            else ("peer_ts_id" if "peer_ts_id" in pairs_df.columns else pairs_df.columns[1])
        )
        score_col = (
            "similarity"
            if "similarity" in pairs_df.columns
            else ("composite_similarity" if "composite_similarity" in pairs_df.columns else None)
        )
        if score_col is not None:
            for row in pairs_df.itertuples(index=False):
                a = str(getattr(row, col_a))
                b = str(getattr(row, col_b))
                s = float(getattr(row, score_col))
                if a != b and np.isfinite(s) and s > 0.0:
                    if a in donor_weights:
                        donor_weights[a].append((b, float(np.clip(s, 0.0, 1.0))))
                    if b in donor_weights:
                        donor_weights[b].append((a, float(np.clip(s, 0.0, 1.0))))

    for ts in ts_ids:
        neighbors = donor_weights.get(ts, [])
        if neighbors:
            active_peers = [w for peer, w in neighbors if peer in ts_ids and peer != ts]
            if active_peers:
                peer_sim_map[ts] = float(np.clip(np.mean(active_peers), 0.0, 1.0))
        if ts not in elast_map and use_digital_twin_elasticity:
            valid_donors = [(peer, w) for peer, w in neighbors if peer in elast_map and w > 0.0]
            if not valid_donors and elast_map:
                # Borrow from any in-panel peer with measured own_history elasticity.
                valid_donors = [(peer, 0.5) for peer in elast_map if peer != ts]
            if valid_donors:
                valid_donors.sort(key=lambda item: item[1], reverse=True)
                top_donors = valid_donors[:3]
                w_sum = sum(w for _, w in top_donors)
                borrowed = sum(w * elast_map[p] for p, w in top_donors) / max(w_sum, 1e-9)
                elast_map[ts] = float(np.clip(borrowed, _MIN_ELASTICITY, _MAX_ELASTICITY))
                source_map[ts] = "digital_twin_borrowed"

    for ts in ts_ids:
        if ts not in elast_map:
            elast_map[ts] = fallback_elast
            source_map[ts] = "prior_default"

    return elast_map, source_map, peer_sim_map


def _demand_at_discount(base_units: float, discount_pct: float, elasticity: float) -> float:
    """Compute expected units demanded at ``discount_pct`` under constant-elasticity demand."""
    if base_units <= 0.0:
        return 0.0
    d = float(np.clip(discount_pct, 0.0, 0.95))
    if d <= 0.0:
        return float(base_units)
    elast = float(np.clip(elasticity, _MIN_ELASTICITY, _MAX_ELASTICITY))
    multiplier = float(min(_MAX_DEMAND_MULTIPLIER, (1.0 - d) ** elast))
    return float(base_units * multiplier)


_ADDITIVE_ROLLUP_CANDIDATES: tuple[str, ...] = (
    "initial_inventory",
    "baseline_sell_through_units",
    "optimal_sell_through_units",
    "unsold_terminal_units",
    "salvage_recovery",
    "spoilage_cost_incurred",
    "total_space_freed",
    "baseline_gross_profit",
    "optimal_gross_profit",
    "net_profit_lift",
    "baseline_units",
    "optimal_units",
    "incremental_units",
    "baseline_margin",
    "optimal_margin",
    "rebate_allocated",
    "shortfall_penalty_allocated",
    "net_contract_value",
    "transferred_out_units",
    "transferred_in_units",
    "lost_walk_units",
    "post_assortment_units",
    "fixed_slot_cost",
    "shelf_space",
    "baseline_profit",
    "post_assortment_profit",
    "net_profit_contribution",
)


def _build_hierarchy_rollup_df(
    summary_df: pd.DataFrame,
    *,
    ref_df: pd.DataFrame | None = None,
    source_df: pd.DataFrame | None = None,
    hierarchy_cols: Sequence[str] | str | None = "auto",
    additive_cols: Sequence[str] | None = None,
    ts_id_col: str = "ts_id",
) -> pd.DataFrame | None:
    """Build a bottom-up coherent hierarchical summary across ``__total__``, intermediate levels,
    and ``leaf`` SKUs when ``hierarchy_cols`` are present on ``ref_df`` / ``source_df``.
    """
    eff_ref = ref_df if ref_df is not None else source_df
    if summary_df.empty or eff_ref is None or eff_ref.empty or ts_id_col not in eff_ref.columns:
        return None

    eff_cols: list[str] = []
    if isinstance(hierarchy_cols, str):
        if hierarchy_cols == "auto":
            for cand in (
                "region",
                "category",
                "department",
                "store",
                "brand",
                "similarity_cluster_id",
            ):
                if cand in eff_ref.columns:
                    eff_cols.append(cand)
        elif hierarchy_cols in eff_ref.columns:
            eff_cols.append(hierarchy_cols)
    elif hierarchy_cols is not None:
        for item in hierarchy_cols:
            if isinstance(item, (list, tuple)):
                for sub_c in item:
                    if str(sub_c) in eff_ref.columns and str(sub_c) not in eff_cols:
                        eff_cols.append(str(sub_c))
            elif str(item) in eff_ref.columns and str(item) not in eff_cols:
                eff_cols.append(str(item))

    if not eff_cols:
        return None

    meta = (
        eff_ref[[ts_id_col, *eff_cols]]
        .rename(columns={ts_id_col: "ts_id"})
        .assign(ts_id=lambda d: d["ts_id"].astype(str))
        .drop_duplicates(subset=["ts_id"], keep="last")
    )
    merged = summary_df.merge(meta, on="ts_id", how="left")
    cand_cols = additive_cols if additive_cols is not None else _ADDITIVE_ROLLUP_CANDIDATES
    sum_cols = [c for c in cand_cols if c in merged.columns]
    if not sum_cols:
        return None

    rollup_rows: list[dict[str, Any]] = []
    # 1. Grand total (__total__)
    tot_row: dict[str, Any] = {
        "hierarchy_level": "__total__",
        "node_id": "__total__",
        "n_skus": len(merged),
    }
    for c in sum_cols:
        tot_row[c] = float(merged[c].sum())
    rollup_rows.append(tot_row)

    # 2. Each declared hierarchy level column (e.g. "region", "category")
    for col in eff_cols:
        for val, grp in merged.groupby(col, dropna=False):
            row_d: dict[str, Any] = {
                "hierarchy_level": col,
                "node_id": f"{col}={val}",
                "n_skus": len(grp),
            }
            for c in sum_cols:
                row_d[c] = float(grp[c].sum())
            rollup_rows.append(row_d)

    # 3. Bottom leaf rows
    for _, r in merged.iterrows():
        leaf_d: dict[str, Any] = {
            "hierarchy_level": "leaf",
            "node_id": str(r["ts_id"]),
            "n_skus": 1,
        }
        for c in sum_cols:
            leaf_d[c] = float(r[c])
        rollup_rows.append(leaf_d)

    return pd.DataFrame(rollup_rows)


def _estimate_cvar_downside_fraction(
    *,
    trajectories_df: pd.DataFrame | None,
    fc_df: pd.DataFrame,
    oof_df: pd.DataFrame | None,
    cvar_alpha: float,
    risk_aversion_lambda: float,
) -> float:
    """Estimate portfolio relative downside shortfall fraction at ``cvar_alpha`` from:
    1. Copula trajectory simulations (``trajectories_df``),
    2. Calibrated lower prediction intervals on ``fc_df`` (``aci_lower``, ``yhat_lower``),
    3. Out-of-fold empirical residuals on ``oof_df`` (``y`` vs. ``yhat``), or
    4. Conservative parametric fallback when ``risk_aversion_lambda > 0``.
    """
    q_tail = float(np.clip(1.0 - float(cvar_alpha), 0.01, 0.49))
    if (
        trajectories_df is not None
        and not trajectories_df.empty
        and "yhat_sim" in trajectories_df.columns
    ):
        path_col = "path_id" if "path_id" in trajectories_df.columns else "sample_id"
        if path_col in trajectories_df.columns:
            path_totals = trajectories_df.groupby(path_col)["yhat_sim"].sum().to_numpy(dtype=float)
            if len(path_totals) >= 5:
                mean_tot = float(np.mean(path_totals))
                q_cutoff = float(np.quantile(path_totals, q_tail))
                tail_vals = path_totals[path_totals <= q_cutoff]
                cvar_low = float(np.mean(tail_vals)) if len(tail_vals) else q_cutoff
                if mean_tot > 0.0:
                    return float(np.clip((mean_tot - cvar_low) / mean_tot, 0.0, 0.50))

    # Check calibrated lower prediction bounds (Online ACI, Conformal, or Quantiles) on fc_df.
    if not fc_df.empty and "yhat" in fc_df.columns:
        yhat_sum = float(pd.to_numeric(fc_df["yhat"], errors="coerce").fillna(0.0).sum())
        if yhat_sum > 0.0:
            for low_col in (
                "aci_lower",
                "yhat_lower",
                "q_0.05",
                "q_0.1",
                "q_0.10",
                "q05",
                "q10",
                "lower",
            ):
                if low_col in fc_df.columns:
                    low_vals = pd.to_numeric(fc_df[low_col], errors="coerce")
                    if low_vals.notna().any():
                        low_sum = float(low_vals.fillna(fc_df["yhat"]).clip(lower=0.0).sum())
                        if low_sum < yhat_sum:
                            return float(np.clip((yhat_sum - low_sum) / yhat_sum, 0.0, 0.50))

    # Check empirical OOF relative residuals when oof_df is available.
    if oof_df is not None and not oof_df.empty and {"y", "yhat"}.issubset(oof_df.columns):
        y_oof = pd.to_numeric(oof_df["y"], errors="coerce").to_numpy(dtype=float)
        yh_oof = pd.to_numeric(oof_df["yhat"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_oof) & np.isfinite(yh_oof) & (yh_oof > 1e-6)
        if int(np.sum(mask)) >= 5:
            rel_shortfall = np.clip((yh_oof[mask] - y_oof[mask]) / yh_oof[mask], -1.0, 1.0)
            q_cut = float(np.quantile(rel_shortfall, 1.0 - q_tail))
            tail_s = rel_shortfall[rel_shortfall >= q_cut]
            cvar_s = float(np.mean(tail_s)) if len(tail_s) else q_cut
            if cvar_s > 0.0:
                return float(np.clip(cvar_s, 0.0, 0.50))

    if float(risk_aversion_lambda) > 0.0:
        return 0.12
    return 0.0


def optimize_clearance(
    predictions_df: pd.DataFrame,
    *,
    history_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    initial_inventory: float | Mapping[str, float] | str | None = None,
    expiry_step: int | Mapping[str, int] | None = None,
    shelf_life_days: int | Mapping[str, int] | None = None,
    discount_ladder: Sequence[float] = DEFAULT_DISCOUNT_LADDER,
    monotone_markdown: bool = True,
    allow_price_rebound: bool = False,
    base_price: float | Mapping[str, float] | str = 20.0,
    unit_cost: float | Mapping[str, float] | str = 8.0,
    salvage_value: float | Mapping[str, float] | str = 2.0,
    spoilage_cost: float | Mapping[str, float] | str = 0.0,
    target_sell_through_pct: float | Mapping[str, float] | None = None,
    unit_volume: float | Mapping[str, float] | str = 1.0,
    target_space_freed: float | None = None,
    reclaim_by_step: int | None = None,
    price_elasticity: float | Mapping[str, float] | None = None,
    price_col: str | None = None,
    covariate_cols: Sequence[str] | None = None,
    elasticity_df: pd.DataFrame | None = None,
    use_digital_twin_elasticity: bool = True,
    default_elasticity: float = -1.8,
    cannibalization_penalty: float = 0.15,
    similarity_df: pd.DataFrame | None = None,
    similarity_method: str = "hybrid",
    model_type: str | None = None,
    reconciliation_method: str | None = None,
    prefer_reconciled: bool = True,
    hierarchy_cols: Sequence[str] | str | None = "auto",
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    horizon: int = 14,
    freq: str = "D",
    strict_feasibility: bool = False,
) -> ClearanceOptimizationResult:
    """Solve the multi-SKU lifecycle markdown, perishable expiry, and shelf-space reclamation MILP.

    Formulates a Mixed-Integer Linear Program (`scipy.optimize.milp` / `HiGHS`) that selects a
    discount rung $d_{i,t} \\in \\text{discount\\_ladder}$ for each SKU $i$ and horizon step
    $t = 1 \\dots H$, subject to:

    - **Monotone markdown ladder:** $d_{i,t} \\le d_{i,t+1}$ when ``monotone_markdown=True`` and
      ``allow_price_rebound=False``.
    - **Finite initial inventory & expiry deadline:** Cumulative sales up to ``expiry_step``
      cannot exceed ``initial_inventory`` $I_{i,0}$, and demand after $T_{\\text{expiry},i}$ is 0.
    - **Minimum sell-through target:** Optional per-SKU ``target_sell_through_pct`` $\\in [0, 1]$.
    - **Warehouse / shelf space reclamation:** Optional portfolio capacity target
      $\\sum_i v_i \\sum_{t=1}^{T_{\\text{reclaim}}} q_{i,t} \\ge \\text{target\\_space\\_freed}$.
    """
    fc_df, eff_hist = _prepare_forecast_frame(
        predictions_df,
        history_df=history_df,
        oof_df=oof_df,
        model_type=model_type,
        reconciliation_method=reconciliation_method,
        prefer_reconciled=prefer_reconciled,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        freq=freq,
    )
    if fc_df.empty:
        return ClearanceOptimizationResult(
            schedule_df=pd.DataFrame(columns=CLEARANCE_SCHEDULE_COLUMNS),
            summary_df=pd.DataFrame(columns=CLEARANCE_SUMMARY_COLUMNS),
            status="empty",
            is_feasible=True,
            total_baseline_profit=0.0,
            total_optimal_profit=0.0,
            total_profit_lift=0.0,
            total_space_freed=0.0,
            target_space_freed=target_space_freed,
            overall_sell_through_pct=0.0,
        )

    ladder = sorted({float(np.clip(d, 0.0, 0.95)) for d in discount_ladder})
    if not ladder or ladder[0] > 0.0:
        ladder = [0.0, *ladder]
    n_rungs = len(ladder)

    ts_ids = list(dict.fromkeys(fc_df["ts_id"].astype(str)))
    ref_df = eff_hist if (eff_hist is not None and not eff_hist.empty) else fc_df
    price_map = _resolve_sku_param(base_price, ts_ids, 20.0, df=ref_df, ts_id_col="ts_id")
    cost_map = _resolve_sku_param(unit_cost, ts_ids, 8.0, df=ref_df, ts_id_col="ts_id")
    salvage_map = _resolve_sku_param(salvage_value, ts_ids, 2.0, df=ref_df, ts_id_col="ts_id")
    spoil_map = _resolve_sku_param(spoilage_cost, ts_ids, 0.0, df=ref_df, ts_id_col="ts_id")
    vol_map = _resolve_sku_param(unit_volume, ts_ids, 1.0, df=ref_df, ts_id_col="ts_id")

    elast_map, source_map, peer_sim_map = _resolve_price_elasticities_and_similarity(
        ts_ids,
        history_df=eff_hist,
        price_elasticity=price_elasticity,
        price_col=price_col,
        covariate_cols=covariate_cols,
        elasticity_df=elasticity_df,
        use_digital_twin_elasticity=use_digital_twin_elasticity,
        default_elasticity=default_elasticity,
        similarity_df=similarity_df,
        similarity_method=similarity_method,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
    )

    # Group forecast horizon steps per SKU.
    sku_groups: dict[str, pd.DataFrame] = {}
    sku_horizon: dict[str, int] = {}
    default_inv_map: dict[str, float] = {}
    for ts in ts_ids:
        grp = fc_df[fc_df["ts_id"] == ts].sort_values("ds").reset_index(drop=True)
        sku_groups[ts] = grp
        sku_horizon[ts] = len(grp)
        base_sum = float(grp["yhat"].sum())
        default_inv_map[ts] = max(1.0, 0.85 * base_sum) if base_sum > 0 else 0.0

    inv_map = (
        {ts: default_inv_map[ts] for ts in ts_ids}
        if initial_inventory is None
        else _resolve_sku_param(initial_inventory, ts_ids, 100.0, df=ref_df, ts_id_col="ts_id")
    )
    for ts in ts_ids:
        inv_map[ts] = max(0.0, float(inv_map[ts]))

    raw_expiry = expiry_step if expiry_step is not None else shelf_life_days
    expiry_map: dict[str, int] = {}
    for ts in ts_ids:
        h_i = sku_horizon[ts]
        if raw_expiry is None:
            expiry_map[ts] = h_i
        elif isinstance(raw_expiry, Mapping):
            val = raw_expiry.get(ts, h_i)
            expiry_map[ts] = int(np.clip(int(val if val is not None else h_i), 1, h_i))
        else:
            expiry_map[ts] = int(np.clip(int(raw_expiry), 1, h_i))

    sell_through_map: dict[str, float] = {}
    if target_sell_through_pct is not None:
        raw_st = _resolve_sku_param(target_sell_through_pct, ts_ids, 0.0)
        for ts, st_val in raw_st.items():
            if st_val > 0.0:
                sell_through_map[ts] = float(np.clip(st_val, 0.0, 1.0))

    # Precompute demand table d_hat[ts][t, k] and check feasibility of sell-through & space targets.
    d_hat: dict[str, np.ndarray] = {}
    max_sellable_by_expiry: dict[str, float] = {}
    max_sellable_by_reclaim: dict[str, float] = {}

    for ts in ts_ids:
        grp = sku_groups[ts]
        h_i = sku_horizon[ts]
        exp_i = expiry_map[ts]
        rec_i = min(h_i, int(reclaim_by_step)) if reclaim_by_step is not None else h_i
        elast_i = elast_map[ts]
        mat = np.zeros((h_i, n_rungs), dtype=float)
        for t_idx in range(h_i):
            if t_idx + 1 <= exp_i:
                base_u = float(grp["yhat"].iloc[t_idx])
                for k_idx, disc in enumerate(ladder):
                    mat[t_idx, k_idx] = _demand_at_discount(base_u, disc, elast_i)
        d_hat[ts] = mat
        max_dem_exp = float(mat[:exp_i, -1].sum())
        max_dem_rec = float(mat[: min(exp_i, rec_i), -1].sum())
        max_sellable_by_expiry[ts] = min(inv_map[ts], max_dem_exp)
        max_sellable_by_reclaim[ts] = min(inv_map[ts], max_dem_rec)

    relaxation_notes: list[str] = []
    eff_sell_through_units: dict[str, float] = {}
    for ts, st_pct in sell_through_map.items():
        req_u = st_pct * inv_map[ts]
        max_u = max_sellable_by_expiry[ts]
        if req_u > max_u + 1e-6:
            max_pct = (max_u / inv_map[ts]) if inv_map[ts] > 0 else 0.0
            relaxation_notes.append(
                f"Infeasible target_sell_through_pct={st_pct:.1%} for '{ts}': maximum achievable "
                f"sell-through by expiry_step={expiry_map[ts]} at max discount "
                f"{ladder[-1]:.0%} is {max_pct:.1%} ({max_u:.2f} / {inv_map[ts]:.2f} units)."
            )
            eff_sell_through_units[ts] = max(0.0, 0.9999 * max_u)
        else:
            eff_sell_through_units[ts] = req_u

    max_possible_space = sum(max(0.0, vol_map[ts]) * max_sellable_by_reclaim[ts] for ts in ts_ids)
    eff_target_space: float | None = None
    if target_space_freed is not None and float(target_space_freed) > 0.0:
        req_space = float(target_space_freed)
        if req_space > max_possible_space + 1e-6:
            rec_label = reclaim_by_step if reclaim_by_step is not None else "horizon end"
            relaxation_notes.append(
                f"Infeasible target_space_freed={req_space:.2f}: maximum achievable space freed "
                f"by step {rec_label} at max discount {ladder[-1]:.0%} is "
                f"{max_possible_space:.2f} (capped by initial_inventory and demand elasticity). "
                f"Lower target_space_freed to <= {max_possible_space:.2f}, extend reclaim_by_step, "
                f"or add a deeper discount rung."
            )
            eff_target_space = max(0.0, 0.9999 * max_possible_space)
        else:
            eff_target_space = req_space

    if relaxation_notes and strict_feasibility:
        raise ValueError(" ".join(relaxation_notes))

    # Build MILP:
    # For each (ts, t, k), we have:
    # - binary rung choice z_{ts, t, k} in {0, 1}
    # - continuous realized sales s_{ts, t, k} in [0, d_hat[ts][t, k]]
    idx_z: dict[tuple[str, int, int], int] = {}
    idx_s: dict[tuple[str, int, int], int] = {}
    var_count = 0
    for ts in ts_ids:
        for t_idx in range(sku_horizon[ts]):
            for k_idx in range(n_rungs):
                idx_z[(ts, t_idx, k_idx)] = var_count
                var_count += 1
                idx_s[(ts, t_idx, k_idx)] = var_count
                var_count += 1

    c_obj = np.zeros(var_count, dtype=float)
    integrality = np.zeros(var_count, dtype=float)
    lb_vec = np.zeros(var_count, dtype=float)
    ub_vec = np.ones(var_count, dtype=float)

    enforce_monotone = bool(monotone_markdown) and not bool(allow_price_rebound)
    cannib_weight = max(0.0, float(cannibalization_penalty))

    for ts in ts_ids:
        p0 = price_map[ts]
        c_i = cost_map[ts]
        v_term = salvage_map[ts] - spoil_map[ts]
        base_unit_margin = max(0.0, p0 - c_i)
        sim_i = peer_sim_map[ts]
        exp_i = expiry_map[ts]
        h_i = sku_horizon[ts]
        for t_idx in range(h_i):
            fifo_bonus = 1e-4 * float(h_i - t_idx) / float(max(h_i, 1))
            for k_idx, disc in enumerate(ladder):
                z_col = idx_z[(ts, t_idx, k_idx)]
                s_col = idx_s[(ts, t_idx, k_idx)]
                integrality[z_col] = 1.0
                ub_vec[z_col] = 1.0
                dem_tk = float(d_hat[ts][t_idx, k_idx])
                ub_vec[s_col] = dem_tk

                if t_idx + 1 > exp_i:
                    # Expired steps have zero sales; lock discount to terminal rung or 0.
                    ub_vec[s_col] = 0.0
                    c_obj[z_col] = 1e-5 * disc
                    continue

                p_disc = p0 * (1.0 - disc)
                # Marginal profit of selling 1 unit at rung k vs. leaving it unsold at expiry:
                net_unit_gain = (p_disc - c_i) - v_term
                base_u = float(sku_groups[ts]["yhat"].iloc[t_idx])
                cannib_tk = cannib_weight * sim_i * disc * base_unit_margin * base_u
                # Small markdown signaling penalty prevents redundant markdowns after stockout.
                z_penalty = cannib_tk + 1e-3 * disc * max(p0, 1.0) * max(base_u, 1.0)
                # Negate for minimization in scipy.optimize.milp:
                c_obj[s_col] = -(net_unit_gain + fifo_bonus)
                c_obj[z_col] = z_penalty

    rows_a: list[np.ndarray] = []
    rows_lb: list[float] = []
    rows_ub: list[float] = []

    def _add_constraint(coeffs: dict[int, float], lower: float, upper: float) -> None:
        row = np.zeros(var_count, dtype=float)
        for col_i, val in coeffs.items():
            if val != 0.0:
                row[col_i] = val
        rows_a.append(row)
        rows_lb.append(lower)
        rows_ub.append(upper)

    for ts in ts_ids:
        h_i = sku_horizon[ts]
        exp_i = expiry_map[ts]
        # 1. Exactly one discount rung chosen per (ts, t), and s_{ts, t, k} <= d_hat * z_{ts, t, k}.
        for t_idx in range(h_i):
            _add_constraint(
                {idx_z[(ts, t_idx, k_idx)]: 1.0 for k_idx in range(n_rungs)},
                1.0,
                1.0,
            )
            for k_idx in range(n_rungs):
                dem_tk = float(d_hat[ts][t_idx, k_idx])
                if dem_tk > 0.0:
                    _add_constraint(
                        {
                            idx_s[(ts, t_idx, k_idx)]: 1.0,
                            idx_z[(ts, t_idx, k_idx)]: -dem_tk,
                        },
                        -np.inf,
                        0.0,
                    )

        # 2. Monotone non-decreasing markdown ladder across horizon steps:
        # sum_{m >= k} z_{t, m} - sum_{m >= k} z_{t+1, m} <= 0 for all k >= 1.
        if enforce_monotone and h_i >= 2:
            for t_idx in range(h_i - 1):
                for k_idx in range(1, n_rungs):
                    coeffs: dict[int, float] = {}
                    for m_idx in range(k_idx, n_rungs):
                        coeffs[idx_z[(ts, t_idx, m_idx)]] = 1.0
                        coeffs[idx_z[(ts, t_idx + 1, m_idx)]] = -1.0
                    _add_constraint(coeffs, -np.inf, 0.0)

        # 3. Finite initial inventory cap and optional minimum sell-through target:
        inv_coeffs = {
            idx_s[(ts, t_idx, k_idx)]: 1.0 for t_idx in range(exp_i) for k_idx in range(n_rungs)
        }
        min_st_u = eff_sell_through_units.get(ts, 0.0)
        _add_constraint(inv_coeffs, min_st_u, inv_map[ts])

    # 4. Portfolio warehouse / shelf space reclamation constraint by reclaim_by_step:
    if eff_target_space is not None and eff_target_space > 0.0:
        space_coeffs: dict[int, float] = {}
        for ts in ts_ids:
            v_i = max(0.0, vol_map[ts])
            if v_i <= 0.0:
                continue
            exp_i = expiry_map[ts]
            h_i = sku_horizon[ts]
            rec_i = min(exp_i, int(reclaim_by_step)) if reclaim_by_step is not None else exp_i
            rec_i = min(rec_i, h_i)
            for t_idx in range(rec_i):
                for k_idx in range(n_rungs):
                    space_coeffs[idx_s[(ts, t_idx, k_idx)]] = v_i
        if space_coeffs:
            _add_constraint(space_coeffs, eff_target_space, np.inf)

    a_mat = np.vstack(rows_a) if rows_a else np.zeros((0, var_count), dtype=float)
    constraints = (
        LinearConstraint(a_mat, np.array(rows_lb, dtype=float), np.array(rows_ub, dtype=float))
        if rows_a
        else None
    )
    bounds = Bounds(lb_vec, ub_vec)
    res = milp(
        c=c_obj,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )

    sol_x = res.x if (res.success and res.x is not None) else np.zeros(var_count, dtype=float)
    solver_ok = bool(res.success and res.x is not None)

    # Extract chosen discount rungs and roll forward exact chronological inventory depletion.
    schedule_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_base_profit = 0.0
    total_opt_profit = 0.0
    total_space_by_reclaim = 0.0
    total_initial_inv = 0.0
    total_opt_sold = 0.0

    for ts in ts_ids:
        grp = sku_groups[ts]
        h_i = sku_horizon[ts]
        exp_i = expiry_map[ts]
        rec_i = min(h_i, int(reclaim_by_step)) if reclaim_by_step is not None else h_i
        p0 = price_map[ts]
        c_i = cost_map[ts]
        salv_i = salvage_map[ts]
        spoil_i = spoil_map[ts]
        v_i = max(0.0, vol_map[ts])
        inv_0 = inv_map[ts]
        elast_i = elast_map[ts]
        src_i = source_map[ts]
        sim_i = peer_sim_map[ts]
        base_unit_margin = max(0.0, p0 - c_i)

        rem_base_inv = inv_0
        rem_opt_inv = inv_0
        cum_opt_units = 0.0
        cum_space_freed = 0.0
        space_freed_by_reclaim_i = 0.0
        base_sold_total = 0.0
        base_margin_total = 0.0
        opt_margin_total = 0.0
        cannib_total = 0.0
        chosen_discounts: list[float] = []
        first_md_step: int | None = None

        for t_idx in range(h_i):
            step_num = t_idx + 1
            ds_val = grp["ds"].iloc[t_idx]
            base_dem = float(grp["yhat"].iloc[t_idx])
            is_expired = step_num > exp_i

            if solver_ok:
                z_vals = [float(sol_x[idx_z[(ts, t_idx, k_idx)]]) for k_idx in range(n_rungs)]
                best_k = int(np.argmax(z_vals))
            else:
                best_k = 0
            if enforce_monotone and chosen_discounts:
                prev_d = chosen_discounts[-1]
                if ladder[best_k] < prev_d - 1e-9:
                    best_k = next(
                        (k_idx for k_idx, d in enumerate(ladder) if d >= prev_d - 1e-9),
                        n_rungs - 1,
                    )
            disc = float(ladder[best_k])
            chosen_discounts.append(disc)
            if disc > 0.0 and first_md_step is None and not is_expired:
                first_md_step = step_num

            opt_price = p0 * (1.0 - disc)
            if is_expired:
                base_u_step = 0.0
                opt_u_step = 0.0
                cannib_step = 0.0
            else:
                base_u_step = min(base_dem, rem_base_inv)
                rem_base_inv = max(0.0, rem_base_inv - base_u_step)
                dem_opt = float(d_hat[ts][t_idx, best_k])
                opt_u_step = min(dem_opt, rem_opt_inv)
                rem_opt_inv = max(0.0, rem_opt_inv - opt_u_step)
                cannib_step = (
                    cannib_weight * sim_i * disc * base_unit_margin * base_dem
                    if opt_u_step > 0.0
                    else 0.0
                )

            cum_opt_units += opt_u_step
            space_step = v_i * opt_u_step
            cum_space_freed += space_step
            if step_num <= rec_i:
                space_freed_by_reclaim_i += space_step

            base_rev = p0 * base_u_step
            opt_rev = opt_price * opt_u_step
            base_m = (p0 - c_i) * base_u_step
            opt_m = (opt_price - c_i) * opt_u_step
            net_contrib = opt_m - cannib_step

            base_sold_total += base_u_step
            base_margin_total += base_m
            opt_margin_total += opt_m
            cannib_total += cannib_step

            schedule_rows.append(
                {
                    "ts_id": ts,
                    "ds": ds_val,
                    "horizon_step": step_num,
                    "baseline_price": float(p0),
                    "optimal_discount_pct": float(disc),
                    "optimal_price": float(opt_price),
                    "baseline_units": float(base_u_step),
                    "expected_units": float(opt_u_step),
                    "incremental_units": float(opt_u_step - base_u_step),
                    "cumulative_units": float(cum_opt_units),
                    "remaining_inventory": float(rem_opt_inv),
                    "unit_cost": float(c_i),
                    "salvage_value": float(salv_i),
                    "unit_volume": float(v_i),
                    "space_freed_step": float(space_step),
                    "cumulative_space_freed": float(cum_space_freed),
                    "baseline_revenue": float(base_rev),
                    "optimal_revenue": float(opt_rev),
                    "baseline_margin": float(base_m),
                    "optimal_margin": float(opt_m),
                    "cannibalization_loss": float(cannib_step),
                    "net_profit_contribution": float(net_contrib),
                    "is_expired_step": bool(is_expired),
                    "price_elasticity": float(elast_i),
                    "elasticity_source": src_i,
                }
            )

        unsold_base = max(0.0, inv_0 - base_sold_total)
        unsold_opt = max(0.0, inv_0 - cum_opt_units)
        salv_rec_opt = unsold_opt * salv_i
        spoil_cost_opt = unsold_opt * spoil_i
        base_net_profit = base_margin_total + unsold_base * (salv_i - spoil_i)
        opt_net_profit = opt_margin_total - cannib_total + salv_rec_opt - spoil_cost_opt
        st_pct_achieved = (cum_opt_units / inv_0) if inv_0 > 0 else 1.0

        sku_status = "infeasible" if relaxation_notes else ("optimal" if solver_ok else "feasible")
        summary_rows.append(
            {
                "ts_id": ts,
                "initial_inventory": float(inv_0),
                "expiry_step": int(exp_i),
                "baseline_sell_through_units": float(base_sold_total),
                "optimal_sell_through_units": float(cum_opt_units),
                "sell_through_pct": float(st_pct_achieved),
                "unsold_terminal_units": float(unsold_opt),
                "salvage_recovery": float(salv_rec_opt),
                "spoilage_cost_incurred": float(spoil_cost_opt),
                "total_space_freed": float(space_freed_by_reclaim_i),
                "max_discount_pct": float(max(chosen_discounts) if chosen_discounts else 0.0),
                "first_markdown_step": int(first_md_step) if first_md_step is not None else 0,
                "baseline_gross_profit": float(base_net_profit),
                "optimal_gross_profit": float(opt_net_profit),
                "net_profit_lift": float(opt_net_profit - base_net_profit),
                "price_elasticity": float(elast_i),
                "elasticity_source": src_i,
                "status": sku_status,
            }
        )

        total_base_profit += base_net_profit
        total_opt_profit += opt_net_profit
        total_space_by_reclaim += space_freed_by_reclaim_i
        total_initial_inv += inv_0
        total_opt_sold += cum_opt_units

    is_feasible = len(relaxation_notes) == 0 and solver_ok
    overall_status = "infeasible" if not is_feasible else "optimal"
    overall_st = (total_opt_sold / total_initial_inv) if total_initial_inv > 0 else 1.0
    hint_str = " ".join(relaxation_notes) if relaxation_notes else None
    summary_df_out = pd.DataFrame(summary_rows, columns=CLEARANCE_SUMMARY_COLUMNS)
    rollup_df = _build_hierarchy_rollup_df(
        summary_df_out,
        source_df=ref_df,
        hierarchy_cols=hierarchy_cols,
        ts_id_col="ts_id",
    )

    return ClearanceOptimizationResult(
        schedule_df=pd.DataFrame(schedule_rows, columns=CLEARANCE_SCHEDULE_COLUMNS),
        summary_df=summary_df_out,
        status=overall_status,
        is_feasible=is_feasible,
        total_baseline_profit=float(total_base_profit),
        total_optimal_profit=float(total_opt_profit),
        total_profit_lift=float(total_opt_profit - total_base_profit),
        total_space_freed=float(total_space_by_reclaim),
        target_space_freed=float(target_space_freed) if target_space_freed is not None else None,
        overall_sell_through_pct=float(overall_st),
        relaxation_hint=hint_str,
        hierarchy_rollup_df=rollup_df,
    )


def optimize_clearance_df(
    predictions_df: pd.DataFrame,
    *,
    summary: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience wrapper around :func:`optimize_clearance` returning a tidy DataFrame.

    Returns ``result.summary_df`` when ``summary=True`` and ``result.schedule_df`` otherwise, with
    solver metadata attached to ``df.attrs["clearance_result"]``.
    """
    res = optimize_clearance(predictions_df, **kwargs)
    out = res.summary_df.copy() if summary else res.schedule_df.copy()
    out.attrs["clearance_result"] = res
    out.attrs["status"] = res.status
    out.attrs["is_feasible"] = res.is_feasible
    out.attrs["relaxation_hint"] = res.relaxation_hint
    out.attrs["hierarchy_rollup_df"] = res.hierarchy_rollup_df
    return out


def optimize_contract_volume(
    predictions_df: pd.DataFrame,
    *,
    history_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    trajectories_df: pd.DataFrame | None = None,
    contract_tiers: Sequence[tuple[float, float]] | Mapping[float, float] | None = None,
    min_contract_volume: float | None = None,
    shortfall_penalty_per_unit: float = 5.0,
    discount_ladder: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.25),
    max_discount_pct: float | None = None,
    base_price: float | Mapping[str, float] | str = 20.0,
    unit_cost: float | Mapping[str, float] | str = 8.0,
    initial_inventory: float | Mapping[str, float] | None = None,
    price_elasticity: float | Mapping[str, float] | None = None,
    price_col: str | None = None,
    covariate_cols: Sequence[str] | None = None,
    elasticity_df: pd.DataFrame | None = None,
    use_digital_twin_elasticity: bool = True,
    default_elasticity: float = -1.8,
    risk_aversion_lambda: float = 0.0,
    cvar_alpha: float = 0.95,
    similarity_df: pd.DataFrame | None = None,
    similarity_method: str = "hybrid",
    model_type: str | None = None,
    reconciliation_method: str | None = None,
    prefer_reconciled: bool = True,
    hierarchy_cols: Sequence[str] | str | None = "auto",
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    horizon: int = 14,
    freq: str = "D",
) -> ContractVolumeOptimizationResult:
    """Solve multi-SKU vendor contract volume, tiered rebate, and Take-or-Pay commitment MILP.

    Allocates promotional discounts across a portfolio of SKUs (or energy Take-or-Pay / Cloud
    FinOps commitment lines) to maximize total gross profit plus unlocked all-units vendor rebates
    $R_m$ minus Take-or-Pay volume shortfall penalties and Copula $\\text{CVaR}_{\\alpha}$ tail
    shortfall risk.
    """
    fc_df, eff_hist = _prepare_forecast_frame(
        predictions_df,
        history_df=history_df,
        oof_df=oof_df,
        model_type=model_type,
        reconciliation_method=reconciliation_method,
        prefer_reconciled=prefer_reconciled,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        freq=freq,
    )
    if fc_df.empty:
        return ContractVolumeOptimizationResult(
            schedule_df=pd.DataFrame(columns=CONTRACT_VOLUME_COLUMNS),
            summary_df=pd.DataFrame(columns=CONTRACT_VOLUME_SUMMARY_COLUMNS),
            status="empty",
            is_feasible=True,
            baseline_portfolio_units=0.0,
            optimal_portfolio_units=0.0,
            target_contract_units=0.0,
            achieved_tier_units=0.0,
            rebate_earned=0.0,
            shortfall_units=0.0,
            shortfall_penalty=0.0,
            cvar_shortfall_units=0.0,
            baseline_net_value=0.0,
            optimal_net_value=0.0,
            net_value_lift=0.0,
        )

    ladder = sorted({float(np.clip(d, 0.0, 0.90)) for d in discount_ladder})
    if max_discount_pct is not None:
        ladder = [d for d in ladder if d <= float(max_discount_pct) + 1e-9]
    if not ladder or ladder[0] > 0.0:
        ladder = [0.0, *ladder]
    n_rungs = len(ladder)

    ts_ids = list(dict.fromkeys(fc_df["ts_id"].astype(str)))
    ref_df = eff_hist if (eff_hist is not None and not eff_hist.empty) else fc_df
    price_map = _resolve_sku_param(base_price, ts_ids, 20.0, df=ref_df, ts_id_col="ts_id")
    cost_map = _resolve_sku_param(unit_cost, ts_ids, 8.0, df=ref_df, ts_id_col="ts_id")
    inv_map = (
        None
        if initial_inventory is None
        else _resolve_sku_param(initial_inventory, ts_ids, 1e9, df=ref_df, ts_id_col="ts_id")
    )

    elast_map, source_map, _ = _resolve_price_elasticities_and_similarity(
        ts_ids,
        history_df=eff_hist,
        price_elasticity=price_elasticity,
        price_col=price_col,
        covariate_cols=covariate_cols,
        elasticity_df=elasticity_df,
        use_digital_twin_elasticity=use_digital_twin_elasticity,
        default_elasticity=default_elasticity,
        similarity_df=similarity_df,
        similarity_method=similarity_method,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
    )

    sku_groups: dict[str, pd.DataFrame] = {}
    d_hat: dict[str, np.ndarray] = {}
    base_portfolio_units = 0.0
    max_portfolio_units = 0.0

    for ts in ts_ids:
        grp = fc_df[fc_df["ts_id"] == ts].sort_values("ds").reset_index(drop=True)
        sku_groups[ts] = grp
        h_i = len(grp)
        mat = np.zeros((h_i, n_rungs), dtype=float)
        for t_idx in range(h_i):
            base_u = float(grp["yhat"].iloc[t_idx])
            for k_idx, disc in enumerate(ladder):
                mat[t_idx, k_idx] = _demand_at_discount(base_u, disc, elast_map[ts])
        if inv_map is not None and inv_map[ts] < float(mat[:, -1].sum()):
            scale = float(inv_map[ts]) / max(float(mat[:, -1].sum()), 1e-9)
            mat = mat * min(1.0, scale)
        d_hat[ts] = mat
        base_portfolio_units += float(mat[:, 0].sum())
        max_portfolio_units += float(mat[:, -1].sum())

    # Normalize contract_tiers into sorted list of (target_units, rebate_usd).
    tiers: list[tuple[float, float]] = []
    if contract_tiers is not None:
        raw_items = (
            list(contract_tiers.items())
            if isinstance(contract_tiers, Mapping)
            else list(contract_tiers)
        )
        for q_t, reb in raw_items:
            if float(q_t) > 0.0 and float(reb) >= 0.0:
                tiers.append((float(q_t), float(reb)))
        tiers.sort(key=lambda item: item[0])
    elif min_contract_volume is None:
        # Default 2-tier vendor rebate ladder at +10% and +20% above baseline volume.
        avg_margin = float(np.mean([max(1.0, price_map[ts] - cost_map[ts]) for ts in ts_ids]))
        tiers = [
            (
                round(1.10 * base_portfolio_units, 2),
                round(0.08 * base_portfolio_units * avg_margin, 2),
            ),
            (
                round(1.22 * base_portfolio_units, 2),
                round(0.18 * base_portfolio_units * avg_margin, 2),
            ),
        ]

    n_tiers = len(tiers)
    q_min = float(min_contract_volume) if min_contract_volume is not None else 0.0
    penalty_rate = max(0.0, float(shortfall_penalty_per_unit))

    # Estimate stochastic Copula / ACI / Conformal / OOF CVaR downside buffer.
    cvar_downside_frac = _estimate_cvar_downside_fraction(
        fc_df=fc_df,
        trajectories_df=trajectories_df,
        oof_df=oof_df,
        cvar_alpha=cvar_alpha,
        risk_aversion_lambda=risk_aversion_lambda,
    )

    risk_lam = max(0.0, float(risk_aversion_lambda))
    eff_q_min = q_min * (1.0 + risk_lam * cvar_downside_frac)

    relaxation_hint: str | None = None
    is_feasible = True
    if q_min > max_portfolio_units + 1e-6:
        is_feasible = False
        relaxation_hint = (
            f"min_contract_volume={q_min:.2f} exceeds maximum achievable portfolio demand "
            f"{max_portfolio_units:.2f} at max discount {ladder[-1]:.0%}; minimum unavoidable "
            f"shortfall is {q_min - max_portfolio_units:.2f} units."
        )

    # Build MILP variables:
    # - binary z_{ts, t, k} in {0, 1} for discount selection
    # - binary u_m in {0, 1} for contract tier m achievement
    # - continuous shortfall xi >= 0 for Take-or-Pay commitment
    idx_z: dict[tuple[str, int, int], int] = {}
    var_count = 0
    for ts in ts_ids:
        for t_idx in range(len(sku_groups[ts])):
            for k_idx in range(n_rungs):
                idx_z[(ts, t_idx, k_idx)] = var_count
                var_count += 1
    idx_u: list[int] = []
    for _ in range(n_tiers):
        idx_u.append(var_count)
        var_count += 1
    idx_xi = var_count
    var_count += 1

    c_obj = np.zeros(var_count, dtype=float)
    integrality = np.zeros(var_count, dtype=float)
    lb_vec = np.zeros(var_count, dtype=float)
    ub_vec = np.ones(var_count, dtype=float)

    for ts in ts_ids:
        p0 = price_map[ts]
        c_i = cost_map[ts]
        for t_idx in range(len(sku_groups[ts])):
            for k_idx, disc in enumerate(ladder):
                col = idx_z[(ts, t_idx, k_idx)]
                integrality[col] = 1.0
                dem_tk = float(d_hat[ts][t_idx, k_idx])
                margin_tk = (p0 * (1.0 - disc) - c_i) * dem_tk
                # Negate for minimization:
                c_obj[col] = -margin_tk + 1e-4 * disc

    for m_idx, (_, reb_val) in enumerate(tiers):
        u_col = idx_u[m_idx]
        integrality[u_col] = 1.0
        c_obj[u_col] = -float(reb_val)

    integrality[idx_xi] = 0.0
    ub_vec[idx_xi] = max(eff_q_min, max_portfolio_units, 1.0) * 2.0
    c_obj[idx_xi] = penalty_rate

    rows_a: list[np.ndarray] = []
    rows_lb: list[float] = []
    rows_ub: list[float] = []

    def _add_row(coeffs: dict[int, float], lower: float, upper: float) -> None:
        row = np.zeros(var_count, dtype=float)
        for c_i, val in coeffs.items():
            if val != 0.0:
                row[c_i] = val
        rows_a.append(row)
        rows_lb.append(lower)
        rows_ub.append(upper)

    # 1. One discount rung per (ts, t).
    for ts in ts_ids:
        for t_idx in range(len(sku_groups[ts])):
            _add_row(
                {idx_z[(ts, t_idx, k_idx)]: 1.0 for k_idx in range(n_rungs)},
                1.0,
                1.0,
            )

    # 2. At most one rebate tier selected, and portfolio units >= sum_m tau_m * u_m.
    if n_tiers > 0:
        _add_row(dict.fromkeys(idx_u, 1.0), -np.inf, 1.0)
        tier_link: dict[int, float] = {
            idx_z[(ts, t_idx, k_idx)]: float(d_hat[ts][t_idx, k_idx])
            for ts in ts_ids
            for t_idx in range(len(sku_groups[ts]))
            for k_idx in range(n_rungs)
        }
        for m_idx, (tau_m, _) in enumerate(tiers):
            tier_link[idx_u[m_idx]] = -float(tau_m)
        _add_row(tier_link, 0.0, np.inf)

    # 3. Take-or-Pay minimum volume constraint: portfolio_units + xi >= eff_q_min.
    if eff_q_min > 0.0:
        top_link: dict[int, float] = {
            idx_z[(ts, t_idx, k_idx)]: float(d_hat[ts][t_idx, k_idx])
            for ts in ts_ids
            for t_idx in range(len(sku_groups[ts]))
            for k_idx in range(n_rungs)
        }
        top_link[idx_xi] = 1.0
        _add_row(top_link, eff_q_min, np.inf)

    a_mat = np.vstack(rows_a) if rows_a else np.zeros((0, var_count), dtype=float)
    constraints = (
        LinearConstraint(a_mat, np.array(rows_lb, dtype=float), np.array(rows_ub, dtype=float))
        if rows_a
        else None
    )
    res = milp(
        c=c_obj,
        integrality=integrality,
        bounds=Bounds(lb_vec, ub_vec),
        constraints=constraints,
    )
    sol_x = res.x if (res.success and res.x is not None) else np.zeros(var_count, dtype=float)
    solver_ok = bool(res.success and res.x is not None)

    schedule_rows: list[dict[str, Any]] = []
    sku_stats: dict[str, dict[str, Any]] = {}
    opt_portfolio_units = 0.0
    base_margin_tot = 0.0
    opt_margin_tot = 0.0

    for ts in ts_ids:
        grp = sku_groups[ts]
        p0 = price_map[ts]
        c_i = cost_map[ts]
        base_u_i = 0.0
        opt_u_i = 0.0
        base_m_i = 0.0
        opt_m_i = 0.0
        discs_i: list[float] = []

        for t_idx in range(len(grp)):
            if solver_ok:
                z_vals = [float(sol_x[idx_z[(ts, t_idx, k_idx)]]) for k_idx in range(n_rungs)]
                best_k = int(np.argmax(z_vals))
            else:
                best_k = 0
            disc = float(ladder[best_k])
            discs_i.append(disc)
            opt_p = p0 * (1.0 - disc)
            b_u = float(d_hat[ts][t_idx, 0])
            o_u = float(d_hat[ts][t_idx, best_k])
            b_m = (p0 - c_i) * b_u
            o_m = (opt_p - c_i) * o_u

            base_u_i += b_u
            opt_u_i += o_u
            base_m_i += b_m
            opt_m_i += o_m
            opt_portfolio_units += o_u

            schedule_rows.append(
                {
                    "ts_id": ts,
                    "ds": grp["ds"].iloc[t_idx],
                    "horizon_step": t_idx + 1,
                    "baseline_price": float(p0),
                    "optimal_discount_pct": float(disc),
                    "optimal_price": float(opt_p),
                    "baseline_units": float(b_u),
                    "expected_units": float(o_u),
                    "incremental_units": float(o_u - b_u),
                    "cumulative_portfolio_units": float(opt_portfolio_units),
                    "unit_cost": float(c_i),
                    "baseline_margin": float(b_m),
                    "optimal_margin": float(o_m),
                }
            )

        base_margin_tot += base_m_i
        opt_margin_tot += opt_m_i
        sku_stats[ts] = {
            "baseline_units": base_u_i,
            "optimal_units": opt_u_i,
            "mean_discount_pct": float(np.mean(discs_i)) if discs_i else 0.0,
            "max_discount_pct": float(np.max(discs_i)) if discs_i else 0.0,
            "baseline_margin": base_m_i,
            "optimal_margin": opt_m_i,
        }

    # Evaluate earned rebate & shortfall on both baseline and optimal volume.
    base_rebate = 0.0
    achieved_tier_u = 0.0
    opt_rebate = 0.0
    for tau_m, reb_m in tiers:
        if base_portfolio_units + 1e-6 >= tau_m:
            base_rebate = max(base_rebate, reb_m)
        if opt_portfolio_units + 1e-6 >= tau_m and reb_m >= opt_rebate:
            opt_rebate = reb_m
            achieved_tier_u = tau_m

    base_shortfall = max(0.0, q_min - base_portfolio_units)
    opt_shortfall = max(0.0, q_min - opt_portfolio_units)
    base_penalty = base_shortfall * penalty_rate
    opt_penalty = opt_shortfall * penalty_rate
    cvar_shortfall = max(0.0, q_min - opt_portfolio_units * (1.0 - cvar_downside_frac))

    base_net_val = base_margin_tot + base_rebate - base_penalty
    opt_net_val = opt_margin_tot + opt_rebate - opt_penalty

    summary_rows: list[dict[str, Any]] = []
    for ts in ts_ids:
        st = sku_stats[ts]
        share = (
            st["optimal_units"] / opt_portfolio_units
            if opt_portfolio_units > 0
            else 1.0 / max(len(ts_ids), 1)
        )
        reb_alloc = opt_rebate * share
        pen_alloc = opt_penalty * share
        summary_rows.append(
            {
                "ts_id": ts,
                "baseline_units": float(st["baseline_units"]),
                "optimal_units": float(st["optimal_units"]),
                "incremental_units": float(st["optimal_units"] - st["baseline_units"]),
                "mean_discount_pct": float(st["mean_discount_pct"]),
                "max_discount_pct": float(st["max_discount_pct"]),
                "baseline_margin": float(st["baseline_margin"]),
                "optimal_margin": float(st["optimal_margin"]),
                "rebate_allocated": float(reb_alloc),
                "shortfall_penalty_allocated": float(pen_alloc),
                "net_contract_value": float(st["optimal_margin"] + reb_alloc - pen_alloc),
                "price_elasticity": float(elast_map[ts]),
                "elasticity_source": source_map[ts],
            }
        )

    target_u = max([q_min, *[t[0] for t in tiers]], default=0.0)
    status_str = "infeasible" if not is_feasible else ("optimal" if solver_ok else "feasible")
    summary_df_out = pd.DataFrame(summary_rows, columns=CONTRACT_VOLUME_SUMMARY_COLUMNS)
    rollup_df = _build_hierarchy_rollup_df(
        summary_df_out,
        source_df=ref_df,
        hierarchy_cols=hierarchy_cols,
        ts_id_col="ts_id",
    )

    return ContractVolumeOptimizationResult(
        schedule_df=pd.DataFrame(schedule_rows, columns=CONTRACT_VOLUME_COLUMNS),
        summary_df=summary_df_out,
        status=status_str,
        is_feasible=is_feasible,
        baseline_portfolio_units=float(base_portfolio_units),
        optimal_portfolio_units=float(opt_portfolio_units),
        target_contract_units=float(target_u),
        achieved_tier_units=float(achieved_tier_u),
        rebate_earned=float(opt_rebate),
        shortfall_units=float(opt_shortfall),
        shortfall_penalty=float(opt_penalty),
        cvar_shortfall_units=float(cvar_shortfall),
        baseline_net_value=float(base_net_val),
        optimal_net_value=float(opt_net_val),
        net_value_lift=float(opt_net_val - base_net_val),
        relaxation_hint=relaxation_hint,
        hierarchy_rollup_df=rollup_df,
    )


def optimize_contract_volume_df(
    predictions_df: pd.DataFrame,
    *,
    summary: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience wrapper around :func:`optimize_contract_volume` returning a tidy DataFrame.

    Returns ``result.summary_df`` when ``summary=True`` and ``result.schedule_df`` otherwise, with
    the full :class:`ContractVolumeOptimizationResult` attached to ``df.attrs["contract_result"]``.
    """
    res = optimize_contract_volume(predictions_df, **kwargs)
    out = res.summary_df.copy() if summary else res.schedule_df.copy()
    out.attrs["contract_result"] = res
    out.attrs["status"] = res.status
    out.attrs["is_feasible"] = res.is_feasible
    out.attrs["relaxation_hint"] = res.relaxation_hint
    out.attrs["hierarchy_rollup_df"] = res.hierarchy_rollup_df
    return out


def plot_clearance_schedule(
    result_or_df: ClearanceOptimizationResult | pd.DataFrame,
    *,
    ts_id: str | None = None,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot the optimal markdown discount ladder, inventory depletion trajectory, and profit/space
    summary for a clearance optimization run.
    """
    import matplotlib.pyplot as plt

    if isinstance(result_or_df, ClearanceOptimizationResult):
        sched_df = result_or_df.schedule_df
        sum_df = result_or_df.summary_df
    elif isinstance(result_or_df, pd.DataFrame):
        stored = result_or_df.attrs.get("clearance_result")
        if isinstance(stored, ClearanceOptimizationResult):
            sched_df = stored.schedule_df
            sum_df = stored.summary_df
        elif "horizon_step" in result_or_df.columns:
            sched_df = result_or_df
            sum_df = pd.DataFrame()
        else:
            sched_df = pd.DataFrame()
            sum_df = result_or_df
    else:
        raise TypeError("result_or_df must be a ClearanceOptimizationResult or pandas DataFrame.")

    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        ax_left, ax_right = axes[0], axes[1]
    elif isinstance(ax, Sequence) and len(ax) >= 2:
        ax_left, ax_right = ax[0], ax[1]
    else:
        ax_left = ax
        ax_right = None

    if not sched_df.empty:
        chosen_ts = (
            str(ts_id)
            if ts_id is not None and str(ts_id) in set(sched_df["ts_id"].astype(str))
            else str(sched_df["ts_id"].astype(str).iloc[0])
        )
        sub = sched_df[sched_df["ts_id"].astype(str) == chosen_ts].sort_values("horizon_step")
        steps = sub["horizon_step"].to_numpy(dtype=float)
        disc_pct = sub["optimal_discount_pct"].to_numpy(dtype=float) * 100.0
        rem_inv = sub["remaining_inventory"].to_numpy(dtype=float)

        ax_left.step(
            steps,
            disc_pct,
            where="post",
            color="#d93025",
            linewidth=2.2,
            label="Markdown Discount (%)",
        )
        ax_left.set_xlabel("Horizon Step")
        ax_left.set_ylabel("Markdown Discount (%)", color="#d93025")
        ax_left.set_ylim(bottom=-2.0, top=max(55.0, float(np.max(disc_pct)) + 8.0))
        ax_left.grid(True, linestyle="--", alpha=0.35)

        ax_inv = ax_left.twinx()
        ax_inv.plot(
            steps,
            rem_inv,
            color="#1a73e8",
            marker="o",
            markersize=4,
            linewidth=1.8,
            label="Remaining Inventory (units)",
        )
        ax_inv.set_ylabel("Remaining Inventory (units)", color="#1a73e8")
        if "is_expired_step" in sub.columns and sub["is_expired_step"].any():
            first_exp = int(sub.loc[sub["is_expired_step"], "horizon_step"].min())
            ax_left.axvline(
                first_exp - 0.5,
                color="#5f6368",
                linestyle=":",
                linewidth=1.5,
                label="Expiry Deadline",
            )
        ax_left.set_title(title or f"Clearance Ladder & Inventory Depletion ({chosen_ts})")

    if ax_right is not None and not sum_df.empty:
        skus = sum_df["ts_id"].astype(str).tolist()
        x = np.arange(len(skus))
        width = 0.36
        ax_right.bar(
            x - width / 2,
            sum_df["baseline_gross_profit"].to_numpy(dtype=float),
            width=width,
            color="#9aa0a6",
            label="Baseline Profit (0% Markdown)",
        )
        ax_right.bar(
            x + width / 2,
            sum_df["optimal_gross_profit"].to_numpy(dtype=float),
            width=width,
            color="#1e8e3e",
            label="Optimal Clearance Profit",
        )
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(skus, rotation=20, ha="right")
        ax_right.set_ylabel("Net Profit ($)")
        ax_right.set_title("Baseline vs. Optimal Clearance Profit by SKU")
        ax_right.legend(loc="best", frameon=False)
        ax_right.grid(True, axis="y", linestyle="--", alpha=0.35)

    return ax_left


def _build_pairwise_similarity_map(
    ts_ids: Sequence[str],
    *,
    similarity_df: pd.DataFrame | None,
    history_df: pd.DataFrame | None,
    preds_df: pd.DataFrame,
    similarity_method: str = "hybrid",
    text_cols: Sequence[str] | None = None,
    attribute_cols: Sequence[str] | None = None,
    covariate_cols: Sequence[str] | None = None,
    hierarchy_cols: Sequence[str] | str | None = "auto",
) -> dict[tuple[str, str], float]:
    """Return symmetric pairwise similarity ``{(sku_i, sku_j): sim}`` in ``[0, 1]``."""
    pair_map: dict[tuple[str, str], float] = {}
    ts_set = {str(t) for t in ts_ids}
    if len(ts_set) < 2:
        return pair_map

    pairs_df = similarity_df
    if pairs_df is None or pairs_df.empty:
        ref = history_df if (history_df is not None and not history_df.empty) else preds_df
        val_col = "y" if "y" in ref.columns else ("yhat" if "yhat" in ref.columns else None)
        if val_col is not None and "ts_id" in ref.columns and ref["ts_id"].nunique() >= 2:
            from .similarity import compute_similarity_matrix

            eff_hcols: list[str] | None = None
            if isinstance(hierarchy_cols, str):
                if hierarchy_cols != "auto" and hierarchy_cols in ref.columns:
                    eff_hcols = [hierarchy_cols]
            elif isinstance(hierarchy_cols, Sequence):
                eff_hcols = [str(c) for c in hierarchy_cols if str(c) in ref.columns]
            eff_acols = (
                list(attribute_cols)
                if attribute_cols
                else (
                    [str(c) for c in covariate_cols if str(c) in ref.columns]
                    if covariate_cols
                    else None
                )
            )
            sim_res = compute_similarity_matrix(
                ref,
                method=similarity_method,
                ts_id_col="ts_id",
                date_col="ds" if "ds" in ref.columns else "forecast_date",
                target_col=val_col,
                text_cols=list(text_cols) if text_cols else None,
                attribute_cols=eff_acols,
                hierarchy_cols=eff_hcols,
            )
            pairs_df = sim_res.pairs_df

    if pairs_df is not None and not pairs_df.empty:
        col_a = (
            "ts_id"
            if "ts_id" in pairs_df.columns
            else ("query_ts_id" if "query_ts_id" in pairs_df.columns else str(pairs_df.columns[0]))
        )
        col_b = (
            "neighbor_ts_id"
            if "neighbor_ts_id" in pairs_df.columns
            else (
                "peer_ts_id"
                if "peer_ts_id" in pairs_df.columns
                else (
                    "substitute_ts_id"
                    if "substitute_ts_id" in pairs_df.columns
                    else str(pairs_df.columns[1])
                )
            )
        )
        score_col = next(
            (
                c
                for c in (
                    "similarity",
                    "similarity_score",
                    "composite_similarity",
                    "substitute_score",
                    "trajectory_sim",
                )
                if c in pairs_df.columns
            ),
            None,
        )
        if score_col is not None:
            for row in pairs_df.itertuples(index=False):
                a = str(getattr(row, col_a))
                b = str(getattr(row, col_b))
                s = float(getattr(row, score_col))
                if a != b and a in ts_set and b in ts_set and np.isfinite(s):
                    val = float(np.clip(s, 0.0, 1.0))
                    pair_map[(a, b)] = max(pair_map.get((a, b), 0.0), val)
                    pair_map[(b, a)] = max(pair_map.get((b, a), 0.0), val)

    # Fallback: if no similarity pairs were resolved, assign moderate 0.35 within-panel affinity.
    if not pair_map and len(ts_ids) >= 2:
        for i, a in enumerate(ts_ids):
            for b in ts_ids[i + 1 :]:
                pair_map[(a, b)] = 0.35
                pair_map[(b, a)] = 0.35

    return pair_map


def optimize_assortment(
    predictions_df: pd.DataFrame,
    *,
    history_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    similarity_df: pd.DataFrame | None = None,
    elasticity_df: pd.DataFrame | None = None,
    similarity_method: str = "hybrid",
    max_skus: int | None = None,
    min_skus: int = 1,
    max_shelf_space: float | None = None,
    shelf_space: float | Mapping[str, float] | str = 1.0,
    base_price: float | Mapping[str, float] | str = 20.0,
    unit_cost: float | Mapping[str, float] | str = 10.0,
    fixed_slot_cost: float | Mapping[str, float] | str = 0.0,
    walk_rate: float | Mapping[str, float] | str | None = None,
    retention_scale: float = 0.85,
    min_walk_rate: float = 0.05,
    max_walk_rate: float = 0.95,
    min_similarity: float = 0.10,
    top_k_substitutes: int = 5,
    walk_penalty_weight: float = 0.0,
    min_category_retention: float | None = None,
    must_keep_skus: Sequence[str] | None = None,
    must_delist_skus: Sequence[str] | None = None,
    text_cols: Sequence[str] | None = None,
    attribute_cols: Sequence[str] | None = None,
    covariate_cols: Sequence[str] | None = None,
    hierarchy_cols: Sequence[str] | str | None = "auto",
    model_type: str | None = None,
    reconciliation_method: str | None = None,
    prefer_reconciled: bool = True,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    horizon: int = 14,
    freq: str = "D",
) -> AssortmentOptimizationResult:
    """Solve the assortment rationalization & delisting demand-transference MILP (`HiGHS`).

    Given $N$ candidate SKUs with horizon demand $D_i$, unit margin $m_i = p_i - c_i$, fixed slot
    cost $F_i$, and shelf space $s_i$, selects which SKUs to retain ($z_i = 1$) vs. delist
    ($z_i = 0$). When SKU $i$ is delisted, a fraction $t_{ij}$ of its demand transfers to retained
    substitute SKUs $j$ based on the multi-modal similarity matrix $\\mathbf{S}_{ij}$, while the
    remaining fraction $w_i^{\\text{walk}} = 1 - \\sum_{j \\text{ kept}} t_{ij}$ walks away from the
    category (`lost_walk_units`).
    """
    preds, eff_hist = _prepare_forecast_frame(
        predictions_df,
        history_df=history_df,
        oof_df=oof_df,
        model_type=model_type,
        reconciliation_method=reconciliation_method,
        prefer_reconciled=prefer_reconciled,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        freq=freq,
    )
    if preds.empty:
        return AssortmentOptimizationResult(
            assortment_df=pd.DataFrame(columns=ASSORTMENT_OPTIMIZER_COLUMNS),
            transference_df=pd.DataFrame(columns=ASSORTMENT_TRANSFERENCE_COLUMNS),
            status="optimal",
            is_feasible=True,
            total_candidate_skus=0,
            retained_skus=0,
            delisted_skus=0,
            baseline_category_units=0.0,
            post_assortment_units=0.0,
            transferred_units=0.0,
            lost_walk_units=0.0,
            category_retention_pct=1.0,
            mean_walk_rate=0.0,
            baseline_net_profit=0.0,
            optimal_net_profit=0.0,
            net_profit_lift=0.0,
            shelf_space_used=0.0,
        )

    ts_ids = sorted(preds["ts_id"].astype(str).unique().tolist())
    n_skus = len(ts_ids)
    sku_to_idx = {ts: idx for idx, ts in enumerate(ts_ids)}
    ref_df = eff_hist if (eff_hist is not None and not eff_hist.empty) else preds

    demand_series = preds.groupby("ts_id")["yhat"].sum()
    base_demand = {ts: float(max(0.0, demand_series.get(ts, 0.0))) for ts in ts_ids}
    price_map = _resolve_sku_param(base_price, ts_ids, 20.0, df=ref_df, ts_id_col="ts_id")
    cost_map = _resolve_sku_param(unit_cost, ts_ids, 10.0, df=ref_df, ts_id_col="ts_id")
    slot_map = _resolve_sku_param(fixed_slot_cost, ts_ids, 0.0, df=ref_df, ts_id_col="ts_id")
    space_map = _resolve_sku_param(shelf_space, ts_ids, 1.0, df=ref_df, ts_id_col="ts_id")
    for ts in ts_ids:
        price_map[ts] = max(0.01, price_map[ts])
        cost_map[ts] = max(0.0, min(cost_map[ts], price_map[ts] * 0.98))
        space_map[ts] = max(1e-6, space_map[ts])

    pair_sim = _build_pairwise_similarity_map(
        ts_ids,
        similarity_df=similarity_df,
        history_df=eff_hist,
        preds_df=preds,
        similarity_method=similarity_method,
        text_cols=text_cols,
        attribute_cols=attribute_cols,
        covariate_cols=covariate_cols,
        hierarchy_cols=hierarchy_cols,
    )

    # Integrate Double-ML cross-SKU cannibalization (gamma < 0) and positive Halo lift (gamma > 0)
    # when `elasticity_df` is supplied.
    halo_units_map: dict[str, float] = dict.fromkeys(ts_ids, 0.0)
    if elasticity_df is not None and not elasticity_df.empty and "ts_id" in elasticity_df.columns:
        ts_set = set(ts_ids)
        for _, erow in elasticity_df.iterrows():
            ts_e = str(erow["ts_id"])
            if ts_e not in ts_set:
                continue
            # Positive complementary Halo units driven by SKU `ts_e` across the category.
            for h_col in ("total_halo_units", "halo_lift", "halo_units"):
                if h_col in erow.index and pd.notna(erow[h_col]):
                    val_h = float(erow[h_col])
                    if np.isfinite(val_h) and val_h > 0.0:
                        halo_units_map[ts_e] = max(halo_units_map[ts_e], val_h)
            # Substitute cannibalization target boosts transference similarity when delisted.
            for c_col in ("top_cannibalized_ts_id", "substitute_ts_id"):
                if c_col in erow.index and pd.notna(erow[c_col]):
                    peer_c = str(erow[c_col])
                    if peer_c in ts_set and peer_c != ts_e:
                        boosted = min(1.0, pair_sim.get((ts_e, peer_c), 0.35) + 0.20)
                        pair_sim[(ts_e, peer_c)] = max(pair_sim.get((ts_e, peer_c), 0.0), boosted)
                        pair_sim[(peer_c, ts_e)] = max(pair_sim.get((peer_c, ts_e), 0.0), boosted)

    # Resolve intrinsic walk_rate and candidate substitute transference edges.
    explicit_walk: dict[str, float] | None = None
    if walk_rate is not None:
        explicit_walk = _resolve_sku_param(walk_rate, ts_ids, 0.30, df=ref_df, ts_id_col="ts_id")

    ret_scale = float(np.clip(retention_scale, 0.0, 1.0))
    w_lo = float(np.clip(min_walk_rate, 0.0, 1.0))
    w_hi = float(np.clip(max_walk_rate, w_lo, 1.0))
    sim_floor = float(np.clip(min_similarity, 0.0, 1.0))
    top_k = max(1, int(top_k_substitutes))

    walk_map: dict[str, float] = {}
    max_sim_map: dict[str, float] = {}
    primary_sub_map: dict[str, str] = {}
    # Directed edges: (i_idx, j_idx, similarity, transfer_share_t_ij)
    edges: list[tuple[int, int, float, float]] = []

    for i_idx, ts_i in enumerate(ts_ids):
        peers = [
            (ts_j, pair_sim.get((ts_i, ts_j), 0.0))
            for ts_j in ts_ids
            if ts_j != ts_i and pair_sim.get((ts_i, ts_j), 0.0) >= sim_floor
        ]
        peers.sort(key=lambda item: (-item[1], item[0]))
        top_peers = peers[:top_k]
        max_s = float(top_peers[0][1]) if top_peers else 0.0
        max_sim_map[ts_i] = max_s
        primary_sub_map[ts_i] = str(top_peers[0][0]) if top_peers else ""

        if explicit_walk is not None and ts_i in explicit_walk:
            w_i = float(np.clip(explicit_walk[ts_i], 0.0, 1.0))
        elif not top_peers:
            w_i = 1.0
        else:
            w_i = float(np.clip(1.0 - ret_scale * max_s, w_lo, w_hi))
        walk_map[ts_i] = w_i

        transfer_budget = max(0.0, 1.0 - w_i)
        if top_peers and transfer_budget > 0.0:
            weights = np.array([max(s, 1e-6) ** 1.5 for _, s in top_peers], dtype=float)
            w_sum = float(np.sum(weights))
            for (ts_j, s_val), w_raw in zip(top_peers, weights, strict=True):
                t_ij = float(transfer_budget * (w_raw / max(w_sum, 1e-9)))
                edges.append((i_idx, sku_to_idx[ts_j], float(s_val), t_ij))

    n_edges = len(edges)
    n_vars = n_skus + n_edges  # z_0..z_{N-1} in {0,1}, y_0..y_{E-1} in [0,1]

    keep_set = {str(s) for s in (must_keep_skus or []) if str(s) in sku_to_idx}
    delist_set = {str(s) for s in (must_delist_skus or []) if str(s) in sku_to_idx}
    conflict_set = keep_set & delist_set

    eff_min_skus = max(0, int(min_skus)) if n_skus > 0 else 0
    eff_max_skus = int(max_skus) if max_skus is not None else n_skus
    total_base_units = float(sum(base_demand.values()))

    # Feasibility pre-checks and relaxation hints.
    is_feasible = True
    relaxation_hint: str | None = None
    if conflict_set:
        is_feasible = False
        relaxation_hint = (
            f"Conflicting must_keep_skus and must_delist_skus on {sorted(conflict_set)}."
        )
    elif eff_max_skus < len(keep_set):
        is_feasible = False
        relaxation_hint = (
            f"max_skus={eff_max_skus} is smaller than len(must_keep_skus)={len(keep_set)}. "
            f"Increase max_skus to at least {len(keep_set)}."
        )
    elif eff_min_skus > n_skus - len(delist_set):
        is_feasible = False
        relaxation_hint = (
            f"min_skus={eff_min_skus} exceeds available non-delisted SKUs "
            f"({n_skus - len(delist_set)}). Lower min_skus to <= {n_skus - len(delist_set)}."
        )
    elif eff_max_skus < eff_min_skus:
        is_feasible = False
        relaxation_hint = (
            f"max_skus={eff_max_skus} < min_skus={eff_min_skus}. Set max_skus >= {eff_min_skus}."
        )
    elif max_shelf_space is not None:
        keep_space = sum(space_map[s] for s in keep_set)
        sorted_spaces = sorted(
            space_map[s] for s in ts_ids if s not in keep_set and s not in delist_set
        )
        need_more = max(0, eff_min_skus - len(keep_set))
        min_possible_space = keep_space + sum(sorted_spaces[:need_more])
        if float(max_shelf_space) + 1e-9 < min_possible_space:
            is_feasible = False
            relaxation_hint = (
                f"max_shelf_space={float(max_shelf_space):.2f} is below the minimum required "
                f"shelf space {min_possible_space:.2f} for min_skus={eff_min_skus} and "
                f"must_keep_skus. Increase max_shelf_space to >= {min_possible_space:.2f}."
            )

    # Build MILP objective (scipy.optimize.milp minimizes c^T x).
    c_vec = np.zeros(n_vars, dtype=float)
    lb_vars = np.zeros(n_vars, dtype=float)
    ub_vars = np.ones(n_vars, dtype=float)
    integrality = np.zeros(n_vars, dtype=int)
    integrality[:n_skus] = 1  # z_i binary

    for i_idx, ts_i in enumerate(ts_ids):
        m_i = price_map[ts_i] - cost_map[ts_i]
        d_i = base_demand[ts_i]
        f_i = slot_map[ts_i]
        halo_bonus = m_i * halo_units_map.get(ts_i, 0.0)
        walk_pen = float(walk_penalty_weight) * m_i * d_i
        # Retaining SKU i yields (m_i * d_i + halo_bonus - f_i) and avoids walk_pen.
        c_vec[i_idx] = -(m_i * d_i + halo_bonus - f_i + walk_pen)
        if ts_i in keep_set:
            lb_vars[i_idx] = 1.0
            ub_vars[i_idx] = 1.0
        elif ts_i in delist_set and ts_i not in keep_set:
            lb_vars[i_idx] = 0.0
            ub_vars[i_idx] = 0.0

    for e_idx, (i_idx, j_idx, _sim, t_ij) in enumerate(edges):
        ts_i = ts_ids[i_idx]
        ts_j = ts_ids[j_idx]
        d_i = base_demand[ts_i]
        m_i = price_map[ts_i] - cost_map[ts_i]
        m_j = price_map[ts_j] - cost_map[ts_j]
        walk_saved = float(walk_penalty_weight) * m_i * d_i * t_ij
        c_vec[n_skus + e_idx] = -(m_j * d_i * t_ij + walk_saved)

    rows_a: list[np.ndarray] = []
    rows_lb: list[float] = []
    rows_ub: list[float] = []

    # 1. Edge coupling constraints:
    #    y_{i->j} + z_i <= 1  (only transfers if i is delisted)
    #    y_{i->j} - z_j <= 0  (only transfers to j if j is kept)
    for e_idx, (i_idx, j_idx, _sim, _t_ij) in enumerate(edges):
        row1 = np.zeros(n_vars, dtype=float)
        row1[i_idx] = 1.0
        row1[n_skus + e_idx] = 1.0
        rows_a.append(row1)
        rows_lb.append(-np.inf)
        rows_ub.append(1.0)

        row2 = np.zeros(n_vars, dtype=float)
        row2[j_idx] = -1.0
        row2[n_skus + e_idx] = 1.0
        rows_a.append(row2)
        rows_lb.append(-np.inf)
        rows_ub.append(0.0)

    # 2. SKU count bounds: min_skus <= sum_i z_i <= max_skus
    if is_feasible:
        row_cnt = np.zeros(n_vars, dtype=float)
        row_cnt[:n_skus] = 1.0
        rows_a.append(row_cnt)
        rows_lb.append(float(eff_min_skus))
        rows_ub.append(float(eff_max_skus))

    # 3. Shelf space cap: sum_i s_i z_i <= max_shelf_space
    if is_feasible and max_shelf_space is not None:
        row_sp = np.zeros(n_vars, dtype=float)
        for i_idx, ts_i in enumerate(ts_ids):
            row_sp[i_idx] = space_map[ts_i]
        rows_a.append(row_sp)
        rows_lb.append(-np.inf)
        rows_ub.append(float(max_shelf_space))

    # 4. Category retention floor: sum_i D_i z_i + sum_e D_i t_ij y_e >= rho_min * sum_i D_i
    if is_feasible and min_category_retention is not None and total_base_units > 0.0:
        rho = float(np.clip(min_category_retention, 0.0, 1.0))
        row_ret = np.zeros(n_vars, dtype=float)
        for i_idx, ts_i in enumerate(ts_ids):
            row_ret[i_idx] = base_demand[ts_i]
        for e_idx, (i_idx, _j_idx, _sim, t_ij) in enumerate(edges):
            row_ret[n_skus + e_idx] = base_demand[ts_ids[i_idx]] * t_ij
        rows_a.append(row_ret)
        rows_lb.append(rho * total_base_units)
        rows_ub.append(np.inf)

    z_sol = np.ones(n_skus, dtype=float)
    y_sol = np.zeros(n_edges, dtype=float)
    solver_ok = False

    if is_feasible:
        constraints = LinearConstraint(np.vstack(rows_a), rows_lb, rows_ub) if rows_a else None
        res = milp(
            c=c_vec,
            integrality=integrality,
            bounds=Bounds(lb_vars, ub_vars),
            constraints=constraints,
        )
        if res.success and res.x is not None:
            solver_ok = True
            z_sol = np.round(np.clip(res.x[:n_skus], 0.0, 1.0))
            y_sol = np.clip(res.x[n_skus:], 0.0, 1.0)
        elif min_category_retention is not None:
            is_feasible = False
            relaxation_hint = (
                f"min_category_retention={float(min_category_retention):.2f} is infeasible under "
                f"max_skus={eff_max_skus} / max_shelf_space={max_shelf_space} because delisted "
                f"SKUs incur walk_rate losses. Lower min_category_retention or increase max_skus."
            )

    if not solver_ok:
        # Deterministic greedy fallback ranking by net standalone profit per unit of shelf space.
        scores = []
        for i_idx, ts_i in enumerate(ts_ids):
            m_i = price_map[ts_i] - cost_map[ts_i]
            prof = m_i * base_demand[ts_i] - slot_map[ts_i]
            scores.append((prof / max(space_map[ts_i], 1e-6), i_idx))
        scores.sort(key=lambda item: (-item[0], item[1]))
        z_sol = np.zeros(n_skus, dtype=float)
        used_sp = 0.0
        kept_cnt = 0
        for ts_k in keep_set:
            idx_k = sku_to_idx[ts_k]
            z_sol[idx_k] = 1.0
            used_sp += space_map[ts_k]
            kept_cnt += 1
        for _sc, idx_cand in scores:
            ts_c = ts_ids[idx_cand]
            if z_sol[idx_cand] > 0.5 or ts_c in delist_set:
                continue
            if kept_cnt >= eff_max_skus:
                break
            if (
                max_shelf_space is not None
                and used_sp + space_map[ts_c] > float(max_shelf_space) + 1e-9
            ):
                continue
            if (_sc > 0.0) or (kept_cnt < eff_min_skus):
                z_sol[idx_cand] = 1.0
                used_sp += space_map[ts_c]
                kept_cnt += 1
        for e_idx, (i_idx, j_idx, _sim, _t_ij) in enumerate(edges):
            y_sol[e_idx] = 1.0 if (z_sol[i_idx] < 0.5 and z_sol[j_idx] > 0.5) else 0.0

    # Assemble transference_df and assortment_df.
    trans_out_units = dict.fromkeys(ts_ids, 0.0)
    trans_in_units = dict.fromkeys(ts_ids, 0.0)
    trans_rows: list[dict[str, Any]] = []

    for e_idx, (i_idx, j_idx, s_val, t_ij) in enumerate(edges):
        act = float(y_sol[e_idx])
        if act > 1e-6 and z_sol[i_idx] < 0.5 and z_sol[j_idx] > 0.5:
            ts_i = ts_ids[i_idx]
            ts_j = ts_ids[j_idx]
            u_flow = float(base_demand[ts_i] * t_ij * act)
            m_j = price_map[ts_j] - cost_map[ts_j]
            trans_out_units[ts_i] += u_flow
            trans_in_units[ts_j] += u_flow
            trans_rows.append(
                {
                    "delisted_ts_id": ts_i,
                    "recipient_ts_id": ts_j,
                    "similarity": float(s_val),
                    "transfer_share": float(t_ij * act),
                    "transferred_units": u_flow,
                    "transferred_margin": float(u_flow * m_j),
                }
            )

    status_str = "infeasible" if not is_feasible else ("optimal" if solver_ok else "feasible")
    assort_rows: list[dict[str, Any]] = []
    base_net_profit = 0.0
    opt_net_profit = 0.0
    total_post_units = 0.0
    total_lost_walk = 0.0
    total_space_used = 0.0

    for i_idx, ts_i in enumerate(ts_ids):
        kept = bool(z_sol[i_idx] >= 0.5)
        d_i = base_demand[ts_i]
        p_i = price_map[ts_i]
        c_i = cost_map[ts_i]
        m_i = p_i - c_i
        f_i = slot_map[ts_i]
        sp_i = space_map[ts_i]
        w_i = walk_map[ts_i]

        t_out = 0.0 if kept else float(min(d_i, trans_out_units[ts_i]))
        t_in = float(trans_in_units[ts_i]) if kept else 0.0
        lost_u = 0.0 if kept else float(max(0.0, d_i - t_out))
        post_u = float(d_i + t_in) if kept else 0.0

        base_prof_i = float(m_i * d_i - f_i)
        post_prof_i = float(m_i * post_u - f_i) if kept else 0.0
        base_net_profit += base_prof_i
        opt_net_profit += post_prof_i
        total_post_units += post_u
        total_lost_walk += lost_u
        if kept:
            total_space_used += sp_i

        assort_rows.append(
            {
                "ts_id": ts_i,
                "action": "KEEP" if kept else "DELIST",
                "is_retained": kept,
                "baseline_units": float(d_i),
                "walk_rate": float(w_i),
                "transferable_share": float(1.0 - w_i),
                "transferred_out_units": float(t_out),
                "transferred_in_units": float(t_in),
                "lost_walk_units": float(lost_u),
                "post_assortment_units": float(post_u),
                "unit_price": float(p_i),
                "unit_cost": float(c_i),
                "unit_margin": float(m_i),
                "fixed_slot_cost": float(f_i),
                "shelf_space": float(sp_i),
                "baseline_profit": float(base_prof_i),
                "post_assortment_profit": float(post_prof_i),
                "net_profit_contribution": float(post_prof_i - base_prof_i),
                "primary_substitute_ts_id": primary_sub_map[ts_i],
                "max_peer_similarity": float(max_sim_map[ts_i]),
                "status": status_str,
            }
        )

    assort_df = pd.DataFrame(assort_rows, columns=ASSORTMENT_OPTIMIZER_COLUMNS)
    trans_df = pd.DataFrame(trans_rows, columns=ASSORTMENT_TRANSFERENCE_COLUMNS)
    retained_cnt = int(sum(1 for r in assort_rows if r["is_retained"]))
    delisted_cnt = int(n_skus - retained_cnt)
    cat_ret_pct = float(total_post_units / total_base_units) if total_base_units > 1e-9 else 1.0
    rollup_df = _build_hierarchy_rollup_df(
        assort_df,
        source_df=ref_df,
        hierarchy_cols=hierarchy_cols,
        ts_id_col="ts_id",
    )

    return AssortmentOptimizationResult(
        assortment_df=assort_df,
        transference_df=trans_df,
        status=status_str,
        is_feasible=is_feasible,
        total_candidate_skus=n_skus,
        retained_skus=retained_cnt,
        delisted_skus=delisted_cnt,
        baseline_category_units=float(total_base_units),
        post_assortment_units=float(total_post_units),
        transferred_units=float(sum(trans_in_units.values())),
        lost_walk_units=float(total_lost_walk),
        category_retention_pct=float(np.clip(cat_ret_pct, 0.0, 1.0)),
        mean_walk_rate=float(np.mean(list(walk_map.values()))) if walk_map else 0.0,
        baseline_net_profit=float(base_net_profit),
        optimal_net_profit=float(opt_net_profit),
        net_profit_lift=float(opt_net_profit - base_net_profit),
        shelf_space_used=float(total_space_used),
        relaxation_hint=relaxation_hint,
        hierarchy_rollup_df=rollup_df,
    )


def optimize_assortment_df(
    predictions_df: pd.DataFrame,
    *,
    transference: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience wrapper around :func:`optimize_assortment` returning a tidy DataFrame.

    Returns ``result.transference_df`` when ``transference=True`` and ``result.assortment_df``
    otherwise, with the full :class:`AssortmentOptimizationResult` attached to
    ``df.attrs["assortment_result"]``.
    """
    res = optimize_assortment(predictions_df, **kwargs)
    out = res.transference_df.copy() if transference else res.assortment_df.copy()
    out.attrs["assortment_result"] = res
    out.attrs["status"] = res.status
    out.attrs["is_feasible"] = res.is_feasible
    out.attrs["relaxation_hint"] = res.relaxation_hint
    out.attrs["hierarchy_rollup_df"] = res.hierarchy_rollup_df
    return out
