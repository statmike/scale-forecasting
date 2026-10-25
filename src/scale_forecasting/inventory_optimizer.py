"""Multi-Echelon & Portfolio-Constrained Inventory Optimizer — pure DataFrame analytics.

Extends single-SKU inventory policy translation
(:func:`~scale_forecasting.diagnostics.compute_inventory_policy`)
to portfolio-constrained multi-SKU replenishment under shared working-capital budgets, warehouse
cubic-volume ceilings, total unit caps, portfolio fill-rate targets, and discrete supplier minimum
order quantities (`vendor_moq` / `min_order_qty`) and case-pack multiples (`order_multiple`):

1. **Copula-Backed or Analytical Protection-Window Uncertainty (`uncertainty_source`):**
   Computes each SKU's protection-window demand distribution $D_{i,P}$ over $P_i = L_i + R_i$ steps
   either from joint multi-horizon copula trajectories (:mod:`~scale_forecasting.copula`,
   ``uncertainty_source="copula"``) or from AR(1)-adjusted parametric error covariance
   (``uncertainty_source="analytical"``).
2. **Stage 1 — Continuous KKT / Lagrangian Dual Shadow Prices:**
   Solves the multi-constraint Lagrangian dual via ``scipy.optimize.root_scalar`` for exact
   economic shadow prices
   ($\\lambda_{\\text{budget}}^* \\ge 0$, $\\lambda_{\\text{volume}}^* \\ge 0$,
   $\\lambda_{\\text{units}}^* \\ge 0$) equalizing marginal expected stockout cost reduction per
   dollar/volume unit across all active SKUs:
   $$\\alpha_i^*(\\boldsymbol{\\lambda}) = \\frac{p_{i,\\text{eff}}}{p_{i,\\text{eff}} + h_i
     + \\lambda_{\\text{budget}} c_i + \\lambda_{\\text{volume}} v_i + \\lambda_{\\text{units}}}$$
   Guarantees ``shadow_price_budget > 0`` if and only if ``max_budget`` binds, and
   ``shadow_price_volume > 0`` if and only if ``max_volume`` binds.
3. **Stage 2 — Discrete Lot-Sizing via C++ `HiGHS` MILP (`scipy.optimize.milp`):**
   When discrete ``vendor_moq`` (``min_order_qty``) or ``order_multiple`` constraints are active,
   solves a Mixed-Integer Linear Program over binary MOQ activation indicators $u_i \\in \\{0, 1\\}$
   and integer case-pack counts $n_i \\in \\mathbb{Z}_{\\ge 0}$
   ($Q_i = \\text{base}_i u_i + m_i n_i$)
   using ``scipy.optimize.milp`` (`HiGHS`) so working-capital (``max_budget``), warehouse capacity
   (``max_volume``), and discrete lot-sizing constraints are satisfied to ``< 1e-6``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp, root_scalar

from .diagnostics import (
    INVENTORY_POLICY_COLUMNS,
    compute_inventory_policy,
)
from .errors import ConfigError, GoalSeekInfeasibleError, require_extra

__all__ = [
    "INVENTORY_OPTIMIZER_COLUMNS",
    "INVENTORY_POLICY_COLUMNS",
    "compute_inventory_frontier",
    "optimize_inventory_df",
    "optimize_inventory_portfolio",
    "plot_inventory_frontier",
]

_APPENDED_OPTIMIZER_COLUMNS: tuple[str, ...] = (
    "unconstrained_service_level",
    "unconstrained_safety_stock",
    "unconstrained_order_qty",
    "unconstrained_total_cost",
    "unit_cost",
    "unit_volume",
    "vendor_moq",
    "order_cost",
    "order_spend",
    "order_volume",
    "allocation_ratio",
    "shadow_price_budget",
    "shadow_price_volume",
    "shadow_price_units",
    "portfolio_fill_rate",
    "binding_constraint",
    "uncertainty_source",
)

INVENTORY_OPTIMIZER_COLUMNS: tuple[str, ...] = (
    *INVENTORY_POLICY_COLUMNS,
    *_APPENDED_OPTIMIZER_COLUMNS,
)


class _InventoryConfigError(ConfigError, ValueError):
    """Inventory optimizer error inheriting from both `ConfigError` and `ValueError`."""


def _normal_loss(z: float) -> float:
    """Standard normal first-order loss function $G(z) = \\phi(z) - z\\,(1 - \\Phi(z))$."""
    zf = float(z)
    if zf >= 8.0:
        return 0.0
    if zf <= -8.0:
        return -zf
    return float(stats.norm.pdf(zf) - zf * stats.norm.sf(zf))


def _solve_z_for_fill_rate(
    fill_rate: float,
    protection_demand: float,
    protection_std: float,
) -> float:
    """Invert $G(z) = (1 - \\beta)\\,\\mu_P / \\sigma_P$ for the non-negative safety factor $z$."""
    if protection_std <= 1e-12:
        return 0.0
    target_g = (1.0 - float(fill_rate)) * max(float(protection_demand), 1e-9) / protection_std
    g_zero = _normal_loss(0.0)
    if target_g >= g_zero:
        return 0.0
    if target_g <= _normal_loss(6.0):
        return 6.0
    lo, hi = 0.0, 6.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if _normal_loss(mid) > target_g:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _window_sum_1d(values: np.ndarray, steps: float) -> float:
    """Sum the first ``steps`` elements of 1D ``values``, extrapolating with the mean."""
    if values.size == 0 or steps <= 0.0:
        return 0.0
    mean_v = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
    full_steps = math.floor(steps)
    frac = float(steps) - full_steps
    avail = min(full_steps, int(values.size))
    total = float(np.nansum(values[:avail])) if avail > 0 else 0.0
    if full_steps > avail:
        total += float(full_steps - avail) * mean_v
    if frac > 1e-9:
        next_val = float(values[avail]) if avail < int(values.size) else mean_v
        total += frac * (next_val if math.isfinite(next_val) else mean_v)
    return max(0.0, total)


def _window_sum_2d(paths_matrix: np.ndarray, steps: float) -> np.ndarray:
    """Compute protection-window demand per path across ``(n_paths, horizon)`` matrix."""
    n_paths, n_steps = paths_matrix.shape
    if n_paths == 0 or n_steps == 0 or steps <= 0.0:
        return np.zeros(max(1, n_paths), dtype=float)
    row_means = np.nanmean(paths_matrix, axis=1)
    row_means = np.where(np.isfinite(row_means), row_means, 0.0)
    full_steps = math.floor(steps)
    frac = float(steps) - full_steps
    avail = min(full_steps, n_steps)
    total = np.nansum(paths_matrix[:, :avail], axis=1) if avail > 0 else np.zeros(n_paths)
    if full_steps > avail:
        total = total + float(full_steps - avail) * row_means
    if frac > 1e-9:
        if avail < n_steps:
            next_col = paths_matrix[:, avail]
            next_col = np.where(np.isfinite(next_col), next_col, row_means)
        else:
            next_col = row_means
        total = total + frac * next_col
    return np.maximum(0.0, total)


def _extract_predictions_and_oof(
    predictions_or_obj: Any,
    oof_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Extract `(predictions_df, oof_df)` from a DataFrame, `Forecaster`, or `BakeOff`."""
    if isinstance(predictions_or_obj, pd.DataFrame):
        return predictions_or_obj, oof_df
    if hasattr(predictions_or_obj, "predictions") and isinstance(
        predictions_or_obj.predictions, pd.DataFrame
    ):
        pred = predictions_or_obj.predictions
        oof = oof_df if oof_df is not None else getattr(predictions_or_obj, "oof", None)
        return pred, oof
    if hasattr(predictions_or_obj, "_resolve_track_k_frames"):
        pred, resolved_oof, _ = predictions_or_obj._resolve_track_k_frames(oof_df=oof_df)
        return pred, resolved_oof
    raise _InventoryConfigError(
        "predictions_df must be a DataFrame or a fitted Forecaster / BakeOff / PlaygroundRun."
    )


def _extract_history_df(
    predictions_or_obj: Any,
    history_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Resolve `history_df` from an explicit DataFrame or a fitted `Forecaster` / `BakeOff`."""
    if history_df is not None:
        return history_df
    if isinstance(predictions_or_obj, pd.DataFrame):
        hist_attr = predictions_or_obj.attrs.get("history_df")
        return hist_attr if isinstance(hist_attr, pd.DataFrame) else None
    for attr in ("_history_df", "history_df", "series"):
        cand = getattr(predictions_or_obj, attr, None)
        if isinstance(cand, pd.DataFrame):
            return cand
    return None


def _resolve_sku_param(
    param: Any,
    tid: str,
    sku_override: Mapping[str, Any],
    field_name: str,
    default_val: Any,
    *,
    alt_field_names: tuple[str, ...] = (),
) -> Any:
    """Resolve a scalar or per-SKU mapping/dataframe override for ``tid``."""
    for key in (field_name, *alt_field_names):
        if key in sku_override and sku_override[key] is not None:
            return sku_override[key]
    if isinstance(param, Mapping):
        if tid in param and param[tid] is not None:
            return param[tid]
        return default_val
    if param is not None:
        return param
    return default_val


def _normalize_inventory_overrides(
    inventory_df: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize per-SKU inventory overrides into ``{ts_id: {field: value}}``."""
    if inventory_df is None:
        return {}
    if isinstance(inventory_df, Mapping):
        return {str(k): dict(v) for k, v in inventory_df.items() if isinstance(v, Mapping)}
    if isinstance(inventory_df, pd.DataFrame):
        if inventory_df.empty:
            return {}
        df = inventory_df.copy()
        if "ts_id" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "ts_id"})
        out: dict[str, dict[str, Any]] = {}
        for row in df.to_dict(orient="records"):
            tid = str(row.get("ts_id", ""))
            if tid:
                out[tid] = {
                    str(k): v
                    for k, v in row.items()
                    if k != "ts_id"
                    and v is not None
                    and not (isinstance(v, float) and math.isnan(v))
                }
        return out
    return {}


def _fallback_sample_trajectories(
    predictions_df: pd.DataFrame,
    oof_df: pd.DataFrame | None,
    *,
    n_paths: int = 500,
    copula_type: str = "auto",
    seed: int = 42,
    confidence_level: float = 0.80,
) -> pd.DataFrame:
    """Internal Gaussian / Student-t AR(1) trajectory sampler when `copula.py` is not invoked."""
    rng = np.random.default_rng(int(seed))
    df = predictions_df.copy()
    if "segment" in df.columns and (df["segment"] == "forecast").any():
        df = df[df["segment"] == "forecast"].copy()
    date_col = (
        "forecast_date"
        if "forecast_date" in df.columns
        else ("target_date" if "target_date" in df.columns else "ds")
    )
    yhat_col = "yhat" if "yhat" in df.columns else "forecast"
    lo_col = "yhat_lower" if "yhat_lower" in df.columns else "lower_bound"
    hi_col = "yhat_upper" if "yhat_upper" in df.columns else "upper_bound"
    if "ts_id" not in df.columns:
        df["ts_id"] = "series_0"
    if "model_type" not in df.columns:
        df["model_type"] = "model"

    eff_conf = float(confidence_level) if 0.0 < float(confidence_level) < 1.0 else 0.80
    z_conf = max(float(stats.norm.ppf(0.5 + 0.5 * eff_conf)), 1e-6)

    # Estimate per-series lag-1 residual autocorrelation and excess kurtosis from oof_df
    rho_map: dict[tuple[str, str], float] = {}
    kurt_list: list[float] = []
    if oof_df is not None and not oof_df.empty:
        o_tid = "ts_id" if "ts_id" in oof_df.columns else None
        o_mt = "model_type" if "model_type" in oof_df.columns else None
        y_col = "y_true" if "y_true" in oof_df.columns else ("y" if "y" in oof_df.columns else None)
        yh_o = "yhat" if "yhat" in oof_df.columns else None
        if o_tid and y_col and yh_o:
            work_o = oof_df.copy()
            if o_mt is None:
                work_o["model_type"] = str(df["model_type"].iloc[0])
                o_mt = "model_type"
            for (tid_k, mt_k), grp_o in work_o.groupby([o_tid, o_mt], sort=False):
                e = (
                    (
                        pd.to_numeric(grp_o[y_col], errors="coerce")
                        - pd.to_numeric(grp_o[yh_o], errors="coerce")
                    )
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if e.size >= 4:
                    e_c = e - float(np.mean(e))
                    denom = float(np.dot(e_c, e_c))
                    if denom > 1e-12:
                        rho_est = float(np.dot(e_c[:-1], e_c[1:]) / denom)
                        rho_map[(str(tid_k), str(mt_k))] = float(np.clip(rho_est, -0.5, 0.92))
                    if e.size >= 8:
                        kurt_list.append(float(stats.kurtosis(e, fisher=True, bias=False)))

    use_t = copula_type == "student_t" or (
        copula_type == "auto" and bool(kurt_list) and float(np.nanmedian(kurt_list)) > 1.0
    )
    df_t = 4.0

    out_parts: list[pd.DataFrame] = []
    df["_dt"] = pd.to_datetime(df[date_col])
    df = df.sort_values(["ts_id", "model_type", "_dt"])

    for (tid, mtype), grp in df.groupby(["ts_id", "model_type"], sort=False):
        yh = np.maximum(
            0.0,
            pd.to_numeric(grp[yhat_col], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        )
        h_steps = len(yh)
        if h_steps == 0:
            continue
        if hi_col in grp.columns:
            yu_raw = pd.to_numeric(grp[hi_col], errors="coerce").to_numpy(dtype=float)
            yu = np.where(np.isfinite(yu_raw), yu_raw, yh)
            if lo_col in grp.columns:
                yl_raw = pd.to_numeric(grp[lo_col], errors="coerce").to_numpy(dtype=float)
                yl = np.where(np.isfinite(yl_raw), yl_raw, yh)
                half_w = np.maximum(yu - yh, 0.5 * (yu - yl))
            else:
                half_w = np.maximum(0.0, yu - yh)
            sigmas = np.maximum(half_w / z_conf, 1e-6)
        else:
            sigmas = np.maximum(0.15 * yh, 1e-6)

        rho = rho_map.get((str(tid), str(mtype)), 0.35)
        idx = np.arange(h_steps)
        corr = rho ** np.abs(idx[:, None] - idx[None, :])
        corr = 0.98 * corr + 0.02 * np.eye(h_steps)
        chol = np.linalg.cholesky(corr)

        z = rng.standard_normal((int(n_paths), h_steps)) @ chol.T
        if use_t:
            chi = rng.chisquare(df_t, size=(int(n_paths), 1))
            z = z / np.sqrt(np.maximum(chi / df_t, 1e-6))

        samples = np.maximum(0.0, yh[None, :] + z * sigmas[None, :])
        path_ids = np.repeat(np.arange(int(n_paths), dtype=int), h_steps)
        step_ids = np.tile(np.arange(1, h_steps + 1, dtype=int), int(n_paths))
        ds_vals = np.tile(grp[date_col].to_numpy(), int(n_paths))
        yh_tile = np.tile(yh, int(n_paths))
        part = pd.DataFrame(
            {
                "path_id": path_ids,
                "ts_id": str(tid),
                "model_type": str(mtype),
                "forecast_date": ds_vals,
                "ds": ds_vals,
                "horizon_step": step_ids,
                "yhat_sample": samples.reshape(-1),
                "yhat": yh_tile,
            }
        )
        out_parts.append(part)

    if not out_parts:
        return pd.DataFrame(
            columns=[
                "path_id",
                "ts_id",
                "model_type",
                "forecast_date",
                "ds",
                "horizon_step",
                "yhat_sample",
                "yhat",
            ]
        )
    res = pd.concat(out_parts, ignore_index=True)
    res.attrs["copula_type_selected"] = "student_t" if use_t else "gaussian"
    return res


def _get_or_sample_copula_paths(
    predictions_df: pd.DataFrame,
    oof_df: pd.DataFrame | None,
    *,
    history_df: pd.DataFrame | None = None,
    trajectories_df: pd.DataFrame | None,
    copula_paths: pd.DataFrame | None,
    copula_type: str,
    n_paths: int,
    seed: int,
    confidence_level: float,
    similarity_shrinkage: bool = True,
    similarity_weight: float | Literal["auto"] = "auto",
    similarity_matrix: pd.DataFrame | np.ndarray | None = None,
    similarity_method: str = "hybrid",
) -> pd.DataFrame:
    """Resolve pre-computed copula trajectories or sample them via `copula.py` (with fallback)."""
    if trajectories_df is not None and not trajectories_df.empty:
        return trajectories_df
    if copula_paths is not None and not copula_paths.empty:
        return copula_paths

    try:
        import importlib

        copula_mod = importlib.import_module("scale_forecasting.copula")
        sample_fn = getattr(copula_mod, "sample_trajectories", None)
        if sample_fn is not None:
            return sample_fn(
                predictions_df,
                oof_df=oof_df,
                history_df=history_df,
                n_paths=int(n_paths),
                copula_type=copula_type,
                cross_series=True,
                reconcile=False,
                similarity_shrinkage=bool(similarity_shrinkage),
                similarity_weight=similarity_weight,
                similarity_matrix=similarity_matrix,
                similarity_method=similarity_method,
                seed=int(seed),
            )
    except Exception as exc:  # noqa: BLE001 - fallback when copula module is unavailable or fails
        logging.getLogger(__name__).debug(
            "Copula trajectory sampling fell back to parametric simulation: %s", exc
        )

    return _fallback_sample_trajectories(
        predictions_df,
        oof_df=oof_df,
        n_paths=int(n_paths),
        copula_type=copula_type,
        seed=int(seed),
        confidence_level=confidence_level,
    )


def optimize_inventory_portfolio(
    predictions_df: pd.DataFrame,
    oof_df: pd.DataFrame | None = None,
    *,
    history_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None = None,
    trajectories_df: pd.DataFrame | None = None,
    copula_paths: pd.DataFrame | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    max_budget: float | None = None,
    max_volume: float | None = None,
    max_total_units: float | None = None,
    target_portfolio_fill_rate: float | None = None,
    min_portfolio_fill_rate: float | None = None,
    unit_cost: float | Mapping[str, float] = 10.0,
    unit_volume: float | Mapping[str, float] = 1.0,
    holding_cost: float | Mapping[str, float] = 1.0,
    stockout_cost: float | Mapping[str, float] = 9.0,
    lead_time: float | Mapping[str, float] = 7.0,
    lead_time_std: float | Mapping[str, float] = 0.0,
    review_period: float | Mapping[str, float] = 0.0,
    service_level: float | str | Mapping[str, float | str] | None = 0.95,
    fill_rate: float | Mapping[str, float] | None = None,
    min_service_level: float = 0.50,
    on_hand: float | Mapping[str, float] = 0.0,
    on_order: float | Mapping[str, float] = 0.0,
    backorders: float | Mapping[str, float] = 0.0,
    vendor_moq: float | Mapping[str, float] | None = None,
    min_order_qty: float | Mapping[str, float] = 0.0,
    order_multiple: float | Mapping[str, float] | None = None,
    uncertainty_source: Literal["copula", "analytical"] = "copula",
    copula_type: str = "auto",
    n_paths: int = 500,
    seed: int = 42,
    random_seed: int | None = None,
    confidence_level: float = 0.80,
    strict_feasibility: bool = False,
    similarity_shrinkage: bool = True,
    similarity_weight: float | Literal["auto"] = "auto",
    similarity_matrix: pd.DataFrame | np.ndarray | None = None,
    similarity_method: str = "hybrid",
) -> pd.DataFrame:
    """Optimize multi-SKU replenishment policies under shared budget, volume, and lot constraints.

    Solves a two-stage Operations Research replenishment problem:

    1. **Continuous KKT / Lagrangian Dual Stage:** Solves for exact economic shadow prices
       ($\\lambda_{\\text{budget}}^* \\ge 0$, $\\lambda_{\\text{volume}}^* \\ge 0$,
       $\\lambda_{\\text{units}}^* \\ge 0$) equalizing marginal expected stockout cost reduction per
       resource unit across all SKUs. Guarantees ``shadow_price_budget > 0`` if and only if
       ``max_budget`` binds, and ``shadow_price_volume > 0`` if and only if ``max_volume`` binds.
    2. **Discrete Lot-Sizing MILP Stage (`HiGHS`):** When ``vendor_moq`` (``min_order_qty``) or
       ``order_multiple`` are active, refines SKU order quantities via ``scipy.optimize.milp``
       over binary MOQ activation indicators $u_i \\in \\{0, 1\\}$ and integer case-pack variables
       $n_i \\in \\mathbb{Z}_{\\ge 0}$ so working-capital (``max_budget``), warehouse capacity
       (``max_volume``), and discrete lot-sizing constraints hold to ``< 1e-6``.
    """
    eff_seed = int(random_seed) if random_seed is not None else int(seed)
    eff_target_fill = (
        target_portfolio_fill_rate
        if target_portfolio_fill_rate is not None
        else min_portfolio_fill_rate
    )

    if max_budget is not None and float(max_budget) < 0.0:
        raise _InventoryConfigError(f"max_budget={max_budget!r} must be >= 0.")
    if max_volume is not None and float(max_volume) < 0.0:
        raise _InventoryConfigError(f"max_volume={max_volume!r} must be >= 0.")
    if max_total_units is not None and float(max_total_units) < 0.0:
        raise _InventoryConfigError(f"max_total_units={max_total_units!r} must be >= 0.")
    if not (0.0 < float(min_service_level) < 1.0):
        raise _InventoryConfigError(
            f"min_service_level={min_service_level!r} must lie strictly in (0.0, 1.0)."
        )
    if eff_target_fill is not None and not (0.0 < float(eff_target_fill) < 1.0):
        raise _InventoryConfigError(
            f"target_portfolio_fill_rate={eff_target_fill!r} must lie strictly in (0.0, 1.0)."
        )
    if uncertainty_source not in ("copula", "analytical"):
        raise _InventoryConfigError(
            f"uncertainty_source={uncertainty_source!r} must be 'copula' or 'analytical'."
        )

    resolved_hist = _extract_history_df(predictions_df, history_df)
    pred_df, resolved_oof = _extract_predictions_and_oof(predictions_df, oof_df)
    if pred_df is None or pred_df.empty:
        empty_df = pd.DataFrame(columns=INVENTORY_OPTIMIZER_COLUMNS)
        empty_df.attrs["portfolio_summary"] = {
            "n_skus": 0,
            "uncertainty_source": str(uncertainty_source),
            "shadow_price_budget": 0.0,
            "shadow_price_volume": 0.0,
            "shadow_price_units": 0.0,
            "portfolio_fill_rate": 1.0,
            "binding_constraint": "none",
            "feasibility_status": "empty",
        }
        return empty_df

    # Merge mapping-typed arguments into per-SKU overrides so `compute_inventory_policy`
    # receives clean scalar defaults and per-SKU dictionary overrides.
    base_overrides = _normalize_inventory_overrides(inventory_df)
    all_tids: list[str] = (
        [str(x) for x in pred_df["ts_id"].drop_duplicates().tolist()]
        if "ts_id" in pred_df.columns
        else ["series_0"]
    )

    def _scalar_default(val: Any, fallback: Any) -> Any:
        if isinstance(val, Mapping):
            vals = [v for v in val.values() if v is not None]
            return vals[0] if vals else fallback
        return fallback if val is None else val

    scalar_lt = float(_scalar_default(lead_time, 7.0))
    scalar_lt_std = float(_scalar_default(lead_time_std, 0.0))
    scalar_rev = float(_scalar_default(review_period, 0.0))
    scalar_oh = float(_scalar_default(on_hand, 0.0))
    scalar_oo = float(_scalar_default(on_order, 0.0))
    scalar_bo = float(_scalar_default(backorders, 0.0))
    scalar_hc = float(_scalar_default(holding_cost, 1.0))
    scalar_sc = float(_scalar_default(stockout_cost, 9.0))

    raw_sl_default = _scalar_default(service_level, None)
    if isinstance(raw_sl_default, str) and raw_sl_default.lower() == "newsvendor":
        scalar_sl: float | None = None
    elif raw_sl_default is not None:
        scalar_sl = float(raw_sl_default)
    else:
        scalar_sl = None

    raw_fr_default = _scalar_default(fill_rate, None)
    scalar_fr: float | None = float(raw_fr_default) if raw_fr_default is not None else None

    merged_overrides: dict[str, dict[str, Any]] = {}
    for tid in all_tids:
        sku_ov = dict(base_overrides.get(tid, {}))
        for fname, pval in (
            ("lead_time", lead_time),
            ("lead_time_std", lead_time_std),
            ("review_period", review_period),
            ("on_hand", on_hand),
            ("on_order", on_order),
            ("backorders", backorders),
            ("holding_cost", holding_cost),
            ("stockout_cost", stockout_cost),
            ("service_level", service_level),
            ("fill_rate", fill_rate),
        ):
            if isinstance(pval, Mapping) and tid in pval and fname not in sku_ov:
                sku_ov[fname] = pval[tid]
        # Compute continuous baseline without discrete rounding first; we apply discrete
        # MOQ / order_multiple lot-sizing explicitly below.
        sku_ov["min_order_qty"] = 0.0
        sku_ov["order_multiple"] = 0.0
        merged_overrides[tid] = sku_ov

    base_policy = compute_inventory_policy(
        pred_df,
        oof_df=resolved_oof,
        inventory_df=merged_overrides,
        model_name=model_name,
        model_type=model_type,
        lead_time=scalar_lt,
        lead_time_std=scalar_lt_std,
        review_period=scalar_rev,
        service_level=scalar_sl,
        fill_rate=scalar_fr,
        on_hand=scalar_oh,
        on_order=scalar_oo,
        backorders=scalar_bo,
        holding_cost=scalar_hc,
        stockout_cost=scalar_sc,
        min_order_qty=0.0,
        order_multiple=0.0,
        confidence_level=confidence_level,
    )
    if base_policy.empty:
        empty_df = pd.DataFrame(columns=INVENTORY_OPTIMIZER_COLUMNS)
        empty_df.attrs["portfolio_summary"] = {
            "n_skus": 0,
            "uncertainty_source": str(uncertainty_source),
            "shadow_price_budget": 0.0,
            "shadow_price_volume": 0.0,
            "shadow_price_units": 0.0,
            "portfolio_fill_rate": 1.0,
            "binding_constraint": "none",
            "feasibility_status": "empty",
        }
        return empty_df

    # Resolve copula paths if uncertainty_source == "copula" or explicit trajectories were passed
    eff_source = (
        "copula"
        if (
            uncertainty_source == "copula"
            or (trajectories_df is not None and not trajectories_df.empty)
            or (copula_paths is not None and not copula_paths.empty)
        )
        else "analytical"
    )
    paths_df: pd.DataFrame | None = None
    copula_type_used = str(copula_type)
    if eff_source == "copula":
        paths_df = _get_or_sample_copula_paths(
            pred_df,
            resolved_oof,
            history_df=resolved_hist,
            trajectories_df=trajectories_df,
            copula_paths=copula_paths,
            copula_type=copula_type,
            n_paths=n_paths,
            seed=eff_seed,
            confidence_level=confidence_level,
            similarity_shrinkage=similarity_shrinkage,
            similarity_weight=similarity_weight,
            similarity_matrix=similarity_matrix,
            similarity_method=similarity_method,
        )
        copula_type_used = str(paths_df.attrs.get("copula_type_selected", copula_type))

    # Build per-SKU optimization state
    n_rows = len(base_policy)
    rows_list = base_policy.to_dict(orient="records")

    # Index copula paths by (ts_id, model_type) or ts_id
    paths_by_key: dict[tuple[str, str], np.ndarray] = {}
    paths_by_tid: dict[str, np.ndarray] = {}
    if paths_df is not None and not paths_df.empty and "yhat_sample" in paths_df.columns:
        p_work = paths_df.copy()
        if "ts_id" not in p_work.columns:
            p_work["ts_id"] = "series_0"
        if "model_type" not in p_work.columns:
            p_work["model_type"] = str(rows_list[0]["model_type"])
        sort_cols = ["ts_id", "model_type", "path_id"]
        if "horizon_step" in p_work.columns:
            sort_cols.append("horizon_step")
        elif "ds" in p_work.columns:
            sort_cols.append("ds")
        elif "forecast_date" in p_work.columns:
            sort_cols.append("forecast_date")
        p_work = p_work.sort_values(sort_cols)
        for (tid_p, mt_p), grp_p in p_work.groupby(["ts_id", "model_type"], sort=False):
            n_p = int(grp_p["path_id"].nunique())
            vals = (
                pd.to_numeric(grp_p["yhat_sample"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            if n_p > 0 and vals.size % n_p == 0:
                mat = vals.reshape(n_p, vals.size // n_p)
                paths_by_key[(str(tid_p), str(mt_p))] = mat
                paths_by_tid.setdefault(str(tid_p), mat)

    c_vec = np.zeros(n_rows, dtype=float)
    v_vec = np.zeros(n_rows, dtype=float)
    h_vec = np.zeros(n_rows, dtype=float)
    s_vec = np.zeros(n_rows, dtype=float)
    p_eff_vec = np.zeros(n_rows, dtype=float)
    moq_vec = np.zeros(n_rows, dtype=float)
    mult_vec = np.zeros(n_rows, dtype=float)

    prot_dem_vec = np.zeros(n_rows, dtype=float)
    lt_dem_vec = np.zeros(n_rows, dtype=float)
    inv_pos_vec = np.zeros(n_rows, dtype=float)
    rev_p_vec = np.zeros(n_rows, dtype=float)
    tot_std_vec = np.zeros(n_rows, dtype=float)
    fcst_std_vec = np.zeros(n_rows, dtype=float)

    unconstrained_sl = np.zeros(n_rows, dtype=float)
    unconstrained_z = np.zeros(n_rows, dtype=float)
    unconstrained_ss = np.zeros(n_rows, dtype=float)
    unconstrained_q_cont = np.zeros(n_rows, dtype=float)
    unconstrained_q_disc = np.zeros(n_rows, dtype=float)
    unconstrained_cost = np.zeros(n_rows, dtype=float)

    copula_prot_draws: list[np.ndarray | None] = [None] * n_rows

    for idx, r in enumerate(rows_list):
        tid = str(r["ts_id"])
        mtype = str(r["model_type"])
        sku_ov = base_overrides.get(tid, {})

        uc = float(_resolve_sku_param(unit_cost, tid, sku_ov, "unit_cost", 10.0))
        uv = float(_resolve_sku_param(unit_volume, tid, sku_ov, "unit_volume", 1.0))
        hc = float(
            _resolve_sku_param(
                holding_cost, tid, sku_ov, "holding_cost", float(r["expected_holding_cost"])
            )
        )
        if "holding_cost" not in sku_ov and not isinstance(holding_cost, Mapping):
            hc = float(holding_cost)
        sc = float(
            _resolve_sku_param(
                stockout_cost, tid, sku_ov, "stockout_cost", float(r["expected_stockout_cost"])
            )
        )
        if "stockout_cost" not in sku_ov and not isinstance(stockout_cost, Mapping):
            sc = float(stockout_cost)

        # Resolve MOQ from either `vendor_moq` or `min_order_qty`
        moq_val_a = float(
            _resolve_sku_param(
                vendor_moq, tid, sku_ov, "vendor_moq", 0.0, alt_field_names=("min_order_qty",)
            )
        )
        moq_val_b = float(
            _resolve_sku_param(
                min_order_qty, tid, sku_ov, "min_order_qty", 0.0, alt_field_names=("vendor_moq",)
            )
        )
        moq = max(moq_val_a, moq_val_b)
        mult = float(_resolve_sku_param(order_multiple, tid, sku_ov, "order_multiple", 0.0))

        if uc <= 0.0:
            raise _InventoryConfigError(f"unit_cost={uc!r} for ts_id={tid!r} must be > 0.")
        if uv < 0.0:
            raise _InventoryConfigError(f"unit_volume={uv!r} for ts_id={tid!r} must be >= 0.")
        if moq < 0.0 or mult < 0.0:
            raise _InventoryConfigError(
                f"vendor_moq={moq!r} and order_multiple={mult!r} for ts_id={tid!r} must be >= 0."
            )

        c_vec[idx] = uc
        v_vec[idx] = uv
        h_vec[idx] = max(hc, 1e-6)
        s_vec[idx] = max(sc, 1e-6)
        moq_vec[idx] = moq
        mult_vec[idx] = mult

        prot_steps = float(r["protection_steps"])
        prot_dem = float(r["protection_demand"])
        lt_dem = float(r["lead_time_demand"])
        lt_var_std = float(r["lead_time_var_std"])
        inv_pos = float(r["inventory_position"])
        rev_p = float(r["review_period"])
        target_mode = str(r["target_mode"])
        sl0 = float(r["target_service_level"])

        fcst_std = float(r["forecast_error_std"])
        tot_std = float(r["total_protection_std"])
        z0 = float(r["safety_factor_z"])

        mat_sku: np.ndarray | None = paths_by_key.get((tid, mtype))
        if mat_sku is None:
            mat_sku = paths_by_tid.get(tid)
        if eff_source == "copula" and mat_sku is not None and mat_sku.shape[0] >= 5:
            raw_draws = _window_sum_2d(mat_sku, prot_steps)
            cop_fcst_std = max(float(np.std(raw_draws, ddof=1)), 1e-6)
            cop_tot_std = math.sqrt(cop_fcst_std**2 + lt_var_std**2)
            # Center draws on protection_demand and scale by total protection std
            draw_mean = float(np.mean(raw_draws))
            centered_draws = np.maximum(
                0.0,
                prot_dem + (raw_draws - draw_mean) * (cop_tot_std / cop_fcst_std),
            )
            copula_prot_draws[idx] = centered_draws
            fcst_std = cop_fcst_std
            tot_std = cop_tot_std
            if target_mode == "fill_rate":
                sku_fr_val = float(
                    _resolve_sku_param(fill_rate, tid, sku_ov, "fill_rate", r["implied_fill_rate"])
                )
                z0 = _solve_z_for_fill_rate(sku_fr_val, prot_dem, tot_std)
                sl0 = float(stats.norm.cdf(z0))
            else:
                q_emp = float(np.quantile(centered_draws, np.clip(sl0, 0.01, 0.999)))
                ss_emp = max(0.0, q_emp - prot_dem)
                z_norm = max(0.0, float(stats.norm.ppf(np.clip(sl0, 0.01, 0.999))))
                # Blend empirical quantile safety factor with normal safety factor
                z0 = max(0.0, 0.5 * (ss_emp / max(tot_std, 1e-9)) + 0.5 * z_norm)

        prot_dem_vec[idx] = prot_dem
        lt_dem_vec[idx] = lt_dem
        inv_pos_vec[idx] = inv_pos
        rev_p_vec[idx] = rev_p
        fcst_std_vec[idx] = fcst_std
        tot_std_vec[idx] = max(tot_std, 1e-9)

        sl0_clipped = float(np.clip(sl0, max(float(min_service_level), 0.01), 0.9999))
        unconstrained_sl[idx] = sl0_clipped
        unconstrained_z[idx] = max(0.0, z0)
        ss0 = max(0.0, unconstrained_z[idx] * tot_std_vec[idx])
        unconstrained_ss[idx] = ss0

        # Effective KKT stockout penalty p_{i,eff} such that p / (p + h) == sl0_clipped
        p_eff_vec[idx] = h_vec[idx] * (sl0_clipped / max(1.0 - sl0_clipped, 1e-6))

        rop0 = lt_dem + ss0
        out0 = prot_dem + ss0
        trig0 = bool(inv_pos < out0) if rev_p > 0.0 else bool(inv_pos <= rop0)
        q0_cont = max(0.0, out0 - inv_pos) if trig0 else 0.0
        unconstrained_q_cont[idx] = q0_cont

        q0_disc = q0_cont
        if q0_disc > 0.0:
            if moq > 0.0:
                q0_disc = max(q0_disc, moq)
            if mult > 0.0:
                q0_disc = math.ceil(q0_disc / mult - 1e-12) * mult
        unconstrained_q_disc[idx] = q0_disc

        es0 = _expected_shortage_for_sku(
            ss0,
            prot_dem,
            tot_std_vec[idx],
            copula_prot_draws[idx],
        )
        unconstrained_cost[idx] = h_vec[idx] * (ss0 + es0) + s_vec[idx] * es0

    min_sl = float(np.clip(min_service_level, 0.01, 0.99))

    # Helper: map (lambda_B, lambda_V, lambda_U, fill_boost) -> continuous order quantities Q_i
    def _eval_continuous_policy(
        lam_b: float,
        lam_v: float,
        lam_u: float,
        fill_boost: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(q_cont, ss_vec, z_vec, sl_vec)` for given multipliers."""
        q_out = np.zeros(n_rows, dtype=float)
        ss_out = np.zeros(n_rows, dtype=float)
        z_out = np.zeros(n_rows, dtype=float)
        sl_out = np.zeros(n_rows, dtype=float)

        for i in range(n_rows):
            eta = max(0.0, lam_b * c_vec[i] + lam_v * v_vec[i] + lam_u)
            p_eff_i = p_eff_vec[i] * max(fill_boost, 1e-6)
            alpha_raw = p_eff_i / (p_eff_i + h_vec[i] + eta)
            sl0_i = unconstrained_sl[i]
            z0_i = unconstrained_z[i]
            z_norm0 = max(float(stats.norm.ppf(np.clip(sl0_i, 0.5001, 0.9999))), 1e-6)
            z_scale = z0_i / z_norm0 if sl0_i > 0.5001 else 1.0

            if eta <= 1e-12 and abs(fill_boost - 1.0) <= 1e-12:
                sl_i = sl0_i
                z_i = z0_i
                ss_i = unconstrained_ss[i]
                q_i = unconstrained_q_cont[i]
            elif alpha_raw >= min_sl:
                sl_i = float(np.clip(alpha_raw, min_sl, 0.9999))
                z_i = max(0.0, float(stats.norm.ppf(sl_i)) * z_scale)
                ss_i = max(0.0, z_i * tot_std_vec[i])
                rop_i = lt_dem_vec[i] + ss_i
                out_i = prot_dem_vec[i] + ss_i
                trig_i = (
                    bool(inv_pos_vec[i] < out_i)
                    if rev_p_vec[i] > 0.0
                    else bool(inv_pos_vec[i] <= rop_i)
                )
                q_i = max(0.0, out_i - inv_pos_vec[i]) if trig_i else 0.0
            else:
                # Below min_sl, safety stock reaches its floor at min_sl and order quantity scales
                # smoothly down to 0 as alpha_raw / min_sl -> 0 so arbitrary caps can bind.
                sl_i = min_sl
                z_min_i = max(0.0, float(stats.norm.ppf(min_sl)) * z_scale)
                ss_min_i = max(0.0, z_min_i * tot_std_vec[i])
                out_min_i = prot_dem_vec[i] + ss_min_i
                q_at_min_sl = max(0.0, out_min_i - inv_pos_vec[i])
                frac = float(np.clip(alpha_raw / max(min_sl, 1e-6), 0.0, 1.0))
                q_i = q_at_min_sl * frac
                out_eff = inv_pos_vec[i] + q_i
                ss_i = max(0.0, out_eff - prot_dem_vec[i])
                z_i = ss_i / tot_std_vec[i]

            q_out[i] = q_i
            ss_out[i] = ss_i
            z_out[i] = z_i
            sl_out[i] = sl_i
        return q_out, ss_out, z_out, sl_out

    def _portfolio_fill_from_ss(ss_arr: np.ndarray) -> float:
        tot_dem = float(np.sum(prot_dem_vec))
        if tot_dem <= 1e-9:
            return 1.0
        tot_short = 0.0
        for i in range(n_rows):
            tot_short += _expected_shortage_for_sku(
                float(ss_arr[i]),
                float(prot_dem_vec[i]),
                float(tot_std_vec[i]),
                copula_prot_draws[i],
            )
        return float(np.clip(1.0 - tot_short / tot_dem, 0.0, 1.0))

    # Optional pre-step: if target_portfolio_fill_rate is requested, calibrate `fill_boost`
    fill_boost = 1.0
    feasibility_status = "optimal"
    if eff_target_fill is not None:
        target_f = float(eff_target_fill)
        base_f = _portfolio_fill_from_ss(unconstrained_ss)
        if abs(base_f - target_f) > 1e-5:

            def _fill_gap(log_boost: float) -> float:
                _, ss_b, _, _ = _eval_continuous_policy(0.0, 0.0, 0.0, math.exp(log_boost))
                return _portfolio_fill_from_ss(ss_b) - target_f

            f_lo = _fill_gap(-6.0)
            f_hi = _fill_gap(8.0)
            if f_lo <= 0.0 <= f_hi:
                sol_f = root_scalar(_fill_gap, bracket=[-6.0, 8.0], method="brentq", xtol=1e-6)
                fill_boost = math.exp(float(sol_f.root))
            elif f_hi < 0.0:
                fill_boost = math.exp(8.0)
            else:
                fill_boost = math.exp(-6.0)

    q_base, _, _, _ = _eval_continuous_policy(0.0, 0.0, 0.0, fill_boost)

    # Stage 1: Continuous KKT / Lagrangian Dual Optimization for Shadow Prices
    spend_0 = float(np.dot(c_vec, q_base))
    vol_0 = float(np.dot(v_vec, q_base))
    units_0 = float(np.sum(q_base))

    budget_binds = max_budget is not None and spend_0 > float(max_budget) + 1e-7
    volume_binds = max_volume is not None and vol_0 > float(max_volume) + 1e-7
    units_binds = max_total_units is not None and units_0 > float(max_total_units) + 1e-7

    lambda_budget = 0.0
    lambda_volume = 0.0
    lambda_units = 0.0

    def _solve_1d_multiplier(
        eval_resource_fn: Any,
        cap_val: float,
    ) -> float:
        if cap_val <= 1e-12:
            return 1e6

        def _res_gap(lam: float) -> float:
            return float(eval_resource_fn(lam)) - cap_val

        hi = 1.0
        while _res_gap(hi) > 0.0 and hi < 1e9:
            hi *= 4.0
        if _res_gap(hi) > 0.0:
            return hi
        sol = root_scalar(_res_gap, bracket=[0.0, hi], method="brentq", xtol=1e-9, rtol=1e-9)
        return max(1e-9, float(sol.root))

    n_active_caps = int(budget_binds) + int(volume_binds) + int(units_binds)
    if n_active_caps == 1:
        if budget_binds and max_budget is not None:
            lambda_budget = _solve_1d_multiplier(
                lambda lb: np.dot(c_vec, _eval_continuous_policy(lb, 0.0, 0.0, fill_boost)[0]),
                float(max_budget),
            )
        elif volume_binds and max_volume is not None:
            lambda_volume = _solve_1d_multiplier(
                lambda lv: np.dot(v_vec, _eval_continuous_policy(0.0, lv, 0.0, fill_boost)[0]),
                float(max_volume),
            )
        elif units_binds and max_total_units is not None:
            lambda_units = _solve_1d_multiplier(
                lambda lu: np.sum(_eval_continuous_policy(0.0, 0.0, lu, fill_boost)[0]),
                float(max_total_units),
            )
    elif n_active_caps >= 2:
        # Gauss-Seidel KKT coordinate root-finding: each multiplier is > 0 iff its constraint
        # is violated when that multiplier is 0 given the other active multipliers.
        cand_lb = 0.0
        cand_lv = 0.0
        cand_lu = 0.0
        for _ in range(30):
            prev_lb, prev_lv, prev_lu = cand_lb, cand_lv, cand_lu
            if budget_binds and max_budget is not None:
                q_at_zero_b, _, _, _ = _eval_continuous_policy(0.0, cand_lv, cand_lu, fill_boost)
                if float(np.dot(c_vec, q_at_zero_b)) <= float(max_budget) + 1e-9:
                    cand_lb = 0.0
                else:
                    cand_lb = _solve_1d_multiplier(
                        lambda lb, _lv=cand_lv, _lu=cand_lu: np.dot(
                            c_vec, _eval_continuous_policy(lb, _lv, _lu, fill_boost)[0]
                        ),
                        float(max_budget),
                    )
            if volume_binds and max_volume is not None:
                q_at_zero_v, _, _, _ = _eval_continuous_policy(cand_lb, 0.0, cand_lu, fill_boost)
                if float(np.dot(v_vec, q_at_zero_v)) <= float(max_volume) + 1e-9:
                    cand_lv = 0.0
                else:
                    cand_lv = _solve_1d_multiplier(
                        lambda lv, _lb=cand_lb, _lu=cand_lu: np.dot(
                            v_vec, _eval_continuous_policy(_lb, lv, _lu, fill_boost)[0]
                        ),
                        float(max_volume),
                    )
            if units_binds and max_total_units is not None:
                q_at_zero_u, _, _, _ = _eval_continuous_policy(cand_lb, cand_lv, 0.0, fill_boost)
                if float(np.sum(q_at_zero_u)) <= float(max_total_units) + 1e-9:
                    cand_lu = 0.0
                else:
                    cand_lu = _solve_1d_multiplier(
                        lambda lu, _lb=cand_lb, _lv=cand_lv: np.sum(
                            _eval_continuous_policy(_lb, _lv, lu, fill_boost)[0]
                        ),
                        float(max_total_units),
                    )
            if (
                abs(cand_lb - prev_lb) <= 1e-9
                and abs(cand_lv - prev_lv) <= 1e-9
                and abs(cand_lu - prev_lu) <= 1e-9
            ):
                break

        lambda_budget = cand_lb
        lambda_volume = cand_lv
        lambda_units = cand_lu

    q_cont, ss_cont, z_cont, sl_cont = _eval_continuous_policy(
        lambda_budget, lambda_volume, lambda_units, fill_boost
    )

    # Final safety clamp on continuous Q in case max_budget == 0 or numerical tolerance > 1e-9
    if max_budget is not None:
        sp = float(np.dot(c_vec, q_cont))
        if sp > float(max_budget) and sp > 1e-12:
            q_cont = q_cont * (float(max_budget) / sp)
    if max_volume is not None:
        vl = float(np.dot(v_vec, q_cont))
        if vl > float(max_volume) and vl > 1e-12:
            q_cont = q_cont * (float(max_volume) / vl)
    if max_total_units is not None:
        un = float(np.sum(q_cont))
        if un > float(max_total_units) and un > 1e-12:
            q_cont = q_cont * (float(max_total_units) / un)

    # Stage 2: Discrete Lot-Sizing via C++ HiGHS MILP (`scipy.optimize.milp`) when `vendor_moq` or
    # `order_multiple` constraints are active.
    has_discrete = bool((moq_vec > 1e-9).any() or (mult_vec > 1e-9).any())
    if has_discrete:
        q_final = _solve_discrete_lot_sizing_milp(
            q_cont=q_cont,
            q_unconstrained=unconstrained_q_disc,
            c_vec=c_vec,
            v_vec=v_vec,
            h_vec=h_vec,
            s_vec=s_vec,
            p_eff_vec=p_eff_vec,
            moq_vec=moq_vec,
            mult_vec=mult_vec,
            prot_dem_vec=prot_dem_vec,
            inv_pos_vec=inv_pos_vec,
            tot_std_vec=tot_std_vec,
            max_budget=max_budget,
            max_volume=max_volume,
            max_total_units=max_total_units,
        )
        # Check if discrete MOQ rounding pushed budget/volume/units into binding territory when
        # continuous Q was slack
        if (
            max_budget is not None
            and float(np.dot(c_vec, unconstrained_q_disc)) > float(max_budget) + 1e-7
            and lambda_budget <= 0.0
        ):
            lambda_budget = float(np.mean((s_vec - h_vec) / np.maximum(c_vec, 1e-6))) * 0.1
        if (
            max_volume is not None
            and float(np.dot(v_vec, unconstrained_q_disc)) > float(max_volume) + 1e-7
            and lambda_volume <= 0.0
        ):
            lambda_volume = float(np.mean((s_vec - h_vec) / np.maximum(v_vec, 1e-6))) * 0.1
        if (
            max_total_units is not None
            and float(np.sum(unconstrained_q_disc)) > float(max_total_units) + 1e-7
            and lambda_units <= 0.0
        ):
            lambda_units = float(np.mean(s_vec - h_vec)) * 0.1
    else:
        q_final = q_cont.copy()

    # Recompute consistent SKU inventory policy metrics from `q_final`
    ss_final = np.zeros(n_rows, dtype=float)
    z_final = np.zeros(n_rows, dtype=float)
    sl_final = np.zeros(n_rows, dtype=float)
    rop_final = np.zeros(n_rows, dtype=float)
    out_final = np.zeros(n_rows, dtype=float)
    trig_final: list[bool] = []
    es_final = np.zeros(n_rows, dtype=float)
    fill_final = np.zeros(n_rows, dtype=float)
    hold_cost_final = np.zeros(n_rows, dtype=float)
    stock_cost_final = np.zeros(n_rows, dtype=float)
    tot_cost_final = np.zeros(n_rows, dtype=float)

    any_constraint_active = (
        lambda_budget > 0.0
        or lambda_volume > 0.0
        or lambda_units > 0.0
        or eff_target_fill is not None
        or has_discrete
    )

    for i in range(n_rows):
        if not any_constraint_active or abs(q_final[i] - unconstrained_q_cont[i]) <= 1e-9:
            ss_i = ss_cont[i] if any_constraint_active else unconstrained_ss[i]
            z_i = z_cont[i] if any_constraint_active else unconstrained_z[i]
            sl_i = sl_cont[i] if any_constraint_active else unconstrained_sl[i]
        else:
            out_eff_i = inv_pos_vec[i] + q_final[i]
            ss_i = max(0.0, out_eff_i - prot_dem_vec[i])
            z_i = ss_i / tot_std_vec[i]
            sl_i = float(np.clip(stats.norm.cdf(z_i), min_sl if q_final[i] > 0 else 0.50, 0.9999))

        rop_i = lt_dem_vec[i] + ss_i
        out_i = prot_dem_vec[i] + ss_i
        trig_i = (
            bool(q_final[i] > 1e-9)
            if any_constraint_active
            else (
                bool(inv_pos_vec[i] < out_i)
                if rev_p_vec[i] > 0.0
                else bool(inv_pos_vec[i] <= rop_i)
            )
        )
        es_i = _expected_shortage_for_sku(
            ss_i,
            prot_dem_vec[i],
            tot_std_vec[i],
            copula_prot_draws[i],
        )
        imp_fill_i = float(np.clip(1.0 - es_i / max(prot_dem_vec[i], 1e-9), 0.0, 1.0))
        hc_i = h_vec[i] * (ss_i + es_i)
        sc_i = s_vec[i] * es_i

        ss_final[i] = ss_i
        z_final[i] = z_i
        sl_final[i] = sl_i
        rop_final[i] = rop_i
        out_final[i] = out_i
        trig_final.append(trig_i)
        es_final[i] = es_i
        fill_final[i] = imp_fill_i
        hold_cost_final[i] = hc_i
        stock_cost_final[i] = sc_i
        tot_cost_final[i] = hc_i + sc_i

    tot_prot_dem = float(np.sum(prot_dem_vec))
    portfolio_fill_rate = (
        float(np.clip(1.0 - float(np.sum(es_final)) / tot_prot_dem, 0.0, 1.0))
        if tot_prot_dem > 1e-9
        else 1.0
    )
    unconstrained_fill_rate = _portfolio_fill_from_ss(unconstrained_ss)

    if eff_target_fill is not None and portfolio_fill_rate + 1e-3 < float(eff_target_fill):
        feasibility_status = "infeasible_target_fill_rate"
        if strict_feasibility:
            raise GoalSeekInfeasibleError(
                f"Target portfolio fill rate {eff_target_fill:.4f} is infeasible under the "
                f"active budget/volume caps (max achievable fill={portfolio_fill_rate:.4f})."
            )
    elif (
        has_discrete
        and (max_budget is not None or max_volume is not None or max_total_units is not None)
        and float(np.sum(unconstrained_q_disc)) > 1e-6
        and float(np.sum(q_final)) <= 1e-6
    ):
        feasibility_status = "infeasible_discrete_moq"
        if strict_feasibility:
            raise GoalSeekInfeasibleError(
                "No positive discrete order satisfies vendor_moq / order_multiple under the active "
                "budget, volume, or unit caps."
            )

    active_bindings: list[str] = []
    if lambda_budget > 0.0:
        active_bindings.append("budget")
    if lambda_volume > 0.0:
        active_bindings.append("volume")
    if lambda_units > 0.0:
        active_bindings.append("units")
    if eff_target_fill is not None and abs(fill_boost - 1.0) > 1e-5:
        active_bindings.append("fill_rate")
    binding_constraint = "+".join(active_bindings) if active_bindings else "none"

    out_records: list[dict[str, Any]] = []
    for i, r in enumerate(rows_list):
        q_i = float(q_final[i])
        q0_i = float(unconstrained_q_disc[i])
        alloc_ratio = float(q_i / q0_i) if q0_i > 1e-9 else 1.0
        spend_i = float(c_vec[i] * q_i)
        vol_i = float(v_vec[i] * q_i)

        rec: dict[str, Any] = {str(k): v for k, v in r.items()}
        rec.update(
            {
                "forecast_error_std": float(fcst_std_vec[i]),
                "total_protection_std": float(tot_std_vec[i]),
                "target_service_level": float(sl_final[i]),
                "implied_fill_rate": float(fill_final[i]),
                "safety_factor_z": float(z_final[i]),
                "safety_stock": float(ss_final[i]),
                "reorder_point": float(rop_final[i]),
                "order_up_to_level": float(out_final[i]),
                "reorder_triggered": bool(trig_final[i]),
                "recommended_order_qty": q_i,
                "expected_units_short": float(es_final[i]),
                "expected_holding_cost": float(hold_cost_final[i]),
                "expected_stockout_cost": float(stock_cost_final[i]),
                "expected_total_cost": float(tot_cost_final[i]),
                "unconstrained_service_level": float(unconstrained_sl[i]),
                "unconstrained_safety_stock": float(unconstrained_ss[i]),
                "unconstrained_order_qty": q0_i,
                "unconstrained_total_cost": float(unconstrained_cost[i]),
                "unit_cost": float(c_vec[i]),
                "unit_volume": float(v_vec[i]),
                "vendor_moq": float(moq_vec[i]),
                "order_cost": spend_i,
                "order_spend": spend_i,
                "order_volume": vol_i,
                "allocation_ratio": alloc_ratio,
                "shadow_price_budget": float(lambda_budget),
                "shadow_price_volume": float(lambda_volume),
                "shadow_price_units": float(lambda_units),
                "portfolio_fill_rate": float(portfolio_fill_rate),
                "binding_constraint": str(binding_constraint),
                "uncertainty_source": str(eff_source),
            }
        )
        out_records.append(rec)

    out_df = pd.DataFrame.from_records(out_records, columns=INVENTORY_OPTIMIZER_COLUMNS)
    total_spend = float(out_df["order_spend"].sum())
    unconstrained_spend = float(np.dot(c_vec, unconstrained_q_disc))
    total_volume = float(out_df["order_volume"].sum())
    unconstrained_volume = float(np.dot(v_vec, unconstrained_q_disc))
    total_units = float(out_df["recommended_order_qty"].sum())
    unconstrained_units = float(np.sum(unconstrained_q_disc))

    out_df.attrs["portfolio_summary"] = {
        "n_skus": int(n_rows),
        "uncertainty_source": str(eff_source),
        "copula_type": str(copula_type_used),
        "similarity_shrinkage_weight": (
            float(paths_df.attrs.get("similarity_shrinkage_weight", 0.0))
            if paths_df is not None
            else 0.0
        ),
        "max_budget": float(max_budget) if max_budget is not None else None,
        "total_order_spend": total_spend,
        "total_order_cost": total_spend,
        "unconstrained_order_spend": unconstrained_spend,
        "budget_utilization_pct": (
            float(100.0 * total_spend / float(max_budget))
            if max_budget is not None and float(max_budget) > 0.0
            else None
        ),
        "max_volume": float(max_volume) if max_volume is not None else None,
        "total_order_volume": total_volume,
        "unconstrained_order_volume": unconstrained_volume,
        "volume_utilization_pct": (
            float(100.0 * total_volume / float(max_volume))
            if max_volume is not None and float(max_volume) > 0.0
            else None
        ),
        "max_total_units": float(max_total_units) if max_total_units is not None else None,
        "total_order_units": total_units,
        "unconstrained_order_units": unconstrained_units,
        "shadow_price_budget": float(lambda_budget),
        "shadow_price_volume": float(lambda_volume),
        "shadow_price_units": float(lambda_units),
        "portfolio_fill_rate": float(portfolio_fill_rate),
        "unconstrained_portfolio_fill_rate": float(unconstrained_fill_rate),
        "total_expected_cost": float(out_df["expected_total_cost"].sum()),
        "unconstrained_total_cost": float(out_df["unconstrained_total_cost"].sum()),
        "binding_constraint": str(binding_constraint),
        "feasibility_status": str(feasibility_status),
    }
    return out_df


def _expected_shortage_for_sku(
    safety_stock: float,
    protection_demand: float,
    total_std: float,
    copula_draws: np.ndarray | None,
) -> float:
    """Compute expected shortage $E[\\max(0, D_P - (\\mu_P + SS))]$ analytically or via copula."""
    z = float(safety_stock) / max(float(total_std), 1e-9)
    es_norm = float(total_std) * _normal_loss(z)
    if copula_draws is None or copula_draws.size == 0:
        return es_norm
    order_up_to = float(protection_demand) + float(safety_stock)
    es_emp = float(np.mean(np.maximum(0.0, copula_draws - order_up_to)))
    return 0.5 * es_emp + 0.5 * es_norm


def _solve_discrete_lot_sizing_milp(
    *,
    q_cont: np.ndarray,
    q_unconstrained: np.ndarray,
    c_vec: np.ndarray,
    v_vec: np.ndarray,
    h_vec: np.ndarray,
    s_vec: np.ndarray,
    p_eff_vec: np.ndarray,
    moq_vec: np.ndarray,
    mult_vec: np.ndarray,
    prot_dem_vec: np.ndarray,
    inv_pos_vec: np.ndarray,
    tot_std_vec: np.ndarray,
    max_budget: float | None,
    max_volume: float | None,
    max_total_units: float | None,
) -> np.ndarray:
    """Solve Stage 2 discrete MOQ and case-pack lot-sizing via C++ `HiGHS` (`scipy.optimize.milp`).

    For each SKU $i$, represents the discrete order quantity $Q_i$ using a binary MOQ activation
    indicator $u_i \\in \\{0, 1\\}$ and a non-negative variable $n_i$ (integer case-pack count when
    ``mult_vec[i] > 0``, or continuous excess above MOQ when ``mult_vec[i] == 0``):

    $$Q_i = \\text{base}_i u_i + \\text{step}_i n_i, \\quad 0 \\le n_i \\le N_{\\max, i} u_i$$

    where $\\text{base}_i = \\lceil \\text{MOQ}_i / m_i \\rceil m_i$ when $m_i > 0$
    (or $\\text{MOQ}_i$ when $m_i = 0$). Whenever $u_i = 0$, $Q_i = 0$; whenever $u_i = 1$,
    $Q_i \\ge \\text{MOQ}_i$ and $Q_i$ is an exact multiple of $m_i$.
    """
    n_skus = len(q_cont)
    base_vec = np.zeros(n_skus, dtype=float)
    step_vec = np.zeros(n_skus, dtype=float)
    n_max_vec = np.zeros(n_skus, dtype=float)
    is_int_step = np.zeros(n_skus, dtype=bool)

    for i in range(n_skus):
        moq = float(moq_vec[i])
        mult = float(mult_vec[i])
        q_target = max(float(q_cont[i]), float(q_unconstrained[i]))
        if mult > 1e-9:
            step_i = mult
            base_i = math.ceil(max(moq, mult) / mult - 1e-12) * mult if moq > 1e-9 else mult
            q_hi = max(base_i, math.ceil(q_target / mult + 1.0) * mult)
            if max_budget is not None and c_vec[i] > 0:
                q_hi = min(q_hi, math.floor(float(max_budget) / c_vec[i] / mult + 1e-9) * mult)
            if max_volume is not None and v_vec[i] > 0:
                q_hi = min(q_hi, math.floor(float(max_volume) / v_vec[i] / mult + 1e-9) * mult)
            if max_total_units is not None:
                q_hi = min(q_hi, math.floor(float(max_total_units) / mult + 1e-9) * mult)
            n_max_i = max(0.0, math.floor((q_hi - base_i) / mult + 1e-9))
            base_vec[i] = base_i
            step_vec[i] = step_i
            n_max_vec[i] = float(n_max_i)
            is_int_step[i] = True
        elif moq > 1e-9:
            base_i = moq
            q_hi = max(base_i, q_target * 1.25 + base_i)
            if max_budget is not None and c_vec[i] > 0:
                q_hi = min(q_hi, float(max_budget) / c_vec[i])
            if max_volume is not None and v_vec[i] > 0:
                q_hi = min(q_hi, float(max_volume) / v_vec[i])
            if max_total_units is not None:
                q_hi = min(q_hi, float(max_total_units))
            n_max_i = max(0.0, q_hi - base_i)
            base_vec[i] = base_i
            step_vec[i] = 1.0
            n_max_vec[i] = float(n_max_i)
            is_int_step[i] = False
        else:
            base_vec[i] = 0.0
            step_vec[i] = 1.0
            q_hi = max(q_target * 1.25, 1.0)
            if max_budget is not None and c_vec[i] > 0:
                q_hi = min(q_hi, float(max_budget) / c_vec[i])
            if max_volume is not None and v_vec[i] > 0:
                q_hi = min(q_hi, float(max_volume) / v_vec[i])
            if max_total_units is not None:
                q_hi = min(q_hi, float(max_total_units))
            n_max_vec[i] = max(0.0, q_hi)
            is_int_step[i] = False

    # Decision variables x = [u_0..u_{N-1}, n_0..n_{N-1}, d_pos_0..d_pos_{N-1},
    # d_neg_0..d_neg_{N-1}] where Q_i = base_i * u_i + step_i * n_i, and
    # Q_i - q_cont_i = d_pos_i - d_neg_i.
    # Minimizes weighted deviation from Stage-1 KKT target `q_cont_i` while rewarding valid order
    # fulfillment on high marginal-stockout-value SKUs.
    n_vars = 4 * n_skus
    obj = np.zeros(n_vars, dtype=float)
    integrality = np.zeros(n_vars, dtype=int)
    lb = np.zeros(n_vars, dtype=float)
    ub = np.zeros(n_vars, dtype=float)

    for i in range(n_skus):
        # Priority weight = marginal stockout penalty per unit
        priority_w = float(p_eff_vec[i] + s_vec[i]) / max(float(tot_std_vec[i]), 1.0)
        priority_w = float(np.clip(priority_w, 0.1, 1e4))

        # u_i in {0, 1}
        integrality[i] = 1
        lb[i] = 0.0
        ub[i] = 1.0
        # If even `base_i` alone violates a global cap, force u_i = 0
        if (
            (max_budget is not None and c_vec[i] * base_vec[i] > float(max_budget) + 1e-9)
            or (max_volume is not None and v_vec[i] * base_vec[i] > float(max_volume) + 1e-9)
            or (max_total_units is not None and base_vec[i] > float(max_total_units) + 1e-9)
            or (q_cont[i] <= 1e-9 and q_unconstrained[i] <= 1e-9)
        ):
            ub[i] = 0.0

        # n_i in [0, n_max_i]
        integrality[n_skus + i] = 1 if is_int_step[i] else 0
        lb[n_skus + i] = 0.0
        ub[n_skus + i] = n_max_vec[i] if ub[i] > 0.0 else 0.0

        # d_pos_i >= 0 (over-ordering relative to q_cont_i)
        lb[2 * n_skus + i] = 0.0
        ub[2 * n_skus + i] = np.inf
        obj[2 * n_skus + i] = priority_w * 1.05 + h_vec[i] * 0.01

        # d_neg_i >= 0 (under-ordering relative to q_cont_i — penalized more heavily than
        # over-ordering when budget allows, proportional to marginal stockout priority)
        lb[3 * n_skus + i] = 0.0
        ub[3 * n_skus + i] = np.inf
        obj[3 * n_skus + i] = priority_w * 2.50 + s_vec[i] * 0.05

    rows_A: list[np.ndarray] = []
    rows_lb: list[float] = []
    rows_ub: list[float] = []

    # 1. Linking constraint: n_i - n_max_i * u_i <= 0
    for i in range(n_skus):
        row = np.zeros(n_vars, dtype=float)
        row[i] = -max(n_max_vec[i], 0.0)
        row[n_skus + i] = 1.0
        rows_A.append(row)
        rows_lb.append(-np.inf)
        rows_ub.append(0.0)

    # 2. Target deviation identity: base_i * u_i + step_i * n_i - d_pos_i + d_neg_i == q_cont_i
    for i in range(n_skus):
        row = np.zeros(n_vars, dtype=float)
        row[i] = base_vec[i]
        row[n_skus + i] = step_vec[i]
        row[2 * n_skus + i] = -1.0
        row[3 * n_skus + i] = 1.0
        rows_A.append(row)
        rows_lb.append(float(q_cont[i]))
        rows_ub.append(float(q_cont[i]))

    # 3. Working-capital budget constraint: sum_i c_i * (base_i * u_i + step_i * n_i) <= max_budget
    if max_budget is not None:
        row_b = np.zeros(n_vars, dtype=float)
        for i in range(n_skus):
            row_b[i] = c_vec[i] * base_vec[i]
            row_b[n_skus + i] = c_vec[i] * step_vec[i]
        rows_A.append(row_b)
        rows_lb.append(-np.inf)
        rows_ub.append(float(max_budget))

    # 4. Warehouse volume constraint: sum_i v_i * (base_i * u_i + step_i * n_i) <= max_volume
    if max_volume is not None:
        row_v = np.zeros(n_vars, dtype=float)
        for i in range(n_skus):
            row_v[i] = v_vec[i] * base_vec[i]
            row_v[n_skus + i] = v_vec[i] * step_vec[i]
        rows_A.append(row_v)
        rows_lb.append(-np.inf)
        rows_ub.append(float(max_volume))

    # 5. Total units constraint: sum_i (base_i * u_i + step_i * n_i) <= max_total_units
    if max_total_units is not None:
        row_u = np.zeros(n_vars, dtype=float)
        for i in range(n_skus):
            row_u[i] = base_vec[i]
            row_u[n_skus + i] = step_vec[i]
        rows_A.append(row_u)
        rows_lb.append(-np.inf)
        rows_ub.append(float(max_total_units))

    A_mat = np.vstack(rows_A)
    constraints = LinearConstraint(
        A_mat, np.asarray(rows_lb, dtype=float), np.asarray(rows_ub, dtype=float)
    )
    bounds = Bounds(lb, ub)

    res_milp = milp(c=obj, integrality=integrality, bounds=bounds, constraints=constraints)
    if res_milp.success and res_milp.x is not None:
        x_sol = np.asarray(res_milp.x, dtype=float)
        u_sol = np.round(x_sol[:n_skus])
        n_sol = x_sol[n_skus : 2 * n_skus]
        for i in range(n_skus):
            if is_int_step[i]:
                n_sol[i] = round(float(n_sol[i]))
        q_out = base_vec * u_sol + step_vec * n_sol
        q_out = np.where(u_sol > 0.5, q_out, 0.0)
        return np.maximum(0.0, q_out)

    # Greedy fallback if MILP solver ever reports non-convergence
    q_fallback = np.zeros(n_skus, dtype=float)
    order_idx = np.argsort(-p_eff_vec / np.maximum(c_vec, 1e-6))
    rem_b = float(max_budget) if max_budget is not None else np.inf
    rem_v = float(max_volume) if max_volume is not None else np.inf
    rem_u = float(max_total_units) if max_total_units is not None else np.inf
    for i in order_idx:
        if q_cont[i] <= 1e-9:
            continue
        q_cand = float(q_cont[i])
        if moq_vec[i] > 0.0:
            q_cand = max(q_cand, float(moq_vec[i]))
        if mult_vec[i] > 0.0:
            q_cand = math.floor(q_cand / float(mult_vec[i]) + 1e-9) * float(mult_vec[i])
            if moq_vec[i] > 0.0 and q_cand < float(moq_vec[i]):
                q_cand = math.ceil(float(moq_vec[i]) / float(mult_vec[i]) - 1e-12) * float(
                    mult_vec[i]
                )
        if (
            c_vec[i] * q_cand <= rem_b + 1e-9
            and v_vec[i] * q_cand <= rem_v + 1e-9
            and q_cand <= rem_u + 1e-9
        ):
            q_fallback[i] = q_cand
            rem_b -= c_vec[i] * q_cand
            rem_v -= v_vec[i] * q_cand
            rem_u -= q_cand
    return q_fallback


optimize_inventory_df = optimize_inventory_portfolio


def compute_inventory_frontier(
    predictions_df: pd.DataFrame,
    oof_df: pd.DataFrame | None = None,
    *,
    budgets: list[float] | np.ndarray | None = None,
    n_points: int = 8,
    **kwargs: Any,
) -> pd.DataFrame:
    """Evaluate portfolio replenishment policies across a working-capital budget grid.

    Returns a tidy Pareto efficiency frontier DataFrame with columns:
    ``["budget_cap", "total_order_spend", "total_order_volume", "total_order_units",
    "portfolio_fill_rate", "total_expected_cost", "shadow_price_budget", "binding_constraint"]``.
    """
    kw = dict(kwargs)
    kw.pop("max_budget", None)
    unconstrained = optimize_inventory_portfolio(
        predictions_df, oof_df=oof_df, max_budget=None, **kw
    )
    if unconstrained.empty:
        return pd.DataFrame(
            columns=[
                "budget_cap",
                "total_order_spend",
                "total_order_volume",
                "total_order_units",
                "portfolio_fill_rate",
                "total_expected_cost",
                "shadow_price_budget",
                "binding_constraint",
            ]
        )

    max_spend = max(float(unconstrained["order_spend"].sum()), 1.0)
    if budgets is None:
        pts = max(2, int(n_points))
        budget_grid = np.linspace(0.20 * max_spend, 1.05 * max_spend, pts)
    else:
        budget_grid = np.asarray(budgets, dtype=float)

    records: list[dict[str, Any]] = []
    for b_cap in budget_grid:
        pol = optimize_inventory_portfolio(
            predictions_df,
            oof_df=oof_df,
            max_budget=float(b_cap),
            **kw,
        )
        summary = pol.attrs.get("portfolio_summary", {})
        records.append(
            {
                "budget_cap": float(b_cap),
                "total_order_spend": float(
                    summary.get("total_order_spend", pol["order_spend"].sum())
                ),
                "total_order_volume": float(
                    summary.get("total_order_volume", pol["order_volume"].sum())
                ),
                "total_order_units": float(
                    summary.get("total_order_units", pol["recommended_order_qty"].sum())
                ),
                "portfolio_fill_rate": float(
                    summary.get("portfolio_fill_rate", pol["portfolio_fill_rate"].mean())
                ),
                "total_expected_cost": float(
                    summary.get("total_expected_cost", pol["expected_total_cost"].sum())
                ),
                "shadow_price_budget": float(
                    summary.get("shadow_price_budget", pol["shadow_price_budget"].iloc[0])
                ),
                "binding_constraint": str(
                    summary.get("binding_constraint", pol["binding_constraint"].iloc[0])
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_inventory_frontier(
    policy_or_predictions_df: pd.DataFrame,
    oof_df: pd.DataFrame | None = None,
    *,
    budgets: list[float] | np.ndarray | None = None,
    n_points: int = 8,
    max_rows: int = 15,
    ax: Any = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot the Portfolio Inventory Trade-off Frontier or Constrained vs. Unconstrained Allocation.

    Accepts either:
    - An already-computed portfolio policy DataFrame (from :func:`optimize_inventory_portfolio`),
    - A budget frontier DataFrame (from :func:`compute_inventory_frontier`), or
    - A forecast ``predictions_df`` DataFrame (evaluating :func:`compute_inventory_frontier`
      across ``n_points`` budget levels).
    """
    require_extra("notebook", purpose="Plotting portfolio inventory frontier")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    df, resolved_oof = _extract_predictions_and_oof(policy_or_predictions_df, oof_df)
    if df is None or df.empty:
        ax.set_title(title or "Portfolio Inventory Frontier (no rows)")
        return ax

    # Case 1: Frontier table with `budget_cap` and `portfolio_fill_rate`
    if "budget_cap" in df.columns and "portfolio_fill_rate" in df.columns:
        frontier_df = df.sort_values("budget_cap")
        ax.plot(
            frontier_df["total_order_spend"].to_numpy(dtype=float),
            frontier_df["portfolio_fill_rate"].to_numpy(dtype=float) * 100.0,
            marker="o",
            linewidth=2.2,
            color="#2563eb",
            label="Portfolio Fill Rate (%)",
        )
        ax.set_xlabel("Total Order Spend ($)")
        ax.set_ylabel("Portfolio Fill Rate (%)")
        ax.set_title(title or "Working-Capital Budget vs. Portfolio Fill Rate Frontier")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=True)
        return ax

    # Case 2: Policy table from `optimize_inventory_portfolio`
    if "recommended_order_qty" in df.columns and "unconstrained_order_qty" in df.columns:
        sub = df.head(max(1, int(max_rows))).copy()
        labels = [f"{r.ts_id} ({r.model_type})" for r in sub.itertuples(index=False)]
        y_pos = np.arange(len(labels))
        q_unc = sub["unconstrained_order_qty"].to_numpy(dtype=float)
        q_opt = sub["recommended_order_qty"].to_numpy(dtype=float)
        height = 0.38
        ax.barh(
            y_pos - height / 2,
            q_unc,
            height=height,
            color="#94a3b8",
            alpha=0.85,
            label="Unconstrained Order Qty",
        )
        ax.barh(
            y_pos + height / 2,
            q_opt,
            height=height,
            color="#2563eb",
            alpha=0.90,
            label="Constrained Optimal Order Qty",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Order Quantity (Units)")
        summary = df.attrs.get("portfolio_summary", {})
        lam_b = float(summary.get("shadow_price_budget", sub["shadow_price_budget"].iloc[0]))
        fill_r = float(summary.get("portfolio_fill_rate", sub["portfolio_fill_rate"].iloc[0]))
        default_title = (
            f"Portfolio Inventory Allocation (Fill Rate={fill_r * 100:.1f}%, λ_budget={lam_b:.3f})"
        )
        ax.set_title(title or default_title)
        ax.legend(loc="best", frameon=True)
        ax.grid(axis="x", alpha=0.25)
        return ax

    # Case 3: Raw predictions DataFrame -> compute and plot Pareto frontier
    frontier_df = compute_inventory_frontier(
        df,
        oof_df=resolved_oof,
        budgets=budgets,
        n_points=n_points,
        **kwargs,
    )
    return plot_inventory_frontier(frontier_df, ax=ax, title=title)
