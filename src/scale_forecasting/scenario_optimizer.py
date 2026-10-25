"""Prescriptive Price & Promotional Calendar Optimizer — pure DataFrame analytics.

Solves mixed continuous/discrete prescriptive optimization problems over causal demand response
surfaces (`@monotone` price guardrails, Robinson / Double-ML `promo_elasticity_df` promotional
lifts, post-promotion `pull_forward_dip` pantry-loading lags, and cross-SKU cannibalization):

1. **Structural Response Surface Extraction:**
   Estimates per-SKU own-price elasticity $\\epsilon_i < 0$ (strictly monotone downward-sloping
   demand per :mod:`~scale_forecasting.monotonicity`), direct promotional lift
   $\\beta_{\\text{promo}, i} \\ge 0$, post-promo pull-forward lag penalties
   $\\gamma_{i, 1 \\dots L} \\le 0$, and cross-SKU cannibalization $\\Gamma_{i,j} \\le 0$ via
   :func:`~scale_forecasting.elasticity.compute_promo_elasticity_decomposition`.
2. **Phase 1 — Mixed-Integer Linear Programming via C++ `HiGHS` (`scipy.optimize.milp`):**
   Optimizes discrete price ladders ($w_{i,h,k} \\in \\{0, 1\\}$) and promotional pulse calendars
   ($z_{i,h} \\in \\{0, 1\\}$) subject to promotional frequency caps (``max_promo_days``),
   minimum inter-promo cooldown spacing (``min_cooldown_days`` clique inequalities
   $\\sum_{\\tau=h}^{h+d} z_{i,\\tau} \\le 1$), portfolio gross-margin percentage floors
   (``min_margin_pct``), volume capacity bounds (``min_total_units``, ``max_total_units``),
   promotional spend caps (``max_promo_budget``), and Rockafellar–Uryasev $\\text{CVaR}_\\alpha$
   downside profit risk.
3. **Phase 2 — Continuous Price Refinement (`scipy.optimize.minimize(method="SLSQP")`) &
   Full-Chain Propagation:**
   When ``price_grid is None``, polishes continuous prices $P_{i,h} \\in [P_{\\min}, P_{\\max}]$
   with the optimal promotional calendar fixed, then propagates the winning schedule through
   spatial/temporal hierarchy reconciliation (``< 1e-6`` coherence error) and `(s, S)` inventory
   policy shifts (``delta_reorder_point``, ``delta_safety_stock``, ``delta_order_qty``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp, minimize

from .config import RunConfig
from .diagnostics import compute_inventory_policy
from .elasticity import compute_promo_elasticity_decomposition, detect_treatment_cols
from .errors import ConfigError, GoalSeekInfeasibleError, require_extra
from .reconciliation import TOTAL_NODE_ID, HierarchySpec, build_hierarchy

__all__ = [
    "SCENARIO_OPTIMIZATION_COLUMNS",
    "SCENARIO_OPTIMIZATION_SUMMARY_COLUMNS",
    "ScenarioOptimizationResult",
    "optimize_scenario",
    "plot_optimal_scenario",
]

SCENARIO_OPTIMIZATION_COLUMNS: tuple[str, ...] = (
    "ts_id",
    "model_type",
    "ds",
    "forecast_date",
    "horizon_step",
    "baseline_price",
    "optimal_price",
    "price",
    "price_change_pct",
    "baseline_promo",
    "optimal_promo",
    "promo",
    "baseline_yhat",
    "optimal_yhat",
    "yhat",
    "yhat_lower",
    "yhat_upper",
    "delta_yhat",
    "delta_pct",
    "price_lift",
    "incremental_promo_lift",
    "pull_forward_dip",
    "pull_forward_effect",
    "cannibalization_effect",
    "cannibalization_dip",
    "net_incremental_units",
    "unit_cost",
    "baseline_revenue",
    "optimal_revenue",
    "scenario_revenue",
    "revenue_delta",
    "revenue_delta_pct",
    "baseline_gross_margin",
    "optimal_gross_margin",
    "scenario_gross_margin",
    "gross_margin_delta",
    "gross_margin_delta_pct",
    "margin_pct",
    "baseline_reorder_point",
    "optimal_reorder_point",
    "delta_reorder_point",
    "baseline_safety_stock",
    "optimal_safety_stock",
    "delta_safety_stock",
    "baseline_order_qty",
    "optimal_order_qty",
    "delta_order_qty",
)

SCENARIO_OPTIMIZATION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "ts_id",
    "model_type",
    "horizon",
    "promo_days",
    "mean_baseline_price",
    "mean_optimal_price",
    "mean_price_change_pct",
    "baseline_total_units",
    "optimal_total_units",
    "delta_units",
    "delta_units_pct",
    "total_pull_forward_dip",
    "total_cannibalization_effect",
    "baseline_total_revenue",
    "optimal_total_revenue",
    "revenue_delta",
    "revenue_delta_pct",
    "baseline_total_gross_margin",
    "optimal_total_gross_margin",
    "gross_margin_delta",
    "gross_margin_delta_pct",
    "optimal_margin_pct",
    "delta_reorder_point",
    "delta_safety_stock",
    "delta_order_qty",
)


class _ScenarioOptimizerError(ConfigError, ValueError):
    """Scenario optimizer error inheriting from both `ConfigError` and `ValueError`."""


@dataclass(frozen=True)
class ScenarioOptimizationResult:
    """Container for prescriptive price and promotional calendar optimization results.

    Provides both structured attributes (``.schedule_df``, ``.summary_df``,
    ``.inventory_delta_df``, ``.optimal_objective_value``, ``.recommended_shocks``) and
    DataFrame-like delegation (``len(res)``, ``res["optimal_price"]``, ``res.columns``,
    ``res.head()``) for ergonomic interactive and programmatic use.
    """

    schedule_df: pd.DataFrame
    summary_df: pd.DataFrame
    inventory_delta_df: pd.DataFrame
    objective: str
    feasibility_status: str
    baseline_objective_value: float
    optimal_objective_value: float
    objective_improvement: float
    objective_improvement_pct: float
    baseline_total_units: float
    optimal_total_units: float
    delta_units_pct: float
    baseline_total_revenue: float
    optimal_total_revenue: float
    revenue_delta: float
    revenue_delta_pct: float
    baseline_total_gross_margin: float
    optimal_total_gross_margin: float
    gross_margin_delta: float
    gross_margin_delta_pct: float
    cvar_profit: float
    baseline_cvar_profit: float
    promo_budget_used: float
    coherence_max_abs_error: float
    recommended_shocks: dict[str, Any]

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the step-level optimal schedule DataFrame (``self.schedule_df``)."""
        return self.schedule_df

    @property
    def df(self) -> pd.DataFrame:
        """Alias for ``self.schedule_df``."""
        return self.schedule_df

    @property
    def baseline_profit(self) -> float:
        """Alias for ``self.baseline_total_gross_margin``."""
        return float(self.baseline_total_gross_margin)

    @property
    def optimal_profit(self) -> float:
        """Alias for ``self.optimal_total_gross_margin``."""
        return float(self.optimal_total_gross_margin)

    @property
    def columns(self) -> pd.Index:
        """Delegate ``.columns`` to ``self.schedule_df``."""
        return self.schedule_df.columns

    @property
    def empty(self) -> bool:
        """Delegate ``.empty`` to ``self.schedule_df``."""
        return bool(self.schedule_df.empty)

    @property
    def shape(self) -> tuple[int, int]:
        """Delegate ``.shape`` to ``self.schedule_df``."""
        return self.schedule_df.shape

    @property
    def attrs(self) -> dict[Any, Any]:
        """Delegate ``.attrs`` to ``self.schedule_df.attrs``."""
        return self.schedule_df.attrs

    def __len__(self) -> int:
        return len(self.schedule_df)

    def __getitem__(self, key: Any) -> Any:
        return self.schedule_df[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.schedule_df)

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` rows of ``self.schedule_df``."""
        return self.schedule_df.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Return the last ``n`` rows of ``self.schedule_df``."""
        return self.schedule_df.tail(n)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar summary metrics and recommended shocks to a plain dictionary."""
        return {
            "objective": self.objective,
            "feasibility_status": self.feasibility_status,
            "baseline_objective_value": float(self.baseline_objective_value),
            "optimal_objective_value": float(self.optimal_objective_value),
            "objective_improvement": float(self.objective_improvement),
            "objective_improvement_pct": float(self.objective_improvement_pct),
            "baseline_total_units": float(self.baseline_total_units),
            "optimal_total_units": float(self.optimal_total_units),
            "delta_units_pct": float(self.delta_units_pct),
            "baseline_total_revenue": float(self.baseline_total_revenue),
            "optimal_total_revenue": float(self.optimal_total_revenue),
            "revenue_delta": float(self.revenue_delta),
            "revenue_delta_pct": float(self.revenue_delta_pct),
            "baseline_total_gross_margin": float(self.baseline_total_gross_margin),
            "optimal_total_gross_margin": float(self.optimal_total_gross_margin),
            "gross_margin_delta": float(self.gross_margin_delta),
            "gross_margin_delta_pct": float(self.gross_margin_delta_pct),
            "cvar_profit": float(self.cvar_profit),
            "baseline_cvar_profit": float(self.baseline_cvar_profit),
            "promo_budget_used": float(self.promo_budget_used),
            "coherence_max_abs_error": float(self.coherence_max_abs_error),
            "recommended_shocks": dict(self.recommended_shocks),
        }

    def summary_text(self) -> str:
        """Render a concise plain-English executive summary of the optimal prescription."""
        return (
            f"ScenarioOptimizationResult(objective={self.objective!r}, "
            f"status={self.feasibility_status!r}): "
            f"objective {self.baseline_objective_value:,.2f} -> "
            f"{self.optimal_objective_value:,.2f} "
            f"({self.objective_improvement:+,.2f}, {self.objective_improvement_pct:+.2f}%), "
            f"units {self.baseline_total_units:,.1f} -> {self.optimal_total_units:,.1f} "
            f"({self.delta_units_pct:+.2f}%), "
            f"revenue ${self.baseline_total_revenue:,.2f} -> ${self.optimal_total_revenue:,.2f} "
            f"({self.revenue_delta_pct:+.2f}%), "
            f"gross_margin ${self.baseline_total_gross_margin:,.2f} -> "
            f"${self.optimal_total_gross_margin:,.2f} ({self.gross_margin_delta_pct:+.2f}%), "
            f"CVaR=${self.cvar_profit:,.2f}, coherence_err={self.coherence_max_abs_error:.2e}"
        )


def _resolve_inputs(
    forecaster_or_df: Any,
    *,
    config: RunConfig | Mapping[str, Any] | None,
    predictions_df: pd.DataFrame | None,
    oof_df: pd.DataFrame | None,
    history_df: pd.DataFrame | None,
    ts_id_col: str,
    date_col: str,
    target_col: str,
    horizon: int | None,
    freq: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, RunConfig | None]:
    """Resolve `(predictions_df, oof_df, history_df, parsed_cfg)` from flexible caller inputs."""
    parsed_cfg: RunConfig | None = None
    if isinstance(config, RunConfig):
        parsed_cfg = config
    elif isinstance(config, Mapping):
        try:
            parsed_cfg = RunConfig.model_validate(config)
        except Exception:  # noqa: BLE001 - partial dict fallback
            parsed_cfg = None

    if isinstance(forecaster_or_df, tuple):
        forecaster_or_df = forecaster_or_df[0]
    if isinstance(history_df, tuple):
        history_df = history_df[0]

    pred = predictions_df
    oof = oof_df
    hist = history_df

    if not isinstance(forecaster_or_df, pd.DataFrame) and forecaster_or_df is not None:
        if parsed_cfg is None and hasattr(forecaster_or_df, "config"):
            cfg_attr = getattr(forecaster_or_df, "config", None)
            if isinstance(cfg_attr, RunConfig):
                parsed_cfg = cfg_attr
        if hasattr(forecaster_or_df, "_resolve_track_k_frames"):
            p_res, o_res, h_res = forecaster_or_df._resolve_track_k_frames(
                forecasts_df=pred,
                oof_df=oof,
                data=hist,
            )
            pred = p_res if pred is None else pred
            oof = o_res if oof is None else oof
            hist = h_res if hist is None else hist
        elif hasattr(forecaster_or_df, "predictions"):
            p_attr = getattr(forecaster_or_df, "predictions", None)
            if pred is None and isinstance(p_attr, pd.DataFrame):
                pred = p_attr
            if oof is None and isinstance(getattr(forecaster_or_df, "oof", None), pd.DataFrame):
                oof = forecaster_or_df.oof
            if hist is None and isinstance(getattr(forecaster_or_df, "series", None), pd.DataFrame):
                hist = forecaster_or_df.series
    elif isinstance(forecaster_or_df, pd.DataFrame):
        df_in = forecaster_or_df.copy()
        has_yhat = "yhat" in df_in.columns or "forecast" in df_in.columns
        has_y = target_col in df_in.columns or "y" in df_in.columns
        if has_yhat and not has_y:
            if pred is None:
                pred = df_in
        elif has_y and not has_yhat:
            if hist is None:
                hist = df_in
        else:
            # Combined frame or predictions frame with actuals
            if "segment" in df_in.columns and (df_in["segment"] == "forecast").any():
                if pred is None:
                    pred = df_in[df_in["segment"] == "forecast"].copy()
                if hist is None:
                    hist = df_in[df_in["segment"] != "forecast"].copy()
            elif has_yhat:
                if pred is None:
                    pred = df_in
                if hist is None and has_y:
                    hist = df_in
            else:
                if hist is None:
                    hist = df_in

    # If only historical panel was provided, build a baseline forward forecast DataFrame
    if (pred is None or pred.empty) and hist is not None and not hist.empty:
        eff_h = (
            int(horizon)
            if horizon is not None
            else (int(parsed_cfg.data.horizon) if parsed_cfg is not None else 14)
        )
        eff_freq = (
            str(freq)
            if freq is not None
            else (str(parsed_cfg.data.freq) if parsed_cfg is not None else "D")
        )
        pred = _build_baseline_forecasts_from_history(
            hist,
            ts_id_col=ts_id_col,
            date_col=date_col,
            target_col=target_col,
            horizon=eff_h,
            freq=eff_freq,
        )

    if pred is None:
        pred = pd.DataFrame()
    return pred, oof, hist, parsed_cfg


def _build_baseline_forecasts_from_history(
    hist_df: pd.DataFrame,
    *,
    ts_id_col: str,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
) -> pd.DataFrame:
    """Construct baseline forward predictions from history when `predictions_df` is omitted."""
    work = hist_df.copy()
    if ts_id_col not in work.columns:
        work[ts_id_col] = "series_0"
    eff_date = date_col if date_col in work.columns else "ds"
    eff_target = target_col if target_col in work.columns else "y"
    work[eff_date] = pd.to_datetime(work[eff_date])
    work = work.sort_values([ts_id_col, eff_date])

    decomp = compute_promo_elasticity_decomposition(
        work,
        ts_id_col=ts_id_col,
        date_col=eff_date,
        target_col=eff_target,
    )
    base_by_tid: dict[str, np.ndarray] = {}
    if not decomp.empty and "baseline_demand" in decomp.columns:
        for tid_d, grp_d in decomp.groupby("ts_id", sort=False):
            base_by_tid[str(tid_d)] = grp_d["baseline_demand"].to_numpy(dtype=float)

    promo_col, price_col = detect_treatment_cols(
        work,
        exclude_cols={ts_id_col, eff_date, eff_target},
    )

    parts: list[pd.DataFrame] = []
    for tid, grp in work.groupby(ts_id_col, sort=False):
        tid_s = str(tid)
        y_vals = pd.to_numeric(grp[eff_target], errors="coerce").dropna().to_numpy(dtype=float)
        if y_vals.size == 0:
            continue
        base_hist = base_by_tid.get(tid_s, y_vals)
        period = min(7, len(base_hist))
        tail_pattern = base_hist[-period:] if period > 0 else np.array([float(np.mean(y_vals))])
        yh = np.resize(tail_pattern, int(horizon)).astype(float)
        yh = np.maximum(0.1, yh)
        sigma = max(
            float(np.std(y_vals - base_hist[-len(y_vals) :]))
            if len(base_hist) >= len(y_vals)
            else float(np.std(y_vals)) * 0.15,
            1e-3,
        )

        last_dt = pd.Timestamp(grp[eff_date].max())
        try:
            fut_dates = pd.date_range(start=last_dt, periods=int(horizon) + 1, freq=freq)[1:]
        except Exception:  # noqa: BLE001 - fallback to daily frequency
            fut_dates = pd.date_range(start=last_dt, periods=int(horizon) + 1, freq="D")[1:]

        part_dict: dict[str, Any] = {
            "ts_id": tid_s,
            "model_type": "causal_baseline",
            "ds": fut_dates,
            "forecast_date": fut_dates,
            "yhat": yh,
            "yhat_lower": np.maximum(0.0, yh - 1.2816 * sigma),
            "yhat_upper": yh + 1.2816 * sigma,
        }
        if price_col and price_col in grp.columns:
            pr_vals = pd.to_numeric(grp[price_col], errors="coerce").dropna().to_numpy(dtype=float)
            reg_p = float(np.nanpercentile(pr_vals, 75)) if pr_vals.size > 0 else 10.0
            part_dict[price_col] = reg_p
        if promo_col and promo_col in grp.columns:
            part_dict[promo_col] = 0.0
        parts.append(pd.DataFrame(part_dict))

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _estimate_sku_structural_params(
    *,
    bottom_ids: Sequence[str],
    horizon: int,
    base_yhat: np.ndarray,
    base_price: np.ndarray,
    base_promo: np.ndarray,
    history_df: pd.DataFrame | None,
    ts_id_col: str,
    date_col: str,
    target_col: str,
    promo_col: str | None,
    price_col: str | None,
    pull_forward_lags: int,
    cannibalization_weight: float,
    use_similarity_priors: bool = True,
    similarity_matrix: pd.DataFrame | np.ndarray | None = None,
    similarity_method: str = "hybrid",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate `(price_elasticity, promo_lift_rate, lag_dip_rates, cannib_matrix)` for leaf SKUs.

    Guarantees `@monotone` sign compliance:
    - Own-price elasticity $\\epsilon_i < 0$ for every SKU $i$,
    - Direct promotional lift rate $\\beta_{\\text{promo}, i} > 0$,
    - Post-promotion pull-forward dip lags $\\gamma_{i, 1 \\dots L} \\le 0$,
    - Cross-SKU cannibalization matrix $\\Gamma_{i, j} \\le 0$ ($j \\ne i$).
    """
    n_b = len(bottom_ids)
    eff_lags = max(1, int(pull_forward_lags))

    # Sensible causal defaults when history lacks independent price/promo variation
    eps_vec = np.full(n_b, -1.35, dtype=float)
    promo_lift_rate = np.full(n_b, 0.28, dtype=float)
    # Post-promo pull-forward dip decays geometrically across lags 1..L
    default_lag_weights = np.array([-0.08 * (0.6**k) for k in range(eff_lags)], dtype=float)
    lag_dip_rates = np.tile(default_lag_weights[None, :], (n_b, 1))
    decomp_sim_weights: pd.DataFrame | None = None

    if history_df is not None and not history_df.empty:
        work = history_df.copy()
        h_tid = (
            ts_id_col
            if ts_id_col in work.columns
            else ("ts_id" if "ts_id" in work.columns else None)
        )
        h_dt = date_col if date_col in work.columns else ("ds" if "ds" in work.columns else None)
        h_y = target_col if target_col in work.columns else ("y" if "y" in work.columns else None)
        if h_tid and h_dt and h_y:
            decomp = compute_promo_elasticity_decomposition(
                work,
                ts_id_col=h_tid,
                date_col=h_dt,
                target_col=h_y,
                promo_col=promo_col,
                price_col=price_col,
                pull_forward_lags=eff_lags,
                use_similarity_priors=use_similarity_priors,
                similarity_matrix=similarity_matrix,
                similarity_method=similarity_method,
            )
            sw_attr = decomp.attrs.get("similarity_weights")
            if isinstance(sw_attr, pd.DataFrame):
                decomp_sim_weights = sw_attr
            _, det_price = detect_treatment_cols(
                work,
                promo_col=promo_col,
                price_col=price_col,
                exclude_cols={h_tid, h_dt, h_y},
            )
            for idx_b, bid in enumerate(bottom_ids):
                sub_h = work[work[h_tid].astype(str) == str(bid)]
                sub_d = (
                    decomp[decomp["ts_id"].astype(str) == str(bid)]
                    if not decomp.empty
                    else pd.DataFrame()
                )
                if not sub_d.empty:
                    base_mean = max(float(sub_d["baseline_demand"].mean()), 1.0)
                    pos_lift = sub_d["incremental_promo_lift"].to_numpy(dtype=float)
                    active_mask = pos_lift > 1e-6
                    if active_mask.any():
                        est_lift = float(np.mean(pos_lift[active_mask])) / base_mean
                        promo_lift_rate[idx_b] = float(np.clip(est_lift, 0.08, 1.50))

                    pf_vals = sub_d["pull_forward_dip"].to_numpy(dtype=float)
                    pf_mask = pf_vals < -1e-6
                    if pf_mask.any():
                        n_promo_ev = max(int(np.sum(active_mask)), 1)
                        total_pf_rate = float(np.sum(pf_vals[pf_mask])) / (n_promo_ev * base_mean)
                        total_pf_rate = float(
                            np.clip(total_pf_rate, -1.50 * promo_lift_rate[idx_b], -0.02)
                        )
                        decay = np.array([0.6**k for k in range(eff_lags)], dtype=float)
                        decay = decay / np.sum(decay)
                        lag_dip_rates[idx_b, :] = total_pf_rate * decay

                # Estimate own-price elasticity if price column varies in history
                if det_price and det_price in sub_h.columns and len(sub_h) >= 8:
                    pr_s = pd.to_numeric(sub_h[det_price], errors="coerce").to_numpy(dtype=float)
                    y_s = pd.to_numeric(sub_h[h_y], errors="coerce").to_numpy(dtype=float)
                    valid = np.isfinite(pr_s) & np.isfinite(y_s) & (pr_s > 1e-6) & (y_s > 0.0)
                    if int(np.sum(valid)) >= 8 and float(np.std(pr_s[valid])) > 1e-6:
                        p_rel = (pr_s[valid] - float(np.mean(pr_s[valid]))) / float(
                            np.mean(pr_s[valid])
                        )
                        y_rel = (y_s[valid] - float(np.mean(y_s[valid]))) / max(
                            float(np.mean(y_s[valid])), 1e-6
                        )
                        denom = float(np.dot(p_rel, p_rel))
                        if denom > 1e-9:
                            raw_eps = float(np.dot(p_rel, y_rel) / denom)
                            # Enforce `@monotone` negative own-price elasticity in [-3.5, -0.45]
                            eps_vec[idx_b] = float(np.clip(min(raw_eps, -0.45), -3.50, -0.45))

    # Cross-SKU cannibalization matrix Gamma_{i, j} <= 0 for j != i
    cannib_matrix = np.zeros((n_b, n_b), dtype=float)
    if n_b >= 2 and float(cannibalization_weight) > 0.0:
        w_tilde = np.zeros((n_b, n_b), dtype=float)
        b_list = [str(b) for b in bottom_ids]
        if use_similarity_priors:
            if decomp_sim_weights is not None:
                w_arr = (
                    decomp_sim_weights.reindex(index=b_list, columns=b_list)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
            elif similarity_matrix is not None:
                if isinstance(similarity_matrix, pd.DataFrame):
                    s_arr = (
                        similarity_matrix.reindex(index=b_list, columns=b_list)
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
                else:
                    s_arr = np.asarray(similarity_matrix, dtype=float).copy()
                w_arr = np.maximum(0.0, s_arr) if s_arr.shape == (n_b, n_b) else np.ones((n_b, n_b))
            else:
                w_arr = np.ones((n_b, n_b), dtype=float)

            np.fill_diagonal(w_arr, 0.0)
            for i in range(n_b):
                r_sum = float(np.sum(w_arr[i]))
                if r_sum > 1e-12:
                    w_tilde[i] = w_arr[i] / r_sum
                else:
                    for j in range(n_b):
                        if i != j:
                            w_tilde[i, j] = 1.0 / float(n_b - 1)
        else:
            for i in range(n_b):
                for j in range(n_b):
                    if i != j:
                        w_tilde[i, j] = 1.0 / float(n_b - 1)

        for i in range(n_b):
            total_row_cannib = -0.06 * float(cannibalization_weight) * promo_lift_rate[i]
            for j in range(n_b):
                if i != j:
                    cannib_matrix[i, j] = total_row_cannib * float(w_tilde[i, j])

    _ = (horizon, base_yhat, base_price, base_promo)
    return eps_vec, promo_lift_rate, lag_dip_rates, cannib_matrix


def optimize_scenario(
    forecaster_or_df: Any,
    *,
    config: RunConfig | Mapping[str, Any] | None = None,
    cfg: RunConfig | Mapping[str, Any] | None = None,
    predictions_df: pd.DataFrame | None = None,
    forecasts_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
    data: pd.DataFrame | None = None,
    future_covariates_df: pd.DataFrame | None = None,
    trajectories_df: pd.DataFrame | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    objective: Literal[
        "gross_profit",
        "gross_margin",
        "revenue",
        "units",
        "risk_adjusted_profit",
        "target_demand",
    ] = "gross_profit",
    price_col: str | None = None,
    promo_col: str | None = None,
    price_bounds: tuple[float, float] | Mapping[str, tuple[float, float]] = (-0.15, 0.15),
    price_grid: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    promo_bounds: tuple[float, float] = (0.0, 1.0),
    binary_promo: bool = True,
    max_promo_days: int | Mapping[str, int] | None = None,
    min_cooldown_days: int = 0,
    cooldown_days: int | None = None,
    unit_cost: float | Mapping[str, float] | None = None,
    promo_fixed_cost: float | Mapping[str, float] = 0.0,
    min_margin_pct: float | None = None,
    min_gross_margin_pct: float | None = None,
    min_gross_margin: float | None = None,
    min_total_units: float | None = None,
    max_total_units: float | None = None,
    target_total_units: float | None = None,
    max_promo_budget: float | None = None,
    inventory_cap: float | Mapping[str, float] | None = None,
    risk_aversion_lambda: float = 0.0,
    risk_aversion: float | None = None,
    cvar_alpha: float = 0.95,
    risk_alpha: float | None = None,
    pull_forward_lags: int = 3,
    cannibalization_weight: float = 1.0,
    use_similarity_priors: bool = True,
    similarity_matrix: pd.DataFrame | np.ndarray | None = None,
    similarity_method: str = "hybrid",
    reconcile: bool = True,
    hier_spec: HierarchySpec | None = None,
    temp_spec: Any = None,
    include_inventory_impact: bool = True,
    lead_time: float = 7.0,
    lead_time_std: float = 0.0,
    service_level: float | None = 0.95,
    n_paths: int = 250,
    copula_type: str = "auto",
    seed: int = 42,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    horizon: int | None = None,
    freq: str | None = None,
    strict_feasibility: bool = False,
) -> ScenarioOptimizationResult:
    """Prescribe the optimal future price and promotional calendar across a forecast horizon.

    Solves a two-phase mixed-integer / continuous optimization problem over causal response
    surfaces (`@monotone` price elasticity, Double-ML `promo_elasticity_df` promotional lift,
    post-promotion `pull_forward_dip` lags, and cross-SKU cannibalization), enforcing operational
    guardrails (`price_bounds` / `price_grid`, `max_promo_days`, `min_cooldown_days`,
    `min_margin_pct`, `min_total_units`, `max_total_units`, `max_promo_budget`, and CVaR
    risk-adjusted profit), then propagates the optimal schedule through 3-mode hierarchy
    reconciliation and `(s, S)` inventory policy deltas.
    """
    eff_cfg = config if config is not None else cfg
    eff_preds_in = predictions_df if predictions_df is not None else forecasts_df
    eff_hist_in = history_df if history_df is not None else data
    eff_cooldown = int(cooldown_days) if cooldown_days is not None else int(min_cooldown_days)
    eff_min_margin_pct = (
        float(min_margin_pct)
        if min_margin_pct is not None
        else (float(min_gross_margin_pct) if min_gross_margin_pct is not None else None)
    )
    if eff_min_margin_pct is not None and eff_min_margin_pct > 1.0:
        eff_min_margin_pct = eff_min_margin_pct / 100.0

    eff_risk_lambda = (
        float(risk_aversion) if risk_aversion is not None else float(risk_aversion_lambda)
    )
    eff_cvar_alpha = float(risk_alpha) if risk_alpha is not None else float(cvar_alpha)

    obj_norm = str(objective).strip().lower()
    if obj_norm == "gross_margin":
        obj_norm = "gross_profit"
    if obj_norm not in (
        "gross_profit",
        "revenue",
        "units",
        "risk_adjusted_profit",
        "target_demand",
    ):
        raise _ScenarioOptimizerError(
            f"Unsupported objective={objective!r}; expected 'gross_profit', 'revenue', "
            "'units', or 'risk_adjusted_profit'."
        )
    if obj_norm == "risk_adjusted_profit" and eff_risk_lambda <= 0.0:
        eff_risk_lambda = 0.35

    if eff_cooldown < 0:
        raise _ScenarioOptimizerError(f"min_cooldown_days={eff_cooldown!r} must be >= 0.")
    if not (0.0 < eff_cvar_alpha < 1.0):
        raise _ScenarioOptimizerError(f"cvar_alpha={eff_cvar_alpha!r} must lie strictly in (0, 1).")
    if eff_min_margin_pct is not None and not (-1.0 < eff_min_margin_pct < 1.0):
        raise _ScenarioOptimizerError(
            f"min_margin_pct={min_margin_pct!r} must lie strictly in (-1.0, 1.0)."
        )
    if (
        min_total_units is not None
        and max_total_units is not None
        and float(min_total_units) > float(max_total_units) + 1e-9
    ):
        raise GoalSeekInfeasibleError(
            f"Infeasible unit bounds: min_total_units={min_total_units} > "
            f"max_total_units={max_total_units}."
        )

    pred_df, resolved_oof, resolved_hist, parsed_cfg = _resolve_inputs(
        forecaster_or_df,
        config=eff_cfg,
        predictions_df=eff_preds_in,
        oof_df=oof_df,
        history_df=eff_hist_in,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        freq=freq,
    )

    if pred_df is None or pred_df.empty:
        empty_sched = pd.DataFrame(columns=SCENARIO_OPTIMIZATION_COLUMNS)
        empty_sum = pd.DataFrame(columns=SCENARIO_OPTIMIZATION_SUMMARY_COLUMNS)
        empty_inv = pd.DataFrame(
            columns=[
                "ts_id",
                "model_type",
                "baseline_reorder_point",
                "optimal_reorder_point",
                "delta_reorder_point",
                "baseline_safety_stock",
                "optimal_safety_stock",
                "delta_safety_stock",
                "baseline_order_qty",
                "optimal_order_qty",
                "delta_order_qty",
            ]
        )
        return ScenarioOptimizationResult(
            schedule_df=empty_sched,
            summary_df=empty_sum,
            inventory_delta_df=empty_inv,
            objective=obj_norm,
            feasibility_status="empty",
            baseline_objective_value=0.0,
            optimal_objective_value=0.0,
            objective_improvement=0.0,
            objective_improvement_pct=0.0,
            baseline_total_units=0.0,
            optimal_total_units=0.0,
            delta_units_pct=0.0,
            baseline_total_revenue=0.0,
            optimal_total_revenue=0.0,
            revenue_delta=0.0,
            revenue_delta_pct=0.0,
            baseline_total_gross_margin=0.0,
            optimal_total_gross_margin=0.0,
            gross_margin_delta=0.0,
            gross_margin_delta_pct=0.0,
            cvar_profit=0.0,
            baseline_cvar_profit=0.0,
            promo_budget_used=0.0,
            coherence_max_abs_error=0.0,
            recommended_shocks={},
        )

    work = pred_df.copy()
    if "segment" in work.columns and (work["segment"] == "forecast").any():
        work = work[work["segment"] == "forecast"].copy()

    p_tid_col = (
        ts_id_col if ts_id_col in work.columns else ("ts_id" if "ts_id" in work.columns else None)
    )
    if p_tid_col is None:
        work["ts_id"] = "series_0"
        p_tid_col = "ts_id"
    elif p_tid_col != "ts_id":
        work["ts_id"] = work[p_tid_col].astype(str)

    p_dt_col = (
        "forecast_date"
        if "forecast_date" in work.columns
        else (
            date_col
            if date_col in work.columns
            else ("ds" if "ds" in work.columns else "target_date")
        )
    )
    work["_dt"] = pd.to_datetime(work[p_dt_col])
    if "model_type" not in work.columns:
        work["model_type"] = model_name or model_type or "model"

    eff_model = model_name or model_type
    if eff_model is not None and str(eff_model).lower() not in ("all", "*"):
        sub_m = work[work["model_type"].astype(str) == str(eff_model)]
        if not sub_m.empty:
            work = sub_m.copy()
    else:
        # Select single primary model_type if multiple models exist in predictions_df
        unique_models = [str(x) for x in work["model_type"].drop_duplicates().tolist()]
        if len(unique_models) > 1:
            ens_models = [m for m in unique_models if m.startswith("ensemble_")]
            chosen_m = ens_models[0] if ens_models else unique_models[0]
            work = work[work["model_type"].astype(str) == chosen_m].copy()

    chosen_model_type = str(work["model_type"].iloc[0])
    work = work.sort_values(["ts_id", "_dt"]).reset_index(drop=True)

    # Resolve hierarchy specification (`HierarchySpec`) if hierarchical reconciliation is active
    resolved_hier: HierarchySpec | None = hier_spec
    if (
        resolved_hier is None
        and parsed_cfg is not None
        and (
            parsed_cfg.hierarchy.levels
            or getattr(parsed_cfg.hierarchy, "similarity_hierarchy", False)
        )
        and resolved_hist is not None
        and not resolved_hist.empty
    ):
        needed_cols = {c for lvl in parsed_cfg.hierarchy.levels for c in lvl} - {
            "similarity_cluster_id"
        }
        if needed_cols.issubset(resolved_hist.columns):
            try:
                _, resolved_hier = build_hierarchy(resolved_hist, parsed_cfg)
            except Exception:  # noqa: BLE001 - fallback if hierarchy build fails
                resolved_hier = None

    all_present_ids = [str(x) for x in work["ts_id"].drop_duplicates().tolist()]
    if resolved_hier is not None:
        bottom_ids = [bid for bid in resolved_hier.bottom_ids if bid in all_present_ids]
        if not bottom_ids:
            bottom_ids = [tid for tid in all_present_ids if tid != TOTAL_NODE_ID]
    else:
        if TOTAL_NODE_ID in all_present_ids and len(all_present_ids) > 1:
            bottom_ids = [tid for tid in all_present_ids if tid != TOTAL_NODE_ID]
            n_b = len(bottom_ids)
            s_mat = np.vstack([np.ones((1, n_b), dtype=float), np.eye(n_b, dtype=float)])
            resolved_hier = HierarchySpec(
                node_ids=(TOTAL_NODE_ID, *bottom_ids),
                bottom_ids=tuple(bottom_ids),
                level_nodes={TOTAL_NODE_ID: (TOTAL_NODE_ID,), "__bottom__": tuple(bottom_ids)},
                summing_matrix=s_mat,
            )
        else:
            bottom_ids = list(all_present_ids)

    n_bottom = len(bottom_ids)
    # Build aligned date grid across bottom_ids
    first_grp = work[work["ts_id"].astype(str) == bottom_ids[0]].sort_values("_dt")
    dates_grid = first_grp["_dt"].to_numpy()
    h_steps = len(dates_grid)

    # Detect price_col and promo_col from future_covariates_df, work, or resolved_hist
    search_df = (
        future_covariates_df
        if (future_covariates_df is not None and not future_covariates_df.empty)
        else (resolved_hist if (resolved_hist is not None and not resolved_hist.empty) else work)
    )
    det_promo, det_price = detect_treatment_cols(
        search_df,
        promo_col=promo_col,
        price_col=price_col,
        exclude_cols={
            "ts_id",
            "ds",
            "forecast_date",
            "target_date",
            "y",
            "yhat",
            "yhat_lower",
            "yhat_upper",
        },
    )
    eff_promo_col = promo_col or det_promo or "promo"
    eff_price_col = price_col or det_price or "price"

    # Extract baseline yhat, lower/upper intervals, baseline price, baseline promo, and unit cost
    base_yhat = np.zeros((n_bottom, h_steps), dtype=float)
    base_lo = np.zeros((n_bottom, h_steps), dtype=float)
    base_hi = np.zeros((n_bottom, h_steps), dtype=float)
    base_price = np.zeros((n_bottom, h_steps), dtype=float)
    base_promo = np.zeros((n_bottom, h_steps), dtype=float)
    cost_vec = np.zeros(n_bottom, dtype=float)
    promo_cost_vec = np.zeros(n_bottom, dtype=float)
    inv_cap_vec = np.full(n_bottom, np.inf, dtype=float)

    yhat_col_name = "yhat" if "yhat" in work.columns else "forecast"

    for i, bid in enumerate(bottom_ids):
        grp_b = work[work["ts_id"].astype(str) == bid].sort_values("_dt")
        yh_b = (
            pd.to_numeric(grp_b[yhat_col_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        yh_b = np.resize(yh_b, h_steps) if len(yh_b) < h_steps else yh_b[:h_steps]
        yh_b = np.maximum(0.0, yh_b)
        base_yhat[i, :] = yh_b

        if "yhat_lower" in grp_b.columns:
            yl_raw = pd.to_numeric(grp_b["yhat_lower"], errors="coerce").to_numpy(dtype=float)[
                :h_steps
            ]
            yl_b = np.where(np.isfinite(yl_raw), yl_raw, 0.8 * yh_b)
        else:
            yl_b = np.maximum(0.0, 0.85 * yh_b)
        if "yhat_upper" in grp_b.columns:
            yu_raw = pd.to_numeric(grp_b["yhat_upper"], errors="coerce").to_numpy(dtype=float)[
                :h_steps
            ]
            yu_b = np.where(np.isfinite(yu_raw), yu_raw, 1.2 * yh_b)
        else:
            yu_b = 1.15 * yh_b
        base_lo[i, :] = yl_b
        base_hi[i, :] = yu_b

        # Resolve unit_cost first so default baseline_price is always profitable (> unit_cost)
        if isinstance(unit_cost, Mapping):
            uc_i = float(unit_cost.get(bid, 4.50))
        elif unit_cost is not None:
            uc_i = float(unit_cost)
        elif "unit_cost" in grp_b.columns:
            uc_i = float(pd.to_numeric(grp_b["unit_cost"], errors="coerce").fillna(4.50).iloc[0])
        else:
            uc_i = 4.50
        cost_vec[i] = max(0.0, uc_i)

        default_p0 = max(10.0, cost_vec[i] * 2.2)
        p_arr: np.ndarray | None = None
        for src_df in (future_covariates_df, grp_b, resolved_hist):
            if src_df is not None and not src_df.empty and eff_price_col in src_df.columns:
                sub_s = (
                    src_df[src_df["ts_id"].astype(str) == bid]
                    if "ts_id" in src_df.columns
                    else src_df
                )
                if not sub_s.empty:
                    vals_p = (
                        pd.to_numeric(sub_s[eff_price_col], errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    if vals_p.size > 0:
                        if src_df is resolved_hist:
                            p_arr = np.full(
                                h_steps, float(np.nanpercentile(vals_p, 75)), dtype=float
                            )
                        else:
                            p_arr = np.resize(vals_p, h_steps).astype(float)
                        break
        base_price[i, :] = np.maximum(
            1e-3, p_arr if p_arr is not None else np.full(h_steps, default_p0)
        )

        # If unit_cost was not explicitly provided and default cost >= base_price,
        # set cost = 0.45 * base_price
        if unit_cost is None and cost_vec[i] >= 0.85 * float(np.mean(base_price[i, :])):
            cost_vec[i] = 0.45 * float(np.mean(base_price[i, :]))

        d_arr: np.ndarray | None = None
        for src_df in (future_covariates_df, grp_b):
            if src_df is not None and not src_df.empty and eff_promo_col in src_df.columns:
                sub_s = (
                    src_df[src_df["ts_id"].astype(str) == bid]
                    if "ts_id" in src_df.columns
                    else src_df
                )
                if not sub_s.empty:
                    vals_d = (
                        pd.to_numeric(sub_s[eff_promo_col], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
                    if vals_d.size > 0:
                        d_arr = np.clip(np.resize(vals_d, h_steps), 0.0, 1.0)
                        break
        base_promo[i, :] = d_arr if d_arr is not None else np.zeros(h_steps, dtype=float)

        if isinstance(promo_fixed_cost, Mapping):
            pc_i = float(promo_fixed_cost.get(bid, 0.0))
        else:
            pc_i = float(promo_fixed_cost)
        if pc_i <= 0.0 and max_promo_budget is not None:
            pc_i = 1.0
        promo_cost_vec[i] = max(0.0, pc_i)

        if isinstance(inventory_cap, Mapping) and bid in inventory_cap:
            inv_cap_vec[i] = float(inventory_cap[bid])
        elif isinstance(inventory_cap, (int, float)):
            inv_cap_vec[i] = float(inventory_cap)

    # Estimate structural elasticity parameters (own-price elasticity, promo lift,
    # pull-forward lags, and cross-SKU cannibalization)
    eps_vec, promo_lift_rate, lag_dip_rates, cannib_matrix = _estimate_sku_structural_params(
        bottom_ids=bottom_ids,
        horizon=h_steps,
        base_yhat=base_yhat,
        base_price=base_price,
        base_promo=base_promo,
        history_df=resolved_hist,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        promo_col=eff_promo_col,
        price_col=eff_price_col,
        pull_forward_lags=pull_forward_lags,
        cannibalization_weight=cannibalization_weight,
        use_similarity_priors=use_similarity_priors,
        similarity_matrix=similarity_matrix,
        similarity_method=similarity_method,
    )
    n_lags = lag_dip_rates.shape[1]

    # Build candidate price grids per SKU `i` and step `h`
    price_candidates_per_sku: list[np.ndarray] = []
    price_bounds_per_sku: list[tuple[float, float]] = []
    for i, bid in enumerate(bottom_ids):
        p0_mean = float(np.mean(base_price[i, :]))
        if price_grid is not None:
            raw_grid = (
                price_grid[bid]
                if isinstance(price_grid, Mapping) and bid in price_grid
                else (
                    next(iter(price_grid.values()))
                    if isinstance(price_grid, Mapping)
                    else price_grid
                )
            )
            g_arr = np.asarray(list(raw_grid), dtype=float)
            if g_arr.size == 0:
                raise _ScenarioOptimizerError(f"Empty price_grid for ts_id={bid!r}.")
            # If grid values look like relative percentage shifts in [-0.8, 0.8]
            if float(np.max(np.abs(g_arr))) <= 0.80 and p0_mean > 1.5:
                rel_grid = np.sort(np.unique(np.append(g_arr, 0.0)))
            else:
                rel_grid = np.sort((g_arr - p0_mean) / max(p0_mean, 1e-6))
            price_candidates_per_sku.append(rel_grid)
            price_bounds_per_sku.append((float(np.min(rel_grid)), float(np.max(rel_grid))))
        else:
            b_tuple = (
                price_bounds[bid]
                if isinstance(price_bounds, Mapping) and bid in price_bounds
                else (
                    next(iter(price_bounds.values()))
                    if isinstance(price_bounds, Mapping)
                    else price_bounds
                )
            )
            lo_b, hi_b = float(b_tuple[0]), float(b_tuple[1])
            if lo_b > hi_b:
                raise _ScenarioOptimizerError(
                    f"Invalid price_bounds=({lo_b}, {hi_b}) for ts_id={bid!r}: lower > upper."
                )
            # Interpret bounds as relative shifts if within [-0.95, 2.0] and lo_b < 0 or hi_b < 1.0,
            # else convert literal price bounds to relative shifts
            if lo_b >= 0.5 and hi_b > 1.0 and p0_mean > 1.0 and lo_b > 0.2 * p0_mean:
                rel_lo = (lo_b - p0_mean) / p0_mean
                rel_hi = (hi_b - p0_mean) / p0_mean
            else:
                rel_lo, rel_hi = lo_b, hi_b
            rel_grid = np.sort(np.unique(np.append(np.linspace(rel_lo, rel_hi, 11), 0.0)))
            price_candidates_per_sku.append(rel_grid)
            price_bounds_per_sku.append((rel_lo, rel_hi))

    # Resolve per-SKU max_promo_days
    max_promo_per_sku = np.full(n_bottom, h_steps, dtype=int)
    for i, bid in enumerate(bottom_ids):
        if isinstance(max_promo_days, Mapping) and bid in max_promo_days:
            max_promo_per_sku[i] = max(0, int(max_promo_days[bid]))
        elif isinstance(max_promo_days, (int, float)):
            max_promo_per_sku[i] = max(0, int(max_promo_days))

    # Sample zero-mean demand residual paths for CVaR calculation
    rng = np.random.default_rng(int(seed))
    half_w = np.maximum(0.5 * (base_hi - base_lo), 0.10 * base_yhat)
    sigma_mat = np.maximum(half_w / 1.28155, 1e-4)
    n_cvar_paths = max(50, min(int(n_paths), 400))
    # AR(1) correlated residual paths across horizon steps
    idx_h = np.arange(h_steps)
    corr_h = 0.40 ** np.abs(idx_h[:, None] - idx_h[None, :])
    chol_h = np.linalg.cholesky(0.98 * corr_h + 0.02 * np.eye(h_steps))
    noise_paths = np.zeros((n_cvar_paths, n_bottom, h_steps), dtype=float)
    for i in range(n_bottom):
        z_draw = rng.standard_normal((n_cvar_paths, h_steps)) @ chol_h.T
        noise_paths[:, i, :] = z_draw * sigma_mat[i, None, :]

    # Helper: evaluate demand components for a given `(rel_price_mat, promo_mat)`
    def _evaluate_schedule(
        rel_p: np.ndarray,
        promo_m: np.ndarray,
    ) -> dict[str, Any]:
        price_m = base_price * (1.0 + rel_p)
        # Price demand lift: downward-sloping `@monotone` response
        price_lift_m = base_yhat * (eps_vec[:, None] * rel_p)
        # Direct promo lift
        promo_lift_m = base_yhat * (promo_lift_rate[:, None] * (promo_m - base_promo))
        # Post-promo pull-forward dip lags (lags 1..L)
        pf_dip_m = np.zeros((n_bottom, h_steps), dtype=float)
        for lag_idx in range(n_lags):
            lag_step = lag_idx + 1
            if h_steps > lag_step:
                pf_dip_m[:, lag_step:] += (
                    base_yhat[:, lag_step:]
                    * lag_dip_rates[:, lag_idx : lag_idx + 1]
                    * promo_m[:, :-lag_step]
                )
        # Cross-SKU cannibalization dip
        cannib_m = np.zeros((n_bottom, h_steps), dtype=float)
        if n_bottom >= 2:
            for i in range(n_bottom):
                for j in range(n_bottom):
                    if i != j and cannib_matrix[i, j] != 0.0:
                        cannib_m[i, :] += base_yhat[i, :] * cannib_matrix[i, j] * promo_m[j, :]

        raw_yhat = base_yhat + price_lift_m + promo_lift_m + pf_dip_m + cannib_m
        opt_yhat = np.maximum(0.0, raw_yhat)
        # Apply optional per-SKU horizon inventory cap
        for i in range(n_bottom):
            if np.isfinite(inv_cap_vec[i]):
                tot_i = float(np.sum(opt_yhat[i, :]))
                if tot_i > inv_cap_vec[i] and tot_i > 1e-9:
                    opt_yhat[i, :] *= inv_cap_vec[i] / tot_i

        rev_m = price_m * opt_yhat
        promo_spend = float(np.sum(promo_cost_vec[:, None] * promo_m))
        gm_m = (price_m - cost_vec[:, None]) * opt_yhat - (promo_cost_vec[:, None] * promo_m)

        tot_units = float(np.sum(opt_yhat))
        tot_rev = float(np.sum(rev_m))
        tot_gm = float(np.sum(gm_m))
        margin_pct_val = tot_gm / max(tot_rev, 1e-9)

        # Compute profit distribution across `noise_paths` for CVaR_(cvar_alpha)
        # Downside profit risk increases with unit margin (price - cost) and promo fixed spend
        unit_margin_m = price_m - cost_vec[:, None]
        path_Demand = np.maximum(0.0, opt_yhat[None, :, :] + noise_paths)
        path_profits = np.sum(unit_margin_m[None, :, :] * path_Demand, axis=(1, 2)) - promo_spend
        tail_frac = max(1.0 - eff_cvar_alpha, 0.01)
        n_tail = max(1, round(tail_frac * n_cvar_paths))
        sorted_profits = np.sort(path_profits)
        cvar_val = float(np.mean(sorted_profits[:n_tail]))
        expected_profit = float(np.mean(path_profits))

        if obj_norm == "gross_profit":
            obj_val = (
                tot_gm
                if eff_risk_lambda <= 0.0
                else (1.0 - eff_risk_lambda) * tot_gm + eff_risk_lambda * cvar_val
            )
        elif obj_norm == "revenue":
            obj_val = tot_rev
        elif obj_norm == "units":
            obj_val = tot_units
        elif obj_norm == "risk_adjusted_profit":
            obj_val = (1.0 - eff_risk_lambda) * expected_profit + eff_risk_lambda * cvar_val
        else:  # target_demand
            target_u = (
                float(target_total_units)
                if target_total_units is not None
                else float(np.sum(base_yhat)) * 1.05
            )
            obj_val = -abs(tot_units - target_u) + 1e-3 * tot_gm

        return {
            "rel_p": rel_p,
            "promo_m": promo_m,
            "price_m": price_m,
            "price_lift_m": price_lift_m,
            "promo_lift_m": promo_lift_m,
            "pf_dip_m": pf_dip_m,
            "cannib_m": cannib_m,
            "opt_yhat": opt_yhat,
            "rev_m": rev_m,
            "gm_m": gm_m,
            "tot_units": tot_units,
            "tot_rev": tot_rev,
            "tot_gm": tot_gm,
            "margin_pct": margin_pct_val,
            "cvar_profit": cvar_val,
            "promo_spend": promo_spend,
            "obj_val": float(obj_val),
        }

    baseline_eval = _evaluate_schedule(
        np.zeros((n_bottom, h_steps), dtype=float), base_promo.copy()
    )

    # Phase 1: Solve Mixed-Integer Linear Program (`scipy.optimize.milp` HiGHS) over discrete
    # price ladder w_{i,h,k} in {0, 1} and promo calendar z_{i,h} in {0, 1}.
    rel_p_milp, promo_milp, milp_status = _solve_phase1_milp(
        n_bottom=n_bottom,
        h_steps=h_steps,
        base_yhat=base_yhat,
        base_price=base_price,
        base_promo=base_promo,
        cost_vec=cost_vec,
        promo_cost_vec=promo_cost_vec,
        eps_vec=eps_vec,
        promo_lift_rate=promo_lift_rate,
        lag_dip_rates=lag_dip_rates,
        cannib_matrix=cannib_matrix,
        sigma_mat=sigma_mat,
        price_candidates_per_sku=price_candidates_per_sku,
        promo_bounds=promo_bounds,
        binary_promo=binary_promo,
        max_promo_per_sku=max_promo_per_sku,
        cooldown_days=eff_cooldown,
        min_margin_pct=eff_min_margin_pct,
        min_gross_margin=min_gross_margin,
        min_total_units=min_total_units,
        max_total_units=max_total_units,
        max_promo_budget=max_promo_budget,
        obj_norm=obj_norm,
        risk_lambda=eff_risk_lambda,
        cvar_alpha=eff_cvar_alpha,
        strict_feasibility=strict_feasibility,
    )

    best_eval = _evaluate_schedule(rel_p_milp, promo_milp)
    feasibility_status = milp_status

    # Phase 2: Continuous SLSQP price refinement when `price_grid is None`
    if price_grid is None:
        rel_p_slsqp = _polish_prices_slsqp(
            init_rel_p=rel_p_milp,
            fixed_promo=promo_milp,
            price_bounds_per_sku=price_bounds_per_sku,
            evaluate_fn=_evaluate_schedule,
            min_margin_pct=eff_min_margin_pct,
            min_gross_margin=min_gross_margin,
            min_total_units=min_total_units,
            max_total_units=max_total_units,
        )
        cand_eval = _evaluate_schedule(rel_p_slsqp, promo_milp)
        if (
            _is_schedule_feasible(
                cand_eval,
                min_margin_pct=eff_min_margin_pct,
                min_gross_margin=min_gross_margin,
                min_total_units=min_total_units,
                max_total_units=max_total_units,
            )
            and cand_eval["obj_val"] >= best_eval["obj_val"] - 1e-9
        ):
            best_eval = cand_eval

    # Build bottom-level and reconciled hierarchy rows
    opt_yhat_b = best_eval["opt_yhat"]
    opt_price_b = best_eval["price_m"]
    opt_promo_b = best_eval["promo_m"]
    rel_p_b = best_eval["rel_p"]
    price_lift_b = best_eval["price_lift_m"]
    promo_lift_b = best_eval["promo_lift_m"]
    pf_dip_b = best_eval["pf_dip_m"]
    cannib_b = best_eval["cannib_m"]
    opt_rev_b = best_eval["rev_m"]
    opt_gm_b = best_eval["gm_m"]
    base_rev_b = base_price * base_yhat
    base_gm_b = (base_price - cost_vec[:, None]) * base_yhat - (
        promo_cost_vec[:, None] * base_promo
    )

    # Assemble step-level records for bottom-level series and (when reconcile=True)
    # exact bottom-up reconciled aggregate hierarchy nodes (`S @ b_t`)
    if reconcile and resolved_hier is not None and resolved_hier.n_aggregated > 0:
        s_mat = np.asarray(resolved_hier.summing_matrix, dtype=float)
        # Ensure s_mat columns match bottom_ids
        if s_mat.shape[1] == n_bottom:
            all_node_ids = list(resolved_hier.node_ids)
            full_base_yhat = s_mat @ base_yhat
            full_opt_yhat = s_mat @ opt_yhat_b
            full_base_lo = s_mat @ base_lo
            full_base_hi = s_mat @ base_hi
            full_price_lift = s_mat @ price_lift_b
            full_promo_lift = s_mat @ promo_lift_b
            full_pf_dip = s_mat @ pf_dip_b
            full_cannib = s_mat @ cannib_b
            full_base_rev = s_mat @ base_rev_b
            full_opt_rev = s_mat @ opt_rev_b
            full_base_gm = s_mat @ base_gm_b
            full_opt_gm = s_mat @ opt_gm_b
            node_counts = np.maximum(s_mat.sum(axis=1), 1.0)
            full_base_promo = (s_mat @ base_promo) / node_counts[:, None]
            full_opt_promo = (s_mat @ opt_promo_b) / node_counts[:, None]
            full_cost = (s_mat @ cost_vec) / node_counts
            full_base_price = np.where(
                full_base_yhat > 1e-9,
                full_base_rev / np.maximum(full_base_yhat, 1e-9),
                (s_mat @ base_price) / node_counts[:, None],
            )
            full_opt_price = np.where(
                full_opt_yhat > 1e-9,
                full_opt_rev / np.maximum(full_opt_yhat, 1e-9),
                (s_mat @ opt_price_b) / node_counts[:, None],
            )
            full_rel_p = np.where(
                full_base_price > 1e-9,
                (full_opt_price - full_base_price) / np.maximum(full_base_price, 1e-9),
                0.0,
            )
        else:
            all_node_ids = list(bottom_ids)
            full_base_yhat = base_yhat
            full_opt_yhat = opt_yhat_b
            full_base_lo = base_lo
            full_base_hi = base_hi
            full_price_lift = price_lift_b
            full_promo_lift = promo_lift_b
            full_pf_dip = pf_dip_b
            full_cannib = cannib_b
            full_base_rev = base_rev_b
            full_opt_rev = opt_rev_b
            full_base_gm = base_gm_b
            full_opt_gm = opt_gm_b
            full_base_promo = base_promo
            full_opt_promo = opt_promo_b
            full_cost = cost_vec
            full_base_price = base_price
            full_opt_price = opt_price_b
            full_rel_p = rel_p_b
    else:
        all_node_ids = list(bottom_ids)
        full_base_yhat = base_yhat
        full_opt_yhat = opt_yhat_b
        full_base_lo = base_lo
        full_base_hi = base_hi
        full_price_lift = price_lift_b
        full_promo_lift = promo_lift_b
        full_pf_dip = pf_dip_b
        full_cannib = cannib_b
        full_base_rev = base_rev_b
        full_opt_rev = opt_rev_b
        full_base_gm = base_gm_b
        full_opt_gm = opt_gm_b
        full_base_promo = base_promo
        full_opt_promo = opt_promo_b
        full_cost = cost_vec
        full_base_price = base_price
        full_opt_price = opt_price_b
        full_rel_p = rel_p_b

    # Compute coherence error across hierarchy if applicable
    coherence_err = 0.0
    if (
        reconcile
        and resolved_hier is not None
        and resolved_hier.n_aggregated > 0
        and len(all_node_ids) == resolved_hier.n_nodes
    ):
        s_mat_check = np.asarray(resolved_hier.summing_matrix, dtype=float)
        reconstructed = s_mat_check @ full_opt_yhat[resolved_hier.n_aggregated :, :]
        coherence_err = float(np.max(np.abs(full_opt_yhat - reconstructed)))

    # Compute operational inventory policy deltas (`delta_reorder_point`, `delta_safety_stock`,
    # `delta_order_qty`)
    inv_delta_by_tid: dict[str, dict[str, float]] = {}
    inv_delta_records: list[dict[str, Any]] = []
    if include_inventory_impact:
        base_pred_rows: list[dict[str, Any]] = []
        opt_pred_rows: list[dict[str, Any]] = []
        for idx_n, nid in enumerate(all_node_ids):
            for h in range(h_steps):
                b_yh = float(full_base_yhat[idx_n, h])
                o_yh = float(full_opt_yhat[idx_n, h])
                half_b = max(
                    0.5 * float(full_base_hi[idx_n, h] - full_base_lo[idx_n, h]), 0.1 * b_yh
                )
                scale_ratio = o_yh / b_yh if b_yh > 1e-6 else 1.0
                half_o = half_b * max(0.5, scale_ratio)
                dt_val = dates_grid[h]
                base_pred_rows.append(
                    {
                        "ts_id": nid,
                        "model_type": chosen_model_type,
                        "ds": dt_val,
                        "forecast_date": dt_val,
                        "yhat": b_yh,
                        "yhat_lower": max(0.0, b_yh - half_b),
                        "yhat_upper": b_yh + half_b,
                    }
                )
                opt_pred_rows.append(
                    {
                        "ts_id": nid,
                        "model_type": chosen_model_type,
                        "ds": dt_val,
                        "forecast_date": dt_val,
                        "yhat": o_yh,
                        "yhat_lower": max(0.0, o_yh - half_o),
                        "yhat_upper": o_yh + half_o,
                    }
                )
        base_pol_df = compute_inventory_policy(
            pd.DataFrame.from_records(base_pred_rows),
            oof_df=resolved_oof,
            lead_time=float(lead_time),
            lead_time_std=float(lead_time_std),
            service_level=service_level,
        )
        opt_pol_df = compute_inventory_policy(
            pd.DataFrame.from_records(opt_pred_rows),
            oof_df=resolved_oof,
            lead_time=float(lead_time),
            lead_time_std=float(lead_time_std),
            service_level=service_level,
        )
        base_pol_map = {str(r["ts_id"]): r for r in base_pol_df.to_dict(orient="records")}
        opt_pol_map = {str(r["ts_id"]): r for r in opt_pol_df.to_dict(orient="records")}
        for nid in all_node_ids:
            br = base_pol_map.get(nid, {})
            orow = opt_pol_map.get(nid, {})
            b_rop = float(br.get("reorder_point", 0.0))
            o_rop = float(orow.get("reorder_point", 0.0))
            b_ss = float(br.get("safety_stock", 0.0))
            o_ss = float(orow.get("safety_stock", 0.0))
            b_q = float(br.get("recommended_order_qty", 0.0))
            o_q = float(orow.get("recommended_order_qty", 0.0))
            inv_delta_by_tid[nid] = {
                "baseline_reorder_point": b_rop,
                "optimal_reorder_point": o_rop,
                "delta_reorder_point": o_rop - b_rop,
                "baseline_safety_stock": b_ss,
                "optimal_safety_stock": o_ss,
                "delta_safety_stock": o_ss - b_ss,
                "baseline_order_qty": b_q,
                "optimal_order_qty": o_q,
                "delta_order_qty": o_q - b_q,
            }
            inv_delta_records.append(
                {
                    "ts_id": nid,
                    "model_type": chosen_model_type,
                    **inv_delta_by_tid[nid],
                }
            )
    else:
        for nid in all_node_ids:
            inv_delta_by_tid[nid] = {
                "baseline_reorder_point": 0.0,
                "optimal_reorder_point": 0.0,
                "delta_reorder_point": 0.0,
                "baseline_safety_stock": 0.0,
                "optimal_safety_stock": 0.0,
                "delta_safety_stock": 0.0,
                "baseline_order_qty": 0.0,
                "optimal_order_qty": 0.0,
                "delta_order_qty": 0.0,
            }
            inv_delta_records.append(
                {
                    "ts_id": nid,
                    "model_type": chosen_model_type,
                    **inv_delta_by_tid[nid],
                }
            )

    inventory_delta_df = pd.DataFrame.from_records(inv_delta_records)

    # Build `schedule_df` and `summary_df`
    sched_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    for idx_n, nid in enumerate(all_node_ids):
        inv_d = inv_delta_by_tid[nid]
        uc_n = float(full_cost[idx_n])
        for h in range(h_steps):
            b_p = float(full_base_price[idx_n, h])
            o_p = float(full_opt_price[idx_n, h])
            p_pct = float(full_rel_p[idx_n, h] * 100.0)
            b_pr = float(full_base_promo[idx_n, h])
            o_pr = float(full_opt_promo[idx_n, h])
            b_y = float(full_base_yhat[idx_n, h])
            o_y = float(full_opt_yhat[idx_n, h])
            dy = o_y - b_y
            dy_pct = (100.0 * dy / b_y) if b_y > 1e-9 else 0.0

            half_b = max(0.5 * float(full_base_hi[idx_n, h] - full_base_lo[idx_n, h]), 0.1 * b_y)
            scale_r = o_y / b_y if b_y > 1e-6 else 1.0
            half_o = half_b * max(0.5, scale_r)

            plift = float(full_price_lift[idx_n, h])
            prlift = float(full_promo_lift[idx_n, h])
            pfdip = float(full_pf_dip[idx_n, h])
            candip = float(full_cannib[idx_n, h])
            net_inc = plift + prlift + pfdip + candip

            b_rev = float(full_base_rev[idx_n, h])
            o_rev = float(full_opt_rev[idx_n, h])
            d_rev = o_rev - b_rev
            d_rev_pct = (100.0 * d_rev / abs(b_rev)) if abs(b_rev) > 1e-9 else 0.0

            b_gm = float(full_base_gm[idx_n, h])
            o_gm = float(full_opt_gm[idx_n, h])
            d_gm = o_gm - b_gm
            d_gm_pct = (100.0 * d_gm / abs(b_gm)) if abs(b_gm) > 1e-9 else 0.0
            m_pct = (o_gm / o_rev) if o_rev > 1e-9 else ((o_p - uc_n) / max(o_p, 1e-9))

            dt_ts = pd.Timestamp(dates_grid[h])
            sched_records.append(
                {
                    "ts_id": nid,
                    "model_type": chosen_model_type,
                    "ds": dt_ts,
                    "forecast_date": dt_ts,
                    "horizon_step": int(h + 1),
                    "baseline_price": b_p,
                    "optimal_price": o_p,
                    "price": o_p,
                    "price_change_pct": p_pct,
                    "baseline_promo": b_pr,
                    "optimal_promo": o_pr,
                    "promo": o_pr,
                    "baseline_yhat": b_y,
                    "optimal_yhat": o_y,
                    "yhat": o_y,
                    "yhat_lower": max(0.0, o_y - half_o),
                    "yhat_upper": o_y + half_o,
                    "delta_yhat": dy,
                    "delta_pct": dy_pct,
                    "price_lift": plift,
                    "incremental_promo_lift": prlift,
                    "pull_forward_dip": pfdip,
                    "pull_forward_effect": pfdip,
                    "cannibalization_effect": candip,
                    "cannibalization_dip": candip,
                    "net_incremental_units": net_inc,
                    "unit_cost": uc_n,
                    "baseline_revenue": b_rev,
                    "optimal_revenue": o_rev,
                    "scenario_revenue": o_rev,
                    "revenue_delta": d_rev,
                    "revenue_delta_pct": d_rev_pct,
                    "baseline_gross_margin": b_gm,
                    "optimal_gross_margin": o_gm,
                    "scenario_gross_margin": o_gm,
                    "gross_margin_delta": d_gm,
                    "gross_margin_delta_pct": d_gm_pct,
                    "margin_pct": m_pct,
                    "baseline_reorder_point": inv_d["baseline_reorder_point"],
                    "optimal_reorder_point": inv_d["optimal_reorder_point"],
                    "delta_reorder_point": inv_d["delta_reorder_point"],
                    "baseline_safety_stock": inv_d["baseline_safety_stock"],
                    "optimal_safety_stock": inv_d["optimal_safety_stock"],
                    "delta_safety_stock": inv_d["delta_safety_stock"],
                    "baseline_order_qty": inv_d["baseline_order_qty"],
                    "optimal_order_qty": inv_d["optimal_order_qty"],
                    "delta_order_qty": inv_d["delta_order_qty"],
                }
            )

        b_tot_u = float(np.sum(full_base_yhat[idx_n, :]))
        o_tot_u = float(np.sum(full_opt_yhat[idx_n, :]))
        du = o_tot_u - b_tot_u
        du_pct = (100.0 * du / b_tot_u) if b_tot_u > 1e-9 else 0.0
        b_tot_r = float(np.sum(full_base_rev[idx_n, :]))
        o_tot_r = float(np.sum(full_opt_rev[idx_n, :]))
        dr = o_tot_r - b_tot_r
        dr_pct = (100.0 * dr / abs(b_tot_r)) if abs(b_tot_r) > 1e-9 else 0.0
        b_tot_gm = float(np.sum(full_base_gm[idx_n, :]))
        o_tot_gm = float(np.sum(full_opt_gm[idx_n, :]))
        dgm = o_tot_gm - b_tot_gm
        dgm_pct = (100.0 * dgm / abs(b_tot_gm)) if abs(b_tot_gm) > 1e-9 else 0.0
        summary_records.append(
            {
                "ts_id": nid,
                "model_type": chosen_model_type,
                "horizon": int(h_steps),
                "promo_days": int(np.sum(full_opt_promo[idx_n, :] >= 0.5)),
                "mean_baseline_price": float(np.mean(full_base_price[idx_n, :])),
                "mean_optimal_price": float(np.mean(full_opt_price[idx_n, :])),
                "mean_price_change_pct": float(np.mean(full_rel_p[idx_n, :]) * 100.0),
                "baseline_total_units": b_tot_u,
                "optimal_total_units": o_tot_u,
                "delta_units": du,
                "delta_units_pct": du_pct,
                "total_pull_forward_dip": float(np.sum(full_pf_dip[idx_n, :])),
                "total_cannibalization_effect": float(np.sum(full_cannib[idx_n, :])),
                "baseline_total_revenue": b_tot_r,
                "optimal_total_revenue": o_tot_r,
                "revenue_delta": dr,
                "revenue_delta_pct": dr_pct,
                "baseline_total_gross_margin": b_tot_gm,
                "optimal_total_gross_margin": o_tot_gm,
                "gross_margin_delta": dgm,
                "gross_margin_delta_pct": dgm_pct,
                "optimal_margin_pct": (o_tot_gm / o_tot_r) if o_tot_r > 1e-9 else 0.0,
                "delta_reorder_point": inv_d["delta_reorder_point"],
                "delta_safety_stock": inv_d["delta_safety_stock"],
                "delta_order_qty": inv_d["delta_order_qty"],
            }
        )

    schedule_df = pd.DataFrame.from_records(sched_records, columns=SCENARIO_OPTIMIZATION_COLUMNS)
    summary_df = pd.DataFrame.from_records(
        summary_records, columns=SCENARIO_OPTIMIZATION_SUMMARY_COLUMNS
    )

    # Construct `recommended_shocks` compatible with `simulate_scenarios`
    mean_rel_p = float(np.mean(rel_p_b))
    promo_rules: list[dict[str, Any]] = []
    for h in range(h_steps):
        if float(np.mean(opt_promo_b[:, h])) >= 0.5:
            dt_str = str(pd.Timestamp(dates_grid[h]).date())
            promo_rules.append(
                {"op": "set", "value": 1.0, "start_date": dt_str, "end_date": dt_str}
            )
    recommended_shocks: dict[str, Any] = {
        "optimal_scenario": {
            eff_price_col: {"op": "mul", "value": round(1.0 + mean_rel_p, 6)},
            **(
                {eff_promo_col: promo_rules}
                if promo_rules
                else {eff_promo_col: {"op": "set", "value": 0.0}}
            ),
        }
    }

    base_obj = float(baseline_eval["obj_val"])
    opt_obj = float(best_eval["obj_val"])
    obj_imp = opt_obj - base_obj
    obj_imp_pct = (100.0 * obj_imp / abs(base_obj)) if abs(base_obj) > 1e-9 else 0.0

    b_u = float(baseline_eval["tot_units"])
    o_u = float(best_eval["tot_units"])
    du_pct_tot = (100.0 * (o_u - b_u) / b_u) if b_u > 1e-9 else 0.0

    b_r = float(baseline_eval["tot_rev"])
    o_r = float(best_eval["tot_rev"])
    dr_tot = o_r - b_r
    dr_pct_tot = (100.0 * dr_tot / abs(b_r)) if abs(b_r) > 1e-9 else 0.0

    b_gm = float(baseline_eval["tot_gm"])
    o_gm = float(best_eval["tot_gm"])
    dgm_tot = o_gm - b_gm
    dgm_pct_tot = (100.0 * dgm_tot / abs(b_gm)) if abs(b_gm) > 1e-9 else 0.0

    schedule_df.attrs["optimization_summary"] = {
        "objective": obj_norm,
        "feasibility_status": feasibility_status,
        "baseline_objective_value": base_obj,
        "optimal_objective_value": opt_obj,
        "objective_improvement": obj_imp,
        "coherence_max_abs_error": coherence_err,
    }
    if temp_spec is not None:
        schedule_df.attrs["temporal_spec"] = temp_spec

    return ScenarioOptimizationResult(
        schedule_df=schedule_df,
        summary_df=summary_df,
        inventory_delta_df=inventory_delta_df,
        objective=obj_norm,
        feasibility_status=feasibility_status,
        baseline_objective_value=base_obj,
        optimal_objective_value=opt_obj,
        objective_improvement=obj_imp,
        objective_improvement_pct=obj_imp_pct,
        baseline_total_units=b_u,
        optimal_total_units=o_u,
        delta_units_pct=du_pct_tot,
        baseline_total_revenue=b_r,
        optimal_total_revenue=o_r,
        revenue_delta=dr_tot,
        revenue_delta_pct=dr_pct_tot,
        baseline_total_gross_margin=b_gm,
        optimal_total_gross_margin=o_gm,
        gross_margin_delta=dgm_tot,
        gross_margin_delta_pct=dgm_pct_tot,
        cvar_profit=float(best_eval["cvar_profit"]),
        baseline_cvar_profit=float(baseline_eval["cvar_profit"]),
        promo_budget_used=float(best_eval["promo_spend"]),
        coherence_max_abs_error=coherence_err,
        recommended_shocks=recommended_shocks,
    )


def _solve_phase1_milp(
    *,
    n_bottom: int,
    h_steps: int,
    base_yhat: np.ndarray,
    base_price: np.ndarray,
    base_promo: np.ndarray,
    cost_vec: np.ndarray,
    promo_cost_vec: np.ndarray,
    eps_vec: np.ndarray,
    promo_lift_rate: np.ndarray,
    lag_dip_rates: np.ndarray,
    cannib_matrix: np.ndarray,
    sigma_mat: np.ndarray,
    price_candidates_per_sku: list[np.ndarray],
    promo_bounds: tuple[float, float],
    binary_promo: bool,
    max_promo_per_sku: np.ndarray,
    cooldown_days: int,
    min_margin_pct: float | None,
    min_gross_margin: float | None,
    min_total_units: float | None,
    max_total_units: float | None,
    max_promo_budget: float | None,
    obj_norm: str,
    risk_lambda: float,
    cvar_alpha: float,
    strict_feasibility: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Solve Phase 1 Mixed-Integer Linear Program via C++ `HiGHS` (`scipy.optimize.milp`)."""
    n_lags = lag_dip_rates.shape[1]
    # Variable indexing:
    # 1. One-hot price choice variables w_{i,h,k} for i in 0..n_bottom-1, h in 0..h_steps-1
    # 2. Promo variables z_{i,h} for i in 0..n_bottom-1, h in 0..h_steps-1
    w_offsets: dict[tuple[int, int], tuple[int, int]] = {}
    idx_cursor = 0
    for i in range(n_bottom):
        k_len = len(price_candidates_per_sku[i])
        for h in range(h_steps):
            w_offsets[(i, h)] = (idx_cursor, k_len)
            idx_cursor += k_len
    n_w_vars = idx_cursor
    n_z_vars = n_bottom * h_steps
    n_vars = n_w_vars + n_z_vars

    def _z_idx(i: int, h: int) -> int:
        return n_w_vars + i * h_steps + h

    # Downside tail risk factor z_alpha for CVaR approximation in linear objective
    tail_prob = max(1.0 - float(cvar_alpha), 0.01)
    from scipy import stats as _sp_stats

    phi_inv = float(_sp_stats.norm.ppf(tail_prob))
    cvar_sigma_mult = float(_sp_stats.norm.pdf(phi_inv) / tail_prob)

    # Precompute linear unit, revenue, and gross margin coefficients for every variable
    unit_coeff = np.zeros(n_vars, dtype=float)
    rev_coeff = np.zeros(n_vars, dtype=float)
    gm_coeff = np.zeros(n_vars, dtype=float)
    risk_gm_coeff = np.zeros(n_vars, dtype=float)

    # Constant offset from (-base_promo) term in promo lift
    const_units = float(np.sum(-base_yhat * promo_lift_rate[:, None] * base_promo))
    const_rev = float(np.sum(-base_price * base_yhat * promo_lift_rate[:, None] * base_promo))
    const_gm = float(
        np.sum(
            -(base_price - cost_vec[:, None]) * base_yhat * promo_lift_rate[:, None] * base_promo
        )
    )

    lb = np.zeros(n_vars, dtype=float)
    ub = np.ones(n_vars, dtype=float)
    integrality = np.ones(n_vars, dtype=int)

    for i in range(n_bottom):
        rel_grid = price_candidates_per_sku[i]
        c_i = cost_vec[i]
        eps_i = eps_vec[i]
        # Evaluate candidate price points across horizon steps
        for h in range(h_steps):
            start_idx, k_len = w_offsets[(i, h)]
            y0 = base_yhat[i, h]
            p0 = base_price[i, h]
            sig_ih = sigma_mat[i, h]
            for k in range(k_len):
                v_idx = start_idx + k
                delta_k = float(rel_grid[k])
                p_k = p0 * (1.0 + delta_k)
                q_k = max(0.05 * y0, y0 * (1.0 + eps_i * delta_k))
                r_k = p_k * q_k
                gm_k = (p_k - c_i) * q_k
                # CVaR downside profit on step (i, h): E[GM] - cvar_sigma_mult * |p_k - c_i| * sig
                # Plus quadratic risk penalty on extreme price hikes that amplify profit variance
                downside_pen = (
                    cvar_sigma_mult * max(0.0, p_k - c_i) * sig_ih * (1.0 + 1.5 * max(0.0, delta_k))
                )
                cvar_k = gm_k - downside_pen

                unit_coeff[v_idx] = q_k
                rev_coeff[v_idx] = r_k
                gm_coeff[v_idx] = gm_k
                risk_gm_coeff[v_idx] = (1.0 - risk_lambda) * gm_k + risk_lambda * cvar_k

    # Promo variable coefficients (including direct lift at step h, future pull-forward dips
    # at h+1..h+L, and concurrent cannibalization on peers j != i at step h)
    use_int_promo = bool(binary_promo or (max_promo_per_sku < h_steps).any() or cooldown_days > 0)
    p_lo, p_hi = float(promo_bounds[0]), float(promo_bounds[1])
    for i in range(n_bottom):
        for h in range(h_steps):
            z_v = _z_idx(i, h)
            lb[z_v] = p_lo if max_promo_per_sku[i] > 0 else 0.0
            ub[z_v] = p_hi if max_promo_per_sku[i] > 0 else 0.0
            integrality[z_v] = 1 if use_int_promo else 0

            # Direct lift on SKU i at step h
            dq_direct = base_yhat[i, h] * promo_lift_rate[i]
            p0_ih = base_price[i, h]
            c_i = cost_vec[i]
            dr_direct = p0_ih * dq_direct
            dgm_direct = (p0_ih - c_i) * dq_direct - promo_cost_vec[i]

            # Lagged pull-forward dips on SKU i at steps h + 1 .. h + L:
            # `dq_lags` / `dr_lags` / `dgm_lags` track in-horizon steps (`h_fut < h_steps`) for
            # exact in-horizon constraint accounting, while `obj_*_lags` also include steady-state
            # post-horizon carryover (`h_fut >= h_steps`) so end-of-horizon promotions are never
            # artificially exempted from their post-promo dip in the objective.
            dq_lags = 0.0
            dr_lags = 0.0
            dgm_lags = 0.0
            obj_dq_lags = 0.0
            obj_dr_lags = 0.0
            obj_dgm_lags = 0.0
            for lag_idx in range(n_lags):
                h_fut = h + lag_idx + 1
                h_ref = min(h_fut, h_steps - 1)
                dq_l = base_yhat[i, h_ref] * lag_dip_rates[i, lag_idx]
                p0_fut = base_price[i, h_ref]
                obj_dq_lags += dq_l
                obj_dr_lags += p0_fut * dq_l
                obj_dgm_lags += (p0_fut - c_i) * dq_l
                if h_fut < h_steps:
                    dq_lags += dq_l
                    dr_lags += p0_fut * dq_l
                    dgm_lags += (p0_fut - c_i) * dq_l

            # Concurrent cannibalization on peer SKUs j != i at step h
            dq_cannib = 0.0
            dr_cannib = 0.0
            dgm_cannib = 0.0
            if n_bottom >= 2:
                for j in range(n_bottom):
                    if j != i and cannib_matrix[j, i] != 0.0:
                        dq_c = base_yhat[j, h] * cannib_matrix[j, i]
                        p0_jh = base_price[j, h]
                        c_j = cost_vec[j]
                        dq_cannib += dq_c
                        dr_cannib += p0_jh * dq_c
                        dgm_cannib += (p0_jh - c_j) * dq_c

            tot_dq = dq_direct + dq_lags + dq_cannib
            tot_dr = dr_direct + dr_lags + dr_cannib
            tot_dgm = dgm_direct + dgm_lags + dgm_cannib
            obj_tot_dgm = dgm_direct + obj_dgm_lags + dgm_cannib
            promo_risk_pen = cvar_sigma_mult * 0.25 * max(0.0, p0_ih - c_i) * sigma_mat[i, h]

            unit_coeff[z_v] = tot_dq
            rev_coeff[z_v] = tot_dr
            gm_coeff[z_v] = tot_dgm
            risk_gm_coeff[z_v] = obj_tot_dgm - risk_lambda * promo_risk_pen
            if obj_tot_dgm < 0.0 and tot_dgm > obj_tot_dgm:
                # Net-negative lifecycle promotion: reflect full lifecycle dip in objective
                gm_coeff[z_v] = obj_tot_dgm

    # Objective vector for minimization (`milp` minimizes `c @ x`)
    if obj_norm == "gross_profit":
        c_obj = -(risk_gm_coeff if risk_lambda > 0.0 else gm_coeff)
    elif obj_norm == "revenue":
        c_obj = -rev_coeff - 1e-4 * gm_coeff
    elif obj_norm == "units":
        c_obj = -unit_coeff - 1e-4 * gm_coeff
    elif obj_norm == "risk_adjusted_profit":
        c_obj = -risk_gm_coeff
    else:
        c_obj = -gm_coeff

    # Assemble linear constraints:
    # Base constraints: one-hot price choice, max_promo_days, min_cooldown_days, max_promo_budget
    base_rows: list[np.ndarray] = []
    base_lb: list[float] = []
    base_ub: list[float] = []

    # 1. One-hot price choice per (i, h): sum_k w_{i,h,k} == 1
    for i in range(n_bottom):
        for h in range(h_steps):
            start_idx, k_len = w_offsets[(i, h)]
            row = np.zeros(n_vars, dtype=float)
            row[start_idx : start_idx + k_len] = 1.0
            base_rows.append(row)
            base_lb.append(1.0)
            base_ub.append(1.0)

    # 2. Max promo days per SKU i: sum_h z_{i,h} <= max_promo_per_sku[i]
    for i in range(n_bottom):
        if max_promo_per_sku[i] < h_steps:
            row = np.zeros(n_vars, dtype=float)
            for h in range(h_steps):
                row[_z_idx(i, h)] = 1.0
            base_rows.append(row)
            base_lb.append(-np.inf)
            base_ub.append(float(max_promo_per_sku[i]))

    # 3. Minimum cooldown days between promo pulses: sum_{tau=h}^{h+cooldown_days} z_{i,tau} <= 1
    if cooldown_days >= 1:
        for i in range(n_bottom):
            for h in range(h_steps):
                end_h = min(h_steps, h + int(cooldown_days) + 1)
                if end_h - h >= 2:
                    row = np.zeros(n_vars, dtype=float)
                    for tau in range(h, end_h):
                        row[_z_idx(i, tau)] = 1.0
                    base_rows.append(row)
                    base_lb.append(-np.inf)
                    base_ub.append(1.0)

    # 4. Max promo budget: sum_{i,h} promo_cost_i * z_{i,h} <= max_promo_budget
    if max_promo_budget is not None:
        row = np.zeros(n_vars, dtype=float)
        for i in range(n_bottom):
            for h in range(h_steps):
                row[_z_idx(i, h)] = promo_cost_vec[i]
        base_rows.append(row)
        base_lb.append(-np.inf)
        base_ub.append(float(max_promo_budget))

    # Optional portfolio guardrails (margin floor, unit bounds)
    extra_rows: list[np.ndarray] = []
    extra_lb: list[float] = []
    extra_ub: list[float] = []

    if min_margin_pct is not None:
        # Portfolio gross margin >= min_margin_pct * portfolio revenue
        # <=> (gm_coeff - m * rev_coeff) @ x >= m * const_rev - const_gm
        m_val = float(min_margin_pct)
        row_m = gm_coeff - m_val * rev_coeff
        rhs_m = m_val * const_rev - const_gm
        extra_rows.append(row_m)
        extra_lb.append(float(rhs_m))
        extra_ub.append(np.inf)

    if min_gross_margin is not None:
        extra_rows.append(gm_coeff.copy())
        extra_lb.append(float(min_gross_margin) - const_gm)
        extra_ub.append(np.inf)

    if min_total_units is not None or max_total_units is not None:
        u_lo = float(min_total_units) - const_units if min_total_units is not None else -np.inf
        u_hi = float(max_total_units) - const_units if max_total_units is not None else np.inf
        extra_rows.append(unit_coeff.copy())
        extra_lb.append(u_lo)
        extra_ub.append(u_hi)

    all_rows = base_rows + extra_rows
    all_lb = base_lb + extra_lb
    all_ub = base_ub + extra_ub

    res = milp(
        c=c_obj,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(
            np.vstack(all_rows),
            np.asarray(all_lb, dtype=float),
            np.asarray(all_ub, dtype=float),
        ),
    )

    status = "optimal"
    if not res.success or res.x is None:
        if strict_feasibility:
            raise GoalSeekInfeasibleError(
                "Scenario optimization constraints (min_margin_pct / unit bounds / price_bounds) "
                "are jointly infeasible."
            )
        # Solve with hard operational constraints (price grid, max_promo_days, cooldown_days,
        # promo_budget) and soft penalty on margin/unit guardrails
        status = "infeasible_relaxed"
        c_pen = c_obj.copy()
        if min_margin_pct is not None:
            c_pen = c_pen - 10.0 * (gm_coeff - float(min_margin_pct) * rev_coeff)
        if min_total_units is not None:
            c_pen = c_pen - 10.0 * unit_coeff
        if max_total_units is not None:
            c_pen = c_pen + 10.0 * unit_coeff
        res = milp(
            c=c_pen,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=LinearConstraint(
                np.vstack(base_rows),
                np.asarray(base_lb, dtype=float),
                np.asarray(base_ub, dtype=float),
            ),
        )

    rel_p_out = np.zeros((n_bottom, h_steps), dtype=float)
    promo_out = np.zeros((n_bottom, h_steps), dtype=float)
    if res.success and res.x is not None:
        x_sol = np.asarray(res.x, dtype=float)
        for i in range(n_bottom):
            rel_grid = price_candidates_per_sku[i]
            for h in range(h_steps):
                start_idx, k_len = w_offsets[(i, h)]
                w_slice = x_sol[start_idx : start_idx + k_len]
                best_k = int(np.argmax(w_slice))
                rel_p_out[i, h] = float(rel_grid[best_k])
                z_val = float(x_sol[_z_idx(i, h)])
                promo_out[i, h] = (
                    round(z_val) if use_int_promo else float(np.clip(z_val, p_lo, p_hi))
                )
    return rel_p_out, promo_out, status


def _is_schedule_feasible(
    eval_dict: Mapping[str, Any],
    *,
    min_margin_pct: float | None,
    min_gross_margin: float | None,
    min_total_units: float | None,
    max_total_units: float | None,
    tol: float = 1e-5,
) -> bool:
    """Return True if `eval_dict` satisfies portfolio margin and unit guardrails within `tol`."""
    if min_margin_pct is not None and float(eval_dict["margin_pct"]) < float(min_margin_pct) - tol:
        return False
    if min_gross_margin is not None and float(eval_dict["tot_gm"]) < float(min_gross_margin) - tol:
        return False
    if min_total_units is not None and float(eval_dict["tot_units"]) < float(min_total_units) - tol:
        return False
    return not (
        max_total_units is not None and float(eval_dict["tot_units"]) > float(max_total_units) + tol
    )


def _polish_prices_slsqp(
    *,
    init_rel_p: np.ndarray,
    fixed_promo: np.ndarray,
    price_bounds_per_sku: list[tuple[float, float]],
    evaluate_fn: Any,
    min_margin_pct: float | None,
    min_gross_margin: float | None,
    min_total_units: float | None,
    max_total_units: float | None,
) -> np.ndarray:
    """Refine continuous relative prices via `scipy.optimize.minimize(..., method="SLSQP")`."""
    n_bottom, h_steps = init_rel_p.shape
    x0 = init_rel_p.reshape(-1).copy()
    bounds_list: list[tuple[float, float]] = []
    for i in range(n_bottom):
        lo_i, hi_i = price_bounds_per_sku[i]
        for _ in range(h_steps):
            bounds_list.append((lo_i, hi_i))

    def _obj(x_vec: np.ndarray) -> float:
        rp = x_vec.reshape(n_bottom, h_steps)
        ev = evaluate_fn(rp, fixed_promo)
        return -float(ev["obj_val"])

    slsqp_constraints: list[dict[str, Any]] = []
    if min_margin_pct is not None:
        m_floor = float(min_margin_pct)
        slsqp_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x_vec: (
                    float(evaluate_fn(x_vec.reshape(n_bottom, h_steps), fixed_promo)["margin_pct"])
                    - m_floor
                ),
            }
        )
    if min_gross_margin is not None:
        gm_floor = float(min_gross_margin)
        slsqp_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x_vec: (
                    float(evaluate_fn(x_vec.reshape(n_bottom, h_steps), fixed_promo)["tot_gm"])
                    - gm_floor
                ),
            }
        )
    if min_total_units is not None:
        u_floor = float(min_total_units)
        slsqp_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x_vec: (
                    float(evaluate_fn(x_vec.reshape(n_bottom, h_steps), fixed_promo)["tot_units"])
                    - u_floor
                ),
            }
        )
    if max_total_units is not None:
        u_cap = float(max_total_units)
        slsqp_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x_vec: (
                    u_cap
                    - float(evaluate_fn(x_vec.reshape(n_bottom, h_steps), fixed_promo)["tot_units"])
                ),
            }
        )

    res = minimize(
        _obj,
        x0,
        method="SLSQP",
        bounds=bounds_list,
        constraints=slsqp_constraints,
        options={"ftol": 1e-8, "maxiter": 80},
    )
    if res.success and res.x is not None:
        return np.asarray(res.x, dtype=float).reshape(n_bottom, h_steps)
    return init_rel_p


def plot_optimal_scenario(
    result_or_df: ScenarioOptimizationResult | pd.DataFrame,
    *,
    ts_id: str | None = None,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot Baseline vs. Optimal Scenario Demand, Price Schedule, and Promotional Pulses."""
    require_extra("notebook", purpose="Plotting optimal scenario schedule")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    sched_df = (
        result_or_df.schedule_df
        if isinstance(result_or_df, ScenarioOptimizationResult)
        else result_or_df
    )
    if sched_df is None or sched_df.empty:
        ax.set_title(title or "Optimal Scenario Schedule (no rows)")
        return ax

    sub = sched_df.copy()
    if ts_id is not None and "ts_id" in sub.columns:
        sub = sub[sub["ts_id"].astype(str) == str(ts_id)].copy()
    elif "ts_id" in sub.columns:
        all_ids = [str(x) for x in sub["ts_id"].drop_duplicates().tolist()]
        chosen_id = TOTAL_NODE_ID if TOTAL_NODE_ID in all_ids else all_ids[0]
        sub = sub[sub["ts_id"].astype(str) == chosen_id].copy()

    if sub.empty:
        ax.set_title(title or "Optimal Scenario Schedule (empty selection)")
        return ax

    dt_col = "ds" if "ds" in sub.columns else "forecast_date"
    sub = sub.sort_values(dt_col)
    x_dates = pd.to_datetime(sub[dt_col])
    base_y = sub["baseline_yhat"].to_numpy(dtype=float)
    opt_y = sub["optimal_yhat"].to_numpy(dtype=float)

    ax.plot(
        x_dates, base_y, linestyle="--", linewidth=1.8, color="#64748b", label="Baseline Demand"
    )
    ax.plot(
        x_dates, opt_y, marker="o", linewidth=2.2, color="#2563eb", label="Optimal Scenario Demand"
    )

    if "optimal_promo" in sub.columns:
        promo_mask = sub["optimal_promo"].to_numpy(dtype=float) >= 0.5
        if promo_mask.any():
            ax.scatter(
                x_dates[promo_mask],
                opt_y[promo_mask],
                color="#dc2626",
                s=65,
                zorder=5,
                label="Active Promo Pulse",
            )

    if isinstance(result_or_df, ScenarioOptimizationResult):
        default_title = (
            f"Prescriptive Scenario Optimization ({result_or_df.objective}: "
            f"{result_or_df.objective_improvement_pct:+.1f}% Lift)"
        )
    else:
        default_title = "Prescriptive Scenario Schedule: Baseline vs. Optimal Demand"

    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast Demand (Units)")
    ax.set_title(title or default_title)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.25)
    return ax
