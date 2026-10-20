"""Counterfactual "What-If" scenario, gross-margin & inventory-policy simulator — pure, offline.

Allows planners and AI agents to perturb future exogenous covariates (promotions, pricing, weather,
macro indices, capacity caps) across the entire forecast horizon or within specific date windows
and propagate those shocks end-to-end across five stages:

1. **Stage 1 — Base Model Shocks:** Evaluates every candidate model (honoring `@<tag>` variants and
   `monotonic_constraints` sign guardrails) on unshocked `"baseline"` and shocked scenario grids.
2. **Stage 2 — Winning Ensemble Propagation:** When `cfg.ensemble.enabled` is active, blends base
   scenario trajectories through the configured calculated and learned ensemble strategies
   (`ensemble_<strategy>`).
3. **Stage 3 — Hierarchy Reconciliation:** When `reconcile=True` and `cfg.hierarchy.enabled` is
   active, reconciles every scenario trajectory across all hierarchy levels (`__total__`,
   intermediate nodes, and bottom leaves) via `reconcile_forecasts` (`mint_nonneg` / configured
   method) so additive coherence (`verify_coherence < 1e-6`) holds under every shock.
4. **Stage 4 — Gross-Margin Delta:** Computes `baseline_revenue`, `scenario_revenue`,
   `revenue_delta`, `revenue_delta_pct`, `implied_elasticity`, and—when `unit_cost` is supplied—
   `unit_cost`, `baseline_gross_margin`, `scenario_gross_margin`, `gross_margin_delta`, and
   `gross_margin_delta_pct`.
5. **Stage 5 — Operational Inventory Policy Shift:** When `include_inventory_impact=True`, invokes
   `compute_inventory_policy` under baseline and shocked demand to quantify
   `baseline_reorder_point`, `scenario_reorder_point`, `delta_reorder_point`,
   `baseline_safety_stock`, `scenario_safety_stock`, `delta_safety_stock`, `baseline_order_qty`,
   `scenario_order_qty`, and `delta_order_qty`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from .backtest import backtest_cell
from .calibration import (
    Calibration,
    apply_calibration,
    apply_output_constraints,
    apply_quantile_and_cost_rules,
    calibrate_from_oof,
    has_output_constraints,
    resolve_arm,
)
from .config import LEARNED_STRATEGIES, RunConfig, corrected_arm_for
from .features import (
    build_features,
    build_future_features,
    extract_static_covariates,
    fit_transform_lambda,
    regularize_series_grid,
    resolve_model_variant,
)
from .models import base_model_name, get_model
from .models.base_model import ModelContext
from .monotonicity import audit_scenario_monotonicity, infer_shock_sign_from_spec
from .seasonality import seasonal_period
from .worker import is_panel_model

__all__ = [
    "apply_scenario_shocks",
    "audit_scenario_monotonicity",
    "compare_scenario_margin_curves",
    "plot_scenarios",
    "simulate_scenarios",
    "summarize_scenarios",
]


def _parse_single_shock(spec: Any) -> list[dict[str, Any]]:
    """Normalize a user shock specification into a list of ``{op, value, start_date, end_date}``."""
    if (
        isinstance(spec, tuple)
        and len(spec) == 2
        and isinstance(spec[0], str)
        and spec[0].strip().lower()
        in ("=", "*", "+", "-", "%", "set", "mul", "multiply", "add", "delta", "pct", "percent")
    ):
        return _parse_single_shock({"op": spec[0], "value": spec[1]})

    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, Mapping)):
        out: list[dict[str, Any]] = []
        for item in spec:
            out.extend(_parse_single_shock(item))
        return out

    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return [{"op": "set", "value": float(spec), "start_date": None, "end_date": None}]

    if isinstance(spec, str):
        s = spec.strip()
        if s.endswith("%"):
            pct = float(s[:-1].strip()) / 100.0
            return [{"op": "mul", "value": 1.0 + pct, "start_date": None, "end_date": None}]
        if s.startswith("*"):
            return [
                {"op": "mul", "value": float(s[1:].strip()), "start_date": None, "end_date": None}
            ]
        if s.startswith("="):
            return [
                {"op": "set", "value": float(s[1:].strip()), "start_date": None, "end_date": None}
            ]
        if s.startswith(("+", "-")):
            return [{"op": "add", "value": float(s), "start_date": None, "end_date": None}]
        return [{"op": "set", "value": float(s), "start_date": None, "end_date": None}]

    if isinstance(spec, Mapping):
        start_d = spec.get("start_date") or spec.get("start")
        end_d = spec.get("end_date") or spec.get("end")
        if "op" in spec:
            op_raw = str(spec["op"]).lower().strip()
            val = float(spec["value"])
            if op_raw in ("mul", "multiply", "*"):
                op = "mul"
            elif op_raw in ("pct", "percent", "%"):
                op = "mul"
                val = 1.0 + (val / 100.0 if abs(val) > 1.0 else val)
            elif op_raw in ("add", "delta", "+"):
                op = "add"
            elif op_raw in ("set", "override", "="):
                op = "set"
            else:
                raise ValueError(
                    f"Unsupported shock op {op_raw!r}; expected 'mul', 'add', or 'set'"
                )
            return [{"op": op, "value": val, "start_date": start_d, "end_date": end_d}]
        if "factor" in spec:
            return [
                {
                    "op": "mul",
                    "value": float(spec["factor"]),
                    "start_date": start_d,
                    "end_date": end_d,
                }
            ]
        if "pct" in spec:
            pct_v = float(spec["pct"])
            factor = 1.0 + (pct_v / 100.0 if abs(pct_v) > 1.0 else pct_v)
            return [{"op": "mul", "value": factor, "start_date": start_d, "end_date": end_d}]
        if "delta" in spec or "add" in spec:
            dv = float(spec.get("delta", spec.get("add", 0.0)))
            return [{"op": "add", "value": dv, "start_date": start_d, "end_date": end_d}]
        if "set" in spec or "value" in spec:
            sv = float(spec.get("set", spec.get("value", 0.0)))
            return [{"op": "set", "value": sv, "start_date": start_d, "end_date": end_d}]

    raise ValueError(
        f"Invalid scenario shock specification {spec!r}. Use a scalar (1.0), string ('*0.85', "
        "'-15%', '+5', '=1'), or dict ({'op': 'mul', 'value': 0.85, 'start_date': ...})."
    )


def apply_scenario_shocks(
    df: pd.DataFrame,
    shocks: Mapping[str, Any],
    *,
    date_col: str = "ds",
) -> pd.DataFrame:
    """Apply counterfactual covariate shocks to a copy of ``df`` (pure).

    Supports full-horizon or date-windowed perturbations:

    - Multiplicative: ``{"price_index": "*0.85"}``, ``{"price_index": "-15%"}``, or
      ``{"price_index": {"op": "mul", "value": 0.85, "start_date": "2026-03-01",
      "end_date": "2026-03-07"}}``
    - Additive: ``{"temperature": "+5.0"}`` or ``{"temperature": {"op": "add", "value": 5.0}}``
    - Override: ``{"promo": "=1"}``, ``{"promo": 1}``, or ``{"promo": {"op": "set", "value": 1.0}}``
    """
    if df is None or df.empty or not shocks:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    eff_date_col = (
        date_col
        if date_col in out.columns
        else (
            "ds"
            if "ds" in out.columns
            else ("forecast_date" if "forecast_date" in out.columns else None)
        )
    )
    dt_series = pd.to_datetime(out[eff_date_col]) if eff_date_col is not None else None

    for col, raw_spec in shocks.items():
        if col not in out.columns:
            out[col] = 0.0
        col_vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for rule in _parse_single_shock(raw_spec):
            mask = np.ones(len(out), dtype=bool)
            if dt_series is not None:
                if rule["start_date"] is not None:
                    mask &= (dt_series >= pd.Timestamp(rule["start_date"])).to_numpy(dtype=bool)
                if rule["end_date"] is not None:
                    mask &= (dt_series <= pd.Timestamp(rule["end_date"])).to_numpy(dtype=bool)
            op = rule["op"]
            val = float(rule["value"])
            if op == "mul":
                col_vals[mask] = col_vals[mask] * val
            elif op == "add":
                col_vals[mask] = col_vals[mask] + val
            elif op == "set":
                col_vals[mask] = val
        out[col] = col_vals
    return out


def _build_baseline_future_frame(
    hist_df: pd.DataFrame,
    cfg: RunConfig,
    cov_cols: Sequence[str],
    future_covariates_df: pd.DataFrame | None,
    ts_id: str,
) -> pd.DataFrame:
    """Construct unshocked future covariate DataFrame for ``ts_id`` over ``cfg.data.horizon``."""
    d = cfg.data
    horizon = d.horizon
    last_dt = pd.Timestamp(pd.to_datetime(hist_df[d.date_col]).max())
    future_idx = pd.date_range(start=last_dt, periods=horizon + 1, freq=d.freq)[1:]
    base_df = pd.DataFrame({d.date_col: future_idx, d.ts_id_col: ts_id})

    if future_covariates_df is not None and not future_covariates_df.empty:
        sub_fut = future_covariates_df
        if d.ts_id_col in sub_fut.columns:
            sub_fut = sub_fut[sub_fut[d.ts_id_col].astype(str) == str(ts_id)]
        if not sub_fut.empty:
            f_date_col = (
                d.date_col
                if d.date_col in sub_fut.columns
                else ("ds" if "ds" in sub_fut.columns else None)
            )
            if f_date_col is not None:
                mapped = (
                    sub_fut.assign(_dt=pd.to_datetime(sub_fut[f_date_col]))
                    .drop_duplicates(subset=["_dt"], keep="last")
                    .set_index("_dt")
                )
                for c in cov_cols:
                    if c in mapped.columns:
                        raw_mapped = base_df[d.date_col].map(mapped[c])
                        base_df[c] = pd.to_numeric(raw_mapped, errors="coerce")

    period = max(1, seasonal_period(d.freq))
    for c in cov_cols:
        if c in base_df.columns and base_df[c].notna().all():
            continue
        if c in hist_df.columns:
            hist_vals = (
                pd.to_numeric(hist_df[c], errors="coerce").ffill().bfill().to_numpy(dtype=float)
            )
            if hist_vals.size >= period and period > 1:
                cycle = hist_vals[-period:]
                tiled = np.resize(cycle, horizon)
            elif hist_vals.size > 0:
                tiled = np.full(horizon, float(hist_vals[-1]), dtype=float)
            else:
                tiled = np.zeros(horizon, dtype=float)
        else:
            tiled = np.zeros(horizon, dtype=float)
        if c in base_df.columns:
            existing = pd.to_numeric(base_df[c], errors="coerce").to_numpy(dtype=float)
            base_df[c] = np.where(np.isfinite(existing), existing, tiled)
        else:
            base_df[c] = tiled
    return base_df


def _infer_price_col(
    explicit_price_col: str | None,
    cov_cols: Sequence[str],
    scenarios: Mapping[str, Mapping[str, Any]],
    panel_cols: Sequence[str] = (),
) -> str | None:
    """Resolve ``price_col`` when explicitly provided or when a price column is shocked/present."""
    if explicit_price_col is not None:
        return explicit_price_col
    shocked_cols = {str(k) for shocks in scenarios.values() for k in shocks}
    for cand in ("price", "unit_price", "sell_price", "price_index", "price_idx", "avg_price"):
        if cand in shocked_cols or cand in cov_cols or cand in panel_cols:
            return cand
    return None


def _resolve_unit_cost_array(
    unit_cost: float | Mapping[str, float] | str | None,
    *,
    ts_id: str,
    fut_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    horizon: int,
) -> np.ndarray | None:
    """Resolve per-step unit cost vector of length ``horizon`` for ``ts_id``, or ``None``."""
    if unit_cost is None:
        return None
    if isinstance(unit_cost, (int, float)) and not isinstance(unit_cost, bool):
        return np.full(horizon, float(unit_cost), dtype=float)
    if isinstance(unit_cost, Mapping):
        if str(ts_id) in unit_cost:
            return np.full(horizon, float(unit_cost[str(ts_id)]), dtype=float)
        if unit_cost:
            mean_c = float(np.mean([float(v) for v in unit_cost.values()]))
            return np.full(horizon, mean_c, dtype=float)
        return np.zeros(horizon, dtype=float)
    if isinstance(unit_cost, str):
        if unit_cost in fut_df.columns:
            vals = pd.to_numeric(fut_df[unit_cost], errors="coerce").ffill().bfill().fillna(0.0)
            return vals.to_numpy(dtype=float)[:horizon]
        if unit_cost in hist_df.columns:
            h_vals = pd.to_numeric(hist_df[unit_cost], errors="coerce").dropna()
            last_v = float(h_vals.iloc[-1]) if not h_vals.empty else 0.0
            return np.full(horizon, last_v, dtype=float)
    return None


def _expected_scenario_dir(
    shocks: Mapping[str, Any],
    constraints: Mapping[str, int],
) -> int:
    """Return ``-1`` or ``+1`` when all active constrained shocks agree on direction, else ``0``."""
    if not shocks or not constraints:
        return 0
    dirs: list[int] = []
    for col, spec in shocks.items():
        c_sign = int(constraints.get(str(col), 0))
        sh_sign = infer_shock_sign_from_spec(spec)
        if c_sign != 0 and sh_sign != 0:
            dirs.append(c_sign * sh_sign)
    if dirs and all(d == dirs[0] for d in dirs):
        return int(dirs[0])
    return 0


def _clamp_monotonic_trajectories(
    df: pd.DataFrame,
    scenarios: Mapping[str, Mapping[str, Any]],
    model_mono_map: Mapping[str, dict[str, int]],
    model_native_mono: Mapping[str, bool],
) -> pd.DataFrame:
    """Clamp counterfactual trajectories to respect ``monotonic_constraints`` for models without
    native training-time sign constraints (e.g. ``sarimax@monotone``, ``prophet@monotone``, or
    unconstrained ``ensemble_ridge`` / ``ensemble_xgb`` over all-monotone base models).
    """
    if df.empty:
        return df
    all_base_monotone = bool(model_mono_map) and all(
        any(v != 0 for v in cmap.values()) for cmap in model_mono_map.values()
    )
    merged_base_constraints: dict[str, int] = (
        {k: v for cmap in model_mono_map.values() for k, v in cmap.items() if v != 0}
        if all_base_monotone
        else {}
    )

    base_rows = df[df["scenario"] == "baseline"]
    if base_rows.empty:
        return df
    base_lookup = (
        base_rows.drop_duplicates(subset=["ts_id", "model_type", "ds"])
        .set_index(["ts_id", "model_type", "ds"])["yhat"]
        .to_dict()
    )

    out_records: list[dict[Any, Any]] = []
    for rec in df.to_dict(orient="records"):
        scen = str(rec["scenario"])
        mtype = str(rec["model_type"])
        if scen == "baseline" or scen not in scenarios:
            out_records.append(rec)
            continue

        if mtype.startswith("ensemble_"):
            if not all_base_monotone:
                out_records.append(rec)
                continue
            c_map = merged_base_constraints
        else:
            if model_native_mono.get(mtype, False):
                out_records.append(rec)
                continue
            c_map = model_mono_map.get(mtype, {})
            if not any(v != 0 for v in c_map.values()):
                out_records.append(rec)
                continue

        exp_dir = _expected_scenario_dir(scenarios[scen], c_map)
        if exp_dir == 0:
            out_records.append(rec)
            continue

        key = (str(rec["ts_id"]), mtype, pd.Timestamp(rec["ds"]))
        b_yh = base_lookup.get(key)
        if b_yh is None:
            out_records.append(rec)
            continue

        yh = float(rec["yhat"])
        if (exp_dir == -1 and yh > float(b_yh)) or (exp_dir == 1 and yh < float(b_yh)):
            shift = float(b_yh) - yh
            rec = dict(rec)
            rec["yhat"] = float(b_yh)
            rec["yhat_lower"] = float(rec["yhat_lower"]) + shift
            rec["yhat_upper"] = float(rec["yhat_upper"]) + shift
        out_records.append(rec)

    return pd.DataFrame.from_records(out_records)


def _apply_learned_ensemble_rows(
    base_df: pd.DataFrame,
    wmap: dict[str, float],
    strategy: str,
) -> list[dict[str, Any]]:
    """Blend base scenario predictions by learned stacking weights into ``ensemble_<strategy>``."""
    model_type = f"ensemble_{strategy}"
    rows: list[dict[str, Any]] = []
    if base_df.empty:
        return rows
    models = [m for m in wmap if m in set(base_df["model_type"].unique())]
    if not models:
        return rows
    weights = np.array([wmap[m] for m in models], dtype=float)

    def _blend(value_col: str) -> tuple[pd.Index, np.ndarray]:
        wide = base_df.pivot_table(
            index=["ts_id", "forecast_date"], columns="model_type", values=value_col
        ).reindex(columns=models)
        vals = wide.to_numpy(dtype=float)
        present = ~np.isnan(vals)
        wrow = present * weights
        denom = wrow.sum(axis=1)
        num = np.nansum(np.where(present, vals, 0.0) * weights, axis=1)
        blended = np.where(denom > 0.0, num / np.where(denom > 0.0, denom, 1.0), np.nan)
        return wide.index, blended

    index, yhat = _blend("yhat")
    _, lower = _blend("yhat_lower")
    _, upper = _blend("yhat_upper")
    for (ts_id, forecast_date), yh, lo, up in zip(index, yhat, lower, upper, strict=True):
        if np.isnan(yh):
            continue
        rows.append(
            {
                "ts_id": str(ts_id),
                "model_type": model_type,
                "forecast_date": forecast_date,
                "yhat": float(yh),
                "yhat_lower": float(yh) if np.isnan(lo) else float(lo),
                "yhat_upper": float(yh) if np.isnan(up) else float(up),
            }
        )
    return rows


def simulate_scenarios(
    data: pd.DataFrame,
    cfg: RunConfig | Mapping[str, Any],
    scenarios: Mapping[str, Mapping[str, Any]],
    *,
    models: Sequence[str] | None = None,
    future_covariates_df: pd.DataFrame | None = None,
    price_col: str | None = None,
    unit_cost: float | Mapping[str, float] | str | None = None,
    reconcile: bool = True,
    include_inventory_impact: bool = False,
    lead_time: float = 7.0,
    lead_time_std: float = 0.0,
    service_level: float | None = 0.95,
    oof_df: pd.DataFrame | None = None,
    inventory_params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run full-chain counterfactual scenarios across base models, ensembles, hierarchy, margin,
    and inventory policy (pure, offline).

    1. **Base Model Shocks:** Fits each candidate model once per series (applying `@<tag>` variant
       resolution and `monotonic_constraints` sign guardrails) and evaluates `"baseline"` plus each
       named scenario in ``scenarios``.
    2. **Ensemble Propagation:** When ``cfg.ensemble.enabled`` is True and at least 2 base models
       are evaluated, propagates each scenario through ``combine_calculated`` and ``fit_learned``.
    3. **Hierarchy Reconciliation:** When ``reconcile=True`` and ``cfg.hierarchy.enabled`` is True,
       projects every scenario across the hierarchy summing matrix ``S`` via ``reconcile_forecasts``
       so additive coherence (`verify_coherence < 1e-6`) holds across all nodes.
    4. **Revenue & Gross-Margin Delta:** Computes revenue deltas (`price * yhat`) and—when
       ``unit_cost`` is supplied—gross-margin deltas (`(price - unit_cost) * yhat`).
    5. **Inventory Policy Shift:** When ``include_inventory_impact=True`` (or ``inventory_params``
       is supplied), computes baseline vs. scenario Reorder Points (`ROP`), Safety Stock (`SS`), and
       Recommended Order Quantities (`Q`).
    """
    from .errors import ConfigError

    if isinstance(cfg, Mapping):
        cfg = RunConfig.model_validate(cfg)
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        raise ConfigError("simulate_scenarios requires a non-empty DataFrame.")
    if not scenarios:
        raise ConfigError("simulate_scenarios requires at least one named scenario.")

    shock_cols = sorted({str(k) for shocks in scenarios.values() for k in shocks})
    known_covs = list(dict.fromkeys([*cfg.features.known_future_covariates, *shock_cols]))
    if price_col and price_col not in known_covs:
        known_covs.append(price_col)

    work_cfg = cfg
    if set(known_covs) != set(cfg.features.known_future_covariates):
        work_cfg = cfg.model_copy(
            update={"features": cfg.features.model_copy(update={"future_covariates": known_covs})}
        )

    d = work_cfg.data
    raw_panel = data.copy()
    if d.ts_id_col not in raw_panel.columns:
        raw_panel[d.ts_id_col] = "series_0"
    if d.target_col not in raw_panel.columns or d.date_col not in raw_panel.columns:
        raise ConfigError(
            f"simulate_scenarios data is missing required columns "
            f"{[c for c in (d.date_col, d.target_col) if c not in raw_panel.columns]}"
        )

    # Separate historical rows from any trailing future rows embedded in `data`
    y_num = pd.to_numeric(raw_panel[d.target_col], errors="coerce")
    if y_num.isna().any() and y_num.notna().any():
        embedded_fut = raw_panel[y_num.isna()].copy()
        panel = raw_panel[y_num.notna()].copy()
        if future_covariates_df is None or future_covariates_df.empty:
            future_covariates_df = embedded_fut
    else:
        panel = raw_panel

    raw_leaf_panel = panel
    raw_leaf_future = future_covariates_df

    eff_price_col = _infer_price_col(
        price_col,
        known_covs,
        scenarios,
        panel_cols=list(panel.columns),
    )
    if eff_price_col and eff_price_col not in known_covs:
        known_covs.append(eff_price_col)
        work_cfg = work_cfg.model_copy(
            update={
                "features": work_cfg.features.model_copy(update={"future_covariates": known_covs})
            }
        )

    candidate_models = list(models) if models is not None else list(work_cfg.models)
    requested_ensembles = [m for m in candidate_models if str(m).startswith("ensemble_")]
    base_candidates = [m for m in candidate_models if not str(m).startswith("ensemble_")]
    if not base_candidates:
        base_candidates = list(work_cfg.models)

    py_models: list[str] = []
    for m in base_candidates:
        try:
            m_cls = get_model(base_model_name(m))
            if m_cls.runtime == "python" and m_cls.is_available():
                py_models.append(m)
        except Exception:  # noqa: BLE001 - skip unknown or unavailable non-python models
            continue
    if not py_models:
        py_models = ["naive_seasonal"]

    all_covs = list(dict.fromkeys([*work_cfg.features.all_covariates, *known_covs]))
    for c in all_covs:
        if c not in panel.columns:
            panel[c] = 1.0 if c == eff_price_col else 0.0

    # Stage 3 preparation: if hierarchy is enabled and reconcile=True, build full hierarchy panel
    hier_spec = None
    if reconcile and work_cfg.hierarchy.enabled and work_cfg.hierarchy.levels:
        from .reconciliation import TOTAL_NODE_ID, build_hierarchy

        req_h_cols = {col for lvl in work_cfg.hierarchy.levels for col in lvl}
        if req_h_cols.issubset(panel.columns):
            panel, hier_spec = build_hierarchy(panel, work_cfg)
            # Also aggregate future_covariates_df across hierarchy nodes if supplied for leaves only
            if (
                future_covariates_df is not None
                and not future_covariates_df.empty
                and d.ts_id_col in future_covariates_df.columns
            ):
                fut_ids = set(future_covariates_df[d.ts_id_col].astype(str).unique())
                if TOTAL_NODE_ID not in fut_ids:
                    f_dt_col = (
                        d.date_col
                        if d.date_col in future_covariates_df.columns
                        else ("ds" if "ds" in future_covariates_df.columns else None)
                    )
                    if f_dt_col is not None:
                        num_f_cols = [
                            c
                            for c in all_covs
                            if c in future_covariates_df.columns
                            and pd.api.types.is_numeric_dtype(future_covariates_df[c])
                        ]
                        if num_f_cols:
                            s_mat = hier_spec.summing_matrix
                            b_ids = list(hier_spec.bottom_ids)
                            agg_fut_parts: list[pd.DataFrame] = [future_covariates_df]
                            for idx_n, nid in enumerate(
                                hier_spec.node_ids[: hier_spec.n_aggregated]
                            ):
                                member_b = [
                                    b_ids[j] for j in range(len(b_ids)) if s_mat[idx_n, j] > 0.0
                                ]
                                sub_f = future_covariates_df[
                                    future_covariates_df[d.ts_id_col].astype(str).isin(member_b)
                                ]
                                if not sub_f.empty:
                                    agg_f = (
                                        sub_f.groupby(f_dt_col, as_index=False)[num_f_cols]
                                        .mean()
                                        .assign(**{d.ts_id_col: nid})
                                    )
                                    agg_fut_parts.append(agg_f)
                            future_covariates_df = pd.concat(agg_fut_parts, ignore_index=True)

    all_scenarios: dict[str, Mapping[str, Any]] = {"baseline": {}, **dict(scenarios)}
    target_qs = work_cfg.output.effective_quantiles
    fallback_arm = corrected_arm_for(work_cfg.backtest.decision_metric)
    horizon = d.horizon

    # Collect per-(scenario, ts_id, model_type, ds) raw predictions + prices + unit costs
    raw_scen_rows: list[dict[str, Any]] = []
    price_lookup: dict[tuple[str, str, pd.Timestamp], float] = {}
    cost_lookup: dict[tuple[str, int], float] = {}
    collected_oof: list[pd.DataFrame] = []

    s_clean_by_tid: dict[str, pd.DataFrame] = {}
    lam_by_tid: dict[str, float | None] = {}
    base_fut_by_tid: dict[str, pd.DataFrame] = {}
    static_map: dict[str, dict[str, Any]] = {}

    for raw_tid, s_group in panel.groupby(d.ts_id_col, sort=True):
        tid = str(raw_tid)
        s_clean = regularize_series_grid(s_group, work_cfg)
        raw_target = s_clean[d.target_col].astype(float)
        lam = fit_transform_lambda(raw_target, work_cfg.features.transform)
        base_fut = _build_baseline_future_frame(
            s_clean, work_cfg, known_covs, future_covariates_df, tid
        )
        s_clean_by_tid[tid] = s_clean
        lam_by_tid[tid] = lam
        base_fut_by_tid[tid] = base_fut
        try:
            sc = extract_static_covariates(s_clean, work_cfg)
            if sc:
                static_map[tid] = sc
        except Exception:  # noqa: BLE001 - static covariates optional
            pass

        hist_for_cost = s_clean
        fut_for_cost = base_fut
        if isinstance(unit_cost, str):
            if unit_cost not in hist_for_cost.columns and unit_cost in raw_leaf_panel.columns:
                raw_h_sub = raw_leaf_panel[
                    raw_leaf_panel[d.ts_id_col].astype(str) == tid
                ].sort_values(d.date_col)
                if not raw_h_sub.empty:
                    hist_for_cost = raw_h_sub
            if (
                unit_cost not in fut_for_cost.columns
                and raw_leaf_future is not None
                and not raw_leaf_future.empty
                and unit_cost in raw_leaf_future.columns
            ):
                raw_f_sub = (
                    raw_leaf_future[raw_leaf_future[d.ts_id_col].astype(str) == tid]
                    if d.ts_id_col in raw_leaf_future.columns
                    else raw_leaf_future
                )
                if not raw_f_sub.empty:
                    fut_for_cost = raw_f_sub

        uc_arr = _resolve_unit_cost_array(
            unit_cost,
            ts_id=tid,
            fut_df=fut_for_cost,
            hist_df=hist_for_cost,
            horizon=horizon,
        )
        if uc_arr is not None:
            for step_i, uc_v in enumerate(uc_arr, start=1):
                cost_lookup[(tid, step_i)] = float(uc_v)

        for scen_name, shocks in all_scenarios.items():
            fut_cov_df = apply_scenario_shocks(base_fut, shocks, date_col=d.date_col)
            if eff_price_col and eff_price_col in fut_cov_df.columns:
                p_vals = pd.to_numeric(fut_cov_df[eff_price_col], errors="coerce").to_numpy(
                    dtype=float
                )
                dt_vals = pd.to_datetime(fut_cov_df[d.date_col]).tolist()
                for dt_v, pv in zip(dt_vals, p_vals, strict=False):
                    price_lookup[(scen_name, tid, pd.Timestamp(dt_v))] = float(pv)
            elif unit_cost is not None:
                dt_vals = pd.to_datetime(fut_cov_df[d.date_col]).tolist()
                for dt_v in dt_vals:
                    price_lookup[(scen_name, tid, pd.Timestamp(dt_v))] = 1.0

    uids = sorted(s_clean_by_tid.keys())
    panel_py_models: list[str] = []
    local_py_models: list[str] = []
    model_mono_map: dict[str, dict[str, int]] = {}
    model_native_mono: dict[str, bool] = {}

    for mname in py_models:
        m_cls = get_model(base_model_name(mname))
        spec = resolve_model_variant(work_cfg, mname, model_cls=m_cls)
        model_mono_map[mname] = dict(spec.effective_cfg.features.monotonic_constraints)
        model_native_mono[mname] = bool(m_cls.supports_monotonic_constraints)
        if is_panel_model(mname, work_cfg) and (m_cls.supports_global or m_cls.supports_hybrid):
            panel_py_models.append(mname)
        else:
            local_py_models.append(mname)

    # Execute @global / @hybrid panel models once across all series via fit_panel / predict_panel
    for mname in tuple(panel_py_models):
        m_cls = get_model(base_model_name(mname))
        spec = resolve_model_variant(work_cfg, mname, model_cls=m_cls)
        model_cfg = spec.effective_cfg
        m_params: dict[str, Any] = dict(spec.clean_params)
        panel_ctx = ModelContext(
            freq=d.freq,
            horizon=model_cfg.max_horizon,
            transform=model_cfg.features.transform,
            past_covariates=tuple(model_cfg.features.past_covariates),
            monotonic_constraints=dict(model_cfg.features.monotonic_constraints),
        )
        try:
            full_series_map: dict[str, tuple[pd.Series, pd.DataFrame | None]] = {}
            for tid in uids:
                full_series_map[tid] = build_features(
                    s_clean_by_tid[tid],
                    model_cfg,
                    lam_by_tid[tid],
                    m_cls.lags_covariates_internally,
                )
            est = m_cls(m_params, panel_ctx)
            est.fit_panel(full_series_map, static_map=static_map if static_map else None)

            cal_by_tid: dict[str, Calibration | None] = dict.fromkeys(uids, None)
            arm_by_tid: dict[str, str] = {}
            for tid in uids:
                cell_oof: pd.DataFrame | None = None
                if (
                    oof_df is not None
                    and not oof_df.empty
                    and {"ts_id", "model_type"}.issubset(oof_df.columns)
                ):
                    sub_oof = oof_df[
                        (oof_df["ts_id"].astype(str) == tid) & (oof_df["model_type"] == mname)
                    ]
                    if not sub_oof.empty:
                        cell_oof = sub_oof.copy()
                        if "ds" not in cell_oof.columns and "forecast_date" in cell_oof.columns:
                            cell_oof["ds"] = pd.to_datetime(cell_oof["forecast_date"])
                        cal_by_tid[tid] = calibrate_from_oof(cell_oof, target_qs)
                        oof_tagged = cell_oof.copy()
                        oof_tagged["ts_id"] = tid
                        oof_tagged["model_type"] = mname
                        if "forecast_date" not in oof_tagged.columns and "ds" in oof_tagged.columns:
                            oof_tagged["forecast_date"] = pd.to_datetime(oof_tagged["ds"])
                        collected_oof.append(oof_tagged)
                arm_val, _ = resolve_arm(
                    model_cfg.output.point_forecast,
                    cell_oof,
                    model_cfg.backtest.decision_metric,
                    fallback_arm,
                )
                arm_by_tid[tid] = arm_val

            for scen_name, shocks in all_scenarios.items():
                scen_futr_map: dict[str, pd.DataFrame | None] = {}
                fut_cov_by_tid: dict[str, pd.DataFrame] = {}
                for tid in uids:
                    fut_cov_df = apply_scenario_shocks(
                        base_fut_by_tid[tid], shocks, date_col=d.date_col
                    )
                    fut_cov_by_tid[tid] = fut_cov_df
                    y_hist_t, X_hist_t = full_series_map[tid]
                    scen_futr_map[tid] = build_future_features(
                        y_hist_t,
                        X_hist_t,
                        model_cfg,
                        horizon=horizon,
                        future_covariates_df=fut_cov_df,
                    )
                preds_map = est.predict_panel(
                    horizon,
                    scen_futr_map,
                    quantiles=target_qs,
                    transform_lambdas=lam_by_tid,
                )
                for tid in uids:
                    raw_pred = preds_map[tid]
                    cal_pred, _ = apply_calibration(raw_pred, cal_by_tid[tid], arm_by_tid[tid])
                    cal_pred, _ = apply_quantile_and_cost_rules(cal_pred, model_cfg.output)
                    if has_output_constraints(model_cfg.output):
                        cal_pred = apply_output_constraints(
                            cal_pred, model_cfg.output, bounds_df=fut_cov_by_tid[tid]
                        )
                    yh = pd.to_numeric(cal_pred["yhat"], errors="coerce").to_numpy(dtype=float)
                    yl = pd.to_numeric(cal_pred["yhat_lower"], errors="coerce").to_numpy(
                        dtype=float
                    )
                    yu = pd.to_numeric(cal_pred["yhat_upper"], errors="coerce").to_numpy(
                        dtype=float
                    )
                    ds_vals = pd.to_datetime(cal_pred["ds"]).tolist()
                    for idx in range(len(cal_pred)):
                        raw_scen_rows.append(
                            {
                                "scenario": scen_name,
                                "ts_id": tid,
                                "model_type": mname,
                                "ds": pd.Timestamp(ds_vals[idx]),
                                "horizon_step": idx + 1,
                                "yhat": float(yh[idx]),
                                "yhat_lower": float(yl[idx]),
                                "yhat_upper": float(yu[idx]),
                            }
                        )
        except Exception:  # noqa: BLE001 - fall back to per-series execution if panel fit fails
            local_py_models.append(mname)

    for tid in uids:
        s_clean = s_clean_by_tid[tid]
        lam = lam_by_tid[tid]
        base_fut = base_fut_by_tid[tid]

        for mname in local_py_models:
            m_cls = get_model(base_model_name(mname))
            spec = resolve_model_variant(work_cfg, mname, model_cls=m_cls)
            model_cfg = spec.effective_cfg
            m_params = dict(spec.clean_params)
            ctx = ModelContext(
                freq=d.freq,
                horizon=model_cfg.max_horizon,
                transform=model_cfg.features.transform,
                transform_lambda=lam,
                monotonic_constraints=dict(model_cfg.features.monotonic_constraints),
            )
            y_hist, X_hist = build_features(
                s_clean, model_cfg, lam, m_cls.lags_covariates_internally
            )
            est = m_cls(m_params, ctx)
            est.fit(y_hist, X_hist)

            cal: Calibration | None = None
            cell_oof = None
            if (
                oof_df is not None
                and not oof_df.empty
                and {"ts_id", "model_type"}.issubset(oof_df.columns)
            ):
                sub_oof = oof_df[
                    (oof_df["ts_id"].astype(str) == tid) & (oof_df["model_type"] == mname)
                ]
                if not sub_oof.empty:
                    cell_oof = sub_oof.copy()
                    if "ds" not in cell_oof.columns and "forecast_date" in cell_oof.columns:
                        cell_oof["ds"] = pd.to_datetime(cell_oof["forecast_date"])
            if cell_oof is None and model_cfg.backtest.enabled:
                bt_cfg_for_cell = model_cfg
                needed_obs = (
                    model_cfg.backtest.min_train
                    + max(0, model_cfg.backtest.n_folds - 1) * model_cfg.backtest.step
                    + model_cfg.backtest.horizon
                )
                if len(s_clean) < needed_obs and len(s_clean) > model_cfg.backtest.horizon + 14:
                    adapted_min_train = max(
                        14,
                        len(s_clean)
                        - max(0, model_cfg.backtest.n_folds - 1) * model_cfg.backtest.step
                        - model_cfg.backtest.horizon,
                    )
                    bt_cfg_for_cell = model_cfg.model_copy(
                        update={
                            "backtest": model_cfg.backtest.model_copy(
                                update={"min_train": adapted_min_train}
                            )
                        }
                    )
                oof_raw, _, _ = backtest_cell(
                    s_clean,
                    partial(m_cls, m_params, ctx),
                    bt_cfg_for_cell,
                    lam,
                    None,
                    m_cls.lags_covariates_internally,
                )
                cell_oof = oof_raw if not oof_raw.empty else None
            if cell_oof is not None and not cell_oof.empty:
                cal = calibrate_from_oof(cell_oof, target_qs)
                oof_tagged = cell_oof.copy()
                oof_tagged["ts_id"] = tid
                oof_tagged["model_type"] = mname
                if "forecast_date" not in oof_tagged.columns and "ds" in oof_tagged.columns:
                    oof_tagged["forecast_date"] = pd.to_datetime(oof_tagged["ds"])
                collected_oof.append(oof_tagged)

            arm, _ = resolve_arm(
                model_cfg.output.point_forecast,
                cell_oof,
                model_cfg.backtest.decision_metric,
                fallback_arm,
            )

            for scen_name, shocks in all_scenarios.items():
                fut_cov_df = apply_scenario_shocks(base_fut, shocks, date_col=d.date_col)
                X_fut = build_future_features(
                    y_hist,
                    X_hist,
                    model_cfg,
                    horizon=horizon,
                    future_covariates_df=fut_cov_df,
                )
                raw_pred = est.predict(horizon, X_fut, quantiles=target_qs)
                cal_pred, _ = apply_calibration(raw_pred, cal, arm)
                cal_pred, _ = apply_quantile_and_cost_rules(cal_pred, model_cfg.output)
                if has_output_constraints(model_cfg.output):
                    cal_pred = apply_output_constraints(
                        cal_pred, model_cfg.output, bounds_df=fut_cov_df
                    )
                yh = pd.to_numeric(cal_pred["yhat"], errors="coerce").to_numpy(dtype=float)
                yl = pd.to_numeric(cal_pred["yhat_lower"], errors="coerce").to_numpy(dtype=float)
                yu = pd.to_numeric(cal_pred["yhat_upper"], errors="coerce").to_numpy(dtype=float)
                ds_vals = pd.to_datetime(cal_pred["ds"]).tolist()
                for idx in range(len(cal_pred)):
                    raw_scen_rows.append(
                        {
                            "scenario": scen_name,
                            "ts_id": tid,
                            "model_type": mname,
                            "ds": pd.Timestamp(ds_vals[idx]),
                            "horizon_step": idx + 1,
                            "yhat": float(yh[idx]),
                            "yhat_lower": float(yl[idx]),
                            "yhat_upper": float(yu[idx]),
                        }
                    )

    combined_oof_df = (
        pd.concat(collected_oof, ignore_index=True)
        if collected_oof
        else (oof_df.copy() if oof_df is not None else pd.DataFrame())
    )

    scen_pred_df = pd.DataFrame.from_records(raw_scen_rows)
    if scen_pred_df.empty:
        return scen_pred_df

    # Stage 2: Winning Ensemble Propagation
    run_ensembles = (
        work_cfg.ensemble.enabled
        and len(py_models) >= 2
        and (models is None or bool(requested_ensembles))
    ) or bool(requested_ensembles)
    if run_ensembles and len(py_models) >= 2:
        from .ensembler import combine_calculated, fit_learned

        ens_cfg = work_cfg.model_copy(update={"models": list(py_models)})
        if requested_ensembles:
            req_strats = [m.removeprefix("ensemble_") for m in requested_ensembles]
            valid_strats = [s for s in req_strats if s in ens_cfg.ensemble.strategies] or req_strats
            ens_cfg = ens_cfg.model_copy(
                update={
                    "ensemble": ens_cfg.ensemble.model_copy(
                        update={"enabled": True, "strategies": valid_strats}
                    )
                }
            )

        learned_weights: dict[str, dict[str, float]] = {}
        if (
            any(s in LEARNED_STRATEGIES for s in ens_cfg.ensemble.strategies)
            and not combined_oof_df.empty
            and ens_cfg.backtest.enabled
        ):
            try:
                learned_weights, _, _ = fit_learned(combined_oof_df, ens_cfg)
            except Exception:  # noqa: BLE001 - fallback to calculated strategies if OOF incomplete
                learned_weights = {}

        ens_rows: list[dict[str, Any]] = []
        for scen_key, s_grp in scen_pred_df.groupby("scenario", sort=False):
            base_for_ens = s_grp.rename(columns={"ds": "forecast_date"})[
                ["ts_id", "model_type", "forecast_date", "yhat", "yhat_lower", "yhat_upper"]
            ].copy()
            calc_rows = combine_calculated(
                base_for_ens,
                ens_cfg,
                oof_df=combined_oof_df if not combined_oof_df.empty else None,
            )
            for strategy, wmap in learned_weights.items():
                calc_rows.extend(_apply_learned_ensemble_rows(base_for_ens, wmap, strategy))

            step_map = (
                s_grp.drop_duplicates(subset=["ts_id", "ds"])
                .set_index(["ts_id", "ds"])["horizon_step"]
                .to_dict()
            )
            for er in calc_rows:
                if requested_ensembles and er["model_type"] not in requested_ensembles:
                    continue
                dt_ts = pd.Timestamp(er["forecast_date"])
                h_step = int(step_map.get((str(er["ts_id"]), dt_ts), 1))
                ens_rows.append(
                    {
                        "scenario": str(scen_key),
                        "ts_id": str(er["ts_id"]),
                        "model_type": str(er["model_type"]),
                        "ds": dt_ts,
                        "horizon_step": h_step,
                        "yhat": float(er["yhat"]),
                        "yhat_lower": float(
                            er["yhat"] if er.get("yhat_lower") is None else er["yhat_lower"]
                        ),
                        "yhat_upper": float(
                            er["yhat"] if er.get("yhat_upper") is None else er["yhat_upper"]
                        ),
                    }
                )
        if ens_rows:
            scen_pred_df = pd.concat(
                [scen_pred_df, pd.DataFrame.from_records(ens_rows)], ignore_index=True
            )

    scen_pred_df = _clamp_monotonic_trajectories(
        scen_pred_df,
        scenarios,
        model_mono_map,
        model_native_mono,
    )

    # Stage 3: Hierarchy Reconciliation across every scenario and model
    if hier_spec is not None:
        from .reconciliation import reconcile_forecasts

        primary_method = (
            work_cfg.hierarchy.reconciliation_methods[0]
            if work_cfg.hierarchy.reconciliation_methods
            else "mint_nonneg"
        )
        hist_for_rec = panel.rename(columns={d.ts_id_col: "ts_id", d.date_col: "ds"})
        rec_parts: list[pd.DataFrame] = []
        for scen_key, s_grp in scen_pred_df.groupby("scenario", sort=False):
            rec_s = reconcile_forecasts(
                s_grp,
                hier_spec,
                primary_method,
                history_df=hist_for_rec,
                oof_df=combined_oof_df if not combined_oof_df.empty else None,
                middle_level=work_cfg.hierarchy.middle_level,
                date_col="ds",
                ts_id_col="ts_id",
                target_col=d.target_col,
                stamp_method=False,
            )
            rec_s["scenario"] = str(scen_key)
            rec_parts.append(rec_s)
        if rec_parts:
            scen_pred_df = pd.concat(rec_parts, ignore_index=True)

    # Stage 4: Baseline comparison, Revenue, Implied Elasticity & Gross-Margin Delta
    base_slice = scen_pred_df[scen_pred_df["scenario"] == "baseline"]
    base_yhat_map = (
        base_slice.drop_duplicates(subset=["ts_id", "model_type", "ds"])
        .set_index(["ts_id", "model_type", "ds"])["yhat"]
        .to_dict()
    )

    has_price_or_cost = bool(price_lookup) or (unit_cost is not None)
    has_unit_cost = unit_cost is not None

    final_rows: list[dict[str, Any]] = []
    for r in scen_pred_df.to_dict(orient="records"):
        scen_s = str(r["scenario"])
        tid_s = str(r["ts_id"])
        mtype_s = str(r["model_type"])
        dt_ts = pd.Timestamp(r["ds"])
        h_step = int(r["horizon_step"])
        s_yh = float(r["yhat"])
        b_yh = float(base_yhat_map.get((tid_s, mtype_s, dt_ts), s_yh))
        d_yh = s_yh - b_yh
        d_pct = (d_yh / abs(b_yh)) if abs(b_yh) > 1e-12 else 0.0

        rec: dict[str, Any] = {
            "scenario": scen_s,
            "ts_id": tid_s,
            "model_type": mtype_s,
            "ds": dt_ts,
            "horizon_step": h_step,
            "yhat": s_yh,
            "yhat_lower": float(r["yhat_lower"]),
            "yhat_upper": float(r["yhat_upper"]),
            "baseline_yhat": b_yh,
            "delta_yhat": d_yh,
            "delta_pct": d_pct,
        }
        if has_price_or_cost:
            bp = float(price_lookup.get(("baseline", tid_s, dt_ts), 1.0))
            sp = float(price_lookup.get((scen_s, tid_s, dt_ts), bp))
            brev = bp * b_yh
            srev = sp * s_yh
            rev_delta = srev - brev
            rev_pct = (rev_delta / abs(brev)) if abs(brev) > 1e-12 else 0.0
            p_pct = ((sp - bp) / abs(bp)) if abs(bp) > 1e-12 else 0.0
            elasticity = (d_pct / p_pct) if abs(p_pct) > 1e-9 else float("nan")
            rec.update(
                {
                    "baseline_price": bp,
                    "price": sp,
                    "baseline_revenue": brev,
                    "scenario_revenue": srev,
                    "revenue_delta": rev_delta,
                    "revenue_delta_pct": rev_pct,
                    "implied_elasticity": elasticity,
                }
            )
            if has_unit_cost:
                uc = float(cost_lookup.get((tid_s, h_step), 0.0))
                bgm = (bp - uc) * b_yh
                sgm = (sp - uc) * s_yh
                gm_delta = sgm - bgm
                gm_pct = (gm_delta / abs(bgm)) if abs(bgm) > 1e-12 else 0.0
                rec.update(
                    {
                        "unit_cost": uc,
                        "baseline_gross_margin": bgm,
                        "scenario_gross_margin": sgm,
                        "gross_margin_delta": gm_delta,
                        "gross_margin_delta_pct": gm_pct,
                    }
                )
        final_rows.append(rec)

    if hier_spec is not None and has_price_or_cost and hier_spec.n_aggregated > 0:
        b_ids = list(hier_spec.bottom_ids)
        agg_ids = list(hier_spec.node_ids[: hier_spec.n_aggregated])
        s_agg = np.asarray(hier_spec.summing_matrix[: hier_spec.n_aggregated, :], dtype=float)

        rows_by_slice: dict[tuple[str, str, pd.Timestamp], dict[str, dict[str, Any]]] = {}
        for rec in final_rows:
            key = (str(rec["scenario"]), str(rec["model_type"]), pd.Timestamp(rec["ds"]))
            rows_by_slice.setdefault(key, {})[str(rec["ts_id"])] = rec

        for tid_map in rows_by_slice.values():
            if not all(bid in tid_map for bid in b_ids):
                continue
            leaf_brev = np.array(
                [float(tid_map[bid]["baseline_revenue"]) for bid in b_ids], dtype=float
            )
            leaf_srev = np.array(
                [float(tid_map[bid]["scenario_revenue"]) for bid in b_ids], dtype=float
            )
            agg_brev = s_agg @ leaf_brev
            agg_srev = s_agg @ leaf_srev

            if has_unit_cost:
                leaf_bgm = np.array(
                    [float(tid_map[bid]["baseline_gross_margin"]) for bid in b_ids], dtype=float
                )
                leaf_sgm = np.array(
                    [float(tid_map[bid]["scenario_gross_margin"]) for bid in b_ids], dtype=float
                )
                agg_bgm = s_agg @ leaf_bgm
                agg_sgm = s_agg @ leaf_sgm
            else:
                agg_bgm = np.zeros(len(agg_ids), dtype=float)
                agg_sgm = np.zeros(len(agg_ids), dtype=float)

            for idx_a, aid in enumerate(agg_ids):
                if aid not in tid_map:
                    continue
                arec = tid_map[aid]
                b_yh_a = float(arec["baseline_yhat"])
                s_yh_a = float(arec["yhat"])
                d_pct_a = float(arec["delta_pct"])
                brev_a = float(agg_brev[idx_a])
                srev_a = float(agg_srev[idx_a])
                rev_delta_a = srev_a - brev_a
                rev_pct_a = (rev_delta_a / abs(brev_a)) if abs(brev_a) > 1e-12 else 0.0
                bp_a = (brev_a / b_yh_a) if abs(b_yh_a) > 1e-12 else float(arec["baseline_price"])
                sp_a = (srev_a / s_yh_a) if abs(s_yh_a) > 1e-12 else float(arec["price"])
                p_pct_a = ((sp_a - bp_a) / abs(bp_a)) if abs(bp_a) > 1e-12 else 0.0
                elast_a = (d_pct_a / p_pct_a) if abs(p_pct_a) > 1e-9 else float("nan")
                arec.update(
                    {
                        "baseline_price": bp_a,
                        "price": sp_a,
                        "baseline_revenue": brev_a,
                        "scenario_revenue": srev_a,
                        "revenue_delta": rev_delta_a,
                        "revenue_delta_pct": rev_pct_a,
                        "implied_elasticity": elast_a,
                    }
                )
                if has_unit_cost:
                    bgm_a = float(agg_bgm[idx_a])
                    sgm_a = float(agg_sgm[idx_a])
                    gm_delta_a = sgm_a - bgm_a
                    gm_pct_a = (gm_delta_a / abs(bgm_a)) if abs(bgm_a) > 1e-12 else 0.0
                    uc_a = (
                        ((srev_a - sgm_a) / s_yh_a)
                        if abs(s_yh_a) > 1e-12
                        else (
                            ((brev_a - bgm_a) / b_yh_a)
                            if abs(b_yh_a) > 1e-12
                            else float(arec["unit_cost"])
                        )
                    )
                    arec.update(
                        {
                            "unit_cost": uc_a,
                            "baseline_gross_margin": bgm_a,
                            "scenario_gross_margin": sgm_a,
                            "gross_margin_delta": gm_delta_a,
                            "gross_margin_delta_pct": gm_pct_a,
                        }
                    )

    out_df = pd.DataFrame.from_records(final_rows)

    # Stage 5: Operational Inventory Policy Shift
    if (include_inventory_impact or inventory_params is not None) and not out_df.empty:
        from .diagnostics import compute_inventory_policy

        inv_kw: dict[str, Any] = {
            "lead_time": float(lead_time),
            "lead_time_std": float(lead_time_std),
            "service_level": service_level,
        }
        if inventory_params:
            inv_kw.update(dict(inventory_params))

        inv_by_scen: dict[tuple[str, str, str], dict[str, float]] = {}
        for scen_key, s_grp in out_df.groupby("scenario", sort=False):
            pol_df = compute_inventory_policy(
                s_grp,
                oof_df=combined_oof_df if not combined_oof_df.empty else None,
                **inv_kw,
            )
            for pr in pol_df.to_dict(orient="records"):
                inv_by_scen[(str(scen_key), str(pr["ts_id"]), str(pr["model_type"]))] = {
                    "rop": float(pr["reorder_point"]),
                    "ss": float(pr["safety_stock"]),
                    "q": float(pr["recommended_order_qty"]),
                }

        b_rop_col: list[float] = []
        s_rop_col: list[float] = []
        d_rop_col: list[float] = []
        b_ss_col: list[float] = []
        s_ss_col: list[float] = []
        d_ss_col: list[float] = []
        b_q_col: list[float] = []
        s_q_col: list[float] = []
        d_q_col: list[float] = []

        for r in out_df.to_dict(orient="records"):
            scen_key = str(r["scenario"])
            ti = str(r["ts_id"])
            mt = str(r["model_type"])
            base_inv = inv_by_scen.get(("baseline", ti, mt), {"rop": 0.0, "ss": 0.0, "q": 0.0})
            scen_inv = inv_by_scen.get((scen_key, ti, mt), base_inv)
            b_rop, s_rop = base_inv["rop"], scen_inv["rop"]
            b_ss, s_ss = base_inv["ss"], scen_inv["ss"]
            b_q, s_q = base_inv["q"], scen_inv["q"]
            b_rop_col.append(b_rop)
            s_rop_col.append(s_rop)
            d_rop_col.append(s_rop - b_rop)
            b_ss_col.append(b_ss)
            s_ss_col.append(s_ss)
            d_ss_col.append(s_ss - b_ss)
            b_q_col.append(b_q)
            s_q_col.append(s_q)
            d_q_col.append(s_q - b_q)

        out_df["baseline_reorder_point"] = b_rop_col
        out_df["scenario_reorder_point"] = s_rop_col
        out_df["delta_reorder_point"] = d_rop_col
        out_df["reorder_point_delta"] = d_rop_col
        out_df["baseline_safety_stock"] = b_ss_col
        out_df["scenario_safety_stock"] = s_ss_col
        out_df["delta_safety_stock"] = d_ss_col
        out_df["safety_stock_delta"] = d_ss_col
        out_df["baseline_order_qty"] = b_q_col
        out_df["scenario_order_qty"] = s_q_col
        out_df["delta_order_qty"] = d_q_col
        out_df["recommended_order_qty_delta"] = d_q_col

    return out_df


def summarize_scenarios(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Roll up step-level `simulate_scenarios` output into horizon totals per
    ``(scenario, ts_id, model_type)``.
    """
    base_cols = [
        "scenario",
        "ts_id",
        "model_type",
        "n_steps",
        "baseline_units",
        "scenario_units",
        "delta_units",
        "delta_pct",
    ]
    rev_cols = [
        "baseline_revenue",
        "scenario_revenue",
        "revenue_delta",
        "revenue_delta_pct",
        "mean_implied_elasticity",
    ]
    gm_cols = [
        "baseline_gross_margin",
        "scenario_gross_margin",
        "gross_margin_delta",
        "gross_margin_delta_pct",
    ]
    inv_cols = [
        "baseline_reorder_point",
        "scenario_reorder_point",
        "delta_reorder_point",
        "reorder_point_delta",
        "baseline_safety_stock",
        "scenario_safety_stock",
        "delta_safety_stock",
        "safety_stock_delta",
        "baseline_order_qty",
        "scenario_order_qty",
        "delta_order_qty",
        "recommended_order_qty_delta",
    ]
    if sim_df is None or sim_df.empty:
        return pd.DataFrame(columns=base_cols)

    has_rev = "scenario_revenue" in sim_df.columns
    has_gm = "scenario_gross_margin" in sim_df.columns
    has_inv = "scenario_reorder_point" in sim_df.columns
    records: list[dict[str, Any]] = []
    for (scen, tid, mtype), grp in sim_df.groupby(["scenario", "ts_id", "model_type"], sort=False):
        b_u = float(pd.to_numeric(grp["baseline_yhat"], errors="coerce").sum())
        s_u = float(pd.to_numeric(grp["yhat"], errors="coerce").sum())
        d_u = s_u - b_u
        d_pct = (d_u / abs(b_u)) if abs(b_u) > 1e-12 else 0.0
        rec: dict[str, Any] = {
            "scenario": str(scen),
            "ts_id": str(tid),
            "model_type": str(mtype),
            "n_steps": len(grp),
            "baseline_units": b_u,
            "scenario_units": s_u,
            "delta_units": d_u,
            "delta_pct": d_pct,
        }
        if has_rev:
            b_r = float(pd.to_numeric(grp["baseline_revenue"], errors="coerce").sum())
            s_r = float(pd.to_numeric(grp["scenario_revenue"], errors="coerce").sum())
            d_r = s_r - b_r
            r_pct = (d_r / abs(b_r)) if abs(b_r) > 1e-12 else 0.0
            el_vals = pd.to_numeric(grp["implied_elasticity"], errors="coerce").dropna()
            finite_el = el_vals[np.isfinite(el_vals)]
            rec.update(
                {
                    "baseline_revenue": b_r,
                    "scenario_revenue": s_r,
                    "revenue_delta": d_r,
                    "revenue_delta_pct": r_pct,
                    "mean_implied_elasticity": (
                        float(finite_el.mean()) if not finite_el.empty else float("nan")
                    ),
                }
            )
        if has_gm:
            b_gm = float(pd.to_numeric(grp["baseline_gross_margin"], errors="coerce").sum())
            s_gm = float(pd.to_numeric(grp["scenario_gross_margin"], errors="coerce").sum())
            d_gm = s_gm - b_gm
            gm_pct = (d_gm / abs(b_gm)) if abs(b_gm) > 1e-12 else 0.0
            rec.update(
                {
                    "baseline_gross_margin": b_gm,
                    "scenario_gross_margin": s_gm,
                    "gross_margin_delta": d_gm,
                    "gross_margin_delta_pct": gm_pct,
                }
            )
        if has_inv:
            first_row = grp.iloc[0]
            for ic in inv_cols:
                if ic in first_row:
                    rec[ic] = float(first_row[ic])
                elif ic == "reorder_point_delta" and "delta_reorder_point" in first_row:
                    rec[ic] = float(first_row["delta_reorder_point"])
                elif ic == "safety_stock_delta" and "delta_safety_stock" in first_row:
                    rec[ic] = float(first_row["delta_safety_stock"])
                elif ic == "recommended_order_qty_delta" and "delta_order_qty" in first_row:
                    rec[ic] = float(first_row["delta_order_qty"])
                else:
                    rec[ic] = 0.0
        records.append(rec)

    cols = (
        base_cols
        + (rev_cols if has_rev else [])
        + (gm_cols if has_gm else [])
        + (inv_cols if has_inv else [])
    )
    return pd.DataFrame.from_records(records, columns=cols)


def compare_scenario_margin_curves(
    sim_df: pd.DataFrame,
    *,
    ts_id: str | None = None,
    model_type: str | None = None,
) -> pd.DataFrame:
    """Compare counterfactual scenarios ranked by gross-margin delta (or revenue/unit delta).

    Wraps :func:`summarize_scenarios` and sorts candidate scenarios descending by
    ``gross_margin_delta`` (when ``unit_cost`` was supplied), ``revenue_delta`` (when price
    is present), or ``delta_units``.
    """
    summary = summarize_scenarios(sim_df)
    if summary.empty:
        return summary
    if ts_id is not None and "ts_id" in summary.columns:
        summary = summary[summary["ts_id"].astype(str) == str(ts_id)].copy()
    if model_type is not None and "model_type" in summary.columns:
        summary = summary[summary["model_type"].astype(str) == str(model_type)].copy()
    sort_col = (
        "gross_margin_delta"
        if "gross_margin_delta" in summary.columns
        else ("revenue_delta" if "revenue_delta" in summary.columns else "delta_units")
    )
    if sort_col in summary.columns:
        summary = summary.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return summary


def plot_scenarios(
    sim_df: pd.DataFrame,
    *,
    ts_id: str | None = None,
    model_type: str | None = None,
    show_revenue: bool = False,
    show_gross_margin: bool = False,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot baseline vs. counterfactual scenario forecast trajectories."""
    from .errors import require_extra

    require_extra("notebook", purpose="Plotting scenario simulations")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if sim_df is None or sim_df.empty:
        ax.set_title(title or "Scenario Simulation (empty)")
        return ax

    sub = sim_df.copy()
    chosen_ts = ts_id or str(sub["ts_id"].iloc[0])
    sub = sub[sub["ts_id"].astype(str) == str(chosen_ts)]
    if sub.empty:
        ax.set_title(title or f"Scenario Simulation ({chosen_ts}: no rows)")
        return ax
    chosen_model = model_type or str(sub["model_type"].iloc[0])
    sub = sub[sub["model_type"].astype(str) == str(chosen_model)]

    if show_gross_margin and "scenario_gross_margin" in sub.columns:
        val_col = "scenario_gross_margin"
        y_label = "Gross Margin"
    elif show_revenue and "scenario_revenue" in sub.columns:
        val_col = "scenario_revenue"
        y_label = "Revenue"
    else:
        val_col = "yhat"
        y_label = "Forecast Units (yhat)"

    for scen, grp in sub.groupby("scenario", sort=False):
        ordered = grp.sort_values("ds")
        style = "--" if scen == "baseline" else "-"
        lw = 2.2 if scen == "baseline" else 1.8
        ax.plot(ordered["ds"], ordered[val_col], linestyle=style, linewidth=lw, label=str(scen))
        if val_col == "yhat" and scen == "baseline":
            ax.fill_between(
                ordered["ds"],
                ordered["yhat_lower"],
                ordered["yhat_upper"],
                alpha=0.15,
                label="baseline 80% PI",
            )

    ax.set_xlabel("Forecast Date")
    ax.set_ylabel(y_label)
    ax.set_title(title or f"What-If Scenarios — {chosen_ts} ({chosen_model})")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.25)
    return ax
