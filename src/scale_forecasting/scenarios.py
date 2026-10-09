"""Counterfactual "What-If" scenario & price-elasticity simulator — pure, zero GCP imports.

Allows planners and AI agents to perturb future exogenous covariates (promotions, pricing, weather,
macro indices, capacity caps) across the entire forecast horizon or within specific date windows
(e.g., a 2-week promotion followed by post-promo dip) and compare baseline vs. counterfactual unit
volume, prediction intervals, and dollar revenue (`price * yhat`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

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
from .config import RunConfig, corrected_arm_for
from .features import (
    build_features,
    build_future_features,
    fit_transform_lambda,
    regularize_series_grid,
)
from .models import get_model
from .models.base_model import ModelContext
from .seasonality import seasonal_period

if TYPE_CHECKING:
    pass

__all__ = [
    "apply_scenario_shocks",
    "plot_scenarios",
    "simulate_scenarios",
    "summarize_scenarios",
]


def _parse_single_shock(spec: Any) -> list[dict[str, Any]]:
    """Normalize a user shock specification into a list of ``{op, value, start_date, end_date}``."""
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
) -> str | None:
    """Resolve ``price_col`` when explicitly provided or when a price column is shocked."""
    if explicit_price_col is not None:
        return explicit_price_col
    shocked_cols = {str(k) for shocks in scenarios.values() for k in shocks}
    for cand in ("price", "unit_price", "price_index", "avg_price"):
        if cand in shocked_cols or cand in cov_cols:
            return cand
    return None


def simulate_scenarios(
    data: pd.DataFrame,
    cfg: RunConfig,
    scenarios: Mapping[str, Mapping[str, Any]],
    *,
    models: Sequence[str] | None = None,
    future_covariates_df: pd.DataFrame | None = None,
    price_col: str | None = None,
) -> pd.DataFrame:
    """Run counterfactual covariate scenarios across ``data`` and ``models`` (pure, offline).

    Fits each candidate model once on each series' historical trajectory, then evaluates both the
    unshocked ``"baseline"`` future covariate trajectory and each named scenario in ``scenarios``.
    When ``price_col`` is provided (or automatically detected from ``"price"`` / ``"price_index"``),
    computes ``baseline_revenue``, ``scenario_revenue``, ``revenue_delta``, ``revenue_delta_pct``,
    and ``implied_elasticity`` ($\\frac{\\%\\Delta \\hat{y}}{\\%\\Delta p}$).
    """
    shock_cols = sorted({str(k) for shocks in scenarios.values() for k in shocks})
    known_covs = list(dict.fromkeys([*cfg.features.known_future_covariates, *shock_cols]))
    if price_col and price_col not in known_covs:
        known_covs.append(price_col)

    work_cfg = cfg
    if set(known_covs) != set(cfg.features.known_future_covariates):
        work_cfg = cfg.model_copy(
            update={"features": cfg.features.model_copy(update={"future_covariates": known_covs})}
        )

    eff_price_col = _infer_price_col(price_col, known_covs, scenarios)

    candidate_models = list(models) if models is not None else list(work_cfg.models)
    py_models = [
        m
        for m in candidate_models
        if get_model(m).runtime == "python" and get_model(m).is_available()
    ]
    if not py_models:
        py_models = ["naive_seasonal"]

    d = work_cfg.data
    panel = data.copy()
    if d.ts_id_col not in panel.columns:
        panel[d.ts_id_col] = "series_0"
    for c in known_covs:
        if c not in panel.columns:
            panel[c] = 1.0 if c == eff_price_col else 0.0

    all_scenarios: dict[str, Mapping[str, Any]] = {"baseline": {}, **dict(scenarios)}
    target_qs = work_cfg.output.effective_quantiles
    fallback_arm = corrected_arm_for(work_cfg.backtest.decision_metric)
    horizon = d.horizon

    rows: list[dict[str, Any]] = []
    for raw_tid, s_group in panel.groupby(d.ts_id_col, sort=True):
        tid = str(raw_tid)
        s_clean = regularize_series_grid(s_group, work_cfg)
        raw_target = s_clean[d.target_col].astype(float)
        lam = fit_transform_lambda(raw_target, work_cfg.features.transform)
        base_fut = _build_baseline_future_frame(
            s_clean, work_cfg, known_covs, future_covariates_df, tid
        )
        base_prices = (
            pd.to_numeric(base_fut[eff_price_col], errors="coerce").to_numpy(dtype=float)
            if eff_price_col and eff_price_col in base_fut.columns
            else None
        )

        for mname in py_models:
            m_cls = get_model(mname)
            m_params: dict[str, Any] = dict(work_cfg.model_params.get(mname, {}))
            ctx = ModelContext(
                freq=d.freq,
                horizon=work_cfg.max_horizon,
                transform=work_cfg.features.transform,
                transform_lambda=lam,
            )
            y_hist, X_hist = build_features(
                s_clean, work_cfg, lam, m_cls.lags_covariates_internally
            )
            est = m_cls(m_params, ctx)
            est.fit(y_hist, X_hist)

            cal: Calibration | None = None
            oof_df: pd.DataFrame | None = None
            if work_cfg.backtest.enabled:
                oof_raw, _, _ = backtest_cell(
                    s_clean,
                    partial(m_cls, m_params, ctx),
                    work_cfg,
                    lam,
                    None,
                    m_cls.lags_covariates_internally,
                )
                oof_df = oof_raw if not oof_raw.empty else None
                if oof_df is not None:
                    cal = calibrate_from_oof(oof_df, target_qs)

            arm, _ = resolve_arm(
                work_cfg.output.point_forecast,
                oof_df,
                work_cfg.backtest.decision_metric,
                fallback_arm,
            )

            baseline_yhat_arr: np.ndarray | None = None
            scen_frames: list[tuple[str, pd.DataFrame, np.ndarray | None]] = []

            for scen_name, shocks in all_scenarios.items():
                fut_cov_df = apply_scenario_shocks(base_fut, shocks, date_col=d.date_col)
                X_fut = build_future_features(
                    y_hist,
                    X_hist,
                    work_cfg,
                    horizon=horizon,
                    future_covariates_df=fut_cov_df,
                )
                raw_pred = est.predict(horizon, X_fut, quantiles=target_qs)
                cal_pred, _ = apply_calibration(raw_pred, cal, arm)
                cal_pred, _ = apply_quantile_and_cost_rules(cal_pred, work_cfg.output)
                if has_output_constraints(work_cfg.output):
                    cal_pred = apply_output_constraints(
                        cal_pred, work_cfg.output, bounds_df=fut_cov_df
                    )
                scen_prices = (
                    pd.to_numeric(fut_cov_df[eff_price_col], errors="coerce").to_numpy(dtype=float)
                    if eff_price_col and eff_price_col in fut_cov_df.columns
                    else None
                )
                if scen_name == "baseline":
                    baseline_yhat_arr = pd.to_numeric(cal_pred["yhat"], errors="coerce").to_numpy(
                        dtype=float
                    )
                scen_frames.append((scen_name, cal_pred, scen_prices))

            assert baseline_yhat_arr is not None
            for scen_name, cal_pred, scen_prices in scen_frames:
                yh = pd.to_numeric(cal_pred["yhat"], errors="coerce").to_numpy(dtype=float)
                yl = pd.to_numeric(cal_pred["yhat_lower"], errors="coerce").to_numpy(dtype=float)
                yu = pd.to_numeric(cal_pred["yhat_upper"], errors="coerce").to_numpy(dtype=float)
                ds_vals = pd.to_datetime(cal_pred["ds"]).tolist()
                for idx in range(len(cal_pred)):
                    b_yh = float(baseline_yhat_arr[idx])
                    s_yh = float(yh[idx])
                    d_yh = s_yh - b_yh
                    d_pct = (d_yh / abs(b_yh)) if abs(b_yh) > 1e-12 else 0.0
                    rec: dict[str, Any] = {
                        "scenario": scen_name,
                        "ts_id": tid,
                        "model_type": mname,
                        "ds": ds_vals[idx],
                        "horizon_step": idx + 1,
                        "yhat": s_yh,
                        "yhat_lower": float(yl[idx]),
                        "yhat_upper": float(yu[idx]),
                        "baseline_yhat": b_yh,
                        "delta_yhat": d_yh,
                        "delta_pct": d_pct,
                    }
                    if base_prices is not None and scen_prices is not None:
                        bp = float(base_prices[idx])
                        sp = float(scen_prices[idx])
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
                    rows.append(rec)

    return pd.DataFrame.from_records(rows)


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
    if sim_df is None or sim_df.empty:
        return pd.DataFrame(columns=base_cols)

    has_rev = "scenario_revenue" in sim_df.columns
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
        records.append(rec)

    cols = base_cols + (rev_cols if has_rev else [])
    return pd.DataFrame.from_records(records, columns=cols)


def plot_scenarios(
    sim_df: pd.DataFrame,
    *,
    ts_id: str | None = None,
    model_type: str | None = None,
    show_revenue: bool = False,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot baseline vs. counterfactual scenario forecast trajectories (or revenue)."""
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

    val_col = "scenario_revenue" if show_revenue and "scenario_revenue" in sub.columns else "yhat"
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
    ax.set_ylabel("Revenue" if val_col == "scenario_revenue" else "Forecast Units (yhat)")
    ax.set_title(title or f"What-If Scenarios — {chosen_ts} ({chosen_model})")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.25)
    return ax
