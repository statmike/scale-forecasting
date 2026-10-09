"""Self-contained interactive HTML & JSON executive forecast report — pure, zero GCP imports.

Generates a single-file, zero-server, offline-capable ``.html`` report (plus optional ``.json``
summary payload) combining:

1. **Executive Summary & KPI Cards** — run identity, series count, model count, winning model,
   headline Forecast Value Added (FVA %), and active exception count.
2. **Model Leaderboard & FVA Waterfall** — full metric table alongside an inline SVG Forecast Value
   Added bar chart with Harvey–Leybourne–Newbold (1997) modified Diebold–Mariano significance
   indicators (``p < 0.05``).
3. **Top-N Exception Triage Queue** — interactive filterable table of flagged
   ``(ts_id, model_type)`` cells across ``worse_than_naive``, ``chronic_bias``, ``realized_drift``,
   ``whipsaw_jump``, and ``capacity_ceiling_hit`` with plain-English explanations.
4. **Lead-Time Horizon Decay & Bias Audit** — inline SVG step-by-step error curve ($h = 1 \\dots H$)
   and Trigg Tracking Signal table.
5. **Interactive Series & Multi-Quantile Fan Explorer** — client-side ``ts_id`` and ``model_type``
   selector rendering an interactive SVG chart of historical actuals, point forecast, $80\\%$
   calibrated interval, and multi-percentile fan (`p10`–`p95`).
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import (
    compute_bias_report,
    compute_exceptions_queue,
    compute_fva,
    compute_horizon_decay,
)

__all__ = [
    "build_report_payload",
    "export_report",
    "render_html_report",
]


def _clean_json_val(val: Any) -> Any:
    """Convert numpy/pandas scalars and NaNs into JSON-safe Python primitives."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        f = float(val)
        return round(f, 6) if math.isfinite(f) else None
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(val).strftime("%Y-%m-%d %H:%M:%S")).removesuffix(" 00:00:00")
    return str(val)


def _df_to_records(df: pd.DataFrame | None, max_rows: int = 200) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    sub = df.head(max_rows)
    out: list[dict[str, Any]] = []
    for row in sub.to_dict(orient="records"):
        out.append({str(k): _clean_json_val(v) for k, v in row.items()})
    return out


def _render_fva_svg(fva_records: list[dict[str, Any]]) -> str:
    """Render a crisp, standalone inline SVG horizontal bar chart for the FVA ladder."""
    if not fva_records:
        return '<p class="muted">No backtest out-of-fold rows available to compute FVA.</p>'

    width = 760
    row_h = 34
    top_pad = 30
    bot_pad = 36
    left_pad = 210
    right_pad = 90
    n = len(fva_records)
    height = top_pad + n * row_h + bot_pad
    plot_w = width - left_pad - right_pad

    pcts = [float(r.get("fva_pct") or 0.0) * 100.0 for r in fva_records]
    max_abs = max([abs(p) for p in pcts] + [10.0]) * 1.15
    zero_x = left_pad + plot_w * 0.5

    lines = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
        'aria-label="Forecast Value Added Waterfall">',
        f'<line x1="{zero_x:.1f}" y1="{top_pad - 8}" x2="{zero_x:.1f}" '
        f'y2="{height - bot_pad + 6}" stroke="#475569" stroke-width="1.5" '
        'stroke-dasharray="4,3" />',
    ]
    for idx, rec in enumerate(fva_records):
        y = top_pad + idx * row_h
        pct = float(rec.get("fva_pct") or 0.0) * 100.0
        sig = bool(rec.get("significant_at_05"))
        mtype = html.escape(str(rec.get("model_type", "")))
        stage = html.escape(str(rec.get("fva_stage", "")))
        label = f"{mtype} ({stage})" + (" *" if sig else "")

        bar_w = abs(pct) / max_abs * (plot_w * 0.5)
        bar_x = zero_x if pct >= 0 else zero_x - bar_w
        color = "#16a34a" if pct > 0 else ("#64748b" if pct == 0 else "#dc2626")

        lines.append(
            f'<text x="{left_pad - 10}" y="{y + 18}" text-anchor="end" '
            f'class="svg-label">{label}</text>'
        )
        lines.append(
            f'<rect x="{bar_x:.1f}" y="{y + 5}" width="{max(bar_w, 2.0):.1f}" '
            f'height="{row_h - 12}" rx="3" fill="{color}" opacity="0.88" />'
        )
        val_txt = f"{pct:+.1f}%"
        txt_x = (bar_x + bar_w + 6) if pct >= 0 else (bar_x - 6)
        anchor = "start" if pct >= 0 else "end"
        lines.append(
            f'<text x="{txt_x:.1f}" y="{y + 18}" text-anchor="{anchor}" '
            f'class="svg-val">{val_txt}</text>'
        )

    lines.append(
        f'<text x="{zero_x:.1f}" y="{height - 10}" text-anchor="middle" class="svg-axis">'
        "FVA % Accuracy Gain vs. Baseline (* p &lt; 0.05 HLN Diebold-Mariano)</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines)


def _render_decay_svg(decay_records: list[dict[str, Any]]) -> str:
    """Render an inline SVG multi-line chart for Lead-Time Horizon Accuracy Decay."""
    if not decay_records:
        return '<p class="muted">No horizon step data available.</p>'

    width, height = 760, 290
    left_pad, right_pad, top_pad, bot_pad = 64, 150, 24, 42
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bot_pad

    by_model: dict[str, list[tuple[int, float]]] = {}
    all_steps: list[int] = []
    all_vals: list[float] = []
    metric_name = str(decay_records[0].get("metric", "wape")).upper()

    for r in decay_records:
        m = str(r.get("model_type", "model"))
        s = int(r.get("horizon_step") or 1)
        v = r.get("metric_value")
        if v is None:
            continue
        vf = float(v)
        by_model.setdefault(m, []).append((s, vf))
        all_steps.append(s)
        all_vals.append(vf)

    if not all_vals:
        return '<p class="muted">No finite horizon step metrics available.</p>'

    min_s, max_s = min(all_steps), max(all_steps)
    min_v, max_v = 0.0, max(all_vals) * 1.12 or 1.0
    palette = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#0891b2", "#4f46e5"]

    def _sx(s: int) -> float:
        if max_s <= min_s:
            return left_pad + plot_w * 0.5
        return left_pad + (s - min_s) / (max_s - min_s) * plot_w

    def _sy(v: float) -> float:
        return top_pad + plot_h - (v - min_v) / (max_v - min_v) * plot_h

    lines = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
        'aria-label="Horizon Accuracy Decay">',
        f'<rect x="{left_pad}" y="{top_pad}" width="{plot_w}" height="{plot_h}" '
        'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" />',
    ]
    for idx, (mtype, pts) in enumerate(by_model.items()):
        pts_sorted = sorted(pts, key=lambda x: x[0])
        color = palette[idx % len(palette)]
        poly = " ".join(f"{_sx(s):.1f},{_sy(v):.1f}" for s, v in pts_sorted)
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{poly}" />'
        )
        for s, v in pts_sorted:
            lines.append(f'<circle cx="{_sx(s):.1f}" cy="{_sy(v):.1f}" r="3" fill="{color}" />')
        leg_y = top_pad + 16 + idx * 20
        lines.append(
            f'<line x1="{left_pad + plot_w + 14}" y1="{leg_y - 4}" '
            f'x2="{left_pad + plot_w + 30}" y2="{leg_y - 4}" stroke="{color}" stroke-width="2.5"/>'
            f'<text x="{left_pad + plot_w + 36}" y="{leg_y}" class="svg-label">'
            f"{html.escape(mtype)}</text>"
        )

    lines.append(
        f'<text x="{left_pad + plot_w * 0.5:.1f}" y="{height - 10}" text-anchor="middle" '
        f'class="svg-axis">Forecast Step h ({min_s} .. {max_s}) — {html.escape(metric_name)}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def _build_explorer_series_map(
    predictions_df: pd.DataFrame | None,
    history_df: pd.DataFrame | None,
    *,
    max_series: int = 25,
    max_history_points: int = 90,
) -> dict[str, Any]:
    """Assemble a compact JSON dictionary of historical + forecast trajectories for the explorer."""
    if predictions_df is None or predictions_df.empty:
        return {"series": {}, "ts_ids": [], "models": []}

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

    ts_ids = list(dict.fromkeys(pred["ts_id"].astype(str)))[:max_series]
    models = list(dict.fromkeys(pred["model_type"].astype(str)))

    hist_map: dict[str, list[dict[str, Any]]] = {}
    if history_df is not None and not history_df.empty:
        h_df = history_df.copy()
        h_ts = "ts_id" if "ts_id" in h_df.columns else None
        h_date = "ds" if "ds" in h_df.columns else ("date" if "date" in h_df.columns else None)
        h_y = "y" if "y" in h_df.columns else ("y_true" if "y_true" in h_df.columns else None)
        if h_date and h_y:
            if h_ts is None:
                h_df["ts_id"] = ts_ids[0] if ts_ids else "series_0"
                h_ts = "ts_id"
            for tid, grp in h_df.groupby(h_ts, sort=False):
                tid_s = str(tid)
                if tid_s not in ts_ids:
                    continue
                tail = grp.sort_values(h_date).tail(max_history_points)
                hist_map[tid_s] = [
                    {
                        "ds": _clean_json_val(r[h_date]),
                        "y": _clean_json_val(r[h_y]),
                    }
                    for r in tail.to_dict(orient="records")
                ]

    series_payload: dict[str, Any] = {}
    pred_sub = pred[pred["ts_id"].astype(str).isin(ts_ids)]
    for (tid_p, mtype_p), grp in pred_sub.groupby(["ts_id", "model_type"], sort=False):
        tid_s, m_s = str(tid_p), str(mtype_p)
        ordered = grp.sort_values(p_date)
        f_rows: list[dict[str, Any]] = []
        for r in ordered.to_dict(orient="records"):
            q_val = r.get("quantiles")
            if isinstance(q_val, str) and q_val.startswith("{"):
                try:
                    q_parsed = json.loads(q_val)
                except Exception:  # noqa: BLE001 - malformed JSON falls back to empty dict
                    q_parsed = {}
            elif isinstance(q_val, dict):
                q_parsed = {str(k): _clean_json_val(v) for k, v in q_val.items()}
            else:
                q_parsed = {}
            f_rows.append(
                {
                    "ds": _clean_json_val(r.get(p_date)),
                    "yhat": _clean_json_val(r.get(p_yhat)),
                    "lower": _clean_json_val(r.get(p_lo)) if p_lo else None,
                    "upper": _clean_json_val(r.get(p_hi)) if p_hi else None,
                    "quantiles": q_parsed,
                }
            )
        series_payload.setdefault(tid_s, {"history": hist_map.get(tid_s, []), "models": {}})[
            "models"
        ][m_s] = f_rows

    return {"series": series_payload, "ts_ids": ts_ids, "models": models}


def build_report_payload(
    *,
    title: str = "Scale-Forecasting Executive Workbench Report",
    run_id: str = "local-workbench",
    leaderboard_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    predictions_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
    fva_df: pd.DataFrame | None = None,
    horizon_decay_df: pd.DataFrame | None = None,
    bias_df: pd.DataFrame | None = None,
    exceptions_df: pd.DataFrame | None = None,
    realized_df: pd.DataFrame | None = None,
    stability_df: pd.DataFrame | None = None,
    metric: str = "wape",
) -> dict[str, Any]:
    """Assemble the structured dictionary backing both HTML and JSON report exports."""
    if fva_df is None and oof_df is not None and not oof_df.empty:
        fva_df = compute_fva(oof_df, metric=metric, series_df=history_df)
    if horizon_decay_df is None and oof_df is not None and not oof_df.empty:
        horizon_decay_df = compute_horizon_decay(oof_df, metric=metric)
    if bias_df is None and oof_df is not None and not oof_df.empty:
        bias_df = compute_bias_report(oof_df)
    if exceptions_df is None:
        exceptions_df = compute_exceptions_queue(
            oof_df=oof_df,
            predictions_df=predictions_df,
            realized_df=realized_df,
            stability_df=stability_df,
            metric=metric,
        )

    fva_recs = _df_to_records(fva_df)
    lb_recs = _df_to_records(leaderboard_df)
    exc_recs = _df_to_records(exceptions_df)
    decay_recs = _df_to_records(horizon_decay_df)
    bias_recs = _df_to_records(bias_df)
    real_recs = _df_to_records(realized_df)
    stab_recs = _df_to_records(stability_df)
    explorer = _build_explorer_series_map(predictions_df, history_df)

    best_model = "n/a"
    best_fva_pct: float | None = None
    if fva_recs:
        best_row = min(
            fva_recs,
            key=lambda r: float(r["mean_metric"]) if r.get("mean_metric") is not None else 1e9,
        )
        best_model = str(best_row.get("model_type", "n/a"))
        if best_row.get("fva_pct") is not None:
            best_fva_pct = round(float(best_row["fva_pct"]) * 100.0, 2)
    elif lb_recs:
        first_row = lb_recs[0]
        best_model = str(first_row.get("model_type") or first_row.get("model") or "n/a")

    return {
        "title": title,
        "run_id": run_id,
        "metric": metric,
        "kpis": {
            "n_series": len(explorer["ts_ids"]),
            "n_models": len(explorer["models"]) or len(lb_recs) or len(fva_recs),
            "champion_model": best_model,
            "best_fva_pct": best_fva_pct,
            "n_exceptions": len(exc_recs),
        },
        "leaderboard": lb_recs,
        "fva": fva_recs,
        "exceptions": exc_recs,
        "horizon_decay": decay_recs,
        "bias_report": bias_recs,
        "realized_accuracy": real_recs,
        "revision_stability": stab_recs,
        "explorer": explorer,
    }


def _html_table(records: list[dict[str, Any]], *, table_id: str = "", empty_msg: str = "") -> str:
    """Render a clean HTML table from a list of flat dictionaries."""
    if not records:
        return f'<p class="muted">{html.escape(empty_msg or "No rows available.")}</p>'
    cols = list(records[0].keys())
    id_attr = f' id="{html.escape(table_id)}"' if table_id else ""
    thead = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    tbody_rows: list[str] = []
    for r in records:
        flags = html.escape(str(r.get("exception_flags", "")))
        tr_attr = f' data-flags="{flags}"' if flags else ""
        cells: list[str] = []
        for c in cols:
            v = r.get(c)
            if v is None:
                txt = "—"
            elif isinstance(v, float):
                txt = f"{v:.4f}"
            else:
                txt = html.escape(str(v))
            cells.append(f"<td>{txt}</td>")
        tbody_rows.append(f"<tr{tr_attr}>{''.join(cells)}</tr>")
    return (
        f'<div class="table-wrap"><table class="data-table"{id_attr}>'
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(tbody_rows)}</tbody></table></div>"
    )


def render_html_report(
    *,
    title: str = "Scale-Forecasting Executive Workbench Report",
    run_id: str = "local-workbench",
    leaderboard_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    predictions_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
    fva_df: pd.DataFrame | None = None,
    horizon_decay_df: pd.DataFrame | None = None,
    bias_df: pd.DataFrame | None = None,
    exceptions_df: pd.DataFrame | None = None,
    realized_df: pd.DataFrame | None = None,
    stability_df: pd.DataFrame | None = None,
    metric: str = "wape",
) -> str:
    """Render a complete, self-contained single-file interactive HTML report."""
    payload = build_report_payload(
        title=title,
        run_id=run_id,
        leaderboard_df=leaderboard_df,
        oof_df=oof_df,
        predictions_df=predictions_df,
        history_df=history_df,
        fva_df=fva_df,
        horizon_decay_df=horizon_decay_df,
        bias_df=bias_df,
        exceptions_df=exceptions_df,
        realized_df=realized_df,
        stability_df=stability_df,
        metric=metric,
    )
    kpis = payload["kpis"]
    fva_svg = _render_fva_svg(payload["fva"])
    decay_svg = _render_decay_svg(payload["horizon_decay"])
    lb_html = _html_table(
        payload["leaderboard"] or payload["fva"],
        empty_msg="No leaderboard rows available.",
    )
    fva_html = _html_table(payload["fva"], empty_msg="No backtest OOF rows available for FVA.")
    exc_html = _html_table(
        payload["exceptions"],
        table_id="exceptions-table",
        empty_msg="Zero exceptions flagged — all series passed operational checks.",
    )
    decay_html = _html_table(payload["horizon_decay"], empty_msg="No horizon decay rows.")
    bias_html = _html_table(payload["bias_report"], empty_msg="No bias report rows.")

    fva_kpi_str = f"{kpis['best_fva_pct']:+.1f}%" if kpis["best_fva_pct"] is not None else "n/a"
    explorer_json = json.dumps(payload["explorer"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)} — {html.escape(run_id)}</title>
<style>
  :root {{
    --bg: #f8fafc;
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --primary: #2563eb;
    --success: #16a34a;
    --warn: #d97706;
    --danger: #dc2626;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px 32px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}
  header {{
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 20px;
  }}
  h1 {{ margin: 0; font-size: 1.45rem; }}
  .badge-run {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #e0e7ff; color: #1e40af; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem;
  }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px; margin-bottom: 22px;
  }}
  .kpi-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }}
  .kpi-label {{ font-size: 0.78rem; text-transform: uppercase; color: var(--muted); }}
  .kpi-val {{ font-size: 1.35rem; font-weight: 700; margin-top: 4px; }}
  .tabs {{
    display: flex; gap: 8px; border-bottom: 1px solid var(--border); margin-bottom: 20px;
    flex-wrap: wrap;
  }}
  .tab-btn {{
    background: none; border: none; border-bottom: 3px solid transparent;
    padding: 10px 16px; cursor: pointer; font-weight: 600; color: var(--muted);
    font-size: 0.92rem;
  }}
  .tab-btn.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px; margin-bottom: 20px;
  }}
  .card h2 {{ margin-top: 0; font-size: 1.1rem; }}
  .table-wrap {{ overflow-x: auto; }}
  table.data-table {{
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
  }}
  table.data-table th, table.data-table td {{
    padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left;
  }}
  table.data-table th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
  .chart-svg {{ width: 100%; max-width: 820px; height: auto; display: block; margin: 8px 0; }}
  .svg-label {{ font-size: 12px; fill: #1e293b; }}
  .svg-val {{ font-size: 11px; font-weight: 600; fill: #0f172a; }}
  .svg-axis {{ font-size: 12px; fill: #475569; font-weight: 600; }}
  .filter-bar {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
  .pill {{
    border: 1px solid var(--border); background: #f8fafc; border-radius: 999px;
    padding: 4px 12px; font-size: 0.8rem; cursor: pointer;
  }}
  .pill.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
  .controls {{
    display: flex; gap: 14px; align-items: center; margin-bottom: 14px; flex-wrap: wrap;
  }}
  select {{
    padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); font-size: 0.9rem;
  }}
  .muted {{ color: var(--muted); font-style: italic; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <span class="badge-run">run_id: {html.escape(run_id)}</span>
</header>

<section class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Series Evaluated</div>
    <div class="kpi-val">{kpis["n_series"]}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Models &amp; Ensembles</div>
    <div class="kpi-val">{kpis["n_models"]}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Champion Model ({html.escape(metric.upper())})</div>
    <div class="kpi-val">{html.escape(str(kpis["champion_model"]))}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Headline FVA vs. Baseline</div>
    <div class="kpi-val">{fva_kpi_str}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Flagged Exceptions</div>
    <div class="kpi-val">{kpis["n_exceptions"]}</div>
  </div>
</section>

<nav class="tabs">
  <button class="tab-btn active" onclick="switchTab('tab-fva', this)">
    Leaderboard &amp; FVA
  </button>
  <button class="tab-btn" onclick="switchTab('tab-exceptions', this)">
    Exception Triage ({kpis["n_exceptions"]})
  </button>
  <button class="tab-btn" onclick="switchTab('tab-decay', this)">
    Horizon Decay &amp; Bias
  </button>
  <button class="tab-btn" onclick="switchTab('tab-explorer', this)">
    Interactive Series &amp; Fan Explorer
  </button>
</nav>

<section id="tab-fva" class="tab-panel active">
  <div class="card">
    <h2>Forecast Value Added (FVA) Waterfall &amp; Significance</h2>
    {fva_svg}
    {fva_html}
  </div>
  <div class="card">
    <h2>Model Leaderboard</h2>
    {lb_html}
  </div>
</section>

<section id="tab-exceptions" class="tab-panel">
  <div class="card">
    <h2>Top-N Exception Triage Queue</h2>
    <div class="filter-bar">
      <button class="pill active" onclick="filterExceptions('all', this)">All</button>
      <button class="pill" onclick="filterExceptions('worse_than_naive', this)">
        Worse Than Naive
      </button>
      <button class="pill" onclick="filterExceptions('chronic_bias', this)">
        Chronic Bias
      </button>
      <button class="pill" onclick="filterExceptions('realized_drift', this)">
        Realized Drift
      </button>
      <button class="pill" onclick="filterExceptions('whipsaw_jump', this)">
        Whipsaw Jump
      </button>
      <button class="pill" onclick="filterExceptions('capacity_ceiling_hit', this)">
        Capacity Hit
      </button>
    </div>
    {exc_html}
  </div>
</section>

<section id="tab-decay" class="tab-panel">
  <div class="card">
    <h2>Lead-Time Horizon Accuracy Decay</h2>
    {decay_svg}
    {decay_html}
  </div>
  <div class="card">
    <h2>Directional Bias &amp; Trigg Tracking Signal Audit</h2>
    {bias_html}
  </div>
</section>

<section id="tab-explorer" class="tab-panel">
  <div class="card">
    <h2>Interactive Series &amp; Multi-Quantile Fan Explorer</h2>
    <div class="controls">
      <label>Series (<code>ts_id</code>):
        <select id="sel-ts" onchange="renderFanChart()"></select>
      </label>
      <label>Model (<code>model_type</code>):
        <select id="sel-model" onchange="renderFanChart()"></select>
      </label>
    </div>
    <div id="fan-chart-container"></div>
  </div>
</section>

<script>
const EXPLORER_DATA = {explorer_json};

function switchTab(panelId, btn) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(panelId).classList.add('active');
  btn.classList.add('active');
}}

function filterExceptions(flag, btn) {{
  document.querySelectorAll('.filter-bar .pill').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  const tbl = document.getElementById('exceptions-table');
  if (!tbl) return;
  tbl.querySelectorAll('tbody tr').forEach(tr => {{
    const flags = tr.getAttribute('data-flags') || '';
    tr.style.display = (flag === 'all' || flags.indexOf(flag) !== -1) ? '' : 'none';
  }});
}}

function initExplorer() {{
  const tsSel = document.getElementById('sel-ts');
  const modSel = document.getElementById('sel-model');
  if (!tsSel || !modSel) return;
  (EXPLORER_DATA.ts_ids || []).forEach(t => {{
    const o = document.createElement('option');
    o.value = t; o.textContent = t; tsSel.appendChild(o);
  }});
  (EXPLORER_DATA.models || []).forEach(m => {{
    const o = document.createElement('option');
    o.value = m; o.textContent = m; modSel.appendChild(o);
  }});
  renderFanChart();
}}

function renderFanChart() {{
  const box = document.getElementById('fan-chart-container');
  const tsSel = document.getElementById('sel-ts');
  const modSel = document.getElementById('sel-model');
  if (!box || !tsSel || !modSel) return;
  const tid = tsSel.value;
  const mtype = modSel.value;
  const sObj = (EXPLORER_DATA.series || {{}})[tid];
  if (!sObj || !sObj.models || !sObj.models[mtype]) {{
    box.innerHTML = '<p class="muted">No forecast trajectory available for this selection.</p>';
    return;
  }}
  const hist = sObj.history || [];
  const fc = sObj.models[mtype] || [];
  const allY = [];
  hist.forEach(p => {{ if (p.y !== null && p.y !== undefined) allY.push(p.y); }});
  fc.forEach(p => {{
    if (p.yhat !== null) allY.push(p.yhat);
    if (p.lower !== null) allY.push(p.lower);
    if (p.upper !== null) allY.push(p.upper);
    const q = p.quantiles || {{}};
    if (q.p10 !== undefined && q.p10 !== null) allY.push(q.p10);
    if (q.p90 !== undefined && q.p90 !== null) allY.push(q.p90);
    if (q.p95 !== undefined && q.p95 !== null) allY.push(q.p95);
  }});
  if (!allY.length) {{
    box.innerHTML = '<p class="muted">Empty series coordinates.</p>';
    return;
  }}
  const W = 780, H = 310, LP = 58, RP = 30, TP = 22, BP = 38;
  const PW = W - LP - RP, PH = H - TP - BP;
  const totalN = hist.length + fc.length;
  const minY = Math.min(...allY), maxY = Math.max(...allY);
  const spanY = (maxY - minY) || 1.0;
  const sx = i => LP + (totalN <= 1 ? PW * 0.5 : (i / (totalN - 1)) * PW);
  const sy = v => TP + PH - ((v - minY) / spanY) * PH;

  let polyBand = '';
  const bandTop = [], bandBot = [];
  fc.forEach((p, idx) => {{
    const x = sx(hist.length + idx);
    const lo = (p.lower !== null && p.lower !== undefined) ? p.lower : p.yhat;
    const hi = (p.upper !== null && p.upper !== undefined) ? p.upper : p.yhat;
    bandTop.push(`${{x.toFixed(1)}},${{sy(hi).toFixed(1)}}`);
    bandBot.unshift(`${{x.toFixed(1)}},${{sy(lo).toFixed(1)}}`);
  }});
  if (bandTop.length) {{
    const pts = bandTop.concat(bandBot).join(' ');
    polyBand = `<polygon points="${{pts}}" fill="#3b82f6" opacity="0.20" />`;
  }}

  const histPts = hist.map((p, idx) => `${{sx(idx).toFixed(1)}},${{sy(p.y).toFixed(1)}}`).join(' ');
  const fcPts = fc.map(
    (p, idx) => `${{sx(hist.length + idx).toFixed(1)}},${{sy(p.yhat).toFixed(1)}}`
  ).join(' ');
  const splitX = sx(Math.max(0, hist.length - 1));
  const sxStr = splitX.toFixed(1);
  const histLine = histPts
    ? `<polyline fill="none" stroke="#0f172a" stroke-width="2" points="${{histPts}}" />`
    : '';
  const fcLine = fcPts
    ? `<polyline fill="none" stroke="#2563eb" stroke-width="2.4" points="${{fcPts}}" />`
    : '';
  const lblTxt = `${{tid}} · ${{mtype}} (min=${{minY.toFixed(1)}}, max=${{maxY.toFixed(1)}})`;

  box.innerHTML = `
    <svg viewBox="0 0 ${{W}} ${{H}}" class="chart-svg">
      <rect x="${{LP}}" y="${{TP}}" width="${{PW}}" height="${{PH}}"
        fill="#f8fafc" stroke="#cbd5e1" />
      <line x1="${{sxStr}}" y1="${{TP}}" x2="${{sxStr}}" y2="${{TP + PH}}"
        stroke="#94a3b8" stroke-dasharray="4,3" />
      ${{polyBand}}
      ${{histLine}}
      ${{fcLine}}
      <text x="${{LP + 8}}" y="${{TP + 16}}" class="svg-label">${{lblTxt}}</text>
    </svg>`;
}}
window.addEventListener('DOMContentLoaded', initExplorer);
</script>
</body>
</html>
"""


def export_report(
    target: Any,
    out_path: str | Path,
    *,
    title: str | None = None,
    metric: str | None = None,
    write_json: bool = False,
) -> Path:
    """Export a self-contained `.html` report (and optional `.json` sidecar) from a `BakeOff`,
    `PlaygroundRun`, `Forecaster`, or dictionary of DataFrames.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    run_id = "local-workbench"
    eff_metric = metric or "wape"
    leaderboard_df: pd.DataFrame | None = None
    oof_df: pd.DataFrame | None = None
    predictions_df: pd.DataFrame | None = None
    history_df: pd.DataFrame | None = None
    fva_df: pd.DataFrame | None = None
    horizon_decay_df: pd.DataFrame | None = None
    bias_df: pd.DataFrame | None = None
    exceptions_df: pd.DataFrame | None = None

    # 1. BakeOff duck-typing
    if hasattr(target, "leaderboard") and hasattr(target, "predictions") and hasattr(target, "oof"):
        leaderboard_df = target.leaderboard
        predictions_df = target.predictions
        oof_df = target.oof
        history_df = getattr(target, "series", None)
        if hasattr(target, "config"):
            run_id = getattr(target.config, "run_name", run_id)
            if metric is None:
                eff_metric = target.config.backtest.decision_metric
    # 2. PlaygroundRun duck-typing
    elif hasattr(target, "result") and hasattr(target, "series"):
        res = target.result
        run_id = getattr(res, "run_id", run_id)
        predictions_df = res.predictions.assign(ts_id=res.ts_id, model_type=res.model_type)
        if res.oof is not None and not res.oof.empty:
            oof_df = res.oof.assign(ts_id=res.ts_id, model_type=res.model_type)
        history_df = target.series
    # 3. Forecaster duck-typing
    elif hasattr(target, "leaderboard_df") and hasattr(target, "fva_df"):
        run_id = getattr(target, "run_id", run_id)
        leaderboard_df = target.leaderboard_df(run_id)
        fva_df = target.fva_df(run_id, metric=metric)
        horizon_decay_df = target.horizon_decay_df(run_id, metric=metric)
        bias_df = target.bias_report_df(run_id)
        predictions_df = target.predictions_df(run_id)
        if hasattr(target, "exceptions_df"):
            exceptions_df = target.exceptions_df(run_id)
    # 4. Mapping of DataFrames
    elif isinstance(target, dict):
        run_id = str(target.get("run_id", run_id))
        leaderboard_df = target.get("leaderboard_df")
        oof_df = target.get("oof_df")
        predictions_df = target.get("predictions_df")
        history_df = target.get("history_df")
        fva_df = target.get("fva_df")
        horizon_decay_df = target.get("horizon_decay_df")
        bias_df = target.get("bias_df")
        exceptions_df = target.get("exceptions_df")

    eff_title = title or f"Scale-Forecasting Executive Report ({run_id})"
    html_str = render_html_report(
        title=eff_title,
        run_id=run_id,
        leaderboard_df=leaderboard_df,
        oof_df=oof_df,
        predictions_df=predictions_df,
        history_df=history_df,
        fva_df=fva_df,
        horizon_decay_df=horizon_decay_df,
        bias_df=bias_df,
        exceptions_df=exceptions_df,
        metric=eff_metric,
    )
    out.write_text(html_str, encoding="utf-8")

    if write_json:
        payload = build_report_payload(
            title=eff_title,
            run_id=run_id,
            leaderboard_df=leaderboard_df,
            oof_df=oof_df,
            predictions_df=predictions_df,
            history_df=history_df,
            fva_df=fva_df,
            horizon_decay_df=horizon_decay_df,
            bias_df=bias_df,
            exceptions_df=exceptions_df,
            metric=eff_metric,
        )
        json_path = out.with_suffix(".json")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m scale_forecasting.report --out report.html``."""
    parser = argparse.ArgumentParser(
        prog="python -m scale_forecasting.report",
        description="Generate a self-contained HTML Forecaster Workbench report.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .html file path.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["naive_seasonal", "theta", "holtwinters"],
        help="Models to include when running an offline bake-off report.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Forecast horizon for offline bake-off report.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=2,
        help="Backtest folds for offline bake-off report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write a companion .json summary payload next to --out.",
    )
    args = parser.parse_args(argv)

    from .playground import bakeoff

    bo = bakeoff(models=list(args.models), horizon=int(args.horizon), n_folds=int(args.n_folds))
    written = export_report(bo, args.out, write_json=bool(args.json))
    print(f"Wrote interactive HTML report to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
