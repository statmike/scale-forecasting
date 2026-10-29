"""Pure-offline natural-language router for in-memory `Forecaster` / `PlaygroundRun` / `BakeOff`.

Allows ``f.ask("Which model won and what was the FVA lift?")`` to work cleanly in pure offline
environments (Path 1 Playground, unit tests, air-gapped CI) with **zero GCP credentials**,
returning the exact same `BQAgentResponse` dataclass (`.answer`, `.sql`, `.df`, `.mode="offline"`)
paired with the canonical BigQuery Golden SQL template from `golden_queries`.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from .client import BQAgentResponse
from .golden_queries import GoldenQuery, build_golden_queries

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fva", ("fva", "value added", "waterfall", "complexity", "naive", "lift")),
    (
        "exceptions",
        (
            "exception",
            "bias",
            "over-forecast",
            "under-forecast",
            "tracking signal",
            "triage",
            "stockout",
            "attention",
            "sbc",
            "regime",
            "error",
        ),
    ),
    (
        "explainability",
        ("driver", "feature", "attribution", "shap", "covariate", "explain", "weight"),
    ),
    (
        "horizon_and_anomalies",
        ("horizon", "decay", "lead time", "step", "coverage", "anomal", "spike", "staleness"),
    ),
    ("forecasts", ("forecast", "yhat", "quantile", "p10", "p50", "p90", "uncertainty", "volume")),
    ("finops_and_realized", ("gpu", "provision", "runtime", "finops", "realized", "actuals")),
    ("leaderboard", ("win", "won", "best", "leaderboard", "rank", "compare", "champion", "wape")),
    ("executive_summary", ("summary", "scorecard", "s&op", "sop", "overview", "recent")),
)


def match_golden_query(
    question: str,
    *,
    project_id: str = "your-gcp-project-id",
    dataset_id: str = "scale_forecasting",
) -> GoldenQuery:
    """Match a natural-language ``question`` to the closest `GoldenQuery` template."""
    q_lower = question.lower()
    queries = build_golden_queries(project_id, dataset_id)

    best_score = -1
    best_query = queries[0]
    q_tokens = {tok.strip("?.,!():;'\"") for tok in q_lower.split() if len(tok) > 2}

    target_category: str | None = None
    for cat, kws in _CATEGORY_KEYWORDS:
        if any(kw in q_lower for kw in kws):
            target_category = cat
            break

    for candidate in queries:
        cand_tokens = {
            tok.strip("?.,!():;'\"") for tok in candidate.question.lower().split() if len(tok) > 2
        }
        overlap = len(q_tokens & cand_tokens) * 3
        if target_category and candidate.category == target_category:
            overlap += 5
        if overlap > best_score:
            best_score = overlap
            best_query = candidate
    return best_query


def _safe_extract_df(target: Any, *names: str) -> pd.DataFrame:
    """Try zero-arg methods or DataFrame attributes on ``target`` in priority order."""
    import pandas as pd

    for name in names:
        attr = getattr(target, name, None)
        if isinstance(attr, pd.DataFrame) and not attr.empty:
            return attr.copy()
        if callable(attr):
            try:
                res = attr()
                if isinstance(res, pd.DataFrame) and not res.empty:
                    return res.copy()
            except Exception:  # noqa: BLE001 - graceful offline fallback across target types
                continue
    # Single-model PlaygroundRun fallback (`target.result` -> CellResult).
    cell_res = getattr(target, "result", None)
    if cell_res is not None:
        if "leaderboard_df" in names or "leaderboard" in names:
            metrics = getattr(cell_res, "metrics", None)
            if isinstance(metrics, dict) and metrics:
                return pd.DataFrame.from_records(
                    [{"model_type": getattr(cell_res, "model_type", "model"), **metrics}]
                )
        if "predictions_df" in names or "predictions" in names:
            preds = getattr(cell_res, "predictions", None)
            if isinstance(preds, pd.DataFrame) and not preds.empty:
                return preds.copy()
    return pd.DataFrame()


def _infer_run_id(target: Any) -> str:
    rid = getattr(target, "run_id", None)
    if isinstance(rid, str) and rid:
        return rid
    cfg = getattr(target, "config", None) or getattr(target, "_config", None)
    if cfg is not None:
        from ..registry.ids import make_run_id

        try:
            return make_run_id(cfg)
        except Exception:
            logger.debug("offline._infer_run_id could not hash config; falling back", exc_info=True)
    return "offline-run"


def build_offline_sop_summary_df(
    target: Any,
    *,
    run_id: str | None = None,
    default_n_series: int = 0,
) -> pd.DataFrame:
    """Compute a 1-row offline Executive S&OP Scorecard DataFrame mirroring
    `v_executive_sop_summary`.
    """
    import pandas as pd

    rid = run_id or _infer_run_id(target)
    cfg = getattr(target, "_config", None) or getattr(target, "config", None)
    run_name = getattr(cfg, "run_name", rid) if cfg is not None else rid
    lb = _safe_extract_df(target, "leaderboard_df", "leaderboard")
    preds = _safe_extract_df(target, "predictions_df", "predictions")
    model_col = (
        "model_type" if "model_type" in lb.columns else ("model" if "model" in lb.columns else None)
    )
    champ_model = str(lb.iloc[0][model_col]) if not lb.empty and model_col is not None else None
    champ_wape = (
        float(lb.iloc[0]["wape"])
        if not lb.empty and "wape" in lb.columns and pd.notna(lb.iloc[0]["wape"])
        else None
    )
    champ_mase = (
        float(lb.iloc[0]["mase"])
        if not lb.empty and "mase" in lb.columns and pd.notna(lb.iloc[0]["mase"])
        else None
    )
    champ_cov = (
        float(lb.iloc[0]["coverage"])
        if not lb.empty and "coverage" in lb.columns and pd.notna(lb.iloc[0]["coverage"])
        else None
    )
    champ_preds = (
        preds[preds["model_type"].astype(str) == champ_model]
        if not preds.empty and champ_model is not None and "model_type" in preds.columns
        else preds
    )
    date_col = "forecast_date" if "forecast_date" in champ_preds.columns else "ds"
    fva_lift: float | None = None
    with contextlib.suppress(Exception):
        fva = _safe_extract_df(target, "fva_df", "fva_cascade_df")
        fcol = (
            "model_type"
            if "model_type" in fva.columns
            else ("model" if "model" in fva.columns else None)
        )
        if not fva.empty and champ_model is not None and fcol is not None:
            frow = fva[fva[fcol].astype(str) == champ_model]
            for lcol in ("fva_pct", "fva_relative_lift_pct"):
                if not frow.empty and lcol in frow.columns and pd.notna(frow.iloc[0][lcol]):
                    fva_lift = float(frow.iloc[0][lcol])
                    break
    n_bias = 0
    with contextlib.suppress(Exception):
        bias_df = _safe_extract_df(target, "bias_report_df", "bias_df")
        if not bias_df.empty and "bias_alert" in bias_df.columns:
            bsub = (
                bias_df[bias_df["model_type"].astype(str) == champ_model]
                if champ_model is not None and "model_type" in bias_df.columns
                else bias_df
            )
            n_bias = int(bsub["bias_alert"].fillna(False).astype(bool).sum())
    return pd.DataFrame.from_records(
        [
            {
                "run_id": rid,
                "run_name": run_name,
                "status": "COMPLETED",
                "created_at": None,
                "n_series": (
                    int(champ_preds["ts_id"].nunique())
                    if not champ_preds.empty and "ts_id" in champ_preds.columns
                    else default_n_series
                ),
                "champion_model": champ_model,
                "champion_wape": champ_wape,
                "champion_mase": champ_mase,
                "champion_coverage": champ_cov,
                "fva_lift_pct": fva_lift,
                "horizon_dates": (
                    int(champ_preds[date_col].nunique())
                    if not champ_preds.empty and date_col in champ_preds.columns
                    else 0
                ),
                "total_forecast_units": (
                    float(champ_preds["yhat"].sum())
                    if not champ_preds.empty and "yhat" in champ_preds.columns
                    else 0.0
                ),
                "total_lower_bound_units": (
                    float(champ_preds["yhat_lower"].sum())
                    if not champ_preds.empty and "yhat_lower" in champ_preds.columns
                    else 0.0
                ),
                "total_upper_bound_units": (
                    float(champ_preds["yhat_upper"].sum())
                    if not champ_preds.empty and "yhat_upper" in champ_preds.columns
                    else 0.0
                ),
                "n_bias_alert_skus": n_bias,
            }
        ]
    )


def ask_offline(
    target: Any,
    question: str,
    *,
    project_id: str = "your-gcp-project-id",
    dataset_id: str = "scale_forecasting",
    conversation_id: str | None = None,
) -> BQAgentResponse:
    """Answer ``question`` from in-memory ``target`` DataFrames and return a `BQAgentResponse`."""
    import pandas as pd

    if not question or not question.strip():
        from ..errors import ConfigError

        raise ConfigError("question must be a non-empty string")

    golden = match_golden_query(question, project_id=project_id, dataset_id=dataset_id)
    run_id = _infer_run_id(target)
    conv_id = conversation_id or f"offline-{run_id}"

    df = pd.DataFrame()
    summary_lines: list[str] = []

    if golden.category == "executive_summary":
        df = _safe_extract_df(target, "sop_summary_df")
        if not df.empty:
            champ = df.iloc[0].get("champion_model", "champion")
            units = df.iloc[0].get("total_forecast_units", 0.0)
            summary_lines.append(
                f"Executive S&OP rollup for `{run_id}`: champion `{champ}` forecasts "
                f"{float(units or 0.0):,.2f} total units."
            )
    elif golden.category == "fva":
        df = _safe_extract_df(target, "fva_df", "fva_cascade_df")
        if not df.empty:
            summary_lines.append(
                f"Computed Forecast Value Added (FVA) across {len(df)} model stage(s) "
                f"for `{run_id}`."
            )
    elif golden.category == "exceptions":
        df = _safe_extract_df(target, "exceptions_df", "bias_report_df", "bias_df")
        summary_lines.append(
            f"Evaluated planner bias & exception triage for `{run_id}` ({len(df)} flagged row(s))."
        )
    elif golden.category == "horizon_and_anomalies":
        if "anomal" in question.lower() or "spike" in question.lower():
            df = _safe_extract_df(target, "detect_anomalies_df")
            summary_lines.append(
                f"Inspected historical anomalies for `{run_id}` ({len(df)} row(s))."
            )
        else:
            df = _safe_extract_df(target, "horizon_decay_df")
            summary_lines.append(
                f"Computed step-by-step horizon accuracy decay for `{run_id}` "
                f"({len(df)} step row(s))."
            )
    elif golden.category == "forecasts":
        df = _safe_extract_df(target, "predictions_df", "predictions")
        if not df.empty and "yhat" in df.columns:
            total_vol = float(df["yhat"].sum())
            summary_lines.append(
                f"Forward forecast table for `{run_id}` contains {len(df)} row(s) "
                f"(cumulative `yhat` = {total_vol:,.2f})."
            )
    elif golden.category == "explainability":
        df = _safe_extract_df(target, "feature_attributions_df", "promo_elasticity_df")
        summary_lines.append(f"Extracted driver attributions for `{run_id}` ({len(df)} row(s)).")

    if df.empty:
        df = _safe_extract_df(
            target, "leaderboard_df", "leaderboard", "predictions_df", "predictions"
        )
        model_col = (
            "model_type"
            if "model_type" in df.columns
            else ("model" if "model" in df.columns else None)
        )
        if not df.empty and model_col is not None:
            top_model = str(df.iloc[0][model_col])
            wape_txt = ""
            for col in ("pooled_wape", "wape", "mean_wape", "score"):
                if col in df.columns and pd.notna(df.iloc[0][col]):
                    wape_txt = f" ({col}={float(df.iloc[0][col]):.4f})"
                    break
            summary_lines.append(
                f"Top-ranked model on `{run_id}` is `{top_model}`{wape_txt} across "
                f"{len(df)} candidate(s)."
            )

    if not summary_lines:
        summary_lines.append(
            f"Answered offline for `{run_id}` using matched BigQuery Golden SQL template "
            f"(`{golden.category}`)."
        )

    return BQAgentResponse(
        question=question.strip(),
        answer=" ".join(summary_lines),
        sql=golden.sql,
        df=df,
        chart_spec=None,
        conversation_id=conv_id,
        agent_name="offline-workbench-router",
        run_id=run_id,
        mode="offline",
    )
