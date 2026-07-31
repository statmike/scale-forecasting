"""Execute + score the run's ensembles into the registry (DESIGN §5.2, the B5 orchestration seam).

:mod:`ensembler` is **pure** — it renders the calculated-ensemble SQL and fits the learned
meta-learners, but touches no GCP. This module is the thin I/O orchestrator that makes those
consensuses *real*: it runs the SQL, applies the learned weights, and — the missing leaderboard
link — **scores every ensemble pseudo-model into ``forecast_metadata``** so it appears on
``v_model_leaderboard`` beside the base models. It is engine-agnostic (it only reads the shared
registry) and is called from :func:`main.run` after both engines join, under the one shared
``run_id``.

Three responsibilities, in order:

1. **Calculated** (``mean`` / ``median`` / ``inverse_error``) — run
   :func:`ensembler.build_ensemble_sql` as a single multi-statement BigQuery script (``@run_id``
   bound as a query parameter, visible to every statement), which writes ``ensemble_<s>`` rows into
   ``forecast_predictions`` with ``compute_engine='ensemble'``.
2. **Learned** (``nnls`` / ``ridge`` / ``xgb``) — read the base predictions + ``backtest_oof``,
   ``fit_learned`` on the OOF, then apply the weights **in pandas** (``yhat = Σ wₘ·yhatₘ``,
   renormalized over whichever base models are present per ``(ts_id, forecast_date)`` — robust when
   the Spark future window and the native held-out window don't overlap), and append the resulting
   ``ensemble_<s>`` prediction rows via the Storage Write API. Each fitted meta-learner is uploaded
   as a GCS artifact and linked from its scored metadata row.
3. **Score** — read every ``ensemble_*`` prediction back, **join to ``source_series`` actuals** on
   ``(ts_id, forecast_date)``, and run the shared :func:`metrics.compute_metrics` per ``(model,
   ts_id)`` → ``forecast_metadata`` rows with ``fold_id=NULL`` and ``compute_engine='ensemble'``.
   The join is what makes scoring correct across the two forecast windows: only dates with ground
   truth (the held-out window) contribute, so a future-dated Spark-blended row simply drops out —
   the ensemble is scored on exactly the window the native models are scored on. Once these
   ``fold_id IS NULL`` rows land, the leaderboard shows the ensembles automatically — **no view
   change**.

**Idempotency (append-only + dedupe-on-read, §3.4).** ``run_id`` is a pure digest of the config and
every ensemble row is deterministic, so a re-run lands byte-identical rows. The calculated
``INSERT … SELECT`` is plain DML (not the Write API), so a re-run *does* append a second copy of the
calculated prediction rows — correct-but-wasteful: serving views dedupe on read, and duplicated
identical ``(y_true, yhat)`` pairs leave the ratio/mean metrics unchanged, so the leaderboard is
unaffected. (No pre-delete: a ``DELETE`` matching rows still in the base models' ~90-min Write API
streaming buffer is rejected for the whole window — the constraint the cell writers already live
under.)

Public surface: :func:`run_ensembles`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .ensembler import build_ensemble_sql, fit_learned

if TYPE_CHECKING:
    from .config import RunConfig
    from .settings import Settings


def run_ensembles(
    cfg: RunConfig, run_id: str, *, settings: Settings
) -> None:  # pragma: no cover - GCP I/O, @gcp ensemble smoke
    """Execute + score every requested ensemble for ``run_id`` into the shared registry.

    A no-op when ``cfg.ensemble.enabled`` is false. Otherwise runs the calculated SQL, applies the
    learned weights, and scores all ensemble pseudo-models into ``forecast_metadata`` (see the
    module docstring). Raises on any failure so :func:`main.run` can finalize the shared header
    FAILED — mirroring how an engine error is surfaced. ``settings`` is the orchestrator's
    already-resolved infra (never re-resolved here, so one identity governs the whole run).
    """
    import json
    from datetime import UTC, datetime

    from google.cloud import bigquery

    from .engines import bigquery_engine
    from .engines.bigquery_engine import _source_ref, build_history_query
    from .errors import get_logger
    from .metrics import METRIC_NAMES, compute_metrics
    from .registry import bq
    from .registry.artifacts import upload_artifact_bytes
    from .registry.ids import make_model_hash

    if not cfg.ensemble.enabled:
        return

    log = get_logger(__name__)
    dataset = settings.dataset_ref
    source = _source_ref(cfg, dataset)
    client = bigquery.Client(project=settings.project_id)
    created_at = datetime.now(UTC)
    log.info("ensemble run start: run_id=%s strategies=%s", run_id, cfg.ensemble.strategies)

    def _query(sql: str) -> Any:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
        )
        return client.query(sql, job_config=job_config).result()

    # 1. Calculated ensembles — one multi-statement script; @run_id reaches every statement.
    calc_sql = build_ensemble_sql(cfg, dataset)
    if calc_sql:
        _query(calc_sql)
        log.info("ensemble calculated SQL executed: run_id=%s", run_id)

    # 2. Learned ensembles — fit on the OOF, apply the weights in pandas, append prediction rows.
    artifact_uris: dict[str, str] = {}
    learned_weights: dict[str, dict[str, float]] = {}
    models = list(cfg.models)
    model_list = ", ".join(f"'{m}'" for m in models)
    base_pred_sql = (
        "SELECT ts_id, model_type, forecast_date, yhat, yhat_lower, yhat_upper\n"
        f"FROM `{dataset}.forecast_predictions`\n"
        f"WHERE run_id = @run_id AND model_type IN ({model_list})"
    )
    oof_sql = (
        "SELECT ts_id, model_type, fold_id, forecast_date, y_true, yhat\n"
        f"FROM `{dataset}.backtest_oof`\n"
        "WHERE run_id = @run_id"
    )
    oof_df = _query(oof_sql).to_dataframe()
    learned_weights, artifacts = fit_learned(oof_df, cfg)
    if learned_weights:
        base_df = _query(base_pred_sql).to_dataframe()
        pred_rows: list[dict[str, Any]] = []
        for strategy, wmap in learned_weights.items():
            pred_rows.extend(_apply_weights(base_df, wmap, run_id, strategy))
            artifact_uris[strategy] = upload_artifact_bytes(
                artifacts[strategy], f"ensemble_{strategy}.pkl", run_id, settings.warehouse_uri
            )
        bigquery_engine._append_rows(settings, "forecast_predictions", bq._PRED_SPEC, pred_rows)
        log.info(
            "ensemble learned applied: run_id=%s strategies=%s rows=%d",
            run_id,
            list(learned_weights),
            len(pred_rows),
        )

    # 3. Score every ensemble_* pseudo-model into forecast_metadata (fold_id=NULL) — the join to
    #    actuals scores only the window that has ground truth, so the metric is comparable to the
    #    base models on the leaderboard.
    idc, datec, targetc = cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col
    eval_sql = (
        "SELECT p.ts_id AS ts_id, p.model_type AS model_type, p.forecast_date AS forecast_date,\n"
        f"       s.{targetc} AS y_true, p.yhat AS yhat,\n"
        "       p.yhat_lower AS yhat_lower, p.yhat_upper AS yhat_upper\n"
        f"FROM `{dataset}.forecast_predictions` p\n"
        f"JOIN `{source}` s ON s.{idc} = p.ts_id AND s.{datec} = p.forecast_date\n"
        "WHERE p.run_id = @run_id AND p.compute_engine = 'ensemble'\n"
        "ORDER BY p.model_type, p.ts_id, p.forecast_date"
    )
    eval_df = _query(eval_sql).to_dataframe()
    if eval_df.empty:
        log.warning("ensemble scoring: no ensemble rows with actuals for run_id=%s", run_id)
        return

    history = _query(build_history_query(cfg, dataset)).to_dataframe()
    hist_by_id = {tid: g["y"].to_numpy() for tid, g in history.groupby("ts_id")}

    meta_rows: list[dict[str, Any]] = []
    for (model_type, ts_id), g in eval_df.groupby(["model_type", "ts_id"]):
        g = g.sort_values("forecast_date")
        panel = compute_metrics(
            g["y_true"].to_numpy(),
            g["yhat"].to_numpy(),
            y_train=hist_by_id.get(ts_id),
            lower=g["yhat_lower"].to_numpy(),
            upper=g["yhat_upper"].to_numpy(),
        )
        strategy = str(model_type).removeprefix("ensemble_")
        meta_rows.append(
            {
                "run_id": run_id,
                "ts_id": ts_id,
                "model_type": model_type,
                "compute_engine": "ensemble",
                "model_hash": make_model_hash(run_id, str(ts_id), str(model_type), cfg),
                "fold_id": None,
                **{name: panel[name] for name in METRIC_NAMES},
                "fit_seconds": None,
                "best_params": (
                    json.dumps(learned_weights[strategy], sort_keys=True)
                    if strategy in learned_weights
                    else None
                ),
                "model_artifact": artifact_uris.get(strategy),
                "created_at": created_at,
            }
        )
    bigquery_engine._append_rows(settings, "forecast_metadata", bq._META_SPEC, meta_rows)
    log.info(
        "ensemble run done: run_id=%s scored=%d models=%s",
        run_id,
        len(meta_rows),
        sorted(eval_df["model_type"].unique()),
    )


def _apply_weights(
    base_df: pd.DataFrame,
    wmap: dict[str, float],
    run_id: str,
    strategy: str,
) -> list[dict[str, Any]]:
    """Blend base predictions by learned weights → ``ensemble_<strategy>`` prediction rows (pure).

    ``base_df`` is long-format ``(ts_id, model_type, forecast_date, yhat, yhat_lower, yhat_upper)``.
    For each ``(ts_id, forecast_date)`` the weighted mean is taken over **whichever base models are
    present**, with the weights renormalized to sum to 1 over that present subset — so a date where
    only some base models forecast (the disjoint Spark-future / native-held-out case) still yields a
    well-defined blend rather than a NULL. Rows where no weighted base model is present, or the
    present weights sum to zero, are dropped. Pure (no GCP) so it is unit-tested offline."""
    model_type = f"ensemble_{strategy}"
    rows: list[dict[str, Any]] = []
    if base_df.empty:
        return rows
    models = [m for m in wmap if m in set(base_df["model_type"].unique())]
    if not models:
        return rows
    weights = np.array([wmap[m] for m in models], dtype=float)

    def _blend(value_col: str) -> Any:
        wide = base_df.pivot_table(
            index=["ts_id", "forecast_date"], columns="model_type", values=value_col
        ).reindex(columns=models)
        vals = wide.to_numpy(dtype=float)
        present = ~np.isnan(vals)
        wrow = present * weights  # zero out absent models per row
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
                "run_id": run_id,
                "ts_id": ts_id,
                "model_type": model_type,
                "compute_engine": "ensemble",
                "forecast_date": forecast_date,
                "yhat": float(yh),
                "yhat_lower": None if np.isnan(lo) else float(lo),
                "yhat_upper": None if np.isnan(up) else float(up),
                "quantiles": None,
            }
        )
    return rows
