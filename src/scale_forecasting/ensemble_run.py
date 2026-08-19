"""Execute + score the run's ensembles into the registry (the orchestration seam).

`ensembler` is **pure** — it blends the base forecasts (calculated) and fits the learned
meta-learners, but touches no GCP. This module is the thin I/O orchestrator that makes those
consensuses *real*: it reads the base rows, computes both families **in pandas**, appends the
``ensemble_<s>`` prediction rows via the Storage Write API, and — the missing leaderboard link —
**scores every ensemble pseudo-model into ``forecast_metadata``** so it appears on
``v_model_leaderboard`` beside the base models. It is engine-agnostic (it only reads the shared
registry) and runs either inline from `main.run` (after both engines join, under the one
shared ``run_id``) or standalone via ``python -m scale_forecasting.ensemble_run`` (re-ensembling an
already-completed run — see `_main`).

**Config-keyed ensembles.** Every ensemble row carries an ``ensemble_id =
make_ensemble_id(cfg.ensemble)`` — a digest of the ensemble configuration alone — so *several*
ensemble configs can be scored under one ``run_id`` without their ``ensemble_<strategy>``
pseudo-models colliding. Re-running the *same* ensemble config lands the same ``ensemble_id`` (and,
being deterministic, byte-identical rows); a *different* config lands a different ``ensemble_id``
and sits beside the first on the leaderboard, distinctly keyed. The leaderboard view groups by
``(run_id, model_type, ensemble_id)`` so the two never merge.

Three responsibilities, in order:

1. **Calculated** (``mean`` / ``median`` / ``inverse_error``) — read the base
   ``forecast_predictions`` (+ ``forecast_metadata`` for the inverse-error weights / pruning) and
   blend them **in pandas** (`ensembler.combine_calculated`), then append the ``ensemble_<s>``
   rows via the **Storage Write API** — the same append path the learned strategies use (no
   ``INSERT…SELECT`` DML). ``compute_engine='ensemble'``.
2. **Learned** (``nnls`` / ``ridge`` / ``xgb``) — read the base predictions + ``backtest_oof``,
   ``fit_learned`` on the OOF, then apply the weights **in pandas** (``yhat = Σ wₘ·yhatₘ``,
   renormalized over whichever base models are present per ``(ts_id, forecast_date)`` — robust when
   the Spark future window and the native held-out window don't overlap), and append the resulting
   ``ensemble_<s>`` prediction rows via the Storage Write API. Each fitted meta-learner is uploaded
   as a GCS artifact and linked from its scored metadata row. These prediction rows are a **true
   beyond-data forecast** (the base predictions they blend are, too), so — like the base
   models — they carry no ground truth of their own.
3. **Score** — blend the base ``backtest_oof`` into an **ensemble OOF** with the same consensus
   rules (`ensembler.combine_oof`) and run the shared `metrics.compute_metrics` per
   ``(model, ts_id)`` → ``forecast_metadata`` rows with ``fold_id=NULL`` and
   ``compute_engine='ensemble'``. Scoring lives on the OOF window because the base
   predictions (and therefore every ensemble prediction) are a true beyond-data forecast with no
   actuals to join — so an ensemble earns its leaderboard metric on **exactly the window the base
   models are scored on** (``backtest_oof``, where ``y_true`` lives). OOF carries no interval
   bounds, so ensemble coverage/pinball are NaN — consistent with the base models' OOF metrics.
   Learned consensuses are scored on the folds their meta-learner trained on (mildly optimistic,
   the price of stacking having no held-out-of-held-out window). Once these ``fold_id IS NULL`` rows
   land, the leaderboard shows the ensembles automatically — **no view change** beyond the
   ``ensemble_id`` group key.

**Idempotency (append-only + dedupe-on-read).** Every ensemble row is now written through the
Write API and is deterministic in ``(run_id, ensemble_id, ts_id, model_type)``, so a re-run of the
same ensemble config lands byte-identical rows — correct-but-wasteful (a re-append is a duplicate):
the leaderboard view dedupes on read (``GROUP BY run_id, model_type, ensemble_id``), and duplicated
identical ``(y_true, yhat)`` pairs leave the ratio/mean metrics unchanged, so the leaderboard is
unaffected. No pre-delete — a ``DELETE`` matching rows still in the ~90-min Write API streaming
buffer is rejected for the whole window (the constraint every cell writer already lives under).
A *different* ensemble config keys distinctly (different ``ensemble_id``), so it never overwrites
and never collides — both coexist.

Public surface: `run_ensembles`, ``python -m scale_forecasting.ensemble_run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .ensembler import combine_calculated, fit_learned

if TYPE_CHECKING:
    from .config import RunConfig
    from .settings import Settings


def run_ensembles(
    cfg: RunConfig, run_id: str, *, settings: Settings
) -> None:  # pragma: no cover - GCP I/O, @gcp ensemble smoke
    """Execute + score every requested ensemble for ``run_id`` into the shared registry.

    A no-op when ``cfg.ensemble.enabled`` is false. Otherwise blends the calculated + learned
    consensuses in pandas, appends their ``ensemble_<s>`` prediction rows via the Storage Write API,
    and scores every ensemble pseudo-model into ``forecast_metadata`` (see the module docstring),
    all stamped with ``ensemble_id = make_ensemble_id(cfg.ensemble)`` so multiple ensemble configs
    coexist under one ``run_id``. Raises on any failure so `main.run` can finalize the shared
    header FAILED — mirroring how an engine error is surfaced. ``settings`` is the orchestrator's
    already-resolved infra (never re-resolved here, so one identity governs the whole run).
    """
    import json
    from datetime import UTC, datetime

    from google.cloud import bigquery

    from .engines import bigquery_engine
    from .ensembler import combine_oof
    from .errors import get_logger
    from .metrics import METRIC_NAMES, compute_metrics
    from .registry import bq
    from .registry.artifacts import upload_artifact_bytes
    from .registry.ids import make_ensemble_id, make_model_hash
    from .worker import _rollup_metrics

    if not cfg.ensemble.enabled:
        return

    log = get_logger(__name__)
    dataset = settings.dataset_ref
    ensemble_id = make_ensemble_id(cfg.ensemble)
    client = bigquery.Client(project=settings.project_id)
    created_at = datetime.now(UTC)
    log.info(
        "ensemble run start: run_id=%s ensemble_id=%s strategies=%s",
        run_id,
        ensemble_id,
        cfg.ensemble.strategies,
    )

    def _query(sql: str) -> Any:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
        )
        return client.query(sql, job_config=job_config).result()

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
    metric = cfg.backtest.decision_metric
    metric_sql = (
        f"SELECT ts_id, model_type, {metric}\n"
        f"FROM `{dataset}.forecast_metadata`\n"
        f"WHERE run_id = @run_id AND fold_id IS NULL AND model_type IN ({model_list})"
    )
    base_df = _query(base_pred_sql).to_dataframe()
    oof_df = _query(oof_sql).to_dataframe()
    metric_df = _query(metric_sql).to_dataframe()

    # A single Write-API append of every ensemble prediction row (calculated + learned). Each row is
    # stamped with run_id + ensemble_id here so the pure blenders stay config-only.
    pred_rows: list[dict[str, Any]] = []

    # 1. Calculated ensembles — blend the base predictions in pandas (Write API, not DML).
    for row in combine_calculated(base_df, cfg, metric_df):
        row.update(
            run_id=run_id, ensemble_id=ensemble_id, compute_engine="ensemble", quantiles=None
        )
        pred_rows.append(row)

    # 2. Learned ensembles — fit on the OOF, apply the weights in pandas, append prediction rows.
    artifact_uris: dict[str, str] = {}
    learned_weights, artifacts = fit_learned(oof_df, cfg)
    for strategy, wmap in learned_weights.items():
        for row in _apply_weights(base_df, wmap, run_id, strategy):
            row["ensemble_id"] = ensemble_id
            pred_rows.append(row)
        artifact_uris[strategy] = upload_artifact_bytes(
            artifacts[strategy], f"ensemble_{ensemble_id}_{strategy}.pkl", run_id,
            settings.warehouse_uri,
        )

    if pred_rows:
        bigquery_engine._append_rows(settings, "forecast_predictions", bq._PRED_SPEC, pred_rows)
    log.info(
        "ensemble predictions appended: run_id=%s ensemble_id=%s rows=%d strategies=%s",
        run_id,
        ensemble_id,
        len(pred_rows),
        cfg.ensemble.strategies,
    )

    # 3. Score every ensemble_* pseudo-model into forecast_metadata (fold_id=NULL). The base
    #    predictions are a true beyond-data forecast (no actuals to join), so ensembles earn their
    #    metric on the backtest OOF window — the same window the base models are scored on. We blend
    #    the base OOF with the same consensus rules, then compute_metrics per (model, ts_id) and
    #    roll the folds up exactly as worker.py does for the base models.
    if oof_df.empty:
        log.warning("ensemble scoring: backtest_oof empty for run_id=%s — nothing to score", run_id)
        return
    ens_oof = combine_oof(oof_df, cfg, learned_weights)
    if ens_oof.empty:
        log.warning("ensemble scoring: no ensemble OOF produced for run_id=%s", run_id)
        return

    # y_train (for MASE/RMSSE scale) is per-series history; the base OOF has no in-sample rows, so
    # read the full series history once, matching the natives' history read.
    history = _query(bigquery_engine.build_history_query(cfg, dataset)).to_dataframe()
    hist_by_id = {tid: g["y"].to_numpy() for tid, g in history.groupby("ts_id")}

    meta_rows: list[dict[str, Any]] = []
    for (model_type, ts_id), g in ens_oof.groupby(["model_type", "ts_id"]):
        # Score per fold, then roll up (NaN-ignoring mean) — identical to the base-model path
        # (worker._rollup_metrics), so ensemble and base metrics are computed the same way.
        fold_panels: list[dict[str, float]] = []
        for _fold, fg in g.sort_values("forecast_date").groupby("fold_id"):
            fold_panels.append(
                compute_metrics(
                    fg["y_true"].to_numpy(),
                    fg["yhat"].to_numpy(),
                    y_train=hist_by_id.get(ts_id),
                )
            )
        panel = _rollup_metrics(fold_panels)
        strategy = str(model_type).removeprefix("ensemble_")
        meta_rows.append(
            {
                "run_id": run_id,
                "ts_id": ts_id,
                "model_type": model_type,
                "compute_engine": "ensemble",
                "model_hash": make_model_hash(run_id, str(ts_id), str(model_type), cfg),
                "ensemble_id": ensemble_id,
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
        "ensemble run done: run_id=%s ensemble_id=%s scored=%d models=%s",
        run_id,
        ensemble_id,
        len(meta_rows),
        sorted(ens_oof["model_type"].unique()),
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


# --- standalone CLI (re-ensemble an already-completed run) ---------------------


def _override_ensemble(cfg: RunConfig, strategies: list[str] | None) -> RunConfig:
    """Return ``cfg`` with its ``ensemble`` block overridden to ``strategies`` (pure).

    ``None`` leaves the config's own ensemble block untouched; a list rebuilds
    `EnsembleConfig` (so the strategies are **re-validated**
    against the known calculated/learned sets and enabled), preserving the original
    ``prune_threshold``. This is what lets one base run be ensembled several ways from the CLI —
    each override is a distinct ``EnsembleConfig`` and thus a distinct ``ensemble_id`` per run.
    """
    from .config import EnsembleConfig

    if strategies is None:
        return cfg
    # Rebuild via validated model construction so CLI strings are checked against the known
    # strategy set (raises a clear pydantic error on a typo) rather than trusted blindly.
    ensemble = EnsembleConfig.model_validate(
        {
            "enabled": True,
            "strategies": strategies,
            "prune_threshold": cfg.ensemble.prune_threshold,
        }
    )
    return cfg.model_copy(update={"ensemble": ensemble})


def _main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    """``python -m scale_forecasting.ensemble_run --config c.json [--run-id …] [--strategies …]``.

    Re-runs the ensemble stage against an *already-completed* run's base predictions — the
    standalone counterpart to the inline call `main.run` makes. Loads the config, optionally
    overrides the ensemble strategies (``--strategies mean,median``), resolves the infra identity
    from the ``SF_*`` environment (``--sf-*`` promoted first), and calls `run_ensembles`.

    ``--run-id`` is the **base run whose forecasts are blended**; it defaults to the config's own
    ``make_run_id`` (re-ensembling the run that config produced), but is passed explicitly to
    ensemble a run whose base models were computed under a *different* config revision. The
    ``ensemble_id`` is always derived from the (possibly overridden) ensemble block, so re-running
    with new ``--strategies`` lands a *new* ``ensemble_id`` beside the existing one — never
    overwriting (append-only), never colliding (distinctly keyed).
    """
    import argparse

    from ._infra_args import add_infra_args, export_infra_env
    from .config import load_config
    from .errors import get_logger
    from .registry.ids import make_run_id
    from .settings import Settings

    parser = argparse.ArgumentParser(
        prog="ensemble_run", description="Re-run the ensemble stage for a completed run."
    )
    parser.add_argument("--config", required=True, help="path to the run config JSON")
    parser.add_argument(
        "--run-id",
        default=None,
        help="base run to ensemble (default: derived from --config via make_run_id)",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="comma-separated override of ensemble.strategies (default: the config's own block)",
    )
    add_infra_args(parser)
    ns = parser.parse_args(argv)
    export_infra_env(ns)

    strategies = (
        [s.strip() for s in ns.strategies.split(",") if s.strip()]
        if ns.strategies is not None
        else None
    )
    cfg = _override_ensemble(load_config(ns.config), strategies)
    run_id = ns.run_id or make_run_id(cfg)
    get_logger(__name__).info(
        "ensemble_run CLI: run_id=%s strategies=%s", run_id, cfg.ensemble.strategies
    )
    run_ensembles(cfg, run_id, settings=Settings.resolve())


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    _main()
