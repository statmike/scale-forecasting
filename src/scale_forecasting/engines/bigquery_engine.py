"""BigQuery-native models — the SQL runner for ARIMA_PLUS / TimesFM.

The BigQuery runtime executes forecasting *as SQL inside BigQuery* — the opposite of the Spark
track's per-cell fan-out. ``ARIMA_PLUS`` with ``time_series_id_col`` trains **all series in one
``CREATE MODEL`` statement``; ``AI.FORECAST`` (TimesFM) forecasts every series in one call with no
training at all. Both land in the *same* three-tier registry as the Python models, so a native model
and a Spark model are directly comparable on ``v_model_leaderboard``.

This module is the executor: `run` resolves `Settings`, owns the ``run_registry`` header lifecycle
exactly like `spark_explode.run`, executes `bigquery_sql`'s statements via ``bigquery.Client``,
reads the fold forecasts back, computes the metric panel through the shared `compute_metrics` (no
formula drift), and writes all three cell tables via the registry's Storage Write API row-dict path.
The SQL it runs is rendered by `bigquery_sql`; the object names in that SQL come from
`bigquery_names`.

**Alignment with the Spark track.** The native models mean the *same thing* as the
Python models in every table:

* ``forecast_predictions`` **always** holds a **true beyond-data forecast** — the final model is fit
  on *all* history and forecasts the next ``data.horizon`` steps, exactly like the Spark path's
  final-fit-then-forecast. It is never a scored within-history window.
* Scored evaluation lives **entirely in the backtest path**, for both engines. When
  ``backtest.enabled`` is on, a **BQML fold loop** (per fold: ``CREATE MODEL`` on ``ds <= cutoff`` +
  ``ML.FORECAST``) mirrors `backtest.make_folds`'s anchored-from-end geometry, writing
  ``backtest_oof`` with real ``fold_id``s and a rolled-up ``forecast_metadata`` panel
  (``fold_id=NULL``). When backtest is off, the engine writes a ``fold_id=NULL`` metadata row per
  ``(series, model)`` with a NaN metric panel — precise parity with the Python worker, which also
  emits an unscored metadata row when backtesting is off.

**Transform.** ``cfg.features.transform`` (e.g. ``log1p``) is intentionally **not** applied here:
ARIMA_PLUS runs its own decomposition, and TimesFM is a pretrained foundation model.
Holidays *are* honored — the custom-holiday CTE is built from the same
`holiday_frame` the Python suite uses, so "holiday" is identical
across runtimes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..registry.tables import bigquery_client

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from ..config import RunConfig
    from ..settings import Settings


@dataclass(frozen=True)
class BqOutcome:
    """The BigQuery engine's run summary — what `main.run` folds into the shared header.

    ``status`` is COMPLETED (the engine raises on any SQL failure rather than returning FAILED, so a
    returned outcome is always COMPLETED today; the field is explicit for symmetry with the Spark
    roll-up and future partial-success handling). ``n_series`` is the distinct series count observed
    in the source subset; ``models`` is the executed native subset (feeds ``bq_models``).
    """

    status: str
    n_series: int
    models: list[str]


def run(
    cfg: RunConfig,
    models: list[str],
    *,
    manage_header: bool = True,
    settings: Settings | None = None,
    job_id_prefix: str | None = None,
) -> BqOutcome:  # pragma: no cover - GCP I/O, @gcp smoke
    """Execute the BigQuery-native subset end-to-end, mirroring `spark_explode.run`.

    Header lifecycle: resolve `Settings`, derive the config-pinned
    ``run_id``, ``ensure_tables`` → ``write_header`` (RUNNING), run the SQL, then ``update_header``
    with the aggregated status, wall-clock ``runtime_seconds``, ``n_series``, ``n_models``, and the
    ``bq_models`` array.

    Two phases per run (see the module docstring):

    * **Final forecast (always).** Each model's `build_setup_statements` fits on all history
      and INSERTs a true beyond-data forecast into ``forecast_predictions`` — parity with Spark.
    * **Scored evaluation (backtest only).** When ``backtest.enabled``, a fold loop
      (`fold_plan`) trains one model per fold on ``ds <= cutoff``, reads each fold's forecast
      joined to actuals via `build_eval_query`, writes ``backtest_oof`` with real
      ``fold_id``s, and rolls the per-fold panels up (via ``worker.rollup_metrics``) into a
      ``fold_id=NULL``
      ``forecast_metadata`` row. When backtest is off, a single unscored ``fold_id=NULL`` metadata
      row per ``(series, model)`` (NaN panel) is written instead — parity with the Python worker.

    ``manage_header=False`` is **contributor mode**: `main.run` owns the single shared
    header, so the engine skips ``ensure_tables`` / ``write_header`` / ``update_header`` and only
    runs SQL + writes the cell tables. ``settings`` may be passed to reuse the orchestrator's
    already-resolved infra; ``None`` resolves it here (standalone default).

    Idempotent: ``run_id`` is a pure function of the config and every write is append-only /
    dedupe-on-read, so a re-run of the same config lands byte-identical rows.

    ``job_id_prefix`` (the orchestrator's deterministic ``system_job_id`` for the native family)
    prefixes every BigQuery job this run submits, so the family's jobs are console-resolvable under
    that id — the reverse-trace hook. ``None`` (the standalone default) lets BigQuery auto-name it.
    """
    import time
    from datetime import UTC, datetime

    import pandas as pd
    from google.cloud import bigquery

    from ..errors import RegistryError, get_logger
    from ..metrics import METRIC_NAMES
    from ..registry.ids import make_run_id
    from ..registry.lifecycle import run_header
    from ..registry.write_api import META_SPEC as _META_SPEC
    from ..registry.write_api import OOF_SPEC as _OOF_SPEC
    from ..settings import Settings
    from ..worker import rollup_metrics
    from .bigquery_sql import (
        bqml_options,
        build_arima_coefficients_query,
        build_arima_evaluate_query,
        build_eval_query,
        build_explain_forecast_query,
        build_fold_create_statements,
        build_fold_drop_statements,
        build_history_query,
        build_series_ids_query,
        build_setup_statements,
        fold_plan,
        xreg_cols,
    )

    log = get_logger(__name__)
    settings = settings or Settings.resolve()
    run_id = make_run_id(cfg)
    # Source reads resolve against `dataset`; model objects and forecast_predictions land in
    # `registry_dataset`. Identical strings unless the deployment split them (`registry_of`).
    dataset = settings.dataset_ref
    registry_dataset = settings.registry_dataset_ref
    client = bigquery_client(settings.project_id)
    log.info(
        "bigquery run start: run_id=%s models=%s manage_header=%s backtest=%s",
        run_id,
        models,
        manage_header,
        cfg.backtest.enabled,
    )

    def _query(sql: str) -> Any:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)],
            use_query_cache=False,
        )
        return client.query(sql, job_config=job_config, job_id_prefix=job_id_prefix).result()

    # Header first (run_header): RUNNING on entry, finalized on a clean exit; a crash records FAILED
    # on the owned header before re-raising. Contributor mode (main.run owns the shared header) is a
    # no-op wrapper, so main.run's finalize sees the raised RegistryError and records the status.
    with run_header(cfg, run_id, settings=settings, manage=manage_header) as hdr:
        started = time.perf_counter()
        created_at = datetime.now(UTC)
        status = "COMPLETED"
        nan_panel = {name: float("nan") for name in METRIC_NAMES}
        # The Spark and Ray jobs pin every source read to the snapshot the header recorded, so they
        # time-travel to one identical input state. The BigQuery-native subset cannot join them:
        # BQML CREATE MODEL rejects a constant FOR SYSTEM_TIME AS OF as "in the future" — even for a
        # timestamp committed hours ago that a plain SELECT against the same table time-travels to
        # fine (only CREATE MODEL is affected, and it accepts only a CURRENT_TIMESTAMP()-relative
        # expression, not a fixed snapshot instant; this holds for both native and BigLake Iceberg
        # sources). So the native subset reads un-pinned (live), keeping its reads internally
        # consistent (all un-pinned) rather than mixing pinned SELECTs with an un-pinnable CREATE
        # MODEL. Safe because a run's source data is static for its duration.
        snapshot_millis = None
        series_ids = [
            str(r.ts_id)
            for r in _query(build_series_ids_query(cfg, dataset, snapshot_millis=snapshot_millis))
        ]
        n_series = len(series_ids)
        try:
            # --- Phase 1: final true-future forecast → forecast_predictions (always) ----------
            tier1_by_model: dict[str, dict[str, dict[str, float]]] = {}
            arima_diag_by_model: dict[str, dict[str, dict[str, Any]]] = {}
            for model_name in models:
                for stmt in build_setup_statements(
                    cfg,
                    model_name,
                    dataset,
                    registry_dataset=registry_dataset,
                    snapshot_millis=snapshot_millis,
                ):
                    _query(stmt)
                explain_sql = build_explain_forecast_query(
                    cfg,
                    model_name,
                    dataset,
                    registry_dataset=registry_dataset,
                    snapshot_millis=snapshot_millis,
                )
                if explain_sql is not None:
                    try:
                        explain_df = _query(explain_sql).to_dataframe()
                        t1, _ = extract_bqml_explanations(
                            explain_df,
                            ts_id_col=cfg.data.ts_id_col,
                            xreg_cols=xreg_cols(cfg, model_name),
                        )
                        tier1_by_model[model_name] = t1
                    except Exception as exc:  # noqa: BLE001 - explainability is best-effort
                        log.warning("ML.EXPLAIN_FORECAST failed for %s: %s", model_name, exc)
                arima_eval_sql = build_arima_evaluate_query(
                    cfg,
                    model_name,
                    dataset,
                    registry_dataset=registry_dataset,
                )
                if arima_eval_sql is not None:
                    try:
                        arima_eval_df = _query(arima_eval_sql).to_dataframe()
                        arima_coef_sql = build_arima_coefficients_query(
                            cfg,
                            model_name,
                            dataset,
                            registry_dataset=registry_dataset,
                        )
                        arima_coef_df = (
                            _query(arima_coef_sql).to_dataframe()
                            if arima_coef_sql is not None
                            else None
                        )
                        arima_diag_by_model[model_name] = extract_arima_diagnostics(
                            arima_eval_df,
                            arima_coef_df,
                            ts_id_col=cfg.data.ts_id_col,
                        )
                    except Exception as exc:  # noqa: BLE001 - ARIMA diagnostics are best-effort
                        log.warning("ML.ARIMA_EVALUATE failed for %s: %s", model_name, exc)

            # --- Phase 2: scored evaluation ---------------------------------------------------
            oof_rows: list[dict[str, Any]] = []
            meta_rows: list[dict[str, Any]] = []

            if cfg.backtest.enabled:
                history = _query(
                    build_history_query(cfg, dataset, snapshot_millis=snapshot_millis)
                ).to_dataframe()
                # Dates ride along with the values: the scale denominator is the *fold's* training
                # window, not the whole series, and `backtest.training_window` cuts it at the
                # fold's cutoff. Keyed off `ORDER BY ts_id, ds`, so each group is already sorted.
                hist_by_id = {
                    str(tid): (pd.to_datetime(g["ds"]).to_numpy(), g["y"].to_numpy())
                    for tid, g in history.groupby("ts_id")
                }
                plan = fold_plan(cfg)
                for model_name in models:
                    best_params = json.dumps(bqml_options(cfg, model_name), sort_keys=True)
                    panels_by_ts: dict[str, list[dict[str, float]]] = {}
                    # The span of dates each series was actually scored over, accumulated from the
                    # rows as they are produced. The native fold grid is uniform across the panel,
                    # so this is less surprising here than on the Python path — but it is recorded
                    # for the same reason the Python path records it: one reader has to be able to
                    # compare a native model against a Python one without knowing which engine
                    # wrote the row, and a column that only one engine fills breaks that.
                    span_by_ts: dict[str, tuple[Any, Any]] = {}
                    for fold_id, back_steps in plan:
                        for stmt in build_fold_create_statements(
                            cfg,
                            model_name,
                            dataset,
                            fold_id,
                            back_steps,
                            registry_dataset=registry_dataset,
                            snapshot_millis=snapshot_millis,
                        ):
                            _query(stmt)
                        eval_df = _query(
                            build_eval_query(
                                cfg,
                                model_name,
                                dataset,
                                registry_dataset=registry_dataset,
                                back_steps=back_steps,
                                fold_id=fold_id,
                                snapshot_millis=snapshot_millis,
                            )
                        ).to_dataframe()
                        # The fold model has served its forecast — drop it so backtest runs don't
                        # leave orphaned sf_model_*_f{k} objects behind. Best-effort: a failed
                        # cleanup must not sink an otherwise-good run (results already read above).
                        for stmt in build_fold_drop_statements(
                            cfg, model_name, dataset, fold_id, registry_dataset=registry_dataset
                        ):
                            try:
                                _query(stmt)
                            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                                log.warning(
                                    "fold model cleanup failed (%s f%d): %s",
                                    model_name,
                                    fold_id,
                                    exc,
                                )
                        fold_oof, fold_panels = _score_fold(
                            eval_df,
                            hist_by_id,
                            run_id=run_id,
                            model_name=model_name,
                            fold_id=fold_id,
                            cfg=cfg,
                            created_at=created_at,
                        )
                        oof_rows.extend(fold_oof)
                        for row in fold_oof:
                            d = row["forecast_date"]
                            lo, hi = span_by_ts.get(row["ts_id"], (d, d))
                            span_by_ts[row["ts_id"]] = (min(lo, d), max(hi, d))
                        for ts_id, panel in fold_panels.items():
                            panels_by_ts.setdefault(ts_id, []).append(panel)
                    for ts_id, panels in panels_by_ts.items():
                        rolled = rollup_metrics(panels)
                        first_val, last_val = span_by_ts.get(ts_id, (None, None))
                        meta_rows.append(
                            _meta_row(
                                run_id,
                                ts_id,
                                model_name,
                                rolled,
                                best_params,
                                created_at,
                                cfg,
                                n_folds_achieved=len(panels),
                                first_val_date=first_val,
                                last_val_date=last_val,
                                feature_attributions=tier1_by_model.get(model_name, {}).get(ts_id),
                                arima_diagnostics=arima_diag_by_model.get(model_name, {}).get(
                                    ts_id
                                ),
                            )
                        )
            else:
                # Backtest off: one unscored fold_id=NULL metadata row per (series, model) — parity
                # with the Python worker, which also emits an unscored metadata row when off.
                for model_name in models:
                    best_params = json.dumps(bqml_options(cfg, model_name), sort_keys=True)
                    for ts_id in series_ids:
                        meta_rows.append(
                            _meta_row(
                                run_id,
                                ts_id,
                                model_name,
                                nan_panel,
                                best_params,
                                created_at,
                                cfg,
                                feature_attributions=tier1_by_model.get(model_name, {}).get(ts_id),
                                arima_diagnostics=arima_diag_by_model.get(model_name, {}).get(
                                    ts_id
                                ),
                            )
                        )

            _append_rows(settings, "backtest_oof", _OOF_SPEC, oof_rows)
            _append_rows(settings, "forecast_metadata", _META_SPEC, meta_rows)
        except Exception as exc:  # run_header records FAILED as this propagates
            # Wrap the cause so the failure reads clearly; run_header (owner mode) or main.run's
            # finalize (contributor mode) records the FAILED/PARTIAL header status.
            raise RegistryError(f"bigquery run failed for {run_id}: {exc}") from exc

        runtime_seconds = time.perf_counter() - started
        hdr.finalize(
            status=status,
            n_series=n_series,
            n_models=len(models),
            bq_models=list(models),
        )
    log.info(
        "bigquery run done: run_id=%s status=%s models=%d series=%d runtime=%.1fs",
        run_id,
        status,
        len(models),
        n_series,
        runtime_seconds,
    )
    return BqOutcome(status=status, n_series=n_series, models=list(models))


# --- the native row shapes (pure) ----------------------------------------------
# `_append_rows` encodes by walking the *spec*, so a key these builders emit that the spec does not
# name is dropped in silence — no error anywhere, just a column that reads NULL for native models
# only. That makes the row shapes a contract worth asserting with no cloud, which is why these two
# are plain functions rather than dict literals inside `run`'s GCP body.


def _score_fold(
    eval_df: pd.DataFrame,
    hist_by_id: Mapping[str, Any],
    *,
    run_id: str,
    model_name: str,
    fold_id: int,
    cfg: RunConfig,
    created_at: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """One fold's eval frame → its ``backtest_oof`` rows and one metric panel per series (pure).

    This is where a native model *earns its leaderboard number*, so it is worth having out of the
    GCP body. Four things it has to get right, none of which a live run would complain about if it
    got them wrong — it would just report different metrics:

    * the rows are sorted by ``forecast_date`` before scoring, so a horizon-weighted metric sees the
      horizon in order;
    * ``y_train`` — the scale denominator MASE and RMSSE divide by — is *this fold's* training
      window of the series' own history, cut at the fold's ``cutoff_date`` by
      `backtest.training_window`. Looked up per series rather than shared, and cut rather than
      passed whole: the full history contains the window being scored, and a denominator that has
      seen the future is not comparable to the Python engines', which never do;
    * the interval bounds are passed through, so ``coverage``, ``pinball``, ``interval_score`` and
      ``interval_width`` are real numbers here rather than the NaNs the Python worker's OOF path
      produces;
    * the seasonal period comes from ``cfg.data.freq`` rather than a default, so ``mase_seasonal``
      divides by the right naive.

    ``hist_by_id`` maps ``ts_id`` to a ``(ds, y)`` pair of aligned, date-sorted arrays.

    Panels are keyed by ``str(ts_id)`` to match `_meta_row`'s key type — the caller accumulates one
    list per series across folds and rolls them up exactly as `worker.rollup_metrics` does.
    """
    from ..backtest import training_window
    from ..calibration import apply_output_constraints, has_output_constraints
    from ..metrics import compute_metrics
    from ..seasonality import seasonal_period

    period = seasonal_period(cfg.data.freq)
    oof_rows: list[dict[str, Any]] = []
    panels: dict[str, dict[str, float]] = {}
    for ts_id, g in eval_df.groupby("ts_id"):
        g = g.sort_values("forecast_date").copy()
        if has_output_constraints(cfg.output):
            g["yhat_raw"] = g["yhat"].copy()
            g = apply_output_constraints(g, cfg.output)
        for rec in g.to_dict(orient="records"):
            oof_rows.append(_oof_row(run_id, str(ts_id), model_name, fold_id, rec, created_at))
        hist = hist_by_id.get(str(ts_id))
        # One cutoff per fold — the native path trains every series to the same global origin — so
        # any row of the group carries it. Absent on a hand-built or pre-cutoff frame, in which
        # case `training_window` keeps the whole history, which is what this path used to do.
        cutoff = g["cutoff_date"].iloc[0] if "cutoff_date" in g.columns else None
        panels[str(ts_id)] = compute_metrics(
            g["y_true"].to_numpy(),
            g["yhat"].to_numpy(),
            y_train=None if hist is None else training_window(hist[0], hist[1], cutoff, cfg),
            lower=g["yhat_lower"].to_numpy(),
            upper=g["yhat_upper"].to_numpy(),
            seasonal_period=period,
        )
    return oof_rows, panels


def _oof_row(
    run_id: str,
    ts_id: str,
    model_name: str,
    fold_id: int,
    row: Mapping[Any, Any],
    created_at: Any = None,
) -> dict[str, Any]:
    """Assemble one ``backtest_oof`` row from a fold's eval-query result row (pure).

    ``row`` is one record of `bigquery_sql.build_eval_query`'s output — the fold forecast joined to
    actuals — so this is the seam where that query's column names become table columns.
    """
    from ..registry.rows import as_date as _as_date

    return {
        "run_id": run_id,
        "ts_id": ts_id,
        "model_type": model_name,
        "fold_id": fold_id,
        "forecast_date": _as_date(row["forecast_date"]),
        "y_true": row["y_true"],
        "yhat": row["yhat"],
        # `build_eval_query` has always selected these — ML.FORECAST returns them and the metric
        # panel above is already scored on them — and this row assembler dropped them on the way
        # to the table.
        # One arm, said twice. BigQuery ML applies no bias correction, so `yhat_raw` and
        # `yhat_adjusted` are both the forecast it returned. Copying the value rather than leaving
        # the pair NULL keeps `y_true - yhat_raw` a valid residual for every engine, which is what
        # lets a cross-engine calibration diagnostic be one query instead of two.
        "yhat_raw": row.get("yhat_raw", row["yhat"]),
        "yhat_adjusted": row["yhat"],
        "yhat_lower": row.get("yhat_lower"),
        "yhat_upper": row.get("yhat_upper"),
        # The fold's real geometry, not its ordinal. `cutoff_date` is what an ensemble joins on
        # across engines — this path's fold `k` and the Python path's fold `k` are the same window
        # only when every series ends on the same date. See `ensembler._fold_key`.
        "cutoff_date": _as_date(row.get("cutoff_date")),
        "horizon_step": row.get("horizon_step"),
        # The write's clock, so a re-scored fold beats the fold it re-scored on any dedupe-on-read.
        # See `registry.rows.cell_dedup_key`.
        "created_at": created_at,
    }


_SEASONAL_COLS: tuple[str, ...] = (
    "seasonal_period_yearly",
    "seasonal_period_quarterly",
    "seasonal_period_monthly",
    "seasonal_period_weekly",
    "seasonal_period_daily",
)


def extract_bqml_explanations(
    explain_df: pd.DataFrame,
    *,
    ts_id_col: str = "ts_id",
    xreg_cols: Sequence[str] = (),
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], str]]:
    """Extract Tier 1 and Tier 2 feature attributions from an ``ML.EXPLAIN_FORECAST`` frame (pure).

    * **Tier 1 (`tier1_by_ts`):** per-series normalized mean absolute attribution across ``trend``,
      ``seasonal`` (sum of ``seasonal_period_*``), ``holiday_effect``, and each exogenous column in
      ``xreg_cols`` (``attribution_<col>``). Stored in
      ``forecast_metadata.fit_diagnostics["feature_attributions"]``.
    * **Tier 2 (`tier2_by_ts_date`):** per-step JSON string ``{"baseline_score": float,
      "attributions": {...}}`` keyed by ``(ts_id, forecast_date_iso)`` for
      ``forecast_predictions.explanations``.
    """
    import numpy as np
    import pandas as pd

    if explain_df.empty or ts_id_col not in explain_df.columns:
        return {}, {}

    tier1_by_ts: dict[str, dict[str, float]] = {}
    tier2_by_ts_date: dict[tuple[str, str], str] = {}

    for ts_id_raw, g in explain_df.groupby(ts_id_col):
        ts_id = str(ts_id_raw)
        n = len(g)
        if n == 0:
            continue
        trend_arr = (
            np.nan_to_num(g["trend"].to_numpy(dtype=float), nan=0.0)
            if "trend" in g.columns
            else np.zeros(n, dtype=float)
        )
        seas_arr = np.zeros(n, dtype=float)
        for scol in _SEASONAL_COLS:
            if scol in g.columns:
                seas_arr = seas_arr + np.nan_to_num(g[scol].to_numpy(dtype=float), nan=0.0)
        hol_arr = (
            np.nan_to_num(g["holiday_effect"].to_numpy(dtype=float), nan=0.0)
            if "holiday_effect" in g.columns
            else np.zeros(n, dtype=float)
        )
        comp_arrays: dict[str, np.ndarray] = {
            "trend": trend_arr,
            "seasonal": seas_arr,
        }
        if np.any(np.abs(hol_arr) > 0.0) or "holiday_effect" in g.columns:
            comp_arrays["holiday_effect"] = hol_arr
        for col in xreg_cols:
            acol = f"attribution_{col}"
            if acol in g.columns:
                comp_arrays[col] = np.nan_to_num(g[acol].to_numpy(dtype=float), nan=0.0)
            else:
                comp_arrays[col] = np.zeros(n, dtype=float)

        mean_abs = {k: float(np.mean(np.abs(arr))) for k, arr in comp_arrays.items()}
        total_abs = sum(mean_abs.values())
        if total_abs > 0.0:
            tier1_by_ts[ts_id] = {
                k: round(v / total_abs, 6) for k, v in mean_abs.items() if v > 0.0 or k in xreg_cols
            }
        else:
            tier1_by_ts[ts_id] = dict.fromkeys(comp_arrays, 0.0)

        if "time_series_timestamp" in g.columns:
            ts_dates = [
                d.date().isoformat() if hasattr(d, "date") else str(d)[:10]
                for d in pd.to_datetime(g["time_series_timestamp"])
            ]
            for idx, ds_iso in enumerate(ts_dates):
                baseline = round(float(trend_arr[idx] + seas_arr[idx]), 6)
                attrs = {k: round(float(arr[idx]), 6) for k, arr in comp_arrays.items()}
                tier2_by_ts_date[(ts_id, ds_iso)] = json.dumps(
                    {"baseline_score": baseline, "attributions": attrs},
                    sort_keys=True,
                )

    return tier1_by_ts, tier2_by_ts_date


def extract_arima_diagnostics(
    eval_df: pd.DataFrame,
    coef_df: pd.DataFrame | None = None,
    *,
    ts_id_col: str = "ts_id",
) -> dict[str, dict[str, Any]]:
    """Extract per-series diagnostics from ``ML.ARIMA_EVALUATE`` and ``ML.ARIMA_COEFFICIENTS``.

    Populates ``non_seasonal_order`` (``[p, d, q]``), ``has_drift``, ``aic``, ``log_likelihood``,
    ``variance``, ``seasonal_periods``, ``has_holiday_effect``, ``has_spikes_and_dips``,
    ``has_step_changes``, ``ar_coefficients``, ``ma_coefficients``, ``intercept_or_drift``, and
    ``xreg_weights`` (for ``ARIMA_PLUS_XREG`` numeric and categorical regressors).
    """
    import numpy as np
    import pandas as pd

    out: dict[str, dict[str, Any]] = {}
    if not eval_df.empty and ts_id_col in eval_df.columns:
        for _, row in eval_df.iterrows():
            ts_id = str(row[ts_id_col])
            entry: dict[str, Any] = {}
            p = row.get("non_seasonal_p")
            d = row.get("non_seasonal_d")
            q = row.get("non_seasonal_q")
            if (
                p is not None
                and d is not None
                and q is not None
                and not (pd.isna(p) or pd.isna(d) or pd.isna(q))
            ):
                entry["non_seasonal_order"] = [int(p), int(d), int(q)]
            for bkey in (
                "has_drift",
                "has_holiday_effect",
                "has_spikes_and_dips",
                "has_step_changes",
            ):
                bval = row.get(bkey)
                if bval is not None and not pd.isna(bval):
                    entry[bkey] = bool(bval)
            for fkey, out_key in (
                ("AIC", "aic"),
                ("aic", "aic"),
                ("log_likelihood", "log_likelihood"),
                ("variance", "variance"),
            ):
                fval = row.get(fkey)
                if fval is not None and not pd.isna(fval) and np.isfinite(float(fval)):
                    entry[out_key] = round(float(fval), 6)
            s_periods = row.get("seasonal_periods")
            if (
                s_periods is not None
                and hasattr(s_periods, "__iter__")
                and not isinstance(s_periods, (str, bytes))
            ):
                entry["seasonal_periods"] = [str(s) for s in s_periods if s is not None]
            err_msg = row.get("error_message")
            if err_msg is not None and not pd.isna(err_msg) and str(err_msg).strip():
                entry["error_message"] = str(err_msg).strip()
            out[ts_id] = entry

    if coef_df is not None and not coef_df.empty and ts_id_col in coef_df.columns:
        for ts_id_raw, g in coef_df.groupby(ts_id_col):
            ts_id = str(ts_id_raw)
            entry = out.setdefault(ts_id, {})
            xreg_weights: dict[str, Any] = {}
            for _, crow in g.iterrows():
                proc_in = crow.get("processed_input")
                is_arima_row = proc_in is None or (
                    not isinstance(proc_in, (list, np.ndarray)) and pd.isna(proc_in)
                )
                if is_arima_row:
                    for ckey in ("ar_coefficients", "ma_coefficients"):
                        cval = crow.get(ckey)
                        if (
                            cval is not None
                            and hasattr(cval, "__iter__")
                            and not isinstance(cval, (str, bytes))
                        ):
                            entry[ckey] = [
                                round(float(x), 6)
                                for x in cval
                                if x is not None and np.isfinite(float(x))
                            ]
                    idrift = crow.get("intercept_or_drift")
                    if idrift is not None and not pd.isna(idrift) and np.isfinite(float(idrift)):
                        entry["intercept_or_drift"] = round(float(idrift), 6)
                else:
                    feat_name = str(proc_in)
                    w = crow.get("weight")
                    if w is not None and not pd.isna(w) and np.isfinite(float(w)):
                        xreg_weights[feat_name] = round(float(w), 6)
                    else:
                        cat_w = crow.get("category_weights")
                        if (
                            cat_w is not None
                            and hasattr(cat_w, "__iter__")
                            and not isinstance(cat_w, (str, bytes))
                        ):
                            cat_map: dict[str, float] = {}
                            for item in cat_w:
                                if (
                                    isinstance(item, Mapping)
                                    and "category" in item
                                    and "weight" in item
                                    and item["weight"] is not None
                                    and np.isfinite(float(item["weight"]))
                                ):
                                    cat_map[str(item["category"])] = round(float(item["weight"]), 6)
                            if cat_map:
                                xreg_weights[feat_name] = cat_map
            if xreg_weights:
                entry["xreg_weights"] = xreg_weights

    return out


def _meta_row(
    run_id: str,
    ts_id: str,
    model_name: str,
    panel: dict[str, float],
    best_params: str,
    created_at: Any,
    cfg: RunConfig,
    n_folds_achieved: int | None = None,
    first_val_date: Any = None,
    last_val_date: Any = None,
    feature_attributions: Mapping[str, float] | None = None,
    arima_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one ``forecast_metadata`` row (``fold_id=NULL``) for a native model (pure).

    The per-cell lanes a Python worker fills — ``worker_id``, the cell timestamps, the harvested
    ``cpu_seconds``/RSS/thread counts — are deliberately absent: a native family has no per-cell
    worker and traces at the ``run_jobs`` grain instead, so those columns are NULL by design rather
    than by omission.

    Metrics go through `registry.rows._as_float` for the same reason the Python worker's do: an
    unscored or non-finite metric must land as NULL, not NaN. This row used to pass the panel
    through raw, which is how a backtest-off native run wrote NaN into columns the Python engines
    wrote NULL into — same table, same run, same meaning, two encodings. NaN also sorts *ahead* of
    every real number in a BigQuery ``ORDER BY wape``, so an unscored native model would head a
    leaderboard it had not competed in. Found live 2026-09-02 in smoke 13.

    ``n_folds_achieved`` is how many folds this series was actually scored on — ``None`` when
    backtesting was off, which is what leaves all three scoring columns NULL. The Python worker
    fills the same three columns, and the whole point of them is that one reader can compare a
    native model against a Python one without knowing which engine wrote the row; a native family
    that left them NULL while backtesting would read as "never asked".

    ``first_val_date`` / ``last_val_date`` are the span this series was scored over, accumulated by
    the caller from the fold rows. They complete the same parity argument one level down: the
    scored-geometry columns are only useful in a cross-engine ``GROUP BY``, so an engine that fills
    some of them and not others is worse than one that fills none — the gaps read as differences.
    """
    from ..features import should_stamp_variant_diagnostics, variant_diagnostics_for
    from ..metrics import METRIC_NAMES
    from ..registry.ids import make_model_hash
    from ..registry.rows import as_date as _as_date
    from ..registry.rows import as_float as _as_float

    status, note = None, None
    if n_folds_achieved is not None:
        requested = cfg.backtest.n_folds
        if n_folds_achieved >= requested:
            status = "full"
        else:
            # No shortfall arithmetic to quote here, unlike the Python path: the fold grid is
            # uniform across the panel, so a native series scores short only because BigQuery ML
            # left it out of a fold's evaluation — the reason lives on the BigQuery side.
            status = "reduced" if n_folds_achieved else "unscored"
            note = f"scored on {n_folds_achieved} of {requested} folds"

    diag_dict: dict[str, Any] = {}
    if should_stamp_variant_diagnostics(model_name, cfg):
        diag_dict.update(variant_diagnostics_for(model_name, cfg))
    if feature_attributions:
        diag_dict["feature_attributions"] = dict(feature_attributions)
    if arima_diagnostics:
        diag_dict["arima_diagnostics"] = dict(arima_diagnostics)
    fit_diag: str | None = json.dumps(diag_dict, sort_keys=True) if diag_dict else None

    return {
        "run_id": run_id,
        "ts_id": ts_id,
        "model_type": model_name,
        "compute_engine": "bigquery",
        "model_hash": make_model_hash(run_id, str(ts_id), model_name, cfg),
        "fold_id": None,
        **{name: _as_float(panel[name]) for name in METRIC_NAMES},
        "fit_seconds": None,
        "best_params": best_params,
        "model_artifact": None,
        "created_at": created_at,
        "backtest_status": status,
        "backtest_note": note,
        "n_folds_achieved": n_folds_achieved,
        # A native row exists only because the model was built and forecast, so "ok" is a fact
        # rather than an assumption. Filled even though `error_class`/`error_detail` are not: a
        # native failure takes the whole job down and writes no rows at all, so there is never a
        # native error row to describe — but leaving `cell_status` NULL would make
        # `WHERE cell_status = 'ok'` quietly skip every native model in the run.
        "cell_status": "ok",
        "error_class": None,
        "error_detail": None,
        # BQML returns `prediction_interval_lower_bound` / `_upper_bound` from the fitted model,
        # so a native cell's bounds are always the model's own. Filled for the same reason the
        # Python path fills it: an interval-metric comparison between the two engines is only
        # like-for-like if a reader can see which kind of interval each row was scored on.
        "interval_source": "native",
        # The native path ships BigQuery ML's own point forecast, uncorrected, and BigQuery ML's own
        # band, uncalibrated. Both columns say so rather than reading NULL, because NULL here would
        # be indistinguishable from "an older run wrote this row before the columns existed", and
        # the whole use of these columns is a fleet-wide GROUP BY that mixes engines. There is no
        # margin to report: a comparison needs two arms and this engine produces one.
        "point_forecast_source": "raw",
        # Not "configured": nothing chose this, the engine has no other arm to offer. Saying
        # `configured` would imply an `output.point_forecast` setting was honoured here, and a
        # run set to `auto` would then look as though the native models had opted out of selection
        # rather than never having been eligible for it.
        "point_forecast_decision": "engine-native",
        "interval_calibration": "native",
        "point_forecast_margin": None,
        # BigQuery ML rebuilds the model for every fold's cutoff — `CREATE MODEL` is the only way
        # to fit it — so the native path is `per_fold` by construction, whatever `backtest.scheme`
        # the config names. Stated rather than left NULL for the same reason as `cell_status`: a
        # reader grouping the leaderboard by `backtest_refit` would otherwise drop every native
        # model out of the comparison. There is no frozen arm to compare against, so no gap.
        "backtest_refit": "per_fold" if n_folds_achieved is not None else None,
        "staleness_gap": None,
        # The scored geometry. There is no `short_series` policy on this path — `fold_plan` steps a
        # single global cutoff back uniformly — so the effective step and training floor are the
        # authored ones, and saying so is a fact rather than an assumption. They are filled for the
        # same reason `backtest_refit` above is: a reader grouping a leaderboard by scored geometry
        # would otherwise drop every native model out of the comparison, which is the one thing
        # these columns exist to make possible.
        "achieved_step": cfg.backtest.step if n_folds_achieved else None,
        "achieved_min_train": cfg.backtest.min_train if n_folds_achieved else None,
        "first_val_date": _as_date(first_val_date),
        "last_val_date": _as_date(last_val_date),
        "fit_diagnostics": fit_diag,
    }


def _append_rows(  # pragma: no cover - GCP I/O, @gcp smoke
    settings: Settings,
    table: str,
    spec: tuple[tuple[str, str], ...],
    rows: list[dict[str, Any]],
) -> None:
    """Append plain row dicts to a cell table via the registry's Storage Write API path.

    Reuses the same ``_proto_for`` / ``_encode_rows`` / ``_append_via_write_api`` machinery that
    `registry.cells.write_cells` uses — the ``CellResult`` requirement lives only in the
    ``assemble_*`` wrappers, not the write path, so the native engine feeds ``_*_SPEC``-shaped dicts
    directly. Empty input is a no-op.
    """
    from ..registry.write_api import (
        append_via_write_api as _append_via_write_api,
    )
    from ..registry.write_api import (
        encode_rows as _encode_rows,
    )
    from ..registry.write_api import (
        get_write_client,
    )
    from ..registry.write_api import (
        proto_for as _proto_for,
    )

    if not rows:
        return
    write_client = get_write_client()
    msg_cls, proto_descriptor = _proto_for(table, spec)
    serialized = _encode_rows(msg_cls, spec, rows)
    _append_via_write_api(
        write_client,
        settings.project_id,
        settings.registry_dataset_id,
        table,
        proto_descriptor,
        serialized,
    )


# --- CLI -----------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    """``python -m scale_forecasting.engines.bigquery_engine --config path.json``.

    Loads the config, routes its models to the BigQuery subset via
    `split_by_runtime`, and runs the engine on that subset. A config
    with no native models is a no-op (the Python runtime owns the rest).
    """
    import argparse

    from ..config import load_config
    from ..errors import configure_cli_logging, get_logger, require_extra
    from ..router import split_by_runtime

    configure_cli_logging()
    require_extra("gcp", purpose="The BigQuery-native engine")
    parser = argparse.ArgumentParser(description="Run the BigQuery-native forecasting models.")
    parser.add_argument("--config", required=True, help="Path to the run config JSON.")
    args = parser.parse_args(argv)

    log = get_logger(__name__)
    cfg = load_config(args.config)
    _, bq_models = split_by_runtime(cfg)
    if not bq_models:
        log.warning("no BigQuery-native models in config %s; nothing to run", args.config)
        return
    run(cfg, bq_models)


if __name__ == "__main__":  # pragma: no cover
    _main()
