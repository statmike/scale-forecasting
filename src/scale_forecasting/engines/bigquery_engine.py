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

if TYPE_CHECKING:
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
      ``fold_id``s, and rolls the per-fold panels up (via ``worker._rollup_metrics``) into a
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

    from google.cloud import bigquery

    from ..errors import RegistryError, get_logger
    from ..metrics import METRIC_NAMES
    from ..registry.ids import make_run_id
    from ..registry.lifecycle import run_header
    from ..registry.write_api import _META_SPEC, _OOF_SPEC
    from ..settings import Settings
    from ..worker import _rollup_metrics
    from .bigquery_sql import (
        bqml_options,
        build_eval_query,
        build_fold_create_statements,
        build_fold_drop_statements,
        build_history_query,
        build_series_ids_query,
        build_setup_statements,
        fold_plan,
    )

    log = get_logger(__name__)
    settings = settings or Settings.resolve()
    run_id = make_run_id(cfg)
    # Source reads resolve against `dataset`; model objects and forecast_predictions land in
    # `registry_dataset`. Identical strings unless the deployment split them (`_registry_of`).
    dataset = settings.dataset_ref
    registry_dataset = settings.registry_dataset_ref
    client = bigquery.Client(project=settings.project_id)
    log.info(
        "bigquery run start: run_id=%s models=%s manage_header=%s backtest=%s",
        run_id,
        models,
        manage_header,
        cfg.backtest.enabled,
    )

    def _query(sql: str) -> Any:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
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
        # sources — see CONSIDERATIONS.md). So the native subset reads un-pinned (live), keeping its
        # reads internally consistent (all un-pinned) rather than mixing pinned SELECTs with an
        # un-pinnable CREATE MODEL. Safe because a run's source data is static for its duration.
        snapshot_millis = None
        series_ids = [
            str(r.ts_id)
            for r in _query(build_series_ids_query(cfg, dataset, snapshot_millis=snapshot_millis))
        ]
        n_series = len(series_ids)
        try:
            # --- Phase 1: final true-future forecast → forecast_predictions (always) ----------
            for model_name in models:
                for stmt in build_setup_statements(
                    cfg,
                    model_name,
                    dataset,
                    registry_dataset=registry_dataset,
                    snapshot_millis=snapshot_millis,
                ):
                    _query(stmt)

            # --- Phase 2: scored evaluation ---------------------------------------------------
            oof_rows: list[dict[str, Any]] = []
            meta_rows: list[dict[str, Any]] = []

            if cfg.backtest.enabled:
                history = _query(
                    build_history_query(cfg, dataset, snapshot_millis=snapshot_millis)
                ).to_dataframe()
                hist_by_id = {tid: g["y"].to_numpy() for tid, g in history.groupby("ts_id")}
                plan = fold_plan(cfg)
                for model_name in models:
                    best_params = json.dumps(bqml_options(cfg, model_name), sort_keys=True)
                    panels_by_ts: dict[str, list[dict[str, float]]] = {}
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
                        )
                        oof_rows.extend(fold_oof)
                        for ts_id, panel in fold_panels.items():
                            panels_by_ts.setdefault(ts_id, []).append(panel)
                    for ts_id, panels in panels_by_ts.items():
                        rolled = _rollup_metrics(panels)
                        meta_rows.append(
                            _meta_row(
                                run_id, ts_id, model_name, rolled, best_params, created_at, cfg
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
                                run_id, ts_id, model_name, nan_panel, best_params, created_at, cfg
                            )
                        )

            _append_rows(settings, "backtest_oof", _OOF_SPEC, oof_rows)
            _append_rows(settings, "forecast_metadata", _META_SPEC, meta_rows)
        except Exception as exc:  # noqa: BLE001 - run_header records FAILED as this propagates
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
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """One fold's eval frame → its ``backtest_oof`` rows and one metric panel per series (pure).

    This is where a native model *earns its leaderboard number*, so it is worth having out of the
    GCP body. Three things it has to get right, none of which a live run would complain about if it
    got them wrong — it would just report different metrics:

    * the rows are sorted by ``forecast_date`` before scoring, so a horizon-weighted metric sees the
      horizon in order;
    * ``y_train`` comes from the series' own history (the scale denominator MASE and RMSSE divide
      by), looked up per series rather than shared;
    * the interval bounds are passed through, so ``coverage`` and ``pinball`` are real numbers here
      rather than the NaNs the Python worker's OOF path produces.

    Panels are keyed by ``str(ts_id)`` to match `_meta_row`'s key type — the caller accumulates one
    list per series across folds and rolls them up exactly as `worker._rollup_metrics` does.
    """
    from ..metrics import compute_metrics

    oof_rows: list[dict[str, Any]] = []
    panels: dict[str, dict[str, float]] = {}
    for ts_id, g in eval_df.groupby("ts_id"):
        g = g.sort_values("forecast_date")
        for _, row in g.iterrows():
            oof_rows.append(_oof_row(run_id, str(ts_id), model_name, fold_id, row))
        panels[str(ts_id)] = compute_metrics(
            g["y_true"].to_numpy(),
            g["yhat"].to_numpy(),
            y_train=hist_by_id.get(ts_id),
            lower=g["yhat_lower"].to_numpy(),
            upper=g["yhat_upper"].to_numpy(),
        )
    return oof_rows, panels


def _oof_row(
    run_id: str, ts_id: str, model_name: str, fold_id: int, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble one ``backtest_oof`` row from a fold's eval-query result row (pure).

    ``row`` is one record of `bigquery_sql.build_eval_query`'s output — the fold forecast joined to
    actuals — so this is the seam where that query's column names become table columns.
    """
    return {
        "run_id": run_id,
        "ts_id": ts_id,
        "model_type": model_name,
        "fold_id": fold_id,
        "forecast_date": row["forecast_date"],
        "y_true": row["y_true"],
        "yhat": row["yhat"],
    }


def _meta_row(
    run_id: str,
    ts_id: str,
    model_name: str,
    panel: dict[str, float],
    best_params: str,
    created_at: Any,
    cfg: RunConfig,
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
    """
    from ..metrics import METRIC_NAMES
    from ..registry.ids import make_model_hash
    from ..registry.rows import _as_float

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
    from google.cloud import bigquery_storage_v1

    from ..registry.write_api import _append_via_write_api, _encode_rows, _proto_for

    if not rows:
        return
    write_client = bigquery_storage_v1.BigQueryWriteClient()
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
    from ..errors import get_logger
    from ..router import split_by_runtime

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
