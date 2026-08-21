"""BigQuery-native models — SQL runner for ARIMA_PLUS / TimesFM.

The BigQuery runtime executes forecasting *as SQL inside BigQuery* — the opposite of the Spark
track's per-cell fan-out. ``ARIMA_PLUS`` with ``time_series_id_col`` trains **all series in one
``CREATE MODEL`` statement``; ``AI.FORECAST`` (TimesFM) forecasts every series in one call with no
training at all. Both land in the *same* three-tier registry as the Python models, so a native model
and a Spark model are directly comparable on ``v_model_leaderboard``.

This module has two halves:

* **Pure SQL builders** (``build_*`` / ``render_*``) — deterministic string renderers: a ``dataset``
  argument defaulting to a ``{dataset}`` template token so a config-only call renders, ``@run_id``
  bound as a **query parameter**, and identifiers (dataset, columns, model names) interpolated.
  Snapshot-tested offline, no GCP.
* **The engine** (`run`) — resolves `Settings`, owns the
  ``run_registry`` header lifecycle exactly like `spark_explode.run`, executes the builders'
  SQL via ``bigquery.Client``, reads the fold forecasts back, computes the metric panel through
  the shared `compute_metrics` (no formula drift), and writes all
  three cell tables via the registry's Storage Write API row-dict path.

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

Public surface: ``run(cfg, models)``; builders ``build_create_model_sql``,
``build_forecast_insert_sql``, ``build_eval_query``, ``build_history_query``,
``build_series_ids_query``, ``build_custom_holiday_cte``, ``build_fold_create_statements``,
``build_fold_drop_statements``, ``render_setup_sql``, ``bqml_options``, ``fold_plan``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..features import holiday_frame
from ..registry.ids import make_run_id

if TYPE_CHECKING:
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


# --- constants -----------------------------------------------------------------

# 80% prediction interval → matches the Python models' 0.1/0.9 quantile bounds (worker.predict),
# so coverage/pinball are computed on the same interval width across runtimes.
_CONFIDENCE_LEVEL = 0.8

# Symmetric one-step window around each custom holiday (BQML defaults; small + generic).
_PREHOLIDAY_DAYS = 1
_POSTHOLIDAY_DAYS = 1

# BQML model_type per native model name. TimesFM has no CREATE MODEL (AI.FORECAST is serverless).
_MODEL_TYPE: dict[str, str] = {
    "arima_plus": "ARIMA_PLUS",
}

# pandas offset alias → (BQML data_frequency, DATE_SUB INTERVAL unit). Covers the common cadences;
# anything else falls back to daily, which is the shipped generator's cadence.
_FREQ_MAP: dict[str, tuple[str, str]] = {
    "D": ("DAILY", "DAY"),
    "W": ("WEEKLY", "WEEK"),
    "M": ("MONTHLY", "MONTH"),
    "MS": ("MONTHLY", "MONTH"),
    "H": ("HOURLY", "HOUR"),
    "Q": ("QUARTERLY", "QUARTER"),
    "Y": ("YEARLY", "YEAR"),
    "A": ("YEARLY", "YEAR"),
}


# --- small helpers -------------------------------------------------------------


def _freq(cfg: RunConfig) -> tuple[str, str]:
    """Return ``(data_frequency, interval_unit)`` for the run's cadence (defaults to daily)."""
    return _FREQ_MAP.get(cfg.data.freq.upper(), ("DAILY", "DAY"))


def _sanitize_identifier(text: str) -> str:
    """Coerce arbitrary text into a valid BigQuery identifier fragment.

    Custom holiday names must be valid column names — no spaces — because ``ML.EXPLAIN_FORECAST``
    surfaces them as columns. Model object names must likewise be plain identifiers, so the
    hyphenated ``run_id`` is folded here too. Non-alphanumerics collapse to ``_``.
    """
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return out.strip("_") or "x"


def _source_ref(cfg: RunConfig, dataset: str) -> str:
    """Fully-qualify the source table: pass through a dotted name, else qualify against ``dataset``.

    Mirrors ``spark_io._resolve_source_table`` so both runtimes read the identical table.
    """
    src = cfg.data.source_table
    return src if "." in src else f"{dataset}.{src}"


def _model_ref(cfg: RunConfig, model_name: str, dataset: str, *, fold_id: int | None = None) -> str:
    """The backtick-quoted BQML model object path for one ``(model, run[, fold])`` (persisted).

    The object name embeds the config-pinned ``run_id`` so re-running the same config targets the
    *same* model (``CREATE OR REPLACE`` is idempotent) while a different config gets a distinct
    object. ``fold_id`` (when set) appends an ``_f{k}`` suffix so each backtest fold trains its own
    model object without clobbering the final (true-future) model or the other folds. A model object
    name is an identifier — it cannot be a bound query parameter — so the ``run_id`` is interpolated
    here; it is a pure function of ``cfg`` (`make_run_id`), keeping every builder pure.
    """
    run_id = make_run_id(cfg)
    suffix = f"_f{fold_id}" if fold_id is not None else ""
    return f"`{dataset}.sf_model_{model_name}_{_sanitize_identifier(run_id)}{suffix}`"


def _series_filter(cfg: RunConfig, source: str, id_expr: str) -> str:
    """A ``ts_id IN (...)`` fragment for the deterministic ``series_limit`` subset (``""`` if none).

    Same rule as ``spark_io._limit_series``: distinct ids → ordered → first N. ``id_expr`` is the
    column reference to constrain (bare ``ts_id`` in a scan, ``s.ts_id`` under a join alias).
    """
    limit = cfg.data.series_limit
    if limit is None:
        return ""
    idc = cfg.data.ts_id_col
    return (
        f"{id_expr} IN (SELECT {idc} FROM `{source}` GROUP BY {idc} ORDER BY {idc} LIMIT {limit})"
    )


def _cutoff_expr(cfg: RunConfig, source: str, back_steps: int) -> str:
    """Scalar subquery for a training cutoff date: ``MAX(ds) - back_steps`` over the source.

    ``back_steps`` is the number of cadence units to step back from the last observed date, so the
    model trains on ``ds <= cutoff`` and is scored on the ``horizon`` window after it. A single
    global cutoff (not per-series) keeps every series' fold window aligned — the shipped data shares
    one date span, so this is exact; note it if your series end on different dates.
    """
    _, unit = _freq(cfg)
    return (
        f"(SELECT DATE_SUB(MAX({cfg.data.date_col}), INTERVAL {back_steps} {unit}) FROM `{source}`)"
    )


def fold_plan(cfg: RunConfig) -> list[tuple[int, int]]:
    """The backtest folds as ``[(fold_id, back_steps)]`` — pure, mirrors ``backtest.make_folds``.

    ``make_folds`` anchors folds from the end: fold ``k``'s validation window starts at position
    ``n - horizon - (n_folds - 1 - k) * step``. In date space anchored on ``MAX(ds)`` that makes
    the last *training* date ``MAX(ds) - back_steps`` where
    ``back_steps = horizon + (n_folds-1-k)*step`` — independent of each series' length, so this is a
    pure function of ``cfg.backtest``. ``fold_id`` ordering matches ``make_folds`` (fold 0 is the
    earliest / largest step-back), so native and Python OOF fold ids line up. The per-series
    min-train feasibility guard ``make_folds`` enforces is *not* replicated in SQL (BQML trains on
    whatever history precedes the cutoff); series too short for a fold simply train on less.
    """
    bt = cfg.backtest
    return [(k, bt.horizon + (bt.n_folds - 1 - k) * bt.step) for k in range(bt.n_folds)]


# --- pure SQL builders ---------------------------------------------------------


def bqml_options(cfg: RunConfig, model_name: str) -> dict[str, Any]:
    """The resolved model parameters, as an ordered dict — one source of truth for two uses.

    For the ARIMA models this is the ``CREATE MODEL`` ``OPTIONS(...)`` body *and* the
    ``best_params`` JSON on each ``forecast_metadata`` row — the registry records what trained.
    TimesFM has no ``CREATE MODEL``; `_render_options` never sees its dict, but ``run`` still
    stamps ``best_params`` for every model, so we return the resolved ``AI.FORECAST`` arguments here
    — keeping the metadata row's provenance non-NULL and meaningful across both native shapes.
    """
    freq, _ = _freq(cfg)
    if model_name not in _MODEL_TYPE:  # timesfm — serverless AI.FORECAST, no OPTIONS clause
        return {
            "model_type": "TimesFM (AI.FORECAST)",
            "id_cols": [cfg.data.ts_id_col],
            "timestamp_col": cfg.data.date_col,
            "data_col": cfg.data.target_col,
            "horizon": cfg.data.horizon,
            "confidence_level": _CONFIDENCE_LEVEL,
        }
    opts: dict[str, Any] = {
        "model_type": _MODEL_TYPE[model_name],
        "time_series_id_col": cfg.data.ts_id_col,
        "time_series_timestamp_col": cfg.data.date_col,
        "time_series_data_col": cfg.data.target_col,
        "horizon": cfg.data.horizon,
        "data_frequency": freq,
    }
    return opts


def _render_options(opts: dict[str, Any]) -> str:
    """Render an options dict to the ``OPTIONS(...)`` body (strings quoted, numbers bare)."""
    lines = []
    for key, value in opts.items():
        rendered = f"'{value}'" if isinstance(value, str) else str(value)
        lines.append(f"  {key} = {rendered}")
    return "OPTIONS(\n" + ",\n".join(lines) + "\n)"


def build_custom_holiday_cte(cfg: RunConfig) -> str:
    """Render the ``custom_holiday`` CTE from `features.holiday_frame`, or ``""`` if none.

    Emits one row per (holiday, occurrence) — ``region``, sanitized ``holiday_name``,
    ``primary_date``, and a symmetric ``preholiday_days`` / ``postholiday_days`` window — via a
    ``UNNEST([STRUCT(...), ...])`` literal. ``region`` is the joined ISO code(s) so the model treats
    these as *the* holiday calendar (no built-in ``holiday_region`` option, matching the Python
    suite which models only the configured countries). BQML only applies holidays that fall inside a
    daily/weekly training span longer than a year, so the generous calendar window is self-trimming.
    """
    codes = cfg.features.holidays
    if not codes:
        return ""
    frame = holiday_frame(cfg)
    if frame.empty:
        return ""
    region = _sanitize_identifier("_".join(codes))
    rows = []
    for ds, name in zip(frame["ds"], frame["holiday"], strict=True):
        holiday_name = _sanitize_identifier(str(name))
        primary = ds.date().isoformat()
        rows.append(
            f"    STRUCT('{region}' AS region, '{holiday_name}' AS holiday_name, "
            f"DATE '{primary}' AS primary_date, "
            f"{_PREHOLIDAY_DAYS} AS preholiday_days, {_POSTHOLIDAY_DAYS} AS postholiday_days)"
        )
    return "custom_holiday AS (\n  SELECT * FROM UNNEST([\n" + ",\n".join(rows) + "\n  ])\n)"


def _train_window_where(cfg: RunConfig, source: str, back_steps: int | None) -> list[str]:
    """The training-window date predicates for a model fit (``[]`` = train on all history).

    ``back_steps=None`` is the **final, true-future** fit: no date bound, train on everything.
    Otherwise the fit is a backtest fold: ``ds <= cutoff`` (expanding), plus a lower
    ``ds > cutoff - min_train`` bound for the sliding scheme so the window is fixed-width —
    mirroring ``backtest.make_folds``'s ``expanding`` vs ``sliding`` ``train_start``.
    """
    if back_steps is None:
        return []
    datec = cfg.data.date_col
    conds = [f"{datec} <= {_cutoff_expr(cfg, source, back_steps)}"]
    if cfg.backtest.scheme == "sliding":
        lower = _cutoff_expr(cfg, source, back_steps + cfg.backtest.min_train)
        conds.append(f"{datec} > {lower}")
    return conds


def _training_select(cfg: RunConfig, source: str, *, back_steps: int | None = None) -> str:
    """The ``training_data`` SELECT: id/timestamp/target over the fit's training window.

    ``back_steps=None`` trains on **all** history (the final true-future fit); an int restricts to a
    backtest fold's window (see `_train_window_where`).
    """
    cols = [cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col]
    where = _train_window_where(cfg, source, back_steps)
    sfilter = _series_filter(cfg, source, cfg.data.ts_id_col)
    if sfilter:
        where.append(sfilter)
    clause = f"\n  WHERE {' AND '.join(where)}" if where else ""
    return f"SELECT {', '.join(cols)}\n  FROM `{source}`{clause}"


def build_create_model_sql(
    cfg: RunConfig,
    model_name: str,
    dataset: str = "{dataset}",
    *,
    back_steps: int | None = None,
    fold_id: int | None = None,
) -> str:
    """``CREATE OR REPLACE MODEL`` for an ARIMA_PLUS native model.

    ``back_steps=None`` (default) is the **final** model: trained on *all* history so its forecast
    is a true beyond-data forecast (parity with the Spark final fit). A ``back_steps``/``fold_id``
    pair builds a **backtest fold** model — trained on ``ds <= cutoff`` — into a fold-suffixed
    object.
    When holidays are configured the ``AS`` clause takes the named-subquery form
    (``training_data AS (...), custom_holiday AS (...)``); otherwise it is a plain training query.
    TimesFM has no CREATE MODEL — see `build_forecast_insert_sql`.
    """
    ref = _model_ref(cfg, model_name, dataset, fold_id=fold_id)
    source = _source_ref(cfg, dataset)
    options = _render_options(bqml_options(cfg, model_name))
    training = _training_select(cfg, source, back_steps=back_steps)
    holiday_cte = build_custom_holiday_cte(cfg)
    if holiday_cte:
        body = f"  training_data AS (\n    {training}\n  ),\n  {holiday_cte}"
        as_clause = f"AS (\n{body}\n)"
    else:
        as_clause = f"AS (\n  {training}\n)"
    return f"CREATE OR REPLACE MODEL {ref}\n{options}\n{as_clause};"


def _forecast_source(
    cfg: RunConfig,
    model_name: str,
    dataset: str,
    *,
    back_steps: int | None = None,
    fold_id: int | None = None,
    horizon: int | None = None,
) -> str:
    """The ``ML.FORECAST`` / ``AI.FORECAST`` table expression producing a forecast.

    ``back_steps=None`` is the **final true-future** forecast (from the all-history model /
    all-history TimesFM history); an int is a **backtest fold** forecast (from the fold model /
    ``ds <= cutoff`` TimesFM history). ``horizon`` defaults to ``data.horizon`` for the final
    forecast and ``backtest.horizon`` for a fold. Both yield ``forecast_timestamp`` /
    ``forecast_value`` / ``prediction_interval_{lower,upper}_bound`` plus the id column.
    """
    source = _source_ref(cfg, dataset)
    idc, datec, targetc = cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col
    h = (
        horizon
        if horizon is not None
        else (cfg.data.horizon if back_steps is None else cfg.backtest.horizon)
    )

    if model_name == "timesfm":
        where = _train_window_where(cfg, source, back_steps)
        sfilter = _series_filter(cfg, source, idc)
        if sfilter:
            where.append(sfilter)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        inner = f"SELECT {idc}, {datec}, {targetc} FROM `{source}`{clause}"
        return (
            "AI.FORECAST(\n"
            f"    ({inner}),\n"
            f"    data_col => '{targetc}',\n"
            f"    timestamp_col => '{datec}',\n"
            f"    id_cols => ['{idc}'],\n"
            f"    horizon => {h},\n"
            f"    confidence_level => {_CONFIDENCE_LEVEL})"
        )

    ref = _model_ref(cfg, model_name, dataset, fold_id=fold_id)
    struct = f"STRUCT({h} AS horizon, {_CONFIDENCE_LEVEL} AS confidence_level)"
    return f"ML.FORECAST(MODEL {ref}, {struct})"


# Columns written to forecast_predictions, in DDL order (shared by the INSERT).
_PRED_COLS = (
    "run_id, ts_id, model_type, compute_engine, forecast_date, "
    "yhat, yhat_lower, yhat_upper, quantiles"
)


def build_forecast_insert_sql(cfg: RunConfig, model_name: str, dataset: str = "{dataset}") -> str:
    """``INSERT INTO forecast_predictions`` the **true beyond-data** forecast for one native model.

    The showpiece statement: one query forecasts every series' next ``data.horizon`` steps from a
    model fit on *all* history and lands canonical prediction rows with
    ``compute_engine='bigquery'`` — directly comparable to the Spark models' true-future rows.
    ``forecast_timestamp`` → ``DATE()`` → ``forecast_date``; the interval bounds map to
    ``yhat_lower`` / ``yhat_upper``;
    ``quantiles`` is NULL (native models emit an interval, not an arbitrary quantile set).
    """
    dataset_q = f"`{dataset}.forecast_predictions`"
    forecast = _forecast_source(cfg, model_name, dataset)
    return (
        f"INSERT INTO {dataset_q}\n"
        f"  ({_PRED_COLS})\n"
        f"SELECT\n"
        f"  @run_id, {cfg.data.ts_id_col}, '{model_name}', 'bigquery',\n"
        f"  DATE(forecast_timestamp), forecast_value,\n"
        f"  prediction_interval_lower_bound, prediction_interval_upper_bound, NULL\n"
        f"FROM {forecast};"
    )


def build_eval_query(
    cfg: RunConfig,
    model_name: str,
    dataset: str = "{dataset}",
    *,
    back_steps: int,
    fold_id: int,
) -> str:
    """A read-back ``SELECT`` of a **backtest fold's** forecast joined to actuals (OOF + metrics).

    Returns ``(ts_id, forecast_date, y_true, yhat, yhat_lower, yhat_upper)`` for the fold's
    validation window (the ``backtest.horizon`` dates after ``cutoff``, all of which have ground
    truth). The engine groups these by ``ts_id`` and feeds `metrics.compute_metrics` — so the
    metric math is byte-identical to the Python models. Intervals are carried so coverage/pinball
    are real. ``@run_id`` only names the model here; this query writes no row.
    """
    source = _source_ref(cfg, dataset)
    idc, datec, targetc = cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col
    forecast = _forecast_source(cfg, model_name, dataset, back_steps=back_steps, fold_id=fold_id)
    return (
        f"SELECT\n"
        f"  f.{idc} AS ts_id, DATE(f.forecast_timestamp) AS forecast_date,\n"
        f"  s.{targetc} AS y_true, f.forecast_value AS yhat,\n"
        f"  f.prediction_interval_lower_bound AS yhat_lower,\n"
        f"  f.prediction_interval_upper_bound AS yhat_upper\n"
        f"FROM {forecast} f\n"
        f"JOIN `{source}` s\n"
        f"  ON s.{idc} = f.{idc} AND s.{datec} = DATE(f.forecast_timestamp)\n"
        f"ORDER BY ts_id, forecast_date;"
    )


def build_history_query(cfg: RunConfig, dataset: str = "{dataset}") -> str:
    """A ``SELECT`` of **all** training history ``(ts_id, ds, y)`` (for MASE/RMSSE scale).

    The scale-free metrics need each series' training actuals as ``y_train``; the engine loads this
    once, groups by ``ts_id``, and passes the per-series history to `metrics.compute_metrics`.
    Post-alignment this is the full series history (the natives train on all of it for the final
    forecast) — a robust, freq-agnostic scale for the fold metrics.
    """
    source = _source_ref(cfg, dataset)
    idc, datec, targetc = cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col
    sfilter = _series_filter(cfg, source, idc)
    clause = f"\nWHERE {sfilter}" if sfilter else ""
    return (
        f"SELECT {idc} AS ts_id, {datec} AS ds, {targetc} AS y\n"
        f"FROM `{source}`{clause}\n"
        f"ORDER BY ts_id, ds;"
    )


def build_series_ids_query(cfg: RunConfig, dataset: str = "{dataset}") -> str:
    """A ``SELECT DISTINCT ts_id`` over the source subset — the run's series list (engine-agnostic).

    Feeds the ``n_series`` count and, when backtesting is off, the one unscored ``fold_id=NULL``
    metadata row per ``(series, model)`` (NaN metric panel) that keeps native parity with the Python
    worker's always-emitted metadata row.
    """
    source = _source_ref(cfg, dataset)
    idc = cfg.data.ts_id_col
    sfilter = _series_filter(cfg, source, idc)
    clause = f"\nWHERE {sfilter}" if sfilter else ""
    return f"SELECT DISTINCT {idc} AS ts_id\nFROM `{source}`{clause}\nORDER BY ts_id;"


def build_setup_statements(
    cfg: RunConfig, model_name: str, dataset: str = "{dataset}"
) -> list[str]:
    """The mutating statements for one native model's **final true-future forecast**, in order.

    ARIMA models: ``[CREATE MODEL (all history), INSERT INTO forecast_predictions]``. TimesFM: just
    the INSERT (no training). The backtest fold statements (`build_create_model_sql` with
    ``back_steps``) and the read-back eval (`build_eval_query`) are *not* here — this renders
    only the always-run true-future path.
    """
    statements: list[str] = []
    if model_name in _MODEL_TYPE:
        statements.append(build_create_model_sql(cfg, model_name, dataset))
    statements.append(build_forecast_insert_sql(cfg, model_name, dataset))
    return statements


def build_fold_create_statements(
    cfg: RunConfig, model_name: str, dataset: str, fold_id: int, back_steps: int
) -> list[str]:
    """The training statements for one backtest fold (``[CREATE MODEL]`` for ARIMA, ``[]`` else).

    TimesFM needs no model object — its fold forecast reads the ``ds <= cutoff`` history directly in
    `build_eval_query` — so it has no fold-training statement.
    """
    if model_name in _MODEL_TYPE:
        return [
            build_create_model_sql(cfg, model_name, dataset, back_steps=back_steps, fold_id=fold_id)
        ]
    return []


def build_fold_drop_statements(
    cfg: RunConfig, model_name: str, dataset: str, fold_id: int
) -> list[str]:
    """The cleanup statement for one backtest fold (``[DROP MODEL]`` for ARIMA, ``[]`` else).

    Each fold trains a persisted ``sf_model_{model}_{run_id}_f{k}`` object solely to produce that
    fold's held-out forecast; once `build_eval_query` has read it back the object has no
    further use. Without this it would linger in the dataset — orphaned fold models accumulating
    every run. The *final* true-future model (``fold_id=None``) is deliberately **not** dropped: it
    backs ``forecast_predictions`` and its ``CREATE OR REPLACE`` idempotency (lineage).
    ``IF EXISTS`` keeps the drop safe if a fold's CREATE failed. TimesFM trains no object, so
    nothing to drop.
    """
    if model_name in _MODEL_TYPE:
        return [f"DROP MODEL IF EXISTS {_model_ref(cfg, model_name, dataset, fold_id=fold_id)};"]
    return []


def render_setup_sql(cfg: RunConfig, model_name: str, dataset: str = "{dataset}") -> str:
    """All of a model's true-future setup statements joined into one script (snapshot + reading)."""
    return "\n\n".join(build_setup_statements(cfg, model_name, dataset))


# --- engine --------------------------------------------------------------------


def run(
    cfg: RunConfig,
    models: list[str],
    *,
    manage_header: bool = True,
    settings: Settings | None = None,
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
    """
    import time
    from datetime import UTC, datetime

    from google.cloud import bigquery

    from ..errors import RegistryError, get_logger
    from ..metrics import METRIC_NAMES, compute_metrics
    from ..registry import bq
    from ..registry.ids import make_run_id
    from ..settings import Settings
    from ..worker import _rollup_metrics

    log = get_logger(__name__)
    settings = settings or Settings.resolve()
    run_id = make_run_id(cfg)
    dataset = settings.dataset_ref
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
        return client.query(sql, job_config=job_config).result()

    # Header first (run_header): RUNNING on entry, finalized on a clean exit; a crash records FAILED
    # on the owned header before re-raising. Contributor mode (main.run owns the shared header) is a
    # no-op wrapper, so main.run's finalize sees the raised RegistryError and records the status.
    with bq.run_header(cfg, run_id, settings=settings, manage=manage_header) as hdr:
        started = time.perf_counter()
        created_at = datetime.now(UTC)
        status = "COMPLETED"
        nan_panel = {name: float("nan") for name in METRIC_NAMES}
        series_ids = [str(r.ts_id) for r in _query(build_series_ids_query(cfg, dataset))]
        n_series = len(series_ids)
        try:
            # --- Phase 1: final true-future forecast → forecast_predictions (always) ----------
            for model_name in models:
                for stmt in build_setup_statements(cfg, model_name, dataset):
                    _query(stmt)

            # --- Phase 2: scored evaluation ---------------------------------------------------
            oof_rows: list[dict[str, Any]] = []
            meta_rows: list[dict[str, Any]] = []

            if cfg.backtest.enabled:
                history = _query(build_history_query(cfg, dataset)).to_dataframe()
                hist_by_id = {tid: g["y"].to_numpy() for tid, g in history.groupby("ts_id")}
                plan = fold_plan(cfg)
                for model_name in models:
                    best_params = json.dumps(bqml_options(cfg, model_name), sort_keys=True)
                    panels_by_ts: dict[str, list[dict[str, float]]] = {}
                    for fold_id, back_steps in plan:
                        for stmt in build_fold_create_statements(
                            cfg, model_name, dataset, fold_id, back_steps
                        ):
                            _query(stmt)
                        eval_df = _query(
                            build_eval_query(
                                cfg, model_name, dataset, back_steps=back_steps, fold_id=fold_id
                            )
                        ).to_dataframe()
                        # The fold model has served its forecast — drop it so backtest runs don't
                        # leave orphaned sf_model_*_f{k} objects behind. Best-effort: a failed
                        # cleanup must not sink an otherwise-good run (results already read above).
                        for stmt in build_fold_drop_statements(cfg, model_name, dataset, fold_id):
                            try:
                                _query(stmt)
                            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                                log.warning(
                                    "fold model cleanup failed (%s f%d): %s",
                                    model_name,
                                    fold_id,
                                    exc,
                                )
                        for ts_id, g in eval_df.groupby("ts_id"):
                            g = g.sort_values("forecast_date")
                            for _, r in g.iterrows():
                                oof_rows.append(
                                    {
                                        "run_id": run_id,
                                        "ts_id": ts_id,
                                        "model_type": model_name,
                                        "fold_id": fold_id,
                                        "forecast_date": r["forecast_date"],
                                        "y_true": r["y_true"],
                                        "yhat": r["yhat"],
                                    }
                                )
                            panel = compute_metrics(
                                g["y_true"].to_numpy(),
                                g["yhat"].to_numpy(),
                                y_train=hist_by_id.get(ts_id),
                                lower=g["yhat_lower"].to_numpy(),
                                upper=g["yhat_upper"].to_numpy(),
                            )
                            panels_by_ts.setdefault(str(ts_id), []).append(panel)
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

            _append_rows(settings, "backtest_oof", bq._OOF_SPEC, oof_rows)
            _append_rows(settings, "forecast_metadata", bq._META_SPEC, meta_rows)
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


def _meta_row(  # pragma: no cover - GCP I/O helper, exercised by the @gcp smoke
    run_id: str,
    ts_id: str,
    model_name: str,
    panel: dict[str, float],
    best_params: str,
    created_at: Any,
    cfg: RunConfig,
) -> dict[str, Any]:
    """Assemble one ``forecast_metadata`` row (``fold_id=NULL``) for a native model."""
    from ..metrics import METRIC_NAMES
    from ..registry.ids import make_model_hash

    return {
        "run_id": run_id,
        "ts_id": ts_id,
        "model_type": model_name,
        "compute_engine": "bigquery",
        "model_hash": make_model_hash(run_id, str(ts_id), model_name, cfg),
        "fold_id": None,
        **{name: panel[name] for name in METRIC_NAMES},
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
    `registry.bq.write_cells` uses — the ``CellResult`` requirement lives only in the
    ``assemble_*`` wrappers, not the write path, so the native engine feeds ``_*_SPEC``-shaped dicts
    directly. Empty input is a no-op.
    """
    from google.cloud import bigquery_storage_v1

    from ..registry import bq

    if not rows:
        return
    write_client = bigquery_storage_v1.BigQueryWriteClient()
    msg_cls, proto_descriptor = bq._proto_for(table, spec)
    serialized = bq._encode_rows(msg_cls, spec, rows)
    bq._append_via_write_api(
        write_client,
        settings.project_id,
        settings.dataset_id,
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
