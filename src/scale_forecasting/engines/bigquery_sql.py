"""The pure SQL a BigQuery-native run executes — every statement, rendered from config alone.

Deterministic string renderers, no client and no I/O: a ``dataset`` argument defaulting to a
``{dataset}`` template token so a config-only call still renders, ``@run_id`` bound as a **query
parameter**, and identifiers (dataset, columns, model names) interpolated because BigQuery cannot
bind those. Snapshot-tested offline; `bigquery_engine` is the only thing that runs them.

Two layers, in file order:

* **Fragments** — the sub-statement expressions the builders compose: the cadence map (`_freq`),
  the snapshot pin (`_snapshot_clause`), the deterministic ``series_limit`` subset
  (`_series_filter`), the fold cutoff date (`_cutoff_expr` / `fold_plan`), the resolved model
  parameters (`bqml_options`), and the training/forecast table expressions.
* **Statements** — the complete, executable SQL (``build_*`` / `render_setup_sql`): create a model,
  insert the true-future forecast, read a fold back against actuals, and the per-fold
  create/drop pairs.

**Backtest geometry.** `fold_plan` and `_cutoff_expr` are the date-space mirror of
`backtest.make_folds`'s anchored-from-end index arithmetic. They are the piece most able to drift
from the Python path silently, so they are rendered from the same three config values and nothing
else.

Object naming and table qualification live next door in `bigquery_names`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..features import holiday_frame
from .bigquery_names import _model_ref, _registry_of, _sanitize_identifier, _source_ref

if TYPE_CHECKING:
    from ..config import RunConfig


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

def _freq(cfg: RunConfig) -> tuple[str, str]:
    """Return ``(data_frequency, interval_unit)`` for the run's cadence (defaults to daily)."""
    return _FREQ_MAP.get(cfg.data.freq.upper(), ("DAILY", "DAY"))

def _snapshot_clause(snapshot_millis: int | None) -> str:
    """The ``FOR SYSTEM_TIME AS OF`` clause pinning a source read to the run's snapshot (else "").

    Appended immediately *after* each backtick-quoted source-table reference so every native SQL
    statement time-travels to the identical instant every other job in the run reads — the same
    snapshot the Spark and Ray readers pin to (`registry.header.snapshot_millis_for`). ``None``
    renders nothing, so an un-pinned run's SQL is byte-identical to the pre-snapshot behavior (and
    only the source tables carry it — model objects and the registry tables are never
    time-travelled).
    """
    if snapshot_millis is None:
        return ""
    return f" FOR SYSTEM_TIME AS OF TIMESTAMP_MILLIS({snapshot_millis})"

def _series_filter(
    cfg: RunConfig, source: str, id_expr: str, *, snapshot_millis: int | None = None
) -> str:
    """A ``ts_id IN (...)`` fragment for the deterministic ``series_limit`` subset (``""`` if none).

    Same rule as ``spark_io._limit_series``: distinct ids → ordered → first N. ``id_expr`` is the
    column reference to constrain (bare ``ts_id`` in a scan, ``s.ts_id`` under a join alias). The
    subquery reads the source, so it carries the run's snapshot pin too (`_snapshot_clause`).
    """
    limit = cfg.data.series_limit
    if limit is None:
        return ""
    idc = cfg.data.ts_id_col
    snap = _snapshot_clause(snapshot_millis)
    return (
        f"{id_expr} IN (SELECT {idc} FROM `{source}`{snap} "
        f"GROUP BY {idc} ORDER BY {idc} LIMIT {limit})"
    )

def _cutoff_expr(
    cfg: RunConfig, source: str, back_steps: int, *, snapshot_millis: int | None = None
) -> str:
    """Scalar subquery for a training cutoff date: ``MAX(ds) - back_steps`` over the source.

    ``back_steps`` is the number of cadence units to step back from the last observed date, so the
    model trains on ``ds <= cutoff`` and is scored on the ``horizon`` window after it. A single
    global cutoff (not per-series) keeps every series' fold window aligned — the shipped data shares
    one date span, so this is exact; note it if your series end on different dates.
    """
    _, unit = _freq(cfg)
    snap = _snapshot_clause(snapshot_millis)
    return (
        f"(SELECT DATE_SUB(MAX({cfg.data.date_col}), INTERVAL {back_steps} {unit}) "
        f"FROM `{source}`{snap})"
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

# --- statements ----------------------------------------------------------------

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

def _train_window_where(
    cfg: RunConfig, source: str, back_steps: int | None, *, snapshot_millis: int | None = None
) -> list[str]:
    """The training-window date predicates for a model fit (``[]`` = train on all history).

    ``back_steps=None`` is the **final, true-future** fit: no date bound, train on everything.
    Otherwise the fit is a backtest fold: ``ds <= cutoff`` (expanding), plus a lower
    ``ds > cutoff - min_train`` bound for the sliding scheme so the window is fixed-width —
    mirroring ``backtest.make_folds``'s ``expanding`` vs ``sliding`` ``train_start``.
    """
    if back_steps is None:
        return []
    datec = cfg.data.date_col
    conds = [f"{datec} <= {_cutoff_expr(cfg, source, back_steps, snapshot_millis=snapshot_millis)}"]
    if cfg.backtest.scheme == "sliding":
        lower = _cutoff_expr(
            cfg, source, back_steps + cfg.backtest.min_train, snapshot_millis=snapshot_millis
        )
        conds.append(f"{datec} > {lower}")
    return conds

def _training_select(
    cfg: RunConfig,
    source: str,
    *,
    back_steps: int | None = None,
    snapshot_millis: int | None = None,
) -> str:
    """The ``training_data`` SELECT: id/timestamp/target over the fit's training window.

    ``back_steps=None`` trains on **all** history (the final true-future fit); an int restricts to a
    backtest fold's window (see `_train_window_where`).
    """
    cols = [cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col]
    where = _train_window_where(cfg, source, back_steps, snapshot_millis=snapshot_millis)
    sfilter = _series_filter(cfg, source, cfg.data.ts_id_col, snapshot_millis=snapshot_millis)
    if sfilter:
        where.append(sfilter)
    clause = f"\n  WHERE {' AND '.join(where)}" if where else ""
    snap = _snapshot_clause(snapshot_millis)
    return f"SELECT {', '.join(cols)}\n  FROM `{source}`{snap}{clause}"

def build_create_model_sql(
    cfg: RunConfig,
    model_name: str,
    dataset: str = "{dataset}",
    *,
    registry_dataset: str | None = None,
    back_steps: int | None = None,
    fold_id: int | None = None,
    snapshot_millis: int | None = None,
) -> str:
    """``CREATE OR REPLACE MODEL`` for an ARIMA_PLUS native model.

    ``back_steps=None`` (default) is the **final** model: trained on *all* history so its forecast
    is a true beyond-data forecast (parity with the Spark final fit). A ``back_steps``/``fold_id``
    pair builds a **backtest fold** model — trained on ``ds <= cutoff`` — into a fold-suffixed
    object.
    When holidays are configured the ``AS`` clause takes the named-subquery form
    (``training_data AS (...), custom_holiday AS (...)``); otherwise it is a plain training query.
    TimesFM has no CREATE MODEL — see `build_forecast_insert_sql`.

    The model object goes to ``registry_dataset`` (default: ``dataset``) while the training data is
    read from ``dataset`` — see `_registry_of`.
    """
    ref = _model_ref(cfg, model_name, _registry_of(dataset, registry_dataset), fold_id=fold_id)
    source = _source_ref(cfg, dataset)
    options = _render_options(bqml_options(cfg, model_name))
    training = _training_select(
        cfg, source, back_steps=back_steps, snapshot_millis=snapshot_millis
    )
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
    registry_dataset: str | None = None,
    back_steps: int | None = None,
    fold_id: int | None = None,
    horizon: int | None = None,
    snapshot_millis: int | None = None,
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
        where = _train_window_where(cfg, source, back_steps, snapshot_millis=snapshot_millis)
        sfilter = _series_filter(cfg, source, idc, snapshot_millis=snapshot_millis)
        if sfilter:
            where.append(sfilter)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        snap = _snapshot_clause(snapshot_millis)
        inner = f"SELECT {idc}, {datec}, {targetc} FROM `{source}`{snap}{clause}"
        return (
            "AI.FORECAST(\n"
            f"    ({inner}),\n"
            f"    data_col => '{targetc}',\n"
            f"    timestamp_col => '{datec}',\n"
            f"    id_cols => ['{idc}'],\n"
            f"    horizon => {h},\n"
            f"    confidence_level => {_CONFIDENCE_LEVEL})"
        )

    ref = _model_ref(cfg, model_name, _registry_of(dataset, registry_dataset), fold_id=fold_id)
    struct = f"STRUCT({h} AS horizon, {_CONFIDENCE_LEVEL} AS confidence_level)"
    return f"ML.FORECAST(MODEL {ref}, {struct})"

# Columns written to forecast_predictions, in DDL order (shared by the INSERT).
_PRED_COLS = (
    "run_id, ts_id, model_type, compute_engine, forecast_date, "
    "yhat, yhat_lower, yhat_upper, quantiles"
)

def build_forecast_insert_sql(
    cfg: RunConfig,
    model_name: str,
    dataset: str = "{dataset}",
    *,
    registry_dataset: str | None = None,
    snapshot_millis: int | None = None,
) -> str:
    """``INSERT INTO forecast_predictions`` the **true beyond-data** forecast for one native model.

    The showpiece statement: one query forecasts every series' next ``data.horizon`` steps from a
    model fit on *all* history and lands canonical prediction rows with
    ``compute_engine='bigquery'`` — directly comparable to the Spark models' true-future rows.
    ``forecast_timestamp`` → ``DATE()`` → ``forecast_date``; the interval bounds map to
    ``yhat_lower`` / ``yhat_upper``;
    ``quantiles`` is NULL (native models emit an interval, not an arbitrary quantile set).
    """
    dataset_q = f"`{_registry_of(dataset, registry_dataset)}.forecast_predictions`"
    forecast = _forecast_source(
        cfg, model_name, dataset, registry_dataset=registry_dataset,
        snapshot_millis=snapshot_millis,
    )
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
    registry_dataset: str | None = None,
    back_steps: int,
    fold_id: int,
    snapshot_millis: int | None = None,
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
    forecast = _forecast_source(
        cfg, model_name, dataset, registry_dataset=registry_dataset,
        back_steps=back_steps, fold_id=fold_id, snapshot_millis=snapshot_millis,
    )
    snap = _snapshot_clause(snapshot_millis)
    return (
        f"SELECT\n"
        f"  f.{idc} AS ts_id, DATE(f.forecast_timestamp) AS forecast_date,\n"
        f"  s.{targetc} AS y_true, f.forecast_value AS yhat,\n"
        f"  f.prediction_interval_lower_bound AS yhat_lower,\n"
        f"  f.prediction_interval_upper_bound AS yhat_upper\n"
        f"FROM {forecast} f\n"
        f"JOIN `{source}`{snap} s\n"
        f"  ON s.{idc} = f.{idc} AND s.{datec} = DATE(f.forecast_timestamp)\n"
        f"ORDER BY ts_id, forecast_date;"
    )

def build_history_query(
    cfg: RunConfig, dataset: str = "{dataset}", *, snapshot_millis: int | None = None
) -> str:
    """A ``SELECT`` of **all** training history ``(ts_id, ds, y)`` (for MASE/RMSSE scale).

    The scale-free metrics need each series' training actuals as ``y_train``; the engine loads this
    once, groups by ``ts_id``, and passes the per-series history to `metrics.compute_metrics`.
    Post-alignment this is the full series history (the natives train on all of it for the final
    forecast) — a robust, freq-agnostic scale for the fold metrics.
    """
    source = _source_ref(cfg, dataset)
    idc, datec, targetc = cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col
    sfilter = _series_filter(cfg, source, idc, snapshot_millis=snapshot_millis)
    clause = f"\nWHERE {sfilter}" if sfilter else ""
    snap = _snapshot_clause(snapshot_millis)
    return (
        f"SELECT {idc} AS ts_id, {datec} AS ds, {targetc} AS y\n"
        f"FROM `{source}`{snap}{clause}\n"
        f"ORDER BY ts_id, ds;"
    )

def build_series_ids_query(
    cfg: RunConfig, dataset: str = "{dataset}", *, snapshot_millis: int | None = None
) -> str:
    """A ``SELECT DISTINCT ts_id`` over the source subset — the run's series list (engine-agnostic).

    Feeds the ``n_series`` count and, when backtesting is off, the one unscored ``fold_id=NULL``
    metadata row per ``(series, model)`` (NaN metric panel) that keeps native parity with the Python
    worker's always-emitted metadata row.
    """
    source = _source_ref(cfg, dataset)
    idc = cfg.data.ts_id_col
    sfilter = _series_filter(cfg, source, idc, snapshot_millis=snapshot_millis)
    clause = f"\nWHERE {sfilter}" if sfilter else ""
    snap = _snapshot_clause(snapshot_millis)
    return f"SELECT DISTINCT {idc} AS ts_id\nFROM `{source}`{snap}{clause}\nORDER BY ts_id;"

def build_setup_statements(
    cfg: RunConfig,
    model_name: str,
    dataset: str = "{dataset}",
    *,
    registry_dataset: str | None = None,
    snapshot_millis: int | None = None,
) -> list[str]:
    """The mutating statements for one native model's **final true-future forecast**, in order.

    ARIMA models: ``[CREATE MODEL (all history), INSERT INTO forecast_predictions]``. TimesFM: just
    the INSERT (no training). The backtest fold statements (`build_create_model_sql` with
    ``back_steps``) and the read-back eval (`build_eval_query`) are *not* here — this renders
    only the always-run true-future path.
    """
    statements: list[str] = []
    if model_name in _MODEL_TYPE:
        statements.append(
            build_create_model_sql(
                cfg, model_name, dataset, registry_dataset=registry_dataset,
                snapshot_millis=snapshot_millis,
            )
        )
    statements.append(
        build_forecast_insert_sql(
            cfg, model_name, dataset, registry_dataset=registry_dataset,
            snapshot_millis=snapshot_millis,
        )
    )
    return statements

def build_fold_create_statements(
    cfg: RunConfig,
    model_name: str,
    dataset: str,
    fold_id: int,
    back_steps: int,
    *,
    registry_dataset: str | None = None,
    snapshot_millis: int | None = None,
) -> list[str]:
    """The training statements for one backtest fold (``[CREATE MODEL]`` for ARIMA, ``[]`` else).

    TimesFM needs no model object — its fold forecast reads the ``ds <= cutoff`` history directly in
    `build_eval_query` — so it has no fold-training statement.
    """
    if model_name in _MODEL_TYPE:
        return [
            build_create_model_sql(
                cfg, model_name, dataset, registry_dataset=registry_dataset,
                back_steps=back_steps, fold_id=fold_id, snapshot_millis=snapshot_millis,
            )
        ]
    return []

def build_fold_drop_statements(
    cfg: RunConfig,
    model_name: str,
    dataset: str,
    fold_id: int,
    *,
    registry_dataset: str | None = None,
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
        ref = _model_ref(cfg, model_name, _registry_of(dataset, registry_dataset), fold_id=fold_id)
        return [f"DROP MODEL IF EXISTS {ref};"]
    return []

def render_setup_sql(
    cfg: RunConfig,
    model_name: str,
    dataset: str = "{dataset}",
    *,
    registry_dataset: str | None = None,
    snapshot_millis: int | None = None,
) -> str:
    """All of a model's true-future setup statements joined into one script (snapshot + reading)."""
    return "\n\n".join(
        build_setup_statements(
            cfg, model_name, dataset, registry_dataset=registry_dataset,
            snapshot_millis=snapshot_millis,
        )
    )
