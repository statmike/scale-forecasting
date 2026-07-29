"""BigQuery-native models — SQL runner for ARIMA_PLUS / ARIMA_PLUS_XREG / TimesFM (CONTRACTS §5).

The BigQuery runtime executes forecasting *as SQL inside BigQuery* — the opposite of the Spark
track's per-cell fan-out. ``ARIMA_PLUS`` with ``time_series_id_col`` trains **all series in one
``CREATE MODEL`` statement**; ``AI.FORECAST`` (TimesFM) forecasts every series in one call with no
training at all. Both land in the *same* three-tier registry as the Python models, so a native model
and a Spark model are directly comparable on ``v_model_leaderboard`` (DESIGN §3.3).

This module has two halves:

* **Pure SQL builders** (``build_*`` / ``render_*``) — deterministic string renderers, mirroring the
  ``ensembler.build_ensemble_sql`` house style: a ``dataset`` argument defaulting to a ``{dataset}``
  template token so a config-only call renders, ``@run_id`` bound as a **query parameter**, and
  identifiers (dataset, columns, model names) interpolated. Snapshot-tested offline, no GCP.
* **The engine** (:func:`run`) — resolves :class:`~scale_forecasting.settings.Settings`, owns the
  ``run_registry`` header lifecycle exactly like :func:`spark_naive.run`, executes the builders'
  SQL via ``bigquery.Client``, reads the held-out forecast back, computes the metric panel through
  the shared :func:`~scale_forecasting.metrics.compute_metrics` (no formula drift), and writes all
  three cell tables via the registry's Storage Write API row-dict path.

**Held-out semantics (single fold).** Every native model trains on ``ds <= cutoff`` (where
``cutoff = MAX(ds) - horizon``) and forecasts the last ``horizon`` window ``ds > cutoff`` — a window
we have ground truth for. So ``forecast_predictions``, ``backtest_oof`` (``fold_id=0``), and the
metric panel are all real, not synthetic-future. Beyond-data forecasting and alignment with the
Spark track under one shared ``run_id`` is Arc B (``main`` orchestration), out of B3's scope.

**Transform.** ``cfg.features.transform`` (e.g. ``log1p``) is intentionally **not** applied here:
ARIMA_PLUS runs its own decomposition, and TimesFM is a pretrained foundation model (DESIGN §4).
Holidays *are* honored — the custom-holiday CTE is built from the same
:func:`~scale_forecasting.features.holiday_frame` the Python suite uses, so "holiday" is identical
across runtimes.

Public surface: ``run(cfg, models)``; builders ``build_create_model_sql``,
``build_forecast_insert_sql``, ``build_eval_query``, ``build_history_query``,
``build_custom_holiday_cte``, ``render_setup_sql``, ``bqml_options``.
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
    """The BigQuery engine's run summary — what :func:`main.run` folds into the shared header.

    ``status`` is COMPLETED (the engine raises on any SQL failure rather than returning FAILED, so a
    returned outcome is always COMPLETED today; the field is explicit for symmetry with the Spark
    roll-up and future partial-success handling). ``n_series`` is the distinct series count observed
    in the held-out eval; ``models`` is the executed native subset (feeds ``bq_models``).
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
    "arima_plus_xreg": "ARIMA_PLUS_XREG",
}
_XREG_MODELS = frozenset({"arima_plus_xreg"})

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

    Mirrors ``spark_io._resolve_source_table`` so both runtimes read the identical table (G1).
    """
    src = cfg.data.source_table
    return src if "." in src else f"{dataset}.{src}"


def _model_ref(cfg: RunConfig, model_name: str, dataset: str) -> str:
    """The backtick-quoted BQML model object path for one ``(model, run)`` (persisted, reusable).

    The object name embeds the config-pinned ``run_id`` so re-running the same config targets the
    *same* model (``CREATE OR REPLACE`` is idempotent) while a different config gets a distinct
    object. A model object name is an identifier — it cannot be a bound query parameter — so the
    ``run_id`` is interpolated here; it is a pure function of ``cfg`` (:func:`make_run_id`), which
    keeps every builder a pure function of its config.
    """
    run_id = make_run_id(cfg)
    return f"`{dataset}.sf_model_{model_name}_{_sanitize_identifier(run_id)}`"


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


def _cutoff_expr(cfg: RunConfig, source: str) -> str:
    """Scalar subquery for the held-out cutoff date: ``MAX(ds) - horizon`` over the source.

    The cutoff is the last *training* date: models train on ``ds <= cutoff`` and are scored on
    ``ds > cutoff`` — exactly the ``horizon`` days ``cutoff+1 … MAX(ds)``, every one of which has
    ground truth. A single global cutoff (not per-series) keeps every series' held-out window
    aligned — the shipped data shares one date span, so this is exact; note it if your series end on
    different dates.
    """
    _, unit = _freq(cfg)
    return (
        f"(SELECT DATE_SUB(MAX({cfg.data.date_col}), INTERVAL {cfg.data.horizon} {unit}) "
        f"FROM `{source}`)"
    )


# --- pure SQL builders ---------------------------------------------------------


def bqml_options(cfg: RunConfig, model_name: str) -> dict[str, Any]:
    """The resolved model parameters, as an ordered dict — one source of truth for two uses.

    For the ARIMA models this is the ``CREATE MODEL`` ``OPTIONS(...)`` body *and* the
    ``best_params`` JSON on each ``forecast_metadata`` row — the registry records what trained.
    TimesFM has no ``CREATE MODEL``; :func:`_render_options` never sees its dict, but ``run`` still
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
    """Render the ``custom_holiday`` CTE from :func:`features.holiday_frame`, or ``""`` if none.

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


def _training_select(cfg: RunConfig, source: str, model_name: str) -> str:
    """The ``training_data`` SELECT: id/timestamp/target (+ exog for XREG) over ``ds <= cutoff``."""
    cols = [cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col]
    if model_name in _XREG_MODELS:
        cols += list(cfg.features.exog)
    cutoff = _cutoff_expr(cfg, source)
    where = [f"{cfg.data.date_col} <= {cutoff}"]
    sfilter = _series_filter(cfg, source, cfg.data.ts_id_col)
    if sfilter:
        where.append(sfilter)
    return f"SELECT {', '.join(cols)}\n  FROM `{source}`\n  WHERE {' AND '.join(where)}"


def build_create_model_sql(cfg: RunConfig, model_name: str, dataset: str = "{dataset}") -> str:
    """``CREATE OR REPLACE MODEL`` for an ARIMA_PLUS / ARIMA_PLUS_XREG native model (held-out fit).

    Trains on ``ds <= cutoff`` so the model is scored on the last ``horizon`` window it never saw.
    When holidays are configured the ``AS`` clause takes the named-subquery form
    (``training_data AS (...), custom_holiday AS (...)``); otherwise it is a plain training query.
    TimesFM has no CREATE MODEL — see :func:`build_forecast_insert_sql`.
    """
    ref = _model_ref(cfg, model_name, dataset)
    source = _source_ref(cfg, dataset)
    options = _render_options(bqml_options(cfg, model_name))
    training = _training_select(cfg, source, model_name)
    holiday_cte = build_custom_holiday_cte(cfg)
    if holiday_cte:
        body = f"  training_data AS (\n    {training}\n  ),\n  {holiday_cte}"
        as_clause = f"AS (\n{body}\n)"
    else:
        as_clause = f"AS (\n  {training}\n)"
    return f"CREATE OR REPLACE MODEL {ref}\n{options}\n{as_clause};"


def _forecast_source(cfg: RunConfig, model_name: str, dataset: str) -> str:
    """The ``ML.FORECAST`` / ``AI.FORECAST`` table expression producing the held-out forecast.

    ARIMA_PLUS: ``ML.FORECAST(MODEL m, STRUCT(...))``. ARIMA_PLUS_XREG: the same with the held-out
    window's *real* future features supplied as a query *after* the STRUCT (the position BQML
    requires). TimesFM: ``AI.FORECAST`` over the ``ds <= cutoff`` history — no model object. All
    three yield ``forecast_timestamp`` / ``forecast_value`` /
    ``prediction_interval_{lower,upper}_bound`` plus the id column.
    """
    source = _source_ref(cfg, dataset)
    idc, datec, targetc = cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col
    horizon = cfg.data.horizon
    struct = f"STRUCT({horizon} AS horizon, {_CONFIDENCE_LEVEL} AS confidence_level)"

    if model_name == "timesfm":
        cutoff = _cutoff_expr(cfg, source)
        where = [f"{datec} <= {cutoff}"]
        sfilter = _series_filter(cfg, source, idc)
        if sfilter:
            where.append(sfilter)
        inner = f"SELECT {idc}, {datec}, {targetc} FROM `{source}` WHERE {' AND '.join(where)}"
        return (
            "AI.FORECAST(\n"
            f"    ({inner}),\n"
            f"    data_col => '{targetc}',\n"
            f"    timestamp_col => '{datec}',\n"
            f"    id_cols => ['{idc}'],\n"
            f"    horizon => {horizon},\n"
            f"    confidence_level => {_CONFIDENCE_LEVEL})"
        )

    ref = _model_ref(cfg, model_name, dataset)
    if model_name in _XREG_MODELS:
        cutoff = _cutoff_expr(cfg, source)
        exog_cols = ", ".join([idc, datec, *cfg.features.exog])
        where = [f"{datec} > {cutoff}"]
        sfilter = _series_filter(cfg, source, idc)
        if sfilter:
            where.append(sfilter)
        future = f"SELECT {exog_cols} FROM `{source}` WHERE {' AND '.join(where)}"
        return f"ML.FORECAST(MODEL {ref},\n    {struct},\n    ({future}))"
    return f"ML.FORECAST(MODEL {ref}, {struct})"


# Columns written to forecast_predictions, in DDL order (shared by the INSERT).
_PRED_COLS = (
    "run_id, ts_id, model_type, compute_engine, forecast_date, "
    "yhat, yhat_lower, yhat_upper, quantiles"
)


def build_forecast_insert_sql(cfg: RunConfig, model_name: str, dataset: str = "{dataset}") -> str:
    """``INSERT INTO forecast_predictions`` the held-out forecast for one native model.

    The showpiece statement: one query forecasts every series and lands canonical prediction rows
    with ``compute_engine='bigquery'``. ``forecast_timestamp`` → ``DATE()`` → ``forecast_date``;
    the interval bounds map to ``yhat_lower`` / ``yhat_upper``; ``quantiles`` is NULL (native models
    emit an interval, not an arbitrary quantile set).
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


def build_eval_query(cfg: RunConfig, model_name: str, dataset: str = "{dataset}") -> str:
    """A read-back ``SELECT`` of the held-out forecast joined to actuals (for OOF + metrics).

    Returns ``(ts_id, forecast_date, y_true, yhat, yhat_lower, yhat_upper)`` for the held-out span.
    The engine groups these by ``ts_id`` and feeds :func:`metrics.compute_metrics` — so the metric
    math is byte-identical to the Python models. Intervals are carried (unlike ``backtest_oof``) so
    coverage/pinball are real. ``@run_id`` only names the model here; this query writes no row.
    """
    source = _source_ref(cfg, dataset)
    idc, datec, targetc = cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col
    forecast = _forecast_source(cfg, model_name, dataset)
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
    """A ``SELECT`` of training history ``(ts_id, ds, y)`` over ``ds <= cutoff`` (for MASE/RMSSE).

    The scale-free metrics need each series' training actuals as ``y_train``; the engine loads this
    once, groups by ``ts_id``, and passes the per-series history to :func:`metrics.compute_metrics`.
    """
    source = _source_ref(cfg, dataset)
    idc, datec, targetc = cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col
    cutoff = _cutoff_expr(cfg, source)
    where = [f"{datec} <= {cutoff}"]
    sfilter = _series_filter(cfg, source, idc)
    if sfilter:
        where.append(sfilter)
    return (
        f"SELECT {idc} AS ts_id, {datec} AS ds, {targetc} AS y\n"
        f"FROM `{source}`\n"
        f"WHERE {' AND '.join(where)}\n"
        f"ORDER BY ts_id, ds;"
    )


def build_setup_statements(
    cfg: RunConfig, model_name: str, dataset: str = "{dataset}"
) -> list[str]:
    """The mutating statements for one native model, in execution order.

    ARIMA models: ``[CREATE MODEL, INSERT INTO forecast_predictions]``. TimesFM: just the INSERT
    (no training). The read-back eval query (:func:`build_eval_query`) is *not* here — it returns
    rows, it doesn't mutate — and neither is the metadata write (that's the Write API path).
    """
    statements: list[str] = []
    if model_name in _MODEL_TYPE:
        statements.append(build_create_model_sql(cfg, model_name, dataset))
    statements.append(build_forecast_insert_sql(cfg, model_name, dataset))
    return statements


def render_setup_sql(cfg: RunConfig, model_name: str, dataset: str = "{dataset}") -> str:
    """All of a model's setup statements joined into one script (snapshot + human reading)."""
    return "\n\n".join(build_setup_statements(cfg, model_name, dataset))


# --- engine --------------------------------------------------------------------


def run(
    cfg: RunConfig,
    models: list[str],
    *,
    manage_header: bool = True,
    settings: Settings | None = None,
) -> BqOutcome:  # pragma: no cover - GCP I/O, @gcp smoke
    """Execute the BigQuery-native subset end-to-end, mirroring :func:`spark_naive.run`.

    Header lifecycle (CONTRACTS §8.2): resolve :class:`Settings`, derive the config-pinned
    ``run_id``, ``ensure_tables`` → ``write_header`` (RUNNING), run each native model's SQL, then
    ``update_header`` with the aggregated status, wall-clock ``runtime_seconds``, ``n_series``,
    ``n_models``, and the ``bq_models`` array. Per model: execute the setup statements (CREATE +
    forecast INSERT), read the held-out eval + history back, compute the metric panel through
    :func:`metrics.compute_metrics`, and append ``backtest_oof`` + ``forecast_metadata`` rows via
    the registry's Storage Write API row-dict path.

    ``manage_header=False`` is **contributor mode** (Arc B): :func:`main.run` owns the single shared
    header, so the engine skips ``ensure_tables`` / ``write_header`` / ``update_header`` and only
    runs SQL + writes the cell tables. It returns a :class:`BqOutcome` (status + ``n_series``) so
    the orchestrator can fold it into the combined header finalize. ``settings`` may be passed to
    reuse the orchestrator's already-resolved infra; ``None`` resolves it here (standalone default).

    Idempotent: ``run_id`` is a pure function of the config and every write is append-only /
    dedupe-on-read, so a re-run of the same config lands byte-identical rows (§3.4).
    """
    import time
    from datetime import UTC, datetime

    from google.cloud import bigquery

    from ..errors import RegistryError, get_logger
    from ..metrics import METRIC_NAMES, compute_metrics
    from ..registry import bq
    from ..registry.ids import make_model_hash, make_run_id
    from ..settings import Settings

    log = get_logger(__name__)
    settings = settings or Settings.resolve()
    run_id = make_run_id(cfg)
    dataset = settings.dataset_ref
    client = bigquery.Client(project=settings.project_id)
    log.info(
        "bigquery run start: run_id=%s models=%s manage_header=%s", run_id, models, manage_header
    )

    def _query(sql: str) -> Any:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
        )
        return client.query(sql, job_config=job_config).result()

    if manage_header:
        bq.ensure_tables(cfg, settings=settings)
        bq.write_header(cfg, run_id, settings=settings)

    started = time.perf_counter()
    created_at = datetime.now(UTC)
    status = "COMPLETED"
    n_series = 0
    try:
        history = _query(build_history_query(cfg, dataset)).to_dataframe()
        hist_by_id = {tid: g["y"].to_numpy() for tid, g in history.groupby("ts_id")}

        oof_rows: list[dict[str, Any]] = []
        meta_rows: list[dict[str, Any]] = []
        for model_name in models:
            for stmt in build_setup_statements(cfg, model_name, dataset):
                _query(stmt)
            eval_df = _query(build_eval_query(cfg, model_name, dataset)).to_dataframe()
            n_series = max(n_series, int(eval_df["ts_id"].nunique()))
            best_params = json.dumps(bqml_options(cfg, model_name), sort_keys=True)
            for ts_id, g in eval_df.groupby("ts_id"):
                g = g.sort_values("forecast_date")
                for _, r in g.iterrows():
                    oof_rows.append(
                        {
                            "run_id": run_id,
                            "ts_id": ts_id,
                            "model_type": model_name,
                            "fold_id": 0,
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
                meta_rows.append(
                    {
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
                )

        _append_rows(settings, "backtest_oof", bq._OOF_SPEC, oof_rows)
        _append_rows(settings, "forecast_metadata", bq._META_SPEC, meta_rows)
    except Exception as exc:  # noqa: BLE001 - header must record the failure before re-raising
        status = "FAILED"
        # Owner mode records the failure on its own header before re-raising; contributor mode
        # leaves the shared header to main.run's finalize (which sees the raised RegistryError).
        if manage_header:
            bq.update_header(
                run_id,
                settings=settings,
                status=status,
                runtime_seconds=time.perf_counter() - started,
            )
        raise RegistryError(f"bigquery run failed for {run_id}: {exc}") from exc

    runtime_seconds = time.perf_counter() - started
    if manage_header:
        bq.update_header(
            run_id,
            settings=settings,
            status=status,
            runtime_seconds=runtime_seconds,
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


def _append_rows(  # pragma: no cover - GCP I/O, @gcp smoke
    settings: Settings,
    table: str,
    spec: tuple[tuple[str, str], ...],
    rows: list[dict[str, Any]],
) -> None:
    """Append plain row dicts to a cell table via the registry's Storage Write API path.

    Reuses the same ``_proto_for`` / ``_encode_rows`` / ``_append_via_write_api`` machinery that
    :func:`registry.bq.write_cells` uses — the ``CellResult`` requirement lives only in the
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
    :func:`~scale_forecasting.router.split_by_runtime`, and runs the engine on that subset. A config
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
