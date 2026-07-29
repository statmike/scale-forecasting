"""CREATE TABLE DDL for the registry + example data (CONTRACTS §4, DESIGN §8).

The five tables are the source of truth for the whole system; the column definitions
below are verbatim from CONTRACTS §4. Each is a **BigQuery-managed Apache Iceberg**
table (open format on GCS): the columns/partition/cluster are wrapped with a
``WITH CONNECTION ... OPTIONS(table_format='ICEBERG', ...)`` clause pointing at the
warehouse bucket.

Rendering is a pure string operation (no BigQuery client), so it is snapshot-tested
offline; ``registry/bq.ensure_tables`` (Arc B) executes what this renders.

JSON-typed fields (``raw_config``, ``best_params``, ``quantiles``) are stored as
**STRING**, not the native ``JSON`` type: BigQuery-managed Iceberg tables reject the
JSON column type (verified against a live table, B0.3). The row assemblers in ``bq.py``
already emit JSON *strings* (``json.dumps`` / ``_as_json``), so STRING matches what the
writers produce; query them back with ``PARSE_JSON(col)`` when you need structured access.

Public surface: ``TABLE_NAMES``, ``render_create_tables``.
"""

from __future__ import annotations

# Table bodies: columns + PARTITION BY + CLUSTER BY, verbatim from CONTRACTS §4.
# `{d}` is the dataset ref (`project.dataset` or `dataset`). No trailing semicolon —
# the renderer appends the OPTIONS clause and the semicolon.
_TABLE_BODIES: dict[str, str] = {
    "run_registry": """\
CREATE TABLE IF NOT EXISTS `{d}.run_registry` (
  run_id            STRING NOT NULL,
  created_at        TIMESTAMP NOT NULL,
  user_id           STRING,
  git_sha           STRING,
  python_runtime    STRING,
  spark_method      STRING,
  bq_models         ARRAY<STRING>,
  backtest_on       BOOL,
  decision_metric   STRING,
  ensemble_strategies ARRAY<STRING>,
  raw_config        STRING NOT NULL,
  status            STRING,
  n_series          INT64,
  n_models          INT64,
  runtime_seconds   FLOAT64,
  job_telemetry     STRING
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id""",
    "forecast_metadata": """\
CREATE TABLE IF NOT EXISTS `{d}.forecast_metadata` (
  run_id         STRING NOT NULL,
  ts_id          STRING NOT NULL,
  model_type     STRING NOT NULL,
  compute_engine STRING NOT NULL,
  model_hash     STRING NOT NULL,
  fold_id        INT64,
  mae FLOAT64, rmse FLOAT64, mse FLOAT64, mape FLOAT64, smape FLOAT64,
  wape FLOAT64, mase FLOAT64, rmsse FLOAT64, bias FLOAT64,
  coverage FLOAT64, pinball FLOAT64,
  fit_seconds    FLOAT64,
  best_params    STRING,
  model_artifact STRING,
  created_at     TIMESTAMP NOT NULL
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id, model_type""",
    "forecast_predictions": """\
CREATE TABLE IF NOT EXISTS `{d}.forecast_predictions` (
  run_id        STRING NOT NULL,
  ts_id         STRING NOT NULL,
  model_type    STRING NOT NULL,
  compute_engine STRING,
  forecast_date DATE NOT NULL,
  yhat          FLOAT64,
  yhat_lower    FLOAT64,
  yhat_upper    FLOAT64,
  quantiles     STRING
)
PARTITION BY forecast_date
CLUSTER BY run_id, ts_id""",
    "backtest_oof": """\
CREATE TABLE IF NOT EXISTS `{d}.backtest_oof` (
  run_id        STRING NOT NULL,
  ts_id         STRING NOT NULL,
  model_type    STRING NOT NULL,
  fold_id       INT64 NOT NULL,
  forecast_date DATE NOT NULL,
  y_true        FLOAT64,
  yhat          FLOAT64
)
PARTITION BY forecast_date
CLUSTER BY run_id, ts_id""",
    # Columns carry business names; the config maps each to a role (date→date_col,
    # price_index→features.exog). `price_index` is the example driver the generator emits
    # and the xreg models regress on; swap it for real drivers in your own source table.
    "source_series": """\
CREATE TABLE IF NOT EXISTS `{d}.source_series` (
  ts_id       STRING NOT NULL,
  ds          DATE NOT NULL,
  y           FLOAT64,
  archetype   STRING,
  price_index FLOAT64,
  is_holiday  BOOL
)
PARTITION BY ds
CLUSTER BY ts_id""",
}

TABLE_NAMES: tuple[str, ...] = tuple(_TABLE_BODIES)


def _iceberg_options(table_name: str, warehouse_uri: str) -> str:
    """The OPTIONS block that makes a table a managed Iceberg table (DESIGN §8)."""
    storage_uri = f"{warehouse_uri.rstrip('/')}/{table_name}"
    return (
        "OPTIONS (\n"
        "  file_format = 'PARQUET',\n"
        "  table_format = 'ICEBERG',\n"
        f"  storage_uri = '{storage_uri}'\n"
        ")"
    )


def render_create_tables(
    dataset: str,
    *,
    connection: str | None = None,
    warehouse_uri: str | None = None,
    iceberg: bool = True,
) -> dict[str, str]:
    """Render ``{table_name: CREATE TABLE statement}`` for all five tables.

    Args:
        dataset: dataset ref, ``project.dataset`` or ``dataset``.
        connection: Cloud Resource / BigLake connection ``project.region.name``;
            required when ``iceberg`` is True.
        warehouse_uri: GCS warehouse root (e.g. ``gs://bucket/warehouse``); required
            when ``iceberg`` is True.
        iceberg: when True (default) render managed-Iceberg DDL; when False render
            plain native BigQuery tables (the D1 fallback, or for a BQ emulator).

    Every statement is idempotent (``CREATE TABLE IF NOT EXISTS``).
    """
    if iceberg and (connection is None or warehouse_uri is None):
        raise ValueError("iceberg=True requires both 'connection' and 'warehouse_uri'")

    out: dict[str, str] = {}
    for name, body in _TABLE_BODIES.items():
        stmt = body.format(d=dataset)
        if iceberg:
            assert warehouse_uri is not None  # narrowed by the guard above
            options = _iceberg_options(name, warehouse_uri)
            stmt = f"{stmt}\nWITH CONNECTION `{connection}`\n{options}"
        out[name] = stmt + ";"
    return out
