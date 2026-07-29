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

Schema evolution is additive: when a new NULLABLE column is added to a body below,
``render_migrations`` derives an ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` for it from
the *same* bodies, so a table created under an older schema can be brought up to date
without the CREATE and the migration ever drifting apart (``ensure_tables`` runs both).

Public surface: ``TABLE_NAMES``, ``render_create_tables``, ``additive_columns``,
``render_migrations``.
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


def _column_block(body: str) -> str:
    """Return the text between the CREATE's opening ``(`` and the closing ``)`` (the columns)."""
    open_paren = body.index("(")
    close_paren = body.index("\n)", open_paren)
    return body[open_paren + 1 : close_paren]


def additive_columns(table: str) -> list[tuple[str, str]]:
    """Return the ``(name, type)`` of every NULLABLE column of ``table``, in declaration order.

    These are exactly the columns that can be back-filled onto an already-created table with
    ``ADD COLUMN IF NOT EXISTS`` (a ``NOT NULL`` column can't be added to a populated table, so
    those are excluded). Parsed from the same ``_TABLE_BODIES`` the CREATE renders, so the two can
    never drift. Commas separate columns (types like ``ARRAY<STRING>`` carry none), so a plain
    split is unambiguous; a piece containing ``NOT NULL`` is skipped.
    """
    cols: list[tuple[str, str]] = []
    for piece in _column_block(_TABLE_BODIES[table]).split(","):
        tokens = piece.split()
        if not tokens or "NOT NULL" in " ".join(tokens):
            continue
        name, col_type = tokens[0], " ".join(tokens[1:])
        if col_type:  # skip stray fragments; a real column always has a type
            cols.append((name, col_type))
    return cols


def render_migrations(dataset: str) -> dict[str, str]:
    """Render ``{table_name: ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...}`` for additive columns.

    One idempotent ``ALTER`` per table adds every nullable column if absent, so a table created
    under an older schema is brought current without touching existing rows (they read NULL) and
    without a hand-written migration. Tables with no nullable columns are omitted.
    """
    out: dict[str, str] = {}
    for name in TABLE_NAMES:
        cols = additive_columns(name)
        if not cols:
            continue
        adds = ",\n  ".join(f"ADD COLUMN IF NOT EXISTS {c} {t}" for c, t in cols)
        out[name] = f"ALTER TABLE `{dataset}.{name}`\n  {adds};"
    return out


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
