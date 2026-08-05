"""CREATE TABLE DDL for the registry + example data (CONTRACTS §4, DESIGN §8, D19).

Six tables are the source of truth for the whole system; the column definitions below are
verbatim from CONTRACTS §4. Storage format is split by role (D19):

- The four **run-collection** tables (``run_registry``, ``forecast_metadata``,
  ``forecast_predictions``, ``backtest_oof``) are always **native BigQuery**. Native gives us
  the real ``JSON`` column type (``raw_config`` / ``job_telemetry`` / ``quantiles`` /
  ``best_params``) and ``WRITE_TRUNCATE`` reseed, and needs no BigLake connection.
- The example input table ships in **both** formats — ``source_series_iceberg`` (a
  BigQuery-managed Apache Iceberg table: columns/partition/cluster wrapped with
  ``WITH CONNECTION ... OPTIONS(table_format='ICEBERG', ...)`` at the warehouse bucket) and
  ``source_series_native`` (plain native) — so a deployment can benchmark the identical series
  on either storage. The engines read both transparently through BigQuery's table interface.

Rendering is a pure string operation (no BigQuery client), so it is snapshot-tested
offline; ``registry/bq.ensure_tables`` (Arc B) executes what ``render_deployment_ddl`` renders.

The ``JSON`` columns use the native ``JSON`` type (only the native registry carries them; the
Iceberg source table has none). The row assemblers in ``bq.py`` serialize JSON text, which a
``JSON`` column parses on ingest; read them back with ``.`` field access or ``JSON_VALUE``.

Schema evolution is additive: when a new NULLABLE column is added to a body below,
``render_migrations`` derives an ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` for it from
the *same* bodies, so a table created under an older schema can be brought up to date
without the CREATE and the migration ever drifting apart (``ensure_tables`` runs both).

Public surface: ``TABLE_NAMES``, ``SOURCE_TABLE_ICEBERG``, ``SOURCE_TABLE_NATIVE``,
``render_deployment_ddl``, ``render_create_tables``, ``render_drop_tables``,
``additive_columns``, ``render_migrations``.
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
  raw_config        JSON NOT NULL,
  status            STRING,
  n_series          INT64,
  n_models          INT64,
  runtime_seconds   FLOAT64,
  job_telemetry     JSON
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
  best_params    JSON,
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
  quantiles     JSON
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
}

# The four collection tables above are the run registry. They are ALWAYS native BigQuery
# (D19): native supports the JSON column type (so raw_config/job_telemetry/quantiles/best_params
# are real JSON, not STRING) and WRITE_TRUNCATE (so a reseed is a clean truncate, not a
# streaming-buffer-bounded DELETE). No BigLake connection or warehouse bucket is needed for them.
_REGISTRY_TABLES: tuple[str, ...] = tuple(_TABLE_BODIES)

# The example input table, shipped in BOTH storage formats so a deployment can benchmark the
# identical series as managed Iceberg vs native BigQuery (the engines read either transparently
# through BigQuery's table interface). `{name}` is filled per variant; `{{d}}` survives .format()
# as the `{d}` dataset placeholder every other body uses.
#
# Columns carry business names; the config maps each to a role (date→date_col). The shipped example
# is univariate (`y` history + holidays). To feed an exogenous regressor, add its column here and
# name it in `features.exog` — the generic exog seam consumes it, no code change.
_SOURCE_BODY_TEMPLATE = """\
CREATE TABLE IF NOT EXISTS `{{d}}.{name}` (
  ts_id       STRING NOT NULL,
  ds          DATE NOT NULL,
  y           FLOAT64,
  archetype   STRING,
  is_holiday  BOOL
)
PARTITION BY ds
CLUSTER BY ts_id"""

SOURCE_TABLE_ICEBERG = "source_series_iceberg"
SOURCE_TABLE_NATIVE = "source_series_native"
_SOURCE_TABLES: tuple[str, ...] = (SOURCE_TABLE_ICEBERG, SOURCE_TABLE_NATIVE)

for _src in _SOURCE_TABLES:
    _TABLE_BODIES[_src] = _SOURCE_BODY_TEMPLATE.format(name=_src)

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
    for name in _TABLE_BODIES:
        out[name] = _render_one(name, dataset, connection, warehouse_uri, iceberg=iceberg)
    return out


def _render_one(
    name: str,
    dataset: str,
    connection: str | None,
    warehouse_uri: str | None,
    *,
    iceberg: bool,
) -> str:
    """Render a single ``CREATE TABLE`` statement, Iceberg-wrapped iff ``iceberg``."""
    stmt = _TABLE_BODIES[name].format(d=dataset)
    if iceberg:
        assert warehouse_uri is not None  # callers pass both when iceberg=True
        options = _iceberg_options(name, warehouse_uri)
        stmt = f"{stmt}\nWITH CONNECTION `{connection}`\n{options}"
    return stmt + ";"


def render_deployment_ddl(
    dataset: str,
    *,
    connection: str,
    warehouse_uri: str,
) -> dict[str, str]:
    """Render the CREATE DDL for a real deployment: native registry + both source variants (D19).

    The storage policy is fixed here so callers don't juggle a per-table flag:

    - the four **registry** tables are always **native** BigQuery (native ``JSON`` columns +
      ``WRITE_TRUNCATE`` reseed; no BigLake connection needed);
    - ``source_series_iceberg`` is a managed-Iceberg table (needs ``connection`` +
      ``warehouse_uri``);
    - ``source_series_native`` is a plain native table.

    ``connection``/``warehouse_uri`` are required because the Iceberg source variant is always
    created. Every statement is idempotent (``CREATE TABLE IF NOT EXISTS``).
    """
    out: dict[str, str] = {}
    for name in TABLE_NAMES:
        iceberg = name == SOURCE_TABLE_ICEBERG
        out[name] = _render_one(name, dataset, connection, warehouse_uri, iceberg=iceberg)
    return out


def render_drop_tables(dataset: str) -> dict[str, str]:
    """Render ``{table_name: DROP TABLE IF EXISTS ...;}`` for all six tables (reset path).

    Pure string op (snapshot-testable); the destructive execution lives in ``bq.drop_all``. Used
    to tear a deployment down to bare metal before a clean ``ensure_tables`` recreates it in the
    current native/dual-format shape (the Iceberg→native registry switch is not an ``ALTER``).
    """
    return {name: f"DROP TABLE IF EXISTS `{dataset}.{name}`;" for name in TABLE_NAMES}
