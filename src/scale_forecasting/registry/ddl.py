"""CREATE TABLE DDL for the registry + example data.

Seven tables are the source of truth for the whole system. Storage format is split by role:

- The five **run-collection** tables (``run_registry``, ``run_jobs``, ``forecast_metadata``,
  ``forecast_predictions``, ``backtest_oof``) are always **native BigQuery**. Native gives us
  the real ``JSON`` column type (``raw_config`` / ``job_telemetry`` / ``quantiles`` /
  ``best_params``) and ``WRITE_TRUNCATE`` reseed, and needs no BigLake connection.
- The example input table ships in **both** formats — ``source_series_iceberg`` (a
  BigQuery-managed Apache Iceberg table: columns/partition/cluster wrapped with
  ``WITH CONNECTION ... OPTIONS(table_format='ICEBERG', ...)`` at the warehouse bucket) and
  ``source_series_native`` (plain native) — so a deployment can benchmark the identical series
  on either storage. The engines read both transparently through BigQuery's table interface.

Rendering is a pure string operation (no BigQuery client), so it is snapshot-tested
offline; ``registry/bq.ensure_tables`` executes what ``render_deployment_ddl`` renders.

The ``JSON`` columns use the native ``JSON`` type (only the native registry carries them; the
Iceberg source table has none). The row assemblers in ``bq.py`` serialize JSON text, which a
``JSON`` column parses on ingest; read them back with ``.`` field access or ``JSON_VALUE``.

Schema evolution is additive: when a new NULLABLE column is added to a body below,
``render_migrations`` derives an ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` for it from
the *same* bodies, so a table created under an older schema can be brought up to date
without the CREATE and the migration ever drifting apart (``ensure_tables`` runs both).

**The two families are addressable separately.** ``REGISTRY_TABLE_NAMES`` and
``SOURCE_TABLE_NAMES`` partition ``TABLE_NAMES``, and every renderer takes an optional ``tables``
subset plus its own ``dataset``. That is what lets a deployment put its registry in one dataset and
its source panel in another — the two have separate lifetimes (you redesign source data far less
often than you clear a registry) and nothing about them requires a shared fate. Renderers stay dumb
about the policy; ``render_deployment_ddl`` is the one place that knows which family goes where.

Public surface: ``TABLE_NAMES``, ``REGISTRY_TABLE_NAMES``, ``SOURCE_TABLE_NAMES``,
``SOURCE_TABLE_ICEBERG``, ``SOURCE_TABLE_NATIVE``, ``render_deployment_ddl``,
``render_create_tables``, ``render_drop_tables``, ``additive_columns``, ``render_migrations``.
"""

from __future__ import annotations

from collections.abc import Sequence

# Table bodies: columns + PARTITION BY + CLUSTER BY.
# `{d}` is the dataset ref (`project.dataset` or `dataset`). No trailing semicolon —
# the renderer appends the OPTIONS clause and the semicolon.
_TABLE_BODIES: dict[str, str] = {
    "run_registry": """\
CREATE TABLE IF NOT EXISTS `{d}.run_registry` (
  run_id            STRING NOT NULL,
  created_at        TIMESTAMP NOT NULL,
  snapshot_millis   INT64,
  user_id           STRING,
  git_sha           STRING,
  python_runtime    STRING,
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
    "run_jobs": """\
CREATE TABLE IF NOT EXISTS `{d}.run_jobs` (
  job_id           STRING NOT NULL,
  run_id           STRING NOT NULL,
  family           STRING NOT NULL,
  attempt          INT64 NOT NULL,
  runtime          STRING,
  spark_mode       STRING,
  hardware         STRING,
  gpu_type         STRING,
  system_job_id    STRING,
  status           STRING,
  created_at       TIMESTAMP NOT NULL,
  started_at       TIMESTAMP,
  ended_at         TIMESTAMP,
  runtime_seconds  FLOAT64,
  job_telemetry    JSON
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id, family""",
    "forecast_metadata": """\
CREATE TABLE IF NOT EXISTS `{d}.forecast_metadata` (
  run_id         STRING NOT NULL,
  ts_id          STRING NOT NULL,
  model_type     STRING NOT NULL,
  compute_engine STRING NOT NULL,
  model_hash     STRING NOT NULL,
  ensemble_id    STRING,
  fold_id        INT64,
  mae FLOAT64, rmse FLOAT64, mse FLOAT64, mape FLOAT64, smape FLOAT64,
  wape FLOAT64, mase FLOAT64, rmsse FLOAT64, bias FLOAT64,
  coverage FLOAT64, pinball FLOAT64,
  fit_seconds    FLOAT64,
  best_params    JSON,
  model_artifact STRING,
  created_at     TIMESTAMP NOT NULL,
  worker_id      STRING,
  cell_started_at TIMESTAMP,
  cell_ended_at  TIMESTAMP,
  cpu_seconds    FLOAT64,
  process_rss_bytes INT64,
  peak_gpu_bytes INT64,
  intraop_threads INT64,
  n_obs          INT64
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id, model_type""",
    "forecast_predictions": """\
CREATE TABLE IF NOT EXISTS `{d}.forecast_predictions` (
  run_id        STRING NOT NULL,
  ts_id         STRING NOT NULL,
  model_type    STRING NOT NULL,
  compute_engine STRING,
  ensemble_id   STRING,
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

# The five collection tables above are the run registry. They are ALWAYS native BigQuery:
# native supports the JSON column type (so raw_config/job_telemetry/quantiles/best_params
# are real JSON, not STRING) and WRITE_TRUNCATE (so a reseed is a clean truncate, not a
# streaming-buffer-bounded DELETE). No BigLake connection or warehouse bucket is needed for them.
#
# This tuple is PUBLIC because it is the definition of "a registry": these five names, in one
# dataset. BigQuery's namespace then makes `project.dataset` a guaranteed-unique registry key for
# free — a dataset can hold exactly one table called `run_registry` — so nothing has to validate
# uniqueness, and the key is safe to reuse verbatim as a GCS path segment.
REGISTRY_TABLE_NAMES: tuple[str, ...] = tuple(_TABLE_BODIES)

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
# Public for the same reason as REGISTRY_TABLE_NAMES: these two are the *input panel*, a separate
# concern with a separate lifetime. Clearing a registry must never take them with it.
SOURCE_TABLE_NAMES: tuple[str, ...] = (SOURCE_TABLE_ICEBERG, SOURCE_TABLE_NATIVE)

for _src in SOURCE_TABLE_NAMES:
    _TABLE_BODIES[_src] = _SOURCE_BODY_TEMPLATE.format(name=_src)

# Both families. Kept as the default subset for every renderer so existing callers are unchanged.
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


def render_migrations(dataset: str, *, tables: Sequence[str] | None = None) -> dict[str, str]:
    """Render ``{table_name: ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...}`` for additive columns.

    One idempotent ``ALTER`` per table adds every nullable column if absent, so a table created
    under an older schema is brought current without touching existing rows (they read NULL) and
    without a hand-written migration. Tables with no nullable columns are omitted.

    ``tables`` restricts the output to a subset (default: all of them) — pass
    `REGISTRY_TABLE_NAMES` or `SOURCE_TABLE_NAMES` when the two families live in different
    datasets, so each ``ALTER`` is addressed to the dataset that actually holds the table.
    """
    out: dict[str, str] = {}
    for name in tables if tables is not None else TABLE_NAMES:
        cols = additive_columns(name)
        if not cols:
            continue
        adds = ",\n  ".join(f"ADD COLUMN IF NOT EXISTS {c} {t}" for c, t in cols)
        out[name] = f"ALTER TABLE `{dataset}.{name}`\n  {adds};"
    return out


def _iceberg_options(table_name: str, warehouse_uri: str) -> str:
    """The OPTIONS block that makes a table a managed Iceberg table."""
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
    tables: Sequence[str] | None = None,
) -> dict[str, str]:
    """Render ``{table_name: CREATE TABLE statement}`` for all tables.

    Args:
        dataset: dataset ref, ``project.dataset`` or ``dataset``.
        connection: Cloud Resource / BigLake connection ``project.region.name``;
            required when ``iceberg`` is True.
        warehouse_uri: GCS warehouse root (e.g. ``gs://bucket/warehouse``); required
            when ``iceberg`` is True.
        iceberg: when True (default) render managed-Iceberg DDL; when False render
            plain native BigQuery tables (the native-BigQuery fallback, or for a BQ emulator).
        tables: restrict to a subset (default: all). See `render_migrations`.

    Every statement is idempotent (``CREATE TABLE IF NOT EXISTS``).
    """
    if iceberg and (connection is None or warehouse_uri is None):
        raise ValueError("iceberg=True requires both 'connection' and 'warehouse_uri'")

    out: dict[str, str] = {}
    for name in tables if tables is not None else _TABLE_BODIES:
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
    source_dataset: str | None = None,
) -> dict[str, str]:
    """Render the CREATE DDL for a real deployment: native registry + both source variants.

    The storage policy is fixed here so callers don't juggle a per-table flag:

    - the five **registry** tables are always **native** BigQuery (native ``JSON`` columns +
      ``WRITE_TRUNCATE`` reseed; no BigLake connection needed);
    - ``source_series_iceberg`` is a managed-Iceberg table (needs ``connection`` +
      ``warehouse_uri``);
    - ``source_series_native`` is a plain native table.

    ``source_dataset`` places the two source tables somewhere other than ``dataset`` (default:
    alongside the registry, which is every deployment that has never asked for otherwise). This is
    the only function that knows which family belongs where; the renderers it calls take a plain
    dataset + table list and have no policy in them.

    ``connection``/``warehouse_uri`` are required because the Iceberg source variant is always
    created. Every statement is idempotent (``CREATE TABLE IF NOT EXISTS``).
    """
    out: dict[str, str] = {}
    for name in REGISTRY_TABLE_NAMES:
        out[name] = _render_one(name, dataset, connection, warehouse_uri, iceberg=False)
    for name in SOURCE_TABLE_NAMES:
        iceberg = name == SOURCE_TABLE_ICEBERG
        out[name] = _render_one(
            name, source_dataset or dataset, connection, warehouse_uri, iceberg=iceberg
        )
    return out


def render_drop_tables(dataset: str, *, tables: Sequence[str] | None = None) -> dict[str, str]:
    """Render ``{table_name: DROP TABLE IF EXISTS ...;}`` for a table family (destructive paths).

    Pure string op (snapshot-testable); the execution lives outside the product, in the control
    tower's dev-only teardown tools. ``tables`` defaults to all seven for backwards compatibility,
    but a caller that means "clear the registry" must pass `REGISTRY_TABLE_NAMES` — dropping the
    source panel as part of a registry clear is a different, much more expensive operation, and
    conflating the two is exactly the defect this subset argument exists to prevent.
    """
    names = tables if tables is not None else TABLE_NAMES
    return {name: f"DROP TABLE IF EXISTS `{dataset}.{name}`;" for name in names}
