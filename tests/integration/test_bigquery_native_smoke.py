"""Live BigQuery-native engine smoke (BUILD B3, ``@gcp``).

Runs :func:`scale_forecasting.engines.bigquery_engine.run` against live BigQuery for all three
native models (``arima_plus``, ``arima_plus_xreg``, ``timesfm``) at a small ``series_limit`` and
asserts the B3 contract: each model lands canonical ``forecast_predictions`` + ``backtest_oof`` +
``forecast_metadata`` rows with ``compute_engine='bigquery'`` and a non-NULL metric panel, the run
header closes ``COMPLETED`` with ``bq_models`` populated, and the native models surface on
``v_model_leaderboard`` — the same read surface the Spark models land on (DESIGN §3.3).

Skipped unless ``SF_PROJECT_ID`` (+ ADC) is set (see ``tests/conftest.py``). Run manually::

    SF_PROJECT_ID=statmike-scale-forecasting \\
    SF_CONNECTION=statmike-scale-forecasting.us-central1.sf-iceberg \\
    SF_WAREHOUSE_URI=gs://statmike-scale-forecasting-warehouse/warehouse \\
    SF_DATASET_ID=scale_forecasting \\
        uv run pytest -m gcp tests/integration/test_bigquery_native_smoke.py

**Self-contained data.** The shipped 100k seed was generated with ``with_exog=False``, so its
``price_index`` column is all-NULL and ARIMA_PLUS_XREG cannot train on it. Rather than drop the XREG
model from the parity check, this test seeds its *own* tiny exog-carrying scratch table via the same
:mod:`~scale_forecasting.data_gen.generator` (``with_exog=True``) and tears it down afterward — the
"test owns its data" discipline the registry round-trip uses. Two years of daily history keeps the
training window over a year (so ARIMA_PLUS holiday effects apply) while staying cents-cheap.

The engine derives ``run_id`` deterministically from the config, so this test varies ``run_name``
per invocation (append-only cell tables can't be DELETE-d while buffered, so a unique run_id keeps
each invocation isolated).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.engines import bigquery_engine as be
from scale_forecasting.metrics import METRIC_NAMES
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

pytestmark = pytest.mark.gcp

_MODELS = ["arima_plus", "arima_plus_xreg", "timesfm"]
_SERIES_LIMIT = 20
_HORIZON = 28
_HISTORY = 730  # ~2 years daily → training window > 1 year, so ARIMA_PLUS holidays apply
_SCRATCH_TABLE = "b3_native_smoke_source"


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    return Settings.resolve()


@pytest.fixture(scope="module")
def scratch_source(settings: Settings) -> Iterator[str]:
    """Seed an exog-carrying scratch ``source_series`` table, yield its name, drop it after.

    Uses :func:`generate_panel` (``with_exog=True``) + :func:`_to_source_rows` — the identical
    generator the production seed uses — so the panel is coherent and the exog column is populated,
    which the shipped seed's ``price_index`` is not. Loaded as a plain BQ table (the engine only
    SELECTs / CREATE MODELs against it), truncate-on-write so a rerun is clean.
    """
    from google.cloud import bigquery

    from scale_forecasting.data_gen.generator import GenConfig, generate_panel
    from scale_forecasting.data_gen.seed_spark import _to_source_rows

    client = bigquery.Client(project=settings.project_id)
    table_ref = settings.table_ref(_SCRATCH_TABLE)

    gen = GenConfig(
        history=_HISTORY, freq="D", start="2021-01-01", holidays=("US",), with_exog=True
    )
    panel = generate_panel(_SERIES_LIMIT, gen, seed=7)
    rows = _to_source_rows(panel, ("US",))
    # Plain (non-nullable) dtypes for the load — with_exog=True guarantees no NA in price_index.
    rows = rows.astype({"y": "float64", "price_index": "float64", "is_holiday": "bool"})

    schema = [
        bigquery.SchemaField("ts_id", "STRING"),
        bigquery.SchemaField("ds", "DATE"),
        bigquery.SchemaField("y", "FLOAT"),
        bigquery.SchemaField("archetype", "STRING"),
        bigquery.SchemaField("price_index", "FLOAT"),
        bigquery.SchemaField("is_holiday", "BOOL"),
    ]
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(rows, table_ref, job_config=job_config).result()
    try:
        yield _SCRATCH_TABLE
    finally:
        client.delete_table(table_ref, not_found_ok=True)


def _cfg(source_table: str) -> RunConfig:
    # A per-invocation run_name → a unique deterministic run_id, so append-only cell rows stay
    # isolated across reruns (see module docstring). int(time.time()) is the only nondeterminism.
    return RunConfig(
        run_name=f"b3 native smoke {int(time.time())}",
        data={"source_table": source_table, "horizon": _HORIZON, "series_limit": _SERIES_LIMIT},
        models=_MODELS,
        features={"holidays": ["US"], "exog": ["price_index"]},
    )


def _run_id_params(run_id: str) -> Any:
    from google.cloud import bigquery

    return bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    )


def _rows(client: Any, sql: str, run_id: str) -> list[Any]:
    return list(client.query(sql, job_config=_run_id_params(run_id)).result())


def _poll_rows(client: Any, sql: str, run_id: str, predicate: Any) -> list[Any]:
    """Write API rows are async-visible; poll briefly until ``predicate(rows)`` holds."""
    rows: list[Any] = []
    for _ in range(10):
        rows = _rows(client, sql, run_id)
        if predicate(rows):
            return rows
        time.sleep(3)
    return rows


def test_bigquery_native_smoke(settings: Settings, scratch_source: str) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=settings.project_id)
    cfg = _cfg(scratch_source)
    run_id = make_run_id(cfg)
    d = settings.dataset_ref
    n_models = len(_MODELS)

    # Full engine run: header lifecycle + CREATE MODEL / forecast INSERT per model + metric
    # write-back. Idempotent by run_id; a fresh run_id per invocation keeps this isolated.
    be.run(cfg, _MODELS)

    # Header closed COMPLETED with the native models recorded in the bq_models array.
    header = next(
        iter(
            _rows(
                client,
                f"SELECT status, n_models, n_series, runtime_seconds, "
                f"ARRAY_TO_STRING(bq_models, ',') AS bq_models "
                f"FROM `{d}.run_registry` WHERE run_id=@run_id",
                run_id,
            )
        )
    )
    assert header.status == "COMPLETED"
    assert header.n_models == n_models
    assert header.n_series == _SERIES_LIMIT
    assert header.runtime_seconds > 0
    assert set(header.bq_models.split(",")) == set(_MODELS)

    # forecast_predictions: horizon × series per model, all compute_engine='bigquery'.
    preds = _poll_rows(
        client,
        f"SELECT model_type, COUNT(*) AS n, COUNT(DISTINCT ts_id) AS n_series, "
        f"COUNTIF(compute_engine != 'bigquery') AS wrong_engine, "
        f"COUNTIF(yhat IS NULL) AS null_yhat "
        f"FROM `{d}.forecast_predictions` WHERE run_id=@run_id GROUP BY model_type",
        run_id,
        lambda rows: len(rows) == n_models,
    )
    assert {r.model_type for r in preds} == set(_MODELS)
    for r in preds:
        assert r.n_series == _SERIES_LIMIT, r.model_type
        assert r.n == _SERIES_LIMIT * _HORIZON, r.model_type
        assert r.wrong_engine == 0, r.model_type
        assert r.null_yhat == 0, r.model_type

    # backtest_oof: one held-out fold (fold_id=0), horizon × series per model, y_true present.
    oof = _poll_rows(
        client,
        f"SELECT model_type, COUNT(*) AS n, COUNTIF(fold_id != 0) AS bad_fold, "
        f"COUNTIF(y_true IS NULL) AS null_true "
        f"FROM `{d}.backtest_oof` WHERE run_id=@run_id GROUP BY model_type",
        run_id,
        lambda rows: len(rows) == n_models,
    )
    assert {r.model_type for r in oof} == set(_MODELS)
    for r in oof:
        assert r.n == _SERIES_LIMIT * _HORIZON, r.model_type
        assert r.bad_fold == 0, r.model_type
        assert r.null_true == 0, r.model_type

    # forecast_metadata: one row per (series, model), fold_id NULL, full non-NULL metric panel.
    metric_nonnull = ", ".join(f"COUNTIF({m} IS NULL) AS null_{m}" for m in METRIC_NAMES)
    meta = _poll_rows(
        client,
        f"SELECT model_type, COUNT(*) AS n, COUNTIF(fold_id IS NOT NULL) AS bad_fold, "
        f"COUNTIF(compute_engine != 'bigquery') AS wrong_engine, "
        f"COUNTIF(best_params IS NULL) AS null_params, {metric_nonnull} "
        f"FROM `{d}.forecast_metadata` WHERE run_id=@run_id GROUP BY model_type",
        run_id,
        lambda rows: len(rows) == n_models,
    )
    assert {r.model_type for r in meta} == set(_MODELS)
    for r in meta:
        assert r.n == _SERIES_LIMIT, r.model_type
        assert r.bad_fold == 0, r.model_type
        assert r.wrong_engine == 0, r.model_type
        assert r.null_params == 0, r.model_type
        # Every scored metric is non-NULL for every cell — full parity with the Python models.
        for m in METRIC_NAMES:
            assert getattr(r, f"null_{m}") == 0, f"{r.model_type}.{m}"

    # v_model_leaderboard surfaces the native models with compute_engine='bigquery' — the same read
    # surface the Spark models land on, so a native model and a Spark model are directly comparable.
    board = _rows(
        client,
        f"SELECT model_type, compute_engine, n_cells, mean_wape "
        f"FROM `{d}.v_model_leaderboard` WHERE run_id=@run_id",
        run_id,
    )
    assert {r.model_type for r in board} == set(_MODELS)
    for r in board:
        assert r.compute_engine == "bigquery", r.model_type
        assert r.n_cells == _SERIES_LIMIT, r.model_type
        assert r.mean_wape is not None, r.model_type
