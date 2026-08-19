"""Live BigQuery-native engine smoke (``@gcp``).

Runs :func:`scale_forecasting.engines.bigquery_engine.run` against live BigQuery for both
native models (``arima_plus``, ``timesfm``) at a small ``series_limit`` and
asserts the contract: each model lands canonical ``forecast_predictions`` + ``backtest_oof`` +
``forecast_metadata`` rows with ``compute_engine='bigquery'`` and a non-NULL metric panel, the run
header closes ``COMPLETED`` with ``bq_models`` populated, and the native models surface on
``v_model_leaderboard`` — the same read surface the Spark models land on.

Skipped unless ``SF_PROJECT_ID`` (+ ADC) is set (see ``tests/conftest.py``). Run manually::

    SF_PROJECT_ID=statmike-scale-forecasting \\
    SF_CONNECTION=statmike-scale-forecasting.us-central1.sf-iceberg \\
    SF_WAREHOUSE_URI=gs://statmike-scale-forecasting-warehouse/warehouse \\
    SF_DATASET_ID=scale_forecasting \\
        uv run pytest -m gcp tests/integration/test_bigquery_native_smoke.py

**Self-contained data.** This test seeds its *own* tiny univariate scratch ``source_series`` table
via the same :mod:`~scale_forecasting.data_gen.generator` and tears it down afterward — the "test
owns its data" discipline the registry round-trip uses. Two years of daily history keeps the
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

_MODELS = ["arima_plus", "timesfm"]
_SERIES_LIMIT = 20
_HORIZON = 28
_N_FOLDS = 2  # multi-fold backtest → exercises the fold loop + per-fold DROP MODEL cleanup
_HISTORY = 730  # ~2 years daily → training window > 1 year, so ARIMA_PLUS holidays apply
_SCRATCH_TABLE = "b3_native_smoke_source"


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    return Settings.resolve()


@pytest.fixture(scope="module")
def scratch_source(settings: Settings) -> Iterator[str]:
    """Seed a tiny univariate scratch ``source_series`` table, yield its name, drop it after.

    Uses :func:`generate_panel` + :func:`_to_source_rows` — the identical generator the production
    seed uses — so the panel is coherent. Loaded as a plain BQ table (the engine only SELECTs /
    CREATE MODELs against it), truncate-on-write so a rerun is clean.
    """
    from google.cloud import bigquery

    from scale_forecasting.data_gen.generator import GenConfig, generate_panel
    from scale_forecasting.data_gen.seed_spark import _to_source_rows

    client = bigquery.Client(project=settings.project_id)
    table_ref = settings.table_ref(_SCRATCH_TABLE)

    gen = GenConfig(history=_HISTORY, freq="D", start="2021-01-01", holidays=("US",))
    panel = generate_panel(_SERIES_LIMIT, gen, seed=7)
    rows = _to_source_rows(panel, ("US",))
    rows = rows.astype({"y": "float64", "is_holiday": "bool"})

    schema = [
        bigquery.SchemaField("ts_id", "STRING"),
        bigquery.SchemaField("ds", "DATE"),
        bigquery.SchemaField("y", "FLOAT"),
        bigquery.SchemaField("archetype", "STRING"),
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
        features={"holidays": ["US"]},
        # Scored evaluation lives entirely in the backtest path: the OOF rows and the
        # non-NaN metric panel this test asserts are only produced when backtesting is on. A
        # multi-fold plan also exercises the native fold loop + per-fold DROP MODEL cleanup (#161).
        backtest={"enabled": True, "n_folds": _N_FOLDS, "horizon": _HORIZON, "step": _HORIZON},
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

    # backtest_oof: N_FOLDS held-out folds (fold_id 0..N-1), horizon × series × folds per model,
    # y_true present. Multi-fold exercises the native fold loop + per-fold DROP cleanup (#161).
    oof = _poll_rows(
        client,
        f"SELECT model_type, COUNT(*) AS n, COUNT(DISTINCT fold_id) AS n_folds, "
        f"COUNTIF(fold_id NOT BETWEEN 0 AND {_N_FOLDS - 1}) AS bad_fold, "
        f"COUNTIF(y_true IS NULL) AS null_true "
        f"FROM `{d}.backtest_oof` WHERE run_id=@run_id GROUP BY model_type",
        run_id,
        lambda rows: len(rows) == n_models,
    )
    assert {r.model_type for r in oof} == set(_MODELS)
    for r in oof:
        assert r.n == _SERIES_LIMIT * _HORIZON * _N_FOLDS, r.model_type
        assert r.n_folds == _N_FOLDS, r.model_type
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

    # Fold-model cleanup (#161): each backtest fold trains a persisted sf_model_*_f{k} object solely
    # to read its held-out forecast, then DROPs it. After the run the final true-future model must
    # remain (it backs forecast_predictions) while no orphaned fold objects linger. TimesFM trains
    # no object, so this concerns arima_plus only. list_models is the live ground truth (no
    # INFORMATION_SCHEMA path-quoting to fight).
    from scale_forecasting.engines.bigquery_engine import _sanitize_identifier

    final_model = f"sf_model_arima_plus_{_sanitize_identifier(run_id)}"
    models_seen = {
        m.model_id
        for m in client.list_models(settings.dataset_ref)
        if m.model_id.startswith(final_model)
    }
    assert final_model in models_seen, models_seen
    orphaned_folds = {m for m in models_seen if m.startswith(f"{final_model}_f")}
    assert orphaned_folds == set(), orphaned_folds
