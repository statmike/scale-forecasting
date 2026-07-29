"""Live parallel-orchestration smoke (BUILD Arc B, ``@gcp``).

Runs :func:`scale_forecasting.main.run` end-to-end against live GCP: a **mixed** config (one Spark
model + the three BigQuery-native models) executes both runtimes **in parallel under one shared
``run_id`` and one ``run_registry`` header**, landing a real Dataproc Serverless batch *and* the
in-BigQuery SQL engine in the same run. This is the Arc B contract the offline tests can't reach —
that the two compute tracks share a header and are comparable on ``v_model_leaderboard`` (DESIGN
§3.3, the "wall-clock ≈ max(python, bq), not sum" thesis).

Asserts: exactly **one** ``run_registry`` row, ``COMPLETED``, ``bq_models`` populated with the three
natives, ``n_models`` == the full model count; ``v_model_leaderboard`` shows the Spark model with
``compute_engine='bigquery'`` *and* — critically — the Spark model with ``compute_engine='spark'``,
both under the **same** ``run_id``. That single-run, two-engine leaderboard *is* the showpiece.

Skipped unless ``SF_PROJECT_ID`` is set (see ``tests/conftest.py``). It also needs the Dataproc
batch-infra env (:class:`~scale_forecasting.submit.BatchInfra`) and ADC. Beyond the ``SF_*``
identity the writers resolve (``SF_PROJECT_ID`` / ``SF_CONNECTION`` / ``SF_WAREHOUSE_URI`` /
``SF_DATASET_ID`` / ``SF_REGION``), set the four batch-infra vars — ``SF_CODE_BUCKET``,
``SF_CONTAINER_IMAGE``, ``SF_COMPUTE_SA``, ``SF_SUBNETWORK_URI`` — then run::

    uv run pytest -m gcp tests/integration/test_main_orchestration_smoke.py

This launches a real Dataproc batch (~5-9 min provision→terminal) plus the in-BigQuery engine, so it
is materially slower + costlier than the pure-BQ B3 smoke; that spend is the point of the test.

**Self-contained data.** Like the B3 native smoke, the shipped 100k seed's ``price_index`` is
all-NULL (generated ``with_exog=False``), so ARIMA_PLUS_XREG can't train on it. This test seeds its
own tiny exog-carrying scratch ``source_series`` table via the same generator and tears it down
after — the "test owns its data" discipline. ``run_name`` varies per invocation so the deterministic
``run_id`` is unique (append-only cell tables can't be DELETE-d while buffered).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

pytestmark = pytest.mark.gcp

_SPARK_MODEL = "theta"
_NATIVE_MODELS = ["arima_plus", "arima_plus_xreg", "timesfm"]
_ALL_MODELS = [_SPARK_MODEL, *_NATIVE_MODELS]
_SERIES_LIMIT = 10
_HORIZON = 28
_HISTORY = 730  # ~2 years daily → training window > 1 year, so ARIMA_PLUS holidays apply
_SCRATCH_TABLE = "arc_b_mixed_smoke_source"


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    return Settings.resolve()


@pytest.fixture(scope="module")
def scratch_source(settings: Settings) -> Iterator[str]:
    """Seed an exog-carrying scratch ``source_series`` table, yield its name, drop it after.

    The same generator the production seed uses (``with_exog=True``), loaded as a plain BQ table
    both the Spark connector and the BigQuery engine read. Truncate-on-write so a rerun is clean.
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
    # A per-invocation run_name → a unique deterministic run_id (see module docstring).
    return RunConfig(
        run_name=f"arc b mixed smoke {int(time.time())}",
        python_runtime="spark",
        spark_method="explode",
        data={"source_table": source_table, "horizon": _HORIZON, "series_limit": _SERIES_LIMIT},
        models=_ALL_MODELS,
        features={"holidays": ["US"], "exog": ["price_index"]},
        # Backtest ON so the Spark path emits an OOF metric panel: without it the Python engine only
        # forecasts (no metrics), while the BQ natives always score a held-out fold — so the Spark
        # model would land NULL mean_wape and the two runtimes wouldn't be comparable, which is the
        # whole point of the single-run leaderboard. A small fold count keeps the batch cheap.
        backtest={"enabled": True, "n_folds": 2, "horizon": _HORIZON, "step": _HORIZON},
    )


def _rows(client: Any, sql: str, run_id: str) -> list[Any]:
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    )
    return list(client.query(sql, job_config=job_config).result())


def _poll_rows(client: Any, sql: str, run_id: str, predicate: Any) -> list[Any]:
    """Write API rows are async-visible; poll briefly until ``predicate(rows)`` holds."""
    rows: list[Any] = []
    for _ in range(10):
        rows = _rows(client, sql, run_id)
        if predicate(rows):
            return rows
        time.sleep(3)
    return rows


def test_main_orchestration_parallel_smoke(settings: Settings, scratch_source: str) -> None:
    from google.cloud import bigquery

    from scale_forecasting import main

    client = bigquery.Client(project=settings.project_id)
    cfg = _cfg(scratch_source)
    run_id = make_run_id(cfg)
    d = settings.dataset_ref

    # The whole point: one call, both runtimes, one run_id. Returns the shared run_id it wrote.
    returned = main.run(cfg)
    assert returned == run_id

    # Exactly ONE header row for the run, COMPLETED, with the natives recorded in bq_models and
    # n_models spanning BOTH runtimes' models (the shared-header invariant).
    headers = _rows(
        client,
        f"SELECT status, n_models, n_series, runtime_seconds, python_runtime, "
        f"ARRAY_TO_STRING(bq_models, ',') AS bq_models "
        f"FROM `{d}.run_registry` WHERE run_id=@run_id",
        run_id,
    )
    assert len(headers) == 1, "one config → one shared header row"
    header = headers[0]
    assert header.status == "COMPLETED"
    assert header.n_models == len(_ALL_MODELS)
    assert header.python_runtime == "spark"
    assert header.runtime_seconds > 0
    assert set(header.bq_models.split(",")) == set(_NATIVE_MODELS)

    # v_model_leaderboard: every model surfaces under the SAME run_id, and the compute_engine column
    # cleanly splits the Spark model from the natives — the single-run, two-engine comparison.
    board = _poll_rows(
        client,
        f"SELECT model_type, compute_engine, n_cells, mean_wape "
        f"FROM `{d}.v_model_leaderboard` WHERE run_id=@run_id",
        run_id,
        lambda rows: {r.model_type for r in rows} == set(_ALL_MODELS),
    )
    by_model = {r.model_type: r for r in board}
    assert set(by_model) == set(_ALL_MODELS), "both runtimes' models on one leaderboard"

    # The Spark model ran on Spark; the natives ran in BigQuery — under one run_id.
    assert by_model[_SPARK_MODEL].compute_engine == "spark"
    for m in _NATIVE_MODELS:
        assert by_model[m].compute_engine == "bigquery", m

    # Every model produced scored cells (a real, comparable metric, not an empty row).
    for m in _ALL_MODELS:
        assert by_model[m].n_cells == _SERIES_LIMIT, m
        assert by_model[m].mean_wape is not None, m
