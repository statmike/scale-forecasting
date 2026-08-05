"""Live Spark Connect smoke (BUILD B5 demo arc, ``@spark`` + ``@gcp``).

Drives :func:`scale_forecasting.engines.spark_explode.run` over a **Dataproc Spark Connect**
endpoint from this process — the notebook path — instead of submitting a remote Dataproc batch. It
proves the injectable-session seam end-to-end against live infra: a caller-owned
``DataprocSparkSession`` runs the ``groupBy(bucket).applyInPandas`` cell fan-out, lands
``compute_engine='spark'`` cells in the live registry under one ``run_id``, and — because the engine
did *not* create the session — is left running for the caller to stop.

**Reachability.** Spark Connect needs outbound access to the Dataproc endpoint. This workstation's
egress boundary may block it (the same class of block that gated the Vertex Ray ingress). So the
test does a scratch ``spark.range(5).count()`` reachability check first and **skips** (does not
fail) if the session can't be created or reached — the documented ``main.run`` remote-batch fallback
(proven by ``test_main_orchestration_smoke``) uses the *identical* engine code, so an unreachable
endpoint is an environment limitation, not a product defect. Run where the endpoint is reachable::

    SF_PROJECT_ID=statmike-scale-forecasting \\
    SF_CONNECTION=statmike-scale-forecasting.us-central1.sf-iceberg \\
    SF_WAREHOUSE_URI=gs://statmike-scale-forecasting-warehouse/warehouse \\
    SF_DATASET_ID=scale_forecasting \\
    SF_DATAPROC_SUBNET=projects/<p>/regions/<r>/subnetworks/<s> \\
        uv run pytest -m "spark and gcp" tests/integration/test_spark_connect_smoke.py

**Self-contained data.** Seeds its own tiny univariate scratch ``source_series`` table and tears it
down after. ``run_name`` varies per invocation so the deterministic ``run_id`` is unique.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("pyspark")

from scale_forecasting.config import RunConfig  # noqa: E402
from scale_forecasting.registry.ids import make_run_id  # noqa: E402
from scale_forecasting.settings import Settings  # noqa: E402

pytestmark = [pytest.mark.spark, pytest.mark.gcp]

_MODELS = ["theta", "holtwinters"]
_SERIES_LIMIT = 6
_HORIZON = 28
_HISTORY = 730
_SCRATCH_TABLE = "b5_connect_smoke_source"
_RUNTIME_VERSION = "3.0"  # Spark Connect requires >= 3.0 (the batch default is left untouched)


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings.resolve()


@pytest.fixture(scope="module")
def scratch_source(settings: Settings) -> Iterator[str]:
    """Seed a scratch ``source_series`` table, yield its name, drop it after."""
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


def _connect_session(settings: Settings) -> Any:
    """Build a reachable Spark Connect session, or ``pytest.skip`` if it can't be reached.

    Creates a ``DataprocSparkSession`` pinned to runtime 3.0, then runs a scratch job to confirm the
    endpoint is reachable *and* a one-row ``applyInPandas`` to confirm driver↔worker Python parity
    (Dataproc 3.0 workers are Python 3.12). Any failure — missing client dep, provisioning error,
    egress block, or Python-minor skew — skips; the remote-batch fallback runs the same engine code
    with no local driver and is verified green in the ensemble/orchestration smokes.
    """
    import os

    try:
        from google.cloud.dataproc_spark_connect import DataprocSparkSession
        from google.cloud.dataproc_v1 import Session
    except ImportError as exc:  # pragma: no cover - client-only extra
        pytest.skip(f"dataproc-spark-connect not installed ([spark] extra): {exc}")

    region = os.environ.get("SF_DATAPROC_REGION", settings.region)
    subnet = os.environ.get("SF_DATAPROC_SUBNET")

    session_cfg = Session()
    session_cfg.runtime_config.version = _RUNTIME_VERSION
    if subnet:
        session_cfg.environment_config.execution_config.subnetwork_uri = subnet

    try:
        spark = (
            DataprocSparkSession.builder.projectId(settings.project_id)
            .location(region)
            .dataprocSessionConfig(session_cfg)
            .getOrCreate()
        )
        assert spark.range(5).count() == 5  # reachability check (JVM-only)
    except Exception as exc:  # noqa: BLE001 - any endpoint failure → skip, not fail
        pytest.skip(f"Spark Connect endpoint unreachable (use remote-batch fallback): {exc}")

    # Python-worker parity probe. range().count() is JVM-only, so it can't catch a driver↔worker
    # Python minor-version skew — but the engine's applyInPandas fan-out runs Python on the
    # workers, and Connect refuses to run mismatched minors. Dataproc 3.0 workers are Python 3.12;
    # if this driver venv is a different minor, skip (the remote-batch path runs the same engine
    # with no local driver, and is verified green in the ensemble/orchestration smokes).
    try:
        import pandas as pd

        probe = spark.range(1).groupBy("id").applyInPandas(lambda df: df, "id long")
        assert isinstance(probe.toPandas(), pd.DataFrame)
    except Exception as exc:  # noqa: BLE001 - worker-side failure (e.g. Python skew) → skip
        pytest.skip(
            "Spark Connect endpoint reachable but Python-worker parity unmet (driver Python "
            f"must match Dataproc 3.0 workers = 3.12); use remote-batch fallback: {exc}"
        )
    return spark


def _rows(client: Any, sql: str, run_id: str) -> list[Any]:
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    )
    return list(client.query(sql, job_config=job_config).result())


def _poll_rows(client: Any, sql: str, run_id: str, predicate: Any) -> list[Any]:
    rows: list[Any] = []
    for _ in range(12):
        rows = _rows(client, sql, run_id)
        if predicate(rows):
            return rows
        time.sleep(3)
    return rows


def test_spark_connect_explode_smoke(settings: Settings, scratch_source: str) -> None:
    from google.cloud import bigquery

    from scale_forecasting.engines import spark_explode

    spark = _connect_session(settings)
    client = bigquery.Client(project=settings.project_id)
    d = settings.dataset_ref

    cfg = RunConfig(
        run_name=f"b5 connect smoke {int(time.time())}",
        python_runtime="spark",
        spark_method="explode",
        data={"source_table": scratch_source, "horizon": _HORIZON, "series_limit": _SERIES_LIMIT},
        models=_MODELS,
        features={"holidays": ["US"]},
    )
    run_id = make_run_id(cfg)

    try:
        # Injected caller-owned session → the engine runs the fan-out over Connect and does NOT stop
        # it. Standalone mode (manage_header defaults True) so this run owns its own header.
        spark_explode.run(cfg, settings=settings, spark=spark)

        # The engine left the caller's session live (owns_session was False) — a trivial job runs.
        assert spark.range(3).count() == 3
    finally:
        spark.stop()

    # Header closed COMPLETED for this standalone run.
    header = next(
        iter(
            _rows(
                client,
                f"SELECT status, n_series FROM `{d}.run_registry` WHERE run_id=@run_id",
                run_id,
            )
        )
    )
    assert header.status == "COMPLETED"
    assert header.n_series == _SERIES_LIMIT

    # v_model_leaderboard: both models ran as Spark cells over the Connect endpoint, one run_id.
    board = _poll_rows(
        client,
        f"SELECT model_type, compute_engine, n_cells "
        f"FROM `{d}.v_model_leaderboard` WHERE run_id=@run_id",
        run_id,
        lambda rows: {r.model_type for r in rows} == set(_MODELS),
    )
    by_model = {r.model_type: r for r in board}
    assert set(by_model) == set(_MODELS)
    for m in _MODELS:
        assert by_model[m].compute_engine == "spark", m
        assert by_model[m].n_cells == _SERIES_LIMIT, m
