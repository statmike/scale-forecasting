"""Live CPU-only Vertex Ray smoke (BUILD B4, ``@raylive`` + ``@gcp``).

The GPU-free sibling of ``test_ray_gpu_smoke``. It proves the part that was never in doubt about
the *models* but had never been exercised *live* — the Vertex Ray **lifecycle** itself: stage the
config to GCS, create a fixed-size CPU cluster, connect to its dashboard, submit the Ray Job, poll
it to completion, and tear the cluster down — all under one shared ``run_id`` with the in-BigQuery
native engine running in parallel. No T4, no ``NVIDIA_T4_GPUS`` quota: a CPU-only run still walks
the entire :func:`scale_forecasting.ray_submit.submit_ray` path (the dashboard-connect handshake is
the step that most needs a live confirmation), so a green run here is the standing evidence that
Ray-on-Vertex works end-to-end. The fractional-T4 NeuralProphet showpiece is the *only* thing this
can't cover — that's ``test_ray_gpu_smoke``'s job when T4 quota is in hand.

With ``compute.use_gpu=false`` and no ``deep_learning`` model in the mix,
:func:`~scale_forecasting.engines.ray_io.plan_cluster` sizes the GPU pool to **zero** nodes and
provisions the CPU worker pool only — the same code the GPU path runs, one config flag apart (G2).

Asserts:

* exactly **one** ``run_registry`` row, ``COMPLETED``, ``python_runtime='ray'``, natives in
  ``bq_models``, ``n_models`` == the full model count;
* ``v_model_leaderboard`` shows every model under the same ``run_id`` — the Python-runtime models
  with ``compute_engine='ray'`` (each with a non-NULL ``mean_wape`` from a real OOF fold), the
  natives with ``compute_engine='bigquery'``;
* the header's ``job_telemetry`` records the CPU-only sizing — ``runtime='ray'``,
  ``gpu_node_count == 0``, ``cpu_node_count >= 1``;
* the ephemeral cluster is **gone afterward** — the teardown-in-``finally`` guarantee.

Skipped unless **both** ``SF_ENABLE_RAY`` and ``SF_PROJECT_ID`` are set (see ``tests/conftest.py``):
it provisions a real (billed) Vertex Ray cluster (~15-25 min) — but needs **no GPU quota**. Beyond
the ``SF_*`` identity the writers resolve, set the Ray infra vars (``SF_COMPUTE_SA`` /
``SF_CODE_BUCKET`` required; ``SF_RAY_NETWORK`` optional — unset uses the public endpoint;
optionally ``SF_CONTAINER_IMAGE`` / ``SF_RAY_VERSION``), then run::

    SF_ENABLE_RAY=1 uv run pytest -m raylive tests/integration/test_ray_cpu_smoke.py

**Self-contained data.** Like the GPU smoke, the shipped seed's ``price_index`` is all-NULL, so
ARIMA_PLUS_XREG can't train on it; this test seeds its own tiny exog-carrying scratch
``source_series`` table and drops it after. ``run_name`` varies per invocation so the deterministic
``run_id`` is unique.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

pytestmark = [pytest.mark.raylive, pytest.mark.gcp]

# All CPU models — statistical Python-runtime models plus the in-BigQuery natives. Deliberately no
# ``neuralprophet`` (the only ``deep_learning`` family): its absence is what drives plan_cluster's
# GPU pool to zero, keeping this run GPU-free.
_CPU_MODELS = ["theta", "holtwinters"]
_NATIVE_MODELS = ["arima_plus", "arima_plus_xreg", "timesfm"]
_ALL_MODELS = [*_CPU_MODELS, *_NATIVE_MODELS]
_SERIES_LIMIT = 6
_HORIZON = 28
_HISTORY = 730  # ~2 years daily → training window > 1 year, so ARIMA_PLUS holidays apply
_SCRATCH_TABLE = "b4_ray_cpu_smoke_source"


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    return Settings.resolve()


@pytest.fixture(scope="module")
def scratch_source(settings: Settings) -> Iterator[str]:
    """Seed an exog-carrying scratch ``source_series`` table, yield its name, drop it after."""
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
    # Per-invocation run_name → a unique deterministic run_id (append-only cell tables can't be
    # DELETE-d while buffered, so a fresh id keeps reruns clean).
    return RunConfig(
        run_name=f"b4 ray cpu smoke {int(time.time())}",
        python_runtime="ray",
        data={"source_table": source_table, "horizon": _HORIZON, "series_limit": _SERIES_LIMIT},
        models=_ALL_MODELS,
        features={"holidays": ["US"], "exog": ["price_index"]},
        # Backtest ON so the Ray path emits an OOF metric panel comparable to the natives (which
        # always score a held-out fold). A small fold count keeps the run cheap.
        backtest={"enabled": True, "n_folds": 2, "horizon": _HORIZON, "step": _HORIZON},
        # use_gpu=false → plan_cluster sizes the GPU pool to zero; only the CPU worker pool is
        # provisioned. ray_regions: hop across US regions if one transiently stocks out on capacity.
        compute={
            "use_gpu": False,
            "ray_regions": ["us-central1", "us-east1", "us-west1"],
        },
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


def test_ray_cpu_fixed_cluster_smoke(settings: Settings, scratch_source: str) -> None:
    from google.cloud import bigquery

    from scale_forecasting import main
    from scale_forecasting.engines import ray_io

    client = bigquery.Client(project=settings.project_id)
    cfg = _cfg(scratch_source)
    run_id = make_run_id(cfg)
    d = settings.dataset_ref

    # The pure plan the run will provision — CPU pool only, GPU pool sized to zero. Names the
    # ephemeral cluster so we can confirm it's gone at the end.
    plan = ray_io.plan_cluster(cfg, models=_CPU_MODELS, run_id=run_id)
    assert plan.gpu_node_count == 0, "CPU-only run must not size a GPU pool"
    assert plan.cpu_node_count >= 1

    # One call, both runtimes, one run_id. Returns the shared run_id it wrote.
    returned = main.run(cfg)
    assert returned == run_id

    # Exactly ONE header row for the run, COMPLETED, ray runtime, natives in bq_models, n_models
    # spanning BOTH runtimes (the shared-header invariant).
    headers = _rows(
        client,
        f"SELECT status, n_models, python_runtime, job_telemetry, "
        f"ARRAY_TO_STRING(bq_models, ',') AS bq_models "
        f"FROM `{d}.run_registry` WHERE run_id=@run_id",
        run_id,
    )
    assert len(headers) == 1, "one config → one shared header row"
    header = headers[0]
    assert header.status == "COMPLETED"
    assert header.n_models == len(_ALL_MODELS)
    assert header.python_runtime == "ray"
    assert set(header.bq_models.split(",")) == set(_NATIVE_MODELS)

    # job_telemetry records the CPU-only sizing that actually ran (no GPU pool).
    tel = json.loads(header.job_telemetry)
    assert tel["runtime"] == "ray"
    assert tel["gpu_node_count"] == 0
    assert tel["cpu_node_count"] >= 1

    # v_model_leaderboard: every model under the SAME run_id; compute_engine splits the Python-
    # runtime models (ray) from the natives (bigquery) — the single-run, two-engine comparison.
    board = _poll_rows(
        client,
        f"SELECT model_type, compute_engine, n_cells, mean_wape "
        f"FROM `{d}.v_model_leaderboard` WHERE run_id=@run_id",
        run_id,
        lambda rows: {r.model_type for r in rows} == set(_ALL_MODELS),
    )
    by_model = {r.model_type: r for r in board}
    assert set(by_model) == set(_ALL_MODELS), "both runtimes' models on one leaderboard"

    for m in _CPU_MODELS:
        assert by_model[m].compute_engine == "ray", m
    for m in _NATIVE_MODELS:
        assert by_model[m].compute_engine == "bigquery", m

    # Every model produced scored cells with a real OOF metric — directly comparable across engines.
    for m in _ALL_MODELS:
        assert by_model[m].n_cells == _SERIES_LIMIT, m
        assert by_model[m].mean_wape is not None, m

    # Teardown-in-finally: the ephemeral cluster is gone (no orphaned nodes billing forever).
    from google.cloud.aiplatform import vertex_ray

    live = {c.cluster_resource_name for c in vertex_ray.list_ray_clusters()}
    assert not any(name.endswith(plan.cluster_name) for name in live), (
        f"ephemeral cluster {plan.cluster_name} should have been deleted after the run"
    )
