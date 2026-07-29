"""Live fixed-size T4 Vertex Ray smoke (BUILD B4, ``@gpu``).

Runs a mixed ``python_runtime="ray"`` config end-to-end through :func:`scale_forecasting.main.run`
against live GCP: a **fixed-size T4 Vertex Ray cluster** (sized deterministically from the run's
fan-out, no autoscaling — DESIGN §11.1/D17) runs the Python-runtime models in parallel with the
in-BigQuery native engine, all under one shared ``run_id`` / one ``run_registry`` header. The point
Spark can't reach: NeuralProphet cells pack onto a **fractional** T4 while stats/ML cells run on CPU
(heterogeneous routing), and the calibrated GPU fraction + fixed cluster plan are stamped to the
run.

Asserts:

* exactly **one** ``run_registry`` row, ``COMPLETED``, ``python_runtime='ray'``, natives in
  ``bq_models``, ``n_models`` == the full model count;
* ``v_model_leaderboard`` shows every model under the same ``run_id`` — the Python-runtime models
  with ``compute_engine='ray'`` (NeuralProphet among them, non-NULL ``mean_wape`` — it scored a real
  OOF fold on a T4), the natives with ``compute_engine='bigquery'``;
* the header's ``job_telemetry`` records the fixed sizing that ran — ``runtime='ray'``, a calibrated
  ``sizing_gpu_fraction`` in (0, 1], ``accelerator_type='NVIDIA_TESLA_T4'``, and node counts;
* the ephemeral cluster is **gone afterward** (``vertex_ray.list_ray_clusters`` no longer lists
  it) — the teardown-in-``finally`` guarantee, so no orphaned T4 bills.

The **"resize for scale"** half of the B4 story — a larger ``series_limit`` yields a strictly larger
fixed :func:`~scale_forecasting.engines.ray_io.plan_cluster` (more nodes, no autoscaling) — is
proven offline+free by the ``plan_cluster`` sizing tests in ``tests/unit/test_ray_io.py``; this file
is the one authorized *live* run.

Skipped unless **both** ``SF_ENABLE_GPU`` and ``SF_PROJECT_ID`` are set (see ``tests/conftest.py``):
this provisions a real T4 cluster (cost + ~15-25 min) and needs T4 quota (``NVIDIA_T4_GPUS`` in the
region) + ADC. Beyond the ``SF_*`` identity the writers resolve, set the Ray infra vars
(``SF_RAY_NETWORK`` / ``SF_COMPUTE_SA`` / ``SF_CODE_BUCKET``; optionally ``SF_CONTAINER_IMAGE`` /
``SF_RAY_VERSION``), then run::

    SF_ENABLE_GPU=1 uv run pytest -m gpu tests/integration/test_ray_gpu_smoke.py

**Self-contained data.** Like the Arc B mixed smoke, the shipped seed's ``price_index`` is all-NULL,
so ARIMA_PLUS_XREG can't train on it; this test seeds its own tiny exog-carrying scratch
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

pytestmark = pytest.mark.gpu

_GPU_MODEL = "neuralprophet"
_CPU_MODEL = "theta"
_NATIVE_MODELS = ["arima_plus", "arima_plus_xreg", "timesfm"]
_PYTHON_MODELS = [_GPU_MODEL, _CPU_MODEL]
_ALL_MODELS = [*_PYTHON_MODELS, *_NATIVE_MODELS]
_SERIES_LIMIT = 6
_HORIZON = 28
_HISTORY = 730  # ~2 years daily → training window > 1 year, so ARIMA_PLUS holidays apply
_SCRATCH_TABLE = "b4_ray_gpu_smoke_source"


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    return Settings.resolve()


@pytest.fixture(scope="module")
def scratch_source(settings: Settings) -> Iterator[str]:
    """Seed an exog-carrying scratch ``source_series`` table, yield its name, drop it after.

    The same generator the production seed uses (``with_exog=True``), loaded as a plain BQ table
    both the Ray BigQuery read and the native BigQuery engine consume. Truncate-on-write so a rerun
    starts clean.
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
    # A per-invocation run_name → a unique deterministic run_id (append-only cell tables can't be
    # DELETE-d while buffered, so a fresh id keeps reruns clean).
    return RunConfig(
        run_name=f"b4 ray gpu smoke {int(time.time())}",
        python_runtime="ray",
        data={"source_table": source_table, "horizon": _HORIZON, "series_limit": _SERIES_LIMIT},
        models=_ALL_MODELS,
        features={"holidays": ["US"], "exog": ["price_index"]},
        # Backtest ON so the Ray path emits an OOF metric panel: without it the Python engine only
        # forecasts (no metrics) while the BQ natives always score a held-out fold — so NP
        # would land NULL mean_wape and the two runtimes wouldn't be comparable, which is the whole
        # point of the single-run leaderboard. A small fold count keeps the T4 run cheap.
        backtest={"enabled": True, "n_folds": 2, "horizon": _HORIZON, "step": _HORIZON},
        compute={"use_gpu": True, "gpu_type": "T4", "gpu_fraction": "auto", "accelerator_count": 1},
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


def test_ray_gpu_fixed_cluster_smoke(settings: Settings, scratch_source: str) -> None:
    from google.cloud import bigquery

    from scale_forecasting import main
    from scale_forecasting.engines import ray_io

    client = bigquery.Client(project=settings.project_id)
    cfg = _cfg(scratch_source)
    run_id = make_run_id(cfg)
    d = settings.dataset_ref

    # Capture the plan we expect the run to have provisioned (same pure sizing the submitter uses),
    # so we can name the ephemeral cluster and confirm it's gone at the end.
    plan = ray_io.plan_cluster(cfg, models=_PYTHON_MODELS, run_id=run_id)

    # The whole point: one call, both runtimes, one run_id. Returns the shared run_id it wrote.
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

    # job_telemetry records the fixed T4 sizing that actually ran (the sizing decision is audited).
    tel = json.loads(header.job_telemetry)
    assert tel["runtime"] == "ray"
    assert tel["accelerator_type"] == "NVIDIA_TESLA_T4"
    assert 0.0 < tel["sizing_gpu_fraction"] <= 1.0  # a calibrated fraction, packed onto the T4
    assert tel["gpu_node_count"] >= 1
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

    # The Python-runtime models ran on Ray; the natives in BigQuery — under one run_id.
    for m in _PYTHON_MODELS:
        assert by_model[m].compute_engine == "ray", m
    for m in _NATIVE_MODELS:
        assert by_model[m].compute_engine == "bigquery", m

    # Every model produced scored cells — critically NeuralProphet, which fit on a fractional T4 and
    # scored a real OOF fold (non-NULL metric), so it's directly comparable to the natives.
    for m in _ALL_MODELS:
        assert by_model[m].n_cells == _SERIES_LIMIT, m
        assert by_model[m].mean_wape is not None, m

    # Teardown-in-finally: the ephemeral cluster is gone (no orphaned T4s billing forever).
    from google.cloud.aiplatform import vertex_ray

    live = {c.cluster_resource_name for c in vertex_ray.list_ray_clusters()}
    assert not any(name.endswith(plan.cluster_name) for name in live), (
        f"ephemeral cluster {plan.cluster_name} should have been deleted after the run"
    )
