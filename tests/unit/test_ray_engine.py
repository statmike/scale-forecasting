"""Tests for the Ray on-cluster driver (BUILD B4, ``scale_forecasting.engines.ray_engine``).

Two tiers, mirroring the pure/I-O seam:

* **Offline (no marker):** the pure driver helpers — the heterogeneous GPU/CPU task-options routing
  (:func:`_task_options`), the per-pool chunk count (:func:`_chunk_count`), the pool cell count
  (:func:`_pool_cells`). No Ray, no GPU, no BigQuery.
* **``@ray`` (needs the [ray] extra):** :func:`run` end-to-end on a *real* local Ray session
  (``local_mode`` was removed in Ray 2.x). A session-scoped 2-CPU cluster is started once; the
  engine reuses it (so it doesn't tear it down). The source read is monkeypatched to an in-memory
  panel and the chunk runner to a BigQuery-free stand-in (Ray tasks run in separate processes, so a
  driver-side ``bq`` patch wouldn't reach them), leaving the real
  read→route→chunk→fan→aggregate→header skeleton under test. The live T4 + fractional-GPU path is
  the ``@gpu`` smoke in ``test_ray_gpu_smoke.py``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.engines import ray_engine, ray_io
from scale_forecasting.engines.spark_io import _MODEL_COL, STATUS_COLUMNS
from scale_forecasting.settings import Settings

_CPU = "theta"
_GPU = "neuralprophet"


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "ray engine test",
        "python_runtime": "ray",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 4},
        "models": [_CPU, _GPU],
    }
    base.update(over)
    return RunConfig(**base)


def _compute(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"use_gpu": True, "bucket_target_cells": 2}
    base.update(over)
    return base


def _panel(n_series: int, rows_each: int = 6) -> pd.DataFrame:
    frames = []
    for i in range(n_series):
        frames.append(
            pd.DataFrame(
                {
                    "ts_id": [f"s{i}"] * rows_each,
                    "ds": pd.date_range("2024-01-01", periods=rows_each),
                    "y": range(rows_each),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- offline: pure driver helpers ----------------------------------------------


def test_task_options_gpu_pool_requests_fraction_when_gpu_on() -> None:
    cfg = _cfg(compute=_compute(use_gpu=True))
    assert ray_engine._task_options(cfg, 0.25, gpu=True) == {"num_gpus": 0.25}


def test_task_options_cpu_pool_always_requests_one_cpu() -> None:
    cfg = _cfg(compute=_compute(use_gpu=True))
    assert ray_engine._task_options(cfg, 0.25, gpu=False) == {"num_cpus": 1}


def test_task_options_gpu_pool_falls_back_to_cpu_when_gpu_off() -> None:
    # use_gpu=False: no device to schedule against, so a GPU-model chunk runs as a plain CPU task
    # (NeuralProphet falls back to CPU inside the cell).
    cfg = _cfg(compute=_compute(use_gpu=False))
    assert ray_engine._task_options(cfg, 0.25, gpu=True) == {"num_cpus": 1}


def test_chunk_count_ceils_and_floors() -> None:
    assert ray_engine._chunk_count(0, 8) == 0  # empty pool → no chunks
    assert ray_engine._chunk_count(1, 8) == 1  # a single cell still needs one chunk
    assert ray_engine._chunk_count(20, 8) == 3  # ceil(20 / 8)


def test_pool_cells_counts_series_times_models() -> None:
    src = _panel(4)
    cfg = _cfg()
    assert ray_engine._pool_cells(src, cfg, [_CPU, _GPU]) == 8  # 4 series × 2 models
    assert ray_engine._pool_cells(src, cfg, []) == 0  # empty pool
    assert ray_engine._pool_cells(pd.DataFrame(), cfg, [_CPU]) == 0  # empty panel


# --- offline: Storage Read API read helpers (C-Ray, Q3) ------------------------


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "project_id": "proj-x",
        "connection": "proj-x.us-central1.conn",
        "warehouse_uri": "gs://bkt/warehouse",
        "dataset_id": "ds_x",
    }
    base.update(over)
    return Settings(**base)


def test_storage_table_path_qualifies_bare_name_against_deployment_dataset() -> None:
    # a bare source_table resolves against the deployment project+dataset → storage resource path.
    path = ray_engine._storage_table_path(_cfg(), _settings())
    assert path == "projects/proj-x/datasets/ds_x/tables/source_series_native"


def test_storage_table_path_accepts_fully_qualified_source() -> None:
    cfg = _cfg(data={"source_table": "other_proj.other_ds.series", "horizon": 7})
    path = ray_engine._storage_table_path(cfg, _settings())
    assert path == "projects/other_proj/datasets/other_ds/tables/series"


def test_storage_table_path_prefixes_two_part_source_with_project() -> None:
    # dataset.table (another dataset in the same project) → the deployment project is prepended.
    cfg = _cfg(data={"source_table": "other_ds.series", "horizon": 7})
    path = ray_engine._storage_table_path(cfg, _settings())
    assert path == "projects/proj-x/datasets/other_ds/tables/series"


def test_limit_series_keeps_first_n_ordered_ids() -> None:
    # 5 series, limit 3 → the first three ts_ids by sort order, all their rows, others dropped.
    src = _panel(5)
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7, "series_limit": 3})
    out = ray_engine._limit_series(src, cfg)
    assert sorted(out["ts_id"].unique()) == ["s0", "s1", "s2"]
    assert len(out) == 3 * 6  # 3 series × 6 rows each, nothing else


def test_limit_series_passthrough_when_unset() -> None:
    src = _panel(4)
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7})  # no series_limit
    out = ray_engine._limit_series(src, cfg)
    assert len(out) == len(src)
    assert sorted(out["ts_id"].unique()) == ["s0", "s1", "s2", "s3"]


def test_limit_series_matches_spark_ordered_subset() -> None:
    # Parity with spark_io._limit_series: both keep the SAME first-N ordered distinct ids, so Ray
    # and Spark run identical series at every scale (DESIGN §13.1). Ids ordered lexically.
    src = pd.DataFrame(
        {
            "ts_id": ["s3", "s1", "s2", "s1", "s3", "s2"],
            "ds": pd.date_range("2024-01-01", periods=6),
            "y": range(6),
        }
    )
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7, "series_limit": 2})
    out = ray_engine._limit_series(src, cfg)
    assert sorted(out["ts_id"].unique()) == ["s1", "s2"]  # s1,s2 ordered-first, s3 dropped


# --- @ray: run end-to-end on a real local Ray session --------------------------


@pytest.fixture(scope="module")
def _local_ray() -> Any:
    """A real 2-CPU local Ray session, started once for the module (local_mode is gone in Ray 2.x).

    The engine sees an already-initialized session, so it reuses it and does *not* shut it down; the
    fixture owns teardown.
    """
    ray = pytest.importorskip("ray")
    ray.init(num_cpus=2, include_dashboard=False, configure_logging=False, logging_level="ERROR")
    yield ray
    ray.shutdown()


def _fake_runner(cfg: RunConfig, settings: Settings, models: list[str] | None = None) -> Any:
    """A BigQuery-free stand-in for :func:`ray_io.make_chunk_runner`.

    Ray tasks run in separate processes, so we can't monkeypatch ``bq.write_cells`` inside a worker;
    instead we swap the whole runner for one that emits a status row per ``(ts_id, model)`` cell
    without fitting a model or touching BigQuery. Returned closure is cloudpickle-able (Ray ships it
    to the worker). Every cell reports ``status="ok"`` so the run rolls up COMPLETED.
    """

    def _run(chunk: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (ts_id, model), _sub in chunk.groupby(["ts_id", _MODEL_COL], sort=False):
            rows.append((str(ts_id), str(model), "ok", 0.01))
        return pd.DataFrame(rows, columns=list(STATUS_COLUMNS))

    return _run


@pytest.fixture
def _stubbed_engine(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the driver's I/O seams: Settings, source read, chunk runner, and the header writes."""
    settings = Settings(
        project_id="test-proj",
        connection="test-proj.us-central1.conn",
        warehouse_uri="gs://test/warehouse",
    )
    monkeypatch.setattr(Settings, "resolve", classmethod(lambda cls: settings))
    monkeypatch.setattr(ray_engine, "_read_source_series", lambda cfg, settings: _panel(4))
    monkeypatch.setattr(ray_io, "make_chunk_runner", _fake_runner)

    calls: dict[str, Any] = {"ensure": 0, "write_header": 0, "update_header": None}
    from scale_forecasting.registry import bq

    monkeypatch.setattr(
        bq,
        "ensure_tables",
        lambda cfg, settings=None: calls.__setitem__("ensure", calls["ensure"] + 1),
    )
    monkeypatch.setattr(
        bq,
        "write_header",
        lambda cfg, run_id, settings=None: calls.__setitem__(
            "write_header", calls["write_header"] + 1
        ),
    )
    monkeypatch.setattr(
        bq,
        "update_header",
        lambda run_id, settings=None, **fields: calls.__setitem__("update_header", fields),
    )
    return calls


@pytest.mark.ray
def test_run_owner_mode_fans_all_cells_and_closes_header(
    _local_ray: Any, _stubbed_engine: dict[str, Any]
) -> None:
    # use_gpu=False so NeuralProphet routes to a CPU task on the GPU-less test box; the fake runner
    # emits a status row per cell without fitting. 4 series × 2 models = 8 cells → COMPLETED.
    cfg = _cfg(compute=_compute(use_gpu=False))
    ray_engine.run(cfg, manage_header=True)

    calls = _stubbed_engine
    assert calls["ensure"] == 1
    assert calls["write_header"] == 1
    assert calls["update_header"] is not None
    assert calls["update_header"]["status"] == "COMPLETED"
    assert calls["update_header"]["n_series"] == 4


@pytest.mark.ray
def test_run_contributor_mode_skips_header_lifecycle(
    _local_ray: Any, _stubbed_engine: dict[str, Any]
) -> None:
    # manage_header=False (Arc B): main.run owns the header, so the engine touches none of it.
    cfg = _cfg(compute=_compute(use_gpu=False))
    ray_engine.run(cfg, manage_header=False)

    calls = _stubbed_engine
    assert calls["ensure"] == 0
    assert calls["write_header"] == 0
    assert calls["update_header"] is None


@pytest.mark.ray
def test_run_honors_executed_subset(
    _local_ray: Any, _stubbed_engine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arc B hands only the Python-runtime subset; only those models become Ray cells.
    captured: dict[str, Any] = {}

    def _capturing_runner(
        cfg: RunConfig, settings: Settings, models: list[str] | None = None
    ) -> Any:
        captured["models"] = models

        def _run(chunk: pd.DataFrame) -> pd.DataFrame:
            rows = [
                (str(t), str(m), "ok", 0.01)
                for (t, m), _s in chunk.groupby(["ts_id", _MODEL_COL], sort=False)
            ]
            return pd.DataFrame(rows, columns=list(STATUS_COLUMNS))

        return _run

    monkeypatch.setattr(ray_io, "make_chunk_runner", _capturing_runner)
    cfg = _cfg(compute=_compute(use_gpu=False))
    ray_engine.run(cfg, models=[_CPU], manage_header=True)
    assert captured["models"] == [_CPU]
