"""Tests for the Ray on-cluster driver (``scale_forecasting.engines.ray_engine``).

Two tiers, mirroring the pure/I-O seam:

* **Offline (no marker):** the pure driver helpers — the heterogeneous GPU/CPU pool sizing and its
  task options (:func:`_pool_plans`), the per-pool chunk count (:func:`_chunk_count`,
  :func:`_pool_chunks`), the pool cell count (:func:`_pool_cells`). No Ray, no GPU, no BigQuery.
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


def _plans(cfg: RunConfig, panel: pd.DataFrame | None = None, gpu_fraction: float = 0.25):
    """Both pool plans for the standard CPU/GPU model split, over ``panel`` (default 4 series)."""
    return ray_engine._pool_plans(
        _panel(4) if panel is None else panel,
        cfg,
        "rid",
        [_CPU],
        [_GPU],
        None,
        gpu_fraction,
    )


def test_task_options_gpu_pool_requests_fraction_when_gpu_on() -> None:
    _cpu, gpu = _plans(_cfg(compute=_compute(use_gpu=True)))
    assert gpu.task_options == {"num_gpus": 0.25}


def test_task_options_cpu_pool_always_requests_one_cpu() -> None:
    cpu, _gpu = _plans(_cfg(compute=_compute(use_gpu=True)))
    assert cpu.task_options == {"num_cpus": 1}


def test_task_options_gpu_pool_falls_back_to_cpu_when_gpu_off() -> None:
    # use_gpu=False: no device to schedule against, so a GPU-model chunk runs as a plain CPU task
    # (NeuralProphet falls back to CPU inside the cell).
    _cpu, gpu = _plans(_cfg(compute=_compute(use_gpu=False)))
    assert gpu.task_options == {"num_cpus": 1}


def test_an_unprofiled_pool_asks_for_no_memory() -> None:
    """Ray treats ``memory`` as a hard scheduling resource; a number nobody took wedges tasks."""
    cpu, gpu = _plans(_cfg(compute=_compute(use_gpu=True)))
    assert "memory" not in cpu.task_options
    assert "memory" not in gpu.task_options


def test_the_pools_are_sized_from_the_panel_not_from_series_limit() -> None:
    """``series_limit`` is an upper bound; the sizing should describe the run that is happening."""
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7, "series_limit": 1000})
    cpu, _gpu = _plans(cfg, panel=_panel(3))
    assert cpu.n_cells == 3  # 3 series in the panel x 1 CPU model, not 1000


def test_chunk_count_ceils_and_floors() -> None:
    assert ray_engine._chunk_count(0, 8) == 0  # empty pool → no chunks
    assert ray_engine._chunk_count(1, 8) == 1  # a single cell still needs one chunk
    assert ray_engine._chunk_count(20, 8) == 3  # ceil(20 / 8)


def test_an_empty_pool_gets_no_chunks_however_wide_its_ceiling() -> None:
    """A pool with no work must not be handed tasks to make an autoscaler happy."""
    cfg = _cfg(models=[_CPU], compute=_compute(use_gpu=False))
    _cpu, gpu = ray_engine._pool_plans(_panel(4), cfg, "rid", [_CPU], [], None, 0.5)
    assert ray_engine._pool_chunks(gpu, 8) == 0


def test_the_chunk_count_is_floored_so_the_autoscaler_can_reach_its_ceiling() -> None:
    """Ray grows on *pending* demand: too few tasks and the pool sits at its minimum forever."""
    cfg = _cfg(
        models=[_CPU],
        data={"source_table": "source_series_native", "horizon": 7, "series_limit": 1000},
        compute=_compute(use_gpu=False),
    )
    cpu, _gpu = ray_engine._pool_plans(_panel(40), cfg, "rid", [_CPU], [], None, 0.5)
    # 40 cells at 2 per chunk is only 20 tasks, but the cluster was created able to hold far more.
    assert ray_engine._chunk_count(cpu.n_cells, 2) == 20
    assert ray_engine._pool_chunks(cpu, 2) == cpu.slots_at_ceiling > 20


def test_pool_cells_counts_series_times_models() -> None:
    src = _panel(4)
    cfg = _cfg()
    assert ray_engine._pool_cells(src, cfg, [_CPU, _GPU]) == 8  # 4 series × 2 models
    assert ray_engine._pool_cells(src, cfg, []) == 0  # empty pool
    assert ray_engine._pool_cells(pd.DataFrame(), cfg, [_CPU]) == 0  # empty panel


# --- offline: Storage Read API read helpers ------------------------------------


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


def test_storage_dataset_path_drops_scaffolding_to_dataset_dot_table() -> None:
    # ray.data.read_bigquery(dataset=...) wants D.T; the resource path is stripped back to it.
    ref = ray_engine._storage_dataset_path(_cfg(), _settings())
    assert ref == "ds_x.source_series_native"


def test_storage_dataset_path_uses_source_dataset_not_deployment() -> None:
    cfg = _cfg(data={"source_table": "other_proj.other_ds.series", "horizon": 7})
    assert ray_engine._storage_dataset_path(cfg, _settings()) == "other_ds.series"


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
    # and Spark run identical series at every scale. Ids ordered lexically.
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


# --- _read_source_series: Storage Read stream assembly (no GCP) -----------------


def _install_fake_read_client(
    monkeypatch: pytest.MonkeyPatch,
    per_stream_frames: list[pd.DataFrame],
    captured: dict[str, Any] | None = None,
) -> None:
    """Stub ``google.cloud.bigquery_storage_v1`` so ``_read_source_series`` runs offline.

    Each entry of ``per_stream_frames`` becomes one read stream returning that frame from
    ``read_rows(...).to_dataframe(...)`` — so a list of length > 1 drives the multi-stream
    ``pd.concat`` branch (the one the server picks at scale, and the one the @gpu smoke's
    single-stream reads never hit — where a missing ``pd`` import crashed a 100k run).
    """
    import sys
    import types as pytypes

    class _Stream:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Reader:
        def __init__(self, frame: pd.DataFrame) -> None:
            self._frame = frame

        def to_dataframe(self, _session: Any) -> pd.DataFrame:
            return self._frame

    class _Session:
        def __init__(self, n: int) -> None:
            self.streams = [_Stream(f"stream-{i}") for i in range(n)]

    class _FakeReadClient:
        def __init__(self) -> None:
            self._by_name = {f"stream-{i}": f for i, f in enumerate(per_stream_frames)}

        def create_read_session(
            self, *, parent: str, read_session: Any, max_stream_count: int
        ) -> Any:
            if captured is not None:
                captured["max_stream_count"] = max_stream_count
                captured["data_format"] = read_session.data_format
            return _Session(len(per_stream_frames))

        def read_rows(self, name: str) -> _Reader:
            return _Reader(self._by_name[name])

    # A minimal stand-in module for the lazily-imported storage client + its `types` namespace.
    fake_mod = pytypes.ModuleType("google.cloud.bigquery_storage_v1")
    fake_mod.BigQueryReadClient = _FakeReadClient  # type: ignore[attr-defined]
    fake_types = pytypes.SimpleNamespace(
        DataFormat=pytypes.SimpleNamespace(ARROW="ARROW"),
        ReadSession=lambda **kw: pytypes.SimpleNamespace(**kw),
    )
    fake_types.ReadSession.TableReadOptions = lambda **kw: pytypes.SimpleNamespace(**kw)  # type: ignore[attr-defined]
    fake_mod.types = fake_types  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery_storage_v1", fake_mod)

    # Keep the read fully offline: the snapshot lookup would otherwise construct a real BigQuery
    # client (needs ADC) to fetch the run header. These tests exercise the stream read, not snapshot
    # pinning, so pin it to None (an unpinned read) — the same value the best-effort lookup returns
    # for a run with no recorded snapshot.
    monkeypatch.setattr(ray_engine, "_snapshot_millis", lambda cfg, settings: None)


def test_read_source_series_concats_multiple_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    # The scale path: the Storage Read server fans the table into several streams, so the frames
    # list has length > 1 and _read_source_series must pd.concat them. Regression for a 100k Ray run
    # that died with `NameError: name 'pd' is not defined` on exactly this branch (pandas is
    # TYPE_CHECKING-only at module scope, so the function must import it at runtime).
    s0 = _panel(2)  # ts_ids s0, s1
    s1 = _panel(2).assign(ts_id=lambda d: d["ts_id"].str.replace("s", "t"))  # t0, t1
    _install_fake_read_client(monkeypatch, [s0, s1])
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7})
    out = ray_engine._read_source_series(cfg, _settings())
    assert len(out) == len(s0) + len(s1)  # both streams present
    assert sorted(out["ts_id"].unique()) == ["s0", "s1", "t0", "t1"]


def test_read_driver_collect_requests_arrow_and_honors_read_max_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The read session pins ARROW explicitly, and compute.read_max_streams flows through to the
    # server-side stream cap (0 = let the server choose, the default).
    captured: dict[str, Any] = {}
    _install_fake_read_client(monkeypatch, [_panel(1)], captured=captured)
    cfg = _cfg(
        compute=_compute(read_max_streams=4),
        data={"source_table": "source_series_native", "horizon": 7},
    )
    ray_engine._read_source_series(cfg, _settings())
    assert captured["max_stream_count"] == 4
    assert captured["data_format"] == "ARROW"


def test_read_driver_collect_defaults_to_server_chosen_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_read_client(monkeypatch, [_panel(1)], captured=captured)
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7})  # no read_max_streams
    ray_engine._read_source_series(cfg, _settings())
    assert captured["max_stream_count"] == 0  # server chooses


def test_read_source_series_single_stream_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    # The small-scale path the @gpu smoke exercised: one stream → frames[0], no concat. Kept so both
    # branches of the len(frames) check are covered offline.
    only = _panel(3)
    _install_fake_read_client(monkeypatch, [only])
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7})
    out = ray_engine._read_source_series(cfg, _settings())
    assert len(out) == len(only)
    assert sorted(out["ts_id"].unique()) == ["s0", "s1", "s2"]


# --- ray_read_mode dispatch: default reader vs ray.data reader ------------------


def test_read_source_series_defaults_to_driver_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default mode dispatches to the proven BigQueryReadClient path — the one the morning
    # greenfield Ray smoke runs on. ray.data must NOT be touched when the mode is left unset.
    sentinel = _panel(2)
    seen: dict[str, bool] = {"driver": False, "ray_data": False}

    def _fake_driver(cfg: RunConfig, settings: Settings) -> pd.DataFrame:
        seen["driver"] = True
        return sentinel

    def _fake_ray_data(cfg: RunConfig, settings: Settings) -> pd.DataFrame:
        seen["ray_data"] = True
        return sentinel

    monkeypatch.setattr(ray_engine, "_read_driver_collect", _fake_driver)
    monkeypatch.setattr(ray_engine, "_read_ray_data", _fake_ray_data)
    out = ray_engine._read_source_series(_cfg(), _settings())  # ray_read_mode defaults to driver
    assert seen == {"driver": True, "ray_data": False}
    assert len(out) == len(sentinel)


def test_read_source_series_ray_data_mode_dispatches_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ray_read_mode="ray_data" routes to the ray.data reader, and the shared series_limit subset is
    # still applied on top — so both readers produce the identical panel the fan-out expects.
    def _fake_ray_data(cfg: RunConfig, settings: Settings) -> pd.DataFrame:
        return _panel(5)  # 5 series; the cfg limit below must trim to the first 2

    monkeypatch.setattr(ray_engine, "_read_ray_data", _fake_ray_data)
    cfg = _cfg(
        compute=_compute(ray_read_mode="ray_data"),
        data={"source_table": "source_series_native", "horizon": 7, "series_limit": 2},
    )
    out = ray_engine._read_source_series(cfg, _settings())
    assert sorted(out["ts_id"].unique()) == ["s0", "s1"]  # limit applied after the ray.data read


def test_read_ray_data_reads_by_dataset_and_projects_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_read_ray_data`` calls ``ray.data.read_bigquery(project_id=, dataset=)`` and projects cols.

    Stubs ``ray.data`` so the reader runs offline: assert it passes the deployment project + the
    ``D.T`` dataset ref (never a ``query=``, so no query slots), and that the returned frame is
    column-projected to :func:`_needed_columns` — matching the default reader's shape.
    """
    import sys
    import types as pytypes

    captured: dict[str, Any] = {}

    class _FakeDataset:
        def to_pandas(self) -> pd.DataFrame:
            # An extra column the projection must drop, on top of the needed ts_id/ds/y.
            return _panel(2).assign(extra="drop me")

    def _read_bigquery(**kw: Any) -> _FakeDataset:
        captured.update(kw)
        return _FakeDataset()

    fake_ray = pytypes.ModuleType("ray")
    fake_ray.data = pytypes.SimpleNamespace(read_bigquery=_read_bigquery)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    # Offline: skip the snapshot lookup (it would build a real BigQuery client needing ADC).
    monkeypatch.setattr(ray_engine, "_snapshot_millis", lambda cfg, settings: None)

    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7})
    out = ray_engine._read_ray_data(cfg, _settings())

    assert captured == {"project_id": "proj-x", "dataset": "ds_x.source_series_native"}
    assert "query" not in captured  # a pure table scan, not a slot-consuming query
    assert "extra" not in out.columns  # projected down to the needed columns
    assert set(out.columns) == set(ray_engine._needed_columns(cfg))


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


def _fake_runner(
    cfg: RunConfig,
    settings: Settings,
    models: list[str] | None = None,
    params_by_model: dict[str, Any] | None = None,
) -> Any:
    """A BigQuery-free stand-in for :func:`ray_io.make_chunk_runner`.

    Ray tasks run in separate processes, so we can't monkeypatch ``cells.write_cells`` inside a
    worker; instead we swap the whole runner for one that emits a status row per ``(ts_id, model)``
    cell without fitting a model or touching BigQuery. Returned closure is cloudpickle-able (Ray
    ships it to the worker). Every cell reports ``status="ok"`` so the run rolls up COMPLETED. The
    ``params_by_model`` arg mirrors the real signature (fleetwide HPO) — ignored here.
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
    from scale_forecasting.registry import header, tables

    monkeypatch.setattr(
        tables,
        "ensure_tables",
        lambda cfg, settings=None: calls.__setitem__("ensure", calls["ensure"] + 1),
    )
    monkeypatch.setattr(
        header,
        "write_header",
        lambda cfg, run_id, settings=None: calls.__setitem__(
            "write_header", calls["write_header"] + 1
        ),
    )
    monkeypatch.setattr(
        header,
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
    # manage_header=False: main.run owns the header, so the engine touches none of it.
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
    # main.run hands only the Python-runtime subset; only those models become Ray cells.
    captured: dict[str, Any] = {}

    def _capturing_runner(
        cfg: RunConfig,
        settings: Settings,
        models: list[str] | None = None,
        params_by_model: dict[str, Any] | None = None,
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
