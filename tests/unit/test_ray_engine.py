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


def _scheduling_request(plan: Any) -> dict[str, Any]:
    """``task_options`` minus the thread pin — the part the scheduler reads.

    Every plan also carries a ``runtime_env`` capping the native thread pools at the cores Ray
    assigns the task. That pin asks the scheduler for nothing, so it has no bearing on which pool
    gets what; it is owned and asserted by ``test_resources.py``. Dropping it here keeps these
    exact-dict comparisons about routing rather than about the pin's current contents.
    """
    return {key: value for key, value in plan.task_options.items() if key != "runtime_env"}


def test_task_options_gpu_pool_requests_fraction_when_gpu_on() -> None:
    _cpu, gpu = _plans(_cfg(compute=_compute(use_gpu=True)))
    assert _scheduling_request(gpu) == {"num_gpus": 0.25}


def test_task_options_cpu_pool_always_requests_one_cpu() -> None:
    cpu, _gpu = _plans(_cfg(compute=_compute(use_gpu=True)))
    assert _scheduling_request(cpu) == {"num_cpus": 1}


def test_task_options_gpu_pool_falls_back_to_cpu_when_gpu_off() -> None:
    # use_gpu=False: no device to schedule against, so a GPU-model chunk runs as a plain CPU task
    # (NeuralProphet falls back to CPU inside the cell).
    _cpu, gpu = _plans(_cfg(compute=_compute(use_gpu=False)))
    assert _scheduling_request(gpu) == {"num_cpus": 1}


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


def test_the_calibration_sample_is_the_same_series_whatever_order_the_reader_returned() -> None:
    """These few series decide the whole run's GPU density, so they cannot depend on stream order.

    ``_sample_series`` feeds `calibrate_gpu_fraction`, whose answer sets how many cells share a
    device for the entire run. Taking whichever series arrived first made that a function of Storage
    Read API stream ordering: two runs of the same config on the same table could size differently
    and neither would be wrong in a way anything could catch.
    """
    panel = _panel(6)
    shuffled = panel.sample(frac=1.0, random_state=0).reset_index(drop=True)
    cfg = _cfg(compute=_compute(gpu_calibration_samples=3))

    def ids(frames: list[pd.DataFrame]) -> list[str]:
        return [frame["ts_id"].iloc[0] for frame in frames]

    assert ids(ray_engine._sample_series(panel, cfg)) == ["s0", "s1", "s2"]
    assert ids(ray_engine._sample_series(shuffled, cfg)) == ["s0", "s1", "s2"]


def test_the_calibration_sample_is_the_same_subset_the_rest_of_the_codebase_takes() -> None:
    """One "first k ts_ids" rule everywhere — `_limit_series` is the one this must agree with."""
    panel = _panel(6)
    cfg = _cfg(
        data={"source_table": "source_series_native", "horizon": 7, "series_limit": 2},
        compute=_compute(gpu_calibration_samples=2),
    )
    sample = pd.concat(ray_engine._sample_series(panel, cfg), ignore_index=True)
    limited = ray_engine._limit_series(panel, cfg)
    assert sorted(sample["ts_id"].unique()) == sorted(limited["ts_id"].unique())


def test_the_executed_pool_plans_are_filed_per_family_beside_the_planned_ones() -> None:
    """The submitter's `sizing.<family>` is what was bought; this is what the driver ran on."""
    cpu, gpu = _plans(_cfg(compute=_compute(use_gpu=True)))
    patch = ray_engine._executed_sizing_patch(cpu, gpu, [_CPU], [_GPU])
    assert set(patch) == {"sizing_executed.statistical", "sizing_executed.deep_learning"}
    assert patch["sizing_executed.statistical"] == {"cpu_pool": cpu.to_dict()}
    assert patch["sizing_executed.deep_learning"] == {"gpu_pool": gpu.to_dict()}


def test_a_pool_with_no_models_files_nothing_rather_than_an_empty_record() -> None:
    """No GPU pool is a fact about the run; an entry saying "0 cells" reads as a broken one."""
    cfg = _cfg(models=[_CPU], compute=_compute(use_gpu=False))
    cpu, gpu = ray_engine._pool_plans(_panel(4), cfg, "rid", [_CPU], [], None, 0.5)
    assert ray_engine._executed_sizing_patch(cpu, gpu, [_CPU], []) == {
        "sizing_executed.statistical": {"cpu_pool": cpu.to_dict()}
    }


def test_two_families_sharing_the_cpu_pool_file_under_one_joined_label() -> None:
    """The CPU pool holds everything that is not deep learning, so its label has to say so."""
    cfg = _cfg(models=[_CPU, "xgboost", _GPU])
    cpu, gpu = ray_engine._pool_plans(_panel(4), cfg, "rid", [_CPU, "xgboost"], [_GPU], None, 0.5)
    patch = ray_engine._executed_sizing_patch(cpu, gpu, [_CPU, "xgboost"], [_GPU])
    assert set(patch) == {"sizing_executed.statistical_ml", "sizing_executed.deep_learning"}


def test_pool_cells_counts_series_times_models() -> None:
    src = _panel(4)
    cfg = _cfg()
    assert ray_engine._pool_cells(src, cfg, [_CPU, _GPU]) == 8  # 4 series × 2 models
    assert ray_engine._pool_cells(src, cfg, []) == 0  # empty pool
    assert ray_engine._pool_cells(pd.DataFrame(), cfg, [_CPU]) == 0  # empty panel


# --- offline: one chunk failing costs its cells, not the run -------------------


class _FakeRay:
    """The two `ray` calls `_collect_chunks` makes, over pre-decided per-future outcomes.

    A future here is just a key into ``outcomes``: either a status frame to hand back or an
    exception to raise. That is the whole surface `_collect_chunks` touches, which is the point of
    passing ``ray_mod`` in — the salvage logic is scheduling bookkeeping and a real cluster would
    only make it slower to get wrong.
    """

    def __init__(self, outcomes: dict[str, Any]) -> None:
        self.outcomes = outcomes

    def wait(self, refs: list[str], num_returns: int = 1) -> tuple[list[str], list[str]]:
        return refs[:num_returns], refs[num_returns:]

    def get(self, ref: str) -> pd.DataFrame:
        outcome = self.outcomes[ref]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ok_status(chunk: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """What a chunk task returns on the happy path: one ``ok`` row per cell it ran."""
    cells = [(ts_id, model) for ts_id in chunk["ts_id"].drop_duplicates() for model in models]
    return pd.DataFrame(
        {
            "ts_id": [c[0] for c in cells],
            "model_type": [c[1] for c in cells],
            "status": ["ok"] * len(cells),
            "fit_seconds": [0.1] * len(cells),
        },
        columns=list(STATUS_COLUMNS),
    )


def test_one_failing_chunk_still_yields_the_other_chunks_and_closes_partial() -> None:
    """Three chunks, one raises: the other two land, the lost cells are error, run is PARTIAL.

    `ray.get(futures)` on the list would have propagated the first exception and thrown away the
    two frames already in hand — while those chunks' forecasts sit written in BigQuery. Salvaging
    them is what turns a FAILED run that disowns durable work into a PARTIAL one that reports it.
    """
    cfg = _cfg()
    chunks = [_panel(1).assign(ts_id=f"s{i}") for i in range(3)]
    fake = _FakeRay(
        {
            "a": _ok_status(chunks[0], [_CPU]),
            "b": RuntimeError("worker blew up mid-fit"),
            "c": _ok_status(chunks[2], [_CPU]),
        }
    )
    pending = {
        "a": (chunks[0], [_CPU]),
        "b": (chunks[1], [_CPU]),
        "c": (chunks[2], [_CPU]),
    }

    status_pdf, failures = ray_engine._collect_chunks(fake, pending, cfg)

    assert len(status_pdf) == 3  # every cell accounted for, including the lost one
    assert dict(status_pdf.groupby("status").size()) == {"ok": 2, "error": 1}
    outcome = ray_io.aggregate_status(status_pdf)
    assert outcome.status == "PARTIAL"
    assert (outcome.n_ok, outcome.n_error) == (2, 1)
    assert len(failures) == 1
    assert "worker blew up mid-fit" in failures[0]["error"]
    assert failures[0] == {
        "error": failures[0]["error"],
        "n_series": 1,
        "models": [_CPU],
    }


def test_every_chunk_failing_closes_failed_rather_than_looking_empty() -> None:
    cfg = _cfg()
    chunk = _panel(2)
    fake = _FakeRay({"a": RuntimeError("boom"), "b": RuntimeError("boom")})
    pending = {"a": (chunk, [_CPU]), "b": (chunk, [_GPU])}

    status_pdf, failures = ray_engine._collect_chunks(fake, pending, cfg)

    assert len(failures) == 2
    assert set(status_pdf["status"]) == {"error"}
    assert ray_io.aggregate_status(status_pdf).status == "FAILED"


def test_a_failed_chunk_reports_the_cells_it_lost_not_the_rows_it_held() -> None:
    """The error rows are cells — series × the pool's models — not one row per source row."""
    cfg = _cfg()
    chunk = _panel(3, rows_each=10)  # 30 rows, 3 series, 2 pool models → 6 cells
    lost = ray_engine._failed_chunk_status(chunk, cfg, [_CPU, _GPU], RuntimeError("boom"))
    assert len(lost) == 6
    assert list(lost.columns) == list(STATUS_COLUMNS)
    assert set(lost["model_type"]) == {_CPU, _GPU}
    assert set(lost["status"]) == {"error"}


def test_a_failed_tagged_chunk_takes_its_cells_from_the_tag_not_the_pool_list() -> None:
    """Below one series per chunk `chunk_cells` falls back to tagged frames — honor the tag.

    A tagged chunk carries one model per row, so expanding it against the pool's full model list
    would invent cells that chunk was never going to run and over-count the loss.
    """
    cfg = _cfg()
    tagged = _panel(2).assign(**{_MODEL_COL: _GPU})
    lost = ray_engine._failed_chunk_status(tagged, cfg, [_CPU, _GPU], RuntimeError("boom"))
    assert len(lost) == 2  # 2 series × the one tagged model
    assert set(lost["model_type"]) == {_GPU}


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


# --- offline: series_limit as a read-side row_restriction ----------------------


def test_the_bound_names_the_nth_id_so_one_comparison_selects_the_whole_subset() -> None:
    # 5 ids, limit 3 → "<= the third", which is one comparison however many series it selects.
    ids = ["s4", "s0", "s3", "s1", "s2"]
    assert ray_engine.build_series_bound(ids, 3, "ts_id") == "ts_id <= 's2'"


def test_no_limit_means_no_filter_rather_than_a_filter_that_matches_everything() -> None:
    assert ray_engine.build_series_bound(["s0", "s1"], None, "ts_id") is None


def test_a_limit_the_table_cannot_reach_reads_unfiltered() -> None:
    # Asking for more series than exist: a bound at the last id excludes nothing, so it is pure
    # cost — the service still has to evaluate it against every row.
    assert ray_engine.build_series_bound(["s0", "s1"], 2, "ts_id") is None
    assert ray_engine.build_series_bound(["s0", "s1"], 99, "ts_id") is None


def test_an_empty_source_reads_unfiltered_rather_than_indexing_off_the_end() -> None:
    assert ray_engine.build_series_bound([], 3, "ts_id") is None


def test_the_id_column_is_the_configured_one_not_a_hardcoded_ts_id() -> None:
    assert ray_engine.build_series_bound(["a", "b", "c"], 1, "series_key") == "series_key <= 'a'"


def test_a_quote_in_a_series_id_is_escaped_instead_of_ending_the_literal() -> None:
    # Without escaping this is `ts_id <= 'o'brien'` — a syntax error, and the read session is
    # rejected. Series ids come from the deployer's own keyspace; apostrophes are ordinary.
    assert ray_engine.build_series_bound(["o'brien", "z"], 1, "ts_id") == "ts_id <= 'o\\'brien'"


def test_a_backslash_is_escaped_before_the_quotes_are() -> None:
    # Order matters: escaping quotes first would then double the backslash they introduced.
    assert ray_engine.build_series_bound(["a\\b", "z"], 1, "ts_id") == "ts_id <= 'a\\\\b'"


def test_the_pushed_bound_selects_exactly_what_the_client_side_subset_would_have() -> None:
    # The two rules must agree or the pushdown silently changes which series a run covers. Apply the
    # bound the way BigQuery would (a string <= comparison) and compare frames.
    src = pd.DataFrame(
        {
            "ts_id": ["s10", "s2", "s1", "s10", "s2", "s1"],
            "ds": pd.date_range("2024-01-01", periods=6),
            "y": range(6),
        }
    )
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7, "series_limit": 2})
    bound = ray_engine.build_series_bound(src["ts_id"], 2, "ts_id")
    assert bound == "ts_id <= 's10'"  # lexical, not numeric: s1, s10 come before s2
    boundary = bound.split(" <= ")[1].strip("'")
    pushed = src[src["ts_id"] <= boundary].reset_index(drop=True)
    pd.testing.assert_frame_equal(pushed, ray_engine._limit_series(src, cfg))


def test_a_subsetting_run_resolves_the_boundary_first_then_reads_only_that_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two sessions: a narrow single-column pass to find the Nth id, then the panel read carrying the
    # bound. The point of the whole item is that the second session is filtered server-side.
    captured: dict[str, Any] = {}
    _install_fake_read_client(monkeypatch, [_panel(4)], captured=captured)
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7, "series_limit": 2})
    out = ray_engine._read_source_series(cfg, _settings())
    boundary, panel = captured["sessions"]
    assert boundary["fields"] == ["ts_id"]  # id column only — the cheap pass
    assert boundary["row_restriction"] is None
    assert panel["row_restriction"] == "ts_id <= 's1'"
    assert "y" in panel["fields"]  # the real read, column-projected as before
    # The fake server ignores the restriction, so `_limit_series` still has work to do here — which
    # is exactly the idempotence the client-side subset is kept for.
    assert sorted(out["ts_id"].unique()) == ["s0", "s1"]


def test_a_run_with_no_series_limit_opens_one_session_and_pushes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_read_client(monkeypatch, [_panel(3)], captured=captured)
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7})
    ray_engine._read_source_series(cfg, _settings())
    assert len(captured["sessions"]) == 1  # no boundary pass to pay for
    assert captured["sessions"][0]["row_restriction"] is None


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
                # One list entry per session opened, in order: a subsetting run opens the narrow
                # boundary session first and the panel session second, so the sequence is the
                # evidence that the bound was resolved and then pushed down.
                captured.setdefault("sessions", []).append(
                    {
                        "fields": list(read_session.read_options.selected_fields),
                        "row_restriction": getattr(
                            read_session.read_options, "row_restriction", None
                        ),
                    }
                )
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
