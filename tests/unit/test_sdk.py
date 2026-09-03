"""Offline tests for the public SDK front door (``scale_forecasting`` package + ``sdk.py``).

No GCP, no compute: the :class:`Forecaster` facade only wraps :func:`scale_forecasting.main.run`, so
here we prove the wrapper's contract — construction parity, the offline ``dry_run``/``review``
pointers, and that ``run`` delegates to a spied ``main.run`` — plus the load-bearing **import-cost
contract**: ``import scale_forecasting`` must not pull the heavy model modules; only touching a
heavy public name (``run_cell``/``run``/``Forecaster``…) may. That last one runs in a subprocess
because the rest of the suite has already imported ``worker`` into this process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import scale_forecasting as sf
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.registry.views import VIEW_NAMES
from scale_forecasting.settings import Settings

_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _cfg_dict(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_name": "sdk test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": ["theta", "arima_plus"],
    }
    base.update(over)
    return base


# --- construction --------------------------------------------------------------


def test_from_dict_and_from_file_build_equal_config(tmp_path: Path) -> None:
    import json

    data = _cfg_dict()
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data))

    from_dict = sf.Forecaster.from_dict(data)
    from_file = sf.Forecaster.from_file(path)
    assert isinstance(from_dict.config, RunConfig)
    assert from_dict.config == from_file.config


def test_from_dict_rejects_bad_config() -> None:
    with pytest.raises(ConfigError):
        sf.Forecaster.from_dict({"not_a_field": True})


def test_from_file_rejects_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json ")
    with pytest.raises(ConfigError):
        sf.Forecaster.from_file(bad)


def test_run_id_is_deterministic_config_hash() -> None:
    f = sf.Forecaster.from_dict(_cfg_dict())
    assert f.run_id == make_run_id(f.config)


# --- dry_run: offline plan -----------------------------------------------------


def test_dry_run_reports_id_fanout_and_runtime_split() -> None:
    f = sf.Forecaster.from_dict(_cfg_dict())
    dr = f.dry_run()
    assert dr.run_id == f.run_id  # single source of truth (delegates to main.run dry_run)
    assert dr.python_models == ["theta"]
    assert dr.bq_models == ["arima_plus"]
    assert dr.fanout.n_series == 5
    assert dr.fanout.n_models == 2
    assert dr.fanout.n_cells == 5 * 2  # no folds → cells = series × models


# --- run: delegates to main.run, wraps in a RunResult --------------------------


def test_run_delegates_to_main_run_and_wraps_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting import main

    calls: dict[str, Any] = {}

    def _spy(
        cfg: RunConfig,
        *,
        dry_run: bool = False,
        spark: object | None = None,
        settings: Settings | None = None,
        n_series: int | None = None,
        max_executors: int | None = None,
    ) -> str:
        calls["cfg"] = cfg
        calls["spark"] = spark
        calls["settings"] = settings
        calls["n_series"] = n_series
        calls["max_executors"] = max_executors
        return make_run_id(cfg)

    monkeypatch.setattr(main, "run", _spy)

    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    result = f.run(spark="fake-session")

    assert calls["cfg"] == f.config
    assert calls["spark"] == "fake-session"
    assert calls["settings"] is _SETTINGS  # injected identity is threaded through
    assert result.run_id == f.run_id
    assert result.dataset_ref == _SETTINGS.dataset_ref
    assert result.views == VIEW_NAMES


def test_run_threads_scale_knobs_to_main_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting import main

    calls: dict[str, Any] = {}

    def _spy(
        cfg: RunConfig,
        *,
        dry_run: bool = False,
        spark: object | None = None,
        settings: Settings | None = None,
        n_series: int | None = None,
        max_executors: int | None = None,
    ) -> str:
        calls["n_series"] = n_series
        calls["max_executors"] = max_executors
        return make_run_id(cfg)

    monkeypatch.setattr(main, "run", _spy)
    sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS).run(n_series=1000, max_executors=8)
    assert calls["n_series"] == 1000
    assert calls["max_executors"] == 8


# --- review: offline pointer ---------------------------------------------------


def test_review_is_offline_pointer_with_injected_settings() -> None:
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    rr = f.review()
    assert rr.run_id == f.run_id
    assert rr.dataset_ref == _SETTINGS.dataset_ref
    assert rr.views == VIEW_NAMES


def test_review_is_graceful_when_identity_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SF_PROJECT_ID", "SF_CONNECTION", "SF_WAREHOUSE_URI"):
        monkeypatch.delenv(var, raising=False)
    f = sf.Forecaster.from_dict(_cfg_dict())  # no injected settings
    rr = f.review()
    assert rr.run_id == f.run_id
    assert rr.dataset_ref is None  # unresolved infra → None, not an exception
    assert rr.views == VIEW_NAMES


# --- status / wait / results: the closed lifecycle -----------------------------


def test_status_reads_header_for_this_configs_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import header

    seen: dict[str, Any] = {}

    def _fake_status(run_id: str, *, settings: Any = None) -> str:
        seen["run_id"] = run_id
        seen["settings"] = settings
        return "COMPLETED"

    monkeypatch.setattr(header, "header_status", _fake_status)
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    assert f.status() == "COMPLETED"
    assert seen["run_id"] == f.run_id  # defaults to this config's id
    assert seen["settings"] is _SETTINGS


def test_status_is_none_when_never_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: None)
    assert sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS).status() is None


def test_wait_returns_terminal_status_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: "COMPLETED")
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    assert f.wait(timeout=1.0, poll_seconds=0.0) == "COMPLETED"


def test_wait_polls_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.sdk as sdk_mod
    from scale_forecasting.registry import header

    statuses = iter(["RUNNING", "RUNNING", "PARTIAL"])
    monkeypatch.setattr(header, "header_status", lambda *a, **k: next(statuses))
    slept: list[float] = []
    monkeypatch.setattr(sdk_mod.time, "sleep", lambda s: slept.append(s))

    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    assert f.wait(timeout=100.0, poll_seconds=5.0) == "PARTIAL"
    assert slept == [5.0, 5.0]  # slept once per non-terminal poll


def test_wait_raises_when_run_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: None)
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    with pytest.raises(ConfigError, match="no run found"):
        f.wait(timeout=1.0, poll_seconds=0.0)


def test_wait_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.sdk as sdk_mod
    from scale_forecasting.registry import header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: "RUNNING")
    monkeypatch.setattr(sdk_mod.time, "sleep", lambda s: None)
    # monotonic jumps past the deadline on the second read so the loop exits deterministically.
    clock = iter([0.0, 100.0, 200.0, 300.0])
    monkeypatch.setattr(sdk_mod.time, "monotonic", lambda: next(clock))

    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    with pytest.raises(TimeoutError):
        f.wait(timeout=10.0, poll_seconds=1.0)


def test_results_maps_leaderboard_rows_to_model_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import reads

    seen: dict[str, Any] = {}

    def _fake_leaderboard(run_id: str, *, settings: Any = None) -> list[dict[str, Any]]:
        seen["run_id"] = run_id
        return [
            {
                "model_type": "theta",
                "ensemble_id": None,
                "compute_engine": "spark",
                "n_cells": 5,
                "no_artifact_rate": 0.0,
                "median_fit_seconds": 0.2,
                "mean_wape": 0.11,
                "mean_mae": 3.4,
            },
            {
                "model_type": "arima_plus",
                "ensemble_id": None,
                "compute_engine": "bigquery",
                "n_cells": 5,
                "no_artifact_rate": 1.0,
                "median_fit_seconds": None,
                "mean_wape": None,
                "mean_mae": None,
            },
        ]

    monkeypatch.setattr(reads, "read_leaderboard", _fake_leaderboard)
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    results = f.results()

    assert seen["run_id"] == f.run_id
    assert [r.model_type for r in results] == ["theta", "arima_plus"]
    assert results[0].mean_wape == 0.11
    assert results[1].mean_wape is None and results[1].no_artifact_rate == 1.0
    assert isinstance(results[0], sf.ModelResult)


def test_results_empty_when_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import reads

    monkeypatch.setattr(reads, "read_leaderboard", lambda *a, **k: [])
    assert sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS).results() == []


# --- dag: the offline planned per-job DAG --------------------------------------


def test_dag_returns_planned_nodes_offline() -> None:
    from scale_forecasting.registry.ids import make_job_key

    f = sf.Forecaster.from_dict(_cfg_dict(models=["theta", "arima_plus"]), settings=_SETTINGS)
    nodes = f.dag()
    assert [n.family for n in nodes] == ["statistical", "native"]
    for n in nodes:
        assert n.job_key == make_job_key(f.run_id, n.family, 1)
    assert all(isinstance(n, sf.DagNode) for n in nodes)


# --- jobs: the per-job cross-system trace --------------------------------------


def test_jobs_maps_run_jobs_rows_to_job_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import jobs

    seen: dict[str, Any] = {}

    def _fake_run_jobs(run_id: str, *, settings: Any = None) -> list[dict[str, Any]]:
        seen["run_id"] = run_id
        seen["settings"] = settings
        return [
            {
                "family": "statistical",
                "job_id": "sf-abc-statistical-a1",
                "system_job_id": "sf-abc-statistical-a1",
                "runtime": "spark",
                "spark_mode": "serverless",
                "hardware": "cpu",
                "gpu_type": None,
                "status": "COMPLETED",
                "attempt": 1,
                "runtime_seconds": 42.0,
            },
            {
                "family": "native",
                "job_id": "sf-abc-native-a1",
                "system_job_id": "sf-abc-native-a1",
                "runtime": "bigquery",
                "spark_mode": None,
                "hardware": None,
                "gpu_type": None,
                "status": "RUNNING",
                "attempt": 1,
                "runtime_seconds": None,
            },
        ]

    monkeypatch.setattr(jobs, "read_run_jobs", _fake_run_jobs)
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    traces = f.jobs()

    assert seen["run_id"] == f.run_id
    assert seen["settings"] is _SETTINGS
    assert [t.family for t in traces] == ["statistical", "native"]
    assert traces[0].job_key == "sf-abc-statistical-a1"  # job_id is the canonical key
    assert traces[0].system_job_id == "sf-abc-statistical-a1"
    assert traces[0].runtime == "spark" and traces[0].status == "COMPLETED"
    assert traces[1].runtime == "bigquery" and traces[1].runtime_seconds is None
    assert isinstance(traces[0], sf.JobTrace)


def test_jobs_empty_when_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import jobs

    monkeypatch.setattr(jobs, "read_run_jobs", lambda *a, **k: [])
    assert sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS).jobs() == []


# --- probe: the registry-vs-runtime reconciled drill-down ----------------------


def test_probe_delegates_to_probe_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.probes.reconcile as probes_mod

    seen: dict[str, Any] = {}

    def _fake_probe_run(run_id: str, *, job: str | None = None, settings: Any = None) -> str:
        seen["run_id"] = run_id
        seen["job"] = job
        seen["settings"] = settings
        return "report"

    monkeypatch.setattr(probes_mod, "probe_run", _fake_probe_run)
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)

    assert f.probe(job="statistical") == "report"
    assert seen["run_id"] == f.run_id  # defaults to this config's id
    assert seen["job"] == "statistical"
    assert seen["settings"] is _SETTINGS  # injected identity is threaded through


def test_probe_honors_explicit_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.probes.reconcile as probes_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        probes_mod, "probe_run", lambda run_id, **kw: seen.update(run_id=run_id, **kw)
    )
    sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS).probe(run_id="sf-other")
    assert seen["run_id"] == "sf-other"
    assert seen["job"] is None  # default: whole run, no family narrowing


# --- settle: the probe verdict, written back -----------------------------------


def test_settle_previews_by_default_and_threads_the_injected_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``settle()`` with no arguments must reach `settle_run` with ``yes=False``.

    Settle is the one probe verb that writes, so the default matters more than the delegation: a
    ``yes`` that defaulted true would repair rows on a call an operator made to *look*.
    """
    import scale_forecasting.probes.settle as settle_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        settle_mod, "settle_run", lambda run_id, **kw: seen.update(run_id=run_id, **kw) or "report"
    )
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)

    assert f.settle() == "report"
    assert seen["run_id"] == f.run_id  # defaults to this config's id
    assert seen["yes"] is False and seen["job"] is None and seen["reason"] == ""
    assert seen["settings"] is _SETTINGS


def test_settle_passes_the_confirmation_and_the_audit_reason_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scale_forecasting.probes.settle as settle_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        settle_mod, "settle_run", lambda run_id, **kw: seen.update(run_id=run_id, **kw)
    )
    sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS).settle(
        run_id="sf-other", job="deep_learning", yes=True, reason="driver died mid-write"
    )
    assert seen["run_id"] == "sf-other" and seen["job"] == "deep_learning"
    assert seen["yes"] is True and seen["reason"] == "driver died mid-write"


# --- trace: the per-job + per-cell execution timeline --------------------------


def _dt(second: int) -> Any:
    from datetime import UTC, datetime

    return datetime(2026, 1, 1, 0, 0, second, tzinfo=UTC)


def _job_row(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "family": "statistical",
        "job_id": "sf-abc-statistical-a1",
        "runtime": "spark",
        "status": "COMPLETED",
        "started_at": _dt(0),
        "ended_at": _dt(40),
        "runtime_seconds": 40.0,
    }
    base.update(over)
    return base


def _cell_row(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ts_id": "series-a",
        "model_type": "theta",
        "compute_engine": "spark",
        "worker_id": "host-1:123",
        "cell_started_at": _dt(2),
        "cell_ended_at": _dt(5),
    }
    base.update(over)
    return base


def test_build_trace_frame_stacks_jobs_and_cells() -> None:
    frame = sf.build_trace_frame([_job_row()], [_cell_row()])
    assert list(frame.columns) == [
        "kind",
        "lane",
        "label",
        "start",
        "end",
        "duration_s",
        "status",
        "runtime",
        "model_type",
        "ts_id",
    ]
    job = frame[frame["kind"] == "job"].iloc[0]
    assert job["lane"] == "statistical" and job["label"] == "statistical"
    assert job["duration_s"] == 40.0 and job["runtime"] == "spark"
    assert job["model_type"] is None and job["ts_id"] is None
    cell = frame[frame["kind"] == "cell"].iloc[0]
    assert cell["lane"] == "host-1:123" and cell["label"] == "theta:series-a"
    assert cell["duration_s"] == 3.0 and cell["runtime"] == "spark"
    assert cell["model_type"] == "theta" and cell["ts_id"] == "series-a"
    assert cell["status"] is None


def test_build_trace_frame_drops_rows_without_start_and_zero_widths_open_spans() -> None:
    # A job with no started_at can't be placed → dropped; a started-but-not-ended job → zero-width.
    rows = [
        _job_row(family="never-started", started_at=None, ended_at=None),
        _job_row(family="still-running", ended_at=None, runtime_seconds=None),
    ]
    frame = sf.build_trace_frame(rows, [])
    assert list(frame["lane"]) == ["still-running"]  # the un-started job is gone
    open_span = frame.iloc[0]
    assert open_span["start"] == open_span["end"]  # end defaults to start (zero-width marker)
    assert open_span["duration_s"] is None  # no end + no runtime_seconds → not derivable


def test_build_trace_frame_empty_inputs_keep_columns() -> None:
    frame = sf.build_trace_frame([], [])
    assert frame.empty
    assert list(frame.columns)[:2] == ["kind", "lane"]


def test_trace_reads_both_sources_and_builds_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import jobs, reads

    seen: dict[str, Any] = {}

    def _fake_jobs(run_id: str, *, settings: Any = None) -> list[dict[str, Any]]:
        seen["jobs_run_id"] = run_id
        seen["jobs_settings"] = settings
        return [_job_row()]

    def _fake_cells(
        run_id: str, *, limit: int = 5000, settings: Any = None
    ) -> list[dict[str, Any]]:
        seen["cells_run_id"] = run_id
        seen["limit"] = limit
        return [_cell_row(), _cell_row(ts_id="series-b")]

    monkeypatch.setattr(jobs, "read_run_jobs", _fake_jobs)
    monkeypatch.setattr(reads, "read_cell_timing", _fake_cells)
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    frame = f.trace(cell_limit=100)

    assert seen["jobs_run_id"] == f.run_id and seen["cells_run_id"] == f.run_id
    assert seen["jobs_settings"] is _SETTINGS
    assert seen["limit"] == 100  # cell_limit threads through to the reader
    assert (frame["kind"] == "job").sum() == 1
    assert (frame["kind"] == "cell").sum() == 2


def test_plot_trace_renders_a_lane_per_track() -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed for the smoke check
    frame = sf.build_trace_frame([_job_row()], [_cell_row(), _cell_row(worker_id="host-2:9")])
    ax = sf.plot_trace(frame, title="my run")
    # one y-tick per (kind, lane): 1 job track + 2 distinct worker tracks
    assert len(ax.get_yticklabels()) == 3
    assert ax.get_title() == "my run"


def test_plot_trace_handles_empty_frame() -> None:
    import matplotlib

    matplotlib.use("Agg")
    ax = sf.plot_trace(sf.build_trace_frame([], []))
    assert "no timed rows" in ax.get_title()


# --- public surface + import-cost contract -------------------------------------


def test_every_public_name_resolves() -> None:
    for name in sf.__all__:
        assert getattr(sf, name) is not None, name
    assert dir(sf) == sorted(sf.__all__)


def test_import_does_not_pull_heavy_modules_until_needed() -> None:
    # PEP 562 lazy contract, checked in a clean subprocess (this process already loaded worker).
    script = (
        "import sys, scale_forecasting as sf\n"
        "assert 'scale_forecasting.worker' not in sys.modules, 'import pulled worker'\n"
        "_ = sf.RunConfig, sf.Settings, sf.ConfigError, sf.Fanout\n"
        "assert 'scale_forecasting.worker' not in sys.modules, 'light names pulled worker'\n"
        "_ = sf.run_cell\n"
        "assert 'scale_forecasting.worker' in sys.modules, 'run_cell did not load worker'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
