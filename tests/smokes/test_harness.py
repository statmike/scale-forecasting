"""Offline unit tests for the smoke harness's pure verify/trace helpers (`smoke_harness`).

The live driver (`smoke_harness.run_smoke`) submits real jobs and is exercised by the runbook, not
here. These tests pin the *checking* logic — the part that decides PASS/FAIL from registry rows — so
a reviewer trusts a green smoke means what it says. All inputs are plain dicts + a `RunConfig`; no
GCP, no imports of the live path.
"""

from __future__ import annotations

from typing import Any

import smoke_harness as h

from scale_forecasting.config import RunConfig


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "smoke_test",
        "data": {"source_table": "source_series_iceberg", "horizon": 7, "series_limit": 100},
        "models": ["theta", "arima_plus"],
    }
    base.update(over)
    return RunConfig(**base)


def _job(family: str, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": family,
        "runtime": "spark",
        "spark_mode": None,
        "hardware": "cpu",
        "gpu_type": None,
        "status": "COMPLETED",
        "system_job_id": f"sf-run-{family}-a1",
    }
    row.update(over)
    return row


# --- verify_run_jobs -----------------------------------------------------------


def test_verify_run_jobs_all_green_is_clean() -> None:
    cfg = _cfg()  # theta (statistical, spark) + arima_plus (native)
    rows = [_job("statistical"), _job("native", runtime="bigquery")]
    assert h.verify_run_jobs(rows, cfg) == []


def test_verify_run_jobs_flags_missing_family() -> None:
    cfg = _cfg()
    rows = [_job("statistical")]  # native never ran
    problems = h.verify_run_jobs(rows, cfg)
    assert any("native" in p and "did not run" in p for p in problems)


def test_verify_run_jobs_flags_non_completed_status() -> None:
    cfg = _cfg()
    rows = [_job("statistical", status="FAILED"), _job("native", runtime="bigquery")]
    problems = h.verify_run_jobs(rows, cfg)
    assert any("statistical" in p and "FAILED" in p for p in problems)


def test_verify_run_jobs_flags_missing_system_job_id() -> None:
    # A blank system_job_id means reverse-trace is broken — the point of the real-id stamp-back.
    cfg = _cfg()
    rows = [_job("statistical", system_job_id=""), _job("native", runtime="bigquery")]
    problems = h.verify_run_jobs(rows, cfg)
    assert any("statistical" in p and "system_job_id" in p for p in problems)


# --- verify_leaderboard --------------------------------------------------------


def test_verify_leaderboard_all_models_present_is_clean() -> None:
    cfg = _cfg()
    board = [{"model_type": "theta", "n_cells": 100}, {"model_type": "arima_plus", "n_cells": 100}]
    assert h.verify_leaderboard(board, cfg) == []


def test_verify_leaderboard_flags_missing_model() -> None:
    cfg = _cfg()
    board = [{"model_type": "theta", "n_cells": 100}]  # arima_plus missing
    problems = h.verify_leaderboard(board, cfg)
    assert any("arima_plus" in p for p in problems)


def test_verify_leaderboard_requires_ensemble_rows_when_enabled() -> None:
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    board = [{"model_type": "theta", "n_cells": 100}, {"model_type": "arima_plus", "n_cells": 100}]
    problems = h.verify_leaderboard(board, cfg)
    assert any("ensemble" in p for p in problems)


def test_verify_leaderboard_accepts_ensemble_rows_when_enabled() -> None:
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    board = [
        {"model_type": "theta", "n_cells": 100},
        {"model_type": "arima_plus", "n_cells": 100},
        {"model_type": "ensemble_mean", "n_cells": 100},
    ]
    assert h.verify_leaderboard(board, cfg) == []


# --- verify_rerun --------------------------------------------------------------


def test_verify_rerun_identical_board_is_clean() -> None:
    board = [{"model_type": "theta", "n_cells": 100}, {"model_type": "arima_plus", "n_cells": 100}]
    assert h.verify_rerun(board, [dict(r) for r in board]) == []


def test_verify_rerun_flags_changed_counts() -> None:
    before = [{"model_type": "theta", "n_cells": 100}]
    after = [{"model_type": "theta", "n_cells": 200}]  # dedupe-on-read should have kept this at 100
    problems = h.verify_rerun(before, after)
    assert any("n_cells" in p and "theta" in p for p in problems)


def test_verify_rerun_flags_changed_model_set() -> None:
    before = [{"model_type": "theta", "n_cells": 100}]
    after = [{"model_type": "theta", "n_cells": 100}, {"model_type": "sarimax", "n_cells": 100}]
    problems = h.verify_rerun(before, after)
    assert any("model set" in p for p in problems)


# --- format_trace --------------------------------------------------------------


def test_format_trace_maps_each_runtime_to_its_service() -> None:
    rows = [
        _job("statistical", runtime="spark", spark_mode=None),
        _job("ml", runtime="spark", spark_mode="cluster"),
        _job("deep_learning", runtime="ray", hardware="gpu", gpu_type="T4"),
        _job("native", runtime="bigquery"),
    ]
    text = "\n".join(h.format_trace(rows))
    assert "Dataproc Serverless batch" in text
    assert "Dataproc cluster job" in text
    assert "Vertex Ray submission" in text
    assert "BigQuery job" in text
    # Trace is family-sorted for stable output.
    assert text.index("deep_learning") < text.index("ml") < text.index("native")


def test_format_trace_shows_system_job_id() -> None:
    rows = [_job("statistical", system_job_id="sf-run-abc-statistical-a1")]
    (line,) = h.format_trace(rows)
    assert "sf-run-abc-statistical-a1" in line
