"""Offline tests for the direct/power path — reached entirely through the public front door.

The promise: a user who wants to drive Spark or Ray themselves imports the pure core
(:func:`run_group`), the writer-attached runners (:func:`make_group_runner` /
:func:`make_chunk_runner`), the cell-tagging helper (:func:`chunk_cells`), and the unit of work
(:func:`run_cell`) straight from ``scale_forecasting`` and gets the *same* model machinery the SDK
and CLI use. These tests exercise both embedding modes (untagged whole-series and cell-tagged)
and prove the runner writer-seam calls ``write_cells`` exactly once per group — all via
``from scale_forecasting import ...``, never a private module path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scale_forecasting import (
    CellResult,
    RunConfig,
    chunk_cells,
    make_chunk_runner,
    make_group_runner,
    run_cell,
    run_group,
)
from scale_forecasting.settings import Settings

_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)
_MODELS = ["theta", "holtwinters"]


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "direct use test",
        "data": {"source_table": "t", "freq": "D", "horizon": 7},
        "models": _MODELS,
    }
    base.update(over)
    return RunConfig(**base)


def _series(ts_id: str, n: int = 90) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(10.0, 30.0, n)
    weekly = 3.0 * np.sin(np.arange(n) * 2 * np.pi / 7)
    return pd.DataFrame({"ts_id": ts_id, "ds": idx, "y": trend + weekly})


def _source(*ts_ids: str) -> pd.DataFrame:
    return pd.concat([_series(t) for t in ts_ids], ignore_index=True)


# --- run_group: untagged whole-series embedding --------------------------------


def test_run_group_untagged_runs_every_model_per_series() -> None:
    # No _sf_model column → run_group loops all models per series: 2 series × 2 models = 4 cells.
    pdf = _source("series-a", "series-b")
    results, status = run_group(pdf, _cfg(), models=_MODELS)
    assert len(results) == 4
    assert all(isinstance(r, CellResult) for r in results)
    assert {(r.ts_id, r.model_type) for r in results} == {
        ("series-a", "theta"), ("series-a", "holtwinters"),
        ("series-b", "theta"), ("series-b", "holtwinters"),
    }
    assert list(status.columns) == ["ts_id", "model_type", "status", "fit_seconds"]
    assert len(status) == 4


# --- run_group: cell-tagged embedding via chunk_cells --------------------------


def test_run_group_tagged_via_chunk_cells() -> None:
    cfg = _cfg()
    chunks = chunk_cells(_source("series-a", "series-b"), cfg, _MODELS, n_chunks=1)
    assert len(chunks) == 1  # one chunk holds all 4 cells
    results, _status = run_group(chunks[0], cfg)  # models come from the tag column, not the arg
    assert len(results) == 4
    assert {(r.ts_id, r.model_type) for r in results} == {
        ("series-a", "theta"), ("series-a", "holtwinters"),
        ("series-b", "theta"), ("series-b", "holtwinters"),
    }


# --- run_cell: the unit of work never raises on a bad model --------------------


def test_run_cell_maps_bad_model_to_error_status() -> None:
    res = run_cell(_series("series-z"), "nope", _cfg(models=["theta"]))
    assert isinstance(res, CellResult)
    assert res.status == "error"
    assert res.error is not None


# --- runner writer-seam: write_cells called exactly once per group -------------


def test_group_runner_writes_once(monkeypatch: Any) -> None:
    from scale_forecasting.registry import bq

    calls: list[int] = []

    def _fake_write(results: list[CellResult], *, settings: Settings) -> None:
        calls.append(len(results))

    monkeypatch.setattr(bq, "write_cells", _fake_write)

    runner = make_group_runner(_cfg(), _SETTINGS, models=_MODELS)
    status = runner(_source("series-a", "series-b"))
    assert calls == [4]  # one write, four cells
    assert len(status) == 4


def test_chunk_runner_writes_once(monkeypatch: Any) -> None:
    from scale_forecasting.registry import bq

    calls: list[int] = []

    def _fake_write(results: list[CellResult], *, settings: Settings) -> None:
        calls.append(len(results))

    monkeypatch.setattr(bq, "write_cells", _fake_write)

    cfg = _cfg()
    chunk = chunk_cells(_source("series-a", "series-b"), cfg, _MODELS, n_chunks=1)[0]
    runner = make_chunk_runner(cfg, _SETTINGS)
    status = runner(chunk)
    assert calls == [4]
    assert len(status) == 4
