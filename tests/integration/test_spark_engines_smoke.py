"""Local-Spark smoke for the explode engine (``@spark`` gate).

This runs :func:`scale_forecasting.engines.spark_explode.run` on a real *local* ``SparkSession`` —
no cluster, no BigQuery. The connector read is swapped for a local DataFrame and the executor-side
write is redirected to a temp dir, so what's actually exercised is the Spark machinery that the
offline ``run_group`` tests can't reach: the cross-join, ``groupBy(bucket).applyInPandas``, the
broadcast of a frozen ``Settings`` to the grouped UDF, and the compact status frame surviving real
pickling back to the driver. The live serverless path (``submit_batch`` → Dataproc) is the separate
``@gcp`` smoke.

PySpark runs the ``applyInPandas`` UDF in a *separate* Python worker process (even in ``local[*]``),
so a driver-side ``monkeypatch`` of ``write_cells`` never reaches it. Instead we swap the one seam
that gets cloudpickled out to the worker — :func:`spark_io.make_group_runner` — for a runner that
writes each bucket's results to a shared temp dir (a JSONL file per bucket). The driver's read of
those files after the job is how we observe the executor-side writes across the process boundary.
Everything else in :func:`spark_explode.run` — the real cross-join, bucketing, broadcast, and status
roll-up — runs unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

from scale_forecasting.config import RunConfig  # noqa: E402
from scale_forecasting.settings import Settings  # noqa: E402

pytestmark = pytest.mark.spark


def _panel(n_series: int, n_obs: int = 90) -> pd.DataFrame:
    """A tiny multi-series panel [ts_id, ds, y] with deterministic trend + weekly seasonality."""
    frames = []
    idx = pd.date_range("2023-01-01", periods=n_obs, freq="D")
    for s in range(n_series):
        trend = np.linspace(10.0, 30.0, n_obs) + s
        weekly = 3.0 * np.sin(np.arange(n_obs) * 2 * np.pi / 7)
        frames.append(pd.DataFrame({"ts_id": f"s_{s:03d}", "ds": idx, "y": trend + weekly}))
    return pd.concat(frames, ignore_index=True)


def _settings() -> Settings:
    return Settings(
        project_id="proj-x",
        connection="proj-x.us-central1.conn",
        warehouse_uri="gs://bkt/warehouse",
        dataset_id="ds_x",
        region="us-central1",
    )


def _capturing_runner(cfg: RunConfig, settings: Any, sink_dir: str) -> Any:
    """A :func:`make_group_runner` stand-in: run a bucket, write its results as JSONL to the sink.

    This is what gets cloudpickled to the Spark worker, so the write lands in the worker process.
    ``sink_dir`` is a filesystem path both driver and (local) workers can see; each bucket writes a
    uniquely-named file, so concurrent buckets never collide. Mirrors the real runner exactly except
    the leaf write goes to a file instead of ``bq.write_cells``. ``settings`` is the frozen
    :class:`Settings` captured directly (the Connect-safe seam — no ``sparkContext.broadcast``).
    """
    from scale_forecasting.engines.spark_io import run_group

    def _run(pdf: Any) -> Any:
        import json
        import os
        import uuid

        results, status = run_group(pdf, cfg)
        if results:
            rows = [
                {"ts_id": r.ts_id, "model_type": r.model_type, "status": r.status} for r in results
            ]
            path = os.path.join(sink_dir, f"bucket-{uuid.uuid4().hex}.jsonl")
            with open(path, "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
        return status

    return _run


def _run_engine_locally(
    engine_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drive the engine's ``run(cfg)`` on local Spark; return (written cell rows, closed header).

    The fan-out cross-joins to a cell per (series, model), which is exactly what the caller asserts
    on. The connector read, the executor-side write, and the header writes are swapped as described
    in the module docstring.
    """
    import json

    from pyspark.sql import SparkSession

    from scale_forecasting.engines import spark_io
    from scale_forecasting.registry import bq

    n_series, models = 4, ["theta", "holtwinters"]
    panel = _panel(n_series)
    sink = str(tmp_path)

    def _fake_read(spark: SparkSession, cfg: RunConfig, settings: Settings) -> Any:
        pdf = panel.copy()
        pdf["ds"] = pdf["ds"].dt.date  # source_series.ds is DATE
        return spark.createDataFrame(pdf)

    monkeypatch.setattr(spark_io, "read_source_series", _fake_read)
    monkeypatch.setattr(
        spark_io,
        "make_group_runner",
        lambda cfg, settings, models=None: _capturing_runner(cfg, settings, sink),
    )

    header: dict[str, Any] = {}
    monkeypatch.setattr(bq, "ensure_tables", lambda cfg, *, settings=None: None)
    monkeypatch.setattr(
        bq, "write_header", lambda cfg, run_id, *, settings=None: header.update(run_id=run_id)
    )
    monkeypatch.setattr(
        bq, "update_header", lambda run_id, *, settings=None, **fields: header.update(fields)
    )
    monkeypatch.setattr(Settings, "resolve", staticmethod(_settings))

    cfg = RunConfig(
        run_name="local spark explode",
        data={"source_table": "source_series", "horizon": 7, "series_limit": n_series},
        models=models,
    )
    engine_module.run(cfg)

    written = []
    for path in sorted(tmp_path.glob("bucket-*.jsonl")):
        for line in path.read_text().splitlines():
            written.append(json.loads(line))
    return written, header


def test_explode_run_on_local_spark(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from scale_forecasting.engines import spark_explode

    written, header = _run_engine_locally(spark_explode, monkeypatch, tmp_path)

    # One row per (series, model) cell — the explode cross-join, through applyInPandas and back
    # across the process boundary.
    n_series, models = 4, {"theta", "holtwinters"}
    assert len(written) == n_series * len(models)
    cells = {(r["ts_id"], r["model_type"]) for r in written}
    assert len(cells) == n_series * len(models)
    assert {m for _, m in cells} == models

    assert header["status"] in {"COMPLETED", "PARTIAL"}
    assert header["n_series"] == n_series
    assert header["runtime_seconds"] > 0


def test_injected_session_is_used_but_not_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """An injected (caller-owned) session runs the fan-out but is NOT stopped by the engine.

    This is the Spark Connect / notebook seam: ``spark_explode.run(cfg, spark=session)`` uses the
    passed session (``owns_session = spark is None`` is False) and leaves its lifecycle to the
    caller. A self-created session would be ``stop()``-ed in ``finally``; here the session must stay
    live after ``run`` returns, which we prove by running a trivial job on it afterward.
    """
    import json

    from pyspark.sql import SparkSession

    from scale_forecasting.engines import spark_explode, spark_io
    from scale_forecasting.registry import bq

    n_series, models = 3, ["theta", "holtwinters"]
    panel = _panel(n_series)
    sink = str(tmp_path)

    def _fake_read(spark: SparkSession, cfg: RunConfig, settings: Settings) -> Any:
        pdf = panel.copy()
        pdf["ds"] = pdf["ds"].dt.date
        return spark.createDataFrame(pdf)

    monkeypatch.setattr(spark_io, "read_source_series", _fake_read)
    monkeypatch.setattr(
        spark_io,
        "make_group_runner",
        lambda cfg, settings, models=None: _capturing_runner(cfg, settings, sink),
    )
    header: dict[str, Any] = {}
    monkeypatch.setattr(bq, "ensure_tables", lambda cfg, *, settings=None: None)
    monkeypatch.setattr(
        bq, "write_header", lambda cfg, run_id, *, settings=None: header.update(run_id=run_id)
    )
    monkeypatch.setattr(
        bq, "update_header", lambda run_id, *, settings=None, **fields: header.update(fields)
    )

    session = SparkSession.builder.master("local[2]").appName("injected-smoke").getOrCreate()
    try:
        cfg = RunConfig(
            run_name="injected spark explode",
            data={"source_table": "source_series", "horizon": 7, "series_limit": n_series},
            models=models,
        )
        # Injected session + resolved settings — the notebook/Connect call shape.
        spark_explode.run(cfg, settings=_settings(), spark=session)

        # The engine must NOT have stopped the caller's session: a trivial job still runs.
        assert session.range(5).count() == 5

        written = []
        for path in sorted(tmp_path.glob("bucket-*.jsonl")):
            for line in path.read_text().splitlines():
                written.append(json.loads(line))
        assert len(written) == n_series * len(models)
        assert header["status"] in {"COMPLETED", "PARTIAL"}
    finally:
        session.stop()
