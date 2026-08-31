"""Offline tests for the pure Spark-engine helpers (``engines.spark_io``).

Everything here runs without Spark or BigQuery — it exercises the grouped-UDF body
(:func:`run_group`), the run-level status roll-up (:func:`aggregate_status`), and the bucketing
policy that is the crux of the per-cell scaling story. The Spark shell (connector read,
cross-join, applyInPandas) is covered by the ``@spark``/``@gcp`` gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scale_forecasting.config import RunConfig
from scale_forecasting.engines import spark_explode, spark_io
from scale_forecasting.engines.spark_io import (
    STATUS_COLUMNS,
    aggregate_status,
    bucket_key_cols,
    default_bucket_count,
    run_group,
)

HORIZON = 7


def _series(ts_id: str, n: int = 90) -> pd.DataFrame:
    """One ts_id's rows: deterministic trend + weekly seasonality, columns [ts_id, ds, y]."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(10.0, 30.0, n)
    weekly = 3.0 * np.sin(np.arange(n) * 2 * np.pi / 7)
    return pd.DataFrame({"ts_id": ts_id, "ds": idx, "y": trend + weekly})


def _panel(ids: list[str]) -> pd.DataFrame:
    return pd.concat([_series(i) for i in ids], ignore_index=True)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "spark io test",
        "data": {"source_table": "t", "freq": "D", "horizon": HORIZON},
        "models": ["theta"],
    }
    base.update(over)
    return RunConfig(**base)


# --- bucket_key_cols: the per-cell crux ----------------------------------------


def test_buckets_on_cell() -> None:
    cfg = _cfg(models=["theta", "holtwinters"])
    # The engine isolates each (series, model) cell so a slow cell can't block fast ones.
    assert bucket_key_cols(cfg) == ["ts_id", spark_io._MODEL_COL]


def test_bucket_key_honors_custom_ts_id_col() -> None:
    cfg = _cfg(data={"source_table": "t", "ts_id_col": "series_key"})
    assert bucket_key_cols(cfg) == ["series_key", spark_io._MODEL_COL]


# --- default_bucket_count ------------------------------------------------------


def test_bucket_count_targets_cells_per_bucket() -> None:
    cfg = _cfg(
        models=["theta", "holtwinters", "sarimax"],
        data={"source_table": "t", "series_limit": 100},
        compute={"bucket_target_cells": 8},
    )
    # 100 series × 3 models = 300 cells → ceil(300 / 8) = 38 buckets (~8 cells each).
    assert default_bucket_count(cfg) == 38


def test_bucket_count_scales_with_work_not_max_parallelism() -> None:
    # The 100k OOM regression guard: buckets are decoupled from max_parallelism (a concurrency
    # knob), so a huge run makes many small buckets instead of few giant frames. 100k × 4 = 400k
    # cells → ceil(400k / 8) = 50k buckets, regardless of a small max_parallelism.
    cfg = _cfg(
        models=["theta", "holtwinters", "sarimax", "xgboost"],
        data={"source_table": "t", "series_limit": 100_000},
        compute={"max_parallelism": 50, "bucket_target_cells": 8},
    )
    assert default_bucket_count(cfg) == 50_000


def test_bucket_count_respects_max_buckets_ceiling() -> None:
    # Pathological config: tiny target on a huge run would shatter into too many partitions; the
    # _MAX_BUCKETS safety ceiling caps it.
    cfg = _cfg(
        models=["theta", "holtwinters", "sarimax", "xgboost"],
        data={"source_table": "t", "series_limit": 1_000_000},
        compute={"bucket_target_cells": 1},
    )
    assert default_bucket_count(cfg) == spark_io._MAX_BUCKETS


def test_bucket_count_defaults_to_cap_when_unlimited() -> None:
    cfg = _cfg(compute={"max_parallelism": 123})
    # series_limit unset → cell count unknown offline → fall back to the parallelism cap.
    assert default_bucket_count(cfg) == 123


# --- reachable_bucket_count: the buckets-≥-ceiling invariant --------------------


def test_a_fan_out_narrower_than_the_ceiling_is_widened_to_reach_it() -> None:
    # 100 executors × (8 cores / 1 cpu per task) = 800 tasks before the autoscaler has any reason
    # to grow to 100. At 40 buckets the fleet would sit near its minimum for the whole run.
    assert (
        spark_io.reachable_bucket_count(40, max_executors=100, executor_cores=8, task_cpus=1) == 800
    )


def test_a_fan_out_already_wide_enough_is_left_exactly_alone() -> None:
    assert (
        spark_io.reachable_bucket_count(5000, max_executors=100, executor_cores=8, task_cpus=1)
        == 5000
    )


def test_wide_tasks_lower_the_bar_because_fewer_fit_per_executor() -> None:
    # spark.task.cpus=4 on an 8-core executor → 2 tasks each, so 100 executors need only 200.
    assert (
        spark_io.reachable_bucket_count(40, max_executors=100, executor_cores=8, task_cpus=4) == 200
    )


def test_an_unset_ceiling_leaves_the_policy_count_untouched() -> None:
    # No maxExecutors property → the platform default applies and we did not choose it; sizing
    # fan-out against a guess at someone else's number is worse than not raising at all.
    assert spark_io.reachable_bucket_count(40, max_executors=None, executor_cores=8) == 40
    assert spark_io.reachable_bucket_count(40, max_executors=100, executor_cores=None) == 40


def test_the_widened_count_still_respects_the_safety_ceiling() -> None:
    raised = spark_io.reachable_bucket_count(
        40, max_executors=2000, executor_cores=96, task_cpus=1
    )
    assert raised == spark_io._MAX_BUCKETS


def test_a_zero_core_executor_cannot_drive_the_count_to_zero() -> None:
    assert spark_io.reachable_bucket_count(40, max_executors=10, executor_cores=0) == 40


# --- _conf_int: reading the live conf without trusting it ----------------------


class _ConfStub:
    def __init__(self, values: dict[str, Any], *, raises: bool = False) -> None:
        self._values = values
        self._raises = raises

    def get(self, key: str, default: Any = None) -> Any:
        if self._raises:
            raise RuntimeError("unknown conf key")
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value


class _SessionStub:
    def __init__(self, values: dict[str, Any], *, raises: bool = False) -> None:
        self.conf = _ConfStub(values, raises=raises)


def test_a_set_property_is_read_back_as_an_int() -> None:
    spark = _SessionStub({"spark.executor.cores": "16"})
    assert spark_explode._conf_int(spark, "spark.executor.cores") == 16


def test_an_unset_property_reads_as_none_not_zero() -> None:
    assert spark_explode._conf_int(_SessionStub({}), "spark.executor.cores") is None


def test_a_non_numeric_property_reads_as_none_rather_than_raising() -> None:
    spark = _SessionStub({"spark.executor.cores": "8g"})
    assert spark_explode._conf_int(spark, "spark.executor.cores") is None


def test_a_session_that_rejects_the_key_outright_reads_as_none() -> None:
    # A Spark Connect session may refuse an unknown key instead of returning the default; the
    # bucket count must not die because a conf lookup did.
    spark = _SessionStub({}, raises=True)
    assert spark_explode._conf_int(spark, "spark.dynamicAllocation.maxExecutors") is None


# --- fanout_properties + _widen_fanout: making a bucket actually be a task ------


def test_the_shuffle_width_is_pinned_to_the_bucket_count() -> None:
    # Without this, groupBy(...).applyInPandas plans spark.sql.shuffle.partitions tasks — 200 by
    # default — no matter how many buckets the cells were hashed into.
    props = spark_io.fanout_properties(4000)
    assert props["spark.sql.shuffle.partitions"] == "4000"


def test_aqe_is_stopped_from_coalescing_the_pin_away() -> None:
    # Pinning the partition count is not enough on its own: AQE coalesces small partitions after
    # the fact, which is how a 200-task stage becomes a 1-task stage.
    props = spark_io.fanout_properties(4000)
    assert props["spark.sql.adaptive.coalescePartitions.minPartitionNum"] == "4000"


def _fanout_cfg(**compute: Any) -> RunConfig:
    return _cfg(data={"source_table": "t", "series_limit": 100}, compute=compute)


def test_widen_fanout_raises_the_count_and_pins_the_shuffle_to_the_raised_one() -> None:
    spark = _SessionStub(
        {"spark.dynamicAllocation.maxExecutors": "50", "spark.executor.cores": "8"}
    )
    assert spark_explode._widen_fanout(_fanout_cfg(), spark, 40) == 400
    # The pin must follow the *raised* count, not the policy count it started from.
    assert spark.conf.get("spark.sql.shuffle.partitions") == "400"


def test_widen_fanout_still_pins_the_shuffle_when_the_count_is_already_wide_enough() -> None:
    # The raise and the pin are independent: a fan-out that needs no widening still needs its
    # tasks to exist.
    spark = _SessionStub(
        {"spark.dynamicAllocation.maxExecutors": "2", "spark.executor.cores": "4"}
    )
    assert spark_explode._widen_fanout(_fanout_cfg(), spark, 500) == 500
    assert spark.conf.get("spark.sql.shuffle.partitions") == "500"


def test_widen_fanout_touches_nothing_when_profiling_is_off() -> None:
    # The escape hatch has to survive Serverless writing its own dynamicAllocation defaults into
    # the driver conf — which is why the gate is here and not at the call site.
    spark = _SessionStub(
        {"spark.dynamicAllocation.maxExecutors": "1000", "spark.executor.cores": "4"}
    )
    cfg = _fanout_cfg(profile={"mode": "off"})
    assert spark_explode._widen_fanout(cfg, spark, 500) == 500
    assert spark.conf.get("spark.sql.shuffle.partitions") is None


# --- run_group: tagged frame (cross-joined, model column present) ---------------


def _with_model_col(panel: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Emulate the cross-join: replicate each series row once per model, tagged _sf_model."""
    parts = [panel.assign(**{spark_io._MODEL_COL: m}) for m in models]
    return pd.concat(parts, ignore_index=True)


def test_run_group_tagged_one_result_per_cell() -> None:
    cfg = _cfg(models=["theta", "holtwinters"])
    pdf = _with_model_col(_panel(["s0", "s1"]), ["theta", "holtwinters"])
    results, status = run_group(pdf, cfg)

    # 2 series × 2 models = 4 cells.
    assert len(results) == 4
    assert {(r.ts_id, r.model_type) for r in results} == {
        ("s0", "theta"),
        ("s0", "holtwinters"),
        ("s1", "theta"),
        ("s1", "holtwinters"),
    }
    assert all(r.status == "ok" for r in results)
    # Helper columns never reach run_cell (would break feature building) — all cells succeeded.
    assert list(status.columns) == list(STATUS_COLUMNS)
    assert len(status) == 4


def test_run_group_untagged_loops_models_per_series() -> None:
    cfg = _cfg(models=["theta", "holtwinters"])
    pdf = _panel(["s0", "s1"])  # no model column — an untagged frame groups by ts_id only
    results, status = run_group(pdf, cfg)

    assert len(results) == 4  # 2 series × 2 models, run per series in a loop
    assert {(r.ts_id, r.model_type) for r in results} == {
        ("s0", "theta"),
        ("s0", "holtwinters"),
        ("s1", "theta"),
        ("s1", "holtwinters"),
    }
    assert all(r.status == "ok" for r in results)


def test_run_group_error_cell_becomes_status_row() -> None:
    cfg = _cfg(models=["nonexistent_model"])
    pdf = _with_model_col(_panel(["s0"]), ["nonexistent_model"])
    results, status = run_group(pdf, cfg)

    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].error is not None
    # The batch survives: an error is a status row, not an exception.
    assert status.iloc[0]["status"] == "error"


def test_run_group_status_frame_has_fit_seconds() -> None:
    cfg = _cfg(models=["theta"])
    pdf = _with_model_col(_panel(["s0"]), ["theta"])
    _results, status = run_group(pdf, cfg)
    assert status["fit_seconds"].dtype == np.float64
    assert (status["fit_seconds"] >= 0).all()


def test_run_group_threads_fleetwide_params_to_each_cell() -> None:
    # params_by_model (the driver's fleetwide resolution) reaches every cell of that model, so
    # best_params reflects the tuned params — the seam that carries HPO across the fan-out.
    cfg = _cfg(models=["xgboost", "theta"])
    pdf = _with_model_col(_panel(["s0", "s1"]), ["xgboost", "theta"])
    params = {"xgboost": {"n_estimators": 111, "max_depth": 3, "learning_rate": 0.09}}
    results, _status = run_group(pdf, cfg, params_by_model=params)

    xgb = [r for r in results if r.model_type == "xgboost"]
    theta = [r for r in results if r.model_type == "theta"]
    assert xgb and all(r.best_params == params["xgboost"] for r in xgb)  # tuned params applied
    assert theta and all(r.best_params == {} for r in theta)  # absent from map → {} default


def test_run_group_without_params_is_unchanged_default() -> None:
    # No params_by_model → today's behavior: every cell runs with {} (additive-by-default).
    cfg = _cfg(models=["xgboost"])
    pdf = _with_model_col(_panel(["s0"]), ["xgboost"])
    results, _status = run_group(pdf, cfg)
    assert all(r.best_params == {} for r in results)


# --- make_group_runner: Settings captured directly (Connect-safe) --------------


def _settings() -> Any:
    from scale_forecasting.settings import Settings

    return Settings(
        project_id="proj-x",
        connection="proj-x.us-central1.conn",
        warehouse_uri="gs://bkt/warehouse",
        dataset_id="ds_x",
        region="us-central1",
    )


def test_make_group_runner_passes_captured_settings_to_write_cells(
    monkeypatch: Any,
) -> None:
    """The runner closure captures the frozen ``Settings`` directly (no ``sparkContext.broadcast``).

    Locks the Spark Connect refactor: ``make_group_runner(cfg, settings, models)`` closes over the
    picklable ``Settings`` by value and hands that exact object to ``bq.write_cells(settings=...)``,
    with no ``.value`` broadcast indirection. Driving the returned ``_run`` on a real bucket frame
    (so ``run_group`` produces results) and capturing the ``write_cells`` kwargs proves the seam.
    """
    from scale_forecasting.registry import bq

    captured: dict[str, Any] = {}

    def _fake_write_cells(results: Any, *, settings: Any = None) -> None:
        captured["results"] = results
        captured["settings"] = settings

    monkeypatch.setattr(bq, "write_cells", _fake_write_cells)

    cfg = _cfg(models=["theta"])
    settings = _settings()
    runner = spark_io.make_group_runner(cfg, settings, ["theta"])

    pdf = _with_model_col(_panel(["s0"]), ["theta"])
    status = runner(pdf)

    # The exact Settings object was captured and forwarded (identity, not a broadcast wrapper).
    assert captured["settings"] is settings
    assert not hasattr(captured["settings"], "value")
    assert len(captured["results"]) == 1
    assert list(status.columns) == list(STATUS_COLUMNS)


def test_make_group_runner_skips_write_when_no_results(monkeypatch: Any) -> None:
    """An empty bucket writes nothing but still returns the (empty) status frame."""
    from scale_forecasting.registry import bq

    called = {"n": 0}
    monkeypatch.setattr(bq, "write_cells", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    cfg = _cfg(models=["theta"])
    runner = spark_io.make_group_runner(cfg, _settings(), ["theta"])

    empty = _with_model_col(_panel(["s0"]), ["theta"]).iloc[0:0]
    status = runner(empty)

    assert called["n"] == 0
    assert list(status.columns) == list(STATUS_COLUMNS)
    assert len(status) == 0


# --- aggregate_status: the driver's header roll-up -----------------------------


def _status(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([(t, m, s, 0.1) for t, m, s in rows], columns=list(STATUS_COLUMNS))


def test_aggregate_all_ok_is_completed() -> None:
    out = aggregate_status(_status([("s0", "theta", "ok"), ("s1", "theta", "ok")]))
    assert out.status == "COMPLETED"
    assert out.n_series == 2
    assert out.n_cells == 2
    assert out.n_ok == 2
    assert out.n_error == 0


def test_aggregate_mixed_is_partial() -> None:
    out = aggregate_status(_status([("s0", "theta", "ok"), ("s1", "theta", "error")]))
    assert out.status == "PARTIAL"
    assert out.n_ok == 1
    assert out.n_error == 1


def test_aggregate_all_error_is_failed() -> None:
    out = aggregate_status(_status([("s0", "theta", "error"), ("s1", "theta", "error")]))
    assert out.status == "FAILED"
    assert out.n_ok == 0


def test_aggregate_empty_is_failed() -> None:
    out = aggregate_status(pd.DataFrame(columns=list(STATUS_COLUMNS)))
    assert out.status == "FAILED"
    assert out.n_cells == 0
    assert out.n_series == 0


# --- read_source_series: connector option wiring (no Spark) ---------------------


class _FakeReader:
    """A chainable stand-in for Spark's DataFrameReader that records every ``.option`` call."""

    def __init__(self) -> None:
        self.opts: dict[str, str] = {}

    def format(self, fmt: str) -> _FakeReader:
        self.opts["format"] = fmt
        return self

    def option(self, key: str, value: str) -> _FakeReader:
        self.opts[key] = value
        return self

    def load(self) -> _FakeDF:
        return _FakeDF()


class _FakeDF:
    def select(self, *_cols: Any) -> _FakeDF:
        return self


class _FakeSpark:
    def __init__(self) -> None:
        self.read = _FakeReader()


def test_read_source_series_sets_arrow_explicitly(monkeypatch: Any) -> None:
    # ARROW is set explicitly (not left to the connector default); snapshot pinning is stubbed off
    # here so the test isolates the format/parallelism wiring.
    monkeypatch.setattr(spark_io, "_snapshot_millis", lambda cfg, settings: None)
    spark = _FakeSpark()
    cfg = _cfg(data={"source_table": "source_series_native", "freq": "D", "horizon": HORIZON})
    spark_io.read_source_series(spark, cfg, _settings())
    assert spark.read.opts["format"] == "bigquery"
    assert spark.read.opts["readDataFormat"] == "ARROW"
    assert "maxParallelism" not in spark.read.opts  # unset knob → server chooses


def test_read_source_series_caps_streams_when_read_max_streams_set(monkeypatch: Any) -> None:
    monkeypatch.setattr(spark_io, "_snapshot_millis", lambda cfg, settings: None)
    spark = _FakeSpark()
    cfg = _cfg(
        compute={"read_max_streams": 3},
        data={"source_table": "source_series_native", "freq": "D", "horizon": HORIZON},
    )
    spark_io.read_source_series(spark, cfg, _settings())
    assert spark.read.opts["maxParallelism"] == "3"
