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


def test_bucket_count_defaults_to_cap_when_nobody_knows_how_many_series_there_are() -> None:
    cfg = _cfg(compute={"max_parallelism": 123})
    # series_limit unset and no estimate offered → fall back to the parallelism cap. A guess is
    # still better than one bucket; the estimate below is how a caller does better than a guess.
    assert default_bucket_count(cfg) == 123


def test_an_estimated_series_count_sizes_an_unbounded_run_the_way_a_limit_would() -> None:
    """The 100k OOM guard, on the shape production actually runs.

    ``series_limit=None`` means "forecast the whole table", and it was the last caller still
    sizing off ``max_parallelism`` — the same ``min(cells, cap)`` rule that fattened frames past
    the executor budget. The bounded path got the target-cells fix; this is the unbounded path
    getting it, from a count the caller went and found rather than one the config declared.
    """
    cfg = _cfg(
        models=["theta", "holtwinters", "sarimax", "xgboost"],
        compute={"max_parallelism": 50, "bucket_target_cells": 8},
    )
    assert default_bucket_count(cfg) == 50  # the cap, as before
    assert default_bucket_count(cfg, n_series=100_000) == 50_000  # ceil(400k cells / 8)


def test_a_declared_limit_outranks_an_estimate_of_the_whole_table() -> None:
    """``series_limit`` is what this run will read; the table's size is not.

    An estimate is only ever an answer to "how much is there when nobody said" — letting it win
    would size a deliberately-subsetted run against every series it chose not to touch.
    """
    cfg = _cfg(
        models=["theta"],
        data={"source_table": "t", "series_limit": 80},
        compute={"bucket_target_cells": 8},
    )
    assert default_bucket_count(cfg, n_series=100_000) == 10


# --- cost-weighted bucketing: a bucket is a unit of *work*, not of cells --------


def _profile(**walls: float | None) -> Any:
    """A `ComputeProfile` carrying nothing but per-model median wall times."""
    from scale_forecasting.profiling.cost import ComputeProfile, ModelCost

    models = {
        name: ModelCost(
            model_type=name,
            family="statistical",
            n_fits=1,
            n_ok=1,
            max_n_obs=100,
            max_peak_rss_bytes=None,
            max_peak_gpu_bytes=None,
            median_wall_s=wall,
            median_cpu_s=None,
            max_effective_cores=None,
        )
        for name, wall in walls.items()
    }
    return ComputeProfile(
        families={},
        models=models,
        memory_margin=1.0,
        time_margin=1.0,
        n_measurements=len(models),
        n_ok=len(models),
        sample_ts_ids=(),
    )


def test_cost_weights_are_relative_to_the_median_measured_model() -> None:
    # naive 0.5s, theta 1.0s, neuralprophet 8.0s → median 1.0 is "typical", so the weights are the
    # multiples of it. The median, not the mean: a mean of 3.17 would call theta a third of typical.
    weights = spark_io.model_cost_weights(
        ["naive", "theta", "neuralprophet"],
        _profile(naive=0.5, theta=1.0, neuralprophet=8.0),
    )
    assert weights == {"naive": 0.5, "theta": 1.0, "neuralprophet": 8.0}


def test_an_unmeasured_model_is_priced_as_a_typical_cell() -> None:
    # xgboost was never measured → weight exactly 1.0, which is what makes it fall back to the
    # global bucket_target_cells rather than to whatever the loudest measured model implies.
    weights = spark_io.model_cost_weights(
        ["naive", "neuralprophet", "xgboost"], _profile(naive=0.5, neuralprophet=8.0)
    )
    assert weights["xgboost"] == 1.0


def test_an_impossible_reading_is_dropped_rather_than_believed() -> None:
    # A zero or a negative wall time would make a model's slice one bucket wide however many
    # series it has — the opposite of what a "free" model would need. Treated as unmeasured.
    weights = spark_io.model_cost_weights(["naive", "theta"], _profile(naive=0.0, theta=2.0))
    assert weights == {"naive": 1.0, "theta": 1.0}  # theta alone is the median → everything typical


def test_no_profile_prices_every_model_the_same() -> None:
    assert spark_io.model_cost_weights(["naive", "theta"], None) == {"naive": 1.0, "theta": 1.0}


def test_a_slow_model_is_cut_into_more_buckets_than_a_fast_one() -> None:
    # The whole point: 4x the cost buys 4x the buckets, so a quarter of the cells per frame and
    # about the same wall time per bucket as everyone else's.
    alloc = spark_io.allocate_buckets(["fast", "slow"], 100, {"fast": 1.0, "slow": 4.0})
    assert alloc["fast"][1] == 20
    assert alloc["slow"][1] == 80


def test_a_cheap_model_keeps_the_global_target_instead_of_a_fatter_frame() -> None:
    # naive at a tenth of typical must NOT get a tenth of the buckets — that is ten times the cells
    # in one pandas frame, which is the OOM the bucket target exists to prevent. Floored at 1.0, so
    # naive is sized as typical and only the slow model is cut finer.
    alloc = spark_io.allocate_buckets(["naive", "slow"], 100, {"naive": 0.1, "slow": 4.0})
    assert alloc["naive"][1] == 20  # 1 part in 5, the same as an unmeasured model would get
    assert alloc["slow"][1] == 80


def test_the_slices_tile_the_bucket_space_with_no_gap_and_no_overlap() -> None:
    # The slices ARE the bucket space: a gap is a task with nothing in it, an overlap puts two
    # models in one applyInPandas frame. Checked exhaustively over an awkward, unroundable split.
    models = [f"m{i}" for i in range(7)]
    weights = {"m0": 1.0, "m1": 1.3, "m2": 2.7, "m3": 5.0, "m4": 0.2, "m5": 11.1, "m6": 1.0}
    for n_buckets in range(7, 260):
        alloc = spark_io.allocate_buckets(models, n_buckets, weights)
        covered: list[int] = []
        for offset, width in alloc.values():
            assert width >= 1
            covered.extend(range(offset, offset + width))
        assert sorted(covered) == list(range(n_buckets))


def test_no_evidence_means_no_allocation_so_the_flat_hash_is_untouched() -> None:
    # Equal weights would give equal slices, which buys nothing over the flat hash and costs the
    # mixing it gives. Absence of an allocation is the signal to keep today's behaviour exactly.
    assert spark_io.allocate_buckets(["a", "b", "c"], 90, {"a": 1.0, "b": 1.0, "c": 1.0}) == {}
    assert spark_io.allocate_buckets(["a", "b"], 90, {}) == {}
    # And below-typical models are all floored to typical, so they are uniform too.
    assert spark_io.allocate_buckets(["a", "b"], 90, {"a": 0.2, "b": 0.9}) == {}


def test_a_bucket_space_too_small_to_seat_every_model_declines_to_allocate() -> None:
    # 3 buckets, 4 models: some model would get zero buckets and its cells nowhere to go.
    assert spark_io.allocate_buckets(["a", "b", "c", "d"], 3, {"a": 9.0, "b": 1.0}) == {}
    # One more bucket and it fits — one each, then the remainder to the expensive one.
    alloc = spark_io.allocate_buckets(["a", "b", "c", "d"], 4, {"a": 9.0, "b": 1.0})
    assert sorted(w for _, w in alloc.values()) == [1, 1, 1, 1]


def test_the_same_inputs_always_produce_the_same_slices() -> None:
    # Ties break on model name, so two equally-weighted models never swap places between the
    # driver's allocation and anything that re-derives it from the stamped record.
    models = ["b", "a", "c"]
    weights = {"a": 1.0, "b": 1.0, "c": 4.0}
    first = spark_io.allocate_buckets(models, 47, weights)
    assert all(spark_io.allocate_buckets(models, 47, weights) == first for _ in range(5))
    # Offsets follow the caller's model order, not sorted order — the cross-join's order.
    assert [name for name in first] == ["b", "a", "c"]
    assert first["b"][0] == 0


def test_a_slow_model_raises_the_bucket_count_rather_than_fattening_its_frames() -> None:
    # 100 series, target 8, two models. Uniform: ceil(200/8) = 25 buckets. With neuralprophet
    # measured at 8x typical it needs its own target of 1 cell, so ceil(100/8) + ceil(100*8/8)
    # = 13 + 100 = 113. The count rises; no frame gets bigger.
    cfg = _cfg(
        models=["theta", "neuralprophet"],
        data={"source_table": "t", "series_limit": 100},
        compute={"bucket_target_cells": 8},
    )
    assert default_bucket_count(cfg) == 25
    weights = {"theta": 1.0, "neuralprophet": 8.0}
    assert default_bucket_count(cfg, weights=weights) == 113


def test_uniform_weights_size_the_run_byte_for_byte_the_way_no_weights_do() -> None:
    # sum(ceil(series/target)) and ceil(series*models/target) differ by rounding, and that is not a
    # difference worth introducing to the live-proven path.
    cfg = _cfg(
        models=["theta", "holtwinters", "sarimax"],
        data={"source_table": "t", "series_limit": 100},
        compute={"bucket_target_cells": 8},
    )
    flat = default_bucket_count(cfg)
    assert default_bucket_count(cfg, weights=None) == flat
    assert default_bucket_count(cfg, weights=dict.fromkeys(cfg.models, 1.0)) == flat
    assert default_bucket_count(cfg, weights={"theta": 0.3}) == flat  # floored to typical


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
    raised = spark_io.reachable_bucket_count(40, max_executors=2000, executor_cores=96, task_cpus=1)
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


def test_the_shuffle_width_is_set_wider_than_the_bucket_count() -> None:
    # Without this, groupBy(...).applyInPandas plans spark.sql.shuffle.partitions tasks — 200 by
    # default — no matter how many buckets the cells were hashed into. Wider than the bucket count
    # rather than equal to it: see the occupancy test below.
    props = spark_io.fanout_properties(4000)
    assert props["spark.sql.shuffle.partitions"] == "12000"


def test_aqe_is_stopped_from_coalescing_the_width_away() -> None:
    # Setting the partition count is not enough on its own: AQE coalesces small partitions after
    # the fact, which is how a 200-task stage becomes a 1-task stage. The old form of this,
    # `coalescePartitions.minPartitionNum`, is deprecated and outranked by minPartitionSize.
    props = spark_io.fanout_properties(4000)
    assert props["spark.sql.adaptive.coalescePartitions.enabled"] == "false"
    assert "spark.sql.adaptive.coalescePartitions.minPartitionNum" not in props


def test_widening_the_shuffle_lifts_bucket_occupancy_from_two_thirds_to_six_sevenths() -> None:
    """Why the factor exists, simulated on Spark's own partitioner (pure, no Spark).

    A bucket lands in partition ``pmod(hash(bucket), width)``. At ``width == buckets`` that is
    n balls into n bins, so ``(1 - 1/n)^n → 1/e`` of the bins stay empty and only ~63% of the
    tasks in the stage do any work — the fan-out looks n-wide and runs 0.63n-wide. Widening the
    shuffle costs nothing real (``applyInPandas`` is invoked once per group, not per partition)
    and moves the expected non-empty count to ``3n(1 - e^(-1/3)) ≈ 0.85n``.
    """

    import zlib

    def occupancy(n_buckets: int, width: int) -> float:
        # crc32 stands in for Spark's murmur3 — the arithmetic is about a uniform hash, not about
        # which uniform hash, and crc32 is deterministic across processes where `hash()` is not.
        hit = {zlib.crc32(f"bucket-{i}".encode()) % width for i in range(n_buckets)}
        return len(hit) / n_buckets

    n = 4000
    assert 0.60 < occupancy(n, n) < 0.67
    widened = int(spark_io.fanout_properties(n)["spark.sql.shuffle.partitions"])
    assert 0.82 < occupancy(n, widened) < 0.88


def _fanout_cfg(**compute: Any) -> RunConfig:
    return _cfg(data={"source_table": "t", "series_limit": 100}, compute=compute)


def test_widen_fanout_raises_the_count_and_sizes_the_shuffle_off_the_raised_one() -> None:
    spark = _SessionStub(
        {"spark.dynamicAllocation.maxExecutors": "50", "spark.executor.cores": "8"}
    )
    assert spark_explode._widen_fanout(_fanout_cfg(), spark, 40)["buckets"] == 400
    # The width must follow the *raised* count, not the policy count it started from.
    assert spark.conf.get("spark.sql.shuffle.partitions") == "1200"


def test_widen_fanout_still_sets_the_shuffle_when_the_count_is_already_wide_enough() -> None:
    # The raise and the width are independent: a fan-out that needs no widening still needs its
    # tasks to exist.
    spark = _SessionStub({"spark.dynamicAllocation.maxExecutors": "2", "spark.executor.cores": "4"})
    assert spark_explode._widen_fanout(_fanout_cfg(), spark, 500)["buckets"] == 500
    assert spark.conf.get("spark.sql.shuffle.partitions") == "1500"


def test_the_executed_fanout_record_carries_both_counts_and_the_confs_behind_them() -> None:
    """The shape nothing else can see.

    Every other number about a batch is decided at submit and echoed back by the API. The fan-out
    is decided on a live session against confs that may have come from `submit.sizing_properties`,
    a ``--max-executors`` flag, or the platform's own defaults — so "we asked for 40 buckets and
    ran 400 because something set a 50-executor ceiling" is only ever knowable here.
    """
    spark = _SessionStub(
        {
            "spark.dynamicAllocation.maxExecutors": "50",
            "spark.executor.cores": "8",
            "spark.task.cpus": "2",
        }
    )
    record = spark_explode._widen_fanout(_fanout_cfg(), spark, 40)
    assert record == {
        "buckets_policy": 40,
        "buckets": 200,  # 50 executors x (8 cores / 2 cpus-per-task)
        "shuffle_partitions": 600,  # 3x the buckets — see fanout_properties on occupancy
        "max_executors": 50,
        "executor_cores": 8,
        "task_cpus": 2,
        "widened": True,
    }


def test_a_fanout_that_needed_no_widening_says_so_rather_than_looking_like_one_that_did() -> None:
    spark = _SessionStub({"spark.dynamicAllocation.maxExecutors": "2", "spark.executor.cores": "4"})
    record = spark_explode._widen_fanout(_fanout_cfg(), spark, 500)
    assert record["widened"] is False
    assert record["buckets_policy"] == record["buckets"] == 500
    # An unset conf is None — "the platform default applies and we did not choose it" — not 0.
    assert record["task_cpus"] is None


def test_widen_fanout_does_not_widen_when_profiling_is_off() -> None:
    # The escape hatch has to survive Serverless writing its own dynamicAllocation defaults into
    # the driver conf — which is why the gate is here and not at the call site. A 1000-executor
    # ceiling nobody chose would otherwise raise 500 buckets to 4000.
    spark = _SessionStub(
        {"spark.dynamicAllocation.maxExecutors": "1000", "spark.executor.cores": "4"}
    )
    cfg = _fanout_cfg(profile={"mode": "off"})
    assert spark_explode._widen_fanout(cfg, spark, 500)["buckets"] == 500


def test_the_shuffle_width_is_not_part_of_the_profiling_escape_hatch() -> None:
    """Turning measurement off must not also turn the fan-out off.

    The width does not depend on any measurement — it says "the count we settled on is the count
    of tasks", and without it Spark plans its default 200 however many buckets the cells were
    hashed into. Skipping it here does not restore a pre-profiler run; it caps an unmeasured run
    at 200 tasks, which is a behaviour change smuggled in under a flag that reads as "measure
    nothing".
    """
    spark = _SessionStub({})
    cfg = _fanout_cfg(profile={"mode": "off"})
    assert spark_explode._widen_fanout(cfg, spark, 500)["buckets"] == 500
    assert spark.conf.get("spark.sql.shuffle.partitions") == "1500"
    assert spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled") == "false"


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


def test_status_schema_and_status_columns_cannot_drift() -> None:
    """The Spark ``StructType`` and the pandas column tuple are the same frame, declared twice.

    `run_group` builds its status frame from `STATUS_COLUMNS`; `spark_explode` hands
    `status_schema()` to ``applyInPandas`` as the promised return type. They are maintained by hand
    in two places, and Spark matches by *position*, not by name — so adding a column to one and not
    the other does not raise "unknown column", it silently reads the new column's values under the
    old column's name, or fails deep inside an executor with an arrow conversion error nobody can
    trace back here.

    Names and order both, for that reason. Types are left alone: this pins the drift, not the
    schema's content.
    """
    schema = spark_io.status_schema()
    assert [f.name for f in schema.fields] == list(STATUS_COLUMNS)


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
    picklable ``Settings`` by value and hands that exact object to
    ``cells.write_cells(settings=...)``, with no ``.value`` broadcast indirection. Driving the
    returned ``_run`` on a real bucket frame
    (so ``run_group`` produces results) and capturing the ``write_cells`` kwargs proves the seam.
    """
    from scale_forecasting.registry import cells

    captured: dict[str, Any] = {}

    def _fake_write_cells(results: Any, *, settings: Any = None) -> None:
        captured["results"] = results
        captured["settings"] = settings

    monkeypatch.setattr(cells, "write_cells", _fake_write_cells)

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
    from scale_forecasting.registry import cells

    called = {"n": 0}
    monkeypatch.setattr(
        cells, "write_cells", lambda *a, **k: called.__setitem__("n", called["n"] + 1)
    )

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


# --- the source is cached across the driver's several actions ------------------


class _RecordingSource:
    """A DataFrame stand-in that records the order of the driver's calls against it.

    Spark's laziness is the thing under test and it is invisible from a single call site — what
    matters is the *sequence*: cache the relation before anything reads it, and release it after
    the last read. A fake is the only way to see a sequence.
    """

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def persist(self, level: Any) -> _RecordingSource:
        self.log.append(f"persist:{level}")
        return self

    def unpersist(self) -> _RecordingSource:
        self.log.append("unpersist")
        return self


def _explode_with_fakes(monkeypatch: Any, log: list[str], *, boom: bool = False) -> Any:
    """Run the explode driver with every collaborator faked; return the recording source."""
    from pyspark import StorageLevel

    source = _RecordingSource(log)
    cells = object()

    def _read(spark: Any, cfg: Any, settings: Any) -> Any:
        log.append("read")
        return source

    def _hpo(src: Any, cfg: Any, executed: list[str]) -> dict[str, Any]:
        log.append("hpo")
        assert src is source  # the cached relation, not a fresh read
        return {}

    def _cross(src: Any, cfg: Any, spark: Any, executed: list[str]) -> Any:
        log.append("cross_join")
        if boom:
            raise RuntimeError("fan-out blew up")
        return cells

    class _Status:
        def applyInPandas(self, runner: Any, schema: Any) -> Any:
            return self

        def toPandas(self) -> pd.DataFrame:
            log.append("collect")
            return pd.DataFrame(
                {"ts_id": ["a"], "model_type": ["theta"], "status": ["ok"], "fit_seconds": [0.1]},
                columns=list(STATUS_COLUMNS),
            )

    class _Cells:
        def groupBy(self, col: str) -> _Status:
            return _Status()

    monkeypatch.setattr(spark_io, "read_source_series", _read)
    monkeypatch.setattr(spark_io, "resolve_fleetwide_hpo", _hpo)
    monkeypatch.setattr(spark_io, "cross_join_models", _cross)
    monkeypatch.setattr(spark_io, "add_bucket", lambda cells, cfg, n, alloc=None: _Cells())
    monkeypatch.setattr(spark_io, "make_group_runner", lambda *a, **k: None)
    monkeypatch.setattr(spark_io, "status_schema", lambda: "schema")
    # The cost weights are a registry read; this test is about the cache ordering, not evidence.
    monkeypatch.setattr(spark_explode, "_cost_weights", lambda cfg, executed, settings: {})
    monkeypatch.setattr(spark_explode, "_widen_fanout", lambda cfg, spark, n: {"buckets": 4})
    monkeypatch.setattr(spark_explode, "_stamp_executed_fanout", lambda *a, **k: None)

    cfg = _cfg(data={"source_table": "t", "freq": "D", "horizon": HORIZON, "series_limit": 5})
    spark_explode.run(
        cfg,
        manage_header=False,  # contributor mode: no header writes, so no GCP
        settings=_settings(),
        spark=_SessionStub({}),  # injected → the engine must not stop it
    )
    assert log[1] == f"persist:{StorageLevel.MEMORY_AND_DISK}"
    return source


def test_the_source_is_cached_before_the_first_action_and_released_after_the_last(
    monkeypatch: Any,
) -> None:
    """Persist right after the read, unpersist after the collect — nothing reads it uncached.

    Four actions run against this relation (the series count, the HPO sample, the cross-join, and
    the semi-join `series_limit` does inside it). Uncached, each one re-executes the BigQuery read.
    """
    log: list[str] = []
    _explode_with_fakes(monkeypatch, log)
    assert log[0] == "read"
    assert log[1].startswith("persist:")
    assert log[2:] == ["hpo", "cross_join", "collect", "unpersist"]


def test_the_cache_is_released_even_when_the_fan_out_raises(monkeypatch: Any) -> None:
    """An injected session outlives this call, so a leaked cache leaks in the caller's cluster."""
    import pytest

    log: list[str] = []
    with pytest.raises(RuntimeError, match="fan-out blew up"):
        _explode_with_fakes(monkeypatch, log, boom=True)
    assert log[-1] == "unpersist"
    assert "collect" not in log
