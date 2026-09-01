"""Offline tests for the Ray-engine core (``scale_forecasting.engines.ray_io``).

No Ray, no Vertex, no GPU, no BigQuery: the pure sizing/routing/chunking logic is exercised against
real ``RunConfig`` objects. The live cluster + fractional-GPU path is the ``@gpu`` smoke in
``tests/integration/test_ray_gpu_smoke.py``; the on-cluster driver is ``test_ray_engine.py``.

The two load-bearing properties — sizing to the run's scale, and showing resizing:
:func:`plan_cluster` is a deterministic function of the config, and a larger ``series_limit`` yields
a strictly larger fixed-size-equivalent (and vice-versa). Autoscaling is
the default: the plan carries per-pool ``[min, max]`` bounds that the launcher turns into an
``AutoscalingSpec``; ``ray_autoscale=False`` restores the fixed path.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from scale_forecasting import resources
from scale_forecasting.config import RunConfig
from scale_forecasting.engines import ray_io
from scale_forecasting.engines.spark_io import _MODEL_COL
from scale_forecasting.profiling.cost import build_profile
from scale_forecasting.profiling.measure import MeasuredFit
from scale_forecasting.registry.ids import make_run_id

# theta/holtwinters = CPU (statistical); xgboost = CPU (ml); neuralprophet = GPU (deep_learning).
_CPU = "theta"
_GPU = "neuralprophet"


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "ray io test",
        "python_runtime": "ray",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 10},
        "models": [_CPU, _GPU],
    }
    base.update(over)
    return RunConfig(**base)


def _compute(**over: Any) -> dict[str, Any]:
    """A compute block with GPU on by default (so the GPU pool is sized)."""
    base: dict[str, Any] = {"use_gpu": True}
    base.update(over)
    return base


# --- split_gpu_cpu_models ------------------------------------------------------


def test_split_routes_neuralprophet_to_gpu_rest_to_cpu() -> None:
    gpu, cpu = ray_io.split_gpu_cpu_models(_cfg(models=[_CPU, _GPU, "xgboost", "holtwinters"]))
    assert gpu == [_GPU]
    assert cpu == [_CPU, "xgboost", "holtwinters"]


def test_split_honors_executed_subset() -> None:
    # main.run hands only the Python-runtime subset; split must respect it, not cfg.models.
    gpu, cpu = ray_io.split_gpu_cpu_models(_cfg(), models=[_CPU])
    assert gpu == []
    assert cpu == [_CPU]


def test_split_preserves_order() -> None:
    gpu, cpu = ray_io.split_gpu_cpu_models(_cfg(models=["holtwinters", _CPU]))
    assert cpu == ["holtwinters", _CPU]  # input order, not sorted


# --- calibrate_gpu_fraction ----------------------------------------------------


def test_calibrate_fixed_fraction_passthrough() -> None:
    cfg = _cfg(compute=_compute(gpu_fraction=0.25))
    assert ray_io.calibrate_gpu_fraction(cfg) == 0.25


def test_calibrate_auto_solves_from_injected_peak() -> None:
    # A 4 GiB peak on a 16 GiB T4 with the default 1.3 margin → 4*1.3/16 = 0.325.
    cfg = _cfg(compute=_compute(gpu_fraction="auto", gpu_safety_margin=1.3))
    frac = ray_io.calibrate_gpu_fraction(cfg, measured_peaks_bytes=[4 * 1024**3])
    assert frac == pytest.approx(0.325)


def test_calibrate_auto_takes_worst_case_peak() -> None:
    cfg = _cfg(compute=_compute(gpu_fraction="auto", gpu_safety_margin=1.5))
    frac = ray_io.calibrate_gpu_fraction(
        cfg, measured_peaks_bytes=[1 * 1024**3, 8 * 1024**3, 2 * 1024**3]
    )
    assert frac == pytest.approx(0.75)  # 8 GiB (the max) × 1.5 / 16 GiB


def test_calibrate_auto_clamps_high_to_one() -> None:
    # A peak larger than the device (+ margin) can't exceed a whole GPU.
    cfg = _cfg(compute=_compute(gpu_fraction="auto"))
    assert ray_io.calibrate_gpu_fraction(cfg, measured_peaks_bytes=[20 * 1024**3]) == 1.0


def test_calibrate_auto_clamps_low_to_floor() -> None:
    cfg = _cfg(compute=_compute(gpu_fraction="auto", gpu_safety_margin=1.01))
    frac = ray_io.calibrate_gpu_fraction(cfg, measured_peaks_bytes=[1024])  # ~nothing
    assert frac == ray_io._MIN_FRACTION


def test_calibrate_auto_no_measurements_falls_back_to_nominal() -> None:
    cfg = _cfg(compute=_compute(gpu_fraction="auto"))
    assert (
        ray_io.calibrate_gpu_fraction(cfg, measured_peaks_bytes=[]) == ray_io._NOMINAL_AUTO_FRACTION
    )


def test_device_memory_known_and_unknown() -> None:
    assert ray_io.device_memory_bytes("T4") == 16 * 1024**3
    assert ray_io.device_memory_bytes("L4") == 24 * 1024**3
    # An unknown device assumes the smallest known one: under-pack (waste) beats over-pack (OOM).
    assert ray_io.device_memory_bytes("H100") == 16 * 1024**3
    assert ray_io.device_memory_bytes(None) == 16 * 1024**3


def test_calibrate_auto_sizes_against_the_l4s_larger_memory() -> None:
    """The same measured peak must yield a smaller fraction on a bigger device.

    An L4 is 24 GiB against a T4's 16 GiB. Sizing an L4 with the T4 denominator packed only two
    thirds of the tasks the device could hold — a silent 1.5x GPU over-spend.
    """
    cfg = _cfg(compute=_compute(gpu_fraction="auto", gpu_safety_margin=1.2))
    peak = [6 * 1024**3]
    t4 = ray_io.calibrate_gpu_fraction(cfg, measured_peaks_bytes=peak, gpu_type="T4")
    l4 = ray_io.calibrate_gpu_fraction(cfg, measured_peaks_bytes=peak, gpu_type="L4")
    assert t4 == pytest.approx(0.45)  # 6 x 1.2 / 16 → 2 tasks per device
    assert l4 == pytest.approx(0.30)  # 6 x 1.2 / 24 → 3 tasks per device
    assert ray_io.gpu_slots_per_device(l4) > ray_io.gpu_slots_per_device(t4)


def test_calibrate_gpu_type_defaults_to_the_config() -> None:
    # No explicit gpu_type → the flat compute default, so a single-runtime run needs no plumbing.
    cfg = _cfg(compute=_compute(gpu_fraction="auto", gpu_safety_margin=1.2, gpu_type="L4"))
    frac = ray_io.calibrate_gpu_fraction(cfg, measured_peaks_bytes=[6 * 1024**3])
    assert frac == pytest.approx(0.30)


def test_gpu_slots_per_device() -> None:
    assert ray_io.gpu_slots_per_device(0.25) == 4
    assert ray_io.gpu_slots_per_device(0.5) == 2
    assert ray_io.gpu_slots_per_device(1.0) == 1
    assert ray_io.gpu_slots_per_device(0.75) == 1  # floor(1.33) == 1


# --- plan_cluster: determinism + sizing ----------------------------------------


def test_plan_is_deterministic() -> None:
    cfg = _cfg(compute=_compute())
    rid = make_run_id(cfg)
    a = ray_io.plan_cluster(cfg, run_id=rid)
    b = ray_io.plan_cluster(cfg, run_id=rid)
    assert a == b


def test_plan_autoscale_default_on_with_resolved_bounds() -> None:
    # Autoscaling is the default, and each pool carries resolved [min, max] bounds.
    plan = ray_io.plan_cluster(_cfg(compute=_compute()), run_id="rid")
    assert plan.autoscale is True
    assert plan.cpu_min_nodes == 1
    assert plan.gpu_min_nodes == 1
    # 10 series is a one-node run, so each ceiling sits at the burst floor — NOT the shared
    # ray_max_nodes (16). A small run gets a small elastic pool.
    assert plan.cpu_max_nodes == ray_io._AUTOSCALE_MAX_FLOOR
    assert plan.gpu_max_nodes == ray_io._AUTOSCALE_MAX_FLOOR


def test_plan_autoscale_ceiling_grows_with_the_run() -> None:
    """The point of the whole change: the ceiling tracks the fan-out, it is not a constant."""
    small = ray_io.plan_cluster(
        _cfg(data={"source_table": "s", "series_limit": 10}, compute=_compute()), run_id="r"
    )
    large = ray_io.plan_cluster(
        _cfg(data={"source_table": "s", "series_limit": 5000}, compute=_compute()), run_id="r"
    )
    assert large.cpu_max_nodes > small.cpu_max_nodes


def test_plan_autoscale_ceiling_still_capped_by_the_hard_ceiling() -> None:
    # ray_max_nodes remains the guardrail against a runaway fan-out requesting an unbounded pool.
    plan = ray_io.plan_cluster(
        _cfg(
            data={"source_table": "s", "series_limit": 1_000_000},
            compute=_compute(ray_max_nodes=4),
        ),
        run_id="r",
    )
    assert plan.cpu_max_nodes == 4
    assert plan.gpu_max_nodes == 4


def test_plan_autoscale_ceiling_never_below_the_pool_floor() -> None:
    # A pre-warmed pool (min 4) can't be handed an AutoscalingSpec whose max is below it.
    plan = ray_io.plan_cluster(
        _cfg(data={"source_table": "s", "series_limit": 10}, compute=_compute(ray_cpu_min_nodes=4)),
        run_id="r",
    )
    assert plan.cpu_max_nodes >= plan.cpu_min_nodes == 4


def test_plan_per_pool_max_override_is_a_pin_not_a_derivation() -> None:
    # A run can cap the (expensive) GPU pool independently of the (cheap) CPU pool. An explicit
    # value is honoured verbatim — this 10-series run would otherwise derive the burst floor.
    plan = ray_io.plan_cluster(
        _cfg(compute=_compute(ray_cpu_max_nodes=20, ray_gpu_max_nodes=4)), run_id="rid"
    )
    assert plan.cpu_max_nodes == 20
    assert plan.gpu_max_nodes == 4


def test_plan_pinned_max_below_pool_min_is_rejected_at_config_load() -> None:
    # An incoherent [min, max] can never reach plan_cluster — ComputeConfig rejects it at load, so
    # the sizing math never has to defend against an impossible AutoscalingSpec.
    with pytest.raises(ValueError, match="exceeds the cpu pool max"):
        _cfg(compute=_compute(ray_cpu_min_nodes=8, ray_cpu_max_nodes=4))


def test_plan_per_pool_min_override_respected() -> None:
    plan = ray_io.plan_cluster(
        _cfg(compute=_compute(ray_cpu_min_nodes=2, ray_gpu_min_nodes=1)), run_id="rid"
    )
    assert plan.cpu_min_nodes == 2
    # A used pool's derived node count is floored at its min.
    assert plan.cpu_node_count >= 2


def test_plan_autoscale_false_restores_fixed_plan() -> None:
    plan = ray_io.plan_cluster(_cfg(compute=_compute(ray_autoscale=False)), run_id="rid")
    assert plan.autoscale is False
    # The bounds are still resolved (for telemetry) even though no AutoscalingSpec is attached.
    assert isinstance(plan.cpu_max_nodes, int)
    assert isinstance(plan.gpu_max_nodes, int)


def test_plan_names_ephemeral_cluster_from_run_id() -> None:
    plan = ray_io.plan_cluster(_cfg(compute=_compute()), run_id="run-abc")
    assert plan.cluster_name == "sf-ray-run-abc"
    assert plan.reuse is False


def test_plan_reuse_targets_named_cluster_and_skips_lifecycle() -> None:
    plan = ray_io.plan_cluster(
        _cfg(compute=_compute(ray_cluster_name="my-standing-cluster")), run_id="run-abc"
    )
    assert plan.cluster_name == "my-standing-cluster"
    assert plan.reuse is True


def test_plan_gpu_off_sizes_no_gpu_pool() -> None:
    # use_gpu=False → NeuralProphet still routes to the GPU list, but no GPU nodes are provisioned
    # (the model would fall back to CPU inside the task). The CPU pool still runs the stat model.
    plan = ray_io.plan_cluster(_cfg(compute=_compute(use_gpu=False)), run_id="rid")
    assert plan.gpu_node_count == 0
    assert plan.cpu_node_count >= 1


def test_plan_all_cpu_models_size_no_gpu_pool() -> None:
    plan = ray_io.plan_cluster(_cfg(models=[_CPU, "holtwinters"], compute=_compute()), run_id="rid")
    assert plan.gpu_node_count == 0
    assert plan.n_gpu_cells == 0


def test_plan_accelerator_type_mapped_to_vertex_enum() -> None:
    plan = ray_io.plan_cluster(_cfg(compute=_compute()), run_id="rid")
    assert plan.accelerator_type == "NVIDIA_TESLA_T4"


def test_plan_l4_maps_to_vertex_enum_on_g2_machine() -> None:
    plan = ray_io.plan_cluster(
        _cfg(compute=_compute(gpu_type="L4", ray_gpu_machine_type="g2-standard-8")), run_id="rid"
    )
    assert plan.accelerator_type == "NVIDIA_L4"


def test_plan_l4_on_n1_machine_raises() -> None:
    # L4 attaches only to G2; the default n1 gpu machine is rejected at plan time, not at create.
    with pytest.raises(ValueError, match="requires a 'g2-' machine"):
        ray_io.plan_cluster(_cfg(compute=_compute(gpu_type="L4")), run_id="rid")


def test_plan_use_gpu_override_forces_pool_without_touching_run_id() -> None:
    # The per-family GPU decision flows as an argument, not a cfg change: the flat default is CPU,
    # but the override provisions the GPU pool while the run_id (a cfg digest) stays identical.
    cfg = _cfg(compute=_compute(use_gpu=False))
    off = ray_io.plan_cluster(cfg, run_id="rid")
    on = ray_io.plan_cluster(cfg, run_id="rid", use_gpu=True)
    assert off.gpu_node_count == 0
    assert on.gpu_node_count >= 1


def test_plan_gpu_type_override_maps_without_touching_run_id() -> None:
    cfg = _cfg(compute=_compute(ray_gpu_machine_type="g2-standard-8"))
    plan = ray_io.plan_cluster(cfg, run_id="rid", gpu_type="L4")
    assert plan.accelerator_type == "NVIDIA_L4"


def test_plan_larger_scale_yields_larger_cluster() -> None:
    # The core "resize for the scale of the run" property: 10× the series ⇒ strictly more nodes.
    small = ray_io.plan_cluster(
        _cfg(data={"source_table": "s", "series_limit": 10}, compute=_compute()), run_id="r"
    )
    large = ray_io.plan_cluster(
        _cfg(data={"source_table": "s", "series_limit": 1000}, compute=_compute()), run_id="r"
    )
    assert large.total_worker_nodes > small.total_worker_nodes
    assert large.cpu_node_count > small.cpu_node_count


def test_plan_node_count_clamped_to_max() -> None:
    plan = ray_io.plan_cluster(
        _cfg(
            data={"source_table": "s", "series_limit": 1_000_000},
            compute=_compute(ray_max_nodes=4),
        ),
        run_id="r",
    )
    assert plan.cpu_node_count <= 4
    assert plan.gpu_node_count <= 4


def test_plan_smaller_scale_yields_single_node_each() -> None:
    plan = ray_io.plan_cluster(
        _cfg(data={"source_table": "s", "series_limit": 1}, compute=_compute()), run_id="r"
    )
    assert plan.cpu_node_count == 1
    assert plan.gpu_node_count == 1


def test_plan_unbounded_series_sizes_from_max_parallelism() -> None:
    # No series_limit → sizing uses max_parallelism as the cell basis (best guess), still fixed.
    plan = ray_io.plan_cluster(
        _cfg(
            data={"source_table": "s"},  # no series_limit
            compute=_compute(max_parallelism=100),
        ),
        run_id="r",
    )
    assert plan.cpu_node_count >= 1


def test_plan_finer_gpu_fraction_packs_more_and_needs_fewer_nodes() -> None:
    # A smaller fraction packs more NP tasks per T4, so the same cells need no more GPU nodes.
    coarse = ray_io.plan_cluster(
        _cfg(data={"source_table": "s", "series_limit": 64}, compute=_compute(gpu_fraction=0.5)),
        run_id="r",
    )
    fine = ray_io.plan_cluster(
        _cfg(data={"source_table": "s", "series_limit": 64}, compute=_compute(gpu_fraction=0.25)),
        run_id="r",
    )
    assert fine.gpu_node_count <= coarse.gpu_node_count


# --- plan_pool: the measured-profile translation -------------------------------

_GIB = 1024**3


def _fit(family: str, *, model_type: str, rss: int | None, gpu_bytes: int | None = None):
    """One measurement for ``family``, single-threaded, at the given process footprint."""
    return MeasuredFit(
        ts_id="s1",
        model_type=model_type,
        family=family,
        n_obs=1000,
        wall_s=1.0,
        cpu_s=1.0,
        peak_rss_bytes=1024,
        peak_gpu_bytes=gpu_bytes,
        ok=True,
        error=None,
        intraop_threads=1,
        host_cpu_count=8,
        process_rss_bytes=rss,
    )


def test_an_unprofiled_pool_reproduces_the_constants_the_engine_used_inline() -> None:
    """The safety property the whole wiring rests on: no measurement, no behaviour change."""
    cfg = _cfg(compute=_compute())
    cpu = ray_io.plan_pool(cfg, [_CPU, "xgboost"], 1000, gpu=False)
    gpu = ray_io.plan_pool(cfg, [_GPU], 1000, gpu=True, gpu_type="T4")
    assert cpu.slots_per_unit == resources.machine_cores(cfg.compute.ray_cpu_machine_type)
    assert gpu.slots_per_unit == cfg.compute.accelerator_count * ray_io.gpu_slots_per_device(
        ray_io._sizing_fraction(cfg)
    )
    assert cpu.task_options == {"num_cpus": 1}
    assert gpu.task_options == {"num_gpus": ray_io._sizing_fraction(cfg)}


def test_a_shared_cpu_pool_is_sized_for_the_heaviest_family_that_lands_on_it() -> None:
    """statistical and ml cells go through the same worker, so its slot must hold either one."""
    profile = build_profile(
        [
            _fit("statistical", model_type=_CPU, rss=1 * _GIB),
            _fit("ml", model_type="xgboost", rss=5 * _GIB),
        ],
        memory_margin=1.0,
        time_margin=1.0,
    )
    plan = ray_io.plan_pool(_cfg(compute=_compute()), [_CPU, "xgboost"], 1000, gpu=False,
                            profile=profile)
    assert plan.slot.memory_bytes == 5 * _GIB
    assert plan.family == "statistical+ml"
    assert plan.task_options["memory"] == 5 * _GIB


def test_a_measured_heavy_family_shrinks_the_density_and_widens_the_fleet() -> None:
    """The behaviour change W6 exists for: sizing follows the work, not just the cell count."""
    cfg = _cfg(data={"source_table": "s", "series_limit": 200}, compute=_compute(use_gpu=False))
    light = ray_io.plan_cluster(cfg, [_CPU], run_id="r")
    heavy = ray_io.plan_cluster(
        cfg,
        [_CPU],
        run_id="r",
        profile=build_profile(
            [_fit("statistical", model_type=_CPU, rss=8 * _GIB)],
            memory_margin=1.0,
            time_margin=1.0,
        ),
    )
    assert heavy.cpu_pool.slots_per_unit < light.cpu_pool.slots_per_unit
    assert heavy.cpu_node_count > light.cpu_node_count


def test_a_measured_device_footprint_beats_the_nominal_sizing_fraction() -> None:
    """The L4 under-pack this line of work started from, at the pool seam."""
    profile = build_profile(
        [_fit("deep_learning", model_type=_GPU, rss=None, gpu_bytes=4 * _GIB)],
        memory_margin=1.0,
        time_margin=1.0,
    )
    plan = ray_io.plan_pool(
        _cfg(compute=_compute()), [_GPU], 1000, gpu=True, gpu_type="L4", profile=profile
    )
    assert plan.task_options["num_gpus"] == (4 * _GIB) / ray_io.device_memory_bytes("L4")


def test_a_live_calibrated_fraction_beats_the_submit_time_nominal() -> None:
    """On the cluster the engine has measured a real device; the plan should use that number."""
    plan = ray_io.plan_pool(
        _cfg(compute=_compute()), [_GPU], 1000, gpu=True, gpu_type="T4", gpu_fraction=0.2
    )
    assert plan.task_options == {"num_gpus": 0.2}
    assert plan.slots_per_unit == 5


def test_the_stored_pool_plans_carry_the_ceiling_the_pool_can_actually_reach() -> None:
    """``slots_at_ceiling`` feeds the chunk floor, so it must be the autoscaling max not the cap."""
    plan = ray_io.plan_cluster(
        _cfg(data={"source_table": "s", "series_limit": 1000}, compute=_compute()), run_id="r"
    )
    assert plan.cpu_pool.max_units == plan.cpu_max_nodes
    assert plan.gpu_pool.max_units == plan.gpu_max_nodes
    assert plan.cpu_pool.derived_units == plan.cpu_node_count
    assert plan.gpu_pool.derived_units == plan.gpu_node_count


def test_an_unused_gpu_pool_plans_no_nodes() -> None:
    plan = ray_io.plan_cluster(_cfg(compute=_compute(use_gpu=False)), run_id="r")
    assert plan.gpu_pool.derived_units == 0
    assert plan.gpu_pool.n_cells == 0


# --- chunk_cells ---------------------------------------------------------------


def _source(n_series: int, rows_each: int = 3) -> pd.DataFrame:
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


def test_chunk_cells_tags_model_and_covers_every_cell() -> None:
    src = _source(4)
    chunks = ray_io.chunk_cells(src, _cfg(), [_CPU, _GPU], n_chunks=3)
    assert all(_MODEL_COL in c.columns for c in chunks)
    # 4 series × 2 models = 8 cells, each a distinct (ts_id, model) across all chunks.
    seen = set()
    for c in chunks:
        for (ts_id, model), _sub in c.groupby(["ts_id", _MODEL_COL]):
            seen.add((ts_id, model))
    assert len(seen) == 8


def test_chunk_cells_keeps_a_cells_history_together() -> None:
    src = _source(3, rows_each=5)
    chunks = ray_io.chunk_cells(src, _cfg(), [_CPU], n_chunks=5)
    # Every (ts_id, model) cell appears in exactly one chunk, with its full 5-row history.
    locations: dict[tuple[str, str], int] = {}
    for idx, c in enumerate(chunks):
        for key, sub in c.groupby(["ts_id", _MODEL_COL]):
            assert len(sub) == 5
            locations.setdefault(key, idx)  # type: ignore[arg-type]
            assert locations[key] == idx  # never split across chunks
    assert len(locations) == 3


def test_chunk_cells_is_deterministic() -> None:
    src = _source(6)
    a = ray_io.chunk_cells(src, _cfg(), [_CPU, _GPU], n_chunks=4)
    b = ray_io.chunk_cells(src, _cfg(), [_CPU, _GPU], n_chunks=4)
    assert len(a) == len(b)
    for ca, cb in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(ca, cb)


def test_chunk_cells_empty_source_or_models_yields_nothing() -> None:
    assert ray_io.chunk_cells(pd.DataFrame(), _cfg(), [_CPU], n_chunks=2) == []
    assert ray_io.chunk_cells(_source(2), _cfg(), [], n_chunks=2) == []
