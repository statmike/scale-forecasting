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

import sys
from typing import Any

import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.engines import ray_io
from scale_forecasting.engines.spark_io import _MODEL_COL
from scale_forecasting.profiling.cost import build_profile
from scale_forecasting.profiling.measure import MeasuredFit
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.resources import catalog, fleet
from scale_forecasting.resources.fleet import UnitShape

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


def _scheduling_request(plan: fleet.RuntimeResourcePlan) -> dict[str, Any]:
    """``task_options`` minus the thread pin — what this file is about.

    Every plan also carries a ``runtime_env`` capping the native thread pools at the cores Ray
    assigns the task; that pin belongs to `fleet` and is asserted in ``test_resources.py``. The
    sizing tests here care only about what the *scheduler* is asked for, so they drop it rather
    than restate it and go stale the next time its contents change.
    """
    return {key: value for key, value in plan.task_options.items() if key != "runtime_env"}


# --- split_gpu_cpu_models ------------------------------------------------------


def test_split_routes_neuralprophet_to_gpu_rest_to_cpu() -> None:
    # compute=_compute() so a GPU pool actually exists — the split is hardware-aware, and without
    # a GPU there is no pool for neuralprophet to route to (see the GPU-off test below).
    gpu, cpu = ray_io.split_gpu_cpu_models(
        _cfg(models=[_CPU, _GPU, "xgboost", "holtwinters"], compute=_compute())
    )
    assert gpu == [_GPU]
    assert cpu == [_CPU, "xgboost", "holtwinters"]


def test_split_honors_executed_subset() -> None:
    # main.run hands only the Python-runtime subset; split must respect it, not cfg.models.
    gpu, cpu = ray_io.split_gpu_cpu_models(_cfg(compute=_compute()), models=[_CPU])
    assert gpu == []
    assert cpu == [_CPU]


def test_split_preserves_order() -> None:
    gpu, cpu = ray_io.split_gpu_cpu_models(_cfg(models=["holtwinters", _CPU]))
    assert cpu == ["holtwinters", _CPU]  # input order, not sorted


def test_split_sends_deep_learning_to_the_cpu_pool_when_there_is_no_gpu() -> None:
    # No GPU pool exists, so neuralprophet's cells are CPU work (it falls back inside the cell).
    # Routing them to a pool that will not be created is what left both pools at zero nodes.
    gpu, cpu = ray_io.split_gpu_cpu_models(_cfg(models=[_CPU, _GPU]))
    assert gpu == []
    assert cpu == [_CPU, _GPU]


def test_split_use_gpu_argument_overrides_the_flat_default() -> None:
    # The per-family resolved hardware wins over compute.use_gpu, the same way plan_cluster takes
    # it — a family override must not have to round-trip through cfg (which would move run_id).
    gpu, cpu = ray_io.split_gpu_cpu_models(_cfg(models=[_GPU]), use_gpu=True)
    assert (gpu, cpu) == ([_GPU], [])
    gpu, cpu = ray_io.split_gpu_cpu_models(_cfg(models=[_GPU], compute=_compute()), use_gpu=False)
    assert (gpu, cpu) == ([], [_GPU])


def test_plan_cluster_gives_a_gpu_less_deep_learning_job_real_workers() -> None:
    """A deep-learning-only Ray job without a GPU must still get worker nodes (regression).

    ``{"python_runtime": "ray", "models": ["neuralprophet"]}`` resolves ``deep_learning`` to
    ``hardware="cpu"`` (``compute.use_gpu`` defaults False), which used to plan a cluster with an
    empty GPU pool *and* an empty CPU pool — a head-only cluster the job then waited on forever,
    with no timeout and no error. The cells have to land in the CPU pool.
    """
    plan = ray_io.plan_cluster(_cfg(models=[_GPU]), [_GPU], run_id="x", use_gpu=False)
    assert plan.gpu_node_count == 0
    assert plan.cpu_node_count > 0, (
        "deep-learning cells must size the CPU pool when there is no GPU"
    )
    assert plan.n_gpu_cells == 0
    assert plan.n_cpu_cells > 0


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


def test_a_probe_that_measured_zero_is_a_probe_that_failed_not_a_free_model() -> None:
    """Zeros used to survive into ``max()`` and drag the fraction to the floor.

    ``torch.cuda.max_memory_allocated()`` returns 0 both when a model genuinely allocated nothing
    and when the probe ran somewhere without a device to allocate on. The old code could not tell
    those apart, so a pool whose probes all failed reported a peak of 0 bytes, solved to 0, and
    clamped to `_MIN_FRACTION` — the *densest* possible packing, chosen on the strength of no
    evidence whatsoever. The nominal fallback is the conservative answer, and it is the one an
    absent measurement has to produce however the absence is spelled.
    """
    cfg = _cfg(compute=_compute(gpu_fraction="auto"))
    for peaks in ([0, 0, 0], [None, 0], [None], [None, None]):
        assert ray_io.calibrate_gpu_fraction(cfg, measured_peaks_bytes=peaks) == (
            ray_io._NOMINAL_AUTO_FRACTION
        ), peaks


def test_one_real_probe_outvotes_the_ones_that_came_back_empty() -> None:
    """A partial failure is still a measurement — the surviving probe sizes the pool."""
    cfg = _cfg(compute=_compute(gpu_fraction="auto", gpu_safety_margin=1.2))
    assert ray_io.calibrate_gpu_fraction(
        cfg, measured_peaks_bytes=[None, 0, 6 * 1024**3]
    ) == pytest.approx(0.45)


def test_the_footprint_neuralprophet_actually_has_lands_on_the_floor_not_near_it() -> None:
    """The Phase-0 number, run through the real arithmetic, to show what the clamp is holding up.

    76 KiB is the peak device memory a live NeuralProphet fit reached on a T4 — 0.00045% of the
    card. Solved honestly that is a fraction of about six millionths, which would ask Ray to pack
    ~160,000 cells onto one device. Nothing in the memory arithmetic stops that; `_MIN_FRACTION`
    does, and this test exists so that the floor is never mistaken for a rounding detail. The real
    limit on GPU density is the node's cores, which is `resources.fleet`'s job, not this one.
    """
    cfg = _cfg(compute=_compute(gpu_fraction="auto", gpu_safety_margin=1.3))
    frac = ray_io.calibrate_gpu_fraction(cfg, measured_peaks_bytes=[77_824], gpu_type="T4")
    assert frac == ray_io._MIN_FRACTION
    assert (77_824 * 1.3) / ray_io.device_memory_bytes("T4") < ray_io._MIN_FRACTION


# --- where the calibration probe runs ------------------------------------------


class _FakeRemoteFn:
    def __init__(self, ray: _FakeRay, fn: Any) -> None:
        self._ray, self.fn = ray, fn

    def remote(self, *_args: Any, **_kwargs: Any) -> Any:
        self._ray.calls += 1
        return ("future", self._ray.calls)


class _FakeRay:
    """The three pieces of Ray's API `_measure_np_peaks_on_device` touches, recording the options.

    Deliberately not the real thing. What needs locking down is that the probe asks the scheduler
    for a *whole device*, and that is a property of the call, not of Ray.
    """

    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.options: dict[str, Any] | None = None
        self.timeout: float | None = None
        self.calls = 0

    def remote(self, **options: Any) -> Any:
        self.options = options
        return lambda fn: _FakeRemoteFn(self, fn)

    def get(self, _futures: list[Any], timeout: float | None = None) -> list[Any]:
        self.timeout = timeout
        return self.results


def _raise_timeout(*_args: Any, **_kwargs: Any) -> list[Any]:
    raise TimeoutError("GetTimeoutError: the GPU pool never scaled")


def _sample_frames(n: int) -> list[pd.DataFrame]:
    return [pd.DataFrame({"ts_id": [f"s{i}"], "y": [1.0]}) for i in range(n)]


def test_the_probe_asks_for_a_whole_device_so_it_lands_on_a_node_that_has_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entire fix, in one assertion: ``num_gpus=1``.

    `calibrate_gpu_fraction` runs in the driver section of ``ray_engine.run``, and that driver is
    the Ray job entrypoint — it executes on the **head** node, which carries no accelerator. Calling
    the probe inline therefore measured nothing on every run ever made, and ``"auto"`` quietly
    became whichever constant the no-samples fallback held. Requesting a whole device makes the
    scheduler place the probe on the GPU pool, which is the only place the measurement exists.
    """
    fake = _FakeRay(results=[111, 222, 333])
    monkeypatch.setitem(sys.modules, "ray", fake)
    peaks = ray_io._measure_np_peaks_on_device(_sample_frames(3), _cfg(compute=_compute()))
    assert peaks == [111, 222, 333]
    assert fake.options == {"num_gpus": 1}
    assert fake.calls == 3
    assert fake.timeout == ray_io._CALIBRATION_TIMEOUT_S


def test_a_gpu_pool_that_never_scales_degrades_the_calibration_instead_of_hanging_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe that cannot be scheduled is "not measured", not a stalled driver.

    ``ray.get`` raises on timeout. The caller must land on the nominal fraction — exactly the
    behaviour that existed before the probe was moved — rather than blocking a run that is
    otherwise ready to go.
    """
    fake = _FakeRay(results=[])
    monkeypatch.setattr(fake, "get", _raise_timeout)
    monkeypatch.setitem(sys.modules, "ray", fake)
    cfg = _cfg(compute=_compute(gpu_fraction="auto"))
    assert ray_io._measure_np_peaks_on_device(_sample_frames(2), cfg) == [None, None]
    assert ray_io.calibrate_gpu_fraction(cfg, sample_series=_sample_frames(2)) == (
        ray_io._NOMINAL_AUTO_FRACTION
    )


def test_no_samples_never_reaches_ray_at_all() -> None:
    """Nothing to measure is answered locally — importing Ray to dispatch no tasks is pointless."""
    assert ray_io._measure_np_peaks_on_device([], _cfg(compute=_compute())) == []


def test_calibration_dispatches_the_whole_sample_set_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam `calibrate_gpu_fraction` uses is the on-device batch, not an inline per-series fit.

    A regression here would be invisible to the arithmetic tests above — they all inject
    ``measured_peaks_bytes`` and never exercise the measuring path at all — so the wiring gets its
    own guard. ``gpu_calibration_samples`` still truncates the list before dispatch.
    """
    seen: dict[str, Any] = {}

    def _capture(series: list[pd.DataFrame], _cfg_arg: Any) -> list[int | None]:
        seen["n"] = len(series)
        return [4 * 1024**3]

    monkeypatch.setattr(ray_io, "_measure_np_peaks_on_device", _capture)
    cfg = _cfg(
        compute=_compute(gpu_fraction="auto", gpu_safety_margin=1.3, gpu_calibration_samples=2)
    )
    frac = ray_io.calibrate_gpu_fraction(cfg, sample_series=_sample_frames(5), gpu_type="T4")
    assert seen["n"] == 2  # truncated to gpu_calibration_samples, not all five
    assert frac == pytest.approx(0.325)


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


def test_a_cluster_name_round_trips_back_to_its_run_id() -> None:
    # `ray_reaper` reads the run id back out of a cluster name to decide whether the run that owns
    # the machine has finished. If these two ever stop being inverses the reaper stops recognising
    # its own clusters, so the pairing is pinned here, next to the naming it depends on.
    name = ray_io.cluster_name(_cfg(compute=_compute()), "run-abc-123")
    assert ray_io.run_id_prefix_from_cluster_name(name) == "run-abc-123"


def test_a_long_run_id_comes_back_only_as_a_prefix() -> None:
    # The 63-char clamp means the name cannot hold a long id in full. What comes back is the start
    # of the real one, which is why the reaper matches by prefix and treats an ambiguous match as a
    # reason to leave the cluster running rather than assume equality.
    run_id = "x" * 80
    name = ray_io.cluster_name(_cfg(compute=_compute()), run_id)
    recovered = ray_io.run_id_prefix_from_cluster_name(name)
    assert recovered is not None
    assert run_id.startswith(recovered)
    assert len(recovered) < len(run_id)


def test_an_operator_named_cluster_yields_no_run_id() -> None:
    # A reuse target is not run-derived, so there is no id to recover and nothing for the reaper to
    # judge it against.
    assert ray_io.run_id_prefix_from_cluster_name("my-standing-cluster") is None
    assert ray_io.run_id_prefix_from_cluster_name(ray_io.EPHEMERAL_PREFIX) is None


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


def _fit(
    family: str,
    *,
    model_type: str,
    rss: int | None,
    gpu_bytes: int | None = None,
    cpu_s: float = 1.0,
):
    """One measurement for ``family``, single-threaded, at the given process footprint.

    ``cpu_s`` against a fixed one-second wall clock *is* the effective-core reading, so raising it
    is how a test says "this cell is four cores' worth of work".
    """
    return MeasuredFit(
        ts_id="s1",
        model_type=model_type,
        family=family,
        n_obs=1000,
        wall_s=1.0,
        cpu_s=cpu_s,
        peak_rss_bytes=1024,
        peak_gpu_bytes=gpu_bytes,
        ok=True,
        error=None,
        intraop_threads=1,
        host_cpu_count=8,
        process_rss_bytes=rss,
    )


def test_an_unprofiled_pool_asks_for_the_same_slot_and_packs_it_onto_schedulable_cores() -> None:
    """No measurement still means no *request* changes — but density is the three-way min now.

    The old claim here was that an unprofiled pool reproduced `plan_cluster`'s inline arithmetic
    exactly. What a cell asks Ray for (``task_options``) is untouched by the absence of a profile,
    and that half still holds. Density is where the fleet arithmetic deliberately differs: the CPU
    pool no longer packs a cell onto the core the raylet reports progress on, so it lands one cell
    per node below nameplate. The GPU pool is unmoved here only because the *submit-time nominal*
    fraction is coarse enough that two devices' worth of slots is still the scarcest axis — the
    calibrated fraction is the case where cores take the binding over from devices, and that case
    lives in ``test_resources.py``.
    """
    cfg = _cfg(compute=_compute())
    cpu = ray_io.plan_pool(cfg, [_CPU, "xgboost"], 1000, gpu=False)
    gpu = ray_io.plan_pool(cfg, [_GPU], 1000, gpu=True, gpu_type="T4")

    cpu_unit = UnitShape(cores=catalog.machine_cores(cfg.compute.ray_cpu_machine_type))
    assert cpu.slots_per_unit == fleet.schedulable_cores(cpu_unit)
    assert cpu.binding_axis == "cores"

    devices_alone = cfg.compute.accelerator_count * ray_io.gpu_slots_per_device(
        ray_io._sizing_fraction(cfg)
    )
    gpu_unit = UnitShape(cores=catalog.machine_cores(cfg.compute.ray_gpu_machine_type))
    assert devices_alone < fleet.schedulable_cores(gpu_unit)
    assert gpu.slots_per_unit == devices_alone
    assert gpu.binding_axis == "device"

    assert _scheduling_request(cpu) == {"num_cpus": 1}
    assert _scheduling_request(gpu) == {"num_gpus": ray_io._sizing_fraction(cfg)}


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
    plan = ray_io.plan_pool(
        _cfg(compute=_compute()), [_CPU, "xgboost"], 1000, gpu=False, profile=profile
    )
    assert plan.slot.memory_bytes == 5 * _GIB
    assert plan.family == "statistical+ml"
    assert plan.task_options["memory"] == 5 * _GIB


def test_a_measured_heavy_family_shrinks_the_density_and_widens_the_fleet() -> None:
    """The behaviour change W6 exists for: sizing follows the work, not just the cell count.

    Measured here on cores, which is the axis Ray actually schedules against. A cell that keeps
    four cores busy is four cells' worth of work, and a fleet that has to run the same cell count
    at once has to be wider by the same factor.
    """
    cfg = _cfg(data={"source_table": "s", "series_limit": 200}, compute=_compute(use_gpu=False))
    light = ray_io.plan_cluster(cfg, [_CPU], run_id="r")
    heavy = ray_io.plan_cluster(
        cfg,
        [_CPU],
        run_id="r",
        profile=build_profile(
            [_fit("statistical", model_type=_CPU, rss=None, cpu_s=4.0)],
            memory_margin=1.0,
            time_margin=1.0,
        ),
    )
    assert heavy.cpu_pool.slots_per_unit < light.cpu_pool.slots_per_unit
    assert heavy.cpu_node_count > light.cpu_node_count


def test_a_heavy_memory_footprint_does_not_widen_a_ray_fleet_that_will_not_hear_of_it() -> None:
    """Sizing follows the work, but only along an axis the scheduler is actually told about.

    This assertion used to read the other way: an 8 GiB measured footprint shrank the planned
    density and the cluster was widened to compensate. The fleet that arrived was three times the
    nodes it needed, because the tasks it then ran asked Ray for cores and a GPU fraction and
    nothing else — the memory bound the plan sized around was never communicated to the scheduler,
    so the pool packed itself back to its core density and most of those nodes sat idle. The plan
    now models the scheduler it submits to; the footprint survives as `density_note`.
    """
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
    assert heavy.cpu_pool.slots_per_unit == light.cpu_pool.slots_per_unit
    assert heavy.cpu_node_count == light.cpu_node_count
    assert "is never asked for memory" in (heavy.cpu_pool.density_note or "")


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
    assert _scheduling_request(plan) == {"num_gpus": 0.2}
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


def _cells(chunks: list[pd.DataFrame], models: list[str]) -> list[tuple[str, str]]:
    """Every (ts_id, model) cell the chunks will run, as a list so duplicates are visible.

    Mirrors what `run_group` does with each frame: a tagged chunk runs the cells its tag column
    names; an untagged one runs every model in the runner's list, once per series.
    """
    out: list[tuple[str, str]] = []
    for chunk in chunks:
        if _MODEL_COL in chunk.columns:
            out += [
                (str(a), str(b)) for a, b in chunk[["ts_id", _MODEL_COL]].drop_duplicates().values
            ]
        else:
            out += [(str(ts_id), m) for ts_id in chunk["ts_id"].unique() for m in models]
    return out


def test_chunk_cells_covers_every_cell_exactly_once() -> None:
    # The one contractual property: the chunks a pool gets, run against that pool's models, are
    # the pool's cell set — no cell missed, no cell run twice on two workers.
    models = [_CPU, _GPU]
    chunks = ray_io.chunk_cells(_source(20), _cfg(), models, n_chunks=6)
    cells = _cells(chunks, models)
    assert sorted(cells) == sorted((f"s{i}", m) for i in range(20) for m in models)
    assert len(cells) == len(set(cells))


def test_chunk_cells_shards_by_series_not_by_cell() -> None:
    # The 5.1 change itself: no cross-join, so no model tag and one copy of each row. The frames
    # a chunk carries are a subset of the panel's rows, not a replicate of them.
    src = _source(20)
    chunks = ray_io.chunk_cells(src, _cfg(), [_CPU, _GPU], n_chunks=6)
    assert all(_MODEL_COL not in c.columns for c in chunks)
    assert sum(len(c) for c in chunks) == len(src)  # not len(src) × 2 models


def test_chunk_cells_keeps_a_series_history_together() -> None:
    src = _source(12, rows_each=5)
    chunks = ray_io.chunk_cells(src, _cfg(), [_CPU], n_chunks=4)
    # Each series is whole and in exactly one chunk — a model fit on half a history is not a
    # slower run, it is a wrong number.
    locations: dict[str, int] = {}
    for idx, chunk in enumerate(chunks):
        for ts_id, sub in chunk.groupby("ts_id"):
            assert len(sub) == 5
            assert str(ts_id) not in locations
            locations[str(ts_id)] = idx
    assert len(locations) == 12


def test_chunk_cells_falls_back_to_tagging_when_more_tasks_than_series() -> None:
    # Below one series per task, per-series sharding cannot reach the requested task count — and
    # that count is what keeps the pool's autoscaler fed. So this case cross-joins to cell
    # granularity and tags, buying the finer split at the cost the default path avoids.
    models = [_CPU, _GPU]
    chunks = ray_io.chunk_cells(_source(3), _cfg(), models, n_chunks=6)
    assert all(_MODEL_COL in c.columns for c in chunks)
    # More tasks than there are series, which per-series sharding could not produce at any
    # n_chunks. Not all six: CRC32 modulo leaves some slots empty and empty chunks are dropped,
    # which is why the count is a floor to aim at and never a promise.
    assert len(chunks) > 3
    assert sorted(_cells(chunks, models)) == sorted((f"s{i}", m) for i in range(3) for m in models)


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
