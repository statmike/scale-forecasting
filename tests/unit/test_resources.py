"""Offline tests for the profile→runtime translation (``scale_forecasting.resources``).

Pure arithmetic, no Ray and no GPU — which is the point of the module existing separately from
the engine that consumes it. Profiles here are built through the real `build_profile` from
constructed `MeasuredFit` records rather than by hand-assembling a `FamilyCost`, so
these tests also defend the seam between the two modules: a change to how ``slot_rss_bytes`` or
``slot_cores`` is derived shows up here as a translation change, which is exactly what it is.

The load-bearing properties:

* **Falling back reproduces the pre-profiler behaviour exactly.** No profile, an empty profile,
  and a profile that measured a *different* family must all yield the same plan the engine made
  before any of this existed. Turning the profiler on can add information; it must never remove
  any, and it must never make an unmeasured run worse.
* **Absence is never filled in.** An unmeasured memory axis produces no ``memory`` request, not a
  zero and not a guess. Ray treats ``memory`` as a hard scheduling resource, so a fabricated
  number does not degrade a run, it wedges it.
* **A slot never exceeds its unit.** A task asking for more cores or more RAM than any node has is
  unschedulable, and an unschedulable Ray task hangs rather than fails. The clamp is the
  difference between a slow run and a run that never finishes, and the clamp is recorded.
* **Density is bounded by memory as well as by cores.** The cores-only formula over-packs a
  memory-heavy family onto a node that cannot hold it.
* **Provenance survives.** Every plan can say which axes were measured, which were assumed, and
  what was clamped — the sizing decision has to be auditable after the fact.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from scale_forecasting import resources
from scale_forecasting.engines import ray_io
from scale_forecasting.profiling import ComputeProfile, MeasuredFit, build_profile
from scale_forecasting.resources import (
    ResourceSlot,
    UnitShape,
    machine_memory_bytes,
    merge_slots,
    plan_fleet,
    plan_resources,
    resource_slot,
    slots_per_unit,
    snap_to_legal,
    tasks_for_ceiling,
    translate_serverless,
)

_GIB = 1024**3
_MIB = 1024**2

# An n1-standard-8: 8 cores, 30 GiB nameplate, 21 GiB schedulable.
_N1_STANDARD_8 = UnitShape(cores=8, memory_bytes=30 * _GIB)


# --- helpers -------------------------------------------------------------------


def _fit(
    *,
    model_type: str = "theta",
    family: str = "statistical",
    ts_id: str = "s1",
    wall_s: float = 2.0,
    cpu_s: float = 2.0,
    process_rss_bytes: int | None = 900 * _MIB,
    peak_gpu_bytes: int | None = None,
    ok: bool = True,
) -> MeasuredFit:
    """One measurement, defaulted to a plausible single-threaded statistical fit."""
    return MeasuredFit(
        ts_id=ts_id,
        model_type=model_type,
        family=family,
        n_obs=1460,
        wall_s=wall_s,
        cpu_s=cpu_s,
        peak_rss_bytes=10 * _MIB,
        peak_gpu_bytes=peak_gpu_bytes,
        ok=ok,
        error=None if ok else "boom",
        intraop_threads=1,
        host_cpu_count=8,
        process_rss_bytes=process_rss_bytes,
    )


def _profile(*fits: MeasuredFit, memory_margin: float = 1.0) -> ComputeProfile:
    """A profile over the given measurements; margin 1.0 keeps the arithmetic readable."""
    return build_profile(fits, memory_margin=memory_margin, time_margin=1.0)


# --- machine shapes ------------------------------------------------------------


def test_machine_memory_follows_the_class_not_just_the_core_count() -> None:
    """highmem/highcpu at the same core count differ by ~7x — the whole reason for a table."""
    assert machine_memory_bytes("n1-standard-8") == int(8 * 3.75 * _GIB)
    assert machine_memory_bytes("n1-highmem-8") == int(8 * 6.5 * _GIB)
    assert machine_memory_bytes("n1-highcpu-8") == int(8 * 0.9 * _GIB)
    assert machine_memory_bytes("g2-standard-4") == 4 * 4 * _GIB


def test_an_untabulated_class_assumes_the_smallest_standard_ratio() -> None:
    """Under-counting under-packs the node; over-counting OOMs it. Only one is recoverable."""
    assert machine_memory_bytes("n9-standard-8") == int(8 * 3.75 * _GIB)


def test_an_unparseable_machine_type_reports_unknown_rather_than_empty() -> None:
    """0 means "no memory bound" — a custom type must degrade to cores-only, not to one slot."""
    assert machine_memory_bytes("n1-custom-8-16384") == 0
    assert machine_memory_bytes("some-alias") == 0
    plan = plan_resources(
        _profile(_fit(process_rss_bytes=8 * _GIB)),
        "statistical",
        "ray",
        n_cells=64,
        unit=UnitShape(cores=8, memory_bytes=machine_memory_bytes("n1-custom-8-16384")),
    )
    assert plan.slots_per_unit == 8  # cores-only, exactly as before the memory bound existed


# --- the slot: fallback must reproduce today's behaviour -----------------------


def test_no_profile_yields_exactly_the_hardcoded_slot() -> None:
    """The safety property: enabling the profiler with nothing measured changes nothing."""
    slot = resource_slot(None, "statistical")
    assert slot.cores == 1
    assert slot.memory_bytes is None
    assert slot.gpu_fraction is None
    assert slot.basis == "static"
    assert set(slot.assumed) == {"cores", "memory_bytes"}
    assert slot.measured == ()


def test_a_profile_that_measured_another_family_is_the_same_as_no_profile() -> None:
    """Absence of *this* family must fall back, not borrow a neighbour's numbers."""
    profile = _profile(_fit(model_type="neuralprophet", family="deep_learning"))
    assert resource_slot(profile, "statistical") == resource_slot(None, "statistical")


def test_a_profile_where_every_fit_failed_is_the_same_as_no_profile() -> None:
    """A pre-pass that measured nothing usable must not size anything."""
    profile = _profile(_fit(ok=False), _fit(ok=False, ts_id="s2"))
    assert profile.is_empty
    assert resource_slot(profile, "statistical") == resource_slot(None, "statistical")


def test_the_measured_memory_reaches_the_slot_with_the_margin_already_applied() -> None:
    """``slot_rss_bytes`` carries the margin; the translation must not multiply a second time."""
    profile = _profile(_fit(process_rss_bytes=1000 * _MIB), memory_margin=1.3)
    slot = resource_slot(profile, "statistical", max_memory_bytes=None)
    assert slot.memory_bytes == math.ceil(1000 * _MIB * 1.3)
    assert "memory_bytes" in slot.measured


def test_cores_of_one_is_a_measurement_not_a_fallback() -> None:
    """The pinned probe answering "1" says the hardcoded num_cpus was right — from evidence."""
    slot = resource_slot(_profile(_fit(wall_s=2.0, cpu_s=2.0)), "statistical")
    assert slot.cores == 1
    assert "cores" in slot.measured
    assert "cores" not in slot.assumed
    assert slot.basis == "measured"


def test_a_genuinely_multithreaded_family_widens_the_slot() -> None:
    """The axis is not inert: a fit that really used four cores gets four."""
    slot = resource_slot(_profile(_fit(wall_s=1.0, cpu_s=3.6)), "statistical")
    assert slot.cores == 4


# --- the slot: clamps, because an unschedulable task hangs ---------------------


def test_a_slot_wider_than_the_machine_is_clamped_and_the_clamp_is_recorded() -> None:
    """Ray does not fail an unplaceable task — it waits forever. Clamping beats hanging."""
    slot = resource_slot(_profile(_fit(wall_s=1.0, cpu_s=16.0)), "statistical", max_cores=8)
    assert slot.cores == 8
    assert any("clamped" in note for note in slot.notes)


def test_a_slot_heavier_than_the_machine_is_clamped_to_schedulable_memory() -> None:
    """Same failure mode on the memory axis, and the same answer."""
    plan = plan_resources(
        _profile(_fit(process_rss_bytes=40 * _GIB)),
        "statistical",
        "ray",
        n_cells=10,
        unit=_N1_STANDARD_8,
    )
    assert plan.slot.memory_bytes == int(30 * _GIB * 0.7)
    assert plan.slots_per_unit == 1
    assert any("clamped" in note for note in plan.slot.notes)


# --- the GPU axis --------------------------------------------------------------


def test_a_cpu_family_carries_no_gpu_fraction_at_all() -> None:
    """A fraction on a CPU pool would schedule against a device that is not there."""
    profile = _profile(_fit(peak_gpu_bytes=4 * _GIB))
    assert resource_slot(profile, "statistical", use_gpu=False).gpu_fraction is None


def test_the_measured_device_footprint_sets_the_fraction() -> None:
    """4 GiB measured against a 16 GiB T4 packs four cells per device."""
    profile = _profile(
        _fit(model_type="neuralprophet", family="deep_learning", peak_gpu_bytes=4 * _GIB)
    )
    slot = resource_slot(
        profile, "deep_learning", use_gpu=True, device_bytes=ray_io.device_memory_bytes("T4")
    )
    assert slot.gpu_fraction == 0.25
    assert "gpu_fraction" in slot.measured
    assert slots_per_unit(slot, UnitShape(cores=8, accelerators=1)) == 4


def test_the_same_footprint_packs_more_onto_a_bigger_device() -> None:
    """The L4 under-pack this whole line of work started from: 24 GiB is not 16 GiB."""
    profile = _profile(
        _fit(model_type="neuralprophet", family="deep_learning", peak_gpu_bytes=4 * _GIB)
    )
    t4 = resource_slot(
        profile, "deep_learning", use_gpu=True, device_bytes=ray_io.device_memory_bytes("T4")
    )
    l4 = resource_slot(
        profile, "deep_learning", use_gpu=True, device_bytes=ray_io.device_memory_bytes("L4")
    )
    unit = UnitShape(cores=8, accelerators=1)
    assert slots_per_unit(l4, unit) > slots_per_unit(t4, unit)


def test_an_unmeasured_device_falls_back_to_the_operators_pin_then_the_nominal() -> None:
    """Three sources, in that order — and the fallbacks are labelled as assumptions."""
    pinned = resource_slot(None, "deep_learning", use_gpu=True, static_gpu_fraction=0.25)
    assert pinned.gpu_fraction == 0.25
    assert "gpu_fraction" in pinned.assumed
    nominal = resource_slot(None, "deep_learning", use_gpu=True)
    assert nominal.gpu_fraction == resources._NOMINAL_GPU_FRACTION


def test_the_fraction_is_clamped_into_the_schedulable_band() -> None:
    """A cell measured larger than the whole device asks for the whole device, not more."""
    profile = _profile(
        _fit(model_type="neuralprophet", family="deep_learning", peak_gpu_bytes=40 * _GIB)
    )
    slot = resource_slot(profile, "deep_learning", use_gpu=True, device_bytes=16 * _GIB)
    assert slot.gpu_fraction == 1.0


def test_the_gpu_band_matches_the_engine_it_replaces() -> None:
    """Drift test: two modules cannot each own the clamp band and disagree about it."""
    assert resources._MIN_GPU_FRACTION == ray_io._MIN_FRACTION
    assert resources._NOMINAL_GPU_FRACTION == ray_io._NOMINAL_AUTO_FRACTION


# --- merging: one pool, several families ---------------------------------------


_HOST_AXES = ("cores", "memory_bytes")


def _slot(
    family: str,
    *,
    cores: int = 1,
    memory_bytes: int | None = None,
    gpu_fraction: float | None = None,
    device_bytes: int | None = None,
    measured: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> ResourceSlot:
    """A hand-built slot; ``measured`` names the axes with a basis, the rest are assumed."""
    axes = {"cores", "memory_bytes"} | ({"gpu_fraction"} if gpu_fraction is not None else set())
    return ResourceSlot(
        family=family,
        cores=cores,
        memory_bytes=memory_bytes,
        gpu_fraction=gpu_fraction,
        device_bytes=device_bytes,
        measured=measured,
        assumed=tuple(sorted(axes - set(measured))),
        notes=notes,
    )


def test_a_shared_pool_is_sized_for_its_heaviest_family() -> None:
    """A Ray CPU pool runs statistical and ML cells through one worker; the slot holds both."""
    merged = merge_slots(
        [
            _slot("statistical", cores=1, memory_bytes=1 * _GIB, measured=_HOST_AXES),
            _slot("ml", cores=4, memory_bytes=6 * _GIB, measured=_HOST_AXES),
        ],
        family="statistical+ml",
    )
    assert (merged.cores, merged.memory_bytes) == (4, 6 * _GIB)
    assert merged.family == "statistical+ml"


def test_the_axes_are_maxed_independently_of_each_other() -> None:
    """The winner is per axis, not per family — a wide-but-light cell must not shrink memory."""
    merged = merge_slots(
        [
            _slot("statistical", cores=8, memory_bytes=1 * _GIB, measured=_HOST_AXES),
            _slot("ml", cores=1, memory_bytes=6 * _GIB, measured=_HOST_AXES),
        ],
        family="pool",
    )
    assert (merged.cores, merged.memory_bytes) == (8, 6 * _GIB)


def test_an_axis_is_measured_only_if_the_family_that_won_it_measured_it() -> None:
    """Provenance follows the winning number, not the majority — that is the honest answer."""
    merged = merge_slots(
        [
            _slot("statistical", memory_bytes=6 * _GIB, measured=("memory_bytes",)),
            _slot("ml", memory_bytes=1 * _GIB),  # unmeasured, and loses anyway
        ],
        family="pool",
    )
    assert "memory_bytes" in merged.measured

    flipped = merge_slots(
        [
            _slot("statistical", memory_bytes=1 * _GIB, measured=("memory_bytes",)),
            _slot("ml", memory_bytes=6 * _GIB),  # unmeasured, and it wins
        ],
        family="pool",
    )
    assert "memory_bytes" in flipped.assumed
    assert "memory_bytes" not in flipped.measured


def test_a_family_that_measured_nothing_is_named_in_a_note() -> None:
    """"Measured" would over-claim if a family on the pool contributed no evidence."""
    merged = merge_slots(
        [
            _slot("statistical", cores=2, memory_bytes=6 * _GIB, measured=_HOST_AXES),
            _slot("ml", cores=1),
        ],
        family="pool",
    )
    assert any("ml" in note for note in merged.notes)
    assert not any("statistical" in note for note in merged.notes)


def test_a_pool_where_nothing_was_measured_earns_no_note_and_no_claim() -> None:
    """Uniform ignorance is already visible in ``assumed``; a per-family note adds nothing."""
    merged = merge_slots([_slot("statistical"), _slot("ml")], family="pool")
    assert merged.notes == ()
    assert set(merged.assumed) == {"cores", "memory_bytes"}
    assert merged.measured == ()


def test_merging_unmeasured_slots_reproduces_the_hardcoded_slot() -> None:
    """The no-op guarantee again, at the pool seam: no profile in, today's behaviour out."""
    merged = merge_slots(
        [resource_slot(None, "statistical"), resource_slot(None, "ml")], family="pool"
    )
    assert (merged.cores, merged.memory_bytes, merged.gpu_fraction) == (1, None, None)


def test_a_cpu_pool_carries_no_device_axis_at_all() -> None:
    """An absent GPU axis is neither measured nor assumed — there is no device to size for."""
    merged = merge_slots([_slot("statistical"), _slot("ml")], family="pool")
    assert merged.gpu_fraction is None
    assert "gpu_fraction" not in set(merged.measured) | set(merged.assumed)


def test_a_gpu_pool_keeps_the_largest_fraction_and_its_device() -> None:
    """Two DL families on one card: the pool packs for whichever cell needs the most of it."""
    merged = merge_slots(
        [
            _slot("deep_learning", gpu_fraction=0.25, device_bytes=16 * _GIB,
                  measured=("gpu_fraction",)),
            _slot("foundation", gpu_fraction=0.5, device_bytes=16 * _GIB,
                  measured=("gpu_fraction",)),
        ],
        family="pool",
    )
    assert merged.gpu_fraction == 0.5
    assert merged.device_bytes == 16 * _GIB
    assert slots_per_unit(merged, UnitShape(cores=8, accelerators=1)) == 2


def test_clamp_notes_from_every_contributor_survive_the_merge() -> None:
    """A clamp is why a number is what it is; losing it on merge loses the audit trail."""
    merged = merge_slots(
        [
            _slot("statistical", measured=_HOST_AXES, notes=("cores clamped",)),
            _slot("ml", measured=_HOST_AXES, notes=("memory_bytes clamped",)),
        ],
        family="pool",
    )
    assert set(merged.notes) == {"cores clamped", "memory_bytes clamped"}


def test_a_single_family_pool_merges_to_itself() -> None:
    """Merging must be the identity on one contributor, or the shared path diverges."""
    only = resource_slot(_profile(_fit(process_rss_bytes=2 * _GIB)), "statistical")
    merged = merge_slots([only], family="statistical")
    assert (merged.cores, merged.memory_bytes) == (only.cores, only.memory_bytes)
    assert set(merged.measured) == set(only.measured)


def test_merging_nothing_is_a_caller_bug() -> None:
    """A pool with no families is not a slot with no size — it is a mistake upstream."""
    with pytest.raises(ValueError, match="at least one slot"):
        merge_slots([], family="pool")


# --- density: cores are not the only bound -------------------------------------


def test_a_light_family_packs_one_cell_per_core() -> None:
    """900 MiB x 8 fits inside 21 GiB, so cores bind and the old formula stands."""
    plan = plan_resources(
        _profile(_fit(process_rss_bytes=900 * _MIB)),
        "statistical",
        "ray",
        n_cells=1000,
        unit=_N1_STANDARD_8,
    )
    assert plan.slots_per_unit == 8


def test_a_heavy_family_is_packed_by_memory_not_by_cores() -> None:
    """Eight 4 GiB cells do not run on a 30 GiB node however many cores it has."""
    plan = plan_resources(
        _profile(
            _fit(
                model_type="neuralprophet",
                family="deep_learning",
                process_rss_bytes=4 * _GIB,
            )
        ),
        "deep_learning",
        "ray",
        n_cells=1000,
        unit=_N1_STANDARD_8,
    )
    assert plan.slots_per_unit == math.floor(30 * _GIB * 0.7 / (4 * _GIB))  # 5, not 8
    assert plan.slots_per_unit < _N1_STANDARD_8.cores


def test_an_unmeasured_memory_axis_leaves_density_exactly_where_it_was() -> None:
    """No basis → no bound. The memory rule must not shrink a fleet it knows nothing about."""
    assert slots_per_unit(resource_slot(None, "statistical"), _N1_STANDARD_8) == 8


def test_host_memory_bounds_a_gpu_slot_too_because_ray_enforces_the_request() -> None:
    """4 cards would hold 4 cells; 30 GiB of host RAM at 8 GiB a cell holds 2. Ray honours both."""
    unit = UnitShape(cores=8, memory_bytes=30 * _GIB, accelerators=4)
    slot = _slot("deep_learning", memory_bytes=8 * _GIB, gpu_fraction=1.0, measured=_HOST_AXES)
    assert slots_per_unit(slot, unit) == 2


def test_a_gpu_slot_that_fits_in_host_memory_is_still_bound_by_its_devices() -> None:
    """The memory bound is a ceiling, not a replacement — it must not *raise* device density."""
    unit = UnitShape(cores=8, memory_bytes=30 * _GIB, accelerators=2)
    slot = _slot("deep_learning", memory_bytes=1 * _GIB, gpu_fraction=0.5, measured=_HOST_AXES)
    assert slots_per_unit(slot, unit) == 4  # 2 cards x 2 cells, not the 21 the RAM would allow


def test_an_unmeasured_gpu_slot_packs_by_device_alone_exactly_as_before() -> None:
    """The byte-identity guarantee reaches the GPU branch: no memory measured, no new bound."""
    unit = UnitShape(cores=8, memory_bytes=30 * _GIB, accelerators=2)
    slot = _slot("deep_learning", gpu_fraction=0.25)
    assert slots_per_unit(slot, unit) == 8


# --- the fleet -----------------------------------------------------------------


def test_the_fleet_widens_with_the_load_and_stops_at_the_ceiling() -> None:
    """derived = ceil(cells / (slots x target)), clamped — the engine's own arithmetic."""
    def units(n_cells: int, max_units: int = 100) -> int:
        return plan_resources(
            _profile(_fit()),
            "statistical",
            "ray",
            n_cells=n_cells,
            unit=_N1_STANDARD_8,
            target_cells_per_slot=8,
            max_units=max_units,
        ).derived_units

    assert units(64) == 1  # exactly one node's worth: 8 slots x 8 cells
    assert units(65) == 2
    assert units(6400) == 100
    assert units(64_000, max_units=16) == 16


def test_an_empty_pool_provisions_nothing() -> None:
    """A family with no work must not be floored up to the minimum node count."""
    plan = plan_resources(
        _profile(_fit()), "statistical", "ray", n_cells=0, unit=_N1_STANDARD_8, min_units=2
    )
    assert plan.derived_units == 0
    assert plan.saturating_units == 0
    assert tasks_for_ceiling(plan) == 0


def test_the_record_shows_when_the_ceiling_and_not_the_work_bounded_the_run() -> None:
    """saturating_units is deliberately unclamped so throttling is readable off the record."""
    plan = plan_resources(
        _profile(_fit()),
        "statistical",
        "ray",
        n_cells=8000,
        unit=_N1_STANDARD_8,
        max_units=4,
    )
    assert plan.derived_units == 4
    assert plan.saturating_units == 1000
    assert plan.total_slots == 32


def test_a_run_must_produce_enough_tasks_for_its_autoscaler_to_reach_the_ceiling() -> None:
    """"We enabled autoscaling and nothing scaled" is arithmetic, not a platform problem."""
    plan = plan_resources(
        _profile(_fit()),
        "statistical",
        "ray",
        n_cells=5000,
        unit=_N1_STANDARD_8,
        max_units=10,
    )
    assert plan.slots_at_ceiling == 80
    assert tasks_for_ceiling(plan) == 80  # fewer than 80 chunks and the pool never grows


# --- the handover: what the runtime is actually given --------------------------


def test_a_cpu_plan_hands_ray_its_old_options_plus_the_memory_it_never_had() -> None:
    plan = plan_resources(
        _profile(_fit(process_rss_bytes=1 * _GIB)),
        "statistical",
        "ray",
        n_cells=100,
        unit=_N1_STANDARD_8,
    )
    assert plan.task_options == {"num_cpus": 1, "memory": 1 * _GIB}


def test_an_unmeasured_plan_hands_ray_exactly_what_it_handed_it_before() -> None:
    """The no-op guarantee, stated at the seam the engine actually calls."""
    plan = plan_resources(None, "statistical", "ray", n_cells=100, unit=_N1_STANDARD_8)
    assert plan.task_options == {"num_cpus": 1}


def test_a_gpu_plan_requests_a_fraction_and_lets_ray_default_the_cpu() -> None:
    """Parity with ``ray_io._task_options``: a GPU task states num_gpus and nothing else."""
    profile = _profile(
        _fit(
            model_type="neuralprophet",
            family="deep_learning",
            peak_gpu_bytes=4 * _GIB,
            process_rss_bytes=None,
        )
    )
    plan = plan_resources(
        profile,
        "deep_learning",
        "ray",
        n_cells=100,
        unit=UnitShape(cores=8, memory_bytes=30 * _GIB, accelerators=1),
        use_gpu=True,
        device_bytes=16 * _GIB,
    )
    assert plan.task_options == {"num_gpus": 0.25}


# --- provenance ----------------------------------------------------------------


def test_the_plan_is_json_serializable_for_telemetry() -> None:
    """The sizing decision is stamped into job_telemetry; it must survive json.dumps as-is."""
    plan = plan_resources(
        _profile(_fit()), "statistical", "ray", n_cells=100, unit=_N1_STANDARD_8
    )
    payload: dict[str, Any] = json.loads(json.dumps(plan.to_dict()))
    assert payload["runtime"] == "ray"
    assert payload["slot"]["basis"] == "measured"
    assert set(payload) >= {
        "family",
        "n_cells",
        "slot",
        "unit",
        "slots_per_unit",
        "derived_units",
        "saturating_units",
        "min_units",
        "max_units",
        "target_cells_per_slot",
        "total_slots",
        "slots_at_ceiling",
    }


def test_every_axis_is_labelled_measured_or_assumed_exactly_once() -> None:
    """A reader must never have to guess whether a number came from a fit or from a table."""
    profile = _profile(
        _fit(model_type="neuralprophet", family="deep_learning", peak_gpu_bytes=2 * _GIB)
    )
    slot = resource_slot(profile, "deep_learning", use_gpu=True, device_bytes=16 * _GIB)
    assert set(slot.measured) | set(slot.assumed) == {"cores", "memory_bytes", "gpu_fraction"}
    assert not set(slot.measured) & set(slot.assumed)


def test_a_hand_assembled_slot_bigger_than_its_unit_still_yields_one_slot() -> None:
    """Belt and braces: packing must never return 0 slots and stall a pool."""
    oversized = ResourceSlot(
        family="statistical",
        cores=64,
        memory_bytes=100 * _GIB,
        gpu_fraction=None,
        device_bytes=None,
    )
    assert slots_per_unit(oversized, _N1_STANDARD_8) == 1


# --- serverless: the same measurement, spelled backwards ------------------------


def _serverless(
    *,
    cores: int = 1,
    memory_bytes: int | None = None,
    gpu_fraction: float | None = None,
    min_units: int = 1,
    max_units: int = 10,
    n_cells: int = 1000,
    tier: str = "standard",
    notes: tuple[str, ...] = (),
) -> resources.ServerlessTranslation:
    """Translate a hand-built slot straight through `plan_fleet` into properties."""
    slot = _slot(
        "statistical",
        cores=cores,
        memory_bytes=memory_bytes,
        gpu_fraction=gpu_fraction,
        measured=_HOST_AXES,
        notes=notes,
    )
    plan = plan_fleet(
        slot,
        runtime="serverless",
        n_cells=n_cells,
        unit=UnitShape(cores=8, memory_bytes=30 * _GIB, accelerators=1 if gpu_fraction else 0),
        min_units=min_units,
        max_units=max_units,
    )
    return translate_serverless(plan, tier=tier)


def test_snapping_up_takes_the_smallest_legal_value_that_still_fits() -> None:
    assert snap_to_legal(5, (4, 8, 16), up=True) == 8
    assert snap_to_legal(8, (4, 8, 16), up=True) == 8
    assert snap_to_legal(99, (4, 8, 16), up=True) == 16  # off the top → the largest we may ask for


def test_snapping_down_takes_the_largest_legal_value_that_does_not_overshoot() -> None:
    assert snap_to_legal(15, (4, 8, 16), up=False) == 8
    assert snap_to_legal(1, (4, 8, 16), up=False) == 4  # off the bottom → the smallest that exists


def test_snapping_against_an_empty_table_is_a_caller_bug() -> None:
    with pytest.raises(ValueError, match="at least one legal value"):
        snap_to_legal(4, (), up=True)


def test_a_single_threaded_family_lands_on_the_four_core_default_from_evidence() -> None:
    """The cores axis measuring 1 confirms the default rather than changing it."""
    out = _serverless(cores=1, memory_bytes=2 * _GIB)
    assert out.properties["spark.executor.cores"] == "4"
    assert out.ideal_executor_cores == 1.0


def test_a_multi_core_family_snaps_up_to_a_legal_executor_shape() -> None:
    assert _serverless(cores=5).properties["spark.executor.cores"] == "8"
    assert _serverless(cores=9).properties["spark.executor.cores"] == "16"


def test_the_measured_footprint_lands_in_overhead_not_in_the_jvm_heap() -> None:
    """A PySpark fit is charged to memoryOverhead; sizing executor.memory would OOM the worker."""
    out = _serverless(cores=1, memory_bytes=2 * _GIB)
    assert out.properties["spark.executor.memoryOverhead"] == f"{4 * 2 * 1024}m"  # 4 cores x 2 GiB
    assert out.properties["spark.executor.memory"] == "2048m"  # 4 cores x the JVM floor


def test_a_slot_too_fat_for_the_standard_tier_is_clamped_and_says_so() -> None:
    out = _serverless(cores=1, memory_bytes=10 * _GIB)
    assert any("standard-tier ceiling" in note for note in out.notes)
    assert out.properties["spark.executor.memoryOverhead"] == f"{4 * 7424 - 4 * 512}m"


def test_the_premium_tier_widens_the_band_instead_of_clamping() -> None:
    out = _serverless(cores=1, memory_bytes=10 * _GIB, tier="premium")
    assert not any("ceiling" in note for note in out.notes)
    assert out.properties["spark.executor.memoryOverhead"] == f"{4 * 10 * 1024}m"


def test_a_tiny_slot_is_raised_to_the_platform_memory_floor() -> None:
    """Below 1024m per core the pair is illegal, so the ask is snapped up, not down."""
    out = _serverless(cores=1, memory_bytes=64 * _MIB)
    assert out.properties["spark.executor.memoryOverhead"] == f"{4 * 1024 - 4 * 512}m"


def test_an_unmeasured_memory_axis_emits_no_memory_properties_at_all() -> None:
    """Absence propagates here too: no measurement, no request, serverless defaults stand."""
    out = _serverless(cores=1, memory_bytes=None)
    assert "spark.executor.memory" not in out.properties
    assert "spark.executor.memoryOverhead" not in out.properties
    assert any("memory unmeasured" in note for note in out.notes)


def test_native_thread_pools_are_pinned_to_one_because_spark_runs_a_task_per_core() -> None:
    """Ray gets this for free; a Spark executor oversubscribes unless we export it."""
    out = _serverless(cores=1, memory_bytes=_GIB)
    assert out.properties["spark.executorEnv.OMP_NUM_THREADS"] == "1"
    assert out.properties["spark.executorEnv.MKL_NUM_THREADS"] == "1"
    assert "spark.task.cpus" not in out.properties  # one task per core is Spark's own default


def test_a_threaded_family_gets_wider_tasks_and_the_memory_budget_follows() -> None:
    """Tasks, the thread pin and the memory budget are one decision — they may not disagree.

    Without ``spark.task.cpus`` Spark would run 8 tasks on the 8-core executor while each one
    spawned 5 BLAS threads: 40 threads on 8 cores, the exact thrash the pin exists to stop.
    """
    out = _serverless(cores=5, memory_bytes=_GIB)
    assert out.properties["spark.executor.cores"] == "8"
    assert out.properties["spark.task.cpus"] == "5"
    assert out.properties["spark.executorEnv.OMP_NUM_THREADS"] == "5"
    # 8 // 5 == 1 concurrent cell, so one cell's memory — not eight.
    assert out.properties["spark.executor.memoryOverhead"] == f"{8 * 1024 - 8 * 512}m"


def test_the_gpu_fraction_is_inverted_into_a_legal_core_count() -> None:
    """A tenth of a card per cell means ten cells fit; 8 cores on 1 device is the legal shape."""
    out = _serverless(cores=1, memory_bytes=_GIB, gpu_fraction=0.1)
    assert out.properties["spark.executor.cores"] == "8"
    assert out.tasks_per_device == 8


def test_a_cell_that_wants_a_whole_card_still_gets_serverless_minimum_of_four() -> None:
    """Serverless has no shape below 4 tasks per L4, so the honest answer is a warning."""
    out = _serverless(cores=1, memory_bytes=_GIB, gpu_fraction=1.0)
    assert out.tasks_per_device == 4
    assert any("device pressure" in note for note in out.notes)


def test_a_gpu_batch_sizes_memory_through_the_service_owned_overhead_ratio() -> None:
    """Overhead is rejected on the GPU path, so executor.memory is the only handle left."""
    out = _serverless(cores=1, memory_bytes=768 * _MIB, gpu_fraction=0.25)
    assert "spark.executor.memoryOverhead" not in out.properties
    assert out.properties["spark.executor.memory"] == f"{math.ceil(4 * 768 / 0.4)}m"


def test_the_gpu_inversion_is_clamped_to_the_per_config_maximum() -> None:
    """Dividing by 0.4 puts an ordinary DL footprint past the cap — raw, the batch is rejected."""
    out = _serverless(cores=1, memory_bytes=2 * _GIB, gpu_fraction=0.25)
    assert out.properties["spark.executor.memory"] == f"{4 * 3346}m"  # not the raw 20480m
    assert any("L4 maximum" in note for note in out.notes)


def test_a_tiny_gpu_slot_is_raised_to_the_memory_floor_too() -> None:
    out = _serverless(cores=1, memory_bytes=8 * _MIB, gpu_fraction=0.25)
    assert out.properties["spark.executor.memory"] == f"{4 * 1024}m"


def test_the_autoscaler_starts_warm_and_fills_the_whole_gap() -> None:
    """initialExecutors at the derived count and ratio 1.0 — the slow-ramp defaults are the bug."""
    out = _serverless(cores=1, memory_bytes=_GIB, n_cells=1000, min_units=1, max_units=50)
    props = out.properties
    assert props["spark.dynamicAllocation.enabled"] == "true"
    assert props["spark.dynamicAllocation.minExecutors"] == "2"  # 1 is below the platform floor
    assert props["spark.dynamicAllocation.initialExecutors"] == "16"  # the derived fleet, warm
    assert props["spark.dynamicAllocation.maxExecutors"] == "50"
    assert props["spark.dynamicAllocation.executorAllocationRatio"] == "1.0"


def test_a_fleet_already_at_its_ceiling_leaves_the_allocation_ratio_alone() -> None:
    """Nothing to ramp into, so there is no gap to fill and no reason to touch the default."""
    out = _serverless(cores=1, memory_bytes=_GIB, min_units=4, max_units=4)
    assert "spark.dynamicAllocation.executorAllocationRatio" not in out.properties
    assert out.properties["spark.dynamicAllocation.initialExecutors"] == "4"


def test_an_absurd_ceiling_is_clamped_to_the_platform_maximum() -> None:
    out = _serverless(cores=1, memory_bytes=_GIB, n_cells=10**7, max_units=5000)
    assert out.properties["spark.dynamicAllocation.maxExecutors"] == "2000"
    assert any("platform max" in note for note in out.notes)


def test_the_slots_own_notes_survive_into_the_translation() -> None:
    out = _serverless(cores=1, memory_bytes=_GIB, notes=("cores clamped to the unit",))
    assert "cores clamped to the unit" in out.notes


def test_the_translation_serializes_for_telemetry() -> None:
    out = _serverless(cores=1, memory_bytes=_GIB)
    assert json.loads(json.dumps(out.to_dict()))["executor_cores"] == 4


# --- plan_serverless: the two-pass wiring ---------------------------------------


def test_the_executor_shape_is_the_one_the_snapped_core_count_buys() -> None:
    unit = resources.serverless_unit(8, gpu=False)
    assert unit.cores == 8
    assert unit.accelerators == 0
    # The standard tier's per-core ceiling is what an 8-core executor can be given at most.
    assert unit.memory_bytes == 8 * resources._SERVERLESS_MAX_MB_PER_CORE["standard"] * _MIB


def test_a_gpu_executor_shape_carries_its_cards_and_the_narrower_memory_band() -> None:
    unit = resources.serverless_unit(24, gpu=True)
    assert (unit.cores, unit.accelerators) == (24, 2)
    assert unit.memory_bytes == 24 * resources._SERVERLESS_L4_MB_PER_CORE * _MIB


def test_planning_with_no_profile_reproduces_the_unmeasured_translation() -> None:
    plan, translation = resources.plan_serverless(None, ["statistical"], 800)
    assert plan.runtime == "serverless"
    assert plan.family == "statistical"
    # Nothing was measured, so nothing about memory is requested — the platform defaults stand.
    assert "spark.executor.memory" not in translation.properties
    assert "memory unmeasured; serverless memory defaults left in place" in translation.notes


def test_the_second_pass_plans_against_the_executor_the_first_pass_chose() -> None:
    # A 5-core family snaps up to an 8-core executor; the fleet is then packed into *that*
    # shape, not into the 4-core one the first pass had to start from.
    profile = build_profile(
        [_fit(family="ml", model_type="xgboost", wall_s=2.0, cpu_s=10.0)] * 5
    )
    plan, translation = resources.plan_serverless(profile, ["ml"], 400)
    assert translation.executor_cores == 8
    assert plan.unit.cores == 8


def test_the_ceiling_saturates_the_work_so_the_fan_out_can_reach_it() -> None:
    # 800 tasks, 4 per executor (a 1-core family on a 4-core executor) → 200 executors is the
    # widest fleet that has anything to do.
    plan, translation = resources.plan_serverless(None, ["statistical"], 800)
    assert plan.slots_per_unit == 4
    assert plan.max_units == 200
    assert translation.properties["spark.dynamicAllocation.maxExecutors"] == "200"
    # ...and the invariant that motivated it: the fan-out is wide enough to get there.
    assert tasks_for_ceiling(plan) <= 800


def test_an_explicit_operator_cap_wins_over_the_saturating_count() -> None:
    _, translation = resources.plan_serverless(None, ["statistical"], 800, max_executors=3)
    assert translation.properties["spark.dynamicAllocation.maxExecutors"] == "3"


def test_a_cap_below_the_platform_floor_is_raised_to_it() -> None:
    _, translation = resources.plan_serverless(None, ["statistical"], 800, max_executors=1)
    props = translation.properties
    assert props["spark.dynamicAllocation.minExecutors"] == "2"
    assert props["spark.dynamicAllocation.maxExecutors"] == "2"


def test_an_empty_batch_provisions_the_floor_and_nothing_more() -> None:
    plan, translation = resources.plan_serverless(None, ["statistical"], 0)
    assert plan.derived_units == 0
    assert translation.properties["spark.dynamicAllocation.initialExecutors"] == "2"


def test_several_families_share_one_executor_shape_sized_for_the_heaviest() -> None:
    profile = build_profile(
        [_fit(wall_s=2.0, cpu_s=2.0)] * 5
        + [_fit(family="ml", model_type="xgboost", wall_s=2.0, cpu_s=12.0)] * 5
    )
    plan, translation = resources.plan_serverless(profile, ["statistical", "ml"], 400)
    assert plan.family == "statistical+ml"
    # One executor has to hold whichever cell arrives, so the 6-core ML family sets the shape.
    assert translation.executor_cores == 8
    assert translation.properties["spark.task.cpus"] == "6"


def test_a_gpu_batch_plans_against_l4_executors() -> None:
    profile = build_profile(
        [
            _fit(
                family="deep_learning",
                model_type="neuralprophet",
                peak_gpu_bytes=4 * _GIB,
            )
        ]
        * 5
    )
    plan, translation = resources.plan_serverless(
        profile, ["deep_learning"], 200, gpu=True, device_bytes=24 * _GIB
    )
    assert plan.unit.accelerators >= 1
    assert translation.tasks_per_device is not None
    assert translation.executor_cores in resources._SERVERLESS_L4_CORES


def test_the_gpu_fleet_is_sized_off_the_packing_the_core_table_grants() -> None:
    # A sixth of a card is 6 cells per device by arithmetic, but the legal core table has no
    # shape that runs 6 tasks on one L4 — it snaps down to 4. Sizing the fleet off the 6 the
    # measurement asked for would under-provision it by a third, since the executors it counted
    # on never materialize.
    profile = build_profile(
        [_fit(family="deep_learning", model_type="neuralprophet", peak_gpu_bytes=4 * _GIB)] * 5
    )
    plan, translation = resources.plan_serverless(
        profile, ["deep_learning"], 240, gpu=True, device_bytes=24 * _GIB
    )
    granted = plan.unit.accelerators * (translation.tasks_per_device or 1)
    assert plan.slots_per_unit == granted
    assert plan.max_units == math.ceil(240 / granted)
    # The invariant the whole exercise is for: the fan-out can reach the ceiling it asks for.
    assert tasks_for_ceiling(plan) <= 240


def test_the_serverless_density_is_the_executors_task_slots_not_its_devices() -> None:
    # floor(1/0.1) = 10 cells per card by arithmetic; a 1-core task on an 8-core executor runs 8.
    slot = _slot("deep_learning", cores=1, gpu_fraction=0.1)
    assert resources.serverless_tasks_per_executor(slot, 8) == 8


def test_a_threaded_family_takes_whole_task_slots_at_a_time() -> None:
    slot = _slot("ml", cores=6)
    assert resources.serverless_tasks_per_executor(slot, 8) == 1
    assert resources.serverless_tasks_per_executor(slot, 16) == 2


def test_an_explicit_density_overrides_what_the_slot_arithmetic_would_derive() -> None:
    slot = _slot("cpu", cores=1)
    unit = resources.UnitShape(cores=16, memory_bytes=64 * _GIB)
    derived = resources.plan_fleet(slot, runtime="serverless", n_cells=64, unit=unit)
    assert derived.slots_per_unit == 16  # what the cores alone would say
    forced = resources.plan_fleet(
        slot, runtime="serverless", n_cells=64, unit=unit, max_units=99, density=4
    )
    assert forced.slots_per_unit == 4
    assert forced.saturating_units == 16  # 64 cells / 4 per unit, not / 16
