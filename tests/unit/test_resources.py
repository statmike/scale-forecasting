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

from scale_forecasting.engines import ray_io
from scale_forecasting.profiling.cost import ComputeProfile, build_profile
from scale_forecasting.profiling.measure import MeasuredFit
from scale_forecasting.resources import audit, catalog, cluster, fleet, serverless
from scale_forecasting.resources.catalog import machine_memory_bytes
from scale_forecasting.resources.fleet import (
    UnitShape,
    plan_fleet,
    plan_resources,
    slots_per_unit,
    tasks_for_ceiling,
)
from scale_forecasting.resources.serverless import snap_to_legal, translate_serverless
from scale_forecasting.resources.slot import ResourceSlot, merge_slots, resource_slot

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
    assert nominal.gpu_fraction == catalog._NOMINAL_GPU_FRACTION


def test_the_fraction_is_clamped_into_the_schedulable_band() -> None:
    """A cell measured larger than the whole device asks for the whole device, not more."""
    profile = _profile(
        _fit(model_type="neuralprophet", family="deep_learning", peak_gpu_bytes=40 * _GIB)
    )
    slot = resource_slot(profile, "deep_learning", use_gpu=True, device_bytes=16 * _GIB)
    assert slot.gpu_fraction == 1.0


def test_the_gpu_band_matches_the_engine_it_replaces() -> None:
    """Drift test: two modules cannot each own the clamp band and disagree about it."""
    assert catalog._MIN_GPU_FRACTION == ray_io._MIN_FRACTION
    assert catalog._NOMINAL_GPU_FRACTION == ray_io._NOMINAL_AUTO_FRACTION


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
    """ "Measured" would over-claim if a family on the pool contributed no evidence."""
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
            _slot(
                "deep_learning",
                gpu_fraction=0.25,
                device_bytes=16 * _GIB,
                measured=("gpu_fraction",),
            ),
            _slot(
                "foundation", gpu_fraction=0.5, device_bytes=16 * _GIB, measured=("gpu_fraction",)
            ),
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
    """ "We enabled autoscaling and nothing scaled" is arithmetic, not a platform problem."""
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
    plan = plan_resources(_profile(_fit()), "statistical", "ray", n_cells=100, unit=_N1_STANDARD_8)
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
) -> serverless.ServerlessTranslation:
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
    unit = serverless.serverless_unit(8, gpu=False)
    assert unit.cores == 8
    assert unit.accelerators == 0
    # The standard tier's per-core ceiling is what an 8-core executor can be given at most.
    assert unit.memory_bytes == 8 * serverless._SERVERLESS_MAX_MB_PER_CORE["standard"] * _MIB


def test_a_gpu_executor_shape_carries_its_cards_and_the_narrower_memory_band() -> None:
    unit = serverless.serverless_unit(24, gpu=True)
    assert (unit.cores, unit.accelerators) == (24, 2)
    assert unit.memory_bytes == 24 * serverless._SERVERLESS_L4_MB_PER_CORE * _MIB


def test_planning_with_no_profile_reproduces_the_unmeasured_translation() -> None:
    plan, translation = serverless.plan_serverless(None, ["statistical"], 800)
    assert plan.runtime == "serverless"
    assert plan.family == "statistical"
    # Nothing was measured, so nothing about memory is requested — the platform defaults stand.
    assert "spark.executor.memory" not in translation.properties
    assert "memory unmeasured; serverless memory defaults left in place" in translation.notes


def test_the_second_pass_plans_against_the_executor_the_first_pass_chose() -> None:
    # A 5-core family snaps up to an 8-core executor; the fleet is then packed into *that*
    # shape, not into the 4-core one the first pass had to start from.
    profile = build_profile([_fit(family="ml", model_type="xgboost", wall_s=2.0, cpu_s=10.0)] * 5)
    plan, translation = serverless.plan_serverless(profile, ["ml"], 400)
    assert translation.executor_cores == 8
    assert plan.unit.cores == 8


def test_the_ceiling_saturates_the_work_so_the_fan_out_can_reach_it() -> None:
    # 800 tasks, 4 per executor (a 1-core family on a 4-core executor) → 200 executors is the
    # widest fleet that has anything to do.
    plan, translation = serverless.plan_serverless(None, ["statistical"], 800)
    assert plan.slots_per_unit == 4
    assert plan.max_units == 200
    assert translation.properties["spark.dynamicAllocation.maxExecutors"] == "200"
    # ...and the invariant that motivated it: the fan-out is wide enough to get there.
    assert tasks_for_ceiling(plan) <= 800


def test_an_explicit_operator_cap_wins_over_the_saturating_count() -> None:
    _, translation = serverless.plan_serverless(None, ["statistical"], 800, max_executors=3)
    assert translation.properties["spark.dynamicAllocation.maxExecutors"] == "3"


def test_a_cap_below_the_platform_floor_is_raised_to_it() -> None:
    _, translation = serverless.plan_serverless(None, ["statistical"], 800, max_executors=1)
    props = translation.properties
    assert props["spark.dynamicAllocation.minExecutors"] == "2"
    assert props["spark.dynamicAllocation.maxExecutors"] == "2"


def test_an_empty_batch_provisions_the_floor_and_nothing_more() -> None:
    plan, translation = serverless.plan_serverless(None, ["statistical"], 0)
    assert plan.derived_units == 0
    assert translation.properties["spark.dynamicAllocation.initialExecutors"] == "2"


def test_several_families_share_one_executor_shape_sized_for_the_heaviest() -> None:
    profile = build_profile(
        [_fit(wall_s=2.0, cpu_s=2.0)] * 5
        + [_fit(family="ml", model_type="xgboost", wall_s=2.0, cpu_s=12.0)] * 5
    )
    plan, translation = serverless.plan_serverless(profile, ["statistical", "ml"], 400)
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
    plan, translation = serverless.plan_serverless(
        profile, ["deep_learning"], 200, gpu=True, device_bytes=24 * _GIB
    )
    assert plan.unit.accelerators >= 1
    assert translation.tasks_per_device is not None
    assert translation.executor_cores in serverless._SERVERLESS_L4_CORES


def test_the_gpu_fleet_is_sized_off_the_packing_the_core_table_grants() -> None:
    # A sixth of a card is 6 cells per device by arithmetic, but the legal core table has no
    # shape that runs 6 tasks on one L4 — it snaps down to 4. Sizing the fleet off the 6 the
    # measurement asked for would under-provision it by a third, since the executors it counted
    # on never materialize.
    profile = build_profile(
        [_fit(family="deep_learning", model_type="neuralprophet", peak_gpu_bytes=4 * _GIB)] * 5
    )
    plan, translation = serverless.plan_serverless(
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
    assert serverless.spark_tasks_per_executor(slot, 8) == 8


def test_a_threaded_family_takes_whole_task_slots_at_a_time() -> None:
    slot = _slot("ml", cores=6)
    assert serverless.spark_tasks_per_executor(slot, 8) == 1
    assert serverless.spark_tasks_per_executor(slot, 16) == 2


def test_an_explicit_density_overrides_what_the_slot_arithmetic_would_derive() -> None:
    slot = _slot("cpu", cores=1)
    unit = fleet.UnitShape(cores=16, memory_bytes=64 * _GIB)
    derived = fleet.plan_fleet(slot, runtime="serverless", n_cells=64, unit=unit)
    assert derived.slots_per_unit == 16  # what the cores alone would say
    forced = fleet.plan_fleet(
        slot, runtime="serverless", n_cells=64, unit=unit, max_units=99, density=4
    )
    assert forced.slots_per_unit == 4
    assert forced.saturating_units == 16  # 64 cells / 4 per unit, not / 16


# --- dataproc cluster: the worker is the unit, and it is billed whole -----------


def _cluster(
    *,
    cores: int = 1,
    memory_bytes: int | None = None,
    gpu_fraction: float | None = None,
    machine_type: str = "n1-standard-8",
    accelerators: int = 0,
    n_cells: int = 1000,
    max_units: int = 10,
    notes: tuple[str, ...] = (),
) -> cluster.ClusterTranslation:
    """Translate a hand-built slot straight through `plan_fleet` into cluster properties."""
    slot = _slot(
        "statistical",
        cores=cores,
        memory_bytes=memory_bytes,
        gpu_fraction=gpu_fraction,
        measured=_HOST_AXES,
        notes=notes,
    )
    unit = cluster.cluster_unit(machine_type, accelerators=accelerators)
    plan = plan_fleet(
        slot,
        runtime="cluster",
        n_cells=n_cells,
        unit=unit,
        min_units=2,
        max_units=max_units,
        density=cluster._cluster_density(slot, unit)[2],
    )
    return cluster.translate_cluster(plan)


def test_the_worker_is_read_straight_off_its_machine_name() -> None:
    """No legal-value table and no tier band on a cluster — the name is the whole shape."""
    unit = cluster.cluster_unit("n1-standard-8", accelerators=1)
    assert (unit.cores, unit.memory_bytes, unit.accelerators) == (8, 30 * _GIB, 1)


def test_an_unparseable_machine_name_still_yields_a_countable_worker() -> None:
    """Cores have no "unknown" answer — every caller divides by them, and 0 holds no cells."""
    assert catalog.machine_cores("n1-standard-16") == 16
    assert catalog.machine_cores("n1-custom-8-16384") == 8  # not 16384 MiB read as cores
    assert catalog.machine_cores("some-alias") == 8


def test_the_executor_leaves_the_application_master_room_to_land() -> None:
    """A whole-worker executor makes the AM unplaceable and the job hangs in ACCEPTED forever."""
    out = _cluster()
    assert out.executor_cores == 7  # 8 - _CLUSTER_AM_CORES
    assert out.properties["spark.executor.cores"] == "7"
    heap = int(out.properties["spark.executor.memory"].removesuffix("m"))
    overhead = int(out.properties["spark.executor.memoryOverhead"].removesuffix("m"))
    schedulable_mb = int(30 * _GIB * catalog._SCHEDULABLE_MEMORY_FRACTION) // _MIB
    assert heap + overhead == schedulable_mb - cluster._CLUSTER_AM_RESERVE_MB


def test_a_cluster_states_its_memory_even_when_nothing_was_measured() -> None:
    """The one place absence does *not* mean "request nothing".

    Dataproc bakes ``spark.executor.memory`` into the cluster's ``spark-defaults`` at create,
    sized for the default executor shape. Widening the executor at job level and leaving memory
    alone pairs a 7-core executor with a 4-core executor's heap.
    """
    out = _cluster(memory_bytes=None)
    assert out.properties["spark.executor.memory"] == "3584m"  # 7 cores x the JVM floor
    assert out.properties["spark.executor.memoryOverhead"] == "15872m"  # the rest of the worker


def test_an_unparseable_worker_leaves_the_clusters_own_memory_defaults_alone() -> None:
    """Unknown nameplate → no container to carve, so say so rather than emit a guessed split."""
    out = _cluster(machine_type="n1-custom-8-16384")
    assert "spark.executor.memory" not in out.properties
    assert "spark.executor.memoryOverhead" not in out.properties
    assert any("unparseable" in note for note in out.notes)


def test_the_measured_footprint_narrows_the_task_width_and_the_budget_follows() -> None:
    # 15872m of Python pool holds three 4 GiB cells, so tasks widen to 2 cores (7 // 2 = 3).
    out = _cluster(memory_bytes=4 * _GIB)
    assert out.properties["spark.task.cpus"] == "2"
    assert out.tasks_per_executor == 3
    assert out.properties["spark.executor.memoryOverhead"] == f"{3 * 4 * 1024}m"


def test_a_cell_too_fat_for_the_worker_is_clamped_and_says_so() -> None:
    """Unschedulable is worse than tight: run one cell, keep the whole pool, warn."""
    out = _cluster(memory_bytes=20 * _GIB)
    assert out.tasks_per_executor == 1
    assert out.properties["spark.executor.memoryOverhead"] == "15872m"
    assert any("expect host memory pressure" in note for note in out.notes)


def test_the_gpu_fraction_is_what_bounds_how_many_cells_share_the_card() -> None:
    """The cluster's version of the buckets-are-not-tasks bug.

    Nothing else on a cluster limits concurrency — a 7-core executor would run seven cells on
    one T4 whatever fraction was measured, which is the device OOM this exists to prevent.
    """
    out = _cluster(gpu_fraction=0.5, accelerators=1)
    assert out.properties["spark.task.cpus"] == "3"  # narrowest width giving 7 // 3 = 2 tasks
    assert out.tasks_per_executor == 2


def test_a_smaller_fraction_packs_more_cells_onto_the_same_card() -> None:
    out = _cluster(gpu_fraction=0.25, accelerators=1)
    assert out.properties["spark.task.cpus"] == "2"
    assert out.tasks_per_executor == 3


def test_a_cell_that_wants_a_whole_card_runs_alone_on_the_worker() -> None:
    # 4 rather than 7: the narrowest width that already yields one task (7 // 4 == 1). Widening
    # further would idle the same cores and change nothing Spark schedules.
    out = _cluster(gpu_fraction=1.0, accelerators=1)
    assert out.tasks_per_executor == 1
    assert out.properties["spark.task.cpus"] == "4"


def test_a_gpu_cluster_records_why_it_withholds_the_yarn_resource_request() -> None:
    """Dataproc leaves YARN GPU isolation off; the Spark half alone fails every executor."""
    out = _cluster(gpu_fraction=0.5, accelerators=1)
    assert "spark.executor.resource.gpu.amount" not in out.properties
    assert "spark.task.resource.gpu.amount" not in out.properties
    assert any("yarn gpu isolation off" in note for note in out.notes)


def test_the_native_thread_pools_are_pinned_to_the_task_width() -> None:
    """Same rule as the batch path: a task owning N cores may use N threads, no more."""
    single = _cluster()
    wide = _cluster(cores=4)
    for name in catalog._INTRAOP_ENV_VARS:
        assert single.properties[f"spark.executorEnv.{name}"] == "1"
        assert wide.properties[f"spark.executorEnv.{name}"] == "4"
    assert "spark.task.cpus" not in single.properties  # 1 is Spark's own default


def test_the_workers_are_all_asked_for_at_once_because_they_are_already_paid_for() -> None:
    out = _cluster(n_cells=200)
    props = out.properties
    assert props["spark.dynamicAllocation.enabled"] == "true"
    assert props["spark.dynamicAllocation.initialExecutors"] == str(out.worker_count)
    assert props["spark.dynamicAllocation.maxExecutors"] == str(out.worker_count)


def test_a_small_run_keeps_the_two_worker_cluster_it_has_today() -> None:
    """The safety property: a run too small to derive anything gets the pre-profiler cluster."""
    out = _cluster(n_cells=25)
    assert (out.worker_count, out.ideal_workers) == (2, 1)
    assert out.notes == ()


def test_the_worker_count_stops_at_the_spend_ceiling_and_says_what_it_wanted() -> None:
    """A cluster's workers bill create→delete, so the clamp is a spend decision, on the record."""
    out = _cluster(n_cells=1000)  # 7 cells per worker x 8 target = 56 per worker
    assert (out.worker_count, out.ideal_workers) == (10, 18)
    assert any("raise max_workers" in note for note in out.notes)


def test_the_slots_own_notes_survive_into_the_cluster_translation() -> None:
    assert "clamped something" in _cluster(notes=("clamped something",)).notes


def test_the_cluster_translation_serializes_for_telemetry() -> None:
    out = _cluster(gpu_fraction=0.5, accelerators=1)
    record = json.loads(json.dumps(out.to_dict()))
    assert record["executor_cores"] == 7
    assert record["properties"]["spark.task.cpus"] == "3"
    assert record["ideal_workers"] >= record["worker_count"] or record["worker_count"] == 2


# --- plan_dataproc_cluster: the one-pass wiring ---------------------------------


def test_planning_a_cluster_needs_no_second_pass_because_the_machine_is_given() -> None:
    plan, translation = cluster.plan_dataproc_cluster(
        None, ["statistical"], 25, machine_type="n1-standard-8"
    )
    assert plan.runtime == "cluster"
    assert plan.unit.cores == 8
    assert translation.executor_cores == 7
    assert translation.worker_count == 2


def test_planning_with_no_profile_still_shapes_the_executor_to_the_worker() -> None:
    """No measurement anywhere, and the executor/AM split and thread pins are still right."""
    _plan, translation = cluster.plan_dataproc_cluster(
        None, ["statistical", "ml"], 100, machine_type="n1-standard-8"
    )
    assert translation.properties["spark.executor.cores"] == "7"
    assert translation.properties["spark.executor.memory"] == "3584m"
    assert "spark.task.cpus" not in translation.properties


def test_several_families_share_one_worker_shape_sized_for_the_heaviest() -> None:
    profile = build_profile(
        [_fit(wall_s=2.0, cpu_s=2.0)] * 5
        + [_fit(family="ml", model_type="xgboost", wall_s=2.0, cpu_s=12.0)] * 5
    )
    plan, translation = cluster.plan_dataproc_cluster(
        profile, ["statistical", "ml"], 400, machine_type="n1-standard-8"
    )
    assert plan.family == "statistical+ml"
    assert translation.properties["spark.task.cpus"] == "6"  # the 6-core ML family sets it
    assert translation.tasks_per_executor == 1


def test_a_gpu_cluster_plans_against_the_card_its_machine_type_carries() -> None:
    profile = build_profile(
        [_fit(family="deep_learning", model_type="neuralprophet", peak_gpu_bytes=4 * _GIB)] * 5
    )
    plan, translation = cluster.plan_dataproc_cluster(
        profile,
        ["deep_learning"],
        200,
        machine_type="n1-standard-8",
        accelerators=1,
        gpu=True,
        device_bytes=16 * _GIB,
    )
    assert plan.unit.accelerators == 1
    # A quarter of a 16 GiB T4 per cell → 4 cells the card allows, 3 the 7 cores can spell.
    assert translation.tasks_per_executor == 3
    assert plan.slots_per_unit == translation.tasks_per_executor


def test_the_cluster_fleet_is_sized_off_the_density_the_properties_spell() -> None:
    """A fleet sized off a density the job never grants is the bug this whole seam prevents."""
    profile = build_profile(
        [_fit(family="deep_learning", model_type="neuralprophet", peak_gpu_bytes=4 * _GIB)] * 5
    )
    plan, translation = cluster.plan_dataproc_cluster(
        profile,
        ["deep_learning"],
        480,
        machine_type="n1-standard-8",
        accelerators=1,
        gpu=True,
        device_bytes=16 * _GIB,
    )
    assert plan.slots_per_unit == translation.tasks_per_executor
    assert tasks_for_ceiling(plan) <= 480


def test_an_operator_cap_replaces_the_default_spend_ceiling() -> None:
    _plan, tight = cluster.plan_dataproc_cluster(
        None, ["statistical"], 1000, machine_type="n1-standard-8", max_workers=4
    )
    assert tight.worker_count == 4
    _plan, wide = cluster.plan_dataproc_cluster(
        None, ["statistical"], 1000, machine_type="n1-standard-8", max_workers=40
    )
    assert wide.worker_count == wide.ideal_workers == 18
    assert wide.notes == ()


def test_a_cap_below_the_dataproc_floor_is_raised_to_it() -> None:
    """Two workers is Dataproc's own floor for a standard cluster — not ours to undercut."""
    _plan, out = cluster.plan_dataproc_cluster(
        None, ["statistical"], 1000, machine_type="n1-standard-8", max_workers=1
    )
    assert out.worker_count == 2


def test_an_empty_cluster_run_provisions_the_floor_and_nothing_more() -> None:
    _plan, out = cluster.plan_dataproc_cluster(
        None, ["statistical"], 0, machine_type="n1-standard-8"
    )
    assert (out.worker_count, out.ideal_workers) == (2, 0)


# --- the thread pin, and the one run that has to turn it off ---------------------


_INTRAOP_KEYS = frozenset(f"spark.executorEnv.{n}" for n in catalog._INTRAOP_ENV_VARS)


def _intraop(properties: dict[str, str]) -> dict[str, str]:
    """Just the native-thread-cap exports out of a translation's properties."""
    return {k: v for k, v in properties.items() if k in _INTRAOP_KEYS}


def test_both_spark_translators_pin_native_threads_by_default() -> None:
    """Unpinned, N python workers each grab the whole executor and it thrashes."""
    on_serverless = serverless.plan_serverless(None, ["statistical"], 800)[1]
    on_cluster = cluster.plan_dataproc_cluster(
        None, ["statistical"], 800, machine_type="n1-standard-8"
    )[1]
    for translation in (on_serverless, on_cluster):
        exported = _intraop(translation.properties)
        assert exported, "the pin is the default, on both Spark paths"
        assert set(exported.values()) == {translation.properties.get("spark.task.cpus", "1")}


@pytest.mark.parametrize(
    "plan",
    [
        lambda pin: serverless.plan_serverless(None, ["statistical"], 800, pin_threads=pin)[1],
        lambda pin: cluster.plan_dataproc_cluster(
            None, ["statistical"], 800, machine_type="n1-standard-8", pin_threads=pin
        )[1],
    ],
    ids=["serverless", "cluster"],
)
def test_a_controlled_measurement_run_can_unpin_them_and_says_so_on_the_plan(plan: Any) -> None:
    """The pin is self-referential: a pinned fit can only report the pin back as its cores.

    Unpinning is therefore the only way to measure `effective_cores`, and it is also not the
    shape of a real run — so the translation has to carry that warning with it rather than
    leaving a note in a design document.
    """
    unpinned = plan(False)
    assert _intraop(unpinned.properties) == {}
    assert any("effective_cores" in note for note in unpinned.notes)
    # Everything else about the fleet is untouched — only the exports changed.
    pinned = plan(True)
    assert {k: v for k, v in unpinned.properties.items() if k not in _INTRAOP_KEYS} == {
        k: v for k, v in pinned.properties.items() if k not in _INTRAOP_KEYS
    }


# --- the audit record ----------------------------------------------------------


def test_the_sizing_record_carries_the_decision_the_translation_and_the_evidence() -> None:
    """One blob answers all three audit questions, and survives a JSON round-trip."""
    profile = _profile(_fit(), _fit(ts_id="s2"))
    plan, translation = serverless.plan_serverless(profile, ["statistical"], 800)
    record = audit.sizing_telemetry(plan, translation=translation, profile=profile)

    assert record["family"] == "statistical"
    assert record["plans"] == [plan.to_dict()]
    assert record["translation"] == translation.to_dict()
    assert record["profile"] == profile.to_dict()
    # It is written into a native JSON column, so everything in it has to be JSON-safe.
    assert json.loads(json.dumps(record)) == record


def test_the_sizing_record_keeps_the_pools_it_has_and_drops_the_ones_it_does_not() -> None:
    """A Ray run spells two pools; a CPU-only one has no GPU plan and records one, not a null."""
    cpu = serverless.plan_serverless(None, ["statistical"], 800)[0]
    assert len(audit.sizing_telemetry(cpu, None)["plans"]) == 1
    assert len(audit.sizing_telemetry(cpu, cpu)["plans"]) == 2
    # Nothing planned at all (profiling off) still yields a record-shaped dict rather than a crash.
    assert audit.sizing_telemetry()["plans"] == []
    assert audit.sizing_telemetry()["family"] is None


def test_the_record_can_be_filed_under_the_jobs_family_rather_than_a_pools() -> None:
    """A Ray deep-learning job's CPU pool is labelled "cpu" — filing the job there would bury it."""
    cpu = serverless.plan_serverless(None, ["statistical"], 800)[0]
    assert audit.sizing_telemetry(cpu, family="deep_learning")["family"] == "deep_learning"


# --- the legal-value sweep ----------------------------------------------------
#
# The tests above check the interesting *cases*; this checks the *property*, over a grid wide
# enough to include measurements no real fit would produce. An illegal `(cores, memory)` pair is
# not caught by anything we run — it is caught by Dataproc, as an INVALID_ARGUMENT, minutes into a
# run the operator has already walked away from. So the claim under test is total: whatever the
# probe measured, the emitted properties are a shape the service accepts.
#
# Each assertion below names the table it enforces (`serverless._SERVERLESS_*`), so a future
# platform change moves one constant and this sweep re-derives against it rather than going stale.

_SWEEP_CORES = (1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 33, 64)
_SWEEP_MEMORY = (None, 1, 64 * _MIB, 512 * _MIB, _GIB, 4 * _GIB, 10 * _GIB, 96 * _GIB)


def _assert_legal_cpu_properties(out: serverless.ServerlessTranslation, tier: str) -> None:
    props = out.properties
    cores = int(props["spark.executor.cores"])
    assert cores in serverless._SERVERLESS_CPU_CORES
    assert int(props["spark.driver.cores"]) in serverless._SERVERLESS_CPU_CORES
    # spark.task.cpus may never exceed the executor's width — Spark would never schedule a task.
    assert int(props.get("spark.task.cpus", 1)) <= cores
    if "spark.executor.memory" in props:
        total = int(props["spark.executor.memory"].rstrip("m")) + int(
            props["spark.executor.memoryOverhead"].rstrip("m")
        )
        per_core = total / cores
        ceiling = serverless._SERVERLESS_MAX_MB_PER_CORE[tier]
        assert serverless._SERVERLESS_MIN_MB_PER_CORE <= per_core <= ceiling, (
            f"{per_core:.0f}m per core is outside the {tier} band at {cores} cores"
        )
    else:  # absence propagates: no measurement, no request, no half-specified pair
        assert "spark.executor.memoryOverhead" not in props


@pytest.mark.parametrize("tier", ["standard", "premium"])
@pytest.mark.parametrize("cores", _SWEEP_CORES)
@pytest.mark.parametrize("memory_bytes", _SWEEP_MEMORY)
def test_every_cpu_shape_the_probe_can_measure_translates_to_a_legal_batch(
    tier: str, cores: int, memory_bytes: int | None
) -> None:
    _assert_legal_cpu_properties(
        _serverless(cores=cores, memory_bytes=memory_bytes, tier=tier), tier
    )


@pytest.mark.parametrize("cores", _SWEEP_CORES)
@pytest.mark.parametrize("memory_bytes", [m for m in _SWEEP_MEMORY if m])
@pytest.mark.parametrize("gpu_fraction", [0.05, 0.25, 0.5, 1.0])
def test_every_gpu_shape_the_probe_can_measure_translates_to_a_legal_batch(
    cores: int, memory_bytes: int, gpu_fraction: float
) -> None:
    out = _serverless(cores=cores, memory_bytes=memory_bytes, gpu_fraction=gpu_fraction)
    props = out.properties
    granted = int(props["spark.executor.cores"])
    assert granted in serverless._SERVERLESS_L4_CORES
    assert int(props.get("spark.task.cpus", 1)) <= granted
    # The GPU path may not set memoryOverhead at all — the service owns it — and executor.memory
    # is bounded by the extrapolated per-config maximum on both sides.
    assert "spark.executor.memoryOverhead" not in props
    memory_mb = int(props["spark.executor.memory"].rstrip("m"))
    assert granted * serverless._SERVERLESS_MIN_MB_PER_CORE <= memory_mb
    assert memory_mb <= granted * serverless._SERVERLESS_L4_MB_PER_CORE


@pytest.mark.parametrize("max_units", [1, 2, 3, 50, 10_000])
@pytest.mark.parametrize("n_cells", [1, 100, 10_000_000])
def test_every_fleet_size_lands_inside_the_platforms_executor_bounds(
    max_units: int, n_cells: int
) -> None:
    props = _serverless(cores=1, memory_bytes=_GIB, max_units=max_units, n_cells=n_cells).properties
    lo = int(props["spark.dynamicAllocation.minExecutors"])
    initial = int(props["spark.dynamicAllocation.initialExecutors"])
    hi = int(props["spark.dynamicAllocation.maxExecutors"])
    assert serverless._SERVERLESS_MIN_EXECUTORS <= lo <= initial <= hi
    assert hi <= serverless._SERVERLESS_MAX_EXECUTORS
