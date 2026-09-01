"""How big is one cell of this family? — the first of the package's two layers.

Cores, host memory, GPU fraction: three numbers, runtime-neutral by construction. The same
slot describes a cell whether it runs as a Ray task or a Spark partition, and the
translators above turn it into whichever knobs their platform accepts.

**The seam is the point.** Sizing arithmetic that lives inside an engine can only be tested
with that engine running. Everything in this package is pure — no Ray, no Spark, no GPU, no
GCP. An engine hands over a `ComputeProfile` and gets back a decision plus the evidence for
it.

**Absence propagates; it is never filled in silently.** A `ResourceSlot` records which axes
came from measurement (``measured``) and which fell back to a static default (``assumed``),
plus every clamp applied (``notes``). An unmeasured memory axis requests *no* memory —
today's behaviour — rather than requesting a number nobody took. That is the contract
`FamilyCost` keeps, carried one layer out so "we sized this off nothing" survives into
telemetry.

**What measurement actually changed, honestly.** Of the three axes, the interesting result
is that **cores is usually 1, and that is a finding rather than a null one.** The probe pins
every native thread pool to one thread (`profiling.measure._pinned_intraop_threads`), which
is exactly the environment a Ray task runs in — Ray exports ``OMP_NUM_THREADS`` = the task's
``num_cpus``. So the measurement says the hardcoded ``num_cpus=1`` was right, and now says it
from evidence instead of from assumption. The axes that move a run are the other two: a
**memory** request Ray had never been given at all, and a GPU fraction derived from a real
footprint rather than from `ray_io._NOMINAL_AUTO_FRACTION`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .catalog import _DEFAULT_SLOT_CORES, _MIN_GPU_FRACTION, _NOMINAL_GPU_FRACTION

if TYPE_CHECKING:
    from ..profiling.cost import ComputeProfile


@dataclass(frozen=True)
class ResourceSlot:
    """What one cell of one family needs, and how much of that was actually measured (pure).

    Runtime-neutral by construction: three numbers plus their provenance. ``cores`` is
    always present (it has a defensible static default); ``memory_bytes`` and
    ``gpu_fraction`` are ``None`` when there is no basis, and ``None`` means *request
    nothing on this axis* — not zero, and not a guess.

    ``measured`` and ``assumed`` partition the axes that carry a value, so a reader of the
    telemetry can tell a 4 GiB request that came from a fit apart from one that came from a
    table. ``notes`` records every clamp in plain words; a slot that was silently trimmed to
    fit the machine is a sizing decision, and sizing decisions have to be auditable.
    """

    family: str
    cores: int  # >= 1, whole cores per cell
    memory_bytes: int | None  # bytes of host RAM per cell; None == no basis, request none
    gpu_fraction: float | None  # share of one device per cell; None == CPU-only or no basis
    device_bytes: int | None  # the denominator the fraction was computed against
    measured: tuple[str, ...] = ()  # axis names taken from a fit
    assumed: tuple[str, ...] = ()  # axis names that fell back to a static default
    notes: tuple[str, ...] = ()  # clamps applied, most-recent last

    @property
    def basis(self) -> str:
        """``"measured"`` if any axis came from a fit, else ``"static"`` — the headline."""
        return "measured" if self.measured else "static"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for telemetry — plain scalars and lists, no custom encoder."""
        return {
            "family": self.family,
            "basis": self.basis,
            "cores": self.cores,
            "memory_bytes": self.memory_bytes,
            "gpu_fraction": self.gpu_fraction,
            "device_bytes": self.device_bytes,
            "measured": list(self.measured),
            "assumed": list(self.assumed),
            "notes": list(self.notes),
        }


def _clamp_gpu_fraction(fraction: float) -> float:
    """Clamp a GPU fraction to ``[_MIN_GPU_FRACTION, 1.0]`` (pure)."""
    return max(_MIN_GPU_FRACTION, min(1.0, fraction))


def resource_slot(
    profile: ComputeProfile | None,
    family: str,
    *,
    use_gpu: bool = False,
    device_bytes: int | None = None,
    static_gpu_fraction: float | None = None,
    max_cores: int | None = None,
    max_memory_bytes: int | None = None,
) -> ResourceSlot:
    """Size one cell of ``family`` from the profile, falling back per-axis (pure).

    Each axis resolves independently, because they fail independently: a family can have a
    solid wall-time and RSS measurement and no GPU number at all (nothing ran on a device),
    or the reverse. Resolving them as a unit would throw away good evidence because of a
    neighbouring gap.

    * **cores** — ``FamilyCost.slot_cores``, else `_DEFAULT_SLOT_CORES`. Clamped to
      ``max_cores`` when given: a task asking for more cores than any node has is not slow,
      it is *unschedulable*, and Ray will sit on it forever rather than fail. Clamping and
      recording beats hanging.
    * **memory** — ``FamilyCost.slot_rss_bytes`` (the absolute footprint x the profile's
      memory margin), else ``None``. Clamped to ``max_memory_bytes`` for the same
      unschedulable-task reason.
    * **gpu_fraction** — ``None`` unless ``use_gpu``. When measured *and* ``device_bytes``
      is known: ``slot_gpu_bytes / device_bytes``, clamped to the band. Otherwise
      ``static_gpu_fraction`` (the operator's pin), and failing that the nominal.

    ``profile`` may be ``None`` (profiling off, or the pre-pass produced nothing) and the
    family may simply be absent from it; both take every fallback, which reproduces the
    pre-profiler behaviour exactly. That equivalence is the safety property: turning the
    profiler on can add information, never remove it.

    The margin already lives inside the ``slot_*`` properties, so nothing here multiplies
    again. Note that the measured GPU path applies the profile's ``memory_margin`` while the
    legacy `ray_io.calibrate_gpu_fraction` path applies ``compute.gpu_safety_margin``;
    both default to 1.3, so the two agree unless an operator moves one of them.
    """
    cost = profile.for_family(family) if profile is not None else None
    measured: list[str] = []
    assumed: list[str] = []
    notes: list[str] = []

    cores = cost.slot_cores if cost is not None else None
    if cores is None:
        cores = _DEFAULT_SLOT_CORES
        assumed.append("cores")
    else:
        measured.append("cores")
    if max_cores is not None and cores > max_cores:
        notes.append(f"cores {cores} exceeded the unit's {max_cores}; clamped")
        cores = max_cores
    cores = max(1, cores)

    memory_bytes = cost.slot_rss_bytes if cost is not None else None
    if memory_bytes is None:
        assumed.append("memory_bytes")
    else:
        measured.append("memory_bytes")
        if max_memory_bytes is not None and memory_bytes > max_memory_bytes:
            notes.append(
                f"memory {memory_bytes} exceeded the unit's schedulable "
                f"{max_memory_bytes}; clamped"
            )
            memory_bytes = max_memory_bytes

    gpu_fraction = _resolve_gpu_fraction(
        cost,
        use_gpu=use_gpu,
        device_bytes=device_bytes,
        static_gpu_fraction=static_gpu_fraction,
        measured=measured,
        assumed=assumed,
    )

    return ResourceSlot(
        family=family,
        cores=cores,
        memory_bytes=memory_bytes,
        gpu_fraction=gpu_fraction,
        device_bytes=device_bytes if use_gpu else None,
        measured=tuple(measured),
        assumed=tuple(assumed),
        notes=tuple(notes),
    )


def merge_slots(slots: Sequence[ResourceSlot], *, family: str) -> ResourceSlot:
    """Collapse several families' slots into the one slot a shared pool needs (pure).

    A Ray CPU pool runs whatever lands on it — statistical cells and ML cells go through the
    same worker — so its slot has to hold the heaviest of them. Same roll-up rule
    `FamilyCost` applies across a family's models, one level out: **max per axis**,
    because the slot must fit whichever cell arrives, not the average one.

    Provenance is resolved per axis by asking *where the winning number came from*: an axis is
    ``measured`` iff a contributor that measured it supplied the max. That is the honest
    answer — a 4 GiB max taken from a real fit is measured evidence even if a lighter family
    beside it had none. What the lighter family's gap does earn is a **note**, so a reader can
    see that the pool was sized off a subset of the families that will run on it. Without that
    note "measured" would over-claim; with it, both facts are on the record.

    ``family`` is the merged label (``"statistical+ml"``). Raises on an empty sequence: a pool
    with no families is a caller bug, not a slot.
    """
    if not slots:
        raise ValueError("merge_slots needs at least one slot")

    def pick(values: list[tuple[float | None, bool]]) -> tuple[float | None, bool]:
        """The max across contributors, plus whether a *measuring* contributor supplied it."""
        present = [(value, was_measured) for value, was_measured in values if value is not None]
        if not present:
            return None, False
        best = max(value for value, _ in present)
        return best, any(was_measured for value, was_measured in present if value == best)

    def axis_of(slot: ResourceSlot, axis: str) -> tuple[float | None, bool]:
        raw = getattr(slot, axis)
        return (None if raw is None else float(raw)), axis in slot.measured

    cores, cores_measured = pick([axis_of(s, "cores") for s in slots])
    memory, memory_measured = pick([axis_of(s, "memory_bytes") for s in slots])
    fraction, fraction_measured = pick([axis_of(s, "gpu_fraction") for s in slots])

    measured: list[str] = []
    assumed: list[str] = []
    for axis, value, was_measured in (
        ("cores", cores, cores_measured),
        ("memory_bytes", memory, memory_measured),
        ("gpu_fraction", fraction, fraction_measured),
    ):
        if axis == "gpu_fraction" and value is None:
            continue  # a CPU pool has no device axis at all, measured or otherwise
        (measured if was_measured else assumed).append(axis)

    notes = [note for slot in slots for note in slot.notes]
    for axis in ("cores", "memory_bytes", "gpu_fraction"):
        if axis == "gpu_fraction" and fraction is None:
            continue
        blind = sorted(s.family for s in slots if axis not in s.measured)
        if blind and len(blind) < len(slots):
            notes.append(f"{axis} sized without a measurement for {', '.join(blind)}")

    return ResourceSlot(
        family=family,
        cores=max(1, int(cores or _DEFAULT_SLOT_CORES)),
        memory_bytes=None if memory is None else int(memory),
        gpu_fraction=fraction,
        device_bytes=next((s.device_bytes for s in slots if s.device_bytes is not None), None),
        measured=tuple(measured),
        assumed=tuple(assumed),
        notes=tuple(notes),
    )


def _resolve_gpu_fraction(
    cost: Any,
    *,
    use_gpu: bool,
    device_bytes: int | None,
    static_gpu_fraction: float | None,
    measured: list[str],
    assumed: list[str],
) -> float | None:
    """The device share one cell needs: measured, else pinned, else nominal (pure).

    Split out of `resource_slot` because it is the one axis with three sources rather
    than two, and inlining its ladder buried the other two. ``None`` when no GPU is
    provisioned — a CPU-only family must not carry a fraction, or a consumer will schedule
    against a device that isn't there.
    """
    if not use_gpu:
        return None
    slot_gpu = cost.slot_gpu_bytes if cost is not None else None
    if slot_gpu is not None and device_bytes:
        measured.append("gpu_fraction")
        return _clamp_gpu_fraction(slot_gpu / device_bytes)
    assumed.append("gpu_fraction")
    if static_gpu_fraction is not None:
        return _clamp_gpu_fraction(static_gpu_fraction)
    return _NOMINAL_GPU_FRACTION
