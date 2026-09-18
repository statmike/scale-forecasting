"""The numbers no measurement produced — fallbacks, machine shapes, and two Spark facts.

The bottom of this package, and it is here because the layers above it would otherwise
have to import each other sideways. Three groups, each a different kind of "we did not
measure this":

* **Fallbacks** — what a slot is worth when the profile has no basis for it. Every value
  reproduces what the engines hardcoded before the profiler existed, which is the safety
  property: enabling measurement can add information, never make an unmeasured run worse.
* **Machine shapes** — the RAM a GCE machine type implies, because neither Ray nor Vertex
  will tell us a node's size and the type name is the only handle there is.
* **Spark platform facts** — the JVM heap floor and the thread-pin variable names. Shared
  because `serverless` and `cluster` are peers: two Spark translators that must agree about
  the platform, with neither importing the other.

Both fallback directions are chosen, not defaulted. Guessing a node's memory *low*
under-packs it (we pay for capacity we don't use); guessing *high* over-packs it (the run
OOMs). Only one of those is recoverable, and every unknown here leans that way.
"""

from __future__ import annotations

import math
import re

# --- fallbacks: what a slot is worth when nothing measured it ------------------

# Cores per cell when the profile has no basis. One, because that is what every engine
# hardcodes today: falling back must reproduce current behaviour exactly, so enabling the
# profiler can never make an unmeasured run worse than a profiler-less one.
_DEFAULT_SLOT_CORES = 1

# GPU-fraction band, mirrored from ``ray_io``. Below the floor a task barely uses the
# device and packing overhead dominates; above 1.0 is meaningless. Duplicated rather than
# imported because this module must not depend on an engine — a drift test pins the two
# together instead (`tests/unit/test_resources.py`).
#
# The floor is a *default*, not a ceiling on density — `device_floor_fraction` lowers it on a
# node whose cores would have run more. Read that function before changing this number.
_MIN_GPU_FRACTION = 0.1
_NOMINAL_GPU_FRACTION = 0.5

# Cells we want to flow through one slot before widening the fleet, mirroring
# ``compute.ray_target_cells_per_slot``. Only a default for direct callers; every engine
# passes the configured value.
_DEFAULT_TARGET_CELLS_PER_SLOT = 8


def device_floor_fraction(schedulable_cores: int, accelerators: int) -> float:
    """The smallest GPU fraction worth asking for on a node of this shape (pure).

    `_MIN_GPU_FRACTION` unless the node's cores would have run more cells than that floor
    permits, in which case it drops far enough to let them.

    **Why a flat floor was wrong.** ``0.1`` caps a device at ten concurrent cells, and the
    justification for it — below this a task barely uses the card, so packing overhead
    dominates — describes a model that is actually using the accelerator. The deep-learning
    family we ship is not: a NeuralProphet fit peaks at 50–78 KB on a 16 GB card, so the honest
    fraction is about six millionths and *every* auto-calibrated run lands on the floor. The
    fraction stopped being a measurement and became a constant. On an eight-core node that was
    invisible, because seven usable cores bind before ten cells do; on a sixteen-core node with
    the same single card the floor binds at ten and five cores sit idle, and `binding_axis`
    reports ``device`` — sending an operator to change a GPU setting when the real limit is a
    number in this file. It has already cost one config a hand-pinned fraction to work around
    (see the 2026-09-09 note in the validation ledger).

    **What replaces it.** Cores are the density a node can genuinely sustain — one cell needs a
    core to run on whatever the card thinks — so the floor is never allowed to sit above the
    fraction those cores imply. ``ceil`` on the per-device share keeps the device bound at or
    above the core bound rather than one short of it when a node carries several cards. The
    floor only ever moves *down*: a pinned fraction an operator chose is theirs, and a node too
    small to want the relaxation keeps the default it always had.

    **Only the Ray path passes this today, and that is deliberate.** Both Spark translators own a
    second density derivation of their own (`cluster._cluster_density`,
    `serverless._serverless_gpu_cores`), each live-proven on its platform's executor shapes, and
    relaxing a floor underneath arithmetic this has not been measured against would be trading a
    known conservative answer for an unmeasured one. They keep the flat default by not passing an
    argument. `fleet.plan_resources` — the generic, runtime-neutral entry point — does pass it.

    ``accelerators <= 0`` returns the default unchanged — there is no device axis to bound.
    """
    if accelerators <= 0:
        return _MIN_GPU_FRACTION
    per_device = max(1, math.ceil(max(1, schedulable_cores) / accelerators))
    return min(_MIN_GPU_FRACTION, 1.0 / per_device)


# --- machine shapes: memory a GCE machine type implies -------------------------

# GiB of RAM per vCPU, by ``<family>-<class>`` prefix. Concurrency is bounded by memory as
# well as by cores, and neither Ray nor Vertex tells us a node's size — the machine type is
# the only handle we have. Values are the published GCE ratios; the *class* is what
# carries them (a "highmem" node holds roughly seven times the cells of a "highcpu" one at
# the same core count, which is the whole reason this is a table and not a constant).
_MEMORY_PER_CORE_GIB = {
    "n1-standard": 3.75,
    "n1-highmem": 6.5,
    "n1-highcpu": 0.9,
    "n2-standard": 4.0,
    "n2-highmem": 8.0,
    "n2-highcpu": 1.0,
    "n2d-standard": 4.0,
    "n2d-highmem": 8.0,
    "n2d-highcpu": 1.0,
    "e2-standard": 4.0,
    "e2-highmem": 8.0,
    "e2-highcpu": 1.0,
    "c2-standard": 4.0,
    "g2-standard": 4.0,  # L4 machines; the card is bundled into the machine type
}

# Unrecognised machine type → assume the smallest *standard* ratio. Same asymmetry as
# ``ray_io._DEFAULT_DEVICE_MEMORY_BYTES``: guessing low under-packs the node (we pay for
# capacity we don't use), guessing high over-packs it (the run OOMs). Only one of those is
# recoverable.
_DEFAULT_MEMORY_PER_CORE_GIB = 3.75

# Cores assumed for a machine type whose name does not carry a count. Unlike the memory
# axis there is no "unknown" answer available here: every caller divides by this number, and
# a unit with zero cores holds zero slots. Eight is the N1 default and the shape both the Ray
# CPU pool and the Dataproc worker already default to, so the fallback stands in for the
# concrete case it is most likely covering.
_DEFAULT_MACHINE_CORES = 8

# ``<family>-<class>-<cores>``. Both machine-shape readers below match against this one
# pattern so they never disagree about whether a name is legible.
_MACHINE_TYPE_RE = r"^([a-z0-9]+-[a-z]+)-(\d+)$"

# Share of a node's RAM that is actually schedulable. Ray reserves ~30% of available
# memory for the plasma object store by default and subtracts it from the node's ``memory``
# resource, so sizing against the machine's nameplate RAM over-packs by roughly that much.
# Also absorbs the OS and the Ray runtime itself.
_SCHEDULABLE_MEMORY_FRACTION = 0.7

# Share of *schedulable* memory one slot may ask for. The clamp above is an estimate of what the
# scheduler will hand out, and estimating it exactly is not possible from a machine-type name: Ray
# takes its 30% off what the container's OS reports, which is a percent or two below nameplate, so
# a request sized at exactly `_SCHEDULABLE_MEMORY_FRACTION` of nameplate lands just *over* the real
# ceiling and the task is never placed. Live 2026-09-03: a 100k Ray run sat at zero cells for an
# hour with the autoscaler repeating "No available node types can fulfill resource request
# {'CPU': 1.0, 'memory': 22548578304.0}" — 0.7 x n1-standard-8's nameplate, to the byte.
#
# The headroom is not only defensive arithmetic. A slot entitled to a whole node's memory is a slot
# that runs one cell per node with every other core idle, which is a bad plan even where it is a
# legal one. Capping the ask below the node forces the packing arithmetic to stay meaningful.
_MAX_SLOT_MEMORY_FRACTION = 0.85

# Cores held back on every unit, whatever its size. The memory axis has had a schedulable share
# since the packing arithmetic was written; the core axis has been dividing *nameplate* vCPUs,
# which quietly assumes a node runs cells and nothing else. It does not: the raylet, the Spark
# executor's JVM, the log shipper and the OS all want a core, and the cell that gets scheduled onto
# the last one does not fail — it time-slices against the agent that is supposed to be reporting
# its progress. One core is a flat reserve rather than a fraction because the overhead is roughly
# constant per node: a 4-core node loses a quarter and a 96-core node loses one percent, which is
# the right shape for a per-node daemon and the wrong shape for anything proportional.
_RESERVED_CORES_PER_UNIT = 1

_GIB = 1024**3
_MIB = 1024**2


def machine_cores(machine_type: str) -> int:
    """vCPUs a GCE machine type implies — ``n1-standard-8`` → 8 (pure; unparseable → 8).

    The count suffix of a ``<family>-<class>-<cores>`` name — the same shape
    `machine_memory_bytes` parses, so the two axes agree about which names they understand.
    Matching the whole shape rather than just a trailing number is what keeps a custom type
    (``n1-custom-8-16384``, where the trailing number is megabytes) from being read as a
    16384-core machine.
    """
    match = re.match(_MACHINE_TYPE_RE, machine_type)
    return int(match.group(2)) if match else _DEFAULT_MACHINE_CORES


def machine_memory_bytes(machine_type: str) -> int:
    """RAM a GCE machine type implies, in bytes (pure; unknown → the smallest standard ratio).

    Derived as ``cores x GiB-per-vCPU`` from `_MEMORY_PER_CORE_GIB`, keyed on the
    ``<family>-<class>`` prefix — so ``n1-standard-8`` is 30 GiB and ``n1-highmem-8`` is 52
    GiB. Nameplate RAM, not schedulable RAM: `plan_resources` applies
    `_SCHEDULABLE_MEMORY_FRACTION` on top.

    Two different kinds of "we don't know", kept distinct because they warrant different
    answers. A name whose *class* is untabulated but whose shape parses (some future
    ``n4-standard-8``) falls back to the smallest standard ratio — it under-counts memory
    and therefore under-packs the node, the safe direction. A name that does not parse at
    all (``n1-custom-8-16384``, a bare alias) returns **0**, meaning *unknown*: callers
    treat that as no memory bound rather than as a machine with no memory, so an
    unrecognised type degrades to today's cores-only packing instead of to one slot.
    """
    match = re.match(_MACHINE_TYPE_RE, machine_type)
    if match is None:
        return 0
    prefix, cores = match.group(1), int(match.group(2))
    per_core = _MEMORY_PER_CORE_GIB.get(prefix, _DEFAULT_MEMORY_PER_CORE_GIB)
    return int(cores * per_core * _GIB)


# --- Spark platform facts: what both translators have to know ------------------

# JVM heap per core. We do no JVM-side work — every fit runs in the Python worker, which is
# charged to memoryOverhead — so this is deliberately a floor rather than a share: enough for
# the shuffle machinery and the Arrow batches crossing the boundary, and no more.
_SPARK_JVM_MB_PER_CORE = 512

# Native thread-pool caps. Ray sets ``OMP_NUM_THREADS = num_cpus`` per task for free, but only
# that one of the five — the other four fall back to counting the machine's cores, so the cap is
# in force for OpenMP and absent for the libraries beside it. A Spark executor pins nothing at
# all, so N concurrent Python workers each grab the whole executor and the machine thrashes on
# N x cores threads. The profile was measured with these pinned
# to one (`profiling.measure._pinned_intraop_threads`), so exporting them is also what makes the
# measurement describe the environment it is being used to size.
#
# The ``pin_threads=False`` escape hatch on both translators exists for exactly one caller: a
# ``compute.profile.measure="controlled"`` run, which needs `effective_cores` to be real. The pin
# is self-referential — with OMP_NUM_THREADS exported as ``spark.task.cpus``, a fit's measured
# cpu/wall ratio reports the cap back rather than the parallelism the library wanted. Unpinning
# buys that honesty at the price of oversubscription, which is why it is opt-in per run and why
# both translators leave a note on the plan saying the shape is not a real run's shape.
_INTRAOP_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def intraop_env_vars(threads: int, *, include_omp: bool = True) -> dict[str, str]:
    """``{var: str(threads)}`` for every native thread pool a fit might use (pure).

    One definition of the pin, so the two runtimes cannot drift into capping different subsets of
    these libraries and calling the result the same thing.

    ``include_omp=False`` is the Ray caller. Ray sets ``OMP_NUM_THREADS`` itself, from the task's
    assigned cores, and its setter is a no-op when the variable already has a value — so passing
    our own would silently take ownership of a derivation Ray is already doing correctly, and any
    future divergence between the two would resolve in favour of ours without a word. Leaving that
    one variable alone keeps a single owner for it. Spark has no such setter and takes all five.
    """
    return {
        name: str(threads) for name in _INTRAOP_ENV_VARS if include_omp or name != "OMP_NUM_THREADS"
    }
