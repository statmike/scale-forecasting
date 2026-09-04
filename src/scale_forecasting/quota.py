"""Read the meter before asking for the hardware.

`capacity` is what the system does *after* a provision fails. This module is what it does
**before** one is attempted: ask the project what it is actually allowed in the region it is about
to try, compare that against what the profiler sized, and either proceed, lower the ceiling, or
stop with a reason. A `HARD_CEILING` discovered by attempting a create costs ~12 minutes on the Ray
path; discovered here it costs one HTTP GET.

Four parts, in dependency order:

1. **The vocabulary** (`QuotaMetric` and the ``*_metric`` functions under it) — which meter a given
   piece of infra is actually billed to. This is the part that had been wrong.
2. **The reader** (`read_limits`) — live `effectiveLimit` per metric per region from Service Usage.
   It never raises and never blocks a run on its own failure.
3. **The reconciler** (`reconcile`) — pure. Turns a limit and an ask into a verdict and, where the
   ask is too big, a *lowered* set of bounds.
4. **The advisor** (`advise`) — pure. The "your quota is costing you this much wall clock, and a
   raise would buy you this much back" report.

**Why the vocabulary is a module and not a constant.** On 2026-09-04 a Ray GPU pool provisioned
seven T4s in a project documented — in three separate places — as having four. Nothing was broken:
Dataproc's GPUs are billed to Compute Engine's ``nvidia_t4_gpus`` and Ray's to Vertex's
``custom_model_training_nvidia_t4_gpus``, and in ``us-central1`` those are 4 and 12. The same
conflation had also been made for vCPUs (200 vs 2,200). One wrong number in prose is a typo; the
same wrong number reached for three times is a missing abstraction, so the mapping now lives in one
table that both the docs and the preflight read.

Three things learned from the live API that shape the code:

* Limits are returned as **strings**, and an **absent** ``effectiveLimit`` is ambiguous — the proto
  means "unlimited", but ``us-east4`` reads absent for T4 while ``us-east1`` reads ``2``, which is
  plainly "not offered here" rather than "infinite". Neither reading is safe to act on, so absent
  is `QUOTA_UNKNOWN`: reported, never enforced. Only a **literal zero** blocks.
* **Vertex CPU quota is machine-family-specific.** ``custom_model_training_cpus`` covers N1/E2
  only; ``n2-``, ``g2-``, ``a2-``, ``c2-``, ``m1-``, ``n4-`` and ``a3-`` machines each have their
  own metric. Checking the N1 meter for a G2 pool reads a number that has nothing to do with the
  run.
* **Dataproc has no capacity meter.** Every ``dataproc.googleapis.com`` quota is an API request
  rate; a Serverless batch's vCPUs are billed to Compute Engine ``cpus`` in the region. So the
  Serverless path checks a Compute Engine meter, which looks wrong and is right.

**The clamp only ever lowers, and it only ever lowers what it has to.** ``min_units`` is the
profiler's answer to "how much hardware does this load need" and quota is not evidence about load,
so a large allowance never raises a minimum: an eight-node quota does not mean a two-node job wants
eight nodes. Quota is a *ceiling*, and the only case in which it touches the floor is when the
floor is above it — a pool that cannot start is a pool that fails at create.

**Nothing here may enter ``run_id``.** A quota reading is an observation of the world at a moment,
not an input the author chose; folding one into identity would mean the same config resolved to a
different run every time someone's allowance changed, and two runs of genuinely the same work would
stop being comparable. The same rule `BatchInfra` follows for ``SF_SERVERLESS_DEPS``. The reconciler
therefore adjusts a *plan*, never a config, and every function here is either pure or a read.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .errors import get_logger
from .resources.catalog import machine_cores

if TYPE_CHECKING:
    from .engines.ray_io import RayClusterPlan

_log = get_logger(__name__)


# --- verdicts -----------------------------------------------------------------------------------

# The ask fits inside the allowance; nothing to do.
QUOTA_OK = "OK"
# The allowance is below the ceiling asked for. The run proceeds against a lowered ceiling, and the
# gap between what it wanted and what it got is exactly what `advise` reports.
QUOTA_CLAMPED = "CLAMPED"
# The allowance cannot host even the minimum. No create is worth attempting here — this is a
# `capacity.HARD_CEILING` established without spending an attempt on it.
QUOTA_BLOCKED = "BLOCKED"
# The meter could not be read, or read back absent. Bounds pass through untouched.
QUOTA_UNKNOWN = "UNKNOWN"

QUOTA_STATUSES: frozenset[str] = frozenset({QUOTA_OK, QUOTA_CLAMPED, QUOTA_BLOCKED, QUOTA_UNKNOWN})


# --- the vocabulary -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QuotaMetric:
    """One meter: the service that owns it, its id, and what it counts.

    ``scope`` is the part that decides whether a shortfall may be *acted on* or only *reported*:

    * ``"pool"`` — the metric belongs to exactly one worker pool (a device count). Clamping it has
      an unambiguous meaning: make that pool smaller.
    * ``"shared"`` — several pools and the head node bill to it (vCPUs). Dividing a shortfall
      between them needs a policy nobody has asked for, and in practice it never binds: a Ray run
      is limited by 12 T4s long before it is limited by 2,200 vCPUs. So a shared metric reports,
      and blocks when even the minimum will not fit, but never silently reshapes a fleet.
    """

    service: str
    metric: str
    label: str
    unit: str  # "devices" | "vcpus"
    scope: str = "pool"  # "pool" | "shared"

    @property
    def console_name(self) -> str:
        """The id as the quota console and ``gcloud services quota`` spell it."""
        return f"{self.service}/{self.metric.split('/')[-1]}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for telemetry."""
        return {
            "service": self.service,
            "metric": self.metric,
            "label": self.label,
            "unit": self.unit,
            "scope": self.scope,
        }


_VERTEX = "aiplatform.googleapis.com"
_COMPUTE = "compute.googleapis.com"

# Short ``gpu_type`` (the config's spelling) → the metric *suffix* on each service. The two services
# name the same card differently only by prefix, but they are genuinely different allowances, which
# is the entire point of keeping both columns in one table where they can be seen together.
_GPU_METRIC_SUFFIX: dict[str, str] = {
    "T4": "nvidia_t4_gpus",
    "L4": "nvidia_l4_gpus",
    "V100": "nvidia_v100_gpus",
    "P100": "nvidia_p100_gpus",
    "P4": "nvidia_p4_gpus",
    "A100": "nvidia_a100_gpus",
    "H100": "nvidia_h100_gpus",
}

# GCE machine-type prefix → the Vertex training-CPU metric it bills to.
# ``custom_model_training_cpus`` is documented as "CPUs for N1/E2 machine types" — it is *not* the
# general Vertex vCPU meter, and reading it for a G2 pool reports a number about machines the run
# is not using. Anything untabulated falls back to the N1/E2 meter, which is what every default in
# this product uses.
_VERTEX_CPU_METRIC_BY_FAMILY: dict[str, str] = {
    "n1": "custom_model_training_cpus",
    "e2": "custom_model_training_cpus",
    "n2": "custom_model_training_n2_cpus",
    "n4": "custom_model_training_n4_cpus",
    "c2": "custom_model_training_c2_cpus",
    "g2": "custom_model_training_g2_cpus",
    "a2": "custom_model_training_a2_cpus",
    "a3": "custom_model_training_a3_cpus",
    "m1": "custom_model_training_m1_cpus",
}
_DEFAULT_VERTEX_CPU_METRIC = "custom_model_training_cpus"

_FAMILY_RE = re.compile(r"^([a-z0-9]+)-")


def machine_family(machine_type: str) -> str:
    """The GCE family prefix of a machine type — ``n1-standard-8`` → ``n1`` (pure).

    Returns ``""`` for a name with no prefix, which routes to the default metric rather than to a
    lookup miss.
    """
    match = _FAMILY_RE.match(machine_type)
    return match.group(1) if match else ""


def vertex_gpu_metric(gpu_type: str) -> QuotaMetric | None:
    """The Vertex training meter for an accelerator, or ``None`` for one not tabulated (pure)."""
    suffix = _GPU_METRIC_SUFFIX.get(gpu_type.upper())
    if suffix is None:
        return None
    return QuotaMetric(
        service=_VERTEX,
        metric=f"{_VERTEX}/custom_model_training_{suffix}",
        label=f"Ray on Vertex — {gpu_type.upper()} devices",
        unit="devices",
        scope="pool",
    )


def compute_gpu_metric(gpu_type: str) -> QuotaMetric | None:
    """The Compute Engine meter for an accelerator, or ``None`` for one not tabulated (pure).

    What a **Dataproc** cluster's GPUs are billed to. Deliberately a different function from
    `vertex_gpu_metric` rather than a flag on one: the two are separate allowances with separate
    numbers, and a signature that made them a parameter is a signature that invites passing the
    wrong one.
    """
    suffix = _GPU_METRIC_SUFFIX.get(gpu_type.upper())
    if suffix is None:
        return None
    return QuotaMetric(
        service=_COMPUTE,
        metric=f"{_COMPUTE}/{suffix}",
        label=f"Dataproc / Compute Engine — {gpu_type.upper()} devices",
        unit="devices",
        scope="pool",
    )


def vertex_cpu_metric(machine_type: str) -> QuotaMetric:
    """The Vertex training-CPU meter for a machine type (pure; untabulated family → N1/E2)."""
    family = machine_family(machine_type)
    suffix = _VERTEX_CPU_METRIC_BY_FAMILY.get(family, _DEFAULT_VERTEX_CPU_METRIC)
    return QuotaMetric(
        service=_VERTEX,
        metric=f"{_VERTEX}/{suffix}",
        label=f"Ray on Vertex — vCPUs ({family or 'default'} machines)",
        unit="vcpus",
        scope="shared",
    )


def compute_cpu_metric() -> QuotaMetric:
    """The Compute Engine regional vCPU meter.

    Billed by Dataproc clusters *and* by Dataproc Serverless batches — Serverless has no meter of
    its own (every ``dataproc.googleapis.com`` quota is an API request rate), so a batch's vCPUs
    land here.
    """
    return QuotaMetric(
        service=_COMPUTE,
        metric=f"{_COMPUTE}/cpus",
        label="Compute Engine — regional vCPUs (Dataproc clusters and Serverless batches)",
        unit="vcpus",
        scope="shared",
    )


# Named so the omission is deliberate rather than forgotten, mirroring
# `capacity.UNMANAGED_SERVICES`.
# BigQuery capacity is slots, resolved BigQuery-side by reservations or on-demand queueing; there is
# no regional allowance a create could exceed, so there is nothing to preflight.
UNMETERED_SERVICES: frozenset[str] = frozenset({"bigquery"})


# --- the ask ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QuotaDemand:
    """What one pool wants from one meter in one region (pure).

    ``per_unit`` is the metric's units consumed by a single node — accelerators per node for a
    device metric, vCPUs per node for a CPU metric. ``fixed`` is the units-independent share: the
    head node's vCPUs, which are spent whether or not a worker ever starts.

    ``min_units`` / ``max_units`` are the pool's autoscaling bounds *as the profiler sized them*,
    before any quota is consulted. The reconciler returns adjusted copies; these stay as the record
    of what was wanted.
    """

    metric: QuotaMetric
    region: str
    pool: str  # "cpu" | "gpu" — which worker pool this describes
    per_unit: int
    min_units: int
    max_units: int
    fixed: int = 0

    def amount(self, units: int) -> int:
        """Metric units consumed by a pool of ``units`` nodes (pure)."""
        return self.fixed + self.per_unit * max(0, units)

    @property
    def fixed_only(self) -> bool:
        """True when the ask does not scale with a node count.

        The shape a whole-cluster vCPU total takes: a Ray cluster's head node, CPU pool and GPU pool
        all bill to the same Vertex meter at different rates, so the total is a single number rather
        than a per-unit rate, and there is no ``units`` for which asking "how many fit" means
        anything. `reconcile` answers such a demand with a straight does-it-fit.
        """
        return self.per_unit <= 0

    def units_within(self, limit: int) -> int:
        """The largest node count whose consumption fits inside ``limit`` (pure, floored at 0).

        Meaningless for a `fixed_only` demand, which `reconcile` handles before reaching here.
        """
        if self.fixed_only:
            return self.max_units
        return max(0, (limit - self.fixed) // self.per_unit)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for telemetry."""
        return {
            "metric": self.metric.to_dict(),
            "region": self.region,
            "pool": self.pool,
            "per_unit": self.per_unit,
            "min_units": self.min_units,
            "max_units": self.max_units,
            "fixed": self.fixed,
            "amount_at_min": self.amount(self.min_units),
            "amount_at_max": self.amount(self.max_units),
        }


# --- the reading --------------------------------------------------------------------------------


@dataclass(frozen=True)
class QuotaReading:
    """What the meter said, and how confidently (pure data).

    ``limit`` is ``None`` for every kind of "we do not know": the API was unreachable, the caller
    lacks ``serviceusage.services.get``, the metric has no bucket for this region, or the bucket
    came back with an absent ``effectiveLimit``. They are one value because they warrant one
    behaviour — report it, change nothing — and distinguishing them in the type would invite a
    caller to act on a distinction that does not survive contact with the API.

    ``detail`` keeps them apart for a human, which is where the difference actually matters.
    """

    metric: QuotaMetric
    region: str
    limit: int | None
    detail: str

    @property
    def known(self) -> bool:
        """True when a number was actually read."""
        return self.limit is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for telemetry."""
        return {
            "metric": self.metric.metric,
            "region": self.region,
            "limit": self.limit,
            "detail": self.detail,
        }


# Service Usage returns int64 as a JSON string, and ``-1`` is the documented "unlimited".
_UNLIMITED = -1


def parse_buckets(payload: dict[str, Any], metric: QuotaMetric, region: str) -> QuotaReading:
    """Pull one region's ``effectiveLimit`` out of a ``consumerQuotaMetrics`` GET (pure).

    The response holds one or more ``consumerQuotaLimits``, each with a ``quotaBuckets`` list whose
    entries carry a ``dimensions`` map. The regional bucket is the one whose ``dimensions.region``
    matches; the bucket with no dimensions is the project-wide default and is used only as a
    fallback when no regional bucket exists.

    Every not-found path returns an unknown reading rather than raising — a preflight that can fail
    is a preflight that has to be wrapped in a try by every caller, and one of them will forget.
    """
    regional: Any = None
    default: Any = None
    for limit in payload.get("consumerQuotaLimits") or []:
        for bucket in limit.get("quotaBuckets") or []:
            dimensions = bucket.get("dimensions") or {}
            if dimensions.get("region") == region:
                regional = bucket
            elif not dimensions:
                default = bucket

    bucket = regional if regional is not None else default
    if bucket is None:
        return QuotaReading(metric, region, None, f"no quota bucket for {region}")

    raw = bucket.get("effectiveLimit")
    if raw is None:
        # Ambiguous by construction: the proto means "unlimited", but a region that simply does not
        # offer the accelerator reads exactly the same way (us-east4 for T4, while us-east1 reads
        # 2). Acting on either reading can be wrong, so neither is acted on.
        return QuotaReading(metric, region, None, f"{region} reports no explicit limit")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return QuotaReading(metric, region, None, f"unparseable limit {raw!r}")
    if value == _UNLIMITED:
        return QuotaReading(metric, region, None, f"{region} is unlimited")
    source = "regional" if regional is not None else "project default"
    return QuotaReading(metric, region, value, f"{source} limit {value}")


def read_limits(
    project_id: str,
    metrics: list[QuotaMetric],
    regions: list[str],
    *,
    fetch: Any = None,
    timeout: float = 20.0,
) -> dict[tuple[str, str], QuotaReading]:
    """Read every ``(metric, region)`` pair, keyed by ``(metric.metric, region)``.

    One HTTP GET per *metric* — the response already carries every region's bucket, so asking per
    region would multiply the calls for no extra information.

    ``fetch(service, metric) -> dict`` is injected by tests; the default builds an
    ``AuthorizedSession`` from ADC, the same way `identity` does. **Nothing here raises.** A
    preflight is a diagnostic: when it cannot run, the run proceeds exactly as it did before this
    module existed, which is also what makes it safe to deploy under a service account that has not
    yet been granted ``serviceusage.services.get``.
    """
    reader = fetch if fetch is not None else _default_fetch(project_id, timeout)
    readings: dict[tuple[str, str], QuotaReading] = {}
    for metric in metrics:
        try:
            payload = reader(metric.service, metric.metric)
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never sink the run
            _log.debug("quota: could not read %s (%r)", metric.metric, exc)
            detail = f"quota API unavailable ({type(exc).__name__})"
            for region in regions:
                readings[(metric.metric, region)] = QuotaReading(metric, region, None, detail)
            continue
        for region in regions:
            readings[(metric.metric, region)] = parse_buckets(payload, metric, region)
    return readings


_SERVICE_USAGE = "https://serviceusage.googleapis.com/v1beta1"


def _default_fetch(project_id: str, timeout: float) -> Any:  # pragma: no cover - live HTTP
    """An ADC-authed reader for the Service Usage v1beta1 ``consumerQuotaMetrics`` GET.

    v1beta1 and not v1 because the bucket-level ``effectiveLimit`` — the only field that reflects an
    approved quota *increase* rather than the shipped default — exists only there.
    """
    import urllib.parse

    import google.auth
    import google.auth.transport.requests as gtr

    credentials, _ = google.auth.default()
    session = gtr.AuthorizedSession(credentials)

    def fetch(service: str, metric: str) -> dict[str, Any]:
        quoted = urllib.parse.quote(metric, safe="")
        base = f"{_SERVICE_USAGE}/projects/{project_id}/services/{service}"
        url = f"{base}/consumerQuotaMetrics/{quoted}"
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return dict(response.json())

    return fetch


# --- the reconciler -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QuotaOutcome:
    """One demand judged against one reading, with the bounds it leaves behind (pure).

    ``min_units`` / ``max_units`` are the bounds to *use*; ``demand`` still carries what was asked
    for, so the difference between the two is legible without reconstructing it.

    ``autoscale_viable`` is False in the one case that has no autoscaling expression at all: an
    allowance that permits exactly one node. Vertex requires ``min < max`` strictly, so a ceiling of
    one cannot be written as a range and the pool has to be created fixed-size instead. Discovered
    the hard way — a spec with ``min == max`` is rejected outright, and the resulting
    ``InvalidArgument`` was being retried as though the cloud were merely busy.

    ``advisory`` marks the one outcome that is ``QUOTA_OK`` and still worth saying out loud: a
    *shared* meter that will not cover the full fleet. Nothing is reshaped — that is what shared
    scope means — but "this region has 24 vCPUs and your cluster wants 68" is precisely the
    sentence an operator needs, and without this flag `QuotaPreflight.render` would drop it for
    being OK.
    """

    demand: QuotaDemand
    reading: QuotaReading
    status: str
    min_units: int
    max_units: int
    autoscale_viable: bool
    detail: str
    advisory: bool = False

    @property
    def clamped(self) -> bool:
        """True when the usable ceiling came out below the one the profiler asked for."""
        return self.max_units < self.demand.max_units

    @property
    def noteworthy(self) -> bool:
        """True when this outcome has something to tell a reader — anything but a clean pass."""
        return self.status != QUOTA_OK or self.advisory

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict — part of one ``job_telemetry.$.capacity.preflight`` entry."""
        return {
            "status": self.status,
            "advisory": self.advisory,
            "pool": self.demand.pool,
            "region": self.demand.region,
            "metric": self.demand.metric.console_name,
            "limit": self.reading.limit,
            "asked_min_units": self.demand.min_units,
            "asked_max_units": self.demand.max_units,
            "min_units": self.min_units,
            "max_units": self.max_units,
            "autoscale_viable": self.autoscale_viable,
            "detail": self.detail,
        }


def reconcile(demand: QuotaDemand, reading: QuotaReading) -> QuotaOutcome:
    """Judge one ask against one allowance and return the bounds to actually use (pure).

    The rules, and the asymmetry between them is the whole design:

    * **Unknown limit** → `QUOTA_UNKNOWN`, bounds untouched. Never guess.
    * **Zero** → `QUOTA_BLOCKED`. The region cannot host this pool at all.
    * **A shared metric that cannot seat the minimum** → `QUOTA_BLOCKED`. Reported, not reshaped:
      see `QuotaMetric.scope` for why a shared shortfall is not divided up automatically.
    * **Fewer units allowed than the ceiling asks** → `QUOTA_CLAMPED`, ceiling lowered to what fits.
    * **Fewer units allowed than the *floor* asks** → the floor comes down too, because a pool whose
      minimum exceeds quota does not start. This is the only circumstance in which a quota reading
      touches ``min_units``: a *generous* allowance never raises a floor, since quota is evidence
      about permission and ``min_units`` is evidence about load. An eight-node allowance does not
      mean a two-node job wants eight nodes.
    * **Otherwise** → `QUOTA_OK`.

    Finally ``min < max`` is re-established, because lowering a ceiling onto a floor produces
    exactly the malformed spec Vertex rejects. Where the ceiling has room the floor steps back one;
    where the ceiling is one node there is nowhere to step, so ``autoscale_viable`` goes False and
    the caller creates the pool fixed-size.
    """
    asked_min, asked_max = demand.min_units, demand.max_units

    if reading.limit is None:
        return QuotaOutcome(
            demand, reading, QUOTA_UNKNOWN, asked_min, asked_max, True, reading.detail
        )

    if reading.limit <= 0:
        return QuotaOutcome(
            demand,
            reading,
            QUOTA_BLOCKED,
            asked_min,
            asked_max,
            True,
            f"{demand.metric.console_name} is 0 in {demand.region}",
        )

    if demand.fixed_only:
        # A whole-cluster total, not a rate. It either fits or it does not; there is no ceiling to
        # lower, because the number is the sum of three differently-shaped pools.
        if demand.fixed > reading.limit:
            return QuotaOutcome(
                demand,
                reading,
                QUOTA_BLOCKED,
                asked_min,
                asked_max,
                True,
                f"{demand.pool} needs {demand.fixed} {demand.metric.unit} but "
                f"{demand.metric.console_name} allows {reading.limit} in {demand.region}",
            )
        return QuotaOutcome(
            demand,
            reading,
            QUOTA_OK,
            asked_min,
            asked_max,
            True,
            f"{demand.fixed}/{reading.limit} {demand.metric.unit} in {demand.region}",
        )

    allowed = demand.units_within(reading.limit)

    if allowed < 1:
        needed = demand.amount(1)
        return QuotaOutcome(
            demand,
            reading,
            QUOTA_BLOCKED,
            asked_min,
            asked_max,
            True,
            f"one {demand.pool} node needs {needed} {demand.metric.unit} but "
            f"{demand.region} allows {reading.limit}",
        )

    if demand.metric.scope == "shared":
        # Report-only, except for the one case that is not a judgement call: the floor does not fit,
        # so the create fails no matter how the shortfall would have been apportioned.
        if allowed < asked_min:
            return QuotaOutcome(
                demand,
                reading,
                QUOTA_BLOCKED,
                asked_min,
                asked_max,
                True,
                f"{demand.pool} floor of {asked_min} needs {demand.amount(asked_min)} "
                f"{demand.metric.unit} but {demand.region} allows {reading.limit}",
            )
        if allowed < asked_max:
            return QuotaOutcome(
                demand,
                reading,
                QUOTA_OK,
                asked_min,
                asked_max,
                True,
                f"{demand.metric.console_name} ({reading.limit}) is below the "
                f"{demand.amount(asked_max)} a full {demand.pool} pool would use; shared meter, "
                f"not clamped",
                advisory=True,
            )
        return QuotaOutcome(demand, reading, QUOTA_OK, asked_min, asked_max, True, reading.detail)

    if allowed >= asked_max:
        return QuotaOutcome(demand, reading, QUOTA_OK, asked_min, asked_max, True, reading.detail)

    max_units = allowed
    min_units = min(asked_min, max_units)
    autoscale_viable = max_units > 1
    if autoscale_viable and min_units >= max_units:
        min_units = max_units - 1

    detail = (
        f"{demand.metric.console_name} allows {reading.limit} in {demand.region}, so the "
        f"{demand.pool} pool tops out at {max_units} node(s) instead of {asked_max}"
    )
    if not autoscale_viable:
        detail += "; a one-node ceiling cannot be autoscaled, so the pool is fixed-size"
    elif min_units < asked_min:
        detail += f"; floor lowered {asked_min} -> {min_units} to fit"

    return QuotaOutcome(
        demand, reading, QUOTA_CLAMPED, min_units, max_units, autoscale_viable, detail
    )


# --- the advisor --------------------------------------------------------------------------------

# Multiples of the current allowance to price out in the "what would a raise buy" table. Small and
# concrete on purpose: an operator files a quota increase for a number, and 2x/4x are the numbers
# that get approved. A table running to 100x reads as a sales pitch rather than as advice.
_RAISE_MULTIPLES: tuple[int, ...] = (2, 4)

# How far above the current ceiling the "no longer throttled" width may be before it stops being
# advice. A 10,000-cell run at two cells per T4 saturates at 5,000 devices; printing "at 5000
# node(s): ~2m" beside a real allowance of 12 is not a suggestion anyone can act on, it is just a
# reminder that the work is large. Beyond this multiple the saturating width is still reported in
# the throttle line — as a fact about the run — but it is dropped from the table of asks.
_MAX_PROJECTION_MULTIPLE = 8


@dataclass(frozen=True)
class QuotaAdvice:
    """Why a run will take as long as it will, and what a quota increase would change (pure).

    The user-facing half of this module, and the reason it is not merely a guard. A clamp that
    silently succeeds leaves an operator with a run that is four times slower than it needed to be
    and no way to know that quota was the cause — the wall clock looks like the model's fault. This
    turns the throttle into a number, and then into a number someone can put on a quota request.
    """

    pool: str
    region: str
    metric: QuotaMetric
    limit: int | None
    ceiling_units: int
    saturating_units: int
    n_cells: int
    slots_per_unit: int
    seconds_per_cell: float | None
    projections: tuple[tuple[int, float | None], ...] = field(default=())

    @property
    def throttled(self) -> bool:
        """True when the fleet cannot grow wide enough to run every cell at once."""
        return self.saturating_units > self.ceiling_units > 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for telemetry."""
        return {
            "pool": self.pool,
            "region": self.region,
            "metric": self.metric.console_name,
            "limit": self.limit,
            "ceiling_units": self.ceiling_units,
            "saturating_units": self.saturating_units,
            "throttled": self.throttled,
            "projections": [
                {"units": units, "estimated_wall_s": seconds} for units, seconds in self.projections
            ],
        }

    def render(self) -> list[str]:
        """Human-readable lines — what gets logged at plan time and on a clamp."""
        head = f"quota: {self.pool} pool in {self.region} — {self.metric.console_name}"
        limit = "unknown" if self.limit is None else str(self.limit)
        lines = [f"{head} = {limit}, usable nodes = {self.ceiling_units}"]
        if self.throttled:
            lines.append(
                f"  this run would saturate at {self.saturating_units} nodes "
                f"({self.n_cells} cells / {self.slots_per_unit} per node); the ceiling holds it "
                f"to {self.ceiling_units}"
            )
        # A column of "~unknown" is not a projection table, it is three rows of noise. With nothing
        # measured, the ceiling and the throttle ratio above are the whole of what can be said.
        for units, seconds in self.projections:
            if seconds is None:
                continue
            marker = "  now" if units == self.ceiling_units else "  at "
            lines.append(f"{marker} {units:>4} node(s): ~{format_duration(seconds)}")
        return lines


def format_duration(seconds: float) -> str:
    """``4512.0`` → ``1h15m`` (pure). Coarse on purpose — these are estimates, not measurements."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


def estimate_wall_seconds(
    n_cells: int, slots_per_unit: int, units: int, seconds_per_cell: float | None
) -> float | None:
    """Wall clock for ``n_cells`` at a given fleet width (pure; unmeasured → ``None``).

    ``ceil(n_cells / concurrency) x seconds_per_cell``. Deliberately the crudest possible model: it
    ignores stragglers, scheduling gaps and the cluster create, all of which make it an
    *under*-estimate of the absolute time. That is fine, because the number it exists to produce is
    a **ratio** — halve the fleet and the estimate doubles — and every one of those omissions is
    roughly constant across the widths being compared.
    """
    if seconds_per_cell is None or units <= 0 or slots_per_unit <= 0 or n_cells <= 0:
        return None
    concurrency = slots_per_unit * units
    return math.ceil(n_cells / concurrency) * seconds_per_cell


def advise(
    outcome: QuotaOutcome,
    *,
    n_cells: int,
    slots_per_unit: int,
    saturating_units: int,
    seconds_per_cell: float | None,
) -> QuotaAdvice:
    """Price the current ceiling against the alternatives it is being compared to (pure).

    The projection set is the current ceiling, then `_RAISE_MULTIPLES` of it, then — when the run is
    throttled and the width that would un-throttle it is a number anyone could actually be granted
    (`_MAX_PROJECTION_MULTIPLE`) — that width too. It is the one worth putting on a quota request,
    which is why it is singled out; it is also the one that goes absurd fastest on a large fan-out,
    which is why it is bounded.
    """
    ceiling = outcome.max_units
    widths = {ceiling}
    for multiple in _RAISE_MULTIPLES:
        widths.add(ceiling * multiple)
    if ceiling < saturating_units <= ceiling * _MAX_PROJECTION_MULTIPLE:
        widths.add(saturating_units)

    projections = tuple(
        (units, estimate_wall_seconds(n_cells, slots_per_unit, units, seconds_per_cell))
        for units in sorted(widths)
    )
    return QuotaAdvice(
        pool=outcome.demand.pool,
        region=outcome.demand.region,
        metric=outcome.demand.metric,
        limit=outcome.reading.limit,
        ceiling_units=ceiling,
        saturating_units=saturating_units,
        n_cells=n_cells,
        slots_per_unit=slots_per_unit,
        seconds_per_cell=seconds_per_cell,
        projections=projections,
    )


# --- the whole answer for one region --------------------------------------------------------------


@dataclass(frozen=True)
class QuotaPreflight:
    """Every outcome for one region, plus the one-line answer: may this region be attempted?

    ``blocked`` is what turns a preflight into a saved 12 minutes. A region that answers True here
    is recorded as a `capacity.HARD_CEILING` and dropped from the candidate walk **without a create
    attempt being spent on it** — the same conclusion the walk would eventually reach, reached
    before the money.
    """

    region: str
    outcomes: tuple[QuotaOutcome, ...]
    advice: tuple[QuotaAdvice, ...] = field(default=())

    @property
    def blocked(self) -> bool:
        """True when any pool cannot be placed here at all."""
        return any(outcome.status == QUOTA_BLOCKED for outcome in self.outcomes)

    @property
    def clamped(self) -> bool:
        """True when any pool's ceiling was lowered to fit."""
        return any(outcome.status == QUOTA_CLAMPED for outcome in self.outcomes)

    @property
    def block_reason(self) -> str:
        """Every blocking detail joined — the message the `HARD_CEILING` ledger entry carries."""
        return "; ".join(o.detail for o in self.outcomes if o.status == QUOTA_BLOCKED)

    def for_pool(self, pool: str) -> QuotaOutcome | None:
        """The pool-scoped outcome for ``pool``, or ``None`` when no meter covered it.

        Pool-scoped only: a shared vCPU outcome names a pool too, but it never carries adjusted
        bounds (see `QuotaMetric.scope`), so returning one to a caller asking "what should this
        pool's bounds be" would hand back the unclamped numbers as though they had been checked.
        """
        for outcome in self.outcomes:
            if outcome.demand.pool == pool and outcome.demand.metric.scope == "pool":
                return outcome
        return None

    def render(self) -> list[str]:
        """Human-readable lines for the plan output and the launch log."""
        lines: list[str] = []
        for outcome in self.outcomes:
            if outcome.noteworthy:
                lines.append(f"quota [{outcome.status}] {outcome.detail}")
        for advice in self.advice:
            lines.extend(advice.render())
        return lines

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict — one entry under ``job_telemetry.$.capacity.preflight``.

        It rides in `capacity.CapacityLedger` rather than a column of its own because it answers
        the same question the attempt list answers — why this run landed where it landed — and a
        reader should not have to join two places for the two halves of that answer.
        """
        return {
            "region": self.region,
            "blocked": self.blocked,
            "clamped": self.clamped,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "advice": [a.to_dict() for a in self.advice],
        }


# --- Ray on Vertex: assembling the ask, and applying the answer -------------------------------


def gpu_type_from_accelerator(accelerator_type: str) -> str | None:
    """Vertex's accelerator enum back to the config's short name — ``NVIDIA_TESLA_T4`` → ``T4``.

    The plan carries the enum (that is what the SDK wants) while the quota table is keyed on the
    short name (that is what a person types and what the metric id echoes). Deriving one from the
    other here means the preflight needs nothing threaded down from the config.

    Pure. ``None`` for a name whose remainder is not in `_GPU_METRIC_SUFFIX` — an accelerator we
    have no meter for is one the preflight declines to judge, not one it guesses about.
    """
    name = accelerator_type.upper().removeprefix("NVIDIA_").removeprefix("TESLA_")
    return name if name in _GPU_METRIC_SUFFIX else None


def ray_demands(plan: RayClusterPlan, region: str) -> list[QuotaDemand]:
    """The meters a Vertex Ray cluster of this shape will draw on, in one region (pure).

    Three demands at most, and the third is the one the old documentation missed entirely:

    * the **GPU device** meter, pool-scoped, so a shortfall lowers the GPU pool's ceiling;
    * the **CPU pool** against Vertex's training-vCPU meter for its machine family, shared-scope;
    * the **whole cluster's** vCPU total — head node plus both pools at full scale — as a
      `QuotaDemand.fixed_only` ask, because those three bill to the same meter at three different
      rates and no per-node figure describes the sum.

    A pool with no planned nodes contributes nothing: a run with no deep-learning work should not be
    blocked by an accelerator allowance it was never going to spend.

    Note both vCPU demands are Vertex's, not Compute Engine's. A Ray worker is a Vertex training
    node and is billed as one; ``compute.googleapis.com/cpus`` has nothing to say about it.
    """
    demands: list[QuotaDemand] = []

    gpu_type = gpu_type_from_accelerator(plan.accelerator_type)
    if plan.gpu_node_count > 0 and plan.accelerator_count > 0 and gpu_type:
        metric = vertex_gpu_metric(gpu_type)
        if metric is not None:
            demands.append(
                QuotaDemand(
                    metric=metric,
                    region=region,
                    pool="gpu",
                    per_unit=plan.accelerator_count,
                    min_units=plan.gpu_min_nodes,
                    max_units=plan.gpu_max_nodes,
                )
            )

    if plan.cpu_node_count > 0:
        demands.append(
            QuotaDemand(
                metric=vertex_cpu_metric(plan.cpu_machine_type),
                region=region,
                pool="cpu",
                per_unit=machine_cores(plan.cpu_machine_type),
                min_units=plan.cpu_min_nodes,
                max_units=plan.cpu_max_nodes,
            )
        )

    total_vcpus = machine_cores(plan.head_machine_type)
    if plan.cpu_node_count > 0:
        total_vcpus += plan.cpu_max_nodes * machine_cores(plan.cpu_machine_type)
    if plan.gpu_node_count > 0:
        total_vcpus += plan.gpu_max_nodes * machine_cores(plan.gpu_machine_type)
    demands.append(
        QuotaDemand(
            metric=vertex_cpu_metric(plan.head_machine_type),
            region=region,
            pool="cluster",
            per_unit=0,
            min_units=0,
            max_units=0,
            fixed=total_vcpus,
        )
    )
    return demands


def preflight_ray(
    plan: RayClusterPlan,
    regions: list[str],
    project_id: str,
    *,
    fetch: Any = None,
    cpu_seconds_per_cell: float | None = None,
    gpu_seconds_per_cell: float | None = None,
) -> dict[str, QuotaPreflight]:
    """Read every candidate region's meters and judge this cluster shape against each.

    Returns one `QuotaPreflight` per region, in the order given. Regions are judged **all at once**
    rather than lazily per attempt because the reading is one GET per metric for every region at
    once — so knowing that two of three candidates are hopeless costs exactly as much as knowing it
    about one, and the walk can then be handed only the candidates worth spending a create on.

    ``*_seconds_per_cell`` come from the measured profile when there is one; without them the advice
    still reports the ceiling and the throttle ratio, just not a wall clock.
    """
    demands_by_region = {region: ray_demands(plan, region) for region in regions}
    metrics = {d.metric.metric: d.metric for ds in demands_by_region.values() for d in ds}
    readings = read_limits(project_id, list(metrics.values()), regions, fetch=fetch)

    result: dict[str, QuotaPreflight] = {}
    for region, demands in demands_by_region.items():
        outcomes: list[QuotaOutcome] = []
        advice: list[QuotaAdvice] = []
        for demand in demands:
            reading = readings.get(
                (demand.metric.metric, region),
                QuotaReading(demand.metric, region, None, "not read"),
            )
            outcome = reconcile(demand, reading)
            outcomes.append(outcome)
            pool_plan = plan.gpu_pool if demand.pool == "gpu" else plan.cpu_pool
            if demand.metric.scope == "pool" and pool_plan is not None:
                advice.append(
                    advise(
                        outcome,
                        n_cells=pool_plan.n_cells,
                        slots_per_unit=pool_plan.slots_per_unit,
                        saturating_units=pool_plan.saturating_units,
                        seconds_per_cell=(
                            gpu_seconds_per_cell if demand.pool == "gpu" else cpu_seconds_per_cell
                        ),
                    )
                )
        result[region] = QuotaPreflight(
            region=region, outcomes=tuple(outcomes), advice=tuple(advice)
        )
    return result


def apply_to_ray_plan(plan: RayClusterPlan, preflight: QuotaPreflight) -> RayClusterPlan:
    """A copy of ``plan`` with each pool's bounds lowered to what the region will grant (pure).

    Only the *plan* changes. The config is untouched and so is ``run_id``: the same authored config
    still resolves to the same run in a region that grants less, which is what makes two runs of the
    same work comparable across a quota change. The adjustment is recorded in telemetry instead.

    ``autoscale`` is switched off for the whole cluster when a pool comes back with a one-node
    ceiling, because Vertex requires ``min < max`` strictly and a single node has no such range. The
    switch is cluster-wide rather than per-pool only because `RayClusterPlan.autoscale` is one flag;
    the alternative — a fixed pool beside an autoscaling one — is a shape the plan cannot express,
    and inventing it to avoid a rare degenerate case is not worth the surface.

    ``[cpu|gpu]_node_count`` is lowered alongside the ceiling. It is the fixed-size-equivalent, so
    leaving it above a ceiling that was just lowered would mean the fixed path provisions a fleet
    the region has already said no to.
    """
    changes: dict[str, Any] = {}
    autoscale = plan.autoscale

    for pool, min_field, max_field, count_field in (
        ("cpu", "cpu_min_nodes", "cpu_max_nodes", "cpu_node_count"),
        ("gpu", "gpu_min_nodes", "gpu_max_nodes", "gpu_node_count"),
    ):
        outcome = preflight.for_pool(pool)
        if outcome is None or outcome.status != QUOTA_CLAMPED:
            continue
        changes[min_field] = outcome.min_units
        changes[max_field] = outcome.max_units
        changes[count_field] = min(getattr(plan, count_field), outcome.max_units)
        if not outcome.autoscale_viable:
            autoscale = False

    if not changes:
        return plan
    if autoscale != plan.autoscale:
        changes["autoscale"] = autoscale
    return replace(plan, **changes)


# --- Dataproc clusters: assembling the ask, and applying the answer -----------------------------


def cluster_demands(
    region: str,
    *,
    master_machine_type: str,
    worker_machine_type: str,
    worker_count: int,
    gpu_type: str | None = None,
    accelerators_per_worker: int = 0,
) -> list[QuotaDemand]:
    """The meters a Dataproc cluster of this shape will draw on, in one region (pure).

    Two demands, and both are **Compute Engine's**. A Dataproc cluster is a set of GCE VMs and is
    billed as one; nothing under ``dataproc.googleapis.com`` is a capacity meter, so there is no
    third, Dataproc-specific number to go looking for.

    * the **GPU device** meter, pool-scoped, so a shortfall lowers the worker count. Note this is
      `compute_gpu_metric` and not `vertex_gpu_metric` — the same T4 reads 4 here and 12 on the Ray
      path, and reaching for the wrong one is the mistake this module exists to make impossible.
    * the **cluster's vCPUs**, master plus workers, against `compute_cpu_metric`. Shared scope, so
      a shortfall is reported rather than acted on (see `QuotaMetric.scope`): those vCPUs are the
      same pool every other VM in the project draws from, and shrinking *this* cluster because
      something else is using the region is a judgement call belonging to a person. The one case it
      does stop a run is when even a single worker will not fit, which is not a judgement call.

    ``min_units`` is **1, not the worker count**, and that difference is the whole behaviour. A
    Dataproc cluster does not autoscale, so its "floor" is not a shape requirement the way a Ray
    pool's is — it is just the smallest thing that is still a cluster. An allowance that seats two
    of the eight workers asked for should build a two-worker cluster and say so, not refuse to
    build anything.

    No wall-clock advice comes back from this path, and the omission is deliberate rather than
    unfinished: `advise` prices a ceiling in cells-per-slot, and a cluster is sized against *tasks*
    — buckets of `compute.bucket_target_cells` cells that run sequentially inside one frame (see
    `dataproc_cluster.cluster_sizing`). Quoting the Ray arithmetic over a different unit would put a
    confidently wrong number in front of an operator.
    """
    demands: list[QuotaDemand] = []
    workers = max(1, worker_count)

    if gpu_type and accelerators_per_worker > 0:
        metric = compute_gpu_metric(gpu_type)
        if metric is not None:
            demands.append(
                QuotaDemand(
                    metric=metric,
                    region=region,
                    pool="gpu",
                    per_unit=accelerators_per_worker,
                    min_units=1,
                    max_units=workers,
                )
            )

    demands.append(
        QuotaDemand(
            metric=compute_cpu_metric(),
            region=region,
            pool="worker",
            per_unit=machine_cores(worker_machine_type),
            min_units=1,
            max_units=workers,
            fixed=machine_cores(master_machine_type),
        )
    )
    return demands


def preflight_cluster(
    regions: list[str],
    project_id: str,
    *,
    master_machine_type: str,
    worker_machine_type: str,
    worker_count: int,
    gpu_type: str | None = None,
    accelerators_per_worker: int = 0,
    fetch: Any = None,
) -> dict[str, QuotaPreflight]:
    """Read each candidate region's Compute Engine meters and judge this cluster shape against them.

    The Dataproc analog of `preflight_ray`, and the same economics: one GET per metric returns every
    regional bucket, so judging three candidate regions costs exactly what judging one does. Returns
    one `QuotaPreflight` per region, in the order given.

    The Dataproc candidate list is *zonal* (`compute_fallback.Candidate`) while these meters are
    regional, so several candidates share a verdict. Callers group by region before calling.
    """
    demands_by_region = {
        region: cluster_demands(
            region,
            master_machine_type=master_machine_type,
            worker_machine_type=worker_machine_type,
            worker_count=worker_count,
            gpu_type=gpu_type,
            accelerators_per_worker=accelerators_per_worker,
        )
        for region in regions
    }
    metrics = {d.metric.metric: d.metric for ds in demands_by_region.values() for d in ds}
    readings = read_limits(project_id, list(metrics.values()), regions, fetch=fetch)

    result: dict[str, QuotaPreflight] = {}
    for region, demands in demands_by_region.items():
        outcomes = [
            reconcile(
                demand,
                readings.get(
                    (demand.metric.metric, region),
                    QuotaReading(demand.metric, region, None, "not read"),
                ),
            )
            for demand in demands
        ]
        result[region] = QuotaPreflight(region=region, outcomes=tuple(outcomes))
    return result


def clamp_worker_count(worker_count: int, preflight: QuotaPreflight) -> int:
    """The worker count this region will actually grant, never above the one asked for (pure).

    Only the device meter moves this number — the vCPU meter is shared and reports rather than
    reshapes. Floored at one, because a zero-worker cluster is not a smaller cluster, it is a
    different thing; a region that cannot seat even one worker comes back `QuotaPreflight.blocked`
    and is dropped from the walk before this is ever consulted.
    """
    outcome = preflight.for_pool("gpu")
    if outcome is None or outcome.status != QUOTA_CLAMPED:
        return worker_count
    return max(1, min(worker_count, outcome.max_units))


# --- regional prerequisites: things that must exist here, quota aside ---------------------------

_ATTACHMENT_REGION_RE = re.compile(r"/regions/([^/]+)/networkAttachments/")


def attachment_region(network_attachment: str) -> str | None:
    """The region a network-attachment resource name lives in (pure; unparseable → ``None``)."""
    match = _ATTACHMENT_REGION_RE.search(network_attachment)
    return match.group(1) if match else None


def regions_without_attachment(
    network_attachment: str | None, regions: list[str]
) -> dict[str, str]:
    """Candidate regions a PSC-I cluster cannot be created in, mapped to why (pure).

    **A network attachment is a regional resource, and Terraform builds exactly one.** So a config
    listing three ``ray_regions`` while `RayInfra.network_attachment` names an attachment in
    ``us-central1`` has, in fact, one usable region — the other two 404 on the attachment, and they
    do it *before* quota is ever consulted, which is why no amount of quota-reading finds them.

    Found live 2026-09-04, where a single-candidate walk against a region with no attachment churned
    for 35 minutes on a config that could never have succeeded. The check costs a regex.

    Returns ``{}`` when no attachment is configured (public or VPC-peered clusters have no such
    constraint) or when the name does not parse — an unreadable prerequisite is not evidence of a
    missing one, and blocking a launch on a failed *parse* would be the worst kind of false
    negative.
    """
    if not network_attachment:
        return {}
    home = attachment_region(network_attachment)
    if home is None:
        return {}
    return {
        region: (
            f"the PSC-I network attachment is in {home}; network attachments are regional and "
            f"there is none in {region}"
        )
        for region in regions
        if region != home
    }


# --- the operator-facing report -----------------------------------------------------------------


def report_for_run(cfg: Any, *, settings: Any = None) -> list[str]:
    """The whole quota picture for a config, as lines to print. Reads only; changes nothing.

    This is the ``--quota`` verb. Everything under it already runs inside a launch — the launch path
    reads the same meters and clamps the same plans — so the report exists for the one thing the
    launch path cannot do, which is tell you *before* you spend anything. An operator deciding
    whether to file a quota increase should not have to start a run to find out what it would buy.

    One block per metered family node in the DAG (`dag.dag_nodes`), because the ask genuinely
    differs between them: a ``deep_learning`` job on Vertex T4s and a ``statistical`` job on a
    Dataproc cluster draw on *different services'* meters for the same card, with different
    allowances. Three shapes come back:

    * **Ray** — the Vertex pools, their ceilings, and what a raise would buy in wall clock.
    * **Spark on an ephemeral cluster** — the Compute Engine device and vCPU meters, and the worker
      count the region will actually grant.
    * **Spark on Serverless, BigQuery, and a reused cluster** — named, not metered. A reused cluster
      already exists so nothing is being asked for; a Serverless batch has no fixed allocation to
      pre-read (its vCPUs bill to Compute Engine as they are used); BigQuery has no capacity meter
      at all (`UNMETERED_SERVICES`).

    Best-effort throughout: an unresolvable environment or an unreadable meter produces a line
    saying so, never an exception. A reporting verb that can fail is one an operator stops running.
    """
    from . import ray_cluster
    from .dag import dag_nodes, plan_dag
    from .profiling.source import profile_for_run
    from .registry.ids import make_run_id
    from .settings import Settings

    lines: list[str] = []
    try:
        settings = settings or Settings.resolve()
    except Exception as exc:  # noqa: BLE001 - no SF_* env is a thing to report, not to raise on
        return [f"quota: cannot resolve settings ({exc}); set the SF_* environment and retry"]

    run_id = make_run_id(cfg)
    regions = ray_cluster._resolve_regions(cfg, settings)
    profile = profile_for_run(cfg, settings=settings)
    run_dag = plan_dag(cfg)
    # A family told to reuse a named cluster provisions nothing, so there is no ask to preflight.
    # `DagNode` does not carry the name (it is not part of a job's identity), so it comes from the
    # resolved compute the DAG was planned from.
    reused = {
        job.family
        for job in run_dag.jobs
        if job.compute is not None and job.compute.spark_cluster_name
    }
    lines.append(f"quota report for {run_id} — candidate regions: {', '.join(regions)}")

    for node in dag_nodes(run_dag):
        if node.runtime == "ray":
            lines.extend(_report_ray_node(cfg, node, run_id, regions, settings, profile))
        elif node.runtime == "spark" and node.spark_mode == "cluster" and node.family not in reused:
            lines.extend(_report_cluster_node(cfg, node, settings, profile))
        else:
            lines.append(
                f"  {node.family}: on {_placement(node, node.family in reused)}, "
                f"no capacity meter to pre-read"
            )
    return lines


def _placement(node: Any, reused: bool = False) -> str:
    """How a node's runtime reads in the report — ``spark/serverless``, ``bigquery`` (pure)."""
    if reused:
        return f"{node.runtime}/cluster (reused)"
    return f"{node.runtime}/{node.spark_mode}" if node.spark_mode else node.runtime


def _report_ray_node(
    cfg: Any, node: Any, run_id: str, regions: list[str], settings: Any, profile: Any
) -> list[str]:
    """One Ray family's block: its pools, and each candidate region's verdict on them."""
    from .engines import ray_io

    plan = ray_io.plan_cluster(
        cfg,
        list(node.models),
        run_id=run_id,
        use_gpu=node.hardware == "gpu",
        gpu_type=node.gpu_type,
        profile=profile,
    )
    seconds = _seconds_per_cell(profile, node.family)
    preflights = preflight_ray(
        plan,
        regions,
        settings.project_id,
        cpu_seconds_per_cell=seconds,
        gpu_seconds_per_cell=seconds,
    )
    unreachable = regions_without_attachment(_attachment_or_none(), regions)

    lines = [f"  {node.family} on ray/{node.hardware}: {_describe_pools(plan)}"]
    if seconds is None:
        # Said out loud rather than left as a row of "~unknown": the reason a projection is missing
        # is that nothing has measured this family yet, which is a fixable state and a different
        # thing from the meter being unreadable.
        lines.append(
            f"    no measured profile for {node.family}; ceilings are reported without "
            f"wall-clock projections"
        )
    for region in regions:
        if region in unreachable:
            lines.append(f"    {region}: UNREACHABLE — {unreachable[region]}")
            continue
        preflight = preflights.get(region)
        if preflight is not None:
            lines.extend(_render_region(region, preflight))
    return lines


def _report_cluster_node(cfg: Any, node: Any, settings: Any, profile: Any) -> list[str]:
    """One ephemeral-Dataproc-cluster family's block: its worker fleet, judged per region.

    The candidate list is `compute_fallback.resolve_candidates` rather than ``ray_regions`` — the
    Dataproc walk is zonal and hops on its own catalogue, so reporting the Ray region list here
    would describe a walk that will not happen. Distinct regions only, since these meters are
    regional and every zone in a region shares one verdict.
    """
    from .batch_infra import BatchInfra
    from .compute_fallback import resolve_candidates
    from .dataproc_cluster import (
        _DEFAULT_WORKER_COUNT,
        cluster_sizing,
        master_machine_type,
        worker_machine_type,
    )

    hardware = node.hardware or "cpu"
    family = cfg.compute.machine_family
    derived, _properties, _audit = cluster_sizing(
        cfg,
        list(node.models),
        hardware=hardware,
        gpu_type=node.gpu_type,
        profile=profile,
    )
    workers = derived if derived is not None else _DEFAULT_WORKER_COUNT
    candidates = resolve_candidates(settings=settings, infra=BatchInfra.resolve())
    regions = list(dict.fromkeys(c.region for c in candidates))

    preflights = preflight_cluster(
        regions,
        settings.project_id,
        master_machine_type=master_machine_type(family),
        worker_machine_type=worker_machine_type(hardware, node.gpu_type, family),
        worker_count=workers,
        gpu_type=node.gpu_type if hardware == "gpu" else None,
        accelerators_per_worker=1 if hardware == "gpu" else 0,
    )

    lines = [f"  {node.family} on spark/cluster/{hardware}: {workers} worker(s) + 1 master"]
    for region in regions:
        preflight = preflights.get(region)
        if preflight is None:
            continue
        lines.extend(_render_region(region, preflight))
        granted = clamp_worker_count(workers, preflight)
        if granted < workers:
            lines.append(
                f"      the cluster would be built with {granted} worker(s), not {workers}"
            )
    return lines


def _render_region(region: str, preflight: QuotaPreflight) -> list[str]:
    """A region's verdict headline plus every non-trivial finding under it (pure)."""
    verdict = "BLOCKED" if preflight.blocked else "CLAMPED" if preflight.clamped else "OK"
    return [f"    {region}: {verdict}", *(f"      {line}" for line in preflight.render())]


def _describe_pools(plan: RayClusterPlan) -> str:
    """The pools this job will actually create, as ``cpu[1,20]`` (pure).

    A pool with no planned nodes is omitted rather than printed at its default bounds. A CPU-only
    job's plan still carries ``gpu_min_nodes``/``gpu_max_nodes`` — they are fields on every plan —
    and printing them beside a job that will never ask for a GPU reads as though it will.
    """
    parts = []
    if plan.cpu_node_count > 0:
        parts.append(f"cpu[{plan.cpu_min_nodes},{plan.cpu_max_nodes}]")
    if plan.gpu_node_count > 0:
        parts.append(f"gpu[{plan.gpu_min_nodes},{plan.gpu_max_nodes}]")
    return " ".join(parts) or "no worker pools"


def _attachment_or_none() -> str | None:
    """The configured PSC-I attachment, or ``None`` when the infra cannot be resolved (pure-ish)."""
    from .ray_infra import RayInfra

    try:
        return RayInfra.resolve().network_attachment
    except Exception:  # noqa: BLE001 - an unresolvable infra rules out no region
        return None


def _seconds_per_cell(profile: Any, family: str) -> float | None:
    """The measured per-cell wall clock for one family, or ``None`` when nothing measured it.

    ``planning_wall_s`` rather than the raw median: it is the number the sizing advisor already
    consumes (median x the time margin), so the wall clock this report quotes is the same one the
    fleet was sized against. Quoting a different one would put two contradictory estimates in front
    of the same operator.
    """
    if profile is None:
        return None
    cost = profile.for_family(family)
    return cost.planning_wall_s if cost is not None else None
