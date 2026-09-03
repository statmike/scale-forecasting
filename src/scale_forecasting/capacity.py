"""Resource unavailability as a *state*, not an exception.

The system has always **reacted** to "the cloud has no room": Dataproc walks zone candidates,
Vertex Ray walks region candidates. It has never **represented** it. A reaction lives in a stack
frame, shows up only in a log line, and when it runs out raises something indistinguishable from a
``ModuleNotFoundError``. A state has a name, a counter, a deadline, and a row someone can query.

This module is the representation. It owns four things, in dependency order:

1. **`classify`** — one classifier, three verdicts. Previously there were two classifiers with
   opposite defaults (`compute_fallback` for Dataproc, private helpers in `ray_cluster` for Vertex)
   and no shared vocabulary. Two classifiers that disagree is worse than one that is wrong.
2. **`CapacityPolicy`** — how much patience to spend, bounded by attempts *and* wall-clock.
3. **`CapacityLedger`** — the attempt log that ends up in ``job_telemetry`` under ``$.capacity``.
4. **`walk`** — the loop: try candidates, then back off and try them again, until one works or the
   budget is gone.

Everything here is pure and import-free (stdlib only) so the whole thing unit-tests offline with an
injected clock. The cluster submitters drive it.

**Why three verdicts and not a boolean.** The old question was *hop, or raise?* The right question
is *why*, because only one of the three answers is worth waiting on:

| Verdict | Meaning | Action |
|---|---|---|
| `TRANSIENT_CAPACITY` | no room *right now* | hop **and** retry — time will plausibly fix it |
| `HARD_CEILING` | not allowed this much *here* | hop only; waiting cannot raise a quota |
| `CONFIG_FAULT` | the request is malformed | stop; hopping ends in the same error |

The middle one was earned on 2026-09-02: ``us-east1`` allows 2 training T4s and a smoke asked for 7,
so no amount of retrying could ever succeed there — while ``us-central1``'s contentless error the
same afternoon *was* worth retrying and succeeded untouched two hours later. Same accelerator, same
day, opposite correct behaviour. A policy that treated both as transient would have burned the
entire time budget on the region that was arithmetically impossible.

**An unrecognised message is `TRANSIENT_CAPACITY`, not `CONFIG_FAULT`** — see `classify`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from .errors import EngineError, get_logger

_log = get_logger(__name__)

C = TypeVar("C")
T = TypeVar("T")


# --- verdicts ---------------------------------------------------------------------------------

TRANSIENT_CAPACITY = "TRANSIENT_CAPACITY"
HARD_CEILING = "HARD_CEILING"
CONFIG_FAULT = "CONFIG_FAULT"

VERDICTS: frozenset[str] = frozenset({TRANSIENT_CAPACITY, HARD_CEILING, CONFIG_FAULT})


# --- the registry status ------------------------------------------------------------------------

# The ``run_jobs.status`` a family wears while the walk is between attempts. It is **non-terminal**:
# the job has not failed, it has not been abandoned, and it is not running either — it is waiting on
# a cloud that has no room yet. That third thing had no name before this module, so a run in it
# looked exactly like a run that had hung.
#
# The literal lives *here*, next to the walk that writes it, and every reader imports it. There are
# already four separately-spelled copies of the terminal-status set in this codebase
# (`registry.ops`, `probes.vocabulary`, `airflow_tasks`, `sdk`) for import-cycle reasons; a fifth
# divergent spelling of a status meaning "do not close this run yet" is where that would bite.
#
# It is deliberately **not** terminal. See `CapacityExhausted`: when the budget runs out the row
# goes ``FAILED`` with ``failure_reason=CAPACITY_EXHAUSTED``, because FAILED is already
# load-bearing in the header roll-up, `close_runs`, and the validation ledger.
AWAITING_CAPACITY = "AWAITING_CAPACITY"

# The ``failure_reason`` stamped on the FAILED row when the walk gives up.
CAPACITY_EXHAUSTED = "CAPACITY_EXHAUSTED"


# --- classification ---------------------------------------------------------------------------

# Phrases that mark a *capacity* shortage: the place is out of the thing, right now. Pooled from the
# two lists this module replaces — `compute_fallback._CAPACITY_ERROR_MARKERS` (Compute Engine /
# Dataproc phrasings) and `ray_cluster._CAPACITY_ERROR_MARKERS` (Vertex phrasings). The union is
# deliberate: a phrasing observed on one service is evidence about the wording of clouds in general,
# and nothing is lost by recognising it everywhere.
_CAPACITY_MARKERS: tuple[str, ...] = (
    "resources are insufficient in region",
    "does not have enough resources",
    "insufficient resources",
    "resource exhausted",
    "resource pool exhausted",
    "zone_resource_pool_exhausted",
    "out of resources",
    "try a different",
    "capacity",
)

# api_core exception class names that are transient by nature (gRPC UNAVAILABLE=14,
# RESOURCE_EXHAUSTED=8). Matched by *name* so this module needs no google import — the same trick
# `compute_fallback` used, kept for the same reason.
_CAPACITY_EXCEPTION_NAMES: frozenset[str] = frozenset({"ServiceUnavailable", "ResourceExhausted"})

# Quota is matched *compositionally* — a quota word near an exhaustion word — rather than as fixed
# phrases. Every phrase the old fixed list held ("quota exceeded", "exceeds quota", "quota limit")
# says quota and says exceed-or-limit, so nothing is lost; what is gained is the wordings nobody
# enumerated. Found live 2026-09-01: `us-east1` answered "The following quotas are exceeded:
# CustomModelTrainingT4GPUsPerProjectPerRegion" — plural, in an order no marker matched — so a
# textbook quota ceiling was misread as a config fault and the third region was never tried.
_QUOTA_WORDS: tuple[str, ...] = ("quota", "quotas")
_EXHAUSTION_WORDS: tuple[str, ...] = ("exceed", "exceeds", "exceeded", "limit", "limits")

# Causes that are the same in every region, and so the only reason to stop walking. Everything here
# names something about the *request* rather than the *place*: who is asking, what they asked for,
# whether the API is even on. Trying another region will not change any of them.
_CONFIG_FAULT_MARKERS: tuple[str, ...] = (
    "permission denied",
    "permission_denied",
    "does not have permission",
    "not authorized",
    "unauthorized",
    "iam",
    "service account",
    "invalid argument",
    "invalid_argument",
    # Broad on purpose: any complaint *about the machine type* is a complaint about what was asked
    # for. The capacity check wins the tie below — "machine type X is unavailable in this zone" is
    # about the place.
    "machine type",
    "unsupported accelerator",
    "has not been used in project",  # API disabled
    "api is not enabled",
    "billing",
    # A retired Dataproc image is nowhere and never again, which is the sharpest possible
    # config fault: no zone has it either, and only a version bump fixes it.
    "can no longer be used to create new clusters",
    "no longer supported",
)


def classify(message: str, exc: BaseException | None = None) -> str:
    """Read a provisioning failure and return one of `VERDICTS` (pure).

    ``message`` should be the richest text available. On Vertex that means the *resource's* own
    ``error.message`` concatenated with the raised exception — the SDK's exception is a generic
    "returned an error" while the "Resources are insufficient in region" / quota text lives only on
    the resource, so classifying on the exception alone would never detect a stockout. ``exc`` is
    consulted only for its class name (`_CAPACITY_EXCEPTION_NAMES`), which is how a Compute Engine
    stockout surfaces from ``create_cluster`` with no useful string at all.

    **Precedence, and every step of it is load-bearing:**

    1. **Capacity wins outright.** A message can name a machine type *and* say the zone ran out of
       it; that is about the place, and hopping is exactly right.
    2. **Quota, when it is not also capacity, is a `HARD_CEILING`.** The region has room; this
       project is simply not allowed more. Hop, never wait.
    3. **A named region-invariant cause is a `CONFIG_FAULT`.** Stop.
    4. **Everything else is `TRANSIENT_CAPACITY`.**

    Step 4 is the asymmetry, and it is deliberate. The classifier used to work the other way — hop
    only on reasons we recognised — and that inverted default cost the feature three times in one
    afternoon (2026-09-01), each to a different contentless string: "An internal error occurred on
    your cluster", "Unexpected response.", and the plural quota message above. Each read as a
    diagnosed config fault and was re-raised in the first region, so a config naming three regions
    tried one. The costs are not symmetric: retrying a config fault wastes minutes and still ends in
    an error that names every candidate tried, while refusing to retry a stock-out loses the
    feature silently. So: give up only when the message names a cause that travels with the request.

    Note this makes the **Dataproc** path more patient than it was. `compute_fallback` re-raised on
    anything it did not recognise as capacity; it now hops and retries instead. That is a real
    behaviour change, made on purpose, so both cluster paths answer to one rule.
    """
    low = message.lower()
    if exc is not None and type(exc).__name__ in _CAPACITY_EXCEPTION_NAMES:
        return TRANSIENT_CAPACITY
    if any(marker in low for marker in _CAPACITY_MARKERS):
        return TRANSIENT_CAPACITY
    if any(q in low for q in _QUOTA_WORDS) and any(e in low for e in _EXHAUSTION_WORDS):
        return HARD_CEILING
    if any(marker in low for marker in _CONFIG_FAULT_MARKERS):
        return CONFIG_FAULT
    return TRANSIENT_CAPACITY


# --- policy -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityPolicy:
    """How much patience to spend looking for room, bounded two ways at once.

    **Both bounds, because attempt cost spans two orders of magnitude across services.** A Ray GPU
    provision is ~12 minutes per attempt (measured 2026-09-02); a Serverless batch rejection comes
    back in seconds. ``max_attempts=5`` therefore means an hour on one and half a minute on the
    other, so neither bound alone means anything. Whichever is reached first ends the walk.

    ``max_attempts`` counts *individual candidate attempts*, not passes over the candidate list — a
    three-region config with ``max_attempts=6`` gets two full passes. ``0`` disables the bound.

    ``max_wall_seconds`` is measured from the first attempt. It is checked *before* starting an
    attempt and *before* sleeping, never mid-attempt: a create that is already running is allowed to
    finish, because killing it would leak the very resource `ray_cluster._clear_stale_resource`
    exists to clean up. So the real ceiling is this plus one attempt's duration, which is honest and
    documented rather than pretended away. ``0.0`` disables the bound.

    ``max_passes`` bounds complete sweeps of the candidate list. ``0`` disables it; ``1`` reproduces
    the pre-retry behaviour exactly — try each place once, then give up — which is what
    `config.CapacityConfig.enabled = false` resolves to. It exists because "no retry" cannot be
    expressed with the other two bounds: the walk does not know how many candidates it will get.

    Back-off applies only *between passes*, never between candidates within a pass — hopping is free
    and waiting is not, so the walk exhausts the places it can try before it spends any time.
    """

    max_attempts: int = 6
    max_wall_seconds: float = 3600.0
    max_passes: int = 0
    backoff_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    backoff_max_seconds: float = 600.0

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0 (0 disables the attempt bound)")
        if self.max_passes < 0:
            raise ValueError("max_passes must be >= 0 (0 disables the pass bound)")
        if self.max_wall_seconds < 0:
            raise ValueError("max_wall_seconds must be >= 0 (0 disables the time bound)")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1 (a shrinking back-off is a bug)")
        if self.backoff_max_seconds < 0:
            raise ValueError("backoff_max_seconds must be >= 0")

    def backoff_for(self, completed_passes: int) -> float:
        """Seconds to wait after ``completed_passes`` exhausted passes (pure, capped)."""
        if completed_passes < 1:
            return 0.0
        delay = self.backoff_seconds * (self.backoff_multiplier ** (completed_passes - 1))
        return min(delay, self.backoff_max_seconds)


# Shipped per-service defaults. The services genuinely differ in attempt cost, so one number would
# be wrong for three of them. Overridable per run via `compute.capacity` (config), which is excluded
# from the run_id digest — a retry policy is an operational knob, and if it moved identity then
# changing your patience would fork your run.
DEFAULT_POLICIES: dict[str, CapacityPolicy] = {
    # ~12 min per GPU provision: the fewest attempts and the longest per-attempt budget.
    "ray": CapacityPolicy(
        max_attempts=6, max_wall_seconds=3600.0, backoff_seconds=120.0, backoff_max_seconds=600.0
    ),
    # ~5-7 min per create, and the richest candidate list (zones as well as regions).
    "dataproc_cluster": CapacityPolicy(
        max_attempts=8, max_wall_seconds=2700.0, backoff_seconds=60.0, backoff_max_seconds=300.0
    ),
    # Seconds to reject, so patience is cheap; the bound that matters here is the clock.
    "dataproc_serverless": CapacityPolicy(
        max_attempts=10, max_wall_seconds=1800.0, backoff_seconds=30.0, backoff_max_seconds=120.0
    ),
}

# Named so the omission is deliberate rather than forgotten: BigQuery slot contention is resolved
# BigQuery-side (reservations, on-demand queueing) and surfaces as latency, not as a create that
# fails and could be retried elsewhere. There is no candidate list to walk, so there is no policy.
UNMANAGED_SERVICES: frozenset[str] = frozenset({"bigquery"})


# --- the attempt ledger -------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityAttempt:
    """One attempt at one candidate, and what the cloud said about it.

    ``message`` is kept **verbatim**, truncated only to keep a pathological traceback out of a
    telemetry column. Three of this campaign's defects were classifiers failing on strings nobody
    had thought of, and the only durable fix is keeping the strings: the next unrecognised wording
    is discoverable from a query instead of from a live failure.
    """

    candidate: str
    verdict: str
    message: str
    elapsed_seconds: float

    def to_json(self) -> dict[str, object]:
        """The telemetry shape — plain JSON types only, so it merges into ``job_telemetry``."""
        return {
            "candidate": self.candidate,
            "verdict": self.verdict,
            "message": self.message,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


# Long enough to hold any real cloud error message, short enough that a runaway traceback cannot
# bloat the telemetry column.
_MESSAGE_LIMIT = 2000


@dataclass
class CapacityLedger:
    """Every attempt a walk made, in order — what lands under ``job_telemetry.$.capacity``.

    Mutable and append-only by design: `walk` writes to it as it goes, so a caller holding the
    ledger can publish partial state *while the walk is still running*. That is what makes
    ``AWAITING_CAPACITY`` observable rather than merely nameable — without it, the only view of a
    job in the middle of a three-region walk would be the same ``RUNNING`` row a computing job has.
    """

    service: str
    attempts: list[CapacityAttempt] = field(default_factory=list)
    exhausted: bool = False

    def record(self, candidate: str, verdict: str, message: str, elapsed_seconds: float) -> None:
        """Append one attempt, clipping an over-long message."""
        self.attempts.append(
            CapacityAttempt(
                candidate=candidate,
                verdict=verdict,
                message=message[:_MESSAGE_LIMIT],
                elapsed_seconds=elapsed_seconds,
            )
        )

    @property
    def dead_candidates(self) -> frozenset[str]:
        """Candidates that answered `HARD_CEILING` — never worth another attempt this run.

        A quota ceiling does not move while a run is in flight, so re-trying one is guaranteed
        waste. Dropping them is what keeps a long walk from spending its whole budget re-asking a
        region that has already said no on arithmetic grounds.
        """
        return frozenset(a.candidate for a in self.attempts if a.verdict == HARD_CEILING)

    def to_json(self) -> dict[str, object]:
        """The ``$.capacity`` payload: the service, the attempt list, and whether it gave up."""
        return {
            "service": self.service,
            "exhausted": self.exhausted,
            "n_attempts": len(self.attempts),
            "attempts": [a.to_json() for a in self.attempts],
        }


class CapacityExhausted(EngineError):
    """Every candidate was tried until the attempt or time budget ran out.

    Carries the ledger so the caller can write it to telemetry and stamp the job row ``FAILED`` with
    ``failure_reason: CAPACITY_EXHAUSTED`` — the one failure worth re-running unchanged, and the
    thing an operator could not previously tell apart from a broken import.
    """

    def __init__(self, message: str, ledger: CapacityLedger) -> None:
        super().__init__(message)
        self.ledger = ledger


# --- the walk ---------------------------------------------------------------------------------


@dataclass
class _Budget:
    """Bookkeeping for one walk: what has been spent and what is still worth trying."""

    policy: CapacityPolicy
    started: float
    attempts: int = 0

    def attempts_left(self, now: float) -> bool:
        """True if both bounds still permit starting another attempt."""
        if self.policy.max_attempts and self.attempts >= self.policy.max_attempts:
            return False
        return not (
            self.policy.max_wall_seconds and now - self.started >= self.policy.max_wall_seconds
        )

    def sleep_fits(self, now: float, delay: float) -> bool:
        """True if waiting ``delay`` would still leave time to attempt something afterwards.

        Sleeping into the deadline is pure waste: the walk would wake up only to discover it has no
        budget left. Checking first turns "gave up after an hour, most of it asleep" into "gave up
        after an hour of trying".
        """
        if not self.policy.max_wall_seconds:
            return True
        return (now - self.started) + delay < self.policy.max_wall_seconds


def walk(
    candidates: Sequence[C],
    attempt: Callable[[C], T],
    *,
    ledger: CapacityLedger,
    policy: CapacityPolicy,
    label: Callable[[C], str] = str,
    describe_failure: Callable[[C, Exception], str] | None = None,
    on_state: Callable[[CapacityLedger], None] | None = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Try ``candidates`` until one succeeds, retrying the list until the budget is gone.

    The loop the design calls for, and the whole reason this module exists::

        attempt 1 .......... TRANSIENT_CAPACITY in us-central1
          hop to the next candidate; no wait (another region may have room now)
        candidates exhausted, budget remains
          back off, then restart the candidate walk from the top
        budget exhausted (attempts OR wall-clock)
          terminal: CapacityExhausted, ledger intact

    Returns whatever ``attempt`` returns for the candidate that worked.

    **Raises immediately on `CONFIG_FAULT`** — re-raising the original exception, not a wrapper, so
    the caller's existing handling and the operator's traceback are both unchanged. Hopping on a
    malformed request wastes minutes and ends in the same error.

    **Raises `CapacityExhausted`** when the bounds run out, carrying the ledger.

    ``describe_failure`` lets a caller enrich the message before it is classified and recorded. The
    Vertex path needs it: the SDK raises a generic "returned an error" while the reason lives on the
    failed resource, so that path reads the resource and concatenates. Defaults to ``str(exc)``.

    ``on_state`` is called after every recorded attempt, with the ledger, so a caller can publish
    ``AWAITING_CAPACITY`` and the partial ledger *during* the walk rather than only at the end. It
    is best-effort: a publish that raises must not sink a walk that might still succeed, so it is
    caught and logged.

    ``now``/``sleep`` are injected so the whole loop — including the back-off schedule and both
    bounds — unit-tests offline in microseconds with no real clock.
    """
    if not candidates:
        raise ValueError("walk needs at least one candidate")

    budget = _Budget(policy=policy, started=now())
    last_exc: Exception | None = None
    completed_passes = 0

    while True:
        live = _live_candidates(candidates, ledger, label)
        if not live:
            # Every candidate has answered HARD_CEILING. Waiting cannot raise a quota, so there is
            # nothing left to try and no point spending the remaining budget discovering that.
            break

        for candidate in live:
            if not budget.attempts_left(now()):
                break
            budget.attempts += 1
            started = now()
            try:
                return attempt(candidate)
            # `Exception`, not `BaseException`: a KeyboardInterrupt or SystemExit during a create is
            # an operator ending the run, not the cloud running out of room. Classifying one as
            # TRANSIENT_CAPACITY would make Ctrl-C start the next region.
            except Exception as exc:  # noqa: BLE001 - classified below; CONFIG_FAULT re-raises
                message = (
                    describe_failure(candidate, exc) if describe_failure else str(exc)
                ) or repr(exc)
                verdict = classify(message, exc)
                ledger.record(label(candidate), verdict, message, now() - started)
                _publish(on_state, ledger)
                _log.warning(
                    "capacity: %s attempt %d on %s -> %s (%s)",
                    ledger.service,
                    budget.attempts,
                    label(candidate),
                    verdict,
                    message.splitlines()[0][:200] if message else "no message",
                )
                if verdict == CONFIG_FAULT:
                    raise
                last_exc = exc

        completed_passes += 1
        if policy.max_passes and completed_passes >= policy.max_passes:
            break
        if not budget.attempts_left(now()):
            break
        # Re-check *before* sleeping, not just at the top of the next pass. A pass in which every
        # candidate answered HARD_CEILING has nothing left to come back to, so backing off first
        # would spend real minutes on the way to a conclusion already reached.
        if not _live_candidates(candidates, ledger, label):
            break
        delay = policy.backoff_for(completed_passes)
        if not budget.sleep_fits(now(), delay):
            break
        if delay:
            _log.warning(
                "capacity: %s exhausted %d candidate(s) on pass %d; backing off %.0fs",
                ledger.service,
                len(live),
                completed_passes,
                delay,
            )
            sleep(delay)

    ledger.exhausted = True
    _publish(on_state, ledger)
    raise CapacityExhausted(
        f"{ledger.service}: no capacity after {len(ledger.attempts)} attempt(s) across "
        f"{len({a.candidate for a in ledger.attempts})} candidate(s) "
        f"({', '.join(sorted({a.candidate for a in ledger.attempts})) or 'none'}); "
        f"last error {last_exc!r}",
        ledger,
    )


def _live_candidates(
    candidates: Iterable[C], ledger: CapacityLedger, label: Callable[[C], str]
) -> list[C]:
    """``candidates`` minus the ones a `HARD_CEILING` has already ruled out (pure)."""
    dead = ledger.dead_candidates
    return [c for c in candidates if label(c) not in dead]


def _publish(on_state: Callable[[CapacityLedger], None] | None, ledger: CapacityLedger) -> None:
    """Call ``on_state``, swallowing anything it raises.

    A telemetry write that fails must never sink a walk that might still find room — the state is a
    diagnostic, and losing the diagnostic is strictly better than losing the run.
    """
    if on_state is None:
        return
    try:
        on_state(ledger)
    except Exception as exc:  # noqa: BLE001 - observability must not sink the walk
        _log.debug("capacity state publish failed (non-fatal): %r", exc)
