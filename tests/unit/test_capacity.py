"""Offline tests for capacity classification, policy bounds, the attempt ledger, and the walk.

Everything in `scale_forecasting.capacity` is pure or clock-injected, so the whole loop — including
the back-off schedule and both budget bounds — runs here in microseconds with no network and no
real sleep. That is the point of the module's shape: the thing that only fails at 3am against a
stocked-out region is exactly the thing that must be provable at a desk.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scale_forecasting import capacity as cap
from scale_forecasting.config import CapacityConfig
from scale_forecasting.errors import EngineError

# --- classification -----------------------------------------------------------------------------


# The classifier matches exception types by class *name* (so it needs no google import); these
# stand in for the real google.api_core.exceptions types and must carry their exact names.
class ServiceUnavailable(Exception):
    pass


class ResourceExhausted(Exception):
    pass


class InvalidArgument(Exception):
    pass


@pytest.mark.parametrize(
    "message",
    [
        "Resources are insufficient in region: us-central1",
        "The zone does not have enough resources available",
        "Insufficient resources in zone us-east1-b",
        "ZONE_RESOURCE_POOL_EXHAUSTED",
        "resource pool exhausted",
        "Machine type is out of resources, try a different zone",
        "Not enough capacity for the requested accelerators",
    ],
)
def test_capacity_phrasings_read_as_transient(message: str) -> None:
    assert cap.classify(message) == cap.TRANSIENT_CAPACITY


@pytest.mark.parametrize(
    "message",
    [
        "Quota exceeded for NVIDIA_T4_GPUS",
        "The request exceeds quota for this region",
        "You have reached your quota limit",
        # The plural, differently-ordered wording that a fixed marker list missed live on
        # 2026-09-01 and that cost a third region.
        "The following quotas are exceeded: CustomModelTrainingT4GPUsPerProjectPerRegion",
    ],
)
def test_quota_phrasings_read_as_a_hard_ceiling(message: str) -> None:
    assert cap.classify(message) == cap.HARD_CEILING


@pytest.mark.parametrize(
    "message",
    [
        "Permission denied on resource project foo",
        "The caller does not have permission",
        "Service account is missing a role",
        "Invalid argument: bad accelerator spec",
        "Machine type n1-nope-8 is not recognised",
        "Compute Engine API has not been used in project 12345 before",
        "Billing account is not open",
        "Image 2.2.1-debian12 can no longer be used to create new clusters",
    ],
)
def test_request_shaped_causes_read_as_a_config_fault(message: str) -> None:
    assert cap.classify(message) == cap.CONFIG_FAULT


@pytest.mark.parametrize(
    "message",
    [
        "An internal error occurred on your cluster",
        "Unexpected response.",
        "",
        "the cloud said nothing useful whatsoever",
    ],
)
def test_an_unrecognised_message_is_transient_not_a_config_fault(message: str) -> None:
    """The asymmetry, asserted directly.

    Refusing to retry a stock-out loses the feature silently; retrying a config fault wastes
    minutes and still ends in an error naming every candidate. Three contentless strings cost the
    Ray fallback three times in one afternoon, which is why the default points this way.
    """
    assert cap.classify(message) == cap.TRANSIENT_CAPACITY


def test_capacity_wins_over_a_machine_type_complaint() -> None:
    """A message can name the machine type AND say the place ran out of it.

    "machine type" is a config-fault marker on purpose (a complaint about what was asked for), so
    the tie-break has to favour capacity or every stockout phrased this way stops the walk.
    """
    message = "Machine type n1-standard-8 is unavailable in zone us-central1-a: out of resources"
    assert cap.classify(message) == cap.TRANSIENT_CAPACITY


def test_quota_loses_to_capacity_but_beats_a_config_fault() -> None:
    assert cap.classify("insufficient resources; quota exceeded") == cap.TRANSIENT_CAPACITY
    assert cap.classify("permission issue: quota exceeded for iam") == cap.HARD_CEILING


def test_transient_exception_types_are_recognised_without_a_message() -> None:
    """A Compute Engine stockout surfaces from create_cluster with no useful string at all."""
    assert cap.classify("", ServiceUnavailable()) == cap.TRANSIENT_CAPACITY
    assert cap.classify("", ResourceExhausted()) == cap.TRANSIENT_CAPACITY


def test_a_transient_exception_type_outranks_a_config_fault_message() -> None:
    """gRPC UNAVAILABLE is about the moment, whatever prose the server attached to it."""
    assert cap.classify("invalid argument", ServiceUnavailable()) == cap.TRANSIENT_CAPACITY


def test_an_unrelated_exception_type_does_not_make_a_message_transient() -> None:
    assert cap.classify("permission denied", InvalidArgument()) == cap.CONFIG_FAULT


def test_classification_is_case_insensitive() -> None:
    assert cap.classify("PERMISSION DENIED") == cap.CONFIG_FAULT
    assert cap.classify("QUOTA EXCEEDED") == cap.HARD_CEILING


def test_every_verdict_classify_can_return_is_a_declared_verdict() -> None:
    for message in ("out of resources", "quota exceeded", "permission denied", "???"):
        assert cap.classify(message) in cap.VERDICTS


# --- policy -------------------------------------------------------------------------------------


def test_backoff_grows_geometrically_and_is_capped() -> None:
    policy = cap.CapacityPolicy(
        backoff_seconds=10.0, backoff_multiplier=3.0, backoff_max_seconds=100.0
    )
    assert policy.backoff_for(0) == 0.0  # nothing has been exhausted yet
    assert policy.backoff_for(1) == 10.0
    assert policy.backoff_for(2) == 30.0
    assert policy.backoff_for(3) == 90.0
    assert policy.backoff_for(4) == 100.0  # capped
    assert policy.backoff_for(40) == 100.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": -1},
        {"max_wall_seconds": -1.0},
        {"backoff_seconds": -1.0},
        # A multiplier below 1 shrinks the wait each pass, which is a bug wearing a config's
        # clothing: it would hammer a stocked-out region harder the longer it stayed stocked out.
        {"backoff_multiplier": 0.5},
        {"backoff_max_seconds": -1.0},
    ],
)
def test_a_nonsensical_policy_is_rejected_at_construction(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        cap.CapacityPolicy(**kwargs)  # type: ignore[arg-type]


def test_shipped_policies_cover_every_service_with_a_candidate_walk() -> None:
    assert set(cap.DEFAULT_POLICIES) == {"ray", "dataproc_cluster", "dataproc_serverless"}
    assert cap.UNMANAGED_SERVICES == frozenset({"bigquery"})
    assert not set(cap.DEFAULT_POLICIES) & cap.UNMANAGED_SERVICES


def test_per_service_policies_reflect_per_service_attempt_cost() -> None:
    """One number would be wrong for two of the three: attempt cost spans two orders of magnitude.

    A Ray GPU provision is ~12 min; a Serverless rejection is seconds. So Ray gets the fewest
    attempts and the longest back-off, Serverless the most attempts and the shortest.
    """
    ray = cap.DEFAULT_POLICIES["ray"]
    serverless = cap.DEFAULT_POLICIES["dataproc_serverless"]
    assert ray.max_attempts < serverless.max_attempts
    assert ray.backoff_seconds > serverless.backoff_seconds


# --- the ledger ---------------------------------------------------------------------------------


def test_the_ledger_keeps_the_message_verbatim() -> None:
    """Verbatim because three campaign defects were classifiers failing on unforeseen strings.

    The next unrecognised wording has to be discoverable from a query, not from a live failure.
    """
    ledger = cap.CapacityLedger(service="ray")
    ledger.record("us-central1", cap.TRANSIENT_CAPACITY, "Weird Vertex Prose #7", 12.0)
    assert ledger.attempts[0].message == "Weird Vertex Prose #7"
    assert ledger.to_json()["attempts"][0]["message"] == "Weird Vertex Prose #7"


def test_a_pathological_message_is_clipped_so_telemetry_cannot_bloat() -> None:
    ledger = cap.CapacityLedger(service="ray")
    ledger.record("us-central1", cap.TRANSIENT_CAPACITY, "x" * 50_000, 1.0)
    assert len(ledger.attempts[0].message) == cap._MESSAGE_LIMIT


def test_dead_candidates_are_the_hard_ceilings_only() -> None:
    ledger = cap.CapacityLedger(service="ray")
    ledger.record("us-east1", cap.HARD_CEILING, "quota exceeded", 1.0)
    ledger.record("us-central1", cap.TRANSIENT_CAPACITY, "out of resources", 1.0)
    assert ledger.dead_candidates == frozenset({"us-east1"})


def test_the_ledger_json_is_plain_types_all_the_way_down() -> None:
    """It merges into a BigQuery JSON column: a dataclass leaking through would fail the write."""
    import json

    ledger = cap.CapacityLedger(service="dataproc_cluster")
    ledger.record("us-central1/auto", cap.TRANSIENT_CAPACITY, "out of resources", 3.14159)
    ledger.exhausted = True
    payload = json.loads(json.dumps(ledger.to_json()))
    assert payload["service"] == "dataproc_cluster"
    assert payload["exhausted"] is True
    assert payload["n_attempts"] == 1
    assert payload["attempts"][0]["elapsed_seconds"] == 3.142


# --- the walk -----------------------------------------------------------------------------------


class _Clock:
    """A monotonic clock that only moves when something asks it to."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _ledger(service: str = "ray") -> cap.CapacityLedger:
    return cap.CapacityLedger(service=service)


_FAST = cap.CapacityPolicy(
    max_attempts=6, max_wall_seconds=0.0, backoff_seconds=10.0, backoff_multiplier=2.0
)


def test_the_first_candidate_that_works_ends_the_walk() -> None:
    clock = _Clock()
    tried: list[str] = []

    def attempt(candidate: str) -> str:
        tried.append(candidate)
        return f"cluster-in-{candidate}"

    ledger = _ledger()
    got = cap.walk(
        ["us-central1", "us-east1"],
        attempt,
        ledger=ledger,
        policy=_FAST,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert got == "cluster-in-us-central1"
    assert tried == ["us-central1"]
    assert ledger.attempts == []  # nothing failed, so there is nothing to record


def test_a_transient_failure_hops_to_the_next_candidate_without_waiting() -> None:
    """Candidates before back-off: hopping is free and waiting is not."""
    clock = _Clock()

    def attempt(candidate: str) -> str:
        if candidate == "us-central1":
            raise RuntimeError("Resources are insufficient in region")
        return candidate

    ledger = _ledger()
    got = cap.walk(
        ["us-central1", "us-east1"],
        attempt,
        ledger=ledger,
        policy=_FAST,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert got == "us-east1"
    assert clock.slept == []
    assert [(a.candidate, a.verdict) for a in ledger.attempts] == [
        ("us-central1", cap.TRANSIENT_CAPACITY)
    ]


def test_the_walk_restarts_the_candidate_list_after_backing_off() -> None:
    """The behaviour that did not exist before this module: retry, not just hop.

    Modelled on the real event — the identical spec that failed everywhere succeeded untouched two
    hours later. Without a second pass, that run is simply lost.
    """
    clock = _Clock()
    calls: list[str] = []

    def attempt(candidate: str) -> str:
        calls.append(candidate)
        if len(calls) < 3:
            raise RuntimeError("An internal error occurred on your cluster")
        return candidate

    ledger = _ledger()
    got = cap.walk(
        ["us-central1", "us-east1"],
        attempt,
        ledger=ledger,
        policy=_FAST,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert got == "us-central1"  # pass 2 starts at the top of the list again
    assert calls == ["us-central1", "us-east1", "us-central1"]
    assert clock.slept == [10.0]  # exactly one back-off, between the two passes


def test_a_config_fault_stops_immediately_and_re_raises_the_original() -> None:
    """Not a wrapper: the caller's handling and the operator's traceback both stay intact."""
    clock = _Clock()
    calls: list[str] = []
    boom = RuntimeError("Permission denied on resource")

    def attempt(candidate: str) -> str:
        calls.append(candidate)
        raise boom

    ledger = _ledger()
    with pytest.raises(RuntimeError) as excinfo:
        cap.walk(
            ["us-central1", "us-east1"],
            attempt,
            ledger=ledger,
            policy=_FAST,
            now=clock.now,
            sleep=clock.sleep,
        )
    assert excinfo.value is boom
    assert calls == ["us-central1"]  # the second region was never tried
    assert ledger.attempts[0].verdict == cap.CONFIG_FAULT
    assert not ledger.exhausted  # it did not run out of patience; it was told to stop


def test_a_hard_ceiling_candidate_is_never_tried_again() -> None:
    """Waiting cannot raise a quota, so re-asking is guaranteed waste.

    This is the us-east1-allows-2-T4s case: arithmetic, not weather.
    """
    clock = _Clock()
    calls: list[str] = []

    def attempt(candidate: str) -> str:
        calls.append(candidate)
        if candidate == "us-east1":
            raise RuntimeError("The following quotas are exceeded: T4GPUsPerProjectPerRegion")
        if calls.count("us-central1") < 2:
            raise RuntimeError("Unexpected response.")
        return candidate

    ledger = _ledger()
    got = cap.walk(
        ["us-central1", "us-east1"],
        attempt,
        ledger=ledger,
        policy=_FAST,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert got == "us-central1"
    assert calls == ["us-central1", "us-east1", "us-central1"]
    assert calls.count("us-east1") == 1
    assert ledger.dead_candidates == frozenset({"us-east1"})


def test_a_walk_whose_every_candidate_hard_ceilings_gives_up_without_waiting() -> None:
    """Nothing is left to try, so spending the remaining budget discovering that is pure waste."""
    clock = _Clock()

    def attempt(candidate: str) -> str:
        raise RuntimeError("quota exceeded")

    ledger = _ledger()
    with pytest.raises(cap.CapacityExhausted):
        cap.walk(
            ["us-central1", "us-east1"],
            attempt,
            ledger=ledger,
            policy=_FAST,
            now=clock.now,
            sleep=clock.sleep,
        )
    assert len(ledger.attempts) == 2  # one each, then it stopped
    assert clock.slept == []


def test_the_attempt_bound_ends_the_walk_and_the_ledger_says_so() -> None:
    clock = _Clock()
    policy = cap.CapacityPolicy(max_attempts=3, max_wall_seconds=0.0, backoff_seconds=5.0)

    def attempt(candidate: str) -> str:
        raise RuntimeError("Unexpected response.")

    ledger = _ledger()
    with pytest.raises(cap.CapacityExhausted) as excinfo:
        cap.walk(
            ["a", "b"], attempt, ledger=ledger, policy=policy, now=clock.now, sleep=clock.sleep
        )
    assert len(ledger.attempts) == 3
    assert ledger.exhausted
    assert excinfo.value.ledger is ledger
    assert "3 attempt(s)" in str(excinfo.value)


def test_the_wall_clock_bound_ends_the_walk_even_with_attempts_left() -> None:
    """Both bounds, because attempts alone say nothing about how long they take."""
    clock = _Clock()
    policy = cap.CapacityPolicy(max_attempts=100, max_wall_seconds=100.0, backoff_seconds=1.0)

    def attempt(candidate: str) -> str:
        clock.advance(40.0)  # each attempt is expensive
        raise RuntimeError("Unexpected response.")

    ledger = _ledger()
    with pytest.raises(cap.CapacityExhausted):
        cap.walk(["a"], attempt, ledger=ledger, policy=policy, now=clock.now, sleep=clock.sleep)
    assert len(ledger.attempts) == 3  # 0s, 40s, 80s start; 120s > 100s deadline
    assert ledger.exhausted


def test_the_walk_never_sleeps_past_its_own_deadline() -> None:
    """Sleeping into the deadline means waking with no budget — an hour of it spent asleep."""
    clock = _Clock()
    policy = cap.CapacityPolicy(
        max_attempts=100, max_wall_seconds=50.0, backoff_seconds=600.0, backoff_multiplier=1.0
    )

    def attempt(candidate: str) -> str:
        raise RuntimeError("Unexpected response.")

    ledger = _ledger()
    with pytest.raises(cap.CapacityExhausted):
        cap.walk(["a"], attempt, ledger=ledger, policy=policy, now=clock.now, sleep=clock.sleep)
    assert clock.slept == []  # a 600s back-off never fits in a 50s budget


def test_zero_bounds_disable_themselves_rather_than_forbidding_everything() -> None:
    """`0` has to mean "no bound"; read as a literal ceiling it would forbid the first attempt."""
    clock = _Clock()
    policy = cap.CapacityPolicy(max_attempts=0, max_wall_seconds=0.0, backoff_seconds=1.0)
    calls: list[str] = []

    def attempt(candidate: str) -> str:
        calls.append(candidate)
        if len(calls) < 5:
            raise RuntimeError("Unexpected response.")
        return candidate

    assert (
        cap.walk(["a"], attempt, ledger=_ledger(), policy=policy, now=clock.now, sleep=clock.sleep)
        == "a"
    )
    assert len(calls) == 5


def test_describe_failure_enriches_the_message_before_it_is_classified() -> None:
    """The Vertex path needs this: the reason lives on the resource, not on the exception.

    Classifying the SDK's generic "returned an error" alone would never detect a stockout.
    """
    clock = _Clock()

    def attempt(candidate: str) -> str:
        raise RuntimeError("returned an error")

    def describe(candidate: str, exc: Exception) -> str:
        return f"{exc} | Resources are insufficient in region {candidate}"

    ledger = _ledger()
    with pytest.raises(cap.CapacityExhausted):
        cap.walk(
            ["us-central1"],
            attempt,
            ledger=ledger,
            policy=cap.CapacityPolicy(max_attempts=1, max_wall_seconds=0.0),
            describe_failure=describe,
            now=clock.now,
            sleep=clock.sleep,
        )
    assert ledger.attempts[0].verdict == cap.TRANSIENT_CAPACITY
    assert "insufficient in region us-central1" in ledger.attempts[0].message


def test_on_state_sees_the_ledger_while_the_walk_is_still_running() -> None:
    """What makes AWAITING_CAPACITY observable rather than merely nameable.

    Without a mid-walk publish, a job three regions into a walk has the same registry row as a job
    that is computing — which is exactly the hour of "it looked dead" this design exists to end.
    """
    clock = _Clock()
    seen: list[int] = []
    calls: list[str] = []

    def attempt(candidate: str) -> str:
        calls.append(candidate)
        if len(calls) < 3:
            raise RuntimeError("Unexpected response.")
        return candidate

    cap.walk(
        ["a", "b"],
        attempt,
        ledger=_ledger(),
        policy=_FAST,
        on_state=lambda led: seen.append(len(led.attempts)),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert seen == [1, 2]  # published after each failure, before the walk finished


def test_a_publish_that_raises_cannot_sink_a_walk_that_still_succeeds() -> None:
    """Losing the diagnostic is strictly better than losing the run."""
    clock = _Clock()
    calls: list[str] = []

    def attempt(candidate: str) -> str:
        calls.append(candidate)
        if len(calls) < 2:
            raise RuntimeError("Unexpected response.")
        return candidate

    def explode(_led: cap.CapacityLedger) -> None:
        raise RuntimeError("telemetry is down")

    assert (
        cap.walk(
            ["a", "b"],
            attempt,
            ledger=_ledger(),
            policy=_FAST,
            on_state=explode,
            now=clock.now,
            sleep=clock.sleep,
        )
        == "b"
    )


def test_exhaustion_publishes_the_final_state_too() -> None:
    clock = _Clock()
    seen: list[bool] = []

    def attempt(candidate: str) -> str:
        raise RuntimeError("Unexpected response.")

    with pytest.raises(cap.CapacityExhausted):
        cap.walk(
            ["a"],
            attempt,
            ledger=_ledger(),
            policy=cap.CapacityPolicy(max_attempts=1, max_wall_seconds=0.0),
            on_state=lambda led: seen.append(led.exhausted),
            now=clock.now,
            sleep=clock.sleep,
        )
    assert seen[-1] is True  # the last publish carries exhausted=True


def test_capacity_exhausted_is_an_engine_error() -> None:
    """So every existing `except EngineError` keeps working — the reason is an addition."""
    assert issubclass(cap.CapacityExhausted, EngineError)


def test_the_exhaustion_message_names_the_candidates_it_tried() -> None:
    """The diagnosis is delayed by the walk, never lost by it."""
    clock = _Clock()

    def attempt(candidate: str) -> str:
        raise RuntimeError("Unexpected response.")

    with pytest.raises(cap.CapacityExhausted) as excinfo:
        cap.walk(
            ["us-central1", "us-east1"],
            attempt,
            ledger=_ledger(),
            policy=cap.CapacityPolicy(max_attempts=2, max_wall_seconds=0.0),
            now=clock.now,
            sleep=clock.sleep,
        )
    assert "us-central1" in str(excinfo.value)
    assert "us-east1" in str(excinfo.value)


def test_an_empty_candidate_list_is_a_programming_error_not_an_exhausted_walk() -> None:
    """Silently "exhausting" zero candidates would report a stockout where there was a bad call."""
    with pytest.raises(ValueError, match="at least one candidate"):
        cap.walk([], lambda c: c, ledger=_ledger(), policy=_FAST)


def test_a_keyboard_interrupt_is_not_a_capacity_problem() -> None:
    """An operator ending the run must not be classified as the cloud running out of room."""

    def attempt(candidate: str) -> str:
        raise KeyboardInterrupt

    ledger = _ledger()
    with pytest.raises(KeyboardInterrupt):
        cap.walk(["a", "b"], attempt, ledger=ledger, policy=_FAST, now=_Clock().now)
    assert ledger.attempts == []


# --- max_passes: the bound the other two cannot express -------------------------------------------


def _always_stocked_out(candidate: str) -> str:
    raise RuntimeError("Resources are insufficient in region")


def test_one_pass_tries_every_candidate_once_and_never_waits() -> None:
    """`max_passes=1` is the pre-retry behaviour, exactly: hop the list, then give up.

    It cannot be expressed with the other two bounds. `max_attempts` counts attempts, and the walk
    does not know how many candidates it will be handed — capping attempts at len(candidates) at
    the call site would be the caller doing the loop's arithmetic for it.
    """
    clock = _Clock()
    tried: list[str] = []

    def attempt(candidate: str) -> str:
        tried.append(candidate)
        return _always_stocked_out(candidate)

    with pytest.raises(cap.CapacityExhausted):
        cap.walk(
            ["us-central1", "us-east1", "us-west1"],
            attempt,
            ledger=_ledger(),
            policy=cap.CapacityPolicy(max_passes=1, max_attempts=0, max_wall_seconds=0.0),
            now=clock.now,
            sleep=clock.sleep,
        )
    assert tried == ["us-central1", "us-east1", "us-west1"]
    assert clock.slept == []


def test_max_passes_bounds_how_many_times_the_list_is_walked() -> None:
    clock = _Clock()
    tried: list[str] = []

    def attempt(candidate: str) -> str:
        tried.append(candidate)
        return _always_stocked_out(candidate)

    with pytest.raises(cap.CapacityExhausted):
        cap.walk(
            ["a", "b"],
            attempt,
            ledger=_ledger(),
            policy=cap.CapacityPolicy(
                max_passes=3, max_attempts=0, max_wall_seconds=0.0, backoff_seconds=10.0
            ),
            now=clock.now,
            sleep=clock.sleep,
        )
    assert tried == ["a", "b", "a", "b", "a", "b"]
    # Two waits, not three: the walk does not back off on its way out the door.
    assert clock.slept == [10.0, 20.0]


def test_zero_max_passes_leaves_the_pass_count_unbounded() -> None:
    """0 disables the bound, like the other two — here the attempt budget is what stops it."""
    clock = _Clock()
    tried: list[str] = []

    def attempt(candidate: str) -> str:
        tried.append(candidate)
        return _always_stocked_out(candidate)

    with pytest.raises(cap.CapacityExhausted):
        cap.walk(
            ["a", "b"],
            attempt,
            ledger=_ledger(),
            policy=cap.CapacityPolicy(
                max_passes=0, max_attempts=5, max_wall_seconds=0.0, backoff_seconds=1.0
            ),
            now=clock.now,
            sleep=clock.sleep,
        )
    assert len(tried) == 5


# --- the config surface -------------------------------------------------------------------------


def test_an_unconfigured_run_gets_the_shipped_per_service_defaults() -> None:
    capacity_cfg = CapacityConfig()
    for service, shipped in cap.DEFAULT_POLICIES.items():
        assert capacity_cfg.policy_for(service) == shipped


def test_an_override_lays_only_its_own_fields_over_the_shipped_default() -> None:
    """The partial-override promise: name one number, inherit the rest."""
    capacity_cfg = CapacityConfig(ray={"max_wall_seconds": 7200.0})
    resolved = capacity_cfg.policy_for("ray")
    shipped = cap.DEFAULT_POLICIES["ray"]
    assert resolved.max_wall_seconds == 7200.0
    assert resolved.max_attempts == shipped.max_attempts
    assert resolved.backoff_seconds == shipped.backoff_seconds
    assert resolved.backoff_max_seconds == shipped.backoff_max_seconds
    # ...and tuning one service leaves the others exactly as shipped.
    assert capacity_cfg.policy_for("dataproc_cluster") == cap.DEFAULT_POLICIES["dataproc_cluster"]


def test_disabling_capacity_retry_collapses_every_service_to_a_single_pass() -> None:
    """The escape hatch has to be exact, not approximate — one pass, no back-off, all services."""
    capacity_cfg = CapacityConfig(enabled=False)
    for service in cap.DEFAULT_POLICIES:
        assert capacity_cfg.policy_for(service).max_passes == 1


def test_disabling_capacity_retry_beats_an_authored_pass_count() -> None:
    """`enabled: false` is the operator overruling the config, so it must win the merge."""
    capacity_cfg = CapacityConfig(enabled=False, ray={"max_passes": 9})
    assert capacity_cfg.policy_for("ray").max_passes == 1


def test_a_service_with_no_candidate_walk_has_no_policy_to_resolve() -> None:
    """BigQuery slot contention is not a create that failed somewhere retryable (UNMANAGED)."""
    for service in cap.UNMANAGED_SERVICES:
        with pytest.raises(KeyError):
            CapacityConfig().policy_for(service)


def test_a_bad_bound_fails_at_config_load_naming_the_field() -> None:
    """Validated in the config as well as the dataclass, so it fails while the run is cheap."""
    with pytest.raises(ValidationError, match="max_attempts"):
        CapacityConfig(ray={"max_attempts": -1})
    with pytest.raises(ValidationError, match="backoff_multiplier"):
        CapacityConfig(ray={"backoff_multiplier": 0.5})


def test_an_unknown_capacity_field_is_rejected_rather_than_silently_ignored() -> None:
    """A typo'd knob that parses is a knob that does nothing — and looks like it worked."""
    with pytest.raises(ValidationError):
        CapacityConfig(ray={"max_attemps": 3})
    with pytest.raises(ValidationError):
        CapacityConfig(bigquery={"max_attempts": 3})


def test_a_resolved_policy_is_the_type_the_walk_takes() -> None:
    """`policy_for` is the only bridge from config to the loop; it must land on the dataclass."""
    assert isinstance(CapacityConfig().policy_for("ray"), cap.CapacityPolicy)
