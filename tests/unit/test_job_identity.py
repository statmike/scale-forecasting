"""Tests for deterministic job identity: job_key, its parse, and the re-run attempt policy."""

from __future__ import annotations

import pytest

from scale_forecasting.registry.ids import (
    BASE_JOB_FAMILIES,
    JOB_FAMILIES,
    REPAIR_JOB_FAMILIES,
    base_family,
    bigquery_job_id,
    dataproc_job_id,
    decide_attempt,
    is_repair_family,
    make_job_key,
    parse_job_key,
    ray_submission_id,
    repair_family,
)

RUN_ID = "my-run-0123456789ab"


# --- make_job_key --------------------------------------------------------------


def test_job_key_shape() -> None:
    assert make_job_key(RUN_ID, "statistical") == f"sf-{RUN_ID}-statistical-a1"
    assert make_job_key(RUN_ID, "ensemble", 3) == f"sf-{RUN_ID}-ensemble-a3"


def test_job_key_is_deterministic() -> None:
    assert make_job_key(RUN_ID, "ml", 2) == make_job_key(RUN_ID, "ml", 2)


def test_job_key_distinct_per_family_and_attempt() -> None:
    keys = {
        make_job_key(RUN_ID, "statistical", 1),
        make_job_key(RUN_ID, "ml", 1),
        make_job_key(RUN_ID, "statistical", 2),
    }
    assert len(keys) == 3


def test_job_key_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unknown job family"):
        make_job_key(RUN_ID, "stats")


def test_job_key_rejects_non_positive_attempt() -> None:
    with pytest.raises(ValueError, match="attempt must be >= 1"):
        make_job_key(RUN_ID, "ml", 0)


def test_every_family_produces_a_valid_key() -> None:
    for family in JOB_FAMILIES:
        assert make_job_key(RUN_ID, family).endswith(f"-{family}-a1")


# --- parse_job_key (backward 1:1) ----------------------------------------------


def test_parse_round_trips_make() -> None:
    for family in JOB_FAMILIES:
        for attempt in (1, 2, 17):
            job_id = make_job_key(RUN_ID, family, attempt)
            assert parse_job_key(job_id) == (RUN_ID, family, attempt)


def test_parse_recovers_run_id_with_hyphens() -> None:
    # A run whose slug ends in something family-like still parses: the true family+attempt suffix
    # is the final one, and the 12-hex digest before it never matches a family token.
    run_id = "go-native-a-0123456789ab"
    job_id = make_job_key(run_id, "statistical", 2)
    assert parse_job_key(job_id) == (run_id, "statistical", 2)


def test_parse_rejects_malformed() -> None:
    for bad in ("not-a-job", f"sf-{RUN_ID}-statistical", f"sf-{RUN_ID}-bogus-a1", "statistical-a1"):
        with pytest.raises(ValueError, match="malformed job id"):
            parse_job_key(bad)


# --- repair family tokens ------------------------------------------------------
#
# A repair files its own ``run_jobs`` row instead of a second attempt of the family it repairs,
# because ``v_run_jobs`` keeps the highest attempt per (run_id, family) — so a forty-cell repair
# filed as attempt 2 would replace the row for a hundred-thousand-cell family and report the whole
# family COMPLETED. These check that the token is a real, parseable member of the id vocabulary and
# that it always leads back to the family it came from.


def test_every_repairable_family_has_a_token_and_the_ensemble_does_not() -> None:
    assert REPAIR_JOB_FAMILIES == (
        "statistical_repair",
        "ml_repair",
        "deep_learning_repair",
        "native_repair",
    )
    # A repair never re-runs the ensemble — `dag.narrow_to_models` returns ensemble_enabled=False.
    assert "ensemble_repair" not in JOB_FAMILIES
    with pytest.raises(ValueError, match="no repair token"):
        repair_family("ensemble")


def test_a_repair_token_round_trips_through_the_job_key_regex() -> None:
    """The parse is what a trace runs on, and it has to survive the longer token."""
    for family in REPAIR_JOB_FAMILIES:
        for attempt in (1, 2, 17):
            assert parse_job_key(make_job_key(RUN_ID, family, attempt)) == (RUN_ID, family, attempt)


def test_the_shorter_family_never_swallows_the_repair_token() -> None:
    """``statistical`` comes first in the alternation; on a repair token it must not win."""
    assert parse_job_key(f"sf-{RUN_ID}-statistical_repair-a4")[1] == "statistical_repair"
    assert parse_job_key(f"sf-{RUN_ID}-statistical-a4")[1] == "statistical"


def test_a_repair_key_never_collides_with_its_family_at_any_attempt() -> None:
    keys = {
        make_job_key(RUN_ID, f, a) for f in ("statistical", "statistical_repair") for a in (1, 2)
    }
    assert len(keys) == 4


def test_base_family_leads_every_token_back_to_the_family_it_repairs() -> None:
    for family in BASE_JOB_FAMILIES:
        assert base_family(family) == family and not is_repair_family(family)
    for family in REPAIR_JOB_FAMILIES:
        assert is_repair_family(family)
        assert base_family(family) in BASE_JOB_FAMILIES
        assert repair_family(base_family(family)) == family


def test_taking_the_repair_token_twice_is_the_same_token() -> None:
    """Narrowing an already-narrowed DAG is a mistake, not a corruption."""
    assert repair_family(repair_family("ml")) == "ml_repair"


def test_a_repair_key_maps_to_a_legal_dataproc_id() -> None:
    dp = dataproc_job_id(make_job_key(RUN_ID, "deep_learning_repair", 1))
    assert dp == "sf-my-run-0123456789ab-deep-learning-repair-a1"
    assert len(dp) <= 63 and dp[:1].isalpha() and not dp.endswith("-")


# --- per-system platform ids ---------------------------------------------------


def test_dataproc_id_maps_underscore_and_stays_legal() -> None:
    dp = dataproc_job_id(make_job_key(RUN_ID, "deep_learning", 1))
    assert dp == "sf-my-run-0123456789ab-deep-learning-a1"  # underscore → hyphen
    assert dp[:1].isalpha() and not dp.endswith("-")
    assert all(c.isalnum() or c == "-" for c in dp) and dp.islower()


def test_dataproc_id_preserves_unique_tail_when_truncated() -> None:
    long_run = "a-very-long-descriptive-run-name-that-keeps-going-0123456789ab"
    dp = dataproc_job_id(make_job_key(long_run, "deep_learning", 2))
    assert len(dp) <= 63
    assert dp[:1].isalpha() and not dp.endswith("-")
    # the tail (digest + family + attempt = the unique part) survives truncation
    assert dp.endswith("0123456789ab-deep-learning-a2")


def test_dataproc_ids_distinct_across_attempts_and_families() -> None:
    ids = {
        dataproc_job_id(make_job_key(RUN_ID, "statistical", 1)),
        dataproc_job_id(make_job_key(RUN_ID, "ml", 1)),
        dataproc_job_id(make_job_key(RUN_ID, "statistical", 2)),
    }
    assert len(ids) == 3


def test_ray_and_bigquery_ids_are_the_canonical_key() -> None:
    key = make_job_key(RUN_ID, "deep_learning", 3)
    assert ray_submission_id(key) == key  # Ray accepts the key unchanged (closes the auto-id gap)
    assert bigquery_job_id(key) == key  # BQ accepts underscores/hyphens up to 1024 chars


# --- decide_attempt (re-run policy) --------------------------------------------


def test_first_run_is_attempt_one_and_new() -> None:
    assert decide_attempt(None, force=False) == (1, True)
    assert decide_attempt(None, force=True) == (1, True)


def test_unforced_rerun_reuses_existing_job() -> None:
    assert decide_attempt(1, force=False) == (1, False)
    assert decide_attempt(4, force=False) == (4, False)


def test_forced_rerun_takes_next_attempt() -> None:
    assert decide_attempt(1, force=True) == (2, True)
    assert decide_attempt(4, force=True) == (5, True)
