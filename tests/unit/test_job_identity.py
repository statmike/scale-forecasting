"""Tests for deterministic job identity: job_key, its parse, and the re-run attempt policy."""

from __future__ import annotations

import pytest

from scale_forecasting.registry.ids import (
    JOB_FAMILIES,
    bigquery_job_id,
    dataproc_job_id,
    decide_attempt,
    make_job_key,
    parse_job_key,
    ray_submission_id,
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
