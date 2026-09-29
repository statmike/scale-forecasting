"""Tests for deterministic run_id and model_hash."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from scale_forecasting.config import RunConfig
from scale_forecasting.registry.ids import _canonical_config, make_model_hash, make_run_id


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "my run",
        "data": {"source_table": "p.d.source_series_native"},
        "models": ["theta"],
    }
    base.update(over)
    return RunConfig(**base)


# --- run_id --------------------------------------------------------------------


def test_run_id_is_stable_for_same_config() -> None:
    assert make_run_id(_cfg()) == make_run_id(_cfg())


def test_run_id_changes_when_config_changes() -> None:
    assert make_run_id(_cfg()) != make_run_id(_cfg(models=["theta", "sarimax"]))


def test_run_id_has_readable_slug_prefix() -> None:
    rid = make_run_id(_cfg(run_name="My Run!!"))
    assert rid.startswith("my-run-")
    # prefix + 12-hex digest
    assert len(rid.split("-")[-1]) == 12


def test_run_id_independent_of_key_order() -> None:
    a = RunConfig(run_name="r", data={"source_table": "t"}, models=["theta"])
    b = RunConfig(models=["theta"], data={"source_table": "t"}, run_name="r")
    assert make_run_id(a) == make_run_id(b)


def test_empty_slug_falls_back_to_run() -> None:
    assert make_run_id(_cfg(run_name="!!!")).startswith("run-")


# --- what is deliberately NOT part of the identity -------------------------------
#
# A run's identity is *what was asked for*. Provenance the launcher resolves, and operational
# patience, are neither — and both failure modes here are silent, so each gets a test that says so.


def test_how_long_we_wait_for_capacity_does_not_move_the_run_id() -> None:
    """Patience is an operational knob, not a description of the experiment.

    If `compute.capacity` moved the digest, an operator who raised the GPU wait after a stock-out
    would land on a *different* run_id — a second run instead of a resumed one, and dedupe-on-read
    would never see the two as the same work.
    """
    patient = _cfg(compute={"capacity": {"ray": {"max_wall_seconds": 7200.0}}})
    assert make_run_id(patient) == make_run_id(_cfg())


def test_disabling_capacity_retry_does_not_move_the_run_id() -> None:
    """The escape hatch must not fork identity either — same ask, same id."""
    assert make_run_id(_cfg(compute={"capacity": {"enabled": False}})) == make_run_id(_cfg())


def test_sizing_a_repair_does_not_move_the_run_id() -> None:
    """A repair that runs narrower than the attempt it repairs is still the same run.

    This is the whole reason `config.RetryResources` sits under ``compute.capacity`` rather than
    beside ``compute.max_executors``. If it forked the id, a repair would write its ``run_jobs``
    rows and its cells under a run nobody is looking at — the failed run would still read as failed
    and the recovered cells would be invisible to it.
    """
    narrow = _cfg(compute={"capacity": {"retry": {"max_executors": 4}}})
    assert make_run_id(narrow) == make_run_id(_cfg())
    # The sibling it is deliberately *not*: a run-wide ceiling describes what was asked for.
    assert make_run_id(_cfg(compute={"max_executors": 4})) != make_run_id(_cfg())


def test_the_resolved_profile_source_does_not_move_the_run_id() -> None:
    """Observed live (smoke 01): a pinned harvest in the digest never converges on a re-run."""
    pinned = _cfg(compute={"profile": {"source": "prior-run-0123456789ab"}})
    assert make_run_id(pinned) == make_run_id(_cfg())


def test_the_rest_of_the_profile_block_still_moves_the_run_id() -> None:
    """Only `source` is exempt — the sizing knobs themselves describe a different experiment."""
    assert make_run_id(_cfg(compute={"profile": {"measure": "controlled"}})) != make_run_id(_cfg())


# --- removed fields, pinned back into the digest ---------------------------------
#
# `test_prebreak_snapshots` is what actually proves no id moved — it pins the digest of all 44
# shipped configs, and deleting `_REMOVED_DEFAULTS` turns every one of them red at once. What that
# failure cannot do is *say why*: it reports 44 digests that changed, which reads like an intended
# surface break rather than a removed compatibility pin. These two tests are the label on it.


def test_the_digest_still_carries_the_removed_ensemble_fields() -> None:
    """The three `EnsembleCompute` fields deleted 2026-09-25 are pinned at their old defaults.

    They were read by nothing (the ensemble node is hard-wired to the driver), so removing them
    changed no behaviour — but the digest includes defaults, so without this pin every run in the
    registry would be re-keyed and the whole validation ledger staled to buy a tidier config model.
    """
    ensemble = json.loads(_canonical_config(_cfg()))["compute"]["ensemble"]
    assert ensemble["runtime"] == "spark"
    assert ensemble["spark_mode"] is None
    assert ensemble["spark_cluster_name"] is None


def test_a_removed_field_cannot_be_set_even_though_the_digest_names_it() -> None:
    """The pin lives in the digest only. The config surface still refuses the field.

    This is the pairing that keeps the shim honest: were the model to start accepting these again,
    the pin would silently swallow whatever was set and two different configs would share an id.
    """
    with pytest.raises(ValidationError):
        _cfg(compute={"ensemble": {"runtime": "ray"}})


def test_the_digest_no_longer_carries_the_removed_target_lags() -> None:
    """`features.lags`, deleted 2026-09-26, is pinned back at the `[]` every run on record had.

    Target lags are model-owned (see `FeaturesConfig`), and the field reached no model at all: the
    lag-native models stripped its columns and the rest failed on its NaN head. So the pin is not
    an approximation of what old ids meant — it is exactly what they meant.
    """
    assert json.loads(_canonical_config(_cfg()))["features"]["lags"] == []
    with pytest.raises(ValidationError):
        _cfg(features={"lags": [1, 7]})


# --- added fields, elided from the digest while they hold their default ----------
#
# The mirror of the block above, and it carries the same risk in the other direction: an elision
# that stops matching the pre-break payload merges two different questions under one id instead of
# splitting one question across two.


def test_a_default_exog_lags_is_absent_from_the_digest() -> None:
    """Adding `features.exog_lags` did not re-key a single run, and this is the reason.

    The digest hashes `model_dump()` including defaults, so a new field would ordinarily move every
    id in existence. Eliding it while empty makes a config that asks for no covariate lags hash
    exactly as it did before the field existed — which is faithful, because that *is* the same run.
    """
    assert "exog_lags" not in json.loads(_canonical_config(_cfg()))["features"]


def test_asking_for_covariate_lags_still_moves_the_run_id() -> None:
    """The other half of the bargain: elision must never merge two different questions.

    Only the default is dropped. Any real value stays in the payload and keys its own id, so a run
    that lags a covariate can never collide with the run that does not.
    """
    plain = _cfg(features={"exog": ["promo"]})
    lagged = _cfg(features={"exog": ["promo"], "exog_lags": {"promo": [1]}})
    assert make_run_id(plain) != make_run_id(lagged)
    assert json.loads(_canonical_config(lagged))["features"]["exog_lags"] == {"promo": [1]}


def test_an_explicitly_empty_exog_lags_is_the_same_run_as_an_absent_one() -> None:
    """ "Absent" and "at default" describe the same run, so they must share an id."""
    assert make_run_id(_cfg(features={"exog_lags": {}})) == make_run_id(_cfg())


@pytest.mark.parametrize(
    "tier",
    ["static_covariates", "future_covariates", "past_covariates"],
)
def test_tiered_covariates_elided_at_default_and_move_run_id_when_set(tier: str) -> None:
    base = _cfg()
    assert tier not in json.loads(_canonical_config(base))["features"]
    assert make_run_id(_cfg(features={tier: []})) == make_run_id(base)
    with_tier = _cfg(features={tier: ["col_a"]})
    assert make_run_id(with_tier) != make_run_id(base)
    assert json.loads(_canonical_config(with_tier))["features"][tier] == ["col_a"]


# --- model_hash ----------------------------------------------------------------


def test_model_hash_is_stable() -> None:
    cfg = _cfg()
    rid = make_run_id(cfg)
    assert make_model_hash(rid, "s1", "theta", cfg) == make_model_hash(rid, "s1", "theta", cfg)


def test_model_hash_unique_per_series() -> None:
    cfg = _cfg()
    rid = make_run_id(cfg)
    assert make_model_hash(rid, "s1", "theta", cfg) != make_model_hash(rid, "s2", "theta", cfg)


def test_model_hash_unique_per_model() -> None:
    cfg = _cfg(models=["theta", "sarimax"])
    rid = make_run_id(cfg)
    assert make_model_hash(rid, "s1", "theta", cfg) != make_model_hash(rid, "s1", "sarimax", cfg)


def test_model_hash_unique_per_run() -> None:
    cfg_a = _cfg(run_name="a")
    cfg_b = _cfg(run_name="b")
    ha = make_model_hash(make_run_id(cfg_a), "s1", "theta", cfg_a)
    hb = make_model_hash(make_run_id(cfg_b), "s1", "theta", cfg_b)
    assert ha != hb


def test_model_hash_is_hex_sha256() -> None:
    cfg = _cfg()
    h = make_model_hash(make_run_id(cfg), "s1", "theta", cfg)
    assert len(h) == 64
    int(h, 16)  # raises if not hex
