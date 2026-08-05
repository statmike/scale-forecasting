"""Tests for deterministic run_id and model_hash (CONTRACTS §3, TESTING §2)."""

from __future__ import annotations

from typing import Any

from scale_forecasting.config import RunConfig
from scale_forecasting.registry.ids import make_model_hash, make_run_id


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
