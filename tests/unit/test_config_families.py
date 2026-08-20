"""Tests for per-family compute overrides, their validation, and the resolver."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from scale_forecasting.config import (
    EnsembleCompute,
    FamilyCompute,
    ResolvedFamilyCompute,
    RunConfig,
)


def _cfg(**compute: Any) -> RunConfig:
    """A minimal valid config with the given ``compute`` block."""
    return RunConfig(
        run_name="t",
        data={"source_table": "p.d.source_series_native"},
        models=["theta"],
        compute=compute,
    )


# --- defaults / back-compat ----------------------------------------------------


def test_omitting_families_keeps_defaults() -> None:
    cfg = _cfg()
    assert cfg.compute.families == {}
    assert cfg.compute.ensemble == EnsembleCompute()  # runtime=spark, mode=barrier


def test_resolve_inherits_flat_defaults() -> None:
    # No override, default python_runtime=spark → serverless CPU, no GPU.
    r = _cfg().resolve_family_compute("statistical")
    assert r == ResolvedFamilyCompute(
        family="statistical",
        runtime="spark",
        spark_mode="serverless",
        spark_cluster_name=None,
        hardware="cpu",
        gpu_type=None,
    )


def test_resolve_inherits_run_level_ray() -> None:
    cfg = RunConfig(
        run_name="t",
        data={"source_table": "p.d.s"},
        models=["theta"],
        python_runtime="ray",
    )
    r = cfg.resolve_family_compute("ml")
    assert r.runtime == "ray"
    assert r.spark_mode is None and r.spark_cluster_name is None
    assert r.hardware == "cpu" and r.gpu_type is None


# --- per-family overrides resolve correctly ------------------------------------


def test_per_family_runtime_override() -> None:
    cfg = _cfg(families={"ml": {"runtime": "ray"}})
    assert cfg.resolve_family_compute("ml").runtime == "ray"
    # A sibling family with no override still inherits the flat default.
    assert cfg.resolve_family_compute("statistical").runtime == "spark"


def test_spark_cluster_mode_with_reuse_name() -> None:
    cfg = _cfg(families={"statistical": {"spark_mode": "cluster", "spark_cluster_name": "warm-1"}})
    r = cfg.resolve_family_compute("statistical")
    assert r.spark_mode == "cluster"
    assert r.spark_cluster_name == "warm-1"


def test_deep_learning_gpu_serverless_forces_l4() -> None:
    cfg = _cfg(families={"deep_learning": {"hardware": "gpu"}})  # spark serverless inherited
    r = cfg.resolve_family_compute("deep_learning")
    assert r.hardware == "gpu"
    assert r.gpu_type == "L4"  # Serverless offers L4 only


def test_deep_learning_gpu_ray_uses_flat_gpu_type() -> None:
    cfg = _cfg(gpu_type="T4", families={"deep_learning": {"runtime": "ray", "hardware": "gpu"}})
    r = cfg.resolve_family_compute("deep_learning")
    assert r.runtime == "ray" and r.hardware == "gpu" and r.gpu_type == "T4"


def test_deep_learning_gpu_cluster_allows_t4() -> None:
    cfg = _cfg(
        families={"deep_learning": {"spark_mode": "cluster", "hardware": "gpu", "gpu_type": "T4"}}
    )
    r = cfg.resolve_family_compute("deep_learning")
    assert r.spark_mode == "cluster" and r.gpu_type == "T4"


def test_use_gpu_flag_drives_deep_learning_default() -> None:
    # Cluster mode so the flat T4 default survives (serverless would force L4).
    cfg = _cfg(use_gpu=True, families={"deep_learning": {"spark_mode": "cluster"}})
    r = cfg.resolve_family_compute("deep_learning")
    assert r.hardware == "gpu" and r.gpu_type == "T4"
    # Without use_gpu and no override, deep_learning stays CPU.
    assert _cfg().resolve_family_compute("deep_learning").hardware == "cpu"


# --- validation: rejected combinations -----------------------------------------


def test_native_is_not_a_valid_family_key() -> None:
    with pytest.raises(ValidationError):
        _cfg(families={"native": {"runtime": "ray"}})


def test_unknown_family_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _cfg(families={"stats": {"runtime": "ray"}})


def test_gpu_only_for_deep_learning() -> None:
    with pytest.raises(ValidationError):
        _cfg(families={"ml": {"hardware": "gpu"}})
    with pytest.raises(ValidationError):
        _cfg(families={"statistical": {"gpu_type": "L4"}})


def test_ray_rejects_spark_only_fields() -> None:
    with pytest.raises(ValidationError):
        _cfg(families={"ml": {"runtime": "ray", "spark_mode": "cluster"}})


def test_cluster_name_requires_cluster_mode() -> None:
    with pytest.raises(ValidationError):
        _cfg(families={"ml": {"spark_mode": "serverless", "spark_cluster_name": "x"}})


def test_explicit_serverless_t4_rejected() -> None:
    with pytest.raises(ValidationError):
        _cfg(families={"deep_learning": {"spark_mode": "serverless", "gpu_type": "T4"}})


def test_gpu_type_with_cpu_hardware_rejected() -> None:
    with pytest.raises(ValidationError):
        _cfg(families={"deep_learning": {"hardware": "cpu", "gpu_type": "L4"}})


def test_inherited_serverless_t4_rejected_at_load() -> None:
    # spark_mode unset → resolves to serverless; hardware=gpu + T4 is only caught at resolution,
    # which _normalize runs at load. So construction must fail.
    with pytest.raises(ValidationError):
        _cfg(families={"deep_learning": {"hardware": "gpu", "gpu_type": "T4"}})


# --- ensemble compute node -----------------------------------------------------


def test_ensemble_compute_defaults_and_override() -> None:
    cfg = _cfg(ensemble={"runtime": "ray", "mode": "microbatch"})
    assert cfg.compute.ensemble.runtime == "ray"
    assert cfg.compute.ensemble.mode == "microbatch"


def test_ensemble_ray_rejects_spark_fields() -> None:
    with pytest.raises(ValidationError):
        _cfg(ensemble={"runtime": "ray", "spark_mode": "cluster"})


# --- resolver guards -----------------------------------------------------------


def test_resolve_native_raises() -> None:
    with pytest.raises(ValueError, match="BigQuery"):
        _cfg().resolve_family_compute("native")


def test_family_block_is_frozen() -> None:
    fc = FamilyCompute(runtime="ray")
    with pytest.raises(ValidationError):
        fc.runtime = "spark"  # type: ignore[misc]


def test_families_shift_run_id() -> None:
    from scale_forecasting.registry.ids import make_run_id

    base = make_run_id(_cfg())
    with_family = make_run_id(_cfg(families={"ml": {"runtime": "ray"}}))
    assert base != with_family  # compute is part of the definition digest
