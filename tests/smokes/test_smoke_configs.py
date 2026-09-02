"""Offline validity + coverage checks for the smoke config library (`configs/smokes/*.json`).

No GCP: every smoke config must load, validate, and plan a DAG purely offline — so a broken config
(a typo'd model, an invalid runtime/hardware combo, a field the schema rejects) fails here in the
offline gate, long before anyone spends money submitting it. The coverage test pins that the library
still spans every runtime/hardware/ensemble combination the live campaign is meant to prove.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale_forecasting.config import RunConfig, load_config
from scale_forecasting.dag import plan_dag
from scale_forecasting.router import split_by_runtime

_SMOKE_DIR = Path(__file__).resolve().parents[2] / "configs" / "smokes"
_CONFIGS = sorted(_SMOKE_DIR.glob("*.json"))


def test_smoke_dir_is_populated() -> None:
    # Guard against a path/glob regression silently collecting zero configs (which would make every
    # parametrized test below vacuously pass).
    assert len(_CONFIGS) >= 12, f"expected the smoke library under {_SMOKE_DIR}, found {_CONFIGS}"


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_smoke_config_loads_and_plans(path: Path) -> None:
    cfg = load_config(str(path))
    assert isinstance(cfg, RunConfig)
    # Plans a DAG offline (resolves per-family compute + the runtime split) — the same resolution
    # the live run does, so an invalid combo surfaces here.
    dag = plan_dag(cfg)
    assert dag.families, f"{path.name} planned no families"


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_smoke_config_conventions(path: Path) -> None:
    raw = json.loads(path.read_text())
    cfg = load_config(str(path))
    # run_name mirrors the file stem so a run is traceable back to its config from the registry.
    assert raw["run_name"] == f"smoke_{path.stem}", (
        f"{path.name}: run_name {raw['run_name']!r} should be 'smoke_{path.stem}'"
    )
    # Bounded scale — a smoke is ~100 series, never an unbounded (full-table) run.
    assert cfg.data.series_limit is not None and cfg.data.series_limit <= 1000, (
        f"{path.name}: set a bounded series_limit for a smoke"
    )


def test_library_covers_every_runtime_combo() -> None:
    """The library must still span the combinations the live campaign proves (coverage tripwire)."""
    seen: set[str] = set()
    for path in _CONFIGS:
        cfg = load_config(str(path))
        dag = plan_dag(cfg)
        python_models, bq_models = split_by_runtime(cfg)
        if bq_models:
            seen.add("native")
        for job in dag.python_jobs:
            rc = job.compute
            if rc.runtime == "ray":
                seen.add("ray_gpu" if rc.hardware == "gpu" else "ray_cpu")
            elif rc.runtime == "spark":
                if rc.hardware == "gpu":
                    seen.add("spark_gpu")
                elif rc.spark_mode == "cluster":
                    seen.add("spark_cluster")
                else:
                    seen.add("spark_serverless")
        if cfg.ensemble.enabled:
            seen.add(f"ensemble_{cfg.compute.ensemble.mode}")

    required = {
        "spark_serverless",
        "spark_cluster",
        "spark_gpu",
        "ray_cpu",
        "ray_gpu",
        "native",
        "ensemble_barrier",
        "ensemble_microbatch",
    }
    missing = required - seen
    assert not missing, f"smoke library no longer covers: {sorted(missing)}"


def test_a_smoke_needs_two_dataproc_clusters_at_once() -> None:
    """Some smoke must force the per-hardware cluster split, or the branch ships unexercised.

    A Dataproc cluster has one worker machine type, so a run whose ephemeral cluster families span
    CPU and GPU gets one cluster each. That is a different code path from the single-cluster case —
    a second create, two distinct names, a per-cluster region, two teardowns — and *no config
    reached it* when the split was written: smoke 04 has two cluster families and both are CPU, so
    it takes the single-group path unchanged.

    This is the config-side half of that gap. It cannot prove the two clusters actually come up
    (that needs live spend), but it does guarantee a config exists that would, so the branch is
    never silently uncovered again.
    """
    from scale_forecasting.shared_clusters import shared_spark_inputs

    split = {
        path.name: sorted(groups)
        for path in _CONFIGS
        if (groups := shared_spark_inputs(plan_dag(load_config(str(path))).python_jobs) or {})
        and len(groups) > 1
    }
    assert split, (
        "no smoke config produces a multi-hardware Dataproc cluster split; add one with two "
        "ephemeral spark_mode=cluster families on different hardware"
    )


def test_at_least_one_native_source_format_smoke() -> None:
    # Dual-format coverage: the library must exercise both the managed-Iceberg and native BigQuery
    # source tables so a live campaign proves reads work against each.
    tables = {load_config(str(p)).data.source_table for p in _CONFIGS}
    assert "source_series_native" in tables, "add a smoke reading the native-format source table"
    assert "source_series_iceberg" in tables, "add a smoke reading the Iceberg source table"
