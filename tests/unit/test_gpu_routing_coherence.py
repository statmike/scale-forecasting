"""The device a run buys must be the device it routes work to (offline, pure — no Ray, no GPU).

There are two ways to ask for a GPU. The flat ``compute.use_gpu`` / ``compute.gpu_type`` pair is the
original one; ``compute.families.deep_learning.hardware`` is the documented per-family one, and it
is what every current example config uses. `config.RunConfig.resolve_family_compute` reconciles
them into one answer, and the submitter provisions from that answer.

The engine used to re-derive the same decision from the flat field alone, which meant the two halves
of a run could disagree — and they did, in both directions:

* **Buy and don't use.** A per-family GPU config leaves ``compute.use_gpu`` at ``False``. The
  submitter provisioned accelerators; the engine routed every deep-learning cell to the CPU pool.
  The devices were never even visible to a task, for the whole run, with nothing reporting it. The
  registry signature is unmistakable in hindsight: ``peak_gpu_bytes`` NULL on 100% of
  ``neuralprophet`` cells while ``cpu_seconds`` is recorded on every one.
* **Use and don't buy.** Flat ``use_gpu: true`` with a family override of ``hardware: "cpu"``
  provisioned zero GPU nodes while the engine still asked Ray for ``num_gpus`` — tasks that can
  never be scheduled, and no error to say so.

Neither shape raises anything. Both are silent, and the second is worse than a crash. So the checks
here are structural rather than behavioural where they can be: the engine is required not to read
the flat field at all, because a fix that only corrects today's five call sites does not stop the
sixth from being added next month.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.dag import plan_dag
from scale_forecasting.engines import ray_engine, ray_io
from scale_forecasting.registry.ids import make_run_id

_ROOT = Path(__file__).resolve().parents[2]
_ENGINE_SRC = _ROOT / "src" / "scale_forecasting" / "engines" / "ray_engine.py"

# Infra catalogs that live beside the run configs but have their own schema (see
# test_shipped_configs.py, which excludes the same file for the same reason).
_NON_RUNCONFIG = {"compute_fallback.json"}
_CONFIGS = sorted(
    p
    for p in [
        *(_ROOT / "configs").glob("*.json"),
        *(_ROOT / "configs" / "smokes").glob("*.json"),
    ]
    if p.name not in _NON_RUNCONFIG
)

_CPU = "theta"
_GPU = "neuralprophet"


def _cfg(compute: dict[str, Any] | None = None, **over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "gpu routing test",
        "python_runtime": "ray",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 10},
        "models": [_CPU, _GPU],
    }
    if compute is not None:
        base["compute"] = compute
    return RunConfig(**{**base, **over})


def _panel(n_series: int = 10) -> pd.DataFrame:
    """A minimal read panel — `_pool_cells` only counts distinct ids."""
    return pd.DataFrame({"ts_id": [f"s_{i:04d}" for i in range(n_series)]})


def _plans(cfg: RunConfig) -> tuple[Any, Any]:
    """Size both pools exactly the way ``ray_engine.run`` does, without needing Ray."""
    has_gpu, _ = ray_io.resolve_job_gpu(cfg)
    gpu_models, cpu_models = ray_io.split_gpu_cpu_models(cfg, cfg.models, use_gpu=has_gpu)
    return ray_engine._pool_plans(
        _panel(), cfg, make_run_id(cfg), cpu_models, gpu_models, None, 0.5
    )


# --- structural: the engine must have exactly one source of GPU intent -----------------


def _flat_gpu_reads(path: Path) -> list[tuple[int, str]]:
    """Every ``<something>.compute.use_gpu`` / ``.compute.gpu_type`` read in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in {"use_gpu", "gpu_type"}
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "compute"
        ):
            found.append((node.lineno, f"...compute.{node.attr}"))
    return sorted(found)


def test_ray_engine_never_re_derives_gpu_intent_from_the_flat_field() -> None:
    """``ray_engine`` must ask `ray_io.resolve_job_gpu`, never ``cfg.compute.use_gpu``.

    The behavioural tests below cover the shapes we know about. This one covers the shape nobody
    has written yet: `ray_engine.run` needs a live Ray cluster to execute, so a sixth flat read
    added tomorrow would not be exercised by any offline test. A static check is the only guard
    that scales to the whole file, and it costs one ``ast`` walk.
    """
    reads = _flat_gpu_reads(_ENGINE_SRC)
    assert not reads, (
        "ray_engine re-derives GPU intent from the flat compute field at "
        + ", ".join(f"line {ln} ({expr})" for ln, expr in reads)
        + ". Use ray_io.resolve_job_gpu(cfg), which reads the same resolved per-family compute "
        "the submitter provisions from — see that function's docstring for what disagreeing "
        "cost us."
    )


# --- behavioural: routing follows provisioning, in both directions ---------------------


def test_per_family_gpu_config_routes_deep_learning_to_the_gpu_pool() -> None:
    """The 'buy and don't use' shape: per-family GPU with the flat field left alone.

    This is what `per_family_runtimes_demo` and smokes 08/09/10/15 all look like, and every one
    of them provisioned accelerators that no cell ever saw.
    """
    cfg = _cfg(compute={"families": {"deep_learning": {"hardware": "gpu", "gpu_type": "T4"}}})
    assert cfg.compute.use_gpu is False  # the flat field really is off; that was the trap
    assert ray_io.resolve_job_gpu(cfg) == (True, "T4")

    _, gpu_plan = _plans(cfg)
    assert "num_gpus" in gpu_plan.task_options, (
        "deep-learning cells were routed to the CPU pool while the submitter provisioned a GPU"
    )
    assert gpu_plan.n_cells > 0, "the GPU pool was provisioned but given no cells"


def test_flat_use_gpu_config_still_routes_to_the_gpu_pool() -> None:
    """The legacy shape must keep working — ``ray_gpu_demo`` is one of these."""
    cfg = _cfg(compute={"use_gpu": True, "gpu_type": "T4"})
    assert ray_io.resolve_job_gpu(cfg) == (True, "T4")
    _, gpu_plan = _plans(cfg)
    assert "num_gpus" in gpu_plan.task_options
    assert gpu_plan.n_cells > 0


def test_family_cpu_override_beats_flat_use_gpu_and_asks_for_no_device() -> None:
    """The 'use and don't buy' shape: flat ``use_gpu: true`` overridden to CPU per family.

    The submitter provisions zero GPU nodes here. If the engine still requests ``num_gpus`` the
    tasks are unschedulable forever, which presents as a hang rather than as an error.
    """
    cfg = _cfg(
        compute={
            "use_gpu": True,
            "gpu_type": "T4",
            "families": {"deep_learning": {"hardware": "cpu"}},
        }
    )
    assert ray_io.resolve_job_gpu(cfg) == (False, None)

    cpu_plan, gpu_plan = _plans(cfg)
    assert "num_gpus" not in gpu_plan.task_options
    assert gpu_plan.n_cells == 0, "no device was provisioned, so no cell may be routed to one"
    assert cpu_plan.n_cells > 0, "the deep-learning cells have to land somewhere"


def test_gpu_type_reaches_the_pool_from_the_family_override() -> None:
    """A per-family ``gpu_type`` must not fall back to the flat default.

    ``device_memory_bytes`` is the denominator of the auto-fraction calibration, so resolving a
    T4 where an L4 was provisioned (or the reverse) misprices every slot in the pool.
    """
    cfg = _cfg(
        compute={
            "gpu_type": "T4",
            "families": {"deep_learning": {"hardware": "gpu", "gpu_type": "L4"}},
        }
    )
    assert ray_io.resolve_job_gpu(cfg) == (True, "L4")


# --- the seam, across everything we ship -----------------------------------------------


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_routes_where_it_provisions(path: Path) -> None:
    """For every shipped config: the pool that gets cells is the pool that gets hardware.

    Parametrized over the demo *and* smoke configs because the smokes are the ones that reach
    live hardware, and five of them were provisioning accelerators they never used.
    """
    cfg = RunConfig(**json.loads(path.read_text()))
    ray_jobs = [j for j in plan_dag(cfg).python_jobs if j.runtime == "ray"]
    if not ray_jobs:
        pytest.skip("no Ray job in this config")

    has_gpu, gpu_type = ray_io.resolve_job_gpu(cfg)
    provisions_gpu = any(j.compute is not None and j.compute.hardware == "gpu" for j in ray_jobs)
    assert has_gpu == provisions_gpu, (
        f"{path.name}: engine resolves has_gpu={has_gpu} but the DAG provisions "
        f"gpu={provisions_gpu}"
    )

    routed_gpu, _ = ray_io.split_gpu_cpu_models(cfg, cfg.models, use_gpu=has_gpu)
    if provisions_gpu:
        assert routed_gpu, f"{path.name}: a GPU is provisioned but no model routes to it"
        assert gpu_type, f"{path.name}: a GPU is provisioned with no resolved gpu_type"
    else:
        assert not routed_gpu, f"{path.name}: models route to a GPU that is never provisioned"
