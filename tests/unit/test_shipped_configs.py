"""Every JSON config shipped in ``configs/`` must load and plan a DAG (offline, GCP-free).

The example configs are the front door — the guides, workshop, and notebooks all point at them, so a
config that no longer parses under the current schema is a broken demo. This walks the shipped set
and asserts each one loads (strict ``extra="forbid"`` validation) and resolves to a run DAG, so a
knob rename or removal that leaves an example behind is caught here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.dag import plan_dag

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_CONFIGS = sorted(_CONFIGS_DIR.glob("*.json"))


def test_configs_dir_is_present_and_nonempty() -> None:
    assert _CONFIGS, f"no shipped configs found under {_CONFIGS_DIR}"


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_shipped_config_loads_and_plans(path: Path) -> None:
    cfg = RunConfig(**json.loads(path.read_text()))
    # run_name should match the file stem so a rename can't silently drift the two apart
    assert cfg.run_name == path.stem, f"{path.name}: run_name {cfg.run_name!r} != file stem"
    run_dag = plan_dag(cfg)
    assert run_dag.jobs, f"{path.name}: config planned no family jobs"
