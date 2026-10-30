"""Zero-drift tripwires for generated AI agent surfaces, JSON Schema, and skill manifests.

Verifies that:
1. ``check_agent_surfaces()`` reports zero drift across all 6 generated files:
   - ``docs/schemas/run_config.schema.json``
   - ``docs/llms.txt``
   - ``docs/llms-full.txt``
   - ``skills/scale-forecasting/references/config_reference.md``
   - ``skills/scale-forecasting/references/catalog_reference.md``
   - ``skills/scale-forecasting/references/execution_paths_and_ops.md``
2. Every shipped JSON configuration (``configs/*.json`` and ``configs/smokes/*.json``)
   validates cleanly against ``RunConfig`` both as-is and with ``"$schema"`` injected,
   preserving its exact ``make_run_id`` digest while still rejecting unknown fields.
3. ``ALL_EXTRA_MODULES`` keys match ``pyproject.toml`` ``[project.optional-dependencies]``.
4. ``probe_environment()`` and the ``python -m scale_forecasting.agent_surfaces`` CLI flags
   (``--check``, ``--write``, ``--probe-env``) behave deterministically.
5. ``skills/scale-forecasting/SKILL.md``, ``.agents/skills.json``, ``plugin.json``,
   ``mcp_config.json``, ``.mcp.json``, and ``cloudshell_tutorial.md`` conform to their respective
   specifications.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from scale_forecasting import agent_surfaces
from scale_forecasting.config import RUN_CONFIG_SCHEMA_URI, RunConfig
from scale_forecasting.registry.ids import make_run_id

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_agent_surfaces_zero_drift() -> None:
    drifted = agent_surfaces.check_agent_surfaces(REPO_ROOT)
    assert not drifted, (
        f"Agent surface files are out of sync with Python reflection: {drifted}. "
        "Run `make agent-surfaces` (`python -m scale_forecasting.agent_surfaces --write`)."
    )
    assert agent_surfaces.main(["--check"]) == 0


def test_write_and_check_agent_surfaces_in_tmp_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copy2(REPO_ROOT / "mkdocs.yml", tmp_path / "mkdocs.yml")
    configs_dir = tmp_path / "configs" / "smokes"
    configs_dir.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "configs" / "mixed_demo.json",
        tmp_path / "configs" / "mixed_demo.json",
    )
    shutil.copy2(
        REPO_ROOT / "configs" / "smokes" / "01_serverless_cpu.json",
        configs_dir / "01_serverless_cpu.json",
    )

    written = agent_surfaces.write_agent_surfaces(tmp_path)
    assert len(written) == 6
    assert agent_surfaces.check_agent_surfaces(tmp_path) == []

    # Mutate one generated file and verify check_agent_surfaces detects the drift.
    llms_path = tmp_path / "docs" / "llms.txt"
    llms_path.write_text("# drifted\n", encoding="utf-8")
    drifted = agent_surfaces.check_agent_surfaces(tmp_path)
    assert len(drifted) == 1
    assert drifted[0].startswith("docs/llms.txt:")

    monkeypatch.setattr(agent_surfaces, "_default_repo_root", lambda: tmp_path)
    assert agent_surfaces.main(["--check"]) == 1
    assert agent_surfaces.main(["--write"]) == 0
    assert agent_surfaces.main(["--check"]) == 0


def test_all_shipped_configs_validate_with_schema_key_and_preserve_run_id() -> None:
    schema = agent_surfaces.build_run_config_json_schema()
    assert schema["$id"] == RUN_CONFIG_SCHEMA_URI
    assert "$schema" in schema["properties"]
    assert schema.get("additionalProperties") is False

    demo_paths = [
        p
        for p in sorted((REPO_ROOT / "configs").glob("*.json"))
        if p.name != "compute_fallback.json"
    ]
    smoke_paths = sorted((REPO_ROOT / "configs" / "smokes").glob("*.json"))
    config_paths = demo_paths + smoke_paths
    assert len(config_paths) == 62

    schema_props = set(schema["properties"].keys())
    for path in config_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert set(raw.keys()).issubset(schema_props), f"Unlisted keys in {path.name}"
        cfg_plain = RunConfig.model_validate(raw)
        cfg_with_schema = RunConfig.model_validate({"$schema": RUN_CONFIG_SCHEMA_URI, **raw})
        assert make_run_id(cfg_plain) == make_run_id(cfg_with_schema)

    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            {
                "$schema": RUN_CONFIG_SCHEMA_URI,
                "run_name": "bad",
                "models": ["holtwinters"],
                "unknown_top_level_key": 123,
            }
        )


def test_all_extra_modules_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared_extras = set(pyproject["project"]["optional-dependencies"].keys())
    assert set(agent_surfaces.ALL_EXTRA_MODULES.keys()) == declared_extras


def test_probe_environment_structure_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = agent_surfaces.probe_environment(REPO_ROOT)
    assert "python_version" in env
    assert set(env["extras"].keys()) == set(agent_surfaces.ALL_EXTRA_MODULES.keys())
    assert env["models"]["total"] == 37
    assert env["models"]["available_count"] >= 14
    assert env["execution_paths_ready"]["path_1_offline_playground_and_dry_run"] is True
    assert env["execution_paths_ready"]["path_5_airflow_dag_emission"] is True
    assert env["auto_forecaster"]["benchmarks_count"] == 20
    assert env["auto_forecaster"]["practitioner_questions_count"] == 7
    assert "first_light" in env["auto_forecaster"]["exposure_tiers"]
    assert "hyperscale" in env["auto_forecaster"]["scale_regimes"]
    assert isinstance(env["recommended_actions"], list)

    rc = agent_surfaces.main(["--probe-env"])
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["models"]["total"] == 37


def test_skill_md_and_extension_manifests() -> None:
    skill_path = REPO_ROOT / "skills" / "scale-forecasting" / "SKILL.md"
    assert skill_path.is_file()
    text = skill_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter = text.split("---\n", 2)[1]
    assert "name: scale-forecasting" in frontmatter
    assert "description:" in frontmatter

    # Progressive disclosure: SKILL.md stays concise (< 300 lines) and links to references/*.md
    lines = text.splitlines()
    assert len(lines) < 300, f"SKILL.md should stay under 300 lines, got {len(lines)}"
    for ref_name in (
        "references/config_reference.md",
        "references/catalog_reference.md",
        "references/execution_paths_and_ops.md",
    ):
        assert ref_name in text
        assert (skill_path.parent / ref_name).is_file()
    for auto_token in (
        "Forecaster.profile_data",
        "Forecaster.build_config",
        "Forecaster.auto",
        "Forecaster.evolve",
        "practitioner_assessment_df",
        "promo_elasticity_df",
        "temporal_reconcile_df",
        "allocate_target",
        "cold_start_transfer_df",
        "similarity_df",
        "find_similar_series_df",
        "cluster_by_similarity_df",
        "recommend_substitutes_df",
        "emerging_trends_df",
        "aci_df",
        "sample_trajectories",
        "tail_risk_df",
        "optimize_inventory_df",
        "optimize_clearance_df",
        "optimize_contract_volume_df",
        "optimize_scenario",
    ):
        assert auto_token in text
    for fm_token in (
        "AlphaEvolve",
        "Double-ML",
        "THieF",
        "Goal-Seek",
        "cold-start NPI",
        "time-series similarity",
        "copula",
        "VaR/CVaR",
        "optimize_clearance_df",
        "optimize_contract_volume_df",
    ):
        assert fm_token in frontmatter

    # Google Antigravity workspace auto-discovery (.agents/skills.json + .agents/skills/ symlink)
    agents_skills_json = REPO_ROOT / ".agents" / "skills.json"
    assert agents_skills_json.is_file()
    skills_cfg = json.loads(agents_skills_json.read_text(encoding="utf-8"))
    assert skills_cfg == {"entries": [{"path": "skills"}]}
    symlinked_skill = REPO_ROOT / ".agents" / "skills" / "scale-forecasting" / "SKILL.md"
    assert symlinked_skill.is_file()
    assert symlinked_skill.resolve() == skill_path.resolve()

    # Google Antigravity plugin.json manifest (no legacy gemini-extension.json or `author` key)
    import tomllib

    expected_ver = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    assert f"version: {expected_ver}" in frontmatter

    assert not (REPO_ROOT / "gemini-extension.json").exists()
    plugin_path = REPO_ROOT / "plugin.json"
    assert plugin_path.is_file()
    plugin = json.loads(plugin_path.read_text(encoding="utf-8"))
    assert plugin["name"] == "scale-forecasting"
    assert plugin["version"] == expected_ver
    assert "author" not in plugin
    assert isinstance(plugin.get("suggestedPrompts"), list)
    assert 1 <= len(plugin["suggestedPrompts"]) <= 3

    # MCP server manifests (Antigravity mcp_config.json and portable project-root .mcp.json)
    for mcp_filename in ("mcp_config.json", ".mcp.json"):
        mcp_path = REPO_ROOT / mcp_filename
        assert mcp_path.is_file()
        mcp_cfg = json.loads(mcp_path.read_text(encoding="utf-8"))
        assert "scale-forecasting" in mcp_cfg["mcpServers"]
        assert mcp_cfg["mcpServers"]["scale-forecasting"]["args"] == ["-m", "scale_forecasting.mcp"]

    tutorial_path = REPO_ROOT / "cloudshell_tutorial.md"
    assert tutorial_path.is_file()
    tutorial_text = tutorial_path.read_text(encoding="utf-8")
    assert "<walkthrough-conclusion-trophy" in tutorial_text


def test_bump_version_updates_all_version_locked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `agent_surfaces.bump_version` updates all 7 version-locked files."""
    for rel in (
        "pyproject.toml",
        "src/scale_forecasting/__init__.py",
        "plugin.json",
        "skills/scale-forecasting/SKILL.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "docker/cloudbuild.yaml",
    ):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)

    updated = agent_surfaces.bump_version("9.8.7", tmp_path, release_date="2026-10-14")
    assert {p.relative_to(tmp_path).as_posix() for p in updated} == {
        "pyproject.toml",
        "src/scale_forecasting/__init__.py",
        "plugin.json",
        "skills/scale-forecasting/SKILL.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "docker/cloudbuild.yaml",
    }
    assert 'version = "9.8.7"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "9.8.7"' in (tmp_path / "src/scale_forecasting/__init__.py").read_text(
        encoding="utf-8"
    )
    assert json.loads((tmp_path / "plugin.json").read_text(encoding="utf-8"))["version"] == "9.8.7"
    assert "  version: 9.8.7" in (tmp_path / "skills/scale-forecasting/SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "## [9.8.7] - 2026-10-14" in (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert 'version: "9.8.7"' in (tmp_path / "CITATION.cff").read_text(encoding="utf-8")
    assert '_VERSION_TAG: "v9.8.7"' in (tmp_path / "docker/cloudbuild.yaml").read_text(
        encoding="utf-8"
    )

    monkeypatch.setattr(agent_surfaces, "_default_repo_root", lambda: tmp_path)
    monkeypatch.setattr(agent_surfaces, "write_agent_surfaces", lambda root=None: [])
    assert agent_surfaces.main(["--bump-version", "9.8.8"]) == 0
    assert 'version = "9.8.8"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid semantic version"):
        agent_surfaces.bump_version("not-a-semver", tmp_path)


def test_variant_tags_and_track_l_enums_have_zero_drift() -> None:
    """R3.3 tripwire: verify `_VARIANT_SYNTAX_ROWS` and `probe_environment()` stay in sync with
    `config.VALID_VARIANT_TAGS`, `config.COLD_START_METHODS`, and `copula.COPULA_TYPES`.
    """
    import re

    from scale_forecasting import config
    from scale_forecasting.copula import COPULA_TYPES

    syntax_text = " ".join(row[0] for row in agent_surfaces._VARIANT_SYNTAX_ROWS)
    documented_tags = set(re.findall(r"<model>@([a-z0-9_]+)", syntax_text))
    assert documented_tags == set(config.VALID_VARIANT_TAGS)

    probe = agent_surfaces.probe_environment()
    assert tuple(probe["auto_forecaster"]["cold_start_archetypes"]) == config.COLD_START_METHODS
    assert tuple(probe["auto_forecaster"]["calibration_methods"]) == config.CALIBRATION_METHODS
    assert "foundation_latent" in probe["auto_forecaster"]["similarity_methods"]
    assert "similarity_kernel" in probe["auto_forecaster"]["evolve_targets"]
    for wb_cap in (
        "time_series_similarity",
        "sku_substitutes",
        "emerging_trends",
        "clearance_optimizer",
        "contract_volume_optimizer",
    ):
        assert wb_cap in probe["auto_forecaster"]["workbench_capabilities"]

    norm_text = " ".join(agent_surfaces._NORMALIZATION_RULES)
    assert "output.include_trajectories" not in norm_text
    assert "optimize_clearance_df" in norm_text
    assert "optimize_contract_volume_df" in norm_text
    for cop_t in COPULA_TYPES:
        assert f'"{cop_t}"' in norm_text


def test_visual_architecture_sync_and_drift_detection(tmp_path: Path) -> None:
    """Verify check_agent_surfaces detects stale counts in docs/visual_architecture.md and
    write_agent_surfaces heals them back to zero drift.
    """
    shutil.copy2(REPO_ROOT / "mkdocs.yml", tmp_path / "mkdocs.yml")
    configs_dir = tmp_path / "configs" / "smokes"
    configs_dir.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "configs" / "mixed_demo.json",
        tmp_path / "configs" / "mixed_demo.json",
    )
    shutil.copy2(
        REPO_ROOT / "configs" / "smokes" / "01_serverless_cpu.json",
        configs_dir / "01_serverless_cpu.json",
    )
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    visual_dst = docs_dir / "visual_architecture.md"
    shutil.copy2(REPO_ROOT / "docs" / "visual_architecture.md", visual_dst)

    written = agent_surfaces.write_agent_surfaces(tmp_path)
    assert len(written) == 6
    assert agent_surfaces.check_agent_surfaces(tmp_path) == []

    # Simulate adding/removing a model or glossary term so Mermaid & inventory counts drift.
    stale_text = visual_dst.read_text(encoding="utf-8").replace(
        "(37 Models Total)", "(99 Models Total)"
    )
    visual_dst.write_text(stale_text, encoding="utf-8")

    drifted = agent_surfaces.check_agent_surfaces(tmp_path)
    assert any(msg.startswith("docs/visual_architecture.md:") for msg in drifted)

    agent_surfaces.write_agent_surfaces(tmp_path)
    assert agent_surfaces.check_agent_surfaces(tmp_path) == []
    assert "(37 Models Total)" in visual_dst.read_text(encoding="utf-8")
