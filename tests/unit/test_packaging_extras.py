"""The dependency-extras contract: what a bare install is, and that every surface agrees on it.

``pip install scale-forecasting`` is the pure forecasting layer — models, metrics, backtests, the
playground — with no Google Cloud client, no notebook kernel, no plotting library. Everything that
reaches Google Cloud sits behind ``[gcp]``; the runtime clients that are not Google libraries sit
behind ``[spark]`` and ``[ray]``; plotting behind ``[notebook]``; the model libraries behind the
``[models-*]`` family extras. Four other surfaces restate that layout — the container build, the
``docker/requirements.txt`` export, the CI lock check, and the notebooks' Colab bootstrap cells —
and the docs name the extras in install lines. These tests read the one source of truth
(``pyproject.toml``) and fail the moment any of those drift from it, so an install line in the
README can never name an extra pip cannot resolve (which is how ``[models-automl]`` spent a while
documented but undeclared).

The same file holds two facts of the same kind about the offline gate — that ``make test``,
``make ci-offline``, and the CI ``offline`` job run one identical pytest command, and that the
coverage floor is declared once, in ``[tool.coverage.report]`` — because they are also build and CI
surfaces restating ``pyproject.toml``.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))

# Distributions the pure layer must never pull: Google clients, the kernel, plotting, runtimes.
_NOT_IN_CORE = re.compile(
    r"^(google-|db-dtypes|protobuf|ipykernel|matplotlib|pyspark|ray\b|torch|xgboost|lightgbm|"
    r"catboost|prophet|neuralprophet|neuralforecast|statsforecast)"
)
# The Google clients `[gcp]` must carry — each is imported by name somewhere under src/.
_GCP_CLIENTS = {
    "google-cloud-bigquery",
    "google-cloud-bigquery-storage",
    "google-cloud-storage",
    "google-cloud-dataproc",
    "google-cloud-aiplatform",
    "google-auth",
    "google-api-core",
    "db-dtypes",
    "protobuf",
}
# The notebooks that drive Google Cloud (everything but the two offline sandboxes).
_OFFLINE_NOTEBOOKS = {"00_model_playground", "09_custom_models_and_metrics"}
_SELF_EXTRA = re.compile(r"^scale-forecasting\[([a-z0-9,\- ]+)\]$")


def _project() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _extras() -> dict[str, list[str]]:
    return _project()["project"]["optional-dependencies"]


def _names(requirements: list[str]) -> set[str]:
    """Distribution names in a requirement list, self-references excluded."""
    return {Requirement(r).name for r in requirements if not _SELF_EXTRA.match(r)}


def _self_refs(requirements: list[str]) -> set[str]:
    """The extras a requirement list pulls in through `scale-forecasting[...]` self-references."""
    refs: set[str] = set()
    for r in requirements:
        if m := _SELF_EXTRA.match(r):
            refs.update(e.strip() for e in m.group(1).split(","))
    return refs


def _closure(extra: str, extras: dict[str, list[str]]) -> set[str]:
    """Every distribution `extra` installs, following self-references."""
    seen: set[str] = set()
    todo = [extra]
    names: set[str] = set()
    while todo:
        e = todo.pop()
        if e in seen:
            continue
        seen.add(e)
        names |= _names(extras[e])
        todo.extend(_self_refs(extras[e]))
    return names


def _extra_flags(text: str) -> set[str]:
    """The `--extra <name>` flags in a shell/Make/YAML snippet."""
    return set(re.findall(r"--extra\s+([a-z0-9\-]+)", text))


# --- the layout ---------------------------------------------------------------------------------


def test_core_dependencies_are_the_pure_layer_only() -> None:
    core = _names(_project()["project"]["dependencies"])
    offenders = sorted(n for n in core if _NOT_IN_CORE.match(n))
    assert not offenders, f"core `dependencies` must stay Google- and runtime-free: {offenders}"


def test_gcp_extra_carries_every_google_client_the_package_imports() -> None:
    missing = _GCP_CLIENTS - _names(_extras()["gcp"])
    assert not missing, f"[gcp] is missing clients src/ imports by name: {sorted(missing)}"


def test_no_other_extra_redeclares_a_gcp_client() -> None:
    # `[spark]`, `[ray]`, `[models-automl]` reach the clients through `scale-forecasting[gcp]`;
    # a second copy of a pin is the drift the composition exists to prevent.
    extras = _extras()
    for extra, reqs in extras.items():
        if extra == "gcp":
            continue
        dup = _GCP_CLIENTS & _names(reqs)
        assert not dup, f"[{extra}] redeclares [gcp] pins: {sorted(dup)}"


@pytest.mark.parametrize("extra", ["spark", "ray", "models-automl"])
def test_cloud_runtime_extras_include_gcp(extra: str) -> None:
    assert "gcp" in _self_refs(_extras()[extra]), f"[{extra}] must include scale-forecasting[gcp]"


def test_submit_is_the_thin_launch_client_alias() -> None:
    # The operations runbook says `uv sync --extra submit`; with every Google launch client in
    # [gcp] and Ray's in [ray], the thin client is [ray] spelled under the name operators know.
    assert _extras()["submit"] == ["scale-forecasting[ray]"]


def test_models_is_composed_from_the_family_extras() -> None:
    extras = _extras()
    families = {e for e in extras if e.startswith("models-")}
    assert _self_refs(extras["models"]) == families
    assert not _names(extras["models"]), "[models] must declare no pins of its own"


def test_every_version_floor_is_declared_exactly_once() -> None:
    extras = _extras()
    where: dict[str, list[str]] = {}
    for extra, reqs in extras.items():
        for name in _names(reqs):
            where.setdefault(name, []).append(extra)
    dup = {n: e for n, e in where.items() if len(e) > 1}
    assert not dup, f"pins declared in more than one extra (compose instead): {dup}"


def test_all_covers_every_other_extra() -> None:
    extras = _extras()
    assert _closure("all", extras) == set().union(
        *(_closure(e, extras) for e in extras if e != "all")
    )


def test_dev_group_restores_a_cloud_capable_venv() -> None:
    # `uv sync` (core + dev) must still yield a venv that can reach BigQuery and plot — the
    # notebooks and the live tests depend on that — now that core is the pure layer.
    dev = _project()["dependency-groups"]["dev"]
    assert {"gcp", "notebook"} <= _self_refs([r for r in dev if isinstance(r, str)])


# --- the surfaces that restate it ---------------------------------------------------------------


def test_container_export_and_ci_share_one_extras_set_that_includes_gcp() -> None:
    makefile = _extra_flags((ROOT / "Makefile").read_text())
    ci = _extra_flags((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    dockerfile = _extra_flags((ROOT / "docker" / "Dockerfile").read_text())
    header = _extra_flags((ROOT / "docker" / "requirements.txt").read_text().splitlines()[1])
    assert makefile == ci == dockerfile == header, {
        "Makefile": makefile,
        "ci.yml": ci,
        "Dockerfile": dockerfile,
        "requirements.txt": header,
    }
    assert "gcp" in makefile, "the runtime image must carry the Google clients"
    assert makefile <= set(_extras()), f"undeclared extras in the export flags: {makefile}"


def test_offline_pytest_command_is_the_same_in_make_and_ci() -> None:
    # `make test`, `make ci-offline`, and the CI `offline` job must run one pytest invocation —
    # same marker selection, same `--cov` flags — or "the CI job, step for step" stops being true.
    makefile = (ROOT / "Makefile").read_text()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    make_cmd = re.search(r"^OFFLINE_PYTEST\s*:=\s*(.+)$", makefile, re.M)
    ci_cmd = re.search(r"- name: offline test gate\n\s+run: uv run (.+)$", ci, re.M)
    assert make_cmd and ci_cmd, "could not find the offline pytest command in both files"
    assert make_cmd.group(1).strip() == ci_cmd.group(1).strip()
    assert "--cov" in make_cmd.group(1), "the offline gate must measure coverage"
    assert makefile.count("$(OFFLINE_PYTEST)") == 2, "both `test` and `ci-offline` must use it"


def test_coverage_floor_is_declared_once_in_pyproject_and_is_a_ratchet() -> None:
    # The floor lives in [tool.coverage.report] so there is one number to raise; a
    # `--cov-fail-under` on a command line would be a second one. 85 is where the ratchet
    # started (85.77 % measured).
    report = _project()["tool"]["coverage"]["report"]
    assert report["fail_under"] >= 85, "the coverage floor only moves up"
    assert _project()["tool"]["coverage"]["run"]["source"] == ["scale_forecasting"]
    for rel in ("Makefile", ".github/workflows/ci.yml"):
        text = (ROOT / rel).read_text()
        assert "--cov-fail-under" not in text, f"{rel}: the floor belongs in pyproject"
    addopts = _project()["tool"]["pytest"]["ini_options"].get("addopts", "")
    assert "--cov" not in addopts, "addopts would break the bare-venv core-install job"


def test_documented_install_lines_name_declared_extras() -> None:
    declared = set(_extras())
    root_md = [
        ROOT / name
        for name in (
            "README.md",
            "AGENTS.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "SECURITY.md",
            "CHANGELOG.md",
        )
        if (ROOT / name).exists()
    ]
    pages = [*root_md, *(ROOT / "docs").glob("*.md")]
    for tree in ("configs", "docker", "notebooks", "src", "terraform", "tests"):
        pages += list((ROOT / tree).rglob("README.md"))
    bad: dict[str, set[str]] = {}
    for page in pages:
        text = page.read_text()
        named = set()
        for group in re.findall(r"scale-forecasting\[([a-z0-9,\- ]+)\]", text):
            named.update(e.strip() for e in group.split(","))
        named |= _extra_flags(text)
        if undeclared := named - declared:
            bad[str(page.relative_to(ROOT))] = undeclared
    assert not bad, f"docs name extras pyproject does not declare: {bad}"


def test_pypi_metadata_and_urls_declared() -> None:
    proj = _project()["project"]
    assert proj["version"] == "1.0.0"
    assert set(proj["urls"]) >= {"Homepage", "Documentation", "Repository", "Changelog", "Issues"}
    classifiers = set(proj["classifiers"])
    assert "Development Status :: 5 - Production/Stable" in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Typing :: Typed" in classifiers
    assert {"time-series", "forecasting", "google-cloud", "bigquery"} <= set(proj["keywords"])


def _bootstrap_extras(notebook: Path) -> list[str]:
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if m := re.search(r"^EXTRAS\s*=\s*(\[.*?\])\s*$", source, re.M):
            return json.loads(m.group(1).replace("'", '"'))
    raise AssertionError(f"{notebook.name}: no bootstrap cell with an EXTRAS line")


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_bootstrap_extras_are_declared_and_cloud_notebooks_lock_the_clients(
    notebook: Path,
) -> None:
    extras = _bootstrap_extras(notebook)
    assert set(extras) <= set(_extras()), f"{notebook.name} bootstraps undeclared extras: {extras}"
    if notebook.stem in _OFFLINE_NOTEBOOKS:
        assert "gcp" not in extras, f"{notebook.name} is the offline sandbox; keep it client-free"
    else:
        # The bootstrap exists to install the LOCKED dependency set; a cloud notebook that leaves
        # [gcp] out would silently run against whatever client versions the runtime image has.
        assert "gcp" in extras, f"{notebook.name} drives Google Cloud but does not lock [gcp]"


# --- the models reference restates the model → extra mapping ------------------------------------


def _models_reference_table() -> dict[str, set[str]]:
    """extra → model names, as the "Granular installation extras" table in the reference says."""
    from scale_forecasting.models import list_models

    known = set(list_models())
    text = (ROOT / "docs" / "models_reference.md").read_text()
    section = text.split("### Granular installation extras", 1)[1].split("\n### ", 1)[0]
    table: dict[str, set[str]] = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0].startswith(":") or cells[0] == "Extra":
            continue
        models = {m for m in re.findall(r"`([a-z0-9_]+)`", cells[2]) if m in known}
        if not models:
            continue  # the "[models] = everything" row names no individual model
        key = "core" if "core" in cells[0] else re.search(r"\[([a-z\-]+)\]", cells[0]).group(1)
        table[key] = models
    return table


def test_models_reference_extras_table_matches_the_models_own_declarations() -> None:
    # Each model declares `optional_extra`; the reference table is prose that must say the same.
    # This is the table that once filed eleven statsmodels models under an extra they never needed.
    from scale_forecasting.models import get_model, list_models

    declared: dict[str, set[str]] = {}
    for name in list_models():
        declared.setdefault(get_model(name).optional_extra or "core", set()).add(name)
    assert _models_reference_table() == declared
