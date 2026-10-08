"""Structural tripwires for the API Reference (`docs/api/`) and folder README spine.

Guarantees that:
1. Every module re-exported by `scale_forecasting.__init__` (eager + `_LAZY`) has a corresponding
   `::: scale_forecasting.<module>` directive in `docs/api/*.md`.
2. Every `::: scale_forecasting...` directive across `docs/api/*.md` resolves to a real source file
   under `src/scale_forecasting/`.
3. Every `docs/api/*.md` page is wired into `mkdocs.yml` (and vice versa).
4. Every `scale_forecasting` subpackage is represented in `docs/api/` and carries its own
   `README.md` overview with a Mermaid diagram.
"""

from __future__ import annotations

import re
from pathlib import Path

import scale_forecasting

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PKG = REPO_ROOT / "src" / "scale_forecasting"
DOCS_API = REPO_ROOT / "docs" / "api"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"

_DIRECTIVE_RE = re.compile(r"^:::\s+(scale_forecasting(?:\.[a-zA-Z0-9_]+)+)\s*$", re.MULTILINE)
_NAV_API_RE = re.compile(r":\s+(api/[a-zA-Z0-9_]+\.md)\s*$", re.MULTILINE)


def _documented_modules() -> set[str]:
    modules: set[str] = set()
    for md in DOCS_API.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        modules.update(_DIRECTIVE_RE.findall(text))
    return modules


def _module_to_source_path(dotted: str) -> Path:
    parts = dotted.split(".")
    assert parts[0] == "scale_forecasting"
    rel = Path(*parts[1:])
    as_file = SRC_PKG / f"{rel}.py"
    if as_file.is_file():
        return as_file
    return SRC_PKG / rel / "__init__.py"


def test_every_api_directive_resolves_to_source_file() -> None:
    documented = _documented_modules()
    assert documented, "Expected ::: scale_forecasting.* directives in docs/api/*.md"
    missing = [mod for mod in sorted(documented) if not _module_to_source_path(mod).is_file()]
    assert not missing, f"docs/api/ directives do not resolve to source files: {missing}"


def test_every_root_exported_module_is_in_api_docs() -> None:
    documented = _documented_modules()
    eager_modules = {
        "scale_forecasting.config",
        "scale_forecasting.errors",
        "scale_forecasting.settings",
    }
    lazy_modules = {
        f"scale_forecasting{mod_rel}" for mod_rel, _attr in scale_forecasting._LAZY.values()
    }
    required = eager_modules | lazy_modules
    missing = sorted(required - documented)
    assert not missing, (
        f"Modules exported by scale_forecasting.__init__ are missing from docs/api/: {missing}"
    )


def test_docs_api_pages_and_mkdocs_nav_stay_in_sync() -> None:
    on_disk = {f"api/{p.name}" for p in DOCS_API.glob("*.md")}
    in_nav = set(_NAV_API_RE.findall(MKDOCS_YML.read_text(encoding="utf-8")))
    assert on_disk == in_nav, (
        f"docs/api/*.md and mkdocs.yml nav differ: "
        f"unlisted={sorted(on_disk - in_nav)}, missing_files={sorted(in_nav - on_disk)}"
    )


def test_every_subpackage_is_documented_and_has_readme() -> None:
    documented = _documented_modules()
    subpackages = sorted(
        p.name for p in SRC_PKG.iterdir() if p.is_dir() and (p / "__init__.py").is_file()
    )
    assert subpackages == [
        "data_gen",
        "engines",
        "metrics",
        "models",
        "probes",
        "profiling",
        "registry",
        "resources",
    ]
    for subpkg in subpackages:
        prefix = f"scale_forecasting.{subpkg}"
        assert any(m == prefix or m.startswith(f"{prefix}.") for m in documented), (
            f"Subpackage {prefix} has no ::: directive in docs/api/"
        )
        readme = SRC_PKG / subpkg / "README.md"
        assert readme.is_file(), f"Missing subpackage README: {readme}"
        assert "```mermaid" in readme.read_text(encoding="utf-8"), (
            f"Expected a Mermaid diagram in {readme}"
        )


def test_folder_readme_spine_has_mermaid_diagrams() -> None:
    spine = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "configs" / "README.md",
        REPO_ROOT / "configs" / "smokes" / "README.md",
        REPO_ROOT / "docs" / "README.md",
        REPO_ROOT / "notebooks" / "README.md",
        REPO_ROOT / "skills" / "README.md",
        REPO_ROOT / "terraform" / "README.md",
        REPO_ROOT / "docker" / "README.md",
        REPO_ROOT / "src" / "scale_forecasting" / "README.md",
        REPO_ROOT / "tests" / "README.md",
    ]
    for readme in spine:
        assert readme.is_file(), f"Missing folder README: {readme}"
        assert "```mermaid" in readme.read_text(encoding="utf-8"), (
            f"Expected a Mermaid diagram in {readme}"
        )
