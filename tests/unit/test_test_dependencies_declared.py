"""Every third-party module the test-suite imports must be in the CI ``offline`` environment.

Why this file exists: on 2026-10-06 ``main`` went red with ``ModuleNotFoundError: No module named
'pymdownx'`` from ``test_docs_integrity.py``, and it stayed red for two merges because no local
environment could reproduce it. The package was present at every desk — ``make docs`` had pulled it
in through the ``docs`` dependency group — but the CI gate syncs ``--all-extras`` *without* that
group, so the one machine whose verdict counts was the one machine that lacked it. A test can only
be as portable as the dependencies it declares, and "it imports fine here" is not a declaration.

What it checks: the closure of what ``uv sync --frozen --all-extras`` installs, computed from
``uv.lock`` itself (the project's core dependencies, every extra, and the default dependency
groups, followed through each package's own ``dependencies`` and any requested
``optional-dependencies``), against every third-party top-level module that any file under
``tests/`` imports — module-level or inside a function, since the failure above was a lazy import.
A module whose distribution is outside that closure fails here, at the desk, in under a second.

What it deliberately leaves alone:

* imports guarded by ``try: ... except ImportError`` or preceded by ``pytest.importorskip`` for the
  same top-level name — those are optional by construction and skip cleanly where absent;
* environment markers — the closure is the union over platforms, which is a superset of any one
  runner. That can only make the check more lenient, never produce a false alarm.

The fix for a failure is one of two things, and the message says which: declare the distribution
in ``[dependency-groups].dev`` (or an extra the gate installs) in ``pyproject.toml`` and re-lock, or
make the import optional in the test that needs it.
"""

from __future__ import annotations

import ast
import importlib.metadata
import sys
import tomllib
from collections import deque
from pathlib import Path

from packaging.utils import canonicalize_name

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK = _REPO_ROOT / "uv.lock"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_TESTS = _REPO_ROOT / "tests"
_PROJECT = "scale-forecasting"

# Top-level names that are never third-party distributions: the package under test, the test tree
# itself, and the interpreter's own modules. Sibling test modules (``smoke_harness``,
# ``test_validation_ledger``, …) are importable by bare name because pytest puts each test file's
# directory on ``sys.path``, so every ``.py`` stem and package directory under ``tests/`` counts
# too.
_FIRST_PARTY = frozenset(
    {"scale_forecasting", "tests", "conftest"}
    | {p.stem for p in _TESTS.rglob("*.py") if "__pycache__" not in p.parts}
    | {p.name for p in _TESTS.rglob("*") if p.is_dir() and "__pycache__" not in p.parts}
)


def _default_groups() -> list[str]:
    """The groups ``uv sync`` installs when none are named: ``dev`` unless overridden."""
    tool_uv = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8")).get("tool", {}).get("uv", {})
    return list(tool_uv.get("default-groups", ["dev"]))


def ci_offline_closure() -> set[str]:
    """Distribution names ``uv sync --frozen --all-extras`` installs, per ``uv.lock``.

    Mirrors the ``offline`` job in ``.github/workflows/ci.yml`` and the Makefile ``sync`` target:
    the project's core dependencies, every extra, and the default groups, followed transitively.
    """
    lock = tomllib.loads(_LOCK.read_text(encoding="utf-8"))
    packages = {pkg["name"]: pkg for pkg in lock["package"]}
    project = packages[_PROJECT]

    roots: list[dict] = list(project.get("dependencies", []))
    for extra_deps in project.get("optional-dependencies", {}).values():
        roots.extend(extra_deps)
    for group in _default_groups():
        roots.extend(project.get("dev-dependencies", {}).get(group, []))

    closure: set[str] = set()
    seen: set[tuple[str, tuple[str, ...]]] = set()
    queue: deque[tuple[str, tuple[str, ...]]] = deque(
        (dep["name"], tuple(dep.get("extra", []))) for dep in roots
    )
    while queue:
        name, extras = queue.popleft()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        closure.add(name)
        pkg = packages.get(name)
        if pkg is None:  # pragma: no cover - a lock that references an absent package is corrupt
            continue
        for dep in pkg.get("dependencies", []):
            queue.append((dep["name"], tuple(dep.get("extra", []))))
        for extra in extras:
            for dep in pkg.get("optional-dependencies", {}).get(extra, []):
                queue.append((dep["name"], tuple(dep.get("extra", []))))
    return closure


def _guarded_import_nodes(tree: ast.AST) -> set[int]:
    """``id()`` of every import node inside a ``try`` whose handlers swallow an ImportError."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        swallows = False
        for handler in node.handlers:
            if handler.type is None:
                swallows = True
                break
            names = (
                [handler.type]
                if not isinstance(handler.type, ast.Tuple)
                else list(handler.type.elts)
            )
            if any(
                isinstance(n, ast.Name) and n.id in {"ImportError", "ModuleNotFoundError"}
                for n in names
            ):
                swallows = True
                break
        if swallows:
            for inner in node.body:
                for sub in ast.walk(inner):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        guarded.add(id(sub))
    return guarded


def _importorskip_roots(tree: ast.AST) -> set[str]:
    """Top-level names a file declares optional via ``pytest.importorskip("name[.sub]")``."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "importorskip"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            roots.add(node.args[0].value.split(".")[0])
    return roots


def third_party_imports() -> dict[str, list[str]]:
    """Map each third-party top-level module imported under ``tests/`` to ``file:line`` sites."""
    sites: dict[str, list[str]] = {}
    for path in sorted(_TESTS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guarded = _guarded_import_nodes(tree)
        optional = _importorskip_roots(tree)
        rel = path.relative_to(_REPO_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            if id(node) in guarded:
                continue
            for dotted in names:
                top = dotted.split(".")[0]
                if top in _FIRST_PARTY or top in sys.stdlib_module_names or top in optional:
                    continue
                sites.setdefault(top, []).append(f"{rel}:{node.lineno}")
    return sites


def test_every_test_import_is_installed_by_the_ci_offline_sync() -> None:
    """A module the tests import by name is a distribution the CI gate installs by name."""
    closure = ci_offline_closure()
    module_to_dists = importlib.metadata.packages_distributions()

    errors: list[str] = []
    for module, sites in sorted(third_party_imports().items()):
        dists = [canonicalize_name(d) for d in module_to_dists.get(module, [])]
        where = ", ".join(sites[:3]) + (" …" if len(sites) > 3 else "")
        if not dists:
            errors.append(
                f"{module!r} (imported at {where}) is not installed in this environment, so its "
                "distribution cannot be checked against uv.lock — install it, or make the import "
                "optional (try/except ImportError or pytest.importorskip)."
            )
        elif not any(d in closure for d in dists):
            errors.append(
                f"{module!r} (imported at {where}) comes from {sorted(set(dists))}, which "
                "`uv sync --frozen --all-extras` does not install — CI's offline gate will raise "
                "ModuleNotFoundError. Declare it in [dependency-groups].dev (or an extra) in "
                "pyproject.toml and run `make lock`, or make the import optional."
            )
    assert not errors, "test-suite imports outside the CI offline environment:\n" + "\n".join(
        errors
    )


def test_the_closure_reads_the_lock_the_way_ci_syncs_it() -> None:
    """Sanity pins for the closure, so a quiet lock-format change cannot make the guard a no-op."""
    closure = ci_offline_closure()
    # Core dependencies and the dev group are always in.
    assert {"pandas", "pydantic", "pytest", "ruff"} <= closure
    # Extras are in (``--all-extras``): the Spark and Ray clients.
    assert {"pyspark", "ray"} <= closure
    # A transitive dependency reached only through a package's requested extra is in.
    assert "grpcio" in closure  # google-api-core[grpc]
    # The docs group is NOT in — that is precisely the gap this file guards.
    assert "mkdocs" not in closure
