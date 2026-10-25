"""Repo-wide static checks on the source tree that ruff does not make.

Four checks. The first two exist because of a bug class that is genuinely invisible until it
runs on a launch point that is not a developer's laptop; the last two keep a deliberate
duplication and a naming convention honest (see their own docstrings at the bottom).

The first is a relative import at the **wrong level**. Ruff resolves undefined names and unused
imports, not whether ``from .models import get_model`` names a module that exists. When that
import sits at the top of a file, any test that imports the file catches it immediately. When
it sits inside a function body — which this codebase does deliberately, to keep heavy model,
Spark and BigQuery imports off the submit path — nothing catches it until that branch runs,
and the branch that runs it is often on a cluster, mid-job, an hour in.

Moving a module into a package is exactly when this breaks: every ``from .x import y`` in the
moved code silently starts meaning something else. The check is a few lines of ``ast`` and it
covers the whole tree, so it costs nothing to keep and pays for itself on the first move.

The second is a module whose name **shadows a stdlib module**. Inside a package this is normally
harmless — ``scale_forecasting.profiling.numbers`` and stdlib ``numbers`` coexist fine, because
only the package root is ever on ``sys.path``. It stops being harmless the moment something puts
an inner directory on ``sys.path``, and a Composer plugins delivery does exactly that: a live
Airflow run died on ``cannot import name 'Number' from 'numbers'`` because a *third-party*
library's ``import numbers`` resolved to our file. Nothing local reproduces it — the same code
imports cleanly from a repo checkout — so a static name check is the only cheap guard.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "scale_forecasting"


def _relative_imports() -> list[tuple[Path, int, str, int]]:
    """Every relative import in the source tree as ``(file, lineno, module, level)``."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                found.append((path, node.lineno, node.module or "", node.level))
    return found


def _resolves(path: Path, module: str, level: int) -> bool:
    """Does ``from <'.' * level><module> import ...`` inside ``path`` name something on disk?"""
    # `level` 1 means "this file's own package", 2 means its parent, and so on.
    base = path.parent
    for _ in range(level - 1):
        base = base.parent
    target = base.joinpath(*module.split(".")) if module else base
    return (target / "__init__.py").is_file() or target.with_suffix(".py").is_file()


def test_every_relative_import_resolves_to_a_module_that_exists() -> None:
    """A wrong-level relative import inside a lazy function body must fail here, not in a job.

    The failure message names file, line and the exact import, because the fix is always to
    add or drop a dot and the only hard part is finding which one.
    """
    broken = [
        f"{path.relative_to(SRC)}:{lineno}: from {'.' * level}{module} import ..."
        for path, lineno, module, level in _relative_imports()
        if not _resolves(path, module, level)
    ]
    assert not broken, "relative imports that name no module:\n  " + "\n  ".join(broken)


def test_the_check_would_notice_a_wrong_level_import() -> None:
    """Guard the guard: a resolver that always returns True would pass the test above silently."""
    package_module = SRC / "profiling" / "cost.py"
    assert _resolves(package_module, "models", 2), "`..models` from inside profiling/ is real"
    assert not _resolves(package_module, "models", 1), "`.models` from inside profiling/ is not"


def test_the_source_tree_actually_has_relative_imports_to_check() -> None:
    """Guard the guard: an rglob that matched nothing would also report zero breakage."""
    assert len(_relative_imports()) > 50


def test_no_module_name_shadows_a_stdlib_module() -> None:
    # See the module docstring: a Composer plugins delivery puts inner directories on sys.path, so
    # `profiling/numbers.py` made an unrelated library's `import numbers` resolve to our file and
    # killed a live Ray submit with `cannot import name 'Number' from 'numbers'`. A repo checkout
    # never reproduces it, so nothing but this check stands between us and the next one.
    offenders = [
        f"{path.relative_to(SRC.parent)} shadows stdlib '{path.stem}'"
        for path in sorted(SRC.rglob("*.py"))
        if path.stem in sys.stdlib_module_names
    ]
    assert not offenders, "rename these — stdlib-shadowing module names:\n" + "\n".join(offenders)


def test_the_shadowing_check_knows_what_a_stdlib_name_looks_like() -> None:
    # Guard the guard: if `sys.stdlib_module_names` ever came back empty the check above would pass
    # vacuously and we would learn about it on a cluster instead of here.
    assert {"numbers", "json", "types"} <= sys.stdlib_module_names


# --- the deliberately-duplicated terminal-status set ---------------------------


def test_every_copy_of_the_terminal_status_set_agrees() -> None:
    """Four modules each keep their own copy of "a run/job has stopped changing". They must match.

    The duplication is on purpose and stays: `probes` is imported low and `sdk` high, so a shared
    constant would mean importing the SDK from the probe layer (and the registry layer from the
    probes package) purely to name one frozenset. Each site says so in its own comment.

    What duplication costs is drift, and drift here is not cosmetic — the four copies answer
    "should I keep waiting?", "should I probe the runtime?", "may I close this header?", and "did
    this family finish?". A status added to one and not the others makes `Forecaster.wait` hang on
    a run the registry considers settled, or lets close-runs write a verdict on a live job. This
    test is the cheap thing that makes the deliberate copy safe; it is not a hint to merge them.
    """
    from scale_forecasting.airflow_tasks import _TERMINAL_STATUSES as airflow_set
    from scale_forecasting.probes.vocabulary import _TERMINAL as probe_set
    from scale_forecasting.registry.ops import _TERMINAL_JOB_STATUSES as ops_set
    from scale_forecasting.sdk import _TERMINAL_STATUSES as sdk_set

    assert sdk_set == airflow_set == probe_set == ops_set
    # Named outright rather than compared to each other alone: four copies of the *wrong* set would
    # also be equal, and CANCELLED is the member most likely to be dropped by someone reasoning that
    # a stopped run "never finished".
    assert sdk_set == frozenset({"COMPLETED", "FAILED", "PARTIAL", "CANCELLED"})


# --- cross-module imports of underscore-prefixed names (a ratchet) ---------------------------

# Every `from .somewhere import _name` in the tree today, as `<file> <relative module> <name>`.
# The test below fails on any entry that is NOT here (so the list can only shrink) and on any
# entry here that no longer exists (so it is never stale). To clear a failure, promote the helper
# — drop the underscore, give it a docstring — or move it to a shared module; do not add here.
# Burned down from 92 to 0 in Track F.2 (v1.8.2): zero private cross-module imports are permitted.
_PRIVATE_CROSS_IMPORTS = ""


def _parse_private_cross_imports(block: str) -> set[tuple[str, str, str]]:
    """The allowlist block as ``{(file, relative_module, name)}``; blank lines ignored."""
    found: set[tuple[str, str, str]] = set()
    for line in block.splitlines():
        if line.strip():
            file, module, name = line.split()
            found.add((file, module, name))
    return found


def _private_cross_imports() -> set[tuple[str, str, str]]:
    """Every relative import of an underscore-prefixed *name* in the tree.

    Importing an underscore-prefixed *module* (``from . import _lag_forecaster``) is not counted:
    that is how a package keeps a shared implementation file out of its public surface, and the
    module's own leading underscore is the convention working as intended.
    """
    found: set[tuple[str, str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level):
                continue
            module = node.module or ""
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            target = base.joinpath(*module.split(".")) if module else base
            for alias in node.names:
                if not alias.name.startswith("_"):
                    continue
                if target.is_dir() and (target / f"{alias.name}.py").is_file():
                    continue  # a private module, not a private name
                found.add((str(path.relative_to(SRC)), "." * node.level + module, alias.name))
    return found


def test_cross_module_private_imports_only_ratchet_down() -> None:
    """A leading underscore means "not for use outside this module"; importing it elsewhere says
    otherwise. Every such import is a small lie about the module's surface, and the honest fix is
    cheap: drop the underscore (the helper is shared, so say so) or move it to a module whose job
    is to be shared. The list above was burned down to zero in v1.8.2 and must remain empty.
    """
    live = _private_cross_imports()
    allowed = _parse_private_cross_imports(_PRIVATE_CROSS_IMPORTS)

    added = sorted(live - allowed)
    assert not added, (
        "new cross-module imports of underscore-prefixed names:\n  "
        + "\n  ".join(" ".join(entry) for entry in added)
        + "\n\nPromote the helper (drop the underscore, add a docstring) or move it to a shared"
        " module. Do not add it to _PRIVATE_CROSS_IMPORTS; that list is permanently zero."
    )

    removed = sorted(allowed - live)
    assert not removed, (
        "allowlist entries that no longer exist — delete them from _PRIVATE_CROSS_IMPORTS so the"
        " list stays an exact picture of the remaining debt:\n  "
        + "\n  ".join(" ".join(entry) for entry in removed)
    )


def test_the_ratchet_sees_private_names_but_not_private_modules() -> None:
    """Guard the guard: the walker must skip private module imports
    (``from . import _lag_forecaster``) that the lag-based models rely on, while
    `_parse_private_cross_imports` parses 3-tuples.
    """
    live = _private_cross_imports()
    assert live == set()
    assert not any(name == "_lag_forecaster" for _, _, name in live)
    sample = _parse_private_cross_imports("worker.py .resources.catalog _INTRAOP_ENV_VARS")
    assert sample == {("worker.py", ".resources.catalog", "_INTRAOP_ENV_VARS")}


# --- Track F.2 Anti-"Vibe-Code" & Enterprise Hardening AST Tripwires ---------------------------


def test_no_silent_except_pass() -> None:
    """Forbid bare `except ...: pass` in production source (`src/scale_forecasting/`).

    Every exception handler must either log with context, record a structured diagnostic, re-raise,
    or use `contextlib.suppress(SpecificException)` when suppression is genuinely the intent.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, "silent `except ...: pass` blocks in src/:\n  " + "\n  ".join(offenders)


def test_no_local_file_workarounds_or_hardcoded_tmp() -> None:
    """Forbid hardcoded `/tmp/` file paths in Python runtime logic (`use tempfile` instead).

    Allowed exceptions:
    - `cluster_deps.py` and `gce_submit.py`: bash startup scripts executed inside remote VMs.
    - `agent_surfaces.py` and `benchmarks.py`: CLI docstring/markdown examples.
    """
    allowed_files = {
        "cluster_deps.py",
        "gce_submit.py",
        "agent_surfaces.py",
        "benchmarks.py",
    }
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        if rel in allowed_files:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("/tmp/")
            ):
                offenders.append(f"{rel}:{node.lineno}: {node.value!r}")
    assert not offenders, "hardcoded `/tmp/` literals in src/:\n  " + "\n  ".join(offenders)


def test_no_sys_path_or_monkeypatch_in_src() -> None:
    """Forbid `sys.path` mutation outside `code_delivery.py` and `unittest.mock` in `src/`."""
    sys_path_offenders: list[str] = []
    mock_offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        text = path.read_text(encoding="utf-8")
        if "sys.path" in text and rel != "code_delivery.py":
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "path"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "sys"
                ):
                    sys_path_offenders.append(f"{rel}:{node.lineno}")
        if "unittest.mock" in text or "monkeypatch" in text:
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "unittest.mock"
                ):
                    mock_offenders.append(f"{rel}:{node.lineno}")
    assert not sys_path_offenders, (
        "`sys.path` access outside `code_delivery.py`:\n  " + "\n  ".join(sys_path_offenders)
    )
    assert not mock_offenders, "`unittest.mock` imports in src/:\n  " + "\n  ".join(mock_offenders)


def test_no_eval_or_unsandboxed_exec() -> None:
    """Forbid `eval()` anywhere in `src/` and restrict `exec()` to `evolve/sandbox.py`."""
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "eval":
                    offenders.append(f"{rel}:{node.lineno}: eval()")
                elif node.func.id == "exec" and rel != "evolve/sandbox.py":
                    offenders.append(f"{rel}:{node.lineno}: exec() outside evolve/sandbox.py")
    assert not offenders, "unsafe eval()/exec() calls in src/:\n  " + "\n  ".join(offenders)


def test_hub_module_line_count_ceilings() -> None:
    """Enforce Anti-Bloat Guardrail 3: hub modules (`sdk.py`, `playground.py`, `workbench.py`) must
    remain below 3,500 lines.
    """
    ceilings = {
        "sdk.py": 3500,
        "playground.py": 3500,
        "workbench.py": 3500,
    }
    offenders: list[str] = []
    for rel, max_lines in ceilings.items():
        n_lines = len((SRC / rel).read_text(encoding="utf-8").splitlines())
        if n_lines >= max_lines:
            offenders.append(f"{rel}: {n_lines} lines >= ceiling {max_lines}")
    assert not offenders, "hub modules exceeding line-count ceiling:\n  " + "\n  ".join(offenders)
