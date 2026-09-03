"""Repo-wide static checks on the source tree that ruff does not make.

Two checks, and both exist because of a bug class that is genuinely invisible until it runs on
a launch point that is not a developer's laptop.

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
