"""Scaffold smoke test — the package imports and reports its version.

Replaced/extended by real unit tests as each capability lands.
"""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import version as pkg_version
from pathlib import Path

import scale_forecasting

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_and_has_single_sourced_version() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = pyproject["project"]["version"]
    assert scale_forecasting.__version__ == expected
    assert pkg_version("scale-forecasting") == expected
    init_src = (_REPO_ROOT / "src" / "scale_forecasting" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert not re.search(r'^__version__\s*(?::\s*str\s*)?=\s*["\']', init_src, re.M), (
        "__version__ in src/scale_forecasting/__init__.py must be derived from importlib.metadata, "
        "not hardcoded"
    )
