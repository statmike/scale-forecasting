"""Scaffold smoke test — the package imports and reports its version.

Replaced/extended by real unit tests as each capability lands.
"""

from __future__ import annotations

import scale_forecasting


def test_package_imports_and_has_version() -> None:
    assert scale_forecasting.__version__
