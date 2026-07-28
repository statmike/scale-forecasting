"""Shared test configuration.

Markers are declared in ``pyproject.toml``; ``@gcp``/``@spark``/``@ray``/``@gpu`` tests
are collected but skipped unless their environment is available (wired per phase in Arc B).
"""

from __future__ import annotations

import os

import pytest

# A ``@gcp`` test needs live infra identity in the environment (the same ``SF_*`` vars the
# writers resolve). Absent it, collect-but-skip so the offline gate stays green everywhere.
_GCP_ENV = "SF_PROJECT_ID"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get(_GCP_ENV):
        return
    skip_gcp = pytest.mark.skip(reason=f"@gcp: set {_GCP_ENV} (+ ADC) to run against live infra")
    for item in items:
        if "gcp" in item.keywords:
            item.add_marker(skip_gcp)
