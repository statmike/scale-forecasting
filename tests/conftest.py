"""Shared test configuration.

Markers are declared in ``pyproject.toml``; ``@gcp``/``@spark``/``@ray``/``@gpu`` tests
are collected but skipped unless their environment is available (wired per phase in Arc B).
"""

from __future__ import annotations

import importlib.util
import os

import pytest

# A ``@gcp`` test needs live infra identity in the environment (the same ``SF_*`` vars the
# writers resolve). Absent it, collect-but-skip so the offline gate stays green everywhere.
_GCP_ENV = "SF_PROJECT_ID"

# A ``@spark`` test needs a local Spark session — i.e. the optional ``[spark]`` extra (pyspark)
# installed. Absent it, collect-but-skip so the core offline gate stays green without pyspark.
_HAS_PYSPARK = importlib.util.find_spec("pyspark") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    have_gcp = bool(os.environ.get(_GCP_ENV))
    skip_gcp = pytest.mark.skip(reason=f"@gcp: set {_GCP_ENV} (+ ADC) to run against live infra")
    skip_spark = pytest.mark.skip(reason="@spark: install the [spark] extra (pyspark) to run")
    for item in items:
        if "gcp" in item.keywords and not have_gcp:
            item.add_marker(skip_gcp)
        if "spark" in item.keywords and not _HAS_PYSPARK:
            item.add_marker(skip_spark)
