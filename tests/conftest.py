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

# A ``@ray`` test needs a local Ray runtime — the optional ``[ray]`` extra installed. It runs Ray
# in ``local_mode`` (no cluster), so ray-importable is the only requirement; absent it, skip.
_HAS_RAY = importlib.util.find_spec("ray") is not None

# A ``@gpu`` test provisions a live T4 Vertex Ray cluster (cost + ~15-25 min) and needs live infra.
# Opt-in only: run when both ``SF_ENABLE_GPU`` is set *and* GCP identity is present, so the T4 smoke
# never fires by accident in the offline gate or a plain ``@gcp`` run.
_GPU_ENV = "SF_ENABLE_GPU"

# A ``@raylive`` test provisions a live *CPU-only* Vertex Ray cluster (cost + ~15-25 min, no GPU
# quota). Same opt-in shape as ``@gpu`` but its own switch: it must not fire in a plain ``@gcp`` run
# (it costs money) yet must run without T4 quota. Gated on ``SF_ENABLE_RAY`` + GCP identity.
_RAYLIVE_ENV = "SF_ENABLE_RAY"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    have_gcp = bool(os.environ.get(_GCP_ENV))
    have_gpu = have_gcp and bool(os.environ.get(_GPU_ENV))
    have_raylive = have_gcp and bool(os.environ.get(_RAYLIVE_ENV))
    skip_gcp = pytest.mark.skip(reason=f"@gcp: set {_GCP_ENV} (+ ADC) to run against live infra")
    skip_spark = pytest.mark.skip(reason="@spark: install the [spark] extra (pyspark) to run")
    skip_ray = pytest.mark.skip(reason="@ray: install the [ray] extra to run against local Ray")
    skip_gpu = pytest.mark.skip(
        reason=f"@gpu: set {_GPU_ENV} + {_GCP_ENV} (+ T4 quota + ADC) to run the live T4 smoke"
    )
    skip_raylive = pytest.mark.skip(
        reason=f"@raylive: set {_RAYLIVE_ENV} + {_GCP_ENV} (+ ADC) to run the live CPU Ray smoke"
    )
    for item in items:
        if "gcp" in item.keywords and not have_gcp:
            item.add_marker(skip_gcp)
        if "spark" in item.keywords and not _HAS_PYSPARK:
            item.add_marker(skip_spark)
        if "ray" in item.keywords and not _HAS_RAY:
            item.add_marker(skip_ray)
        if "gpu" in item.keywords and not have_gpu:
            item.add_marker(skip_gpu)
        if "raylive" in item.keywords and not have_raylive:
            item.add_marker(skip_raylive)
