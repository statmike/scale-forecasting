"""GCS artifact upload + ObjectRef construction (CONTRACTS §3.4, §6).

Owned by BUILD steps 1.4 + B1. Public surface: ``upload_artifact``.
"""

from __future__ import annotations


def upload_artifact(local_path: str, run_id: str) -> object:  # pragma: no cover - stub
    raise NotImplementedError("registry.artifacts.upload_artifact — BUILD step 1.4/B1")
