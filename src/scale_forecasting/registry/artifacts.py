"""GCS artifact upload + ObjectRef construction (CONTRACTS §3.4, §6).

A fitted model (or any per-cell binary) is written to GCS under a deterministic,
run-scoped path and referenced from ``forecast_metadata.model_artifact`` so every
prediction is traceable back to the exact object that produced it (lineage, DESIGN §8.3).

Split along the pure/I-O seam (CONTRACTS §0):

- **Pure** (tested offline now): ``artifact_gcs_uri`` builds the deterministic
  ``gs://.../<run_id>/<basename>`` destination from a local path — no client, no upload.
- **I/O** (structured now, GCP-verified in Arc B step B1): ``upload_artifact`` streams
  the bytes and returns the URI.

Public surface: ``artifact_gcs_uri``, ``upload_artifact``.
"""

from __future__ import annotations

from pathlib import PurePath, PurePosixPath


def artifact_gcs_uri(local_path: str, run_id: str, warehouse_uri: str) -> str:
    """Deterministic GCS destination for one artifact (pure, no I/O).

    Artifacts live under ``<warehouse>/artifacts/<run_id>/<basename>`` so they share the
    run's lineage and are easy to sweep on cleanup. The basename is taken from the local
    path; the same ``(local_path, run_id)`` always maps to the same URI (idempotent
    re-runs overwrite in place).

    Args:
        local_path: path to the artifact on the worker's local disk.
        run_id: the owning run (path scope).
        warehouse_uri: GCS warehouse root, e.g. ``gs://bucket/warehouse``.
    """
    basename = PurePath(local_path).name
    if not basename:
        raise ValueError(f"local_path has no filename component: '{local_path}'")
    root = warehouse_uri.rstrip("/")
    rel = PurePosixPath("artifacts") / run_id / basename
    return f"{root}/{rel}"


def upload_artifact(
    local_path: str, run_id: str, warehouse_uri: str
) -> str:  # pragma: no cover - Arc B (B1)
    raise NotImplementedError("registry.artifacts.upload_artifact — BUILD step B1")
