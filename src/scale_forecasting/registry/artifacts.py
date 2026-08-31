"""GCS artifact upload + ObjectRef construction.

A fitted model (or any per-cell binary) is written to GCS under a deterministic,
run-scoped path and referenced from ``forecast_metadata.model_artifact`` so every
prediction is traceable back to the exact object that produced it (lineage).

Split along the pure/I-O seam:

- **Pure** (tested offline): ``artifact_gcs_uri`` builds the deterministic
  ``<artifact_root>/<run_id>/<basename>`` destination from a local path — no client, no upload.
- **I/O** (GCP-verified by the ``@gcp`` round-trip test): ``upload_artifact`` streams a
  local file, ``upload_artifact_bytes`` streams in-memory bytes (the executor-side path —
  the fitted model is serialized to bytes in ``run_cell`` and never touches local disk);
  both return the URI.

Public surface: ``artifact_gcs_uri``, ``upload_artifact``, ``upload_artifact_bytes``.
"""

from __future__ import annotations

from pathlib import PurePath, PurePosixPath


def artifact_gcs_uri(local_path: str, run_id: str, artifact_root: str) -> str:
    """Deterministic GCS destination for one artifact (pure, no I/O).

    Artifacts live under ``<artifact_root>/<run_id>/<basename>``, where the root is
    `Settings.artifact_root` — ``<warehouse>/artifacts/<project>/<registry_dataset>``. The
    registry key in the root is what makes cleanup well-defined: every object under it belongs to
    exactly one registry, so a sweep can identify orphans (a ``run_id`` prefix with no
    ``run_registry`` row) without risking another registry's data in the same bucket.

    The basename is taken from the local path; the same ``(local_path, run_id)`` always maps to the
    same URI (idempotent re-runs overwrite in place).

    Args:
        local_path: path to the artifact on the worker's local disk (or a bare basename,
            e.g. ``<model_hash>.pkl`` for the in-memory bytes path).
        run_id: the owning run (path scope).
        artifact_root: the registry's artifact prefix, i.e. ``Settings.artifact_root``.
    """
    basename = PurePath(local_path).name
    if not basename:
        raise ValueError(f"local_path has no filename component: '{local_path}'")
    root = artifact_root.rstrip("/")
    rel = PurePosixPath(run_id) / basename
    return f"{root}/{rel}"


def upload_artifact(
    local_path: str, run_id: str, artifact_root: str
) -> str:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Upload one local artifact to its deterministic GCS destination; return the URI.

    Destination is `artifact_gcs_uri` (pure), so re-running a cell overwrites the
    same object in place — idempotent, matching the run-scoped registry writes. Raises
    `RegistryError` if the local file is missing or the upload fails.
    """
    from google.cloud import storage

    from ..errors import RegistryError

    uri = artifact_gcs_uri(local_path, run_id, artifact_root)
    # gs://<bucket>/<blob path> -> (bucket, blob)
    without_scheme = uri[len("gs://") :]
    bucket_name, _, blob_path = without_scheme.partition("/")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
    except Exception as exc:  # noqa: BLE001 - re-raised as a package error with context
        raise RegistryError(f"artifact upload failed for {local_path!r} -> {uri}: {exc}") from exc
    return uri


def upload_artifact_bytes(
    data: bytes, basename: str, run_id: str, artifact_root: str
) -> str:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Upload in-memory artifact bytes to the run-scoped GCS destination; return the URI.

    The executor-side path: ``run_cell`` serializes the fitted model to bytes (no local
    disk), and the registry writer calls this with a deterministic ``basename`` (the cell's
    ``model_hash`` + extension) so each cell maps to its own object and a re-run overwrites
    in place — idempotent, matching the run-scoped registry writes. Raises
    `RegistryError` if the upload fails.
    """
    from google.cloud import storage

    from ..errors import RegistryError

    uri = artifact_gcs_uri(basename, run_id, artifact_root)
    without_scheme = uri[len("gs://") :]
    bucket_name, _, blob_path = without_scheme.partition("/")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(data, content_type="application/octet-stream")
    except Exception as exc:  # noqa: BLE001 - re-raised as a package error with context
        raise RegistryError(f"artifact upload failed for {basename!r} -> {uri}: {exc}") from exc
    return uri
