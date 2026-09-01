"""The GCS artifact layout — ``<artifact_root>/<run_id>/<basename>`` — written and read back.

A fitted model (or any per-cell binary) is written to GCS under a deterministic,
run-scoped path and referenced from ``forecast_metadata.model_artifact`` so every
prediction is traceable back to the exact object that produced it (lineage).

**One module owns the layout in both directions**, because the two directions have to agree
exactly or lineage breaks. ``artifact_gcs_uri`` composes the path on the way out;
``run_id_from_blob`` decomposes it on the way back, and it is the *only* way to answer "which run
owns this object" — nothing else records the mapping. A change to one is a change to the other, and
keeping them apart is how they would silently drift.

Reading the layout back is what makes registry cleanup well-defined. ``list_prefixes`` buckets
every object under the root by its owning run; ``orphan_run_ids`` names the prefixes the registry
has no row for; ``delete_prefixes`` removes them. Those three are the machinery behind
`ops.sweep_orphans` and the *first* step of `ops.drop_run`'s ordering rule (enumerate before
deleting the rows that index the objects) — the policy lives there, the layout arithmetic lives
here.

Split along the pure/I-O seam:

- **Pure** (tested offline): ``artifact_gcs_uri`` builds the deterministic destination from a
  local path; ``split_gcs_uri`` and ``run_id_from_blob`` parse a URI and a blob name back;
  ``orphan_run_ids`` is set arithmetic over the two id sources. No client, no network.
- **I/O** (GCP-verified by the ``@gcp`` round-trip test): ``upload_artifact`` streams a
  local file, ``upload_artifact_bytes`` streams in-memory bytes (the executor-side path —
  the fitted model is serialized to bytes in ``run_cell`` and never touches local disk);
  both return the URI. ``list_prefixes`` and ``delete_prefixes`` enumerate and remove.

Public surface: ``ArtifactPrefix``, ``artifact_gcs_uri``, ``split_gcs_uri``, ``run_id_from_blob``,
``orphan_run_ids``, ``upload_artifact``, ``upload_artifact_bytes``, ``list_prefixes``,
``delete_prefixes``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import Settings


@dataclass(frozen=True)
class ArtifactPrefix:
    """One run's GCS artifact prefix and what is under it."""

    run_id: str
    object_count: int
    byte_total: int


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


def split_gcs_uri(uri: str) -> tuple[str, str]:
    """``gs://bucket/a/b`` → ``("bucket", "a/b")``. A bare bucket yields an empty prefix.

    Raises `ValueError` on anything that is not a ``gs://`` URI, so a misconfigured warehouse root
    fails here rather than as a confusing 404 from the storage client.
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"not a GCS URI: {uri!r}")
    bucket, _, prefix = uri[len("gs://") :].partition("/")
    if not bucket:
        raise ValueError(f"GCS URI has no bucket: {uri!r}")
    return bucket, prefix.strip("/")


def run_id_from_blob(blob_name: str, root_prefix: str) -> str | None:
    """The ``run_id`` owning an artifact object, or ``None`` if it is not under ``root_prefix``.

    The inverse of `artifact_gcs_uri`. The layout is ``<root_prefix>/<run_id>/<basename>``, so the
    run id is the first path segment after the root. An object sitting directly in the root (no run
    segment) returns ``None`` rather than claiming a bogus id — a sweep must never delete something
    it cannot attribute.
    """
    root = root_prefix.strip("/")
    head = f"{root}/" if root else ""
    if not blob_name.startswith(head):
        return None
    rest = blob_name[len(head) :]
    run_id, sep, _ = rest.partition("/")
    return run_id if sep and run_id else None


def orphan_run_ids(seen: Iterable[str], known: Iterable[str]) -> tuple[str, ...]:
    """Run ids present in GCS but absent from the registry — sorted, deduplicated.

    Deliberately set arithmetic on the *whole* known set, not a per-id lookup: the caller reads
    ``run_registry`` once, so a sweep costs one query no matter how many prefixes exist.
    """
    known_set = set(known)
    return tuple(sorted({r for r in seen if r not in known_set}))


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
    try:
        # Parsed by the shared splitter, not inline: a malformed ``artifact_root`` then fails here
        # (as a `RegistryError` naming the URI) instead of silently uploading to a bucket named
        # after whatever the first path segment happened to be.
        bucket_name, blob_path = split_gcs_uri(uri)
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
    try:
        # Parsed by the shared splitter, not inline: a malformed ``artifact_root`` then fails here
        # (as a `RegistryError` naming the URI) instead of silently uploading to a bucket named
        # after whatever the first path segment happened to be.
        bucket_name, blob_path = split_gcs_uri(uri)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(data, content_type="application/octet-stream")
    except Exception as exc:  # noqa: BLE001 - re-raised as a package error with context
        raise RegistryError(f"artifact upload failed for {basename!r} -> {uri}: {exc}") from exc
    return uri


def list_prefixes(
    settings: Settings, run_ids: Sequence[str] | None = None
) -> dict[str, ArtifactPrefix]:  # pragma: no cover - GCP I/O, @gcp smoke
    """Artifact prefixes under this registry's root, with object counts and byte totals.

    Lists ``<warehouse>/artifacts/<project>/<registry_dataset>/`` and buckets by the ``run_id`` path
    segment. Objects that cannot be attributed to a run are ignored, never deleted.

    ``run_ids`` narrows to those runs' prefixes — one list call each instead of one sweep over the
    whole root. `ops.plan_drop_run` passes it because it already knows the runs;
    `ops.plan_sweep_orphans` cannot, since finding the unattributed prefixes *is* the job.
    """
    from google.cloud import storage

    bucket_name, root_prefix = split_gcs_uri(settings.artifact_root)
    client = storage.Client(project=settings.project_id)
    scans = [f"{root_prefix}/{r}/" for r in run_ids] if run_ids is not None else [f"{root_prefix}/"]
    counts: dict[str, list[int]] = {}
    for scan in scans:
        for blob in client.list_blobs(bucket_name, prefix=scan):
            run_id = run_id_from_blob(blob.name, root_prefix)
            if run_id is None:
                continue
            acc = counts.setdefault(run_id, [0, 0])
            acc[0] += 1
            acc[1] += int(blob.size or 0)
    return {r: ArtifactPrefix(r, n, b) for r, (n, b) in counts.items()}


def delete_prefixes(
    settings: Settings, run_ids: Sequence[str]
) -> int:  # pragma: no cover - GCP I/O, @gcp smoke
    """Delete every object under the named runs' artifact prefixes; return the object count."""
    from google.cloud import storage

    bucket_name, root_prefix = split_gcs_uri(settings.artifact_root)
    client = storage.Client(project=settings.project_id)
    bucket = client.bucket(bucket_name)
    deleted = 0
    for run_id in run_ids:
        for blob in client.list_blobs(bucket_name, prefix=f"{root_prefix}/{run_id}/"):
            bucket.blob(blob.name).delete()
            deleted += 1
    return deleted
