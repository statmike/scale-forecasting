"""Ray run telemetry — flatten the cluster plan into the header's ``job_telemetry`` JSON.

The Ray sibling of `batch_telemetry` and `cluster_telemetry`, answering the same operability
questions for a third runtime: *how big was the pool (and its elastic bounds), what did it cost in
wall-clock, and what sizing produced it*. Split out so all three runtimes' observe leg is found in
the same shape of file, and so the pure flattening step is testable with no Ray or Vertex import at
all.

The extract half is pure; the stamp half is the one best-effort write, deliberately swallowing its
own failures — telemetry is an overlay on an already-complete run, never a reason to fail it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .engines import ray_io
from .errors import get_logger

if TYPE_CHECKING:
    from .settings import Settings

_log = get_logger(__name__)


def extract_ray_telemetry(
    plan: ray_io.RayClusterPlan,
    *,
    cluster: object,
    job_id: str,
    job_status: str,
    total_wall_s: float | None,
    reuse: bool,
) -> dict[str, Any]:
    """Flatten the plan + cluster into the JSON-able telemetry dict stamped on the header (pure).

    The Ray analog of `extract_job_telemetry`, answering the same
    operability questions — *how big was the pool (and its elastic bounds), what did it cost in
    wall-clock, and what sizing produced it* — so a Ray run is as auditable on ``v_run_summary`` as
    a Spark one. Reads only fields already on the ``plan`` and the ``cluster`` object; every cluster
    field is optional (a missing attr degrades to None, never a raise) so this is safe on any object
    ``get_ray_cluster`` returns.
    """
    return {
        "runtime": "ray",
        "cluster_name": plan.cluster_name,
        "reuse": reuse,
        "job_id": job_id,
        "job_status": job_status,
        "total_wall_s": total_wall_s,
        "cpu_node_count": plan.cpu_node_count,
        "gpu_node_count": plan.gpu_node_count,
        "total_worker_nodes": plan.total_worker_nodes,
        # Elastic spec: the flag + per-pool bounds the cluster was created with, so
        # v_run_summary shows whether/how the pools autoscaled. node_count above is the derived
        # fixed-size-equivalent (the reference size; under autoscaling the pool starts at min).
        "autoscale": plan.autoscale,
        "cpu_min_nodes": plan.cpu_min_nodes,
        "cpu_max_nodes": plan.cpu_max_nodes,
        "gpu_min_nodes": plan.gpu_min_nodes,
        "gpu_max_nodes": plan.gpu_max_nodes,
        "head_machine_type": plan.head_machine_type,
        "cpu_machine_type": plan.cpu_machine_type,
        "gpu_machine_type": plan.gpu_machine_type,
        "accelerator_type": plan.accelerator_type,
        "accelerator_count": plan.accelerator_count,
        "sizing_gpu_fraction": plan.sizing_gpu_fraction,
        "n_gpu_cells": plan.n_gpu_cells,
        "n_cpu_cells": plan.n_cpu_cells,
        "ray_version": getattr(cluster, "ray_version", None) or None,
        "python_version": getattr(cluster, "python_version", None) or None,
        "dashboard_address": getattr(cluster, "dashboard_address", None) or None,
    }


def _stamp_ray_telemetry(
    telemetry: dict[str, Any],
    run_id: str,
    settings: Settings,
    *,
    sizing: dict[str, Any] | None = None,
) -> None:
    """Write the Ray telemetry dict to the run header's native JSON column (best-effort).

    The pure `extract_ray_telemetry` output, merged into ``job_telemetry`` a key at a time
    (`registry.header.merge_header_telemetry`) rather than written whole — several family jobs of
    one run each land here, and a whole-column write would leave only whichever finished last. The
    column is a native ``JSON`` type whose query parameter serializes the value itself, so we pass
    **dicts** (not pre-serialized strings, which would double-encode).

    ``sizing`` (`resources.audit.sizing_telemetry` over the two pool plans) is filed under
    ``$.sizing.<family>``: what the pools were sized to hold, and off whose measurements.

    Wrapped so any failure (API error, header not yet written) is logged and swallowed: telemetry
    is a nice-to-have overlay on an already-complete run, never a reason to fail it.
    """
    from .registry.header import merge_header_telemetry, sizing_telemetry_path

    patch = dict(telemetry)
    if sizing:
        patch[sizing_telemetry_path(sizing)] = sizing
    try:
        merge_header_telemetry(run_id, patch, settings=settings)
        _log.info("Ray telemetry stamped for run %s: %s", run_id, telemetry)
    except Exception as exc:  # noqa: BLE001 - telemetry is best-effort, never fatal
        _log.warning("Ray telemetry capture failed (non-fatal): %r", exc)
