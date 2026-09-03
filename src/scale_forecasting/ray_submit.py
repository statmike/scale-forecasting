"""Submit a forecast run to Ray on Vertex AI — the autoscaling-cluster launcher.

The Ray analog of `submit`: the ``[ray]``-extra, ADC-authenticated helper
that turns a validated `RunConfig` into a run on an **autoscaling**
Vertex Ray cluster (``ray_autoscale=False`` selects fixed
sizing). It owns the cluster lifecycle the way ``submit`` owns the Dataproc batch; the on-cluster
compute is
`run`, reached through the
`ray_entry` Jobs entrypoint.

What `submit_ray` does:

1. **Size the cluster to the run's fan-out** — `plan_cluster` turns the config into a
   ``RayClusterPlan`` (a GPU worker pool for NeuralProphet + a CPU worker pool for everything else).
   By default each pool carries a Vertex ``AutoscalingSpec(min, max)`` and scales with Ray's task
   demand; ``ray_autoscale=False`` gives each pool a fixed ``node_count`` and no spec. Either way
   the whole spec is a pure product of the config — logged and stamped to the run.
2. **Stage the run config** — write the validated config to ``gs://<code>/runs/<run_id>.json`` and
   pass it as ``--config-uri`` (the lossless reproducibility record — same contract as Spark).
3. **Provision (ephemeral default) or target (reuse opt-in) the cluster** — ephemeral:
   `ray_cluster._create_cluster_across_regions` from the planned spec, run, then
   `ray_cluster._delete_cluster` in a
   ``finally`` (teardown guaranteed even on failure); reuse: ``compute.ray_cluster_name`` /
   ``cluster_name=`` targets a standing cluster by name and skips both create and delete.
4. **Submit the on-cluster driver** — a Ray Job via
   `ray_jobs._submit_and_poll` (``vertex_ray://<dashboard>``) whose entrypoint
   is ``python -m scale_forecasting.ray_entry`` with the same ``--config-uri`` / ``--models`` /
   ``--manage-header`` / ``--sf-*`` contract the Spark entry uses. Current ``src/`` ships as the
   job's ``runtime_env`` working dir (runtime code delivery, never baked into the image — the same
   code runs locally and in the cloud), with ``requirements.txt`` for the on-cluster deps.
5. **Poll to terminal + stamp telemetry** — with ``wait``, block until the job is terminal, stamp a
   Ray analog of Spark's ``job_telemetry`` (cluster name, node counts, machine/accelerator types,
   calibrated-vs-sizing GPU fraction, wall-clock, job id) **plus the whole sizing decision** (both
   pool plans and the profile they were sized off, filed under ``$.sizing.<family>``) into
   ``run_registry.job_telemetry`` via ``header.merge_header_telemetry`` — **no schema change** (the
   JSON column already exists), and a merge rather than a whole-column write so the several family
   jobs of one run don't overwrite each other — and raise on a non-SUCCEEDED terminal state so a
   failed run never exits 0.

This module is the *ordering* of those five steps and nothing else; each step's machinery is a
sibling. The infra envelope is `ray_infra.RayInfra`, the cluster's whole lifetime (including the
region fallback) is `ray_cluster`, the Jobs-client connect/submit/poll is `ray_jobs`, the telemetry
flatten + stamp is `ray_telemetry`, and the job's ``runtime_env`` — current ``src/`` plus the
on-cluster deps — is built by `code_delivery.build_runtime_env`, alongside the zip the two Spark
surfaces ship, so all three code-delivery mechanisms read one ``src/``.

Public surface: ``submit_ray``, ``build_entrypoint``, ``main``.
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

from . import ray_cluster, ray_jobs, ray_telemetry
from .code_delivery import build_runtime_env
from .commands import build_driver_args
from .engines import ray_io
from .errors import EngineError, get_logger
from .ray_infra import RayInfra
from .resources.audit import sizing_telemetry
from .staging import stage_config

if TYPE_CHECKING:
    from .config import RunConfig
    from .settings import Settings

_log = get_logger(__name__)

# --- pure: job spec assembly (no network) --------------------------------------


def build_entrypoint(
    config_uri: str,
    settings: Settings,
    *,
    models: list[str] | None = None,
    manage_header: bool = True,
) -> str:
    """The Ray Job entrypoint shell command — ``python -m scale_forecasting.ray_entry ...`` (pure).

    The Ray analog of `build_batch`'s arg list, as a single shell
    string (what the Jobs API runs on the cluster head). Carries the same contract: ``--config-uri``
    (the staged config, whose digest is the shared ``run_id``), the ``--sf-*`` infra identity, and
    — only when non-default — ``--models m1,m2`` (executed subset) and ``--manage-header
    false`` (contributor mode; `main.run` owns the shared header). Defaults omit these
    flags so a standalone submit builds the plain command.
    """
    parts = ["python", "-m", "scale_forecasting.ray_entry"]
    parts += build_driver_args(config_uri, settings, models=models, manage_header=manage_header)
    return " ".join(parts)


# --- I/O: config staging -------------------------------------------------------


def _stage_config(cfg: RunConfig, run_id: str, infra: RayInfra) -> str:
    """Stage the run config to GCS and return its URI (see `staging.stage_config`).

    Shares the one staging helper with the Spark path, so a mixed run stages one config the same
    way regardless of runtime — the JSON *is* the shared reproducibility record, and its digest is
    the shared ``run_id``.
    """
    return stage_config(cfg, run_id, infra.code_bucket)


def submit_ray(
    cfg: RunConfig,
    *,
    models: list[str] | None = None,
    manage_header: bool = True,
    settings: Settings | None = None,
    infra: RayInfra | None = None,
    cluster_name: str | None = None,
    n_series: int | None = None,
    wait: bool = True,
    submission_id: str | None = None,
    use_gpu: bool | None = None,
    gpu_type: str | None = None,
    cluster_region: str | None = None,
) -> tuple[str, str, str]:
    """Size, provision, run, and (ephemeral) tear down a Ray-on-Vertex forecast run.

    Returns ``(job_id, cluster_resource_name, region)`` — the Ray job's id plus the coordinates a
    reader needs to probe it later: the cluster's Vertex persistent-resource path and the region it
    actually landed in (the reuse target's region, or the region an ephemeral create settled on
    after any capacity hop).

    The Ray analog of `submit_batch`. Resolves infra from the
    environment when not passed, sizes an **autoscaling** cluster to the run's fan-out
    (`plan_cluster`; ``ray_autoscale=False`` for fixed),
    stages the full config to GCS (so its ``run_id`` matches `main.run`'s), then runs the
    lifecycle:

    * **ephemeral (default):** create the planned cluster → submit the on-cluster driver as a Ray
      Job → (with ``wait``) poll to terminal + stamp telemetry → ``delete_ray_cluster`` in a
      ``finally`` so teardown happens even if the job raises.
    * **reuse (opt-in):** ``cluster_name`` (or ``compute.ray_cluster_name``) targets a standing
      cluster by name — skip create *and* skip delete; the plan still records the size it *should*
      be.

    ``n_series`` overrides ``series_limit`` at submit time (the scale knob — a different scale is a
    different fixed plan *and* a distinct ``run_id``/header, so each scale is its own queryable run;
    this is how "resize for a larger/smaller scale" is driven). ``models`` / ``manage_header`` carry
    the on-cluster contract: the full ``cfg`` is always staged (shared ``run_id``) while ``models``
    restricts the on-cluster executed subset and ``manage_header=False`` runs the engine in
    contributor mode (`main.run` owns the shared header). With ``wait`` a non-SUCCEEDED
    terminal state raises so a failed run never exits 0; the telemetry stamp precedes the raise so a
    failed run still records its sizing.

    ``submission_id``, when set (the orchestrator passes the family's deterministic ``job_key`` via
    `registry.ids.ray_submission_id`), becomes the Ray job's own id — so families sharing a
    ``run_id`` get distinct, directly-queryable Ray jobs instead of random auto-assigned ids. When
    ``None`` Ray assigns one (the standalone path).

    ``use_gpu``/``gpu_type`` override the flat ``compute`` GPU defaults for this family's job (the
    DAG orchestrator passes the family's resolved hardware — e.g. the deep-learning family gets a
    GPU pool even when the flat default is CPU). They size the cluster only; they're kept out of
    ``cfg`` so the staged config's ``run_id`` stays identical across every family in the run.

    ``cluster_region`` pins where a *reused* cluster is targeted (SDK init + resource path). It
    defaults to ``settings.region`` — the standing-cluster case — but the DAG orchestrator passes
    the region a shared ephemeral cluster landed in (which may differ after a capacity hop), so
    every Ray family's job finds the one shared cluster by name in the right region.
    """
    from .profiling.source import profile_for_run
    from .registry.ids import make_run_id
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or RayInfra.resolve()
    cfg = cfg.with_series_limit(n_series)
    run_id = make_run_id(cfg)
    # A past run's measurements, when `compute.profile.source` points at any. The pools are sized
    # here, before any node exists, so the only evidence available is somebody else's run — the
    # same position both Spark submit paths are in. The engine's own in-run `resolve_profile` sizes
    # *tasks* on the cluster this produces; it cannot reach back and change the cluster.
    profile = profile_for_run(cfg, settings=settings)
    plan = ray_io.plan_cluster(
        cfg, models, run_id=run_id, use_gpu=use_gpu, gpu_type=gpu_type, profile=profile
    )

    # Reuse when the config names a standing cluster or the caller overrides the name; else create.
    reuse = plan.reuse or cluster_name is not None
    name = cluster_name or plan.cluster_name

    config_uri = _stage_config(cfg, run_id, infra)
    entrypoint = build_entrypoint(config_uri, settings, models=models, manage_header=manage_header)
    runtime_env = build_runtime_env()
    regions = ray_cluster._resolve_regions(cfg, settings)
    _log.info(
        "ray submit: run_id=%s cluster=%s reuse=%s cpu_nodes=%d gpu_nodes=%d regions=%s",
        run_id,
        name,
        reuse,
        plan.cpu_node_count,
        plan.gpu_node_count,
        regions,
    )

    cluster_resource_name: str | None = None
    # For an ephemeral run, the teardown target is the *deterministic* resource path. The create
    # helper cleans up each stocked-out region as it advances; this final target covers the region
    # that actually got a cluster (torn down after the job) or a create that reached ERROR after the
    # loop settled on a region. Reuse leaves the standing cluster alone.
    teardown_target: str | None = None
    try:
        if reuse:
            # A reused cluster (a standing one, or the run's shared ephemeral one) lives in a known
            # region — the data-plane region by default, or the region a shared cluster landed in
            # after a capacity hop (passed as cluster_region). Pin the SDK there and target it.
            region = cluster_region or settings.region
            ray_cluster._init_vertex(settings, region)
            cluster_resource_name = ray_cluster.cluster_resource_path(settings, name, region)
        else:
            cluster_resource_name, cluster_region = ray_cluster._create_cluster_across_regions(
                plan, infra, name, settings, regions, policy=cfg.compute.capacity.policy_for("ray")
            )
            region = cluster_region
            teardown_target = ray_cluster.cluster_resource_path(settings, name, cluster_region)

        cluster = ray_cluster._get_cluster(cluster_resource_name)
        started = time.perf_counter()
        job_id, status, detail = ray_jobs._submit_and_poll(
            cluster_resource_name, entrypoint, runtime_env, wait=wait, submission_id=submission_id
        )
        wall_s = round(time.perf_counter() - started, 1) if wait else None

        if wait:
            telemetry = ray_telemetry.extract_ray_telemetry(
                plan,
                cluster=cluster,
                job_id=job_id,
                job_status=status,
                total_wall_s=wall_s,
                reuse=reuse,
            )
            ray_telemetry._stamp_ray_telemetry(
                telemetry,
                run_id,
                settings,
                sizing=sizing_telemetry(
                    plan.cpu_pool,
                    plan.gpu_pool,
                    profile=profile,
                    # This job's own label, not a pool's: a deep-learning job still has a (fallback)
                    # CPU pool, and filing the record under that pool's ``"cpu"`` would bury it.
                    family="+".join(ray_io.pool_families(models or cfg.models)) or "cpu",
                ),
            )
            if status != "SUCCEEDED":
                # Fold the driver diagnosis (captured before the log stream ages out) into the
                # raised error so the *cause* — not just "FAILED" — reaches the operator.
                suffix = f"\n{detail}" if detail else ""
                raise EngineError(f"ray job {job_id} terminal state {status}{suffix}")
        return job_id, cluster_resource_name, region
    finally:
        # Guaranteed teardown of an ephemeral cluster — even on a raised job *or a create that
        # errored mid-provision*. teardown_target is set (to the
        # deterministic path) for every ephemeral run and None for reuse, so a reused cluster is
        # left standing while a half-created ephemeral one is still cleaned up. _delete_cluster is
        # best-effort, so a target that never actually materialized is a harmless no-op.
        if teardown_target is not None:
            ray_cluster._delete_cluster(teardown_target)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m scale_forecasting.ray_submit --config run.json [--cluster-name ...]``."""
    from .config import load_config_uri

    p = argparse.ArgumentParser(
        prog="ray_submit", description="Submit a forecast run to Vertex Ray."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="path to the run config JSON")
    src.add_argument("--config-uri", help="gs:// URI of a staged config (portable source)")
    p.add_argument("--n-series", type=int, default=None, help="override series_limit (scale knob)")
    p.add_argument(
        "--cluster-name",
        default=None,
        help="reuse a standing cluster by name (skip create + teardown); else ephemeral",
    )
    p.add_argument("--no-wait", action="store_true", help="return once submitted (don't block)")
    ns = p.parse_args(argv)

    cfg = load_config_uri(ns.config or ns.config_uri)
    job_id, _, _ = submit_ray(
        cfg, cluster_name=ns.cluster_name, n_series=ns.n_series, wait=not ns.no_wait
    )
    _log.info("submitted: %s", job_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
