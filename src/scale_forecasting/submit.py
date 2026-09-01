"""Submit a forecast run to Dataproc Serverless — the local/Composer launcher.

This is the ``[spark]``-extra, ADC-authenticated helper that turns a validated
`RunConfig` into a running Dataproc Serverless batch. It is the
same call path a scheduled Composer DAG would use (that orchestration is in development):
reproducing at runtime the exact delivery the Terraform ``seed`` module does for the seed job, but
for *forecast* runs and driven from Python (runs live in the registry, not Terraform state).

What `submit_batch` does:

1. **Package the code at runtime** — zip ``src/`` and upload it to the code bucket, so the batch
   loads current code via ``python_file_uris`` rather than anything baked into the container image.
   Upload the standalone ``spark_main`` shim as the ``gs://``
   main file (Dataproc runs it as ``__main__``; it absolute-imports the in-package dispatch logic).
2. **Stage the run config** — write the validated config to ``gs://<code>/runs/<run_id>.json`` and
   pass it as ``--config-uri``. The JSON is the lossless reproducibility record.
3. **Deliver infra identity as args** — the ``--sf-*`` flags (Dataproc rejects driver-env), built
   from `Settings` via `infra_args_from`.
4. **Submit** through `BatchControllerClient` (regional endpoint),
   optionally capping executors (``--max-executors`` → ``spark.dynamicAllocation.maxExecutors``, how
   a run is throttled), and return the batch id.

The two neighbours this leans on: `batch_infra` answers *what infrastructure we have* (including
which of the two dependency envelopes delivers the locked environment), and `batch_telemetry`
answers *what the batch did* once it is terminal. Both have consumers that never submit anything,
which is why they are not folded in here.

Public surface: ``submit_batch``, ``build_batch``, ``sizing_properties``, ``plan_sizing``, ``main``.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from .batch_infra import _DEFAULT_TTL_SECONDS, BatchInfra, serverless_dep_properties
from .commands import build_driver_args
from .errors import ConfigError, EngineError, get_logger
from .staging import stage_code, stage_config

if TYPE_CHECKING:
    from .config import RunConfig
    from .profiling.cost import ComputeProfile
    from .settings import Settings

_log = get_logger(__name__)

# Dataproc Serverless offers L4 only (no T4 on serverless — the config resolver forces L4 there and
# rejects a T4). A single accelerator per executor is attached; the deep-learning fit runs inside
# the pandas UDF (torch/NeuralProphet), so the GPU just needs to be visible to the executor's Python
# worker — we don't enable the RAPIDS SQL plugin (our SQL isn't the GPU workload).
_SERVERLESS_GPU_TYPE = "L4"

# How long ``wait=True`` blocks on the batch LRO before giving up. The google-api-core polling
# default is 900s (15 min) — shorter than a 100k forecast batch, so the bare ``operation.result()``
# would raise a client-side TimeoutError on a batch that is still running perfectly server-side
# (the batch is unaffected — only the local wait aborts). We poll for up to 2h so a full-scale run
# is actually waited out (and its DCU/wall-clock telemetry stamped). Not a cost knob — the batch's
# own runtime is what it is; this only bounds how long the submitter blocks.
_WAIT_TIMEOUT_SECONDS = 7200.0


# --- pure: batch spec assembly (no network) ------------------------------------


def _batch_id(run_id: str) -> str:
    """A Dataproc batch id: ``sf-<run_id>``, clamped to the 4-63 char / alnum+hyphen rule.

    ``run_id`` is already a slug + hex digest; prefix ``sf-`` and trim to fit. This is a cosmetic
    platform batch name only — never persisted to BigQuery, never used by idempotency.
    """
    raw = f"sf-{run_id}"
    return raw[:63].rstrip("-")


def _serverless_gpu_properties(gpu_type: str) -> dict[str, str]:
    """Dataproc Serverless runtime properties that attach an L4 to each executor (pure).

    Serverless manages GPU attachment itself: naming the accelerator type and selecting the premium
    compute *and* disk tiers is sufficient for the executor VM to carry an L4 that the
    torch/NeuralProphet fit inside the pandas UDF can use. The premium disk tier is mandatory
    whenever an L4 is requested.

    The Spark-level GPU resource-scheduling properties (``spark.executor.resource.gpu.*``,
    ``spark.task.resource.gpu.amount``) are omitted because Serverless owns them: it applies
    ``executor.resource.gpu.amount=1`` and ``task.resource.gpu.amount=1/spark.executor.cores`` as
    service defaults and rejects explicit values. GPU scheduling *is* therefore fractional here —
    the per-task share is chosen indirectly, by choosing executor cores. See ``CONSIDERATIONS.md``
    C2 for what that couples together.
    """
    if gpu_type != _SERVERLESS_GPU_TYPE:
        raise ConfigError(
            f"Dataproc Serverless supports {_SERVERLESS_GPU_TYPE} only, not {gpu_type!r}; "
            "use spark_mode='cluster' or runtime='ray' for other accelerators"
        )
    return {
        "spark.dataproc.executor.compute.tier": "premium",
        "spark.dataproc.executor.disk.tier": "premium",
        "spark.dataproc.executor.resource.accelerator.type": gpu_type.lower(),
    }


def sizing_properties(
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    max_executors: int | None = None,
    profile: ComputeProfile | None = None,
) -> dict[str, str]:
    """The ``spark.*`` overlay alone — `plan_sizing` without the audit record (pure).

    The shape every caller that only *submits* wants. A caller that also records the decision
    (`submit_batch`) calls `plan_sizing` and keeps both halves; a caller that renders a portable
    command (`main._assemble_commands`) has nowhere to record one and wants only these.
    """
    return plan_sizing(
        cfg,
        models,
        hardware=hardware,
        gpu_type=gpu_type,
        max_executors=max_executors,
        profile=profile,
    )[0]


def plan_sizing(
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    max_executors: int | None = None,
    profile: ComputeProfile | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """The ``spark.*`` overlay this batch's shape implies, **and** the audit record behind it.

    Returns ``(properties, sizing)`` — pure, and ``({}, {})`` when profiling is off. ``properties``
    is merged into the batch's ``RuntimeConfig``; ``sizing`` (`resources.audit.sizing_telemetry`) is
    the plan + translation + evidence, stamped onto the run header so the decision survives the
    driver log it would otherwise only appear in.

    A Serverless executor's shape is fixed at batch *creation*, so unlike Ray — where the engine
    sizes tasks on a cluster that already exists — this has to be decided here, before anything
    runs. `resources.serverless.plan_serverless` does the arithmetic; this only assembles its inputs
    from the config.

    **Most of the win needs no measurement.** The executor cores, the thread pins, the warm
    ``initialExecutors`` and the allocation ratio all follow from the task count and the family
    list alone, and they are emitted whether or not a profile arrives.

    **The memory sizing is what a profile buys, and it cannot be measured here.** There is no
    submit-side probe — the fleetwide pre-pass runs on the Spark driver *inside* the batch, by
    which point the executor shape is already fixed — so ``profile`` is a measurement of a
    *previous* run, resolved by `profiling.source.profile_for_run` from ``compute.profile.source``
    and handed in. This function stays pure and is only ever *given* one; it never goes and looks.
    ``None`` (no evidence, or none wanted) leaves the memory properties unemitted and Serverless'
    own defaults standing, exactly as before.

    **The unit sized against is a task, not a cell.** The engine shuffles cells into buckets and
    runs one task per bucket (`engines.spark_io.default_bucket_count`), each holding
    ``compute.bucket_target_cells`` cells that execute *sequentially* inside one pandas frame.
    So the widest useful fleet is the one that runs every bucket at once, not every cell — sizing
    against cells would ask for `bucket_target_cells`x more executors than the fan-out can ever
    keep busy. The driver enforces the other side of the same identity
    (`engines.spark_io.reachable_bucket_count`): whatever ceiling ends up on the batch, the
    bucket count is raised to match it.

    **A GPU batch is sized against the device the config names, not a nominal one.** On the GPU
    path ``executor.cores`` *is* the per-task device share, so the config's own
    ``compute.gpu_fraction`` and the accelerator's memory decide the executor's shape. A fixed
    fraction is passed through; ``"auto"`` cannot be calibrated here (there is no device to
    measure yet, exactly as on the Ray submit path) and falls back to the nominal share.

    ``compute.profile.mode == "off"`` returns ``({}, {})`` — the documented escape hatch back to
    the pre-profiler batch, reused rather than adding a field that would move every
    ``run_id``. Nothing is decided, so there is nothing to record either.
    """
    if cfg.compute.profile.mode == "off":
        return {}, {}

    # `device_memory_bytes` lives with the Ray engine because that is where the device table is
    # maintained; one table, consulted by both runtimes, beats a second copy that drifts.
    from .engines.ray_io import device_memory_bytes
    from .engines.spark_io import default_bucket_count
    from .models import get_model
    from .resources.audit import sizing_telemetry
    from .resources.serverless import plan_serverless

    executed = models if models is not None else cfg.models
    families: list[str] = []
    for name in executed:
        family = get_model(name).family
        if family not in families:
            families.append(family)

    gpu = hardware == "gpu"
    fraction = cfg.compute.gpu_fraction
    n_tasks = default_bucket_count(cfg, executed)
    plan, translation = plan_serverless(
        profile,
        families,
        n_tasks,
        gpu=gpu,
        device_bytes=device_memory_bytes(gpu_type or cfg.compute.gpu_type) if gpu else None,
        static_gpu_fraction=float(fraction) if isinstance(fraction, float) else None,
        max_executors=max_executors,
        # A controlled-measurement run wants the native thread pools uncapped, because a pinned
        # fit can only ever report the pin back as its `effective_cores` (see resources).
        pin_threads=not cfg.compute.profile.unpins_threads,
    )
    _log.info("serverless sizing: %s", translation.to_dict())
    return translation.properties, sizing_telemetry(
        plan, translation=translation, profile=profile
    )


def build_batch(
    *,
    infra: BatchInfra,
    settings: Settings,
    package_uri: str,
    launcher_uri: str,
    config_uri: str,
    max_executors: int | None = None,
    models: list[str] | None = None,
    manage_header: bool = True,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    properties: dict[str, str] | None = None,
) -> object:
    """Assemble the ``dataproc_v1.Batch`` for one forecast run (pure — builds the message only).

    Mirrors the Terraform seed batch: runtime container + package zip on ``python_file_uris``, the
    ``spark_main`` shim as the ``gs://`` main file, ``--config-uri`` + the ``--sf-*`` infra args.
    ``max_executors`` caps ``spark.dynamicAllocation.maxExecutors`` (executor throttle).

    ``models`` / ``manage_header`` carry the on-cluster contract: ``--models m1,m2`` restricts
    the executed subset (run_id still derives from the full staged config) and ``--manage-header
    false`` puts the on-cluster engine in contributor mode (``main.run`` owns the shared header).
    Both are appended to ``args`` **only when non-default**, so a standalone submit builds the exact
    same arg list as before (existing batches / snapshot tests unchanged).

    ``hardware="gpu"`` attaches an L4 per executor (`_serverless_gpu_properties`) — the
    deep-learning family's serverless job. ``gpu_type`` names the accelerator (serverless is
    L4-only; the resolver already forces this). A CPU batch adds no accelerator properties, so its
    message is unchanged.

    ``properties`` is the sizing overlay — `resources.serverless.translate_serverless` spelled as
    ``spark.*`` — applied *first*, so the two things a caller states explicitly still win over
    it: an explicit ``max_executors`` and the GPU attachment. Omitted (the default) the message
    is byte-identical to the pre-profiler one.

    The dependency envelope comes from `serverless_dep_properties` and is laid down *before* the
    overlay: the default ``container`` mode contributes the image and no properties (so the message
    is unchanged), while ``packed_venv`` contributes no image and the archive properties instead.
    """
    from datetime import timedelta

    from google.cloud import dataproc_v1 as dataproc

    args = build_driver_args(config_uri, settings, models=models, manage_header=manage_header)
    container_image, props = serverless_dep_properties(infra)
    props.update(properties or {})
    if max_executors is not None:
        props["spark.dynamicAllocation.maxExecutors"] = str(max_executors)
    if hardware == "gpu":
        props.update(_serverless_gpu_properties(gpu_type or _SERVERLESS_GPU_TYPE))

    return dataproc.Batch(
        pyspark_batch=dataproc.PySparkBatch(
            main_python_file_uri=launcher_uri,
            python_file_uris=[package_uri],
            args=args,
        ),
        runtime_config=dataproc.RuntimeConfig(
            version=infra.runtime_version,
            container_image=container_image,
            properties=props,
        ),
        environment_config=dataproc.EnvironmentConfig(
            execution_config=dataproc.ExecutionConfig(
                service_account=infra.compute_sa,
                subnetwork_uri=infra.subnetwork_uri,
                # Explicit max-runtime cap — overrides Dataproc's silent 4h default that would
                # cancel a healthy long 100k run mid-flight (see _DEFAULT_TTL_SECONDS).
                ttl=timedelta(seconds=infra.ttl_seconds),
            )
        ),
    )


# --- I/O: staging + submit -----------------------------------------------------


def _stage_config(cfg: RunConfig, run_id: str, infra: BatchInfra) -> str:
    """Stage the run config to GCS and return its URI (see `staging.stage_config`)."""
    return stage_config(cfg, run_id, infra.code_bucket)


def submit_batch(
    cfg: RunConfig,
    *,
    n_series: int | None = None,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    max_executors: int | None = None,
    models: list[str] | None = None,
    manage_header: bool = True,
    batch_id: str | None = None,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    wait: bool = True,
    wait_timeout: float = _WAIT_TIMEOUT_SECONDS,
) -> str:
    """Stage code + config and submit one Dataproc Serverless forecast batch; return its batch id.

    Resolves infra from the environment when not passed. ``n_series`` overrides
    ``data.series_limit`` at submit time — the scale knob for the 10 → 100 → 1k → 100k story;
    because it changes the config it yields a distinct ``run_id``/header per scale (each scale is
    its own queryable run). With ``wait`` the call blocks until the batch is terminal (parity with
    the Terraform seed apply) and then stamps Dataproc job telemetry onto the header
    (`batch_telemetry._stamp_job_telemetry`, best-effort); otherwise it returns once submitted (no
    telemetry).

    ``models`` / ``manage_header`` carry the on-cluster contract. The **full** ``cfg`` is
    always staged (so its ``run_id`` matches `main.run`'s), while ``models`` restricts the
    executed subset on-cluster and ``manage_header=False`` runs the engine in contributor mode
    (``main.run`` owns the shared header). Both default to standalone behavior, so every existing
    caller stages and submits exactly as before.

    ``batch_id`` overrides the derived ``sf-<run_id>`` id. A caller that fans out several
    batches under **one** shared ``run_id`` (each staging the same full cfg) supplies a distinct
    per-batch id, since the derived id would otherwise collide. When ``None`` the id is derived.

    ``hardware="gpu"`` attaches an L4 per executor (the deep-learning family's serverless job);
    ``gpu_type`` names the accelerator (serverless is L4-only). Both default to the CPU batch, so an
    existing caller submits exactly as before.
    """
    # `batch_telemetry`'s two names are bound per call, not at module load. They are what a test
    # substitutes to run this function without a network, and a module-level import would freeze
    # the originals here where `monkeypatch.setattr(batch_telemetry, ...)` can no longer reach them.
    from .batch_telemetry import _batch_client, _stamp_job_telemetry
    from .profiling.source import profile_for_run
    from .registry.ids import make_run_id
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    cfg = cfg.with_series_limit(n_series)
    run_id = make_run_id(cfg)
    batch_id = batch_id or _batch_id(run_id)

    package_uri, launcher_uri = stage_code(infra.code_bucket)
    config_uri = _stage_config(cfg, run_id, infra)
    properties, sizing = plan_sizing(
        cfg,
        models,
        hardware=hardware,
        gpu_type=gpu_type,
        max_executors=max_executors,
        # A past run's measurements, if `compute.profile.source` points at any (memoized, so
        # every family job of one run sizes off the same evidence rather than re-discovering).
        profile=profile_for_run(cfg, settings=settings),
    )
    batch = build_batch(
        infra=infra,
        settings=settings,
        package_uri=package_uri,
        launcher_uri=launcher_uri,
        config_uri=config_uri,
        max_executors=max_executors,
        models=models,
        manage_header=manage_header,
        hardware=hardware,
        gpu_type=gpu_type,
        properties=properties,
    )

    client = _batch_client(settings.region)
    parent = f"projects/{settings.project_id}/locations/{settings.region}"
    _log.info("submitting batch %s to %s", batch_id, parent)
    operation = client.create_batch(parent=parent, batch=batch, batch_id=batch_id)  # type: ignore[attr-defined]
    if wait:
        # Block until terminal, but with a timeout long enough for a full-scale (100k) batch — the
        # api-core polling default is only 900s, which a 100k run exceeds (_WAIT_TIMEOUT_SECONDS).
        result = operation.result(timeout=wait_timeout)  # blocks until terminal
        state = getattr(result, "state", None)
        state_name = getattr(state, "name", str(state))
        _log.info("batch %s finished: state=%s", batch_id, state_name)
        # Stamp Dataproc-level telemetry (cluster sizing, wall/overhead split, DCU usage) onto the
        # header — before the raise below, so even a FAILED batch (whose on-cluster update_header
        # never ran) still gets its sizing recorded. Best-effort: any failure here is logged and
        # swallowed, never sinking the run (the forecasts + registry rows already landed).
        _stamp_job_telemetry(client, parent, batch_id, run_id, settings, sizing=sizing)
        # A non-SUCCEEDED terminal state must fail loudly — the caller/CLI otherwise exits 0 on a
        # failed batch (the header stays RUNNING and the failure is silent). SUCCEEDED is the one
        # green state; CANCELLED/FAILED and anything else raise with the batch's own status message.
        if state_name != "SUCCEEDED":
            detail = getattr(result, "state_message", "") or "(no state_message)"
            raise EngineError(f"batch {batch_id} terminal state {state_name}: {detail}")
    return batch_id


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m scale_forecasting.submit --config run.json``."""
    from .config import load_config_uri

    p = argparse.ArgumentParser(prog="submit", description="Submit a forecast run to Dataproc.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="path to the run config JSON")
    src.add_argument("--config-uri", help="gs:// URI of a staged config (portable source)")
    p.add_argument("--n-series", type=int, default=None, help="override series_limit (scale knob)")
    p.add_argument(
        "--max-executors", type=int, default=None, help="cap dynamicAllocation executors"
    )
    p.add_argument("--no-wait", action="store_true", help="return once submitted (don't block)")
    p.add_argument(
        "--wait-timeout",
        type=float,
        default=_WAIT_TIMEOUT_SECONDS,
        help=f"seconds to block on the batch when waiting (default {_WAIT_TIMEOUT_SECONDS:.0f}; a "
        "100k batch exceeds the 900s api-core default)",
    )
    p.add_argument(
        "--ttl",
        type=int,
        default=_DEFAULT_TTL_SECONDS,
        help=f"batch max-runtime cap in seconds (default {_DEFAULT_TTL_SECONDS}; overrides "
        "Dataproc's silent 4h default that cancels a healthy long 100k run)",
    )
    ns = p.parse_args(argv)

    cfg = load_config_uri(ns.config or ns.config_uri)
    # Build infra once so --ttl overrides the default cap on every (child) batch this run submits.
    infra = BatchInfra.resolve()
    if ns.ttl != _DEFAULT_TTL_SECONDS:
        from dataclasses import replace

        infra = replace(infra, ttl_seconds=ns.ttl)
    batch_id = submit_batch(
        cfg,
        n_series=ns.n_series,
        infra=infra,
        max_executors=ns.max_executors,
        wait=not ns.no_wait,
        wait_timeout=ns.wait_timeout,
    )
    _log.info("submitted: %s", batch_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
