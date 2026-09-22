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
The wait itself — including the watchdog that cancels a batch producing nothing — is `job_wait`,
shared with the cluster submitter.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from .batch_infra import _DEFAULT_TTL_SECONDS, BatchInfra, serverless_dep_properties
from .commands import build_driver_args
from .errors import ConfigError, EngineError, JobIdTaken, get_logger
from .hardware import spark_executor_env
from .job_wait import wait_for_job
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

# Serverless' own executor shape for an L4 batch, used only when no sizing overlay named one —
# `compute.profile.mode == "off"`, or a profile with no memory measurement for this family.
_SERVERLESS_DEFAULT_GPU_CORES = 4

# How many executors a GPU batch may lose before Spark gives up on the application, and the window
# those losses have to fall inside.
#
# The failure this bounds is not a slow run, it is a run that cannot start and will not stop. An
# executor whose card is missing dies inside the RAPIDS plugin at init, *before* it runs a task, so
# no task-failure counter ever advances — Spark simply asks the service for a replacement, and the
# replacement does the same thing. On 2026-09-10 that batch was still churning executors 49 minutes
# in, against 28 minutes for the same run when the card was there, with no forecast and nothing to
# stop it short of the TTL — and our TTL is deliberately hours long so that a healthy 100k run is
# not cancelled mid-flight. Unattended, that is fleet burned for nothing.
#
# Scaled to the fleet rather than fixed, because on a large batch losing a node now and then is
# ordinary attrition and must not kill the run; windowed for the same reason, so failures spread
# thinly across a long run never accumulate into a verdict. A batch that cannot keep a single
# executor alive blows through both in minutes.
_GPU_EXECUTOR_FAILURE_FACTOR = 2
_GPU_MIN_EXECUTOR_FAILURES = 8
_GPU_EXECUTOR_FAILURE_WINDOW = "30m"


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

    **The RAPIDS pool is released, and it has to be.** Serverless GPU runtimes ship the RAPIDS
    accelerator switched on, and its default allocator reserves nearly the whole card for Spark SQL
    before a single fit starts — ``Initializing RMM ASYNC pool size = 21632.125 MB on gpuId 0``, out
    of an L4's ~22.5 GB. The fits do not run in that JVM. They run in the PySpark Python workers,
    which then find a few hundred megabytes between them, and the ones that lose the race die with
    ``CUDA error: out of memory`` while asking for a model that needs 64 KB. On 2026-09-09 that cost
    smoke 03 thirty-seven of a hundred cells; the same run on a Dataproc cluster T4 and on Ray T4,
    neither of which loads RAPIDS, lost none. ``pool=NONE`` drops the reservation and leaves RAPIDS
    allocating on demand, so SQL still runs on the GPU and the fits can reach it too.

    **Naming any ``spark.rapids.*`` property costs us the service's memory defaults**, which is why
    `build_batch` states ``spark.executor.memory`` right after calling this. Left alone, Serverless
    sizes a GPU executor at 9560m over 4 cores and derives the overhead from that. Supply one
    RAPIDS property and it stops: the batch resolves to ``spark.executor.memory=3346m`` with
    ``spark.executor.memoryOverhead=0m``, and the service then rejects its own default, because 0
    is below the 256m-per-core floor it validates against. There is no way to keep the defaulting
    and release the pool, so we restate the number the platform would have chosen.
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
        "spark.rapids.memory.gpu.pool": "NONE",
    }


def _gpu_executor_failures(max_executors: int | None) -> int:
    """How many executor losses this GPU batch tolerates before Spark fails it (pure).

    Two per executor the batch may run, floored so a small batch still gets a few retries. ``None``
    — no cap named, so the service decides the fleet size — takes the floor.
    """
    if max_executors is None:
        return _GPU_MIN_EXECUTOR_FAILURES
    return max(_GPU_MIN_EXECUTOR_FAILURES, _GPU_EXECUTOR_FAILURE_FACTOR * max_executors)


def apply_gpu_properties(
    props: dict[str, str], *, gpu_type: str | None = None, max_executors: int | None = None
) -> None:
    """Add the Serverless GPU block to ``props`` in place — the accelerator and what it costs.

    Three separable things, all consequences of asking for a device: the accelerator attachment
    itself (`_serverless_gpu_properties`), the executor memory the service stops defaulting once a
    ``spark.rapids.*`` property is named, and a bound on executor churn. The last two are
    ``setdefault`` — a measured sizing overlay or a config override has a better number and keeps
    it; these are floors under a failure mode, not policy.

    **This exists as one function because two callers must produce the same batch.**
    `build_batch` builds the message the SDK submits and `commands.build_spark_commands` prints the
    ``gcloud`` line a reader copies. If the GPU block lived only in the first, the printed command
    would say ``--provisioned-hardware gpu`` — telling the code it has a device — while provisioning
    none, and the fits would silently run on CPU.
    """
    from .resources.serverless import serverless_gpu_executor_memory_mb

    props.update(_serverless_gpu_properties(gpu_type or _SERVERLESS_GPU_TYPE))
    cores = int(props.get("spark.executor.cores", _SERVERLESS_DEFAULT_GPU_CORES))
    props.setdefault("spark.executor.memory", f"{serverless_gpu_executor_memory_mb(cores)}m")
    props.setdefault("spark.executor.maxNumFailures", str(_gpu_executor_failures(max_executors)))
    props.setdefault("spark.executor.failuresValidityInterval", _GPU_EXECUTOR_FAILURE_WINDOW)


def _estimated_series(
    cfg: RunConfig, settings: Settings
) -> int | None:  # pragma: no cover - GCP I/O, exercised by the @gcp smokes
    """Roughly how many distinct series the source holds, or ``None`` when it need not be asked.

    Only the unbounded run has to ask: with ``series_limit`` set the count is known offline and
    `engines.spark_io.default_bucket_count` uses it directly. Without one, the alternative to
    asking is sizing the fleet against a parallelism cap, so a thousand-series table and a
    hundred-thousand-series table would be handed the same executors.

    ``APPROX_COUNT_DISTINCT`` because the number decides a *task count*, not a result. HyperLogLog
    is a scan of one column with no shuffle and its few-percent error moves the bucket count by a
    few percent; an exact count would shuffle the whole id column to answer a sizing question. The
    driver runs the same estimate again on the data it actually reads
    (`engines.spark_explode._estimated_series`) — this one only has to be close enough to pick an
    executor shape at submit time, before there is a session to ask.

    ``None`` on any failure. A batch that cannot be sized well should still be submitted; the
    fallback is the cap that was the only answer before this existed.
    """
    if cfg.data.series_limit is not None:
        return None
    from google.cloud import bigquery

    from .engines.bigquery_names import _source_ref

    table = _source_ref(cfg, settings.dataset_ref)
    sql = f"SELECT APPROX_COUNT_DISTINCT(`{cfg.data.ts_id_col}`) AS n FROM `{table}`"
    try:
        rows = list(bigquery.Client(project=settings.project_id).query(sql).result())
    except Exception as exc:  # noqa: BLE001 - an estimate is an optimisation, not a precondition
        _log.warning("could not estimate the series count (%s); sizing off max_parallelism", exc)
        return None
    n = rows[0]["n"] if rows else None
    return int(n) if n else None


def sizing_properties(
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    max_executors: int | None = None,
    profile: ComputeProfile | None = None,
    estimated_series: int | None = None,
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
        estimated_series=estimated_series,
    )[0]


def plan_sizing(
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    max_executors: int | None = None,
    profile: ComputeProfile | None = None,
    estimated_series: int | None = None,
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

    ``estimated_series`` is how an *unbounded* run gets a task count worth sizing against.
    ``series_limit=None`` means "forecast the whole table", and without a series count the fan-out
    falls back to a parallelism cap — so the fleet would be sized for a few hundred tasks whether
    the table holds a thousand series or a hundred thousand. `submit_batch` resolves it with one
    ``APPROX_COUNT_DISTINCT`` before calling here; this function stays pure and is only ever
    *given* the number, the same arrangement as ``profile``.

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
    n_tasks = default_bucket_count(cfg, executed, n_series=estimated_series)
    plan, translation = plan_serverless(
        profile,
        families,
        n_tasks,
        gpu=gpu,
        device_bytes=device_memory_bytes(gpu_type or cfg.compute.gpu_type) if gpu else None,
        static_gpu_fraction=float(fraction) if isinstance(fraction, float) else None,
        # An explicit argument wins over the config's ceiling; absent both, the fan-out decides.
        max_executors=max_executors if max_executors is not None else cfg.compute.max_executors,
        # A controlled-measurement run wants the native thread pools uncapped, because a pinned
        # fit can only ever report the pin back as its `effective_cores` (see resources).
        pin_threads=not cfg.compute.profile.unpins_threads,
    )
    _log.info("serverless sizing: %s", translation.to_dict())
    return translation.properties, sizing_telemetry(plan, translation=translation, profile=profile)


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

    A GPU batch also *tells the code* it has a device, twice, because a driver and an executor are
    two processes: the ``--provisioned-hardware`` driver arg and the
    ``spark.executorEnv.SF_PROVISIONED_HARDWARE`` property (see `hardware`). Attaching an
    accelerator and never saying so is how a run could be billed for L4s while every cell asked
    Lightning to pick a device and it picked the CPU.

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

    args = build_driver_args(
        config_uri,
        settings,
        models=models,
        manage_header=manage_header,
        provisioned_hardware=hardware,
    )
    container_image, props = serverless_dep_properties(infra)
    props.update(properties or {})
    if max_executors is not None:
        props["spark.dynamicAllocation.maxExecutors"] = str(max_executors)
    if hardware == "gpu":
        apply_gpu_properties(props, gpu_type=gpu_type, max_executors=max_executors)
    props.update(spark_executor_env(hardware))

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
    wait_timeout: float | None = None,
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

    ``wait_timeout`` left ``None`` takes ``infra.batch_job_wait_seconds`` — how long to block is a
    deployment-level patience setting, not something a caller should have to know. Pass a number
    only to override one submit.
    """
    # `batch_telemetry`'s two names are bound per call, not at module load. They are what a test
    # substitutes to run this function without a network, and a module-level import would freeze
    # the originals here where `monkeypatch.setattr(batch_telemetry, ...)` can no longer reach them.
    from google.api_core.exceptions import AlreadyExists

    from .batch_telemetry import _batch_client, _stamp_job_telemetry
    from .job_outcome import launch_window_start
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
        # Unbounded runs only: one APPROX_COUNT_DISTINCT so the fleet is sized against the table
        # that exists rather than against a parallelism cap. Resolved after `with_series_limit`,
        # so an `--n-series` override is treated as the known count it is and no query runs.
        estimated_series=_estimated_series(cfg, settings),
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
    since = launch_window_start()  # before submit: nothing this batch writes can predate it
    try:
        operation = client.create_batch(parent=parent, batch=batch, batch_id=batch_id)  # type: ignore[attr-defined]
    except AlreadyExists as exc:
        # The platform holds this batch id already. Translated here rather than left to surface as
        # a raw 409 so the row records *why* and the operator is told the two ways out — see
        # `errors.JobIdTaken` for how the id is reachable while the registry has no row for it.
        raise JobIdTaken(
            f"batch {batch_id} already exists in {parent}: the platform holds this job id but the "
            f"registry has no attempt for it. Re-run with a different run_id, or bump the attempt "
            f"by letting --force walk past it (job_launch checks the platform before stamping)."
        ) from exc
    if wait:
        # Block until terminal, with a patience that outlasts the batch's own ttl — the api-core
        # polling default is 900s and even the old 2h ceiling was short of a 100k run, and a wait
        # expiring on a healthy batch costs the telemetry stamp and an honest exit code for nothing
        # (see `batch_infra._ENV_BATCH_JOB_WAIT`). The watchdog inside cancels a batch that never
        # writes a cell (`job_wait.wait_for_job`, shared with the cluster submitter); the `since`
        # bound is what stops a re-run reading the previous attempt's rows as its own life signs,
        # the same trap `job_outcome` closed for the status audit.
        result = wait_for_job(
            operation,
            run_id=run_id,
            label=f"batch {batch_id}",
            wait_timeout=(
                float(infra.batch_job_wait_seconds) if wait_timeout is None else wait_timeout
            ),
            grace_s=infra.stall_grace_seconds,
            since=since,
        )
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
    """CLI: ``python -m scale_forecasting.submit --config run.json``.

    ``--models`` / ``--batch-id`` / ``--hardware`` / ``--gpu-type`` are the *reproduce one family of
    a multi-family run* flags. A run fans out one batch per model family, each shaped for its own
    device and carrying its own id under the shared ``run_id``; the launch-plan emitter prints one
    command per family, and these are what let a plain CLI line be one of them. Left off, the CLI
    submits the whole config as a single CPU batch under the derived ``sf-<run_id>`` — the
    standalone behavior every existing caller already gets.
    """
    from .config import load_config_uri

    p = argparse.ArgumentParser(prog="submit", description="Submit a forecast run to Dataproc.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="path to the run config JSON")
    src.add_argument("--config-uri", help="gs:// URI of a staged config (portable source)")
    p.add_argument("--n-series", type=int, default=None, help="override series_limit (scale knob)")
    p.add_argument(
        "--max-executors", type=int, default=None, help="cap dynamicAllocation executors"
    )
    p.add_argument(
        "--models",
        default=None,
        help="comma-separated subset to execute on-cluster; the FULL config is still staged, so "
        "the run_id is unchanged (default: every model in the config)",
    )
    p.add_argument(
        "--batch-id",
        default=None,
        help="override the derived sf-<run_id> batch id — needed when several batches fan out "
        "under one run_id, since the derived id would collide",
    )
    p.add_argument(
        "--hardware",
        choices=("cpu", "gpu"),
        default="cpu",
        help="device class to provision per executor (default cpu)",
    )
    p.add_argument(
        "--gpu-type", default=None, help="accelerator name when --hardware gpu (serverless is L4)"
    )
    p.add_argument("--no-wait", action="store_true", help="return once submitted (don't block)")
    p.add_argument(
        "--wait-timeout",
        type=float,
        default=None,
        help="seconds to block on the batch when waiting (default SF_BATCH_JOB_WAIT_S, 24h; "
        "giving up early costs the telemetry stamp and exits non-zero on a healthy batch)",
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
        models=ns.models.split(",") if ns.models else None,
        batch_id=ns.batch_id,
        hardware=ns.hardware,
        gpu_type=ns.gpu_type,
        wait=not ns.no_wait,
        wait_timeout=ns.wait_timeout,
    )
    _log.info("submitted: %s", batch_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
