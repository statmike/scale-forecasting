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

Dependencies reach the batch in one of two envelopes around the same locked environment — the
shared runtime image (default) or the packed-venv archive — resolved by `serverless_dep_properties`
and switched with ``SF_SERVERLESS_DEPS``. See the note above that function.

Public surface: ``BatchInfra``, ``submit_batch``, ``serverless_dep_properties``,
``sizing_properties``, ``plan_sizing``, ``main``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .commands import build_driver_args
from .errors import ConfigError, EngineError, get_logger
from .staging import stage_config

if TYPE_CHECKING:
    from .config import RunConfig
    from .profiling.cost import ComputeProfile
    from .settings import Settings

_log = get_logger(__name__)

# The package root that gets zipped + uploaded (contains only scale_forecasting/, so it sits at the
# zip root and is importable once on sys.path — same layout the Terraform seed module relies on).
_SRC_DIR = Path(__file__).resolve().parent.parent

# Batch-submission infra env vars (beyond the SF_* identity Settings resolves). Kept together so the
# docstring, resolve(), and any tooling agree.
_ENV_CODE_BUCKET = "SF_CODE_BUCKET"
_ENV_CONTAINER_IMAGE = "SF_CONTAINER_IMAGE"
_ENV_COMPUTE_SA = "SF_COMPUTE_SA"
_ENV_SUBNETWORK = "SF_SUBNETWORK_URI"
# Optional: the packed-venv archive URI (gs://…/envs/<hash>.tar.gz). The Dataproc-cluster runtime's
# dependency-delivery mechanism — clusters can't use the custom container, so a cluster job attaches
# this archive instead (see dataproc_cluster.build_job). Unset for serverless/Ray-only deployments.
_ENV_VENV_ARCHIVE = "SF_VENV_ARCHIVE"
# Optional: the custom GPU cluster image URI (a Compute image with the NVIDIA driver pre-baked,
# built from the same 2.2 line). GPU Dataproc *clusters* boot from it so the driver is already
# present at create time instead of being compiled on every node. Unset → GPU clusters install the
# driver at create via the stock init action (the fallback). CPU clusters, serverless, Ray ignore.
_ENV_GPU_IMAGE = "SF_GPU_IMAGE"

_DEFAULT_RUNTIME_VERSION = "2.2"

# How a Dataproc SERVERLESS batch gets its dependencies. Two envelopes around the *same* locked
# environment:
#   "container"   (default) — the shared runtime image. Nothing installs at launch, and the batch's
#                             Python is ours regardless of which Python the runtime version ships.
#   "packed_venv"           — the self-contained venv archive the Dataproc-*cluster* path uses,
#                             attached via ``spark.archives``. Lets a deployment with no Artifact
#                             Registry run Spark, at the cost of a per-node fetch of the archive.
#
# This is deployment infrastructure, NOT a run parameter, so it lives on `BatchInfra` (env-resolved,
# like the archive URI itself) and deliberately not in `ComputeConfig`: both envelopes deliver the
# byte-identical uv.lock environment, so the science of a run is the same either way, and folding
# the choice into the config would fold it into the ``run_id`` — making one experiment two runs.
# ``compute.spark_deps`` stays what it has always been: the Dataproc-*cluster* knob.
#
# ⚠️ ``packed_venv`` on serverless is UNPROVEN and is the reason this switch exists. Job archives are
# localized to the *executors'* working dirs; whether the serverless *driver* gets one is the open
# question (on a cluster it does not — see ``dataproc_cluster._VENV_DIR``, where an init action
# lands the venv at an absolute path instead, a fix serverless has no equivalent of). Default stays
# "container" until a live batch says otherwise.
_ENV_SERVERLESS_DEPS = "SF_SERVERLESS_DEPS"
_SERVERLESS_DEPS_CONTAINER = "container"
_SERVERLESS_DEPS_PACKED_VENV = "packed_venv"

# Where ``spark.archives``' ``#env`` fragment unpacks, and the interpreter inside it — relative to
# the working directory Spark localizes into.
_VENV_UNPACK_DIR = "env"
_VENV_ARCHIVE_PYTHON = f"./{_VENV_UNPACK_DIR}/bin/python"

# Dataproc Serverless offers L4 only (no T4 on serverless — the config resolver forces L4 there and
# rejects a T4). A single accelerator per executor is attached; the deep-learning fit runs inside
# the pandas UDF (torch/NeuralProphet), so the GPU just needs to be visible to the executor's Python
# worker — we don't enable the RAPIDS SQL plugin (our SQL isn't the GPU workload).
_SERVERLESS_GPU_TYPE = "L4"

# Batch max-runtime cap (``ExecutionConfig.ttl``). Dataproc Serverless applies a DEFAULT ttl of 4h
# when none is set — which silently CANCELS a longer-running batch mid-flight (a full 100k explode
# run can exceed 4h, and the cancel kills it before it writes its run_registry summary row, so the
# efficiency views render blank). We set an explicit, generous 24h so a full-scale run finishes on
# its own. Override per-submit with ``--ttl``. This bounds the batch's lifetime, NOT the client wait
# (that's _WAIT_TIMEOUT_SECONDS): a serverless batch bills only for what it uses, so a high ceiling
# costs nothing extra — it just stops the platform from guillotining a healthy long run.
_DEFAULT_TTL_SECONDS = 86400

# How long ``wait=True`` blocks on the batch LRO before giving up. The google-api-core polling
# default is 900s (15 min) — shorter than a 100k forecast batch, so the bare ``operation.result()``
# would raise a client-side TimeoutError on a batch that is still running perfectly server-side
# (the batch is unaffected — only the local wait aborts). We poll for up to 2h so a full-scale run
# is actually waited out (and its DCU/wall-clock telemetry stamped). Not a cost knob — the batch's
# own runtime is what it is; this only bounds how long the submitter blocks.
_WAIT_TIMEOUT_SECONDS = 7200.0


@dataclass(frozen=True)
class BatchInfra:
    """Dataproc-batch infra identity — what submitting a batch needs beyond `Settings`.

    Resolved from ``SF_*`` env (parity with ``Settings``) or ``terraform output``. Frozen and
    passed down so a run's batch targets the resolved infra.
    """

    code_bucket: str  # bucket the package zip + launcher + config JSON are staged to
    container_image: str  # full runtime image incl. tag
    compute_sa: str  # runtime SA the batch runs as (scale-forecasting-compute)
    subnetwork_uri: str  # subnet with Private Google Access + internal-ingress firewall
    runtime_version: str = _DEFAULT_RUNTIME_VERSION
    ttl_seconds: int = _DEFAULT_TTL_SECONDS  # batch max-runtime cap; > default 4h so 100k finishes
    # Packed-venv archive URI for the Dataproc-*cluster* path (clusters can't use the container).
    # Optional: only cluster families with spark_deps="packed_venv" need it; serverless/Ray ignore.
    venv_archive_uri: str | None = None
    # Custom GPU cluster image URI (NVIDIA driver pre-baked). Optional: only GPU cluster families
    # use it; when unset a GPU cluster installs the driver at create. CPU/serverless/Ray ignore it.
    gpu_image_uri: str | None = None
    # Which envelope delivers deps to a SERVERLESS batch — see `_ENV_SERVERLESS_DEPS`. Clusters and
    # Ray ignore it (they have exactly one mechanism each).
    serverless_deps: str = _SERVERLESS_DEPS_CONTAINER

    @classmethod
    def resolve(cls) -> BatchInfra:
        """Build from the ``SF_*`` batch-infra environment; raise naming the first missing var.

        ``SF_CONTAINER_IMAGE`` is required for the default ``container`` envelope and *not* required
        under ``SF_SERVERLESS_DEPS=packed_venv`` — a deployment that delivers deps by archive has no
        Artifact Registry to name, which is the point of the switch.
        """
        serverless_deps = os.environ.get(_ENV_SERVERLESS_DEPS) or _SERVERLESS_DEPS_CONTAINER
        required = {
            "code_bucket": _ENV_CODE_BUCKET,
            "compute_sa": _ENV_COMPUTE_SA,
            "subnetwork_uri": _ENV_SUBNETWORK,
        }
        if serverless_deps == _SERVERLESS_DEPS_CONTAINER:
            required["container_image"] = _ENV_CONTAINER_IMAGE
        values: dict[str, str] = {"container_image": os.environ.get(_ENV_CONTAINER_IMAGE) or ""}
        for field_name, env_name in required.items():
            raw = os.environ.get(env_name)
            if not raw:
                raise ConfigError(
                    f"missing required environment variable {env_name} "
                    f"(set it, or use BatchInfra.from_terraform_outputs for local dev)"
                )
            values[field_name] = raw
        return cls(
            code_bucket=values["code_bucket"],
            container_image=values["container_image"],
            compute_sa=values["compute_sa"],
            subnetwork_uri=values["subnetwork_uri"],
            runtime_version=os.environ.get("SF_RUNTIME_VERSION") or _DEFAULT_RUNTIME_VERSION,
            venv_archive_uri=os.environ.get(_ENV_VENV_ARCHIVE) or None,
            gpu_image_uri=os.environ.get(_ENV_GPU_IMAGE) or None,
            serverless_deps=serverless_deps,
        )

    @classmethod
    def from_terraform_outputs(
        cls, outputs: dict[str, str], image_tag: str = "latest"
    ) -> BatchInfra:
        """Build from a ``terraform output -json`` value map (local dev/tests).

        Reads the keys the ``terraform/main`` stage emits — ``code_bucket``, ``runtime_image_repo``
        (a base path; ``image_tag`` is appended), ``compute_sa``, ``subnetwork_uri``, the optional
        ``venv_archive_uri`` (the packed-venv archive for the cluster path), and the optional
        ``gpu_image_uri`` (the pre-baked GPU cluster image).
        """
        try:
            return cls(
                code_bucket=outputs["code_bucket"],
                container_image=f"{outputs['runtime_image_repo']}:{image_tag}",
                compute_sa=outputs["compute_sa"],
                subnetwork_uri=outputs["subnetwork_uri"],
                venv_archive_uri=outputs.get("venv_archive_uri") or None,
                gpu_image_uri=outputs.get("gpu_image_uri") or None,
            )
        except KeyError as exc:
            raise ConfigError(f"terraform outputs missing key: {exc.args[0]}") from exc


# --- pure: batch spec assembly (no network) ------------------------------------


def serverless_dep_properties(infra: BatchInfra) -> tuple[str, dict[str, str]]:
    """Resolve serverless dependency delivery → ``(container_image, extra properties)`` (pure).

    The one place the two envelopes are spelled out, shared by `build_batch` and the ``gcloud``
    emitter so the submitted batch and the printed command can't disagree about how deps arrive:

    - ``container`` (default) — the image, no properties.
    - ``packed_venv`` — no image, and three properties: ``spark.archives`` attaches the
      self-contained venv archive under ``#env``, and ``PYSPARK_PYTHON`` is repointed at the
      interpreter inside it for **both** sides (``spark.dataproc.driverEnv.*`` for the driver,
      ``spark.executorEnv.*`` for the executors — the driver-side prefix is Dataproc-specific).

    Raises `ConfigError` on an unknown mode, and on ``packed_venv`` with no archive URI resolved:
    a batch submitted without either envelope would run against the stock runtime's Python and fail
    deep inside a model fit, long after the point where the mistake was fixable.
    """
    if infra.serverless_deps == _SERVERLESS_DEPS_CONTAINER:
        return infra.container_image, {}
    if infra.serverless_deps != _SERVERLESS_DEPS_PACKED_VENV:
        raise ConfigError(
            f"unknown {_ENV_SERVERLESS_DEPS}={infra.serverless_deps!r}; expected "
            f"{_SERVERLESS_DEPS_CONTAINER!r} or {_SERVERLESS_DEPS_PACKED_VENV!r}"
        )
    if not infra.venv_archive_uri:
        raise ConfigError(
            f"{_ENV_SERVERLESS_DEPS}={_SERVERLESS_DEPS_PACKED_VENV!r} needs the packed-venv "
            f"archive; set {_ENV_VENV_ARCHIVE} (terraform output venv_archive_uri)"
        )
    return "", {
        "spark.archives": f"{infra.venv_archive_uri}#{_VENV_UNPACK_DIR}",
        "spark.dataproc.driverEnv.PYSPARK_PYTHON": _VENV_ARCHIVE_PYTHON,
        "spark.executorEnv.PYSPARK_PYTHON": _VENV_ARCHIVE_PYTHON,
    }


def _rfc3339_seconds(a: object, b: object) -> float | None:
    """Whole seconds between two Dataproc timestamp fields (``b - a``), or None.

    Dataproc stamps ``create_time``/``state_time`` as ``google.protobuf.Timestamp``; both expose
    ``.timestamp()`` (via the proto's datetime helper). Returns None if either is missing so a
    partial batch object degrades cleanly rather than raising.
    """
    ts_a = getattr(a, "timestamp", None)
    ts_b = getattr(b, "timestamp", None)
    if not callable(ts_a) or not callable(ts_b):
        return None
    try:
        return round(ts_b() - ts_a(), 1)
    except Exception:  # noqa: BLE001 - telemetry is best-effort, never fatal
        return None


def extract_job_telemetry(batch: object) -> dict[str, Any]:
    """Flatten a Dataproc ``Batch`` into the JSON-able telemetry dict stamped on the run header.

    Pure (no network): reads only fields already on the ``batch`` object that ``get_batch`` returns.
    Answers the operability questions the registry couldn't before — *how big was the cluster, did
    it autoscale, how much did it cost, and where did the wall-clock go* (provision + startup +
    closeout vs. our own ``runtime_seconds``):

    - ``total_wall_s`` — ``state_time − create_time``: the full provision→terminal wall-clock. The
      gap between this and the engine's ``runtime_seconds`` is Dataproc overhead (autoscaling
      warm-up + teardown), which amortizes as scale grows — the efficiency half of the scale story.
    - ``dcu_milli_seconds`` / ``shuffle_storage_gb_seconds`` — approximate usage (billing proxy +
      shuffle pressure).
    - ``driver_cores`` / ``executor_cores`` / ``executor_instances`` / ``max_executors`` /
      ``executor_memory`` / ``executor_memory_overhead`` — the resolved cluster sizing and the
      autoscaling cap (the executor throttle shows up here). This is the *echoed* shape — what
      Dataproc says it ran — as against the ``sizing`` record, which is what we asked for and why;
      the two disagreeing is a finding, so both are kept.
    - ``runtime_version`` / ``container_image`` — what actually ran (reproducibility).
    - ``service_account`` / ``subnetwork_uri`` — the identity + network the batch had access to.

    Every field is individually optional: a missing sub-message yields None for its keys, never a
    raise, so this is safe to call on any batch object the API returns.
    """
    tel: dict[str, Any] = {}

    tel["total_wall_s"] = _rfc3339_seconds(
        getattr(batch, "create_time", None), getattr(batch, "state_time", None)
    )

    runtime_info = getattr(batch, "runtime_info", None)
    usage = getattr(runtime_info, "approximate_usage", None) if runtime_info else None
    tel["dcu_milli_seconds"] = (
        int(getattr(usage, "milli_dcu_seconds", 0)) or None if usage else None
    )
    tel["shuffle_storage_gb_seconds"] = (
        int(getattr(usage, "shuffle_storage_gb_seconds", 0)) or None if usage else None
    )

    runtime_config = getattr(batch, "runtime_config", None)
    props = dict(getattr(runtime_config, "properties", {}) or {}) if runtime_config else {}

    def _prop_int(key: str) -> int | None:
        raw = props.get(key)
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    tel["driver_cores"] = _prop_int("spark.driver.cores")
    tel["executor_cores"] = _prop_int("spark.executor.cores")
    tel["executor_instances"] = _prop_int("spark.executor.instances")
    tel["max_executors"] = _prop_int("spark.dynamicAllocation.maxExecutors")
    # The memory half of the resolved shape, and the only half a profile actually moves (§3.10):
    # cores and the executor counts follow from fan-out with or without evidence. Strings, because
    # Spark spells them ``"8g"`` / ``"3891m"`` — kept verbatim rather than parsed to bytes so the
    # record shows what the platform was told, not our reading of it. Absent on a batch that left
    # Serverless' own defaults standing, which is itself the answer to "what sized this".
    tel["executor_memory"] = props.get("spark.executor.memory") or None
    tel["executor_memory_overhead"] = props.get("spark.executor.memoryOverhead") or None
    tel["runtime_version"] = (
        getattr(runtime_config, "version", None) or None if runtime_config else None
    )
    tel["container_image"] = (
        getattr(runtime_config, "container_image", None) or None if runtime_config else None
    )
    # The other way a batch can get its dependencies (`serverless_dep_properties`). Exactly one of
    # these two is set on any batch, so the pair reads as "which envelope delivered the env" — and a
    # batch with neither is one running against the stock runtime's Python, which is a finding.
    tel["venv_archive"] = props.get("spark.archives") or None

    env = getattr(batch, "environment_config", None)
    exec_cfg = getattr(env, "execution_config", None) if env else None
    tel["service_account"] = (
        getattr(exec_cfg, "service_account", None) or None if exec_cfg else None
    )
    tel["subnetwork_uri"] = getattr(exec_cfg, "subnetwork_uri", None) or None if exec_cfg else None

    return tel


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
    is merged into the batch's ``RuntimeConfig``; ``sizing`` (`resources.sizing_telemetry`) is the
    plan + translation + evidence, stamped onto the run header so the decision survives the driver
    log it would otherwise only appear in.

    A Serverless executor's shape is fixed at batch *creation*, so unlike Ray — where the
    engine sizes tasks on a cluster that already exists — this has to be decided here, before
    anything runs. `resources.plan_serverless` does the arithmetic; this only assembles its
    inputs from the config.

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
    from .resources import plan_serverless, sizing_telemetry

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

    ``properties`` is the sizing overlay — `resources.translate_serverless` spelled as
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


def _stage_code(infra: BatchInfra) -> tuple[str, str]:
    """Zip ``src/`` + upload it and the standalone launcher shim to the code bucket.

    Returns ``(package_uri, launcher_uri)``. The zip name carries an md5 so a code change is a new
    object (no in-place overwrite races), matching the seed module's runtime-delivery contract. The
    launcher is ``src/spark_main.py`` — a top-level shim (absolute import), *not* the in-package
    ``spark_entry`` module: Dataproc runs the main file as ``__main__`` with no package context, so
    a file with relative imports would ``ImportError``. The zip supplies the package it imports.

    The zip itself is built by `build_package_zip` — the SAME builder the
    interactive Spark Connect path (notebook 01) uses to ship code to its workers, so worker code
    can't drift between the batch and Connect delivery mechanisms.
    """
    from google.cloud import storage

    from .code_delivery import build_package_zip

    # Build the zip in memory (deterministic walk) and hash it for the object name — shared with the
    # Connect path so both deliver byte-identical package code.
    data, code_hash = build_package_zip()

    client = storage.Client()
    bucket = client.bucket(infra.code_bucket)
    pkg_name = f"runs/scale_forecasting-{code_hash}.zip"
    bucket.blob(pkg_name).upload_from_string(data, content_type="application/zip")

    launcher_name = "runs/spark_main.py"
    launcher_local = _SRC_DIR / "spark_main.py"
    bucket.blob(launcher_name).upload_from_filename(str(launcher_local))

    return (
        f"gs://{infra.code_bucket}/{pkg_name}",
        f"gs://{infra.code_bucket}/{launcher_name}",
    )


def _stage_config(cfg: RunConfig, run_id: str, infra: BatchInfra) -> str:
    """Stage the run config to GCS and return its URI (see `staging.stage_config`)."""
    return stage_config(cfg, run_id, infra.code_bucket)


def _batch_client(region: str) -> object:
    """A regional `BatchControllerClient` (Dataproc batches are a regional resource)."""
    from google.api_core.client_options import ClientOptions
    from google.cloud import dataproc_v1 as dataproc

    return dataproc.BatchControllerClient(
        client_options=ClientOptions(api_endpoint=f"{region}-dataproc.googleapis.com:443")
    )


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
    (`_stamp_job_telemetry`, best-effort); otherwise it returns once submitted (no telemetry).

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
    from .profiling.source import profile_for_run
    from .registry.ids import make_run_id
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    cfg = cfg.with_series_limit(n_series)
    run_id = make_run_id(cfg)
    batch_id = batch_id or _batch_id(run_id)

    package_uri, launcher_uri = _stage_code(infra)
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


def _stamp_job_telemetry(
    client: Any,
    parent: str,
    batch_id: str,
    run_id: str,
    settings: Settings,
    *,
    sizing: dict[str, Any] | None = None,
) -> None:
    """Read the finished batch's telemetry and write it to the run header (best-effort).

    A fresh ``get_batch`` (the LRO result can carry incomplete ``approximate_usage``) → the pure
    `extract_job_telemetry` → the header's ``job_telemetry``. The header column is a
    native ``JSON`` type whose query parameter serializes the value itself, so we pass the telemetry
    **dict** (not a pre-serialized string, which would double-encode). Wrapped so any failure (API
    error, missing field, header not yet written) is logged and swallowed: telemetry is a
    nice-to-have overlay on an already-complete run, never a reason to fail it.

    ``sizing`` (`plan_sizing`'s second half) rides along under ``$.sizing.<family>``. It is decided
    at *submit* and stamped at *finish* so one write carries both halves, and a batch that never
    reaches terminal has no telemetry worth reading anyway.

    The write **merges** (`registry.bq.merge_header_telemetry`) rather than replacing the column:
    several family jobs of one run each land here, and a whole-column write would leave only
    whichever finished last.
    """
    from .registry import bq

    try:
        fetched = client.get_batch(name=f"{parent}/batches/{batch_id}")
        telemetry = extract_job_telemetry(fetched)
        if sizing:
            telemetry[bq.sizing_telemetry_path(sizing)] = sizing
        bq.merge_header_telemetry(run_id, telemetry, settings=settings)
        _log.info("batch %s telemetry stamped: %s", batch_id, telemetry)
    except Exception as exc:  # noqa: BLE001 - telemetry is best-effort, never fatal
        _log.warning("batch %s telemetry capture failed (non-fatal): %r", batch_id, exc)


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
