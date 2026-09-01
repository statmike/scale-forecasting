"""The resolved Dataproc deployment envelope — what infrastructure a run has to work with.

`BatchInfra` is the frozen answer to "where does code go, what image runs it, as whom, on which
subnet" — resolved once from ``SF_*`` environment (parity with `Settings`) or from a
``terraform output -json`` map, then passed down unchanged.

**It is deliberately not part of `submit`.** Most of what consults it never submits a serverless
batch: the Dataproc-*cluster* path reads the packed-venv archive and GPU image URIs off it,
`commands` renders a portable ``gcloud`` line from it, `compute_fallback` inspects it to decide
whether a runtime is even reachable, and `main` threads it through every family launch. Living in
the submitter made it look like a submitter detail; it is the deployment's shape.

`serverless_dep_properties` lives here for the same reason — it is the one place the two dependency
envelopes (shared runtime image vs. packed-venv archive) are spelled out, and both the real submit
and the printed ``gcloud`` command read it, so they cannot disagree about how deps arrive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigError

# Batch-submission infra env vars (beyond the SF_* identity Settings resolves). Kept together so the
# docstring, resolve(), and any tooling agree.
_ENV_CODE_BUCKET = "SF_CODE_BUCKET"
_ENV_CONTAINER_IMAGE = "SF_CONTAINER_IMAGE"
_ENV_COMPUTE_SA = "SF_COMPUTE_SA"
_ENV_SUBNETWORK = "SF_SUBNETWORK_URI"

# Optional: the packed-venv archive URI (gs://…/envs/<hash>.tar.gz). The Dataproc-cluster runtime's
# dependency-delivery mechanism — clusters can't use the custom container, so a cluster job attaches
# this archive instead (see cluster_deps). Unset for serverless/Ray-only deployments.
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
# question (on a cluster it does not — see ``cluster_deps._VENV_DIR``, where an init action
# lands the venv at an absolute path instead, a fix serverless has no equivalent of). Default stays
# "container" until a live batch says otherwise.
_ENV_SERVERLESS_DEPS = "SF_SERVERLESS_DEPS"
_SERVERLESS_DEPS_CONTAINER = "container"
_SERVERLESS_DEPS_PACKED_VENV = "packed_venv"

# Where ``spark.archives``' ``#env`` fragment unpacks, and the interpreter inside it — relative to
# the working directory Spark localizes into.
_VENV_UNPACK_DIR = "env"
_VENV_ARCHIVE_PYTHON = f"./{_VENV_UNPACK_DIR}/bin/python"

# Batch max-runtime cap (``ExecutionConfig.ttl``). Dataproc Serverless applies a DEFAULT ttl of 4h
# when none is set — which silently CANCELS a longer-running batch mid-flight (a full 100k explode
# run can exceed 4h, and the cancel kills it before it writes its run_registry summary row, so the
# efficiency views render blank). We set an explicit, generous 24h so a full-scale run finishes on
# its own. Override per-submit with ``--ttl``. This bounds the batch's lifetime, NOT the client wait
# (that's _WAIT_TIMEOUT_SECONDS): a serverless batch bills only for what it uses, so a high ceiling
# costs nothing extra — it just stops the platform from guillotining a healthy long run.
_DEFAULT_TTL_SECONDS = 86400


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
