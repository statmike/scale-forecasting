"""Vertex-Ray infra identity — the ``SF_*`` envelope a Ray cluster launch needs.

The Ray sibling of `batch_infra`: the deployment-scoped facts that are *not* the run config and not
`Settings` — which service account the cluster runs as, which bucket the run config is staged to,
and how the cluster is attached to the network. Kept in its own module because two very different
callers need it and neither should have to import a launcher to get it: `ray_submit.submit_ray`
resolves it to create a cluster, and `main` resolves it to plan one.

Also the home of the Vertex-Ray *version* defaults, because the supported-version constraint is a
property of the deployment envelope (which image Vertex will accept) rather than of any one run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigError

# Ray-cluster infra env vars (beyond the SF_* identity Settings resolves). Kept together so the
# docstring, resolve(), and any tooling agree. code_bucket + compute_sa are shared with the Spark
# batch; network (optional) is a VPC for a private endpoint. There is deliberately no custom-image
# var: Ray always runs on Vertex's prebuilt image + a uv runtime_env (a custom node image fails
# Vertex Ray GPU-node provisioning — see `code_delivery.build_runtime_env`). SF_CONTAINER_IMAGE
# stays Spark-only.
_ENV_NETWORK = "SF_RAY_NETWORK"
_ENV_NETWORK_ATTACHMENT = "SF_RAY_NETWORK_ATTACHMENT"
_ENV_COMPUTE_SA = "SF_COMPUTE_SA"
_ENV_CODE_BUCKET = "SF_CODE_BUCKET"
_ENV_RAY_VERSION = "SF_RAY_VERSION"

# Vertex Ray's supported Ray version + our runtime Python. Vertex AI accepts only a fixed set of Ray
# versions for the cluster image (2.9.3 / 2.33.0 / 2.42.0 / 2.47.1; on Python 3.11 only 2.42 or
# 2.47), and the client-side Ray MUST match the cluster's: the JobSubmissionClient handshake (GET
# /api/version) hangs on a version-skewed dashboard rather than erroring cleanly. So the [ray] extra
# is capped to a supported range (see pyproject.toml) and this default matches. Overridable via
# SF_RAY_VERSION to select a different *supported* image without a code change — but the client Ray
# must still equal it.
_DEFAULT_RAY_VERSION = "2.47"
_DEFAULT_PYTHON_VERSION = "3.11"


@dataclass(frozen=True)
class RayInfra:
    """Vertex-Ray infra identity — what launching a cluster needs beyond `Settings`.

    Resolved from ``SF_*`` env (parity with ``Settings`` / ``BatchInfra``) or ``terraform output``.

    Connectivity is one of three modes, in precedence order — the first that is set wins:

    * ``network_attachment`` (**PSC-I**, the supported private path): a network-attachment
      resource name. Vertex's tenant attaches an interface into the VPC through it, and — critically
      — this is the *only* mode under which the managed Ray dashboard / ``JobSubmissionClient``
      handshake (``GET /api/version``) is reachable off-cluster on this org; both public and VPC
      peering leave the proxy→head-node hop dead (a 30s hang → HTTP 524). Excludes ``network``.
    * ``network`` (VPC peering): a VPC (with a private-services connection) for a peered private
      endpoint. Kept for deployments that already run this way, but note the dashboard-handshake
      caveat above — prefer ``network_attachment``.
    * neither set: a public endpoint (Vertex's default; same handshake caveat).

    ``compute_sa`` is the runtime SA the cluster runs as; ``code_bucket`` where the run config JSON
    is staged. There is no custom-image field: Ray always runs on Vertex's prebuilt image and the
    uv ``runtime_env`` installs the deps on top. Unlike the Spark path, Ray never uses a custom node
    image — one fails Vertex Ray GPU-node provisioning (see `code_delivery.build_runtime_env`), so
    ``SF_CONTAINER_IMAGE`` stays Spark-only and is never read here.
    """

    compute_sa: str
    code_bucket: str
    network: str | None = None
    network_attachment: str | None = None
    ray_version: str = _DEFAULT_RAY_VERSION
    python_version: str = _DEFAULT_PYTHON_VERSION

    @classmethod
    def resolve(cls) -> RayInfra:
        """Build from the ``SF_*`` Ray-infra environment; raise naming the first missing var.

        ``SF_COMPUTE_SA`` and ``SF_CODE_BUCKET`` are required; ``SF_RAY_NETWORK_ATTACHMENT``
        (PSC-I, preferred) and ``SF_RAY_NETWORK`` (VPC peering) are optional — set at most one; if
        both are set the attachment wins. Neither set → public endpoint. Ray always runs on Vertex's
        prebuilt image + a uv ``runtime_env`` (no custom-image var).
        """
        required = {
            "compute_sa": _ENV_COMPUTE_SA,
            "code_bucket": _ENV_CODE_BUCKET,
        }
        values: dict[str, str] = {}
        for field_name, env_name in required.items():
            raw = os.environ.get(env_name)
            if not raw:
                raise ConfigError(
                    f"missing required environment variable {env_name} "
                    f"(set it, or use RayInfra.from_terraform_outputs for local dev)"
                )
            values[field_name] = raw
        return cls(
            compute_sa=values["compute_sa"],
            code_bucket=values["code_bucket"],
            network=os.environ.get(_ENV_NETWORK) or None,
            network_attachment=os.environ.get(_ENV_NETWORK_ATTACHMENT) or None,
            ray_version=os.environ.get(_ENV_RAY_VERSION) or _DEFAULT_RAY_VERSION,
        )

    @classmethod
    def from_terraform_outputs(cls, outputs: dict[str, str]) -> RayInfra:
        """Build from a ``terraform output -json`` value map (local dev/tests).

        Reads the keys the ``terraform/main`` stage emits — ``compute_sa``, ``code_bucket``, an
        optional ``network_attachment_id`` (PSC-I, preferred) and/or ``network_id`` (VPC peering);
        both absent → public endpoint, and if both are present the attachment wins. Ray always runs
        on Vertex's prebuilt image + a uv ``runtime_env``, so no image is read here.
        """
        try:
            return cls(
                compute_sa=outputs["compute_sa"],
                code_bucket=outputs["code_bucket"],
                network=outputs.get("network_id") or None,
                network_attachment=outputs.get("network_attachment_id") or None,
            )
        except KeyError as exc:
            raise ConfigError(f"terraform outputs missing key: {exc.args[0]}") from exc
