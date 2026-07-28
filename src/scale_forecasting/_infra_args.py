"""Infra-identity CLI args ↔ ``SF_*`` environment — the one delivery mapping (G1).

Dataproc Serverless allowlists Spark property prefixes and rejects driver-env
(``spark.kubernetes.driverEnv.*`` → "unsupported properties"), so a batch cannot hand the
infra identity to the driver as environment variables. Instead every entrypoint accepts the
identity as ``--sf-*`` job args and exports them into ``os.environ`` before
:meth:`~scale_forecasting.settings.Settings.resolve` runs — keeping env-based ``Settings`` the
single G1 seam rather than forking a "resolve from args" path.

This module owns that mapping in ONE place so the seed job, the Spark forecast launcher
(``spark_entry``), the local submit helper (``submit``), and the Terraform modules that build the
``--sf-*`` arg list all stay in agreement. Local runs pass no ``--sf-*`` and use the ambient
``SF_*`` environment untouched.

Public surface: ``INFRA_ARG_ENV``, ``add_infra_args``, ``export_infra_env``, ``infra_args_from``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from .settings import Settings

# The (argparse dest → SF_* env var) mapping. ``add_infra_args`` registers ``--sf-*`` flags whose
# argparse dests are these first elements; ``export_infra_env`` copies each provided value into the
# matching env var before Settings.resolve().
INFRA_ARG_ENV: tuple[tuple[str, str], ...] = (
    ("sf_project_id", "SF_PROJECT_ID"),
    ("sf_connection", "SF_CONNECTION"),
    ("sf_warehouse_uri", "SF_WAREHOUSE_URI"),
    ("sf_dataset_id", "SF_DATASET_ID"),
    ("sf_region", "SF_REGION"),
)


def add_infra_args(parser: argparse.ArgumentParser) -> None:
    """Register the ``--sf-*`` infra-identity flags on a parser (all default ``None``).

    When unset (local runs) the ambient ``SF_*`` environment is used as-is; when set (a cluster
    batch) :func:`export_infra_env` promotes them to ``os.environ`` before resolution.
    """
    for dest, env_name in INFRA_ARG_ENV:
        flag = "--" + dest.replace("_", "-")
        parser.add_argument(
            flag, type=str, default=None, help=f"sets {env_name} in the environment"
        )


def export_infra_env(ns: argparse.Namespace) -> None:
    """Copy any provided ``--sf-*`` args from ``ns`` into ``os.environ`` (only when set).

    Keeps env-based ``Settings.resolve()`` the single G1 seam without a parallel "resolve from
    args" path. Values left unset are skipped, so a local run's ambient environment is untouched.
    """
    for dest, env_name in INFRA_ARG_ENV:
        value = getattr(ns, dest, None)
        if value:
            os.environ[env_name] = value


def infra_args_from(settings: Settings) -> list[str]:
    """Build the ``--sf-*`` arg list carrying ``settings`` to a cluster batch (submit side).

    The inverse of :func:`add_infra_args` / :func:`export_infra_env`: turns a resolved
    :class:`~scale_forecasting.settings.Settings` into the flat ``["--sf-project-id", ...]`` list a
    Dataproc batch passes so the driver re-materializes the same identity via env.
    """
    values = {
        "sf_project_id": settings.project_id,
        "sf_connection": settings.connection,
        "sf_warehouse_uri": settings.warehouse_uri,
        "sf_dataset_id": settings.dataset_id,
        "sf_region": settings.region,
    }
    args: list[str] = []
    for dest, _env_name in INFRA_ARG_ENV:
        value = values[dest]
        if value:
            args += ["--" + dest.replace("_", "-"), value]
    return args
