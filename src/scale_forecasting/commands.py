"""Portable launch commands for a run — the two-tier command builder (pure, no network).

A run is fully described by its staged config, so the exact command that launches it can be
reconstructed offline and handed to any machine. This module builds that command in two tiers:

* **universal** — ``python -m scale_forecasting.<launcher> --config-uri gs://…`` — runs anywhere
  the package and ADC are present; it re-stages the current code and submits.
* **native** — ``gcloud dataproc batches submit pyspark …`` — runs with only ``gcloud`` + ADC, no
  Python package. It exists **only for Spark**: a Dataproc Serverless batch maps onto a single
  ``gcloud`` verb. A Ray run is submitted through a Ray ``JobSubmissionClient`` handshake, which has
  no ``gcloud`` equivalent, so its native form is ``None`` and the universal form is the portable
  one.

The driver-arg assembly (`build_driver_args`) is the **same** one `submit.build_batch` and
`ray_submit.build_entrypoint` use to build what actually runs, so an emitted command cannot drift
from the real submission. The native command references the staged GCS artifacts, so it is
byte-faithful to the exact batch; the universal command reproduces the config-driven run with the
launcher's current code.

Public surface: ``LaunchCommands``, ``build_driver_args``, ``build_main_command``,
``build_spark_commands``, ``build_ray_commands``, ``shell_join``.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._infra_args import infra_args_from

if TYPE_CHECKING:
    from .settings import Settings
    from .submit import BatchInfra


@dataclass(frozen=True)
class LaunchCommands:
    """The command(s) that launch a run, in two portability tiers.

    ``universal`` always exists; ``native`` is set only where a package-free form exists (Spark).
    """

    runtime: str  # "spark" | "ray"
    universal: str  # python -m … (needs the package + ADC)
    native: str | None  # gcloud … (needs only gcloud + ADC); None where none exists


def shell_join(parts: list[str]) -> str:
    """Join argv into a single copy-pasteable shell line, quoting each token as needed."""
    return " ".join(shlex.quote(p) for p in parts)


def build_driver_args(
    config_uri: str,
    settings: Settings,
    *,
    models: list[str] | None = None,
    manage_header: bool = True,
) -> list[str]:
    """The on-cluster driver arg list shared by the Spark batch, the Ray entrypoint, and emission.

    ``--config-uri`` (the staged config, whose digest is the shared ``run_id``) + the ``--sf-*``
    infra identity, plus — only when non-default — ``--models m1,m2`` (executed subset) and
    ``--manage-header false`` (contributor mode). Defaults omit the optional flags so a standalone
    run builds the plain arg list. Each runtime runs its single built-in engine (Spark's
    cross-join/explode strategy, Ray's), so there is no method flag.
    """
    args: list[str] = ["--config-uri", config_uri, *infra_args_from(settings)]
    if models is not None:
        args += ["--models", ",".join(models)]
    if not manage_header:
        args += ["--manage-header", "false"]
    return args


def build_spark_commands(
    *,
    settings: Settings,
    infra: BatchInfra,
    batch_id: str,
    package_uri: str,
    launcher_uri: str,
    config_uri: str,
    max_executors: int | None = None,
    models: list[str] | None = None,
    manage_header: bool = True,
) -> LaunchCommands:
    """Both command tiers for a Dataproc Serverless (Spark) run.

    The native ``gcloud dataproc batches submit pyspark`` command reconstructs the exact
    `submit.build_batch` spec — same launcher/py-files/runtime/image/SA/subnet/ttl/properties and
    the same ``build_driver_args`` list — so it is byte-faithful to the batch that would be
    submitted. The universal command is the standalone re-submission via the ``submit`` launcher.
    """
    driver = build_driver_args(config_uri, settings, models=models, manage_header=manage_header)

    gcloud: list[str] = [
        "gcloud",
        "dataproc",
        "batches",
        "submit",
        "pyspark",
        launcher_uri,
        f"--project={settings.project_id}",
        f"--region={settings.region}",
        f"--batch={batch_id}",
        f"--py-files={package_uri}",
        f"--version={infra.runtime_version}",
        f"--container-image={infra.container_image}",
        f"--service-account={infra.compute_sa}",
        f"--subnet={infra.subnetwork_uri}",
        f"--ttl={infra.ttl_seconds}s",
    ]
    if max_executors is not None:
        gcloud.append(f"--properties=spark.dynamicAllocation.maxExecutors={max_executors}")
    gcloud += ["--", *driver]

    universal_argv = ["python", "-m", "scale_forecasting.submit", "--config-uri", config_uri]
    if max_executors is not None:
        universal_argv += ["--max-executors", str(max_executors)]

    return LaunchCommands(
        runtime="spark",
        universal=shell_join(universal_argv),
        native=shell_join(gcloud),
    )


def build_main_command(config_uri: str) -> LaunchCommands:
    """The orchestrator command that reproduces a full run from its staged config (universal-only).

    ``python -m scale_forecasting.main --config-uri gs://…/<run_id>.json`` runs the whole config —
    the Python-runtime engine (Spark or Ray) in parallel with the BigQuery-native engine under one
    ``run_id`` — so it is the faithful "run this config" form for a *mixed* run, where a single
    per-runtime command would cover only one engine. There is no ``gcloud`` verb that orchestrates
    both engines, so ``native`` is ``None``.
    """
    argv = ["python", "-m", "scale_forecasting.main", "--config-uri", config_uri]
    return LaunchCommands(runtime="main", universal=shell_join(argv), native=None)


def build_ray_commands(
    *,
    config_uri: str,
    cluster_name: str | None = None,
) -> LaunchCommands:
    """The (universal-only) command for a Ray-on-Vertex run.

    Ray job submission is a ``JobSubmissionClient`` handshake to the cluster dashboard, with no
    ``gcloud`` verb that submits a job to an existing cluster, so ``native`` is ``None`` and the
    universal ``ray_submit`` command is the portable form. ``--cluster-name`` is emitted only when
    a standing cluster is being reused.
    """
    argv = ["python", "-m", "scale_forecasting.ray_submit", "--config-uri", config_uri]
    if cluster_name is not None:
        argv += ["--cluster-name", cluster_name]
    return LaunchCommands(runtime="ray", universal=shell_join(argv), native=None)
