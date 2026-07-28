"""Submit a forecast run to Dataproc Serverless (BUILD B2) — the local/Composer launcher.

This is the ``[spark]``-extra, ADC-authenticated helper that turns a validated
:class:`~scale_forecasting.config.RunConfig` into a running Dataproc Serverless batch. It is the
same call path a Composer DAG uses later (B6): reproducing at runtime the exact delivery the
Terraform ``seed`` module does for the seed job, but for *forecast* runs and driven from Python
(runs live in the registry, not Terraform state).

What :func:`submit_batch` does:

1. **Package the code at runtime** — zip ``src/`` and upload it to the code bucket, so the batch
   loads current code via ``python_file_uris`` rather than anything baked into the container image
   (the B0.4 code-delivery decision). Upload the thin ``spark_entry`` launcher as the ``gs://`` main
   file.
2. **Stage the run config** — write the validated config to ``gs://<code>/runs/<run_id>.json`` and
   pass it as ``--config-uri``. The JSON is the lossless reproducibility record (G3).
3. **Deliver infra identity as args** — the ``--sf-*`` flags (Dataproc rejects driver-env), built
   from :class:`~scale_forecasting.settings.Settings` via :func:`._infra_args.infra_args_from`.
4. **Submit** through :class:`~google.cloud.dataproc_v1.BatchControllerClient` (regional endpoint),
   optionally capping executors (``--max-executors`` → ``spark.dynamicAllocation.maxExecutors``, how
   the naive demo is throttled), and return the batch id.

``multi`` is orchestrated here too (:func:`submit_multi`): it fans out one child ``explode`` batch
per model family, because ``google-cloud-dataproc`` lives in the ``[spark]`` extra and is absent
from the runtime container — so family-splitting can only happen submit-side, not on-cluster.

Public surface: ``BatchInfra``, ``submit_batch``, ``submit_multi``, ``main``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ._infra_args import infra_args_from
from .errors import ConfigError, get_logger

if TYPE_CHECKING:
    from .config import RunConfig
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

_DEFAULT_RUNTIME_VERSION = "2.2"


@dataclass(frozen=True)
class BatchInfra:
    """Dataproc-batch infra identity — what submitting a batch needs beyond :class:`Settings`.

    Resolved from ``SF_*`` env (parity with ``Settings``) or ``terraform output``. Frozen and
    passed down so a run's every child batch (multi) targets the same infra.
    """

    code_bucket: str  # bucket the package zip + launcher + config JSON are staged to
    container_image: str  # full runtime image incl. tag
    compute_sa: str  # runtime SA the batch runs as (scale-forecasting-compute)
    subnetwork_uri: str  # subnet with Private Google Access + internal-ingress firewall
    runtime_version: str = _DEFAULT_RUNTIME_VERSION

    @classmethod
    def resolve(cls) -> BatchInfra:
        """Build from the ``SF_*`` batch-infra environment; raise naming the first missing var."""
        required = {
            "code_bucket": _ENV_CODE_BUCKET,
            "container_image": _ENV_CONTAINER_IMAGE,
            "compute_sa": _ENV_COMPUTE_SA,
            "subnetwork_uri": _ENV_SUBNETWORK,
        }
        values: dict[str, str] = {}
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
        )

    @classmethod
    def from_terraform_outputs(
        cls, outputs: dict[str, str], image_tag: str = "latest"
    ) -> BatchInfra:
        """Build from a ``terraform output -json`` value map (local dev/tests).

        Reads the keys the ``terraform/main`` stage emits — ``code_bucket``, ``runtime_image_repo``
        (a base path; ``image_tag`` is appended), ``compute_sa``, ``subnetwork_uri``.
        """
        try:
            return cls(
                code_bucket=outputs["code_bucket"],
                container_image=f"{outputs['runtime_image_repo']}:{image_tag}",
                compute_sa=outputs["compute_sa"],
                subnetwork_uri=outputs["subnetwork_uri"],
            )
        except KeyError as exc:
            raise ConfigError(f"terraform outputs missing key: {exc.args[0]}") from exc


# --- pure: batch spec assembly (no network) ------------------------------------


def _batch_id(run_id: str, engine: str) -> str:
    """A Dataproc batch id: ``sf-<engine>-<run_id>``, clamped to the 4-63 char / alnum+hyphen rule.

    ``run_id`` is already a slug + hex digest; prefix the engine and trim to fit.
    """
    raw = f"sf-{engine}-{run_id}"
    return raw[:63].rstrip("-")


def build_batch(
    *,
    infra: BatchInfra,
    settings: Settings,
    engine: str,
    package_uri: str,
    launcher_uri: str,
    config_uri: str,
    max_executors: int | None = None,
) -> object:
    """Assemble the ``dataproc_v1.Batch`` for one forecast run (pure — builds the message only).

    Mirrors the Terraform seed batch: runtime container + package zip on ``python_file_uris``,
    ``spark_entry`` as the ``gs://`` main file, ``--engine``/``--config-uri`` + the ``--sf-*`` infra
    args. ``max_executors`` caps ``spark.dynamicAllocation.maxExecutors`` (the naive-demo throttle).
    """
    from google.cloud import dataproc_v1 as dataproc

    args = ["--engine", engine, "--config-uri", config_uri, *infra_args_from(settings)]
    properties = {}
    if max_executors is not None:
        properties["spark.dynamicAllocation.maxExecutors"] = str(max_executors)

    return dataproc.Batch(
        pyspark_batch=dataproc.PySparkBatch(
            main_python_file_uri=launcher_uri,
            python_file_uris=[package_uri],
            args=args,
        ),
        runtime_config=dataproc.RuntimeConfig(
            version=infra.runtime_version,
            container_image=infra.container_image,
            properties=properties,
        ),
        environment_config=dataproc.EnvironmentConfig(
            execution_config=dataproc.ExecutionConfig(
                service_account=infra.compute_sa,
                subnetwork_uri=infra.subnetwork_uri,
            )
        ),
    )


# --- I/O: staging + submit -----------------------------------------------------


def _stage_code(infra: BatchInfra) -> tuple[str, str]:
    """Zip ``src/`` + upload it and the ``spark_entry`` launcher to the code bucket.

    Returns ``(package_uri, launcher_uri)``. The zip name carries an md5 so a code change is a new
    object (no in-place overwrite races), matching the seed module's runtime-delivery contract.
    """
    import hashlib
    import io
    import zipfile

    from google.cloud import storage

    # Build the zip in memory (deterministic walk) and hash it for the object name.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pkg_root = _SRC_DIR / "scale_forecasting"
        for path in sorted(pkg_root.rglob("*.py")):
            zf.write(path, arcname=str(path.relative_to(_SRC_DIR)))
    data = buf.getvalue()
    code_hash = hashlib.md5(data).hexdigest()[:8]  # noqa: S324 - non-crypto object-name tag

    client = storage.Client()
    bucket = client.bucket(infra.code_bucket)
    pkg_name = f"runs/scale_forecasting-{code_hash}.zip"
    bucket.blob(pkg_name).upload_from_string(data, content_type="application/zip")

    launcher_name = "runs/spark_entry.py"
    launcher_local = _SRC_DIR / "scale_forecasting" / "spark_entry.py"
    bucket.blob(launcher_name).upload_from_filename(str(launcher_local))

    return (
        f"gs://{infra.code_bucket}/{pkg_name}",
        f"gs://{infra.code_bucket}/{launcher_name}",
    )


def _stage_config(cfg: RunConfig, run_id: str, infra: BatchInfra) -> str:
    """Write the validated config to ``gs://<code>/runs/<run_id>.json`` and return the URI (G3)."""
    import json

    from google.cloud import storage

    client = storage.Client()
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, indent=2)
    name = f"runs/{run_id}.json"
    storage.Client.bucket(client, infra.code_bucket).blob(name).upload_from_string(
        payload, content_type="application/json"
    )
    return f"gs://{infra.code_bucket}/{name}"


def _batch_client(region: str) -> object:
    """A regional :class:`BatchControllerClient` (Dataproc batches are a regional resource)."""
    from google.api_core.client_options import ClientOptions
    from google.cloud import dataproc_v1 as dataproc

    return dataproc.BatchControllerClient(
        client_options=ClientOptions(api_endpoint=f"{region}-dataproc.googleapis.com:443")
    )


def submit_batch(
    cfg: RunConfig,
    *,
    engine: str = "explode",
    n_series: int | None = None,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    max_executors: int | None = None,
    wait: bool = True,
) -> str:
    """Stage code + config and submit one Dataproc Serverless forecast batch; return its batch id.

    Resolves infra from the environment when not passed (G1). ``engine`` is the Spark method
    (``explode``/``naive``); ``multi`` fans out via :func:`submit_multi`. ``n_series`` overrides
    ``data.series_limit`` at submit time — the scale knob for the 10 → 100 → 1k → 100k story;
    because it changes the config it yields a distinct ``run_id``/header per scale (each scale is
    its own queryable run). With ``wait`` the call blocks until the batch is terminal (parity with
    the Terraform seed apply); otherwise it returns once submitted.
    """
    from .registry.ids import make_run_id
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    if n_series is not None:
        cfg = cfg.model_copy(
            update={"data": cfg.data.model_copy(update={"series_limit": n_series})}
        )
    run_id = make_run_id(cfg)
    batch_id = _batch_id(run_id, engine)

    package_uri, launcher_uri = _stage_code(infra)
    config_uri = _stage_config(cfg, run_id, infra)
    batch = build_batch(
        infra=infra,
        settings=settings,
        engine=engine,
        package_uri=package_uri,
        launcher_uri=launcher_uri,
        config_uri=config_uri,
        max_executors=max_executors,
    )

    client = _batch_client(settings.region)
    parent = f"projects/{settings.project_id}/locations/{settings.region}"
    _log.info("submitting batch %s (engine=%s) to %s", batch_id, engine, parent)
    operation = client.create_batch(parent=parent, batch=batch, batch_id=batch_id)  # type: ignore[attr-defined]
    if wait:
        result = operation.result()  # blocks until terminal
        _log.info("batch %s finished: state=%s", batch_id, getattr(result, "state", "?"))
    return batch_id


def submit_multi(
    cfg: RunConfig,
    *,
    n_series: int | None = None,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    wait: bool = True,
) -> list[str]:
    """Fan a run out into one child ``explode`` batch per model family (the ``multi`` method).

    Splits ``cfg.models`` by each model's ``family`` (statistical / ml / deep_learning / native)
    and submits an independent explode batch per family — separate autoscaling + failure domains,
    and a distinct ``run_id``/header each (the family subset changes the config, hence the id).
    ``n_series`` (if set) is threaded to every child so all families run the same scale.
    Orchestrated here rather than on-cluster because ``google-cloud-dataproc`` isn't in the runtime
    container. Returns the child batch ids.
    """
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()

    families = split_models_by_family(cfg)
    _log.info("multi: %d family batches for run '%s'", len(families), cfg.run_name)
    batch_ids: list[str] = []
    for family, models in families.items():
        child = cfg.model_copy(update={"models": models, "spark_method": "explode"})
        batch_ids.append(
            submit_batch(
                child,
                engine="explode",
                n_series=n_series,
                settings=settings,
                infra=infra,
                wait=wait,
            )
        )
        _log.info("multi: submitted family=%s models=%s", family, models)
    return batch_ids


def split_models_by_family(cfg: RunConfig) -> dict[str, list[str]]:
    """Group ``cfg.models`` by each model's registered ``family`` (pure; order-preserving).

    The grouping ``multi`` fans out on. Unknown model names surface as a :class:`ModelError` from
    the factory rather than being silently dropped.
    """
    from .models import get_model

    families: dict[str, list[str]] = {}
    for name in cfg.models:
        family = get_model(name).family
        families.setdefault(family, []).append(name)
    return families


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m scale_forecasting.submit --config run.json [--engine ...]``."""
    from .config import load_config

    p = argparse.ArgumentParser(prog="submit", description="Submit a forecast run to Dataproc.")
    p.add_argument("--config", required=True, help="path to the run config JSON")
    p.add_argument("--engine", default="explode", choices=("explode", "naive", "multi"))
    p.add_argument("--n-series", type=int, default=None, help="override series_limit (scale knob)")
    p.add_argument(
        "--max-executors", type=int, default=None, help="cap dynamicAllocation executors"
    )
    p.add_argument("--no-wait", action="store_true", help="return once submitted (don't block)")
    ns = p.parse_args(argv)

    cfg = load_config(ns.config)
    if ns.engine == "multi":
        ids = submit_multi(cfg, n_series=ns.n_series, wait=not ns.no_wait)
        _log.info("multi submitted: %s", ids)
    else:
        batch_id = submit_batch(
            cfg,
            engine=ns.engine,
            n_series=ns.n_series,
            max_executors=ns.max_executors,
            wait=not ns.no_wait,
        )
        _log.info("submitted: %s", batch_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
