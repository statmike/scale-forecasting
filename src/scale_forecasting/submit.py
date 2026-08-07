"""Submit a forecast run to Dataproc Serverless (BUILD B2) — the local/Composer launcher.

This is the ``[spark]``-extra, ADC-authenticated helper that turns a validated
:class:`~scale_forecasting.config.RunConfig` into a running Dataproc Serverless batch. It is the
same call path a Composer DAG uses later (B6): reproducing at runtime the exact delivery the
Terraform ``seed`` module does for the seed job, but for *forecast* runs and driven from Python
(runs live in the registry, not Terraform state).

What :func:`submit_batch` does:

1. **Package the code at runtime** — zip ``src/`` and upload it to the code bucket, so the batch
   loads current code via ``python_file_uris`` rather than anything baked into the container image
   (the B0.4 code-delivery decision). Upload the standalone ``spark_main`` shim as the ``gs://``
   main file (Dataproc runs it as ``__main__``; it absolute-imports the in-package dispatch logic).
2. **Stage the run config** — write the validated config to ``gs://<code>/runs/<run_id>.json`` and
   pass it as ``--config-uri``. The JSON is the lossless reproducibility record (G3).
3. **Deliver infra identity as args** — the ``--sf-*`` flags (Dataproc rejects driver-env), built
   from :class:`~scale_forecasting.settings.Settings` via :func:`._infra_args.infra_args_from`.
4. **Submit** through :class:`~google.cloud.dataproc_v1.BatchControllerClient` (regional endpoint),
   optionally capping executors (``--max-executors`` → ``spark.dynamicAllocation.maxExecutors``, how
   the naive demo is throttled), and return the batch id.

``multi`` is orchestrated here too (:func:`submit_multi`): it fans out one child ``explode`` batch
per model family — all under **one** shared ``run_id`` and one ``run_registry`` header (C3), the
same contributor-mode contract :func:`main.run` uses. Family-splitting happens submit-side (not
on-cluster) because ``google-cloud-dataproc`` lives in the ``[spark]`` extra and is absent from the
runtime container.

Public surface: ``BatchInfra``, ``submit_batch``, ``submit_multi``, ``main``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._infra_args import infra_args_from
from .errors import ConfigError, EngineError, get_logger

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
    - ``driver_cores`` / ``executor_cores`` / ``executor_instances`` / ``max_executors`` — the
      resolved cluster sizing and the autoscaling cap (our naive throttle shows up here).
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
    tel["runtime_version"] = (
        getattr(runtime_config, "version", None) or None if runtime_config else None
    )
    tel["container_image"] = (
        getattr(runtime_config, "container_image", None) or None if runtime_config else None
    )

    env = getattr(batch, "environment_config", None)
    exec_cfg = getattr(env, "execution_config", None) if env else None
    tel["service_account"] = (
        getattr(exec_cfg, "service_account", None) or None if exec_cfg else None
    )
    tel["subnetwork_uri"] = getattr(exec_cfg, "subnetwork_uri", None) or None if exec_cfg else None

    return tel


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
    models: list[str] | None = None,
    manage_header: bool = True,
) -> object:
    """Assemble the ``dataproc_v1.Batch`` for one forecast run (pure — builds the message only).

    Mirrors the Terraform seed batch: runtime container + package zip on ``python_file_uris``, the
    ``spark_main`` shim as the ``gs://`` main file, ``--engine``/``--config-uri`` + the ``--sf-*``
    infra args. ``max_executors`` caps ``spark.dynamicAllocation.maxExecutors`` (naive throttle).

    ``models`` / ``manage_header`` carry the Arc B contract on-cluster: ``--models m1,m2`` restricts
    the executed subset (run_id still derives from the full staged config) and ``--manage-header
    false`` puts the on-cluster engine in contributor mode (``main.run`` owns the shared header).
    Both are appended to ``args`` **only when non-default**, so a standalone submit builds the exact
    same arg list as before (existing batches / snapshot tests unchanged).
    """
    from google.cloud import dataproc_v1 as dataproc

    args = ["--engine", engine, "--config-uri", config_uri, *infra_args_from(settings)]
    if models is not None:
        args += ["--models", ",".join(models)]
    if not manage_header:
        args += ["--manage-header", "false"]
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
    """Zip ``src/`` + upload it and the standalone launcher shim to the code bucket.

    Returns ``(package_uri, launcher_uri)``. The zip name carries an md5 so a code change is a new
    object (no in-place overwrite races), matching the seed module's runtime-delivery contract. The
    launcher is ``src/spark_main.py`` — a top-level shim (absolute import), *not* the in-package
    ``spark_entry`` module: Dataproc runs the main file as ``__main__`` with no package context, so
    a file with relative imports would ``ImportError``. The zip supplies the package it imports.
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

    launcher_name = "runs/spark_main.py"
    launcher_local = _SRC_DIR / "spark_main.py"
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
    models: list[str] | None = None,
    manage_header: bool = True,
    batch_id: str | None = None,
    wait: bool = True,
) -> str:
    """Stage code + config and submit one Dataproc Serverless forecast batch; return its batch id.

    Resolves infra from the environment when not passed (G1). ``engine`` is the Spark method
    (``explode``/``naive``); ``multi`` fans out via :func:`submit_multi`. ``n_series`` overrides
    ``data.series_limit`` at submit time — the scale knob for the 10 → 100 → 1k → 100k story;
    because it changes the config it yields a distinct ``run_id``/header per scale (each scale is
    its own queryable run). With ``wait`` the call blocks until the batch is terminal (parity with
    the Terraform seed apply) and then stamps Dataproc job telemetry onto the header
    (:func:`_stamp_job_telemetry`, best-effort); otherwise it returns once submitted (no telemetry).

    ``models`` / ``manage_header`` carry the Arc B contract to the cluster. The **full** ``cfg`` is
    always staged (so its ``run_id`` matches :func:`main.run`'s), while ``models`` restricts the
    executed subset on-cluster and ``manage_header=False`` runs the engine in contributor mode
    (``main.run`` owns the shared header). Both default to standalone behavior, so every existing
    caller stages and submits exactly as before.

    ``batch_id`` overrides the derived ``sf-<engine>-<run_id>`` id. It exists for
    :func:`submit_multi`, where every family child stages the **same** full cfg (one shared
    ``run_id``) as ``explode`` — so the derived id would collide across families; the caller
    supplies a per-family id instead. When ``None`` (every standalone caller) the id is derived as
    before.
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
    batch_id = batch_id or _batch_id(run_id, engine)

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
        models=models,
        manage_header=manage_header,
    )

    client = _batch_client(settings.region)
    parent = f"projects/{settings.project_id}/locations/{settings.region}"
    _log.info("submitting batch %s (engine=%s) to %s", batch_id, engine, parent)
    operation = client.create_batch(parent=parent, batch=batch, batch_id=batch_id)  # type: ignore[attr-defined]
    if wait:
        result = operation.result()  # blocks until terminal
        state = getattr(result, "state", None)
        state_name = getattr(state, "name", str(state))
        _log.info("batch %s finished: state=%s", batch_id, state_name)
        # Stamp Dataproc-level telemetry (cluster sizing, wall/overhead split, DCU usage) onto the
        # header — before the raise below, so even a FAILED batch (whose on-cluster update_header
        # never ran) still gets its sizing recorded. Best-effort: any failure here is logged and
        # swallowed, never sinking the run (the forecasts + registry rows already landed).
        _stamp_job_telemetry(client, parent, batch_id, run_id, settings)
        # A non-SUCCEEDED terminal state must fail loudly — the caller/CLI otherwise exits 0 on a
        # failed batch (the header stays RUNNING and the failure is silent). SUCCEEDED is the one
        # green state; CANCELLED/FAILED and anything else raise with the batch's own status message.
        if state_name != "SUCCEEDED":
            detail = getattr(result, "state_message", "") or "(no state_message)"
            raise EngineError(f"batch {batch_id} terminal state {state_name}: {detail}")
    return batch_id


def _stamp_job_telemetry(
    client: Any, parent: str, batch_id: str, run_id: str, settings: Settings
) -> None:
    """Read the finished batch's telemetry and write it to the run header (best-effort).

    A fresh ``get_batch`` (the LRO result can carry incomplete ``approximate_usage``) → the pure
    :func:`extract_job_telemetry` → ``update_header(job_telemetry=<dict>)``. The header column is a
    native ``JSON`` type whose query parameter serializes the value itself, so we pass the telemetry
    **dict** (not a pre-serialized string, which would double-encode). Wrapped so any failure (API
    error, missing field, header not yet written) is logged and swallowed: telemetry is a
    nice-to-have overlay on an already-complete run, never a reason to fail it (CONTRACTS §3.3).
    """
    from .registry import bq

    try:
        fetched = client.get_batch(name=f"{parent}/batches/{batch_id}")
        telemetry = extract_job_telemetry(fetched)
        bq.update_header(run_id, settings=settings, job_telemetry=telemetry)
        _log.info("batch %s telemetry stamped: %s", batch_id, telemetry)
    except Exception as exc:  # noqa: BLE001 - telemetry is best-effort, never fatal
        _log.warning("batch %s telemetry capture failed (non-fatal): %r", batch_id, exc)


def submit_multi(
    cfg: RunConfig,
    *,
    n_series: int | None = None,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    wait: bool = True,
) -> list[str]:
    """Fan a run out into one child ``explode`` batch per family — under **one** shared run_id.

    Splits ``cfg.models`` by each model's ``family`` (statistical / ml / deep_learning / native) and
    submits an independent explode batch per family — separate autoscaling + failure domains — but
    all under a **single** ``run_id`` and a **single** ``run_registry`` header (C3). This is the
    contributor-mode contract :func:`main.run` already uses:

    1. **One run_id from the full cfg.** ``run_id = make_run_id(cfg)`` is computed once over the
       whole config (``n_series`` applied first so a scale override still yields one id); every
       child stages that same full cfg, so all children derive the identical id — the leaderboard
       shows the whole multi run as one ``run_id`` with every family under it, not one per family.
    2. **One header owner.** :func:`submit_multi` writes the shared header (RUNNING) up front and
       finalizes it after every child joins; each child runs the engine with ``manage_header=False``
       (contributor mode), so no child touches the header and there is no UPDATE race.
    3. **Per-family executed subset + batch id.** Each child gets ``models=<family>`` (restricting
       what it runs on-cluster while the full cfg stays staged) and an explicit per-family
       ``batch_id`` (``sf-multi-<family>-<run_id>``), because the derived ``sf-explode-<run_id>`` id
       would be identical across families (same run_id) and collide in Dataproc.

    Orchestrated here rather than on-cluster because ``google-cloud-dataproc`` isn't in the runtime
    container. Blocks per child when ``wait`` (families run sequentially — B2 keeps the submit path
    simple; each child's own batch still autoscales independently). The shared header is finalized
    COMPLETED iff every child succeeded, else FAILED (finalized before re-raising the first failure,
    so the run stays queryable and the CLI exits non-zero). Returns the child batch ids.
    """
    import time

    from .registry import bq
    from .registry.ids import make_run_id
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()

    # One run_id for the whole multi run: apply the scale override first (so it's part of the hashed
    # cfg), then hash the full cfg once. Every child stages this same cfg → same run_id.
    if n_series is not None:
        cfg = cfg.model_copy(
            update={"data": cfg.data.model_copy(update={"series_limit": n_series})}
        )
    run_id = make_run_id(cfg)

    families = split_models_by_family(cfg)
    _log.info(
        "multi: %d family batches for run '%s' under one run_id=%s",
        len(families),
        cfg.run_name,
        run_id,
    )

    # One header owner (mirrors main.run): write RUNNING once, finalize after all children join.
    bq.ensure_tables(cfg, settings=settings)
    bq.write_header(cfg, run_id, settings=settings)

    started = time.perf_counter()
    batch_ids: list[str] = []
    first_error: BaseException | None = None
    for family, models in families.items():
        try:
            batch_ids.append(
                submit_batch(
                    cfg,  # full cfg → shared run_id; models= restricts the on-cluster subset
                    engine="explode",
                    settings=settings,
                    infra=infra,
                    models=models,
                    manage_header=False,  # contributor mode: this function owns the shared header
                    batch_id=_batch_id(f"{family}-{run_id}", "multi"),
                    wait=wait,
                )
            )
            _log.info("multi: submitted family=%s models=%s", family, models)
        except Exception as exc:  # noqa: BLE001 - captured, header finalized below, re-raised
            first_error = first_error or exc
            _log.warning("multi: family=%s failed: %r", family, exc)

    runtime_seconds = time.perf_counter() - started
    bq.update_header(
        run_id,
        settings=settings,
        status="COMPLETED" if first_error is None else "FAILED",
        runtime_seconds=runtime_seconds,
        n_series=cfg.data.series_limit,
    )
    if first_error is not None:
        raise first_error
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
