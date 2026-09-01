"""Stage a run's artifacts to GCS — the shared submission-prep seam.

Every launcher (Dataproc Serverless batch, Dataproc cluster, Ray on Vertex) stages the validated run
config to ``gs://<code>/runs/<run_id>.json`` in exactly the same way: the JSON is the lossless
reproducibility record, and its digest is the shared ``run_id``, so a mixed run stages one config
identically regardless of runtime. This module single-sources that write so the paths cannot drift.

`stage_code` is here for the same reason. The package zip and the launcher shim are *the same two
objects* on the batch and cluster surfaces — same builder, same md5-named blob, same bucket — and
three callers want them: `submit.submit_batch`, `cluster_submit.submit_cluster_job`, and
`main.stage_run` (which stages without submitting anything). It lived on the batch submitter, so the
cluster path had to import a private name out of it to run a job at all.

Everything here takes a plain ``code_bucket`` string rather than an infra object: the two Dataproc
surfaces carry a `BatchInfra` and the Ray surface a `RayInfra`, and the only field any of this needs
is the bucket. Keeping the seam infra-agnostic is what lets all three share it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RunConfig

# The repo's ``src/`` — the parent of this package, where the standalone launcher shim lives.
_SRC_DIR = Path(__file__).resolve().parent.parent


def stage_config(cfg: RunConfig, run_id: str, code_bucket: str) -> str:
    """Write the validated config to ``gs://<code_bucket>/runs/<run_id>.json``; return the URI.

    The payload is ``sort_keys``-stable, indented JSON — a deterministic, human-readable record
    that any ADC-authenticated reader can fetch, which is what makes the staged URI a portable
    handle to the run.
    """
    from google.cloud import storage

    client = storage.Client()
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, indent=2)
    name = f"runs/{run_id}.json"
    client.bucket(code_bucket).blob(name).upload_from_string(
        payload, content_type="application/json"
    )
    return f"gs://{code_bucket}/{name}"


def stage_code(code_bucket: str) -> tuple[str, str]:
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
    bucket = client.bucket(code_bucket)
    pkg_name = f"runs/scale_forecasting-{code_hash}.zip"
    bucket.blob(pkg_name).upload_from_string(data, content_type="application/zip")

    launcher_name = "runs/spark_main.py"
    launcher_local = _SRC_DIR / "spark_main.py"
    bucket.blob(launcher_name).upload_from_filename(str(launcher_local))

    return (
        f"gs://{code_bucket}/{pkg_name}",
        f"gs://{code_bucket}/{launcher_name}",
    )


def stage_dag(dag_source: str, run_id: str, code_bucket: str) -> str:
    """Write a rendered Airflow DAG to ``gs://<code_bucket>/runs/dag_<run_id>.py``; return the URI.

    The Composer counterpart to `stage_config`: `airflow_emit.emit_airflow_dag` renders a run's
    ``dag_<run_id>.py`` (whose ``CONFIG_URI`` points at the config staged alongside it), and this
    uploads it next to that config so a deployment can sync it into the Airflow DAGs folder. The
    source is the deterministic emitter output, so re-staging an unchanged config overwrites with
    byte-identical text.
    """
    from google.cloud import storage

    client = storage.Client()
    name = f"runs/dag_{run_id}.py"
    client.bucket(code_bucket).blob(name).upload_from_string(
        dag_source, content_type="text/x-python"
    )
    return f"gs://{code_bucket}/{name}"


def stage_manifest(manifest: dict[str, object], run_id: str, code_bucket: str) -> str:
    """Write the run's reproducibility manifest to ``gs://<code_bucket>/runs/<run_id>.plan.json``.

    The manifest sits next to the staged config and records what would launch this run — the config
    digest, fan-out, both command tiers, staged URIs, and runtime — so "what command produced run
    X?" stays answerable forever. Deterministic (``sort_keys``, indented) like the config, and
    written by the caller after staging so the URIs it records are the real ones.
    """
    from google.cloud import storage

    client = storage.Client()
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    name = f"runs/{run_id}.plan.json"
    client.bucket(code_bucket).blob(name).upload_from_string(
        payload, content_type="application/json"
    )
    return f"gs://{code_bucket}/{name}"
