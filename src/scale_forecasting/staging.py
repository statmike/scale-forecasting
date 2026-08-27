"""Stage a run's artifacts to GCS — the shared submission-prep seam.

Both launchers (Dataproc Serverless batch and Ray on Vertex) stage the validated run config to
``gs://<code>/runs/<run_id>.json`` in exactly the same way: the JSON is the lossless
reproducibility record, and its digest is the shared ``run_id``, so a mixed run stages one config
identically regardless of runtime. This module single-sources that write so the two paths cannot
drift.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RunConfig


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
