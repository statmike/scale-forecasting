"""Headless notebook acceptance — run every notebook on its Colab Enterprise template.

The 8 notebooks in ``notebooks/`` are first-class run drivers, and Terraform ships two Colab
Enterprise runtime templates that carry the full ``SF_*`` run identity in their env (see
``terraform/main/modules/colab``). This module is the repeatable way to *prove* each notebook runs
green on the right template — and to re-prove it whenever a notebook changes — without a human
driving the Colab browser UI.

It uses the Vertex AI **NotebookExecutionJob** API: submit a notebook to run headless on a template,
poll to a terminal state, download the executed ``.ipynb`` from GCS, and assert no cell errored. The
notebook runs in a **fresh, empty kernel**, so it relies entirely on the ``SF_*`` env baked into the
template — the same identity a human gets when they open the template — which is why this harness
and the human-open path validate the *same* thing (G1).

**Auth / endpoint choices** (learned from prior art, see docs/notebook_runtimes.md):

* **serviceAccount mode** (the run executes as the runner SA), which requires the **v1** endpoint —
  the v1beta1/``executionUser`` path needs OAuth consent and hangs in ``PENDING`` on v1. So this is
  fully headless: no browser consent, runnable from CI.
* **Never** send a custom ``notebookExecutionJobId`` — that also hangs in ``PENDING``; let the API
  mint the id and recover it from the returned operation name.
* **Refresh the token every call.** NB04 (Ray) can outlive a ~60-min OAuth token; refreshing each
  poll (mirroring ``ray_submit``'s client-refresh) keeps a long poll from 401-ing.
* The executed notebook lands at ``{gcsOutputUri}/{JOB_ID}/content.ipynb``.

**Tiers** bound cost — most notebooks orchestrate real Dataproc/Ray compute, so the harness
escalates deliberately (smoke → batch → full); see :data:`REGISTRY` and :func:`notebooks_for_tier`.

Public surface: :data:`REGISTRY`, :class:`AcceptanceResult`, :func:`run_acceptance`,
:func:`notebooks_for_tier`, and a ``python -m scale_forecasting.notebook_acceptance`` CLI.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import EngineError, get_logger

_log = get_logger(__name__)

# Template routing keys. The caller maps these to the two runtime-template resource names Terraform
# outputs (colab_main_runtime_template_id / colab_spark_runtime_template_id) — the harness stays
# id-agnostic so the same registry works across deploys.
TEMPLATE_MAIN = "main"
TEMPLATE_SPARK = "spark"

# Acceptance tiers, cheapest first. A tier RUNS its own notebooks plus every cheaper tier's, so
# "batch" implies "smoke" and "full" implies everything — escalate deliberately (each step adds real
# Dataproc/Ray spend). See notebooks_for_tier().
TIER_SMOKE = "smoke"
TIER_BATCH = "batch"
TIER_FULL = "full"
_TIER_ORDER = (TIER_SMOKE, TIER_BATCH, TIER_FULL)


@dataclass(frozen=True)
class NotebookSpec:
    """One notebook's acceptance spec: which template runs it, its tier, and a timeout."""

    name: str  # file stem under notebooks/, e.g. "02_bigquery_native"
    template: str  # TEMPLATE_MAIN | TEMPLATE_SPARK
    tier: str  # TIER_SMOKE | TIER_BATCH | TIER_FULL
    timeout_s: int  # executionTimeout budget (also the local poll ceiling)


# The acceptance matrix. Tier rationale:
#   * smoke  — BQ-only / fully-local notebooks: cheap, fast, safe to run on every change.
#   * batch  — notebooks that submit a Dataproc Serverless batch (real, small spend). 01 is here
#              too: its interactive Spark Connect path needs sf-spark-connect (py312). 03 is here
#              because its combo run is python_runtime="spark" → main.run launches a Dataproc batch
#              (not BQ-only): it incurs the same Spark spend as 05/06 and outlasts the smoke budget.
#   * full   — 04_ray_on_vertex provisions a live Vertex Ray cluster (biggest cost + wall-clock).
# Routing: only NB01 runs on sf-spark-connect; everything else on sf-main (py311, matches the pin).
REGISTRY: dict[str, NotebookSpec] = {
    spec.name: spec
    for spec in (
        NotebookSpec("model_playground", TEMPLATE_MAIN, TIER_SMOKE, 900),
        NotebookSpec("02_bigquery_native", TEMPLATE_MAIN, TIER_SMOKE, 900),
        NotebookSpec("07_scale_review", TEMPLATE_MAIN, TIER_SMOKE, 900),
        NotebookSpec("01_spark_via_connect", TEMPLATE_SPARK, TIER_BATCH, 1800),
        NotebookSpec("03_combo_and_ensemble", TEMPLATE_MAIN, TIER_BATCH, 1800),
        NotebookSpec("05_spark_naive", TEMPLATE_MAIN, TIER_BATCH, 1800),
        # 06's "multi" method fans out one child explode-batch per model family, so its wall-clock
        # is the sum of the slowest cell across families — it ran ~34 min live, past the 1800s
        # sibling budget. 3600s gives headroom without masking a genuine hang (natives < 20 min).
        NotebookSpec("06_spark_multi", TEMPLATE_MAIN, TIER_BATCH, 3600),
        NotebookSpec("04_ray_on_vertex", TEMPLATE_MAIN, TIER_FULL, 5400),
    )
}


def notebooks_for_tier(tier: str) -> list[NotebookSpec]:
    """Every notebook at ``tier`` or a cheaper one (cumulative), in registry order.

    ``smoke`` → the 3 BQ/local notebooks; ``batch`` → those + the 4 Dataproc ones; ``full`` →
    all 8 (adds Ray). Raises :class:`EngineError` on an unknown tier so a CLI typo fails clearly.
    """
    if tier not in _TIER_ORDER:
        raise EngineError(f"unknown tier {tier!r}; choose one of {', '.join(_TIER_ORDER)}")
    max_rank = _TIER_ORDER.index(tier)
    included = set(_TIER_ORDER[: max_rank + 1])
    return [spec for spec in REGISTRY.values() if spec.tier in included]


@dataclass(frozen=True)
class AcceptanceResult:
    """Outcome of one headless notebook execution."""

    name: str
    job_id: str
    state: str  # terminal jobState, e.g. "JOB_STATE_SUCCEEDED"
    n_cell_errors: int  # error outputs found in the executed notebook (0 = clean)
    detail: str  # error message on a non-terminal-SUCCEEDED state or cell errors; else ""
    executed_uri: str  # gs:// path to the executed notebook (empty if never produced)

    @property
    def ok(self) -> bool:
        """True iff the job SUCCEEDED and no cell emitted an error output."""
        return self.state == "JOB_STATE_SUCCEEDED" and self.n_cell_errors == 0


# --- Vertex AI NotebookExecutionJob REST helpers ------------------------------------------------
# serviceAccount mode → v1 endpoint (v1beta1 is for executionUser + OAuth consent, which hangs in
# PENDING on v1). Token refreshed on every request so a long poll (NB04 > 60 min) never 401s.

_TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}


def _endpoint(region: str) -> str:
    """Regional aiplatform v1 endpoint the NotebookExecutionJob API lives behind."""
    return f"https://{region}-aiplatform.googleapis.com/v1"


def _token(credentials: object) -> str:
    """Mint a fresh access token from ADC credentials (refresh-on-read; see module docstring)."""
    from google.auth.transport.requests import Request

    credentials.refresh(Request())  # type: ignore[attr-defined]
    return credentials.token  # type: ignore[attr-defined]


def submit_job(
    *,
    project_id: str,
    region: str,
    notebook_path: Path,
    template_resource_name: str,
    service_account: str,
    gcs_output_uri: str,
    timeout_s: int,
    credentials: object,
    display_name: str,
) -> str:
    """Submit one notebook to run headless on a template; return the API-minted job id.

    The notebook is sent inline as base64 (``directNotebookSource`` — no pre-upload to GCS). We do
    NOT set ``notebookExecutionJobId`` (a custom id hangs in PENDING) and recover the id from the
    returned operation name. serviceAccount mode → the run executes as ``service_account``.
    """
    import requests

    nb_b64 = base64.b64encode(notebook_path.read_bytes()).decode("ascii")
    url = f"{_endpoint(region)}/projects/{project_id}/locations/{region}/notebookExecutionJobs"
    body: dict[str, Any] = {
        "displayName": display_name,
        "directNotebookSource": {"content": nb_b64},
        "notebookRuntimeTemplateResourceName": template_resource_name,
        "gcsOutputUri": gcs_output_uri,
        "serviceAccount": service_account,
        "executionTimeout": f"{timeout_s}s",
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {_token(credentials)}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=60,
    )
    resp.raise_for_status()
    op_name = resp.json().get("name", "")
    parts = op_name.split("/")
    try:
        return parts[parts.index("notebookExecutionJobs") + 1]
    except (ValueError, IndexError) as exc:
        raise EngineError(f"could not parse job id from operation name {op_name!r}") from exc


def poll_to_terminal(
    *,
    project_id: str,
    region: str,
    job_id: str,
    timeout_s: int,
    credentials: object,
    poll_interval_s: int = 30,
) -> tuple[str, str]:
    """Poll a job until a terminal ``jobState``; return ``(state, detail)``.

    ``detail`` carries the API error message on FAILED/CANCELLED (empty on SUCCEEDED). Refreshes the
    token every poll. If the local budget (``timeout_s`` + a grace margin) is spent before the job
    goes terminal, returns the last observed state with a timeout note rather than hanging forever.
    """
    import requests

    url = (
        f"{_endpoint(region)}/projects/{project_id}/locations/{region}"
        f"/notebookExecutionJobs/{job_id}"
    )
    # Local ceiling = the job's own executionTimeout plus a margin for provisioning + upload, so we
    # stop polling shortly after the server would have killed the job.
    deadline = time.monotonic() + timeout_s + 600
    state = "JOB_STATE_UNSPECIFIED"
    while time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {_token(credentials)}"}, timeout=60
        )
        resp.raise_for_status()
        job = resp.json()
        state = job.get("jobState", "")
        if state in _TERMINAL_STATES:
            if state == "JOB_STATE_SUCCEEDED":
                return state, ""
            detail = (
                job.get("status", {}).get("message")
                or (job.get("error") or {}).get("message")
                or state
            )
            return state, detail
        _log.info("notebook job %s: %s", job_id, state)
    return state, f"local poll timed out after {timeout_s + 600}s (last state {state})"


def download_executed(
    *, gcs_output_uri: str, job_id: str, project_id: str
) -> tuple[str, bytes | None]:
    """Fetch the executed notebook bytes from ``{gcsOutputUri}/{job_id}/content.ipynb``.

    Returns ``(uri, bytes)``; ``bytes`` is None if the object doesn't exist (e.g. the job failed
    before writing output), so the caller can still record the terminal state.
    """
    from google.cloud import storage

    uri = f"{gcs_output_uri}/{job_id}/content.ipynb"
    without_scheme = uri.removeprefix("gs://")
    bucket_name, _, blob_path = without_scheme.partition("/")
    client = storage.Client(project=project_id)
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        return uri, None
    return uri, blob.download_as_bytes()


def assert_no_cell_errors(notebook_bytes: bytes) -> int:
    """Count cell error outputs in an executed notebook (0 = clean run).

    Scans every code cell's outputs for ``output_type == 'error'`` — the shape nbclient records when
    a cell raises. A clean acceptance run has zero.
    """
    import nbformat

    nb = nbformat.reads(notebook_bytes.decode("utf-8"), as_version=4)
    errors = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors += 1
    return errors


def run_acceptance(
    *,
    specs: list[NotebookSpec],
    project_id: str,
    region: str,
    notebooks_dir: Path,
    template_ids: dict[str, str],
    service_account: str,
    gcs_output_uri: str,
    credentials: object | None = None,
    run_label: str,
) -> list[AcceptanceResult]:
    """Run each notebook headless on its template and collect per-notebook results.

    Submits, polls to terminal, downloads the executed notebook, and scans it for cell errors. One
    notebook's failure never stops the others — every spec produces an :class:`AcceptanceResult`.
    ``template_ids`` maps ``TEMPLATE_MAIN``/``TEMPLATE_SPARK`` to the runtime-template resource
    names Terraform outputs. ``run_label`` disambiguates concurrent runs' output paths + names.
    """
    if credentials is None:
        import google.auth

        credentials, _ = google.auth.default()

    results: list[AcceptanceResult] = []
    for spec in specs:
        notebook_path = notebooks_dir / f"{spec.name}.ipynb"
        if not notebook_path.exists():
            results.append(
                AcceptanceResult(spec.name, "", "JOB_STATE_FAILED", 0, "notebook file missing", "")
            )
            continue
        template_resource = template_ids.get(spec.template)
        if not template_resource:
            results.append(
                AcceptanceResult(
                    spec.name, "", "JOB_STATE_FAILED", 0, f"no template id for {spec.template}", ""
                )
            )
            continue

        # Per-notebook output prefix keeps executed artifacts from colliding across notebooks/runs.
        out_uri = f"{gcs_output_uri.rstrip('/')}/acceptance/{run_label}/{spec.name}"
        _log.info("submitting %s on template %s", spec.name, spec.template)
        job_id = submit_job(
            project_id=project_id,
            region=region,
            notebook_path=notebook_path,
            template_resource_name=template_resource,
            service_account=service_account,
            gcs_output_uri=out_uri,
            timeout_s=spec.timeout_s,
            credentials=credentials,
            display_name=f"sf-accept-{spec.name}-{run_label}",
        )
        state, detail = poll_to_terminal(
            project_id=project_id,
            region=region,
            job_id=job_id,
            timeout_s=spec.timeout_s,
            credentials=credentials,
        )

        n_errors = 0
        executed_uri = ""
        if state == "JOB_STATE_SUCCEEDED":
            executed_uri, nb_bytes = download_executed(
                gcs_output_uri=out_uri, job_id=job_id, project_id=project_id
            )
            if nb_bytes is None:
                detail = "job SUCCEEDED but executed notebook not found in GCS"
            else:
                n_errors = assert_no_cell_errors(nb_bytes)
                if n_errors:
                    detail = f"{n_errors} cell error output(s) in executed notebook"
        results.append(
            AcceptanceResult(spec.name, job_id, state, n_errors, detail, executed_uri)
        )
        _log.info("%s → %s (%d cell errors)", spec.name, state, n_errors)
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scale_forecasting.notebook_acceptance",
        description="Run the notebooks headless on their Colab Enterprise templates and assert "
        "no cell errored. Tiers escalate cost: smoke (BQ/local) → batch (+Dataproc) → full (+Ray).",
    )
    parser.add_argument("--tier", default=TIER_SMOKE, choices=_TIER_ORDER, help="acceptance tier")
    parser.add_argument("--project", required=True, help="GCP project id (explicit, never ambient)")
    parser.add_argument("--region", default="us-central1", help="region the templates live in")
    parser.add_argument(
        "--main-template", required=True, help="sf-main runtime template resource name"
    )
    parser.add_argument(
        "--spark-template", required=True, help="sf-spark-connect runtime template resource name"
    )
    parser.add_argument(
        "--service-account", required=True, help="runner SA the notebooks execute as"
    )
    parser.add_argument("--gcs-output", required=True, help="gs:// prefix for executed notebooks")
    parser.add_argument(
        "--notebooks-dir",
        default=None,
        help="path to notebooks/ (default: repo notebooks/ relative to this file)",
    )
    parser.add_argument(
        "--run-label", default="cli", help="label disambiguating this run's outputs"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the selected tier, print a per-notebook table; return non-zero on any failure."""
    args = _parse_args(argv)
    notebooks_dir = (
        Path(args.notebooks_dir)
        if args.notebooks_dir
        else Path(__file__).resolve().parents[2] / "notebooks"
    )
    specs = notebooks_for_tier(args.tier)
    print(f"Acceptance tier={args.tier}: {len(specs)} notebook(s) on project {args.project}")
    results = run_acceptance(
        specs=specs,
        project_id=args.project,
        region=args.region,
        notebooks_dir=notebooks_dir,
        template_ids={TEMPLATE_MAIN: args.main_template, TEMPLATE_SPARK: args.spark_template},
        service_account=args.service_account,
        gcs_output_uri=args.gcs_output,
        run_label=args.run_label,
    )
    print()
    print(f"{'notebook':<24} {'state':<22} {'cell_err':>8}  detail")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<24} {r.state:<22} {r.n_cell_errors:>8}  {r.detail}")
    failed = [r for r in results if not r.ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
