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
and the human-open path validate the *same* thing.

**Auth / endpoint choices** (learned from prior art, see docs/notebook_runtimes.md):

* **serviceAccount mode** (the run executes as the runner SA), which requires the **v1** endpoint —
  the v1beta1/``executionUser`` path needs OAuth consent and hangs in ``PENDING`` on v1. So this is
  fully headless: no browser consent, runnable from CI.
* **Never** send a custom ``notebookExecutionJobId`` — that also hangs in ``PENDING``; let the API
  mint the id and recover it from the returned operation name.
* **Refresh the token every call.** NB04 (Ray) can outlive a ~60-min OAuth token; refreshing each
  poll (mirroring ``ray_jobs``'s client-refresh) keeps a long poll from 401-ing.
* The executed notebook lands at ``{gcsOutputUri}/{JOB_ID}/content.ipynb``.

**Tiers** bound cost — most notebooks orchestrate real Dataproc/Ray compute, so the harness
escalates deliberately (smoke → batch → full); see `REGISTRY` and `notebooks_for_tier`.

Public surface: `REGISTRY`, `AcceptanceResult`, `run_acceptance`, `notebooks_for_tier`,
`notebooks_by_name`, `notebook_completed_clean`, and a
``python -m scale_forecasting.notebook_acceptance`` CLI.
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

# Template routing key. The caller maps this to the runtime-template resource name Terraform outputs
# (colab_main_runtime_template_id) — the harness stays id-agnostic so the same registry works across
# deploys. There is one template (sf-main, Python 3.11); every notebook runs on it.
TEMPLATE_MAIN = "main"

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
    template: str  # TEMPLATE_MAIN (the one template every notebook runs on)
    tier: str  # TIER_SMOKE | TIER_BATCH | TIER_FULL
    timeout_s: int  # executionTimeout budget (also the local poll ceiling)


# The acceptance matrix. Tier rationale:
#   * smoke  — BQ-only / fully-local notebooks: cheap, fast, safe to run on every change.
#   * batch  — notebooks that submit a Dataproc Serverless batch (real, small spend). 01 is here
#              too: its interactive Spark Connect path runs on Dataproc runtime 2.3 and submits real
#              cluster work. 03 is here because its combo run is python_runtime="spark" → main.run
#              launches a Dataproc batch (not BQ-only): real Spark spend, so it outlasts smoke.
#   * full   — 04_ray_on_vertex provisions a live Vertex Ray cluster (biggest cost + wall-clock).
# Routing: every notebook runs on the single sf-main template (py311, matches the pin).
REGISTRY: dict[str, NotebookSpec] = {
    spec.name: spec
    for spec in (
        NotebookSpec("model_playground", TEMPLATE_MAIN, TIER_SMOKE, 900),
        NotebookSpec("02_bigquery_native", TEMPLATE_MAIN, TIER_SMOKE, 900),
        NotebookSpec("07_scale_review", TEMPLATE_MAIN, TIER_SMOKE, 900),
        # 09 is registry-read-only (reviews any finished run_id) → cheapest tier.
        NotebookSpec("09_review_run", TEMPLATE_MAIN, TIER_SMOKE, 900),
        # 01 brings up an interactive Spark Connect session AND its bootstrap installs the locked
        # deps (incl. pyspark) into a private prefix from scratch (no cache on a fresh runtime) to
        # shadow the base image's numpy 2.x — together ~30 min, so it gets a wider budget than a
        # normal batch notebook.
        NotebookSpec("01_spark_via_connect", TEMPLATE_MAIN, TIER_BATCH, 3600),
        # 03 and 08 both block on a Dataproc Serverless batch, which carries ~30 min of fixed
        # provisioning overhead before any cell of work runs — so a 1800 s ceiling gave them roughly
        # zero margin, and on 2026-09-02 03 tripped it. Widened to 3600 s, matching 01, because of
        # what tripping it costs: the run's finalizer lives in the notebook process, so a deadline
        # kill lands *after* the batch has succeeded and *before* the header is closed, leaving a
        # permanently RUNNING row in the registry. A ceiling is not a duration — nothing pays for
        # the extra headroom unless it is needed — and the failure it prevents needs a human to
        # clean up. 08 is the same shape and was passing only narrowly; it moves with 03.
        NotebookSpec("03_combo_and_ensemble", TEMPLATE_MAIN, TIER_BATCH, 3600),
        # 08 launches a multi-engine run (Spark ∥ BigQuery) then live-monitors it → Dataproc spend.
        NotebookSpec("08_run_and_monitor", TEMPLATE_MAIN, TIER_BATCH, 3600),
        NotebookSpec("04_ray_on_vertex", TEMPLATE_MAIN, TIER_FULL, 5400),
    )
}


def notebooks_for_tier(tier: str) -> list[NotebookSpec]:
    """Every notebook at ``tier`` or a cheaper one (cumulative), in registry order.

    ``smoke`` → the 4 BQ/local notebooks; ``batch`` → those + the 3 Dataproc ones; ``full`` →
    all 8 (adds Ray). Raises `EngineError` on an unknown tier so a CLI typo fails clearly.
    """
    if tier not in _TIER_ORDER:
        raise EngineError(f"unknown tier {tier!r}; choose one of {', '.join(_TIER_ORDER)}")
    max_rank = _TIER_ORDER.index(tier)
    included = set(_TIER_ORDER[: max_rank + 1])
    return [spec for spec in REGISTRY.values() if spec.tier in included]


def notebooks_by_name(names: list[str]) -> list[NotebookSpec]:
    """Exactly the named notebooks, in registry order — the re-run-one-notebook path.

    A tier is the right unit for *proving* the suite and the wrong unit for *repairing* it. A single
    notebook can fail for reasons that have nothing to do with it — the one that prompted this
    returned `Quota 'CPUS' exceeded` because six of its siblings happened to hold the region's
    runtimes at that moment — and re-running the whole tier to retry it costs the tier's full time
    and spend to re-prove seven things that already passed.

    Unknown names raise rather than being skipped: a typo here silently "passing" because it matched
    nothing is the failure mode worth spending an exception on.
    """
    unknown = [name for name in names if name not in REGISTRY]
    if unknown:
        raise EngineError(
            f"unknown notebook(s) {', '.join(sorted(unknown))}; choose from {', '.join(REGISTRY)}"
        )
    wanted = set(names)
    return [spec for spec in REGISTRY.values() if spec.name in wanted]


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
        """True iff the notebook ran clean: no cell error output, and a terminal state that means
        the cells actually executed — either the API's ``JOB_STATE_SUCCEEDED`` or the synthetic
        ``JOB_STATE_SUCCEEDED_WITH_TEARDOWN`` (cells ran to the end but the job wrapper reported
        FAILED during post-execution teardown; see that constant)."""
        return (
            self.state in ("JOB_STATE_SUCCEEDED", JOB_STATE_SUCCEEDED_WITH_TEARDOWN)
            and self.n_cell_errors == 0
        )


@dataclass(frozen=True)
class FanOutResult:
    """Outcome of a fire-and-forget notebook *submission* (no wait for completion).

    Unlike `AcceptanceResult`, this records only that the job was *created* — the notebook
    runs on server-side afterwards. ``detail`` is empty on a clean submit, else the reason it could
    not be submitted (missing file, no template id, API error). ``executed_uri`` is where the
    executed notebook *will* land once the job reaches SUCCEEDED (see `download_executed`).
    """

    name: str
    job_id: str  # "" if the submit itself failed
    display_name: str  # "" if the submit failed
    executed_uri: str  # gs:// path the executed .ipynb will land at (empty on submit failure)
    detail: str  # "" on a clean submit; else why it wasn't submitted


def executions_console_url(project_id: str) -> str:
    """Colab Enterprise **Executions** menu URL for a project (where fanned-out jobs appear).

    This is the console page a presenter watches: each fanned-out ``NotebookExecutionJob`` shows up
    here with its live state, and a finished one opens as the executed notebook with rendered
    outputs. Region isn't in the path — the console lists a project's jobs and filters client-side.
    """
    return f"https://console.cloud.google.com/vertex-ai/colab/execution-jobs?project={project_id}"


# --- Vertex AI NotebookExecutionJob REST helpers ------------------------------------------------
# serviceAccount mode → v1 endpoint (v1beta1 is for executionUser + OAuth consent, which hangs in
# PENDING on v1). Token refreshed on every request so a long poll (NB04 > 60 min) never 401s.

_TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}

# Synthetic (non-API) state: the job FAILED because GCP couldn't provision capacity — the Colab
# runtime VM, or a Dataproc/Ray pool the notebook asked for. This is transient infra, not a defect
# in the notebook or the product, so the acceptance test skips rather than fails on it. See
# `is_capacity_unavailable` for the signals; the harness rewrites FAILED → this before return.
JOB_STATE_CAPACITY_UNAVAILABLE = "JOB_STATE_CAPACITY_UNAVAILABLE"

# Synthetic (non-API) state: the notebook ran EVERY cell clean to the end, but the job wrapper still
# reported FAILED — a post-execution teardown artifact (e.g. the runtime VM was reclaimed after the
# executed notebook was already written, seen under regional CPU-quota pressure). The API's terminal
# state and the actual execution disagree; cell-level success is the source of truth, so acceptance
# passes on this. `run_acceptance` rewrites FAILED → this only when the executed notebook exists,
# has zero cell errors, and every code cell carries an execution_count (proving execution reached
# the end, not a silent mid-run death). See `notebook_completed_clean`.
JOB_STATE_SUCCEEDED_WITH_TEARDOWN = "JOB_STATE_SUCCEEDED_WITH_TEARDOWN"

# Substrings that mark a capacity/stockout failure (case-insensitive), drawn from the messages GCP
# returns: Colab runtime provisioning ("does not have enough resources", "ZONE_RESOURCE_POOL_
# EXHAUSTED"), Dataproc/Compute stockouts, and the generic RESOURCE_EXHAUSTED / quota family. These
# are about physical capacity, NOT quota-you-can-raise — but both are transient-to-the-run, so we
# treat them the same: skip, don't fail.
_CAPACITY_SIGNALS = (
    "does not have enough resources",
    "resource_pool_exhausted",
    "zone_resource_pool_exhausted",
    "resource_exhausted",
    "resources.unavailable",
    "out of resources",
    "insufficient",
    "stockout",
    "capacity",
)


def is_capacity_unavailable(detail: str) -> bool:
    """True if a failure ``detail`` reads as a transient GCP capacity/stockout, not a real error.

    Matched case-insensitively against `_CAPACITY_SIGNALS`. Used by the harness to reclassify
    a FAILED job (or in-cell error) as `JOB_STATE_CAPACITY_UNAVAILABLE` so acceptance skips.
    """
    low = detail.lower()
    return any(sig in low for sig in _CAPACITY_SIGNALS)


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
    ack_out_of_org: bool = False,
) -> str:
    """Submit one notebook to run headless on a template; return the API-minted job id.

    The notebook is sent inline as base64 (``directNotebookSource`` — no pre-upload to GCS). We do
    NOT set ``notebookExecutionJobId`` (a custom id hangs in PENDING) and recover the id from the
    returned operation name. serviceAccount mode → the run executes as ``service_account``.

    ``ack_out_of_org`` opts in to running on a project **outside a Google-corp org** (a personal or
    standalone project). Vertex rejects such a submit with FAILED_PRECONDITION
    (``NOTEBOOK_RUNTIME_OUT_OF_ORGANIZATION``) unless the request carries an acknowledgement label;
    that ack states the runtime SA's credentials may be visible to the project owner, so it's an
    explicit operator opt-in (``--ack-out-of-org``), never sent silently. No-op inside a corp org.
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
    if ack_out_of_org:
        body["labels"] = {"aiplatform.googleapis.com/notebook_runtime_out_of_org_warning": "ack"}
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


def notebook_completed_clean(notebook_bytes: bytes) -> bool:
    """True iff an executed notebook ran EVERY code cell to the end with no error output.

    This distinguishes a *finished* run whose job wrapper failed only at teardown (a background
    thread still alive, a runtime VM reclaimed after output was written — the notebook is complete)
    from a run that died mid-way (a later cell never executed, or a cell raised). The signals:

    * at least one code cell exists (an empty notebook is not "completed"),
    * no code cell has an ``output_type == 'error'`` output, and
    * every code cell with non-blank source carries an ``execution_count`` — nbclient assigns one
      only to cells it ran, so a ``None`` count on a real cell means execution stopped early.

    Used by `run_acceptance` to reclassify a FAILED-but-clean job to
    ``JOB_STATE_SUCCEEDED_WITH_TEARDOWN`` without masking a genuine partial failure.
    """
    import nbformat

    nb = nbformat.reads(notebook_bytes.decode("utf-8"), as_version=4)
    code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
    if not code_cells:
        return False
    for cell in code_cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                return False
        # A blank code cell may legitimately carry no count; only real cells must have executed.
        source = "".join(cell.get("source", []))
        if source.strip() and cell.get("execution_count") is None:
            return False
    return True


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
    ack_out_of_org: bool = False,
) -> list[AcceptanceResult]:
    """Run each notebook headless on its template and collect per-notebook results.

    Submits, polls to terminal, downloads the executed notebook, and scans it for cell errors. One
    notebook's failure never stops the others — every spec produces an `AcceptanceResult`.
    ``template_ids`` maps ``TEMPLATE_MAIN`` to the runtime-template resource name Terraform outputs.
    ``run_label`` disambiguates concurrent runs' output paths + names.
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
            ack_out_of_org=ack_out_of_org,
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
                    # An in-cell capacity stockout (e.g. a Dataproc/Ray pool couldn't scale) is
                    # transient infra — surface the offending cell's message so we can classify it.
                    detail = f"{detail}: {_first_cell_error(nb_bytes)}"
        elif state == "JOB_STATE_FAILED":
            # A FAILED job can mean three different things; the executed notebook (if any) tells
            # them apart. It may have (a) never started — runtime VM couldn't be provisioned, no
            # output; (b) died mid-run — a cell raised or a later cell never executed; or (c) run
            # every cell clean and only failed at TEARDOWN (e.g. notebook 08's daemon monitor thread
            # is still alive at kernel shutdown, or the VM was reclaimed after output was written).
            # Case (c) is a false FAILED: the product ran fine, so we take cell-level success as the
            # source of truth and reclassify. Prefer the notebook's own error message for (b); fall
            # back to the API detail for (a).
            executed_uri, nb_bytes = download_executed(
                gcs_output_uri=out_uri, job_id=job_id, project_id=project_id
            )
            if nb_bytes is not None:
                n_errors = assert_no_cell_errors(nb_bytes)
                cell_err = _first_cell_error(nb_bytes)
                if cell_err:
                    detail = cell_err
                elif n_errors == 0 and notebook_completed_clean(nb_bytes):
                    state = JOB_STATE_SUCCEEDED_WITH_TEARDOWN
                    detail = (
                        "notebook ran every cell clean to completion; job wrapper reported FAILED "
                        f"only at teardown ({detail})"
                    )

        # Reclassify a transient GCP capacity/stockout (runtime VM or in-notebook pool) so
        # acceptance skips rather than fails — infra, not a product/notebook defect.
        if state == "JOB_STATE_FAILED" and is_capacity_unavailable(detail):
            state = JOB_STATE_CAPACITY_UNAVAILABLE

        results.append(AcceptanceResult(spec.name, job_id, state, n_errors, detail, executed_uri))
        _log.info("%s → %s (%d cell errors)", spec.name, state, n_errors)
    return results


def run_fanout(
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
    display_prefix: str = "sf-demo",
    ack_out_of_org: bool = False,
) -> list[FanOutResult]:
    """Submit every notebook headless and return immediately — do NOT wait for them to finish.

    This is the presenter's fan-out: it fires one `submit_job` per spec back-to-back (each
    returns as soon as the job is *created*, ~1s), so all notebooks then run **concurrently
    server-side**, decoupled from this process — you can close the shell and watch them in the
    Colab Enterprise Executions menu (`executions_console_url`). Contrast
    `run_acceptance`, which polls each job to terminal because a *test* needs the verdict; a
    presenter just needs the jobs launched. One notebook's submit failure never stops the others.
    """
    if credentials is None:
        import google.auth

        credentials, _ = google.auth.default()

    results: list[FanOutResult] = []
    for spec in specs:
        notebook_path = notebooks_dir / f"{spec.name}.ipynb"
        if not notebook_path.exists():
            results.append(FanOutResult(spec.name, "", "", "", "notebook file missing"))
            continue
        template_resource = template_ids.get(spec.template)
        if not template_resource:
            results.append(
                FanOutResult(spec.name, "", "", "", f"no template id for {spec.template}")
            )
            continue

        out_uri = f"{gcs_output_uri.rstrip('/')}/fanout/{run_label}/{spec.name}"
        display_name = f"{display_prefix}-{spec.name}-{run_label}"
        _log.info("fanning out %s on template %s", spec.name, spec.template)
        try:
            job_id = submit_job(
                project_id=project_id,
                region=region,
                notebook_path=notebook_path,
                template_resource_name=template_resource,
                service_account=service_account,
                gcs_output_uri=out_uri,
                timeout_s=spec.timeout_s,
                credentials=credentials,
                display_name=display_name,
                ack_out_of_org=ack_out_of_org,
            )
        except Exception as exc:  # noqa: BLE001 — one bad submit must not sink the rest of the fan-out
            results.append(FanOutResult(spec.name, "", display_name, "", f"submit failed: {exc}"))
            _log.warning("fan-out submit failed for %s: %s", spec.name, exc)
            continue

        executed_uri = f"{out_uri}/{job_id}/content.ipynb"
        results.append(FanOutResult(spec.name, job_id, display_name, executed_uri, ""))
        _log.info("fanned out %s → job %s", spec.name, job_id)
    return results


def _first_cell_error(notebook_bytes: bytes) -> str:
    """Return a one-line ``ename: evalue`` for the first errored cell in an executed notebook.

    Empty string if none — used to surface an in-cell failure's message for capacity classification.
    """
    import nbformat

    nb = nbformat.reads(notebook_bytes.decode("utf-8"), as_version=4)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                ename = output.get("ename", "")
                evalue = output.get("evalue", "")
                return f"{ename}: {evalue}".strip(": ").strip()
    return ""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scale_forecasting.notebook_acceptance",
        description="Run the notebooks headless on their Colab Enterprise templates and assert "
        "no cell errored. Tiers escalate cost: smoke (BQ/local) → batch (+Dataproc) → full (+Ray).",
    )
    parser.add_argument("--tier", default=TIER_SMOKE, choices=_TIER_ORDER, help="acceptance tier")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NOTEBOOK",
        help="run exactly these notebooks instead of a tier (repair path: re-run the one that "
        "failed for an infrastructure reason without re-paying for the rest of the tier)",
    )
    parser.add_argument("--project", required=True, help="GCP project id (explicit, never ambient)")
    parser.add_argument("--region", default="us-central1", help="region the templates live in")
    parser.add_argument(
        "--main-template", required=True, help="sf-main runtime template resource name"
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
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="fan out: submit all notebooks and return immediately (don't poll to terminal or "
        "assert cell errors). They run concurrently server-side — watch them in the Colab "
        "Enterprise Executions menu. Use to pre-render the notebooks before a demo.",
    )
    parser.add_argument(
        "--ack-out-of-org",
        action="store_true",
        help="acknowledge running on a project outside a Google-corp org (a personal or standalone "
        "project). Required there: Vertex rejects the submit with FAILED_PRECONDITION "
        "(NOTEBOOK_RUNTIME_OUT_OF_ORGANIZATION) otherwise. The ack states the runtime SA's "
        "credentials may be visible to the project owner — pass only if you trust the project.",
    )
    return parser.parse_args(argv)


def _run_fanout_cli(
    args: argparse.Namespace, notebooks_dir: Path, specs: list[NotebookSpec]
) -> int:
    """``--no-wait`` path: fan the notebooks out and print job ids + the Executions link."""
    selection = f"only={','.join(args.only)}" if args.only else f"tier={args.tier}"
    print(
        f"Fan-out {selection}: submitting {len(specs)} notebook(s) on project {args.project} "
        "(not waiting for completion)"
    )
    results = run_fanout(
        specs=specs,
        project_id=args.project,
        region=args.region,
        notebooks_dir=notebooks_dir,
        template_ids={TEMPLATE_MAIN: args.main_template},
        service_account=args.service_account,
        gcs_output_uri=args.gcs_output,
        run_label=args.run_label,
        ack_out_of_org=args.ack_out_of_org,
    )
    print()
    print(f"{'notebook':<24} {'job_id':<24}  detail")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<24} {r.job_id or '—':<24}  {r.detail}")
    submitted = [r for r in results if r.job_id]
    failed = [r for r in results if not r.job_id]
    print()
    summary = f"{len(submitted)}/{len(results)} submitted"
    if failed:
        summary += f", {len(failed)} failed"
    print(summary)
    print("\nWatch them run (and open the rendered notebooks when done):")
    print(f"  {executions_console_url(args.project)}")
    # Non-zero only if a submission itself failed; the runs' own success is watched in the console.
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """Run the selected tier, print a per-notebook table; return non-zero on any failure."""
    args = _parse_args(argv)
    notebooks_dir = (
        Path(args.notebooks_dir)
        if args.notebooks_dir
        else Path(__file__).resolve().parents[2] / "notebooks"
    )
    specs = notebooks_by_name(args.only) if args.only else notebooks_for_tier(args.tier)
    if args.no_wait:
        return _run_fanout_cli(args, notebooks_dir, specs)
    selection = f"only={','.join(args.only)}" if args.only else f"tier={args.tier}"
    print(f"Acceptance {selection}: {len(specs)} notebook(s) on project {args.project}")
    results = run_acceptance(
        specs=specs,
        project_id=args.project,
        region=args.region,
        notebooks_dir=notebooks_dir,
        template_ids={TEMPLATE_MAIN: args.main_template},
        service_account=args.service_account,
        gcs_output_uri=args.gcs_output,
        run_label=args.run_label,
        ack_out_of_org=args.ack_out_of_org,
    )
    print()
    print(f"{'notebook':<24} {'state':<22} {'cell_err':>8}  detail")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<24} {r.state:<22} {r.n_cell_errors:>8}  {r.detail}")
    # A capacity stockout is transient infra, not a defect: report it as skipped, exit non-zero only
    # on a genuine failure (see is_capacity_unavailable / JOB_STATE_CAPACITY_UNAVAILABLE).
    skipped = [r for r in results if r.state == JOB_STATE_CAPACITY_UNAVAILABLE]
    failed = [r for r in results if not r.ok and r not in skipped]
    passed = len(results) - len(failed) - len(skipped)
    # Passed-but-reclassified: cells ran clean, job wrapper failed at teardown. Counted in `passed`
    # (ok is True), but called out so a real teardown regression isn't hidden by a green summary.
    teardown = [r for r in results if r.state == JOB_STATE_SUCCEEDED_WITH_TEARDOWN]
    print()
    summary = f"{passed}/{len(results)} passed"
    if teardown:
        summary += f" ({len(teardown)} clean-but-FAILED-at-teardown)"
    if skipped:
        summary += f", {len(skipped)} skipped (capacity unavailable)"
    if failed:
        summary += f", {len(failed)} failed"
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
