"""Live headless notebook acceptance (``@gcp``, tiered).

A thin pytest wrapper over :mod:`scale_forecasting.notebook_acceptance`: it runs each notebook in
the selected tier headless on its Colab Enterprise template (via the Vertex AI NotebookExecutionJob
API, serviceAccount mode) and asserts the job SUCCEEDED with zero cell errors. This is the
repeatable proof that every notebook runs green on the right template — and the regression guard
when a notebook changes. The templates carry the ``SF_*`` run identity in their env, so the kernel
resolves ``Settings`` exactly as a human who just opens the template would (G1).

**Tiers escalate cost** (see the harness registry), each gated so money-spending runs are opt-in —
the same shape as ``@gpu``/``@raylive`` in ``tests/conftest.py``:

* **smoke** (default, ``@gcp``) — the 3 BQ-only / fully-local notebooks. Cheap; runs whenever
  ``SF_PROJECT_ID`` + ADC are present.
* **batch** (``SF_ENABLE_NB_BATCH``) — adds the 4 notebooks that submit a Dataproc Serverless batch
  (real, small spend), including NB01 on sf-spark-connect and NB03's Spark ∥ BQ combo.
* **full** (``SF_ENABLE_NB_FULL``) — adds ``04_ray_on_vertex``, which provisions a live Vertex Ray
  cluster (biggest cost + wall-clock).

Infra identity (template ids, runner SA, code bucket, project/region) is read from
``terraform output -json`` in ``terraform/main`` — the values a local operator wires from too. Run::

    uv run --active pytest -m gcp tests/integration/test_notebook_acceptance.py            # smoke
    SF_ENABLE_NB_BATCH=1 uv run --active pytest -m gcp tests/integration/test_notebook_acceptance.py
    SF_ENABLE_NB_FULL=1  uv run --active pytest -m gcp tests/integration/test_notebook_acceptance.py
"""

from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

from scale_forecasting import notebook_acceptance as na

pytestmark = pytest.mark.gcp

# Repo root: this file is tests/integration/…, so parents[2] is the repo; notebooks/ +
# terraform/main hang off it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TERRAFORM_DIR = _REPO_ROOT / "terraform" / "main"
_NOTEBOOKS_DIR = _REPO_ROOT / "notebooks"

# Opt-in switches for the money-spending tiers (mirrors SF_ENABLE_GPU / SF_ENABLE_RAY).
_ENABLE_BATCH = "SF_ENABLE_NB_BATCH"
_ENABLE_FULL = "SF_ENABLE_NB_FULL"


@lru_cache(maxsize=1)
def _terraform_outputs() -> dict[str, str]:
    """Read ``terraform output -json`` from terraform/main → a flat ``{key: value}`` map.

    Cached so the (slow) shell-out runs once per session. Skips the test cleanly if Terraform isn't
    available or the stage isn't applied — this is a live acceptance test, not an offline gate.
    """
    try:
        proc = subprocess.run(
            ["terraform", f"-chdir={_TERRAFORM_DIR}", "output", "-json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"terraform outputs unavailable ({exc}); apply terraform/main first")
    raw = json.loads(proc.stdout)
    return {key: entry["value"] for key, entry in raw.items() if entry.get("value") is not None}


def _selected_tier() -> str:
    """The highest tier whose opt-in switch is set (full > batch > smoke-default)."""
    if os.environ.get(_ENABLE_FULL):
        return na.TIER_FULL
    if os.environ.get(_ENABLE_BATCH):
        return na.TIER_BATCH
    return na.TIER_SMOKE


def _specs() -> list[na.NotebookSpec]:
    return na.notebooks_for_tier(_selected_tier())


@pytest.fixture(scope="module")
def acceptance_results() -> dict[str, na.AcceptanceResult]:
    """Run the selected tier once and key the results by notebook name.

    Module-scoped so all notebooks in the tier submit + run in a single pass (each execution is
    already independent on its own Colab runtime), then each parametrized test just asserts its own
    notebook's result — one clean assertion per notebook in the pytest report.
    """
    outputs = _terraform_outputs()
    main_template = outputs.get("colab_main_runtime_template_id")
    spark_template = outputs.get("colab_spark_runtime_template_id")
    if not main_template or not spark_template:
        pytest.skip(
            "colab runtime templates not in terraform outputs "
            "(set create_colab_templates = true and apply)"
        )

    results = na.run_acceptance(
        specs=_specs(),
        project_id=outputs["project_id"],
        region=outputs.get("region") or "us-central1",
        notebooks_dir=_NOTEBOOKS_DIR,
        template_ids={na.TEMPLATE_MAIN: main_template, na.TEMPLATE_SPARK: spark_template},
        service_account=outputs["runner_sa"],
        gcs_output_uri=f"gs://{outputs['code_bucket']}",
        run_label="pytest",
    )
    return {r.name: r for r in results}


@pytest.mark.slow
@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_notebook_runs_clean(
    spec: na.NotebookSpec, acceptance_results: dict[str, na.AcceptanceResult]
) -> None:
    """The notebook ran to SUCCEEDED on its template with zero cell errors.

    A transient GCP capacity stockout (runtime VM or an in-notebook Dataproc/Ray pool) is skipped,
    not failed — it's infra availability, not a defect in the notebook or the product. Everything
    else must be a clean SUCCEEDED with zero cell errors.
    """
    result = acceptance_results[spec.name]
    if result.state == na.JOB_STATE_CAPACITY_UNAVAILABLE:
        pytest.skip(f"{spec.name}: GCP capacity unavailable (transient) — {result.detail}")
    assert result.state == "JOB_STATE_SUCCEEDED", (
        f"{spec.name} on {spec.template}: {result.state} — {result.detail}"
    )
    assert result.n_cell_errors == 0, f"{spec.name}: {result.detail} ({result.executed_uri})"
