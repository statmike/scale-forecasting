"""The launch-point-lean invariant: a submitter can import the run driver without the model stack.

A *launch point* — a Composer worker, or an SDK/CLI process that only submits runs — carries the
submit-side deps (GCP clients, pandas) but **not** the model stack (statsmodels, scipy, xgboost,
torch, …), because model code ships per-job to Dataproc/Ray/BigQuery and runs there, never on the
launch point. So importing the driver path the worker actually uses —
``scale_forecasting.airflow_tasks`` (the DAG task callables) → ``main`` → ``router`` →
``models`` — must resolve every model's *metadata* (name/family/runtime, for routing and
``plan_dag``) without importing any model's *implementation*. Each model file therefore keeps its
heavy imports inside ``fit``/``predict``; this test is the tripwire that keeps them there. Regressed
once live: a top-level ``from statsmodels… import …`` in a model file crashed every family task on a
Composer worker with ``ModuleNotFoundError: statsmodels``.

Checked in a clean subprocess: the rest of the suite fits real models, so this process has already
imported the model stack — only a fresh interpreter can prove the launch-point import stays lean.
"""

from __future__ import annotations

import subprocess
import sys

# Top-level distribution names no launch-point import may pull. These are the model
# implementations' deps — present in this all-extras test env, so a regression (an import hoisted
# back to a model's module top) lands them in ``sys.modules`` and trips this test.
_MODEL_STACK = ("statsmodels", "scipy", "xgboost", "lightgbm", "neuralprophet", "prophet", "torch")


def test_worker_import_path_does_not_pull_the_model_stack() -> None:
    script = (
        "import sys\n"
        "import scale_forecasting.airflow_tasks  # the Composer worker's entry import\n"
        "import scale_forecasting.main  # run_family: from . import main -> router -> models\n"
        f"stack = {_MODEL_STACK!r}\n"
        "pulled = sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] in stack})\n"
        "assert not pulled, f'launch-point import pulled the model stack: {pulled}'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
