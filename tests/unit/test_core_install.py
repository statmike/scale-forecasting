"""The core-install contract, proven in a fresh interpreter with every optional package blocked.

``pip install scale-forecasting`` — no extra — must be a working forecasting toolkit (every model's
metadata, every metric, the playground's fit + backtest) and must fail *helpfully*, not with a
``ModuleNotFoundError`` from six frames down, the moment something reaches Google Cloud or
matplotlib. The test environment has every extra installed, so the subprocess forces the
core-only world with a meta-path finder that refuses each optional top-level package: ``import
google`` raises ``ModuleNotFoundError`` exactly as it would when the distribution is absent, and
``importlib.util.find_spec`` reports it missing, which is what `errors.require_extra` probes. CI
runs the same assertions against a genuinely bare ``uv pip install .`` venv (the ``core-install``
job); this is the offline mirror every developer gets from ``pytest``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "mixed_demo.json"

# Top-level packages that only an extra installs. Blocking the parent blocks every submodule.
_OPTIONAL_TOP_LEVEL = (
    "google",
    "google_cloud_pipeline_components",
    "db_dtypes",
    "matplotlib",
    "ipykernel",
    "pyspark",
    "ray",
    "torch",
    "xgboost",
    "lightgbm",
    "catboost",
    "prophet",
    "neuralprophet",
    "neuralforecast",
    "statsforecast",
    "immutabledict",
)

_SCRIPT = f"""
import importlib.abc, os, sys

BLOCKED = frozenset({_OPTIONAL_TOP_LEVEL!r})

class _NotInstalled(importlib.abc.MetaPathFinder):
    # First on sys.meta_path: an import of a blocked distribution raises exactly what a missing
    # one raises, and sys.modules never gains an entry for it — the state a bare install is in.
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None

# Site-packages `.pth` shims can pre-register namespace packages (`google`) at interpreter
# start; only what appears *after* the finder is installed is an import the package made.
baseline = set(sys.modules)
sys.meta_path.insert(0, _NotInstalled())
# An identity in the environment must not rescue a core-only install: the extra check has to come
# before the Settings check, and the dry-run plan has to degrade gracefully with it present.
os.environ.update(SF_PROJECT_ID="example-project", SF_REGION="us-central1",
                  SF_REGISTRY_DATASET_ID="sf_registry", SF_SOURCE_DATASET_ID="sf_source",
                  SF_CODE_BUCKET="example-bucket")

import scale_forecasting
from scale_forecasting import agent_surfaces, backtest, calibration, config, dag, ensembler
from scale_forecasting import features, hpo, launch_plan, main, mcp, playground, reconciliation
from scale_forecasting import review, sdk, worker
from scale_forecasting.errors import MissingExtraError
from scale_forecasting.metrics import METRIC_NAMES
from scale_forecasting.models import list_models

assert len(list_models()) == 34, len(list_models())
assert len(METRIC_NAMES) == 21, len(METRIC_NAMES)

# Built-in agent probe and MCP server work in the pure-core install with zero optional extras.
env_probe = agent_surfaces.probe_environment()
assert env_probe["extras"]["gcp"]["installed"] is False
server = mcp.McpServer(allow_launch=False)
dry_resp = server.handle_message({{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {{
        "name": "validate_and_dry_run",
        "arguments": {{"config": {str(CONFIG)!r}, "ignore_unavailable_models": True}},
    }},
}})
assert dry_resp is not None and "result" in dry_resp and not dry_resp["result"]["isError"]

# The playground is the whole point of the bare install: fit + 3-fold backtest, no cloud.
assert playground._main(["--model", "holtwinters", "--horizon", "7", "--backtest"]) == 0

cfg = config.load_config({str(CONFIG)!r})

# Offline verbs still work, identity present or not.
run_id = main.run(cfg, dry_run=True)
assert run_id == sdk.Forecaster(cfg).run_id, (run_id, sdk.Forecaster(cfg).run_id)
assert sdk.Forecaster(cfg).dag()
main._main(["--config", {str(CONFIG)!r}, "--dry-run"])

def expect(extra, fn):
    try:
        fn()
    except MissingExtraError as exc:
        text = str(exc)
        assert f'pip install "scale-forecasting[{{extra}}]"' in text, text
        return
    raise AssertionError(f"expected MissingExtraError naming [{{extra}}] from {{fn}}")

# Everything that reaches Google Cloud names [gcp] (or the runtime's extra) before touching it.
expect("gcp", lambda: main.run(cfg))
expect("gcp", lambda: sdk.Forecaster(cfg).run())
expect("gcp", lambda: main._main(["--config", {str(CONFIG)!r}, "--probe"]))
expect("gcp", lambda: main._main(["--config", {str(CONFIG)!r}, "--dry-run", "--feasibility"]))
from scale_forecasting import submit, vertex_submit, gce_submit, ray_submit, automl_submit
from scale_forecasting.registry import ops
expect("gcp", lambda: submit.main(["--config", {str(CONFIG)!r}]))
expect("gcp", lambda: vertex_submit.main(["--config", {str(CONFIG)!r}]))
expect("gcp", lambda: gce_submit.main(["--config", {str(CONFIG)!r}]))
expect("ray", lambda: ray_submit.main(["--config", {str(CONFIG)!r}]))
expect("models-automl", lambda: automl_submit.main(["--config", {str(CONFIG)!r}]))
expect("gcp", lambda: ops.main(["doctor"]))

# Plotting names [notebook].
import pandas as pd
expect("notebook", lambda: review._pyplot())
expect("notebook", lambda: sdk.plot_trace(pd.DataFrame()))

pulled = sorted(m for m in set(sys.modules) - baseline if m.split(".")[0] in BLOCKED)
assert not pulled, f"an optional package was imported anyway: {{pulled}}"
print("ok")
"""


def test_core_install_runs_the_pure_layer_and_names_the_missing_extra(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,  # nothing from the repo root on sys.path by accident
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    assert proc.stdout.rstrip().endswith("ok"), proc.stdout
