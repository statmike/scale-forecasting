"""Offline unit tests for the Airflow smoke's pure command-builders (`airflow_smoke`).

The live driver (`airflow_smoke.run_airflow_smoke`) shells out to ``gcloud composer`` and is
exercised by the runbook, not here. These tests pin the pure argv builders and the dag_id derivation
— the parts that decide *which* environment/DAG the live path targets — so a typo can't silently
point the smoke at the wrong place. No GCP, no subprocess.
"""

from __future__ import annotations

from typing import Any

import airflow_smoke as a

from scale_forecasting.config import RunConfig
from scale_forecasting.dag import plan_dag


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "smoke_airflow_test",
        "data": {"source_table": "source_series_iceberg", "horizon": 7, "series_limit": 100},
        "models": ["theta", "arima_plus"],
    }
    base.update(over)
    return RunConfig(**base)


def test_dag_id_matches_the_emitter_default() -> None:
    # the trigger/list commands must target exactly the dag_id emit_airflow_dag assigns by default.
    cfg = _cfg()
    assert a.dag_id_for(cfg) == f"scale_forecasting_{plan_dag(cfg).run_id}"


def test_dags_import_command_targets_env_and_source() -> None:
    argv = a.dags_import_command("scale-forecasting", "us-central1", "/tmp/dag_x.py")
    assert argv[:6] == ["gcloud", "composer", "environments", "storage", "dags", "import"]
    assert "--environment" in argv and argv[argv.index("--environment") + 1] == "scale-forecasting"
    assert "--location" in argv and argv[argv.index("--location") + 1] == "us-central1"
    assert "--source" in argv and argv[argv.index("--source") + 1] == "/tmp/dag_x.py"
    assert "--project" not in argv  # omitted when no project given (uses gcloud's ambient project)


def test_dags_list_command_runs_against_the_env() -> None:
    argv = a.dags_list_command("scale-forecasting", "us-central1")
    assert argv[:5] == ["gcloud", "composer", "environments", "run", "scale-forecasting"]
    assert argv[-2:] == ["dags", "list"]


def test_dags_trigger_command_passes_dag_id_after_the_separator() -> None:
    argv = a.dags_trigger_command("scale-forecasting", "us-central1", "scale_forecasting_abc")
    # everything after "--" is forwarded to the Airflow CLI verbatim: -d <dag_id>.
    assert argv[-3:] == ["--", "-d", "scale_forecasting_abc"]
    assert "trigger" in argv and argv[argv.index("trigger") + 1] == "--"


def test_commands_thread_the_deployment_project_when_given() -> None:
    # The env lives in the deployment project (SF_PROJECT_ID), which may differ from gcloud's
    # ambient project — every command must name it explicitly or gcloud reports the env NOT_FOUND.
    proj = "my-deployment-project"
    imp = a.dags_import_command("scale-forecasting", "us-central1", "/tmp/dag_x.py", proj)
    lst = a.dags_list_command("scale-forecasting", "us-central1", proj)
    trg = a.dags_trigger_command("scale-forecasting", "us-central1", "scale_forecasting_abc", proj)
    for argv in (imp, lst, trg):
        assert "--project" in argv and argv[argv.index("--project") + 1] == proj
    # --project stays on the gcloud side of the "--" separator (not forwarded to the Airflow CLI).
    assert trg.index("--project") < trg.index("--")
