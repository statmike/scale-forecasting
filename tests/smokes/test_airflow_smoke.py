"""Offline unit tests for the Airflow smoke's pure builders (`airflow_smoke`).

The live driver (`airflow_smoke.run_airflow_smoke`) shells out to ``gcloud composer`` and calls the
Airflow REST API; it is exercised by the runbook, not here. These tests pin the pure argv builders,
the REST URL builders and the dag_id derivation — the parts that decide *which* environment/DAG the
live path targets — so a typo can't silently point the smoke at the wrong place. No GCP, no
subprocess, no network.
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


def test_airflow_uri_command_is_a_plain_describe() -> None:
    # NOT `gcloud composer environments run`: that routes through executeAirflowCommand, which
    # spins a pod per call and was observed to hang against a healthy environment. A describe is a
    # control-plane read that returns in about a second, so it is safe ahead of a billing run.
    argv = a.airflow_uri_command("scale-forecasting", "us-central1")
    assert argv[:5] == ["gcloud", "composer", "environments", "describe", "scale-forecasting"]
    assert "run" not in argv
    assert argv[-2:] == ["--format", "value(config.airflowUri)"]


def test_rest_urls_hang_off_the_airflow_uri() -> None:
    uri = "https://abc-dot-us-central1.composer.googleusercontent.com"
    dag_id = "scale_forecasting_abc"
    assert a.dag_url(uri, dag_id) == f"{uri}/api/v1/dags/{dag_id}"
    assert a.dag_runs_url(uri, dag_id) == f"{uri}/api/v1/dags/{dag_id}/dagRuns"


def test_rest_urls_tolerate_a_trailing_slash_on_the_uri() -> None:
    # `gcloud ... --format=value(config.airflowUri)` has returned both forms; a doubled slash makes
    # Airflow answer 404 and the smoke would read that as "never parsed".
    plain = a.dag_url("https://x.example.com", "d")
    assert a.dag_url("https://x.example.com/", "d") == plain
    assert "//api" not in a.dag_runs_url("https://x.example.com/", "d")


def test_commands_thread_the_deployment_project_when_given() -> None:
    # The env lives in the deployment project (SF_PROJECT_ID), which may differ from gcloud's
    # ambient project — every command must name it explicitly or gcloud reports the env NOT_FOUND.
    proj = "my-deployment-project"
    imp = a.dags_import_command("scale-forecasting", "us-central1", "/tmp/dag_x.py", proj)
    uri = a.airflow_uri_command("scale-forecasting", "us-central1", proj)
    for argv in (imp, uri):
        assert "--project" in argv and argv[argv.index("--project") + 1] == proj
