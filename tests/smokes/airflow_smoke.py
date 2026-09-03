"""Airflow/Composer smoke: orchestrate one run through Composer, then verify it like a local run.

The direct-launch smokes (`smoke_harness.run_smoke`) prove each family runs on its runtime; this
harness proves the **Airflow layer** actually orchestrates them. It takes the SAME config, resolves
its ``run_id`` and stages its artifacts exactly as a local run does, then emits the run's
``dag_<run_id>.py`` (`airflow_emit.emit_airflow_dag`), imports it into a Composer environment,
triggers it, and waits for the run to land in the registry — the same terminal signal
`run_smoke` polls. Because the ``run_id`` is a digest of the config, a run orchestrated by Composer
writes the registry under the *identical* id a local run would, so success proves the emitted DAG
runs the same building blocks (same code local↔Composer). Verification reuses the pure checkers from
`smoke_harness` (`verify_run_jobs`, `verify_leaderboard`, `verify_predictions`), so the two smokes
hold the result to one standard.

Two control planes, deliberately: the DAG file is delivered with ``gcloud composer environments
storage dags import`` (a GCS copy — fast and reliable), but *confirming the parse* and *triggering
the run* go through the **Airflow REST API** on the environment's ``airflowUri``. The
``gcloud composer environments run`` verbs are not used: they route through
``executeAirflowCommand``, which spins up a pod per invocation and was observed to hang
indefinitely against a healthy environment — unacceptable in a loop that gates a live billing run.
The REST API answers the same questions in under a second and reports ``has_import_errors``
directly, which the CLI does not.

The builders below are pure (offline-unit-tested in `test_airflow_smoke`); the live driver
(`run_airflow_smoke`) shells out once for the environment lookup, calls the REST API, polls the
registry, and is ``@gcp`` — it provisions nothing but needs a running Composer environment
(``create_composer=true``) with the package delivered to its workers (``make composer-sync``).
The runbook (`docs/smoke_testing.md`) drives it. Usage:

    .venv/bin/python tests/smokes/airflow_smoke.py configs/smokes/15_airflow_multi_engine.json \
        --composer-env scale-forecasting --location us-central1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scale_forecasting.config import RunConfig


# --- pure helpers (offline-testable) -------------------------------------------


def dag_id_for(cfg: RunConfig) -> str:
    """The dag_id the emitter assigns this config's run — ``scale_forecasting_<run_id>``.

    Mirrors `airflow_emit.emit_airflow_dag`'s default so the trigger/list commands target exactly
    the DAG the import step uploads.
    """
    from scale_forecasting.dag import plan_dag

    return f"scale_forecasting_{plan_dag(cfg).run_id}"


def _project_flag(project: str | None) -> list[str]:
    """``--project <id>`` when a project is given, else nothing.

    The environment lives in the *deployment* project (``SF_PROJECT_ID``), which is not necessarily
    gcloud's ambient ``core/project`` — so the smoke must name it explicitly or ``gcloud`` looks in
    the wrong project and reports the environment ``NOT_FOUND``.
    """
    return ["--project", project] if project else []


def dags_import_command(
    env: str, location: str, local_dag_path: str, project: str | None = None
) -> list[str]:
    """``gcloud`` argv to copy a rendered DAG file into a Composer environment's DAG folder."""
    return [
        "gcloud",
        "composer",
        "environments",
        "storage",
        "dags",
        "import",
        "--environment",
        env,
        "--location",
        location,
        *_project_flag(project),
        "--source",
        local_dag_path,
    ]


def airflow_uri_command(env: str, location: str, project: str | None = None) -> list[str]:
    """``gcloud`` argv to read an environment's Airflow web URI — the REST API's base address.

    A plain ``describe`` against the Composer control plane: it returns in about a second and does
    NOT go through ``executeAirflowCommand``, so it is safe to call before a live run.
    """
    return [
        "gcloud",
        "composer",
        "environments",
        "describe",
        env,
        "--location",
        location,
        *_project_flag(project),
        "--format",
        "value(config.airflowUri)",
    ]


def dag_url(airflow_uri: str, dag_id: str) -> str:
    """Airflow REST URL for one DAG's detail — the parse check (``has_import_errors``) reads it."""
    return f"{airflow_uri.rstrip('/')}/api/v1/dags/{dag_id}"


def dag_runs_url(airflow_uri: str, dag_id: str) -> str:
    """Airflow REST URL for one DAG's runs collection — POST here to trigger a manual run."""
    return f"{dag_url(airflow_uri, dag_id)}/dagRuns"


# --- live orchestration (@gcp; needs a running Composer env) --------------------


@dataclass
class AirflowSmokeResult:
    """The end-to-end outcome of one Airflow-orchestrated smoke, for the CLI report."""

    config_path: str
    run_id: str
    dag_id: str
    run_status: str | None
    problems: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    leaderboard: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and self.run_status == "COMPLETED"


def run_airflow_smoke(
    config_path: str,
    *,
    composer_env: str,
    location: str | None = None,
    timeout_s: float = 7200.0,
    poll_interval_s: float = 30.0,
) -> AirflowSmokeResult:  # pragma: no cover - @gcp: drives Composer + submits real jobs
    """Drive one config via Composer: stage → emit → import → trigger → wait → verify. Live (@gcp).

    ``composer_env``/``location`` name the environment (``location`` defaults to ``SF_REGION``).
    Waits up to ``timeout_s`` for the run to reach a terminal status in the registry, polling every
    ``poll_interval_s``. Returns an `AirflowSmokeResult`; the checks are the same the direct smoke
    applies, plus the implicit proof that the registry row appeared under the config-derived id.
    """
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    from smoke_harness import (  # sibling module (tests/smokes is on sys.path); flat, not a package
        format_trace,
        verify_leaderboard,
        verify_predictions,
        verify_run_jobs,
    )

    from scale_forecasting import airflow_emit, launch_plan
    from scale_forecasting.config import load_config
    from scale_forecasting.registry.jobs import read_run_jobs
    from scale_forecasting.registry.reads import (
        read_leaderboard,
        read_prediction_counts,
        read_run_summary,
    )
    from scale_forecasting.settings import Settings

    cfg = load_config(config_path)
    settings = Settings.resolve()
    location = location or settings.region
    project = settings.project_id  # the DEPLOYMENT project — gcloud's ambient project may differ
    dag_id = dag_id_for(cfg)

    # 1. plan + stage — resolve the run_id and upload the config (+ code zip for Spark), exactly as
    #    a local run does, so the DAG's CONFIG_URI points at real staged artifacts.
    plan = launch_plan.plan_run(cfg, settings=settings)
    staged = launch_plan.stage_run(cfg, settings=settings)
    run_id = plan.run_id

    # gcloud composer's run/storage verbs shell into the environment (the `run` verbs spin up a pod
    # via executeAirflowCommand) and intermittently 500, so every one-shot call gets a bounded retry
    # — a single transient blip must not abort a live smoke that's about to bill compute.
    def _cli(argv: list[str], *, retries: int = 4, backoff_s: float = 15.0):
        last = subprocess.run(argv, capture_output=True, text=True)
        for _ in range(retries):
            if last.returncode == 0:
                return last
            time.sleep(backoff_s)
            last = subprocess.run(argv, capture_output=True, text=True)
        if last.returncode != 0:
            raise subprocess.CalledProcessError(last.returncode, argv, last.stdout, last.stderr)
        return last

    # 2. emit the DAG for this run and import it into the Composer environment's DAG folder.
    source = airflow_emit.emit_airflow_dag(cfg, staged.config_uri, dag_id=dag_id)
    with tempfile.TemporaryDirectory() as tmp:
        dag_path = Path(tmp) / f"dag_{run_id}.py"
        dag_path.write_text(source)
        _cli(dags_import_command(composer_env, location, str(dag_path), project))

    # 3. wait for Airflow to parse the new file, then trigger a manual run — over the REST API, not
    #    the `gcloud composer environments run` CLI (see the module docstring for why).
    #    AuthorizedSession signs each call with ADC and refreshes on its own, so a slow poll cannot
    #    outlive its token the way a captured bearer string would.
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    airflow_uri = _cli(airflow_uri_command(composer_env, location, project)).stdout.strip()
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    detail_url = dag_url(airflow_uri, dag_id)

    # A DAG appears here only once the scheduler has parsed the file, so a 200 IS the parse
    # confirmation — and `has_import_errors` tells us the parse *succeeded*, which the CLI's
    # `dags list` could not. Transient errors just mean "not confirmed yet"; keep waiting.
    deadline = time.monotonic() + timeout_s
    parsed = False
    while time.monotonic() < deadline:
        resp = session.get(detail_url, timeout=60)
        if resp.status_code == 200:
            detail = resp.json()
            if detail.get("has_import_errors"):
                raise RuntimeError(f"Airflow reports import errors for {dag_id}: {detail}")
            parsed = True
            break
        time.sleep(poll_interval_s)
    if not parsed:
        raise RuntimeError(f"Airflow never parsed {dag_id} within {timeout_s}s")

    # Composer can create DAGs paused (`dags_are_paused_at_creation`); an unpause is a no-op when
    # it does not, and the difference is a run that silently never starts.
    session.patch(detail_url, json={"is_paused": False}, timeout=60)

    triggered = session.post(dag_runs_url(airflow_uri, dag_id), json={"conf": {}}, timeout=60)
    if triggered.status_code not in (200, 201):
        raise RuntimeError(f"trigger failed ({triggered.status_code}): {triggered.text}")

    # 4. wait for the run to reach a terminal header status in the registry (same signal run_smoke
    #    uses — the DAG writes the same registry a local run does).
    run_status: str | None = None
    terminal = {"COMPLETED", "FAILED", "PARTIAL"}
    while time.monotonic() < deadline:
        summary = read_run_summary(run_id, settings=settings)
        run_status = str(summary.get("status")) if summary else None
        if run_status in terminal:
            break
        time.sleep(poll_interval_s)

    # 5. verify — read the views back and hold the run to the same standard as the direct smoke.
    job_rows = read_run_jobs(run_id, settings=settings)
    board = read_leaderboard(run_id, settings=settings)
    pred_counts = read_prediction_counts(run_id, settings=settings)
    problems = (
        verify_run_jobs(job_rows, cfg)
        + verify_leaderboard(board, cfg)
        + verify_predictions(pred_counts, cfg)
    )
    if run_status != "COMPLETED":
        problems.append(f"run status is {run_status!r}, expected COMPLETED")

    return AirflowSmokeResult(
        config_path=config_path,
        run_id=run_id,
        dag_id=dag_id,
        run_status=run_status,
        problems=problems,
        trace=format_trace(job_rows),
        leaderboard=board,
    )


def _report(result: AirflowSmokeResult) -> str:  # pragma: no cover - formatting for the CLI
    """Render an `AirflowSmokeResult` as a human-readable block for the runbook's results log."""
    lines = [
        f"airflow smoke: {result.config_path}",
        f"  dag_id:   {result.dag_id}",
        f"  run_id:   {result.run_id}",
        f"  status:   {result.run_status}",
        "  reverse-trace:",
        *result.trace,
        "  leaderboard (model → mean_wape, n_cells):",
    ]
    for row in result.leaderboard:
        lines.append(
            f"    {str(row.get('model_type')):20s} "
            f"wape={row.get('mean_wape')}  n_cells={row.get('n_cells')}"
        )
    lines.append(f"  RESULT:   {'PASS' if result.ok else 'FAIL'}")
    if result.problems:
        lines.append("  problems:")
        lines += [f"    - {p}" for p in result.problems]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - @gcp CLI wrapper
    """``airflow_smoke.py <config.json> --composer-env <env> [...]`` → process exit code."""
    import argparse

    parser = argparse.ArgumentParser(description="Run one smoke config via Composer (live @gcp).")
    parser.add_argument("config", help="path to a configs/smokes/*.json config")
    parser.add_argument(
        "--composer-env", default="scale-forecasting", help="Composer environment name"
    )
    parser.add_argument("--location", default=None, help="Composer region (defaults to SF_REGION)")
    ns = parser.parse_args(argv)
    result = run_airflow_smoke(ns.config, composer_env=ns.composer_env, location=ns.location)
    print(_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    import sys

    sys.exit(main())
