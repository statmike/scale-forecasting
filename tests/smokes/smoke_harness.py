"""Smoke-test driver: dry → stage → run → verify → rerun → reverse-trace, one config at a time.

Each config under ``configs/smokes/`` exercises one runtime/hardware/ensemble combination of the
family→runtime DAG at ~100 series. This module drives one such config through the full lifecycle a
reviewer would run by hand and checks the result end to end:

1. **dry** — `launch_plan.plan_run`: resolve the run_id, the per-runtime model split, the fanout,
   and the exists-vs-new verdict, touching no GCS.
2. **stage** — `launch_plan.stage_run`: upload the config (+ code zip for Spark) and write the
   reproducibility manifest ``runs/<run_id>.plan.json``; capture the runnable launch commands.
3. **run** — `main.run`: submit every family on its runtime under one run_id and block to terminal.
4. **verify** — read the registry views back (`registry.reads.read_run_summary` / ``read_run_jobs``
   / ``read_leaderboard``) and assert the run reached COMPLETED, every expected family ran and
   succeeded with a real platform job id, and every configured model (plus the ensembles when
   enabled) scored onto the leaderboard.
5. **rerun / collision** — re-run the same config with no ``--force``: it must resolve the *same*
   run_id and, via append-only + dedupe-on-read, leave the leaderboard counts unchanged.
6. **reverse-trace** — print each family's stored ``system_job_id`` and the service it resolves to
   (Dataproc batch / Dataproc cluster job / Vertex Ray submission / BigQuery job), so a human can
   click straight through to the underlying job.

The pure helpers (`expected_families`, `verify_run_jobs`, `verify_leaderboard`, `verify_rerun`,
`format_trace`) take plain row dicts + a `RunConfig` and are unit-tested offline (`test_harness`).
The live orchestration (`run_smoke`) and the CLI are ``@gcp`` — they submit real jobs and cost
money, so they never run in the offline gate; the runbook (`docs/smoke_testing.md`) drives them one
at a time. Usage:

    .venv/bin/python tests/smokes/smoke_harness.py configs/smokes/01_serverless_cpu.json
    .venv/bin/python tests/smokes/smoke_harness.py configs/smokes/01_serverless_cpu.json --force
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scale_forecasting.config import RunConfig


# --- pure helpers (offline-testable) -------------------------------------------


def expected_families(cfg: RunConfig) -> list[str]:
    """The base families this config's DAG will run (excludes the ensemble node), in DAG order."""
    from scale_forecasting.dag import plan_dag

    return list(plan_dag(cfg).families)


def _service_for(runtime: str, spark_mode: str | None) -> str:
    """The GCP service a family's job runs on — where its ``system_job_id`` resolves in-console."""
    if runtime == "bigquery":
        return "BigQuery job"
    if runtime == "ray":
        return "Vertex Ray submission"
    if runtime == "spark":
        return "Dataproc cluster job" if spark_mode == "cluster" else "Dataproc Serverless batch"
    return runtime


def verify_run_jobs(job_rows: list[dict[str, Any]], cfg: RunConfig) -> list[str]:
    """Check the ``v_run_jobs`` rows against the config's DAG. Returns problems (empty list = OK).

    Asserts every expected base family ran, each reached COMPLETED, and each carries a real platform
    job id (``system_job_id``) for reverse-trace.
    """
    problems: list[str] = []
    by_family = {str(r["family"]): r for r in job_rows}
    for fam in expected_families(cfg):
        row = by_family.get(fam)
        if row is None:
            problems.append(f"family {fam!r} did not run (no run_jobs row)")
            continue
        status = str(row.get("status") or "")
        if status != "COMPLETED":
            problems.append(f"family {fam!r} status is {status or 'MISSING'!r}, expected COMPLETED")
        if not row.get("system_job_id"):
            problems.append(f"family {fam!r} has no system_job_id (reverse-trace broken)")
    return problems


def verify_leaderboard(board_rows: list[dict[str, Any]], cfg: RunConfig) -> list[str]:
    """Check the ``v_model_leaderboard`` rows cover every configured model (+ ensembles when on).

    Returns a list of problems (empty=OK). Learned ensemble strategies are only present when
    backtest is enabled (they need the OOF); calculated strategies always land when the ensemble is
    enabled, so the check only requires that *some* ``ensemble_*`` pseudo-model scored.
    """
    problems: list[str] = []
    present = {str(r["model_type"]) for r in board_rows}
    for model in cfg.models:
        if model not in present:
            problems.append(f"model {model!r} did not score onto the leaderboard")
    if cfg.ensemble.enabled:
        if not any(m.startswith("ensemble_") for m in present):
            problems.append("ensemble enabled but no ensemble_* pseudo-model scored")
    return problems


def verify_predictions(pred_counts: dict[str, int], cfg: RunConfig) -> list[str]:
    """Check every configured model actually produced forecasts (not just metadata).

    The leaderboard is built from ``forecast_metadata``, so a model whose cells **all failed** still
    shows there with ``n_cells`` set — but writes zero rows to ``forecast_predictions``. This is the
    guard that turns that silent failure into a FAIL: each configured model must have a non-zero
    prediction count (``pred_counts`` is ``model_type -> row count`` from
    `reads.read_prediction_counts`).
    """
    problems: list[str] = []
    for model in cfg.models:
        if pred_counts.get(model, 0) <= 0:
            problems.append(
                f"model {model!r} produced no forecast rows (fits failed? metadata without "
                "predictions) — its cells did not land in forecast_predictions"
            )
    return problems


def verify_rerun(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[str]:
    """Check a no-``--force`` re-run left the leaderboard unchanged (append-only + dedupe-on-read).

    A re-run of the same config lands byte-identical rows under the same run_id; the view dedupes on
    read, so both the model set and the per-model cell counts must match the first run exactly.
    """
    problems: list[str] = []

    def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {str(r["model_type"]): int(r.get("n_cells") or 0) for r in rows}

    b, a = _counts(before), _counts(after)
    if set(b) != set(a):
        problems.append(f"rerun changed the model set: {sorted(set(b) ^ set(a))}")
    for model in set(b) & set(a):
        if b[model] != a[model]:
            problems.append(
                f"rerun changed n_cells for {model!r}: {b[model]} -> {a[model]} "
                "(dedupe-on-read should keep this constant)"
            )
    return problems


def format_trace(job_rows: list[dict[str, Any]]) -> list[str]:
    """One reverse-trace line per family: id + runtime/hardware + the service it resolves to."""
    lines: list[str] = []
    for row in sorted(job_rows, key=lambda r: str(r.get("family"))):
        runtime = str(row.get("runtime") or "")
        spark_mode = row.get("spark_mode")
        hw = str(row.get("hardware") or "cpu")
        gpu = row.get("gpu_type")
        accel = f"/{gpu}" if gpu else ""
        service = _service_for(runtime, spark_mode if spark_mode is None else str(spark_mode))
        lines.append(
            f"  {str(row.get('family')):14s} {runtime}/{hw}{accel:6s} "
            f"{row.get('system_job_id')!s:40s} [{service}]"
        )
    return lines


# --- live orchestration (@gcp; costs money) ------------------------------------


@dataclass
class SmokeResult:
    """The end-to-end outcome of one smoke, for the CLI report and the living results log."""

    config_path: str
    run_id: str
    dry_verdict: str
    manifest_uri: str | None
    run_status: str | None
    reran: bool
    problems: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    leaderboard: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and self.run_status == "COMPLETED"


def run_smoke(
    config_path: str, *, force: bool = False, do_rerun: bool = True
) -> SmokeResult:  # pragma: no cover - @gcp: submits real jobs
    """Drive one smoke config through dry → stage → run → verify → rerun → trace. Live (@gcp)."""
    from scale_forecasting import launch_plan
    from scale_forecasting import main as main_mod
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

    # 1. dry — resolve id + verdict, no GCS.
    dry = launch_plan.plan_run(cfg, settings=settings, force=force)
    verdict = "exists" if dry.idempotency.exists else "new"

    # 2. stage — upload artifacts + manifest, capture runnable commands.
    staged = launch_plan.stage_run(cfg, settings=settings, force=force)

    # 3. run — submit every family + block to terminal.
    run_id = main_mod.run(cfg, settings=settings, force=force)

    # 4. verify — read the views back.
    summary = read_run_summary(run_id, settings=settings)
    run_status = str(summary.get("status")) if summary else None
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

    # 5. rerun / collision — same config, no force → same id, dedupe keeps the board constant.
    reran = False
    if do_rerun:
        run_id2 = main_mod.run(cfg, settings=settings)
        reran = True
        if run_id2 != run_id:
            problems.append(f"rerun resolved a different run_id: {run_id} -> {run_id2}")
        else:
            problems += verify_rerun(board, read_leaderboard(run_id2, settings=settings))

    # 6. reverse-trace — id → service per family.
    trace = format_trace(job_rows)

    return SmokeResult(
        config_path=config_path,
        run_id=run_id,
        dry_verdict=verdict,
        manifest_uri=staged.config_uri,
        run_status=run_status,
        reran=reran,
        problems=problems,
        trace=trace,
        leaderboard=board,
    )


def _report(result: SmokeResult) -> str:  # pragma: no cover - formatting for the CLI
    """Render a `SmokeResult` as a human-readable block for the runbook's living results log."""
    lines = [
        f"smoke: {result.config_path}",
        f"  run_id:   {result.run_id}  ({result.dry_verdict} at dry-run)",
        f"  status:   {result.run_status}",
        f"  manifest: {result.manifest_uri}",
        f"  rerun:    {'checked (same id, board unchanged)' if result.reran else 'skipped'}",
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
    """``python tests/smokes/smoke_harness.py <config.json> [--force] [--no-rerun]`` → exit code."""
    import argparse

    parser = argparse.ArgumentParser(description="Run one smoke config end to end (live @gcp).")
    parser.add_argument("config", help="path to a configs/smokes/*.json config")
    parser.add_argument(
        "--force", action="store_true", help="bump the attempt (fresh job under the same run_id)"
    )
    parser.add_argument(
        "--no-rerun", action="store_true", help="skip the rerun/collision check (one run only)"
    )
    ns = parser.parse_args(argv)
    result = run_smoke(ns.config, force=ns.force, do_rerun=not ns.no_rerun)
    print(_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    import sys

    sys.exit(main())
