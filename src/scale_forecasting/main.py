"""Orchestrate one run end-to-end — Spark + BigQuery-native in parallel, one run_id.

A single config can mix Python-runtime models (run
by the Spark engine as per-cell fan-out) and BigQuery-native models (run as SQL inside BigQuery);
`run` executes **both runtimes in parallel under one shared ``run_id`` and one
``run_registry`` header row**, so a native model and a Spark model land in the same run and are
directly comparable on ``v_model_leaderboard`` (the "wall-clock ≈ max(python, bq), not
sum" thesis).

Two invariants make the single-run parallelism work (see `ids`):

1. **One run_id from the full config.** ``make_run_id`` is a pure digest over the *whole* config
   incl. ``cfg.models``; both engines receive the full ``cfg`` so they derive the same id. Each is
   handed only its own executed subset (``python_models`` / ``bq_models``) via the ``models``
   argument, so the BigQuery-native models never become Spark cells (which would raise
   ``NotImplementedError`` in ``worker.run_cell``) and vice-versa.
2. **One header owner.** `run` alone writes the header (RUNNING) up front and finalizes it;
   both engines run in **contributor mode** (``manage_header=False``), skipping the header lifecycle
   and only writing their cell rows. No two writers ever touch the header, so there is no UPDATE
   race — the only in-window header write is ``submit_batch``'s best-effort telemetry stamp, which
   completes inside the joined future before `run`'s finalize.

**Parallelism.** The remote Spark batch is launched on a worker thread (``submit_batch(wait=True)``)
while the in-process BigQuery engine runs on the main thread; the BQ work (minutes, in-process)
overlaps the Spark provisioning floor. `run` joins both, rolls the two outcomes into one
combined status (COMPLETED iff both green, else FAILED — finalized *before* re-raising so the run
stays queryable and the CLI exits non-zero), and returns the shared ``run_id``.

**Coarsening (documented).** A remote contributor batch can't return its run-level PARTIAL (some
cells errored) to the orchestrator, so a SUCCEEDED batch is reported COMPLETED; per-model failure
stays visible on ``v_model_leaderboard`` (a failed model → NULL metric AVGs).

Both Python runtimes are supported and dispatched by ``cfg.python_runtime``: ``"spark"`` launches a
Dataproc Serverless batch (`submit_batch`), ``"ray"`` an autoscaling
Vertex Ray cluster (`submit_ray`) — either way on the worker
thread, in contributor mode, in parallel with the in-process BigQuery engine under one run_id.

Public surface: ``run(cfg, *, dry_run=False) -> run_id``, the offline ``plan_run(cfg) ->
LaunchPlan`` (id + fanout + launch-command templates), ``stage_run(cfg) -> LaunchPlan`` (upload
artifacts + runnable commands + reproducibility manifest, no submit), and
``python -m scale_forecasting.main (--config … | --config-uri gs://…) [--dry-run | --stage-only]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import job_launch, launch_plan, shared_clusters
from .errors import get_logger
from .registry.ids import make_run_id

if TYPE_CHECKING:
    from .config import RunConfig
    from .dag import RunDag
    from .probes.cancel import CancelReport
    from .probes.reconcile import ProbeReport
    from .settings import Settings

_log = get_logger(__name__)


def run(
    cfg: RunConfig,
    *,
    dry_run: bool = False,
    spark: object | None = None,
    settings: Settings | None = None,
    force: bool = False,
    n_series: int | None = None,
    max_executors: int | None = None,
) -> str:
    """Execute one run as a DAG: every model family in parallel under one run_id; return that id.

    Plans the run's DAG (`dag.plan_dag`) — one job per model family present in the config
    (statistical / ml / deep_learning each on its resolved runtime, native in BigQuery) plus the
    downstream ensemble node — then:

    * ``dry_run=True`` → delegate to `launch_plan.plan_run` (the offline "what would this
      schedule" path): resolve the run_id + `estimate_fanout` + emit the launch-command templates,
      touching no GCP, and return the run_id.
    * otherwise → resolve `Settings`, ``ensure_tables`` + ``write_header`` (RUNNING, the one shared
      header), launch each Python family's job on its own worker thread
      (`job_launch.launch_family_job`) and run the BigQuery-native family inline
      (`job_launch.launch_native_job`) so they all overlap — each in contributor mode with its own
      ``run_jobs`` row. Once every family job joins and *all* of them succeeded, and
      ``cfg.ensemble.enabled``, run the ensemble DAG node (`job_launch.launch_ensemble_job`, which
      reads the just-written base predictions/OOF and scores each consensus onto the leaderboard
      under its own ``run_jobs`` row), then finalize the header with the combined status +
      wall-clock
      ``runtime_seconds`` + ``bq_models`` (+ the native engine's observed ``n_series`` when it ran).
      On any family *or* ensemble failure the header is finalized FAILED/PARTIAL before the first
      error re-raises, so the run stays queryable and the CLI exits non-zero.

    ``spark`` is an optional injected `SparkSession` (incl. a Spark Connect
    ``DataprocSparkSession``). When supplied, a Spark-runtime family runs **in-process against that
    session** instead of a remote Dataproc batch — the notebook / Connect demo path — using the
    identical engine code (the injectable-session seam). The default (``None``) keeps the
    remote-batch behavior every CLI/Composer caller relies on.

    ``settings`` optionally injects a resolved `Settings` (the GCP infra identity); the default
    (``None``) resolves it from the ``SF_*`` environment. The SDK ``Forecaster`` uses this to thread
    an explicit identity instead of relying on process env. ``force`` bumps each family's
    ``run_jobs`` attempt (a fresh, distinctly-keyed job under the same run_id) and shapes the
    ``dry_run`` plan's re-run guidance.

    ``n_series`` overrides ``data.series_limit`` (the 10→100→1k→100k scale knob) before anything
    else, so it changes the ``run_id`` and every family sees the same limit. ``max_executors`` caps
    a remote Spark batch's dynamic-allocation ceiling (ignored by the in-process/Ray paths).

    Idempotent: the config-pinned run_id + append-only/dedupe-on-read cell writes mean re-running
    the same config lands byte-identical rows. ``compute.profile.source: "auto"`` is locked to what
    it resolves to (`launch_plan.lock_profile_source`), exactly as the plan/stage verbs do it, but
    that resolved pointer is excluded from the id (`registry.ids`) — otherwise a re-run sized from
    newer evidence would land under a new id and the dedupe above would never fire. A source an
    operator pinned by hand is checked for drift first (`profiling.source.check_pinned_source`) and
    fails the run.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from .dag import plan_dag
    from .profiling.source import check_pinned_source
    from .registry.header import header_status
    from .registry.lifecycle import run_header
    from .settings import Settings

    # The series-limit override is applied first so it flows into the run_id and every family — a
    # different scale is a distinct, independently-queryable run.
    cfg = cfg.with_series_limit(n_series)

    if dry_run:
        # Single-source the offline plan: plan_run resolves the id + fanout + runtime split, reports
        # the exists-vs-new verdict, and emits the launch-command templates. It returns the run_id,
        # so this contract is unchanged.
        return launch_plan.plan_run(cfg, settings=settings, force=force).run_id

    settings = settings or Settings.resolve()

    # A pinned ``compute.profile.source`` is a human assertion that one specific run's measurements
    # apply here; if the data has moved under it, say so once, now, rather than six times from
    # inside each family's launch. Before the lock below, which would otherwise turn the system's
    # own ``auto`` choice into something that looks pinned. ``force`` overrides, as it does the
    # idempotency guard.
    check_pinned_source(cfg, settings=settings, force=force)
    # Lock `source: "auto"` before the digest — the same step `launch_plan.plan_run` / ``stage_run``
    # take, and it has to happen here too or the three verbs would disagree about this run's
    # identity: ``--dry-run`` would report an id the real run then never uses.
    cfg = launch_plan.lock_profile_source(cfg, settings=settings)

    run_dag = plan_dag(cfg)
    run_id = run_dag.run_id

    # Idempotency guard: the run_id is config-pinned, so re-running the same config is a no-op once
    # it has COMPLETED — return without relaunching. Relaunching would resubmit each family's
    # deterministic platform job id (a Dataproc batch_id / Ray submission_id) and collide
    # (AlreadyExists), and the reused-attempt terminal write would clobber the completed run's job
    # rows. ``force`` re-executes as a fresh attempt (distinct job ids); a run that never completed
    # (never ran, or a prior FAILED/PARTIAL) falls through and runs.
    if not force and header_status(run_id, settings=settings) == "COMPLETED":
        _log.info("run %s already COMPLETED; skipping relaunch (pass force=True to re-run)", run_id)
        return run_id

    _log.info(
        "run %s start: families=%s ensemble=%s",
        run_id,
        run_dag.families,
        run_dag.ensemble_enabled,
    )

    bq_outcome = None
    # One error slot per family job (keyed by family name), plus the ensemble node's.
    job_errors: dict[str, BaseException] = {}
    ensemble_error: BaseException | None = None
    native = run_dag.native_job
    python_jobs = run_dag.python_jobs

    # Microbatch ensemble overlaps the base jobs: it drains series as each one's full base set lands
    # (rather than waiting for the join like the barrier), so it runs as a concurrent pool task with
    # an ``upstream_done`` predicate the join flips. ``base_done`` is set once every family (Python
    # + native) has joined, telling the drain loop no further base predictions will land. A family
    # that fails leaves its models absent, so no series ever reaches the full-base-set readiness bar
    # and the concurrent node drains nothing — same "no ensembles when a base family failed" outcome
    # as the barrier's post-join skip, just reached by readiness rather than an up-front gate.
    ensemble_concurrent = run_dag.ensemble_enabled and cfg.compute.ensemble.mode == "microbatch"
    base_done = threading.Event()

    # One header owner: run_header writes RUNNING on entry and finalizes once, after every family
    # job joins, with the combined status computed below. Every job runs with manage_header=False so
    # nothing else touches this row. Per-family errors are captured (not raised through the block)
    # so the finalize records the right status; the first is re-raised after, for a non-zero exit.
    with run_header(cfg, run_id, settings=settings, manage=True) as hdr:
        # Launch each Python family on its own worker thread, and run the BigQuery-native family
        # inline on the main thread, so all families overlap. Each family carries the same
        # contributor-mode contract (its model subset + shared header owned here) and its own
        # run_jobs row, so N heterogeneous families run under one run_id.
        # When the run has several ephemeral Ray (or Dataproc-cluster) families, provision one
        # shared cluster per runtime for the duration of the launch block (each family submits its
        # own job to it, torn down once on exit); otherwise these yield None and each family
        # self-provisions as before.
        # +1 pool worker for the concurrent microbatch ensemble so it never contends with a family
        # for a thread; barrier mode keeps the exact family-only pool it always had.
        max_workers = max(1, len(python_jobs)) + (1 if ensemble_concurrent else 0)
        with (
            shared_clusters.shared_ray_cluster(cfg, run_dag, run_id, settings) as ray_cluster,
            shared_clusters.shared_spark_cluster(cfg, run_dag, run_id, settings) as spark_cluster,
            ThreadPoolExecutor(max_workers=max_workers) as pool,
        ):
            futures = {
                pool.submit(
                    job_launch.launch_family_job,
                    cfg,
                    job,
                    run_id,
                    settings,
                    spark,
                    force=force,
                    max_executors=max_executors,
                    ray_cluster=ray_cluster,
                    spark_cluster=spark_cluster,
                ): job
                for job in python_jobs
            }
            # Microbatch: fire the ensemble now, concurrently with the base jobs, draining ready
            # series until the join flips base_done. Barrier: it stays a post-join step (below).
            ensemble_future = (
                pool.submit(
                    job_launch.launch_ensemble_job,
                    cfg,
                    run_id,
                    settings,
                    force=force,
                    upstream_done=base_done.is_set,
                )
                if ensemble_concurrent
                else None
            )
            if native is not None:
                try:
                    bq_outcome = job_launch.launch_native_job(
                        cfg, native, run_id, settings, force=force
                    )
                except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                    job_errors["native"] = exc
            for future, job in futures.items():
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                    job_errors[job.family] = exc
            # Every base family has joined: no more base predictions will land, so the concurrent
            # drain loop can stop after its final ready-series pass. Set before the ensemble join.
            base_done.set()
            if ensemble_future is not None:
                try:
                    ensemble_future.result()
                except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                    ensemble_error = exc

        # Barrier ensemble: it reads every family's base predictions / backtest_oof, so it runs
        # strictly after the join and only when every family succeeded. (Microbatch already ran
        # concurrently above.) A failure here is captured like a family error — the ensembles are
        # part of the run's success contract.
        if not ensemble_concurrent and not job_errors and run_dag.ensemble_enabled:
            try:
                job_launch.launch_ensemble_job(cfg, run_id, settings, force=force)
            except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                ensemble_error = exc

        # Combined status across the family jobs that ran: all green → COMPLETED, all failed →
        # FAILED, some but not all → PARTIAL (surviving families' forecasts stay usable). An
        # ensemble failure on top of all-green families fails the run: full output undelivered.
        status = _combined_status(run_dag, job_errors, ensemble_error)
        fields: dict[str, object] = {"bq_models": list(native.models) if native else []}
        if bq_outcome is not None:
            fields["n_series"] = bq_outcome.n_series
        hdr.finalize(status=status, **fields)

    first_error = next(iter(job_errors.values()), None) or ensemble_error
    if first_error is not None:
        # Re-raise the first failure so the CLI exits non-zero; the header already records the
        # combined status (FAILED or PARTIAL).
        raise first_error
    _log.info("run %s done: status=%s", run_id, status)
    return run_id


def _combined_status(
    run_dag: RunDag,
    job_errors: dict[str, BaseException],
    ensemble_error: BaseException | None,
) -> str:
    """Roll the per-family job outcomes into one run status (pure).

    Over the family jobs that ran (one per family in the DAG): every job green → ``COMPLETED``;
    every job failed → ``FAILED``; a mix → ``PARTIAL``. An ensemble failure downgrades an
    otherwise-``COMPLETED`` run to ``FAILED`` (the requested output is incomplete); it never masks a
    family ``PARTIAL``/``FAILED``.
    """
    n_jobs = len(run_dag.jobs)
    n_failed = len(job_errors)
    if n_failed == 0:
        engine_status = "COMPLETED"
    elif n_failed == n_jobs:
        engine_status = "FAILED"
    else:
        engine_status = "PARTIAL"

    if engine_status == "COMPLETED" and ensemble_error is not None:
        return "FAILED"
    return engine_status


def _emit_airflow(cfg: RunConfig, config_uri: str, *, out_path: str | None = None) -> str:
    """Render this run's Airflow DAG to a local file and return the path (offline — touches no GCP).

    The "emit" verb: resolves the run's DAG and renders it to a standalone ``dag_<run_id>.py``
    (`airflow_emit.emit_airflow_dag`) whose tasks load the config from ``config_uri`` — the same
    ``--config``/``--config-uri`` value passed in, embedded verbatim. Pass a ``gs://``
    ``--config-uri`` (a staged config, digest == run_id) to emit a DAG a Composer environment can
    run directly; a local ``--config`` path emits a DAG suitable for local parse/compile checks and
    inspection. ``out_path`` overrides the default ``./dag_<run_id>.py`` destination. Writing is the
    only side effect.
    """
    from pathlib import Path

    from .airflow_emit import emit_airflow_dag

    run_id = make_run_id(cfg)
    source = emit_airflow_dag(cfg, config_uri)
    out = Path(out_path) if out_path else Path(f"dag_{run_id}.py")
    out.write_text(source, encoding="utf-8")
    return str(out)


def _print_probe_report(report: ProbeReport) -> None:
    """Print a `ProbeReport` as one compact per-family table — one job or a whole run, same shape.

    A header line (``run <id>  status=…  escalated=…  disagreement=…``) then one row per family:
    ``family · runtime · registry · native · verdict · n_done/n_expected · detail``. Written to
    stdout (not the logger) because the report *is* the ``--probe`` verb's output — it must show
    regardless of the log level — and it stays plain text (no plots, no colour) so it reads
    identically in a terminal, a notebook, or a Composer task log.
    """
    row_fmt = "  %-14s %-8s %-10s %-10s %-17s %-9s %s"
    print(
        f"run {report.run_id}  status={report.status}  "
        f"escalated={report.escalated}  disagreement={report.disagreement}"
    )
    print(row_fmt % ("family", "runtime", "registry", "native", "verdict", "done/exp", "detail"))
    for fv in report.families:
        expected = fv.n_expected if fv.n_expected is not None else "?"
        print(
            row_fmt
            % (
                fv.family,
                fv.runtime or "-",
                fv.registry_status or "-",
                fv.native_state or "-",
                fv.verdict,
                f"{fv.n_done}/{expected}",
                fv.detail or "",
            )
        )


def _print_cancel_report(report: CancelReport) -> None:
    """Print a `CancelReport` — the blast-radius preview, then (if executed) the per-family outcome.

    A no-``--force`` call is a **preview only**: it lists what *would* stop and what data is kept,
    then a "Confirm with --force" line. With ``--force`` it also prints each family's outcome and
    the run's rolled-up header status. Written to stdout (not the logger): the report *is* the
    ``--cancel`` verb's output and must show regardless of log level.
    """
    plan = report.plan
    verb = "Cancelled" if report.executed else "Cancel"
    print(
        f"{verb} run {report.run_id}: {plan.n_cancellable} in-flight job(s) "
        f"{'stopped' if report.executed else 'will be stopped'}; partial results are RETAINED"
    )
    row_fmt = "  %-14s %-16s %-10s %s"
    print(row_fmt % ("family", "runtime", "status", "effect"))
    for item in plan.items:
        print(row_fmt % (item.family, item.runtime or "-", item.registry_status or "-", item.note))
    if plan.ensemble_suppressed:
        print("  note: the ensemble node is suppressed because a base family is cancelled")
    if not report.executed:
        print("Confirm with --force (CLI) / confirm=True (SDK) to stop these jobs.")
        return
    print(f"actor={report.actor}  reason={report.reason or '-'}  header={report.header_status}")
    for oc in report.outcomes:
        state = "cancelled" if oc.cancelled else "NOT cancelled"
        print(f"  {oc.family:<14} {state:<14} {oc.detail}")


def _main(argv: list[str] | None = None) -> None:
    """CLI: ``main (--config …|--config-uri …) [--dry-run|--stage-only|--probe|--cancel|…]``."""
    import argparse

    from .config import load_config_uri

    p = argparse.ArgumentParser(prog="main", description="Run a forecast (Spark + BigQuery).")
    # Accept either a local path (--config, the interactive UX) or a gs:// URI (--config-uri, what
    # the emitted portable "main" command references — the staged config, digest == run_id). Exactly
    # one is required; load_config_uri resolves both forms.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="path to the run config JSON")
    src.add_argument("--config-uri", help="gs:// (or local) URI of a staged run config JSON")
    verbs = p.add_mutually_exclusive_group()
    verbs.add_argument(
        "--dry-run", action="store_true", help="resolve + estimate fanout offline; touch no GCP"
    )
    verbs.add_argument(
        "--stage-only",
        action="store_true",
        help="stage artifacts to GCS + emit the runnable command + manifest; do not submit",
    )
    verbs.add_argument(
        "--emit-airflow",
        action="store_true",
        help="render this run's Airflow DAG to a local dag_<run_id>.py; touch no GCP",
    )
    verbs.add_argument(
        "--probe",
        action="store_true",
        help="registry-first reconciled status of this config's run; escalate incomplete/stale "
        "jobs to their runtime; touch no runtime if the run is already terminal",
    )
    verbs.add_argument(
        "--cancel",
        action="store_true",
        help="stop this config's in-flight jobs and finalize the registry to CANCELLED; without "
        "--force this only PREVIEWS the blast radius and stops nothing",
    )
    p.add_argument(
        "--job",
        help="with --probe/--cancel: narrow to one family "
        "(statistical/ml/deep_learning/native/ensemble)",
    )
    p.add_argument(
        "--reason",
        help="with --cancel --force: free-text reason recorded in the cancel audit trail",
    )
    p.add_argument(
        "--emit-out",
        help="where to write the emitted DAG (default: ./dag_<run_id>.py); implies --emit-airflow",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="acknowledge re-running an already-run config (shapes the exists-vs-new guidance)",
    )
    ns = p.parse_args(argv)

    cfg = load_config_uri(ns.config or ns.config_uri)
    if ns.emit_airflow or ns.emit_out:
        out = _emit_airflow(cfg, ns.config or ns.config_uri, out_path=ns.emit_out)
        _log.info("wrote Airflow DAG: %s", out)
        return
    if ns.stage_only:
        result = launch_plan.stage_run(cfg, force=ns.force)
        _log.info("staged: %s", result.run_id)
        return
    if ns.probe:
        from .probes.reconcile import probe_run

        report = probe_run(make_run_id(cfg), job=ns.job)
        _print_probe_report(report)
        return
    if ns.cancel:
        from .probes.cancel import cancel_run

        # --force is the cancel confirmation gate: without it, cancel_run only previews the blast
        # radius and stops nothing (the "never implicit" rule, §7.1).
        cancel_report = cancel_run(
            make_run_id(cfg), job=ns.job, confirm=ns.force, reason=ns.reason or ""
        )
        _print_cancel_report(cancel_report)
        return
    run_id = run(cfg, dry_run=ns.dry_run, force=ns.force)
    _log.info("%s: %s", "planned" if ns.dry_run else "submitted", run_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    _main()
