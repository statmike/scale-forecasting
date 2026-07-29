"""Orchestrate one run end-to-end — Spark + BigQuery-native in parallel, one run_id (CONTRACTS §6).

This is the top of the run spine (BUILD Arc B). A single config can mix Python-runtime models (run
by the Spark engine as per-cell fan-out) and BigQuery-native models (run as SQL inside BigQuery);
:func:`run` executes **both runtimes in parallel under one shared ``run_id`` and one
``run_registry`` header row**, so a native model and a Spark model land in the same run and are
directly comparable on ``v_model_leaderboard`` (the DESIGN §3.3 "wall-clock ≈ max(python, bq), not
sum" thesis).

Two invariants make the single-run parallelism work (see :func:`~scale_forecasting.registry.ids`
and the Arc B engine seam):

1. **One run_id from the full config.** ``make_run_id`` is a pure digest over the *whole* config
   incl. ``cfg.models``; both engines receive the full ``cfg`` so they derive the same id. Each is
   handed only its own executed subset (``python_models`` / ``bq_models``) via the ``models``
   argument, so the BigQuery-native models never become Spark cells (which would raise
   ``NotImplementedError`` in ``worker.run_cell``) and vice-versa.
2. **One header owner.** :func:`run` alone writes the header (RUNNING) up front and finalizes it;
   both engines run in **contributor mode** (``manage_header=False``), skipping the header lifecycle
   and only writing their cell rows. No two writers ever touch the header, so there is no UPDATE
   race — the only in-window header write is ``submit_batch``'s best-effort telemetry stamp, which
   completes inside the joined future before :func:`run`'s finalize.

**Parallelism.** The remote Spark batch is launched on a worker thread (``submit_batch(wait=True)``)
while the in-process BigQuery engine runs on the main thread; the BQ work (minutes, in-process)
overlaps the Spark provisioning floor. :func:`run` joins both, rolls the two outcomes into one
combined status (COMPLETED iff both green, else FAILED — finalized *before* re-raising so the run
stays queryable and the CLI exits non-zero), and returns the shared ``run_id``.

**Coarsening (documented).** A remote contributor batch can't return its run-level PARTIAL (some
cells errored) to the orchestrator, so a SUCCEEDED batch is reported COMPLETED; per-model failure
stays visible on ``v_model_leaderboard`` (a failed model → NULL metric AVGs).

Out of scope here (rejected with a clear pointer): ``python_runtime="ray"`` (unbuilt B4 stub) and
``spark_method="multi"`` (inherently multi-run — each family child gets its own run_id, so it can't
share one header; use ``python -m scale_forecasting.submit --engine multi``).

Public surface: ``run(cfg, *, dry_run=False) -> run_id`` and
``python -m scale_forecasting.main --config ... [--dry-run]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import ConfigError, get_logger
from .registry.ids import make_run_id
from .router import split_by_runtime

if TYPE_CHECKING:
    from .config import RunConfig

_log = get_logger(__name__)


@dataclass(frozen=True)
class _RunPlan:
    """The offline-resolvable shape of a run: its id and the per-runtime executed subsets.

    Pure product of the config (:func:`_plan`) — no GCP. ``spark_method`` is the engine the Spark
    batch runs as (``None`` when there are no Python models to run).
    """

    run_id: str
    python_models: list[str]
    bq_models: list[str]
    spark_method: str | None


def _plan(cfg: RunConfig) -> _RunPlan:
    """Resolve the run_id + per-runtime model split, rejecting shapes this orchestrator can't run.

    Pure and offline: computes ``make_run_id(cfg)`` and :func:`router.split_by_runtime`, then guards
    the two out-of-scope shapes — but only when they would actually bite, i.e. when there *are*
    Python-runtime models to run:

    * ``python_runtime="ray"`` → :class:`ConfigError` (the Ray engine is an unbuilt B4 stub).
    * ``spark_method="multi"`` → :class:`ConfigError` pointing at ``submit --engine multi`` (multi
      fans out one batch *per family*, each with its own run_id, so it can't share a single header).

    An all-BigQuery config is unaffected by either (the Python runtime is never used), so it plans
    and runs regardless of ``python_runtime`` / ``spark_method``.
    """
    run_id = make_run_id(cfg)
    python_models, bq_models = split_by_runtime(cfg)

    if python_models:
        if cfg.python_runtime == "ray":
            raise ConfigError(
                "main.run does not support python_runtime='ray' yet (the Ray engine is an unbuilt "
                "B4 stub); use python_runtime='spark' or a BigQuery-native-only config."
            )
        if cfg.spark_method == "multi":
            raise ConfigError(
                "main.run cannot run spark_method='multi' under one run_id: multi fans out one "
                "batch per model family, each with its own run_id and header. Run it standalone "
                "with `python -m scale_forecasting.submit --engine multi`, or choose "
                "spark_method='explode'/'naive' to orchestrate it in parallel with BigQuery here."
            )

    return _RunPlan(
        run_id=run_id,
        python_models=python_models,
        bq_models=bq_models,
        spark_method=cfg.spark_method,
    )


def run(cfg: RunConfig, *, dry_run: bool = False) -> str:
    """Execute one run: Spark + BigQuery-native in parallel under one run_id; return that run_id.

    Resolves the plan (:func:`_plan`, which rejects ray/multi), then:

    * ``dry_run=True`` → log the run_id + :func:`~scale_forecasting.config.estimate_fanout` and
      return, touching no GCP. The offline "what would this schedule" path (DESIGN §11).
    * otherwise → resolve :class:`Settings`, ``ensure_tables`` + ``write_header`` (RUNNING, the one
      shared header), launch the remote Spark batch on a worker thread and run the BigQuery engine
      inline (both in contributor mode), join, and finalize the header with the combined status +
      wall-clock ``runtime_seconds`` + ``bq_models`` (+ the BQ engine's observed ``n_series`` when
      it ran). On any engine failure the header is finalized FAILED before the error re-raises, so
      the run stays queryable and the CLI exits non-zero.

    Idempotent: the config-pinned run_id + append-only/dedupe-on-read cell writes mean re-running
    the same config lands byte-identical rows (§3.4).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .config import estimate_fanout
    from .engines import bigquery_engine
    from .registry import bq
    from .settings import Settings
    from .submit import submit_batch

    plan = _plan(cfg)
    run_id = plan.run_id

    if dry_run:
        fanout = estimate_fanout(cfg)
        _log.info(
            "dry-run %s: python=%s bq=%s fanout=%s",
            run_id,
            plan.python_models,
            plan.bq_models,
            fanout,
        )
        return run_id

    settings = Settings.resolve()
    _log.info(
        "run %s start: python=%s (%s) bq=%s",
        run_id,
        plan.python_models,
        plan.spark_method,
        plan.bq_models,
    )

    # One header owner: write RUNNING once, then finalize after both engines join. Both engines run
    # with manage_header=False so nothing else touches this row.
    bq.ensure_tables(cfg, settings=settings)
    bq.write_header(cfg, run_id, settings=settings)

    started = time.perf_counter()
    bq_outcome = None
    spark_error: BaseException | None = None
    bq_error: BaseException | None = None

    # Launch the remote Spark batch on a worker thread (it blocks until terminal + stamps telemetry
    # in-thread), and run the in-process BigQuery engine on the main thread so the two overlap.
    with ThreadPoolExecutor(max_workers=1) as pool:
        spark_future = None
        if plan.python_models:
            spark_future = pool.submit(
                submit_batch,
                cfg,
                engine=plan.spark_method or "explode",
                models=plan.python_models,
                manage_header=False,
                settings=settings,
                wait=True,
            )
        if plan.bq_models:
            try:
                bq_outcome = bigquery_engine.run(
                    cfg, plan.bq_models, manage_header=False, settings=settings
                )
            except Exception as exc:  # noqa: BLE001 - captured, header finalized below, re-raised
                bq_error = exc
        if spark_future is not None:
            try:
                spark_future.result()
            except Exception as exc:  # noqa: BLE001 - captured, header finalized below, re-raised
                spark_error = exc

    runtime_seconds = time.perf_counter() - started
    ok = spark_error is None and bq_error is None
    status = "COMPLETED" if ok else "FAILED"

    fields: dict[str, object] = {
        "status": status,
        "runtime_seconds": runtime_seconds,
        "bq_models": plan.bq_models,
    }
    if bq_outcome is not None:
        fields["n_series"] = bq_outcome.n_series
    bq.update_header(run_id, settings=settings, **fields)

    if not ok:
        # Re-raise the first failure so the CLI exits non-zero; the header already records FAILED.
        raise spark_error or bq_error  # type: ignore[misc]
    _log.info("run %s done: status=%s runtime=%.1fs", run_id, status, runtime_seconds)
    return run_id


def _main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m scale_forecasting.main --config run.json [--dry-run]``."""
    import argparse

    from .config import load_config

    p = argparse.ArgumentParser(prog="main", description="Run a forecast (Spark + BigQuery).")
    p.add_argument("--config", required=True, help="path to the run config JSON")
    p.add_argument(
        "--dry-run", action="store_true", help="resolve + estimate fanout offline; touch no GCP"
    )
    ns = p.parse_args(argv)

    cfg = load_config(ns.config)
    run_id = run(cfg, dry_run=ns.dry_run)
    _log.info("%s: %s", "planned" if ns.dry_run else "submitted", run_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    _main()
