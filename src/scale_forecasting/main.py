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

Both Python runtimes are supported and dispatched by ``cfg.python_runtime``: ``"spark"`` launches a
Dataproc Serverless batch (:func:`~scale_forecasting.submit.submit_batch`), ``"ray"`` an autoscaling
Vertex Ray cluster (:func:`~scale_forecasting.ray_submit.submit_ray`) — either way on the worker
thread, in contributor mode, in parallel with the in-process BigQuery engine under one run_id.

Out of scope here (rejected with a clear pointer): ``spark_method="multi"``. multi fans out *N*
family batches, but :func:`run`'s parallelism drives exactly one Python-runtime future alongside the
in-process BigQuery engine — it can't own multi's N-batch fan-out. multi has its own single-run_id
orchestrator (:func:`~scale_forecasting.submit.submit_multi`, C3): use ``python -m
scale_forecasting.submit --engine multi``. ``multi`` is a Spark-only method, so the guard only
applies when ``python_runtime="spark"``.

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
    from .settings import Settings

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
    the one out-of-scope shape — but only when it would actually bite, i.e. when there *are*
    Python-runtime models to run on the Spark runtime:

    * ``python_runtime="spark"`` + ``spark_method="multi"`` → :class:`ConfigError` pointing at
      ``submit --engine multi`` (multi fans out one batch *per family*; this orchestrator drives a
      single Python-runtime future, not multi's N-batch fan-out — multi shares one run_id via its
      own orchestrator, :func:`submit.submit_multi`). ``multi`` is Spark-only, so the Ray runtime
      never trips it.

    Both ``python_runtime="spark"`` and ``python_runtime="ray"`` are supported (dispatched in
    :func:`run`). An all-BigQuery config is unaffected by the guard (the Python runtime is never
    used), so it plans and runs regardless of ``python_runtime`` / ``spark_method``.
    """
    run_id = make_run_id(cfg)
    python_models, bq_models = split_by_runtime(cfg)

    if python_models and cfg.python_runtime == "spark" and cfg.spark_method == "multi":
        raise ConfigError(
            "main.run cannot run spark_method='multi': multi fans out one batch per model "
            "family, but this orchestrator drives a single Python-runtime future in parallel "
            "with BigQuery, not multi's N-batch fan-out. Run it with its own single-run_id "
            "orchestrator, `python -m scale_forecasting.submit --engine multi`, or choose "
            "spark_method='explode'/'naive' to orchestrate it in parallel with BigQuery here."
        )

    return _RunPlan(
        run_id=run_id,
        python_models=python_models,
        bq_models=bq_models,
        spark_method=cfg.spark_method,
    )


def _launch_python_runtime(
    cfg: RunConfig, plan: _RunPlan, settings: Settings, spark: object | None = None
) -> None:
    """Run the Python-runtime models on the runtime ``cfg.python_runtime`` picks (contributor mode).

    The one dispatch point between the two Python runtimes, called on :func:`run`'s worker thread:
    ``"ray"`` → an autoscaling Ray cluster (:func:`~scale_forecasting.ray_submit.submit_ray`);
    otherwise → a Dataproc Serverless batch (:func:`~scale_forecasting.submit.submit_batch`) as the
    ``plan.spark_method`` engine. Both run ``plan.python_models`` with ``manage_header=False`` (this
    orchestrator owns the single shared header) and block until terminal, so the caller joins one
    future regardless of runtime. Kept a plain module function (not a lambda) so the worker thread's
    traceback names it and the import stays lazy (Ray/Spark extras load only for the chosen path).

    ``spark`` is an optional injected :class:`SparkSession`. When it is supplied **and**
    ``python_runtime == "spark"``, the Spark engine runs **in-process against that session**
    (``spark_explode``/``spark_naive`` with ``spark=…``) instead of submitting a remote Dataproc
    batch — the notebook / Spark Connect path. This is the same engine code the batch runs (the
    injectable-session seam), so no logic forks per environment (G1). ``spark`` is ignored for the
    Ray runtime and for a remote batch (the default when no session is passed).
    """
    if cfg.python_runtime == "ray":
        from .ray_submit import submit_ray

        submit_ray(
            cfg,
            models=plan.python_models,
            manage_header=False,
            settings=settings,
            wait=True,
        )
    elif spark is not None:
        # In-process Spark over an injected (Connect or local) session — no remote batch submit.
        # multi never reaches here (rejected by _plan under one run_id), so it's naive xor explode.
        from .engines import spark_explode, spark_naive

        engine = spark_naive if plan.spark_method == "naive" else spark_explode
        engine.run(
            cfg,
            models=plan.python_models,
            manage_header=False,
            settings=settings,
            spark=spark,
        )
    else:
        from .submit import submit_batch

        submit_batch(
            cfg,
            engine=plan.spark_method or "explode",
            models=plan.python_models,
            manage_header=False,
            settings=settings,
            wait=True,
        )


def run(
    cfg: RunConfig,
    *,
    dry_run: bool = False,
    spark: object | None = None,
    settings: Settings | None = None,
) -> str:
    """Execute one run: Spark + BigQuery-native in parallel under one run_id; return that run_id.

    Resolves the plan (:func:`_plan`, which rejects Spark ``multi``), then:

    * ``dry_run=True`` → log the run_id + :func:`~scale_forecasting.config.estimate_fanout` and
      return, touching no GCP. The offline "what would this schedule" path (DESIGN §11).
    * otherwise → resolve :class:`Settings`, ``ensure_tables`` + ``write_header`` (RUNNING, the one
      shared header), launch the remote Spark batch on a worker thread and run the BigQuery engine
      inline (both in contributor mode), join, then — when both engines succeeded and
      ``cfg.ensemble.enabled`` — run the ensembles (:func:`ensemble_run.run_ensembles`, which reads
      the just-written base predictions/OOF and scores each consensus onto the leaderboard), and
      finalize the header with the combined status + wall-clock ``runtime_seconds`` + ``bq_models``
      (+ the BQ engine's observed ``n_series`` when it ran). On any engine *or* ensemble failure the
      header is finalized FAILED before the error re-raises, so the run stays queryable and the CLI
      exits non-zero.

    ``spark`` is an optional injected :class:`SparkSession` (incl. a Spark Connect
    ``DataprocSparkSession``). When supplied and ``python_runtime == "spark"``, the Spark models run
    **in-process against that session** instead of a remote Dataproc batch — the notebook / Connect
    demo path — using the identical engine code (the injectable-session seam, G1). The default
    (``None``) keeps the remote-batch behavior every CLI/Composer caller relies on. The BigQuery
    engine still runs in parallel on the main thread under the one shared run_id.

    ``settings`` optionally injects a resolved :class:`Settings` (the GCP infra identity); the
    default (``None``) resolves it from the ``SF_*`` environment exactly as before, so every
    existing caller is unchanged. The SDK ``Forecaster`` uses this to thread an explicit identity
    instead of relying on process env.

    Idempotent: the config-pinned run_id + append-only/dedupe-on-read cell writes mean re-running
    the same config lands byte-identical rows (§3.4).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .config import estimate_fanout
    from .engines import bigquery_engine
    from .registry import bq
    from .settings import Settings

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

    settings = settings or Settings.resolve()
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

    # Launch the remote Python-runtime job on a worker thread (it blocks until terminal + stamps
    # telemetry in-thread), and run the in-process BigQuery engine on the main thread so the two
    # overlap. The runtime — Spark batch or autoscaling Vertex Ray cluster — is chosen by
    # cfg.python_runtime; both take the same contributor-mode contract (models subset + shared
    # header owned here), so Ray ∥ BigQuery works under one run_id exactly like Spark ∥ BigQuery.
    with ThreadPoolExecutor(max_workers=1) as pool:
        python_future = None
        if plan.python_models:
            python_future = pool.submit(_launch_python_runtime, cfg, plan, settings, spark)
        if plan.bq_models:
            try:
                bq_outcome = bigquery_engine.run(
                    cfg, plan.bq_models, manage_header=False, settings=settings
                )
            except Exception as exc:  # noqa: BLE001 - captured, header finalized below, re-raised
                bq_error = exc
        if python_future is not None:
            try:
                python_future.result()
            except Exception as exc:  # noqa: BLE001 - captured, header finalized below, re-raised
                spark_error = exc

    # Ensembles run only once both engines have produced their base predictions under this run_id
    # (they read forecast_predictions / backtest_oof), so this is sequenced strictly after the join
    # and skipped when an engine failed. A failure here is captured like an engine error and flips
    # the shared header FAILED — the ensembles are part of the run's success contract.
    ensemble_error: BaseException | None = None
    if spark_error is None and bq_error is None and cfg.ensemble.enabled:
        from .ensemble_run import run_ensembles

        try:
            run_ensembles(cfg, run_id, settings=settings)
        except Exception as exc:  # noqa: BLE001 - captured, header finalized below, re-raised
            ensemble_error = exc

    runtime_seconds = time.perf_counter() - started
    ok = spark_error is None and bq_error is None and ensemble_error is None
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
        raise spark_error or bq_error or ensemble_error  # type: ignore[misc]
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
