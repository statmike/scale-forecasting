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

Out of scope here (rejected with a clear pointer): ``spark_method="multi"``. multi fans out *N*
family batches, but `run`'s parallelism drives exactly one Python-runtime future alongside the
in-process BigQuery engine — it can't own multi's N-batch fan-out. multi has its own single-run_id
orchestrator (`submit_multi`): use ``python -m
scale_forecasting.submit --engine multi``. ``multi`` is a Spark-only method, so the guard only
applies when ``python_runtime="spark"``.

Public surface: ``run(cfg, *, dry_run=False) -> run_id``, the offline ``plan_run(cfg) ->
LaunchPlan`` (id + fanout + launch-command templates), ``stage_run(cfg) -> LaunchPlan`` (upload
artifacts + runnable commands + reproducibility manifest, no submit), and
``python -m scale_forecasting.main --config ... [--dry-run | --stage-only]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import ConfigError, get_logger
from .registry.ids import make_run_id
from .router import split_by_runtime

if TYPE_CHECKING:
    from .commands import LaunchCommands
    from .config import Fanout, RunConfig
    from .settings import Settings

_log = get_logger(__name__)


@dataclass(frozen=True)
class Idempotency:
    """Whether a run's config has already been submitted (the pre-submit existence check).

    ``run_id`` is a pure digest of the config, so an unchanged config re-derives the same id — this
    verdict says whether that id already has a header in the registry. ``checked`` is ``False`` when
    the registry couldn't be consulted (no ``SF_*`` env, or no reachable registry): the verdict is
    simply *unknown*, not "new". ``prior_status`` is the existing run's status when ``exists`` (e.g.
    ``COMPLETED``), else ``None``.
    """

    checked: bool
    exists: bool
    prior_status: str | None


@dataclass(frozen=True)
class LaunchPlan:
    """A run resolved to launch-ready form: id, runtime split, fanout, and the launch commands.

    Produced offline by `plan_run` (``staged=False`` — URIs are the *templates* where artifacts will
    land; no GCS is touched) or by `stage_run` (``staged=True`` — artifacts uploaded, URIs real and
    the commands runnable). ``commands`` maps a tier name (``"main"``/``"spark"``/``"ray"``) to its
    `LaunchCommands`; it is ``None`` only when the GCP infra identity can't be resolved offline (no
    ``SF_*`` env), in which case the run_id + fanout + runtime split are still returned.

    ``idempotency`` is the exists-vs-new verdict for this config's id; ``force`` records that the
    caller intends to re-run an already-run config (it only shapes the emitted guidance — a re-run
    is idempotent regardless, via dedupe-on-read under the shared ``run_id``).
    """

    run_id: str
    python_runtime: str
    python_models: list[str]
    bq_models: list[str]
    spark_method: str | None
    fanout: Fanout
    staged: bool
    config_uri: str | None
    commands: dict[str, LaunchCommands] | None
    idempotency: Idempotency
    force: bool


@dataclass(frozen=True)
class _RunPlan:
    """The offline-resolvable shape of a run: its id and the per-runtime executed subsets.

    Pure product of the config (`_plan`) — no GCP. ``spark_method`` is the engine the Spark
    batch runs as (``None`` when there are no Python models to run).
    """

    run_id: str
    python_models: list[str]
    bq_models: list[str]
    spark_method: str | None


def _plan(cfg: RunConfig) -> _RunPlan:
    """Resolve the run_id + per-runtime model split, rejecting shapes this orchestrator can't run.

    Pure and offline: computes ``make_run_id(cfg)`` and `router.split_by_runtime`, then guards
    the one out-of-scope shape — but only when it would actually bite, i.e. when there *are*
    Python-runtime models to run on the Spark runtime:

    * ``python_runtime="spark"`` + ``spark_method="multi"`` → `ConfigError` pointing at
      ``submit --engine multi`` (multi fans out one batch *per family*; this orchestrator drives a
      single Python-runtime future, not multi's N-batch fan-out — multi shares one run_id via its
      own orchestrator, `submit.submit_multi`). ``multi`` is Spark-only, so the Ray runtime
      never trips it.

    Both ``python_runtime="spark"`` and ``python_runtime="ray"`` are supported (dispatched in
    `run`). An all-BigQuery config is unaffected by the guard (the Python runtime is never
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

    The one dispatch point between the two Python runtimes, called on `run`'s worker thread: it
    looks up the `RuntimeSubmitter` registered for ``cfg.python_runtime`` (`get_submitter`) and
    hands it ``plan.python_models`` with ``manage_header=False`` (this orchestrator owns the single
    shared header). The submitter blocks until terminal, so the caller joins one future regardless
    of runtime — Ray submits to an autoscaling cluster, Spark runs in-process against an injected
    session or submits a Dataproc batch. Kept a plain module function (not a lambda) so the worker
    thread's traceback names it, and the submitter's imports stay lazy (Ray/Spark extras load only
    for the chosen path).

    ``spark`` is an optional injected `SparkSession`, passed through to the submitter. The Spark
    submitter, given one, runs **in-process against that session** (the same engine code the batch
    runs — the injectable-session seam) instead of submitting a remote Dataproc batch; other
    runtimes ignore it.
    """
    from .submitters import get_submitter

    get_submitter(cfg.python_runtime).launch(
        cfg,
        models=plan.python_models,
        manage_header=False,
        settings=settings,
        spark=spark,
    )


def run(
    cfg: RunConfig,
    *,
    dry_run: bool = False,
    spark: object | None = None,
    settings: Settings | None = None,
    force: bool = False,
) -> str:
    """Execute one run: Spark + BigQuery-native in parallel under one run_id; return that run_id.

    Resolves the plan (`_plan`, which rejects Spark ``multi``), then:

    * ``dry_run=True`` → delegate to `plan_run` (the offline "what would this schedule" path):
      resolve the run_id + `estimate_fanout` + emit the launch-command templates, touching no GCP,
      and return the run_id.
    * otherwise → resolve `Settings`, ``ensure_tables`` + ``write_header`` (RUNNING, the one
      shared header), launch the remote Spark batch on a worker thread and run the BigQuery engine
      inline (both in contributor mode), join, then — when both engines succeeded and
      ``cfg.ensemble.enabled`` — run the ensembles (`ensemble_run.run_ensembles`, which reads
      the just-written base predictions/OOF and scores each consensus onto the leaderboard), and
      finalize the header with the combined status + wall-clock ``runtime_seconds`` + ``bq_models``
      (+ the BQ engine's observed ``n_series`` when it ran). On any engine *or* ensemble failure the
      header is finalized FAILED before the error re-raises, so the run stays queryable and the CLI
      exits non-zero.

    ``spark`` is an optional injected `SparkSession` (incl. a Spark Connect
    ``DataprocSparkSession``). When supplied and ``python_runtime == "spark"``, the Spark models run
    **in-process against that session** instead of a remote Dataproc batch — the notebook / Connect
    demo path — using the identical engine code (the injectable-session seam). The default
    (``None``) keeps the remote-batch behavior every CLI/Composer caller relies on. The BigQuery
    engine still runs in parallel on the main thread under the one shared run_id.

    ``settings`` optionally injects a resolved `Settings` (the GCP infra identity); the
    default (``None``) resolves it from the ``SF_*`` environment exactly as before, so every
    existing caller is unchanged. The SDK ``Forecaster`` uses this to thread an explicit identity
    instead of relying on process env.

    ``force`` acknowledges re-running an already-run config; it shapes only the ``dry_run`` plan's
    re-run guidance (a submit is idempotent regardless — see below).

    Idempotent: the config-pinned run_id + append-only/dedupe-on-read cell writes mean re-running
    the same config lands byte-identical rows.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .engines import bigquery_engine
    from .registry import bq
    from .settings import Settings

    if dry_run:
        # Single-source the offline plan: plan_run resolves the id + fanout + runtime split, reports
        # the exists-vs-new verdict, and emits the launch-command templates. It returns the run_id,
        # so this contract is unchanged.
        return plan_run(cfg, settings=settings, force=force).run_id

    plan = _plan(cfg)
    run_id = plan.run_id

    settings = settings or Settings.resolve()
    _log.info(
        "run %s start: python=%s (%s) bq=%s",
        run_id,
        plan.python_models,
        plan.spark_method,
        plan.bq_models,
    )

    bq_outcome = None
    spark_error: BaseException | None = None
    bq_error: BaseException | None = None
    ensemble_error: BaseException | None = None

    # One header owner: run_header writes RUNNING on entry and finalizes once, after both engines
    # join, with the combined status computed below. Both engines run with manage_header=False so
    # nothing else touches this row. Track errors are captured (not raised through the block) so the
    # finalize records the right status; the first one is re-raised after, for a non-zero exit.
    with bq.run_header(cfg, run_id, settings=settings, manage=True) as hdr:
        # Launch the remote Python-runtime job on a worker thread (it blocks until terminal + stamps
        # telemetry in-thread), and run the in-process BigQuery engine on the main thread so the two
        # overlap. The runtime — Spark batch or autoscaling Vertex Ray cluster — is chosen by
        # cfg.python_runtime; both take the same contributor-mode contract (models subset + shared
        # header owned here), so Ray ∥ BigQuery works under one run_id like Spark ∥ BigQuery.
        with ThreadPoolExecutor(max_workers=1) as pool:
            python_future = None
            if plan.python_models:
                python_future = pool.submit(_launch_python_runtime, cfg, plan, settings, spark)
            if plan.bq_models:
                try:
                    bq_outcome = bigquery_engine.run(
                        cfg, plan.bq_models, manage_header=False, settings=settings
                    )
                except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                    bq_error = exc
            if python_future is not None:
                try:
                    python_future.result()
                except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                    spark_error = exc

        # Ensembles run only once both engines have produced their base predictions under this
        # run_id (they read forecast_predictions / backtest_oof), so this is sequenced strictly
        # after the join and skipped when an engine failed. A failure here is captured like an
        # engine error — the ensembles are part of the run's success contract.
        if spark_error is None and bq_error is None and cfg.ensemble.enabled:
            from .ensemble_run import run_ensembles

            try:
                run_ensembles(cfg, run_id, settings=settings)
            except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                ensemble_error = exc

        # Combined status across the engine tracks that ran: COMPLETED iff all succeeded, FAILED iff
        # all failed, PARTIAL when some but not all did (a mixed BigQuery ∥ Spark/Ray outcome — the
        # base forecasts of the surviving engine are still usable). An ensemble failure on top of
        # green engines fails the run: it didn't deliver the full requested output.
        status = _combined_status(plan, spark_error, bq_error, ensemble_error)
        fields: dict[str, object] = {"bq_models": plan.bq_models}
        if bq_outcome is not None:
            fields["n_series"] = bq_outcome.n_series
        hdr.finalize(status=status, **fields)

    first_error = spark_error or bq_error or ensemble_error
    if first_error is not None:
        # Re-raise the first failure so the CLI exits non-zero; the header already records the
        # combined status (FAILED or PARTIAL).
        raise first_error
    _log.info("run %s done: status=%s", run_id, status)
    return run_id


def _combined_status(
    plan: _RunPlan,
    spark_error: BaseException | None,
    bq_error: BaseException | None,
    ensemble_error: BaseException | None,
) -> str:
    """Roll the per-track outcomes into one run status (pure).

    Over the engine tracks that actually ran (Python-runtime and/or BigQuery-native): every track
    green → ``COMPLETED``; every track failed → ``FAILED``; a mix → ``PARTIAL``. An ensemble
    failure downgrades an otherwise-``COMPLETED`` run to ``FAILED`` (the requested output is
    incomplete); it never masks an engine ``PARTIAL``/``FAILED``.
    """
    track_ok: list[bool] = []
    if plan.python_models:
        track_ok.append(spark_error is None)
    if plan.bq_models:
        track_ok.append(bq_error is None)

    n_ok = sum(track_ok)
    if not track_ok or n_ok == len(track_ok):
        engine_status = "COMPLETED"
    elif n_ok == 0:
        engine_status = "FAILED"
    else:
        engine_status = "PARTIAL"

    if engine_status == "COMPLETED" and ensemble_error is not None:
        return "FAILED"
    return engine_status


def _check_idempotency(run_id: str, settings: Settings) -> Idempotency:
    """Best-effort pre-submit existence check: has this exact config already run?

    Queries the registry for ``run_id`` (`registry.bq.header_status`). Any failure — no registry
    table yet, no reachable BigQuery — degrades to ``checked=False`` (unknown), so a plain offline
    dry-run still returns a plan. The check is advisory: it warns before an accidental duplicate run
    but never blocks one (a re-run is idempotent via dedupe-on-read).
    """
    from .registry import bq

    try:
        status = bq.header_status(run_id, settings=settings)
    except Exception:  # noqa: BLE001 - advisory check; unknown on any failure, never fatal
        return Idempotency(checked=False, exists=False, prior_status=None)
    return Idempotency(checked=True, exists=status is not None, prior_status=status)


def _resolve_infra(cfg: RunConfig, infra: object | None) -> object:
    """The runtime's infra identity: the injected ``infra``, else resolved from the ``SF_*`` env.

    Spark resolves a `BatchInfra`, Ray a `RayInfra`; both carry the ``code_bucket`` the config
    stages to. Raises `ConfigError` (from ``resolve``) when the env is unset — plan emission is
    best-effort, so `plan_run` catches that and returns a plan without commands.
    """
    if infra is not None:
        return infra
    if cfg.python_runtime == "ray":
        from .ray_submit import RayInfra

        return RayInfra.resolve()
    from .submit import BatchInfra

    return BatchInfra.resolve()


def _template_uris(
    cfg: RunConfig, plan: _RunPlan, code_bucket: str
) -> tuple[str, str | None, str | None]:
    """The ``gs://`` URIs a run's artifacts *will* land at (pure — mirrors the staging scheme).

    Returns ``(config_uri, package_uri, launcher_uri)``. The config URI always exists; the Spark
    package/launcher URIs are set only for a Spark run with Python models (Ray delivers code via its
    ``runtime_env`` working dir, not a staged zip). The package name carries the code hash from
    `code_delivery.build_package_zip` — a deterministic local build, no network — so the template is
    byte-faithful to what `submit._stage_code` would upload.
    """
    config_uri = f"gs://{code_bucket}/runs/{plan.run_id}.json"
    if cfg.python_runtime == "ray" or not plan.python_models:
        return config_uri, None, None
    from .code_delivery import build_package_zip

    _, code_hash = build_package_zip()
    package_uri = f"gs://{code_bucket}/runs/scale_forecasting-{code_hash}.zip"
    launcher_uri = f"gs://{code_bucket}/runs/spark_main.py"
    return config_uri, package_uri, launcher_uri


def _assemble_commands(
    cfg: RunConfig,
    plan: _RunPlan,
    settings: Settings,
    infra: object,
    *,
    config_uri: str,
    package_uri: str | None,
    launcher_uri: str | None,
) -> dict[str, LaunchCommands]:
    """Build the launch commands for a run, keyed by tier (pure — assembles strings only).

    Always emits ``"main"`` — ``python -m scale_forecasting.main --config-uri …``, the orchestrator
    that reproduces the *full* run (both engines under one run_id). When there are Python-runtime
    models it adds the per-runtime tier: ``"ray"`` (universal only) or ``"spark"`` (native
    ``gcloud`` + universal). The Spark tier restricts to ``--models`` **only** for a mixed run (so
    the batch runs just its subset while BigQuery runs the rest); a Python-only config emits no
    ``--models`` so the standalone batch runs the whole config under its own header.
    """
    from .commands import build_main_command, build_ray_commands, build_spark_commands

    commands: dict[str, LaunchCommands] = {"main": build_main_command(config_uri)}
    if not plan.python_models:
        return commands
    if cfg.python_runtime == "ray":
        commands["ray"] = build_ray_commands(
            config_uri=config_uri, cluster_name=cfg.compute.ray_cluster_name
        )
        return commands

    from .submit import BatchInfra, _batch_id

    assert isinstance(infra, BatchInfra)  # spark runtime → BatchInfra (resolved above)
    engine = plan.spark_method or "explode"
    models_arg = plan.python_models if plan.bq_models else None
    commands["spark"] = build_spark_commands(
        settings=settings,
        infra=infra,
        engine=engine,
        batch_id=_batch_id(plan.run_id, engine),
        package_uri=package_uri or "",
        launcher_uri=launcher_uri or "",
        config_uri=config_uri,
        models=models_arg,
        manage_header=True,
    )
    return commands


def _emit_idempotency(result: LaunchPlan) -> None:
    """Log the exists-vs-new verdict and, when re-running, the ``--force`` guidance."""
    idem = result.idempotency
    if not idem.checked:
        return  # registry not consulted (offline / unreachable) — verdict unknown, say nothing
    if not idem.exists:
        _log.info("  new run — this config has not run before")
        return
    if result.force:
        _log.info(
            "  re-run (--force): this config already ran (%s); it appends to the same run_id "
            "(idempotent, dedupe-on-read)",
            idem.prior_status,
        )
    else:
        _log.warning(
            "  already ran: this config ran before (%s). A re-run is idempotent (dedupe-on-read "
            "under the same run_id); pass --force to acknowledge re-running it.",
            idem.prior_status,
        )


def _emit_plan(result: LaunchPlan) -> None:
    """Log the resolved plan + each launch-command tier (the dry-run/stage-only emission)."""
    verb = "stage" if result.staged else "dry-run"
    _log.info(
        "%s %s: runtime=%s python=%s bq=%s fanout=%s",
        verb,
        result.run_id,
        result.python_runtime,
        result.python_models,
        result.bq_models,
        result.fanout,
    )
    _emit_idempotency(result)
    if result.commands is None:
        _log.info(
            "%s %s: infra unresolved (no SF_* env) — commands not emitted", verb, result.run_id
        )
        return
    for name, lc in result.commands.items():
        _log.info("  [%s] %s", name, lc.universal)
        if lc.native is not None:
            _log.info("  [%s:native] %s", name, lc.native)


def plan_run(
    cfg: RunConfig,
    *,
    settings: Settings | None = None,
    infra: object | None = None,
    force: bool = False,
) -> LaunchPlan:
    """Resolve a run offline to its id, fanout, runtime split, and launch-command *templates*.

    The "plan" verb: pure and GCP-free. Computes `_plan` (rejecting Spark ``multi`` up front, so a
    bad shape fails fast even in a dry run) and `estimate_fanout`, then — best-effort — resolves the
    infra identity, consults the registry for the exists-vs-new verdict (`_check_idempotency`), and
    builds the two-tier launch commands with the URIs the artifacts *will* land at (nothing is
    uploaded). When the infra can't be resolved offline (no ``SF_*`` env), the commands are omitted
    (``commands=None``) and the verdict is *unknown*, but the id/fanout/split are still returned, so
    a plain ``--dry-run`` works with no environment. ``force`` shapes only the emitted re-run
    guidance. Emits the plan to the log and returns it.
    """
    from .config import estimate_fanout

    plan = _plan(cfg)
    fanout = estimate_fanout(cfg)
    config_uri: str | None = None
    commands: dict[str, LaunchCommands] | None = None
    idempotency = Idempotency(checked=False, exists=False, prior_status=None)
    try:
        settings = settings or _resolve_settings()
        idempotency = _check_idempotency(plan.run_id, settings)
        resolved_infra = _resolve_infra(cfg, infra)
        code_bucket = resolved_infra.code_bucket  # type: ignore[attr-defined]
        config_uri, package_uri, launcher_uri = _template_uris(cfg, plan, code_bucket)
        commands = _assemble_commands(
            cfg,
            plan,
            settings,
            resolved_infra,
            config_uri=config_uri,
            package_uri=package_uri,
            launcher_uri=launcher_uri,
        )
    except ConfigError:
        # No SF_* env (or no injected settings/infra): return the plan without commands.
        config_uri = None
        commands = None
    result = LaunchPlan(
        run_id=plan.run_id,
        python_runtime=cfg.python_runtime,
        python_models=plan.python_models,
        bq_models=plan.bq_models,
        spark_method=plan.spark_method,
        fanout=fanout,
        staged=False,
        config_uri=config_uri,
        commands=commands,
        idempotency=idempotency,
        force=force,
    )
    _emit_plan(result)
    return result


def _manifest_dict(result: LaunchPlan, *, created_at: str) -> dict[str, object]:
    """The reproducibility-manifest payload for a staged run (pure — ``created_at`` is caller-set).

    Records the config digest, fan-out, runtime split, both command tiers, and the staged config
    URI — everything needed to answer "what command produced run X?". The timestamp is passed in
    (not read here) so this stays a pure function with no wall-clock.
    """
    commands = {
        name: {"runtime": lc.runtime, "universal": lc.universal, "native": lc.native}
        for name, lc in (result.commands or {}).items()
    }
    return {
        "run_id": result.run_id,
        "created_at": created_at,
        "python_runtime": result.python_runtime,
        "spark_method": result.spark_method,
        "python_models": result.python_models,
        "bq_models": result.bq_models,
        "fanout": {
            "n_series": result.fanout.n_series,
            "n_models": result.fanout.n_models,
            "n_folds": result.fanout.n_folds,
            "n_cells": result.fanout.n_cells,
        },
        "config_uri": result.config_uri,
        "commands": commands,
        "force": result.force,
        "idempotency": {
            "checked": result.idempotency.checked,
            "exists": result.idempotency.exists,
            "prior_status": result.idempotency.prior_status,
        },
    }


def stage_run(
    cfg: RunConfig,
    *,
    settings: Settings | None = None,
    infra: object | None = None,
    force: bool = False,
) -> LaunchPlan:
    """Stage a run's artifacts to GCS and return the *runnable* launch commands — no submit.

    The "stage" verb: uploads the config (and, for Spark, the code zip + launcher shim) to the code
    bucket, builds the launch commands against those **real** URIs (so they run as-is from any ADC
    box), and writes the reproducibility manifest ``runs/<run_id>.plan.json`` next to the config.
    Unlike `plan_run`, the infra identity is required — staging touches GCS — so a missing ``SF_*``
    env raises rather than degrading; the exists-vs-new verdict (`_check_idempotency`) is therefore
    always resolved. ``force`` shapes only the emitted re-run guidance. Returns the `LaunchPlan`
    with ``staged=True``.
    """
    from datetime import UTC, datetime

    from .config import estimate_fanout
    from .staging import stage_config, stage_manifest

    plan = _plan(cfg)
    fanout = estimate_fanout(cfg)
    settings = settings or _resolve_settings()
    idempotency = _check_idempotency(plan.run_id, settings)
    resolved_infra = _resolve_infra(cfg, infra)
    code_bucket: str = resolved_infra.code_bucket  # type: ignore[attr-defined]

    config_uri = stage_config(cfg, plan.run_id, code_bucket)
    package_uri: str | None = None
    launcher_uri: str | None = None
    if cfg.python_runtime != "ray" and plan.python_models:
        from .submit import BatchInfra, _stage_code

        assert isinstance(resolved_infra, BatchInfra)  # spark runtime → BatchInfra
        package_uri, launcher_uri = _stage_code(resolved_infra)

    commands = _assemble_commands(
        cfg,
        plan,
        settings,
        resolved_infra,
        config_uri=config_uri,
        package_uri=package_uri,
        launcher_uri=launcher_uri,
    )
    result = LaunchPlan(
        run_id=plan.run_id,
        python_runtime=cfg.python_runtime,
        python_models=plan.python_models,
        bq_models=plan.bq_models,
        spark_method=plan.spark_method,
        fanout=fanout,
        staged=True,
        config_uri=config_uri,
        commands=commands,
        idempotency=idempotency,
        force=force,
    )
    manifest_uri = stage_manifest(
        _manifest_dict(result, created_at=datetime.now(UTC).isoformat()), plan.run_id, code_bucket
    )
    _log.info("wrote run manifest: %s", manifest_uri)
    _emit_plan(result)
    return result


def _resolve_settings() -> Settings:
    """Resolve `Settings` from the ``SF_*`` env (raises `ConfigError` when unset)."""
    from .settings import Settings

    return Settings.resolve()


def _main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m scale_forecasting.main --config run.json [--dry-run | --stage-only]``."""
    import argparse

    from .config import load_config

    p = argparse.ArgumentParser(prog="main", description="Run a forecast (Spark + BigQuery).")
    p.add_argument("--config", required=True, help="path to the run config JSON")
    verbs = p.add_mutually_exclusive_group()
    verbs.add_argument(
        "--dry-run", action="store_true", help="resolve + estimate fanout offline; touch no GCP"
    )
    verbs.add_argument(
        "--stage-only",
        action="store_true",
        help="stage artifacts to GCS + emit the runnable command + manifest; do not submit",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="acknowledge re-running an already-run config (shapes the exists-vs-new guidance)",
    )
    ns = p.parse_args(argv)

    cfg = load_config(ns.config)
    if ns.stage_only:
        result = stage_run(cfg, force=ns.force)
        _log.info("staged: %s", result.run_id)
        return
    run_id = run(cfg, dry_run=ns.dry_run, force=ns.force)
    _log.info("%s: %s", "planned" if ns.dry_run else "submitted", run_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    _main()
