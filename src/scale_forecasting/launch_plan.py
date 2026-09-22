"""Resolve a config into a launch plan, and stage it — everything before anything launches.

The offline half of a run. `plan_run` answers "what *would* this config schedule, and has it run
already" without touching a thing; `stage_run` goes one step further and uploads the artifacts a
remote launch needs (the config, and for Spark the code zip), then writes a reproducibility
manifest — and still submits nothing. Neither ever starts a job; that is `main.run` and
`job_launch`.

Two things make the plan trustworthy rather than merely informative:

* **Same id as the real run.** ``run_id`` is a pure digest of the whole config, so a plan resolves
  the id a live run would use. That is what lets the CLI's ``--dry-run`` report an *existence
  verdict* (`Idempotency`) — this exact config has already run, and here is its status — and what
  makes `stage_run`'s uploads land at the paths the launch will read.
* **The source is pinned, but not into the digest.** `lock_profile_source` rewrites
  ``profile.source: "auto"`` to the concrete run it resolves to, so a staged config reproduces its
  original fleet instead of re-searching. `registry.ids` then excludes that field from the digest:
  it is resolved, not authored, and a config whose identity depended on what had been run before it
  never converged. Two runs a week apart share a run_id and dedupe on read; their manifests and
  sizing provenance record the different evidence each was sized from.

`LaunchPlan` is the one return type for both, distinguished by ``staged``: `plan_run` returns URI
*templates* (where artifacts would go) and `stage_run` returns the real ones (where they now are),
along with commands that are actually runnable. The command strings are built by `commands` from
the resolved deployment envelope, so what is printed is what a launch would execute — the two-tier
emit (a portable ``python -m`` line, plus the platform-native ``gcloud`` form).

Both take an optional `Settings`; without a reachable ``SF_*`` environment `plan_run` degrades
rather than fails (a plan with an unknown verdict and no commands), because resolving a config is
useful offline. `stage_run` requires it — staging touches GCS.

``feasibility_report`` is the deliberate exception to "the plan is offline": it reads one
aggregation off the source panel, because how many backtest folds each series actually achieves is
a fact about the data and no amount of config inspection will produce it. It is opt-in
(``--feasibility``), reads only, and degrades to a line of text when there is no environment.

Public surface: ``plan_run``, ``stage_run``, ``lock_profile_source``, ``feasibility_report``, and
the ``LaunchPlan`` / ``Idempotency`` result types.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import ConfigError, get_logger
from .registry.ids import make_run_id
from .router import split_by_runtime

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .commands import LaunchCommands
    from .config import Fanout, RunConfig
    from .dag import DagNode
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
    the commands runnable). ``commands`` maps a tier name to its `LaunchCommands`: ``"main"``,
    ``"ray"``, and one ``"spark:<family>"`` per Serverless family (``"spark:statistical"``, …) —
    per-family because each family resolves its own hardware and batch id, so one command cannot
    stand for all of them. It is ``None`` only when the GCP infra identity can't be resolved offline
    (no ``SF_*`` env), in which case the run_id + fanout + runtime split are still returned.

    ``idempotency`` is the exists-vs-new verdict for this config's id; ``force`` records that the
    caller intends to re-run an already-run config (it only shapes the emitted guidance — a re-run
    is idempotent regardless, via dedupe-on-read under the shared ``run_id``).

    ``nodes`` is the run's execution DAG resolved offline (`dag.dag_nodes`): one node per family job
    plus the ensemble, each with its deterministic ``job_key`` and upstream dependencies — the
    plan's cross-system identity map, before anything runs.
    """

    run_id: str
    python_runtime: str
    python_models: list[str]
    bq_models: list[str]
    fanout: Fanout
    staged: bool
    config_uri: str | None
    commands: dict[str, LaunchCommands] | None
    idempotency: Idempotency
    force: bool
    nodes: tuple[DagNode, ...]


@dataclass(frozen=True)
class _RunPlan:
    """The offline-resolvable shape of a run: its id and the per-runtime executed subsets.

    Pure product of the config (`_plan`) — no GCP.
    """

    run_id: str
    python_models: list[str]
    bq_models: list[str]


def _plan(cfg: RunConfig) -> _RunPlan:
    """Resolve the run_id + per-runtime model split (pure and offline).

    Computes ``make_run_id(cfg)`` and `router.split_by_runtime`. Both ``python_runtime="spark"``
    and ``python_runtime="ray"`` are supported (dispatched in `run`); an all-BigQuery config plans
    and runs regardless of ``python_runtime``.
    """
    run_id = make_run_id(cfg)
    python_models, bq_models = split_by_runtime(cfg)
    return _RunPlan(
        run_id=run_id,
        python_models=python_models,
        bq_models=bq_models,
    )


def _check_idempotency(run_id: str, settings: Settings) -> Idempotency:
    """Best-effort pre-submit existence check: has this exact config already run?

    Queries the registry for ``run_id`` (`registry.header.header_status`). Any failure — no registry
    table yet, no reachable BigQuery — degrades to ``checked=False`` (unknown), so a plain offline
    dry-run still returns a plan. The check is advisory: it warns before an accidental duplicate run
    but never blocks one (a re-run is idempotent via dedupe-on-read).
    """
    from .registry.header import header_status

    try:
        status = header_status(run_id, settings=settings)
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
        from .ray_infra import RayInfra

        return RayInfra.resolve()
    from .batch_infra import BatchInfra

    return BatchInfra.resolve()


def _template_uris(
    cfg: RunConfig, plan: _RunPlan, code_bucket: str
) -> tuple[str, str | None, str | None]:
    """The ``gs://`` URIs a run's artifacts *will* land at (pure — mirrors the staging scheme).

    Returns ``(config_uri, package_uri, launcher_uri)``. The config URI always exists; the Spark
    package/launcher URIs are set only for a Spark run with Python models (Ray delivers code via its
    ``runtime_env`` working dir, not a staged zip). The package name carries the code hash from
    `code_delivery.build_package_zip` — a deterministic local build, no network — so the template is
    byte-faithful to what `staging.stage_code` would upload.
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
    nodes: Sequence[DagNode],
    *,
    config_uri: str,
    package_uri: str | None,
    launcher_uri: str | None,
) -> dict[str, LaunchCommands]:
    """Build the launch commands for a run, keyed by tier (pure — assembles strings only).

    Always emits ``"main"`` — ``python -m scale_forecasting.main --config-uri …``, the orchestrator
    that reproduces the *full* run (both engines under one run_id). When there are Python-runtime
    models it adds the per-runtime tier: ``"ray"`` (universal only) or one ``"spark:<family>"``
    entry per Serverless family (native ``gcloud`` + universal).

    **The Spark tier is keyed by family because the run is.** A Serverless executor's shape is
    fixed at batch creation and each family resolves its own hardware, so one command cannot
    describe a run whose statistical family is on CPU and whose deep-learning family holds a GPU.
    Emitting one per family is what makes the printed command reproduce the batch `run` submits:
    same ``--batch`` id (`registry.ids.dataproc_job_id` over the family's planned ``job_key``),
    same ``--models`` subset, same sizing overlay derived at that family's hardware, and
    ``--provisioned-hardware gpu`` where the family bought a device.

    Each command stands alone — ``manage_header=True``, so copy-pasting one reproduces that
    family's batch under its own header rather than needing the orchestrator to own one.

    **Cluster-mode families get no command, deliberately.** `build_spark_commands` renders
    ``gcloud dataproc batches submit``, which is the Serverless form; a cluster family's job goes
    to ``gcloud dataproc jobs submit`` against a cluster the run *creates*, so there is no
    standalone line to copy. Emitting the Serverless form for it would print a command that runs
    and runs the wrong thing.
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

    from .batch_infra import BatchInfra
    from .profiling.source import profile_for_run
    from .registry.ids import dataproc_job_id
    from .submit import sizing_properties

    assert isinstance(infra, BatchInfra)  # spark runtime → BatchInfra (resolved above)
    profile = profile_for_run(cfg, settings=settings)
    for node in nodes:
        if node.runtime != "spark" or node.spark_mode == "cluster":
            continue
        models = list(node.models)
        hardware = node.hardware or "cpu"
        commands[f"spark:{node.family}"] = build_spark_commands(
            settings=settings,
            infra=infra,
            batch_id=dataproc_job_id(node.job_key),
            package_uri=package_uri or "",
            launcher_uri=launcher_uri or "",
            config_uri=config_uri,
            models=models,
            manage_header=True,
            # The emitted gcloud command has to carry the sizing overlay the SDK path applies, or
            # copy-pasting it would submit a differently-shaped batch than `run` would. The overlay
            # is hardware-dependent, which is the whole reason this is resolved per family.
            properties=sizing_properties(
                cfg, models, hardware=hardware, gpu_type=node.gpu_type, profile=profile
            ),
            provisioned_hardware=hardware,
            gpu_type=node.gpu_type,
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
    """Log the resolved plan, its DAG nodes, and each launch-command tier (dry-run/stage emit)."""
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
    for node in result.nodes:
        after = f" after [{', '.join(node.depends_on)}]" if node.depends_on else ""
        _log.info("  node %s: %s on %s%s", node.family, node.job_key, node.runtime, after)
    if result.commands is None:
        _log.info(
            "%s %s: infra unresolved (no SF_* env) — commands not emitted", verb, result.run_id
        )
        return
    for name, lc in result.commands.items():
        _log.info("  [%s] %s", name, lc.universal)
        if lc.native is not None:
            _log.info("  [%s:native] %s", name, lc.native)


def lock_profile_source(cfg: RunConfig, *, settings: Settings | None = None) -> RunConfig:
    """Replace ``compute.profile.source = "auto"`` with the concrete reference it resolves to.

    The lockfile step. Pinning is what makes ``auto`` convenient *and* reproducible: re-running a
    staged config reproduces the original fleet exactly, because what actually resolved is written
    into it rather than re-searched.

    It deliberately does **not** move the ``run_id``. ``registry.ids`` excludes this field from the
    digest, because it is resolved rather than authored — see ``_canonical_config``, which explains
    what went wrong when it was included. Two ``auto`` runs a week apart therefore share an id and
    dedupe on read, while their staged manifests and sizing provenance record the different
    evidence each was sized from. That is the distinction: a run's *identity* is what was asked
    for; its *provenance* is what answered.

    What gets pinned is the **pointer**, never the numbers. The rows behind a ``run_id`` are
    immutable, so the pointer is as good as the values and keeps the config small enough to stay
    readable. When discovery finds nothing, the pin is ``"baseline"`` rather than ``"auto"``: the
    remainder of the chain (shipped baseline, then static) is deterministic, so pinning it makes
    the run reproducible instead of leaving it to re-search later.

    Degrades to the unlocked config when the registry cannot be reached — a plan produced with no
    ``SF_*`` environment is a preview, and it says so elsewhere too (``commands=None``, verdict
    unknown). The run still resolves its source at sizing time; it just is not pinned.
    """
    if not cfg.compute.profile.needs_source_resolution:
        return cfg

    from .profiling.signature import signature_from_config
    from .registry.harvest import discover_harvest_run

    want = signature_from_config(cfg)
    try:
        found = discover_harvest_run(
            source_table=want.source_table,
            freq=want.freq,
            target_series=want.n_series,
            target_runtime=cfg.python_runtime,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - an unpinned plan is a worse plan, not a failed one
        _log.debug("profile source not pinned (registry unreachable): %r", exc)
        return cfg

    locked = found or "baseline"
    _log.info("compute profile source: auto -> %s", locked)
    profile = cfg.compute.profile.model_copy(update={"source": locked})
    return cfg.model_copy(update={"compute": cfg.compute.model_copy(update={"profile": profile})})


def _nodes_with_submit_attempts(
    nodes: Sequence[DagNode], run_id: str, *, force: bool, settings: Settings | None
) -> tuple[DagNode, ...]:
    """Re-stamp every node's ``job_key`` with the attempt a submit *right now* would use.

    `dag.plan_dag` is pure and therefore stamps attempt 1 on everything — "a planned identity has no
    history behind it". That is the right property for the DAG builder and the wrong string to hand
    a human. The emitted ``gcloud dataproc batches submit`` command carries the job key as the
    platform's own ``--batch`` id, so on any run that has already executed, pasting the emitted
    command is rejected with ``ALREADY_EXISTS`` — found 2026-09-22 by pasting one, which is the only
    way it could have been found: every offline test compares the emitted string to what the
    *submitter* would build from the same nodes, so both sides carried the same stale ``a1``.

    The history lookup lives here rather than in `dag` because `plan_run` and `stage_run` already
    consult the registry (`_check_idempotency`) while `plan_dag` must stay callable with no
    environment at all. Purity is preserved where it was load-bearing and paid for where it was not.

    ``depends_on`` is rewritten through the same map. Re-stamping the keys without it would leave
    the ensemble node depending on a family job id that no longer exists anywhere.

    The attempt is whatever the policy hands out, including the unforced-re-run case where that is
    the *existing* job's number rather than a fresh one — printing an id no submit would create
    would be a different lie in place of the old one.

    Best-effort, like everything else `plan_run` layers on top of the pure plan: if the registry
    cannot be reached the original attempt-1 nodes come back with a warning, because a stale
    suggestion is worth more than no plan at all — but it says so rather than looking authoritative.
    """
    from dataclasses import replace

    from .registry.ids import make_job_key
    from .registry.jobs import next_job_attempt

    try:
        attempts: dict[str, int] = {}
        for node in nodes:
            if node.family not in attempts:
                attempts[node.family] = next_job_attempt(
                    run_id, node.family, force=force, settings=settings
                )[0]
    except Exception as exc:  # noqa: BLE001 - a stale attempt is a worse plan, not a failed one
        _log.warning(
            "could not resolve submit attempts from the registry (%r); the emitted commands say "
            "attempt 1, which will be rejected as ALREADY_EXISTS if this run has already executed",
            exc,
        )
        return tuple(nodes)

    rekeyed = {n.job_key: make_job_key(run_id, n.family, attempts[n.family]) for n in nodes}
    return tuple(
        replace(
            node,
            job_key=rekeyed[node.job_key],
            depends_on=tuple(rekeyed.get(dep, dep) for dep in node.depends_on),
        )
        for node in nodes
    )


def _file_emitted_rows(
    cfg: RunConfig, nodes: Sequence[DagNode], run_id: str, settings: Settings
) -> None:
    """File one ``EMITTED`` ``run_jobs`` row per staged node — the attempt numbers we just spent.

    **What this closes.** Staging hands out job ids: every command it prints names its own job
    (``--batch``, a Ray ``submission_id``, a BigQuery job prefix), and the attempt in those ids came
    from `_nodes_with_submit_attempts` reading ``MAX(attempt)`` off this very table. Whoever pastes
    the command creates a job on the platform — and, not being this launcher, writes no row. The
    registry then still reads ``MAX(attempt) = N-1``, so the next ``--force`` hands out ``N`` again
    and the submit is refused with ``ALREADY_EXISTS``. Observed live 2026-09-22; see
    `errors.JobIdTaken` and ``docs/validation.md``.

    Writing the row here makes the handed-out attempt visible to the counter, which is all it takes:
    `registry.jobs.latest_job_attempt` is ``MAX(attempt)`` with no status filter, so an EMITTED row
    occupies its number whether or not anyone ever runs the command.

    **The phantom-row question, and why it is answered rather than traded off.** A command that is
    emitted and never pasted leaves a row for a job that does not exist. Three things keep that from
    being a new kind of mess. The row carries a probe handle, so `probes.reconcile` escalates it
    like any other non-terminal row and `probes.settle` reaps it to FAILED / ``NEVER_LAUNCHED`` once
    the platform confirms the id was never created. Its status is not in
    `registry.ops.LIVE_STATUSES`, so it never blocks a destructive verb. And if the operator *does*
    eventually paste the command, the launch writes a fresh ``RUNNING`` row at the same attempt with
    a later ``created_at`` — and ``v_run_jobs`` orders by ``attempt DESC, created_at DESC``, so the
    real launch supersedes this row with no cleanup at all.

    Best-effort for the same reason as `_nodes_with_submit_attempts`: an unreachable registry must
    not fail a staging whose actual product — the uploaded artifacts and the printed commands — is
    already correct. It says so rather than failing quietly.

    One knock-on worth naming: staging and then launching *in this process* with ``--force`` now
    lands on attempt N+1 rather than N, because the row written here is exactly the evidence that N
    is spent. That is the intended reading — a forced launch asks for an id nobody has taken, and
    the staged command may well have been pasted in the meantime — and it costs one attempt number,
    not one job. ``main.run`` is unaffected: it stages its own artifacts through `staging` and never
    comes through here.
    """
    from datetime import UTC, datetime

    from .job_launch import _entry_handle, _system_job_id
    from .probes.vocabulary import ProbeHandle
    from .registry.ids import parse_job_key
    from .registry.jobs import write_job
    from .registry.rows import EMITTED, assemble_job_row
    from .registry.tables import ensure_tables

    try:
        ensure_tables(cfg, settings=settings)
        created_at = datetime.now(UTC)
        for node in nodes:
            _, family, attempt = parse_job_key(node.job_key)
            system_job_id = _system_job_id(node.job_key, node.runtime)
            if node.runtime == "bigquery":
                # The native and ensemble nodes run as several statements under a shared id prefix,
                # recorded exactly as `job_launch.launch_native_job` records them.
                handle = ProbeHandle(
                    "bigquery",
                    native_id=f"{system_job_id}-",
                    region=settings.region,
                    id_kind="prefix",
                )
            else:
                # The same builder the launcher uses, so a staged row and a launched row are
                # probe-able through one code path. No shared cluster exists at staging time, which
                # is the default the builder already handles.
                handle = _entry_handle(
                    cfg, run_id, cfg.resolve_family_compute(family), system_job_id, settings
                )
            write_job(
                assemble_job_row(
                    run_id,
                    family,
                    attempt,
                    created_at,
                    runtime=node.runtime,
                    spark_mode=node.spark_mode,
                    hardware=node.hardware,
                    gpu_type=node.gpu_type,
                    system_job_id=system_job_id,
                    status=EMITTED,
                    probe_handle=handle.to_blob(),
                ),
                settings=settings,
            )
    except Exception as exc:  # noqa: BLE001 - an unrecorded emit is a worse stage, not a failed one
        _log.warning(
            "could not record the emitted attempts in the registry (%r); the commands above are "
            "still correct, but a later --force may re-issue a job id these ones already took",
            exc,
        )


def plan_run(
    cfg: RunConfig,
    *,
    settings: Settings | None = None,
    infra: object | None = None,
    force: bool = False,
) -> LaunchPlan:
    """Resolve a run offline to its id, fanout, runtime split, and launch-command *templates*.

    The "plan" verb: pure and GCP-free. Computes `_plan` and `estimate_fanout`, then — best-effort —
    resolves the infra identity, consults the registry for the exists-vs-new verdict
    (`_check_idempotency`), and
    builds the two-tier launch commands with the URIs the artifacts *will* land at (nothing is
    uploaded). When the infra can't be resolved offline (no ``SF_*`` env), the commands are omitted
    (``commands=None``) and the verdict is *unknown*, but the id/fanout/split are still returned, so
    a plain ``--dry-run`` works with no environment. ``force`` shapes only the emitted re-run
    guidance. Emits the plan to the log and returns it.
    """
    from .config import estimate_fanout
    from .dag import dag_nodes, plan_dag
    from .profiling.source import check_pinned_source

    # Before the lock, so this only ever judges what a *person* pinned (see `check_pinned_source`),
    # and outside the try, because a rotted pin is the one thing here that is *not* best-effort.
    # It degrades on its own when the registry is unreachable — no drift is visible, so none is
    # reported.
    check_pinned_source(cfg, settings=settings, force=force)

    # The infra identity is resolved first and separately, because `lock_profile_source` needs it
    # and its result is part of the digest — so it has to land before `_plan` computes the run_id.
    # Best-effort, exactly like everything else in this try: no environment means an unpinned
    # preview, not a failure.
    with suppress(Exception):
        settings = settings or _resolve_settings()
        cfg = lock_profile_source(cfg, settings=settings)

    plan = _plan(cfg)
    fanout = estimate_fanout(cfg)
    nodes = dag_nodes(plan_dag(cfg))
    config_uri: str | None = None
    commands: dict[str, LaunchCommands] | None = None
    idempotency = Idempotency(checked=False, exists=False, prior_status=None)
    try:
        settings = settings or _resolve_settings()
        idempotency = _check_idempotency(plan.run_id, settings)
        # Only now that a registry is reachable — see `_nodes_with_submit_attempts`. Rebinding
        # `nodes` rather than using a second name is deliberate: the LaunchPlan below carries these
        # too, so what a reader sees in the plan is what a paste of the command would submit.
        nodes = _nodes_with_submit_attempts(nodes, plan.run_id, force=force, settings=settings)
        resolved_infra = _resolve_infra(cfg, infra)
        code_bucket = resolved_infra.code_bucket  # type: ignore[attr-defined]
        config_uri, package_uri, launcher_uri = _template_uris(cfg, plan, code_bucket)
        commands = _assemble_commands(
            cfg,
            plan,
            settings,
            resolved_infra,
            nodes,
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
        fanout=fanout,
        staged=False,
        config_uri=config_uri,
        commands=commands,
        idempotency=idempotency,
        force=force,
        nodes=nodes,
    )
    _emit_plan(result)
    return result


def read_series_lengths(
    cfg: RunConfig, *, settings: Settings | None = None
) -> list[int]:  # pragma: no cover - GCP I/O, covered by the @gcp smokes
    """Observation count of every series the run will forecast — one BigQuery aggregation.

    ``SELECT ts_id, COUNT(*) … GROUP BY ts_id``, honouring ``data.series_limit`` the same way the
    engines do. One scan of two columns with a single shuffle: cheap enough to run before a launch,
    which is the whole point of `feasibility_report`.

    The counts are raw rows, and raw rows are the exact input a fit sees — `features.build_features`
    creates lag columns but does no ``dropna``, so nothing shrinks the panel between here and the
    model. No lag-warmup correction is needed and applying one would understate every series.
    """
    from google.cloud import bigquery

    from .engines.bigquery_names import _source_ref
    from .settings import Settings as _Settings

    settings = settings or _Settings.resolve()
    table = _source_ref(cfg, settings.dataset_ref)
    limit = "" if cfg.data.series_limit is None else f"\nLIMIT {int(cfg.data.series_limit)}"
    sql = (
        f"SELECT `{cfg.data.ts_id_col}` AS ts_id, COUNT(*) AS n_obs\n"
        f"FROM `{table}`\n"
        f"GROUP BY ts_id\n"
        f"ORDER BY ts_id{limit}"
    )
    rows = bigquery.Client(project=settings.project_id).query(sql).result()
    return [int(r["n_obs"]) for r in rows]


def feasibility_lines(cfg: RunConfig, obs_counts: Sequence[int]) -> list[str]:
    """The feasibility report for a measured panel, as lines to print (pure).

    What ``plan --feasibility`` prints once it has the counts. Three things, in the order an
    operator needs them: how much work this is (cells, fits, and the honest backtest multiplier),
    who actually gets scored (the achieved-fold cohorts, which on a ragged panel are not one
    number), and what to change if the answer is unwelcome (`backtest.suggest_min_train`).

    Advisory only — it reads nothing but the counts and writes nothing at all. ``min_train`` is a
    config field, so acting on the suggestion moves the run_id, which is the caller's decision to
    make deliberately rather than something a planning verb should do for them.
    """
    from .backtest import suggest_min_train
    from .config import estimate_workload

    w = estimate_workload(cfg, obs_counts=obs_counts)
    lines = [
        f"feasibility: {w.n_series} series x {w.n_models} models = {w.n_cells} cells",
        f"  fits: {w.n_fits} ({w.train_rows_total:,} training rows) = "
        f"{w.full_fit_equivalents:.3f} whole-history fits per cell",
    ]
    if not cfg.backtest.enabled:
        lines.append("  backtest: off — one full-history fit per cell, no folds to achieve")
        return lines
    for k, n in sorted(w.fold_histogram.items(), reverse=True):
        share = n / (w.n_series or 1)
        verdict = "all folds" if k == cfg.backtest.n_folds else "UNSCORED" if k == 0 else "reduced"
        lines.append(f"  {k} of {cfg.backtest.n_folds} folds: {n} series ({share:.1%}) — {verdict}")
    if w.n_unscored:
        lines.append(
            f"  {w.n_unscored} series get no folds at all: they are still fit and forecast, but "
            "they contribute nothing to any leaderboard (see v_backtest_coverage)"
        )
    suggested = suggest_min_train(obs_counts, cfg)
    current = cfg.backtest.min_train
    if suggested is None:
        lines.append(
            f"  no min_train reaches 90% of series at all {cfg.backtest.n_folds} folds — the panel "
            f"is short for this geometry; reduce n_folds ({cfg.backtest.n_folds}), step "
            f"({cfg.backtest.step}) or horizon ({cfg.backtest.horizon}) instead"
        )
    elif suggested < current:
        lines.append(
            f"  min_train={current} is above what this panel supports; min_train<={suggested} "
            "brings 90% of series to full folds (changing it moves the run_id)"
        )
    else:
        lines.append(
            f"  min_train={current} is within budget: up to {suggested} still keeps 90% of series "
            "at full folds"
        )
    return lines


def feasibility_report(cfg: RunConfig, *, settings: Settings | None = None) -> list[str]:
    """Read the panel's series lengths and report what the run's fold geometry does to it.

    The ``--feasibility`` half of the plan verb: the one part of planning that cannot be honest
    offline, because how many folds a series achieves is a fact about the data, not the config.
    Best-effort like every other reporting verb — an unreachable environment produces a line saying
    so rather than an exception, so ``--dry-run --feasibility`` still returns a plan with no
    ``SF_*`` env.
    """
    try:
        obs_counts = read_series_lengths(cfg, settings=settings)
    except Exception as exc:  # noqa: BLE001 - a report that can fail is one nobody runs
        return [f"feasibility: could not read the source panel ({exc}); reporting counts only"]
    if not obs_counts:
        return ["feasibility: the source panel returned no series"]
    return feasibility_lines(cfg, obs_counts)


def preflight_short_series(cfg: RunConfig, *, settings: Settings | None = None) -> None:
    """Enforce ``backtest.short_series="error"`` at submit time, where the operator is watching.

    `main.run` calls this once, before it launches anything. It is the *useful* place to refuse a
    run under that policy: the message reaches the terminal the operator typed into, no cluster has
    been provisioned, and no registry rows exist to clean up. Each Python engine's driver holds a
    second check (`spark_io.assert_source_supports_folds` and the Ray twin) for the paths that do
    not come through here — a staged config launched from an emitted command, a Composer task, or
    an engine driven directly.

    Costs one aggregation, and only when the policy is armed; every other value of
    ``short_series`` returns before touching BigQuery, because those policies have already decided
    what a short series gets.

    **A failed read warns rather than refuses.** ``short_series="error"`` means "stop if the panel
    is too short", not "stop if I could not tell" — turning a transient BigQuery hiccup into a
    refused run would be a worse failure than the one this guards against. The driver backstop
    still runs, on a panel the engine has by then definitely read, so nothing is skipped by
    letting this one go.
    """
    bt = cfg.backtest
    if not (bt.enabled and bt.short_series == "error"):
        return
    from .backtest import assert_panel_supports_folds

    try:
        obs_counts = read_series_lengths(cfg, settings=settings)
    except Exception as exc:  # noqa: BLE001 - see the docstring: unreadable is not too short
        _log.warning(
            "short_series='error': could not read the source panel (%s); deferring the check to "
            "the engine driver",
            exc,
        )
        return
    if obs_counts:
        assert_panel_supports_folds(obs_counts, cfg)


def _manifest_dict(result: LaunchPlan, *, created_at: str) -> dict[str, object]:
    """The reproducibility-manifest payload for a staged run (pure — ``created_at`` is caller-set).

    Records the config digest, fan-out, runtime split, both command tiers, the staged config URI,
    and the execution ``dag`` (one entry per family job + the ensemble, each with its deterministic
    ``job_key`` and ``depends_on``) — everything needed to answer "what command produced run X, and
    which jobs did it schedule under what ids?". The timestamp is passed in (not read here) so this
    stays a pure function with no wall-clock.
    """
    commands = {
        name: {"runtime": lc.runtime, "universal": lc.universal, "native": lc.native}
        for name, lc in (result.commands or {}).items()
    }
    dag = [
        {
            "job_key": n.job_key,
            "family": n.family,
            "runtime": n.runtime,
            "models": list(n.models),
            "hardware": n.hardware,
            "gpu_type": n.gpu_type,
            "spark_mode": n.spark_mode,
            "depends_on": list(n.depends_on),
        }
        for n in result.nodes
    ]
    return {
        "run_id": result.run_id,
        "created_at": created_at,
        "dag": dag,
        "python_runtime": result.python_runtime,
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
    from .dag import dag_nodes, plan_dag, preflight
    from .profiling.source import check_pinned_source
    from .staging import stage_config, stage_manifest

    # Refuse an unhonourable config, or an incoherent hardware plan, before anything is staged.
    preflight(cfg)
    # Pin `source: "auto"` to what it resolves to *now*, before the digest — the staged config is
    # the reproducibility artifact, so what actually sized this run has to be written into it.
    settings = settings or _resolve_settings()
    # Before the lock, so this only ever judges what a *person* pinned (see `check_pinned_source`).
    check_pinned_source(cfg, settings=settings, force=force)
    cfg = lock_profile_source(cfg, settings=settings)

    plan = _plan(cfg)
    fanout = estimate_fanout(cfg)
    nodes = dag_nodes(plan_dag(cfg))
    idempotency = _check_idempotency(plan.run_id, settings)
    # Staging is the path whose commands a human actually pastes, so the attempt has to be the one
    # a submit would use rather than the planner's attempt 1 — see `_nodes_with_submit_attempts`.
    nodes = _nodes_with_submit_attempts(nodes, plan.run_id, force=force, settings=settings)
    resolved_infra = _resolve_infra(cfg, infra)
    code_bucket: str = resolved_infra.code_bucket  # type: ignore[attr-defined]

    config_uri = stage_config(cfg, plan.run_id, code_bucket)
    package_uri: str | None = None
    launcher_uri: str | None = None
    if cfg.python_runtime != "ray" and plan.python_models:
        from .batch_infra import BatchInfra
        from .staging import stage_code

        assert isinstance(resolved_infra, BatchInfra)  # spark runtime → BatchInfra
        package_uri, launcher_uri = stage_code(resolved_infra.code_bucket)

    commands = _assemble_commands(
        cfg,
        plan,
        settings,
        resolved_infra,
        nodes,
        config_uri=config_uri,
        package_uri=package_uri,
        launcher_uri=launcher_uri,
    )
    result = LaunchPlan(
        run_id=plan.run_id,
        python_runtime=cfg.python_runtime,
        python_models=plan.python_models,
        bq_models=plan.bq_models,
        fanout=fanout,
        staged=True,
        config_uri=config_uri,
        commands=commands,
        idempotency=idempotency,
        force=force,
        nodes=nodes,
    )
    manifest_uri = stage_manifest(
        _manifest_dict(result, created_at=datetime.now(UTC).isoformat()), plan.run_id, code_bucket
    )
    _log.info("wrote run manifest: %s", manifest_uri)
    # Last, once the commands are real and the artifacts are up: record the attempt numbers those
    # commands just spent, so the counter can see a launch this process will never make. `plan_run`
    # deliberately does not do this — a dry preview hands out nothing, and filing rows for it would
    # burn an attempt every time somebody looked at a plan.
    _file_emitted_rows(cfg, nodes, plan.run_id, settings)
    _emit_plan(result)
    return result


def _resolve_settings() -> Settings:
    """Resolve `Settings` from the ``SF_*`` env (raises `ConfigError` when unset)."""
    from .settings import Settings

    return Settings.resolve()
