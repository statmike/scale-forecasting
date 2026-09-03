"""The driver-side pre-pass — decide whether to measure, then where the numbers come from.

The consumer of everything else in this package, and the two questions a run actually asks.

*Is measuring worth it?* — `should_profile`, a pure function of ``compute.profile.mode`` and
the fan-out size.

*Where do the numbers come from?* — the two answers are deliberately separate functions.
`resolve_profile_source` reuses **prior** evidence: a named run's harvest, the newest
auto-discovered harvest whose signature matches, or the shipped baseline, in that order,
each stamped with `ProfileProvenance` so a sizing decision can always say which of the three
it came from. `resolve_profile` produces **fresh** evidence by sampling the panel and
running real fits. Both return ``None`` for "no evidence", which is a decision rather than a
failure: sizing from declared config is the floor under every path here.

One case does *not* get the graceful floor: an operator who pinned ``profile.source`` to a
specific run id asserted that that evidence applies here, and `check_pinned_source` fails the
run at the entry point when the data has since moved out from under that assertion.

Every loader — and the measurement function itself — is injected, so the whole precedence
chain and the whole pre-pass are exercised offline against deterministic stand-ins.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..errors import ConfigError, DataError, get_logger
from .cost import ComputeProfile, ProfileProvenance, build_profile, harvest_profile
from .measure import MeasuredFit, measure_fit
from .sampling import select_profile_sample
from .signature import DataSignature, compare_signatures, signature_from_config, signature_from_rows
from .stats import series_stats

if TYPE_CHECKING:
    import pandas as pd

    from ..config import RunConfig
    from ..settings import Settings

_log = get_logger(__name__)


def should_profile(cfg: RunConfig, n_cells: int) -> bool:
    """Is a measurement pre-pass worth running for a fan-out of ``n_cells``? (pure)

    The three ``compute.profile.mode`` values, and the reasoning behind the middle one:

    * ``off`` — never. The escape hatch: static config, exactly as before the profiler existed.
    * ``always`` — unconditionally. What the smokes use, so the path is exercised on runs small
      enough that exercising it is cheap.
    * ``auto`` (default) — only when the fan-out is big enough to repay the pre-pass. The
      pre-pass costs ``samples x models`` real fits on the driver; below ``min_cells`` that is a
      measurable fraction of the whole run, spent to size a fleet that barely needs sizing. Above
      it, the same fixed cost is rounding error against the work it is optimising.

    The comparison is on **cells**, not series, because a 100-series run over 6 models is the
    same amount of work as a 600-series run over one, and it is the work — not the panel — that
    the fleet is sized for.
    """
    mode = cfg.compute.profile.mode
    if mode == "off":
        return False
    if mode == "always":
        return True
    return n_cells >= cfg.compute.profile.min_cells


RunHarvestLoader = Callable[[str], "tuple[Sequence[Mapping[str, Any]], str | None] | None"]
# Finds the newest run whose harvest matches a signature, or ``None``. Separate from the loader
# because "which run" is a search and "what did it cost" is a read, and only `auto` does the search.
RunDiscoverer = Callable[[DataSignature], "str | None"]
# The shipped, versioned baseline (W13), or ``None`` before one exists.
BaselineLoader = Callable[[], "ComputeProfile | None"]


def resolve_profile_source(
    cfg: RunConfig,
    *,
    load_run: RunHarvestLoader | None = None,
    load_baseline: BaselineLoader | None = None,
    discover: RunDiscoverer | None = None,
) -> ComputeProfile | None:
    """The profile this run should size from, per ``compute.profile.source`` (pure + injected I/O).

    Returns ``None`` for "no evidence" — the static-config case every consumer already handles
    (`resources.slot.resource_slot` takes ``profile=None`` as its identity case). ``None`` is a
    *decision*, not a failure: sizing from declared config is the behaviour this product shipped
    with, and it stays the floor under every path here.

    The precedence, resolved outside-in:

    1. ``mode == "off"`` or ``source == "none"`` → ``None``. Nothing consulted, nothing loaded.
    2. ``source == "<run_id>"`` → that run's harvest. An operator naming a run has made a
       decision; it is honoured even if the signature has drifted, and the drift comes back as
       warnings on the provenance rather than as a substitution they did not ask for.
    3. ``source == "auto"`` → the newest run whose harvest matches this run's signature, if
       ``discover`` finds one.
    4. the shipped baseline, if one is loadable.
    5. ``None``.

    Every loader is injected. That keeps this function pure enough to test the whole precedence
    chain offline with no BigQuery and no baseline file — the same seam ``resolve_profile`` uses
    for ``measure``. A loader that raises is treated as a loader that found nothing: sizing
    evidence is an optimisation, and a registry hiccup must not sink a run that would otherwise
    size itself from config.
    """
    profile_cfg = cfg.compute.profile
    if not profile_cfg.consumes_evidence:
        return None

    want = signature_from_config(cfg)
    source = profile_cfg.source

    run_id = source if source != "baseline" else None
    if source == "auto":
        run_id = _try(lambda: discover(want)) if discover else None

    if run_id and load_run:
        loaded = _try(lambda: load_run(run_id))
        if loaded:
            rows, source_table = loaded
            profile = harvest_profile(
                rows,
                memory_margin=profile_cfg.memory_margin,
                time_margin=profile_cfg.time_margin,
            )
            have = signature_from_rows(rows, source_table=source_table)
            warnings = compare_signatures(want, have)
            return _with_provenance(
                profile,
                ProfileProvenance(
                    # No drift on any axis both sides can see is as close to "measured on your
                    # data" as harvested evidence gets; anything else is honest about being
                    # someone else's measurement.
                    basis="measured" if not warnings else "reference",
                    source=source,
                    run_id=run_id,
                    measured_at=_measured_at(rows),
                    signature=have,
                    warnings=warnings,
                ),
            )

    if load_baseline:
        baseline = _try(load_baseline)
        if baseline is not None:
            existing = baseline.provenance
            return _with_provenance(
                baseline,
                ProfileProvenance(
                    basis="reference",  # measured, but never on your data — that is what it is for
                    source=source,
                    baseline_version=existing.baseline_version if existing else None,
                    measured_at=existing.measured_at if existing else None,
                    signature=existing.signature if existing else None,
                    warnings=(
                        "sized from the shipped baseline, not from a measurement of your data",
                    ),
                ),
            )
    return None


def pinned_source_drift(
    cfg: RunConfig, *, load_run: RunHarvestLoader | None = None
) -> tuple[str, ...]:
    """Signature drift on an **explicitly pinned** ``profile.source``, or ``()`` (pure + injected).

    Only an explicit ``source: "<run_id>"`` is checked. ``auto``, ``baseline`` and ``none`` are the
    system choosing, and the system has a fallback; a pinned run id is a *person* asserting that
    this specific evidence applies to this run. When the signature says the data has moved out from
    under it, that assertion is false, and this is what `check_pinned_source` turns into an error.

    Returns the same warning strings `resolve_profile_source` would have stamped on the provenance
    — computed by calling it, so the two can never disagree about what "drifted" means. A pin that
    cannot be loaded at all returns ``()``: an unreachable registry is not evidence of drift, and
    the sizing path already degrades to config for it.
    """
    profile_cfg = cfg.compute.profile
    if not profile_cfg.consumes_evidence or profile_cfg.source in ("auto", "baseline", "none"):
        return ()
    resolved = resolve_profile_source(cfg, load_run=load_run)
    if resolved is None or resolved.provenance is None:
        return ()
    return tuple(resolved.provenance.warnings)


def check_pinned_source(
    cfg: RunConfig, *, settings: Settings | None = None, force: bool = False
) -> None:
    """Raise `errors.ConfigError` when a pinned ``profile.source`` has silently rotted (I/O).

    The asymmetry with ``auto`` is deliberate and is about who made the claim. ``auto`` warns and
    degrades — it is a hint, it has a fallback, and hard-failing would make an unattended run
    brittle for no safety gain. An explicit run id is a human assertion; sizing off evidence that
    no longer describes this data is exactly the failure the provenance machinery exists to
    prevent, so it fails loudly and names both sides. ``force=True`` overrides — the same verb that
    already overrides the idempotency guard, so there is no second escape hatch to learn.

    Called once at the entry points (`main.run`, `launch_plan.plan_run` / ``stage_run``) rather than
    at each of the sizing sites: the operator should hear about this before anything is submitted,
    once, not six times from inside a launch.
    """
    if force:
        return
    from ..registry.harvest import read_compute_harvest

    warnings = pinned_source_drift(
        cfg, load_run=lambda run_id: read_compute_harvest(run_id, settings=settings)
    )
    if not warnings:
        return

    detail = "; ".join(warnings)
    raise ConfigError(
        f"compute.profile.source is pinned to {cfg.compute.profile.source!r}, but that run's "
        f"measurements no longer describe this run's data: {detail}. Sizing from it would be "
        "sizing off evidence you did not mean to use. Re-pin to a current run, switch to "
        '"auto"/"baseline", or pass --force to size from it anyway.'
    )


def _try(load: Callable[[], Any]) -> Any:
    """Run a loader, swallowing its failure (see `resolve_profile_source`)."""
    try:
        return load()
    except Exception as e:  # noqa: BLE001 - evidence is an optimisation; never fatal
        _log.warning("profile source lookup failed, falling back: %r", e)
        return None


def _with_provenance(profile: ComputeProfile, provenance: ProfileProvenance) -> ComputeProfile:
    """``profile`` with its provenance stamped on (frozen dataclass → a copy)."""
    return replace(profile, provenance=provenance)


def _measured_at(rows: Iterable[Mapping[str, Any]]) -> str | None:
    """The newest ``created_at`` in a harvest, as a string, or None when the rows carry none."""
    stamps = [str(row["created_at"]) for row in rows if row.get("created_at") is not None]
    return max(stamps) if stamps else None


# Resolved profiles, keyed by what determines them. Sizing is consulted once per family job — four
# jobs on one submit host is four identical BigQuery round-trips otherwise, and (worse) an `auto`
# that re-discovers between them could hand two jobs of the same run different fleets. Process-local
# and bounded by the handful of distinct keys a run produces; nothing here outlives the process.
_RESOLVED: dict[tuple[Any, ...], ComputeProfile | None] = {}


def profile_for_run(cfg: RunConfig, *, settings: Settings | None = None) -> ComputeProfile | None:
    """The profile this run sizes from, with the real registry loaders bound (I/O; memoized).

    The impure half of `resolve_profile_source`: it supplies the BigQuery readers and nothing else.
    Split that way so the whole precedence chain stays testable with no cloud, and so the four
    sizing call sites (`submit.sizing_properties`, `dataproc_cluster.cluster_sizing`) each stay a
    pure function that is *handed* a profile rather than one that goes and finds one.

    Returns ``None`` for "size from declared config" — including whenever the registry is
    unreachable, which `resolve_profile_source` degrades to rather than raising.
    """
    profile_cfg = cfg.compute.profile
    if not profile_cfg.consumes_evidence:
        return None

    signature = signature_from_config(cfg)
    key = (
        profile_cfg.source,
        signature.source_table,
        signature.freq,
        signature.n_series,
        cfg.python_runtime,  # ranks the candidates, so two runtimes can resolve differently
        profile_cfg.memory_margin,
        profile_cfg.time_margin,
    )
    if key in _RESOLVED:
        return _RESOLVED[key]

    from ..registry.harvest import discover_harvest_run, read_compute_harvest

    resolved = resolve_profile_source(
        cfg,
        load_run=lambda run_id: read_compute_harvest(run_id, settings=settings),
        discover=lambda want: discover_harvest_run(
            source_table=want.source_table,
            freq=want.freq,
            target_series=want.n_series,
            target_runtime=cfg.python_runtime,
            settings=settings,
        ),
        load_baseline=load_baseline,
    )
    if resolved is not None and resolved.provenance is not None:
        for warning in resolved.provenance.warnings:
            _log.warning("compute profile: %s", warning)
        _log.info(
            "sizing from a %s profile (source=%s, run_id=%s, measured %s)",
            resolved.provenance.basis,
            resolved.provenance.source,
            resolved.provenance.run_id,
            resolved.provenance.measured_at,
        )
    _RESOLVED[key] = resolved
    return resolved


def load_baseline() -> ComputeProfile | None:
    """The shipped, versioned reference profile — ``None`` until one is measured and committed.

    Deliberately a real function with a real caller rather than a ``TODO``: the precedence chain
    already routes through it, so shipping the baseline is dropping a file in beside this and
    parsing it, not rewiring the resolver. It is the last artifact in the profiler's arc because it
    is a *live-proof* claim — committed numbers from a real run, with a version that moves the
    digest and a row in the validation ledger. A baseline whose provenance is a chat message is the
    exact failure that ledger exists to prevent, so there is nothing to load yet.
    """
    return None


def _profilable_models(models: Sequence[str]) -> list[str]:
    """The subset of ``models`` that runs as a Python fit, in the given order (I/O: imports).

    BigQuery-native models (``runtime == "bigquery"``) execute as SQL inside BigQuery: there is no
    process whose cores, RSS or device bytes we could measure, and no slot to size. Measuring one
    would mean issuing a real ``CREATE MODEL`` from a sizing pre-pass, which is both expensive and
    a write. An unresolvable name is dropped rather than raised on — the router has already
    validated the model list, so a failure here is a probe problem, and a probe must not sink a run.
    """
    from ..models import get_model

    keep: list[str] = []
    for name in models:
        try:
            if get_model(name).runtime != "bigquery":
                keep.append(name)
        except Exception:  # noqa: BLE001 - a name we cannot resolve is simply not profilable
            continue
    return keep


def resolve_profile(
    panel: pd.DataFrame,
    cfg: RunConfig,
    models: Sequence[str],
    *,
    params_by_model: dict[str, dict[str, Any]] | None = None,
    measure: Callable[..., MeasuredFit] | None = None,
) -> ComputeProfile | None:
    """Driver-side measurement pre-pass: sample the panel, fit, aggregate → `ComputeProfile`.

    The structural twin of the fleetwide-HPO pre-pass (``resolve_fleetwide_hpo``) — same seam,
    same place, same "resolve once on the driver before fanning out" shape — and it runs *after*
    that one, so tuned hyperparameters can be measured rather than defaults.

    Returns ``None`` when no measurement was taken: profiling is off, the fan-out is below
    ``min_cells``, nothing profilable is in the model list, or the panel yields no usable
    statistics. ``None`` is the signal to size from static config, and every consumer already
    treats it that way (`resources.slot.resource_slot` takes ``profile=None`` as its
    identity case). A profile that *was* taken but measured nothing usable comes back as an empty
    ``ComputeProfile`` rather than ``None`` — the distinction is "we did not look" versus "we
    looked and found nothing", and only the second is worth an audit record.

    **The sample loop is the outer one, deliberately.** Absolute process RSS only grows within a
    process, so a model measured exactly once is charged whatever happened to be imported by the
    time its turn came — the first model measured looks artificially small because the later
    models' libraries were not loaded yet. Cycling every model across every sampled series means
    each model is measured at least once against a fully warm heap, and the ``max`` aggregation
    picks that measurement up. (At ``samples=1`` there is no second pass to warm into, so a
    one-sample budget under-states early models. It also has no length spread and no complexity
    spread; one sample is a smoke-test setting, not a sizing one.)

    **Hyperparameters.** ``params_by_model`` (the fleetwide pre-pass's output) is passed straight
    through, so the measured fit is the fit that will run. Under **per-series** HPO the pre-pass
    instead passes ``{}``: leaving it ``None`` would make each probe call ``worker._resolve_params``
    and run a full Optuna study, turning an 8-sample pre-pass into hundreds of driver-side fits
    with nothing to bound it (see `measure_fit`). Profiling an untuned fit under per-series
    tuning is a known under-statement; running the tuner ``samples x models`` times to avoid it is
    worse.

    ``measure`` injects the measurement function for the offline gate — the default is
    `measure_fit`, and the tests pass a deterministic stand-in so the whole pre-pass, gate
    included, is testable with no fit, no accelerator, and no cloud.
    """
    measure_one = measure or measure_fit
    profilable = _profilable_models(models if models is not None else cfg.models)
    id_col = cfg.data.ts_id_col
    n_series = int(panel[id_col].nunique()) if len(panel) and id_col in panel.columns else 0
    if not profilable or not should_profile(cfg, n_series * len(profilable)):
        return None

    try:
        stats = series_stats(panel, cfg)
    except DataError:
        # A panel we cannot even describe is a panel we cannot sample. Fall back to static
        # config rather than to an arbitrary subset — the run itself will report the real error.
        return None
    sample = select_profile_sample(stats, samples=cfg.compute.profile.samples)
    if not sample:
        return None

    # One frame per sampled id, taken once: slicing the panel inside the loop would rescan it
    # samples x models times for no benefit.
    wanted = {spec.ts_id for spec in sample}
    frames = {
        str(key): frame.reset_index(drop=True)
        for key, frame in panel.groupby(panel[id_col].astype(str), sort=False)
        if str(key) in wanted
    }

    tuned = dict(params_by_model or {})
    untuned: dict[str, Any] | None = (
        {} if (cfg.hpo.enabled and cfg.hpo.granularity == "per_series") else None
    )

    measurements: list[MeasuredFit] = []
    for spec in sample:
        series = frames.get(spec.ts_id)
        if series is None:
            continue
        for model_name in profilable:
            measurements.append(
                measure_one(series, model_name, cfg, params=tuned.get(model_name, untuned))
            )

    profile = build_profile(
        measurements,
        sample=sample,
        memory_margin=cfg.compute.profile.memory_margin,
        time_margin=cfg.compute.profile.time_margin,
    )
    # The one path that measures this run's own data, on this run's own hardware, in-run: the
    # only place `basis="measured"` is unconditional.
    return _with_provenance(
        profile,
        ProfileProvenance(basis="measured", source="in-run", signature=signature_from_config(cfg)),
    )
