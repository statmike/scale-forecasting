"""Monitor a running run and review a finished one — the run-inspection layer, keyed on a run_id.

Two questions, two entry points, both taking only a ``run_id`` (the run *is* its config, so this
layer reads the run's own ``raw_config`` back to recover what it *planned* to do):

- `monitor_run` — *how far along is a run in flight?* Per-family job state, series done vs. the
  expected total, remaining, and mean fit time per family on its chosen runner. Progress is coarse:
  the registry has no live per-series counter — cells land when a family's ``write_cells`` runs
  (often at job end), so done-counts step up per job. The per-job status (from ``v_run_jobs``) is
  the primary live signal; landed-cell counts refine it.

  A registry row is *written by the job*, so a job that dies without writing leaves its row
  ``RUNNING`` forever and the bar simply stops moving. Two things keep that legible. Every family
  carries ``quiet_seconds`` — how long since its last registry signal — which is derived from rows
  already read, costs **no** runtime call, and is a fact rather than a judgement (a family that
  writes its cells at job end is legitimately quiet for its whole run, so a threshold here would
  cry wolf). And ``probe=True`` escalates the non-terminal families to their runtime via
  `probes.reconcile.probe_run`'s reader, attaching a `probes.reconcile.ProbeReport` that says
  whether the job is actually still alive. Registry-first is the default deliberately: a fleet
  poll must never fan native calls, so escalation stays the deliberate per-run drill-down.
- `review_run` — *how did a finished run do, in data-science detail?* The best model per family and
  overall, the full metric panel aggregated across every series (mean + p10/p50/p90), and each
  ensemble's lift over the best base model.
- `calibration_report` — *should you believe the numbers `review_run` just showed you?* Whether the
  point-forecast correction earned its place on this run's own held-out folds, per model, and
  whether the prediction interval achieves the coverage it claims — broken out by horizon step,
  because a band that averages to nominal can still be wrong at both ends.

Same pure/I-O seam as `sdk`: the ``_assemble_*`` functions are pure (turn reader dicts into the
result dataclasses, unit-tested offline), while `monitor_run` / `review_run` are the thin I/O
callers that read the registry (`registry.reads`, `registry.jobs`) and hand off. Plotting
(`plot_progress`, `plot_leaderboard`, `plot_metric_distribution`) is a convenience over the
dataclasses, with matplotlib imported lazily so it never touches the near-instant
``import scale_forecasting`` path. For the wall-clock execution timeline, reuse
`sdk.build_trace_frame` + `sdk.plot_trace`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .capacity import AWAITING_CAPACITY
from .config import RunConfig
from .dag import group_models_by_family
from .device_audit import verdict_label
from .registry.ids import base_family, is_repair_family
from .registry.reads import parse_ts
from .registry.rows import EMITTED, METRIC_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from .probes.reconcile import ProbeReport
    from .settings import Settings

__all__ = [
    "FamilyProgress",
    "RunProgress",
    "ModelReview",
    "BacktestCohort",
    "EnsembleLift",
    "RunReview",
    "ArmComparison",
    "CoveragePoint",
    "CalibrationReport",
    "family_of",
    "monitor_run",
    "review_run",
    "calibration_report",
    "best_overall",
    "best_per_family",
    "ensemble_lift",
    "build_leaderboard_frame",
    "build_predictions_frame",
    "build_hierarchy_frame",
    "build_cohorts_frame",
    "build_calibration_frames",
    "build_ensemble_weights_frame",
    "explain_forecast_frame",
    "plot_progress",
    "plot_leaderboard",
    "plot_metric_distribution",
    "plot_forecasts_frame",
    "plot_hierarchy_frame",
    "plot_calibration",
    "plot_ensemble_weights",
    "plot_forecast_explanation",
]

# Display order for families in a progress/review readout: the base families in DAG order, then the
# downstream ensemble node last. (Mirrors dag._FAMILY_ORDER + the ensemble node it appends.) Repair
# jobs are listed after all of these, in the same order as the families they repair.
_FAMILY_ORDER: tuple[str, ...] = ("statistical", "ml", "deep_learning", "native", "ensemble")


def _family_rank(family: str) -> int:
    """Where a family token sorts in a readout — a repair beside the family it repairs."""
    base = base_family(family)
    return _FAMILY_ORDER.index(base) if base in _FAMILY_ORDER else len(_FAMILY_ORDER)


@dataclass(frozen=True)
class FamilyProgress:
    """One family's live progress on a run: its job state and how many series it has scored.

    ``runtime`` / ``hardware`` name the runner this family resolved to (a Spark family's mean fit
    time is only comparable to another family on the *same* runner). ``n_expected`` is
    ``n_series × models-in-family`` (``None`` when the run's series count isn't known);
    ``n_done`` counts landed full-fit cells; ``fraction`` is their ratio. ``avg_fit_seconds`` is the
    mean per-cell fit time across this family's landed cells; ``runtime_seconds`` is the job's
    wall-clock once it finishes.

    ``last_signal_at`` is the most recent timestamp the job row carries (``ended_at`` →
    ``started_at`` → ``created_at``, first present wins) and ``quiet_seconds`` is its age at read
    time — both ``None`` for a family with no job row yet, or an unparseable timestamp. They are
    reported, never judged: how long a family may legitimately stay quiet depends on the family,
    and the escalation threshold that *does* judge it lives with the probe
    (`probes.reconcile._DEFAULT_STALE_S`), not here.

    ``device_verdict`` is `device_audit`'s answer to whether the accelerator this family paid for
    did any work — ``None`` for every CPU family, and for a GPU family until its job finishes,
    because the audit runs on the driver once the cells are written. It is the only field here that
    is about *cost* rather than progress, and it is on the progress object because this is where a
    reader is already looking at the per-family runtime and hardware.
    """

    family: str
    runtime: str | None
    hardware: str | None
    status: str | None
    models: tuple[str, ...]
    n_expected: int | None
    n_done: int
    fraction: float | None
    avg_fit_seconds: float | None
    runtime_seconds: float | None
    last_signal_at: datetime | None = None
    quiet_seconds: float | None = None
    device_verdict: str | None = None


@dataclass(frozen=True)
class RunProgress:
    """A run's live progress snapshot: header status plus one `FamilyProgress` per family.

    ``status`` is ``None`` when no run exists for the id yet. ``n_done`` / ``n_expected`` /
    ``fraction`` are the run-wide roll-up across families (``n_expected`` and ``fraction`` are
    ``None`` when the series count — hence the denominator — isn't known).

    ``probe`` carries the reconciled `probes.reconcile.ProbeReport` when `monitor_run` was
    called with ``probe=True``, and is ``None`` for the default registry-only read — so a caller
    can always tell "the runtime agreed the job is alive" apart from "we never asked".
    """

    run_id: str
    status: str | None
    n_series: int | None
    families: tuple[FamilyProgress, ...]
    n_done: int
    n_expected: int | None
    fraction: float | None
    probe: ProbeReport | None = None


@dataclass(frozen=True)
class BacktestCohort:
    """How much of the panel one model was actually scored on, split by outcome.

    The context a leaderboard score is meaningless without. A ragged panel does not give every
    series the same backtest: `backtest.make_folds` drops the folds a short series cannot afford,
    so ``full`` (every requested fold), ``reduced`` (some), ``unscored`` (none — too short to
    backtest at all) and ``failed`` are all ordinary outcomes within a single run, and a model
    whose mean error came mostly off ``reduced`` series won an easier contest than its neighbour.
    ``n_not_requested`` counts series whose ``backtest_status`` is NULL, which means the run never
    asked for a backtest rather than that one was attempted and produced nothing.

    ``fold_histogram`` maps achieved fold count → series count, so the shape of the raggedness is
    visible and not just its worst case. Keyed by the achieved count as an ``int``; series with a
    NULL ``n_folds_achieved`` are left out of it (they are still counted in ``n_series``).

    ``refit_modes`` is the other axis of the same question — not how *much* of the panel was
    scored, but on what. It maps ``backtest_refit`` → series count: ``per_fold`` (a fresh fit at
    every origin), ``recondition`` (one fit carried forward on the new observations),
    ``extrapolate`` (one fit, never told what happened next), and ``unsupported`` (a frozen scheme
    was asked for and this model had no seam for it, so those series refit anyway). A model showing
    ``unsupported`` on a frozen run is not answering the same question as its neighbours. Ensemble
    cohorts inherit their members' mode, or ``mixed`` where the members disagreed — see
    `ensemble_run.ensemble_refit_mode`.

    ``staleness_gap`` is what never refreshing the model cost this panel, in the run's decision
    metric, averaged over the series that ran a control arm and weighted by cohort size. Positive
    means refitting earns its keep. ``None`` on the two refit schemes, which run no control arm.
    """

    n_series: int = 0
    n_full: int = 0
    n_reduced: int = 0
    n_unscored: int = 0
    n_failed: int = 0
    n_not_requested: int = 0
    fold_histogram: dict[int, int] = field(default_factory=dict)
    refit_modes: dict[str, int] = field(default_factory=dict)
    staleness_gap: float | None = None


@dataclass(frozen=True)
class ModelReview:
    """One model's (or ensemble pseudo-model's) outcome on a finished run, across all its series.

    ``score`` is the mean of the run's ``decision_metric`` over every series (lower = better;
    ``None`` when no backtest scored it). ``metric_means`` / ``metric_p10`` / ``metric_p50`` /
    ``metric_p90`` carry the full metric panel — the cross-series mean and the 10th/50th/90th
    percentile of each metric in `METRIC_COLUMNS` — so distribution shape reads off the aggregates
    without pulling per-series rows. ``ensemble_id`` is ``None`` for a base model; ``is_ensemble``
    is its convenience flag. ``n_predictions`` is the forecast-row count (0 flags a model that
    scored metadata but produced no forecasts — a fully-failed fit).
    """

    model_type: str
    family: str
    ensemble_id: str | None
    is_ensemble: bool
    compute_engine: str | None
    n_series: int | None
    score: float | None
    metric_means: dict[str, float | None] = field(default_factory=dict)
    metric_p10: dict[str, float | None] = field(default_factory=dict)
    metric_p50: dict[str, float | None] = field(default_factory=dict)
    metric_p90: dict[str, float | None] = field(default_factory=dict)
    mean_fit_seconds: float | None = None
    median_fit_seconds: float | None = None
    no_artifact_rate: float | None = None
    n_predictions: int = 0
    # The cohort behind the score, and the score recomputed so that cohort is held fixed. `score`
    # above is a mean of per-series errors over whatever panel each model happened to get;
    # `pooled_wape` is one WAPE of the whole panel on the holdout fold alone, and
    # `n_comparable_series` is how many series went into it. Two models with different
    # `n_comparable_series` are not yet comparable, whatever their scores say. All three are None on
    # a run with no backtest, and on any run reviewed before these views existed.
    cohort: BacktestCohort | None = None
    pooled_wape: float | None = None
    n_comparable_series: int | None = None


@dataclass(frozen=True)
class EnsembleLift:
    """How much an ensemble improved on the best base model, in the run's decision metric.

    ``lift`` is ``best_base_score − score`` (positive = the ensemble is better, since lower error is
    better); ``lift_pct`` is that as a fraction of the base score. Compares against the single best
    base model overall (`best_overall`), the bar an ensemble has to clear to be worth keeping.
    """

    model_type: str
    score: float
    best_base_model: str
    best_base_score: float
    lift: float
    lift_pct: float | None


@dataclass(frozen=True)
class RunReview:
    """A finished run's data-science review: the leaderboard plus derived bests and ensemble lift.

    ``models`` is every model best-first (lowest ``score``). ``best_per_family`` maps each base
    family to its champion; ``best_overall`` is the single best base model; ``ensembles`` are the
    ensemble pseudo-models; ``ensemble_lift`` scores each against ``best_overall``.
    """

    run_id: str
    status: str | None
    decision_metric: str
    n_series: int | None
    models: tuple[ModelReview, ...]
    best_per_family: dict[str, ModelReview]
    best_overall: ModelReview | None
    ensembles: tuple[ModelReview, ...]
    ensemble_lift: tuple[EnsembleLift, ...]


@dataclass(frozen=True)
class ArmComparison:
    """One model's point-forecast arm choice on a finished run, and what the choice was worth.

    ``margin`` is always "how much the corrected arm beat the raw one by", as a fraction of the raw
    arm's loss in the run's ``decision_metric``, whichever arm the run actually shipped. Negative
    means the correction lost. ``win_rate`` is the share of compared series where it won — the
    number that matters more than the average, because a correction that helps 51% of series by a
    lot and hurts 49% by a lot is a different proposition from one that helps everything a little.

    ``n_raw_arm`` and ``n_auto_decided`` are counts rather than a single label because under
    ``output.point_forecast="auto"`` the series of one model do not have to agree, and that
    disagreement is the feature — reporting one of them as if it spoke for all would hide it.
    """

    model_type: str
    compute_engine: str | None
    interval_calibration: str | None
    n_series: int
    n_raw_arm: int
    n_auto_decided: int
    n_compared: int
    n_corrected_wins: int
    mean_margin: float | None
    median_margin: float | None

    @property
    def win_rate(self) -> float | None:
        """Share of compared series the corrected arm won, or None when nothing was compared."""
        return None if not self.n_compared else self.n_corrected_wins / self.n_compared

    @property
    def raw_arm_rate(self) -> float | None:
        """Share of this model's series shipping the raw arm, or None when there are none.

        1.0 or 0.0 under a fleetwide setting; anything between means selection actually split the
        model's series, which is the thing `auto` exists to do and the thing worth looking at.
        """
        return None if not self.n_series else self.n_raw_arm / self.n_series


@dataclass(frozen=True)
class CoveragePoint:
    """Achieved interval coverage at one horizon step for one model."""

    model_type: str
    horizon_step: int
    n: int
    coverage: float | None
    mean_width: float | None


@dataclass(frozen=True)
class CalibrationReport:
    """What the run's own data says about its point forecast and its prediction interval.

    Two questions a forecaster asks of any system that corrects a model's output, answered from the
    run's held-out folds rather than from a claim in a doc:

    * **Was the correction worth applying?** ``arms``, per model — win rate and margin.
    * **Does the band mean what it says?** ``coverage`` per horizon step against ``nominal``,
      which is the width of the run's quantile set (0.8 for the shipped default of 0.1/0.5/0.9).

    ``worst_step`` is the single largest deviation from nominal across every model and step: a run
    whose average coverage looks fine while one end of the horizon is badly wrong should not read
    as healthy, and an average will always say it does.
    """

    run_id: str
    decision_metric: str
    nominal_coverage: float
    arms: tuple[ArmComparison, ...]
    coverage: tuple[CoveragePoint, ...]

    @property
    def mean_coverage(self) -> float | None:
        """Row-count-weighted achieved coverage across every model and step."""
        pts = [p for p in self.coverage if p.coverage is not None and p.n]
        total = sum(p.n for p in pts)
        return None if not total else sum(p.coverage * p.n for p in pts) / total  # type: ignore[misc]

    @property
    def worst_step(self) -> CoveragePoint | None:
        """The model/step furthest from nominal — the thing an average is designed to hide."""
        pts = [p for p in self.coverage if p.coverage is not None]
        return max(pts, key=lambda p: abs(p.coverage - self.nominal_coverage), default=None)  # type: ignore[arg-type]


def _assemble_calibration(
    run_id: str,
    decision_metric: str,
    nominal_coverage: float,
    arm_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> CalibrationReport:
    """Compose a `CalibrationReport` from the two reader payloads (pure)."""
    arms = tuple(
        ArmComparison(
            model_type=r["model_type"],
            compute_engine=r.get("compute_engine"),
            interval_calibration=r.get("interval_calibration"),
            n_series=int(r.get("n_series") or 0),
            n_raw_arm=int(r.get("n_raw_arm") or 0),
            n_auto_decided=int(r.get("n_auto_decided") or 0),
            n_compared=int(r.get("n_compared") or 0),
            n_corrected_wins=int(r.get("n_corrected_wins") or 0),
            mean_margin=_num(r.get("mean_margin")),
            median_margin=_num(r.get("median_margin")),
        )
        for r in arm_rows
    )
    coverage = tuple(
        CoveragePoint(
            model_type=r["model_type"],
            horizon_step=int(r["horizon_step"]),
            n=int(r.get("n") or 0),
            coverage=_num(r.get("coverage")),
            mean_width=_num(r.get("mean_width")),
        )
        for r in coverage_rows
    )
    return CalibrationReport(
        run_id=run_id,
        decision_metric=decision_metric,
        nominal_coverage=nominal_coverage,
        arms=arms,
        coverage=coverage,
    )


def calibration_report(run_id: str, *, settings: Settings | None = None) -> CalibrationReport:
    """Read a finished run's point-forecast and interval diagnostic.

    Reads the config (for the decision metric and the quantile set that defines nominal coverage),
    the per-model arm rollup (`registry.reads.read_arm_comparison`) and the per-step coverage panel
    (`registry.reads.read_coverage_by_step`), then composes via `_assemble_calibration`.
    """
    from .registry.reads import read_arm_comparison, read_coverage_by_step, read_run_config

    raw = read_run_config(run_id, settings=settings)
    cfg = RunConfig.model_validate(raw) if raw else None
    decision_metric = cfg.backtest.decision_metric if cfg else "wape"
    # Nominal is the span of the shipped quantile set, not a constant: a run that widened its
    # quantiles is not under-covering just because it left the default behind.
    from .models.base_model import DEFAULT_QUANTILES

    nominal = max(DEFAULT_QUANTILES) - min(DEFAULT_QUANTILES)
    return _assemble_calibration(
        run_id,
        decision_metric,
        nominal,
        read_arm_comparison(run_id, settings=settings),
        read_coverage_by_step(run_id, settings=settings),
    )


def _num(value: Any) -> float | None:
    """Coerce a BigQuery numeric to a finite ``float``, mapping ``None``/``NaN``/non-numeric to
    ``None`` — so downstream sorts and ratios never trip on a ``NaN`` an undefined metric leaves."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def family_of(model_type: str, ensemble_id: str | None = None) -> str:
    """The family a result row belongs to: ``"ensemble"`` if ``ensemble_id`` is set, else the
    model's registered ``family`` (``"unknown"`` for a name this build no longer registers)."""
    if ensemble_id is not None:
        return "ensemble"
    from .errors import ModelError
    from .models import get_model

    try:
        return get_model(model_type).family
    except ModelError:
        return "unknown"


# --- monitor (live) ------------------------------------------------------------


def _last_signal(job_row: dict[str, Any]) -> datetime | None:
    """The most recent timestamp a ``v_run_jobs`` row carries — the family's last registry signal.

    ``ended_at`` → ``started_at`` → ``created_at``, first present wins (they are written in that
    order, so the first present one is the latest). ``None`` when the row has none, or none of them
    parses. Shared by `_assemble_progress`'s ``quiet_seconds`` and, through it, the probe's
    escalation grace — so "how long has this been quiet" has exactly one definition.
    """
    for key in ("ended_at", "started_at", "created_at"):
        ts = parse_ts(job_row.get(key))
        if ts is not None:
            return ts
    return None


def _assemble_progress(
    run_id: str,
    summary: dict[str, Any] | None,
    cfg: RunConfig | None,
    job_rows: list[dict[str, Any]],
    progress_rows: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> RunProgress:
    """Compose a `RunProgress` from a run's header, config, job rows, and landed-cell counts (pure).

    The config gives the *denominator* (models per family × series count = expected cells); the job
    rows give each family's runner + status; the progress rows give landed counts and mean fit time.
    With no config (run never ran) this is an empty snapshot carrying just the header status.
    ``now`` is the clock the per-family ``quiet_seconds`` is measured against — injectable so the
    age arithmetic is deterministic offline, and so a caller that probes in the same pass
    (`probes.reconcile._read_and_probe`) reconciles every family against one instant.
    """
    status = (summary or {}).get("status")
    at = now or datetime.now(UTC)
    if cfg is None:
        return RunProgress(run_id, status, None, (), 0, None, None)

    n_series = (summary or {}).get("n_series") or cfg.data.series_limit
    grouped = group_models_by_family(cfg)  # base families → their models, in DAG order
    models_by_family: dict[str, tuple[str, ...]] = {f: tuple(m) for f, m in grouped.items()}
    if cfg.ensemble.enabled:
        models_by_family["ensemble"] = tuple(f"ensemble_{s}" for s in cfg.ensemble.strategies)

    jobs_by_family = {r["family"]: r for r in job_rows}

    # Fold the per-model progress rows up to their family: total landed cells + a cell-weighted mean
    # fit time (Σ mean·n / Σ n is the true mean across cells, robust to uneven per-model counts).
    done: dict[str, int] = {}
    fit_num: dict[str, float] = {}
    fit_den: dict[str, int] = {}
    for r in progress_rows:
        fam = family_of(r["model_type"], r.get("ensemble_id"))
        n = int(r.get("n_cells_done") or 0)
        done[fam] = done.get(fam, 0) + n
        mean_fit = _num(r.get("mean_fit_seconds"))
        if mean_fit is not None and n:
            fit_num[fam] = fit_num.get(fam, 0.0) + mean_fit * n
            fit_den[fam] = fit_den.get(fam, 0) + n

    families: list[FamilyProgress] = []
    for fam in _FAMILY_ORDER:
        if fam not in models_by_family:
            continue
        fam_models = models_by_family[fam]
        n_expected = n_series * len(fam_models) if n_series is not None else None
        n_done = done.get(fam, 0)
        job = jobs_by_family.get(fam, {})
        signal = _last_signal(job)
        families.append(
            FamilyProgress(
                family=fam,
                runtime=job.get("runtime"),
                hardware=job.get("hardware"),
                status=job.get("status"),
                models=fam_models,
                n_expected=n_expected,
                n_done=n_done,
                fraction=(n_done / n_expected if n_expected else None),
                avg_fit_seconds=(fit_num[fam] / fit_den[fam] if fit_den.get(fam) else None),
                runtime_seconds=_num(job.get("runtime_seconds")),
                last_signal_at=signal,
                quiet_seconds=((at - signal).total_seconds() if signal is not None else None),
                device_verdict=job.get("device_verdict"),
            )
        )

    total_done = sum(f.n_done for f in families)
    expected_known = [f.n_expected for f in families if f.n_expected is not None]
    total_expected = sum(expected_known) if len(expected_known) == len(families) else None
    fraction = (total_done / total_expected) if total_expected else None

    # Repair jobs are listed after the families they repair, and after the run totals are taken.
    #
    # A repair is a *job*, not a family: it re-asks a narrowed subset of one family's cells
    # (`dag.narrow_to_models`), and its ``run_jobs`` row carries no record of how large that subset
    # was. So it contributes a job state — runtime, status, quiet time — and deliberately no
    # denominator. Giving it the whole family's expected count would report a finished forty-cell
    # repair as 0.04% done forever, and would poison the run-level fraction with a second copy of a
    # denominator already counted once. Leaving it out of the list entirely was the other option and
    # is worse: `probes.reconcile` and `--cancel` read this snapshot and nothing else, so an omitted
    # repair is a live job neither of them can see or stop.
    #
    # Landed cells are not attributed to it either. The progress rows are per *model*, and a
    # repaired cell and an original cell of the same model are the same row to that query — so a
    # split would have to be invented, and `n_done` on the base family already counts both.
    repairs: list[FamilyProgress] = []
    for r in sorted(job_rows, key=lambda r: _family_rank(str(r.get("family") or ""))):
        fam = str(r.get("family") or "")
        if not is_repair_family(fam):
            continue
        signal = _last_signal(r)
        repairs.append(
            FamilyProgress(
                family=fam,
                runtime=r.get("runtime"),
                hardware=r.get("hardware"),
                status=r.get("status"),
                models=models_by_family.get(base_family(fam), ()),
                n_expected=None,
                n_done=0,
                fraction=None,
                avg_fit_seconds=None,
                runtime_seconds=_num(r.get("runtime_seconds")),
                last_signal_at=signal,
                quiet_seconds=((at - signal).total_seconds() if signal is not None else None),
                device_verdict=r.get("device_verdict"),
            )
        )

    return RunProgress(
        run_id=run_id,
        status=status,
        n_series=n_series,
        families=tuple(families + repairs),
        n_done=total_done,
        n_expected=total_expected,
        fraction=fraction,
    )


def monitor_run(
    run_id: str,
    *,
    probe: bool = False,
    stale_after_s: float | None = None,
    settings: Settings | None = None,
) -> RunProgress:  # pragma: no cover - GCP I/O
    """Read a run's live progress: header status + per-family job state + series done vs. expected.

    Reads the run's header (`registry.reads.read_run_summary`), its config
    (`registry.reads.read_run_config`, for the expected-work denominator), its jobs
    (`registry.jobs.read_run_jobs`) and its landed-cell counts (`registry.reads.read_progress`),
    then composes them via `_assemble_progress`. Poll it while a run is in flight; returns a
    status-only snapshot when the run id has never run. Every family carries ``quiet_seconds``
    either way — the "is this bar frozen or just coarse" signal, free because it comes off rows
    already read.

    ``probe=True`` additionally escalates the run's non-terminal jobs to their runtime and attaches
    the reconciled `probes.reconcile.ProbeReport` as ``RunProgress.probe`` — the answer to *is this
    job still alive*, which the registry alone cannot give. It shares one pass of reads with the
    registry side (`probes.reconcile._read_and_probe`), so probing costs the native calls and not a
    second set of queries; an already-terminal run short-circuits and touches no runtime at all.
    ``stale_after_s`` overrides the probe's startup grace (see `probes.reconcile.probe_run`) and
    is ignored when ``probe`` is ``False``.
    """
    from .registry.jobs import read_run_jobs
    from .registry.reads import read_progress, read_run_config, read_run_summary

    if probe:
        from .probes.reconcile import _read_and_probe
        from .settings import Settings as _Settings

        s = settings if settings is not None else _Settings.resolve()
        progress, report, _rows = _read_and_probe(
            run_id, job=None, settings=s, stale_after_s=stale_after_s
        )
        return replace(progress, probe=report)

    summary = read_run_summary(run_id, settings=settings)
    raw = read_run_config(run_id, settings=settings)
    cfg = RunConfig.model_validate(raw) if raw else None
    if cfg is None:
        return _assemble_progress(run_id, summary, None, [], [])
    job_rows = read_run_jobs(run_id, settings=settings)
    progress_rows = read_progress(run_id, settings=settings)
    return _assemble_progress(run_id, summary, cfg, job_rows, progress_rows)


# --- review (finished) ---------------------------------------------------------


def best_overall(models: list[ModelReview] | tuple[ModelReview, ...]) -> ModelReview | None:
    """The single best base model (lowest ``score``); ``None`` if no base model was scored."""
    scored = [m for m in models if not m.is_ensemble and m.score is not None]
    return min(scored, key=lambda m: m.score) if scored else None


def best_per_family(
    models: list[ModelReview] | tuple[ModelReview, ...],
) -> dict[str, ModelReview]:
    """Map each base family to its champion (lowest-``score`` scored model in that family)."""
    best: dict[str, ModelReview] = {}
    for m in models:
        if m.is_ensemble or m.score is None:
            continue
        cur = best.get(m.family)
        if cur is None or m.score < cur.score:
            best[m.family] = m
    return best


def ensemble_lift(
    models: list[ModelReview] | tuple[ModelReview, ...],
) -> list[EnsembleLift]:
    """Score each ensemble's improvement over the best base model, best lift first.

    Empty when there is no scored base model to compare against, or no scored ensemble.
    """
    champ = best_overall(models)
    if champ is None:
        return []
    lifts = [
        EnsembleLift(
            model_type=m.model_type,
            score=m.score,
            best_base_model=champ.model_type,
            best_base_score=champ.score,
            lift=champ.score - m.score,
            lift_pct=((champ.score - m.score) / champ.score if champ.score else None),
        )
        for m in models
        if m.is_ensemble and m.score is not None
    ]
    return sorted(lifts, key=lambda x: x.lift, reverse=True)


_STATUS_FIELDS: dict[str, str] = {
    "full": "n_full",
    "reduced": "n_reduced",
    "unscored": "n_unscored",
    "failed": "n_failed",
}


def _cohorts_by_model(
    coverage_rows: list[dict[str, Any]],
) -> dict[tuple[str, str | None], BacktestCohort]:
    """Fold `registry.reads.read_backtest_coverage` rows up into one `BacktestCohort` per model.

    The view emits one row per
    ``(model, ensemble_id, backtest_status, n_folds_achieved, backtest_refit)``; a model with a
    ragged panel therefore arrives as several rows that have to be summed back together. A
    ``backtest_status`` this function does not recognise still lands in ``n_series`` — the total is
    the panel, so an unfamiliar status must not quietly vanish from it.

    ``staleness_gap`` is the one field that is averaged rather than summed, and it is weighted by
    each row's series count: the view already averaged within a cohort, so an unweighted mean of
    the cohorts would let a two-series row count as much as a two-thousand-series one.
    """
    totals: dict[tuple[str, str | None], dict[str, Any]] = {}
    weighted: dict[tuple[str, str | None], tuple[float, int]] = {}
    for row in coverage_rows:
        key = (row["model_type"], row.get("ensemble_id"))
        acc = totals.setdefault(key, {"n_series": 0, "fold_histogram": {}, "refit_modes": {}})
        n = int(row.get("n_series") or 0)
        acc["n_series"] += n
        field_name = _STATUS_FIELDS.get(row.get("backtest_status") or "", "n_not_requested")
        acc[field_name] = acc.get(field_name, 0) + n
        folds = row.get("n_folds_achieved")
        if folds is not None:
            hist = acc["fold_histogram"]
            hist[int(folds)] = hist.get(int(folds), 0) + n
        refit = row.get("backtest_refit")
        if refit is not None:
            modes = acc["refit_modes"]
            modes[str(refit)] = modes.get(str(refit), 0) + n
        gap = _num(row.get("mean_staleness_gap"))
        if gap is not None and n:
            total, count = weighted.get(key, (0.0, 0))
            weighted[key] = (total + gap * n, count + n)
    return {
        key: BacktestCohort(
            **{
                **acc,
                "fold_histogram": dict(sorted(acc["fold_histogram"].items())),
                "refit_modes": dict(sorted(acc["refit_modes"].items())),
                "staleness_gap": (weighted[key][0] / weighted[key][1] if key in weighted else None),
            }
        )
        for key, acc in totals.items()
    }


def _attach_cohorts(
    models: list[ModelReview],
    coverage_rows: list[dict[str, Any]],
    comparable_rows: list[dict[str, Any]],
) -> list[ModelReview]:
    """Hang the cohort counts and the holdout-pooled score on each `ModelReview` (pure).

    Both inputs are keyed on ``(model_type, ensemble_id)`` — the same key the leaderboard and the
    aggregates join on — and both are optional: a model missing from either keeps ``None`` there
    rather than a zero, because "no backtest coverage row" and "a cohort of zero series" are
    different facts and only the first one is true of a run that never backtested.
    """
    cohorts = _cohorts_by_model(coverage_rows)
    comparable = {(r["model_type"], r.get("ensemble_id")): r for r in comparable_rows}
    out: list[ModelReview] = []
    for m in models:
        key = (m.model_type, m.ensemble_id)
        row = comparable.get(key, {})
        out.append(
            replace(
                m,
                cohort=cohorts.get(key),
                pooled_wape=_num(row.get("pooled_wape")),
                n_comparable_series=row.get("n_series"),
            )
        )
    return out


def _model_review_from_aggregate(
    agg: dict[str, Any],
    decision_metric: str,
    lb_row: dict[str, Any],
    prediction_counts: dict[str, int],
) -> ModelReview:
    """Turn one `registry.reads.read_metric_aggregates` row (+ its leaderboard match) into a
    `ModelReview` — the full metric panel plus the leaderboard-only fields (artifact rate,
    median fit time)."""
    ensemble_id = agg.get("ensemble_id")
    model_type = agg["model_type"]
    means = {m: _num(agg.get(f"mean_{m}")) for m in METRIC_COLUMNS}
    p10 = {m: _num(agg.get(f"p10_{m}")) for m in METRIC_COLUMNS}
    p50 = {m: _num(agg.get(f"p50_{m}")) for m in METRIC_COLUMNS}
    p90 = {m: _num(agg.get(f"p90_{m}")) for m in METRIC_COLUMNS}
    return ModelReview(
        model_type=model_type,
        family=family_of(model_type, ensemble_id),
        ensemble_id=ensemble_id,
        is_ensemble=ensemble_id is not None,
        compute_engine=agg.get("compute_engine"),
        n_series=agg.get("n_series"),
        score=means.get(decision_metric),
        metric_means=means,
        metric_p10=p10,
        metric_p50=p50,
        metric_p90=p90,
        mean_fit_seconds=_num(agg.get("mean_fit_seconds")),
        median_fit_seconds=_num(lb_row.get("median_fit_seconds")),
        no_artifact_rate=_num(lb_row.get("no_artifact_rate")),
        n_predictions=int(prediction_counts.get(model_type, 0)),
    )


def _model_review_from_leaderboard(
    row: dict[str, Any], prediction_counts: dict[str, int]
) -> ModelReview:
    """Fallback `ModelReview` from a leaderboard row alone (no backtest aggregates): WAPE as the
    score, empty metric panel."""
    ensemble_id = row.get("ensemble_id")
    model_type = row["model_type"]
    return ModelReview(
        model_type=model_type,
        family=family_of(model_type, ensemble_id),
        ensemble_id=ensemble_id,
        is_ensemble=ensemble_id is not None,
        compute_engine=row.get("compute_engine"),
        n_series=row.get("n_cells"),
        score=_num(row.get("mean_wape")),
        mean_fit_seconds=_num(row.get("median_fit_seconds")),
        median_fit_seconds=_num(row.get("median_fit_seconds")),
        no_artifact_rate=_num(row.get("no_artifact_rate")),
        n_predictions=int(prediction_counts.get(model_type, 0)),
    )


def _assemble_review(
    run_id: str,
    summary: dict[str, Any] | None,
    decision_metric: str,
    n_series: int | None,
    leaderboard_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    prediction_counts: dict[str, int],
    coverage_rows: list[dict[str, Any]] | None = None,
    comparable_rows: list[dict[str, Any]] | None = None,
) -> RunReview:
    """Compose a `RunReview` from the leaderboard, metric aggregates, and prediction counts (pure).

    Aggregates are the primary source (full metric panel); the leaderboard supplies artifact rate +
    median fit time and is the fallback when a run had no backtest (no aggregates), scoring on WAPE.
    ``coverage_rows`` and ``comparable_rows`` are the cohort context beside the ranking
    (`registry.reads.read_backtest_coverage`, `registry.reads.read_comparable_leaderboard`);
    both default to empty so an older caller composes exactly as it did.
    """
    lb = {(r["model_type"], r.get("ensemble_id")): r for r in leaderboard_rows}
    if aggregate_rows:
        models = [
            _model_review_from_aggregate(
                agg,
                decision_metric,
                lb.get((agg["model_type"], agg.get("ensemble_id")), {}),
                prediction_counts,
            )
            for agg in aggregate_rows
        ]
    else:
        models = [_model_review_from_leaderboard(r, prediction_counts) for r in leaderboard_rows]
    models = _attach_cohorts(models, coverage_rows or [], comparable_rows or [])
    # best-first: scored models by ascending error, unscored last (stable within group).
    models.sort(key=lambda m: (m.score is None, m.score if m.score is not None else 0.0))
    return RunReview(
        run_id=run_id,
        status=(summary or {}).get("status"),
        decision_metric=decision_metric,
        n_series=n_series,
        models=tuple(models),
        best_per_family=best_per_family(models),
        best_overall=best_overall(models),
        ensembles=tuple(m for m in models if m.is_ensemble),
        ensemble_lift=tuple(ensemble_lift(models)),
    )


def review_run(
    run_id: str, *, settings: Settings | None = None
) -> RunReview:  # pragma: no cover - GCP I/O
    """Read a finished run's data-science review: bests per family/overall + ensemble lift + panel.

    Reads the header (`registry.reads.read_run_summary`), the config (for the decision metric and
    series count), the leaderboard (`registry.reads.read_leaderboard`), the cross-series aggregates
    (`registry.reads.read_metric_aggregates`), per-model prediction counts
    (`registry.reads.read_prediction_counts`), and the two cohort reads that say whether the
    ranking is comparable at all — `registry.reads.read_backtest_coverage` and
    `registry.reads.read_comparable_leaderboard` — then composes via `_assemble_review`.
    """
    from .registry.reads import (
        read_backtest_coverage,
        read_comparable_leaderboard,
        read_leaderboard,
        read_metric_aggregates,
        read_prediction_counts,
        read_run_config,
        read_run_summary,
    )

    summary = read_run_summary(run_id, settings=settings)
    raw = read_run_config(run_id, settings=settings)
    cfg = RunConfig.model_validate(raw) if raw else None
    decision_metric = cfg.backtest.decision_metric if cfg else "wape"
    n_series = (summary or {}).get("n_series") or (cfg.data.series_limit if cfg else None)
    leaderboard_rows = read_leaderboard(run_id, settings=settings)
    aggregate_rows = read_metric_aggregates(run_id, settings=settings)
    prediction_counts = read_prediction_counts(run_id, settings=settings)
    coverage_rows = read_backtest_coverage(run_id, settings=settings)
    comparable_rows = read_comparable_leaderboard(run_id, settings=settings)
    return _assemble_review(
        run_id,
        summary,
        decision_metric,
        n_series,
        leaderboard_rows,
        aggregate_rows,
        prediction_counts,
        coverage_rows,
        comparable_rows,
    )


# --- plots (lazy matplotlib) ---------------------------------------------------
#
# Palette validated with the dataviz skill's checker (do not eyeball / re-pick by taste):
#   - base vs ensemble #0072B2/#E69F00 — CVD ΔE 29.2 (PASS); the orange's sub-3:1 surface contrast
#     is relieved by the direct value label every bar carries.
#   - status green/blue/vermillion #009E73/#0072B2/#D55E00 (PASS separation); pending is gray
#     #999999 by design (a status, not a categorical hue) and every bar is annotated with its status
#     text, so identity is never colour-alone.
#   - AWAITING_CAPACITY and EMITTED share PENDING's gray on purpose rather than taking a seventh and
#     eighth hue: all three are the same fact to a reader scanning the chart (this family has not
#     started), the status text on the bar carries the difference, and a new hue would have to be
#     re-validated for CVD separation against six existing ones to add nothing.
_BASE_COLOR = "#0072B2"
_ENSEMBLE_COLOR = "#E69F00"
_STATUS_COLORS: dict[str | None, str] = {
    "COMPLETED": "#009E73",
    "RUNNING": "#0072B2",
    "PENDING": "#999999",
    AWAITING_CAPACITY: "#999999",
    EMITTED: "#999999",
    "FAILED": "#D55E00",
    "PARTIAL": "#E69F00",
    "CANCELLED": "#CC79A7",
}
_STATUS_DEFAULT = "#999999"


def _human_age(seconds: float) -> str:
    """A quiet-time as a short human age — ``42s`` / ``22m`` / ``3.1h``, for a bar-end label."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _reportable_verdicts(progress: RunProgress) -> dict[str, str]:
    """``{family: verdict}`` for the probed families whose verdict is worth showing (pure).

    Empty when no probe ran. ``TRUST_REGISTRY`` is dropped: it is what the bar's own status colour
    already says (terminal, or never launched), so surfacing it would put a word on every row and
    bury the two that matter — ``LOST`` and ``STALE_REGISTRY``.
    """
    if progress.probe is None:
        return {}
    from .probes.vocabulary import VERDICT_TRUST_REGISTRY

    return {
        v.family: v.verdict for v in progress.probe.families if v.verdict != VERDICT_TRUST_REGISTRY
    }


def plot_progress(progress: RunProgress, *, ax: Any = None, title: str | None = None) -> Any:
    """Render a `RunProgress` as a per-family progress bar chart and return the matplotlib ``Axes``.

    One horizontal bar per family — length is the fraction of expected cells that have landed,
    colour is the family's job status (`_STATUS_COLORS`) — with the ``done/expected`` count and
    status labelled at the bar end (the label doubles as the status's secondary encoding). Families
    keep DAG order, ensemble last. matplotlib imports lazily so it never touches the package import;
    an empty run renders an empty titled axes rather than raising.

    A non-terminal family also gets its ``quiet_seconds`` in the label (``quiet 22m``): the bar of a
    job that died mid-run stops moving and its status stays ``RUNNING``, so without this a dead run
    and a slow one are pixel-identical. It is reported as an age, not flagged against a threshold —
    a family that writes its cells at job end is legitimately quiet the whole time, and a
    cry-wolf marker teaches the reader to ignore it. When a `probes.reconcile.ProbeReport` is
    attached (``monitor_run(probe=True)``), its verdict replaces the age for the families it
    covers, since a live reading beats an inference from silence.

    A family that finished with a device verdict gets two more words at the end of its label
    (``gpu used`` / ``gpu idle`` / ``no gpu``). A GPU family's bar is otherwise indistinguishable
    from a CPU family's, which is the whole reason the accelerator went twenty-one jobs without
    anyone noticing it was doing nothing.
    """
    import matplotlib.pyplot as plt

    heading = title or f"{progress.run_id} — {progress.status or 'unknown'}"
    if progress.fraction is not None:
        heading += f" — {progress.fraction:.0%} of cells landed"
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(2.0, 0.6 * len(progress.families) + 1)))
    ax.set_title(heading if progress.families else f"{heading} (no families)")
    if not progress.families:
        return ax

    fams = list(reversed(progress.families))  # first family on top
    ys = range(len(fams))
    ax.barh(
        list(ys),
        [f.fraction if f.fraction is not None else 0.0 for f in fams],
        height=0.6,
        color=[_STATUS_COLORS.get(f.status, _STATUS_DEFAULT) for f in fams],
    )
    verdicts = _reportable_verdicts(progress)
    for y, f in zip(ys, fams, strict=True):
        expected = f.n_expected if f.n_expected is not None else "?"
        label = f"{f.n_done}/{expected} · {f.status or 'pending'}"
        if (verdict := verdicts.get(f.family)) is not None:
            label += f" · {verdict.lower().replace('_', ' ')}"
        elif (f.status or "").upper() == "RUNNING" and f.quiet_seconds is not None:
            label += f" · quiet {_human_age(f.quiet_seconds)}"
        # Appended rather than chained into the ladder above: liveness and cost are different
        # questions, and a finished GPU family has an answer to both.
        if (device := verdict_label(f.device_verdict)) is not None:
            label += f" · {device}"
        ax.text(
            (f.fraction if f.fraction is not None else 0.0) + 0.01,
            y,
            label,
            va="center",
            fontsize=9,
        )
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f.family for f in fams])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("fraction of expected cells landed")
    return ax


def plot_leaderboard(
    review: RunReview, *, ax: Any = None, top: int | None = None, title: str | None = None
) -> Any:
    """Render a `RunReview`'s scored models as a ranked bar chart; return the matplotlib ``Axes``.

    One horizontal bar per model, best (lowest decision-metric error) on top, coloured base vs.
    ensemble (`_BASE_COLOR`/`_ENSEMBLE_COLOR`) with a legend when both are present and the score
    labelled at each bar end. Unscored models (no backtest) are dropped. ``top`` caps the bar count.
    matplotlib imports lazily; a review with no scored model renders an empty titled axes.
    """
    import matplotlib.pyplot as plt

    scored = [m for m in review.models if m.score is not None]
    if top is not None:
        scored = scored[:top]
    heading = title or f"{review.run_id} — model leaderboard ({review.decision_metric})"
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(2.0, 0.5 * len(scored) + 1)))
    ax.set_title(heading if scored else f"{heading} (no scored models)")
    if not scored:
        return ax

    ranked = list(reversed(scored))  # best on top
    ys = range(len(ranked))
    ax.barh(
        list(ys),
        [m.score for m in ranked],
        height=0.6,
        color=[_ENSEMBLE_COLOR if m.is_ensemble else _BASE_COLOR for m in ranked],
    )
    for y, m in zip(ys, ranked, strict=True):
        ax.text(m.score, y, f" {m.score:.4g}", va="center", fontsize=9)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([m.model_type for m in ranked])
    ax.set_xlabel(f"mean {review.decision_metric} (lower is better)")
    if any(m.is_ensemble for m in ranked) and any(not m.is_ensemble for m in ranked):
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color=_BASE_COLOR, label="base model"),
                Patch(color=_ENSEMBLE_COLOR, label="ensemble"),
            ],
            loc="lower right",
            fontsize=9,
        )
    return ax


def plot_metric_distribution(
    review: RunReview, *, metric: str | None = None, ax: Any = None, title: str | None = None
) -> Any:
    """Render each model's cross-series spread for one metric (p10–p90 range, p50 dot) as ``Axes``.

    One row per scored model: a thin line from the 10th to the 90th cross-series percentile with a
    marker at the median, coloured base vs. ensemble — the distribution shape (not just the mean) of
    a metric across every series, read straight off the server-side aggregates so it holds at scale.
    ``metric`` defaults to the run's decision metric. matplotlib imports lazily; a review with no
    aggregated percentiles renders an empty titled axes.
    """
    import matplotlib.pyplot as plt

    chosen = metric or review.decision_metric
    rows = [m for m in review.models if m.metric_p50.get(chosen) is not None]
    heading = title or f"{review.run_id} — {chosen} across series (p10–p50–p90)"
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(2.0, 0.5 * len(rows) + 1)))
    ax.set_title(heading if rows else f"{heading} (no aggregated percentiles)")
    if not rows:
        return ax

    ordered = sorted(rows, key=lambda m: m.metric_p50[chosen], reverse=True)  # best (low) on top
    for y, m in enumerate(ordered):
        color = _ENSEMBLE_COLOR if m.is_ensemble else _BASE_COLOR
        lo = m.metric_p10.get(chosen)
        hi = m.metric_p90.get(chosen)
        mid = m.metric_p50[chosen]
        if lo is not None and hi is not None:
            ax.hlines(y, lo, hi, color=color, linewidth=2)
        ax.plot(mid, y, "o", color=color, markersize=8)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([m.model_type for m in ordered])
    ax.set_xlabel(f"{chosen} (p10–p90 range, dot = median)")
    if any(m.is_ensemble for m in ordered) and any(not m.is_ensemble for m in ordered):
        from matplotlib.lines import Line2D

        ax.legend(
            handles=[
                Line2D([0], [0], color=_BASE_COLOR, marker="o", label="base model"),
                Line2D([0], [0], color=_ENSEMBLE_COLOR, marker="o", label="ensemble"),
            ],
            loc="lower right",
            fontsize=9,
        )
    return ax


# --- tabular frames & forecast/hierarchy plots ---------------------------------


def build_leaderboard_frame(review: RunReview, *, all_metrics: bool = False) -> pd.DataFrame:
    """Convert a `RunReview` into a ranked pandas ``DataFrame`` (pure, offline).

    Includes core ranking columns (`rank`, `model_type`, `family`, `is_ensemble`, `compute_engine`,
    `n_series`, `score`, `pooled_wape`, `n_comparable_series`), key accuracy and interval metrics
    (`mean_wape`, `mean_smape`, `mean_mase`, `mean_rmsse`, `mean_mae`, `mean_rmse`,
    `mean_coverage`, `mean_interval_score`), runtime/reliability (`mean_fit_seconds`,
    `median_fit_seconds`, `no_artifact_rate`, `n_predictions`), and ensemble lift
    (`lift_vs_best_base`, `lift_pct`). When ``all_metrics=True``, appends ``mean_<metric>`` and
    ``p50_<metric>`` for every metric in `METRIC_COLUMNS`.
    """
    import pandas as pd

    lift_map = {(e.model_type): e for e in review.ensemble_lift}
    records: list[dict[str, Any]] = []
    for idx, m in enumerate(review.models, start=1):
        lift = lift_map.get(m.model_type)
        row: dict[str, Any] = {
            "rank": idx,
            "model_type": m.model_type,
            "family": m.family,
            "is_ensemble": m.is_ensemble,
            "ensemble_id": m.ensemble_id,
            "compute_engine": m.compute_engine,
            "n_series": m.n_series,
            "decision_metric": review.decision_metric,
            "score": m.score,
            "pooled_wape": m.pooled_wape,
            "n_comparable_series": m.n_comparable_series,
            "mean_wape": m.metric_means.get("wape"),
            "mean_smape": m.metric_means.get("smape"),
            "mean_mase": m.metric_means.get("mase"),
            "mean_rmsse": m.metric_means.get("rmsse"),
            "mean_mae": m.metric_means.get("mae"),
            "mean_rmse": m.metric_means.get("rmse"),
            "mean_coverage": m.metric_means.get("coverage"),
            "mean_interval_score": m.metric_means.get("interval_score"),
            "mean_fit_seconds": m.mean_fit_seconds,
            "median_fit_seconds": m.median_fit_seconds,
            "no_artifact_rate": m.no_artifact_rate,
            "n_predictions": m.n_predictions,
            "lift_vs_best_base": lift.lift if lift is not None else None,
            "lift_pct": lift.lift_pct if lift is not None else None,
        }
        if all_metrics:
            for col in METRIC_COLUMNS:
                row[f"mean_{col}"] = m.metric_means.get(col)
                row[f"p50_{col}"] = m.metric_p50.get(col)
        records.append(row)
    if not records:
        return pd.DataFrame(
            columns=[
                "rank",
                "model_type",
                "family",
                "is_ensemble",
                "ensemble_id",
                "compute_engine",
                "n_series",
                "decision_metric",
                "score",
                "pooled_wape",
                "n_comparable_series",
                "mean_wape",
                "mean_smape",
                "mean_mase",
                "mean_rmsse",
                "mean_mae",
                "mean_rmse",
                "mean_coverage",
                "mean_interval_score",
                "mean_fit_seconds",
                "median_fit_seconds",
                "no_artifact_rate",
                "n_predictions",
                "lift_vs_best_base",
                "lift_pct",
            ]
        )
    return pd.DataFrame.from_records(records)


def build_predictions_frame(
    pred_rows: list[dict[str, Any]],
    *,
    oof_rows: list[dict[str, Any]] | None = None,
    history_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Stack historical observations, out-of-fold backtests, and forward predictions into one tidy
    ``DataFrame`` (pure, offline).

    Columns: ``ts_id``, ``segment`` (``"history"`` / ``"oof"`` / ``"forecast"``), ``model_type``,
    ``ds``, ``y_true``, ``yhat``, ``yhat_lower``, ``yhat_upper``, ``fold_id``.
    """
    import pandas as pd

    cols = [
        "ts_id",
        "segment",
        "model_type",
        "ds",
        "y_true",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "fold_id",
    ]
    records: list[dict[str, Any]] = []
    for r in history_rows or []:
        records.append(
            {
                "ts_id": str(r["ts_id"]),
                "segment": "history",
                "model_type": "actual",
                "ds": pd.to_datetime(r.get("ds") or r.get("forecast_date")),
                "y_true": _num(r.get("y") if "y" in r else r.get("y_true")),
                "yhat": None,
                "yhat_lower": None,
                "yhat_upper": None,
                "fold_id": None,
            }
        )
    for r in oof_rows or []:
        records.append(
            {
                "ts_id": str(r["ts_id"]),
                "segment": "oof",
                "model_type": str(r["model_type"]),
                "ds": pd.to_datetime(r.get("forecast_date") or r.get("ds")),
                "y_true": _num(r.get("y_true")),
                "yhat": _num(r.get("yhat")),
                "yhat_lower": _num(r.get("yhat_lower")),
                "yhat_upper": _num(r.get("yhat_upper")),
                "fold_id": r.get("fold_id"),
            }
        )
    for r in pred_rows:
        records.append(
            {
                "ts_id": str(r["ts_id"]),
                "segment": "forecast",
                "model_type": str(r["model_type"]),
                "ds": pd.to_datetime(r.get("forecast_date") or r.get("ds")),
                "y_true": None,
                "yhat": _num(r.get("yhat")),
                "yhat_lower": _num(r.get("yhat_lower")),
                "yhat_upper": _num(r.get("yhat_upper")),
                "fold_id": None,
            }
        )
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame.from_records(records, columns=cols)
    return df.sort_values(["ts_id", "segment", "model_type", "ds"]).reset_index(drop=True)


def plot_forecasts_frame(
    frame: pd.DataFrame,
    *,
    ts_id: str | None = None,
    ts_ids: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    max_series: int = 3,
    history_tail: int | None = None,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot historical actuals, out-of-fold backtests, and forward predictions with 80% intervals
    from a `build_predictions_frame` ``DataFrame`` (pure, offline).

    Renders up to ``max_series`` subplots (one per ``ts_id``) and returns the matplotlib ``Axes``
    (or array of ``Axes`` when multiple series are plotted).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if frame.empty:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))
        ax.set_title(f"{title or 'Forecasts'} (no rows)")
        return ax

    df = frame.copy()
    if "ds" not in df.columns and "forecast_date" in df.columns:
        df["ds"] = pd.to_datetime(df["forecast_date"])
    if "segment" not in df.columns:
        df["segment"] = "forecast"

    effective_ts_ids = [ts_id] if ts_id is not None else (list(ts_ids) if ts_ids else None)
    all_ts = list(dict.fromkeys(df["ts_id"].astype(str)))
    chosen_ts = [t for t in all_ts if not effective_ts_ids or t in set(effective_ts_ids)][
        : max(1, max_series)
    ]
    if not chosen_ts:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))
        ax.set_title(f"{title or 'Forecasts'} (no matching series)")
        return ax

    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    if ax is None:
        _, axes_raw = plt.subplots(
            len(chosen_ts),
            1,
            figsize=(11, 3.4 * len(chosen_ts)),
            squeeze=False,
        )
        axes = [axes_raw[i, 0] for i in range(len(chosen_ts))]
    else:
        axes = [ax]
        chosen_ts = chosen_ts[:1]

    for sub_ax, tid in zip(axes, chosen_ts, strict=True):
        sub = df[df["ts_id"].astype(str) == tid]
        hist = sub[sub["segment"] == "history"].sort_values("ds")
        if history_tail is not None and history_tail > 0:
            hist = hist.tail(history_tail)
        if not hist.empty:
            sub_ax.plot(
                hist["ds"],
                hist["y_true"],
                color="#222222",
                linewidth=1.6,
                label="actual (history)",
            )
        oof = sub[sub["segment"] == "oof"]
        if not oof.empty and hist.empty and "y_true" in oof.columns:
            actual_oof = (
                oof.dropna(subset=["y_true"]).drop_duplicates(subset=["ds"]).sort_values("ds")
            )
            if not actual_oof.empty:
                sub_ax.plot(
                    actual_oof["ds"],
                    actual_oof["y_true"],
                    color="#222222",
                    linewidth=1.5,
                    label="actual (OOF)",
                )
        fc_models = [
            m
            for m in dict.fromkeys(sub[sub["segment"] != "history"]["model_type"].astype(str))
            if not models or m in set(models)
        ]
        for m_idx, m_name in enumerate(fc_models):
            color = palette[m_idx % len(palette)]
            m_oof = oof[oof["model_type"].astype(str) == m_name].sort_values("ds")
            if not m_oof.empty:
                sub_ax.plot(
                    m_oof["ds"],
                    m_oof["yhat"],
                    linestyle="--",
                    linewidth=1.3,
                    color=color,
                    alpha=0.85,
                    label=f"{m_name} (OOF)",
                )
            m_fc = sub[
                (sub["segment"] == "forecast") & (sub["model_type"].astype(str) == m_name)
            ].sort_values("ds")
            if not m_fc.empty:
                sub_ax.plot(
                    m_fc["ds"],
                    m_fc["yhat"],
                    linestyle="-",
                    linewidth=2.0,
                    color=color,
                    label=f"{m_name} (forecast)",
                )
                if (
                    "yhat_lower" in m_fc.columns
                    and "yhat_upper" in m_fc.columns
                    and m_fc["yhat_lower"].notna().any()
                    and m_fc["yhat_upper"].notna().any()
                ):
                    sub_ax.fill_between(
                        m_fc["ds"],
                        m_fc["yhat_lower"].astype(float),
                        m_fc["yhat_upper"].astype(float),
                        color=color,
                        alpha=0.16,
                    )
        sub_ax.set_title(f"{title + ' — ' if title else ''}series: {tid}")
        sub_ax.set_ylabel("value")
        sub_ax.legend(loc="best", fontsize=8, ncol=2)
    axes[-1].set_xlabel("date")
    return axes[0] if len(axes) == 1 else axes


def _hierarchy_level_label(ts_id: str) -> str:
    """Classify a hierarchical ``ts_id`` into ``total``, ``aggregate``, or ``bottom`` (pure)."""
    if ts_id == "__total__":
        return "total"
    if "/" in ts_id:
        return "aggregate"
    return "bottom"


def build_hierarchy_frame(pred_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Summarize hierarchical predictions by model and level, verifying additive coherence
    ($\\max_t |\\hat{y}_{\\text{total},t} - \\sum_{b \\in \\text{bottom}} \\hat{y}_{b,t}|$) (pure).
    """
    import pandas as pd

    cols = [
        "model_type",
        "level",
        "n_series",
        "n_points",
        "mean_yhat",
        "sum_yhat",
        "max_coherence_residual",
    ]
    if not pred_rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(pred_rows)
    df["ts_id"] = df["ts_id"].astype(str)
    df["level"] = df["ts_id"].map(_hierarchy_level_label)
    date_col = "forecast_date" if "forecast_date" in df.columns else "ds"

    records: list[dict[str, Any]] = []
    for model_type, m_grp in df.groupby("model_type", sort=True):
        tot = m_grp[m_grp["level"] == "total"].groupby(date_col)["yhat"].sum()
        bot = m_grp[m_grp["level"] == "bottom"].groupby(date_col)["yhat"].sum()
        residual: float | None = None
        if not tot.empty and not bot.empty:
            aligned = pd.concat([tot.rename("total"), bot.rename("bottom")], axis=1).dropna()
            if not aligned.empty:
                residual = float((aligned["total"] - aligned["bottom"]).abs().max())
        for lvl in ("total", "aggregate", "bottom"):
            l_grp = m_grp[m_grp["level"] == lvl]
            if l_grp.empty:
                continue
            records.append(
                {
                    "model_type": str(model_type),
                    "level": lvl,
                    "n_series": int(l_grp["ts_id"].nunique()),
                    "n_points": int(len(l_grp)),
                    "mean_yhat": float(l_grp["yhat"].astype(float).mean()),
                    "sum_yhat": float(l_grp["yhat"].astype(float).sum()),
                    "max_coherence_residual": residual,
                }
            )
    return pd.DataFrame.from_records(records, columns=cols)


def plot_hierarchy_frame(
    pred_rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    model_type: str | None = None,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot top-level (``__total__``) forecast against the sum of bottom-level leaf forecasts (or
    a level summary from `build_hierarchy_frame`) to visually verify hierarchical coherence.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = (
        pred_rows.copy()
        if isinstance(pred_rows, pd.DataFrame)
        else pd.DataFrame.from_records(pred_rows)
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    if df.empty:
        ax.set_title(f"{title or 'Hierarchical Coherence'} (no rows)")
        return ax

    # Support passing the summary DataFrame from build_hierarchy_frame directly
    if {"level", "sum_yhat", "model_type"} <= set(df.columns) and "ts_id" not in df.columns:
        chosen_model = model_type or str(df["model_type"].iloc[0])
        sub = df[df["model_type"].astype(str) == chosen_model]
        ax.bar(sub["level"].astype(str), sub["sum_yhat"].astype(float), color="#0072B2", alpha=0.85)
        ax.set_title(f"{title or 'Hierarchical Rollup Totals'}: {chosen_model}")
        ax.set_xlabel("hierarchy level")
        ax.set_ylabel("sum of yhat")
        return ax

    df["ts_id"] = df["ts_id"].astype(str)
    df["level"] = df["ts_id"].map(_hierarchy_level_label)
    date_col = "forecast_date" if "forecast_date" in df.columns else "ds"
    df[date_col] = pd.to_datetime(df[date_col])

    chosen_model = model_type or str(df["model_type"].iloc[0])
    sub = df[df["model_type"].astype(str) == chosen_model]
    if sub.empty:
        ax.set_title(f"{title or 'Hierarchical Coherence'} (no rows for {chosen_model})")
        return ax

    tot = sub[sub["level"] == "total"].groupby(date_col)["yhat"].sum().sort_index()
    bot = sub[sub["level"] == "bottom"].groupby(date_col)["yhat"].sum().sort_index()
    if not tot.empty:
        ax.plot(
            tot.index,
            tot.to_numpy(),
            color="#0072B2",
            linewidth=2.4,
            label=f"__total__ ({chosen_model})",
        )
    if not bot.empty:
        ax.plot(
            bot.index,
            bot.to_numpy(),
            color="#E69F00",
            linestyle="--",
            linewidth=2.0,
            label=f"Σ bottom-level leaves ({chosen_model})",
        )
    residual_str = ""
    if not tot.empty and not bot.empty:
        aligned = pd.concat([tot.rename("t"), bot.rename("b")], axis=1).dropna()
        if not aligned.empty:
            max_err = float((aligned["t"] - aligned["b"]).abs().max())
            residual_str = f" — max |total − Σ bottom| = {max_err:.2e}"
    ax.set_title(f"{title or 'Hierarchical Coherence'}: {chosen_model}{residual_str}")
    ax.set_xlabel("forecast date")
    ax.set_ylabel("forecast value")
    ax.legend(loc="best", fontsize=9)
    return ax


def build_cohorts_frame(review: RunReview) -> pd.DataFrame:
    """Convert a `RunReview`'s per-model `BacktestCohort` and comparable holdout metrics into a
    tidy ``DataFrame`` (pure, offline).

    Exposes ``v_backtest_coverage`` and ``v_model_leaderboard_comparable`` side-by-side so you can
    inspect achieved fold counts (`fold_histogram`), refit modes (`per_fold` / `recondition` /
    `extrapolate` / `unsupported`), `staleness_gap` (cost of never refitting), and holdout-fold
    `pooled_wape` alongside the fleet mean `score`.
    """
    import pandas as pd

    cols = [
        "model_type",
        "family",
        "is_ensemble",
        "n_series",
        "score",
        "pooled_wape",
        "n_comparable_series",
        "n_full",
        "n_reduced",
        "n_unscored",
        "n_failed",
        "n_not_requested",
        "fold_histogram",
        "refit_modes",
        "staleness_gap",
    ]
    if not review.models:
        return pd.DataFrame(columns=cols)

    records: list[dict[str, Any]] = []
    for m in review.models:
        c = m.cohort or BacktestCohort()
        hist_str = ", ".join(f"{k}f:{v}" for k, v in c.fold_histogram.items()) or "none"
        modes_str = ", ".join(f"{k}:{v}" for k, v in c.refit_modes.items()) or "none"
        records.append(
            {
                "model_type": m.model_type,
                "family": m.family,
                "is_ensemble": m.is_ensemble,
                "n_series": m.n_series if m.n_series is not None else c.n_series,
                "score": m.score,
                "pooled_wape": m.pooled_wape,
                "n_comparable_series": m.n_comparable_series,
                "n_full": c.n_full,
                "n_reduced": c.n_reduced,
                "n_unscored": c.n_unscored,
                "n_failed": c.n_failed,
                "n_not_requested": c.n_not_requested,
                "fold_histogram": hist_str,
                "refit_modes": modes_str,
                "staleness_gap": c.staleness_gap,
            }
        )
    return pd.DataFrame.from_records(records, columns=cols)


def build_calibration_frames(
    report: CalibrationReport,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a `CalibrationReport` into ``(arms_df, coverage_df)`` pandas ``DataFrame``s (pure).

    * ``arms_df`` summarizes point-forecast bias-correction arm selection (`raw` vs `corrected`
      win rates and relative margins per model).
    * ``coverage_df`` reports empirical prediction-interval coverage and mean interval width at
      each horizon step ($h = 1 \\dots H$) against `report.nominal_coverage`.
    """
    import pandas as pd

    arm_cols = [
        "model_type",
        "compute_engine",
        "interval_calibration",
        "n_series",
        "n_raw_arm",
        "raw_arm_rate",
        "n_auto_decided",
        "n_compared",
        "n_corrected_wins",
        "win_rate",
        "mean_margin",
        "median_margin",
    ]
    cov_cols = [
        "model_type",
        "horizon_step",
        "n",
        "coverage",
        "nominal_coverage",
        "coverage_error",
        "mean_width",
    ]
    arm_records = [
        {
            "model_type": a.model_type,
            "compute_engine": a.compute_engine,
            "interval_calibration": a.interval_calibration,
            "n_series": a.n_series,
            "n_raw_arm": a.n_raw_arm,
            "raw_arm_rate": a.raw_arm_rate,
            "n_auto_decided": a.n_auto_decided,
            "n_compared": a.n_compared,
            "n_corrected_wins": a.n_corrected_wins,
            "win_rate": a.win_rate,
            "mean_margin": a.mean_margin,
            "median_margin": a.median_margin,
        }
        for a in report.arms
    ]
    cov_records = [
        {
            "model_type": p.model_type,
            "horizon_step": p.horizon_step,
            "n": p.n,
            "coverage": p.coverage,
            "nominal_coverage": report.nominal_coverage,
            "coverage_error": (
                p.coverage - report.nominal_coverage if p.coverage is not None else None
            ),
            "mean_width": p.mean_width,
        }
        for p in report.coverage
    ]
    arms_df = (
        pd.DataFrame.from_records(arm_records, columns=arm_cols)
        if arm_records
        else pd.DataFrame(columns=arm_cols)
    )
    cov_df = (
        pd.DataFrame.from_records(cov_records, columns=cov_cols)
        if cov_records
        else pd.DataFrame(columns=cov_cols)
    )
    return arms_df, cov_df


def plot_calibration(
    report: CalibrationReport,
    *,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot empirical prediction-interval coverage and mean interval width by horizon step from a
    `CalibrationReport` (pure, offline).

    When ``ax=None``, creates a two-panel side-by-side figure:
    * Left panel: Empirical coverage by horizon step ($h = 1 \\dots H$) vs. nominal target
      (dashed horizontal reference line at ``report.nominal_coverage``).
    * Right panel: Mean prediction-interval width ($\\hat{y}_{\\text{upper}} -
      \\hat{y}_{\\text{lower}}$) by horizon step.
    """
    import matplotlib.pyplot as plt

    _, cov_df = build_calibration_frames(report)
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    heading = (
        title or f"{report.run_id} — Interval Calibration (nominal {report.nominal_coverage:.0%})"
    )

    if ax is not None:
        ax_cov = ax
        ax_width = None
    elif not cov_df.empty and cov_df["mean_width"].notna().any():
        _, (ax_cov, ax_width) = plt.subplots(1, 2, figsize=(12, 4.2))
    else:
        _, ax_cov = plt.subplots(figsize=(8, 4.2))
        ax_width = None

    if cov_df.empty:
        ax_cov.set_title(f"{heading} (no OOF coverage rows)")
        return ax_cov

    for idx, (m_name, grp) in enumerate(cov_df.groupby("model_type", sort=True)):
        ordered = grp.sort_values("horizon_step")
        color = palette[idx % len(palette)]
        ax_cov.plot(
            ordered["horizon_step"],
            ordered["coverage"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=str(m_name),
        )
        if ax_width is not None and ordered["mean_width"].notna().any():
            ax_width.plot(
                ordered["horizon_step"],
                ordered["mean_width"],
                marker="s",
                markersize=3.5,
                linewidth=1.8,
                color=color,
                label=str(m_name),
            )

    ax_cov.axhline(
        report.nominal_coverage,
        color="#222222",
        linestyle="--",
        linewidth=1.4,
        label=f"nominal ({report.nominal_coverage:.0%})",
    )
    ax_cov.set_ylim(-0.02, 1.05)
    ax_cov.set_xlabel("horizon step (h)")
    ax_cov.set_ylabel("empirical coverage")
    ax_cov.set_title(heading)
    ax_cov.legend(loc="best", fontsize=8)

    if ax_width is not None:
        ax_width.set_xlabel("horizon step (h)")
        ax_width.set_ylabel("mean interval width (yhat_upper − yhat_lower)")
        ax_width.set_title(f"{report.run_id} — Interval Width by Step")
        ax_width.legend(loc="best", fontsize=8)
        return (ax_cov, ax_width)
    return ax_cov


def build_ensemble_weights_frame(
    best_params_rows: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Unpack learned ensemble stacking weights from `read_best_params` rows or a
    `build_best_params_frame` ``DataFrame`` into a long-form ``DataFrame`` (pure, offline).

    Columns: ``ts_id``, ``ensemble_model``, ``strategy``, ``base_model``, ``weight``, ``wape``.
    """
    import json

    import pandas as pd

    cols = ["ts_id", "ensemble_model", "strategy", "base_model", "weight", "wape"]
    if isinstance(best_params_rows, pd.DataFrame):
        if best_params_rows.empty:
            return pd.DataFrame(columns=cols)
        raw_list = best_params_rows.to_dict(orient="records")
    else:
        raw_list = list(best_params_rows)
    if not raw_list:
        return pd.DataFrame(columns=cols)

    records: list[dict[str, Any]] = []
    for r in raw_list:
        m_type = str(r.get("model_type") or "")
        is_ens = (
            bool(r.get("is_ensemble"))
            or (r.get("ensemble_id") is not None)
            or m_type.startswith("ensemble_")
        )
        if not is_ens:
            continue
        raw_bp = r.get("best_params")
        parsed: Any = None
        if isinstance(raw_bp, dict):
            parsed = raw_bp
        elif isinstance(raw_bp, str) and raw_bp:
            try:
                parsed = json.loads(raw_bp)
            except ValueError:
                parsed = None
        if not isinstance(parsed, dict) or not parsed:
            continue
        strategy = m_type.removeprefix("ensemble_")
        for base_model, weight in sorted(parsed.items()):
            w_val = _num(weight)
            if w_val is None:
                continue
            records.append(
                {
                    "ts_id": str(r.get("ts_id") or ""),
                    "ensemble_model": m_type,
                    "strategy": strategy,
                    "base_model": str(base_model),
                    "weight": w_val,
                    "wape": _num(r.get("wape")),
                }
            )
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame.from_records(records, columns=cols)


def plot_ensemble_weights(
    weights_df: list[dict[str, Any]] | pd.DataFrame,
    *,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot base-model stacking weights across learned ensemble strategies (`nnls`, `ridge`, `xgb`)
    as a horizontal stacked bar chart (pure, offline).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = (
        weights_df
        if isinstance(weights_df, pd.DataFrame) and "base_model" in weights_df.columns
        else build_ensemble_weights_frame(weights_df)
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.8))
    if df.empty:
        ax.set_title(f"{title or 'Ensemble Stacking Weights'} (no learned weight rows)")
        return ax

    pivot = (
        df.groupby(["ensemble_model", "base_model"], as_index=False)["weight"]
        .mean()
        .pivot(index="ensemble_model", columns="base_model", values="weight")
        .fillna(0.0)
    )
    # Normalize each strategy row to sum to 1.0 for clean stacked composition display
    row_sums = pivot.sum(axis=1).replace(0.0, 1.0)
    norm_pivot = pivot.div(row_sums, axis=0)

    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
    strategies = list(norm_pivot.index)
    base_models = list(norm_pivot.columns)
    ys = np.arange(len(strategies))
    left = np.zeros(len(strategies))

    for idx, bm in enumerate(base_models):
        vals = norm_pivot[bm].to_numpy(dtype=float)
        color = palette[idx % len(palette)]
        ax.barh(ys, vals, left=left, height=0.55, color=color, label=bm)
        for y_idx, v in enumerate(vals):
            if v >= 0.08:
                ax.text(
                    left[y_idx] + v / 2.0,
                    y_idx,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    color="white" if idx in (0, 2, 3) else "#111111",
                    fontsize=8.5,
                    fontweight="bold",
                )
        left += vals

    ax.set_yticks(list(ys))
    ax.set_yticklabels(strategies)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("mean normalized base-model weight across series")
    ax.set_title(title or "Learned Ensemble Stacking Weights by Strategy")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(4, len(base_models)), fontsize=8.5
    )
    return ax


def explain_forecast_frame(
    frame: pd.DataFrame,
    *,
    ts_id: str | None = None,
    model_type: str | None = None,
    seasonal_period: int = 7,
    covariate_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Decompose a single series' historical actuals, backtest OOF trajectory, and future forecast
    into interpretable structural components (pure, offline).

    Given a ``frame`` from `build_predictions_frame` (or `Forecaster.predictions_df`), extracts:
    1. ``trend_baseline`` & ``level_shift``: Smooth rolling/piecewise trend plus abrupt regime-jump
       detection via `features.level_shift_step`.
    2. ``seasonal_effect``: Repeating periodic cycle (period ``seasonal_period``, default ``7``)
       estimated from detrended history and projected across the forecast horizon.
    3. ``covariate_effect`` (plus ``cov_<name>`` columns when ``covariate_df`` is provided):
       Ridge-stabilized linear attribution of numeric exogenous covariates across history and the
       future forecast horizon.
    4. ``oof_residual`` & ``interval_width``: Out-of-fold error ($y - \\hat{y}_{\\text{OOF}}$) on
       backtest dates and prediction-interval width ($\\hat{y}_{\\text{upper}} -
       \\hat{y}_{\\text{lower}}$) across the future horizon.
    """
    import numpy as np
    import pandas as pd

    from .features import level_shift_step

    base_cols = [
        "ts_id",
        "model_type",
        "segment",
        "ds",
        "y_true",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "trend_baseline",
        "level_shift",
        "seasonal_effect",
        "covariate_effect",
        "oof_residual",
        "interval_width",
    ]
    if frame.empty:
        return pd.DataFrame(columns=base_cols)

    df = frame.copy()
    if "ds" not in df.columns and "forecast_date" in df.columns:
        df["ds"] = pd.to_datetime(df["forecast_date"])
    else:
        df["ds"] = pd.to_datetime(df["ds"])
    if "segment" not in df.columns:
        df["segment"] = "forecast"

    all_ts = list(dict.fromkeys(df["ts_id"].astype(str)))
    chosen_ts = ts_id if (ts_id is not None and ts_id in set(all_ts)) else all_ts[0]
    sub = df[df["ts_id"].astype(str) == chosen_ts]

    fc_models = list(dict.fromkeys(sub[sub["segment"] != "history"]["model_type"].astype(str)))
    chosen_model = (
        model_type
        if (model_type is not None and model_type in set(fc_models))
        else (fc_models[0] if fc_models else "actual")
    )

    hist = sub[sub["segment"] == "history"].sort_values("ds").drop_duplicates(subset=["ds"])
    fc = (
        sub[(sub["segment"] == "forecast") & (sub["model_type"].astype(str) == chosen_model)]
        .sort_values("ds")
        .drop_duplicates(subset=["ds"])
    )
    oof = (
        sub[(sub["segment"] == "oof") & (sub["model_type"].astype(str) == chosen_model)]
        .sort_values("ds")
        .drop_duplicates(subset=["ds"])
    )

    timeline_parts: list[pd.DataFrame] = []
    if not hist.empty:
        timeline_parts.append(
            pd.DataFrame(
                {
                    "ds": hist["ds"].to_numpy(),
                    "segment": "history",
                    "y_true": hist["y_true"].astype(float).to_numpy(),
                    "yhat": np.nan,
                    "yhat_lower": np.nan,
                    "yhat_upper": np.nan,
                }
            )
        )
    if not fc.empty:
        timeline_parts.append(
            pd.DataFrame(
                {
                    "ds": fc["ds"].to_numpy(),
                    "segment": "forecast",
                    "y_true": np.nan,
                    "yhat": fc["yhat"].astype(float).to_numpy(),
                    "yhat_lower": (
                        fc["yhat_lower"].astype(float).to_numpy()
                        if "yhat_lower" in fc.columns
                        else np.nan
                    ),
                    "yhat_upper": (
                        fc["yhat_upper"].astype(float).to_numpy()
                        if "yhat_upper" in fc.columns
                        else np.nan
                    ),
                }
            )
        )
    if not timeline_parts:
        return pd.DataFrame(columns=base_cols)

    tl = pd.concat(timeline_parts, ignore_index=True).sort_values("ds").reset_index(drop=True)
    tl["ts_id"] = chosen_ts
    tl["model_type"] = chosen_model

    # Overlay OOF yhat & residuals on matching history dates (or append if history was omitted)
    oof_map: dict[pd.Timestamp, tuple[float | None, float | None, float | None, float | None]] = {}
    for r in oof.itertuples(index=False):
        oof_map[pd.Timestamp(r.ds)] = (
            _num(getattr(r, "y_true", None)),
            _num(getattr(r, "yhat", None)),
            _num(getattr(r, "yhat_lower", None)),
            _num(getattr(r, "yhat_upper", None)),
        )

    oof_residuals = np.full(len(tl), np.nan)
    for idx, row_ds in enumerate(tl["ds"]):
        ts_key = pd.Timestamp(row_ds)
        if ts_key in oof_map:
            y_t, yh_oof, lo_oof, hi_oof = oof_map[ts_key]
            if tl.loc[idx, "segment"] == "history" and yh_oof is not None:
                tl.loc[idx, "yhat"] = yh_oof
                if lo_oof is not None:
                    tl.loc[idx, "yhat_lower"] = lo_oof
                if hi_oof is not None:
                    tl.loc[idx, "yhat_upper"] = hi_oof
                actual_val = tl.loc[idx, "y_true"] if pd.notna(tl.loc[idx, "y_true"]) else y_t
                if actual_val is not None and pd.notna(actual_val):
                    oof_residuals[idx] = float(actual_val) - float(yh_oof)

    # Combined continuous signal (y_true on history, yhat on forecast)
    signal = np.where(
        tl["segment"] == "history",
        tl["y_true"].to_numpy(dtype=float),
        tl["yhat"].to_numpy(dtype=float),
    )
    n_hist = int((tl["segment"] == "history").sum())

    # 1. Level-shift detection on history + rolling trend baseline
    level_shift_arr = np.zeros(len(tl), dtype=float)
    if n_hist >= 16:
        hist_series = pd.Series(signal[:n_hist])
        step_dummy = level_shift_step(hist_series)
        if step_dummy.max() > 0:
            pre_mean = float(hist_series[step_dummy == 0].mean())
            post_mean = float(hist_series[step_dummy == 1].mean())
            jump = post_mean - pre_mean
            level_shift_arr[:n_hist] = step_dummy * jump
            level_shift_arr[n_hist:] = jump

    adj_signal = signal - level_shift_arr
    win = max(3, int(seasonal_period))
    trend_adj = (
        pd.Series(adj_signal)
        .rolling(window=win, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    trend_baseline = trend_adj + level_shift_arr
    detrended = signal - trend_baseline

    # 2. Periodic seasonal profile (period = seasonal_period)
    seasonal_effect = np.zeros(len(tl), dtype=float)
    p = max(1, int(seasonal_period))
    if p > 1 and len(tl) >= p:
        ref_len = n_hist if n_hist >= p else len(tl)
        phase_means = np.zeros(p, dtype=float)
        for rem in range(p):
            vals = detrended[:ref_len][np.arange(ref_len) % p == rem]
            vals = vals[np.isfinite(vals)]
            phase_means[rem] = float(vals.mean()) if vals.size else 0.0
        phase_means -= float(phase_means.mean())
        for idx in range(len(tl)):
            seasonal_effect[idx] = phase_means[idx % p]

    # 3. Exogenous covariate attribution (when covariate_df is supplied)
    covariate_effect = np.zeros(len(tl), dtype=float)
    cov_added_cols: list[str] = []
    if covariate_df is not None and not covariate_df.empty:
        cov_work = covariate_df.copy()
        if "ts_id" in cov_work.columns:
            cov_work = cov_work[cov_work["ts_id"].astype(str) == chosen_ts]
        date_c = (
            "ds"
            if "ds" in cov_work.columns
            else ("forecast_date" if "forecast_date" in cov_work.columns else None)
        )
        if date_c is not None and not cov_work.empty:
            cov_work["ds"] = pd.to_datetime(cov_work[date_c])
            cov_work = cov_work.sort_values("ds").drop_duplicates(subset=["ds"])
            exclude = {
                "ts_id",
                "ds",
                "forecast_date",
                "y",
                "y_true",
                "yhat",
                "yhat_lower",
                "yhat_upper",
                "segment",
                "fold_id",
            }
            num_cols = [
                c
                for c in cov_work.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(cov_work[c])
            ]
            if num_cols:
                merged_cov = pd.merge(tl[["ds"]], cov_work[["ds", *num_cols]], on="ds", how="left")
                X_raw = merged_cov[num_cols].ffill().bfill().fillna(0.0).to_numpy(dtype=float)
                fit_n = n_hist if n_hist >= len(num_cols) + 2 else len(tl)
                col_means = X_raw[:fit_n].mean(axis=0)
                X_centered = X_raw - col_means
                target_res = (detrended - seasonal_effect)[:fit_n]
                valid_mask = np.isfinite(target_res)
                if int(valid_mask.sum()) >= 2:
                    X_fit = X_centered[:fit_n][valid_mask]
                    y_fit = target_res[valid_mask]
                    # Small ridge penalty so collinear covariates decompose cleanly
                    gram = X_fit.T @ X_fit + 1e-3 * np.eye(X_fit.shape[1])
                    betas = np.linalg.solve(gram, X_fit.T @ y_fit)
                    for c_idx, c_name in enumerate(num_cols):
                        col_contrib = X_centered[:, c_idx] * float(betas[c_idx])
                        out_col = f"cov_{c_name}"
                        tl[out_col] = col_contrib
                        cov_added_cols.append(out_col)
                        covariate_effect += col_contrib

    tl["trend_baseline"] = trend_baseline
    tl["level_shift"] = level_shift_arr
    tl["seasonal_effect"] = seasonal_effect
    tl["covariate_effect"] = covariate_effect
    tl["oof_residual"] = oof_residuals
    tl["interval_width"] = np.where(
        tl["yhat_upper"].notna() & tl["yhat_lower"].notna(),
        tl["yhat_upper"].astype(float) - tl["yhat_lower"].astype(float),
        np.nan,
    )
    return tl[[*base_cols, *cov_added_cols]]


def plot_forecast_explanation(
    explanation_df: pd.DataFrame,
    *,
    history_tail: int | None = None,
    title: str | None = None,
) -> Any:
    """Render a 4-panel forecast explainability & decomposition chart from an
    `explain_forecast_frame` ``DataFrame`` (pure, offline).

    Panels:
    1. **Trajectory, Trend Baseline & Regime Shift**: History, OOF backtest, future forecast + 80%
       prediction interval, and structural trend baseline.
    2. **Periodic Seasonal Cycle**: Extracted seasonal wave across history and forecast horizon.
    3. **Exogenous Covariate Attribution** (or Detrended Residual Wave when univariate): Individual
       ``cov_<name>`` contributions and total covariate effect across history and the future
       horizon.
    4. **Backtest OOF Residuals & Forecast Horizon Uncertainty**: Out-of-fold errors ($y -
       \\hat{y}_{\\text{OOF}}$) transitioning into future 80% interval width
       ($\\hat{y}_{\\text{upper}} - \\hat{y}_{\\text{lower}}$).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if explanation_df.empty:
        _, ax = plt.subplots(figsize=(10, 3))
        ax.set_title(f"{title or 'Forecast Explanation'} (no rows)")
        return ax

    df = explanation_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    if history_tail is not None and history_tail > 0:
        hist_part = df[df["segment"] == "history"].tail(history_tail)
        fc_part = df[df["segment"] != "history"]
        df = pd.concat([hist_part, fc_part], ignore_index=True)
    tid = str(df["ts_id"].iloc[0])
    m_name = str(df["model_type"].iloc[0])
    heading = title or f"Forecast Decomposition & Attribution — series={tid} · model={m_name}"

    _, axes = plt.subplots(4, 1, figsize=(11, 9.2), sharex=True)
    ax_main, ax_seas, ax_cov, ax_err = axes

    hist = df[df["segment"] == "history"]
    fc = df[df["segment"] == "forecast"]

    # Panel 1: Main trajectory + trend + level shift
    if not hist.empty:
        ax_main.plot(
            hist["ds"],
            hist["y_true"],
            color="#222222",
            linewidth=1.5,
            label="actual (history)",
        )
        oof_valid = hist[hist["yhat"].notna()]
        if not oof_valid.empty:
            ax_main.plot(
                oof_valid["ds"],
                oof_valid["yhat"],
                color="#E69F00",
                linestyle="--",
                linewidth=1.4,
                label=f"{m_name} (OOF)",
            )
    if not fc.empty:
        ax_main.plot(
            fc["ds"],
            fc["yhat"],
            color="#0072B2",
            linewidth=2.1,
            label=f"{m_name} (forecast)",
        )
        if fc["yhat_lower"].notna().any() and fc["yhat_upper"].notna().any():
            ax_main.fill_between(
                fc["ds"],
                fc["yhat_lower"].astype(float),
                fc["yhat_upper"].astype(float),
                color="#0072B2",
                alpha=0.18,
                label="80% interval",
            )
        ax_main.axvline(fc["ds"].iloc[0], color="#666666", linestyle=":", linewidth=1.2)

    ax_main.plot(
        df["ds"],
        df["trend_baseline"],
        color="#009E73",
        linewidth=1.8,
        alpha=0.9,
        label="trend + regime baseline",
    )
    if df["level_shift"].abs().max() > 0:
        shift_idx = df.index[df["level_shift"].abs() > 0][0]
        ax_main.axvline(
            df.loc[shift_idx, "ds"],
            color="#D55E00",
            linestyle="-.",
            linewidth=1.3,
            label="detected level shift",
        )
    ax_main.set_title(heading)
    ax_main.set_ylabel("value")
    ax_main.legend(loc="best", fontsize=8, ncol=3)

    # Panel 2: Seasonal effect
    ax_seas.plot(
        df["ds"], df["seasonal_effect"], color="#0072B2", linewidth=1.4, label="seasonal cycle"
    )
    ax_seas.axhline(0.0, color="#999999", linestyle=":", linewidth=0.9)
    if not fc.empty:
        ax_seas.axvline(fc["ds"].iloc[0], color="#666666", linestyle=":", linewidth=1.2)
    ax_seas.set_ylabel("seasonal")
    ax_seas.legend(loc="upper right", fontsize=8)

    # Panel 3: Exogenous covariate attribution
    cov_cols = [c for c in df.columns if c.startswith("cov_")]
    palette = ["#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    if cov_cols:
        for idx, c_col in enumerate(cov_cols):
            ax_cov.plot(
                df["ds"],
                df[c_col],
                linewidth=1.3,
                color=palette[idx % len(palette)],
                label=c_col.removeprefix("cov_"),
            )
        ax_cov.plot(
            df["ds"],
            df["covariate_effect"],
            color="#222222",
            linestyle="--",
            linewidth=1.4,
            label="total covariate effect",
        )
    else:
        ax_cov.plot(
            df["ds"],
            df["covariate_effect"],
            color="#666666",
            linewidth=1.2,
            label="covariate effect (none configured)",
        )
    ax_cov.axhline(0.0, color="#999999", linestyle=":", linewidth=0.9)
    if not fc.empty:
        ax_cov.axvline(fc["ds"].iloc[0], color="#666666", linestyle=":", linewidth=1.2)
    ax_cov.set_ylabel("covariates")
    ax_cov.legend(loc="upper right", fontsize=8, ncol=min(4, max(1, len(cov_cols) + 1)))

    # Panel 4: OOF residuals & future interval width
    oof_res = df[df["oof_residual"].notna()]
    if not oof_res.empty:
        ax_err.bar(
            oof_res["ds"],
            oof_res["oof_residual"],
            width=0.8,
            color="#E69F00",
            alpha=0.75,
            label="OOF residual (y − yhat)",
        )
    if not fc.empty and fc["interval_width"].notna().any():
        ax_err.plot(
            fc["ds"],
            fc["interval_width"],
            color="#0072B2",
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label="forecast 80% interval width",
        )
        ax_err.axvline(fc["ds"].iloc[0], color="#666666", linestyle=":", linewidth=1.2)
    ax_err.axhline(0.0, color="#999999", linestyle=":", linewidth=0.9)
    ax_err.set_ylabel("error / width")
    ax_err.set_xlabel("date")
    ax_err.legend(loc="best", fontsize=8, ncol=2)
    return axes
