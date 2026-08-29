"""Monitor a running run and review a finished one — the run-inspection layer, keyed on a run_id.

Two questions, two entry points, both taking only a ``run_id`` (the run *is* its config, so this
layer reads the run's own ``raw_config`` back to recover what it *planned* to do):

- `monitor_run` — *how far along is a run in flight?* Per-family job state, series done vs. the
  expected total, remaining, and mean fit time per family on its chosen runner. Progress is coarse:
  the registry has no live per-series counter — cells land when a family's ``write_cells`` runs
  (often at job end), so done-counts step up per job. The per-job status (from ``v_run_jobs``) is
  the primary live signal; landed-cell counts refine it.
- `review_run` — *how did a finished run do, in data-science detail?* The best model per family and
  overall, the full metric panel aggregated across every series (mean + p10/p50/p90), and each
  ensemble's lift over the best base model.

Same pure/I-O seam as `sdk`: the ``_assemble_*`` functions are pure (turn reader dicts into the
result dataclasses, unit-tested offline), while `monitor_run` / `review_run` are the thin I/O
callers that read the registry (`registry.bq`) and hand off. Plotting (`plot_progress`,
`plot_leaderboard`, `plot_metric_distribution`) is a convenience over the dataclasses, with
matplotlib imported lazily so it never touches the near-instant ``import scale_forecasting`` path.
For the wall-clock execution timeline, reuse `sdk.build_trace_frame` + `sdk.plot_trace`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import RunConfig
from .dag import group_models_by_family
from .registry.bq import METRIC_COLUMNS

if TYPE_CHECKING:
    from .settings import Settings

__all__ = [
    "FamilyProgress",
    "RunProgress",
    "ModelReview",
    "EnsembleLift",
    "RunReview",
    "family_of",
    "monitor_run",
    "review_run",
    "best_overall",
    "best_per_family",
    "ensemble_lift",
    "plot_progress",
    "plot_leaderboard",
    "plot_metric_distribution",
]

# Display order for families in a progress/review readout: the base families in DAG order, then the
# downstream ensemble node last. (Mirrors dag._FAMILY_ORDER + the ensemble node it appends.)
_FAMILY_ORDER: tuple[str, ...] = ("statistical", "ml", "deep_learning", "native", "ensemble")


@dataclass(frozen=True)
class FamilyProgress:
    """One family's live progress on a run: its job state and how many series it has scored.

    ``runtime`` / ``hardware`` name the runner this family resolved to (a Spark family's mean fit
    time is only comparable to another family on the *same* runner). ``n_expected`` is
    ``n_series × models-in-family`` (``None`` when the run's series count isn't known);
    ``n_done`` counts landed full-fit cells; ``fraction`` is their ratio. ``avg_fit_seconds`` is the
    mean per-cell fit time across this family's landed cells; ``runtime_seconds`` is the job's
    wall-clock once it finishes.
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


@dataclass(frozen=True)
class RunProgress:
    """A run's live progress snapshot: header status plus one `FamilyProgress` per family.

    ``status`` is ``None`` when no run exists for the id yet. ``n_done`` / ``n_expected`` /
    ``fraction`` are the run-wide roll-up across families (``n_expected`` and ``fraction`` are
    ``None`` when the series count — hence the denominator — isn't known).
    """

    run_id: str
    status: str | None
    n_series: int | None
    families: tuple[FamilyProgress, ...]
    n_done: int
    n_expected: int | None
    fraction: float | None


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


def _assemble_progress(
    run_id: str,
    summary: dict[str, Any] | None,
    cfg: RunConfig | None,
    job_rows: list[dict[str, Any]],
    progress_rows: list[dict[str, Any]],
) -> RunProgress:
    """Compose a `RunProgress` from a run's header, config, job rows, and landed-cell counts (pure).

    The config gives the *denominator* (models per family × series count = expected cells); the job
    rows give each family's runner + status; the progress rows give landed counts and mean fit time.
    With no config (run never ran) this is an empty snapshot carrying just the header status.
    """
    status = (summary or {}).get("status")
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
            )
        )

    total_done = sum(f.n_done for f in families)
    expected_known = [f.n_expected for f in families if f.n_expected is not None]
    total_expected = sum(expected_known) if len(expected_known) == len(families) else None
    fraction = (total_done / total_expected) if total_expected else None
    return RunProgress(
        run_id=run_id,
        status=status,
        n_series=n_series,
        families=tuple(families),
        n_done=total_done,
        n_expected=total_expected,
        fraction=fraction,
    )


def monitor_run(
    run_id: str, *, settings: Settings | None = None
) -> RunProgress:  # pragma: no cover - GCP I/O
    """Read a run's live progress: header status + per-family job state + series done vs. expected.

    Reads the run's header (`registry.bq.read_run_summary`), its config
    (`registry.bq.read_run_config`, for the expected-work denominator), its jobs
    (`registry.bq.read_run_jobs`) and its landed-cell counts (`registry.bq.read_progress`), then
    composes them via `_assemble_progress`. Poll it while a run is in flight; returns a status-only
    snapshot when the run id has never run.
    """
    from .registry import bq

    summary = bq.read_run_summary(run_id, settings=settings)
    raw = bq.read_run_config(run_id, settings=settings)
    cfg = RunConfig.model_validate(raw) if raw else None
    if cfg is None:
        return _assemble_progress(run_id, summary, None, [], [])
    job_rows = bq.read_run_jobs(run_id, settings=settings)
    progress_rows = bq.read_progress(run_id, settings=settings)
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


def _model_review_from_aggregate(
    agg: dict[str, Any],
    decision_metric: str,
    lb_row: dict[str, Any],
    prediction_counts: dict[str, int],
) -> ModelReview:
    """Turn one `registry.bq.read_metric_aggregates` row (+ its leaderboard match) into a
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
) -> RunReview:
    """Compose a `RunReview` from the leaderboard, metric aggregates, and prediction counts (pure).

    Aggregates are the primary source (full metric panel); the leaderboard supplies artifact rate +
    median fit time and is the fallback when a run had no backtest (no aggregates), scoring on WAPE.
    """
    lb = {(r["model_type"], r.get("ensemble_id")): r for r in leaderboard_rows}
    if aggregate_rows:
        models = [
            _model_review_from_aggregate(
                agg, decision_metric, lb.get((agg["model_type"], agg.get("ensemble_id")), {}),
                prediction_counts,
            )
            for agg in aggregate_rows
        ]
    else:
        models = [_model_review_from_leaderboard(r, prediction_counts) for r in leaderboard_rows]
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

    Reads the header (`registry.bq.read_run_summary`), the config (for the decision metric and
    series count), the leaderboard (`registry.bq.read_leaderboard`), the cross-series aggregates
    (`registry.bq.read_metric_aggregates`) and per-model prediction counts
    (`registry.bq.read_prediction_counts`), then composes via `_assemble_review`.
    """
    from .registry import bq

    summary = bq.read_run_summary(run_id, settings=settings)
    raw = bq.read_run_config(run_id, settings=settings)
    cfg = RunConfig.model_validate(raw) if raw else None
    decision_metric = cfg.backtest.decision_metric if cfg else "wape"
    n_series = (summary or {}).get("n_series") or (cfg.data.series_limit if cfg else None)
    leaderboard_rows = bq.read_leaderboard(run_id, settings=settings)
    aggregate_rows = bq.read_metric_aggregates(run_id, settings=settings)
    prediction_counts = bq.read_prediction_counts(run_id, settings=settings)
    return _assemble_review(
        run_id, summary, decision_metric, n_series,
        leaderboard_rows, aggregate_rows, prediction_counts,
    )


# --- plots (lazy matplotlib) ---------------------------------------------------
#
# Palette validated with the dataviz skill's checker (do not eyeball / re-pick by taste):
#   - base vs ensemble #0072B2/#E69F00 — CVD ΔE 29.2 (PASS); the orange's sub-3:1 surface contrast
#     is relieved by the direct value label every bar carries.
#   - status green/blue/vermillion #009E73/#0072B2/#D55E00 (PASS separation); pending is gray
#     #999999 by design (a status, not a categorical hue) and every bar is annotated with its status
#     text, so identity is never colour-alone.
_BASE_COLOR = "#0072B2"
_ENSEMBLE_COLOR = "#E69F00"
_STATUS_COLORS: dict[str | None, str] = {
    "COMPLETED": "#009E73",
    "RUNNING": "#0072B2",
    "PENDING": "#999999",
    "FAILED": "#D55E00",
    "PARTIAL": "#E69F00",
    "CANCELLED": "#CC79A7",
}
_STATUS_DEFAULT = "#999999"


def plot_progress(progress: RunProgress, *, ax: Any = None, title: str | None = None) -> Any:
    """Render a `RunProgress` as a per-family progress bar chart and return the matplotlib ``Axes``.

    One horizontal bar per family — length is the fraction of expected cells that have landed,
    colour is the family's job status (`_STATUS_COLORS`) — with the ``done/expected`` count and
    status labelled at the bar end (the label doubles as the status's secondary encoding). Families
    keep DAG order, ensemble last. matplotlib imports lazily so it never touches the package import;
    an empty run renders an empty titled axes rather than raising.
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
    for y, f in zip(ys, fams, strict=True):
        expected = f.n_expected if f.n_expected is not None else "?"
        ax.text(
            (f.fraction if f.fraction is not None else 0.0) + 0.01,
            y,
            f"{f.n_done}/{expected} · {f.status or 'pending'}",
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
