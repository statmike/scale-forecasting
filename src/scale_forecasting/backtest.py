"""Time-series cross-validation folds + out-of-fold capture — pure.

Backtesting fits on history, validates on a held-out future window, and records the
out-of-fold (OOF) predictions the learned ensembler trains on. Two
entry points:

- ``make_folds(n, cfg) -> list[Fold]`` — integer-indexed CV splits over ``n`` sorted
  observations. Folds are anchored from the end: the latest fold validates on the final
  ``horizon`` points, earlier folds step back by ``step``. ``expanding`` grows the train
  window from 0; ``sliding`` keeps a fixed ``min_train`` window. A series too short for the
  requested folds gets as many as it supports (``achievable_folds``), possibly none — never an
  exception, because a scoring shortfall must not cost the forecast.
- ``backtest_cell(series, model, cfg) -> (oof, fold_metrics)`` — features are built once
  (leakage-free: lags only look backward), then a **fresh** model is fit per fold and
  scored on its validation window.

The no-leakage invariant is ``train_end == val_start`` for every fold: training data
strictly precedes the validation window.

Each fold is scored on the *intervals the model already returned*, not on the point forecast
alone — so ``coverage``, ``pinball``, ``interval_score`` and ``interval_width`` are real numbers on
the Python path rather than the NaNs they were for every run before this.

Public surface: ``Fold``, ``OOF_COLUMNS``, ``achievable_folds``, ``holdout_fold_id``,
``hpo_scoring_claim``, ``make_folds``, ``fit_rows``, ``suggest_min_train``, ``training_window``,
``backtest_cell``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .features import build_features, invert_transform
from .metrics import compute_metrics
from .seasonality import seasonal_period

if TYPE_CHECKING:
    from .config import RunConfig
    from .models.base_model import BaseModel


# The canonical OOF frame `backtest_cell` returns, in order. Named once because the empty case has
# to produce the same columns as the populated one — a fold-less series that returned a
# four-column frame while every other series returned eight is the kind of difference that only
# shows up as a concat warning in a 100k run.
OOF_COLUMNS: tuple[str, ...] = (
    "ds",
    "fold_id",
    "y_true",
    "yhat",
    # The model's own point forecast, before any residual bias correction. Both arms are stored
    # because the comparison between them is a deliverable, and a comparison cannot be run after
    # the fact against a number that was never written down. These are also the residuals
    # `calibration.calibrate_from_oof` learns from — measured against the model's *own* output, so
    # the correction is never estimated from data it has already been applied to.
    "yhat_raw",
    "yhat_adjusted",
    "yhat_lower",
    "yhat_upper",
    "cutoff_date",
    "horizon_step",
)


@dataclass(frozen=True)
class Fold:
    """One CV split as integer positions into the sorted series (half-open ranges)."""

    fold_id: int
    train_start: int
    train_end: int  # == val_start (no leakage)
    val_start: int
    val_end: int
    # "holdout" for the newest fold, "fit" for every older one. See `holdout_fold_id`.
    role: str = "fit"

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        return self.val_end - self.val_start


def holdout_fold_id(cfg: RunConfig) -> int:
    """The fold id reserved from every fit — the newest one, ``n_folds - 1`` (pure).

    **The invariant this exists to state: the fold with the smallest step-back is never used to fit
    anything.** Not stacker weights, not ``inverse_error`` weights, not hyperparameters. Everything
    that learns from the backtest learns from the *inner* folds; the newest fold is what those
    learned things are then judged on. Without it, a learned ensemble's leaderboard number is an
    in-sample fit statistic sitting in the same column as the base models' out-of-fold numbers, and
    a per-series hyperparameter search is scored on the exact folds it optimised against.

    It is a pure function of ``cfg`` — not of the series — and that is what makes it usable as a
    join key rather than a per-cell lookup. Two facts make it safe: `make_folds` keeps a survivor's
    original ``fold_id`` and drops the **oldest** folds first, so every series that achieved any
    fold at all achieved this one; and `engines.bigquery_sql.fold_plan` numbers folds identically,
    so the two runtimes agree on which fold is the holdout without having to compare dates.

    A series that achieved exactly one fold has *only* the holdout, so it has nothing to fit on --
    callers fall back to using every fold and record ``ensemble_scoring`` / ``hpo_scoring`` as
    ``'in_sample'`` rather than pretending. Same at ``n_folds == 1``, where that is true of the
    whole run.
    """
    return cfg.backtest.n_folds - 1


def hpo_scoring_claim(cfg: RunConfig) -> str:
    """The run-level answer to "did the hyperparameter search reserve the newest fold?" (pure).

    ``"off"`` when no search runs at all, ``"holdout"`` when the run's fold geometry leaves an
    inner fold for one to score on, ``"in_sample"`` when ``n_folds == 1`` and it cannot. Written
    once onto the run header's ``job_telemetry`` under ``$.scoring.hpo`` so the claim is legible
    for the whole run without reading a per-cell column — the per-cell ``hpo_scoring`` still says
    what happened for an individual short series under per-series tuning, which this cannot.

    Lives here rather than in `hpo` because `main` needs it on the submit path, and importing
    `hpo` there would drag the whole model stack in behind ``models.get_model``.
    """
    if not cfg.hpo.enabled:
        return "off"
    return "holdout" if cfg.backtest.n_folds >= 2 else "in_sample"


def achievable_folds(n: int, cfg: RunConfig) -> int:
    """How many of the requested folds ``n`` observations can actually support (pure).

    ``0`` when the series cannot even hold one fold, ``cfg.backtest.n_folds`` when it holds them
    all. Split out from `make_folds` because two callers need the count without the folds:
    the cell records ``n_folds_achieved``, and a reader deciding whether a run's leaderboard is
    comparable needs to know a series was scored on fewer folds than its neighbours.

    Fold ``k`` validates on ``[n - horizon - (n_folds-1-k)*step, ...)``, so the binding constraint
    is the *oldest* surviving fold's validation start landing at or after ``min_train``.
    """
    bt = cfg.backtest
    slack = n - bt.horizon - bt.min_train
    if slack < 0:
        return 0
    return min(bt.n_folds, slack // bt.step + 1)


def make_folds(n: int, cfg: RunConfig) -> list[Fold]:
    """Build the CV folds for ``n`` observations — as many as the series supports.

    Uses ``cfg.backtest``: ``n_folds``, ``horizon``, ``step``, ``min_train``, ``scheme``.

    **Clamps rather than raises.** A series too short for the requested folds used to raise
    ``ConfigError``, which `run_cell` caught as a cell error — so the forecast was thrown away
    over a *scoring* shortfall, and short history became the single largest error class in the
    registry. The fit itself was never in question. Now the shortest series in a panel returns the
    folds it can support, possibly none, and the caller still fits and forecasts it.

    **Survivors keep their fold_id from the full plan; the OLDEST folds are the ones dropped.**
    Both halves matter. Dropping the oldest keeps every series scored on the most recent window it
    can reach, which is the window a leaderboard is about. Keeping the original numbering means
    ``fold_id`` compares across series: renumbering the survivors 0..k would make ``MAX(fold_id)``
    meaningless in a panel of mixed-length series, and would silently align a short series' fold 0
    against a long series' fold 0 covering a completely different date range. Fold identity is
    anchored on the date, not the ordinal — see `ensembler._pivot_oof`.

    Dropping the oldest is also what makes the holdout universal: fold ``n_folds - 1`` survives for
    every series that achieved any fold at all, so `holdout_fold_id` can be a function of the config
    instead of a per-series lookup. Each fold carries that verdict as ``role``.
    """
    bt = cfg.backtest
    horizon, step, n_folds, min_train = bt.horizon, bt.step, bt.n_folds, bt.min_train

    achieved = achievable_folds(n, cfg)
    if achieved == 0:
        return []

    folds: list[Fold] = []
    for k in range(n_folds - achieved, n_folds):
        val_start = n - horizon - (n_folds - 1 - k) * step
        val_end = val_start + horizon
        train_end = val_start
        # Membership, not equality: `sliding` is the one scheme with a fixed-width window, and
        # `expanding_frozen` differs from `expanding` in how the model is *refit*, not in where
        # training starts. Written as `== "expanding"`, adding that scheme silently gave it sliding
        # geometry — the kind of thing widening a Literal does for free in the digest and not at
        # all in the code.
        train_start = max(0, train_end - min_train) if bt.scheme == "sliding" else 0
        folds.append(
            Fold(
                fold_id=k,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                role="holdout" if k == n_folds - 1 else "fit",
            )
        )
    return folds


def fit_rows(n: int, cfg: RunConfig) -> list[int]:
    """Training-row count of every fit one cell performs on a series of ``n`` observations (pure).

    ``[n, *fold_windows]`` — the final full-history fit the shipped forecast comes from, then one
    entry per *achieved* fold, in fold order. A cell with backtesting off does one fit, so the list
    is ``[n]``.

    This is the arithmetic behind `config.estimate_workload`, and it exists so that arithmetic
    cannot drift from the geometry: the fold windows are read off `make_folds` rather than
    re-derived from ``horizon``/``step``/``min_train``. A count that re-derived them would go
    quietly wrong the first time the geometry changed — and fold geometry has changed twice
    already (clamping, then oldest-first dropping).

    Why the row counts and not just the fit count: a fold trains on *less* history than the final
    fit, so ``n_folds + 1`` overstates the cost of backtesting. Four years of daily history
    backtested twice at a 28-day horizon is 2.94 whole-history fits, not 3.
    """
    if not cfg.backtest.enabled:
        return [n]
    return [n] + [f.train_end - f.train_start for f in make_folds(n, cfg)]


def suggest_min_train(
    obs_counts: Sequence[int], cfg: RunConfig, target_share: float = 0.9
) -> int | None:
    """The largest ``backtest.min_train`` that still gives ``target_share`` of series all folds.

    ``None`` when no positive ``min_train`` reaches the target — the panel is simply too short for
    the requested fold geometry and the fix is fewer folds or a shorter horizon, not a smaller
    ``min_train``.

    A series gets the full ``n_folds`` exactly when
    ``min_train <= n - horizon - (n_folds - 1) * step`` (rearranged from `achievable_folds`), so
    each series has its own ceiling and the answer is the ``target_share`` quantile of those
    ceilings, taken from the long end. Purely advisory: it reports what the geometry permits and
    changes nothing, because ``min_train`` is a config field and moving it moves the run_id.
    """
    if not obs_counts or not 0.0 < target_share <= 1.0:
        return None
    bt = cfg.backtest
    caps = sorted((n - bt.horizon - (bt.n_folds - 1) * bt.step for n in obs_counts), reverse=True)
    # The series at this rank is the marginal one: keep it at full folds and everything longer
    # follows, which is exactly `target_share` of the panel.
    rank = max(1, math.ceil(target_share * len(caps))) - 1
    return caps[rank] if caps[rank] >= 1 else None


def training_window(ds: np.ndarray, y: np.ndarray, cutoff: object, cfg: RunConfig) -> np.ndarray:
    """The slice of a series' own history one fold trained on, given that fold's cutoff (pure).

    MASE and RMSSE divide by the mean step of the *training* data, so which history goes in is not
    a detail — it is the number. `backtest_cell` hands each fold its own slice and always has. The
    two paths that score from a separate history read did not: the BigQuery-native engine and the
    ensemble scorer both passed the **whole** series, including the very window being scored. That
    makes a native model's MASE and a Python model's MASE for the same series answers to different
    questions, which is exactly the comparison the leaderboard exists to support. This is the one
    rule all of them now apply.

    ``ds`` must be datetime64 and sorted ascending, paired positionally with ``y`` — the callers
    read history with ``ORDER BY ts_id, ds``, so it arrives that way. The window is every
    observation at or before ``cutoff``, which is what `engines.bigquery_sql._fit_filter` trains on
    and what ``fold.train_end`` slices to, narrowed to the last ``min_train`` observations under the
    ``sliding`` scheme, whose window is fixed-width by definition.

    Counting that sliding window in *observations* rather than in dates is deliberate: it is what
    `make_folds` does, so the engines agree. On a series with gaps the native SQL's date-space bound
    would take slightly fewer rows — a real difference, and a smaller one than scoring against a
    denominator that has seen the future.

    Returns the whole of ``y`` when the cutoff is missing. That is the behaviour every run had
    before the cutoff was recorded, and it is the only honest answer for an OOF frame that never
    wrote one down.
    """
    if cutoff is None or pd.isna(cutoff):
        return y
    window = y[np.asarray(ds) <= pd.Timestamp(cutoff).to_datetime64()]
    if cfg.backtest.scheme == "sliding":
        window = window[-cfg.backtest.min_train :]
    return window


def backtest_cell(
    series: pd.DataFrame,
    model_factory: Callable[[], BaseModel],
    cfg: RunConfig,
    lam: float | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Run CV for one series and model factory.

    Args:
        series: one ts_id's raw rows (date/target/exog columns).
        model_factory: returns a freshly-constructed model, called once per fold so no
            fitted state leaks across folds.
        cfg: the run config (drives features and fold geometry).
        lam: the cell's fitted Box-Cox λ (None for stateless transforms), so the forward
            transform here matches the inverse the folds' models apply — one λ per cell.

    Returns:
        ``(oof, fold_metrics)`` where ``oof`` is the canonical OOF frame (`OOF_COLUMNS`)
        concatenated across folds, and ``fold_metrics`` is the per-fold metric panel (list, in
        fold order). The registry later augments this frame with ``ts_id``/``model_type`` and
        renames ``ds``→``forecast_date`` before the ensembler consumes it (see
        ``ensembler._pivot_oof``) — this cell emits the bare form.
    """
    y, X = build_features(series, cfg, lam)
    n = len(y)
    folds = make_folds(n, cfg)

    oof_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float]] = []

    for fold in folds:
        y_train = y.iloc[fold.train_start : fold.train_end]
        X_train = X.iloc[fold.train_start : fold.train_end] if X is not None else None
        y_val = y.iloc[fold.val_start : fold.val_end]
        X_val = X.iloc[fold.val_start : fold.val_end] if X is not None else None

        est = model_factory()
        est.fit(y_train, X_train)
        pred = est.predict(fold.val_size, X_val)

        # Align yhat to the true validation dates by position (folds are contiguous).
        # yhat is already in original units (predict inverts the transform), so
        # y_true / y_train are inverted here to score in the same units.
        # Both arms, always. `pred["yhat"]` is the model's own corrected point — for a
        # residual-band model that is `prediction + median(in-sample residual)` computed from *this
        # fold's* training window, so scoring it against this fold's validation slice is honest.
        # `yhat` is then whichever arm the run selected, and it is what `fold_metrics` scores.
        #
        # `mean` scores as `median` here on purpose: the mean shift is an out-of-fold statistic
        # (`calibration.StepCalibration.mean`) and a fold's model never computed one. Both arms
        # mean "corrected"; only the final forecast can tell them apart.
        yhat_raw = pred["yhat_raw"].to_numpy()[: fold.val_size]
        yhat_adjusted = pred["yhat"].to_numpy()[: fold.val_size]
        yhat = yhat_raw if cfg.output.point_forecast == "raw" else yhat_adjusted
        # Every model returns bounds — natively if it has them, from its residual quantiles if not
        # (`BaseModel.residual_intervals`), so these columns are never absent. They used to be
        # computed on every fold and then dropped on the floor: `coverage`, `pinball`,
        # `interval_score` and `interval_width` were four of the fifteen panel metrics and all four
        # were NaN for every Python cell in every run, while the BigQuery-native path scored them.
        # A leaderboard cannot compare the two engines on a metric only one of them fills.
        lower = pred["yhat_lower"].to_numpy()[: fold.val_size]
        upper = pred["yhat_upper"].to_numpy()[: fold.val_size]
        y_true = invert_transform(y_val.to_numpy(), cfg.features.transform, lam)
        y_train_orig = invert_transform(y_train.to_numpy(), cfg.features.transform, lam)
        val_dates = y_val.index

        oof_parts.append(
            pd.DataFrame(
                {
                    "ds": pd.DatetimeIndex(val_dates).as_unit("ns"),
                    "fold_id": fold.fold_id,
                    "y_true": y_true,
                    "yhat": yhat,
                    "yhat_raw": yhat_raw,
                    "yhat_adjusted": yhat_adjusted,
                    "yhat_lower": lower,
                    "yhat_upper": upper,
                    # The last training date — the origin the fold forecasts from, `ds <= cutoff`.
                    # `fold_id` is an ordinal within one series' plan and the native path derives
                    # its folds from a single global `MAX(ds)`, so on a ragged panel the same
                    # `fold_id` is not the same window. The date is, which is why it is recorded
                    # here rather than reconstructed by a reader who would have to know the
                    # geometry to do it.
                    "cutoff_date": y.index[fold.train_end - 1],
                    # 1-based position within this fold's horizon, so "how fast does this model
                    # decay?" is a GROUP BY instead of a re-run. Every fold answers h=1 and h=28
                    # in the same rows; nothing else in the schema separates them.
                    "horizon_step": range(1, fold.val_size + 1),
                }
            )
        )
        fold_metrics.append(
            {
                **compute_metrics(
                    y_true,
                    yhat,
                    y_train=y_train_orig,
                    lower=lower,
                    upper=upper,
                    seasonal_period=seasonal_period(cfg.data.freq),
                ),
                # Which fold this panel belongs to, so a caller can hold the newest one out of a
                # fit (`holdout_fold_id`). The list is in fold order and a survivor keeps its
                # original id, so position would *usually* work and would be wrong exactly when a
                # series is short — the case the invariant is most delicate on. `_rollup_metrics`
                # walks `METRIC_NAMES`, so this extra key is carried, never averaged.
                "fold_id": fold.fold_id,
            }
        )

    oof = (
        pd.concat(oof_parts, ignore_index=True)
        if oof_parts
        else pd.DataFrame(columns=list(OOF_COLUMNS))
    )
    return oof, fold_metrics
