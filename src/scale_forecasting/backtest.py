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

Public surface: ``Fold``, ``OOF_COLUMNS``, ``achievable_folds``, ``make_folds``, ``backtest_cell``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        return self.val_end - self.val_start


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
            )
        )
    return folds


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
        yhat = pred["yhat"].to_numpy()[: fold.val_size]
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
            compute_metrics(
                y_true,
                yhat,
                y_train=y_train_orig,
                lower=lower,
                upper=upper,
                seasonal_period=seasonal_period(cfg.data.freq),
            )
        )

    oof = (
        pd.concat(oof_parts, ignore_index=True)
        if oof_parts
        else pd.DataFrame(columns=list(OOF_COLUMNS))
    )
    return oof, fold_metrics
