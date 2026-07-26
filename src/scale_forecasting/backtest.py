"""Time-series cross-validation folds + out-of-fold capture — pure (CONTRACTS §2.2).

Backtesting fits on history, validates on a held-out future window, and records the
out-of-fold (OOF) predictions the learned ensembler trains on (DESIGN §5.1/§5.2). Two
entry points:

- ``make_folds(n, cfg) -> list[Fold]`` — integer-indexed CV splits over ``n`` sorted
  observations. Folds are anchored from the end: the latest fold validates on the final
  ``horizon`` points, earlier folds step back by ``step``. ``expanding`` grows the train
  window from 0; ``sliding`` keeps a fixed ``min_train`` window.
- ``backtest_cell(series, model, cfg) -> (oof, fold_metrics)`` — features are built once
  (leakage-free: lags only look backward), then a **fresh** model is fit per fold and
  scored on its validation window.

The no-leakage invariant is ``train_end == val_start`` for every fold: training data
strictly precedes the validation window.

Public surface: ``Fold``, ``make_folds``, ``backtest_cell``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from .errors import ConfigError
from .features import build_features, invert_transform
from .metrics import compute_metrics

if TYPE_CHECKING:
    from .config import RunConfig
    from .models.base_model import BaseModel


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


def make_folds(n: int, cfg: RunConfig) -> list[Fold]:
    """Build the CV folds for ``n`` observations (CONTRACTS §2.2).

    Uses ``cfg.backtest``: ``n_folds``, ``horizon``, ``step``, ``min_train``, ``scheme``.
    Raises ``ConfigError`` if ``n`` is too small to support the requested folds.
    """
    bt = cfg.backtest
    horizon, step, n_folds, min_train = bt.horizon, bt.step, bt.n_folds, bt.min_train

    # Earliest fold's validation window must start at or after min_train.
    earliest_val_start = n - horizon - (n_folds - 1) * step
    if earliest_val_start < min_train:
        need = min_train + horizon + (n_folds - 1) * step
        raise ConfigError(
            f"not enough data for backtest: need >= {need} observations for "
            f"{n_folds} folds (horizon={horizon}, step={step}, min_train={min_train}), got {n}"
        )

    folds: list[Fold] = []
    for k in range(n_folds):
        val_start = n - horizon - (n_folds - 1 - k) * step
        val_end = val_start + horizon
        train_end = val_start
        train_start = 0 if bt.scheme == "expanding" else max(0, train_end - min_train)
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
    model: Callable[[], BaseModel],
    cfg: RunConfig,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Run CV for one series and model factory (CONTRACTS §2.2).

    Args:
        series: one ts_id's raw rows (date/target/exog columns).
        model: a **factory** returning a freshly-constructed model, called once per fold so
            no fitted state leaks across folds.
        cfg: the run config (drives features and fold geometry).

    Returns:
        ``(oof, fold_metrics)`` where ``oof`` is the canonical OOF frame (§2.2: ``ds``,
        ``fold_id``, ``y_true``, ``yhat``) concatenated across folds, and ``fold_metrics``
        is the per-fold metric panel (list, in fold order).
    """
    y, X = build_features(series, cfg)
    n = len(y)
    folds = make_folds(n, cfg)

    oof_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float]] = []

    for fold in folds:
        y_train = y.iloc[fold.train_start : fold.train_end]
        X_train = X.iloc[fold.train_start : fold.train_end] if X is not None else None
        y_val = y.iloc[fold.val_start : fold.val_end]
        X_val = X.iloc[fold.val_start : fold.val_end] if X is not None else None

        est = model()
        est.fit(y_train, X_train)
        pred = est.predict(fold.val_size, X_val)

        # Align yhat to the true validation dates by position (folds are contiguous).
        # yhat is already in original units (predict inverts the transform, §2.1), so
        # y_true / y_train are inverted here to score in the same units.
        yhat = pred["yhat"].to_numpy()[: fold.val_size]
        y_true = invert_transform(y_val.to_numpy(), cfg.features.transform)
        y_train_orig = invert_transform(y_train.to_numpy(), cfg.features.transform)
        val_dates = y_val.index

        oof_parts.append(
            pd.DataFrame(
                {
                    "ds": pd.DatetimeIndex(val_dates).as_unit("ns"),
                    "fold_id": fold.fold_id,
                    "y_true": y_true,
                    "yhat": yhat,
                }
            )
        )
        fold_metrics.append(compute_metrics(y_true, yhat, y_train=y_train_orig))

    oof = (
        pd.concat(oof_parts, ignore_index=True)
        if oof_parts
        else pd.DataFrame(columns=["ds", "fold_id", "y_true", "yhat"])
    )
    return oof, fold_metrics
