"""MASE — mean absolute scaled error. ``mae / mae(one-step naive on y_train)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class MASE(BaseMetric):
    """Error as a multiple of what a one-step naive forecast would have scored in-sample.

    The canonical Hyndman–Koehler definition, whose naive is always ``y_{t-1}``; `mase_seasonal` is
    an *additional* metric against the seasonal naive, not a correction to this one. Below 1 the
    model beats the naive. Because it is unit-free it is the honest way to average a score across
    series of wildly different magnitudes, which is what a leaderboard over a fleet is doing.

    **The denominator is this fold's training window, never the whole series.** The context carries
    whatever the caller passed, and every caller cuts it the same way (`backtest.training_window`),
    because a scale that has seen the scored window is not comparable with one that has not — and
    the two would land in the same column.

    NaN without training history, with fewer than two training points, or when the history is
    exactly flat and there is no naive error to scale by.
    """

    name: ClassVar[str] = "mase"
    direction: ClassVar[MetricDirection] = "lower"
    needs_train_history: ClassVar[bool] = True

    def compute(self, ctx: MetricContext) -> float:
        if ctx.y_train is None or ctx.y_train.size < 2:
            return float("nan")
        scale = float(np.mean(np.abs(np.diff(ctx.y_train))))  # mean |y_t - y_{t-1}|
        if scale == 0:
            return float("nan")  # flat training history → undefined scaling
        return ctx.value("mae") / scale


register(MASE)
