"""Interval score — the mean Winkler score of the prediction interval at α = 0.2."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ._intervals import INTERVAL_ALPHA
from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class IntervalScore(BaseMetric):
    """Sharpness *and* calibration in one number — the honest way to rank an interval.

    `coverage` says whether the actuals fell inside; `interval_width` says how wide the band was.
    Either alone is trivially gamed: an infinite band covers everything, a zero-width one is
    maximally sharp. The Winkler score is the width plus a ``2/α`` penalty for each miss,
    proportional to how far outside it landed, so widening only pays when it buys back more
    penalty than it costs. Lower is better.

    NaN without both bounds.
    """

    name: ClassVar[str] = "interval_score"
    direction: ClassVar[MetricDirection] = "lower"
    needs_intervals: ClassVar[bool] = True

    def compute(self, ctx: MetricContext) -> float:
        if not ctx.has_intervals:
            return float("nan")
        lo, up, yt = ctx.lower, ctx.upper, ctx.y_true
        penalty = 2.0 / INTERVAL_ALPHA
        score = (up - lo) + penalty * np.maximum(lo - yt, 0.0) + penalty * np.maximum(yt - up, 0.0)
        return float(np.mean(score))


register(IntervalScore)
