"""Interval width — the mean width of the prediction interval, in the units of the series."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class IntervalWidth(BaseMetric):
    """The sharpness half of `interval_score`, reported on its own because a planner reads it.

    It is in the units of the series, so "how uncertain is this forecast" needs no calibration
    lesson to interpret — which is exactly why it is here and why it is a diagnostic rather than a
    `decision_metric`. Scored lower-is-better, and in isolation that rewards an interval of zero
    width, which is why it belongs beside `coverage` and not instead of it.

    NaN without both bounds.
    """

    name: ClassVar[str] = "interval_width"
    direction: ClassVar[MetricDirection] = "lower"
    needs_intervals: ClassVar[bool] = True

    def compute(self, ctx: MetricContext) -> float:
        if not ctx.has_intervals:
            return float("nan")
        return float(np.mean(ctx.upper - ctx.lower))


register(IntervalWidth)
