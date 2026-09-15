"""Coverage — the fraction of actuals that fell inside the prediction interval."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class Coverage(BaseMetric):
    """Fraction of ``y_true`` within ``[lower, upper]`` — the only higher-is-better metric.

    It is scored higher-is-better, which is a simplification worth naming: coverage is really best
    *at nominal*, and 98% coverage from a wildly over-wide band is not better than 80% from a
    calibrated one. Judging it against nominal needs the interval's α, which is not in the panel,
    and `interval_score` is the metric that already penalises width honestly. Read the two together
    and coverage on its own stops being misleading.

    NaN without both bounds.
    """

    name: ClassVar[str] = "coverage"
    direction: ClassVar[MetricDirection] = "higher"
    needs_intervals: ClassVar[bool] = True

    def compute(self, ctx: MetricContext) -> float:
        if not ctx.has_intervals:
            return float("nan")
        inside = (ctx.y_true >= ctx.lower) & (ctx.y_true <= ctx.upper)
        return float(np.mean(inside))


register(Coverage)
