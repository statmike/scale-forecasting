"""WAPE — weighted absolute percentage error. ``sum(|e|) / sum(|y_true|)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class WAPE(BaseMetric):
    """Total absolute error as a fraction of total actual volume — the panel's default.

    It is the default `decision_metric` because it behaves on the data real fleets have: a ratio of
    sums rather than a mean of ratios, so a single small actual cannot dominate it the way it
    dominates `mape`, and it is defined wherever the window's volume is non-zero. NaN only when the
    actuals sum to zero, where there is no volume to express the error as a fraction of.
    """

    name: ClassVar[str] = "wape"
    direction: ClassVar[MetricDirection] = "lower"

    def compute(self, ctx: MetricContext) -> float:
        denom = float(np.sum(np.abs(ctx.y_true)))
        if denom == 0:
            return float("nan")
        return float(np.sum(ctx.abs_err) / denom)


register(WAPE)
