"""RMSE — root mean squared error. ``sqrt(mean(e²))``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class RMSE(BaseMetric):
    """Root mean squared error, back in the units of the series.

    Taken as the square root of `mse` from the context rather than recomputed from the residual,
    so the two can never be inconsistent with each other on the same window.
    """

    name: ClassVar[str] = "rmse"
    direction: ClassVar[MetricDirection] = "lower"

    def compute(self, ctx: MetricContext) -> float:
        return math.sqrt(ctx.value("mse"))


register(RMSE)
