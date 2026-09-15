"""MSE — mean squared error. ``mean(e²)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class MSE(BaseMetric):
    """Mean squared error — squared units, so read it beside `rmse` rather than instead of it.

    Its place in the panel is as the quadratic-loss member: `config.MEAN_OPTIMAL_METRICS` uses the
    fact that squared error is minimised by the *mean* to pick the right point forecast.
    """

    name: ClassVar[str] = "mse"
    direction: ClassVar[MetricDirection] = "lower"

    def compute(self, ctx: MetricContext) -> float:
        return float(np.mean(ctx.err**2))


register(MSE)
