"""MSE — mean squared error. ``mean(e²)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class MSE(BaseMetric):
    """Mean squared error — squared units, so read it beside `rmse` rather than instead of it.

    Its place in the panel is as a quadratic-loss member: `config.corrected_arm_for` uses the fact
    that squared error is minimised by the *mean* to decide which point-forecast arm a run
    optimising this metric should be scored on.
    """

    name: ClassVar[str] = "mse"
    direction: ClassVar[MetricDirection] = "lower"
    # Squared loss is minimised by the mean. This is the canonical case.
    mean_optimal: ClassVar[bool] = True

    def compute(self, ctx: MetricContext) -> float:
        return float(np.mean(ctx.err**2))


register(MSE)
