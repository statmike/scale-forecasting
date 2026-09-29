"""Root Mean Squared Logarithmic Error — ``sqrt(mean((log1p(y) - log1p(yhat))^2))``."""

from __future__ import annotations

import math

import numpy as np

from .base_metric import BaseMetric, MetricContext, register


class Rmsle(BaseMetric):
    """Root Mean Squared Logarithmic Error (undefined when any ``y < 0`` or ``yhat < 0``)."""

    name = "rmsle"
    direction = "lower"

    def compute(self, ctx: MetricContext) -> float:
        if np.any(ctx.y_true < 0.0) or np.any(ctx.yhat < 0.0):
            return float("nan")
        log_diff = np.log1p(ctx.y_true) - np.log1p(ctx.yhat)
        return math.sqrt(float(np.mean(log_diff**2)))


register(Rmsle)
