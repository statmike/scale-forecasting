"""Coefficient of Variation of RMSE — ``rmse / |mean(y)|``."""

from __future__ import annotations

import numpy as np

from .base_metric import BaseMetric, MetricContext, register


class Cv(BaseMetric):
    """Coefficient of variation of the root mean squared error: ``rmse / |mean(y)|``."""

    name = "cv"
    direction = "lower"
    mean_optimal = True

    def compute(self, ctx: MetricContext) -> float:
        denom = float(np.abs(np.mean(ctx.y_true)))
        if denom == 0.0:
            return float("nan")
        return ctx.value("rmse") / denom


register(Cv)
