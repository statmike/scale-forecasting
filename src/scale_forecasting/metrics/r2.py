"""Coefficient of Determination — ``1 - sum((y - yhat)^2) / sum((y - mean(y))^2)``."""

from __future__ import annotations

import numpy as np

from .base_metric import BaseMetric, MetricContext, register


class R2(BaseMetric):
    """Coefficient of determination (R-squared) over the evaluation window."""

    name = "r2"
    direction = "higher"
    mean_optimal = True

    def compute(self, ctx: MetricContext) -> float:
        if len(ctx.y_true) < 2:
            return float("nan")
        ss_tot = float(np.sum((ctx.y_true - np.mean(ctx.y_true)) ** 2))
        if ss_tot == 0.0:
            return float("nan")
        ss_res = float(np.sum(ctx.err**2))
        return 1.0 - (ss_res / ss_tot)


register(R2)
