"""Overall Percentage Error — ``|sum(yhat) - sum(y)| / |sum(y)|``."""

from __future__ import annotations

import numpy as np

from .base_metric import BaseMetric, MetricContext, register


class Ope(BaseMetric):
    """Overall Percentage Error over the evaluation window: ``|sum(yhat) - sum(y)| / |sum(y)|``."""

    name = "ope"
    direction = "lower"

    def compute(self, ctx: MetricContext) -> float:
        denom = float(np.abs(np.sum(ctx.y_true)))
        if denom == 0.0:
            return float("nan")
        num = float(np.abs(np.sum(ctx.yhat) - np.sum(ctx.y_true)))
        return num / denom


register(Ope)
