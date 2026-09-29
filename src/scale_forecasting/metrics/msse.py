"""Mean Squared Scaled Error — ``mse / mean(diff(y_train)^2)``."""

from __future__ import annotations

import numpy as np

from .base_metric import BaseMetric, MetricContext, register


class Msse(BaseMetric):
    """Mean Squared Scaled Error (unrooted counterpart of RMSSE)."""

    name = "msse"
    direction = "lower"
    needs_train_history = True
    mean_optimal = True

    def compute(self, ctx: MetricContext) -> float:
        y_train = ctx.y_train
        if y_train is None or len(y_train) < 2:
            return float("nan")
        sq_scale = float(np.mean(np.diff(y_train) ** 2))
        if sq_scale == 0.0:
            return float("nan")
        return ctx.value("mse") / sq_scale


register(Msse)
