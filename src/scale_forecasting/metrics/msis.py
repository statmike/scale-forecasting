"""Mean Scaled Interval Score (M4 competition interval metric)."""

from __future__ import annotations

import numpy as np

from .base_metric import BaseMetric, MetricContext, register


class Msis(BaseMetric):
    """Mean Scaled Interval Score: Winkler interval score scaled by seasonal naive MAE."""

    name = "msis"
    direction = "lower"
    needs_intervals = True
    needs_train_history = True
    needs_seasonal_period = True

    def compute(self, ctx: MetricContext) -> float:
        y_train = ctx.y_train
        m = ctx.seasonal_period
        if y_train is None or m is None or m < 1 or len(y_train) <= m:
            return float("nan")
        scale = float(np.mean(np.abs(y_train[m:] - y_train[:-m])))
        if scale == 0.0:
            return float("nan")
        return ctx.value("interval_score") / scale


register(Msis)
