"""Seasonal MASE — ``mae / mae(seasonal naive on y_train)``, the naive being ``y_{t-m}``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class MASESeasonal(BaseMetric):
    """`mase` against the *seasonal* naive instead of the one-step naive.

    On a strongly seasonal series the one-step naive is an easy baseline: last week's value is a
    poor forecast of this week's, so every model beats it and `mase` stops discriminating between
    them. Dividing by ``y_{t-m}`` instead — last Monday for a daily run, last January for a monthly
    one — restores the spread.

    The period comes from the run's frequency by way of `seasonality.seasonal_period`, passed in
    rather than guessed, because a default here would silently score an hourly run against a weekly
    naive and publish the number as if it meant the same thing.

    NaN when the period is unknown, when the history is not longer than one full cycle, or when the
    seasonal naive is exactly flat.
    """

    name: ClassVar[str] = "mase_seasonal"
    direction: ClassVar[MetricDirection] = "lower"
    needs_train_history: ClassVar[bool] = True
    needs_seasonal_period: ClassVar[bool] = True

    def compute(self, ctx: MetricContext) -> float:
        period = ctx.seasonal_period
        if ctx.y_train is None or period is None or period < 1:
            return float("nan")
        tr = ctx.y_train
        if tr.size <= period:
            return float("nan")
        scale = float(np.mean(np.abs(tr[period:] - tr[:-period])))
        if scale == 0:
            return float("nan")
        return ctx.value("mae") / scale


register(MASESeasonal)
