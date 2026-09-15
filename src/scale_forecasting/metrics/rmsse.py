"""RMSSE — root mean squared scaled error. ``rmse / rmse(one-step naive on y_train)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class RMSSE(BaseMetric):
    """`mase`'s quadratic twin: squared error scaled by the one-step naive's squared error.

    Same scale-free property and the same training-window rule as `mase`, but it punishes a few
    large misses harder than many small ones — which is what you want when a single bad week costs
    more than a month of drift. The M5 competition scored on its weighted form.

    NaN under the same three conditions as `mase`.
    """

    name: ClassVar[str] = "rmsse"
    direction: ClassVar[MetricDirection] = "lower"
    # A positive constant divides `rmse`, which does not move where the minimum is.
    mean_optimal: ClassVar[bool] = True
    needs_train_history: ClassVar[bool] = True

    def compute(self, ctx: MetricContext) -> float:
        if ctx.y_train is None or ctx.y_train.size < 2:
            return float("nan")
        naive_err = np.abs(np.diff(ctx.y_train))
        scale = float(np.sqrt(np.mean(naive_err**2)))
        if scale == 0:
            return float("nan")  # flat training history → undefined scaling
        return ctx.value("rmse") / scale


register(RMSSE)
