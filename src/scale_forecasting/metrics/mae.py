"""MAE — mean absolute error. ``mean(|e|)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class MAE(BaseMetric):
    """Mean absolute error, in the units of the series.

    The plainest metric in the panel and the numerator of three others (`mase`, `mase_seasonal`,
    and by way of `mse` the scale for `rmsse`), which is why it is worth having memoised on the
    context rather than recomputed.
    """

    name: ClassVar[str] = "mae"
    direction: ClassVar[MetricDirection] = "lower"

    def compute(self, ctx: MetricContext) -> float:
        return float(np.mean(ctx.abs_err))


register(MAE)
