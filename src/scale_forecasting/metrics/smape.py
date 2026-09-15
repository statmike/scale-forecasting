"""SMAPE — symmetric MAPE. ``mean(2|e| / (|y_true| + |yhat|))``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class SMAPE(BaseMetric):
    """Symmetric mean absolute percentage error — MAPE that survives a zero actual.

    Symmetric because the denominator carries both the actual and the forecast, so it is bounded
    at 2 and does not blow up as the actual approaches zero. Where both are exactly zero the term
    is 0/0, defined here as 0: the forecast was right.
    """

    name: ClassVar[str] = "smape"
    direction: ClassVar[MetricDirection] = "lower"

    def compute(self, ctx: MetricContext) -> float:
        denom = np.abs(ctx.y_true) + np.abs(ctx.yhat)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(denom == 0, 0.0, 2.0 * ctx.abs_err / denom)
        return float(np.mean(terms))


register(SMAPE)
