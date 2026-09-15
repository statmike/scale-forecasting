"""MAAPE — mean arctangent absolute percentage error. ``mean(arctan(|e| / |y_true|))``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class MAAPE(BaseMetric):
    """MAPE that survives zeros, by bounding the ratio instead of dividing by it.

    `mape` is NaN for the whole window if a single actual is zero. Taking the arctangent of the
    ratio caps a zero actual's contribution at π/2 — the bounded worst case — instead of letting it
    poison everything, so intermittent-demand series get a percentage-flavoured score they can
    actually be ranked by. The range is [0, π/2], which means it is comparable across series but
    not readable as a percentage.

    Where the actual is zero *and* the error is zero the ratio is 0/0: a perfect match at a zero
    actual, defined here as 0 rather than as the worst case.
    """

    name: ClassVar[str] = "maape"
    direction: ClassVar[MetricDirection] = "lower"

    def compute(self, ctx: MetricContext) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(ctx.abs_err / np.abs(ctx.y_true))
        ratio = np.where((ctx.y_true == 0) & (ctx.abs_err == 0), 0.0, ratio)
        return float(np.mean(np.arctan(ratio)))


register(MAAPE)
