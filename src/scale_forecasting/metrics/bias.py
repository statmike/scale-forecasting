"""Bias — mean error. ``mean(e)``, where either sign is a fault."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class Bias(BaseMetric):
    """Mean error — the only signed number in the panel, and the reason directions are a triple.

    Everything else here is an error magnitude where smaller is better. Bias is a direction: −5
    means the forecast runs low by five units on average, +5 that it runs high, and 0 that it is
    unbiased. Ranking it as lower-is-better would hand the prize to the model that under-forecasts
    hardest, which is why `loss_of` maps ``"zero"`` to ``abs(value)``.
    """

    name: ClassVar[str] = "bias"
    direction: ClassVar[MetricDirection] = "zero"

    def compute(self, ctx: MetricContext) -> float:
        return float(np.mean(ctx.err))


register(Bias)
