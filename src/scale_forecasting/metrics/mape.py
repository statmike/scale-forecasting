"""MAPE — mean absolute percentage error. ``mean(|e| / |y_true|)``, NaN if any actual is zero."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class MAPE(BaseMetric):
    """Mean absolute percentage error — the familiar one, and the brittle one.

    A single zero actual makes the ratio undefined, and this returns NaN for the **whole window**
    rather than dropping that point: a MAPE computed over the subset of dates that happened to be
    non-zero is a different metric with the same name, and it would sit in the same leaderboard
    column as the honest one. `maape` is the panel's answer for intermittent series.
    """

    name: ClassVar[str] = "mape"
    direction: ClassVar[MetricDirection] = "lower"

    def compute(self, ctx: MetricContext) -> float:
        if np.any(ctx.y_true == 0):
            return float("nan")  # undefined near zeros
        return float(np.mean(ctx.abs_err / np.abs(ctx.y_true)))


register(MAPE)
