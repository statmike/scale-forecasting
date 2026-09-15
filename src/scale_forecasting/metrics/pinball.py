"""Pinball loss — the quantile loss of the two interval bounds, averaged."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ._intervals import LOWER_Q, UPPER_Q
from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class Pinball(BaseMetric):
    """Average quantile loss across the lower (0.1) and upper (0.9) bounds.

    The proper scoring rule for a quantile forecast: it is minimised exactly when each bound sits
    at the quantile it claims to be, so unlike `coverage` it cannot be improved by widening the
    band. Asymmetric by construction — a 0.9 bound is penalised nine times as hard for falling
    below the actual as for sitting above it — which is what makes it measure the *right* tail
    rather than just any tail.

    NaN without both bounds.
    """

    name: ClassVar[str] = "pinball"
    direction: ClassVar[MetricDirection] = "lower"
    needs_intervals: ClassVar[bool] = True

    def compute(self, ctx: MetricContext) -> float:
        if not ctx.has_intervals:
            return float("nan")
        lo = _pinball_q(ctx.y_true, ctx.lower, LOWER_Q)
        up = _pinball_q(ctx.y_true, ctx.upper, UPPER_Q)
        return float(np.mean([lo, up]))


def _pinball_q(yt: np.ndarray, q_forecast: np.ndarray, q: float) -> float:
    """Pinball loss of one quantile forecast at level ``q``."""
    diff = yt - q_forecast
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


register(Pinball)
