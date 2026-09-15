"""TEMPLATE — copy me into src/scale_forecasting/metrics/ to add a metric.

This is not a registered metric (it lives under docs/, which the factory never imports). To
add a real one:

  1. Copy this file to ``src/scale_forecasting/metrics/<your_metric>.py``.
  2. Rename the class and set ``name`` to a unique, lowercase, snake_case string. It becomes a
     BigQuery column, so it must be a bare identifier.
  3. Set ``direction``, any ``needs_*`` flags and ``mean_optimal``, and fill in ``compute``.
  4. Add one import line to ``src/scale_forecasting/metrics/__init__.py`` and one entry to
     ``METRIC_NAMES`` there, at the position you want the column to sit in the table.

That's the whole checklist. The ``forecast_metadata`` column, the Storage Write API spec, the
leaderboard aggregate projection and the ``ADD COLUMN IF NOT EXISTS`` migration are all derived
from ``METRIC_NAMES``, so none of them is an edit you make — the next run's ``ensure_tables``
adds the column to an existing deployment on its own.

See ``docs/adding_a_metric.md`` for the walkthrough and the contract every metric owes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base_metric import BaseMetric, MetricDirection, register

if TYPE_CHECKING:
    from .base_metric import MetricContext


class TemplateMetric(BaseMetric):
    """A one-line description of what your metric measures, then the caveats worth knowing.

    Say what makes it NaN, and say what it should be read beside. A metric that is undefined for
    some windows is normal; a reader who does not know which ones is not.
    """

    # --- registration metadata (read by the factory) --------------------------
    # Unique, lowercase, snake_case, and a valid SQL identifier — this becomes a FLOAT64 column
    # on `forecast_metadata` and a generated identifier in the leaderboard projection.
    name: ClassVar[str] = "template"
    # "lower" for an error, "higher" for a fraction of successes, "zero" for a signed quantity
    # where either sign is a fault. This is what `loss_of` reads to turn your metric into a
    # comparable loss, so a wrong answer here silently inverts ensemble weighting and pruning.
    direction: ClassVar[MetricDirection] = "lower"
    # Declare what the metric reads. These do not gate anything — your compute() must still be
    # NaN-safe on its own. They let config validation warn at plan time that a run cannot produce
    # this metric, rather than leaving an operator to infer it from an empty column.
    needs_intervals: ClassVar[bool] = False  # True if you read ctx.lower / ctx.upper
    needs_train_history: ClassVar[bool] = False  # True if you read ctx.y_train
    needs_seasonal_period: ClassVar[bool] = False  # True if you read ctx.seasonal_period
    # True only if your loss is quadratic in the error, i.e. the *mean* of the predictive
    # distribution minimises it. Every absolute-error shape and every proper interval score leaves
    # this False, because the median does. `config.corrected_arm_for` reads it to pick which
    # corrected point forecast a run ships under `output.point_forecast="auto"`.
    mean_optimal: ClassVar[bool] = False

    def compute(self, ctx: MetricContext) -> float:
        """Return this window's value, or NaN where it is undefined. Must never raise.

        ``ctx`` is the whole input surface: ``y_true`` and ``yhat`` (equal-length float arrays,
        never empty), the precomputed ``err = yhat - y_true`` and ``abs_err``, and the optional
        ``y_train`` / ``lower`` / ``upper`` / ``seasonal_period``. Nothing else — a metric never
        reads global config, which is what lets the identical code score a Spark cell, a Ray cell
        and a BigQuery-native fold.

        Guard every divisor and every optional input, and return ``float("nan")`` rather than
        raising: one undefined cell must never sink a batch.
        """
        # If your metric builds on another one, ask the context for it by name instead of
        # recomputing it — `ctx.value("mae")` is memoised across the whole panel, so the
        # arithmetic happens once and the math for `mae` stays in mae.py. That is how `mase`
        # is written:
        #
        #     scale = float(np.mean(np.abs(np.diff(ctx.y_train))))
        #     return ctx.value("mae") / scale
        #
        # (placeholder "metric": weighted absolute percentage error, i.e. `wape`)
        denominator = float(np.sum(np.abs(ctx.y_true)))
        if denominator == 0:
            return float("nan")
        return float(np.sum(ctx.abs_err)) / denominator


# The one line that registers the metric with the factory. Without it, the metric is invisible.
register(TemplateMetric)
