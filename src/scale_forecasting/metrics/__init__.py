"""Forecast metric panel — pure, and the factory that assembles it.

One entry point, ``compute_metrics``, returns the full panel every run so users never re-run to get
a different metric; the decision metric is then a pure config choice. All values are floats, with
NaN where a metric is undefined (MAPE with zeros, MASE without training history, coverage without
intervals) rather than raising — a metric that can't be computed for one cell must not sink
the batch.

Importing this package registers every metric file by name (each metric module ends with
``register(...)``); ``get_metric(name)`` returns the class and ``list_metrics()`` lists the
registered names. Adding a metric is one new file, one line in the import block, one line in
`METRIC_NAMES`, and the ``ADD COLUMN IF NOT EXISTS`` migration `registry.ddl.render_migrations`
already emits — see `docs/metric_template.py`.

Public surface: ``compute_metrics``, ``METRIC_NAMES``, ``METRIC_DIRECTION``, ``loss_of``,
``get_metric``, ``list_metrics``, ``BaseMetric``, ``MetricContext``, ``MetricDirection``,
``register``.

Definitions (n = horizon, e = yhat - y_true):

- mae   = mean(|e|)
- rmse  = sqrt(mean(e²));  mse = mean(e²)
- mape  = mean(|e| / |y_true|)          (NaN if any y_true == 0)
- smape = mean(2|e| / (|y_true| + |yhat|))
- wape  = sum(|e|) / sum(|y_true|)      (NaN if sum(|y_true|) == 0)
- mase  = mae / mae_naive,   naive = one-step (m=1) on y_train
- rmsse = rmse / rmse_naive, naive = one-step (m=1) on y_train
- bias  = mean(e)   (mean error / ME)
- coverage = fraction of y_true within [lower, upper]  (needs intervals)
- pinball  = mean quantile loss across the interval bounds (needs intervals)
- mase_seasonal = mae / mae_naive, naive = *seasonal* (m=`seasonal_period`) on y_train
- maape = mean(arctan(|e| / |y_true|))  — defined at y_true == 0, unlike MAPE
- interval_score  = mean Winkler score of [lower, upper] at α = 0.2  (needs intervals)
- interval_width  = mean(upper - lower)  (needs intervals)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import ConfigError

# --- metric registration imports (side-effect: each calls register()) ----------
# One line per metric file. Alphabetical, because the import block does *not* set the panel
# order — `METRIC_NAMES` below does, so that an import sorter can never silently re-order the
# table's columns.
from . import (  # noqa: E402,F401
    bias,
    coverage,
    interval_score,
    interval_width,
    maape,
    mae,
    mape,
    mase,
    mase_seasonal,
    mse,
    pinball,
    rmse,
    rmsse,
    smape,
    wape,
)
from .base_metric import (
    _REGISTRY,
    BaseMetric,
    MetricContext,
    MetricDirection,
    register,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = [
    "METRIC_DIRECTION",
    "METRIC_NAMES",
    "BaseMetric",
    "MetricContext",
    "MetricDirection",
    "compute_metrics",
    "get_metric",
    "list_metrics",
    "loss_of",
    "register",
]

# Panel order — kept identical to config.DecisionMetric / the DDL, all three of which are
# generated from this one tuple. New metrics append at the *tail*: the DDL and the Storage Write
# API spec are generated from this order, and only a tail append leaves the existing columns where
# they are. This is a hand-written list rather than the registry's insertion order precisely
# because insertion order is import order, and import order is whatever the formatter last decided.
METRIC_NAMES: tuple[str, ...] = (
    "mae",
    "rmse",
    "mse",
    "mape",
    "smape",
    "wape",
    "mase",
    "rmsse",
    "bias",
    "coverage",
    "pinball",
    "mase_seasonal",
    "maape",
    "interval_score",
    "interval_width",
)

# A registered metric missing from the panel order would never be computed and would have no
# column; a name in the panel order with no registered metric would be a KeyError on the first
# scored cell of a live run. Both are one-line mistakes in a deployment's own metric file, so they
# are caught here, at import, rather than on a cluster.
_missing = set(_REGISTRY) - set(METRIC_NAMES)
_unknown = set(METRIC_NAMES) - set(_REGISTRY)
if _missing or _unknown:
    raise ConfigError(
        "metric panel and registry disagree: "
        f"registered but not in METRIC_NAMES {sorted(_missing)}; "
        f"in METRIC_NAMES but not registered {sorted(_unknown)}"
    )

# Which way is better, for every metric in the panel — read off the classes so the answer lives
# next to the number it describes. See `base_metric.MetricDirection` for why there are three
# answers and not two.
METRIC_DIRECTION: dict[str, str] = {name: _REGISTRY[name].direction for name in METRIC_NAMES}


def get_metric(name: str) -> type[BaseMetric]:
    """Return the registered metric class for ``name``.

    Raises ``ConfigError`` with the available names when ``name`` is unknown.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise ConfigError(f"unknown metric '{name}'; registered metrics: {known}") from None


def list_metrics() -> list[str]:
    """All registered metric names, sorted."""
    return sorted(_REGISTRY)


def loss_of(metric: str, value: float) -> float:
    """``value`` restated as a **loss**: non-negative, zero is perfect, smaller is better (pure).

    One direction map, one conversion, for every caller that has to rank models — the optimiser,
    the inverse-error weighting and the pruner. Each of those wants the same thing, and each used
    to hard-code its own idea of what "worse" means.

    The conversions:

    * ``lower`` → the value itself.
    * ``higher`` → ``1 - value``. Coverage is the only such metric and it is a fraction, so the
      shortfall from perfect coverage is the natural loss. Returning ``-value`` would rank
      identically but be *negative*, and a negative loss cannot be inverted into a weight.
    * ``zero`` → ``abs(value)``. A bias of −0.1 is better than one of +0.4, and both are worse
      than 0.
    * A metric this table does not know, or a NaN, → ``inf``: unrankable, so it can never win.

    Two caveats worth stating rather than burying. ``coverage`` is treated as higher-is-better
    although it is really best *at nominal* — 98% coverage from a wildly over-wide interval is not
    better than 80% from a calibrated one. Judging it against nominal needs the interval's α,
    which is not in the panel, and `interval_score` is the metric that already penalises width
    honestly. And ``interval_width`` is scored lower-is-better, which in isolation rewards an
    interval of zero width; it is a diagnostic to read beside `coverage`, not a `decision_metric`
    to optimise alone.
    """
    if value != value:  # NaN — a metric that could not be computed ranks last, never first
        return float("inf")
    # Read through the registry rather than the frozen `METRIC_DIRECTION` snapshot, so a metric
    # registered after this module was imported still ranks correctly.
    metric_cls = _REGISTRY.get(metric)
    direction = None if metric_cls is None else metric_cls.direction
    if direction == "higher":
        return 1.0 - value
    if direction == "zero":
        return abs(value)
    if direction == "lower":
        return value
    return float("inf")  # unknown metric: no opinion is safer than a wrong one


def compute_metrics(
    y_true: Sequence[float] | np.ndarray,
    yhat: Sequence[float] | np.ndarray,
    y_train: Sequence[float] | np.ndarray | None = None,
    lower: Sequence[float] | np.ndarray | None = None,
    upper: Sequence[float] | np.ndarray | None = None,
    seasonal_period: int | None = None,
) -> dict[str, float]:
    """Compute the full metric panel for one forecast window.

    Builds one `MetricContext` and asks it for every name in `METRIC_NAMES`. Because the context
    memoises, a metric that another metric already needed — ``mae`` under ``mase`` — is computed
    once no matter how many times it is asked for or what order the panel is in.

    Args:
        y_true: actuals over the evaluation window.
        yhat: point forecasts, aligned to ``y_true``.
        y_train: training-history actuals; required for scale-free MASE/RMSSE (else NaN).
            **This fold's training window, not the whole series.** MASE and RMSSE divide by the
            mean step of whatever is handed in, so passing history that overlaps ``y_true``
            scales the score by data the model was judged on, and two engines that disagree
            about it publish two incomparable numbers into the same leaderboard column. Every
            caller derives it the same way — `backtest.training_window`, or the fold's own
            training slice in `backtest.backtest_cell`.
        lower: lower prediction bound; with ``upper`` enables coverage/pinball (else NaN).
        upper: upper prediction bound.
        seasonal_period: steps in one seasonal cycle, from `seasonality.seasonal_period`;
            required for ``mase_seasonal`` (else NaN). Callers pass the run frequency's
            period rather than a default, because guessing it here would silently score
            an hourly run against a weekly naive.

    Returns:
        ``{name: float}`` for every name in `METRIC_NAMES`, in that order. Undefined metrics
        are NaN.

    Raises:
        ValueError: if ``y_true`` and ``yhat`` have different lengths or are empty.
    """
    ctx = MetricContext.from_arrays(
        y_true,
        yhat,
        y_train=y_train,
        lower=lower,
        upper=upper,
        seasonal_period=seasonal_period,
    )
    return {name: ctx.value(name) for name in METRIC_NAMES}
