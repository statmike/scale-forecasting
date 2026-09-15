"""The metric interface every metric file implements.

This is the linchpin of the one-metric-one-file rule and the metric factory, and it is a
deliberate mirror of `models/base_model.py`: every metric is a subclass of `BaseMetric` living
in its own file that ends with ``register(...)``, and the factory (``metrics/__init__.py``)
builds ``{name: class}`` at import. Adding a metric is a new file plus one import line — plus,
unlike a model, one ``ADD COLUMN IF NOT EXISTS`` migration, because a metric's value lands in a
typed column of ``forecast_metadata`` and a model's predictions do not.

**A metric is computed by the framework, never supplied by a model.** That is the rule the whole
leaderboard rests on. `engines/bigquery_engine._score_fold` goes to the trouble of pulling each
BigQuery-native fold back to the driver and scoring it here, with this fold's training window as
the scale denominator, precisely so a native model's ``wape`` and a Ray model's ``wape`` are the
same quantity. A model handing back its own number would break that silently, in a column that
still looks uniform. Numbers only a fitting library can produce — AIC, a training loss, an early
-stopping iteration — are *fit diagnostics*, not metrics: they exist for some models and not
others and are not comparable across them, so they belong in their own JSON channel beside
``best_params``, never in the panel.

Metrics never read global config: everything a metric may see arrives through `MetricContext`,
and a metric that cannot be computed returns NaN rather than raising, so one undefined cell never
sinks a batch.

Public surface: ``BaseMetric``, ``MetricContext``, ``MetricDirection``, ``register`` (plus the
``_REGISTRY`` the factory reads).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from ..errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Sequence

# Which way is better, for every metric in the panel. Three answers, not two:
#
#   "lower"  — an error. Smaller is better, zero is perfect. Most of the panel.
#   "higher" — a fraction of successes. Larger is better, one is perfect. Only `coverage`.
#   "zero"   — a signed quantity where either sign is a fault. Only `bias`.
#
# This distinction exists because three separate places used to answer the question by assuming
# "lower", and were therefore wrong for two of the fifteen metrics — `inverse_error` gave a model
# with 50% interval coverage nearly twice the weight of one with 95%, `prune_threshold` dropped the
# accurate model and kept the broken one, and a model whose bias happened to be negative got weight
# zero for being *good*. A config can name any registered metric as its `decision_metric`, so a
# lower-is-better assumption is not a safe default; it is a silent inversion. Declaring the
# direction on the metric class is what keeps the answer next to the number it describes.
MetricDirection = Literal["lower", "higher", "zero"]


@dataclass(frozen=True)
class MetricContext:
    """One scored window, and everything any metric is allowed to see.

    Built once per call to `compute_metrics` and shared by the whole panel, so the arrays every
    metric needs — the residual and its absolute value — are computed once rather than fifteen
    times. Construct it with `from_arrays`, which is where the shape validation lives.

    ``value(name)`` is the other half of the sharing, and the reason a per-metric contract does not
    cost anything in arithmetic. The panel is not fifteen independent numbers: ``mase`` divides
    ``mae``, ``rmsse`` divides ``rmse``, and ``mase_seasonal`` divides ``mae`` again. Rather than
    have those three recompute their numerator, each asks the context for it by name and the
    context memoises the answer. The dependency stays visible in the file that has it, the
    arithmetic still happens once, and the math for ``mae`` stays in ``mae.py`` where a reader
    looking for it will go.
    """

    y_true: np.ndarray
    yhat: np.ndarray
    # yhat - y_true, and its absolute value. Every metric in the panel is a function of one of
    # these two, the actuals, or the interval bounds.
    err: np.ndarray
    abs_err: np.ndarray
    # The fold's training window, already an array. None when the caller had none, which is what
    # makes every scale-free metric NaN rather than an error.
    y_train: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    seasonal_period: int | None = None
    # Memo for `value`. Mutated through the frozen dataclass, which is legal because the field
    # itself is never rebound — only the dict it points at. Excluded from equality and repr so a
    # context that has answered questions still compares equal to one that has not.
    _memo: dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_arrays(
        cls,
        y_true: Sequence[float] | np.ndarray,
        yhat: Sequence[float] | np.ndarray,
        y_train: Sequence[float] | np.ndarray | None = None,
        lower: Sequence[float] | np.ndarray | None = None,
        upper: Sequence[float] | np.ndarray | None = None,
        seasonal_period: int | None = None,
    ) -> MetricContext:
        """Normalise a caller's arrays into a context, or raise on a window that cannot be scored.

        The two raises are the only ones in the metric layer, and they are here rather than in any
        metric because they are not about any one metric: a window whose forecast and actuals have
        different lengths, or no actuals at all, is a caller bug that should stop, not fifteen
        NaNs that read like fifteen undefined metrics.
        """
        yt = np.asarray(y_true, dtype=float)
        yh = np.asarray(yhat, dtype=float)
        if yt.shape != yh.shape:
            raise ValueError(f"y_true and yhat shape mismatch: {yt.shape} vs {yh.shape}")
        if yt.size == 0:
            raise ValueError("y_true is empty")
        err = yh - yt
        return cls(
            y_true=yt,
            yhat=yh,
            err=err,
            abs_err=np.abs(err),
            y_train=None if y_train is None else np.asarray(y_train, dtype=float),
            lower=None if lower is None else np.asarray(lower, dtype=float),
            upper=None if upper is None else np.asarray(upper, dtype=float),
            seasonal_period=seasonal_period,
        )

    @property
    def has_intervals(self) -> bool:
        """True when both bounds are present — what the interval metrics check before scoring."""
        return self.lower is not None and self.upper is not None

    def value(self, name: str) -> float:
        """This window's value for the registered metric ``name``, computed at most once.

        Used by a metric that builds on another (``mase`` on ``mae``). A metric that names one
        that is not registered gets NaN rather than an exception, on the same principle as
        everything else here: an undefined input makes an undefined metric, not a failed run.
        """
        if name in self._memo:
            return self._memo[name]
        metric_cls = _REGISTRY.get(name)
        if metric_cls is None:
            return float("nan")
        # Seeded before computing so a metric that (wrongly) depends on itself gets NaN and
        # returns, instead of recursing until the interpreter gives up.
        self._memo[name] = float("nan")
        result = float(metric_cls().compute(self))
        self._memo[name] = result
        return result


class BaseMetric(ABC):
    """One metric: a name, a direction, what it needs, and how to compute it.

    Subclasses set the class-level attributes and implement `compute`. The ``needs_*`` flags are
    **declarative, not enforcement** — every metric is already NaN-safe when its inputs are absent
    and that behaviour must not change. Their job is to let config validation say at plan time that
    a chosen ``decision_metric`` will be NaN for every cell of this run, instead of leaving an
    operator to work it out from an empty leaderboard.
    """

    # The panel column name. Must be a valid BigQuery column identifier, because it becomes one.
    name: ClassVar[str]
    direction: ClassVar[MetricDirection]
    # True if the metric reads `ctx.lower`/`ctx.upper`, i.e. it is NaN without prediction intervals.
    needs_intervals: ClassVar[bool] = False
    # True if the metric reads `ctx.y_train`, i.e. it is NaN on a run with no backtest history.
    needs_train_history: ClassVar[bool] = False
    # True if the metric reads `ctx.seasonal_period`.
    needs_seasonal_period: ClassVar[bool] = False

    @abstractmethod
    def compute(self, ctx: MetricContext) -> float:
        """The metric's value for ``ctx``, or NaN where it is undefined. Must never raise."""


# The factory registry: name → concrete metric class. Populated by register() at import.
# Insertion order is the panel order — see the ordered import block in `metrics/__init__.py`.
_REGISTRY: dict[str, type[BaseMetric]] = {}


def register(metric_cls: type[BaseMetric]) -> type[BaseMetric]:
    """Register a metric class under its ``name``. Returns the class so it
    doubles as a decorator. Raises on a missing, duplicate or unusable name.
    """
    name = getattr(metric_cls, "name", None)
    if not name:
        raise ConfigError(f"{metric_cls.__name__} must set a class-level 'name' before register()")
    # The name becomes a FLOAT64 column on `forecast_metadata` and a generated SQL identifier in
    # the leaderboard projection, so a name that is not a bare identifier would produce SQL that
    # fails at query time, far from the file that caused it.
    if not name.isidentifier():
        raise ConfigError(f"metric name '{name}' is not a valid column identifier")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not metric_cls:
        raise ConfigError(
            f"duplicate metric name '{name}': {existing.__name__} vs {metric_cls.__name__}"
        )
    _REGISTRY[name] = metric_cls
    return metric_cls
