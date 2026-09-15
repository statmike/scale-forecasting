"""Shared metric contract test — the metric-side mirror of `test_models_contract.py`.

Parametrized over every registered metric, so a metric a deployment adds is covered the moment
it registers. The contract is deliberately small, because a metric is a small thing: a valid
column name, an honest direction, honest ``needs_*`` declarations, and a float — never an
exception — for every window it is handed, however degenerate.

Two collision guards live here too, and neither is about any one metric:

* a metric name must not collide with another column of ``forecast_metadata``, because the
  metric block of that table is generated from `METRIC_NAMES` and a duplicate column would
  fail at deploy time, in BigQuery, far from the file that caused it; and
* a metric name must not collide with a model's hyperparameter name — the §"a metric is
  computed by the framework" rule, checked mechanically rather than trusted.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from scale_forecasting.errors import ConfigError
from scale_forecasting.metrics import (
    METRIC_DIRECTION,
    METRIC_NAMES,
    MetricContext,
    compute_metrics,
    get_metric,
    list_metrics,
    register,
)
from scale_forecasting.metrics.base_metric import BaseMetric
from scale_forecasting.models import get_model, list_models
from scale_forecasting.models.base_model import BaseModel
from scale_forecasting.registry.ddl import additive_columns

DIRECTIONS = {"lower", "higher", "zero"}


@pytest.fixture(params=list_metrics())
def metric_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _ctx(**over: Any) -> MetricContext:
    """A well-formed scored window with every optional input present."""
    base: dict[str, Any] = {
        "y_true": [10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.5],
        "yhat": [9.5, 12.4, 11.2, 12.0, 13.0, 13.2, 14.1],
        "y_train": list(np.linspace(5.0, 10.0, 60) + np.sin(np.arange(60))),
        "lower": [8.0, 10.0, 9.5, 10.0, 11.0, 11.5, 12.0],
        "upper": [11.5, 14.0, 13.0, 14.5, 15.0, 15.5, 16.0],
        "seasonal_period": 7,
    }
    base.update(over)
    return MetricContext.from_arrays(**base)


# Windows a metric must survive. Each is a real shape the fleet produces, not a fuzz case:
# a zero actual (intermittent demand), a flat series (a dead SKU), a perfect forecast (the
# divide-by-zero), one observation (a series at the fold-geometry floor), and each optional
# input absent (no backtest history, no prediction band, an unknown frequency).
DEGENERATE: dict[str, dict[str, Any]] = {
    "zeros_in_actuals": {"y_true": [0.0, 0.0, 3.0, 0.0, 5.0, 0.0, 1.0]},
    "all_zero_actuals": {"y_true": [0.0] * 7},
    "flat_actuals": {"y_true": [4.0] * 7},
    "perfect_forecast": {"y_true": [10.0] * 7, "yhat": [10.0] * 7},
    "negatives": {"y_true": [-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 4.0]},
    "huge": {"y_true": [1e18] * 7, "yhat": [1e-18] * 7},
    "single_point": {
        "y_true": [7.0],
        "yhat": [6.0],
        "lower": [5.0],
        "upper": [8.0],
    },
    "no_train_history": {"y_train": None},
    "one_train_point": {"y_train": [3.0]},
    "flat_train_history": {"y_train": [3.0] * 60},
    "short_train_history": {"y_train": [1.0, 2.0, 3.0]},  # shorter than the seasonal period
    "no_intervals": {"lower": None, "upper": None},
    "lower_only": {"upper": None},
    "inverted_intervals": {
        "lower": [99.0] * 7,
        "upper": [-99.0] * 7,
    },
    "no_seasonal_period": {"seasonal_period": None},
    "zero_seasonal_period": {"seasonal_period": 0},
}


# --- the contract --------------------------------------------------------------


def test_the_name_is_the_registry_key_and_a_usable_column(metric_name: str) -> None:
    """It becomes a BigQuery column, so the identifier rules are the column's rules."""
    cls = get_metric(metric_name)
    assert cls.name == metric_name
    assert metric_name.isidentifier()
    assert metric_name == metric_name.lower()
    assert not metric_name.startswith("_")


def test_the_direction_is_declared_and_one_of_the_three(metric_name: str) -> None:
    """A wrong answer here silently inverts ensemble weighting, HPO and pruning."""
    assert get_metric(metric_name).direction in DIRECTIONS
    assert METRIC_DIRECTION[metric_name] == get_metric(metric_name).direction


@pytest.mark.parametrize("case", sorted(DEGENERATE))
def test_compute_returns_a_float_and_never_raises(metric_name: str, case: str) -> None:
    value = get_metric(metric_name)().compute(_ctx(**DEGENERATE[case]))
    assert isinstance(value, float)
    # NaN is a legitimate answer ("undefined here"); ±inf is not — it is an unguarded divide
    # that would survive into a mean and poison a whole model's leaderboard row.
    assert not math.isinf(value)


def test_the_needs_flags_are_honest_about_what_is_missing(metric_name: str) -> None:
    """A metric that declares a need must be NaN without it — the flags gate a plan-time warning.

    The declarations are documentation to `config._check_decision_metric_is_computable`, which
    tells an operator their ranking column will be empty before the run spends anything. A flag
    that does not match the arithmetic makes that warning either a false alarm or silence.
    """
    cls = get_metric(metric_name)
    if cls.needs_intervals:
        assert math.isnan(cls().compute(_ctx(lower=None, upper=None)))
    if cls.needs_train_history:
        assert math.isnan(cls().compute(_ctx(y_train=None)))
    if cls.needs_seasonal_period:
        assert math.isnan(cls().compute(_ctx(seasonal_period=None)))


def test_a_metric_that_declares_no_need_computes_without_the_optional_inputs(
    metric_name: str,
) -> None:
    """The other half: an *undeclared* dependency is the failure this catches."""
    cls = get_metric(metric_name)
    if cls.needs_intervals or cls.needs_train_history or cls.needs_seasonal_period:
        pytest.skip(f"{metric_name} declares a need")
    value = cls().compute(_ctx(y_train=None, lower=None, upper=None, seasonal_period=None))
    assert not math.isnan(value), f"{metric_name} silently depends on an input it does not declare"


# --- the panel as a whole ------------------------------------------------------


def test_compute_metrics_returns_the_whole_panel_as_floats() -> None:
    panel = compute_metrics(
        [10.0, 12.0, 11.0],
        [9.0, 13.0, 11.5],
        y_train=[1.0, 2.0, 3.0, 4.0, 5.0],
        lower=[8.0, 10.0, 9.0],
        upper=[12.0, 14.0, 13.0],
        seasonal_period=2,
    )
    assert tuple(panel) == METRIC_NAMES  # keys, in panel order
    assert all(isinstance(v, float) for v in panel.values())


def test_a_dependent_metric_is_computed_once_per_window() -> None:
    """`mase` asks the context for `mae` rather than recomputing it — the memo is the proof."""
    ctx = _ctx()
    assert "mae" not in ctx._memo
    ctx.value("mase")
    assert ctx._memo["mae"] == pytest.approx(float(np.mean(ctx.abs_err)))


def test_a_metric_that_depends_on_itself_gets_nan_instead_of_a_recursion_error() -> None:
    """The self-dependency guard. A registry author gets a NaN, not a blown interpreter."""

    class SelfReferential(BaseMetric):
        name = "self_referential"
        direction = "lower"

        def compute(self, ctx: MetricContext) -> float:
            return ctx.value("self_referential") + 1.0

    register(SelfReferential)
    try:
        assert math.isnan(_ctx().value("self_referential"))
    finally:
        from scale_forecasting.metrics.base_metric import _REGISTRY

        _REGISTRY.pop("self_referential", None)


def test_an_unregistered_name_is_nan_rather_than_an_exception() -> None:
    assert math.isnan(_ctx().value("no_such_metric"))


def test_a_name_that_is_not_a_column_identifier_is_refused_at_registration() -> None:
    class BadName(BaseMetric):
        name = "not a column"
        direction = "lower"

        def compute(self, ctx: MetricContext) -> float:
            return 0.0

    with pytest.raises(ConfigError, match="not a valid column identifier"):
        register(BadName)


# --- the two collision guards --------------------------------------------------


def test_no_metric_name_collides_with_another_forecast_metadata_column() -> None:
    """The metric block is generated into this table — a duplicate column fails at deploy.

    `additive_columns` parses the rendered CREATE, so the metric columns are in it; strip them
    and what is left must share no name with the panel.
    """
    columns = [name for name, _ in additive_columns("forecast_metadata")]
    non_metric = [c for c in columns if c not in METRIC_NAMES]
    assert set(non_metric).isdisjoint(METRIC_NAMES)
    # And the panel appears exactly once each, which is the duplicate the parse would hide.
    for name in METRIC_NAMES:
        assert columns.count(name) == 1


class _RecordingTrial:
    """A stand-in Optuna trial: `search_space` only needs the suggest_* calls to return a value."""

    def suggest_int(self, name: str, low: int, high: int, **kw: Any) -> int:
        return low

    def suggest_float(self, name: str, low: float, high: float, **kw: Any) -> float:
        return low

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        return choices[0]


def _search_space_keys(cls: type[BaseModel]) -> set[str]:
    return set(cls.search_space(_RecordingTrial()))  # type: ignore[arg-type]


def test_no_model_hyperparameter_is_named_after_a_metric() -> None:
    """A metric is computed by the framework; a model may never supply one.

    There is no channel today by which a model could hand back a value for a metric name — the
    worker scores every cell through `compute_metrics` and nothing else — and this keeps it that
    way. `best_params` and the metric panel are read off one `forecast_metadata` row, so a
    hyperparameter called `mase` would make "which mase is this" a real question the first time
    anything flattens that JSON beside the columns.
    """
    seen: set[str] = set()
    for model_name in list_models():
        keys = _search_space_keys(get_model(model_name))
        collisions = keys & set(METRIC_NAMES)
        assert not collisions, f"model '{model_name}' has hyperparameter(s) named {collisions}"
        seen |= keys
    # Non-vacuity: if `search_space` ever stops returning its keys this way, the loop above would
    # pass by finding nothing at all, which is the one outcome that means the guard is off.
    assert len(seen) > 10, "no hyperparameters were read — the collision guard is not checking"
