"""Tests for the error taxonomy and logger factory."""

from __future__ import annotations

import logging

import pytest

from scale_forecasting.errors import (
    ConfigError,
    EngineError,
    ModelError,
    RegistryError,
    ScaleForecastError,
    get_logger,
)

SUBCLASSES = [ConfigError, ModelError, RegistryError, EngineError]


@pytest.mark.parametrize("exc", SUBCLASSES)
def test_subclasses_derive_from_base(exc: type[ScaleForecastError]) -> None:
    # A caller can catch everything from this package with the one base class.
    assert issubclass(exc, ScaleForecastError)
    with pytest.raises(ScaleForecastError):
        raise exc("boom")


def test_base_is_an_exception() -> None:
    assert issubclass(ScaleForecastError, Exception)


def test_error_carries_message() -> None:
    err = ConfigError("missing horizon")
    assert str(err) == "missing horizon"


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("scale_forecasting.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "scale_forecasting.test"


def test_get_logger_is_idempotent_no_duplicate_handlers() -> None:
    # Calling twice must not stack handlers (would duplicate log lines).
    name = "scale_forecasting.idempotent_check"
    first = get_logger(name)
    n_handlers = len(first.handlers)
    second = get_logger(name)
    assert first is second
    assert len(second.handlers) == n_handlers == 1


def test_get_logger_does_not_propagate() -> None:
    # We attach our own handler, so propagation off avoids double emission.
    logger = get_logger("scale_forecasting.no_propagate")
    assert logger.propagate is False
