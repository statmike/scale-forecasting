"""Tests for the error taxonomy and logger factory."""

from __future__ import annotations

import logging

import pytest

from scale_forecasting.errors import (
    CLI_LOG_FORMAT,
    PACKAGE_LOGGER,
    ConfigError,
    EngineError,
    ModelError,
    RegistryError,
    ScaleForecastError,
    configure_cli_logging,
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


def test_get_logger_attaches_nothing_and_propagates() -> None:
    # A library does not decide where its records go: the named logger carries no handler of its
    # own and propagates, so the host's configuration (Airflow, a notebook, the CLI) routes it.
    logger = get_logger("scale_forecasting.propagation_check")
    assert logger.handlers == []
    assert logger.propagate is True


def test_package_logger_has_exactly_one_null_handler() -> None:
    # Importing `errors` installs the one NullHandler that silences the stdlib's last-resort
    # "no handlers could be found" path; repeated imports / get_logger calls must not stack more.
    get_logger("scale_forecasting.a")
    get_logger("scale_forecasting.b")
    handlers = logging.getLogger(PACKAGE_LOGGER).handlers
    assert [type(h) for h in handlers] == [logging.NullHandler]


def test_package_records_reach_a_host_handler(caplog: pytest.LogCaptureFixture) -> None:
    # The whole point of propagation: a handler the application attached sees the package's lines.
    with caplog.at_level(logging.INFO, logger=PACKAGE_LOGGER):
        get_logger("scale_forecasting.reaches_host").info("hello from the package")
    assert "hello from the package" in caplog.text


def test_configure_cli_logging_is_guarded_by_root_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A host that already configured logging (pytest's capture counts) keeps its handlers: the
    # CLI helper must not add a second copy of every line.
    root = logging.getLogger()
    before = list(root.handlers)
    assert before, "pytest installs a root handler; the guard needs one to be meaningful"
    monkeypatch.setenv("SF_LOG_LEVEL", "DEBUG")
    configure_cli_logging()
    assert root.handlers == before


def test_configure_cli_logging_installs_root_handler_and_honours_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        root.handlers.clear()
        monkeypatch.setenv("SF_LOG_LEVEL", "warning")
        configure_cli_logging()
        assert len(root.handlers) == 1
        assert root.level == logging.WARNING
        assert root.handlers[0].formatter is not None
        assert root.handlers[0].formatter._fmt == CLI_LOG_FORMAT
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
