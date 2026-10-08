"""Tests for the error taxonomy and logger factory."""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import pytest

from scale_forecasting import errors
from scale_forecasting.errors import (
    CLI_LOG_FORMAT,
    EXTRA_MODULES,
    PACKAGE_LOGGER,
    ConfigError,
    EngineError,
    MissingExtraError,
    ModelError,
    RegistryError,
    ScaleForecastError,
    configure_cli_logging,
    get_logger,
    require_extra,
)

SUBCLASSES = [ConfigError, ModelError, RegistryError, EngineError, MissingExtraError]


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


# --- require_extra: the one sentence a core-only install gets instead of a ModuleNotFoundError ---

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _declared_extras() -> set[str]:
    with _PYPROJECT.open("rb") as fh:
        return set(tomllib.load(fh)["project"]["optional-dependencies"])


def test_missing_extra_error_is_also_an_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both idioms must catch it: the package's own base class and the stdlib's ImportError, since
    # "optional dependency absent" is what callers already write `except ImportError` for.
    assert issubclass(MissingExtraError, ImportError)
    assert issubclass(MissingExtraError, ScaleForecastError)
    monkeypatch.setattr(errors, "is_importable", lambda module: False)
    with pytest.raises(ImportError):
        require_extra("gcp", purpose="A test")


def test_require_extra_is_a_no_op_when_the_extra_is_installed() -> None:
    # The test environment installs every extra (`uv sync --all-extras`), so each probe passes and
    # the call returns None without touching sys.modules beyond importlib's parent packages.
    for extra in EXTRA_MODULES:
        assert require_extra(extra, purpose="A test") is None


def test_require_extra_message_names_the_purpose_the_extra_and_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(errors, "is_importable", lambda module: False)
    with pytest.raises(MissingExtraError) as info:
        require_extra("notebook", purpose="Plotting")
    message = str(info.value)
    assert message.startswith("Plotting needs the 'notebook' extra")
    assert "missing: matplotlib" in message
    assert 'pip install "scale-forecasting[notebook]"' in message
    assert "uv sync --extra notebook" in message


def test_require_extra_lists_only_the_modules_that_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A half-installed extra (one client present, another not) names the absent module, not the
    # whole probe list — the reader fixes what is actually wrong.
    monkeypatch.setattr(errors, "is_importable", lambda module: module != "pyspark")
    with pytest.raises(MissingExtraError) as info:
        require_extra("spark", purpose="The Spark Connect path")
    assert "(missing: pyspark)" in str(info.value)


def test_importable_treats_a_missing_parent_package_as_not_installed() -> None:
    # `find_spec("google.cloud.bigquery")` raises ModuleNotFoundError when `google` itself is
    # absent (a core-only install); that is "not installed", not an error to propagate.
    assert errors.is_importable("no_such_parent_package.child") is False
    assert errors.is_importable("json") is True


def test_every_probed_extra_is_declared_in_pyproject() -> None:
    # The install line in the error must name an extra pip can actually resolve.
    undeclared = set(EXTRA_MODULES) - _declared_extras()
    assert not undeclared, f"EXTRA_MODULES names extras pyproject does not declare: {undeclared}"


def test_composed_extras_probe_their_gcp_base_too() -> None:
    # `[spark]`, `[ray]`, `[submit]`, `[models-automl]` each include `[gcp]` in pyproject, so the
    # install line "pip install scale-forecasting[spark]" is a sufficient fix for a missing
    # BigQuery client as well — which is only true if the probe checks the base's modules.
    for extra in ("spark", "ray", "submit", "models-automl"):
        assert set(EXTRA_MODULES["gcp"]) <= set(EXTRA_MODULES[extra]), extra
