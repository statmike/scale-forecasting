"""Model factory.

Importing this package registers every model file by name (each model module ends with
``register(...)``); ``get_model(name)`` returns the class and ``list_models()`` lists the
registered names. Adding a model is one new file + one register call listed below — no
other edits.

The model modules are imported here for their registration side effect. The import block
grows by one line per model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import ModelError, get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

# --- model registration imports (side-effect: each calls register()) -----------
# One line per model file.
from . import (  # noqa: F401
    auto_arima,
    auto_ces,
    auto_theta,
    autoets,
    bigquery_native,
    catboost_model,
    chronos2,
    chronos_bolt,
    croston,
    fft,
    holtwinters,
    kalman,
    lightgbm_model,
    naive_drift,
    naive_mean,
    naive_moving_average,
    naive_seasonal,
    neuralprophet_model,
    patchtst,
    prophet_model,
    random_forest,
    regression_lags,
    sarimax,
    stl_bagging,
    tbats_model,
    tft,
    theta,
    tide,
    timesfm_torch,
    tsmixer,
    ucm,
    vertex_l2l,
    vertex_seq2seq,
    vertex_tft,
    vertex_tide,
    xgboost_model,
)
from .base_model import REGISTRY as _REGISTRY
from .base_model import BaseModel

_log = get_logger(__name__)

# Clarity aliases mapping to their canonical registered base model names.
_MODEL_ALIASES: dict[str, str] = {
    "bq_timesfm": "timesfm",
    "ridge_lags": "regression_lags",
}


def base_model_name(name: str) -> str:
    """Return the canonical registered base model name from a ``<base>@<tag>`` variant string."""
    base = name.split("@", 1)[0].strip()
    return _MODEL_ALIASES.get(base, base)


def get_model(name: str) -> type[BaseModel]:
    """Return the registered model class for ``name`` (or ``<base>@<tag>`` variant).

    Raises ``ModelError`` with the available names when ``name`` is unknown.
    """
    base = base_model_name(name)
    try:
        return _REGISTRY[base]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise ModelError(f"unknown model '{name}'; registered models: {known}") from None


def list_models(
    *,
    available_only: bool = False,
    supports_monotonic_constraints: bool = False,
) -> list[str]:
    """All registered model names, sorted.

    Pass ``available_only=True`` to restrict the returned list to models whose
    upstream Python package is installed in the current environment.
    Pass ``supports_monotonic_constraints=True`` to restrict to models with
    native training-time monotonic sign-constraint support (``@monotone``).
    """
    out: list[str] = []
    for name, cls in _REGISTRY.items():
        if available_only and not cls.is_available():
            continue
        if supports_monotonic_constraints and not cls.supports_monotonic_constraints:
            continue
        out.append(name)
    return sorted(out)


def filter_available_models(models: Sequence[str]) -> list[str]:
    """Filter ``models`` to those whose upstream package is installed in this environment.

    Logs a warning for each omitted model so restricted environments can run a broad
    model list without failing on uninstalled optional packages.
    """
    kept: list[str] = []
    omitted: list[str] = []
    for name in models:
        cls = get_model(name)
        if cls.is_available():
            kept.append(name)
        else:
            omitted.append(f"{name} (requires {cls.package})")
    if omitted:
        _log.warning(
            "Omitting %d model(s) whose optional packages are not installed: %s",
            len(omitted),
            ", ".join(omitted),
        )
    return kept
