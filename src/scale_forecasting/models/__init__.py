"""Model factory.

Importing this package registers every model file by name (each model module ends with
``register(...)``); ``get_model(name)`` returns the class and ``list_models()`` lists the
registered names. Adding a model is one new file + one register call listed below — no
other edits.

The model modules are imported here for their registration side effect. The import block
grows by one line per model.
"""

from __future__ import annotations

from ..errors import ModelError

# --- model registration imports (side-effect: each calls register()) -----------
# One line per model file.
from . import (  # noqa: E402,F401
    autoets,
    bigquery_native,
    croston,
    holtwinters,
    lightgbm_model,
    naive_drift,
    naive_mean,
    naive_moving_average,
    naive_seasonal,
    neuralprophet_model,
    prophet_model,
    regression_lags,
    sarimax,
    stl_bagging,
    theta,
    ucm,
    xgboost_model,
)
from .base_model import _REGISTRY, BaseModel


def get_model(name: str) -> type[BaseModel]:
    """Return the registered model class for ``name``.

    Raises ``ModelError`` with the available names when ``name`` is unknown.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise ModelError(f"unknown model '{name}'; registered models: {known}") from None


def list_models() -> list[str]:
    """All registered model names, sorted."""
    return sorted(_REGISTRY)
