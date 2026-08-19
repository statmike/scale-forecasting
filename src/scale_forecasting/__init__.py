"""scale-forecasting — massively-parallel time-series forecasting on Google Cloud.

The package is deliberately flat and readable: one capability per file. Start with
``DESIGN`` in the README, then read ``config.py`` (the run contract), ``worker.py``
(the unit of work), and one file under ``models/`` to see the whole shape.

Public surface — two ways in, one engine underneath:

* **The easy path.** :class:`Forecaster` — point it at a config and call
  :meth:`~scale_forecasting.sdk.Forecaster.dry_run` / :meth:`~scale_forecasting.sdk.Forecaster.run`
  / :meth:`~scale_forecasting.sdk.Forecaster.review`. Thin wrapper over :func:`run`; no logic of its
  own.
* **The direct path.** Drive Spark or Ray yourself and reuse the *same* model machinery:
  :func:`run_group` (pure, per-bucket), :func:`make_group_runner` / :func:`make_chunk_runner` (the
  writer-attached ``applyInPandas`` / Ray-task closures), :func:`chunk_cells`, and the unit of work
  :func:`run_cell`. Both paths run byte-identical cell code (G1). See ``docs/using_the_sdk.md``.

Import cost: ``import scale_forecasting`` is near-instant. Light names (config, settings, errors)
are eager; the heavy names above load the model modules and are therefore **lazy** — resolved on
first attribute access via :pep:`562`. Touching ``Forecaster``/``run``/``run_cell`` pays the model
import cost; touching ``RunConfig``/``Settings`` does not. ``test_sdk.py`` guards this contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Eager (light): the config contract, infra identity, and the error hierarchy. None of these pull
# the model modules, so they keep ``import scale_forecasting`` cheap.
from .config import Fanout, RunConfig, estimate_fanout, load_config
from .errors import (
    ConfigError,
    DataError,
    EngineError,
    ModelError,
    RegistryError,
    ScaleForecastError,
)
from .settings import Settings

__version__ = "0.1.0"

# Lazy (heavy): name -> (module, attribute). Resolved on first access by ``__getattr__`` (PEP 562)
# so the model modules (statsmodels et al., ~5s) load only when a caller actually reaches for the
# run/model surface — never merely to ``import scale_forecasting``.
_LAZY: dict[str, tuple[str, str]] = {
    # Easy path.
    "Forecaster": (".sdk", "Forecaster"),
    "DryRunResult": (".sdk", "DryRunResult"),
    "RunResult": (".sdk", "RunResult"),
    # Orchestration.
    "run": (".main", "run"),
    # Direct path — pure core + writer-attached runners + unit of work.
    "run_group": (".engines.spark_io", "run_group"),
    "make_group_runner": (".engines.spark_io", "make_group_runner"),
    "RunOutcome": (".engines.spark_io", "RunOutcome"),
    "aggregate_status": (".engines.spark_io", "aggregate_status"),
    "make_chunk_runner": (".engines.ray_io", "make_chunk_runner"),
    "chunk_cells": (".engines.ray_io", "chunk_cells"),
    "run_cell": (".worker", "run_cell"),
    "CellResult": (".worker", "CellResult"),
    # Model registry + runtime routing.
    "get_model": (".models", "get_model"),
    "list_models": (".models", "list_models"),
    "split_by_runtime": (".router", "split_by_runtime"),
}

__all__ = [
    "__version__",
    # Eager.
    "RunConfig",
    "load_config",
    "estimate_fanout",
    "Fanout",
    "Settings",
    "ScaleForecastError",
    "ConfigError",
    "DataError",
    "ModelError",
    "RegistryError",
    "EngineError",
    # Lazy.
    *_LAZY,
]

if TYPE_CHECKING:  # so IDEs / type-checkers see the lazy names as real imports (re-exported below)
    from .engines.ray_io import chunk_cells, make_chunk_runner  # noqa: F401
    from .engines.spark_io import (  # noqa: F401
        RunOutcome,
        aggregate_status,
        make_group_runner,
        run_group,
    )
    from .main import run  # noqa: F401
    from .models import get_model, list_models  # noqa: F401
    from .router import split_by_runtime  # noqa: F401
    from .sdk import DryRunResult, Forecaster, RunResult  # noqa: F401
    from .worker import CellResult, run_cell  # noqa: F401


def __getattr__(name: str) -> object:
    """PEP 562 lazy loader: resolve a heavy public name on first access, then cache it."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr = target
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
