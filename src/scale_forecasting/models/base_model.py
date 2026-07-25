"""The model interface every model file implements (CONTRACTS §1).

This is the linchpin of the one-model-one-file rule and the factory. Every Python model
is a subclass of :class:`BaseModel` living in its own file that ends with ``register(...)``;
the factory (``models/__init__.py``) builds ``{name: class}`` at import. Adding a model is
a new file plus one register call — no edits anywhere else.

Models never read global config: everything they need at fit/predict time arrives through
:class:`ModelContext`. Models that don't emit their own prediction intervals
(``supports_native_intervals = False``) call :meth:`BaseModel.residual_intervals` so every
model still returns the canonical frame with ordered bounds (CONTRACTS §2.1).

Public surface: ``BaseModel``, ``ModelContext``, ``register`` (plus the ``_REGISTRY`` the
factory reads, and the ``Runtime``/``Family`` type aliases).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from ..errors import ModelError

if TYPE_CHECKING:
    import optuna

Runtime = Literal["python", "bigquery"]
Family = Literal["statistical", "ml", "deep_learning", "native"]

# Canonical prediction-frame columns (CONTRACTS §2.1), in order.
PREDICTION_COLUMNS: tuple[str, ...] = ("ds", "yhat", "yhat_lower", "yhat_upper", "quantiles")

# Default quantile set for predict() and the residual helper (CONTRACTS §1, §2.1).
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ModelContext:
    """Per-run context handed to every model so it never reads global config (CONTRACTS §1)."""

    freq: str
    horizon: int
    seed: int = 0
    holidays: pd.DataFrame | None = None
    transform: str = "none"


# The factory registry: name → concrete model class. Populated by register() at import.
_REGISTRY: dict[str, type[BaseModel]] = {}


def register(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Register a model class under its ``name`` (CONTRACTS §1). Returns the class so it
    doubles as a decorator. Raises on a missing or duplicate name.
    """
    name = getattr(model_cls, "name", None)
    if not name:
        raise ModelError(f"{model_cls.__name__} must set a class-level 'name' before register()")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not model_cls:
        raise ModelError(
            f"duplicate model name '{name}': {existing.__name__} vs {model_cls.__name__}"
        )
    _REGISTRY[name] = model_cls
    return model_cls


class BaseModel(ABC):
    """Base class for every forecasting model (CONTRACTS §1)."""

    # --- class-level registration metadata (read by the factory) ---
    name: ClassVar[str]
    runtime: ClassVar[Runtime]
    family: ClassVar[Family]
    supports_exog: ClassVar[bool] = False
    supports_native_intervals: ClassVar[bool] = False

    def __init__(self, params: dict[str, Any], ctx: ModelContext) -> None:
        self.params = dict(params)
        self.ctx = ctx
        # Residuals stashed by fit() for models that lean on the residual-interval helper.
        self._residuals: np.ndarray | None = None

    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        """Fit on one series. ``y`` is indexed by ds (datetime64); ``X`` is aligned exog or None."""

    @abstractmethod
    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Return the canonical prediction frame (CONTRACTS §2.1) in original units."""

    def get_params(self) -> dict[str, Any]:
        """Resolved params actually used (post-HPO). Logged to ``forecast_metadata.best_params``."""
        return dict(self.params)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        """HPO search space (optional; used only when ``hpo.enabled``). Default: no search."""
        return {}

    # --- shared helpers ---------------------------------------------------------

    def residual_intervals(
        self, yhat: np.ndarray, quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    ) -> dict[float, np.ndarray]:
        """Empirical residual-quantile prediction intervals (CONTRACTS §1).

        For models without native intervals: fit() records residuals via
        :meth:`_set_residuals`, and this adds their empirical quantiles to the point
        forecast so bounds are naturally ordered (lower ≤ yhat ≤ upper) for any monotone
        quantile set. Falls back to a point-mass band (bounds == yhat) if no residuals
        were recorded.
        """
        yhat = np.asarray(yhat, dtype=float)
        if self._residuals is None or self._residuals.size == 0:
            return {q: yhat.copy() for q in quantiles}
        return {q: yhat + float(np.quantile(self._residuals, q)) for q in quantiles}

    def _set_residuals(self, residuals: np.ndarray | pd.Series) -> None:
        """Record in-sample residuals (actual − fitted) for :meth:`residual_intervals`."""
        arr = np.asarray(residuals, dtype=float)
        self._residuals = arr[~np.isnan(arr)]

    def _assemble_frame(
        self,
        ds: pd.DatetimeIndex | pd.Series,
        quantile_map: dict[float, np.ndarray],
    ) -> pd.DataFrame:
        """Build the canonical prediction frame from a quantile map (CONTRACTS §2.1).

        ``yhat`` is the 0.5 quantile (median); bounds are the min/max quantiles so they
        stay ordered. ``quantiles`` is the full map serialized to a JSON string per row.
        """
        qs = sorted(quantile_map)
        if not qs:
            raise ModelError("quantile_map is empty")
        median = quantile_map.get(0.5, quantile_map[qs[len(qs) // 2]])
        lower = quantile_map[qs[0]]
        upper = quantile_map[qs[-1]]
        n = len(median)
        quantiles_json = [
            json.dumps({str(q): float(quantile_map[q][i]) for q in qs}) for i in range(n)
        ]
        # Contract requires datetime64[ns] (§2.1); pandas 2.x may infer coarser units.
        ds_ns = pd.DatetimeIndex(ds).as_unit("ns")
        return pd.DataFrame(
            {
                "ds": ds_ns,
                "yhat": np.asarray(median, dtype=float),
                "yhat_lower": np.asarray(lower, dtype=float),
                "yhat_upper": np.asarray(upper, dtype=float),
                "quantiles": pd.array(quantiles_json, dtype="string"),
            },
            columns=list(PREDICTION_COLUMNS),
        )
