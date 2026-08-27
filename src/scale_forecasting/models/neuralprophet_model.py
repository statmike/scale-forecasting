"""NeuralProphet — the one model that benefits from a GPU.

One model, one file. Runtime python, deep_learning family. NeuralProphet is
an optional dependency, imported lazily in ``fit`` so the model registers without it (and
without dragging torch into the base install). It supports quantile regression, but quantiles
are fixed at construction; our contract passes the quantile set at *predict* time, so — like
``prophet`` and ``theta`` — we fit a fixed symmetric band, back the implied sigma out of it,
and place any requested quantile from that Gaussian, honoring arbitrary quantile sets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    import optuna

# Fixed band fit into the network; sigma is backed out of it for arbitrary quantiles.
_BAND = (0.1, 0.9)


class NeuralProphetModel(BaseModel):
    """NeuralProphet forecaster (neural additive model)."""

    name = "neuralprophet"
    runtime = "python"
    family = "deep_learning"
    supports_exog = False
    supports_native_intervals = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from neuralprophet import NeuralProphet, set_log_level, set_random_seed
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ModelError("neuralprophet not installed; install the 'models' extra") from e
        if len(y) < 2:
            raise ModelError("neuralprophet requires at least 2 observations")
        set_log_level("ERROR")
        # Seed torch so a fit is reproducible under a fixed seed, like every other stochastic model
        # (xgboost/lightgbm wire ctx.seed too) — the model contract requires determinism.
        set_random_seed(self.ctx.seed)

        self._last_date = y.index[-1]
        self._train = pd.DataFrame(
            {"ds": pd.DatetimeIndex(y.index), "y": y.astype(float).to_numpy()}
        )
        model = NeuralProphet(
            quantiles=list(_BAND),
            epochs=int(self.params.get("epochs", 50)),
            learning_rate=float(self.params.get("learning_rate", 0.01)),
            trainer_config={"accelerator": "auto"},
        )
        model.fit(self._train, freq=self.ctx.freq, progress=None)
        self._model = model

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm  # lazy: keep scipy off the module top (lean launch point)

        future = self._model.make_future_dataframe(self._train, periods=horizon)
        fc = self._model.predict(future).tail(horizon)
        mean = fc["yhat1"].to_numpy(dtype=float)
        lo = fc[self._band_col(_BAND[0])].to_numpy(dtype=float)
        hi = fc[self._band_col(_BAND[1])].to_numpy(dtype=float)
        # Back out sigma from the symmetric band, then place any requested quantile.
        z = norm.ppf(_BAND[1])
        sigma = (hi - lo) / (2.0 * z)

        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)

    @staticmethod
    def _band_col(q: float) -> str:
        """NeuralProphet names quantile columns like ``yhat1 10.0%``."""
        return f"yhat1 {q * 100:.1f}%"

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 20, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        }


register(NeuralProphetModel)
