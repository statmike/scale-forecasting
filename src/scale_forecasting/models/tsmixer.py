"""TSMixer — Time-Series Mixer with exogenous support (Chen et al., 2023; ``deep_learning`` family).

An all-MLP mixer architecture (``TSMixerx`` in ``neuralforecast``) that alternates time-mixing and
feature-mixing residual blocks across lookback windows, static covariates, historical covariates,
and future covariates, with reversible instance normalization (RevIN). Supports both per-series
``training_mode: "local"`` (``n_series=1``) and cross-series ``training_mode: "global"``
(``n_series=N`` across the panel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._neuralforecast_base import NeuralForecastBaseModel
from .base_model import register

if TYPE_CHECKING:
    import optuna


class TSMixerModel(NeuralForecastBaseModel):
    """Time-Series Mixer (TSMixerx) forecaster via ``neuralforecast``."""

    name = "tsmixer"
    supports_exog = True
    supports_static_covariates = True
    _nf_model_name = "TSMixerx"

    def _arch_kwargs(self, *, n_series: int) -> dict[str, Any]:
        return {
            "n_series": int(n_series),
            "n_block": int(self.params.get("n_block", 2)),
            "ff_dim": int(self.params.get("ff_dim", 32)),
            "dropout": float(self.params.get("dropout", 0.1)),
            "revin": bool(self.params.get("revin", True)),
        }

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_block": trial.suggest_int("n_block", 1, 4),
            "ff_dim": trial.suggest_categorical("ff_dim", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }


register(TSMixerModel)
