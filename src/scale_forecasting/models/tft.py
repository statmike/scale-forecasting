"""TFT — Temporal Fusion Transformer (Lim et al., 2021; ``deep_learning`` family).

An attention-and-gating sequence-to-sequence architecture with Variable Selection Networks for
static attributes, observed historical covariates, and known future covariates, gated residual
networks (GRNs), and interpretable multi-head attention across the forecast horizon. Supports both
per-series ``training_mode: "local"`` (Spark and Ray) and cross-series ``training_mode: "global"``
(Ray).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._neuralforecast_base import NeuralForecastBaseModel
from .base_model import register

if TYPE_CHECKING:
    import optuna


class TFTModel(NeuralForecastBaseModel):
    """Temporal Fusion Transformer (TFT) forecaster via ``neuralforecast``."""

    name = "tft"
    supports_exog = True
    supports_static_covariates = True
    _nf_model_name = "TFT"

    def _arch_kwargs(self, *, n_series: int) -> dict[str, Any]:
        return {
            "hidden_size": int(self.params.get("hidden_size", 32)),
            "n_head": int(self.params.get("n_head", 2)),
            "dropout": float(self.params.get("dropout", 0.1)),
        }

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "n_head": trial.suggest_categorical("n_head", [1, 2, 4]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }


register(TFTModel)
