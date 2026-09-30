"""TiDE — Time-series Dense Encoder (Das et al., 2023; ``deep_learning`` family).

An all-MLP encoder-decoder architecture with residual connections that maps a lookback window of
target observations, past covariates, future covariates, and static attributes directly to a
multi-horizon quantile distribution without recurrence or self-attention. Supports both per-series
``training_mode: "local"`` (Spark and Ray) and cross-series ``training_mode: "global"`` (Ray).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._neuralforecast_base import NeuralForecastBaseModel
from .base_model import register

if TYPE_CHECKING:
    import optuna


class TiDEModel(NeuralForecastBaseModel):
    """Time-series Dense Encoder (TiDE) forecaster via ``neuralforecast``."""

    name = "tide"
    supports_exog = True
    _nf_model_name = "TiDE"

    def _arch_kwargs(self, *, n_series: int) -> dict[str, Any]:
        return {
            "hidden_size": int(self.params.get("hidden_size", 64)),
            "decoder_output_dim": int(self.params.get("decoder_output_dim", 8)),
            "temporal_decoder_dim": int(self.params.get("temporal_decoder_dim", 16)),
            "num_encoder_layers": int(self.params.get("num_encoder_layers", 1)),
            "num_decoder_layers": int(self.params.get("num_decoder_layers", 1)),
            "dropout": float(self.params.get("dropout", 0.1)),
        }

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }


register(TiDEModel)
