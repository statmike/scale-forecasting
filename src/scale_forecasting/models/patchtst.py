"""PatchTST — Patch Time Series Transformer (Nie et al., 2023; ``deep_learning`` family).

A channel-independent Transformer encoder that segments each series' lookback window into subseries
patches before self-attention, reducing attention complexity quadratically in patch length while
capturing local semantic structure. Channel-independent by design (``supports_exog = False``);
supports both per-series ``training_mode: "local"`` (Spark and Ray) and cross-series
``training_mode: "global"`` (Ray).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._neuralforecast_base import NeuralForecastBaseModel
from .base_model import register

if TYPE_CHECKING:
    import optuna


class PatchTSTModel(NeuralForecastBaseModel):
    """Patch Time Series Transformer (PatchTST) forecaster via ``neuralforecast``."""

    name = "patchtst"
    supports_exog = False
    _nf_model_name = "PatchTST"

    def _arch_kwargs(self, *, n_series: int) -> dict[str, Any]:
        patch_len = min(int(self.params.get("patch_len", 8)), max(1, self._input_size))
        stride = min(int(self.params.get("stride", 4)), patch_len)
        return {
            "hidden_size": int(self.params.get("hidden_size", 32)),
            "n_heads": int(self.params.get("n_heads", 2)),
            "encoder_layers": int(self.params.get("encoder_layers", 2)),
            "patch_len": patch_len,
            "stride": stride,
            "dropout": float(self.params.get("dropout", 0.1)),
            "revin": bool(self.params.get("revin", True)),
        }

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }


register(PatchTSTModel)
