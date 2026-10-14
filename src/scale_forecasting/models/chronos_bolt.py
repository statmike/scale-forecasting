"""Amazon Chronos-Bolt zero-shot patch-based T5 foundation model (``chronos-forecasting``).

Provides fast univariate zero-shot quantile forecasting via ``ChronosBoltPipeline`` across four
public Hugging Face checkpoint sizes (``base`` default, ``small``, ``mini``, ``tiny``), supporting
both per-series ``local`` execution and batched ``global`` panel inference on CPU or GPU.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np

from ..errors import ConfigError, ModelError
from ._foundation_base import (
    DECILE_GRID,
    FoundationBaseModel,
    FoundationSeriesInput,
)
from .base_model import register

_CHRONOS_BOLT_SIZES: frozenset[str] = frozenset({"tiny", "mini", "small", "base"})


@register
class ChronosBoltModel(FoundationBaseModel):
    """Amazon Chronos-Bolt zero-shot patch-based foundation model."""

    name = "chronos_bolt"
    supports_exog = False
    supports_future_covariates = False
    supports_past_covariates = False
    supports_static_covariates = False
    package = "chronos-forecasting"
    package_url = "https://github.com/amazon-science/chronos-forecasting"
    optional_import = "chronos"
    default_repo_id: ClassVar[str] = "amazon/chronos-bolt-base"

    def _resolve_size(self) -> str:
        size = str(self.params.get("size", "base")).strip().lower()
        if size not in _CHRONOS_BOLT_SIZES:
            raise ConfigError(
                f"model_params.chronos_bolt.size={size!r} is invalid; "
                f"must be one of {sorted(_CHRONOS_BOLT_SIZES)}."
            )
        return size

    def _resolve_repo_id(self) -> str:
        explicit = self.params.get("repo_id")
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        size = self._resolve_size()
        return f"amazon/chronos-bolt-{size}"

    def _pipeline_cache_key(self, repo_id: str, device_str: str) -> tuple[Any, ...]:
        return (self.name, repo_id, device_str, "float32")

    def _load_pipeline(self, repo_id: str, device_str: str) -> Any:
        try:
            from chronos import BaseChronosPipeline
        except ImportError as e:  # pragma: no cover
            raise ModelError(
                "chronos-forecasting not installed; install the 'models-foundation' extra"
            ) from e

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            return BaseChronosPipeline.from_pretrained(
                repo_id,
                device_map=device_str,
                dtype="float32",
            )

    def _predict_batch_raw(
        self,
        pipeline: Any,
        items: list[FoundationSeriesInput],
        steps: int,
        *,
        is_panel: bool,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        import torch

        ctx_len_raw = self.params.get("context_length")
        ctx_len = int(ctx_len_raw) if ctx_len_raw is not None else None
        batch_size = max(1, int(self.params.get("batch_size", 256)))

        tensors: list[torch.Tensor] = []
        for item in items:
            arr = np.asarray(item.y, dtype=np.float32)
            if ctx_len is not None and len(arr) > ctx_len:
                arr = arr[-ctx_len:]
            tensors.append(torch.from_numpy(arr))

        out: list[tuple[np.ndarray, np.ndarray]] = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            for start in range(0, len(tensors), batch_size):
                chunk = tensors[start : start + batch_size]
                quantiles_t, mean_t = pipeline.predict_quantiles(
                    inputs=chunk,
                    prediction_length=int(steps),
                    quantile_levels=list(DECILE_GRID),
                    limit_prediction_length=False,
                )
                q_np = quantiles_t.detach().cpu().numpy().astype(float)
                m_np = mean_t.detach().cpu().numpy().astype(float)
                for idx in range(len(chunk)):
                    out.append((m_np[idx, :steps], q_np[idx, :steps, :]))
        return out

    def diagnostics(self) -> dict[str, Any]:
        diag = super().diagnostics()
        diag["size"] = self._resolve_size()
        return diag

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        super().validate_params(params, max_horizon=max_horizon)
        if "size" in params and params["size"] is not None:
            size = str(params["size"]).strip().lower()
            if size not in _CHRONOS_BOLT_SIZES:
                raise ConfigError(
                    f"model_params.chronos_bolt.size={params['size']!r} is invalid; "
                    f"must be one of {sorted(_CHRONOS_BOLT_SIZES)}."
                )
