"""Amazon Chronos-2 zero-shot foundation model with covariates & cross-learning.

Wraps ``Chronos2Pipeline`` (`amazon/chronos-2`), which extends the Chronos family with group
self-attention across variates and series so zero-shot inference natively consumes historical-only
covariates (``past_covariates``), known-future covariates (``future_covariates`` / ``exog`` /
calendar / Fourier), and static series attributes (``static_covariates``) in both ``local`` and
``global`` panel modes.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np

from ..errors import ModelError
from ._foundation_base import (
    DECILE_GRID,
    FoundationBaseModel,
    FoundationSeriesInput,
    encode_static_carriers,
)
from .base_model import register


@register
class Chronos2Model(FoundationBaseModel):
    """Amazon Chronos-2 zero-shot foundation model supporting covariates and cross-learning."""

    name = "chronos2"
    supports_exog = True
    supports_future_covariates = True
    supports_past_covariates = True
    supports_static_covariates = True
    package = "chronos-forecasting"
    package_url = "https://github.com/amazon-science/chronos-forecasting"
    optional_import = "chronos"
    default_repo_id: ClassVar[str] = "amazon/chronos-2"

    def _resolve_repo_id(self) -> str:
        explicit = self.params.get("repo_id")
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        return self.default_repo_id

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
        ctx_len_raw = self.params.get("context_length")
        ctx_len = int(ctx_len_raw) if ctx_len_raw is not None else None
        batch_size = max(1, int(self.params.get("batch_size", 256)))
        default_cross = bool(is_panel and len(items) > 1)
        cross_learning = bool(self.params.get("cross_learning", default_cross))
        self._last_cross_learning = cross_learning

        # Collect union of static keys across batch items so all dicts share homogeneous keys.
        all_stat_keys: tuple[str, ...] = ()
        if self.supports_static_covariates:
            stat_key_set: set[str] = set()
            for it in items:
                if it.static_covs:
                    stat_key_set.update(str(k) for k in it.static_covs)
            all_stat_keys = tuple(sorted(stat_key_set))

        input_dicts: list[dict[str, Any]] = []
        for item in items:
            y_arr = np.asarray(item.y, dtype=np.float32)
            X_hist = item.X_hist
            if ctx_len is not None and len(y_arr) > ctx_len:
                y_arr = y_arr[-ctx_len:]
                if X_hist is not None:
                    X_hist = X_hist.iloc[-ctx_len:]
            hist_len = len(y_arr)

            past_covs: dict[str, np.ndarray] = {}
            futr_covs: dict[str, np.ndarray] = {}

            for col in self._hist_cols:
                if X_hist is not None and col in X_hist.columns:
                    past_covs[col] = X_hist[col].to_numpy(dtype=np.float32)
                else:
                    past_covs[col] = np.zeros(hist_len, dtype=np.float32)

            for col in self._futr_cols:
                if X_hist is not None and col in X_hist.columns:
                    p_vals = X_hist[col].to_numpy(dtype=np.float32)
                else:
                    p_vals = np.zeros(hist_len, dtype=np.float32)
                past_covs[col] = p_vals

                if item.X_futr is not None and col in item.X_futr.columns and len(item.X_futr) > 0:
                    f_vals = item.X_futr[col].to_numpy(dtype=np.float32)[:steps]
                    if len(f_vals) < steps:
                        pad = np.full(steps - len(f_vals), float(f_vals[-1]), dtype=np.float32)
                        f_vals = np.concatenate([f_vals, pad])
                else:
                    fill_v = float(p_vals[-1]) if len(p_vals) > 0 else 0.0
                    f_vals = np.full(steps, fill_v, dtype=np.float32)
                futr_covs[col] = f_vals

            if all_stat_keys:
                raw_stat = {k: (item.static_covs or {}).get(k, 0.0) for k in all_stat_keys}
                stat_carriers = encode_static_carriers(raw_stat, hist_len=hist_len, futr_len=steps)
                for skey, (spast, sfutr) in stat_carriers.items():
                    past_covs[skey] = spast
                    futr_covs[skey] = sfutr

            entry: dict[str, Any] = {"target": y_arr}
            if past_covs:
                entry["past_covariates"] = past_covs
            if futr_covs:
                entry["future_covariates"] = futr_covs
            input_dicts.append(entry)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            quantiles_list, mean_list = pipeline.predict_quantiles(
                inputs=input_dicts,
                prediction_length=int(steps),
                quantile_levels=list(DECILE_GRID),
                batch_size=batch_size,
                context_length=ctx_len,
                cross_learning=cross_learning,
                limit_prediction_length=False,
            )

        out: list[tuple[np.ndarray, np.ndarray]] = []
        for q_t, m_t in zip(quantiles_list, mean_list, strict=True):
            m_np = m_t.detach().cpu().numpy().reshape(-1).astype(float)[:steps]
            q_np = q_t.detach().cpu().numpy().reshape(-1, len(DECILE_GRID)).astype(float)[:steps, :]
            out.append((m_np, q_np))
        return out

    def diagnostics(self) -> dict[str, Any]:
        diag = super().diagnostics()
        default_cross = bool(getattr(self, "_last_cross_learning", False))
        diag["cross_learning"] = bool(self.params.get("cross_learning", default_cross))
        return diag
