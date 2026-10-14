"""Google Research TimesFM 3.0 & 2.5 open-weight PyTorch foundation model (``timesfm``).

Executes Google's open-weight ``TimesFM`` foundation models directly in PyTorch on CPU or GPU
(local playground, GCE, Vertex AI, GKE, Ray, or Spark), distinct from the managed BigQuery
``AI.FORECAST`` SQL function (``timesfm`` / ``bq_timesfm``):

- **TimesFM 3.0 (`version="3.0"`, default):** Loads ``google/timesfm-3.0-pytorch`` via
  ``timesfm3.TimesFM3Forecaster`` with native in-transformer **Variate Attention** over
  historical-only covariates (``past_only_covariates``), known-future covariates
  (``past_future_covariates``), and static series attributes.
- **TimesFM 2.5 (`version="2.5"` or ``timesfm_torch@v2.5``):** Loads
  ``google/timesfm-2.5-200m-pytorch`` via ``timesfm.TimesFM_2p5_200M_torch`` with continuous
  quantile heads, flip invariance, and in-context Ridge covariate regression.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, ClassVar

import numpy as np

from ..errors import ConfigError, ModelError
from ._foundation_base import (
    FoundationBaseModel,
    FoundationSeriesInput,
    encode_static_carriers,
)
from .base_model import register

_VALID_TIMESFM_TORCH_VERSIONS: dict[str, str] = {
    "3.0": "3.0",
    "v3.0": "3.0",
    "3": "3.0",
    "v3": "3.0",
    "2.5": "2.5",
    "v2.5": "2.5",
}

_DEFAULT_REPO_BY_VERSION: dict[str, str] = {
    "3.0": "google/timesfm-3.0-pytorch",
    "2.5": "google/timesfm-2.5-200m-pytorch",
}


def _fit_predict_ridge_xreg(
    y: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    ridge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a standardized L2-regularized linear model ``y ~ X_train`` and predict on ``X_test``."""
    n_obs, n_feats = X_train.shape
    if n_obs < 2 or n_feats == 0:
        return np.zeros(n_obs, dtype=np.float32), np.zeros(len(X_test), dtype=np.float32)

    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if not np.isfinite(y_std) or y_std < 1e-6:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    x_std = np.where(x_std > 1e-6, x_std, 1.0)
    X_tr_norm = (X_train - x_mean) / x_std
    X_te_norm = (X_test - x_mean) / x_std

    # Augment with intercept column (unpenalized).
    A_tr = np.column_stack([X_tr_norm, np.ones(n_obs, dtype=np.float64)])
    A_te = np.column_stack([X_te_norm, np.ones(len(X_test), dtype=np.float64)])
    reg = np.eye(n_feats + 1, dtype=np.float64) * max(1e-4, float(ridge))
    reg[-1, -1] = 0.0
    try:
        beta = np.linalg.solve(A_tr.T @ A_tr + reg, A_tr.T @ y_norm)
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(A_tr, y_norm, rcond=None)

    pred_tr = (A_tr @ beta) * y_std + y_mean
    pred_te = (A_te @ beta) * y_std + y_mean
    return pred_tr.astype(np.float32), pred_te.astype(np.float32)


@register
class TimesFMTorchModel(FoundationBaseModel):
    """Google Research TimesFM (3.0 default, 2.5 selectable) PyTorch foundation model."""

    name = "timesfm_torch"
    supports_exog = True
    supports_future_covariates = True
    supports_past_covariates = True
    supports_static_covariates = True
    package = "timesfm"
    package_url = "https://github.com/google-research/timesfm"
    optional_import = "timesfm"
    default_repo_id: ClassVar[str] = "google/timesfm-3.0-pytorch"

    def _resolve_version(self) -> str:
        raw = str(self.params.get("version", "3.0")).strip().lower()
        if raw not in _VALID_TIMESFM_TORCH_VERSIONS:
            raise ConfigError(
                f"model_params.timesfm_torch.version={self.params.get('version')!r} is invalid; "
                "supported versions for 'timesfm_torch' are '3.0' (default) and '2.5'."
            )
        return _VALID_TIMESFM_TORCH_VERSIONS[raw]

    def _resolve_repo_id(self) -> str:
        explicit = self.params.get("repo_id")
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        ver = self._resolve_version()
        return _DEFAULT_REPO_BY_VERSION[ver]

    def _pipeline_cache_key(self, repo_id: str, device_str: str) -> tuple[Any, ...]:
        ver = self._resolve_version()
        batch_size = max(1, int(self.params.get("batch_size", 32)))
        return (self.name, ver, repo_id, device_str, batch_size)

    def _load_pipeline(self, repo_id: str, device_str: str) -> Any:
        ver = self._resolve_version()
        batch_size = max(1, int(self.params.get("batch_size", 32)))

        if ver == "3.0":
            try:
                from timesfm3 import TimesFM3Forecaster
            except ImportError as e:  # pragma: no cover
                raise ModelError(
                    "timesfm3 not installed; install the 'models-foundation' extra"
                ) from e
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                return TimesFM3Forecaster.from_pretrained(
                    repo_id,
                    device=device_str,
                    per_core_batch_size=batch_size,
                )

        try:
            import torch
            from timesfm import TimesFM_2p5_200M_torch
        except ImportError as e:  # pragma: no cover
            raise ModelError("timesfm not installed; install the 'models-foundation' extra") from e

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            tfm = TimesFM_2p5_200M_torch.from_pretrained(repo_id, torch_compile=False)
            target_dev = torch.device("cuda:0" if device_str.startswith("cuda") else "cpu")
            tfm.model.to(target_dev)
            tfm.model.device = target_dev
            tfm.model.device_count = 1
            return tfm

    def _predict_batch_raw(
        self,
        pipeline: Any,
        items: list[FoundationSeriesInput],
        steps: int,
        *,
        is_panel: bool,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        ver = self._resolve_version()
        if ver == "3.0":
            return self._predict_batch_v3(pipeline, items, steps)
        return self._predict_batch_v2p5(pipeline, items, steps)

    def _predict_batch_v3(
        self,
        pipeline: Any,
        items: list[FoundationSeriesInput],
        steps: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        ctx_len_raw = self.params.get("context_length")
        ctx_len = int(ctx_len_raw) if ctx_len_raw is not None else None

        all_stat_keys: tuple[str, ...] = ()
        if self.supports_static_covariates:
            stat_set: set[str] = set()
            for it in items:
                if it.static_covs:
                    stat_set.update(str(k) for k in it.static_covs)
            all_stat_keys = tuple(sorted(stat_set))

        contexts: list[np.ndarray] = []
        po_list: list[np.ndarray | None] = []
        pf_list: list[np.ndarray | None] = []
        has_po = False
        has_pf = False

        for item in items:
            y_arr = np.asarray(item.y, dtype=np.float32)
            X_hist = item.X_hist
            if ctx_len is not None and len(y_arr) > ctx_len:
                y_arr = y_arr[-ctx_len:]
                if X_hist is not None:
                    X_hist = X_hist.iloc[-ctx_len:]
            hist_len = len(y_arr)
            contexts.append(y_arr)

            if self._hist_cols:
                po_rows: list[np.ndarray] = []
                for col in self._hist_cols:
                    if X_hist is not None and col in X_hist.columns:
                        po_rows.append(X_hist[col].to_numpy(dtype=np.float32))
                    else:
                        po_rows.append(np.zeros(hist_len, dtype=np.float32))
                po_list.append(np.stack(po_rows, axis=0))
                has_po = True
            else:
                po_list.append(None)

            pf_rows: list[np.ndarray] = []
            for col in self._futr_cols:
                if X_hist is not None and col in X_hist.columns:
                    p_vals = X_hist[col].to_numpy(dtype=np.float32)
                else:
                    p_vals = np.zeros(hist_len, dtype=np.float32)
                if item.X_futr is not None and col in item.X_futr.columns and len(item.X_futr) > 0:
                    f_vals = item.X_futr[col].to_numpy(dtype=np.float32)[:steps]
                    if len(f_vals) < steps:
                        pad = np.full(steps - len(f_vals), float(f_vals[-1]), dtype=np.float32)
                        f_vals = np.concatenate([f_vals, pad])
                else:
                    fill_v = float(p_vals[-1]) if len(p_vals) > 0 else 0.0
                    f_vals = np.full(steps, fill_v, dtype=np.float32)
                pf_rows.append(np.concatenate([p_vals, f_vals]))

            if all_stat_keys:
                raw_stat = {k: (item.static_covs or {}).get(k, 0.0) for k in all_stat_keys}
                stat_carriers = encode_static_carriers(raw_stat, hist_len=hist_len, futr_len=steps)
                for spast, sfutr in stat_carriers.values():
                    pf_rows.append(np.concatenate([spast, sfutr]))

            if pf_rows:
                pf_list.append(np.stack(pf_rows, axis=0))
                has_pf = True
            else:
                pf_list.append(None)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            outputs = list(
                pipeline.predict_batch(
                    contexts=contexts,
                    horizon=int(steps),
                    past_only_covariates=po_list if has_po else None,
                    past_future_covariates=pf_list if has_pf else None,
                    return_quantiles=True,
                    sort_quantiles=True,
                    padding_mode="edge",
                )
            )

        out: list[tuple[np.ndarray, np.ndarray]] = []
        for fo in outputs:
            point = np.asarray(fo.forecast, dtype=float)[:steps]
            deciles = np.asarray(fo.quantiles, dtype=float)[:steps, :]
            out.append((point, deciles))
        return out

    def _predict_batch_v2p5(
        self,
        pipeline: Any,
        items: list[FoundationSeriesInput],
        steps: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        from timesfm import ForecastConfig

        ctx_len_raw = self.params.get("context_length")
        ctx_len = int(ctx_len_raw) if ctx_len_raw is not None else 2048
        batch_size = max(1, int(self.params.get("batch_size", 32)))
        ridge = float(self.params.get("ridge", 1.0))

        all_stat_keys: tuple[str, ...] = ()
        if self.supports_static_covariates:
            stat_set: set[str] = set()
            for it in items:
                if it.static_covs:
                    stat_set.update(str(k) for k in it.static_covs)
            all_stat_keys = tuple(sorted(stat_set))

        contexts: list[np.ndarray] = []
        xreg_futures: list[np.ndarray] = []
        max_len = 32

        for item in items:
            y_arr = np.asarray(item.y, dtype=np.float32)
            X_hist = item.X_hist
            if len(y_arr) > ctx_len:
                y_arr = y_arr[-ctx_len:]
                if X_hist is not None:
                    X_hist = X_hist.iloc[-ctx_len:]
            hist_len = len(y_arr)
            max_len = max(max_len, hist_len)

            tr_cols: list[np.ndarray] = []
            te_cols: list[np.ndarray] = []

            for col in self._futr_cols:
                if X_hist is not None and col in X_hist.columns:
                    p_vals = X_hist[col].to_numpy(dtype=np.float32)
                else:
                    p_vals = np.zeros(hist_len, dtype=np.float32)
                if item.X_futr is not None and col in item.X_futr.columns and len(item.X_futr) > 0:
                    f_vals = item.X_futr[col].to_numpy(dtype=np.float32)[:steps]
                    if len(f_vals) < steps:
                        pad = np.full(steps - len(f_vals), float(f_vals[-1]), dtype=np.float32)
                        f_vals = np.concatenate([f_vals, pad])
                else:
                    fill_v = float(p_vals[-1]) if len(p_vals) > 0 else 0.0
                    f_vals = np.full(steps, fill_v, dtype=np.float32)
                tr_cols.append(p_vals)
                te_cols.append(f_vals)

            for col in self._hist_cols:
                if X_hist is not None and col in X_hist.columns:
                    p_vals = X_hist[col].to_numpy(dtype=np.float32)
                else:
                    p_vals = np.zeros(hist_len, dtype=np.float32)
                # Lag past-only covariates by `steps` so the future window uses observed history.
                lag = min(steps, max(1, hist_len - 1))
                lagged_tr = np.concatenate(
                    [np.full(lag, float(p_vals[0]), dtype=np.float32), p_vals[:-lag]]
                )
                lagged_te = p_vals[-steps:] if hist_len >= steps else np.resize(p_vals, steps)
                tr_cols.append(lagged_tr)
                te_cols.append(lagged_te.astype(np.float32))

            if all_stat_keys:
                raw_stat = {k: (item.static_covs or {}).get(k, 0.0) for k in all_stat_keys}
                stat_carriers = encode_static_carriers(raw_stat, hist_len=hist_len, futr_len=steps)
                for spast, sfutr in stat_carriers.values():
                    tr_cols.append(spast)
                    te_cols.append(sfutr)

            if tr_cols:
                X_tr_mat = np.column_stack(tr_cols)
                X_te_mat = np.column_stack(te_cols)
                xreg_hist, xreg_fut = _fit_predict_ridge_xreg(
                    y_arr, X_tr_mat, X_te_mat, ridge=ridge
                )
                contexts.append(y_arr - xreg_hist)
                xreg_futures.append(xreg_fut)
            else:
                contexts.append(y_arr)
                xreg_futures.append(np.zeros(steps, dtype=np.float32))

        max_h = max(128, int(math.ceil(steps / 128.0) * 128))
        max_c = max(32, min(int(math.ceil(max_len / 32.0) * 32), 16384 - max_h))
        fc = ForecastConfig(
            max_context=max_c,
            max_horizon=max_h,
            normalize_inputs=True,
            per_core_batch_size=batch_size,
            use_continuous_quantile_head=(max_h <= 1024),
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
            return_backcast=False,
        )
        pipeline.compile(fc)
        point_arr, quant_arr = pipeline.forecast(horizon=int(steps), inputs=contexts)

        out: list[tuple[np.ndarray, np.ndarray]] = []
        for idx in range(len(items)):
            xf = xreg_futures[idx].astype(float)
            p_i = np.asarray(point_arr[idx, :steps], dtype=float) + xf
            # Index 0 in TimesFM 2.5 quantile_forecast is mean; indices 1..9 are deciles 0.1..0.9.
            q_i = np.asarray(quant_arr[idx, :steps, 1:10], dtype=float) + xf[:, None]
            out.append((p_i, q_i))
        return out

    def diagnostics(self) -> dict[str, Any]:
        diag = super().diagnostics()
        diag["version"] = self._resolve_version()
        return diag

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        super().validate_params(params, max_horizon=max_horizon)
        if "version" in params and params["version"] is not None:
            raw = str(params["version"]).strip().lower()
            if raw not in _VALID_TIMESFM_TORCH_VERSIONS:
                raise ConfigError(
                    f"model_params.timesfm_torch.version={params['version']!r} is invalid; "
                    "supported versions for 'timesfm_torch' are '3.0' (default) and '2.5'."
                )
