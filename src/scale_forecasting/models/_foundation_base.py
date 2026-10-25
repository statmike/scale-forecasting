"""Shared base class for open-weight pretrained foundation time-series models.

Provides zero-shot local execution (``fit`` / ``predict`` / ``recondition`` /
``advance_origin``) and cross-series batched panel execution (``fit_panel`` /
``predict_panel``) for ``chronos_bolt``, ``chronos2``, and ``timesfm_torch``, along
with a single-active-model per-process weight cache so sequential foundation models
never hold multiple checkpoints in RAM or VRAM simultaneously.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import math
import os
import shutil
import sys
import tempfile
import threading
import zlib
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from ..errors import ConfigError, ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, ModelContext

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

_log = logging.getLogger(__name__)

DECILE_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_LOCAL_UID = "__local__"
_DEFAULT_MAX_CACHED_PIPELINES = 4

# Bounded LRU per-process foundation pipeline cache and re-entrant thread lock. Keeps active
# foundation checkpoints resident across cells/folds while serializing `from_pretrained` (which
# uses process-wide PyTorch `meta`-device hooks) and module inference across worker threads.
_PIPELINE_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_FOUNDATION_THREAD_LOCK = threading.RLock()


def _has_c_compiler() -> bool:
    """Return True if a usable C compiler is configured or present on ``PATH``."""
    cc_env = os.environ.get("CC", "").strip()
    if cc_env and (os.path.isfile(cc_env) or shutil.which(cc_env) is not None):
        return True
    return any(shutil.which(cmd) is not None for cmd in ("gcc", "clang", "cc"))


def _ensure_torch_runtime_ready() -> None:
    """Disable PyTorch 2.11+ Triton native JIT op overrides when no C compiler is installed.

    Minimal runtime containers (``debian:12-slim``) intentionally omit ``gcc``. PyTorch 2.11+
    registers a Triton JIT override on ``aten::bmm`` for CUDA outer-product shapes
    (``(B, M, 1) @ (B, 1, N)``), which ``ChronosBolt`` and ``Chronos2`` hit during attention/RoPE.
    Without ``gcc``, Triton's ``CudaUtils`` fails trying to compile ``cuda_utils.c``. Deregistering
    the Triton DSL op overrides routes ``aten::bmm`` directly to precompiled cuBLAS kernels.
    """
    if _has_c_compiler():
        return
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    try:
        from torch._native import triton_utils

        triton_utils.deregister_op_overrides()
    except Exception as exc:  # noqa: BLE001 - optional PyTorch internal module across versions
        _log.debug("Triton JIT op override deregistration skipped: %s", exc)


def _is_writable_dir(path: str) -> bool:
    """Return True if ``path`` can be created and written to by the current process."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, f".sf_write_probe_{os.getpid()}")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def _ensure_writable_hf_cache() -> str:
    """Ensure Hugging Face Hub and PyTorch cache directories are writable on worker nodes.

    On Dataproc YARN executors and locked-down container runtimes, ``$HOME`` (e.g. ``/home`` for
    user ``yarn``) may not be writable, causing ``huggingface_hub`` / ``transformers`` to fail
    with ``PermissionError`` when creating ``~/.cache/huggingface``. When the default or configured
    ``HF_HOME`` is not writable, this redirects all cache environment variables and any already-
    imported ``huggingface_hub.constants`` / ``transformers.utils.hub`` module attributes to a
    writable directory under ``tempfile.gettempdir()``.
    """
    candidate = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    if not _is_writable_dir(os.path.join(candidate, "hub")):
        candidate = os.path.join(tempfile.gettempdir(), "scale_forecasting_hf_cache")
        os.makedirs(os.path.join(candidate, "hub"), exist_ok=True)

    hub_dir = os.path.join(candidate, "hub")
    assets_dir = os.path.join(candidate, "assets")
    xet_dir = os.path.join(candidate, "xet")
    modules_dir = os.path.join(candidate, "modules")

    os.environ["HF_HOME"] = candidate
    os.environ["HF_HUB_CACHE"] = hub_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_dir
    os.environ.setdefault("TORCH_HOME", os.path.join(candidate, "torch"))
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(candidate, "triton"))

    hf_constants = sys.modules.get("huggingface_hub.constants")
    if hf_constants is not None:
        hf_constants.HF_HOME = candidate  # type: ignore[attr-defined]
        hf_constants.HF_HUB_CACHE = hub_dir  # type: ignore[attr-defined]
        hf_constants.HUGGINGFACE_HUB_CACHE = hub_dir  # type: ignore[attr-defined]
        hf_constants.HF_ASSETS_CACHE = assets_dir  # type: ignore[attr-defined]
        hf_constants.HUGGINGFACE_ASSETS_CACHE = assets_dir  # type: ignore[attr-defined]
        if hasattr(hf_constants, "HF_XET_CACHE"):
            hf_constants.HF_XET_CACHE = xet_dir  # type: ignore[attr-defined]

    tf_hub = sys.modules.get("transformers.utils.hub")
    if tf_hub is not None and hasattr(tf_hub, "HF_MODULES_CACHE"):
        tf_hub.HF_MODULES_CACHE = modules_dir  # type: ignore[attr-defined]

    return candidate


@contextlib.contextmanager
def _checkpoint_load_lock(cache_dir: str) -> Iterator[None]:
    """Serialize checkpoint downloads across worker processes sharing the same host filesystem."""
    os.makedirs(cache_dir, exist_ok=True)
    lock_path = os.path.join(cache_dir, ".scale_forecasting_load.lock")
    try:
        import fcntl

        lock_file = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115
    except Exception:  # noqa: BLE001 - fallback when fcntl or lock file is unavailable
        yield
        return

    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            lock_file.close()


# Apply environment guards at import time before upstream libraries initialize constants.
_ensure_writable_hf_cache()
if not _has_c_compiler():
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - best-effort CUDA cache cleanup
        _log.debug("Foundation model accelerator cache release skipped: %s", exc)


def clear_foundation_cache() -> None:
    """Evict all cached foundation model pipelines and release host/device memory."""
    with _FOUNDATION_THREAD_LOCK:
        if _PIPELINE_CACHE:
            _PIPELINE_CACHE.clear()
            _release_accelerator_memory()


def get_or_load_foundation_pipeline(
    cache_key: tuple[Any, ...],
    loader: Callable[[], Any],
    *,
    max_entries: int = _DEFAULT_MAX_CACHED_PIPELINES,
) -> Any:
    """Return the cached foundation pipeline for ``cache_key``, or load with LRU eviction."""
    cap = max(1, int(max_entries))
    with _FOUNDATION_THREAD_LOCK:
        if cache_key in _PIPELINE_CACHE:
            _PIPELINE_CACHE.move_to_end(cache_key)
            return _PIPELINE_CACHE[cache_key]
        evicted = False
        while len(_PIPELINE_CACHE) >= cap:
            _PIPELINE_CACHE.popitem(last=False)
            evicted = True
        if evicted:
            _release_accelerator_memory()

        hf_home = _ensure_writable_hf_cache()
        _ensure_torch_runtime_ready()
        with _checkpoint_load_lock(hf_home):
            try:
                obj = loader()
            except MemoryError:
                clear_foundation_cache()
                obj = loader()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and _PIPELINE_CACHE:
                    clear_foundation_cache()
                    obj = loader()
                else:
                    raise
        _PIPELINE_CACHE[cache_key] = obj
        return obj


@dataclass(frozen=True)
class FoundationSeriesInput:
    """Normalized conditioning inputs for one series in a foundation inference batch."""

    uid: str
    y: np.ndarray
    X_hist: pd.DataFrame | None
    X_futr: pd.DataFrame | None
    static_covs: Mapping[str, Any] | None


def encode_static_carriers(
    static_covs: Mapping[str, Any] | None,
    *,
    hist_len: int,
    futr_len: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Encode static covariates into instance-norm-preserving phase carriers ``(past, future)``.

    Pretrained foundation architectures (Chronos-2, TimesFM 3.0) apply per-variate instance
    normalization (RevIN) across the time axis, which collapses any constant or amplitude-scaled
    series ``s * f(t)`` to zero or identical unit curves. Encoding each static attribute ``s`` as a
    phase-modulated carrier ``cos(2*pi*t / L + pi*s_norm)`` preserves distinct per-attribute
    trajectories across patches after instance normalization.
    """
    if not static_covs:
        return {}
    total_len = hist_len + futr_len
    t = np.arange(total_len, dtype=np.float32)
    period = max(8.0, float(hist_len))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in sorted(static_covs.keys()):
        val = static_covs[col]
        if isinstance(val, bool | np.bool_):
            s_norm = 0.75 if val else 0.25
        elif isinstance(val, int | float | np.number):
            fv = float(val)
            s_norm = (math.atan(fv) / math.pi + 0.5) if math.isfinite(fv) else 0.5
        else:
            s_norm = float(zlib.crc32(str(val).encode("utf-8")) & 0xFFFFFFFF) / float(2**32)
        carrier = np.cos(2.0 * np.pi * t / period + math.pi * s_norm).astype(np.float32)
        key = f"stat__{col}"
        out[key] = (carrier[:hist_len], carrier[hist_len:total_len])
    return out


def map_deciles_to_quantiles(
    deciles: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Map a ``(horizon, 9)`` decile matrix (`0.1..0.9`) to arbitrary ``quantiles``.

    Preserves exact linear interpolation across the 9 native decile heads (`0.1..0.9`)
    and smoothly extrapolates extreme tail quantiles (`q < 0.1` or `q > 0.9`) from the
    `0.1`–`0.9` inter-decile Gaussian spread so tail bands remain strictly monotone.
    """
    from scipy.stats import norm

    arr = np.asarray(deciles, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ModelError(f"expected decile matrix of shape (horizon, 9), got {arr.shape}")
    mono = np.maximum.accumulate(arr, axis=1)
    grid = np.asarray(DECILE_GRID, dtype=float)
    q10 = mono[:, 0]
    q90 = mono[:, 8]
    z90 = float(norm.ppf(0.9))
    sigma = np.maximum((q90 - q10) / (2.0 * z90), 0.0)

    out: dict[float, np.ndarray] = {}
    for q in quantiles:
        qf = float(q)
        if qf < 0.1:
            out[q] = q10 + (float(norm.ppf(qf)) + z90) * sigma
        elif qf > 0.9:
            out[q] = q90 + (float(norm.ppf(qf)) - z90) * sigma
        else:
            idx = int(np.searchsorted(grid, qf))
            if idx < len(grid) and math.isclose(float(grid[idx]), qf, abs_tol=1e-9):
                out[q] = mono[:, idx].copy()
            else:
                hi_idx = min(max(idx, 1), len(grid) - 1)
                lo_idx = hi_idx - 1
                w = (qf - float(grid[lo_idx])) / (float(grid[hi_idx]) - float(grid[lo_idx]))
                out[q] = (1.0 - w) * mono[:, lo_idx] + w * mono[:, hi_idx]
    return out


class FoundationBaseModel(BaseModel):
    """Base class for zero-shot pretrained foundation time-series models."""

    runtime = "python"
    family = "deep_learning"
    supports_native_intervals = True
    supports_global = True
    supports_hybrid = False
    gpu_capable = True
    supports_recondition = True
    supports_extrapolate = True
    optional_extra = "models-foundation"

    default_repo_id: ClassVar[str]

    def __init__(self, params: dict[str, Any], ctx: ModelContext) -> None:
        super().__init__(params, ctx)
        self._y: np.ndarray | None = None
        self._X: pd.DataFrame | None = None
        self._futr_cols: list[str] = []
        self._hist_cols: list[str] = []
        self._panel_series: dict[str, tuple[np.ndarray, pd.DataFrame | None]] = {}
        self._panel_last_dates: dict[str, pd.Timestamp] = {}
        self._panel_static_map: dict[str, dict[str, Any]] = {}
        self._fitted_device: str | None = None
        self._resolved_repo_id: str | None = None

    @abstractmethod
    def _resolve_repo_id(self) -> str:
        """Return the Hugging Face repo ID or local path for this model configuration."""

    @abstractmethod
    def _pipeline_cache_key(self, repo_id: str, device_str: str) -> tuple[Any, ...]:
        """Return the process-cache key identifying the loaded foundation pipeline."""

    @abstractmethod
    def _load_pipeline(self, repo_id: str, device_str: str) -> Any:
        """Instantiate and return the pretrained foundation pipeline on ``device_str``."""

    @abstractmethod
    def _predict_batch_raw(
        self,
        pipeline: Any,
        items: list[FoundationSeriesInput],
        steps: int,
        *,
        is_panel: bool,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run batched zero-shot inference returning ``[(point_1d, deciles_2d), ...]``."""

    def _split_exog_cols(self, X: pd.DataFrame | None) -> tuple[list[str], list[str]]:
        """Partition ``X`` columns into ``(futr_cols, hist_cols)``."""
        if not self.supports_exog or X is None or X.empty:
            return [], []
        past_set = set(self.ctx.past_covariates)
        hist_cols = [str(c) for c in X.columns if c in past_set]
        futr_cols = [str(c) for c in X.columns if c not in past_set]
        return futr_cols, hist_cols

    def _resolve_torch_device(self) -> str:
        """Resolve the target PyTorch device (`'cuda'` or `'cpu'`) from ``self.ctx.device``."""
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ModelError(
                f"{self.name} requires PyTorch; install 'scale-forecasting[models-foundation]'"
            ) from e

        if self.ctx.device == "gpu":
            if not torch.cuda.is_available():
                raise ModelError(
                    f"{self.name} requested device='gpu' but torch.cuda.is_available() is False"
                )
            return "cuda"
        if self.ctx.device == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_pipeline(self) -> Any:
        """Ensure the pretrained pipeline is loaded in the process cache and record device."""
        self.require_available()
        device_str = self._resolve_torch_device()
        repo_id = self._resolve_repo_id()
        key = self._pipeline_cache_key(repo_id, device_str)
        pipeline = get_or_load_foundation_pipeline(
            key, lambda: self._load_pipeline(repo_id, device_str)
        )
        self._fitted_device = "cuda" if device_str.startswith("cuda") else "cpu"
        self._resolved_repo_id = repo_id
        return pipeline

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        if len(y) < 2:
            raise ModelError(f"{self.name} requires at least 2 observations")
        self._last_date = pd.Timestamp(y.index[-1])
        self._futr_cols, self._hist_cols = self._split_exog_cols(X)
        self._y = y.astype(float).to_numpy().copy()
        self._X = X.copy() if (self.supports_exog and X is not None and not X.empty) else None
        self._origin_offset = 0
        self._gap_exog = None
        self._ensure_pipeline()

    def recondition(self, y_new: pd.Series, X_new: pd.DataFrame | None = None) -> None:
        """Append new observations to the zero-shot conditioning context."""
        if self._y is None:
            raise ModelError(f"{self.name} must be fit before recondition()")
        if len(y_new) == 0:
            return
        self._y = np.concatenate([self._y, y_new.astype(float).to_numpy()])
        if self._X is not None:
            new_cols: dict[str, np.ndarray] = {}
            for col in self._X.columns:
                if X_new is not None and col in X_new.columns:
                    new_cols[col] = X_new[col].astype(float).to_numpy()
                else:
                    last_val = float(self._X[col].iloc[-1])
                    new_cols[col] = np.full(len(y_new), last_val, dtype=float)
            new_df = pd.DataFrame(new_cols, index=y_new.index)
            self._X = pd.concat([self._X, new_df])
        self._last_date = pd.Timestamp(y_new.index[-1])
        self._origin_offset = 0
        self._gap_exog = None

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        if self._y is None:
            raise ModelError(f"{self.name} must be fit before predict()")
        pipeline = self._ensure_pipeline()
        steps = self._forecast_steps(horizon)
        full_exog = self._forecast_exog(X) if self.supports_exog else None

        item = FoundationSeriesInput(
            uid=_LOCAL_UID,
            y=self._y,
            X_hist=self._X,
            X_futr=full_exog,
            static_covs=self.ctx.static_covariates if self.supports_static_covariates else None,
        )
        with _FOUNDATION_THREAD_LOCK:
            point_all, deciles_all = self._predict_batch_raw(
                pipeline, [item], steps, is_panel=False
            )[0]
        point = np.asarray(point_all[-horizon:], dtype=float)
        deciles = np.asarray(deciles_all[-horizon:, :], dtype=float)

        raw_qmap = map_deciles_to_quantiles(deciles, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(vals, t, lam) for q, vals in raw_qmap.items()}
        ds = self._forecast_index(horizon)
        return self._assemble_frame(ds, qmap, raw=invert_transform(point, t, lam))

    def fit_panel(
        self,
        series_map: Mapping[str, tuple[pd.Series, pd.DataFrame | None]],
        static_map: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        """Condition a shared zero-shot foundation pipeline across a panel of series."""
        if not series_map:
            raise ModelError(f"{self.name}.fit_panel requires a non-empty series_map")
        uids = sorted(series_map.keys())
        min_len = min(len(series_map[uid][0]) for uid in uids)
        if min_len < 2:
            raise ModelError(f"{self.name} requires at least 2 observations per series")

        first_X = series_map[uids[0]][1]
        self._futr_cols, self._hist_cols = self._split_exog_cols(first_X)
        self._panel_series = {}
        self._panel_last_dates = {}
        for uid in uids:
            y, X = series_map[uid]
            self._panel_last_dates[uid] = pd.Timestamp(y.index[-1])
            clean_X = X.copy() if (self.supports_exog and X is not None and not X.empty) else None
            self._panel_series[uid] = (y.astype(float).to_numpy().copy(), clean_X)
        self._panel_static_map = (
            {uid: dict(static_map.get(uid, {})) for uid in uids}
            if (self.supports_static_covariates and static_map)
            else {}
        )
        self._ensure_pipeline()

    def predict_panel(
        self,
        horizon: int,
        future_exog_map: Mapping[str, pd.DataFrame | None] | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        transform_lambdas: Mapping[str, float | None] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Predict ``horizon`` steps across all series in the conditioned panel."""
        if not self._panel_series or not self._panel_last_dates:
            raise ModelError(f"{self.name}.predict_panel called before fit_panel()")
        pipeline = self._ensure_pipeline()
        uids = sorted(self._panel_last_dates.keys())

        items: list[FoundationSeriesInput] = []
        for uid in uids:
            y_arr, X_hist = self._panel_series[uid]
            X_futr = (
                future_exog_map.get(uid)
                if (self.supports_exog and future_exog_map is not None)
                else None
            )
            static_covs = self._panel_static_map.get(uid) if self._panel_static_map else None
            items.append(
                FoundationSeriesInput(
                    uid=uid,
                    y=y_arr,
                    X_hist=X_hist,
                    X_futr=X_futr,
                    static_covs=static_covs,
                )
            )

        with _FOUNDATION_THREAD_LOCK:
            batch_preds = self._predict_batch_raw(pipeline, items, horizon, is_panel=True)
        t = self.ctx.transform
        out: dict[str, pd.DataFrame] = {}
        for uid, (point_all, deciles_all) in zip(uids, batch_preds, strict=True):
            point = np.asarray(point_all[:horizon], dtype=float)
            deciles = np.asarray(deciles_all[:horizon, :], dtype=float)
            raw_qmap = map_deciles_to_quantiles(deciles, quantiles)
            lam = (
                transform_lambdas.get(uid, self.ctx.transform_lambda)
                if transform_lambdas is not None
                else self.ctx.transform_lambda
            )
            qmap = {q: invert_transform(vals, t, lam) for q, vals in raw_qmap.items()}
            ds = self._future_index(self._panel_last_dates[uid], horizon)
            out[uid] = self._assemble_frame(ds, qmap, raw=invert_transform(point, t, lam))
        return out

    def device_used(self) -> str | None:
        return self._fitted_device

    def diagnostics(self) -> dict[str, Any]:
        diag: dict[str, Any] = {}
        if self._resolved_repo_id is not None:
            diag["repo_id"] = str(self._resolved_repo_id)
        if self._fitted_device is not None:
            diag["device"] = str(self._fitted_device)
        return diag

    def serialize(self) -> bytes | None:
        """Opt out of per-cell pickle uploads for pretrained foundation checkpoints."""
        return None

    @classmethod
    def gpu_useful(cls, params: Mapping[str, Any]) -> bool:
        """Pretrained transformer foundation models benefit from GPU tensor cores when enabled."""
        return True

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        super().validate_params(params, max_horizon=max_horizon)
        context_length = params.get("context_length")
        if context_length is not None and int(context_length) < 2:
            raise ConfigError(
                f"model_params.{cls.name}.context_length must be >= 2 (got {context_length!r})."
            )
        batch_size = params.get("batch_size")
        if batch_size is not None and int(batch_size) < 1:
            raise ConfigError(
                f"model_params.{cls.name}.batch_size must be >= 1 (got {batch_size!r})."
            )
        raw_q = params.get("quantiles")
        if raw_q is not None:
            if not isinstance(raw_q, list | tuple) or not raw_q:
                raise ConfigError(
                    f"model_params.{cls.name}.quantiles must be a non-empty list of floats "
                    "in (0, 1)."
                )
            for q in raw_q:
                qf = float(q)
                if not (0.0 < qf < 1.0):
                    raise ConfigError(
                        f"model_params.{cls.name}.quantiles entries must be in (0, 1); got {q!r}."
                    )
