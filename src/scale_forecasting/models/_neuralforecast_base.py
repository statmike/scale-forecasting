"""Shared base class for Nixtla ``neuralforecast`` deep-learning models.

Provides both per-series local execution (``fit`` / ``predict`` / ``recondition`` /
``advance_origin``) and cross-series panel execution (``fit_panel`` / ``predict_panel``) for
``TiDE``, ``TFT``, ``TSMixer``, and ``PatchTST``.

Why ``neuralforecast`` is imported lazily inside ``fit`` / ``fit_panel``:
Every model registers at package import without pulling PyTorch or PyTorch Lightning into the
driver or lean submit environment. Environments that omit the ``models-dl`` extra still see the
registered model metadata and can filter available models cleanly via ``is_available()``.

Covariate routing:
Models with ``supports_exog = True`` split the dynamic design matrix ``X`` using
``ModelContext.past_covariates``: columns listed in ``past_covariates`` become
``hist_exog_list`` (observed-only historical inputs), while all remaining columns in ``X``
(known-in-advance ``future_covariates``, ``exog``, holidays, Fourier terms, and level-shift
indicators) become ``futr_exog_list``. Per-series static attributes from
``ModelContext.static_covariates`` (or ``static_map`` in panel mode) are numeric-encoded and
passed via ``static_df`` / ``stat_exog_list``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
import zlib
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from ..errors import ConfigError, ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    import optuna

# Fixed symmetric quantile band fit via MQLoss; sigma is backed out for arbitrary predict quantiles.
_BAND: tuple[float, float, float] = (0.1, 0.5, 0.9)
_LOCAL_UID = "__local__"


def ensure_mpl_dir() -> None:
    """Ensure Matplotlib has a writable cache directory inside sandboxed / container workers."""
    cur = os.environ.get("MPLCONFIGDIR")
    if cur and os.access(cur, os.W_OK):
        return
    target = os.path.join(tempfile.gettempdir(), "scale_forecasting_mpl")
    try:
        os.makedirs(target, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = target
    except OSError:
        pass


def quiet_lightning() -> None:
    """Suppress verbose PyTorch Lightning banner and step logs during cell execution."""
    for logger_name in (
        "pytorch_lightning",
        "lightning",
        "lightning.pytorch",
        "lightning_fabric",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _encode_scalar(val: Any) -> float:
    """Deterministically map a static covariate scalar (numeric, bool, or string) to float."""
    if isinstance(val, bool | np.bool_):
        return 1.0 if val else 0.0
    if isinstance(val, int | float | np.number):
        return float(val)
    # Deterministic [0, 1) hash for string / categorical attributes in single-series mode.
    return float(zlib.crc32(str(val).encode("utf-8")) & 0xFFFFFFFF) / float(2**32)


def _build_static_df(
    static_map: Mapping[str, Mapping[str, Any]] | None,
    uids: list[str],
) -> tuple[pd.DataFrame | None, list[str]]:
    """Build a purely numeric ``static_df`` with ``unique_id`` and encoded static columns."""
    if not static_map:
        return None, []
    first = next(iter(static_map.values()), None)
    if not first:
        return None, []
    raw_cols = list(first.keys())
    if not raw_cols:
        return None, []

    rows: list[dict[str, Any]] = []
    for uid in uids:
        sdict = static_map.get(uid, {})
        row: dict[str, Any] = {"unique_id": uid}
        for col in raw_cols:
            row[col] = sdict.get(col)
        rows.append(row)
    raw_df = pd.DataFrame(rows)

    encoded_parts: list[pd.DataFrame] = [raw_df[["unique_id"]]]
    stat_cols: list[str] = []
    for col in raw_cols:
        series = raw_df[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            cname = f"stat__{col}"
            encoded_parts.append(pd.DataFrame({cname: series.astype(float)}))
            stat_cols.append(cname)
        else:
            if len(uids) > 1 and series.nunique(dropna=True) > 1:
                dummies = pd.get_dummies(series.astype(str), prefix=f"stat__{col}", dtype=float)
                encoded_parts.append(dummies)
                stat_cols.extend(list(dummies.columns))
            else:
                cname = f"stat__{col}"
                encoded_parts.append(
                    pd.DataFrame({cname: [float(_encode_scalar(v)) for v in series]})
                )
                stat_cols.append(cname)
    static_df = pd.concat(encoded_parts, axis=1)
    return static_df, stat_cols


class NeuralForecastBaseModel(BaseModel):
    """Base class for ``neuralforecast`` sequence-to-sequence architectures."""

    runtime = "python"
    family = "deep_learning"
    supports_native_intervals = True
    supports_global = True
    supports_hybrid = False
    gpu_capable = True
    supports_recondition = True
    supports_extrapolate = True
    package = "neuralforecast"
    package_url = "https://nixtlaverse.nixtla.io/neuralforecast/index.html"
    optional_import = "neuralforecast"
    optional_extra = "models-dl"

    # Upstream class name inside `neuralforecast.models` (e.g., "TiDE", "TFT", "TSMixerx").
    _nf_model_name: ClassVar[str]

    def __init__(self, params: dict[str, Any], ctx: Any) -> None:
        super().__init__(params, ctx)
        self._nf: Any = None
        self._fit_h: int = max(1, int(ctx.horizon))
        self._input_size: int = 2
        self._futr_cols: list[str] = []
        self._hist_cols: list[str] = []
        self._stat_cols: list[str] = []
        self._train_df: pd.DataFrame | None = None
        self._static_df: pd.DataFrame | None = None
        self._panel_last_dates: dict[str, pd.Timestamp] = {}
        self._fitted_device: str | None = None

    @abstractmethod
    def _arch_kwargs(self, *, n_series: int) -> dict[str, Any]:
        """Architecture-specific keyword arguments passed to the ``neuralforecast`` model class."""

    def _split_exog_cols(self, X: pd.DataFrame | None) -> tuple[list[str], list[str]]:
        """Partition ``X`` columns into ``(futr_exog_list, hist_exog_list)``."""
        if not self.supports_exog or X is None or X.empty:
            return [], []
        past_set = set(self.ctx.past_covariates)
        hist_cols = [str(c) for c in X.columns if c in past_set]
        futr_cols = [str(c) for c in X.columns if c not in past_set]
        return futr_cols, hist_cols

    def _resolve_input_size(self, min_len: int) -> int:
        """Resolve lookback window ``input_size`` from params or series length + horizon."""
        if "input_size" in self.params and self.params["input_size"] is not None:
            return max(2, int(self.params["input_size"]))
        return max(2, min(2 * self._fit_h, max(2, min_len // 2)))

    def _trainer_kwargs(self, max_steps: int) -> dict[str, Any]:
        """PyTorch Lightning trainer configuration respecting ``self.ctx.device``."""
        val_check = min(int(self.params.get("val_check_steps", 100)), max_steps)
        kw: dict[str, Any] = {
            "max_steps": max_steps,
            "val_check_steps": max(1, val_check),
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "enable_checkpointing": False,
            "logger": False,
        }
        if self.ctx.device == "gpu":
            kw["accelerator"] = "gpu"
            kw["devices"] = 1
        else:
            kw["accelerator"] = self.ctx.device
        return kw

    def _instantiate_nf(
        self,
        *,
        h: int,
        input_size: int,
        n_series: int,
        futr_cols: list[str],
        hist_cols: list[str],
        stat_cols: list[str],
    ) -> Any:
        """Construct the ``NeuralForecast`` wrapper and inner Lightning model."""
        ensure_mpl_dir()
        quiet_lightning()
        try:
            import neuralforecast.models as nf_models
            from neuralforecast import NeuralForecast
            from neuralforecast.losses.pytorch import MQLoss
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ModelError("neuralforecast not installed; install the 'models-dl' extra") from e

        model_cls = getattr(nf_models, self._nf_model_name)
        max_steps = max(1, int(self.params.get("max_steps", 25)))
        learning_rate = float(self.params.get("learning_rate", 1e-2))
        batch_size = int(self.params.get("batch_size", 32))
        scaler_type = str(self.params.get("scaler_type", "standard"))

        common_kwargs: dict[str, Any] = {
            "h": h,
            "input_size": input_size,
            "loss": MQLoss(quantiles=list(_BAND)),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "scaler_type": scaler_type,
            "random_seed": int(self.ctx.seed),
            "start_padding_enabled": True,
            **self._trainer_kwargs(max_steps),
            **self._arch_kwargs(n_series=n_series),
        }
        if self.supports_exog:
            if futr_cols:
                common_kwargs["futr_exog_list"] = list(futr_cols)
            if hist_cols:
                common_kwargs["hist_exog_list"] = list(hist_cols)
            if stat_cols:
                common_kwargs["stat_exog_list"] = list(stat_cols)

        model = model_cls(**common_kwargs)
        return NeuralForecast(models=[model], freq=self.ctx.freq)

    def _record_device_used(self) -> None:
        """Record the device where the fit executed before or after Lightning teardown."""
        try:
            import torch

            if self.ctx.device == "gpu":
                self._fitted_device = "cuda"
                return
            if self.ctx.device == "cpu":
                self._fitted_device = "cpu"
                return
            self._fitted_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            self._fitted_device = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        if len(y) < 2:
            raise ModelError(f"{self.name} requires at least 2 observations")
        self._last_date = pd.Timestamp(y.index[-1])
        self._fit_h = max(1, int(self.ctx.horizon))
        self._input_size = self._resolve_input_size(len(y))
        self._futr_cols, self._hist_cols = self._split_exog_cols(X)

        data: dict[str, Any] = {
            "unique_id": _LOCAL_UID,
            "ds": pd.DatetimeIndex(y.index).as_unit("ns"),
            "y": y.astype(float).to_numpy(),
        }
        if X is not None:
            for col in self._futr_cols + self._hist_cols:
                data[col] = X[col].astype(float).to_numpy()
        self._train_df = pd.DataFrame(data)

        static_map = (
            {_LOCAL_UID: self.ctx.static_covariates} if self.ctx.static_covariates else None
        )
        self._static_df, self._stat_cols = (
            _build_static_df(static_map, [_LOCAL_UID]) if self.supports_exog else (None, [])
        )

        self._nf = self._instantiate_nf(
            h=self._fit_h,
            input_size=self._input_size,
            n_series=1,
            futr_cols=self._futr_cols,
            hist_cols=self._hist_cols,
            stat_cols=self._stat_cols,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._nf.fit(df=self._train_df, static_df=self._static_df)
        self._record_device_used()

    def recondition(self, y_new: pd.Series, X_new: pd.DataFrame | None = None) -> None:
        """Append new observations to the conditioning history without refitting network weights."""
        if self._train_df is None or self._nf is None:
            raise ModelError(f"{self.name} must be fit before recondition()")
        if len(y_new) == 0:
            return
        new_data: dict[str, Any] = {
            "unique_id": _LOCAL_UID,
            "ds": pd.DatetimeIndex(y_new.index).as_unit("ns"),
            "y": y_new.astype(float).to_numpy(),
        }
        for col in self._futr_cols + self._hist_cols:
            if X_new is not None and col in X_new.columns:
                new_data[col] = X_new[col].astype(float).to_numpy()
            else:
                last_val = float(self._train_df[col].iloc[-1])
                new_data[col] = np.full(len(y_new), last_val, dtype=float)
        new_df = pd.DataFrame(new_data)
        self._train_df = pd.concat([self._train_df, new_df], ignore_index=True)
        self._last_date = pd.Timestamp(y_new.index[-1])
        self._origin_offset = 0
        self._gap_exog = None

    def _predict_single_steps(
        self, steps: int, full_exog: pd.DataFrame | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict ``steps`` forward from ``self._train_df``, rolling if ``steps > self._fit_h``."""
        assert self._train_df is not None and self._nf is not None
        hist_df = self._train_df.copy()
        cur_last = pd.Timestamp(hist_df["ds"].iloc[-1])
        alias = self._nf_model_name

        means: list[np.ndarray] = []
        los: list[np.ndarray] = []
        his: list[np.ndarray] = []
        done = 0

        while done < steps:
            chunk_dates = self._future_index(cur_last, self._fit_h)
            futr_df: pd.DataFrame | None = None
            if self._futr_cols:
                futr_data: dict[str, Any] = {
                    "unique_id": _LOCAL_UID,
                    "ds": chunk_dates,
                }
                for col in self._futr_cols:
                    if full_exog is not None and col in full_exog.columns:
                        src = full_exog[col].astype(float).to_numpy()
                        slice_vals = src[done : done + self._fit_h]
                        if len(slice_vals) < self._fit_h:
                            fill_val = (
                                float(src[-1]) if len(src) > 0 else float(hist_df[col].iloc[-1])
                            )
                            pad = np.full(self._fit_h - len(slice_vals), fill_val, dtype=float)
                            slice_vals = np.concatenate([slice_vals, pad])
                        futr_data[col] = slice_vals
                    else:
                        futr_data[col] = np.full(
                            self._fit_h, float(hist_df[col].iloc[-1]), dtype=float
                        )
                futr_df = pd.DataFrame(futr_data)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                fc = self._nf.predict(df=hist_df, static_df=self._static_df, futr_df=futr_df)

            chunk_mean = fc[f"{alias}-median"].to_numpy(dtype=float)
            chunk_lo = fc[f"{alias}-lo-80.0"].to_numpy(dtype=float)
            chunk_hi = fc[f"{alias}-hi-80.0"].to_numpy(dtype=float)
            means.append(chunk_mean)
            los.append(chunk_lo)
            his.append(chunk_hi)
            done += self._fit_h

            if done < steps:
                next_rows: dict[str, Any] = {
                    "unique_id": _LOCAL_UID,
                    "ds": chunk_dates,
                    "y": chunk_mean,
                }
                for col in self._futr_cols:
                    assert futr_df is not None
                    next_rows[col] = futr_df[col].to_numpy(dtype=float)
                for col in self._hist_cols:
                    next_rows[col] = np.full(self._fit_h, float(hist_df[col].iloc[-1]), dtype=float)
                hist_df = pd.concat([hist_df, pd.DataFrame(next_rows)], ignore_index=True)
                cur_last = pd.Timestamp(chunk_dates[-1])

        return (
            np.concatenate(means)[:steps],
            np.concatenate(los)[:steps],
            np.concatenate(his)[:steps],
        )

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm

        steps = self._forecast_steps(horizon)
        full_exog = self._forecast_exog(X)
        mean_all, lo_all, hi_all = self._predict_single_steps(steps, full_exog)
        mean = mean_all[-horizon:]
        lo = lo_all[-horizon:]
        hi = hi_all[-horizon:]

        z = norm.ppf(_BAND[2])
        sigma = np.maximum((hi - lo) / (2.0 * z), 0.0)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        ds = self._forecast_index(horizon)
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    # --- global / panel execution ---------------------------------------------------

    def fit_panel(
        self,
        series_map: Mapping[str, tuple[pd.Series, pd.DataFrame | None]],
        static_map: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        """Fit a single shared global model across all series in ``series_map``."""
        if not series_map:
            raise ModelError(f"{self.name}.fit_panel requires a non-empty series_map")
        uids = sorted(series_map.keys())
        min_len = min(len(series_map[uid][0]) for uid in uids)
        if min_len < 2:
            raise ModelError(f"{self.name} requires at least 2 observations per series")

        first_X = series_map[uids[0]][1]
        self._futr_cols, self._hist_cols = self._split_exog_cols(first_X)
        self._fit_h = max(1, int(self.ctx.horizon))
        self._input_size = self._resolve_input_size(min_len)
        self._panel_last_dates = {}

        frames: list[pd.DataFrame] = []
        for uid in uids:
            y, X = series_map[uid]
            self._panel_last_dates[uid] = pd.Timestamp(y.index[-1])
            row_data: dict[str, Any] = {
                "unique_id": uid,
                "ds": pd.DatetimeIndex(y.index).as_unit("ns"),
                "y": y.astype(float).to_numpy(),
            }
            if X is not None:
                for col in self._futr_cols + self._hist_cols:
                    row_data[col] = X[col].astype(float).to_numpy()
            frames.append(pd.DataFrame(row_data))
        self._train_df = pd.concat(frames, ignore_index=True)
        self._static_df, self._stat_cols = (
            _build_static_df(static_map, uids) if self.supports_exog else (None, [])
        )

        self._nf = self._instantiate_nf(
            h=self._fit_h,
            input_size=self._input_size,
            n_series=len(uids),
            futr_cols=self._futr_cols,
            hist_cols=self._hist_cols,
            stat_cols=self._stat_cols,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._nf.fit(df=self._train_df, static_df=self._static_df)
        self._record_device_used()

    def predict_panel(
        self,
        horizon: int,
        future_exog_map: Mapping[str, pd.DataFrame | None] | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        transform_lambdas: Mapping[str, float | None] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Predict ``horizon`` steps across all series in the fitted panel."""
        from scipy.stats import norm

        if self._train_df is None or self._nf is None or not self._panel_last_dates:
            raise ModelError(f"{self.name}.predict_panel called before fit_panel()")

        uids = sorted(self._panel_last_dates.keys())
        hist_df = self._train_df.copy()
        cur_last = dict(self._panel_last_dates)
        alias = self._nf_model_name

        means_by_uid: dict[str, list[np.ndarray]] = {u: [] for u in uids}
        los_by_uid: dict[str, list[np.ndarray]] = {u: [] for u in uids}
        his_by_uid: dict[str, list[np.ndarray]] = {u: [] for u in uids}
        done = 0

        while done < horizon:
            futr_df: pd.DataFrame | None = None
            chunk_dates_by_uid = {u: self._future_index(cur_last[u], self._fit_h) for u in uids}
            if self._futr_cols:
                futr_frames: list[pd.DataFrame] = []
                for uid in uids:
                    u_exog = future_exog_map.get(uid) if future_exog_map else None
                    u_hist = hist_df[hist_df["unique_id"] == uid]
                    fdata: dict[str, Any] = {
                        "unique_id": uid,
                        "ds": chunk_dates_by_uid[uid],
                    }
                    for col in self._futr_cols:
                        if u_exog is not None and col in u_exog.columns:
                            src = u_exog[col].astype(float).to_numpy()
                            slice_vals = src[done : done + self._fit_h]
                            if len(slice_vals) < self._fit_h:
                                fill_val = (
                                    float(src[-1]) if len(src) > 0 else float(u_hist[col].iloc[-1])
                                )
                                pad = np.full(self._fit_h - len(slice_vals), fill_val, dtype=float)
                                slice_vals = np.concatenate([slice_vals, pad])
                            fdata[col] = slice_vals
                        else:
                            fdata[col] = np.full(
                                self._fit_h, float(u_hist[col].iloc[-1]), dtype=float
                            )
                    futr_frames.append(pd.DataFrame(fdata))
                futr_df = pd.concat(futr_frames, ignore_index=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                fc = self._nf.predict(df=hist_df, static_df=self._static_df, futr_df=futr_df)

            next_hist_parts: list[pd.DataFrame] = []
            for uid in uids:
                ufc = fc[fc["unique_id"] == uid]
                c_mean = ufc[f"{alias}-median"].to_numpy(dtype=float)
                c_lo = ufc[f"{alias}-lo-80.0"].to_numpy(dtype=float)
                c_hi = ufc[f"{alias}-hi-80.0"].to_numpy(dtype=float)
                means_by_uid[uid].append(c_mean)
                los_by_uid[uid].append(c_lo)
                his_by_uid[uid].append(c_hi)

                if done + self._fit_h < horizon:
                    u_hist = hist_df[hist_df["unique_id"] == uid]
                    nrow: dict[str, Any] = {
                        "unique_id": uid,
                        "ds": chunk_dates_by_uid[uid],
                        "y": c_mean,
                    }
                    if futr_df is not None:
                        u_futr = futr_df[futr_df["unique_id"] == uid]
                        for col in self._futr_cols:
                            nrow[col] = u_futr[col].to_numpy(dtype=float)
                    for col in self._hist_cols:
                        nrow[col] = np.full(self._fit_h, float(u_hist[col].iloc[-1]), dtype=float)
                    next_hist_parts.append(pd.DataFrame(nrow))
                    cur_last[uid] = pd.Timestamp(chunk_dates_by_uid[uid][-1])

            done += self._fit_h
            if done < horizon and next_hist_parts:
                hist_df = pd.concat([hist_df, *next_hist_parts], ignore_index=True)

        z = norm.ppf(_BAND[2])
        t = self.ctx.transform
        out: dict[str, pd.DataFrame] = {}
        for uid in uids:
            mean = np.concatenate(means_by_uid[uid])[:horizon]
            lo = np.concatenate(los_by_uid[uid])[:horizon]
            hi = np.concatenate(his_by_uid[uid])[:horizon]
            sigma = np.maximum((hi - lo) / (2.0 * z), 0.0)
            lam = (
                transform_lambdas.get(uid, self.ctx.transform_lambda)
                if transform_lambdas is not None
                else self.ctx.transform_lambda
            )
            qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
            ds = self._future_index(self._panel_last_dates[uid], horizon)
            out[uid] = self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))
        return out

    def device_used(self) -> str | None:
        return self._fitted_device

    @classmethod
    def gpu_useful(cls, params: Mapping[str, Any]) -> bool:
        """Windowed deep-learning architectures use tensor kernels across sequence windows."""
        return True

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "max_steps": trial.suggest_int("max_steps", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        mode = params.get("training_mode", "local")
        allowed = ["local", "global"] + (["hybrid"] if cls.supports_hybrid else [])
        if mode not in allowed:
            raise ConfigError(
                f"model_params.{cls.name}.training_mode={mode!r} is not supported by "
                f"'{cls.name}'; supported modes: {allowed}."
            )
        input_size = params.get("input_size")
        if input_size is not None and int(input_size) < 2:
            raise ConfigError(
                f"model_params.{cls.name}.input_size must be >= 2 (got {input_size!r})."
            )
