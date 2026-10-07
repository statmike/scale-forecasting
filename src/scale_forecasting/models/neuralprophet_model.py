"""NeuralProphet — a neural additive forecaster (the ``deep_learning`` family).

**On GPUs, measured rather than assumed.** This file used to open by calling NeuralProphet "the one
model that benefits from a GPU". It does not, at the parameters we construct it with. Across 31,356
fits on live T4s, peak device memory was 50–78 KB against a card holding 17,179,869,184, while
``cpu_seconds / fit_seconds`` sat between 0.93 and 0.996 on a single thread. The reason is not a
broken install — CUDA initializes and the tensors are device-resident — it is that ``n_lags``
defaults to 0, so the network is a few hundred trend and Fourier parameters. The Lightning loop,
the dataloader and pandas dwarf the kernels. Autoregression (``n_lags > 0``, AR-Net) is what would
make the device worth attaching.

``n_lags``, ``n_forecasts`` and ``batch_size`` are authorable through
``model_params.neuralprophet``; all three keep the library's own defaults otherwise, so nothing
moves unless a config asks for it. **Autoregression has never been run at scale here** — no
accuracy A/B, no live smoke — so it is an expert opt-in and not a default. Turning it on changes
the shape of what ``predict`` reads back; see `_read_steps`.

One model, one file. Runtime python, deep_learning family. NeuralProphet is
an optional dependency, imported lazily in ``fit`` so the model registers without it (and
without dragging torch into the base install). It supports quantile regression, but quantiles
are fixed at construction; our contract passes the quantile set at *predict* time, so — like
``prophet`` and ``theta`` — we fit a fixed symmetric band, back the implied sigma out of it,
and place any requested quantile from that Gaussian, honoring arbitrary quantile sets.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import ConfigError, ModelError
from ..features import invert_transform
from ._neuralforecast_base import _ensure_mpl_dir, _quiet_lightning
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import optuna

# Fixed band fit into the network; sigma is backed out of it for arbitrary quantiles.
_BAND = (0.1, 0.9)


def _import_neuralprophet() -> tuple[Any, Any, Any]:
    """Import the library lazily and quietly.

    Importing ``neuralprophet`` has side effects a caller cannot opt out of: it turns on
    ``logging.captureWarnings`` process-wide with its own stderr handler, logs an *error* when
    ``plotly`` is absent (it is not a dependency here; interactive plots are not used), and pulls in
    ``tqdm.auto``, which warns when ``ipywidgets`` is missing. None of it is actionable by the user
    of this model, so the import runs with warnings off and the library's logger held at CRITICAL
    until it has installed its own level. The ``ImportError`` for a missing extra still propagates.
    """
    logging.getLogger("NP").setLevel(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from neuralprophet import NeuralProphet, set_log_level, set_random_seed
    return NeuralProphet, set_log_level, set_random_seed


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence the library's deprecation and configuration chatter for one fit or predict call.

    Because the import above has routed every warning through ``logging``, each pandas
    ``FutureWarning`` NeuralProphet triggers inside its own training loop — hundreds per fit — and
    each Lightning ``PossibleUserWarning`` about an unconfigured logger or an inferred batch size
    would otherwise reach the terminal or the notebook, prefixed with the absolute path of the
    site-packages file that raised it. Same categories the ``neuralforecast`` base suppresses.
    """
    _quiet_lightning()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


class NeuralProphetModel(BaseModel):
    """NeuralProphet forecaster (neural additive model)."""

    name = "neuralprophet"
    runtime = "python"
    family = "deep_learning"
    supports_exog = False
    supports_native_intervals = True
    supports_global = True
    supports_hybrid = True
    # One of the models here with a tensor library under it, and so one a device can serve.
    gpu_capable = True
    # Extrapolate only. The network's weights are the estimate and there is no partial-fit seam that
    # absorbs an observation without training, so it declines the frozen scheme and answers the
    # staleness one. Under autoregression (`n_lags > 0`) the reach is bounded by `n_forecasts`: the
    # span it must cover is the horizon *plus* the skipped gap, and `_read_steps` says so if short.
    supports_extrapolate = True
    package = "neuralprophet"
    package_url = "https://neuralprophet.com/"
    optional_import = "neuralprophet"
    optional_extra = "models-dl"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        _ensure_mpl_dir()
        try:
            NeuralProphet, set_log_level, set_random_seed = _import_neuralprophet()
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
        # n_lags/n_forecasts default to NeuralProphet's own 0/1 — no autoregression, one head, the
        # shape every run in the ledger was produced with. Both only move when a config authors
        # them. batch_size is passed through as None (the library's "choose one") unless authored;
        # it is measurably faster at 128-512 but has no accuracy A/B yet, so it is not a default.
        model = NeuralProphet(
            quantiles=list(_BAND),
            epochs=int(self.params.get("epochs", 50)),
            learning_rate=float(self.params.get("learning_rate", 0.01)),
            n_lags=int(self.params.get("n_lags", 0)),
            n_forecasts=int(self.params.get("n_forecasts", 1)),
            batch_size=self._optional_int("batch_size"),
            collect_metrics=False,
            trainer_config=self._trainer_config(),
        )
        with _quiet():
            model.fit(self._train, freq=self.ctx.freq, progress=None, minimal=True)
        self._model = model

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm  # lazy: keep scipy off the module top (lean launch point)

        # Read the whole span from the fit's last observation and keep the tail, so an advanced
        # origin lands on the right steps of the network's own output rather than re-anchoring it.
        steps = self._forecast_steps(horizon)
        with _quiet():
            future = self._model.make_future_dataframe(self._train, periods=steps)
            forecast = self._model.predict(future)
        mean, lo, hi = (a[-horizon:] for a in self._read_steps(forecast, steps))
        # Back out sigma from the symmetric band, then place any requested quantile.
        z = norm.ppf(_BAND[1])
        sigma = (hi - lo) / (2.0 * z)

        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        ds = self._forecast_index(horizon)
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    def fit_panel(
        self,
        series_map: Mapping[str, tuple[pd.Series, pd.DataFrame | None]],
        static_map: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        """Fit one global or hybrid NeuralProphet network across all series in ``series_map``."""
        _ensure_mpl_dir()
        try:
            NeuralProphet, set_log_level, set_random_seed = _import_neuralprophet()
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ModelError("neuralprophet not installed; install the 'models' extra") from e
        if not series_map:
            raise ModelError("neuralprophet.fit_panel requires a non-empty series_map")
        set_log_level("ERROR")
        set_random_seed(self.ctx.seed)

        uids = sorted(series_map.keys())
        frames: list[pd.DataFrame] = []
        self._panel_last_dates: dict[str, pd.Timestamp] = {}
        for uid in uids:
            y, _ = series_map[uid]
            if len(y) < 2:
                raise ModelError("neuralprophet requires at least 2 observations per series")
            self._panel_last_dates[uid] = pd.Timestamp(y.index[-1])
            frames.append(
                pd.DataFrame(
                    {
                        "ID": uid,
                        "ds": pd.DatetimeIndex(y.index),
                        "y": y.astype(float).to_numpy(),
                    }
                )
            )
        self._panel_train = pd.concat(frames, ignore_index=True)

        mode = str(self.params.get("training_mode", "global"))
        default_trend = "local" if mode == "hybrid" else "global"
        default_season = "global"
        trend_gl = str(self.params.get("trend_global_local", default_trend))
        season_gl = str(self.params.get("season_global_local", default_season))

        model = NeuralProphet(
            quantiles=list(_BAND),
            epochs=int(self.params.get("epochs", 50)),
            learning_rate=float(self.params.get("learning_rate", 0.01)),
            n_lags=int(self.params.get("n_lags", 0)),
            n_forecasts=int(self.params.get("n_forecasts", 1)),
            batch_size=self._optional_int("batch_size"),
            trend_global_local=trend_gl,
            season_global_local=season_gl,
            collect_metrics=False,
            trainer_config=self._trainer_config(),
        )
        with _quiet():
            model.fit(self._panel_train, freq=self.ctx.freq, progress=None, minimal=True)
        self._model = model

    def predict_panel(
        self,
        horizon: int,
        future_exog_map: Mapping[str, pd.DataFrame | None] | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        transform_lambdas: Mapping[str, float | None] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Predict ``horizon`` steps for every series in the fitted global/hybrid panel."""
        from scipy.stats import norm

        with _quiet():
            future = self._model.make_future_dataframe(self._panel_train, periods=horizon)
            fc = self._model.predict(future)
        z = norm.ppf(_BAND[1])
        t = self.ctx.transform
        out: dict[str, pd.DataFrame] = {}
        for uid, last_date in self._panel_last_dates.items():
            ufc = fc[fc["ID"] == uid]
            mean, lo, hi = (
                a[-horizon:] for a in self._read_steps(ufc, horizon, last_date=last_date)
            )
            sigma = (hi - lo) / (2.0 * z)
            lam = (
                transform_lambdas.get(uid, self.ctx.transform_lambda)
                if transform_lambdas is not None
                else self.ctx.transform_lambda
            )
            qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
            ds = self._future_index(last_date, horizon)
            out[uid] = self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))
        return out

    def device_used(self) -> str | None:
        """Where the fit ran — read off the trainer that ran it, not off the weights afterwards.

        Reading a parameter tensor from ``self._model.model`` after `fit()` returns ``"cpu"`` even
        for a GPU fit because PyTorch Lightning ends every training run by moving the module back to
        CPU (``Strategy.teardown`` calls ``self.lightning_module.cpu()``). Reading the post-fit
        weights would therefore misclassify GPU cells as ``"cpu"`` and cause `device_audit` to
        report ``MISSING_DEVICE``.

        ``trainer.strategy.root_device`` survives teardown and is not the request restated. It is
        what Lightning's accelerator connector *resolved* the request to against the hardware it
        found, and under an explicit ``accelerator="gpu"`` a fit cannot reach the end of training
        without one — Lightning raises instead. The independent evidence sits beside it in the same
        row: ``peak_gpu_bytes`` is measured by the worker from ``torch.cuda``, and `_require_device`
        has already failed any cell that asked for a device the worker could not see.

        The parameter read stays as the fallback, because it is right whenever nothing moved the
        weights — every CPU fit, and any Lightning version that reshapes the trainer.

        Never raises: an unfitted model, a moved attribute, or an empty parameter list all yield
        ``None`` (unknown), because a probe that sank a good forecast would be worse than no probe.
        """
        try:
            return str(self._model.trainer.strategy.root_device.type)
        except Exception:  # noqa: BLE001 - the evidence is optional; the forecast is not
            pass
        try:
            return str(next(self._model.model.parameters()).device.type)
        except Exception:  # noqa: BLE001 - same
            return None

    def _trainer_config(self) -> dict[str, Any]:
        """The Lightning trainer knobs — chiefly *which device*, stated rather than guessed.

        This used to be a hardcoded ``accelerator="auto"``. Auto can never fail, and that is the
        problem: a run that paid for accelerators and got none silently fitted on CPU, which is how
        every GPU run in the ledger came to be a CPU run without anything reporting it. It also
        cannot push a model *off* a card — on a mixed-hardware Dataproc cluster a CPU family sees
        whatever device its executor exposes.

        ``ctx.device`` says it outright. ``"auto"`` reproduces the old behaviour exactly, and is
        what a local run, an SDK call and every CPU job still get.

        ``devices=1`` on the GPU branch is a *request only*, and this is the one place that says
        so. NeuralProphet's own ``configure_trainer`` overwrites it with ``-1`` — all visible
        devices — for every accelerator it resolves to ``"gpu"``. Every shape we provision puts one
        card in front of a worker, where ``-1`` and ``1`` are the same thing, so it is stated to
        keep the intent on the record rather than because the library honours it.

        **No ``callbacks`` here, deliberately.** A Lightning callback would be the natural way to
        read the fit's device while the weights are still on it, but handing NeuralProphet 0.9.0 a
        ``callbacks`` key sends ``configure_trainer`` down a branch that dereferences
        ``pl.callbacks.ProgressBarBase``, removed in the Lightning we pin, and the fit dies with an
        ``AttributeError`` before it starts. `device_used` reads the trainer instead.
        """
        if self.ctx.device == "gpu":
            return {"accelerator": "gpu", "devices": 1}
        return {"accelerator": self.ctx.device}

    def _optional_int(self, key: str) -> int | None:
        """An authored int, or ``None`` to leave the library's own default in place."""
        value = self.params.get(key)
        return None if value is None else int(value)

    def _read_steps(
        self, fc: pd.DataFrame, steps: int, *, last_date: pd.Timestamp | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pull the mean and the band for steps 1..``steps`` out of a forecast frame.

        ``steps`` is the span from the *fit's* last observation, which is the horizon on the
        ordinary path and horizon-plus-gap when the forecast origin has been advanced.

        NeuralProphet has two output shapes and reading the wrong one is silent rather than loud.

        Without autoregression (``n_lags`` unset, the shipped default) there is a single head:
        every future row carries its forecast in ``yhat1``, and the frame is read straight down
        that column.

        With ``n_lags > 0`` the model builds ``n_forecasts`` **direct** heads and returns them on a
        *diagonal* — the row for step ``i`` populates ``yhat{i}`` and leaves every other ``yhat``
        column NaN. Reading ``yhat1`` down that frame yields one number followed by
        ``horizon - 1`` NaNs.

        Rows are selected by date, not by position, because ``make_future_dataframe`` returns
        ``n_lags + n_forecasts`` rows whatever ``periods`` it was asked for: at ``n_forecasts=7``
        with ``horizon=3``, the tail of the frame is steps 5-7, not steps 1-3.
        """
        anchor = self._last_date if last_date is None else last_date
        future = fc[fc["ds"] > anchor]
        heads = self._n_heads(fc)
        available = len(future) if heads <= 1 else min(heads, len(future))
        if available < steps:
            raise ModelError(
                f"neuralprophet produced {available} forecast steps for a span of {steps}. "
                f"With n_lags > 0 it emits exactly n_forecasts direct steps and does not recurse, "
                f"so model_params.neuralprophet.n_forecasts must be at least {steps}."
            )
        if heads <= 1:
            block = future.head(steps)
            return (
                block["yhat1"].to_numpy(dtype=float),
                block[self._band_col(1, _BAND[0])].to_numpy(dtype=float),
                block[self._band_col(1, _BAND[1])].to_numpy(dtype=float),
            )
        rows = [future.iloc[i] for i in range(steps)]
        return (
            np.array([float(r[f"yhat{i + 1}"]) for i, r in enumerate(rows)]),
            np.array([float(r[self._band_col(i + 1, _BAND[0])]) for i, r in enumerate(rows)]),
            np.array([float(r[self._band_col(i + 1, _BAND[1])]) for i, r in enumerate(rows)]),
        )

    @staticmethod
    def _n_heads(fc: pd.DataFrame) -> int:
        """How many direct-forecast heads the fitted model emitted (``yhat1``…``yhatN``)."""
        n = 0
        while f"yhat{n + 1}" in fc.columns:
            n += 1
        return n

    @staticmethod
    def _band_col(step: int, q: float) -> str:
        """NeuralProphet names quantile columns per head, like ``yhat1 10.0%``."""
        return f"yhat{step} {q * 100:.1f}%"

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 20, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        }

    @classmethod
    def gpu_useful(cls, params: Mapping[str, Any]) -> bool:
        """A device earns its cost here under autoregression (``n_lags > 0``) or panel mode.

        Without it the network is a few hundred trend and Fourier parameters and the Lightning
        loop, the dataloader and pandas dwarf the kernels; that is the shape Phase 0 measured at
        50–78 KB of device memory and 95% CPU-bound. AR-Net or global/hybrid panel training makes
        the workload large enough for the card to matter. ``n_forecasts`` alone does not qualify:
        extra heads without lags are extra output units on the same tiny network.
        """
        mode = str(params.get("training_mode", "local"))
        return int(params.get("n_lags", 0) or 0) > 0 or mode in ("global", "hybrid")

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        """Validate ``training_mode`` and autoregression direct-head reach."""
        mode = params.get("training_mode", "local")
        if mode not in ("local", "global", "hybrid"):
            raise ConfigError(
                f"model_params.neuralprophet.training_mode={mode!r} is invalid; "
                "supported modes: ['local', 'global', 'hybrid']."
            )
        n_lags = int(params.get("n_lags", 0) or 0)
        if n_lags <= 0:
            return
        n_forecasts = int(params.get("n_forecasts", 1) or 1)
        if n_forecasts < max_horizon:
            raise ConfigError(
                f"model_params.neuralprophet sets n_lags={n_lags}, which turns on autoregression, "
                f"but n_forecasts={n_forecasts} is below this run's longest horizon of "
                f"{max_horizon}. NeuralProphet emits exactly n_forecasts direct steps and does not "
                f"recurse, so set n_forecasts to at least {max_horizon}."
            )


register(NeuralProphetModel)
