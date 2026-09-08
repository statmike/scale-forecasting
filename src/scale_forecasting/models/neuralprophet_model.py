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

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import ConfigError, ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    from collections.abc import Mapping

    import optuna

# Fixed band fit into the network; sigma is backed out of it for arbitrary quantiles.
_BAND = (0.1, 0.9)


class NeuralProphetModel(BaseModel):
    """NeuralProphet forecaster (neural additive model)."""

    name = "neuralprophet"
    runtime = "python"
    family = "deep_learning"
    supports_exog = False
    supports_native_intervals = True
    # The only model here with a tensor library under it, and so the only one a device can serve.
    gpu_capable = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from neuralprophet import NeuralProphet, set_log_level, set_random_seed
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
            trainer_config=self._trainer_config(),
        )
        model.fit(self._train, freq=self.ctx.freq, progress=None)
        self._model = model

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm  # lazy: keep scipy off the module top (lean launch point)

        future = self._model.make_future_dataframe(self._train, periods=horizon)
        mean, lo, hi = self._read_steps(self._model.predict(future), horizon)
        # Back out sigma from the symmetric band, then place any requested quantile.
        z = norm.ppf(_BAND[1])
        sigma = (hi - lo) / (2.0 * z)

        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    def device_used(self) -> str | None:
        """Where the fitted weights actually live — read off a parameter tensor, not off config.

        The parameters are the receipt. Lightning's trainer can be asked what accelerator it was
        *configured* with, but that is the request again; a tensor's ``.device`` is where the
        arithmetic happened. NeuralProphet keeps the LightningModule on ``.model`` after ``fit``.

        Never raises: an unfitted model, a library version that moved the attribute, or an empty
        parameter list all yield ``None`` (unknown), because a probe that sank a good forecast
        would be worse than no probe at all.
        """
        try:
            return str(next(self._model.model.parameters()).device.type)
        except Exception:  # noqa: BLE001 - the evidence is optional; the forecast is not
            return None

    def _trainer_config(self) -> dict[str, Any]:
        """The Lightning trainer knobs — chiefly *which device*, stated rather than guessed.

        This used to be a hardcoded ``accelerator="auto"``. Auto can never fail, and that is the
        problem: a run that paid for accelerators and got none silently fitted on CPU, which is how
        every GPU run in the ledger came to be a CPU run without anything reporting it. It also
        cannot push a model *off* a card — on a mixed-hardware Dataproc cluster a CPU family sees
        whatever device its executor exposes.

        ``ctx.device`` says it outright. ``"gpu"`` additionally pins ``devices=1`` so a task that
        packs several cells onto one card does not have each of them claim every visible device.
        ``"auto"`` reproduces the old behaviour exactly, and is what a local run, an SDK call and
        every CPU job still get.
        """
        if self.ctx.device == "gpu":
            return {"accelerator": "gpu", "devices": 1}
        return {"accelerator": self.ctx.device}

    def _optional_int(self, key: str) -> int | None:
        """An authored int, or ``None`` to leave the library's own default in place."""
        value = self.params.get(key)
        return None if value is None else int(value)

    def _read_steps(
        self, fc: pd.DataFrame, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pull the mean and the band for steps 1..``horizon`` out of a forecast frame.

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
        future = fc[fc["ds"] > self._last_date]
        heads = self._n_heads(fc)
        available = len(future) if heads <= 1 else min(heads, len(future))
        if available < horizon:
            raise ModelError(
                f"neuralprophet produced {available} forecast steps for a horizon of {horizon}. "
                f"With n_lags > 0 it emits exactly n_forecasts direct steps and does not recurse, "
                f"so model_params.neuralprophet.n_forecasts must be at least {horizon}."
            )
        if heads <= 1:
            block = future.head(horizon)
            return (
                block["yhat1"].to_numpy(dtype=float),
                block[self._band_col(1, _BAND[0])].to_numpy(dtype=float),
                block[self._band_col(1, _BAND[1])].to_numpy(dtype=float),
            )
        rows = [future.iloc[i] for i in range(horizon)]
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
        """A device earns its cost here only under autoregression — ``n_lags > 0``.

        Without it the network is a few hundred trend and Fourier parameters and the Lightning
        loop, the dataloader and pandas dwarf the kernels; that is the shape Phase 0 measured at
        50–78 KB of device memory and 95% CPU-bound. AR-Net is what makes the model big enough for
        the card to matter. ``n_forecasts`` alone does not qualify: extra heads without lags are
        extra output units on the same tiny network.
        """
        return int(params.get("n_lags", 0) or 0) > 0

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        """Autoregression needs one direct head per step of the horizon.

        With ``n_lags > 0`` NeuralProphet builds ``n_forecasts`` direct heads and forecasts exactly
        that far — it does **not** recurse to fill a longer request. Measured: ``n_lags=7`` with
        the default ``n_forecasts=1``, asked for seven periods, returns one value. Left unchecked
        the horizon comes back short (or, with the diagonal output shape, mostly NaN) after the
        fleet has already been paid for, which is why this is refused at plan time.
        """
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
