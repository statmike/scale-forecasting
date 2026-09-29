"""FFT — Fast Fourier Transform harmonic extrapolation (scipy).

One model, one file. Runtime python, statistical family. Detrends the series
with a low-degree polynomial (linear by default), transforms the detrended
signal into the frequency domain via ``scipy.fft.rfft``, keeps the top
``nr_freqs_to_keep`` dominant harmonic amplitudes, and evaluates their cosine
synthesis at future time indices alongside the extrapolated polynomial trend.
Uses empirical residual quantiles for prediction intervals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    import optuna


class FftModel(BaseModel):
    """Fast Fourier Transform harmonic extrapolation forecaster (scipy)."""

    name = "fft"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = False
    supports_recondition = False
    supports_extrapolate = True
    package = "scipy"
    package_url = "https://scipy.org/"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        from scipy.fft import rfft, rfftfreq

        n = len(y)
        if n < 4:
            raise ModelError("fft requires at least 4 observations")

        degree = int(self.params.get("trend_poly_degree", 1))
        nr_freqs = int(self.params.get("nr_freqs_to_keep", 10))
        if degree < 0 or degree >= n:
            degree = min(max(degree, 0), n - 1)

        y_arr = y.to_numpy(dtype=float)
        t_idx = np.arange(n, dtype=float)
        self._n = n
        self._last_date = y.index[-1]
        self._poly_coeffs = np.polyfit(t_idx, y_arr, deg=degree)
        trend_in = np.polyval(self._poly_coeffs, t_idx)
        detrended = y_arr - trend_in

        spectrum = rfft(detrended)
        freqs = rfftfreq(n, d=1.0)
        magnitudes = np.abs(spectrum)
        k = min(max(nr_freqs, 1), len(spectrum))
        top_idx = np.argsort(magnitudes)[::-1][:k]

        self._freqs = freqs[top_idx]
        self._coeffs = spectrum[top_idx]
        self._nyquist_even = n % 2 == 0

        fitted = trend_in + self._synthesize(t_idx)
        self._set_residuals(y_arr - fitted)

    def _synthesize(self, t_idx: np.ndarray) -> np.ndarray:
        """Evaluate the retained Fourier components at arbitrary time indices ``t_idx``."""
        signal = np.zeros(len(t_idx), dtype=float)
        n = float(self._n)
        for freq, coeff in zip(self._freqs, self._coeffs, strict=False):
            if freq == 0.0 or (self._nyquist_even and np.isclose(freq, 0.5)):
                weight = 1.0 / n
            else:
                weight = 2.0 / n
            angle = 2.0 * np.pi * freq * t_idx
            signal += weight * (coeff.real * np.cos(angle) - coeff.imag * np.sin(angle))
        return signal

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        steps = self._forecast_steps(horizon)
        t_future = np.arange(self._n, self._n + steps, dtype=float)[-horizon:]
        mean = np.polyval(self._poly_coeffs, t_future) + self._synthesize(t_future)
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        ds = self._forecast_index(horizon)
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "nr_freqs_to_keep": trial.suggest_int("nr_freqs_to_keep", 3, 25),
            "trend_poly_degree": trial.suggest_int("trend_poly_degree", 0, 2),
        }


register(FftModel)
