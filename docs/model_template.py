"""TEMPLATE — copy me into src/scale_forecasting/models/ to add a model.

This is not a registered model (it lives under docs/, which the factory never imports). To
add a real model:

  1. Copy this file to ``src/scale_forecasting/models/<your_model>.py``.
  2. Rename the class and set ``name`` to a unique, lowercase, snake_case string.
  3. Fill in ``fit`` and ``predict`` (delete the parts you don't need).
  4. Add one import line to ``src/scale_forecasting/models/__init__.py`` so it registers.

That's the whole checklist — no other file changes. The contract test
(``tests/unit/test_models_contract.py``) then covers your model automatically, and it shows
up in the playground/CLI (``python -m scale_forecasting.playground --list``).

See ``docs/adding_a_model.md`` for the full walkthrough and the contract every model owes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from ..seasonality import seasonal_period
from .base_model import DEFAULT_QUANTILES, BaseModel, register


class TemplateModel(BaseModel):
    """A one-line description of your model (shown nowhere, but read by the next person)."""

    # --- registration metadata (read by the factory) --------------------------
    name = "template"  # unique, lowercase, snake_case — this is how users select it
    runtime = "python"  # "python" runs in a Spark/Ray cell; "bigquery" is SQL
    family = "statistical"  # "statistical" | "ml" | "deep_learning" | "native"
    supports_exog = False  # True if fit/predict use the X (exogenous) frame
    supports_native_intervals = False  # True if you produce your own prediction bounds

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        """Fit on one series.

        ``y`` is the target indexed by ``ds`` (datetime64), already transformed per config
        (e.g. log1p). ``X`` is the aligned exogenous/feature frame, or None. Stash whatever
        ``predict`` needs on ``self``. Raise :class:`ModelError` on a genuinely unfittable
        series — the worker turns that into an error cell, it never sinks the batch.

        ``self.ctx`` carries per-run context: ``ctx.freq``, ``ctx.horizon``, ``ctx.seed``,
        ``ctx.transform``. ``self.params`` holds any HPO-selected hyperparameters.
        """
        if len(y) < 2:
            raise ModelError(f"{self.name} requires at least 2 observations")

        # Example: a period-aware statistical fit. seasonal_period() is the one shared
        # freq→period source (7 daily, 12 monthly, …) — never hard-code 7.
        self._period = seasonal_period(self.ctx.freq)
        self._last_date = y.index[-1]
        self._level = float(y.iloc[-1])  # (placeholder "model": last-value carry-forward)

        # Models WITHOUT native intervals record residuals here; the base class turns them
        # into prediction bounds in predict() via residual_intervals(). Delete if you emit
        # your own bounds (and set supports_native_intervals = True).
        fitted = np.full(len(y), self._level, dtype=float)
        self._set_residuals(y.to_numpy() - fitted)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Return the canonical prediction frame in ORIGINAL units.

        Build the point forecast, turn it into a ``{quantile: array}`` map, invert the
        transform so values are in original units, and hand it to ``_assemble_frame`` which
        produces the ``ds, yhat, yhat_lower, yhat_upper, quantiles`` frame with ordered
        bounds. ``_future_index`` gives the horizon dates at the run frequency.
        """
        mean = np.full(horizon, self._level, dtype=float)

        # Residual-based intervals (for supports_native_intervals = False). If you produce
        # native bounds, build the quantile map directly instead of calling this.
        qmap_transformed = self.residual_intervals(mean, quantiles)

        # Invert the target transform so the frame is in original units. Pass both the
        # transform name and the cell's fitted λ (None for none/log1p; set for boxcox) — the
        # worker fits λ once per cell and hands it to you on ctx, so predict never refits it.
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_transformed.items()}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)


# The one line that registers the model with the factory. Without it, the model is invisible.
register(TemplateModel)
