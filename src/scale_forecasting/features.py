"""Feature engineering for the Python models — pure.

Everything here is declarative-from-config and runs *inside* the execution node, so no
data crosses the network. Two entry points:

- ``build_features(series, cfg) -> (y, X)`` — turn one raw series frame into the fit
  inputs: ``y`` (target Series indexed by ds, transform applied) and ``X`` (exog + holiday
  + Fourier + level-shift + lag columns aligned to ``y``, or None when no feature is
  configured).
- ``build_future_features(y, X, cfg) -> DataFrame | None`` — the *forecast-horizon* design
  frame, indexed by the future dates, with the same columns in the same order.
- ``holiday_frame(cfg) -> DataFrame`` — the one canonical holiday calendar (``ds``,
  ``holiday`` columns) computed from ``features.holidays``, fed to *both* Python models and
  the BQML custom-holiday input so "holiday" is identical everywhere.

**Two frames, because a forecast row is not a history row.** Most of what we build is a
deterministic function of the *date* — holiday flags, Fourier phase, the level-shift step —
so the horizon's true values are computable, not guessable, and computing them is the only
way an exog-aware model sees the right seasonal phase for the dates it is forecasting. Only
user-supplied ``features.exog`` is genuinely unknown ahead of time; that alone falls back to
a documented stand-in. Backtesting never needed this (its "future" is in-sample, so
`backtest_cell` slices the real ``X``) — which is exactly why the gap was easy to miss: the
folds scored on correct features while the shipped forecast did not.

Transforms come in two flavors. ``none``/``log1p`` are **stateless** — a model inverts with
only ``ctx.transform`` (a name). ``boxcox`` is **stateful**: its λ is fit per series (MLE),
so it must travel with the cell. We fit λ once in the worker (`fit_transform_lambda`),
put it on ``ctx.transform_lambda``, and every ``apply``/``invert`` call passes that same λ —
never refit at predict, so the backtest folds and the final fit use one λ. ``λ = None``
for the stateless transforms.

Public surface: ``build_features``, ``build_future_features``, ``holiday_frame``,
``apply_transform``, ``invert_transform``, ``fit_transform_lambda``, ``level_shift_step``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .errors import ConfigError
from .seasonality import periods_per_year

if TYPE_CHECKING:
    from .config import RunConfig


# --- transforms ----------------------------------------------------------------


def fit_transform_lambda(y: pd.Series, transform: str) -> float | None:
    """Fit the transform's stateful parameter on ``y`` (the full series' target).

    Only ``boxcox`` has one: its λ, fit by maximum likelihood (``scipy.stats.boxcox_normmax``)
    on the raw target. Returns ``None`` for the stateless transforms (``none``/``log1p``), whose
    inverse needs no fitted state. Box-Cox is defined only for **strictly positive** data, so a
    non-positive value raises ``ConfigError`` naming the constraint (symmetric with log1p's
    ``y >= -1`` rule — we *validate*, never silently shift). Deterministic: same series → same λ.
    """
    if transform != "boxcox":
        return None
    from scipy.stats import boxcox_normmax

    arr = y.to_numpy(dtype=float)
    if not np.all(arr > 0):
        raise ConfigError(
            "boxcox transform requires strictly positive y (got values <= 0); "
            "use 'log1p' for non-negative data with zeros"
        )
    return float(boxcox_normmax(arr, method="mle"))


def apply_transform(y: pd.Series, transform: str, lam: float | None = None) -> pd.Series:
    """Apply the configured target transform to ``y`` (forward direction).

    ``lam`` is the fitted Box-Cox λ from `fit_transform_lambda` (ignored by the stateless
    transforms). Required when ``transform == "boxcox"``.
    """
    if transform == "none":
        return y
    if transform == "log1p":
        if (y < -1).any():
            raise ConfigError("log1p transform requires y >= -1 (got values < -1)")
        return pd.Series(np.log1p(y.to_numpy()), index=y.index, name=y.name)
    if transform == "boxcox":
        if lam is None:
            raise ConfigError("boxcox requires a fitted lambda (call fit_transform_lambda first)")
        from scipy.special import boxcox as boxcox_apply

        arr = y.to_numpy(dtype=float)
        if not np.all(arr > 0):
            raise ConfigError("boxcox transform requires strictly positive y (got values <= 0)")
        return pd.Series(boxcox_apply(arr, lam), index=y.index, name=y.name)
    raise ConfigError(f"unknown transform '{transform}'")


def invert_transform(values: np.ndarray, transform: str, lam: float | None = None) -> np.ndarray:
    """Invert a transform on forecasts/bounds so the frame is in original units.

    ``lam`` is the same fitted Box-Cox λ used in the forward direction (ignored by the stateless
    transforms), so ``invert(apply(y)) == y``.
    """
    arr = np.asarray(values, dtype=float)
    if transform == "none":
        return arr
    if transform == "log1p":
        return np.expm1(arr)
    if transform == "boxcox":
        if lam is None:
            raise ConfigError("boxcox requires a fitted lambda (call fit_transform_lambda first)")
        from scipy.special import inv_boxcox

        return np.asarray(inv_boxcox(arr, lam), dtype=float)
    raise ConfigError(f"unknown transform '{transform}'")


# --- holidays ------------------------------------------------------------------


def holiday_frame(cfg: RunConfig) -> pd.DataFrame:
    """Canonical holiday calendar for the run.

    ``features.holidays`` lists ISO country codes (e.g. ``["US"]``). Returns a frame with
    ``ds`` (datetime64[ns]) and ``holiday`` (name) columns, empty when none configured.
    A generous fixed year window keeps the calendar deterministic; callers filter by ds.
    """
    codes = cfg.features.holidays
    if not codes:
        return pd.DataFrame({"ds": pd.to_datetime([]), "holiday": pd.array([], dtype="string")})

    import holidays as holidays_pkg

    years = range(2015, 2036)  # generous, deterministic window
    records: list[tuple[pd.Timestamp, str]] = []
    for code in codes:
        try:
            cal = holidays_pkg.country_holidays(code, years=years)
        except NotImplementedError as e:
            raise ConfigError(f"unknown holiday country code '{code}'") from e
        for day, name in cal.items():
            records.append((pd.Timestamp(day), str(name)))

    records.sort(key=lambda r: r[0])
    ds = pd.DatetimeIndex([r[0] for r in records]).as_unit("ns")
    names = pd.array([r[1] for r in records], dtype="string")
    return pd.DataFrame({"ds": ds, "holiday": names})


# --- level shift ---------------------------------------------------------------

# A candidate split must leave at least this many observations on each side, so the
# statistic is never dominated by a two-point "segment" at the edge of the history.
_MIN_SEGMENT = 8
# Accept a changepoint only when the standardized mean gap clears this many sigma. 3.0 is
# the usual "clearly not noise" bar; below it we emit an all-zero column rather than hand a
# model a spurious regressor, which is the failure that matters at fleet scale (a false
# positive on one series in ten thousand is a forecast nobody reviews).
_LEVEL_SHIFT_SIGMA = 3.0


def level_shift_step(y: pd.Series) -> np.ndarray:
    """Detect one abrupt level shift in ``y`` and return it as a 0/1 step dummy.

    The shipped example data contains exactly this pattern — `data_gen.generator` gives every
    archetype a ``level_shift_prob`` and applies a single additive jump partway through the
    history — and it is the common real shape too (a re-baselined product, a store
    reopening, a units→cases change). A model that cannot see it fits the *average* of two
    regimes and is biased for the whole horizon.

    The encoding is a **step**, not a spike: 0 before the changepoint, 1 from it onward, and
    (in `build_future_features`) 1 across the entire horizon, because a level shift persists
    — that is what distinguishes it from an outlier. A regression model then learns the jump
    as one coefficient and carries it forward.

    Detection is a single-changepoint binary segmentation, computed in one vectorized pass
    (O(n) via cumulative sums) because this runs once per series across millions of them: for
    every admissible split, standardize the gap between segment means by a robust noise
    estimate and take the argmax. Returns all zeros when nothing clears `_LEVEL_SHIFT_SIGMA`
    or the series is too short to split — never a spurious constant column.
    """
    values = y.to_numpy(dtype=float)
    n = values.size
    if n < 2 * _MIN_SEGMENT:
        return np.zeros(n)

    # Noise scale from first differences (MAD-based, so a genuine jump — one large diff —
    # cannot inflate the very yardstick used to judge it). sigma_diff = sqrt(2)*sigma_level.
    diffs = np.diff(values)
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    sigma = mad * 1.4826 / np.sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.zeros(n)

    # Segment means for every split k (left = [0,k), right = [k,n)) straight off the cumsum.
    total = float(values.sum())
    cumulative = np.cumsum(values)
    k = np.arange(_MIN_SEGMENT, n - _MIN_SEGMENT + 1)
    left_mean = cumulative[k - 1] / k
    right_mean = (total - cumulative[k - 1]) / (n - k)
    # Standard two-sample scaling: a gap seen over more observations is stronger evidence.
    statistic = np.abs(left_mean - right_mean) * np.sqrt(k * (n - k) / n) / sigma

    best = int(np.argmax(statistic))
    if statistic[best] < _LEVEL_SHIFT_SIGMA:
        return np.zeros(n)
    step = np.zeros(n)
    step[k[best] :] = 1.0
    return step


# --- feature frames --------------------------------------------------------------


def build_features(
    series: pd.DataFrame, cfg: RunConfig, lam: float | None = None
) -> tuple[pd.Series, pd.DataFrame | None]:
    """Build ``(y, X)`` fit inputs for one series.

    ``series`` is one ts_id's rows with the configured date/target (and optional exog)
    columns. ``y`` is returned indexed by ds, sorted, with the transform applied. ``X``
    carries any configured exog, an ``is_holiday`` flag, Fourier terms, and lag columns —
    aligned to ``y`` — or None when nothing is configured.

    ``lam`` is the fitted Box-Cox λ (from `fit_transform_lambda`), threaded in so the
    forward transform matches the inverse a model applies at predict; ``None`` (the default)
    for the stateless transforms.
    """
    d, f = cfg.data, cfg.features
    if d.date_col not in series or d.target_col not in series:
        raise ConfigError(
            f"series missing required columns '{d.date_col}'/'{d.target_col}'; "
            f"has {list(series.columns)}"
        )

    frame = series.copy()
    frame[d.date_col] = pd.to_datetime(frame[d.date_col]).astype("datetime64[ns]")
    frame = frame.sort_values(d.date_col).set_index(d.date_col)
    frame.index.name = "ds"

    y = frame[d.target_col].astype(float)
    y = apply_transform(y, f.transform, lam)
    y.name = "y"

    cols: dict[str, np.ndarray] = {}

    # Exogenous regressors passed straight through (must exist in the series).
    for name in f.exog:
        if name not in frame:
            raise ConfigError(f"exog column '{name}' not found in series")
        cols[name] = frame[name].astype(float).to_numpy()

    # Holiday flag from the canonical calendar (parity with BQML).
    if f.holidays:
        hol = holiday_frame(cfg)
        holiday_days = set(hol["ds"].to_numpy())
        cols["is_holiday"] = np.isin(frame.index.to_numpy(), list(holiday_days)).astype(float)

    # Fourier seasonality terms (yearly), for ML models.
    if f.fourier:
        cols.update(_fourier_terms(pd.DatetimeIndex(frame.index), d.freq, order=3))

    # Step dummy for one detected regime change (see `level_shift_step`).
    if f.level_shift:
        cols["level_shift"] = level_shift_step(y)

    # Lag features from the (transformed) target.
    for lag in f.lags:
        if lag <= 0:
            raise ConfigError(f"lags must be positive, got {lag}")
        cols[f"lag_{lag}"] = y.shift(lag).to_numpy()

    if not cols:
        return y, None
    X = pd.DataFrame(cols, index=frame.index)
    return y, X


def build_future_features(
    y: pd.Series, X: pd.DataFrame | None, cfg: RunConfig
) -> pd.DataFrame | None:
    """The design frame for the forecast horizon, indexed by the *future* dates.

    Takes the training pair from `build_features` and returns the ``horizon``-row frame a
    model should be handed at predict time, with **the same columns in the same order** (it
    is built from ``X.columns``, so parity holds by construction rather than by convention —
    `_lag_forecaster.recursive_predict` reads exog positionally and a reordered frame would
    silently feed the wrong column to the wrong coefficient).

    Column by column:

    - ``is_holiday``, ``fourier_*`` — recomputed at the real future dates. Deterministic
      functions of the date, so these are *exact*, not estimated.
    - ``level_shift`` — 1.0 throughout: a detected regime change is still in force over the
      horizon. (0.0 throughout when none was detected, matching the historical column.)
    - ``lag_*`` — the configured lags, read off the history extended by a naive persistence
      forecast. For step ``i <= lag`` the value is a genuine observation; beyond that it is
      the last observed level. (Tree models discard these and own their own recursion; see
      `_lag_forecaster._true_exog`.)
    - anything else — user-supplied ``features.exog``, which is genuinely unknown until the
      real future arrives. Falls back to the **most recent** ``horizon`` observed rows, so an
      exog-driven forecast stays indicative-only but at least reflects the current regime.
      Supply real forward exog by extending the source table past the cutoff.
    """
    if X is None:
        return None
    d, f = cfg.data, cfg.features
    horizon = d.horizon
    future = pd.date_range(start=y.index[-1], periods=horizon + 1, freq=d.freq)[1:].as_unit("ns")

    known: dict[str, np.ndarray] = {}
    if f.holidays:
        hol = holiday_frame(cfg)
        holiday_days = set(hol["ds"].to_numpy())
        known["is_holiday"] = np.isin(future.to_numpy(), list(holiday_days)).astype(float)
    if f.fourier:
        known.update(_fourier_terms(future, d.freq, order=3))
    if f.level_shift:
        known["level_shift"] = np.full(horizon, float(X["level_shift"].to_numpy()[-1]))
    if f.lags:
        # Persistence-extended history: lag_k is observed for the first k steps, then holds.
        extended = np.concatenate([y.to_numpy(dtype=float), np.full(horizon, float(y.iloc[-1]))])
        steps = np.arange(len(y), len(y) + horizon)
        for lag in f.lags:
            known[f"lag_{lag}"] = extended[np.clip(steps - lag, 0, len(extended) - 1)]

    # Recency stand-in for the columns we cannot know: the last `horizon` observed rows.
    # Clipped, so a history shorter than the horizon repeats its earliest row instead of
    # returning a frame that silently disagrees in length with the future index.
    tail = X.iloc[np.clip(np.arange(len(X) - horizon, len(X)), 0, len(X) - 1)]
    cols = {
        name: known[name] if name in known else tail[name].to_numpy(dtype=float)
        for name in X.columns
    }
    return pd.DataFrame(cols, index=future)


def _fourier_terms(index: pd.DatetimeIndex, freq: str, order: int) -> dict[str, np.ndarray]:
    """Sine/cosine Fourier features for a yearly seasonal period (pure)."""
    period = periods_per_year(freq)
    # Position within the seasonal cycle, from the day count since epoch.
    nanos = index.to_numpy(dtype="datetime64[ns]").astype("int64")
    t = nanos.astype(float) / (24 * 3600 * 1e9)
    out: dict[str, np.ndarray] = {}
    for k in range(1, order + 1):
        ang = 2.0 * np.pi * k * t / period
        out[f"fourier_sin_{k}"] = np.sin(ang)
        out[f"fourier_cos_{k}"] = np.cos(ang)
    return out
