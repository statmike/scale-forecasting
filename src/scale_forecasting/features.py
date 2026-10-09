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

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .errors import ConfigError
from .seasonality import periods_per_year, seasonal_period

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def extract_static_covariates(series: pd.DataFrame, cfg: RunConfig) -> dict[str, object]:
    """Validate and extract series-constant static covariates for one ``ts_id`` (pure).

    Static covariates (``features.static_covariates``, e.g. ``region``, ``category``, ``archetype``)
    carry series-level metadata for global/hybrid models and hierarchical grouping. Because they are
    constant within a single series (and may be categorical strings), they are deliberately omitted
    from the single-series local design matrix ``X`` — where a constant column is collinear with the
    model intercept — and surfaced here as a clean ``{col: value}`` mapping.
    """
    static_cols = cfg.features.static_covariates
    if not static_cols:
        return {}
    if series.empty:
        raise ConfigError("cannot extract static covariates from an empty series")
    out: dict[str, object] = {}
    for name in static_cols:
        if name not in series.columns:
            raise ConfigError(f"static covariate column '{name}' not found in series")
        col = series[name]
        if col.isna().any():
            raise ConfigError(f"static covariate column '{name}' has missing (null) values")
        if col.nunique(dropna=False) > 1:
            raise ConfigError(
                f"static covariate column '{name}' must be constant within a series; "
                f"got multiple values: {col.unique().tolist()[:5]}"
            )
        val = col.iloc[0]
        out[name] = val.item() if hasattr(val, "item") else val
    return out


def effective_config_for_model(cfg: RunConfig, model_cls: type[Any]) -> RunConfig:
    """Return a `RunConfig` narrowed to the covariate tiers ``model_cls`` actually supports (pure).

    Used inside the worker and HPO so that in a multi-model benchmark under
    ``on_unsupported_covariates="fallback"`` (the default), a univariate model or a model that
    supports only dynamic covariates never has its target truncated by ``exog_lags`` of an
    unsupported covariate tier or receives columns it cannot consume. The run identity (`run_id`
    and `model_hash`) is always computed from the original ``cfg`` before calling this helper.

    When ``on_unsupported_covariates="error"`` and ``model_cls`` lacks support for any configured
    tier, raises `ConfigError` naming the model and the unsupported tier(s).
    """
    f = cfg.features
    unsupported = model_cls.unsupported_covariate_tiers(f)
    if not unsupported:
        return cfg
    if f.on_unsupported_covariates == "error":
        raise ConfigError(
            f"model '{model_cls.name}' does not support configured covariate tier(s) "
            f"{list(unsupported)} (supports_future_covariates="
            f"{model_cls.supports_future_covariates}, "
            f"supports_past_covariates={model_cls.supports_past_covariates}, "
            f"supports_static_covariates={model_cls.supports_static_covariates}). "
            f"Remove '{model_cls.name}' or the unsupported covariate tier(s), or set "
            f"features.on_unsupported_covariates='fallback' to let '{model_cls.name}' fall back "
            f"to the tiers it supports."
        )
    new_exog = list(f.exog) if model_cls.supports_future_covariates else []
    new_future = list(f.future_covariates) if model_cls.supports_future_covariates else []
    new_past = list(f.past_covariates) if model_cls.supports_past_covariates else []
    new_static = list(f.static_covariates) if model_cls.supports_static_covariates else []
    kept_dynamic = set(new_exog) | set(new_future) | set(new_past)
    new_lags = {k: list(v) for k, v in f.exog_lags.items() if k in kept_dynamic}
    new_features = f.model_copy(
        update={
            "exog": new_exog,
            "future_covariates": new_future,
            "past_covariates": new_past,
            "static_covariates": new_static,
            "exog_lags": new_lags,
        }
    )
    return cfg.model_copy(update={"features": new_features})


def _with_inferred_freq(index: pd.Index) -> pd.Index:
    """``index`` with its frequency set, when the dates are regular enough to carry one.

    A series that passed pre-flight is regular by contract, but ``set_index`` does not know that,
    and a ``DatetimeIndex`` without a ``freq`` makes every statsmodels model *re-derive* it at fit
    time — correctly, and with a ``ValueWarning`` whose text is the caller's site-packages path.
    Declaring what the index already is keeps the arithmetic identical (statsmodels used the same
    inferred value) and keeps a warning that is not about the user's data out of their notebook.

    Irregular dates, duplicates, or fewer than the three points inference needs leave the index as
    it was: statsmodels then falls back exactly as it does today, and the warning it raises there
    is a true statement about that series.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.freq is not None or len(index) < 3:
        return index
    try:
        inferred = pd.infer_freq(index)
    except (TypeError, ValueError):
        return index
    if inferred is None:
        return index
    with_freq = pd.DatetimeIndex(index, freq=inferred)
    with_freq.name = index.name
    return with_freq


# --- SBC demand regime profiling & routing -------------------------------------

_INTERMITTENT_MODELS: frozenset[str] = frozenset(
    {
        "croston",
        "naive_mean",
        "naive_seasonal",
        "naive_moving_average",
        "lightgbm",
        "xgboost",
        "catboost",
        "random_forest",
    }
)
_COLD_START_MODELS: frozenset[str] = frozenset(
    {
        "naive_mean",
        "naive_drift",
        "naive_moving_average",
        "theta",
        "croston",
    }
)
_PURE_INTERMITTENT_MODELS: frozenset[str] = frozenset({"croston"})


def compute_demand_profile(
    y: pd.Series | np.ndarray,
    *,
    min_cold_start_obs: int = 14,
) -> dict[str, Any]:
    """Syntetos–Boylan–Croston (2005) demand regime classification (pure).

    Computes the Average Demand Interval (``adi = n_obs / n_nonzero``), the squared coefficient of
    variation of non-zero demand (``cv2 = (std / mean)^2``), ``zero_share``, and ``n_obs``, and
    assigns one of five canonical regimes:

    - ``"cold_start"``: ``n_obs < min_cold_start_obs`` or fewer than 3 non-zero observations
    - ``"smooth"``: ``adi < 1.32`` and ``cv2 < 0.49``
    - ``"erratic"``: ``adi < 1.32`` and ``cv2 >= 0.49``
    - ``"intermittent"``: ``adi >= 1.32`` and ``cv2 < 0.49``
    - ``"lumpy"``: ``adi >= 1.32`` and ``cv2 >= 0.49``
    """
    arr = np.asarray(y, dtype=float)
    finite = arr[np.isfinite(arr)]
    n_obs = int(finite.size)
    if n_obs == 0:
        return {
            "demand_class": "cold_start",
            "adi": None,
            "cv2": None,
            "zero_share": 1.0,
            "n_obs": 0,
        }

    nz = finite[finite > 0.0] if np.any(finite > 0.0) else finite[finite != 0.0]
    n_nz = int(nz.size)
    zero_share = float(np.mean(finite == 0.0))
    adi = float(n_obs / n_nz) if n_nz > 0 else float("inf")
    if n_nz >= 2 and float(np.mean(nz)) != 0.0:
        cv2 = float((np.std(nz, ddof=1) / abs(float(np.mean(nz)))) ** 2)
    else:
        cv2 = 0.0

    if n_obs < min_cold_start_obs or n_nz < 3:
        demand_class = "cold_start"
    elif adi < 1.32 and cv2 < 0.49:
        demand_class = "smooth"
    elif adi < 1.32 and cv2 >= 0.49:
        demand_class = "erratic"
    elif adi >= 1.32 and cv2 < 0.49:
        demand_class = "intermittent"
    else:
        demand_class = "lumpy"

    return {
        "demand_class": demand_class,
        "adi": round(adi, 4) if np.isfinite(adi) else None,
        "cv2": round(cv2, 4) if np.isfinite(cv2) else None,
        "zero_share": round(zero_share, 4),
        "n_obs": n_obs,
    }


def regime_compatible_models(demand_class: str, models: Sequence[str]) -> list[str]:
    """Filter ``models`` to those suited for ``demand_class``, never returning an empty list."""
    model_list = list(models)
    if not model_list:
        return []
    if demand_class in ("intermittent", "lumpy"):
        matched = [m for m in model_list if m in _INTERMITTENT_MODELS]
    elif demand_class == "cold_start":
        matched = [m for m in model_list if m in _COLD_START_MODELS]
    elif demand_class == "smooth":
        non_intermittent = [m for m in model_list if m not in _PURE_INTERMITTENT_MODELS]
        matched = non_intermittent if non_intermittent else model_list
    else:
        matched = model_list
    return matched if matched else model_list


def select_models_for_series(series: pd.DataFrame, cfg: RunConfig) -> list[str]:
    """Return the active model list for ``series`` under ``cfg.features.regime_routing``."""
    if not cfg.features.regime_routing:
        return list(cfg.models)
    reg_df = regularize_series_grid(series, cfg)
    tcol = cfg.data.target_col
    y_vals = (
        pd.to_numeric(reg_df[tcol], errors="coerce").to_numpy(dtype=float)
        if tcol in reg_df.columns
        else np.array([], dtype=float)
    )
    prof = compute_demand_profile(y_vals)
    return regime_compatible_models(str(prof["demand_class"]), cfg.models)


# --- grid regularization & fold-safe target cleaning ---------------------------


def regularize_series_grid(series: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Regularize missing calendar timestamps (`grid_policy`) and trim pre-launch leading zeros
    (`trim_leading_zeros`) for one series (pure).
    """
    d, f = cfg.data, cfg.features
    if d.grid_policy == "strict" and not d.trim_leading_zeros:
        return series
    if series.empty or d.date_col not in series.columns or d.target_col not in series.columns:
        return series

    frame = series.copy()
    frame[d.date_col] = pd.to_datetime(frame[d.date_col]).astype("datetime64[ns]")
    frame = frame.sort_values(d.date_col).drop_duplicates(subset=[d.date_col], keep="last")

    if d.grid_policy != "strict" and len(frame) >= 2:
        full_idx = pd.date_range(
            start=frame[d.date_col].iloc[0],
            end=frame[d.date_col].iloc[-1],
            freq=d.freq,
        ).as_unit("ns")
        indexed = frame.set_index(d.date_col).reindex(full_idx)
        indexed.index.name = d.date_col

        # Carry series identifier and static covariates across newly inserted calendar dates.
        for scol in [d.ts_id_col, *f.static_covariates]:
            if scol in indexed.columns:
                indexed[scol] = indexed[scol].ffill().bfill()

        # Fill target column according to grid_policy.
        t_series = pd.to_numeric(indexed[d.target_col], errors="coerce")
        if d.grid_policy in ("fill_zero", "zero_fill"):
            indexed[d.target_col] = t_series.fillna(0.0)
        elif d.grid_policy == "ffill":
            indexed[d.target_col] = t_series.ffill().bfill().fillna(0.0)
        elif d.grid_policy == "linear":
            indexed[d.target_col] = (
                t_series.interpolate(method="linear").ffill().bfill().fillna(0.0)
            )

        # Fill dynamic covariates and censored_col on inserted calendar rows.
        for ccol in f.dynamic_covariates:
            if ccol in indexed.columns:
                indexed[ccol] = (
                    pd.to_numeric(indexed[ccol], errors="coerce").ffill().bfill().fillna(0.0)
                )
        if f.censored_col and f.censored_col in indexed.columns:
            indexed[f.censored_col] = pd.to_numeric(
                indexed[f.censored_col], errors="coerce"
            ).fillna(0.0)
        frame = indexed.reset_index()

    if d.trim_leading_zeros and len(frame) > 1:
        y_num = (
            pd.to_numeric(frame[d.target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        nonzero_idx = np.flatnonzero(np.abs(y_num) > 0.0)
        if nonzero_idx.size > 0 and int(nonzero_idx[0]) > 0:
            frame = frame.iloc[int(nonzero_idx[0]) :].reset_index(drop=True)

    return frame


def _local_seasonal_baseline(arr: np.ndarray, period: int) -> np.ndarray:
    """Fast, fold-safe local trend + seasonal baseline for outlier and stockout imputation."""
    n = arr.size
    if n < 4:
        return np.full(n, float(np.nanmedian(arr)) if n else 0.0, dtype=float)
    win = max(5, min(n // 2, max(period, 7)))
    if win % 2 == 0:
        win += 1
    trend = (
        pd.Series(arr)
        .rolling(window=win, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    if period >= 2 and n >= 2 * period:
        detrended = arr - trend
        slots = np.arange(n) % period
        seas = np.zeros(n, dtype=float)
        for s in range(period):
            mask = slots == s
            vals = detrended[mask]
            finite = vals[np.isfinite(vals)]
            if finite.size:
                seas[mask] = float(np.median(finite))
        return trend + seas
    return trend


def clean_training_target(
    raw_y: pd.Series,
    frame: pd.DataFrame,
    cfg: RunConfig,
) -> tuple[pd.Series, np.ndarray | None]:
    """Apply fold-safe censored demand recovery (`censored_col`) and outlier treatment
    (`outlier_policy`) strictly on the provided training window (pure).

    Returns ``(cleaned_y, outlier_flag_array_or_None)``.
    """
    d, f = cfg.data, cfg.features
    if not f.censored_col and f.outlier_policy == "none":
        return raw_y, None

    arr = raw_y.to_numpy(dtype=float).copy()
    n = arr.size
    period = seasonal_period(d.freq)

    # 1. Censored demand (stockout/outage) imputation strictly from uncensored training history.
    if f.censored_col:
        if f.censored_col not in frame.columns:
            raise ConfigError(
                f"censored_col '{f.censored_col}' not found in series; has {list(frame.columns)}"
            )
        cens_mask = (
            pd.to_numeric(frame[f.censored_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            > 0.0
        )
        if cens_mask.any() and (~cens_mask).any():
            uncensored_arr = arr.copy()
            uncensored_arr[cens_mask] = np.nan
            # Interpolate uncensored series + seasonal slot medians from uncensored observations.
            interp_uncens = (
                pd.Series(uncensored_arr)
                .interpolate(method="linear")
                .ffill()
                .bfill()
                .to_numpy(dtype=float)
            )
            base_uncens = _local_seasonal_baseline(interp_uncens, period)
            # Censoring (stockout) caps observed sales below true demand, so true >= observed.
            arr[cens_mask] = np.maximum(arr[cens_mask], base_uncens[cens_mask])

    # 2. Fold-safe outlier despiking or flagging.
    outlier_flag: np.ndarray | None = None
    if f.outlier_policy != "none" and n >= 6:
        baseline = _local_seasonal_baseline(arr, period)
        resid = arr - baseline
        med = float(np.nanmedian(resid))
        mad = float(np.nanmedian(np.abs(resid - med)))
        sigma = mad * 1.4826
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = float(np.nanstd(resid))
        if np.isfinite(sigma) and sigma > 1e-12:
            thresh = float(f.outlier_sigma) * sigma
            is_out = np.abs(resid - med) > thresh
            if f.outlier_policy in ("winsorize", "stl_resid"):
                arr = np.clip(arr, baseline + med - thresh, baseline + med + thresh)
            elif f.outlier_policy == "interpolate":
                if is_out.any():
                    arr[is_out] = baseline[is_out] + med
            elif f.outlier_policy == "flag":
                outlier_flag = is_out.astype(float)
        elif f.outlier_policy == "flag":
            outlier_flag = np.zeros(n, dtype=float)
    elif f.outlier_policy == "flag":
        outlier_flag = np.zeros(n, dtype=float)

    return pd.Series(arr, index=raw_y.index, name=raw_y.name), outlier_flag


def _trading_day_terms(index: pd.DatetimeIndex, freq: str) -> dict[str, np.ndarray]:
    """Deterministic X-13 style trading-day calendar features from ``index`` and ``freq`` (pure).

    Emits ``days_in_period``, ``n_weekdays`` (Mon–Fri count), and ``n_weekend_days`` (Sat–Sun count)
    so monthly/quarterly models capture calendar-length and weekend-composition shifts while daily
    and weekly models see exact weekday/weekend indicators.
    """
    n = len(index)
    if n == 0:
        empty = np.array([], dtype=float)
        return {"days_in_period": empty, "n_weekdays": empty, "n_weekend_days": empty}

    freq_upper = str(freq).upper()
    if freq_upper.startswith(("M", "Q", "Y", "A")):
        days_arr = np.zeros(n, dtype=float)
        wk_arr = np.zeros(n, dtype=float)
        we_arr = np.zeros(n, dtype=float)
        for i, ts in enumerate(index):
            if freq_upper.startswith("M"):
                start = pd.Timestamp(year=ts.year, month=ts.month, day=1)
                end = start + pd.offsets.MonthEnd(1)
            elif freq_upper.startswith("Q"):
                q_month = ((ts.month - 1) // 3) * 3 + 1
                start = pd.Timestamp(year=ts.year, month=q_month, day=1)
                end = start + pd.offsets.QuarterEnd(1)
            else:
                start = pd.Timestamp(year=ts.year, month=1, day=1)
                end = pd.Timestamp(year=ts.year, month=12, day=31)
            span = pd.date_range(start, end, freq="D")
            dow = span.dayofweek.to_numpy()
            days_arr[i] = float(len(span))
            wk_arr[i] = float(np.sum(dow < 5))
            we_arr[i] = float(np.sum(dow >= 5))
        return {"days_in_period": days_arr, "n_weekdays": wk_arr, "n_weekend_days": we_arr}

    if freq_upper.startswith("W"):
        return {
            "days_in_period": np.full(n, 7.0, dtype=float),
            "n_weekdays": np.full(n, 5.0, dtype=float),
            "n_weekend_days": np.full(n, 2.0, dtype=float),
        }

    dow = index.dayofweek.to_numpy()
    is_wk = (dow < 5).astype(float)
    is_we = (dow >= 5).astype(float)
    return {
        "days_in_period": index.days_in_month.to_numpy(dtype=float),
        "n_weekdays": is_wk,
        "n_weekend_days": is_we,
    }


def build_features(
    series: pd.DataFrame,
    cfg: RunConfig,
    lam: float | None = None,
    owns_covariate_lags: bool = False,
) -> tuple[pd.Series, pd.DataFrame | None]:
    """Build ``(y, X)`` fit inputs for one series.

    ``series`` is one ts_id's rows with the configured date/target (and optional covariates)
    columns. ``y`` is returned indexed by ds, sorted, with fold-safe target hygiene
    (`grid_policy`, `trim_leading_zeros`, `censored_col`, `outlier_policy`) and the transform
    applied, and with the index's frequency set whenever the dates are regular (see
    `_with_inferred_freq`). ``X`` carries any configured dynamic covariates (``exog``,
    ``future_covariates``, ``past_covariates``), an ``is_holiday`` flag, Fourier terms,
    ``trading_days`` terms, ``level_shift``, ``is_outlier`` (when ``outlier_policy="flag"``),
    and lagged covariates — aligned to ``y`` — or None when nothing is configured.
    """
    d, f = cfg.data, cfg.features
    exog_lags = {} if owns_covariate_lags else f.exog_lags
    if d.date_col not in series or d.target_col not in series:
        raise ConfigError(
            f"series missing required columns '{d.date_col}'/'{d.target_col}'; "
            f"has {list(series.columns)}"
        )
    if f.static_covariates:
        extract_static_covariates(series, cfg)

    frame = regularize_series_grid(series, cfg).copy()
    frame[d.date_col] = pd.to_datetime(frame[d.date_col]).astype("datetime64[ns]")
    frame = frame.sort_values(d.date_col).set_index(d.date_col)
    frame.index.name = "ds"
    frame.index = _with_inferred_freq(frame.index)

    raw_y = frame[d.target_col].astype(float)
    cleaned_y, outlier_flag = clean_training_target(raw_y, frame, cfg)
    y = apply_transform(cleaned_y, f.transform, lam)
    y.name = "y"

    cols: dict[str, np.ndarray] = {}

    # Dynamic exogenous regressors passed straight through (`exog` + `future_covariates` +
    # `past_covariates`, must exist in the series).
    for name in f.dynamic_covariates:
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

    # X-13 style trading-day / calendar composition terms.
    if f.trading_days:
        cols.update(_trading_day_terms(pd.DatetimeIndex(frame.index), d.freq))

    # Step dummy for one detected regime change (see `level_shift_step`).
    if f.level_shift:
        cols["level_shift"] = level_shift_step(y)

    # Outlier dummy when outlier_policy == "flag".
    if outlier_flag is not None:
        cols["is_outlier"] = outlier_flag

    # Lagged covariates. Only ever built from a declared dynamic covariate column — `FeaturesConfig`
    # has already rejected a name that is not in `dynamic_covariates`, so the lookup cannot miss.
    for name, lags in exog_lags.items():
        for lag in lags:
            cols[f"{name}_lag_{lag}"] = frame[name].astype(float).shift(lag).to_numpy()

    if not cols:
        return y, None
    X = pd.DataFrame(cols, index=frame.index)

    if exog_lags:
        head = max(lag for lags in exog_lags.values() for lag in lags)
        X, y = X.iloc[head:], y.iloc[head:]

    return y, X


def build_future_features(
    y: pd.Series,
    X: pd.DataFrame | None,
    cfg: RunConfig,
    *,
    horizon: int | None = None,
    future_covariates_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """The design frame for the forecast horizon, indexed by the *future* dates."""
    if X is None:
        return None
    d, f = cfg.data, cfg.features
    h = horizon if horizon is not None else d.horizon
    future = pd.date_range(start=y.index[-1], periods=h + 1, freq=d.freq)[1:].as_unit("ns")

    known: dict[str, np.ndarray] = {}
    if f.holidays:
        hol = holiday_frame(cfg)
        holiday_days = set(hol["ds"].to_numpy())
        known["is_holiday"] = np.isin(future.to_numpy(), list(holiday_days)).astype(float)
    if f.fourier:
        known.update(_fourier_terms(future, d.freq, order=3))
    if f.trading_days:
        known.update(_trading_day_terms(future, d.freq))
    if f.level_shift and "level_shift" in X.columns:
        known["level_shift"] = np.full(h, float(X["level_shift"].to_numpy()[-1]))
    if "is_outlier" in X.columns:
        known["is_outlier"] = np.zeros(h, dtype=float)

    if future_covariates_df is not None and len(future_covariates_df) >= h:
        for name in f.known_future_covariates:
            if name in future_covariates_df.columns:
                known[name] = future_covariates_df[name].to_numpy(dtype=float)[:h]

    # Recency stand-in for the columns we cannot know: the last `h` observed rows.
    tail_rows = np.clip(np.arange(len(X) - h, len(X)), 0, len(X) - 1)
    tail = X.iloc[tail_rows]

    if f.exog_lags:
        steps = np.arange(len(X), len(X) + h)
        for name, lags in f.exog_lags.items():
            wanted = [lag for lag in lags if f"{name}_lag_{lag}" in X.columns]
            if not wanted:
                continue
            column = X[name].to_numpy(dtype=float)
            forward = known[name] if name in known else column[tail_rows]
            source = np.concatenate([column, forward])
            for lag in wanted:
                known[f"{name}_lag_{lag}"] = source[np.clip(steps - lag, 0, len(source) - 1)]

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
