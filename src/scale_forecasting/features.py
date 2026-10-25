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
from .holiday_window import build_holiday_window_features, coerce_holiday_window_config
from .monotonicity import resolve_monotonic_constraints, validate_monotonic_constraints_map
from .seasonality import periods_per_year, seasonal_period

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from dataclasses import dataclass
    from typing import Literal

    from .config import RunConfig
    from .models.base_model import BaseModel
else:
    from collections.abc import Mapping
    from dataclasses import dataclass
    from typing import Literal


VARIANT_FEATURE_META_KEYS: frozenset[str] = frozenset(
    {
        "covariate_mode",
        "transform",
        "simplex_sum",
        "exog",
        "future_covariates",
        "past_covariates",
        "static_covariates",
        "exog_lags",
        "holidays",
        "fourier",
        "level_shift",
        "trading_days",
        "monotonic_constraints",
        "holiday_window",
        "cold_start_transfer",
        "cold_start_min_obs",
        "cold_start_method",
        "cold_start_matching",
        "similarity_cluster_feature",
        "similarity_method",
        "similarity_n_clusters",
        "calibration_method",
        "aci_gamma",
    }
)
VARIANT_META_KEYS: frozenset[str] = VARIANT_FEATURE_META_KEYS | {"training_mode"}

CovProfile = Literal[
    "cov_none",
    "cov_future_only",
    "cov_past_only",
    "cov_static_only",
    "cov_dynamic",
    "cov_future_static",
    "cov_past_static",
    "cov_all",
]

VALID_COVARIATE_MODES: frozenset[str] = frozenset(
    {
        "univariate",
        "cov_none",
        "cov_future_only",
        "future_only",
        "xreg",
        "cov_past_only",
        "past_only",
        "cov_static_only",
        "static_only",
        "cov_dynamic",
        "dynamic_only",
        "cov_future_static",
        "cov_past_static",
        "cov_all",
        "all",
    }
)

VALID_ABLATION_MODES: frozenset[str] = frozenset(
    {
        "univariate",
        "cov_none",
        "cov_future_only",
        "cov_past_only",
        "cov_static_only",
        "cov_dynamic",
        "cov_future_static",
        "cov_past_static",
        "cov_all",
        "local",
        "global",
        "hybrid",
        "similarity_global",
        "local_univariate",
        "local_cov_none",
        "local_cov_future_only",
        "local_cov_past_only",
        "local_cov_static_only",
        "local_cov_dynamic",
        "local_cov_future_static",
        "local_cov_past_static",
        "local_cov_all",
        "global_univariate",
        "global_cov_none",
        "global_cov_future_only",
        "global_cov_past_only",
        "global_cov_static_only",
        "global_cov_dynamic",
        "global_cov_future_static",
        "global_cov_past_static",
        "global_cov_all",
        "hybrid_univariate",
        "hybrid_cov_none",
        "hybrid_cov_all",
    }
)


# --- transforms ----------------------------------------------------------------


def fit_transform_lambda(
    y: pd.Series,
    transform: str,
    *,
    simplex_sum: float | None = None,
) -> float | None:
    """Fit the transform's stateful parameter on ``y`` (the full series' target).

    * ``boxcox``: its λ, fit by maximum likelihood (``scipy.stats.boxcox_normmax``) on the raw
      target. Box-Cox is defined only for **strictly positive** data.
    * ``clr`` / ``ilr``: the resolved compositional ``simplex_sum`` ($S = 1.0$ or $100.0$, or the
      explicit ``simplex_sum`` override), validating that ``y >= 0``.
    * ``none`` / ``log1p``: returns ``None`` (stateless).
    """
    if transform in ("clr", "ilr"):
        from .simplex import detect_simplex_sum

        arr = y.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0 and float(np.min(finite)) < 0.0:
            raise ConfigError(f"{transform} transform requires non-negative y (got values < 0)")
        return detect_simplex_sum(arr, simplex_sum=simplex_sum)
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

    ``lam`` is the fitted Box-Cox λ from `fit_transform_lambda` (or the resolved ``simplex_sum``
    when ``transform in ("clr", "ilr")``; ignored by the stateless transforms).
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
    if transform in ("clr", "ilr"):
        from .simplex import apply_series_simplex_transform

        return apply_series_simplex_transform(y, transform, simplex_sum=lam)
    raise ConfigError(f"unknown transform '{transform}'")


def invert_transform(values: np.ndarray, transform: str, lam: float | None = None) -> np.ndarray:
    """Invert a transform on forecasts/bounds so the frame is in original units.

    ``lam`` is the same fitted Box-Cox λ (or ``simplex_sum`` for ``clr``/``ilr``) used in the
    forward direction, so ``invert(apply(y)) == y``.
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
    if transform in ("clr", "ilr"):
        from .simplex import invert_series_simplex_transform

        return invert_series_simplex_transform(arr, transform, simplex_sum=lam)
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


def compute_cov_profile(
    features: Any,
    *,
    model_cls: type[Any] | None = None,
) -> CovProfile:
    """Compute the realized covariate profile after narrowing to ``model_cls`` capabilities."""
    has_future = bool(features.exog or features.future_covariates) and (
        model_cls is None or bool(model_cls.supports_future_covariates)
    )
    has_past = bool(features.past_covariates) and (
        model_cls is None or bool(model_cls.supports_past_covariates)
    )
    has_static = bool(features.static_covariates) and (
        model_cls is None or bool(model_cls.supports_static_covariates)
    )
    if has_future and has_past and has_static:
        return "cov_all"
    if has_future and has_past:
        return "cov_dynamic"
    if has_future and has_static:
        return "cov_future_static"
    if has_past and has_static:
        return "cov_past_static"
    if has_future:
        return "cov_future_only"
    if has_past:
        return "cov_past_only"
    if has_static:
        return "cov_static_only"
    return "cov_none"


def clean_model_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``params`` with per-variant feature-override meta-keys stripped (pure)."""
    return {k: v for k, v in params.items() if k not in VARIANT_FEATURE_META_KEYS}


def _parse_mode_token(mode: str) -> tuple[str | None, str | None]:
    """Parse a canonical ablation or tag token into ``(training_mode, covariate_mode)``."""
    token = mode.strip()
    if token in ("similarity_global", "local", "global", "hybrid"):
        return token, None
    if token in VALID_COVARIATE_MODES:
        return None, token
    for prefix in ("similarity_global_", "local_", "global_", "hybrid_"):
        if token.startswith(prefix):
            rest = token[len(prefix) :]
            if rest in VALID_COVARIATE_MODES:
                return prefix[:-1], rest
    return None, None


@dataclass(frozen=True)
class ModelVariantSpec:
    """Resolved execution contract and orthogonal taxonomy for one ``models`` entry."""

    model_name: str
    base_model: str
    variant_tag: str | None
    scope_capability: Literal["local_only", "multi_scope", "global_only"]
    training_mode: Literal["local", "global", "hybrid", "similarity_global"]
    cov_profile: CovProfile
    is_univariate: bool
    covariates_used: tuple[str, ...]
    effective_cfg: RunConfig
    clean_params: dict[str, Any]
    model_cls: type[BaseModel]
    intended_features: Any = None


def _narrow_unsupported_features(
    cfg: RunConfig,
    model_cls: type[Any],
    *,
    allow_unsupported: bool = False,
) -> RunConfig:
    """Narrow ``cfg.features`` to tiers supported by ``model_cls`` or raise when configured."""
    f = cfg.features
    unsupported = model_cls.unsupported_covariate_tiers(f)
    if not unsupported:
        return cfg
    if f.on_unsupported_covariates == "error" and not allow_unsupported:
        raise ConfigError(
            f"features.on_unsupported_covariates='error': model '{model_cls.name}' does not "
            f"support configured covariate tier(s) {list(unsupported)} (supports_future_covariates="
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
    new_mono = {k: v for k, v in f.monotonic_constraints.items() if k in kept_dynamic}
    new_features = f.model_copy(
        update={
            "exog": new_exog,
            "future_covariates": new_future,
            "past_covariates": new_past,
            "static_covariates": new_static,
            "exog_lags": new_lags,
            "monotonic_constraints": new_mono,
        }
    )
    return cfg.model_copy(update={"features": new_features})


def resolve_model_variant(
    cfg: RunConfig,
    model_name: str,
    *,
    model_cls: type[Any] | None = None,
    allow_unsupported: bool = False,
) -> ModelVariantSpec:
    """Resolve a ``<base_model>`` or ``<base_model>@<tag>`` entry into its `ModelVariantSpec`."""
    raw_base, sep, tag = model_name.partition("@")
    raw_base = raw_base.strip()
    variant_tag = tag.strip() if sep else None
    if sep and (not raw_base or not variant_tag):
        raise ConfigError(f"malformed model variant '{model_name}'; expected '<base_model>@<tag>'")

    from .models import base_model_name, get_model

    base = base_model_name(raw_base)
    if model_cls is None:
        model_cls = get_model(base)

    variant_explicit: dict[str, Any] = {}
    if model_name != base:
        if raw_base != base and f"{base}@{variant_tag}" in cfg.model_params:
            variant_explicit.update(cfg.model_params[f"{base}@{variant_tag}"])
        if model_name in cfg.model_params:
            variant_explicit.update(cfg.model_params[model_name])
    raw_params: dict[str, Any] = dict(cfg.model_params.get(base, {}))
    if raw_base != base and raw_base in cfg.model_params:
        raw_params.update(cfg.model_params[raw_base])
    if variant_explicit:
        raw_params.update(variant_explicit)

    f = cfg.features
    has_monotone_tag = False
    if variant_tag is not None:
        norm_tag_tokens = [t for t in variant_tag.replace("+", "_").split("_") if t]
        has_monotone_tag = any(t in ("monotone", "monotonic") for t in norm_tag_tokens)

        declared = set(f.all_covariates)
        inferred_scope, inferred_cov = _parse_mode_token(variant_tag)
        if inferred_scope is not None and "training_mode" not in variant_explicit:
            raw_params["training_mode"] = inferred_scope
        if inferred_cov is not None and "covariate_mode" not in variant_explicit:
            raw_params["covariate_mode"] = inferred_cov

        # Handle single-column drop/keep tags first so column names with underscores stay intact
        matched_col_slice = False
        if inferred_cov is None and not any(
            k in variant_explicit
            for k in ("exog", "future_covariates", "past_covariates", "static_covariates")
        ):
            if variant_tag.startswith("cov_drop_"):
                drop_col = variant_tag[len("cov_drop_") :]
                if drop_col in declared:
                    matched_col_slice = True
                    raw_params["covariate_mode"] = None
                    raw_params["exog"] = [c for c in f.exog if c != drop_col]
                    raw_params["future_covariates"] = [
                        c for c in f.future_covariates if c != drop_col
                    ]
                    raw_params["past_covariates"] = [c for c in f.past_covariates if c != drop_col]
                    raw_params["static_covariates"] = [
                        c for c in f.static_covariates if c != drop_col
                    ]
            elif variant_tag.startswith("cov_") and variant_tag not in VALID_COVARIATE_MODES:
                keep_col = variant_tag[len("cov_") :]
                if keep_col in declared:
                    matched_col_slice = True
                    raw_params["covariate_mode"] = None
                    raw_params["exog"] = [c for c in f.exog if c == keep_col]
                    raw_params["future_covariates"] = [
                        c for c in f.future_covariates if c == keep_col
                    ]
                    raw_params["past_covariates"] = [c for c in f.past_covariates if c == keep_col]
                    raw_params["static_covariates"] = [
                        c for c in f.static_covariates if c == keep_col
                    ]

        if not matched_col_slice:
            # Composable token parser: supports both single tags (@v2.5, @transform_log1p, @direct,
            # @monotone, @similarity_global, @similarity_cold_start) and compound tags.
            rem_tag = variant_tag.replace("+", "_")
            for multi_cov in sorted(VALID_COVARIATE_MODES, key=len, reverse=True):
                if "_" in multi_cov and multi_cov in rem_tag:
                    if "covariate_mode" not in variant_explicit:
                        raw_params["covariate_mode"] = multi_cov
                    rem_tag = rem_tag.replace(multi_cov, "")
            for multi_tr, tr_norm in (
                ("transform_log1p", "log1p"),
                ("transform_boxcox", "boxcox"),
                ("transform_clr", "clr"),
                ("transform_ilr", "ilr"),
                ("transform_none", "none"),
            ):
                if multi_tr in rem_tag:
                    if "transform" not in variant_explicit:
                        raw_params["transform"] = tr_norm
                    rem_tag = rem_tag.replace(multi_tr, "")
            for cs_m in ("analog_transfer", "bass_diffusion", "weibull_lifecycle"):
                if cs_m in rem_tag:
                    if "cold_start_transfer" not in variant_explicit:
                        raw_params["cold_start_transfer"] = True
                    if "cold_start_method" not in variant_explicit:
                        raw_params["cold_start_method"] = cs_m
                    rem_tag = rem_tag.replace(cs_m, "")
            if "similarity_cold_start" in rem_tag:
                if "cold_start_transfer" not in variant_explicit:
                    raw_params["cold_start_transfer"] = True
                if "cold_start_matching" not in variant_explicit:
                    raw_params["cold_start_matching"] = "similarity"
                rem_tag = rem_tag.replace("similarity_cold_start", "")
            if "cold_start" in rem_tag:
                if "cold_start_transfer" not in variant_explicit:
                    raw_params["cold_start_transfer"] = True
                rem_tag = rem_tag.replace("cold_start", "")
            if "similarity_global" in rem_tag:
                if "training_mode" not in variant_explicit:
                    raw_params["training_mode"] = "similarity_global"
                rem_tag = rem_tag.replace("similarity_global", "")
            for aci_tok in ("aci_normalized", "aci_norm"):
                if aci_tok in rem_tag:
                    if "calibration_method" not in variant_explicit:
                        raw_params["calibration_method"] = "aci_normalized"
                    rem_tag = rem_tag.replace(aci_tok, "")

            supported_for_tag = model_cls.supported_training_modes()
            for tok in (t for t in rem_tag.split("_") if t):
                if tok in ("local", "global", "hybrid") and "training_mode" not in variant_explicit:
                    if not has_monotone_tag or tok in supported_for_tag:
                        raw_params["training_mode"] = tok
                elif tok in VALID_COVARIATE_MODES and "covariate_mode" not in variant_explicit:
                    raw_params["covariate_mode"] = tok
                elif (
                    base == "chronos_bolt"
                    and tok in ("tiny", "mini", "small", "base")
                    and "size" not in variant_explicit
                ):
                    raw_params["size"] = tok
                elif (
                    base in ("timesfm_torch", "timesfm")
                    and tok in ("v2.0", "2.0", "v2.5", "2.5", "v3.0", "3.0")
                    and "version" not in variant_explicit
                ):
                    raw_params["version"] = tok.lstrip("v")
                elif (
                    tok in ("log1p", "boxcox", "clr", "ilr", "linear")
                    and "transform" not in variant_explicit
                ):
                    raw_params["transform"] = "none" if tok == "linear" else tok
                elif tok in ("direct", "recursive") and "strategy" not in variant_explicit:
                    raw_params["strategy"] = tok
                elif tok in ("sba", "tsb", "classic") and "variant" not in variant_explicit:
                    raw_params["variant"] = tok
                elif tok == "aci" and "calibration_method" not in variant_explicit:
                    raw_params["calibration_method"] = "aci"

    # If the run includes a `@monotone` sibling for this base model, an untagged base model
    # serves as the unconstrained baseline unless `monotonic_constraints` was explicitly passed
    # in `model_params[base]`.
    has_monotone_sibling = any(
        base_model_name(m) == base
        and "@" in m
        and any(
            t in ("monotone", "monotonic") for t in m.partition("@")[2].replace("+", "_").split("_")
        )
        for m in cfg.models
    )
    if (
        not has_monotone_tag
        and has_monotone_sibling
        and f.monotonic_constraints
        and "monotonic_constraints" not in raw_params
    ):
        raw_params["monotonic_constraints"] = {}
    elif has_monotone_tag:
        explicit_param_mono = raw_params.get("monotonic_constraints")
        resolved_mono = resolve_monotonic_constraints(
            f.monotonic_constraints,
            explicit_param_mono if isinstance(explicit_param_mono, Mapping) else None,
            candidate_columns=f.dynamic_covariates,
            infer_if_empty=True,
        )
        raw_params["monotonic_constraints"] = resolved_mono
        if not resolved_mono:
            raw_params.setdefault("monotone", True)

    scope_cap = model_cls.scope_capability()
    default_tm = "global" if scope_cap == "global_only" else "local"
    raw_tm = raw_params.get("training_mode")
    tm_str = str(raw_tm) if raw_tm is not None else default_tm
    supported_tms = model_cls.supported_training_modes()
    req_tm_check = "global" if tm_str == "similarity_global" else tm_str
    if req_tm_check not in supported_tms:
        raise ConfigError(
            f"model_params.{model_name}.training_mode={tm_str!r} is not supported by "
            f"'{model_name}' (supported modes: {list(supported_tms)})."
        )
    training_mode: Literal["local", "global", "hybrid", "similarity_global"] = (
        "similarity_global"
        if tm_str == "similarity_global"
        else ("global" if tm_str == "global" else ("hybrid" if tm_str == "hybrid" else "local"))
    )

    has_feature_overrides = any(k in raw_params for k in VARIANT_FEATURE_META_KEYS)
    if not has_feature_overrides:
        cand_features = f
        eff_cfg = _narrow_unsupported_features(cfg, model_cls, allow_unsupported=allow_unsupported)
    else:
        new_exog = list(f.exog)
        new_future = list(f.future_covariates)
        new_past = list(f.past_covariates)
        new_static = list(f.static_covariates)
        new_lags = {k: list(v) for k, v in f.exog_lags.items()}
        new_holidays = list(f.holidays)
        new_fourier = f.fourier
        new_level_shift = f.level_shift
        new_trading_days = f.trading_days
        new_transform = f.transform
        new_simplex_sum = f.simplex_sum
        new_mono_constraints: dict[str, Any] = dict(f.monotonic_constraints)
        new_holiday_window = f.holiday_window
        new_sim_cluster_feature = f.similarity_cluster_feature

        if "transform" in raw_params and raw_params["transform"] is not None:
            tr_val = str(raw_params["transform"])
            if tr_val not in ("none", "log1p", "boxcox", "clr", "ilr"):
                raise ConfigError(
                    f"model_params.{model_name}.transform={tr_val!r} is invalid; "
                    "must be 'none', 'log1p', 'boxcox', 'clr', or 'ilr'"
                )
            new_transform = tr_val  # type: ignore[assignment]

        if "simplex_sum" in raw_params:
            raw_ss = raw_params["simplex_sum"]
            if raw_ss is None:
                new_simplex_sum = None
            else:
                ss_val = float(raw_ss)
                if not np.isfinite(ss_val) or ss_val <= 0.0:
                    raise ConfigError(
                        f"model_params.{model_name}.simplex_sum={raw_ss!r} must be > 0"
                    )
                new_simplex_sum = ss_val

        if "covariate_mode" in raw_params and raw_params["covariate_mode"] is not None:
            cm = str(raw_params["covariate_mode"])
            if cm not in VALID_COVARIATE_MODES:
                raise ConfigError(
                    f"model_params.{model_name}.covariate_mode={cm!r} is invalid; "
                    f"must be one of {sorted(VALID_COVARIATE_MODES)}"
                )
            if cm in ("univariate", "cov_none"):
                new_exog, new_future, new_past, new_static, new_lags = [], [], [], [], {}
                new_sim_cluster_feature = False
            elif cm in ("cov_future_only", "future_only", "xreg"):
                new_past, new_static = [], []
            elif cm in ("cov_past_only", "past_only"):
                new_exog, new_future, new_static = [], [], []
            elif cm in ("cov_static_only", "static_only"):
                new_exog, new_future, new_past, new_lags = [], [], [], {}
            elif cm in ("cov_dynamic", "dynamic_only"):
                new_static = []
            elif cm == "cov_future_static":
                new_past = []
            elif cm == "cov_past_static":
                new_exog, new_future = [], []

        declared_all = set(f.all_covariates)
        for tier_key, target_name in (
            ("exog", "new_exog"),
            ("future_covariates", "new_future"),
            ("past_covariates", "new_past"),
            ("static_covariates", "new_static"),
        ):
            if tier_key in raw_params and raw_params[tier_key] is not None:
                raw_val = raw_params[tier_key]
                if not isinstance(raw_val, list):
                    raise ConfigError(
                        f"model_params.{model_name}.{tier_key} must be a list of column names"
                    )
                cols = [str(c) for c in raw_val]
                undeclared = [c for c in cols if c not in declared_all]
                if undeclared:
                    raise ConfigError(
                        f"model_params.{model_name}.{tier_key} references column(s) {undeclared} "
                        f"that are not declared in features.{tier_key} ({sorted(declared_all)}); "
                        "declare all candidate columns in the top-level features block so the "
                        "panel reader loads them."
                    )
                if target_name == "new_exog":
                    new_exog = cols
                elif target_name == "new_future":
                    new_future = cols
                elif target_name == "new_past":
                    new_past = cols
                elif target_name == "new_static":
                    new_static = cols

        if "exog_lags" in raw_params and raw_params["exog_lags"] is not None:
            raw_lags = raw_params["exog_lags"]
            if not isinstance(raw_lags, dict):
                raise ConfigError(f"model_params.{model_name}.exog_lags must be a dict")
            new_lags = {str(k): [int(x) for x in v] for k, v in raw_lags.items()}
        if "holidays" in raw_params and raw_params["holidays"] is not None:
            raw_hol = raw_params["holidays"]
            if not isinstance(raw_hol, list):
                raise ConfigError(f"model_params.{model_name}.holidays must be a list of ISO codes")
            new_holidays = [str(c) for c in raw_hol]
        if "fourier" in raw_params and raw_params["fourier"] is not None:
            new_fourier = bool(raw_params["fourier"])
        if "level_shift" in raw_params and raw_params["level_shift"] is not None:
            new_level_shift = bool(raw_params["level_shift"])
        if "trading_days" in raw_params and raw_params["trading_days"] is not None:
            new_trading_days = bool(raw_params["trading_days"])
        if (
            "monotonic_constraints" in raw_params
            and raw_params["monotonic_constraints"] is not None
        ):
            raw_mc = raw_params["monotonic_constraints"]
            if isinstance(raw_mc, Mapping):
                new_mono_constraints = validate_monotonic_constraints_map(raw_mc)
        if "holiday_window" in raw_params:
            new_holiday_window = coerce_holiday_window_config(raw_params["holiday_window"])
        new_cs_transfer = (
            bool(raw_params["cold_start_transfer"])
            if "cold_start_transfer" in raw_params and raw_params["cold_start_transfer"] is not None
            else f.cold_start_transfer
        )
        new_cs_min_obs = (
            int(raw_params["cold_start_min_obs"])
            if "cold_start_min_obs" in raw_params and raw_params["cold_start_min_obs"] is not None
            else f.cold_start_min_obs
        )
        new_cs_method = (
            str(raw_params["cold_start_method"])
            if "cold_start_method" in raw_params and raw_params["cold_start_method"] is not None
            else f.cold_start_method
        )
        new_cs_matching = (
            str(raw_params["cold_start_matching"])
            if "cold_start_matching" in raw_params and raw_params["cold_start_matching"] is not None
            else f.cold_start_matching
        )
        if (
            "similarity_cluster_feature" in raw_params
            and raw_params["similarity_cluster_feature"] is not None
        ):
            new_sim_cluster_feature = bool(raw_params["similarity_cluster_feature"])
        new_sim_method = (
            str(raw_params["similarity_method"])
            if "similarity_method" in raw_params and raw_params["similarity_method"] is not None
            else f.similarity_method
        )
        new_sim_n_clusters = (
            raw_params["similarity_n_clusters"]
            if (
                "similarity_n_clusters" in raw_params
                and raw_params["similarity_n_clusters"] is not None
            )
            else f.similarity_n_clusters
        )

        kept_dynamic = set(new_exog) | set(new_future) | set(new_past)
        new_lags = {k: list(v) for k, v in new_lags.items() if k in kept_dynamic}
        new_mono_constraints = {k: v for k, v in new_mono_constraints.items() if k in kept_dynamic}
        cand_features = f.model_copy(
            update={
                "transform": new_transform,
                "simplex_sum": new_simplex_sum,
                "exog": new_exog,
                "future_covariates": new_future,
                "past_covariates": new_past,
                "static_covariates": new_static,
                "exog_lags": new_lags,
                "holidays": new_holidays,
                "fourier": new_fourier,
                "level_shift": new_level_shift,
                "trading_days": new_trading_days,
                "monotonic_constraints": new_mono_constraints,
                "holiday_window": new_holiday_window,
                "cold_start_transfer": new_cs_transfer,
                "cold_start_min_obs": new_cs_min_obs,
                "cold_start_method": new_cs_method,
                "cold_start_matching": new_cs_matching,
                "similarity_cluster_feature": new_sim_cluster_feature,
                "similarity_method": new_sim_method,
                "similarity_n_clusters": new_sim_n_clusters,
            }
        )
        cand_cfg = cfg if cand_features == f else cfg.model_copy(update={"features": cand_features})
        eff_cfg = _narrow_unsupported_features(
            cand_cfg,
            model_cls,
            allow_unsupported=allow_unsupported,
        )
        output_updates: dict[str, Any] = {}
        if "calibration_method" in raw_params and raw_params["calibration_method"] is not None:
            cm_val = str(raw_params["calibration_method"])
            if cm_val not in ("conformal", "aci", "aci_normalized"):
                raise ConfigError(
                    f"model_params.{model_name}.calibration_method={cm_val!r} is invalid; "
                    "must be 'conformal', 'aci', or 'aci_normalized'"
                )
            output_updates["calibration_method"] = cm_val
        if "aci_gamma" in raw_params and raw_params["aci_gamma"] is not None:
            gamma_val = float(raw_params["aci_gamma"])
            if not np.isfinite(gamma_val) or gamma_val <= 0.0:
                raise ConfigError(
                    f"model_params.{model_name}.aci_gamma={raw_params['aci_gamma']!r} must be > 0"
                )
            output_updates["aci_gamma"] = gamma_val
        if output_updates:
            eff_cfg = eff_cfg.model_copy(
                update={"output": eff_cfg.output.model_copy(update=output_updates)}
            )

    eff_f = eff_cfg.features
    cov_prof = compute_cov_profile(eff_f, model_cls=model_cls)
    used_fut = list(eff_f.known_future_covariates) if model_cls.supports_future_covariates else []
    used_past = list(eff_f.past_covariates) if model_cls.supports_past_covariates else []
    used_stat = list(eff_f.static_covariates) if model_cls.supports_static_covariates else []
    if (variant_tag == "xreg" or raw_params.get("covariate_mode") == "xreg") and not used_fut:
        raise ConfigError(
            f"model variant {model_name!r} ('ARIMA_PLUS_XREG') requires at least one known-future "
            "covariate in features.future_covariates (or features.exog)."
        )
    covariates_used = tuple(dict.fromkeys([*used_fut, *used_past, *used_stat]))

    cleaned = clean_model_params(raw_params)
    return ModelVariantSpec(
        model_name=model_name,
        base_model=base,
        variant_tag=variant_tag,
        scope_capability=scope_cap,
        training_mode=training_mode,
        cov_profile=cov_prof,
        is_univariate=(cov_prof == "cov_none"),
        covariates_used=covariates_used,
        effective_cfg=eff_cfg,
        clean_params=cleaned,
        model_cls=model_cls,
        intended_features=cand_features,
    )


def effective_config_for_model(
    cfg: RunConfig,
    model_cls: type[Any],
    model_name: str | None = None,
) -> RunConfig:
    """Return a `RunConfig` narrowed to ``model_name``'s variant overrides and ``model_cls`` tiers.

    Used inside the worker and HPO so that in a multi-model or multi-variant benchmark under
    ``on_unsupported_covariates="fallback"`` (the default), a univariate model or a variant with
    narrowed covariates never has its target truncated by ``exog_lags`` of an unused covariate tier
    or receives columns it should not consume. The run identity (`run_id` and `model_hash`) is
    always computed from the original ``cfg`` before calling this helper.
    """
    target_name = model_name if model_name is not None else str(model_cls.name)
    if (
        "@" not in target_name
        and target_name not in cfg.model_params
        and not cfg.features.monotonic_constraints
        and not model_cls.unsupported_covariate_tiers(cfg.features)
    ):
        return cfg
    return resolve_model_variant(cfg, target_name, model_cls=model_cls).effective_cfg


def should_stamp_variant_diagnostics(model_name: str, cfg: RunConfig) -> bool:
    """Whether ``run_cell`` / ``run_panel`` should stamp taxonomy diagnostics onto `CellResult`."""
    base = model_name.split("@", 1)[0].strip()
    return bool(
        "@" in model_name
        or cfg.features.all_covariates
        or cfg.features.include_univariate_baselines
        or cfg.features.ablation_modes
        or cfg.compute.reuse_prior_runs
        or cfg.compute.reuse_from_run_id is not None
        or cfg.model_params.get(model_name)
        or cfg.model_params.get(base)
    )


def variant_diagnostics_for(
    model_name: str,
    cfg: RunConfig,
    *,
    model_cls: type[Any] | None = None,
    borrowed_from_run_id: str | None = None,
) -> dict[str, Any]:
    """Return the canonical scope/covariate taxonomy and spec-hash metadata for ``model_name``."""
    from .registry.ids import make_model_spec_hash

    spec = resolve_model_variant(cfg, model_name, model_cls=model_cls)
    out: dict[str, Any] = {
        "base_model": spec.base_model,
        "variant_tag": spec.variant_tag,
        "scope_capability": spec.scope_capability,
        "training_mode": spec.training_mode,
        "cov_profile": spec.cov_profile,
        "is_univariate": spec.is_univariate,
        "covariates_used": list(spec.covariates_used),
        "model_spec_hash": make_model_spec_hash(model_name, cfg),
    }
    if borrowed_from_run_id is not None:
        out["borrowed_from_run_id"] = borrowed_from_run_id
    return out


def _cov_mode_is_feasible(cov_mode: str | None, features: Any, model_cls: type[Any]) -> bool:
    """Whether ``cov_mode`` can produce a distinct realized covariate tier for ``model_cls``."""
    if cov_mode is None or cov_mode in ("univariate", "cov_none"):
        return True
    has_fut = bool(features.exog or features.future_covariates) and bool(
        model_cls.supports_future_covariates
    )
    has_past = bool(features.past_covariates) and bool(model_cls.supports_past_covariates)
    has_stat = bool(features.static_covariates) and bool(model_cls.supports_static_covariates)
    if cov_mode in ("cov_future_only", "future_only", "xreg"):
        return has_fut
    if cov_mode in ("cov_past_only", "past_only"):
        return has_past
    if cov_mode in ("cov_static_only", "static_only"):
        return has_stat
    if cov_mode in ("cov_dynamic", "dynamic_only"):
        return has_fut and has_past
    if cov_mode == "cov_future_static":
        return has_fut and has_stat
    if cov_mode == "cov_past_static":
        return has_past and has_stat
    if cov_mode in ("cov_all", "all"):
        return has_fut or has_past or has_stat
    return False


def expand_model_variants(
    cfg: RunConfig,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Expand ``include_univariate_baselines`` and ``ablation_modes`` into concrete ``models`` and
    ``model_params`` with capability-aware filtering and semantic deduplication (pure).
    """
    f = cfg.features
    models_out: list[str] = list(cfg.models)
    params_out: dict[str, dict[str, Any]] = {k: dict(v) for k, v in cfg.model_params.items()}

    if not f.include_univariate_baselines and not f.ablation_modes and not f.ablation_models:
        return models_out, params_out

    if f.ablation_models and not f.include_univariate_baselines and not f.ablation_modes:
        raise ValueError(
            "features.ablation_models requires features.ablation_modes or "
            "features.include_univariate_baselines=true"
        )

    for mode in f.ablation_modes:
        if mode not in VALID_ABLATION_MODES:
            raise ValueError(
                f"features.ablation_modes contains unknown mode {mode!r}; "
                f"valid modes: {sorted(VALID_ABLATION_MODES)}"
            )

    existing_bases = {m.split("@", 1)[0].strip() for m in models_out}
    for target in f.ablation_models:
        target_base = target.split("@", 1)[0].strip()
        if target not in models_out and target_base not in existing_bases:
            raise ValueError(
                f"features.ablation_models contains {target!r}, which is not in models {models_out}"
            )

    # Track semantic signatures of all already-scheduled models so we never add a duplicate fit.
    def _semantic_key(spec: ModelVariantSpec) -> tuple[Any, ...]:
        clean_without_tm = {
            k: (tuple(v) if isinstance(v, list) else v)
            for k, v in sorted(spec.clean_params.items())
            if k != "training_mode"
        }
        return (
            spec.base_model,
            spec.training_mode,
            spec.cov_profile,
            spec.covariates_used,
            tuple(clean_without_tm.items()),
        )

    seen_names: set[str] = set(models_out)
    seen_signatures: set[tuple[Any, ...]] = set()
    all_seed_models: list[str] = []
    scoped_seed_models: list[str] = []
    for m in models_out:
        spec = resolve_model_variant(cfg, m)
        seen_signatures.add(_semantic_key(spec))
        if "@" not in m:
            all_seed_models.append(m)
        if f.ablation_models and (m in f.ablation_models or spec.base_model in f.ablation_models):
            scoped_seed_models.append(m)

    if not all_seed_models:
        # All initial models had @ tags; allow expanding unique base models.
        seen_b: set[str] = set()
        for m in models_out:
            b = m.split("@", 1)[0].strip()
            if b not in seen_b:
                seen_b.add(b)
                all_seed_models.append(m)

    if not f.ablation_models:
        scoped_seed_models = list(all_seed_models)

    def _apply_mode(seed: str, mode_str: str) -> None:
        seed_spec = resolve_model_variant(cfg, seed)
        base = seed_spec.base_model
        cls = seed_spec.model_cls
        req_scope, req_cov = _parse_mode_token(mode_str)
        eff_scope = req_scope if req_scope is not None else seed_spec.training_mode
        req_tm_check = "global" if eff_scope == "similarity_global" else eff_scope
        if req_tm_check not in cls.supported_training_modes():
            return
        if not _cov_mode_is_feasible(req_cov, f, cls):
            return

        cand_name = f"{base}@{mode_str}"
        if cand_name in seen_names:
            return

        cand_params: dict[str, Any] = dict(params_out.get(base, {}))
        if req_scope is not None:
            cand_params["training_mode"] = req_scope
        elif "training_mode" in seed_spec.clean_params:
            cand_params["training_mode"] = seed_spec.clean_params["training_mode"]
        if req_cov is not None:
            cand_params["covariate_mode"] = req_cov

        temp_params = {**params_out, cand_name: cand_params}
        temp_cfg = cfg.model_copy(update={"model_params": temp_params})
        cand_spec = resolve_model_variant(temp_cfg, cand_name, model_cls=cls)
        sig = _semantic_key(cand_spec)
        if sig in seen_signatures:
            return

        seen_names.add(cand_name)
        seen_signatures.add(sig)
        models_out.append(cand_name)
        params_out[cand_name] = cand_params

    if f.include_univariate_baselines:
        uni_seeds = all_seed_models if f.ablation_modes else scoped_seed_models
        for seed in uni_seeds:
            _apply_mode(seed, "univariate")

    for seed in scoped_seed_models:
        for mode_str in f.ablation_modes:
            _apply_mode(seed, mode_str)

    return models_out, params_out


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

    def _base(m: str) -> str:
        return m.split("@", 1)[0].strip()

    if demand_class in ("intermittent", "lumpy"):
        matched = [m for m in model_list if _base(m) in _INTERMITTENT_MODELS]
    elif demand_class == "cold_start":
        matched = [m for m in model_list if _base(m) in _COLD_START_MODELS]
    elif demand_class == "smooth":
        non_intermittent = [m for m in model_list if _base(m) not in _PURE_INTERMITTENT_MODELS]
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


def local_seasonal_baseline(arr: np.ndarray, period: int) -> np.ndarray:
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


_local_seasonal_baseline = local_seasonal_baseline


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


def _parse_similarity_cluster_col(series: pd.Series) -> np.ndarray:
    """Parse numeric or string ``similarity_cluster_id`` values into a float array."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=float)
    out = np.zeros(len(series), dtype=float)
    for idx, val in enumerate(series):
        if pd.isna(val):
            out[idx] = 0.0
            continue
        if isinstance(val, (int, float, np.number)):
            out[idx] = float(val)
            continue
        text = str(val).strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        out[idx] = float(int(digits)) if digits else 0.0
    return out


def add_similarity_features(
    panel_df: pd.DataFrame,
    cfg: RunConfig,
    *,
    cutoff_ds: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Attach fold-safe ``similarity_cluster_id`` and ``similarity_to_medoid`` to ``panel_df``.

    When ``cutoff_ds`` is provided (e.g., inside a walk-forward backtest fold), similarity
    clustering is computed strictly on rows with ``ds <= cutoff_ds`` so future validation targets
    never leak into the cluster assignment or medoid proximity score.
    """
    if panel_df.empty:
        return panel_df.copy()
    d, f = cfg.data, cfg.features
    id_col = d.ts_id_col
    date_col = d.date_col
    target_col = d.target_col
    if id_col not in panel_df.columns or target_col not in panel_df.columns:
        out = panel_df.copy()
        out["similarity_cluster_id"] = 0
        out["similarity_to_medoid"] = 1.0
        return out

    source_df = panel_df
    if cutoff_ds is not None and date_col in panel_df.columns:
        cutoff_ts = pd.Timestamp(cutoff_ds)
        ds_parsed = pd.to_datetime(panel_df[date_col])
        sliced = panel_df.loc[ds_parsed <= cutoff_ts]
        if not sliced.empty:
            source_df = sliced

    from .similarity import cluster_by_similarity_df

    clusters_df = cluster_by_similarity_df(
        source_df,
        cfg=cfg,
        n_clusters=f.similarity_n_clusters,
        method=f.similarity_method,
        min_cluster_size=2,
        ts_id_col=id_col,
        date_col=date_col,
        target_col=target_col,
    )
    cluster_int_map: dict[str, int] = {}
    sim_medoid_map: dict[str, float] = {}
    for _, row in clusters_df.iterrows():
        uid = str(row["ts_id"])
        cluster_int_map[uid] = int(row.get("cluster_int_id", 0))
        sim_medoid_map[uid] = float(row.get("similarity_to_medoid", 1.0))

    out = panel_df.copy()
    uid_str = out[id_col].astype(str)
    out["similarity_cluster_id"] = uid_str.map(lambda u: cluster_int_map.get(u, 0)).astype(int)
    out["similarity_to_medoid"] = uid_str.map(lambda u: sim_medoid_map.get(u, 1.0)).astype(float)
    return out


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
    similarity cluster features (when ``similarity_cluster_feature=True``), and lagged covariates
    — aligned to ``y`` — or None when nothing is configured.
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
        if f.holiday_window is not None and f.holiday_window.enabled:
            cols.update(
                build_holiday_window_features(pd.DatetimeIndex(frame.index), hol, f.holiday_window)
            )

    # Fourier seasonality terms (yearly), for ML models.
    if f.fourier:
        cols.update(_fourier_terms(pd.DatetimeIndex(frame.index), d.freq, order=3))

    # X-13 style trading-day / calendar composition terms.
    if f.trading_days:
        cols.update(_trading_day_terms(pd.DatetimeIndex(frame.index), d.freq))

    # Step dummy for one detected regime change (see `level_shift_step`).
    if f.level_shift:
        cols["level_shift"] = level_shift_step(y)

    # Similarity cluster membership and medoid proximity features (fold-safe).
    if f.similarity_cluster_feature:
        if "similarity_cluster_id" in frame.columns:
            cols["similarity_cluster_id"] = _parse_similarity_cluster_col(
                frame["similarity_cluster_id"]
            )
        else:
            cols["similarity_cluster_id"] = np.zeros(len(frame), dtype=float)
        if "similarity_to_medoid" in frame.columns:
            cols["similarity_to_medoid"] = (
                pd.to_numeric(frame["similarity_to_medoid"], errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
        else:
            cols["similarity_to_medoid"] = np.ones(len(frame), dtype=float)

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
        if f.holiday_window is not None and f.holiday_window.enabled:
            known.update(build_holiday_window_features(future, hol, f.holiday_window))
    if f.fourier:
        known.update(_fourier_terms(future, d.freq, order=3))
    if f.trading_days:
        known.update(_trading_day_terms(future, d.freq))
    if f.level_shift and "level_shift" in X.columns:
        known["level_shift"] = np.full(h, float(X["level_shift"].to_numpy()[-1]))
    if f.similarity_cluster_feature:
        if "similarity_cluster_id" in X.columns:
            known["similarity_cluster_id"] = np.full(
                h, float(X["similarity_cluster_id"].to_numpy()[-1])
            )
        if "similarity_to_medoid" in X.columns:
            known["similarity_to_medoid"] = np.full(
                h, float(X["similarity_to_medoid"].to_numpy()[-1])
            )
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
    """Sine/cosine Fourier features for seasonal cycles (pure).

    Computes ``t`` in days since epoch (`nanos / (24 * 3600 * 1e9)`):
    - Daily, weekly, monthly, and quarterly frequencies use the physical annual period in days
      (`365.25` days) so harmonics $k = 1 \\dots \\text{order}$ represent true annual, semi-annual,
      and 4-month cycles regardless of sampling cadence.
    - Sub-daily hourly frequencies (`h` / `H`) encode multi-seasonal harmonics across the ``order``
      pairs: diurnal $24\\text{h}$ fundamental ($P_d = 1.0$ day, $k = 1$), semi-diurnal
      $12\\text{h}$ double-peak ($P_d = 1.0$ day, $k = 2$), and weekly $168\\text{h}$ cycle
      ($P_w = 7.0$ days).
    """
    nanos = index.to_numpy(dtype="datetime64[ns]").astype("int64")
    t = nanos.astype(float) / (24 * 3600 * 1e9)
    f_norm = str(freq).strip().upper()
    out: dict[str, np.ndarray] = {}
    if f_norm in ("H", "1H", "BH"):
        harmonic_specs = (
            (1.0, 1.0),  # 24h diurnal fundamental
            (2.0, 1.0),  # 12h semi-diurnal double-peak
            (1.0, 7.0),  # 168h weekly cycle
        )
        for k in range(1, order + 1):
            mult, period_days = (
                harmonic_specs[k - 1] if k <= len(harmonic_specs) else (float(k), 7.0)
            )
            ang = 2.0 * np.pi * mult * t / period_days
            out[f"fourier_sin_{k}"] = np.sin(ang)
            out[f"fourier_cos_{k}"] = np.cos(ang)
        return out

    period = (
        365.25 if f_norm.startswith(("D", "B", "W", "M", "Q")) else float(periods_per_year(freq))
    )
    for k in range(1, order + 1):
        ang = 2.0 * np.pi * k * t / period
        out[f"fourier_sin_{k}"] = np.sin(ang)
        out[f"fourier_cos_{k}"] = np.cos(ang)
    return out
