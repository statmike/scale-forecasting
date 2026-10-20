"""Econometric monotonicity & sign-constraint resolution and solver utilities.

Provides declarative sign-constraint validation, automatic covariate direction
inference (e.g., ``price* -> -1``, ``promo*/discount* -> +1``), feature-order
constraint vector alignment for gradient-boosted trees (LightGBM, XGBoost,
CatBoost), and bounded Tikhonov-regularized least squares
(:func:`scipy.optimize.lsq_linear`) for ``regression_lags`` / ``ridge_lags``.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

MONOTONE_SUPPORTED_MODELS: tuple[str, ...] = (
    "lightgbm",
    "xgboost",
    "catboost",
    "regression_lags",
    "ridge_lags",
)

_PRICE_EXACT: frozenset[str] = frozenset(
    {
        "price",
        "unit_price",
        "retail_price",
        "net_price",
        "base_price",
        "shelf_price",
        "regular_price",
        "selling_price",
        "log_price",
        "price_index",
        "avg_price",
        "list_price",
        "reg_price",
        "tariff_rate",
    }
)

_PROMO_EXACT: frozenset[str] = frozenset(
    {
        "promo",
        "promotion",
        "onpromotion",
        "is_promo",
        "promo_flag",
        "promo_active",
        "promo_depth",
        "discount",
        "discount_depth",
        "discount_pct",
        "discount_rate",
        "rebate_depth",
        "markdown",
        "markdown_active",
        "coupon",
        "coupon_flag",
        "deal",
        "feature_display",
        "ad_spend",
        "marketing_spend",
        "marketing_impressions",
    }
)

_EXCLUDED_INFERENCE_PREFIXES: tuple[str, ...] = (
    "lag_",
    "roll_",
    "ewm_",
    "fourier_",
    "dow_",
    "month_",
    "hour_",
    "is_holiday",
    "pre_holiday_",
    "post_holiday_",
    "days_to_next_holiday",
    "days_since_prev_holiday",
    "holiday_offset_days",
)


def validate_monotonic_constraints_map(
    raw: Mapping[str, Any] | None,
) -> dict[str, int]:
    """Validate and normalize a ``{column: -1 | 0 | 1}`` constraint dictionary.

    Args:
        raw: Mapping of feature column names to directional sign constraints
            (``-1`` for non-increasing, ``0`` for unconstrained, ``+1`` for
            non-decreasing).

    Returns:
        Normalized ``dict[str, int]`` with integer values in ``{-1, 0, 1}``.

    Raises:
        ValueError: If any key is empty/non-string or any value is not in
            ``{-1, 0, 1}``.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            "monotonic_constraints must be a mapping of {column_name: -1 | 0 | 1}, "
            f"got {type(raw).__name__}."
        )
    out: dict[str, int] = {}
    for key, val in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"monotonic_constraints keys must be non-empty strings, got {key!r}.")
        if isinstance(val, bool) or not isinstance(val, numbers.Integral):
            raise ValueError(
                f"monotonic_constraints[{key!r}] must be an integer in {{-1, 0, 1}}, got {val!r}."
            )
        ival = int(val)
        if ival not in (-1, 0, 1):
            raise ValueError(f"monotonic_constraints[{key!r}] must be -1, 0, or 1; got {ival}.")
        out[key.strip()] = ival
    return out


def infer_monotonic_constraints(columns: Sequence[str]) -> dict[str, int]:
    """Infer economic sign constraints from covariate column names.

    Applies standard demand-theory sign priors:
    - Price covariates (``price``, ``unit_price``, ``retail_price``, ``*_price``,
      ``price_*``) map to ``-1`` (downward-sloping demand).
    - Promotional / discount covariates (``promo``, ``promo_*``, ``*_promo``,
      ``discount*``, ``markdown*``) map to ``+1`` (non-negative promotional lift).

    Args:
        columns: Candidate exogenous covariate column names.

    Returns:
        Dictionary mapping recognized economic covariates to ``-1`` or ``+1``.
    """
    inferred: dict[str, int] = {}
    for col in columns:
        if not isinstance(col, str):
            continue
        clean = col.strip()
        lower = clean.lower()
        if not lower or lower in {"ts_id", "ds", "y", "unit_cost", "cost"}:
            continue
        if lower.startswith(_EXCLUDED_INFERENCE_PREFIXES):
            continue

        if (
            lower in _PROMO_EXACT
            or lower.startswith(("promo_", "promotion_", "discount_", "markdown_"))
            or lower.endswith(("_promo", "_promotion", "_discount", "_markdown"))
        ):
            inferred[clean] = 1
            continue

        if (
            "promo" not in lower
            and "discount" not in lower
            and "cost" not in lower
            and (lower in _PRICE_EXACT or lower.startswith("price_") or lower.endswith("_price"))
        ):
            inferred[clean] = -1
    return inferred


def resolve_monotonic_constraints(
    cfg_constraints: Mapping[str, Any] | None = None,
    param_constraints: Mapping[str, Any] | None = None,
    candidate_columns: Sequence[str] = (),
    *,
    infer_if_empty: bool = False,
) -> dict[str, int]:
    """Resolve effective monotonic constraints from config, model params, and inference.

    Args:
        cfg_constraints: Constraints declared on ``cfg.features.monotonic_constraints``.
        param_constraints: Model-specific override dictionary in ``model_params``.
        candidate_columns: Known future and past covariate columns eligible for
            automatic sign inference when ``infer_if_empty=True``.
        infer_if_empty: When ``True`` (e.g. ``@monotone`` variant tag), infer
            economic signs from ``candidate_columns`` if no explicit non-zero
            constraints were supplied.

    Returns:
        Effective ``dict[str, int]`` mapping column names to ``{-1, 0, 1}``.
    """
    merged = validate_monotonic_constraints_map(cfg_constraints)
    if param_constraints is not None and isinstance(param_constraints, Mapping):
        merged.update(validate_monotonic_constraints_map(param_constraints))

    has_nonzero = any(v != 0 for v in merged.values())
    if infer_if_empty and not has_nonzero:
        inferred = infer_monotonic_constraints(candidate_columns)
        for col, sign in inferred.items():
            merged.setdefault(col, sign)
    return merged


def build_monotonic_constraint_vector(
    feature_names: Sequence[str],
    constraints: Mapping[str, int] | Sequence[int] | None,
) -> list[int]:
    """Align a constraint map or sequence with an ordered design-matrix feature list.

    Args:
        feature_names: Ordered feature names corresponding to columns of the
            tabular design matrix ``X``.
        constraints: Either a ``{feature_name: -1 | 0 | 1}`` mapping or a
            pre-aligned sequence of integers in ``{-1, 0, 1}``.

    Returns:
        List of integers in ``{-1, 0, 1}`` of length ``len(feature_names)``.
    """
    n_features = len(feature_names)
    if n_features == 0 or constraints is None:
        return [0] * n_features

    if not isinstance(constraints, Mapping):
        vec: list[int] = []
        for idx in range(n_features):
            raw = int(constraints[idx]) if idx < len(constraints) else 0
            if raw not in (-1, 0, 1):
                raise ValueError(
                    f"Monotonic constraint at index {idx} must be -1, 0, or 1; got {raw}."
                )
            vec.append(raw)
        return vec

    validated = validate_monotonic_constraints_map(constraints)
    if not validated:
        return [0] * n_features
    return [int(validated.get(name, 0)) for name in feature_names]


def resolve_model_monotonic_vector(
    feature_names: Sequence[str],
    params: Mapping[str, Any] | None = None,
    ctx_constraints: Mapping[str, int] | None = None,
    *,
    recursive_lags_nonnegative: bool = False,
) -> list[int]:
    """Extract and align the monotonic constraint vector for a model's ``fit()`` call.

    Checks ``params["monotonic_constraints"]``, ``params["monotone_constraints"]``,
    and ``ctx_constraints`` (from ``ModelContext.monotonic_constraints``).
    When ``recursive_lags_nonnegative=True`` and at least one feature has a
    non-zero constraint, autoregressive target lags (``lag_<k>``) default to
    ``+1`` so multi-step recursive rollouts preserve exogenous monotonicity
    by composition of monotone functions.

    Args:
        feature_names: Ordered feature names of the model's design matrix.
        params: Model parameter dictionary (`self.params`).
        ctx_constraints: Context-level monotonic constraints (`self.ctx.monotonic_constraints`).
        recursive_lags_nonnegative: Whether to constrain unmapped ``lag_<k>``
            features to ``+1`` when any exogenous constraint is active.

    Returns:
        List of integers in ``{-1, 0, 1}`` aligned to ``feature_names``.
    """
    raw_param: Any = None
    if params:
        raw_param = params.get("monotonic_constraints")
        if raw_param is None:
            raw_param = params.get("monotone_constraints")

    if isinstance(raw_param, (list, tuple)):
        vec = build_monotonic_constraint_vector(feature_names, raw_param)
    else:
        merged = resolve_monotonic_constraints(
            ctx_constraints,
            raw_param if isinstance(raw_param, Mapping) else None,
        )
        vec = build_monotonic_constraint_vector(feature_names, merged)

    if recursive_lags_nonnegative and any(v != 0 for v in vec):
        vec = [
            1 if (v == 0 and fname.startswith("lag_") and fname[4:].isdigit()) else v
            for fname, v in zip(feature_names, vec, strict=False)
        ]
    return vec


def fit_sign_constrained_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    constraint_vec: Sequence[int] | None = None,
) -> np.ndarray:
    """Fit an intercept-augmented L2-regularized linear model with sign constraints.

    Follows the convention of ``RegressionLagsModel._Ridge`` where column ``0``
    of the augmented design matrix is the unpenalized intercept and columns
    ``1..d`` correspond to the ``d`` features in ``x``.

    When ``constraint_vec`` is empty or all zeros, uses the exact normal
    equations ``(X'X + alpha * I_0)^(-1) X'y``. When any entry in
    ``constraint_vec`` is ``-1`` or ``+1``, solves the equivalent stacked
    bounded least-squares problem via :func:`scipy.optimize.lsq_linear` with
    Tikhonov augmentation ``sqrt(alpha) * diag(0, 1, ..., 1)`` and box bounds
    ``[-inf, 0]`` (for ``-1``) or ``[0, +inf]`` (for ``+1``).

    Args:
        x: Design matrix of shape ``(n_samples, n_features)`` (without intercept).
        y: Target vector of shape ``(n_samples,)``.
        alpha: Non-negative L2 regularization strength.
        constraint_vec: Optional sequence of ``{-1, 0, 1}`` of length ``n_features``.

    Returns:
        Weight vector ``w`` of shape ``(n_features + 1,)`` where ``w[0]`` is the
        intercept and ``w[1:]`` are the sign-constrained feature slopes.
    """
    import numpy as np

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n_samples, n_features = x_arr.shape
    xb = np.hstack([np.ones((n_samples, 1), dtype=float), x_arr])
    reg_alpha = max(float(alpha), 0.0)

    c_list = (
        [int(c) for c in constraint_vec[:n_features]]
        if constraint_vec is not None
        else [0] * n_features
    )
    has_constraints = any(c != 0 for c in c_list)

    if not has_constraints:
        penalty = reg_alpha * np.eye(n_features + 1, dtype=float)
        penalty[0, 0] = 0.0
        try:
            return np.linalg.solve(xb.T @ xb + penalty, xb.T @ y_arr)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(xb.T @ xb + penalty, xb.T @ y_arr, rcond=None)[0]

    sqrt_alpha = float(np.sqrt(reg_alpha))
    reg_block = sqrt_alpha * np.eye(n_features + 1, dtype=float)
    reg_block[0, 0] = 0.0
    a_aug = np.vstack([xb, reg_block])
    y_aug = np.concatenate([y_arr, np.zeros(n_features + 1, dtype=float)])

    lb = np.full(n_features + 1, -np.inf, dtype=float)
    ub = np.full(n_features + 1, np.inf, dtype=float)
    for idx, sign in enumerate(c_list):
        if sign == 1:
            lb[idx + 1] = 0.0
        elif sign == -1:
            ub[idx + 1] = 0.0

    from scipy.optimize import lsq_linear

    result = lsq_linear(a_aug, y_aug, bounds=(lb, ub), method="trf")
    w = np.asarray(result.x, dtype=float).copy()
    for idx, sign in enumerate(c_list):
        if (sign == 1 and w[idx + 1] < 0.0) or (sign == -1 and w[idx + 1] > 0.0):
            w[idx + 1] = 0.0
    return w


__all__ = [
    "MONOTONE_SUPPORTED_MODELS",
    "build_monotonic_constraint_vector",
    "fit_sign_constrained_ridge",
    "infer_monotonic_constraints",
    "resolve_model_monotonic_vector",
    "resolve_monotonic_constraints",
    "validate_monotonic_constraints_map",
]
