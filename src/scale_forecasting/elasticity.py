"""Promo & Price Elasticity Decomposition (Robinson / Double-ML Partialing-Out).

Separates true causal promotional and price lift from seasonal confounding, post-promotion
pull-forward (pantry-loading) troughs, and cross-SKU cannibalization dips using the partially
linear Robinson (1988) / Chernozhukov et al. (2018) cross-fitted Double/Debiased Machine
Learning (DML) estimator:

1. **Partially Linear Structural Model:**
   $$y_{i,t} = g_0(W_{i,t}) + \\theta_{\\text{promo}} D_{i,t}
     + \\sum_{k=1}^{L} \\theta_{\\text{lag},k} D_{i,t-k}
     + \\theta_{\\text{cannib}} P^{\\text{peer}}_{i,t} + \\varepsilon_{i,t}$$
   where $W_{i,t}$ contains seasonal Fourier harmonics, day-of-week indicators, secular trend,
   and holiday-window controls, $D_{i,t}$ is the active promotional or price-discount treatment,
   $D_{i,t-k}$ are post-promo lag indicators capturing pantry-loading pull-forward dips, and
   $P^{\\text{peer}}_{i,t}$ is concurrent promotional intensity across peer SKUs in the panel.
2. **Cross-Fitted Residualization (Neyman Orthogonality):**
   Out-of-fold nuisance regressions estimate $\\hat{\\ell}(W) \\approx E[y \\mid W]$ and
   $\\hat{m}(W) \\approx E[T \\mid W]$ across $K$ stratified folds, producing orthogonalized
   residuals $\\tilde{y} = y - \\hat{\\ell}(W)$ and $\\tilde{T} = T - \\hat{m}(W)$.
3. **Second-Stage Structural Decomposition:**
   Sign-constrained least squares (`scipy.optimize.lsq_linear`) on $(\\tilde{T}, \\tilde{y})$
   recovers $\\hat{\\theta}_{\\text{DML}}$ free of seasonal confounding bias, while a naive
   uncontrolled OLS regression of $y$ on $D$ is computed in parallel to quantify exact
   confounding bias (`confounding_bias_pct`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "PROMO_ELASTICITY_COLUMNS",
    "compute_promo_elasticity",
    "compute_promo_elasticity_decomposition",
    "make_confounded_promo_dgp",
    "plot_promo_elasticity",
    "promo_elasticity_df",
    "summarize_promo_elasticity",
]

PROMO_ELASTICITY_COLUMNS: tuple[str, ...] = (
    "ts_id",
    "ds",
    "y",
    "baseline_demand",
    "baseline_yhat",
    "net_causal_lift",
    "incremental_promo_lift",
    "cannibalization_dip",
    "cannibalization_effect",
    "pull_forward_dip",
    "net_incremental_units",
    "net_causal_elasticity",
    "promo_lift_pct",
    "naive_ols_lift",
    "confounding_bias_pct",
)

_PROMO_CANDIDATES: tuple[str, ...] = (
    "promo",
    "promotion",
    "onpromotion",
    "promo_flag",
    "promo_depth",
    "discount",
    "discount_pct",
    "markdown",
    "deal",
    "campaign",
    "rebate",
    "coupon",
)

_PRICE_CANDIDATES: tuple[str, ...] = (
    "price",
    "sell_price",
    "unit_price",
    "price_index",
    "price_idx",
    "msrp",
    "list_price",
    "net_price",
    "avg_price",
)


def _detect_treatment_cols(
    df: pd.DataFrame,
    *,
    promo_col: str | None = None,
    price_col: str | None = None,
    exclude_cols: set[str] | None = None,
) -> tuple[str | None, str | None]:
    """Auto-detect promotional and price columns from ``df`` when not explicitly provided."""
    excl = exclude_cols or set()
    cols_lower = {str(c).lower(): str(c) for c in df.columns if str(c) not in excl}

    resolved_promo = promo_col if (promo_col and promo_col in df.columns) else None
    if resolved_promo is None:
        for cand in _PROMO_CANDIDATES:
            if cand in cols_lower:
                resolved_promo = cols_lower[cand]
                break
        if resolved_promo is None:
            for c_low, c_orig in cols_lower.items():
                if any(tok in c_low for tok in ("promo", "discount", "markdown", "deal")):
                    resolved_promo = c_orig
                    break

    resolved_price = price_col if (price_col and price_col in df.columns) else None
    if resolved_price is None:
        for cand in _PRICE_CANDIDATES:
            if cand in cols_lower and cols_lower[cand] != resolved_promo:
                resolved_price = cols_lower[cand]
                break
        if resolved_price is None:
            for c_low, c_orig in cols_lower.items():
                if c_orig != resolved_promo and "price" in c_low and "cost" not in c_low:
                    resolved_price = c_orig
                    break

    return resolved_promo, resolved_price


def _build_seasonal_nuisance_matrix(
    dates: pd.Series,
    extra_controls: pd.DataFrame | None = None,
) -> np.ndarray:
    """Build nuisance control matrix $W_t$ (trend, day-of-week, and Fourier harmonics)."""
    dt = pd.to_datetime(dates)
    n = len(dt)
    t_idx = np.arange(n, dtype=np.float64)
    t_norm = (t_idx - np.mean(t_idx)) / max(1.0, float(np.std(t_idx)))

    # Day-of-week one-hot (6 dummies to avoid collinearity with intercept)
    dow = dt.dt.dayofweek.to_numpy(dtype=np.int64)
    dow_cols = [(dow == d).astype(np.float64) for d in range(6)]

    # Multi-harmonic Fourier basis for weekly (7d), monthly (30.4375d), and annual/sub-annual cycles
    doy = dt.dt.dayofyear.to_numpy(dtype=np.float64)
    fourier_cols: list[np.ndarray] = []
    for period, orders in ((7.0, (1, 2)), (30.4375, (1, 2)), (91.25, (1, 2)), (365.25, (1, 2, 3))):
        for k in orders:
            angle = 2.0 * np.pi * k * t_idx / period
            fourier_cols.append(np.sin(angle))
            fourier_cols.append(np.cos(angle))
    # Also include day-of-year harmonic phase
    for k in (1, 2):
        angle_doy = 2.0 * np.pi * k * doy / 365.25
        fourier_cols.append(np.sin(angle_doy))
        fourier_cols.append(np.cos(angle_doy))

    cols: list[np.ndarray] = [
        np.ones(n, dtype=np.float64),
        t_norm,
        t_norm**2,
        *dow_cols,
        *fourier_cols,
    ]
    if extra_controls is not None and not extra_controls.empty:
        for c in extra_controls.columns:
            vals = (
                pd.to_numeric(extra_controls[c], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            v_std = float(np.std(vals))
            if v_std > 1e-9:
                cols.append((vals - float(np.mean(vals))) / v_std)
    return np.column_stack(cols)


def _cross_fitted_residuals(
    W: np.ndarray,
    targets: np.ndarray,
    *,
    n_folds: int = 5,
    l2_alpha: float = 1.0,
) -> np.ndarray:
    """Compute cross-fitted Robinson / DML residuals $targets - \\hat{E}[targets \\mid W]$."""
    n = W.shape[0]
    p = W.shape[1]
    eff_folds = max(2, min(int(n_folds), max(2, n // 6)))
    preds = np.zeros_like(targets, dtype=np.float64)

    # Interleaved modulo folds ensure every training fold spans the full seasonal & trend support
    fold_ids = np.arange(n) % eff_folds
    reg = np.eye(p, dtype=np.float64) * float(l2_alpha)
    reg[0, 0] = 0.0  # Unpenalized intercept

    for f_id in range(eff_folds):
        val_mask = fold_ids == f_id
        tr_mask = ~val_mask
        if not np.any(tr_mask) or not np.any(val_mask):
            continue
        W_tr = W[tr_mask]
        y_tr = targets[tr_mask]
        gram = W_tr.T @ W_tr + reg
        try:
            coef = np.linalg.solve(gram, W_tr.T @ y_tr)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(gram, W_tr.T @ y_tr, rcond=None)
        preds[val_mask] = W[val_mask] @ coef

    return targets - preds


def _fit_ridge_nuisance(
    W: np.ndarray,
    y: np.ndarray,
    *,
    l2_alpha: float = 1.0,
) -> np.ndarray:
    """Fit penalized ridge nuisance baseline $\\hat{g}_0(W)$ on de-promoted demand."""
    p = W.shape[1]
    reg = np.eye(p, dtype=np.float64) * float(l2_alpha)
    reg[0, 0] = 0.0
    gram = W.T @ W + reg
    try:
        coef = np.linalg.solve(gram, W.T @ y)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(gram, W.T @ y, rcond=None)
    return W @ coef


_MAJOR_HOLIDAY_TOKENS: tuple[str, ...] = (
    "new year",
    "independence",
    "thanksgiving",
    "christmas",
    "easter",
    "good friday",
    "boxing",
    "national day",
    "lunar new year",
    "mid-autumn",
    "diwali",
    "eid",
)


def _build_holiday_nuisance_controls(
    dates: pd.Series,
    holidays: Sequence[Any] | pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Build exact holiday-day, major-holiday, and pre/post holiday-window nuisance columns."""
    if holidays is None:
        return None
    if isinstance(holidays, pd.DataFrame) and holidays.empty:
        return None
    if isinstance(holidays, Sequence) and not isinstance(holidays, (str, bytes)) and not holidays:
        return None

    from .holiday_window import (
        HolidayWindowConfig,
        build_holiday_window_features,
        extract_holiday_day_ordinals,
    )

    dt_idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(dt_idx) == 0:
        return None
    if dt_idx.tz is not None:
        dt_idx = dt_idx.tz_localize(None)
    norm_dates = dt_idx.normalize()
    epoch = pd.Timestamp("1970-01-01")
    d_ord = ((norm_dates - epoch) // pd.Timedelta(days=1)).to_numpy(dtype=np.int64)

    h_ord = extract_holiday_day_ordinals(holidays, ref_dates=norm_dates)
    if len(h_ord) == 0:
        return None

    is_holiday = np.isin(d_ord, h_ord).astype(np.float64)
    is_major_holiday = is_holiday.copy()

    if isinstance(holidays, (list, tuple)):
        country_codes = [
            str(x).strip().upper()
            for x in holidays
            if isinstance(x, str) and 2 <= len(x.strip()) <= 3 and x.strip().isalpha()
        ]
        explicit_items = [
            x
            for x in holidays
            if not (isinstance(x, str) and 2 <= len(x.strip()) <= 3 and x.strip().isalpha())
        ]
        if country_codes:
            import holidays as holidays_pkg

            y_min = int(norm_dates.min().year) - 1
            y_max = int(norm_dates.max().year) + 1
            years = range(y_min, y_max + 1)
            major_dates: list[pd.Timestamp] = []
            for code in country_codes:
                cal = holidays_pkg.country_holidays(code, years=years)
                for day, name in cal.items():
                    name_low = str(name).lower()
                    if (
                        any(tok in name_low for tok in _MAJOR_HOLIDAY_TOKENS)
                        and "juneteenth" not in name_low
                    ):
                        major_dates.append(pd.Timestamp(day))
            if explicit_items:
                for ts in pd.to_datetime(pd.Index(explicit_items)):
                    major_dates.append(pd.Timestamp(ts))
            if major_dates:
                m_idx = pd.DatetimeIndex(major_dates)
                if m_idx.tz is not None:
                    m_idx = m_idx.tz_localize(None)
                m_ord = ((m_idx.normalize() - epoch) // pd.Timedelta(days=1)).to_numpy(
                    dtype=np.int64
                )
                is_major_holiday = np.isin(d_ord, np.unique(m_ord)).astype(np.float64)

    hw = build_holiday_window_features(
        norm_dates,
        holidays,
        HolidayWindowConfig(pre_holiday_days=3, post_holiday_days=2, kernel="linear"),
    )
    n_obs = len(dates)
    return pd.DataFrame(
        {
            "_hol_exact": is_holiday,
            "_hol_major": is_major_holiday,
            "_hol_pre": hw.get("pre_holiday_window", np.zeros(n_obs, dtype=np.float64)),
            "_hol_post": hw.get("post_holiday_window", np.zeros(n_obs, dtype=np.float64)),
        },
        index=dates.index,
    )


def compute_promo_elasticity_decomposition(
    df: pd.DataFrame | tuple[pd.DataFrame, Any],
    *,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    promo_col: str | None = None,
    price_col: str | None = None,
    control_cols: Sequence[str] | None = None,
    confounders: Sequence[str] | None = None,
    freq: str | None = None,
    holidays: Sequence[str] | None = None,
    cfg: Any = None,
    pull_forward_lags: int = 3,
    n_folds: int = 5,
    l2_alpha: float = 1.0,
) -> pd.DataFrame:
    """Decompose observed demand into baseline, causal promo lift, pull-forward dip,
    and cannibalization.

    Uses cross-fitted Robinson (1988) / Chernozhukov et al. (2018) Double/Debiased ML
    partialing-out to orthogonalize both demand $y_{i,t}$ and promotional/price treatments
    against seasonal Fourier harmonics, day-of-week effects, and trend controls before
    estimating structural treatment effects.

    Args:
        df: Historical panel DataFrame containing ``ts_id_col``, ``date_col``, ``target_col``, and
            at least one promotional or price covariate.
        ts_id_col: Series identifier column name.
        date_col: Timestamp column name.
        target_col: Target demand column name.
        promo_col: Explicit promotional indicator/depth column name (auto-detected if ``None``).
        price_col: Explicit price column name (auto-detected if ``None``).
        control_cols: Optional additional nuisance control columns (e.g. holiday window features).
        confounders: Alias for ``control_cols`` (additional nuisance confounder columns).
        freq: Optional series frequency hint.
        holidays: Optional country holiday codes to include in nuisance controls.
        cfg: Optional :class:`~scale_forecasting.config.RunConfig` or mapping from which
            ``features.holidays`` is resolved when ``holidays`` is ``None``.
        pull_forward_lags: Number of post-promotion lag steps $L$ used to estimate pantry-loading
            pull-forward dips.
        n_folds: Number of cross-fitting folds for DML residualization.
        l2_alpha: L2 regularization penalty on nuisance seasonal/trend controls.

    Returns:
        DataFrame with canonical columns :data:`PROMO_ELASTICITY_COLUMNS`.
    """
    if isinstance(df, tuple):
        df = df[0]
    if df is None or df.empty:
        return pd.DataFrame(columns=list(PROMO_ELASTICITY_COLUMNS))

    eff_controls = list(dict.fromkeys([*(control_cols or ()), *(confounders or ())]))
    eff_holidays: Sequence[Any] | pd.DataFrame | None = holidays
    if eff_holidays is None and cfg is not None:
        feat_obj = getattr(cfg, "features", None)
        if feat_obj is not None:
            eff_holidays = getattr(feat_obj, "holidays", None)
        elif isinstance(cfg, dict):
            eff_holidays = cfg.get("features", {}).get("holidays")
    _ = freq

    work = df.copy()
    if ts_id_col not in work.columns:
        work[ts_id_col] = "series_001"
    if date_col not in work.columns or target_col not in work.columns:
        return pd.DataFrame(columns=list(PROMO_ELASTICITY_COLUMNS))

    # Filter to historical rows where target is observed
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work[work[target_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=list(PROMO_ELASTICITY_COLUMNS))

    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values([ts_id_col, date_col]).reset_index(drop=True)

    excl = {ts_id_col, date_col, target_col, *eff_controls}
    eff_promo_col, eff_price_col = _detect_treatment_cols(
        work,
        promo_col=promo_col,
        price_col=price_col,
        exclude_cols=excl,
    )

    # Construct per-row active promotional intensity D_{i,t} >= 0.
    # If promo_col is present, use its non-negative intensity.
    # If price_col is also/instead present, compute relative discount depth.
    work["_promo_intensity"] = 0.0
    if eff_promo_col is not None and eff_promo_col in work.columns:
        p_vals = (
            pd.to_numeric(work[eff_promo_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        work["_promo_intensity"] = np.maximum(0.0, p_vals)

    work["_price_discount"] = 0.0
    if eff_price_col is not None and eff_price_col in work.columns:
        for _tid, grp in work.groupby(ts_id_col, sort=False):
            pr = (
                pd.to_numeric(grp[eff_price_col], errors="coerce")
                .ffill()
                .bfill()
                .fillna(1.0)
                .to_numpy(dtype=np.float64)
            )
            reg_price = float(np.nanpercentile(pr, 75)) if len(pr) > 0 else 1.0
            disc = np.maximum(0.0, (reg_price - pr) / max(1e-6, reg_price))
            work.loc[grp.index, "_price_discount"] = disc
        promo_std = float(np.nanstd(work["_promo_intensity"].to_numpy(dtype=np.float64)))
        if eff_promo_col is None or promo_std < 1e-8:
            work["_promo_intensity"] = work["_price_discount"]
            eff_promo_col = None

    # Compute peer promotion intensity across other series on the same date
    n_series = int(work[ts_id_col].nunique())
    if n_series >= 2:
        date_sum = work.groupby(date_col)["_promo_intensity"].transform("sum")
        date_cnt = work.groupby(date_col)["_promo_intensity"].transform("count")
        peer_mean = np.where(
            date_cnt > 1,
            (date_sum - work["_promo_intensity"]) / np.maximum(1, date_cnt - 1),
            0.0,
        )
        # Cannibalization occurs when peers promote more strongly than the focal SKU
        work["_peer_promo"] = np.maximum(0.0, peer_mean - work["_promo_intensity"])
    else:
        work["_peer_promo"] = 0.0

    eff_lags = max(1, int(pull_forward_lags))
    out_frames: list[pd.DataFrame] = []

    for tid, grp in work.groupby(ts_id_col, sort=False):
        grp = grp.sort_values(date_col).reset_index(drop=True)
        n_t = len(grp)
        y_vec = grp[target_col].to_numpy(dtype=np.float64)
        d_vec = grp["_promo_intensity"].to_numpy(dtype=np.float64)
        disc_vec = grp["_price_discount"].to_numpy(dtype=np.float64)
        peer_vec = grp["_peer_promo"].to_numpy(dtype=np.float64)

        has_treatment = float(np.std(d_vec)) > 1e-9 or float(np.std(disc_vec)) > 1e-9
        if not has_treatment or n_t < 12:
            out_frames.append(
                pd.DataFrame(
                    {
                        "ts_id": str(tid),
                        "ds": grp[date_col],
                        "y": y_vec,
                        "baseline_demand": y_vec.copy(),
                        "baseline_yhat": y_vec.copy(),
                        "net_causal_lift": np.zeros(n_t, dtype=np.float64),
                        "incremental_promo_lift": np.zeros(n_t, dtype=np.float64),
                        "cannibalization_dip": np.zeros(n_t, dtype=np.float64),
                        "cannibalization_effect": np.zeros(n_t, dtype=np.float64),
                        "pull_forward_dip": np.zeros(n_t, dtype=np.float64),
                        "net_incremental_units": np.zeros(n_t, dtype=np.float64),
                        "net_causal_elasticity": np.zeros(n_t, dtype=np.float64),
                        "promo_lift_pct": np.zeros(n_t, dtype=np.float64),
                        "naive_ols_lift": np.zeros(n_t, dtype=np.float64),
                        "confounding_bias_pct": np.zeros(n_t, dtype=np.float64),
                    }
                )
            )
            continue

        # Build post-promo pull-forward lag features (D_{t} == 0 and D_{t-k} > 0)
        unpromoted_mask = (d_vec <= 1e-9).astype(np.float64)
        lag_cols: list[np.ndarray] = []
        for k in range(1, eff_lags + 1):
            d_lag = np.zeros(n_t, dtype=np.float64)
            if n_t > k:
                d_lag[k:] = d_vec[:-k] * unpromoted_mask[k:]
            lag_cols.append(d_lag)

        # Assemble treatment matrix T = [d_vec, (optional disc_vec), lag_1..lag_L, peer_vec]
        treat_list: list[np.ndarray] = [d_vec]
        lb_list: list[float] = [0.0]
        ub_list: list[float] = [np.inf]

        include_separate_disc = (
            eff_promo_col is not None
            and eff_price_col is not None
            and float(np.std(disc_vec)) > 1e-9
        )
        if include_separate_disc:
            treat_list.append(disc_vec)
            lb_list.append(0.0)
            ub_list.append(np.inf)

        lag_start_idx = len(treat_list)
        for l_col in lag_cols:
            treat_list.append(l_col)
            lb_list.append(-np.inf)
            ub_list.append(0.0)

        has_peer = float(np.std(peer_vec)) > 1e-9
        peer_idx: int | None = None
        if has_peer:
            peer_idx = len(treat_list)
            treat_list.append(peer_vec)
            lb_list.append(-np.inf)
            ub_list.append(0.0)

        T_mat = np.column_stack(treat_list)

        # Build nuisance control matrix W_t (including explicit controls and holiday windows)
        ctrl_parts: list[pd.DataFrame] = []
        if eff_controls:
            present_ctrls = [c for c in eff_controls if c in grp.columns]
            if present_ctrls:
                ctrl_parts.append(grp[present_ctrls])
        hol_ctrl_df = _build_holiday_nuisance_controls(grp[date_col], eff_holidays)
        if hol_ctrl_df is not None:
            ctrl_parts.append(hol_ctrl_df)
        extra_ctrl_df = pd.concat(ctrl_parts, axis=1) if ctrl_parts else None
        W_mat = _build_seasonal_nuisance_matrix(grp[date_col], extra_controls=extra_ctrl_df)

        # Cross-fitted DML residualization: partial out W_mat from both y_vec and T_mat
        residuals = _cross_fitted_residuals(
            W_mat,
            np.column_stack([y_vec, T_mat]),
            n_folds=n_folds,
            l2_alpha=l2_alpha,
        )
        y_tilde = residuals[:, 0]
        T_tilde = residuals[:, 1:]

        # Fit sign-constrained second-stage DML estimator on (T_tilde, y_tilde)
        from scipy.optimize import lsq_linear

        res_dml = lsq_linear(
            T_tilde,
            y_tilde,
            bounds=(np.asarray(lb_list, dtype=np.float64), np.asarray(ub_list, dtype=np.float64)),
            lsmr_tol="auto",
        )
        theta_dml = np.asarray(res_dml.x, dtype=np.float64)

        # Compute structural component series
        causal_lift = theta_dml[0] * d_vec
        if include_separate_disc:
            causal_lift = causal_lift + theta_dml[1] * disc_vec

        pull_forward_dip = np.zeros(n_t, dtype=np.float64)
        for idx_l, l_col in enumerate(lag_cols):
            pull_forward_dip += theta_dml[lag_start_idx + idx_l] * l_col
        pull_forward_dip = np.minimum(0.0, pull_forward_dip)

        cannib_dip = (
            np.minimum(0.0, theta_dml[peer_idx] * peer_vec)
            if peer_idx is not None
            else np.zeros(n_t, dtype=np.float64)
        )

        # Reconstruct de-promoted structural baseline g_0(W_t)
        total_treatment_effect = causal_lift + pull_forward_dip + cannib_dip
        y_depromoted = y_vec - total_treatment_effect
        baseline_demand = _fit_ridge_nuisance(W_mat, y_depromoted, l2_alpha=l2_alpha)
        if np.all(y_vec >= 0.0):
            baseline_demand = np.maximum(0.0, baseline_demand)

        # Naive uncontrolled OLS regression of y on [1, d_vec] (ignores seasonal confounding W_t)
        d_centered = d_vec - float(np.mean(d_vec))
        d_var = float(np.dot(d_centered, d_centered))
        if d_var > 1e-9:
            theta_ols = max(0.0, float(np.dot(d_centered, y_vec - float(np.mean(y_vec))) / d_var))
        else:
            theta_ols = float(theta_dml[0])
        naive_ols_lift = theta_ols * d_vec
        if include_separate_disc:
            disc_c = disc_vec - float(np.mean(disc_vec))
            disc_var = float(np.dot(disc_c, disc_c))
            if disc_var > 1e-9:
                theta_ols_disc = max(
                    0.0, float(np.dot(disc_c, y_vec - float(np.mean(y_vec))) / disc_var)
                )
                naive_ols_lift = naive_ols_lift + theta_ols_disc * disc_vec

        net_incremental = causal_lift + pull_forward_dip + cannib_dip
        series_floor = max(1.0, 0.05 * float(np.nanmean(np.abs(y_vec))))
        safe_base = np.maximum(series_floor, np.abs(baseline_demand))
        net_causal_elasticity = net_incremental / safe_base
        promo_lift_pct = np.where(d_vec > 1e-9, 100.0 * causal_lift / safe_base, 0.0)

        # Confounding bias (%): how much naive OLS over-estimates true causal lift
        confounding_bias_pct = np.where(
            causal_lift > 1e-6,
            100.0 * (naive_ols_lift - causal_lift) / np.maximum(1e-6, causal_lift),
            0.0,
        )

        out_frames.append(
            pd.DataFrame(
                {
                    "ts_id": str(tid),
                    "ds": grp[date_col],
                    "y": np.round(y_vec, 4),
                    "baseline_demand": np.round(baseline_demand, 4),
                    "baseline_yhat": np.round(baseline_demand, 4),
                    "net_causal_lift": np.round(causal_lift, 4),
                    "incremental_promo_lift": np.round(causal_lift, 4),
                    "cannibalization_dip": np.round(cannib_dip, 4),
                    "cannibalization_effect": np.round(cannib_dip, 4),
                    "pull_forward_dip": np.round(pull_forward_dip, 4),
                    "net_incremental_units": np.round(net_incremental, 4),
                    "net_causal_elasticity": np.round(net_causal_elasticity, 6),
                    "promo_lift_pct": np.round(promo_lift_pct, 4),
                    "naive_ols_lift": np.round(naive_ols_lift, 4),
                    "confounding_bias_pct": np.round(confounding_bias_pct, 4),
                }
            )
        )

    if not out_frames:
        return pd.DataFrame(columns=list(PROMO_ELASTICITY_COLUMNS))
    return pd.concat(out_frames, ignore_index=True)


# Public aliases for SDK / diagnostic ergonomics
promo_elasticity_df = compute_promo_elasticity_decomposition
compute_promo_elasticity = compute_promo_elasticity_decomposition


def summarize_promo_elasticity(decomp_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level promo elasticity decomposition frame to one summary row per ``ts_id``."""
    summary_cols = [
        "ts_id",
        "promo_days",
        "mean_baseline_demand",
        "mean_causal_promo_lift",
        "mean_naive_ols_lift",
        "confounding_bias_pct",
        "total_gross_causal_lift",
        "total_pull_forward_dip",
        "total_cannibalization_dip",
        "total_net_incremental_units",
        "net_to_gross_lift_ratio",
    ]
    if decomp_df is None or decomp_df.empty or "ts_id" not in decomp_df.columns:
        return pd.DataFrame(columns=summary_cols)

    rows: list[dict[str, Any]] = []
    for tid, grp in decomp_df.groupby("ts_id", sort=False):
        promo_mask = grp["net_causal_lift"] > 1e-6
        n_promo = int(promo_mask.sum())
        mean_base = float(grp["baseline_demand"].mean())
        mean_causal = float(grp.loc[promo_mask, "net_causal_lift"].mean()) if n_promo > 0 else 0.0
        mean_ols = float(grp.loc[promo_mask, "naive_ols_lift"].mean()) if n_promo > 0 else 0.0
        bias_pct = (
            100.0 * (mean_ols - mean_causal) / max(1e-6, mean_causal) if mean_causal > 1e-6 else 0.0
        )
        tot_gross = float(grp["net_causal_lift"].sum())
        tot_pf = float(grp["pull_forward_dip"].sum())
        tot_can = float(grp["cannibalization_dip"].sum())
        tot_net = float(grp["net_incremental_units"].sum())
        ratio = tot_net / max(1e-6, tot_gross) if tot_gross > 1e-6 else 0.0
        rows.append(
            {
                "ts_id": str(tid),
                "promo_days": n_promo,
                "mean_baseline_demand": round(mean_base, 4),
                "mean_causal_promo_lift": round(mean_causal, 4),
                "mean_naive_ols_lift": round(mean_ols, 4),
                "confounding_bias_pct": round(bias_pct, 2),
                "total_gross_causal_lift": round(tot_gross, 4),
                "total_pull_forward_dip": round(tot_pf, 4),
                "total_cannibalization_dip": round(tot_can, 4),
                "total_net_incremental_units": round(tot_net, 4),
                "net_to_gross_lift_ratio": round(ratio, 4),
            }
        )
    return pd.DataFrame(rows, columns=summary_cols)


def plot_promo_elasticity(
    decomp_df: pd.DataFrame,
    *,
    ts_id: str | None = None,
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12.0, 5.5),
) -> Any:
    """Plot baseline demand, causal promotional lift, pull-forward dip, and cannibalization dip."""
    import matplotlib.pyplot as plt

    _ = ax
    if decomp_df is None or decomp_df.empty:
        fig, ax_empty = plt.subplots(figsize=figsize)
        ax_empty.set_title(title or "Promo Elasticity Decomposition (empty)")
        return fig

    sub = decomp_df.copy()
    if ts_id is not None and "ts_id" in sub.columns:
        matched = sub[sub["ts_id"].astype(str) == str(ts_id)].copy()
        if matched.empty:
            matched = sub[sub["ts_id"].astype(str).str.lower() == str(ts_id).lower()].copy()
        if matched.empty:
            first_id = str(sub["ts_id"].iloc[0])
            matched = sub[sub["ts_id"].astype(str) == first_id].copy()
            ts_id = first_id
        sub = matched
    elif "ts_id" in sub.columns:
        first_id = str(sub["ts_id"].iloc[0])
        sub = sub[sub["ts_id"].astype(str) == first_id].copy()
        ts_id = first_id

    sub = sub.sort_values("ds").reset_index(drop=True)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0]}
    )

    ax_top.plot(sub["ds"], sub["y"], label="Observed Demand (y)", color="#1f2937", linewidth=1.4)
    ax_top.plot(
        sub["ds"],
        sub["baseline_demand"],
        label="DML Structural Baseline g0(W)",
        color="#2563eb",
        linestyle="--",
        linewidth=1.5,
    )
    ax_top.set_ylabel("Units")
    ax_top.set_title(title or f"Double-ML Promo Elasticity Decomposition — {ts_id or 'Panel'}")
    ax_top.legend(loc="upper left")
    ax_top.grid(True, alpha=0.25)

    ax_bot.bar(
        sub["ds"],
        sub["net_causal_lift"],
        label="Net Causal Promo Lift (+)",
        color="#16a34a",
        alpha=0.8,
    )
    ax_bot.bar(
        sub["ds"],
        sub["pull_forward_dip"],
        label="Post-Promo Pull-Forward Dip (-)",
        color="#dc2626",
        alpha=0.8,
    )
    ax_bot.bar(
        sub["ds"],
        sub["cannibalization_dip"],
        bottom=sub["pull_forward_dip"],
        label="Cross-SKU Cannibalization Dip (-)",
        color="#d97706",
        alpha=0.8,
    )
    ax_bot.axhline(0.0, color="#6b7280", linewidth=0.8)
    ax_bot.set_ylabel("Incremental Units")
    ax_bot.set_xlabel("Date")
    ax_bot.legend(loc="upper left")
    ax_bot.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def make_confounded_promo_dgp(
    *,
    n_series: int = 3,
    n_days: int = 180,
    true_promo_lift: float = 40.0,
    seasonal_amplitude: float = 35.0,
    pull_forward_fraction: float = 0.25,
    cannibalization_fraction: float = 0.15,
    true_pull_forward_dip: float | None = None,
    true_cannibalization_dip: float | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a confounded seasonal + promotional retail DGP with known structural parameters.

    Promotions are intentionally scheduled with $4\\times$ higher probability during seasonal peaks
    ($\\sin(2\\pi t / 7) + \\sin(2\\pi t / 30.4375) > 0.35$), causing naive uncontrolled OLS to
    confound seasonal peaks with promotional lift and over-estimate promo lift by $\\ge 25\\%$.
    Includes post-promotion pantry-loading pull-forward dips and cross-SKU cannibalization dips.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    t_idx = np.arange(n_days, dtype=np.float64)

    # Seasonal wave shared across series (weekly + monthly harmonics)
    seasonal_wave = 0.55 * np.sin(2.0 * np.pi * t_idx / 7.0) + 0.45 * np.sin(
        2.0 * np.pi * t_idx / 30.4375
    )
    peak_mask = seasonal_wave > 0.20

    # First generate promo schedules across all series so we can compute peer cannibalization
    promo_by_series: list[np.ndarray] = []
    for _i in range(n_series):
        # High promo probability during seasonal peaks (confounding!), low during troughs
        prob = np.where(peak_mask, 0.48, 0.08)
        promo = (rng.uniform(0.0, 1.0, size=n_days) < prob).astype(np.float64)
        promo_by_series.append(promo)

    promo_mat = np.column_stack(promo_by_series)  # shape (n_days, n_series)
    true_pf_dip = (
        -abs(float(true_pull_forward_dip))
        if true_pull_forward_dip is not None
        else -abs(float(true_promo_lift) * float(pull_forward_fraction))
    )
    true_cannib_dip = (
        -abs(float(true_cannibalization_dip))
        if true_cannibalization_dip is not None
        else -abs(float(true_promo_lift) * float(cannibalization_fraction))
    )

    frames: list[pd.DataFrame] = []
    for i in range(n_series):
        ts_id = f"sku_{i + 1:03d}"
        base_level = 120.0 + 10.0 * i
        trend = 0.08 * t_idx
        baseline = base_level + trend + float(seasonal_amplitude) * seasonal_wave

        d_vec = promo_mat[:, i]
        # Post-promo pull-forward dip on the day immediately following an active promo
        d_lag1 = np.zeros(n_days, dtype=np.float64)
        d_lag1[1:] = d_vec[:-1] * (1.0 - d_vec[1:])

        # Peer promo intensity when focal SKU is not promoted
        if n_series >= 2:
            peer_mean = (np.sum(promo_mat, axis=1) - d_vec) / max(1, n_series - 1)
            peer_active = np.maximum(0.0, peer_mean - d_vec)
        else:
            peer_active = np.zeros(n_days, dtype=np.float64)

        causal_lift = float(true_promo_lift) * d_vec
        pf_dip = true_pf_dip * d_lag1
        cannib_dip = true_cannib_dip * peer_active
        noise = rng.normal(0.0, 2.0, size=n_days)

        y = np.maximum(0.0, baseline + causal_lift + pf_dip + cannib_dip + noise)
        frames.append(
            pd.DataFrame(
                {
                    "ts_id": ts_id,
                    "ds": dates,
                    "y": np.round(y, 4),
                    "promo_flag": d_vec,
                    "true_baseline": np.round(baseline, 4),
                    "true_causal_lift": np.round(causal_lift, 4),
                    "true_pull_forward_dip": np.round(pf_dip, 4),
                    "true_cannibalization_dip": np.round(cannib_dip, 4),
                }
            )
        )

    panel_df = pd.concat(frames, ignore_index=True)
    ground_truth = {
        "true_promo_lift": float(true_promo_lift),
        "true_pull_forward_dip": float(true_pf_dip),
        "true_cannibalization_dip": float(true_cannib_dip),
        "seasonal_amplitude": float(seasonal_amplitude),
    }
    return panel_df, ground_truth
