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

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "PROMO_ELASTICITY_COLUMNS",
    "compute_promo_elasticity",
    "compute_promo_elasticity_decomposition",
    "detect_treatment_cols",
    "estimate_promo_elasticity",
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
    "adstock_effect",
    "halo_lift",
    "halo_effect",
    "cannibalization_dip",
    "cannibalization_effect",
    "pull_forward_dip",
    "net_incremental_units",
    "net_causal_elasticity",
    "promo_lift_pct",
    "naive_ols_lift",
    "confounding_bias_pct",
    "top_cannibal_ts_id",
    "cannibal_similarity_score",
    "top_halo_ts_id",
    "halo_similarity_score",
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


detect_treatment_cols = _detect_treatment_cols


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


def _compute_adstock_vector(d_vec: np.ndarray, decay: float) -> np.ndarray:
    """Apply recursive geometric ad-stock filter $A_t = D_t + \\lambda A_{t-1}$ (pure)."""
    lam = float(np.clip(decay, 0.0, 0.98))
    if lam <= 1e-9 or len(d_vec) == 0:
        return d_vec.copy()
    out = np.empty_like(d_vec, dtype=np.float64)
    acc = 0.0
    for idx, val in enumerate(d_vec):
        acc = float(val) + lam * acc
        out[idx] = acc
    return out


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
    adstock_decay: float | str = 0.0,
    include_halo: bool = True,
    n_folds: int = 5,
    l2_alpha: float = 1.0,
    use_similarity_priors: bool = True,
    similarity_matrix: pd.DataFrame | np.ndarray | None = None,
    similarity_method: str = "hybrid",
) -> pd.DataFrame:
    """Decompose observed demand into baseline, causal promo lift, ad-stock carryover,
    complementary halo lift, pull-forward dip, and substitute cannibalization.

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
        adstock_decay: Geometric promotional ad-stock decay rate $\\lambda \\in [0, 1)$
            ($A_{i,t} = D_{i,t} + \\lambda A_{i,t-1}$) or ``"auto"`` to estimate $\\hat{\\lambda}$
            on orthogonalized DML residuals. Defaults to ``0.0`` (instantaneous promo + lag dips).
        include_halo: When ``True`` (default), partitions peer SKUs into negative substitute
            cannibals ($\\gamma_{ij} < 0$) and positive complementary halo drivers
            ($\\gamma_{ij} > 0$).
        n_folds: Number of cross-fitting folds for DML residualization.
        l2_alpha: L2 regularization penalty on nuisance seasonal/trend controls.
        use_similarity_priors: When ``True`` (default), weights cross-SKU peer promotional
            exposure $P^{\\text{peer}}_{i,t} = \\sum_{j \\ne i} \\tilde{W}_{i,j} P_{j,t}$ by
            pairwise trajectory similarity so substitute SKUs receive higher cannibalization weight
            than unrelated SKUs.
        similarity_matrix: Optional precomputed $(N, N)$ trajectory similarity matrix.
        similarity_method: Similarity kernel method used when computing the similarity prior on
            the fly (default ``"hybrid"``).

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
        elif isinstance(cfg, Mapping):
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

    # Compute peer promotion intensity across other series on the same date, partitioning
    # peers into negative substitutes (cannibalization) and positive complements (halo lift).
    series_ids = [str(x) for x in work[ts_id_col].drop_duplicates().tolist()]
    n_series = len(series_ids)
    top_cannibal_by_series: dict[str, tuple[str | None, float]] = dict.fromkeys(
        series_ids, (None, 0.0)
    )
    top_halo_by_series: dict[str, tuple[str | None, float]] = dict.fromkeys(series_ids, (None, 0.0))
    w_tilde = np.zeros((n_series, n_series), dtype=np.float64)
    w_halo = np.zeros((n_series, n_series), dtype=np.float64)
    adstock_by_series: dict[str, float] = dict.fromkeys(series_ids, 0.0)

    if n_series >= 2:
        if use_similarity_priors:
            sim_prior = similarity_matrix
            if sim_prior is None:
                sim_prior = df.attrs.get("similarity_matrix")
            if sim_prior is None:
                from .similarity import compute_similarity_matrix

                sim_prior = compute_similarity_matrix(
                    work,
                    method=similarity_method,
                    ts_id_col=ts_id_col,
                    date_col=date_col,
                    target_col=target_col,
                ).matrix_df

            if isinstance(sim_prior, pd.DataFrame):
                sim_arr = (
                    sim_prior.reindex(index=series_ids, columns=series_ids)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
            else:
                sim_arr = np.asarray(sim_prior, dtype=np.float64).copy()
                if sim_arr.shape != (n_series, n_series):
                    sim_arr = np.ones((n_series, n_series), dtype=np.float64)
        else:
            sim_arr = np.ones((n_series, n_series), dtype=np.float64)
            np.fill_diagonal(sim_arr, 0.0)

        piv_promo = (
            work.pivot_table(
                index=date_col,
                columns=ts_id_col,
                values="_promo_intensity",
                aggfunc="mean",
            )
            .reindex(columns=series_ids)
            .fillna(0.0)
        )
        promo_mat_aligned = piv_promo.to_numpy(dtype=np.float64)

        # Estimate preliminary own-promo + seasonal residuals per series to detect positive
        # complementary halo peers (t-stat > 1.96) vs negative substitute cannibals.
        halo_affinity = np.zeros((n_series, n_series), dtype=np.float64)
        if include_halo and len(piv_promo) >= 12:
            piv_y = (
                work.pivot_table(
                    index=date_col,
                    columns=ts_id_col,
                    values=target_col,
                    aggfunc="mean",
                )
                .reindex(index=piv_promo.index, columns=series_ids)
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            y_mat_aligned = piv_y.to_numpy(dtype=np.float64)
            W_base = _build_seasonal_nuisance_matrix(
                pd.Series(piv_promo.index, index=range(len(piv_promo)))
            )
            own_resids = np.zeros_like(y_mat_aligned, dtype=np.float64)
            for idx_i in range(n_series):
                d_i = promo_mat_aligned[:, idx_i]
                W_i = np.column_stack([W_base, d_i]) if float(np.std(d_i)) > 1e-9 else W_base
                own_resids[:, idx_i] = y_mat_aligned[:, idx_i] - _fit_ridge_nuisance(
                    W_i, y_mat_aligned[:, idx_i], l2_alpha=l2_alpha
                )
            n_obs_piv = len(piv_promo)
            for idx_i in range(n_series):
                r_i = own_resids[:, idx_i] - float(np.mean(own_resids[:, idx_i]))
                d_i = promo_mat_aligned[:, idx_i] - float(np.mean(promo_mat_aligned[:, idx_i]))
                di_norm2 = float(np.dot(d_i, d_i))
                for idx_j in range(n_series):
                    if idx_i == idx_j:
                        continue
                    p_j = promo_mat_aligned[:, idx_j] - float(np.mean(promo_mat_aligned[:, idx_j]))
                    if di_norm2 > 1e-9:
                        p_j = p_j - (float(np.dot(p_j, d_i)) / di_norm2) * d_i
                    pj_norm2 = float(np.dot(p_j, p_j))
                    if pj_norm2 <= 1e-9:
                        continue
                    beta_ij = float(np.dot(p_j, r_i)) / pj_norm2
                    if beta_ij <= 0.0:
                        continue
                    err_ij = r_i - beta_ij * p_j
                    se_ij = float(
                        np.sqrt(float(np.dot(err_ij, err_ij)) / (max(1, n_obs_piv - 3) * pj_norm2))
                    )
                    t_stat = beta_ij / max(1e-9, se_ij)
                    if t_stat > 2.20:
                        halo_affinity[idx_i, idx_j] = beta_ij

        w_pos = np.maximum(0.0, sim_arr.copy())
        np.fill_diagonal(w_pos, 0.0)
        for idx_i, tid_i in enumerate(series_ids):
            peer_indices = [j for j in range(n_series) if j != idx_i]
            halo_peers = [j for j in peer_indices if halo_affinity[idx_i, j] > 0.0]
            sub_peers = [j for j in peer_indices if j not in halo_peers]

            if halo_peers:
                best_h = max(halo_peers, key=lambda j: float(halo_affinity[idx_i, j]))
                h_sum = float(np.sum(halo_affinity[idx_i, halo_peers]))
                for j in halo_peers:
                    w_halo[idx_i, j] = float(halo_affinity[idx_i, j]) / max(1e-12, h_sum)
                h_sim_score = (
                    float(w_pos[idx_i, best_h])
                    if use_similarity_priors and float(w_pos[idx_i, best_h]) > 0.0
                    else float(w_halo[idx_i, best_h])
                )
                top_halo_by_series[tid_i] = (series_ids[best_h], round(h_sim_score, 4))

            if sub_peers:
                if use_similarity_priors:
                    best_j = max(
                        sub_peers,
                        key=lambda j: (float(w_pos[idx_i, j]), float(sim_arr[idx_i, j])),
                    )
                    top_cannibal_by_series[tid_i] = (
                        series_ids[best_j],
                        round(float(w_pos[idx_i, best_j]), 4),
                    )
                    sub_sum = float(np.sum(w_pos[idx_i, sub_peers]))
                    if sub_sum > 1e-12:
                        for j in sub_peers:
                            w_tilde[idx_i, j] = float(w_pos[idx_i, j]) / sub_sum
                    else:
                        for j in sub_peers:
                            w_tilde[idx_i, j] = 1.0 / float(len(sub_peers))
                else:
                    top_cannibal_by_series[tid_i] = (
                        series_ids[sub_peers[0]],
                        round(1.0 / float(len(sub_peers)), 4),
                    )
                    for j in sub_peers:
                        w_tilde[idx_i, j] = 1.0 / float(len(sub_peers))

        peer_mat = promo_mat_aligned @ w_tilde.T
        peer_wide = pd.DataFrame(peer_mat, index=piv_promo.index, columns=series_ids)
        peer_long = peer_wide.melt(
            ignore_index=False,
            var_name=ts_id_col,
            value_name="_peer_weighted",
        ).reset_index()
        peer_long[ts_id_col] = peer_long[ts_id_col].astype(str)

        halo_mat = promo_mat_aligned @ w_halo.T
        halo_wide = pd.DataFrame(halo_mat, index=piv_promo.index, columns=series_ids)
        halo_long = halo_wide.melt(
            ignore_index=False,
            var_name=ts_id_col,
            value_name="_halo_promo",
        ).reset_index()
        halo_long[ts_id_col] = halo_long[ts_id_col].astype(str)

        work[ts_id_col] = work[ts_id_col].astype(str)
        work = work.merge(peer_long, on=[date_col, ts_id_col], how="left")
        work = work.merge(halo_long, on=[date_col, ts_id_col], how="left")
        peer_weighted = work["_peer_weighted"].fillna(0.0).to_numpy(dtype=np.float64)
        work["_peer_promo"] = np.maximum(
            0.0, peer_weighted - work["_promo_intensity"].to_numpy(dtype=np.float64)
        )
        work["_halo_promo"] = np.maximum(
            0.0, work["_halo_promo"].fillna(0.0).to_numpy(dtype=np.float64)
        )
    else:
        work["_peer_promo"] = 0.0
        work["_halo_promo"] = 0.0

    eff_lags = max(1, int(pull_forward_lags))
    out_frames: list[pd.DataFrame] = []

    for tid, grp in work.groupby(ts_id_col, sort=False):
        tid_str = str(tid)
        top_can_id, top_can_score = top_cannibal_by_series.get(tid_str, (None, 0.0))
        top_halo_id, top_halo_score = top_halo_by_series.get(tid_str, (None, 0.0))
        grp = grp.sort_values(date_col).reset_index(drop=True)
        n_t = len(grp)
        y_vec = grp[target_col].to_numpy(dtype=np.float64)
        d_vec = grp["_promo_intensity"].to_numpy(dtype=np.float64)
        disc_vec = grp["_price_discount"].to_numpy(dtype=np.float64)
        peer_vec = grp["_peer_promo"].to_numpy(dtype=np.float64)
        halo_vec = grp["_halo_promo"].to_numpy(dtype=np.float64)

        has_treatment = (
            float(np.std(d_vec)) > 1e-9
            or float(np.std(disc_vec)) > 1e-9
            or float(np.std(halo_vec)) > 1e-9
        )
        if not has_treatment or n_t < 12:
            out_frames.append(
                pd.DataFrame(
                    {
                        "ts_id": tid_str,
                        "ds": grp[date_col],
                        "y": y_vec,
                        "baseline_demand": y_vec.copy(),
                        "baseline_yhat": y_vec.copy(),
                        "net_causal_lift": np.zeros(n_t, dtype=np.float64),
                        "incremental_promo_lift": np.zeros(n_t, dtype=np.float64),
                        "adstock_effect": np.zeros(n_t, dtype=np.float64),
                        "halo_lift": np.zeros(n_t, dtype=np.float64),
                        "halo_effect": np.zeros(n_t, dtype=np.float64),
                        "cannibalization_dip": np.zeros(n_t, dtype=np.float64),
                        "cannibalization_effect": np.zeros(n_t, dtype=np.float64),
                        "pull_forward_dip": np.zeros(n_t, dtype=np.float64),
                        "net_incremental_units": np.zeros(n_t, dtype=np.float64),
                        "net_causal_elasticity": np.zeros(n_t, dtype=np.float64),
                        "promo_lift_pct": np.zeros(n_t, dtype=np.float64),
                        "naive_ols_lift": np.zeros(n_t, dtype=np.float64),
                        "confounding_bias_pct": np.zeros(n_t, dtype=np.float64),
                        "top_cannibal_ts_id": top_can_id,
                        "cannibal_similarity_score": float(top_can_score),
                        "top_halo_ts_id": top_halo_id,
                        "halo_similarity_score": float(top_halo_score),
                    }
                )
            )
            continue

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

        # Resolve geometric ad-stock decay rate lambda in [0, 0.98)
        eff_lambda = 0.0
        if isinstance(adstock_decay, str) and adstock_decay.strip().lower() == "auto":
            if float(np.std(d_vec)) > 1e-9:
                cand_lambdas = np.linspace(0.0, 0.85, 18)
                cand_cols = [y_vec] + [
                    _compute_adstock_vector(d_vec, float(lam_c)) for lam_c in cand_lambdas
                ]
                cand_res = _cross_fitted_residuals(
                    W_mat,
                    np.column_stack(cand_cols),
                    n_folds=n_folds,
                    l2_alpha=l2_alpha,
                )
                y_r = cand_res[:, 0]
                best_score = -1.0
                base_score = 0.0
                for c_idx, lam_c in enumerate(cand_lambdas):
                    a_r = cand_res[:, c_idx + 1]
                    denom = float(np.dot(a_r, a_r))
                    if denom <= 1e-9:
                        continue
                    dot_pos = max(0.0, float(np.dot(a_r, y_r)))
                    explained = (dot_pos * dot_pos) / denom
                    if c_idx == 0:
                        base_score = explained
                    if explained > best_score:
                        best_score = explained
                        eff_lambda = float(lam_c)
                if best_score <= base_score * 1.02:
                    eff_lambda = 0.0
        elif isinstance(adstock_decay, (int, float)) and not isinstance(adstock_decay, bool):
            eff_lambda = float(np.clip(float(adstock_decay), 0.0, 0.98))

        adstock_by_series[tid_str] = round(eff_lambda, 4)
        a_vec = _compute_adstock_vector(d_vec, eff_lambda) if eff_lambda > 1e-6 else d_vec

        # Build post-promo pull-forward lag features (D_{t} == 0 and D_{t-k} > 0) when not in
        # positive ad-stock carryover regime.
        unpromoted_mask = (d_vec <= 1e-9).astype(np.float64)
        lag_cols: list[np.ndarray] = []
        if eff_lambda <= 1e-6:
            for k in range(1, eff_lags + 1):
                d_lag = np.zeros(n_t, dtype=np.float64)
                if n_t > k:
                    d_lag[k:] = d_vec[:-k] * unpromoted_mask[k:]
                lag_cols.append(d_lag)

        # Assemble treatment matrix T = [a_vec, (opt disc_vec), lag_1..lag_L, peer_vec, halo_vec]
        treat_list: list[np.ndarray] = [a_vec]
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

        has_halo = float(np.std(halo_vec)) > 1e-9
        halo_idx: int | None = None
        if has_halo:
            halo_idx = len(treat_list)
            treat_list.append(halo_vec)
            lb_list.append(0.0)
            ub_list.append(np.inf)

        T_mat = np.column_stack(treat_list)

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
        adstock_effect = (
            np.maximum(0.0, theta_dml[0] * (a_vec - d_vec))
            if eff_lambda > 1e-6
            else np.zeros(n_t, dtype=np.float64)
        )
        causal_lift = theta_dml[0] * a_vec
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
        halo_lift = (
            np.maximum(0.0, theta_dml[halo_idx] * halo_vec)
            if halo_idx is not None
            else np.zeros(n_t, dtype=np.float64)
        )

        # Reconstruct de-promoted structural baseline g_0(W_t)
        total_treatment_effect = causal_lift + pull_forward_dip + cannib_dip + halo_lift
        y_depromoted = y_vec - total_treatment_effect
        baseline_demand = _fit_ridge_nuisance(W_mat, y_depromoted, l2_alpha=l2_alpha)
        if np.all(y_vec >= 0.0):
            baseline_demand = np.maximum(0.0, baseline_demand)

        # Naive uncontrolled OLS regression of y on [1, a_vec] (ignores seasonal confounding W_t)
        d_centered = a_vec - float(np.mean(a_vec))
        d_var = float(np.dot(d_centered, d_centered))
        if d_var > 1e-9:
            theta_ols = max(0.0, float(np.dot(d_centered, y_vec - float(np.mean(y_vec))) / d_var))
        else:
            theta_ols = float(theta_dml[0])
        naive_ols_lift = theta_ols * a_vec
        if include_separate_disc:
            disc_c = disc_vec - float(np.mean(disc_vec))
            disc_var = float(np.dot(disc_c, disc_c))
            if disc_var > 1e-9:
                theta_ols_disc = max(
                    0.0, float(np.dot(disc_c, y_vec - float(np.mean(y_vec))) / disc_var)
                )
                naive_ols_lift = naive_ols_lift + theta_ols_disc * disc_vec

        net_incremental = causal_lift + pull_forward_dip + cannib_dip + halo_lift
        series_floor = max(1.0, 0.05 * float(np.nanmean(np.abs(y_vec))))
        safe_base = np.maximum(series_floor, np.abs(baseline_demand))
        net_causal_elasticity = net_incremental / safe_base
        promo_lift_pct = np.where(a_vec > 1e-9, 100.0 * causal_lift / safe_base, 0.0)

        # Confounding bias (%): how much naive OLS over-estimates true causal lift
        confounding_bias_pct = np.where(
            causal_lift > 1e-6,
            100.0 * (naive_ols_lift - causal_lift) / np.maximum(1e-6, causal_lift),
            0.0,
        )

        out_frames.append(
            pd.DataFrame(
                {
                    "ts_id": tid_str,
                    "ds": grp[date_col],
                    "y": np.round(y_vec, 4),
                    "baseline_demand": np.round(baseline_demand, 4),
                    "baseline_yhat": np.round(baseline_demand, 4),
                    "net_causal_lift": np.round(causal_lift, 4),
                    "incremental_promo_lift": np.round(causal_lift, 4),
                    "adstock_effect": np.round(adstock_effect, 4),
                    "halo_lift": np.round(halo_lift, 4),
                    "halo_effect": np.round(halo_lift, 4),
                    "cannibalization_dip": np.round(cannib_dip, 4),
                    "cannibalization_effect": np.round(cannib_dip, 4),
                    "pull_forward_dip": np.round(pull_forward_dip, 4),
                    "net_incremental_units": np.round(net_incremental, 4),
                    "net_causal_elasticity": np.round(net_causal_elasticity, 6),
                    "promo_lift_pct": np.round(promo_lift_pct, 4),
                    "naive_ols_lift": np.round(naive_ols_lift, 4),
                    "confounding_bias_pct": np.round(confounding_bias_pct, 4),
                    "top_cannibal_ts_id": top_can_id,
                    "cannibal_similarity_score": float(top_can_score),
                    "top_halo_ts_id": top_halo_id,
                    "halo_similarity_score": float(top_halo_score),
                }
            )
        )

    if not out_frames:
        return pd.DataFrame(columns=list(PROMO_ELASTICITY_COLUMNS))
    res_df = pd.concat(out_frames, ignore_index=True)
    res_df.attrs["similarity_weights"] = pd.DataFrame(
        w_tilde,
        index=series_ids,
        columns=series_ids,
    )
    res_df.attrs["halo_weights"] = pd.DataFrame(
        w_halo,
        index=series_ids,
        columns=series_ids,
    )
    res_df.attrs["top_cannibal_by_series"] = top_cannibal_by_series
    res_df.attrs["top_halo_by_series"] = top_halo_by_series
    res_df.attrs["adstock_decay_by_series"] = adstock_by_series
    return res_df


# Public aliases for SDK / diagnostic ergonomics
promo_elasticity_df = compute_promo_elasticity_decomposition
compute_promo_elasticity = compute_promo_elasticity_decomposition
estimate_promo_elasticity = compute_promo_elasticity_decomposition


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
        "total_adstock_effect",
        "total_halo_lift",
        "total_pull_forward_dip",
        "total_cannibalization_dip",
        "total_net_incremental_units",
        "net_to_gross_lift_ratio",
        "top_cannibal_ts_id",
        "cannibal_similarity_score",
        "top_halo_ts_id",
        "halo_similarity_score",
    ]
    if decomp_df is None or decomp_df.empty or "ts_id" not in decomp_df.columns:
        return pd.DataFrame(columns=summary_cols)

    rows: list[dict[str, Any]] = []
    for tid, grp in decomp_df.groupby("ts_id", sort=False):
        direct_lift = (
            grp["net_causal_lift"] - grp["adstock_effect"]
            if "adstock_effect" in grp.columns
            else grp["net_causal_lift"]
        )
        promo_mask = direct_lift > 1e-6
        n_promo = int(promo_mask.sum())
        mean_base = float(grp["baseline_demand"].mean())
        mean_causal = float(grp.loc[promo_mask, "net_causal_lift"].mean()) if n_promo > 0 else 0.0
        mean_ols = float(grp.loc[promo_mask, "naive_ols_lift"].mean()) if n_promo > 0 else 0.0
        bias_pct = (
            100.0 * (mean_ols - mean_causal) / max(1e-6, mean_causal) if mean_causal > 1e-6 else 0.0
        )
        tot_gross = float(grp["net_causal_lift"].sum())
        tot_adstock = float(grp["adstock_effect"].sum()) if "adstock_effect" in grp.columns else 0.0
        tot_halo = float(grp["halo_lift"].sum()) if "halo_lift" in grp.columns else 0.0
        tot_pf = float(grp["pull_forward_dip"].sum())
        tot_can = float(grp["cannibalization_dip"].sum())
        tot_net = float(grp["net_incremental_units"].sum())
        ratio = tot_net / max(1e-6, tot_gross) if tot_gross > 1e-6 else 0.0
        top_can_id = (
            grp["top_cannibal_ts_id"].iloc[0] if "top_cannibal_ts_id" in grp.columns else None
        )
        top_can_score = (
            float(grp["cannibal_similarity_score"].iloc[0])
            if "cannibal_similarity_score" in grp.columns
            else 0.0
        )
        top_halo_id = grp["top_halo_ts_id"].iloc[0] if "top_halo_ts_id" in grp.columns else None
        top_halo_score = (
            float(grp["halo_similarity_score"].iloc[0])
            if "halo_similarity_score" in grp.columns
            else 0.0
        )
        rows.append(
            {
                "ts_id": str(tid),
                "promo_days": n_promo,
                "mean_baseline_demand": round(mean_base, 4),
                "mean_causal_promo_lift": round(mean_causal, 4),
                "mean_naive_ols_lift": round(mean_ols, 4),
                "confounding_bias_pct": round(bias_pct, 2),
                "total_gross_causal_lift": round(tot_gross, 4),
                "total_adstock_effect": round(tot_adstock, 4),
                "total_halo_lift": round(tot_halo, 4),
                "total_pull_forward_dip": round(tot_pf, 4),
                "total_cannibalization_dip": round(tot_can, 4),
                "total_net_incremental_units": round(tot_net, 4),
                "net_to_gross_lift_ratio": round(ratio, 4),
                "top_cannibal_ts_id": top_can_id,
                "cannibal_similarity_score": round(top_can_score, 4),
                "top_halo_ts_id": top_halo_id,
                "halo_similarity_score": round(top_halo_score, 4),
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
    if "halo_lift" in sub.columns and float(sub["halo_lift"].max()) > 1e-6:
        ax_bot.bar(
            sub["ds"],
            sub["halo_lift"],
            bottom=sub["net_causal_lift"],
            label="Complementary Halo Lift (+)",
            color="#0284c7",
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
    halo_fraction: float = 0.0,
    adstock_decay: float = 0.0,
    true_pull_forward_dip: float | None = None,
    true_cannibalization_dip: float | None = None,
    true_halo_lift: float | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a confounded seasonal + promotional retail DGP with known structural parameters.

    Promotions are intentionally scheduled with $4\\times$ higher probability during seasonal peaks
    ($\\sin(2\\pi t / 7) + \\sin(2\\pi t / 30.4375) > 0.35$), causing naive uncontrolled OLS to
    confound seasonal peaks with promotional lift and over-estimate promo lift by $\\ge 25\\%$.
    Includes post-promotion pantry-loading pull-forward dips, cross-SKU cannibalization dips,
    optional complementary halo lift, and optional geometric ad-stock decay.
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
    eff_halo_lift = (
        abs(float(true_halo_lift))
        if true_halo_lift is not None
        else abs(float(true_promo_lift) * float(halo_fraction))
    )
    eff_adstock = float(np.clip(float(adstock_decay), 0.0, 0.98))

    frames: list[pd.DataFrame] = []
    for i in range(n_series):
        ts_id = f"sku_{i + 1:03d}"
        base_level = 120.0 + 10.0 * i
        trend = 0.08 * t_idx
        baseline = base_level + trend + float(seasonal_amplitude) * seasonal_wave

        d_vec = promo_mat[:, i]
        a_vec = _compute_adstock_vector(d_vec, eff_adstock) if eff_adstock > 1e-6 else d_vec
        # Post-promo pull-forward dip on the day immediately following an active promo
        d_lag1 = np.zeros(n_days, dtype=np.float64)
        if eff_adstock <= 1e-6:
            d_lag1[1:] = d_vec[:-1] * (1.0 - d_vec[1:])

        # Peer promo intensity when focal SKU is not promoted
        halo_active = np.zeros(n_days, dtype=np.float64)
        if n_series >= 2:
            if eff_halo_lift > 1e-9 and i == 1:
                # Series 0 (sku_001) acts as a complementary halo driver for Series 1 (sku_002)
                halo_active = promo_mat[:, 0]
                sub_cols = [j for j in range(n_series) if j not in (i, 0)]
                if sub_cols:
                    peer_mean = np.mean(promo_mat[:, sub_cols], axis=1)
                    peer_active = np.maximum(0.0, peer_mean - d_vec)
                else:
                    peer_active = np.zeros(n_days, dtype=np.float64)
            else:
                peer_mean = (np.sum(promo_mat, axis=1) - d_vec) / max(1, n_series - 1)
                peer_active = np.maximum(0.0, peer_mean - d_vec)
        else:
            peer_active = np.zeros(n_days, dtype=np.float64)

        causal_lift = float(true_promo_lift) * a_vec
        pf_dip = true_pf_dip * d_lag1
        cannib_dip = true_cannib_dip * peer_active
        halo_vec = eff_halo_lift * halo_active
        noise = rng.normal(0.0, 2.0, size=n_days)

        y = np.maximum(0.0, baseline + causal_lift + pf_dip + cannib_dip + halo_vec + noise)
        row_dict: dict[str, Any] = {
            "ts_id": ts_id,
            "ds": dates,
            "y": np.round(y, 4),
            "promo_flag": d_vec,
            "true_baseline": np.round(baseline, 4),
            "true_causal_lift": np.round(causal_lift, 4),
            "true_pull_forward_dip": np.round(pf_dip, 4),
            "true_cannibalization_dip": np.round(cannib_dip, 4),
        }
        if eff_halo_lift > 1e-9:
            row_dict["true_halo_lift"] = np.round(halo_vec, 4)
        if eff_adstock > 1e-6:
            row_dict["true_adstock_effect"] = np.round(float(true_promo_lift) * (a_vec - d_vec), 4)
        frames.append(pd.DataFrame(row_dict))

    panel_df = pd.concat(frames, ignore_index=True)
    ground_truth: dict[str, Any] = {
        "true_promo_lift": float(true_promo_lift),
        "true_pull_forward_dip": float(true_pf_dip),
        "true_cannibalization_dip": float(true_cannib_dip),
        "seasonal_amplitude": float(seasonal_amplitude),
    }
    if eff_halo_lift > 1e-9:
        ground_truth["true_halo_lift"] = float(eff_halo_lift)
    if eff_adstock > 1e-6:
        ground_truth["adstock_decay"] = float(eff_adstock)
    return panel_df, ground_truth
