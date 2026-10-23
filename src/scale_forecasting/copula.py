"""Joint Multi-Horizon & Cross-Series Trajectory Copulas and Tail-Risk Quantification.

Bridges marginal per-step prediction intervals (``[yhat_lower, yhat, yhat_upper]`` or arbitrary
quantile grids) to coherent joint multi-horizon and cross-series sample paths ``Y^(s)`` for
downstream cumulative tail-risk (``VaR_alpha``, ``CVaR_alpha``), multi-period inventory safety
stock, and risk-averse scenario optimization.

Mathematical Formulation
------------------------
1. **Horizon & Cross-Series Innovation Dependence**:
   Given historical out-of-fold (OOF) or in-sample residuals, we estimate the $H \\times H$
   multi-horizon correlation matrix $\\Sigma_H$ by combining empirical step-by-step error
   correlations with a Toeplitz lag-autocorrelation target $T_{h, h'} = \\hat{\\rho}_{|h - h'|}$
   regularized via Schäfer-Strimmer shrinkage (`schafer_strimmer_shrinkage`) and projected onto the
   nearest symmetric positive-definite (SPD) correlation matrix. When ``cross_series=True`` across
   $N$ panel series, we also estimate the $N \\times N$ cross-sectional residual correlation matrix
   $\\Sigma_N$ and draw joint $(N, H)$ trajectories using the Kronecker Cholesky factor
   $L_N \\otimes L_H$.
2. **Copula Families**:
   * ``"gaussian"``: Multivariate normal copula $Z \\sim \\mathcal{N}(0, \\Sigma_H)$,
     $U_h = \\Phi(Z_h)$ with Gaussian tail extrapolation beyond calibrated marginal quantiles.
   * ``"student_t"``: Multivariate Student-$t$ copula $T = Z / \\sqrt{V / \\nu}$ with
     $V \\sim \\chi^2_\\nu$ and $U_h = t_\\nu(T_h)$, capturing symmetric upper/lower tail
     dependence $\\lambda_U > 0$ (simultaneous multi-day spikes) and heavy-tailed $t_\\nu$
     extrapolation in the extreme quantiles ($u > 0.90$ or $u < 0.10$).
   * ``"iman_conover"`` (alias ``"empirical"``): Non-parametric rank-reordering copula
     (Iman & Conover 1982) that pairs empirical residual rank structures with target Cholesky
     correlation scores while preserving exact marginal distributions.
   * ``"auto"``: Selects ``"student_t"`` when empirical residual excess kurtosis exceeds ``1.0``,
     ``"iman_conover"`` under strong skewness ($|\\gamma_1| > 1.0$) with sufficient history,
     and ``"gaussian"`` otherwise.
3. **Randomized PIT for Discrete / Intermittent Counts** (Czado et al. 2012):
   When residual or target observations exhibit discrete ties or zero-inflation, rank/PIT mapping
   replaces point-mass ties with uniform jitter $U_{t, h} \\sim \\text{Uniform}(F(y^-), F(y))$
   prior to copula rank/correlation estimation to prevent zero-inflation rank collapse.
4. **Constraint & Coherence Preservation**:
   Every simulated path $s \\in \\{0, \\dots, S-1\\}$ respects configured hard bounds
   (``floor`` / ``cap``), compositional ``simplex_sum`` closure (`< 1e-6`), and 3-mode
   cross-sectional (`HierarchySpec`), temporal (`TemporalHierarchySpec`), or simultaneous
   cross-temporal (`cross_temporal_mint`) additive coherence (`< 1e-6`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from .config import RunConfig
from .errors import ConfigError, DataError
from .reconciliation import (
    HierarchySpec,
    build_hierarchy,
    reconcile_matrix,
    schafer_strimmer_shrinkage,
)
from .seasonality import seasonal_period
from .simplex import detect_simplex_sum, enforce_simplex_closure
from .temporal_reconciliation import (
    TemporalHierarchySpec,
    aggregate_temporal_history,
    build_temporal_hierarchy,
    reconcile_temporal_matrix,
)

CopulaType = Literal["auto", "gaussian", "student_t", "iman_conover", "empirical"]

COPULA_TYPES: tuple[str, ...] = (
    "auto",
    "gaussian",
    "student_t",
    "iman_conover",
    "empirical",
)
VALID_COPULA_TYPES: frozenset[str] = frozenset(COPULA_TYPES)

_Z_80 = 1.2815515655446004  # stats.norm.ppf(0.90)
_PROB_EPS = 1e-6


@dataclass(frozen=True)
class CopulaRiskSummary:
    """Summary container for joint copula trajectory sampling and tail-risk metrics.

    Attributes:
        trajectories_df: Tidy DataFrame of simulated joint trajectories across ``n_paths``.
        tail_risk_df: Per-series cumulative and rolling-window tail-risk DataFrame.
        copula_type_selected: Resolved copula family used for sampling.
        scores: Optional dictionary of multivariate trajectory evaluation scores
            (``crps``, ``energy_score``, ``variogram_score``).
    """

    trajectories_df: pd.DataFrame
    tail_risk_df: pd.DataFrame
    copula_type_selected: str
    scores: dict[str, float]


def _nearest_correlation_matrix(mat: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix onto the space of SPD correlation matrices with unit diagonal."""
    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ConfigError(f"Correlation matrix must be square 2-D, got shape {arr.shape}")
    d = arr.shape[0]
    if d == 0:
        return np.empty((0, 0), dtype=float)
    if d == 1:
        return np.ones((1, 1), dtype=float)

    sym = 0.5 * (arr + arr.T)
    sym = np.where(np.isfinite(sym), sym, 0.0)
    np.fill_diagonal(sym, 1.0)
    sym = np.clip(sym, -0.999, 0.999)
    np.fill_diagonal(sym, 1.0)

    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.maximum(eigvals, float(min_eig))
    rebuilt = (eigvecs * eigvals_clipped[np.newaxis, :]) @ eigvecs.T
    diag = np.sqrt(np.maximum(np.diag(rebuilt), 1e-12))
    corr = rebuilt / np.outer(diag, diag)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    # Final small jitter check to guarantee Cholesky succeeds.
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        corr = (1.0 - min_eig) * corr + min_eig * np.eye(d, dtype=float)
    return corr


def _randomized_pit(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compute randomized Probability Integral Transform (PIT) uniforms in ``(0, 1)``.

    For discrete or zero-inflated series with ties, replaces tie ranks with uniform draws on
    ``[F(y^-), F(y)]`` (Czado et al. 2012) so rank correlation does not collapse at zero.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full(n, 0.5, dtype=float)

    clean = np.where(finite, arr, float(np.nanmedian(arr[finite])))
    u_out = np.empty(n, dtype=float)
    unique_vals, inverse_idx, counts = np.unique(clean, return_inverse=True, return_counts=True)
    if len(unique_vals) == n:
        ranks = np.argsort(np.argsort(clean)) + 1.0
        u_out = (ranks - 0.5) / float(n)
    else:
        cum_counts = np.cumsum(counts)
        low_bounds = np.concatenate([[0], cum_counts[:-1]]) / float(n)
        high_bounds = cum_counts / float(n)
        lo = low_bounds[inverse_idx]
        hi = high_bounds[inverse_idx]
        u_out = rng.uniform(lo, hi)

    return np.clip(u_out, _PROB_EPS, 1.0 - _PROB_EPS)


def _extract_residual_series(
    oof_df: pd.DataFrame | None,
    history_df: pd.DataFrame | None,
    *,
    ts_id: str | None = None,
    model_type: str | None = None,
    ts_id_col: str = "ts_id",
) -> np.ndarray:
    """Extract a 1-D time-ordered residual vector for ``(ts_id, model_type)``."""
    if oof_df is not None and not oof_df.empty and {"y_true", "yhat"}.issubset(oof_df.columns):
        sub = oof_df
        if ts_id is not None and ts_id_col in sub.columns:
            m_ts = sub[ts_id_col].astype(str) == str(ts_id)
            if m_ts.any():
                sub = sub[m_ts]
        if model_type is not None and "model_type" in sub.columns:
            m_mod = sub["model_type"].astype(str) == str(model_type)
            if m_mod.any():
                sub = sub[m_mod]
        for dcol in ("ds", "forecast_date", "date"):
            if dcol in sub.columns:
                sub = sub.sort_values(dcol, kind="mergesort")
                break
        res = sub["y_true"].to_numpy(dtype=float) - sub["yhat"].to_numpy(dtype=float)
        res = res[np.isfinite(res)]
        if len(res) >= 3:
            return res

    if history_df is not None and not history_df.empty:
        sub_h = history_df
        if ts_id is not None and ts_id_col in sub_h.columns:
            m_ts = sub_h[ts_id_col].astype(str) == str(ts_id)
            if m_ts.any():
                sub_h = sub_h[m_ts]
        for dcol in ("ds", "forecast_date", "date"):
            if dcol in sub_h.columns:
                sub_h = sub_h.sort_values(dcol, kind="mergesort")
                break
        for ycol in ("y", "y_true", "yhat"):
            if ycol in sub_h.columns:
                y_vals = sub_h[ycol].to_numpy(dtype=float)
                y_vals = y_vals[np.isfinite(y_vals)]
                if len(y_vals) >= 4:
                    # Detrended innovations capture serial persistence around local trend.
                    t_idx = np.arange(len(y_vals), dtype=float)
                    slope, intercept = np.polyfit(t_idx, y_vals, 1)
                    return y_vals - (intercept + slope * t_idx)
    return np.empty(0, dtype=float)


def _select_copula_type(
    requested: str,
    residuals: np.ndarray,
) -> tuple[str, float]:
    """Resolve ``copula_type`` and Student-$t$ degrees of freedom ``df_t``."""
    norm = str(requested or "auto").strip().lower()
    if norm == "empirical":
        norm = "iman_conover"
    if norm not in VALID_COPULA_TYPES:
        raise ConfigError(
            f"Unknown copula_type {requested!r}; expected one of {sorted(VALID_COPULA_TYPES)}"
        )

    clean = np.asarray(residuals, dtype=float).reshape(-1)
    clean = clean[np.isfinite(clean)]
    if norm != "auto":
        return norm, 4.0

    if len(clean) >= 12 and float(np.std(clean)) > 1e-10:
        kurt_raw = float(stats.kurtosis(clean, fisher=True, bias=False))
        kurt_innov = kurt_raw
        if len(clean) >= 16:
            c0 = clean[:-1] - float(np.mean(clean[:-1]))
            c1 = clean[1:] - float(np.mean(clean[1:]))
            denom = float(np.dot(c0, c0))
            if denom > 1e-12:
                rho1 = float(np.clip(np.dot(c0, c1) / denom, -0.95, 0.95))
                innov = clean[1:] - rho1 * clean[:-1]
                if float(np.std(innov)) > 1e-10:
                    kurt_innov = float(stats.kurtosis(innov, fisher=True, bias=False))
        kurt = max(
            kurt_raw if np.isfinite(kurt_raw) else 0.0,
            kurt_innov if np.isfinite(kurt_innov) else 0.0,
        )
        skew = float(stats.skew(clean, bias=False))
        if kurt > 1.0:
            est_df = float(np.clip(4.0 + 6.0 / max(kurt, 0.2), 3.0, 10.0))
            return "student_t", est_df
        if np.isfinite(skew) and abs(skew) > 1.0 and len(clean) >= 20:
            return "iman_conover", 4.0
    return "gaussian", 4.0


def _estimate_horizon_correlation(
    horizon: int,
    oof_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
    *,
    ts_id: str | None = None,
    model_type: str | None = None,
    ts_id_col: str = "ts_id",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate the $H \\times H$ SPD horizon error correlation matrix $\\Sigma_H$.

    Combines empirical multi-step fold correlations (when ``fold_id`` is available) with a
    Toeplitz lag-autocorrelation matrix $T_{h, h'} = \\hat{\\rho}_{|h - h'|}$ estimated from
    time-ordered residuals so serial correlation across consecutive horizon dates is always
    preserved even when $n_{\\text{folds}} < H$.
    """
    h = max(1, int(horizon))
    if h == 1:
        return np.ones((1, 1), dtype=float)

    res_1d = _extract_residual_series(
        oof_df,
        history_df,
        ts_id=ts_id,
        model_type=model_type,
        ts_id_col=ts_id_col,
    )

    # 1. Build Toeplitz lag-autocorrelation matrix from time-ordered residuals.
    toeplitz_corr = np.eye(h, dtype=float)
    if len(res_1d) >= 4 and float(np.std(res_1d)) > 1e-12:
        # If discrete/zero-inflated ties exist, blend raw and randomized-PIT normal scores.
        unique_ratio = len(np.unique(np.round(res_1d, 8))) / float(len(res_1d))
        if unique_ratio < 0.75 and rng is not None:
            u_pit = _randomized_pit(res_1d, rng)
            series_for_acf = 0.5 * (
                (res_1d - np.mean(res_1d)) / max(float(np.std(res_1d)), 1e-12)
                + stats.norm.ppf(u_pit)
            )
        else:
            series_for_acf = (res_1d - np.mean(res_1d)) / max(float(np.std(res_1d)), 1e-12)

        n_obs = len(series_for_acf)
        centered = series_for_acf - np.mean(series_for_acf)
        var0 = float(np.dot(centered, centered)) / float(n_obs)
        if var0 > 1e-12:
            rho1 = float(np.dot(centered[1:], centered[:-1])) / (float(n_obs) * var0)
            rho1 = float(np.clip(rho1, -0.92, 0.92))
            lags = np.abs(np.subtract.outer(np.arange(h), np.arange(h)))
            ar1_mat = rho1**lags

            emp_acf = np.ones(h, dtype=float)
            max_lag = min(h - 1, max(1, n_obs // 3))
            for k in range(1, max_lag + 1):
                rk = float(np.dot(centered[k:], centered[:-k])) / (float(n_obs) * var0)
                # Bartlett window damping for finite-sample stability.
                weight_k = max(0.0, 1.0 - float(k) / float(max(max_lag + 2, h)))
                emp_acf[k] = float(
                    np.clip(weight_k * rk + (1.0 - weight_k) * (rho1**k), -0.92, 0.92)
                )
            for k in range(max_lag + 1, h):
                emp_acf[k] = float(rho1**k)
            toeplitz_corr = _nearest_correlation_matrix(0.5 * emp_acf[lags] + 0.5 * ar1_mat)

    # 2. Check if multi-fold, multi-step OOF matrix is available.
    if (
        oof_df is not None
        and not oof_df.empty
        and {"y_true", "yhat"}.issubset(oof_df.columns)
        and ("fold_id" in oof_df.columns or "cutoff_date" in oof_df.columns)
    ):
        sub = oof_df
        if ts_id is not None and ts_id_col in sub.columns:
            m_ts = sub[ts_id_col].astype(str) == str(ts_id)
            if m_ts.any():
                sub = sub[m_ts]
        if model_type is not None and "model_type" in sub.columns:
            m_mod = sub["model_type"].astype(str) == str(model_type)
            if m_mod.any():
                sub = sub[m_mod]

        fold_col = "fold_id" if "fold_id" in sub.columns else "cutoff_date"
        dcol = (
            "ds"
            if "ds" in sub.columns
            else ("forecast_date" if "forecast_date" in sub.columns else None)
        )
        if dcol is not None and sub[fold_col].nunique() >= 3:
            tmp = sub.sort_values([fold_col, dcol], kind="mergesort").copy()
            tmp["_step"] = tmp.groupby(fold_col, sort=False).cumcount() + 1
            tmp["_res"] = tmp["y_true"].to_numpy(dtype=float) - tmp["yhat"].to_numpy(dtype=float)
            tmp = tmp[tmp["_step"] <= h]
            piv = tmp.pivot_table(index=fold_col, columns="_step", values="_res", aggfunc="mean")
            if piv.shape[0] >= 3 and piv.shape[1] >= 2:
                piv_reindexed = (
                    piv.reindex(columns=range(1, h + 1)).fillna(0.0).to_numpy(dtype=float)
                )
                cov_shrunk, _ = schafer_strimmer_shrinkage(piv_reindexed)
                d_std = np.sqrt(np.maximum(np.diag(cov_shrunk), 1e-12))
                fold_corr = cov_shrunk / np.outer(d_std, d_std)
                fold_corr = _nearest_correlation_matrix(fold_corr)
                alpha_fold = min(0.65, float(piv.shape[0]) / float(piv.shape[0] + h))
                return _nearest_correlation_matrix(
                    alpha_fold * fold_corr + (1.0 - alpha_fold) * toeplitz_corr
                )

    return toeplitz_corr


def _estimate_cross_series_correlation(
    series_ids: Sequence[str],
    oof_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
    *,
    model_type: str | None = None,
    ts_id_col: str = "ts_id",
) -> np.ndarray:
    """Estimate the $N \\times N$ SPD cross-series residual correlation matrix $\\Sigma_N$."""
    n_series = len(series_ids)
    if n_series <= 1:
        return np.ones((max(1, n_series), max(1, n_series)), dtype=float)

    if oof_df is not None and not oof_df.empty and {"y_true", "yhat"}.issubset(oof_df.columns):
        sub = oof_df
        if model_type is not None and "model_type" in sub.columns:
            m_mod = sub["model_type"].astype(str) == str(model_type)
            if m_mod.any():
                sub = sub[m_mod]
        dcol = (
            "ds"
            if "ds" in sub.columns
            else ("forecast_date" if "forecast_date" in sub.columns else None)
        )
        if dcol is not None and ts_id_col in sub.columns:
            tmp = sub.copy()
            tmp["_res"] = tmp["y_true"].to_numpy(dtype=float) - tmp["yhat"].to_numpy(dtype=float)
            piv = (
                tmp.pivot_table(index=dcol, columns=ts_id_col, values="_res", aggfunc="mean")
                .reindex(columns=list(series_ids))
                .dropna(how="all")
                .fillna(0.0)
            )
            if piv.shape[0] >= 3:
                cov_n, _ = schafer_strimmer_shrinkage(piv.to_numpy(dtype=float))
                std_n = np.sqrt(np.maximum(np.diag(cov_n), 1e-12))
                corr_n = cov_n / np.outer(std_n, std_n)
                return _nearest_correlation_matrix(corr_n)

    if history_df is not None and not history_df.empty and ts_id_col in history_df.columns:
        dcol = (
            "ds"
            if "ds" in history_df.columns
            else ("forecast_date" if "forecast_date" in history_df.columns else None)
        )
        ycol = (
            "y"
            if "y" in history_df.columns
            else ("y_true" if "y_true" in history_df.columns else None)
        )
        if dcol is not None and ycol is not None:
            piv_h = (
                history_df.pivot_table(index=dcol, columns=ts_id_col, values=ycol, aggfunc="mean")
                .reindex(columns=list(series_ids))
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            if piv_h.shape[0] >= 4:
                diffs = np.diff(piv_h.to_numpy(dtype=float), axis=0)
                cov_n, _ = schafer_strimmer_shrinkage(diffs)
                std_n = np.sqrt(np.maximum(np.diag(cov_n), 1e-12))
                corr_n = cov_n / np.outer(std_n, std_n)
                return _nearest_correlation_matrix(corr_n)

    return np.eye(n_series, dtype=float)


def _resolve_similarity_shrinkage_weight(
    n_series: int,
    n_samples: int,
    weight: float | Literal["auto"] = "auto",
) -> float:
    """Resolve the convex similarity shrinkage weight $\\lambda_{\\text{sim}} \\in [0, 1]$."""
    if isinstance(weight, str):
        if weight != "auto":
            raise ConfigError(
                f"similarity_weight must be 'auto' or a float in [0.0, 1.0], got {weight!r}"
            )
        n_s = max(1, int(n_series))
        t_obs = max(0, int(n_samples))
        lam = float(np.clip(n_s / float(n_s + 2 * t_obs), 0.10, 0.75))
        if t_obs < 3:
            lam = max(lam, 0.50)
        return lam

    if isinstance(weight, bool) or not isinstance(weight, (int, float, np.number)):
        raise ConfigError(
            f"similarity_weight must be 'auto' or a float in [0.0, 1.0], got {weight!r}"
        )
    w_val = float(weight)
    if not np.isfinite(w_val) or w_val < 0.0 or w_val > 1.0:
        raise ConfigError(f"similarity_weight must be in [0.0, 1.0], got {weight!r}")
    return w_val


def shrink_correlation_toward_similarity(
    emp_corr: np.ndarray,
    sim_matrix: np.ndarray | pd.DataFrame,
    *,
    n_samples: int = 0,
    weight: float | Literal["auto"] = "auto",
    series_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Shrink an empirical $(N, N)$ correlation matrix toward a trajectory similarity prior.

    Computes
    $$R_{\\text{reg}} = \\text{NearestPSD}\\!\\big((1 - \\lambda_{\\text{sim}})\\,
    \\hat{R}_{\\text{emp}} + \\lambda_{\\text{sim}}\\,\\text{NearestPSD}(S_{\\text{sim}})\\big)$$
    where $\\lambda_{\\text{sim}} = \\text{clip}(N / (N + 2 T_{\\text{OOF}}), 0.10, 0.75)$ (with a
    minimum floor of $0.50$ when $T_{\\text{OOF}} < 3$) when ``weight="auto"``.
    """
    emp_arr = np.asarray(emp_corr, dtype=float)
    if emp_arr.ndim != 2 or emp_arr.shape[0] != emp_arr.shape[1]:
        raise ConfigError(f"emp_corr must be a square 2D matrix, got shape {emp_arr.shape}")
    n_series = int(emp_arr.shape[0])
    if n_series <= 1:
        return np.ones((max(1, n_series), max(1, n_series)), dtype=float)

    if isinstance(sim_matrix, pd.DataFrame):
        if series_ids is not None:
            aligned_df = sim_matrix.reindex(index=list(series_ids), columns=list(series_ids))
            sim_arr = aligned_df.to_numpy(dtype=float).copy()
            nan_mask = ~np.isfinite(sim_arr)
            sim_arr[nan_mask] = 0.0
            np.fill_diagonal(sim_arr, 1.0)
        else:
            sim_arr = sim_matrix.to_numpy(dtype=float).copy()
    else:
        sim_arr = np.asarray(sim_matrix, dtype=float).copy()

    if sim_arr.shape != (n_series, n_series):
        raise ConfigError(
            f"sim_matrix shape {sim_arr.shape} does not match emp_corr shape {(n_series, n_series)}"
        )

    lam = _resolve_similarity_shrinkage_weight(n_series, n_samples, weight)
    emp_psd = _nearest_correlation_matrix(emp_arr)
    if lam <= 0.0:
        return emp_psd
    sim_psd = _nearest_correlation_matrix(sim_arr)
    if lam >= 1.0:
        return sim_psd
    blended = (1.0 - lam) * emp_psd + lam * sim_psd
    return _nearest_correlation_matrix(blended)


def _sample_copula_uniforms(
    corr_h: np.ndarray,
    n_paths: int,
    copula_type: str,
    *,
    df_t: float = 4.0,
    corr_n: np.ndarray | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw uniform copula samples $U \\in (0, 1)^{S \\times N \\times H}$.

    Uses exact stratified marginal rank grids combined with Cholesky correlation projection or
    multivariate Student-$t$ radial scaling so marginal quantiles at $0.10, 0.50, 0.90$ are
    preserved while joint horizon/cross-series dependence follows ``corr_h`` (and ``corr_n``).
    """
    h = int(corr_h.shape[0])
    n_series = int(corr_n.shape[0]) if corr_n is not None else 1
    chol_h = np.linalg.cholesky(_nearest_correlation_matrix(corr_h))
    chol_n = (
        np.linalg.cholesky(_nearest_correlation_matrix(corr_n))
        if corr_n is not None and n_series > 1
        else np.ones((1, 1), dtype=float)
    )

    # Draw standard normal innovations Z_raw of shape (n_paths, n_series, h) and correlate via
    # Kronecker product (L_N \otimes L_H): Z_corr[s] = L_N @ Z_raw[s] @ L_H.T.
    z_raw = rng.standard_normal(size=(n_paths, n_series, h))
    # Center and normalize across paths when n_paths >= 8 to eliminate Monte Carlo bias.
    if n_paths >= 8:
        z_raw = z_raw - np.mean(z_raw, axis=0, keepdims=True)
        z_std = np.std(z_raw, axis=0, keepdims=True)
        z_raw = np.where(z_std > 1e-8, z_raw / z_std, z_raw)

    if n_series > 1:
        z_corr = np.einsum("ij,sjk,lk->sil", chol_n, z_raw, chol_h)
    else:
        z_corr = np.einsum("sjk,lk->sjl", z_raw, chol_h)

    if copula_type == "student_t":
        nu = max(2.1, float(df_t))
        # Common chi-square shock per path induces tail dependence across all H steps and N series.
        chi2_draws = rng.chisquare(df=nu, size=(n_paths, 1, 1))
        scale = np.sqrt(nu / np.maximum(chi2_draws, 1e-9))
        t_scores = z_corr * scale
        u_draws = stats.t.cdf(t_scores, df=nu)
        return np.clip(u_draws, _PROB_EPS, 1.0 - _PROB_EPS)

    if copula_type == "iman_conover":
        # Iman-Conover (1982) rank reordering on stratified Van der Waerden scores.
        u_ic = np.empty((n_paths, n_series, h), dtype=float)
        base_grid = (np.arange(1, n_paths + 1, dtype=float) - 0.5) / float(n_paths)
        vdw_scores = stats.norm.ppf(base_grid)
        for i in range(n_series):
            # Build independent random permutations of Van der Waerden normal scores.
            score_mat = np.column_stack([rng.permutation(vdw_scores) for _ in range(h)])
            if n_paths >= 4 and h >= 2:
                emp_cov = np.cov(score_mat, rowvar=False)
                emp_corr = _nearest_correlation_matrix(emp_cov)
                chol_emp = np.linalg.cholesky(emp_corr)
                # Map uncorrelated scores to target correlation chol_h and z_corr.
                if n_series > 1:
                    target_scores = (
                        0.7 * (score_mat @ np.linalg.solve(chol_emp, chol_h).T)
                        + 0.3 * z_corr[:, i, :]
                    )
                else:
                    target_scores = score_mat @ np.linalg.solve(chol_emp, chol_h).T
            else:
                target_scores = z_corr[:, i, :]
            for step_j in range(h):
                order = np.argsort(np.argsort(target_scores[:, step_j]))
                u_ic[:, i, step_j] = base_grid[order]
        return np.clip(u_ic, _PROB_EPS, 1.0 - _PROB_EPS)

    # Default: Gaussian copula.
    u_gauss = stats.norm.cdf(z_corr)
    return np.clip(u_gauss, _PROB_EPS, 1.0 - _PROB_EPS)


def _invert_marginal_quantiles_for_series(
    u_mat: np.ndarray,
    sub_pred: pd.DataFrame,
    *,
    copula_type: str,
    df_t: float = 4.0,
    residuals: np.ndarray | None = None,
) -> np.ndarray:
    """Map uniform copula draws $U \\in (0, 1)^{S \\times H}$ through marginal inverse-CDFs.

    Preserves explicit quantile columns / dicts and ``[yhat_lower, yhat, yhat_upper]`` (at
    $q = 0.10, 0.50, 0.90$) while applying Gaussian or heavy-tailed Student-$t$ quantile scaling
    in the tails ($u < 0.10$ and $u > 0.90$).
    """
    n_paths, h = u_mat.shape
    yhat = sub_pred["yhat"].to_numpy(dtype=float)[:h]

    if "yhat_lower" in sub_pred.columns and sub_pred["yhat_lower"].notna().any():
        lo_80 = sub_pred["yhat_lower"].to_numpy(dtype=float)[:h]
    else:
        res_std = float(np.std(residuals)) if residuals is not None and len(residuals) >= 2 else 0.0
        fallback_hw = np.maximum(
            res_std * _Z_80 * np.sqrt(1.0 + 0.05 * np.arange(h, dtype=float)),
            0.05 * np.maximum(np.abs(yhat), 1.0),
        )
        lo_80 = yhat - fallback_hw

    if "yhat_upper" in sub_pred.columns and sub_pred["yhat_upper"].notna().any():
        hi_80 = sub_pred["yhat_upper"].to_numpy(dtype=float)[:h]
    else:
        hi_80 = yhat + np.maximum(yhat - lo_80, 1e-3)

    lo_80 = np.minimum(lo_80, yhat)
    hi_80 = np.maximum(hi_80, yhat)

    # Check if per-row `quantiles` dicts are populated with > 3 quantile levels.
    has_rich_quantiles = False
    if "quantiles" in sub_pred.columns:
        first_q = sub_pred["quantiles"].iloc[0] if len(sub_pred) > 0 else None
        if isinstance(first_q, Mapping) and len(first_q) >= 3:
            has_rich_quantiles = True

    nu = max(2.1, float(df_t))
    if copula_type == "student_t":
        q_scores = stats.t.ppf(np.clip(u_mat, _PROB_EPS, 1.0 - _PROB_EPS), df=nu)
        q_90 = float(stats.t.ppf(0.90, df=nu))
    else:
        q_scores = stats.norm.ppf(np.clip(u_mat, _PROB_EPS, 1.0 - _PROB_EPS))
        q_90 = _Z_80

    out = np.empty((n_paths, h), dtype=float)
    for step_idx in range(h):
        u_col = u_mat[:, step_idx]
        q_col = q_scores[:, step_idx]
        med = float(yhat[step_idx])
        lo = float(lo_80[step_idx])
        hi = float(hi_80[step_idx])
        sigma_up = max((hi - med) / max(q_90, 1e-9), 0.0)
        sigma_dn = max((med - lo) / max(q_90, 1e-9), 0.0)

        if has_rich_quantiles:
            q_dict = sub_pred["quantiles"].iloc[step_idx]
            if isinstance(q_dict, Mapping) and q_dict:
                pairs: list[tuple[float, float]] = []
                for k_str, v_val in q_dict.items():
                    try:
                        ql = float(k_str)
                        vv = float(v_val)
                        if 0.0 < ql < 1.0 and np.isfinite(vv):
                            pairs.append((ql, vv))
                    except (TypeError, ValueError):
                        continue
                if len(pairs) >= 2:
                    pairs.sort(key=lambda t: t[0])
                    q_grid = np.array([p[0] for p in pairs], dtype=float)
                    v_grid = np.maximum.accumulate(np.array([p[1] for p in pairs], dtype=float))
                    interp_vals = np.interp(u_col, q_grid, v_grid)
                    # Extrapolate beyond [q_min, q_max] using the copula family tail quantile law.
                    q_min, v_min = float(q_grid[0]), float(v_grid[0])
                    q_max, v_max = float(q_grid[-1]), float(v_grid[-1])
                    q_score_min = float(
                        stats.t.ppf(q_min, df=nu)
                        if copula_type == "student_t"
                        else stats.norm.ppf(q_min)
                    )
                    q_score_max = float(
                        stats.t.ppf(q_max, df=nu)
                        if copula_type == "student_t"
                        else stats.norm.ppf(q_max)
                    )
                    high_mask = u_col > q_max
                    if np.any(high_mask):
                        interp_vals[high_mask] = v_max + sigma_up * (q_col[high_mask] - q_score_max)
                    low_mask = u_col < q_min
                    if np.any(low_mask):
                        interp_vals[low_mask] = v_min + sigma_dn * (q_col[low_mask] - q_score_min)
                    out[:, step_idx] = interp_vals
                    continue

        # Asymmetric quantile transform anchored at u=0.10 -> lo, u=0.50 -> med, u=0.90 -> hi.
        step_draws = np.where(
            q_col >= 0.0,
            med + sigma_up * q_col,
            med + sigma_dn * q_col,
        )
        out[:, step_idx] = step_draws

    return out


def sample_trajectories(
    predictions_df: pd.DataFrame,
    oof_df: pd.DataFrame | None = None,
    *,
    history_df: pd.DataFrame | None = None,
    cfg: RunConfig | None = None,
    hier_spec: HierarchySpec | None = None,
    spec: HierarchySpec | None = None,
    temp_spec: TemporalHierarchySpec | None = None,
    n_paths: int = 500,
    copula_type: CopulaType | str = "auto",
    copula: str | None = None,
    df_t: float = 4.0,
    df: float | None = None,
    cross_series: bool = False,
    reconcile: bool = True,
    simplex_sum: float | None = None,
    floor: float | None = None,
    cap: float | None = None,
    discrete_pit: bool | None = None,
    similarity_shrinkage: bool = True,
    similarity_weight: float | Literal["auto"] = "auto",
    similarity_matrix: pd.DataFrame | np.ndarray | None = None,
    similarity_method: str = "hybrid",
    seed: int = 42,
    random_seed: int | None = None,
    ts_id_col: str = "ts_id",
    date_col: str = "forecast_date",
    target_col: str = "y",
) -> pd.DataFrame:
    """Sample ``n_paths`` coherent multi-horizon and cross-series forecast trajectories (pure).

    Args:
        predictions_df: Horizon forecast DataFrame containing ``ts_id``, ``ds`` or
            ``forecast_date``, ``yhat``, and optional ``yhat_lower``, ``yhat_upper``, ``quantiles``.
        oof_df: Optional out-of-fold backtest DataFrame with ``y_true`` and ``yhat`` used to
            estimate empirical error autocorrelation and cross-series covariance.
        history_df: Optional historical training DataFrame used when ``oof_df`` is absent or to
            build hierarchy specs and trajectory similarity priors.
        cfg: Optional `RunConfig` supplying ``output.floor``, ``output.cap``,
            ``features.simplex_sum``, and ``hierarchy`` settings.
        hier_spec: Optional compiled `HierarchySpec` for cross-sectional or cross-temporal
            reconciliation of every sampled path.
        spec: Alias for ``hier_spec``.
        temp_spec: Optional compiled `TemporalHierarchySpec` for `THieF` temporal or cross-temporal
            reconciliation of every sampled path.
        n_paths: Number of joint Monte Carlo trajectories to generate (``>= 1``).
        copula_type: Copula family in
            ``("auto", "gaussian", "student_t", "iman_conover", "empirical")``.
        copula: Alias for ``copula_type``.
        df_t: Degrees of freedom $\\nu > 2$ for the multivariate Student-$t$ copula.
        df: Alias for ``df_t``.
        cross_series: When ``True``, couples trajectories across all series via a Kronecker
            $(N \\times H)$ correlation structure in addition to across horizon steps.
        reconcile: When ``True``, projects every sampled path onto active cross-sectional,
            temporal, or cross-temporal hierarchy constraints (``< 1e-6`` discrepancy).
        simplex_sum: Optional compositional sum $S > 0$; when provided (or active on ``cfg``),
            closes every sampled path across series to sum to $S$ at each timestamp.
        floor: Optional hard lower bound applied to every sampled path.
        cap: Optional hard upper bound applied to every sampled path.
        discrete_pit: Optional override enabling randomized PIT for discrete count residuals.
        similarity_shrinkage: When ``True`` (default), regularizes the cross-series correlation
            matrix toward the pairwise trajectory similarity matrix when OOF samples are scarce or
            high-dimensional.
        similarity_weight: Shrinkage weight $\\lambda_{\\text{sim}} \\in [0, 1]$ or ``"auto"``.
        similarity_matrix: Optional precomputed $(N, N)$ pairwise trajectory similarity matrix.
        similarity_method: Similarity kernel method used when computing the similarity prior on
            the fly (default ``"hybrid"``).
        seed: Deterministic random seed.
        random_seed: Alias for ``seed``.
        ts_id_col: Series identifier column name.
        date_col: Forecast timestamp column name.
        target_col: Historical target column name.

    Returns:
        Tidy DataFrame with columns ``["path_id", "ts_id", "model_type", "forecast_date",
        "ds", "horizon_step", "yhat_sample", "yhat"]``
        (plus ``"yhat_point"`` holding the base point forecast) and `.attrs` metadata for
        downstream coherence verification and tail-risk computation.
    """
    _ = discrete_pit
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ConfigError(f"n_paths must be a positive integer >= 1, got {n_paths!r}")

    eff_copula_req = str(copula if copula is not None else copula_type).strip().lower()
    if eff_copula_req not in VALID_COPULA_TYPES:
        raise ConfigError(
            f"Unknown copula_type {eff_copula_req!r}; expected one of {sorted(VALID_COPULA_TYPES)}"
        )
    eff_df_t = float(df if df is not None else df_t)
    if eff_df_t <= 2.0:
        raise ConfigError(f"Student-t degrees of freedom df_t must be > 2.0, got {eff_df_t}")
    _ = _resolve_similarity_shrinkage_weight(2, 0, similarity_weight)

    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(
            columns=[
                "path_id",
                ts_id_col,
                "model_type",
                "forecast_date",
                "ds",
                "horizon_step",
                "yhat_sample",
                "yhat",
            ]
        )

    eff_seed = int(random_seed if random_seed is not None else seed)
    rng = np.random.default_rng(eff_seed)

    eff_floor = floor if floor is not None else (cfg.output.floor if cfg is not None else None)
    eff_cap = cap if cap is not None else (cfg.output.cap if cfg is not None else None)
    if eff_floor is not None and eff_cap is not None and float(eff_floor) > float(eff_cap):
        raise ConfigError(f"floor ({eff_floor}) cannot exceed cap ({eff_cap})")

    eff_simplex_sum = simplex_sum
    if eff_simplex_sum is None and cfg is not None:
        if cfg.features.simplex_sum is not None:
            eff_simplex_sum = float(cfg.features.simplex_sum)
        elif cfg.features.transform in ("clr", "ilr"):
            eff_simplex_sum = detect_simplex_sum(predictions_df["yhat"].to_numpy(dtype=float))

    eff_hier_spec = hier_spec if hier_spec is not None else spec
    if eff_hier_spec is None:
        eff_hier_spec = predictions_df.attrs.get("hierarchy_spec")
    if (
        eff_hier_spec is None
        and cfg is not None
        and cfg.hierarchy.enabled
        and (cfg.hierarchy.levels or getattr(cfg.hierarchy, "similarity_hierarchy", False))
        and history_df is not None
        and not history_df.empty
    ):
        needed_cols = {c for lvl in cfg.hierarchy.levels for c in lvl} - {"similarity_cluster_id"}
        if needed_cols.issubset(history_df.columns):
            _, eff_hier_spec = build_hierarchy(history_df, cfg)

    eff_temp_spec = temp_spec
    if eff_temp_spec is None:
        eff_temp_spec = predictions_df.attrs.get("temporal_spec")

    dcol = date_col
    if dcol not in predictions_df.columns:
        for cand in ("forecast_date", "ds", "date"):
            if cand in predictions_df.columns:
                dcol = cand
                break

    work_df = predictions_df.copy()
    if ts_id_col not in work_df.columns:
        work_df[ts_id_col] = "series_0"
    if "model_type" not in work_df.columns:
        work_df["model_type"] = "forecaster"

    eff_freq = cfg.data.freq if cfg is not None else "D"
    base_season = seasonal_period(eff_freq)

    # Resolve overall copula family if "auto" was requested.
    all_res = _extract_residual_series(oof_df, history_df, ts_id_col=ts_id_col)
    selected_copula, auto_df_t = _select_copula_type(eff_copula_req, all_res)
    if eff_copula_req == "auto" and df is None and df_t == 4.0:
        eff_df_t = auto_df_t

    use_temporal_reconcile = bool(
        reconcile
        and (
            eff_temp_spec is not None
            or (
                cfg is not None
                and (
                    cfg.hierarchy.reconcile_temporal
                    or cfg.hierarchy.cross_temporal
                    or cfg.hierarchy.temporal_aggregations is not None
                )
            )
        )
    )
    use_cs_reconcile = bool(reconcile and eff_hier_spec is not None)

    models = work_df["model_type"].astype(str).drop_duplicates().tolist()
    path_frames: list[pd.DataFrame] = []
    agg_by_path_node_k: dict[tuple[int, str, int], np.ndarray] = {}
    agg_by_node_k_p0: dict[tuple[str, int], np.ndarray] = {}
    resolved_tspec: TemporalHierarchySpec | None = eff_temp_spec
    last_sim_shrinkage_weight: float = 0.0
    last_corr_n: np.ndarray | None = None

    for m_name in models:
        m_df = work_df[work_df["model_type"].astype(str) == m_name].copy()
        dates = sorted(m_df[dcol].drop_duplicates().tolist())
        h = len(dates)
        if h == 0:
            continue

        if eff_hier_spec is not None:
            present_ids = set(m_df[ts_id_col].astype(str).unique())
            if set(eff_hier_spec.node_ids).issubset(present_ids):
                series_ids = list(eff_hier_spec.node_ids)
            else:
                series_ids = [
                    tid for tid in eff_hier_spec.bottom_ids if tid in present_ids
                ] or sorted(present_ids)
        else:
            series_ids = m_df[ts_id_col].astype(str).drop_duplicates().tolist()

        n_series = len(series_ids)
        # Estimate horizon correlation matrices per series and average for cross-series sampling.
        corr_h_by_series: dict[str, np.ndarray] = {}
        res_by_series: dict[str, np.ndarray] = {}
        for tid in series_ids:
            res_by_series[tid] = _extract_residual_series(
                oof_df,
                history_df,
                ts_id=tid,
                model_type=m_name,
                ts_id_col=ts_id_col,
            )
            corr_h_by_series[tid] = _estimate_horizon_correlation(
                h,
                oof_df=oof_df,
                history_df=history_df,
                ts_id=tid,
                model_type=m_name,
                ts_id_col=ts_id_col,
                rng=rng,
            )

        if (cross_series or use_cs_reconcile or eff_simplex_sum is not None) and n_series > 1:
            mean_corr_h = _nearest_correlation_matrix(
                np.mean([corr_h_by_series[tid] for tid in series_ids], axis=0)
            )
            corr_n = _estimate_cross_series_correlation(
                series_ids,
                oof_df=oof_df,
                history_df=history_df,
                model_type=m_name,
                ts_id_col=ts_id_col,
            )
            if similarity_shrinkage:
                n_oof_dates = 0
                if oof_df is not None and not oof_df.empty and ts_id_col in oof_df.columns:
                    oof_sub = oof_df[oof_df[ts_id_col].astype(str).isin(series_ids)]
                    if "model_type" in oof_sub.columns:
                        m_sub = oof_sub[oof_sub["model_type"].astype(str) == m_name]
                        if not m_sub.empty:
                            oof_sub = m_sub
                    oof_dcol = (
                        "forecast_date"
                        if "forecast_date" in oof_sub.columns
                        else ("ds" if "ds" in oof_sub.columns else None)
                    )
                    if oof_dcol is not None and not oof_sub.empty:
                        n_oof_dates = int(oof_sub[oof_dcol].nunique())

                sim_prior: pd.DataFrame | np.ndarray | None = similarity_matrix
                if sim_prior is None:
                    sim_prior = predictions_df.attrs.get("similarity_matrix")
                if sim_prior is None:
                    from .similarity import compute_similarity_matrix

                    if (
                        history_df is not None
                        and not history_df.empty
                        and ts_id_col in history_df.columns
                    ):
                        h_dcol = (
                            "ds"
                            if "ds" in history_df.columns
                            else (
                                "forecast_date" if "forecast_date" in history_df.columns else dcol
                            )
                        )
                        h_ycol = (
                            target_col
                            if target_col in history_df.columns
                            else ("y" if "y" in history_df.columns else "y_true")
                        )
                        h_sub = history_df[history_df[ts_id_col].astype(str).isin(series_ids)]
                        if not h_sub.empty and h_dcol in h_sub.columns and h_ycol in h_sub.columns:
                            sim_prior = compute_similarity_matrix(
                                h_sub,
                                method=similarity_method,
                                ts_id_col=ts_id_col,
                                date_col=h_dcol,
                                target_col=h_ycol,
                            ).matrix_df
                    elif (
                        oof_df is not None
                        and not oof_df.empty
                        and ts_id_col in oof_df.columns
                        and "y_true" in oof_df.columns
                    ):
                        o_dcol = "forecast_date" if "forecast_date" in oof_df.columns else "ds"
                        o_sub = oof_df[oof_df[ts_id_col].astype(str).isin(series_ids)]
                        if not o_sub.empty and o_dcol in o_sub.columns:
                            sim_prior = compute_similarity_matrix(
                                o_sub,
                                method=similarity_method,
                                ts_id_col=ts_id_col,
                                date_col=o_dcol,
                                target_col="y_true",
                            ).matrix_df
                    elif h >= 3 and "yhat" in m_df.columns:
                        sim_prior = compute_similarity_matrix(
                            m_df,
                            method=similarity_method,
                            ts_id_col=ts_id_col,
                            date_col=dcol,
                            target_col="yhat",
                        ).matrix_df

                if sim_prior is not None:
                    last_sim_shrinkage_weight = _resolve_similarity_shrinkage_weight(
                        n_series,
                        n_oof_dates,
                        similarity_weight,
                    )
                    corr_n = shrink_correlation_toward_similarity(
                        corr_n,
                        sim_prior,
                        n_samples=n_oof_dates,
                        weight=similarity_weight,
                        series_ids=series_ids,
                    )
            last_corr_n = corr_n
            u_all = _sample_copula_uniforms(
                mean_corr_h,
                n_paths,
                selected_copula,
                df_t=eff_df_t,
                corr_n=corr_n,
                rng=rng,
            )
        else:
            u_all = np.empty((n_paths, n_series, h), dtype=float)
            for idx_s, tid in enumerate(series_ids):
                u_s = _sample_copula_uniforms(
                    corr_h_by_series[tid],
                    n_paths,
                    selected_copula,
                    df_t=eff_df_t,
                    corr_n=None,
                    rng=rng,
                )
                u_all[:, idx_s, :] = u_s[:, 0, :]

        # Map uniform draws through each series' marginal quantile function -> y_draws.
        y_draws = np.empty((n_paths, n_series, h), dtype=float)
        point_mat = np.empty((n_series, h), dtype=float)
        for idx_s, tid in enumerate(series_ids):
            sub_s = (
                m_df[m_df[ts_id_col].astype(str) == tid]
                .sort_values(dcol, kind="mergesort")
                .reset_index(drop=True)
            )
            if len(sub_s) < h:
                sub_s = (
                    sub_s.set_index(dcol).reindex(dates).ffill().bfill().fillna(0.0).reset_index()
                )
            point_mat[idx_s, :] = sub_s["yhat"].to_numpy(dtype=float)[:h]
            y_draws[:, idx_s, :] = _invert_marginal_quantiles_for_series(
                u_all[:, idx_s, :],
                sub_s,
                copula_type=selected_copula,
                df_t=eff_df_t,
                residuals=res_by_series[tid],
            )

        # Apply floor / cap prior to compositional or hierarchical projection.
        if eff_floor is not None:
            y_draws = np.maximum(float(eff_floor), y_draws)
        if eff_cap is not None:
            y_draws = np.minimum(float(eff_cap), y_draws)

        # Apply temporal and/or cross-sectional reconciliation across every path when requested.
        node_ids = list(series_ids)
        if use_temporal_reconcile:
            eff_aggs = (
                list(eff_temp_spec.aggregations)
                if eff_temp_spec is not None
                else (
                    list(cfg.hierarchy.temporal_aggregations)
                    if (cfg is not None and cfg.hierarchy.temporal_aggregations is not None)
                    else None
                )
            )
            resolved_tspec = (
                eff_temp_spec
                if (eff_temp_spec is not None and eff_temp_spec.horizon == h)
                else build_temporal_hierarchy(eff_freq, h, eff_aggs)
            )
            g_temp_full = reconcile_temporal_matrix(
                resolved_tspec, "wls_struct", use_full_horizon=True
            )
            bot_row_indices = np.concatenate(
                [
                    np.arange(
                        cyc * resolved_tspec.n_total_per_cycle
                        + resolved_tspec.level_slices[1].start,
                        cyc * resolved_tspec.n_total_per_cycle
                        + resolved_tspec.level_slices[1].stop,
                    )
                    for cyc in range(resolved_tspec.n_macro_periods)
                ]
            )
            p_temp_bottom = resolved_tspec.full_summing_matrix[bot_row_indices, :] @ g_temp_full
            s_temp_full = resolved_tspec.full_summing_matrix
            pad_h = resolved_tspec.padded_horizon

            for idx_s in range(n_series):
                mat_s = y_draws[:, idx_s, :]
                if pad_h > h:
                    s_period = max(1, min(base_season, h))
                    padded_mat = np.empty((n_paths, pad_h), dtype=float)
                    padded_mat[:, :h] = mat_s
                    for t_pad in range(h, pad_h):
                        padded_mat[:, t_pad] = (
                            padded_mat[:, t_pad - s_period] if s_period > 1 else mat_s[:, -1]
                        )
                else:
                    padded_mat = mat_s[:, :pad_h].copy()

                y_cycles = padded_mat @ s_temp_full.T
                b_proj = y_cycles @ p_temp_bottom.T
                if eff_floor is not None:
                    b_proj = np.maximum(float(eff_floor), b_proj)
                if eff_cap is not None:
                    b_proj = np.minimum(float(eff_cap), b_proj)
                y_draws[:, idx_s, :] = b_proj[:, :h]

        if use_cs_reconcile and eff_hier_spec is not None:
            node_ids = list(eff_hier_spec.node_ids)
            bottom_ids = list(eff_hier_spec.bottom_ids)
            id_to_idx = {tid: i for i, tid in enumerate(series_ids)}
            if set(node_ids).issubset(set(series_ids)):
                # Full hierarchy present: reconcile via G matrix then lift via S_cs @ b_tilde.
                full_idx = [id_to_idx[nid] for nid in node_ids]
                y_full = y_draws[:, full_idx, :]
                point_full = point_mat[full_idx, :]
                cs_method: Any = (
                    cfg.hierarchy.reconciliation_methods[0]
                    if (cfg is not None and cfg.hierarchy.reconciliation_methods)
                    else "wls_struct"
                )
                if cs_method in ("cross_temporal_mint", "mint_nonneg"):
                    cs_method = "wls_struct"
                try:
                    g_cs = reconcile_matrix(eff_hier_spec, cs_method)
                except Exception:  # noqa: BLE001 - fallback to structural WLS if residual matrix is unavailable
                    g_cs = reconcile_matrix(eff_hier_spec, "wls_struct")
                b_tilde = np.einsum("ij,sjt->sit", g_cs, y_full)
                b_point = g_cs @ point_full
            else:
                bot_idx = [id_to_idx[bid] for bid in bottom_ids if bid in id_to_idx]
                b_tilde = y_draws[:, bot_idx, :]
                b_point = point_mat[bot_idx, :]

            if eff_floor is not None:
                b_tilde = np.maximum(float(eff_floor), b_tilde)
                b_point = np.maximum(float(eff_floor), b_point)
            if eff_cap is not None and eff_hier_spec.n_aggregated == 0:
                b_tilde = np.minimum(float(eff_cap), b_tilde)
                b_point = np.minimum(float(eff_cap), b_point)

            y_draws = np.einsum("ij,sjt->sit", eff_hier_spec.summing_matrix, b_tilde)
            point_mat = eff_hier_spec.summing_matrix @ b_point
            n_series = len(node_ids)

        # Populate temporal bucket aggregations for temporal and cross-temporal coherence checks.
        if resolved_tspec is None:
            resolved_tspec = build_temporal_hierarchy(eff_freq, max(1, h))
        pad_h_final = resolved_tspec.padded_horizon
        s_period_final = max(1, min(base_season, h))
        for p_idx in range(n_paths):
            for n_idx, nid in enumerate(node_ids):
                series_h = y_draws[p_idx, n_idx, :]
                if pad_h_final > h:
                    padded_s = np.empty(pad_h_final, dtype=float)
                    padded_s[:h] = series_h
                    for t_pad in range(h, pad_h_final):
                        padded_s[t_pad] = (
                            padded_s[t_pad - s_period_final] if s_period_final > 1 else series_h[-1]
                        )
                else:
                    padded_s = series_h[:pad_h_final].copy()

                agg_by_path_node_k[(p_idx, nid, 1)] = series_h.copy()
                if p_idx == 0:
                    agg_by_node_k_p0[(nid, 1)] = series_h.copy()
                for k_fac in resolved_tspec.aggregations:
                    k_int = int(k_fac)
                    if k_int == 1:
                        continue
                    agg_arr = aggregate_temporal_history(padded_s, k_int)
                    agg_by_path_node_k[(p_idx, nid, k_int)] = agg_arr
                    if p_idx == 0:
                        agg_by_node_k_p0[(nid, k_int)] = agg_arr

        # Vectorized tidy DataFrame construction of length n_paths * n_series * h.
        path_col = np.repeat(np.arange(n_paths, dtype=int), n_series * h)
        ts_col = np.tile(np.repeat(np.asarray(node_ids, dtype=object), h), n_paths)
        dt_col = np.tile(np.asarray(dates), n_paths * n_series)
        step_col = np.tile(np.arange(1, h + 1, dtype=int), n_paths * n_series)
        val_col = y_draws.reshape(-1)
        pt_col = np.tile(point_mat.reshape(-1), n_paths)

        m_frame = pd.DataFrame(
            {
                "path_id": path_col,
                ts_id_col: ts_col,
                "model_type": m_name,
                "forecast_date": dt_col,
                "ds": dt_col,
                "horizon_step": step_col,
                "yhat_sample": val_col,
                "yhat": val_col.copy(),
                "yhat_point": pt_col,
            }
        )
        if eff_simplex_sum is not None:
            m_frame = enforce_simplex_closure(
                m_frame,
                simplex_sum=float(eff_simplex_sum),
                ts_id_col=ts_id_col,
                date_col="forecast_date",
                value_col="yhat_sample",
            )
            m_frame["yhat"] = m_frame["yhat_sample"].to_numpy(dtype=float)

        path_frames.append(m_frame)

    out_df = (
        pd.concat(path_frames, ignore_index=True)
        if path_frames
        else pd.DataFrame(
            columns=[
                "path_id",
                ts_id_col,
                "model_type",
                "forecast_date",
                "ds",
                "horizon_step",
                "yhat_sample",
                "yhat",
            ]
        )
    )
    out_df.attrs["copula_type_selected"] = selected_copula
    out_df.attrs["df_t"] = eff_df_t
    out_df.attrs["cross_series"] = bool(cross_series)
    out_df.attrs["similarity_shrinkage_weight"] = float(last_sim_shrinkage_weight)
    if last_corr_n is not None:
        out_df.attrs["cross_series_correlation"] = last_corr_n
    if eff_hier_spec is not None:
        out_df.attrs["hierarchy_spec"] = eff_hier_spec
    if resolved_tspec is not None:
        out_df.attrs["temporal_spec"] = resolved_tspec
    from .temporal_reconciliation import SafeArrayDict

    out_df.attrs["aggregated_by_node_k"] = SafeArrayDict(agg_by_node_k_p0)
    out_df.attrs["aggregated_by_path_node_k"] = SafeArrayDict(agg_by_path_node_k)
    return out_df


sample_copula_trajectories = sample_trajectories
sample_joint_trajectories = sample_trajectories


def evaluate_trajectory_paths(
    trajectories_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    *,
    p: float = 0.5,
    ts_id_col: str = "ts_id",
    date_col: str = "forecast_date",
    target_col: str = "y",
) -> dict[str, float]:
    """Compute multivariate probabilistic trajectory scores (`CRPS`, `Energy`, `Variogram`).

    Args:
        trajectories_df: Output DataFrame from `sample_trajectories`.
        actuals_df: Realized future actuals DataFrame containing ``ts_id``, ``ds`` or
            ``forecast_date``, and ``y`` or ``y_true``.
        p: Variogram order $p > 0$ (default ``0.5`` per Scheuerer & Hamill 2015).
        ts_id_col: Series identifier column name.
        date_col: Timestamp column name.
        target_col: Realized target value column name.

    Returns:
        Dictionary with keys ``crps``, ``energy_score``, ``variogram_score``, and
        ``variogram_score_p1``.
    """
    if trajectories_df is None or trajectories_df.empty or actuals_df is None or actuals_df.empty:
        return {
            "crps": float("nan"),
            "energy_score": float("nan"),
            "variogram_score": float("nan"),
            "variogram_score_p1": float("nan"),
        }

    t_dcol = (
        date_col
        if date_col in trajectories_df.columns
        else ("ds" if "ds" in trajectories_df.columns else "forecast_date")
    )
    a_dcol = (
        date_col
        if date_col in actuals_df.columns
        else ("ds" if "ds" in actuals_df.columns else "forecast_date")
    )
    a_ycol = (
        target_col
        if target_col in actuals_df.columns
        else (
            "y_true"
            if "y_true" in actuals_df.columns
            else ("y" if "y" in actuals_df.columns else "yhat")
        )
    )
    vcol = "yhat_sample" if "yhat_sample" in trajectories_df.columns else "yhat"

    act = actuals_df.copy()
    if ts_id_col not in act.columns:
        act[ts_id_col] = (
            trajectories_df[ts_id_col].iloc[0]
            if ts_id_col in trajectories_df.columns
            else "series_0"
        )
    act[a_dcol] = pd.to_datetime(act[a_dcol])

    traj = trajectories_df.copy()
    if ts_id_col not in traj.columns:
        traj[ts_id_col] = "series_0"
    traj[t_dcol] = pd.to_datetime(traj[t_dcol])

    crps_vals: list[float] = []
    energy_vals: list[float] = []
    vario_p_vals: list[float] = []
    vario_1_vals: list[float] = []

    group_cols = [ts_id_col] + (["model_type"] if "model_type" in traj.columns else [])
    for key, grp in traj.groupby(group_cols, sort=False):
        tid = key[0] if isinstance(key, tuple) else key
        act_sub = act[act[ts_id_col].astype(str) == str(tid)].sort_values(a_dcol, kind="mergesort")
        if act_sub.empty:
            continue
        piv = grp.pivot_table(index="path_id", columns=t_dcol, values=vcol, aggfunc="first")
        common_dates = [dt for dt in act_sub[a_dcol].tolist() if dt in piv.columns]
        if not common_dates:
            continue

        y_true_vec = act_sub.set_index(a_dcol).reindex(common_dates)[a_ycol].to_numpy(dtype=float)
        valid_mask = np.isfinite(y_true_vec)
        if not np.any(valid_mask):
            continue
        y_true_vec = y_true_vec[valid_mask]
        paths_mat = piv[common_dates].to_numpy(dtype=float)[:, valid_mask]
        n_paths, h_len = paths_mat.shape
        if n_paths < 1 or h_len < 1:
            continue

        # 1. Marginal CRPS averaged across horizon steps using O(S log S) sorted order statistics.
        step_crps: list[float] = []
        for h_idx in range(h_len):
            col_sorted = np.sort(paths_mat[:, h_idx])
            term1 = float(np.mean(np.abs(col_sorted - y_true_vec[h_idx])))
            if n_paths >= 2:
                weights = 2.0 * np.arange(1, n_paths + 1, dtype=float) - float(n_paths) - 1.0
                term2 = float(np.sum(weights * col_sorted)) / float(n_paths * n_paths)
            else:
                term2 = 0.0
            step_crps.append(max(0.0, term1 - term2))
        crps_vals.append(float(np.mean(step_crps)))

        # 2. Multivariate Energy Score across the H-dimensional trajectory vector.
        diff_true = np.linalg.norm(paths_mat - y_true_vec[np.newaxis, :], axis=1)
        es_term1 = float(np.mean(diff_true))
        if n_paths >= 2:
            # Subsample up to 250 paths for O(S^2) pairwise norm calculation to stay fast.
            sub_mat = paths_mat[: min(n_paths, 250)]
            s_sub = len(sub_mat)
            pw_diffs = np.linalg.norm(sub_mat[:, np.newaxis, :] - sub_mat[np.newaxis, :, :], axis=2)
            es_term2 = 0.5 * float(np.sum(pw_diffs)) / float(s_sub * s_sub)
        else:
            es_term2 = 0.0
        energy_vals.append(max(0.0, es_term1 - es_term2))

        # 3. Variogram Score of order p and order 1 across all horizon pairs (i < j).
        if h_len >= 2:
            true_diff = np.abs(y_true_vec[:, np.newaxis] - y_true_vec[np.newaxis, :])
            path_diffs = np.abs(paths_mat[:, :, np.newaxis] - paths_mat[:, np.newaxis, :])
            tri_i, tri_j = np.triu_indices(h_len, k=1)

            true_p = np.power(true_diff[tri_i, tri_j], float(p))
            sim_p = np.mean(np.power(path_diffs[:, tri_i, tri_j], float(p)), axis=0)
            vario_p_vals.append(float(np.mean((true_p - sim_p) ** 2)))

            true_1 = true_diff[tri_i, tri_j]
            sim_1 = np.mean(path_diffs[:, tri_i, tri_j], axis=0)
            vario_1_vals.append(float(np.mean((true_1 - sim_1) ** 2)))
        else:
            vario_p_vals.append(float(crps_vals[-1] ** 2))
            vario_1_vals.append(float(crps_vals[-1] ** 2))

    return {
        "crps": float(np.mean(crps_vals)) if crps_vals else float("nan"),
        "energy_score": float(np.mean(energy_vals)) if energy_vals else float("nan"),
        "variogram_score": float(np.mean(vario_p_vals)) if vario_p_vals else float("nan"),
        "variogram_score_p1": float(np.mean(vario_1_vals)) if vario_1_vals else float("nan"),
    }


def compute_tail_risk(
    trajectories_df: pd.DataFrame,
    *,
    predictions_df: pd.DataFrame | None = None,
    actuals_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    lead_times: Mapping[str, float | int] | float | int | None = None,
    lead_time: int | float | None = None,
    window: int = 7,
    alphas: Sequence[float] = (0.90, 0.95, 0.99),
    alpha: float = 0.95,
    threshold: float | Mapping[str, float] | None = None,
    holding_cost: float | None = None,
    stockout_cost: float | None = None,
    ts_id_col: str = "ts_id",
    date_col: str = "forecast_date",
) -> pd.DataFrame:
    """Compute cumulative multi-horizon and rolling-window tail-risk (`VaR`/`CVaR`) per series.

    Args:
        trajectories_df: Tidy trajectory DataFrame returned by `sample_trajectories` (or a standard
            predictions DataFrame, in which case trajectories are sampled automatically).
        predictions_df: Optional point/interval forecast DataFrame.
        actuals_df: Optional realized actuals DataFrame for computing `crps`, `energy_score`, and
            `variogram_score`.
        oof_df: Optional OOF backtest DataFrame used if trajectories need to be sampled or scored.
        lead_times: Optional per-series or scalar lead-time horizon $L \\le H$ over which cumulative
            demand is summed.
        lead_time: Alias for scalar ``lead_times``.
        window: Rolling peak-demand window length $W \\ge 1$ (default ``7``).
        alphas: Sequence of quantile levels for Value-at-Risk and Conditional Value-at-Risk.
        alpha: Primary quantile level for ``var_alpha`` and ``cvar_alpha`` (default ``0.95``).
        threshold: Optional capacity/SLA threshold $\\tau$ for ``exceedance_prob`` and
            ``expected_shortfall_units``.
        holding_cost: Optional per-unit holding cost $h$ to compute optimal newsvendor quantile.
        stockout_cost: Optional per-unit stockout penalty $b$ for optimal newsvendor quantile.
        ts_id_col: Series identifier column name.
        date_col: Forecast timestamp column name.

    Returns:
        DataFrame with one row per ``(ts_id, model_type)`` containing cumulative demand moments,
        ``iid_std_cumulative_demand``, ``copula_amplification_ratio``, ``var_90``, ``cvar_90``,
        ``var_95``, ``cvar_95``, ``var_99``, ``cvar_99``, ``var_alpha``, ``cvar_alpha``, rolling
        peak-window tail risk, ``exceedance_prob``, ``expected_shortfall_units``, proper scoring
        rules, and ``risk_verdict``.
    """
    if trajectories_df is None or trajectories_df.empty:
        return pd.DataFrame(
            columns=[
                ts_id_col,
                "model_type",
                "copula_type",
                "n_paths",
                "horizon",
                "window",
                "expected_cumulative_demand",
                "std_cumulative_demand",
                "iid_std_cumulative_demand",
                "copula_amplification_ratio",
                "var_90",
                "cvar_90",
                "var_95",
                "cvar_95",
                "var_99",
                "cvar_99",
                "var_alpha",
                "cvar_alpha",
                "peak_window_mean",
                "peak_window_var",
                "peak_window_cvar",
                "exceedance_prob",
                "expected_shortfall_units",
                "crps",
                "energy_score",
                "variogram_score",
                "risk_verdict",
            ]
        )

    # If the caller passed point predictions without `path_id`, sample trajectories first.
    if "path_id" not in trajectories_df.columns:
        traj_df = sample_trajectories(
            trajectories_df,
            oof_df=oof_df,
            ts_id_col=ts_id_col,
            date_col=date_col,
        )
    else:
        traj_df = trajectories_df

    copula_used = str(traj_df.attrs.get("copula_type_selected", "gaussian"))
    dcol = (
        date_col
        if date_col in traj_df.columns
        else ("ds" if "ds" in traj_df.columns else "forecast_date")
    )
    vcol = "yhat_sample" if "yhat_sample" in traj_df.columns else "yhat"

    work = traj_df.copy()
    if ts_id_col not in work.columns:
        work[ts_id_col] = "series_0"
    if "model_type" not in work.columns:
        work["model_type"] = "forecaster"

    eff_alpha = float(alpha)
    if (
        holding_cost is not None
        and stockout_cost is not None
        and (float(holding_cost) + float(stockout_cost)) > 0.0
    ):
        eff_alpha = float(
            np.clip(
                float(stockout_cost) / (float(holding_cost) + float(stockout_cost)),
                0.01,
                0.999,
            )
        )

    all_alphas = sorted({0.90, 0.95, 0.99, eff_alpha, *(float(a) for a in alphas)})
    eff_lead = lead_time if lead_time is not None else lead_times

    rows: list[dict[str, Any]] = []
    for (tid, m_name), grp in work.groupby([ts_id_col, "model_type"], sort=False):
        tid_str = str(tid)
        piv = grp.pivot_table(index="path_id", columns=dcol, values=vcol, aggfunc="first")
        piv = piv.reindex(columns=sorted(piv.columns))
        paths_mat = piv.to_numpy(dtype=float)
        n_paths, h_full = paths_mat.shape
        if n_paths == 0 or h_full == 0:
            continue

        if isinstance(eff_lead, Mapping):
            lt_val = round(float(eff_lead.get(tid_str, h_full)))
        elif isinstance(eff_lead, (int, float, np.number)):
            lt_val = round(float(eff_lead))
        else:
            lt_val = h_full
        lt_steps = max(1, min(lt_val, h_full))

        lead_mat = paths_mat[:, :lt_steps]
        cum_demand = np.sum(lead_mat, axis=1)
        exp_cum = float(np.mean(cum_demand))
        std_cum = float(np.std(cum_demand, ddof=1 if n_paths > 1 else 0))

        # Independent-horizon (i.i.d.) cumulative std from per-step marginal variances.
        step_vars = np.var(lead_mat, axis=0, ddof=1 if n_paths > 1 else 0)
        iid_std = float(np.sqrt(np.maximum(np.sum(step_vars), 0.0)))
        if predictions_df is not None and not predictions_df.empty:
            p_sub = predictions_df[predictions_df[ts_id_col].astype(str) == tid_str]
            if "model_type" in p_sub.columns:
                p_m = p_sub[p_sub["model_type"].astype(str) == str(m_name)]
                if not p_m.empty:
                    p_sub = p_m
            if not p_sub.empty and "yhat_lower" in p_sub.columns and "yhat_upper" in p_sub.columns:
                p_dcol = (
                    dcol if dcol in p_sub.columns else ("ds" if "ds" in p_sub.columns else None)
                )
                if p_dcol is not None:
                    p_sub = p_sub.sort_values(p_dcol, kind="mergesort")
                lo_v = p_sub["yhat_lower"].to_numpy(dtype=float)[:lt_steps]
                hi_v = p_sub["yhat_upper"].to_numpy(dtype=float)[:lt_steps]
                sig_p = 0.5 * np.abs(hi_v - lo_v) / _Z_80
                iid_from_pred = float(np.sqrt(np.maximum(np.sum(sig_p**2), 0.0)))
                if iid_std <= 1e-12 and iid_from_pred > 0.0:
                    iid_std = iid_from_pred

        amp_ratio = float(std_cum / max(iid_std, 1e-12)) if iid_std > 1e-12 else 1.0

        var_map: dict[float, float] = {}
        cvar_map: dict[float, float] = {}
        for a_lvl in all_alphas:
            q_val = float(np.quantile(cum_demand, a_lvl))
            tail_vals = cum_demand[cum_demand >= q_val - 1e-12]
            cvar_val = float(np.mean(tail_vals)) if len(tail_vals) > 0 else q_val
            var_map[a_lvl] = q_val
            cvar_map[a_lvl] = max(q_val, cvar_val)

        # Enforce exact monotonicity across quantile levels.
        var_90 = var_map[0.90]
        var_95 = max(var_90, var_map[0.95])
        var_99 = max(var_95, var_map[0.99])
        cvar_90 = max(var_90, cvar_map[0.90])
        cvar_95 = max(var_95, cvar_90, cvar_map[0.95])
        cvar_99 = max(var_99, cvar_95, cvar_map[0.99])
        var_a = var_map[eff_alpha]
        cvar_a = max(var_a, cvar_map[eff_alpha])

        # Rolling window peak demand across the full horizon H.
        win_len = max(1, min(int(window), h_full))
        if win_len == 1:
            peak_per_path = np.max(paths_mat, axis=1)
        elif win_len >= h_full:
            peak_per_path = np.sum(paths_mat, axis=1)
        else:
            csum = np.pad(np.cumsum(paths_mat, axis=1), ((0, 0), (1, 0)), mode="constant")
            rolling_sums = csum[:, win_len:] - csum[:, :-win_len]
            peak_per_path = np.max(rolling_sums, axis=1)

        peak_mean = float(np.mean(peak_per_path))
        peak_var = float(np.quantile(peak_per_path, eff_alpha))
        peak_tail = peak_per_path[peak_per_path >= peak_var - 1e-12]
        peak_cvar = float(np.mean(peak_tail)) if len(peak_tail) > 0 else peak_var

        if isinstance(threshold, Mapping):
            tau = float(threshold.get(tid_str, var_90))
        elif threshold is not None:
            tau = float(threshold)
        else:
            tau = var_90
        exceed_prob = float(np.mean(cum_demand > tau))
        shortfall_units = float(np.mean(np.maximum(0.0, cum_demand - tau)))

        # Compute trajectory scores if actuals_df (or oof_df) is provided.
        eval_ref = actuals_df if (actuals_df is not None and not actuals_df.empty) else None
        if eval_ref is not None:
            score_dict = evaluate_trajectory_paths(
                grp,
                eval_ref,
                ts_id_col=ts_id_col,
                date_col=dcol,
            )
        else:
            # Self-consistency dispersion scores around point trajectory when actuals are absent.
            if "yhat_point" in grp.columns:
                ref_df = (
                    grp[[ts_id_col, dcol, "yhat_point"]]
                    .drop_duplicates(subset=[ts_id_col, dcol])
                    .rename(columns={"yhat_point": "y"})
                )
            else:
                ref_df = pd.DataFrame(
                    grp.groupby([ts_id_col, dcol], as_index=False)[vcol].mean()
                ).rename(columns={vcol: "y"})
            score_dict = evaluate_trajectory_paths(
                grp,
                ref_df,
                ts_id_col=ts_id_col,
                date_col=dcol,
                target_col="y",
            )

        if amp_ratio >= 1.15 or ((cvar_95 - exp_cum) / max(abs(exp_cum), 1e-6) >= 0.35):
            verdict = "amplified_tail_risk"
        elif amp_ratio >= 1.03:
            verdict = "moderate_tail_risk"
        else:
            verdict = "nominal_tail_risk"

        row_dict: dict[str, Any] = {
            ts_id_col: tid_str,
            "model_type": str(m_name),
            "copula_type": copula_used,
            "n_paths": int(n_paths),
            "horizon": int(lt_steps),
            "window": int(win_len),
            "expected_cumulative_demand": exp_cum,
            "std_cumulative_demand": std_cum,
            "iid_std_cumulative_demand": iid_std,
            "copula_amplification_ratio": amp_ratio,
            "var_90": var_90,
            "cvar_90": cvar_90,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "var_99": var_99,
            "cvar_99": cvar_99,
            "var_alpha": var_a,
            "cvar_alpha": cvar_a,
            "peak_window_mean": peak_mean,
            "peak_window_var": peak_var,
            "peak_window_cvar": max(peak_var, peak_cvar),
            "exceedance_prob": exceed_prob,
            "expected_shortfall_units": shortfall_units,
            "crps": float(score_dict["crps"]),
            "energy_score": float(score_dict["energy_score"]),
            "variogram_score": float(score_dict["variogram_score"]),
            "risk_verdict": verdict,
        }
        rows.append(row_dict)

    return pd.DataFrame(rows)


tail_risk_df = compute_tail_risk
summarize_trajectory_risk = compute_tail_risk


def plot_trajectories(
    trajectories_df: pd.DataFrame,
    *,
    tail_risk: pd.DataFrame | None = None,
    ts_id: str | None = None,
    max_spaghetti_paths: int = 40,
    ax: Any | None = None,
    title: str | None = None,
) -> Any:
    """Plot joint trajectory spaghetti fan chart and cumulative tail-risk markers.

    Args:
        trajectories_df: Output DataFrame from `sample_trajectories`.
        tail_risk: Optional output DataFrame from `compute_tail_risk`.
        ts_id: Optional series ID filter; defaults to the first series in ``trajectories_df``.
        max_spaghetti_paths: Maximum number of individual sample paths to overlay.
        ax: Optional existing matplotlib ``Axes``.
        title: Optional custom plot title.

    Returns:
        The matplotlib ``Axes`` containing the rendered trajectory fan chart.
    """
    import matplotlib.pyplot as plt

    if trajectories_df is None or trajectories_df.empty:
        raise DataError("trajectories_df is empty; nothing to plot")

    work = trajectories_df.copy()
    id_col = "ts_id" if "ts_id" in work.columns else work.columns[1]
    selected_id = str(ts_id) if ts_id is not None else str(work[id_col].iloc[0])
    sub = work[work[id_col].astype(str) == selected_id].copy()
    if sub.empty:
        sub = work.copy()

    dcol = (
        "forecast_date"
        if "forecast_date" in sub.columns
        else ("ds" if "ds" in sub.columns else "horizon_step")
    )
    vcol = "yhat_sample" if "yhat_sample" in sub.columns else "yhat"

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))

    piv = sub.pivot_table(index="path_id", columns=dcol, values=vcol, aggfunc="first")
    piv = piv.reindex(columns=sorted(piv.columns))
    x_vals = list(piv.columns)
    mat = piv.to_numpy(dtype=float)

    n_show = min(int(max_spaghetti_paths), mat.shape[0])
    for idx in range(n_show):
        ax.plot(x_vals, mat[idx], color="#94a3b8", alpha=0.18, linewidth=0.8)

    q10 = np.quantile(mat, 0.10, axis=0)
    q50 = np.quantile(mat, 0.50, axis=0)
    q90 = np.quantile(mat, 0.90, axis=0)
    q99 = np.quantile(mat, 0.99, axis=0)

    ax.fill_between(x_vals, q10, q90, color="#3b82f6", alpha=0.25, label="80% Copula Band")
    ax.plot(x_vals, q50, color="#1d4ed8", linewidth=2.0, label="Median Trajectory")
    ax.plot(
        x_vals,
        q99,
        color="#dc2626",
        linestyle="--",
        linewidth=1.4,
        label="99th Percentile Envelope",
    )

    cop_name = str(trajectories_df.attrs.get("copula_type_selected", "copula"))
    subtitle = f" ({cop_name})"
    if tail_risk is not None and not tail_risk.empty and id_col in tail_risk.columns:
        tr_sub = tail_risk[tail_risk[id_col].astype(str) == selected_id]
        if not tr_sub.empty and "copula_amplification_ratio" in tr_sub.columns:
            amp = float(tr_sub["copula_amplification_ratio"].iloc[0])
            subtitle = f" ({cop_name}, amplification={amp:.2f}x)"

    ax.set_title(title or f"Joint Copula Trajectories — {selected_id}{subtitle}")
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Forecast Value")
    ax.legend(loc="best")
    return ax


__all__ = [
    "COPULA_TYPES",
    "VALID_COPULA_TYPES",
    "CopulaRiskSummary",
    "CopulaType",
    "compute_tail_risk",
    "evaluate_trajectory_paths",
    "plot_trajectories",
    "sample_copula_trajectories",
    "sample_joint_trajectories",
    "sample_trajectories",
    "shrink_correlation_toward_similarity",
    "summarize_trajectory_risk",
    "tail_risk_df",
]
