"""Cold-Start / New Product Introduction (NPI) & Hierarchical Analog Prior Transfer — pure.

Series with fewer than ``cold_start_min_obs`` observations (including ``0``-observation pre-launch
SKUs) lack sufficient history to estimate seasonal state-space models, autoregressive lags, or
multi-fold out-of-fold conformal prediction intervals on their own.

This module implements hierarchical donor-cluster matching across ``static_covariates`` and
``hierarchy.levels`` with graceful fallback (e.g., ``(category, brand, region) -> (category, region)
-> (category,) -> __global__``), three launch trajectory archetypes:

1. ``"analog_transfer"`` — Empirical similarity-weighted donor launch curve + seasonal harmonic
   profile.
2. ``"bass_diffusion"`` — Bass (1969) innovation ($p$) and imitation ($q$) adoption diffusion model
   with market potential $m$.
3. ``"weibull_lifecycle"`` — Weibull surge-and-decay lifecycle kernel
   $f(t) = A (k / \\lambda) (t / \\lambda)^{k-1} \\exp(-(t / \\lambda)^k) + b_{\\text{steady}}$
   for fashion, electronics, media, and promotional launches.
4. ``"auto"`` — Leave-One-Donor-Out Cross-Validation (LODO-CV) within each static donor cluster
   to select the archetype that minimizes held-out launch WAPE.

Local observations ($n_i \\ge 0$) are blended with the donor cluster prior via Empirical-Bayes
precision weighting:

$$w_i = \\frac{n_i}{n_i + \\kappa}$$

and scale-normalized out-of-fold conformal residual quantiles are pooled across matched donors to
produce calibrated Day-1 prediction intervals even when $n_i = 0$.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

from .calibration import CALIBRATED_COLUMNS, rearrange_monotone_quantiles
from .models.base_model import DEFAULT_QUANTILES
from .seasonality import seasonal_period

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .config import ColdStartMatching, ColdStartMethod, RunConfig, SimilarityMethod

ColdStartArchetype = Literal[
    "analog_transfer",
    "bass_diffusion",
    "weibull_lifecycle",
    "arps_decline",
    "none",
]

_EPS = 1e-9
_DEFAULT_KAPPA = 5.0
_MAX_TRAJECTORY_LEN = 120


@dataclass(frozen=True)
class DonorProfile:
    """Extracted launch profile, seasonal indices, and error quantiles of one mature donor."""

    ts_id: str
    n_obs: int
    n_nonzero: int
    mean_level: float
    std_level: float
    cv: float
    zero_share: float
    slope: float
    seasonal_indices: np.ndarray
    normalized_trajectory: np.ndarray
    raw_launch_trajectory: np.ndarray
    static_features: dict[str, Any]
    residual_quantiles: dict[float, float]
    bass_params: dict[str, float] = field(default_factory=dict)
    weibull_params: dict[str, float] = field(default_factory=dict)
    arps_params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ColdStartDonorPool:
    """Catalog of mature donor profiles, hierarchical attribute groups, and LODO-CV winners."""

    freq: str
    seasonal_period: int
    min_obs: int
    donors: tuple[DonorProfile, ...]
    pooled_residual_quantiles: dict[float, float]
    attribute_Hierarchy: tuple[tuple[str, ...], ...] = ()
    group_archetypes: dict[str, str] = field(default_factory=dict)
    group_oof_lift_pct: dict[str, float] = field(default_factory=dict)
    group_kappa: dict[str, float] = field(default_factory=dict)


ColdStartDonorCatalog = ColdStartDonorPool


@dataclass(frozen=True)
class ColdStartForecastResult:
    """Result of cold-start NPI prior transfer and Empirical-Bayes blending for one series."""

    ts_id: str
    cold_start_flag: bool
    is_cold_start: bool
    n_obs: int
    n_nonzero: int
    donor_group: str
    matched_level: str
    selected_archetype: ColdStartArchetype
    method_used: ColdStartArchetype
    donor_ts_ids: tuple[str, ...]
    similarity_weights: tuple[float, ...]
    empirical_bayes_weight: float
    prior_scale: float
    scale_factor: float
    lifecycle_stage: str
    lifecycle_params: dict[str, float]
    pooled_q80_width: float
    oof_wape_lift_pct: float
    predictions: pd.DataFrame
    diagnostics: dict[str, Any]
    matching_mode_used: str = "attribute"
    donor_similarity_score: float = 0.0
    top_donor_ts_ids: tuple[str, ...] = ()


def _extract_seasonal_indices(y: np.ndarray, period: int) -> np.ndarray:
    """Compute multiplicative/additive normalized seasonal ratios of length ``period``."""
    m = max(1, int(period))
    if m <= 1 or y.size < m:
        return np.ones(m, dtype=float)
    mean_y = float(np.mean(np.maximum(y, 0.0)))
    if mean_y <= _EPS:
        return np.ones(m, dtype=float)
    indices = np.ones(m, dtype=float)
    for pos in range(m):
        vals = y[pos::m]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            indices[pos] = max(0.05, float(np.mean(np.maximum(vals, 0.0))) / mean_y)
    idx_mean = float(np.mean(indices))
    if idx_mean > _EPS:
        indices = indices / idx_mean
    return np.clip(indices, 0.1, 4.0)


def _bass_rate_curve(t: np.ndarray, p: float, q: float, m_pot: float) -> np.ndarray:
    """Evaluate the discrete-time Bass (1969) adoption rate $S(t)$ at $t \\ge 1$."""
    p_safe = max(float(p), 1e-5)
    q_safe = max(float(q), 0.0)
    m_safe = max(float(m_pot), 0.0)
    pq = p_safe + q_safe
    exp_term = np.exp(-np.clip(pq * t, -50.0, 50.0))
    denom = (1.0 + (q_safe / p_safe) * exp_term) ** 2
    rate = m_safe * ((pq**2) / p_safe) * exp_term / np.maximum(denom, _EPS)
    return np.maximum(0.0, rate)


def fit_bass_diffusion(
    y: np.ndarray,
    horizon: int,
    *,
    seasonal_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit Bass (1969) innovation ($p$) and imitation ($q$) diffusion parameters and forecast.

    Parameters
    ----------
    y:
        Observed launch history (1-D array of non-negative demand observations). May be empty or
        short; falls back to canonical NPI diffusion priors when $n < 3$.
    horizon:
        Number of future steps $H \\ge 1$ to forecast after ``len(y)``.
    seasonal_indices:
        Optional multiplicative seasonal profile of length $m$.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        ``(future_yhat, {"p": p, "q": q, "m": m_pot})``.
    """
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.maximum(0.0, arr)
    n = int(arr.size)
    h = max(1, int(horizon))

    if n == 0:
        p_hat, q_hat, m_hat = 0.03, 0.38, 100.0
    elif n < 3:
        mean_val = max(float(np.mean(arr)), 1.0)
        p_hat, q_hat = 0.04, 0.35
        # Choose m_hat so initial adoption rate matches mean_val: S(1) ~ m * p
        m_hat = max(mean_val / max(p_hat, 1e-3), float(np.sum(arr)) * 4.0)
    else:
        # Deseasonalize if seasonal_indices provided.
        if seasonal_indices is not None and seasonal_indices.size > 1:
            m_per = seasonal_indices.size
            s_factors = np.array([seasonal_indices[i % m_per] for i in range(n)], dtype=float)
            y_deseas = arr / np.maximum(s_factors, 0.1)
        else:
            y_deseas = arr

        t_obs = np.arange(1, n + 1, dtype=float)
        mean_y = max(float(np.mean(y_deseas)), 1e-3)
        cum_y = float(np.sum(y_deseas))
        m_init = max(cum_y * 2.5, mean_y * (n + h) * 1.5, 1.0)

        def _obj(theta: np.ndarray) -> float:
            p_v, q_v, log_m = float(theta[0]), float(theta[1]), float(theta[2])
            m_v = math.exp(np.clip(log_m, -10.0, 25.0))
            pred = _bass_rate_curve(t_obs, p_v, q_v, m_v)
            resid = (pred - y_deseas) / mean_y
            # Mild prior regularization keeping (p, q) in realistic NPI territory on tiny prefixes
            reg = 0.05 * ((p_v - 0.03) ** 2 + 0.1 * (q_v - 0.35) ** 2)
            return float(np.mean(resid**2) + reg)

        best_x = np.array([0.03, 0.35, math.log(m_init)], dtype=float)
        best_val = _obj(best_x)
        for p0, q0, m_mult in (
            (0.02, 0.30, 2.0),
            (0.05, 0.45, 3.0),
            (0.08, 0.20, 1.8),
        ):
            cand = np.array([p0, q0, math.log(max(cum_y * m_mult, mean_y * 10.0, 1.0))])
            val = _obj(cand)
            if val < best_val:
                best_val = val
                best_x = cand

        res = minimize(
            _obj,
            best_x,
            method="L-BFGS-B",
            bounds=[
                (0.002, 0.45),
                (0.01, 1.20),
                (math.log(max(cum_y * 0.5, 1e-2)), math.log(max(cum_y * 50.0, 1e6))),
            ],
            options={"maxiter": 80},
        )
        p_hat = float(res.x[0])
        q_hat = float(res.x[1])
        m_hat = float(math.exp(np.clip(float(res.x[2]), -10.0, 25.0)))

    t_fut = np.arange(n + 1, n + h + 1, dtype=float)
    fut_curve = _bass_rate_curve(t_fut, p_hat, q_hat, m_hat)
    if seasonal_indices is not None and seasonal_indices.size > 1:
        m_per = seasonal_indices.size
        fut_s = np.array([seasonal_indices[(n + i) % m_per] for i in range(h)], dtype=float)
        fut_curve = fut_curve * fut_s

    return np.maximum(0.0, fut_curve), {
        "p": round(float(p_hat), 6),
        "q": round(float(q_hat), 6),
        "m": round(float(m_hat), 4),
    }


def _weibull_rate_curve(
    t: np.ndarray, k: float, lam: float, amp: float, baseline: float
) -> np.ndarray:
    """Evaluate the Weibull product lifecycle curve at $t \\ge 1$."""
    k_safe = max(float(k), 0.15)
    lam_safe = max(float(lam), 0.5)
    amp_safe = max(float(amp), 0.0)
    base_safe = max(float(baseline), 0.0)
    scaled_t = np.maximum(t, 0.1) / lam_safe
    log_kernel = (
        math.log(k_safe / lam_safe)
        + (k_safe - 1.0) * np.log(scaled_t)
        - np.power(scaled_t, np.clip(k_safe, 0.15, 8.0))
    )
    kernel = np.exp(np.clip(log_kernel, -50.0, 30.0))
    return np.maximum(0.0, amp_safe * kernel + base_safe)


def fit_weibull_lifecycle(
    y: np.ndarray,
    horizon: int,
    *,
    seasonal_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a Weibull surge-and-decay product lifecycle curve and project forward.

    Models product launches with rapid early peak and post-launch decay:
    $$f(t) = A (k / \\lambda) (t / \\lambda)^{k-1} e^{-(t/\\lambda)^k} + b_{\\text{steady}}$$

    Parameters
    ----------
    y:
        Observed launch history (1-D array).
    horizon:
        Number of future steps $H \\ge 1$ to forecast.
    seasonal_indices:
        Optional multiplicative seasonal profile of length $m$.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        ``(future_yhat, {"k": k, "lambda": lam, "amplitude": amp, "baseline": baseline})``.
    """
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.maximum(0.0, arr)
    n = int(arr.size)
    h = max(1, int(horizon))

    if n == 0:
        k_hat, lam_hat, amp_hat, base_hat = 1.6, 10.0, 250.0, 10.0
    elif n < 3:
        mean_val = max(float(np.mean(arr)), 1.0)
        k_hat, lam_hat = 1.5, float(max(n + h // 2, 6))
        base_hat = 0.25 * mean_val
        amp_hat = mean_val * lam_hat * 2.0
    else:
        if seasonal_indices is not None and seasonal_indices.size > 1:
            m_per = seasonal_indices.size
            s_factors = np.array([seasonal_indices[i % m_per] for i in range(n)], dtype=float)
            y_deseas = arr / np.maximum(s_factors, 0.1)
        else:
            y_deseas = arr

        t_obs = np.arange(1, n + 1, dtype=float)
        mean_y = max(float(np.mean(y_deseas)), 1e-3)
        min_y = float(np.min(y_deseas))
        peak_idx = int(np.argmax(y_deseas)) + 1
        lam_init = float(max(peak_idx * 1.25, 3.0))
        amp_init = max(float(np.sum(y_deseas)) * 1.5, mean_y * lam_init * 2.0, 1.0)
        base_init = max(0.0, min_y * 0.5)

        def _obj(theta: np.ndarray) -> float:
            k_v = float(theta[0])
            lam_v = float(theta[1])
            amp_v = math.exp(np.clip(float(theta[2]), -10.0, 25.0))
            base_v = float(theta[3])
            pred = _weibull_rate_curve(t_obs, k_v, lam_v, amp_v, base_v)
            resid = (pred - y_deseas) / mean_y
            reg = 0.02 * ((k_v - 1.6) ** 2 + ((lam_v - lam_init) / max(lam_init, 5.0)) ** 2)
            return float(np.mean(resid**2) + reg)

        x0 = np.array([1.6, lam_init, math.log(amp_init), base_init], dtype=float)
        res = minimize(
            _obj,
            x0,
            method="L-BFGS-B",
            bounds=[
                (0.5, 5.0),
                (1.5, float(max(n + h * 2, 20))),
                (math.log(max(mean_y * 0.1, 1e-3)), math.log(max(amp_init * 30.0, 1e6))),
                (0.0, float(max(mean_y * 1.5, 1.0))),
            ],
            options={"maxiter": 80},
        )
        k_hat = float(res.x[0])
        lam_hat = float(res.x[1])
        amp_hat = float(math.exp(np.clip(float(res.x[2]), -10.0, 25.0)))
        base_hat = float(res.x[3])

    t_fut = np.arange(n + 1, n + h + 1, dtype=float)
    fut_curve = _weibull_rate_curve(t_fut, k_hat, lam_hat, amp_hat, base_hat)
    if seasonal_indices is not None and seasonal_indices.size > 1:
        m_per = seasonal_indices.size
        fut_s = np.array([seasonal_indices[(n + i) % m_per] for i in range(h)], dtype=float)
        fut_curve = fut_curve * fut_s

    return np.maximum(0.0, fut_curve), {
        "k": round(float(k_hat), 6),
        "lambda": round(float(lam_hat), 6),
        "amplitude": round(float(amp_hat), 4),
        "baseline": round(float(base_hat), 4),
    }


def _arps_unit_basis(t: np.ndarray, b: float, d_i: float) -> np.ndarray:
    """Evaluate the unit Arps (1945) decline curve $(1 + b D_i t)^{-1/b}$ (or $e^{-D_i t}$)."""
    b_safe = float(np.clip(b, 0.0, 1.0))
    d_safe = max(float(d_i), 0.0)
    t_safe = np.maximum(0.0, np.asarray(t, dtype=float))
    if d_safe <= 1e-12:
        return np.ones_like(t_safe)
    if b_safe < 1e-5:
        return np.exp(-np.clip(d_safe * t_safe, 0.0, 60.0))
    base = np.maximum(1.0 + b_safe * d_safe * t_safe, 1e-12)
    return np.exp(-np.clip(np.log(base) / b_safe, 0.0, 60.0))


def _arps_rate_curve(t: np.ndarray, q_i: float, b: float, d_i: float) -> np.ndarray:
    """Evaluate the Arps (1945) decline rate $q(t)$ at $t \\ge 0$."""
    q_safe = max(float(q_i), 0.0)
    return np.maximum(0.0, q_safe * _arps_unit_basis(t, b, d_i))


def _arps_cumulative_at(t_end: float, q_i: float, b: float, d_i: float) -> float:
    """Analytical cumulative production $N_p(T) = \\int_0^T q(t)\\,dt$ under Arps decline."""
    t_val = max(0.0, float(t_end))
    q_val = max(0.0, float(q_i))
    b_val = float(np.clip(b, 0.0, 1.0))
    d_val = max(0.0, float(d_i))
    if t_val <= 0.0 or q_val <= 0.0:
        return 0.0
    if d_val <= 1e-12:
        return float(q_val * t_val)
    if b_val < 1e-4:
        return float((q_val / d_val) * (1.0 - math.exp(-min(d_val * t_val, 60.0))))
    if abs(b_val - 1.0) < 1e-4:
        return float((q_val / d_val) * math.log1p(d_val * t_val))
    term = (1.0 + b_val * d_val * t_val) ** (1.0 - (1.0 / b_val))
    return float((q_val / ((1.0 - b_val) * d_val)) * (1.0 - term))


def _arps_cumulative_eur(
    q_i: float,
    b: float,
    d_i: float,
    *,
    n_obs: int = 0,
    q_econ: float = 1.0,
    max_eur_steps: float = 3650.0,
) -> tuple[float, float]:
    """Compute ``(eur_total, eur_remaining)`` from $t=0$ to $q(T) = q_{\\text{econ}}$.

    Integrates $q(t) = q_i (1 + b D_i t)^{-1/b}$ up to $T_{\\text{lim}} = \\min(T_{\\text{econ}},
    T_{\\max})$, ensuring $T_{\\text{lim}} \\ge n_{\\text{obs}}$ so ``eur_total >= cumulative``.
    """
    q_val = max(0.0, float(q_i))
    b_val = float(np.clip(b, 0.0, 1.0))
    d_val = max(0.0, float(d_i))
    t_max = max(float(max_eur_steps), float(max(0, n_obs)), 1.0)
    q_lim = max(0.0, float(q_econ))

    if q_val <= 0.0:
        return 0.0, 0.0
    if d_val <= 1e-9 or q_lim <= 0.0:
        t_lim = t_max
    elif q_val <= q_lim:
        t_lim = float(max(0, n_obs))
    elif b_val < 1e-4:
        t_econ = math.log(q_val / q_lim) / d_val
        t_lim = min(t_econ, t_max)
    else:
        t_econ = (((q_val / q_lim) ** b_val) - 1.0) / (b_val * d_val)
        t_lim = min(t_econ, t_max)

    t_obs = float(max(0, int(n_obs)))
    t_total = max(t_lim, t_obs)
    eur_total = _arps_cumulative_at(t_total, q_val, b_val, d_val)
    cum_obs = _arps_cumulative_at(min(t_obs, t_total), q_val, b_val, d_val)
    eur_rem = max(0.0, eur_total - cum_obs)
    return float(eur_total), float(eur_rem)


def fit_arps_decline(
    y: np.ndarray,
    horizon: int,
    *,
    seasonal_indices: np.ndarray | None = None,
    q_econ: float = 1.0,
    max_eur_steps: float = 3650.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit Arps (1945) hyperbolic/harmonic/exponential decline curve and compute analytical EUR.

    Models physics-constrained depleting assets (oil/gas wells, geothermal reservoirs, battery
    capacity degradation, legacy SaaS cohort decay):

    $$q(t) = \\frac{q_i}{(1 + b D_i t)^{1/b}}, \\qquad 0 \\le b \\le 1, \\quad D_i > 0$$

    with $b = 0$ reducing to exponential decline $q_i e^{-D_i t}$ and $b = 1$ to harmonic decline
    $q_i / (1 + D_i t)$ for $t = 0, 1, \\dots, n-1$.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        ``(future_yhat, {"q_i": q_i, "b": b, "d_i": d_i, "eur_total": eur_total,
        "eur_remaining": eur_rem})``.
    """
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.maximum(0.0, arr)
    n = int(arr.size)
    h = max(1, int(horizon))

    if n == 0:
        qi_hat, b_hat, di_hat = 100.0, 0.5, 0.05
    elif n == 1:
        qi_hat = max(float(arr[0]), 1.0)
        b_hat, di_hat = 0.5, 0.05
    elif n == 2:
        qi_hat = max(float(arr[0]), 1e-3)
        b_hat = 0.5
        if arr[0] > arr[1] > 0.0:
            di_hat = float(np.clip(((arr[0] / arr[1]) ** b_hat - 1.0) / b_hat, 1e-4, 2.0))
        else:
            di_hat = 0.05
    else:
        if seasonal_indices is not None and seasonal_indices.size > 1:
            m_per = seasonal_indices.size
            s_factors = np.array([seasonal_indices[i % m_per] for i in range(n)], dtype=float)
            y_deseas = arr / np.maximum(s_factors, 0.1)
        else:
            y_deseas = arr

        t_obs = np.arange(0, n, dtype=float)
        scale_y = max(float(np.max(y_deseas)), 1e-6)

        def _solve_qi_and_resid(b_v: float, d_v: float) -> tuple[float, np.ndarray]:
            g = _arps_unit_basis(t_obs, b_v, d_v)
            denom = float(np.dot(g, g))
            q_opt = max(float(np.dot(y_deseas, g) / max(denom, 1e-12)), 1e-6)
            return q_opt, (q_opt * g - y_deseas) / scale_y

        # 1. Coarse 2D grid search over (b, d_i) with variable projection for q_i
        b_grid = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)
        d_grid = np.logspace(-3.5, 0.3, 18)
        best_b, best_d = 0.5, 0.05
        _, r0 = _solve_qi_and_resid(best_b, best_d)
        best_loss = float(np.mean(r0**2))
        for b_cand in b_grid:
            for d_cand in d_grid:
                _, r_vec = _solve_qi_and_resid(b_cand, float(d_cand))
                loss = float(np.mean(r_vec**2))
                if loss < best_loss:
                    best_loss = loss
                    best_b = b_cand
                    best_d = float(d_cand)

        # 2. Polish (b, d_i) jointly via Trust-Region Reflective nonlinear least squares
        res_2d = least_squares(
            lambda theta: _solve_qi_and_resid(float(theta[0]), float(theta[1]))[1],
            x0=np.array([best_b, best_d], dtype=float),
            bounds=([0.0, 1e-6], [1.0, 5.0]),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=200,
        )
        b_hat = float(np.clip(res_2d.x[0], 0.0, 1.0))
        di_hat = float(max(res_2d.x[1], 1e-6))
        qi_hat, r_2d = _solve_qi_and_resid(b_hat, di_hat)
        best_mse = float(np.mean(r_2d**2))

        # 3. Also test exact boundary regimes b=0.0 (exponential) and b=1.0 (harmonic)
        for b_bound in (0.0, 1.0):
            res_1d = least_squares(
                lambda d_arr, _b=b_bound: _solve_qi_and_resid(_b, float(d_arr[0]))[1],
                x0=np.array([di_hat], dtype=float),
                bounds=([1e-6], [5.0]),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=100,
            )
            d_b = float(max(res_1d.x[0], 1e-6))
            q_b, r_b = _solve_qi_and_resid(b_bound, d_b)
            mse_b = float(np.mean(r_b**2))
            if mse_b <= best_mse + 1e-11:
                best_mse = mse_b
                b_hat = b_bound
                di_hat = d_b
                qi_hat = q_b

    t_fut = np.arange(n, n + h, dtype=float)
    fut_curve = _arps_rate_curve(t_fut, qi_hat, b_hat, di_hat)
    if seasonal_indices is not None and seasonal_indices.size > 1:
        m_per = seasonal_indices.size
        fut_s = np.array([seasonal_indices[(n + i) % m_per] for i in range(h)], dtype=float)
        fut_curve = fut_curve * fut_s

    eur_tot, eur_rem = _arps_cumulative_eur(
        qi_hat,
        b_hat,
        di_hat,
        n_obs=n,
        q_econ=q_econ,
        max_eur_steps=max_eur_steps,
    )
    return np.maximum(0.0, fut_curve), {
        "q_i": round(float(qi_hat), 4),
        "b": round(float(b_hat), 6),
        "d_i": round(float(di_hat), 6),
        "eur_total": round(float(eur_tot), 4),
        "eur_remaining": round(float(eur_rem), 4),
    }


def apply_supersession_map(
    panel_df: pd.DataFrame,
    supersession_map: Mapping[str, str] | None,
    *,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    overlap_mode: Literal["auto", "sum", "successor_only"] = "auto",
    drop_predecessors: bool = False,
) -> pd.DataFrame:
    """Splice predecessor SKU histories onto successor SKUs across multi-generation chains.

    Supports both ``{successor_id: predecessor_id}`` and ``{predecessor_id: successor_id}`` map
    orientations by automatically orienting each linked pair from the earlier-starting series
    (predecessor) to the later-starting series (successor), resolving multi-generation chains
    (``v1 -> v2 -> v3``) in topological order, and reconciling phase-out/phase-in overlap windows.
    """
    if (
        panel_df is None
        or panel_df.empty
        or not supersession_map
        or ts_id_col not in panel_df.columns
    ):
        return panel_df.copy() if panel_df is not None else pd.DataFrame()

    work = panel_df.copy()
    has_dates = date_col in work.columns
    has_target = target_col in work.columns

    # Summarize start date and observation count per ts_id to orient edges deterministically.
    min_ds_by_id: dict[str, Any] = {}
    n_obs_by_id: dict[str, int] = {}
    frames_by_id: dict[str, pd.DataFrame] = {}
    for raw_id, grp in work.groupby(ts_id_col, sort=False):
        tid = str(raw_id)
        sorted_grp = grp.sort_values(date_col).reset_index(drop=True) if has_dates else grp.copy()
        frames_by_id[tid] = sorted_grp
        if has_dates and not sorted_grp.empty:
            non_na_ds = sorted_grp[date_col].dropna()
            if not non_na_ds.empty:
                min_ds_by_id[tid] = non_na_ds.iloc[0]
        if has_target and not sorted_grp.empty:
            n_obs_by_id[tid] = int(
                pd.to_numeric(sorted_grp[target_col], errors="coerce").notna().sum()
            )
        else:
            n_obs_by_id[tid] = len(sorted_grp)

    # Orient each (k, v) pair into pred -> succ
    pred_of: dict[str, str] = {}
    for raw_k, raw_v in supersession_map.items():
        k, v = str(raw_k).strip(), str(raw_v).strip()
        if not k or not v or k == v:
            continue
        ds_k, ds_v = min_ds_by_id.get(k), min_ds_by_id.get(v)
        n_k, n_v = n_obs_by_id.get(k, 0), n_obs_by_id.get(v, 0)
        if ds_k is not None and ds_v is not None and ds_k != ds_v:
            if ds_k < ds_v:
                pred, succ = k, v
            else:
                pred, succ = v, k
        elif n_k != n_v:
            if n_k > n_v:
                pred, succ = k, v
            else:
                pred, succ = v, k
        else:
            # Default orientation: {successor_id: predecessor_id}
            pred, succ = v, k
        pred_of[succ] = pred

    if not pred_of:
        return work

    # Resolve topological order across multi-generation chains (v1 -> v2 -> v3)
    chains_by_succ: dict[str, list[str]] = {}
    for succ in sorted(pred_of.keys()):
        chain: list[str] = []
        curr = succ
        visited: set[str] = {succ}
        while curr in pred_of:
            p = pred_of[curr]
            if p in visited:
                break
            visited.add(p)
            chain.append(p)
            curr = p
        chain.reverse()  # oldest ancestor first: [v1, v2] for succ=v3
        chains_by_succ[succ] = chain

    # Sort successors by chain depth so v1 -> v2 splices before v2 -> v3
    ordered_succs = sorted(chains_by_succ.keys(), key=lambda s: (len(chains_by_succ[s]), s))
    all_predecessors: set[str] = set(pred_of.values())

    meta_cols = [c for c in work.columns if c not in {ts_id_col, date_col, target_col}]

    for succ in ordered_succs:
        pred = pred_of[succ]
        df_pred = frames_by_id.get(pred)
        df_succ = frames_by_id.get(succ)
        if df_pred is None or df_pred.empty:
            continue
        if df_succ is None or df_succ.empty:
            spliced = df_pred.copy()
            spliced[ts_id_col] = succ
            frames_by_id[succ] = spliced
            continue

        if not has_dates:
            spliced = pd.concat([df_pred.assign(**{ts_id_col: succ}), df_succ], ignore_index=True)
            frames_by_id[succ] = spliced
            continue

        succ_min_ds = df_succ[date_col].min()
        df_before = df_pred[df_pred[date_col] < succ_min_ds].copy()
        df_overlap_pred = df_pred[df_pred[date_col] >= succ_min_ds].copy()

        # Stamp successor ID and constant static metadata onto prepended predecessor rows
        df_before[ts_id_col] = succ
        for mcol in meta_cols:
            if mcol in df_succ.columns:
                non_na = df_succ[mcol].dropna()
                if not non_na.empty and non_na.nunique() == 1:
                    df_before[mcol] = non_na.iloc[0]

        df_succ_adj = df_succ.copy()
        if not df_overlap_pred.empty and has_target:
            ov_map = df_overlap_pred.groupby(date_col, sort=False)[target_col].sum().to_dict()
            common_mask = df_succ_adj[date_col].isin(ov_map.keys())
            if common_mask.any():
                y_succ_ov = pd.to_numeric(
                    df_succ_adj.loc[common_mask, target_col], errors="coerce"
                ).fillna(0.0)
                y_pred_ov = (
                    df_succ_adj.loc[common_mask, date_col].map(ov_map).astype(float).fillna(0.0)
                )
                should_sum = overlap_mode == "sum"
                if overlap_mode == "auto":
                    if not df_before.empty:
                        mu_pre = float(pd.to_numeric(df_before[target_col], errors="coerce").mean())
                        mu_succ = float(y_succ_ov.mean())
                        mu_sum = float((y_succ_ov + y_pred_ov).mean())
                        should_sum = abs(mu_sum - mu_pre) <= abs(mu_succ - mu_pre)
                    else:
                        should_sum = True
                if should_sum:
                    df_succ_adj.loc[common_mask, target_col] = (y_succ_ov + y_pred_ov).to_numpy(
                        dtype=float
                    )

        spliced = (
            pd.concat([df_before, df_succ_adj], ignore_index=True)
            .sort_values(date_col)
            .reset_index(drop=True)
        )
        frames_by_id[succ] = spliced

    kept_ids = [
        tid
        for tid in frames_by_id
        if not (drop_predecessors and tid in all_predecessors and tid not in chains_by_succ)
    ]
    if drop_predecessors:
        kept_ids = [tid for tid in frames_by_id if tid not in all_predecessors]

    out_frames = [frames_by_id[tid] for tid in kept_ids if not frames_by_id[tid].empty]
    out_df = pd.concat(out_frames, ignore_index=True) if out_frames else work.iloc[0:0].copy()
    out_df.attrs["supersession_chains"] = {
        succ: {
            "predecessors": chains_by_succ[succ],
            "spliced_n_obs": len(frames_by_id.get(succ, ())),
        }
        for succ in ordered_succs
    }
    return out_df


def _build_fallback_levels(
    static_covariates: Sequence[str],
    hierarchy_cols: Sequence[str],
    available_cols: set[str],
) -> tuple[tuple[str, ...], ...]:
    """Construct ordered attribute tuples from finest to coarsest for hierarchical donor fallback.

    For example, given ``("category", "brand", "region")``, produces:
    ``(("category", "brand", "region"), ("category", "region"),
    ("category", "brand"), ("category",))``
    followed by ``()`` (representing ``"__global__"``).
    """
    ordered_cols: list[str] = []
    for col in (*static_covariates, *hierarchy_cols):
        c_str = str(col)
        if c_str in available_cols and c_str not in ordered_cols:
            ordered_cols.append(c_str)

    if not ordered_cols:
        return ((),)

    levels: list[tuple[str, ...]] = []
    full = tuple(ordered_cols)
    levels.append(full)
    if len(ordered_cols) >= 3:
        # e.g. (category, brand, region) -> (category, region) -> (category, brand)
        first_last = (ordered_cols[0], ordered_cols[-1])
        if first_last not in levels:
            levels.append(first_last)
        first_two = (ordered_cols[0], ordered_cols[1])
        if first_two not in levels:
            levels.append(first_two)
    if len(ordered_cols) >= 2:
        first_only = (ordered_cols[0],)
        if first_only not in levels:
            levels.append(first_only)
        for c in ordered_cols[1:]:
            single = (c,)
            if single not in levels:
                levels.append(single)
    levels.append(())
    return tuple(levels)


def _format_donor_group(attrs: Mapping[str, Any], level_cols: Sequence[str]) -> tuple[str, str]:
    """Return ``(donor_group_key, matched_level_name)`` for ``level_cols``."""
    if not level_cols:
        return "__global__", "__global__"
    parts = [f"{col}={attrs.get(col, 'NA')}" for col in level_cols]
    return "|".join(parts), "+".join(level_cols)


def _match_donors_hierarchical(
    target_attrs: Mapping[str, Any],
    donors: Sequence[DonorProfile],
    fallback_levels: Sequence[tuple[str, ...]],
    *,
    exclude_ts_id: str | None = None,
) -> tuple[list[DonorProfile], str, str]:
    """Find mature donors matching ``target_attrs`` at the finest non-empty hierarchy level."""
    candidates = [d for d in donors if d.ts_id != exclude_ts_id]
    if not candidates:
        return [], "__global__", "__global__"

    for level_cols in fallback_levels:
        if not level_cols:
            return candidates, "__global__", "__global__"
        if not all(col in target_attrs for col in level_cols):
            continue
        matched = [
            d
            for d in candidates
            if all(
                str(d.static_features.get(col)) == str(target_attrs.get(col)) for col in level_cols
            )
        ]
        if matched:
            grp_key, lvl_name = _format_donor_group(target_attrs, level_cols)
            return matched, grp_key, lvl_name

    return candidates, "__global__", "__global__"


def _compute_donor_launch_similarity(
    y_obs: np.ndarray,
    donors: Sequence[DonorProfile],
    *,
    similarity_method: str = "hybrid",
    target_attrs: Mapping[str, Any] | None = None,
    text_cols: Sequence[str] | str | None = None,
    attribute_cols: Sequence[str] | str | None = None,
    hierarchy_cols: Sequence[str] | str | None = None,
    embedding_col: str | None = None,
    min_obs: int = 14,
) -> dict[str, float]:
    """Score mature ``donors`` against the cold-start series' multi-modal metadata and/or observed
    launch prefix ``y_obs`` with continuous Day-0 ($n_i = 0$) to Day-14 handoff."""
    n_obs = int(y_obs.size)
    if not donors:
        return {}

    text_list = (
        [text_cols] if isinstance(text_cols, str) else [str(c) for c in (text_cols or ()) if str(c)]
    )
    attr_list = (
        [attribute_cols]
        if isinstance(attribute_cols, str)
        else [str(c) for c in (attribute_cols or ()) if str(c)]
    )
    hier_list = (
        [hierarchy_cols]
        if isinstance(hierarchy_cols, str)
        else [str(c) for c in (hierarchy_cols or ()) if str(c)]
    )
    has_meta = bool(target_attrs) and bool(text_list or attr_list or hier_list or embedding_col)

    if n_obs < 2 and not has_meta:
        return {d.ts_id: 0.5 for d in donors}

    from .similarity import compute_similarity_matrix, find_similar_series_df

    meta_scores: dict[str, float] = {d.ts_id: 0.5 for d in donors}
    if has_meta and target_attrs is not None:
        meta_rows: list[dict[str, Any]] = [
            {"ts_id": "__cs_query__", "ds": 0, "y": 1.0, **dict(target_attrs)}
        ]
        for d in donors:
            meta_rows.append({"ts_id": d.ts_id, "ds": 0, "y": 1.0, **dict(d.static_features)})
        sem_active = bool(text_list or attr_list or embedding_col)
        hier_active = bool(hier_list)
        w_s = 0.6 if (sem_active and hier_active) else (1.0 if sem_active else 0.0)
        w_h = 0.4 if (sem_active and hier_active) else (1.0 if hier_active else 0.0)
        sim_res = compute_similarity_matrix(
            pd.DataFrame(meta_rows),
            method=similarity_method,
            text_cols=text_list or None,
            embedding_col=embedding_col,
            attribute_cols=attr_list or None,
            hierarchy_cols=hier_list or None,
            semantic_weight=w_s,
            hierarchy_weight=w_h,
        )
        if "__cs_query__" in sim_res.matrix_df.index:
            q_row = sim_res.matrix_df.loc["__cs_query__"].to_dict()
            for tid_d, raw_sim in q_row.items():
                if str(tid_d) != "__cs_query__":
                    meta_scores[str(tid_d)] = float(np.clip(float(raw_sim), 0.0, 1.0))

    if n_obs == 0:
        return meta_scores

    traj_scores: dict[str, float] = {d.ts_id: 0.5 for d in donors}
    if n_obs >= 2:
        pref_len = max(2, n_obs)
        panel_rows: list[dict[str, Any]] = []
        for t_idx, val in enumerate(y_obs[:pref_len]):
            panel_rows.append({"ts_id": "__cs_query__", "ds": t_idx, "y": float(val)})
        for d in donors:
            d_slice = d.raw_launch_trajectory[
                : max(pref_len, min(d.raw_launch_trajectory.size, 14))
            ]
            for t_idx, val in enumerate(d_slice):
                panel_rows.append({"ts_id": d.ts_id, "ds": t_idx, "y": float(val)})

        sim_df = find_similar_series_df(
            pd.DataFrame(panel_rows),
            query_ts_id="__cs_query__",
            top_k=len(donors),
            mode="launch_prefix",
            method=similarity_method,
            prefix_len=n_obs,
        )
        for row in sim_df.to_dict(orient="records"):
            nbr = str(row["neighbor_ts_id"])
            traj_scores[nbr] = float(np.clip(float(row["similarity_score"]), 0.0, 1.0))
    else:
        # n_obs == 1: initial level proximity against donor Day-1 observation
        y0 = max(0.0, float(y_obs[0]))
        for d in donors:
            d0 = (
                max(0.0, float(d.raw_launch_trajectory[0]))
                if d.raw_launch_trajectory.size > 0
                else max(0.0, float(d.mean_level))
            )
            traj_scores[d.ts_id] = float(
                np.clip(math.exp(-abs(math.log((y0 + 1.0) / (d0 + 1.0)))), 0.0, 1.0)
            )

    if not has_meta:
        return traj_scores

    alpha_traj = float(np.clip(float(n_obs) / float(max(int(min_obs), 1)), 0.0, 1.0))
    return {
        d.ts_id: float(
            np.clip(
                (1.0 - alpha_traj) * meta_scores.get(d.ts_id, 0.5)
                + alpha_traj * traj_scores.get(d.ts_id, 0.5),
                0.0,
                1.0,
            )
        )
        for d in donors
    }


def _project_archetype_from_donors(
    method: Literal["analog_transfer", "bass_diffusion", "weibull_lifecycle", "arps_decline"],
    y_obs: np.ndarray,
    horizon: int,
    matched_donors: Sequence[DonorProfile],
    *,
    seasonal_indices: np.ndarray,
    top_k_donors: int = 3,
    kappa: float = _DEFAULT_KAPPA,
    prior_scale_override: float | None = None,
    donor_similarity_map: dict[str, float] | None = None,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[float, ...],
    float,
    float,
    float,
    dict[str, float],
]:
    """Project ``horizon`` steps for one cold-start series using ``method`` and ``matched_donors``.

    Returns
    -------
    tuple
        ``(yhat, donor_ids, weights, eb_weight, prior_scale, blended_scale, lifecycle_params)``
    """
    n_obs = int(y_obs.size)
    h = max(1, int(horizon))
    m_per = max(1, int(seasonal_indices.size))

    if not matched_donors:
        # Pure single-series fallback when panel has no mature donors.
        prior_scale = (
            float(prior_scale_override)
            if prior_scale_override is not None
            else (float(np.mean(y_obs)) if n_obs > 0 else 10.0)
        )
        eb_weight = float(n_obs / (n_obs + max(kappa, 1e-3)))
        local_scale = float(np.mean(y_obs)) if n_obs > 0 else prior_scale
        blended_scale = eb_weight * local_scale + (1.0 - eb_weight) * prior_scale
        if method == "bass_diffusion":
            yhat, params = fit_bass_diffusion(y_obs, h, seasonal_indices=seasonal_indices)
            if n_obs == 0:
                yhat = yhat * (blended_scale / max(float(np.mean(yhat)), _EPS))
            return yhat, (), (), eb_weight, prior_scale, blended_scale, params
        if method == "weibull_lifecycle":
            yhat, params = fit_weibull_lifecycle(y_obs, h, seasonal_indices=seasonal_indices)
            if n_obs == 0:
                yhat = yhat * (blended_scale / max(float(np.mean(yhat)), _EPS))
            return yhat, (), (), eb_weight, prior_scale, blended_scale, params
        if method == "arps_decline":
            yhat, params = fit_arps_decline(y_obs, h, seasonal_indices=seasonal_indices)
            if n_obs == 0:
                scale_mult = blended_scale / max(float(np.mean(yhat)), _EPS)
                yhat = yhat * scale_mult
                qi_scaled = params["q_i"] * scale_mult
                eur_t, eur_r = _arps_cumulative_eur(qi_scaled, params["b"], params["d_i"], n_obs=0)
                params = {
                    **params,
                    "q_i": round(float(qi_scaled), 4),
                    "eur_total": round(float(eur_t), 4),
                    "eur_remaining": round(float(eur_r), 4),
                }
            return yhat, (), (), eb_weight, prior_scale, blended_scale, params
        base_curve = np.array(
            [blended_scale * seasonal_indices[(n_obs + step) % m_per] for step in range(h)],
            dtype=float,
        )
        return np.maximum(0.0, base_curve), (), (), eb_weight, prior_scale, blended_scale, {}

    # Compute similarity weights across matched donors over observed prefix 0..n_obs.
    raw_scores: list[float] = []
    donor_prefix_scales: list[float] = []
    for d in matched_donors:
        traj = d.raw_launch_trajectory
        if donor_similarity_map is not None and d.ts_id in donor_similarity_map:
            sim_01 = float(np.clip(donor_similarity_map[d.ts_id], 0.0, 1.0))
            score = math.exp(3.0 * sim_01)
        elif n_obs >= 2 and traj.size >= n_obs:
            d_pref = traj[:n_obs]
            y_norm = y_obs - float(np.mean(y_obs))
            d_norm = d_pref - float(np.mean(d_pref))
            denom = float(np.linalg.norm(y_norm) * np.linalg.norm(d_norm))
            cos_sim = float(np.dot(y_norm, d_norm) / denom) if denom > _EPS else 0.0
            # Map [-1, 1] similarity into positive softmax-like weight
            score = math.exp(1.5 * cos_sim)
        else:
            score = 1.0
        raw_scores.append(score)
        if n_obs >= 1 and traj.size >= n_obs:
            # Expected ratio of full-launch mean to prefix mean on donor
            pref_mean = float(np.mean(traj[:n_obs]))
            donor_prefix_scales.append(pref_mean if pref_mean > _EPS else max(d.mean_level, _EPS))
        else:
            donor_prefix_scales.append(max(d.mean_level, _EPS))

    order = np.argsort(raw_scores)[::-1][: max(1, min(int(top_k_donors), len(matched_donors)))]
    top_donors = [matched_donors[int(i)] for i in order]
    top_scores = np.array([raw_scores[int(i)] for i in order], dtype=float)
    top_weights = top_scores / max(float(np.sum(top_scores)), _EPS)
    top_pref_scales = np.array([donor_prefix_scales[int(i)] for i in order], dtype=float)

    prior_scale = (
        float(prior_scale_override)
        if prior_scale_override is not None
        else float(np.sum(top_weights * np.array([d.mean_level for d in top_donors], dtype=float)))
    )
    prior_scale = max(prior_scale, _EPS)

    # Empirical-Bayes precision weight w_i = n_i / (n_i + kappa)
    eb_weight = float(n_obs / (n_obs + max(float(kappa), _EPS)))
    if n_obs > 0:
        obs_mean = float(np.mean(np.maximum(0.0, y_obs)))
        # Infer implied series mean accounting for where prefix sits on the donor launch curve
        expected_pref_ratio = float(
            np.sum(
                top_weights
                * (
                    top_pref_scales
                    / np.maximum(np.array([d.mean_level for d in top_donors], dtype=float), _EPS)
                )
            )
        )
        expected_pref_ratio = float(np.clip(expected_pref_ratio, 0.2, 5.0))
        local_scale_est = obs_mean / expected_pref_ratio
    else:
        local_scale_est = prior_scale

    blended_scale = float(eb_weight * local_scale_est + (1.0 - eb_weight) * prior_scale)
    blended_scale = max(0.0, blended_scale)

    # Blend seasonal profile from top donors
    blended_seasonal = np.zeros(m_per, dtype=float)
    for w, d in zip(top_weights, top_donors, strict=False):
        if d.seasonal_indices.size == m_per:
            blended_seasonal += float(w) * d.seasonal_indices
        else:
            blended_seasonal += float(w) * seasonal_indices
    s_mean = float(np.mean(blended_seasonal))
    blended_seasonal = blended_seasonal / s_mean if s_mean > _EPS else np.ones(m_per, dtype=float)

    donor_ids = tuple(d.ts_id for d in top_donors)
    weights_tuple = tuple(round(float(w), 6) for w in top_weights)

    if method == "analog_transfer":
        # Normalized trajectory shape from top donors at steps n_obs .. n_obs + h - 1
        shape_vec = np.zeros(h, dtype=float)
        for w, d in zip(top_weights, top_donors, strict=False):
            ntraj = d.normalized_trajectory
            for step_idx in range(h):
                pos = n_obs + step_idx
                if pos < ntraj.size:
                    val = float(ntraj[pos])
                else:
                    # Carry forward terminal normalized level modulated by donor seasonality
                    tail_mean = float(np.mean(ntraj[-min(m_per, ntraj.size) :]))
                    val = tail_mean * float(
                        d.seasonal_indices[pos % d.seasonal_indices.size]
                        if d.seasonal_indices.size > 0
                        else 1.0
                    )
                shape_vec[step_idx] += float(w) * val

        donor_yhat = np.maximum(0.0, blended_scale * shape_vec)
        if n_obs >= 2:
            # Also compute local linear/level continuation modulated by donor seasonality
            last_deseas = float(y_obs[-1]) / max(float(blended_seasonal[(n_obs - 1) % m_per]), 0.1)
            local_anchor = 0.5 * last_deseas + 0.5 * local_scale_est
            local_yhat = np.array(
                [
                    max(0.0, local_anchor * float(blended_seasonal[(n_obs + s) % m_per]))
                    for s in range(h)
                ],
                dtype=float,
            )
            yhat = eb_weight * local_yhat + (1.0 - eb_weight) * donor_yhat
        else:
            yhat = donor_yhat
        return (
            np.maximum(0.0, yhat),
            donor_ids,
            weights_tuple,
            eb_weight,
            prior_scale,
            blended_scale,
            {},
        )

    if method == "bass_diffusion":
        # Weighted donor (p, q) prior + optional local refinement
        p_prior = float(
            np.sum(top_weights * np.array([d.bass_params.get("p", 0.03) for d in top_donors]))
        )
        q_prior = float(
            np.sum(top_weights * np.array([d.bass_params.get("q", 0.35) for d in top_donors]))
        )
        m_ratio_prior = float(
            np.sum(
                top_weights
                * np.array(
                    [
                        d.bass_params.get("m", d.mean_level * 30.0) / max(d.mean_level, _EPS)
                        for d in top_donors
                    ]
                )
            )
        )
        if n_obs >= 4:
            _, local_b = fit_bass_diffusion(y_obs, h, seasonal_indices=blended_seasonal)
            p_blend = eb_weight * local_b["p"] + (1.0 - eb_weight) * p_prior
            q_blend = eb_weight * local_b["q"] + (1.0 - eb_weight) * q_prior
        else:
            p_blend, q_blend = p_prior, q_prior

        # Calibrate m_pot so the Bass trajectory matches blended_scale and observed prefix
        t_all = np.arange(1, n_obs + h + 1, dtype=float)
        unit_curve = _bass_rate_curve(t_all, p_blend, q_blend, 1.0)
        if n_obs > 0:
            obs_s = np.array([blended_seasonal[i % m_per] for i in range(n_obs)], dtype=float)
            unit_pref_mean = float(np.mean(unit_curve[:n_obs] * obs_s))
            m_from_obs = (
                float(np.mean(y_obs)) / max(unit_pref_mean, _EPS)
                if unit_pref_mean > _EPS
                else blended_scale * m_ratio_prior
            )
            m_prior = blended_scale * m_ratio_prior
            m_blend = eb_weight * m_from_obs + (1.0 - eb_weight) * m_prior
        else:
            m_blend = blended_scale * m_ratio_prior

        fut_t = np.arange(n_obs + 1, n_obs + h + 1, dtype=float)
        fut_s = np.array([blended_seasonal[(n_obs + s) % m_per] for s in range(h)], dtype=float)
        yhat = _bass_rate_curve(fut_t, p_blend, q_blend, m_blend) * fut_s
        return (
            np.maximum(0.0, yhat),
            donor_ids,
            weights_tuple,
            eb_weight,
            prior_scale,
            blended_scale,
            {
                "p": round(float(p_blend), 6),
                "q": round(float(q_blend), 6),
                "m": round(float(m_blend), 4),
            },
        )

    if method == "arps_decline":
        b_prior = float(
            np.sum(top_weights * np.array([d.arps_params.get("b", 0.5) for d in top_donors]))
        )
        di_prior = float(
            np.sum(top_weights * np.array([d.arps_params.get("d_i", 0.05) for d in top_donors]))
        )
        qi_ratio_prior = float(
            np.sum(
                top_weights
                * np.array(
                    [
                        d.arps_params.get("q_i", d.mean_level * 1.5) / max(d.mean_level, _EPS)
                        for d in top_donors
                    ]
                )
            )
        )
        local_a: dict[str, float] = {}
        if n_obs >= 3:
            _, local_a = fit_arps_decline(y_obs, h, seasonal_indices=blended_seasonal)
            w_arps = 1.0 if n_obs >= 14 else eb_weight
            b_blend = float(np.clip(w_arps * local_a["b"] + (1.0 - w_arps) * b_prior, 0.0, 1.0))
            di_blend = float(max(w_arps * local_a["d_i"] + (1.0 - w_arps) * di_prior, 1e-6))
        else:
            b_blend, di_blend = b_prior, di_prior

        if n_obs >= 14 and "q_i" in local_a:
            qi_blend = float(local_a["q_i"])
        elif n_obs > 0:
            t_pref = np.arange(0, n_obs, dtype=float)
            obs_s = np.array([blended_seasonal[i % m_per] for i in range(n_obs)], dtype=float)
            g_pref = _arps_unit_basis(t_pref, b_blend, di_blend) * obs_s
            denom_g = float(np.dot(g_pref, g_pref))
            qi_from_obs = (
                float(np.dot(y_obs, g_pref) / max(denom_g, _EPS))
                if denom_g > _EPS
                else float(y_obs[0])
            )
            qi_prior = blended_scale * qi_ratio_prior
            qi_blend = max(float(eb_weight * qi_from_obs + (1.0 - eb_weight) * qi_prior), _EPS)
        else:
            qi_blend = max(blended_scale * qi_ratio_prior, _EPS)

        fut_t = np.arange(n_obs, n_obs + h, dtype=float)
        fut_s = np.array([blended_seasonal[(n_obs + s) % m_per] for s in range(h)], dtype=float)
        yhat = _arps_rate_curve(fut_t, qi_blend, b_blend, di_blend) * fut_s
        eur_tot, eur_rem = _arps_cumulative_eur(qi_blend, b_blend, di_blend, n_obs=n_obs)
        return (
            np.maximum(0.0, yhat),
            donor_ids,
            weights_tuple,
            eb_weight,
            prior_scale,
            blended_scale,
            {
                "q_i": round(float(qi_blend), 4),
                "b": round(float(b_blend), 6),
                "d_i": round(float(di_blend), 6),
                "eur_total": round(float(eur_tot), 4),
                "eur_remaining": round(float(eur_rem), 4),
            },
        )

    # method == "weibull_lifecycle"
    k_prior = float(
        np.sum(top_weights * np.array([d.weibull_params.get("k", 1.6) for d in top_donors]))
    )
    lam_prior = float(
        np.sum(top_weights * np.array([d.weibull_params.get("lambda", 10.0) for d in top_donors]))
    )
    amp_ratio_prior = float(
        np.sum(
            top_weights
            * np.array(
                [
                    d.weibull_params.get("amplitude", d.mean_level * 15.0) / max(d.mean_level, _EPS)
                    for d in top_donors
                ]
            )
        )
    )
    base_ratio_prior = float(
        np.sum(
            top_weights
            * np.array(
                [
                    d.weibull_params.get("baseline", d.mean_level * 0.2) / max(d.mean_level, _EPS)
                    for d in top_donors
                ]
            )
        )
    )
    if n_obs >= 4:
        _, local_w = fit_weibull_lifecycle(y_obs, h, seasonal_indices=blended_seasonal)
        k_blend = eb_weight * local_w["k"] + (1.0 - eb_weight) * k_prior
        lam_blend = eb_weight * local_w["lambda"] + (1.0 - eb_weight) * lam_prior
    else:
        k_blend, lam_blend = k_prior, lam_prior

    amp_blend = blended_scale * amp_ratio_prior
    base_blend = blended_scale * base_ratio_prior
    if n_obs > 0:
        t_pref = np.arange(1, n_obs + 1, dtype=float)
        obs_s = np.array([blended_seasonal[i % m_per] for i in range(n_obs)], dtype=float)
        pred_pref = _weibull_rate_curve(t_pref, k_blend, lam_blend, amp_blend, base_blend) * obs_s
        pref_mean = float(np.mean(pred_pref))
        if pref_mean > _EPS:
            obs_ratio = float(np.mean(y_obs)) / pref_mean
            scale_mult = eb_weight * obs_ratio + (1.0 - eb_weight) * 1.0
            amp_blend *= scale_mult
            base_blend *= scale_mult

    fut_t = np.arange(n_obs + 1, n_obs + h + 1, dtype=float)
    fut_s = np.array([blended_seasonal[(n_obs + s) % m_per] for s in range(h)], dtype=float)
    yhat = _weibull_rate_curve(fut_t, k_blend, lam_blend, amp_blend, base_blend) * fut_s
    return (
        np.maximum(0.0, yhat),
        donor_ids,
        weights_tuple,
        eb_weight,
        prior_scale,
        blended_scale,
        {
            "k": round(float(k_blend), 6),
            "lambda": round(float(lam_blend), 6),
            "amplitude": round(float(amp_blend), 4),
            "baseline": round(float(base_blend), 4),
        },
    )


def _evaluate_lodo_cv(
    donors: Sequence[DonorProfile],
    *,
    seasonal_indices: np.ndarray,
    horizon: int = 14,
    prefix_len: int = 5,
    kappa: float = _DEFAULT_KAPPA,
) -> tuple[ColdStartArchetype, float]:
    """Run Leave-One-Donor-Out CV across ``donors`` to pick the best cold-start archetype.

    Returns ``(winning_archetype, wape_lift_pct_over_local_naive)``.
    """
    if len(donors) < 2:
        return "analog_transfer", 0.0

    h_eval = max(3, min(int(horizon), 14))
    candidates: tuple[
        Literal["analog_transfer", "bass_diffusion", "weibull_lifecycle", "arps_decline"], ...
    ] = (
        "analog_transfer",
        "bass_diffusion",
        "weibull_lifecycle",
        "arps_decline",
    )
    abs_err_by_arch: dict[str, float] = dict.fromkeys(candidates, 0.0)
    naive_abs_err = 0.0
    total_actual = 0.0

    for idx, held_out in enumerate(donors):
        traj = held_out.raw_launch_trajectory
        if traj.size < prefix_len + h_eval:
            continue
        others = [d for j, d in enumerate(donors) if j != idx]
        y_pref = traj[:prefix_len]
        y_true = traj[prefix_len : prefix_len + h_eval]
        denom_sum = float(np.sum(np.abs(y_true)))
        if denom_sum <= _EPS:
            continue
        total_actual += denom_sum
        # Local naive/mean baseline from prefix
        naive_val = float(y_pref[-1]) if y_pref.size > 0 else 0.0
        naive_abs_err += float(np.sum(np.abs(y_true - naive_val)))

        for arch in candidates:
            yhat_cv, _, _, _, _, _, _ = _project_archetype_from_donors(
                arch,
                y_pref,
                h_eval,
                others,
                seasonal_indices=seasonal_indices,
                kappa=kappa,
            )
            abs_err_by_arch[arch] += float(np.sum(np.abs(y_true - yhat_cv)))

    if total_actual <= _EPS:
        return "analog_transfer", 0.0

    wape_by_arch = {arch: err / total_actual for arch, err in abs_err_by_arch.items()}
    best_arch = min(candidates, key=lambda a: wape_by_arch[a])
    naive_wape = naive_abs_err / total_actual
    best_wape = wape_by_arch[best_arch]
    lift_pct = (
        100.0 * (naive_wape - best_wape) / max(naive_wape, _EPS) if naive_wape > _EPS else 0.0
    )
    return best_arch, round(float(lift_pct), 4)


def build_cold_start_donor_pool(
    panel_df: pd.DataFrame,
    cfg: RunConfig | None = None,
    *,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
    min_obs: int | None = None,
    horizon: int | None = None,
    static_covariates: Sequence[str] = (),
    hierarchy_cols: Sequence[str] = (),
    text_cols: Sequence[str] | str | None = (),
    attribute_cols: Sequence[str] | str | None = (),
    embedding_col: str | None = None,
    oof_df: pd.DataFrame | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> ColdStartDonorPool:
    """Build a `ColdStartDonorPool` from all mature series (`n_obs >= min_obs`) in ``panel_df``."""
    _ = (text_cols, attribute_cols, embedding_col)
    eff_ts_col = cfg.data.ts_id_col if cfg is not None else ts_id_col
    eff_date_col = cfg.data.date_col if cfg is not None else date_col
    eff_target_col = cfg.data.target_col if cfg is not None else target_col
    eff_freq = cfg.data.freq if cfg is not None else freq
    eff_min_obs = (
        int(min_obs)
        if min_obs is not None
        else (int(cfg.features.cold_start_min_obs) if cfg is not None else 14)
    )
    eff_horizon = (
        int(horizon) if horizon is not None else (int(cfg.data.horizon) if cfg is not None else 14)
    )

    stat_cols = list(static_covariates)
    if cfg is not None:
        for c in cfg.features.static_covariates:
            if c not in stat_cols:
                stat_cols.append(c)
    hier_cols = list(hierarchy_cols)
    if cfg is not None and cfg.hierarchy.enabled:
        for lvl in cfg.hierarchy.levels:
            for c in lvl:
                if c not in hier_cols:
                    hier_cols.append(c)

    m_per = max(1, seasonal_period(eff_freq))
    q_levels = tuple(sorted({round(float(q), 6) for q in quantiles} | {0.1, 0.5, 0.9}))

    if panel_df is None or panel_df.empty or eff_ts_col not in panel_df.columns:
        default_qs = {q: (q - 0.5) * 0.4 for q in q_levels}
        return ColdStartDonorPool(
            freq=eff_freq,
            seasonal_period=m_per,
            min_obs=eff_min_obs,
            donors=(),
            pooled_residual_quantiles=default_qs,
        )

    if cfg is not None and cfg.features.supersession_map:
        panel_df = apply_supersession_map(
            panel_df,
            cfg.features.supersession_map,
            ts_id_col=eff_ts_col,
            date_col=eff_date_col,
            target_col=eff_target_col,
        )

    avail_cols = set(panel_df.columns) - {eff_ts_col, eff_date_col, eff_target_col}
    if not stat_cols and not hier_cols:
        # Auto-detect series-constant categorical/attribute columns in panel_df when not explicitly
        # passed so novice calls `cold_start_transfer_df(panel_df)` work with zero configuration.
        for cand_col in ("category", "brand", "region", "store", "department", "segment", "group"):
            if cand_col in avail_cols:
                stat_cols.append(cand_col)

    fallback_levels = _build_fallback_levels(stat_cols, hier_cols, avail_cols)

    # Index OOF residuals per ts_id if provided.
    oof_resid_by_id: dict[str, np.ndarray] = {}
    if (
        oof_df is not None
        and not oof_df.empty
        and eff_ts_col in oof_df.columns
        and "y_true" in oof_df.columns
    ):
        pred_c = "yhat_raw" if "yhat_raw" in oof_df.columns else "yhat"
        if pred_c in oof_df.columns:
            for raw_id, o_grp in oof_df.groupby(eff_ts_col, sort=False):
                r_vals = pd.to_numeric(o_grp["y_true"], errors="coerce").to_numpy(
                    dtype=float
                ) - pd.to_numeric(o_grp[pred_c], errors="coerce").to_numpy(dtype=float)
                r_vals = r_vals[np.isfinite(r_vals)]
                if r_vals.size > 0:
                    oof_resid_by_id[str(raw_id)] = r_vals

    donors: list[DonorProfile] = []
    all_scaled_resids: list[np.ndarray] = []

    for raw_id, grp in panel_df.groupby(eff_ts_col, sort=True):
        tid = str(raw_id)
        sub = grp.sort_values(eff_date_col) if eff_date_col in grp.columns else grp
        y_raw = (
            pd.to_numeric(sub[eff_target_col], errors="coerce").to_numpy(dtype=float)
            if eff_target_col in sub.columns
            else np.empty(0, dtype=float)
        )
        y_valid = y_raw[np.isfinite(y_raw)]
        n_obs = int(y_valid.size)
        if n_obs < eff_min_obs:
            continue

        y_pos = np.maximum(0.0, y_valid)
        n_nz = int(np.count_nonzero(y_pos > _EPS))
        mean_lvl = max(float(np.mean(y_pos)), _EPS)
        std_lvl = float(np.std(y_pos))
        cv_val = std_lvl / mean_lvl
        zero_sh = float(1.0 - (n_nz / max(1, n_obs)))
        x_idx = np.arange(n_obs, dtype=float)
        x_centered = x_idx - float(np.mean(x_idx))
        denom_x = float(np.sum(x_centered**2))
        slope_val = (
            float(np.sum(x_centered * (y_pos - mean_lvl)) / denom_x) if denom_x > _EPS else 0.0
        )

        s_indices = _extract_seasonal_indices(y_pos, m_per)
        launch_slice = y_pos[: min(n_obs, _MAX_TRAJECTORY_LEN)].copy()
        norm_traj = launch_slice / mean_lvl

        static_feats: dict[str, Any] = {}
        meta_cand_cols = [
            c for c in sub.columns if c not in {eff_ts_col, eff_date_col, eff_target_col}
        ]
        for col in dict.fromkeys([*stat_cols, *hier_cols, *meta_cand_cols]):
            if col in sub.columns:
                non_na = sub[col].dropna()
                if not non_na.empty:
                    static_feats[str(col)] = non_na.iloc[0]

        # Extract scale-normalized residuals (from OOF if available, else deseasonalized diff)
        if tid in oof_resid_by_id:
            scaled_res = oof_resid_by_id[tid] / mean_lvl
        else:
            deseas = np.array([y_pos[i] / s_indices[i % m_per] for i in range(n_obs)], dtype=float)
            if n_obs >= 4:
                smooth = pd.Series(deseas).rolling(window=min(5, n_obs), min_periods=1).mean()
                expected = np.array(
                    [float(smooth.iloc[max(0, i - 1)]) * s_indices[i % m_per] for i in range(n_obs)]
                )
                scaled_res = (y_pos - expected) / mean_lvl
            else:
                scaled_res = (y_pos - mean_lvl) / mean_lvl
        scaled_res = scaled_res[np.isfinite(scaled_res)]
        if scaled_res.size > 0:
            all_scaled_resids.append(scaled_res)
            res_qs = {float(q): float(np.quantile(scaled_res, q)) for q in q_levels}
        else:
            res_qs = {float(q): (float(q) - 0.5) * 0.4 for q in q_levels}

        _, bass_p = fit_bass_diffusion(launch_slice, eff_horizon, seasonal_indices=s_indices)
        _, weib_p = fit_weibull_lifecycle(launch_slice, eff_horizon, seasonal_indices=s_indices)
        _, arps_p = fit_arps_decline(launch_slice, eff_horizon, seasonal_indices=s_indices)

        donors.append(
            DonorProfile(
                ts_id=tid,
                n_obs=n_obs,
                n_nonzero=n_nz,
                mean_level=mean_lvl,
                std_level=std_lvl,
                cv=cv_val,
                zero_share=zero_sh,
                slope=slope_val,
                seasonal_indices=s_indices,
                normalized_trajectory=norm_traj,
                raw_launch_trajectory=launch_slice,
                static_features=static_feats,
                residual_quantiles=res_qs,
                bass_params=bass_p,
                weibull_params=weib_p,
                arps_params=arps_p,
            )
        )

    if all_scaled_resids:
        pooled_arr = np.concatenate(all_scaled_resids)
        pooled_qs = {float(q): float(np.quantile(pooled_arr, q)) for q in q_levels}
    else:
        pooled_qs = {float(q): (float(q) - 0.5) * 0.4 for q in q_levels}

    # Pre-compute LODO-CV archetype winner & Empirical-Bayes kappa per donor group
    group_archetypes: dict[str, str] = {}
    group_oof_lift: dict[str, float] = {}
    group_kappa: dict[str, float] = {}

    if donors:
        # Evaluate global group + each discovered attribute group
        grouped_donors: dict[str, list[DonorProfile]] = {"__global__": list(donors)}
        for level_cols in fallback_levels:
            if not level_cols:
                continue
            for d in donors:
                if all(col in d.static_features for col in level_cols):
                    g_key, _ = _format_donor_group(d.static_features, level_cols)
                    grouped_donors.setdefault(g_key, []).append(d)

        global_seasonal = np.mean(np.stack([d.seasonal_indices for d in donors], axis=0), axis=0)
        for g_key, g_list in grouped_donors.items():
            # Empirical-Bayes kappa = sigma^2_within / max(sigma^2_between, eps) clipped to [2, 12]
            means = np.array([d.mean_level for d in g_list], dtype=float)
            stds = np.array([d.std_level for d in g_list], dtype=float)
            var_between = float(np.var(means)) if means.size >= 2 else 0.0
            var_within = float(np.mean(stds**2))
            if var_between > _EPS and var_within > _EPS:
                kappa_g = float(np.clip(var_within / var_between, 2.0, 12.0))
            else:
                kappa_g = _DEFAULT_KAPPA
            group_kappa[g_key] = kappa_g

            arch_win, lift_val = _evaluate_lodo_cv(
                g_list,
                seasonal_indices=global_seasonal,
                horizon=eff_horizon,
                prefix_len=max(2, min(eff_min_obs // 2, 6)),
                kappa=kappa_g,
            )
            group_archetypes[g_key] = arch_win
            group_oof_lift[g_key] = lift_val

    return ColdStartDonorPool(
        freq=eff_freq,
        seasonal_period=m_per,
        min_obs=eff_min_obs,
        donors=tuple(donors),
        pooled_residual_quantiles=pooled_qs,
        attribute_Hierarchy=fallback_levels,
        group_archetypes=group_archetypes,
        group_oof_lift_pct=group_oof_lift,
        group_kappa=group_kappa,
    )


build_cold_start_donor_catalog = build_cold_start_donor_pool


def forecast_cold_start_series(
    series_df: pd.DataFrame,
    donor_pool: ColdStartDonorPool | pd.DataFrame | None = None,
    *,
    ts_id: str | None = None,
    horizon: int = 14,
    freq: str = "D",
    method: ColdStartMethod = "auto",
    matching: ColdStartMatching = "auto",
    cold_start_matching: ColdStartMatching | None = None,
    similarity_method: SimilarityMethod = "hybrid",
    min_obs: int = 14,
    top_k_donors: int = 3,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    static_covariates: Sequence[str] = (),
    hierarchy_cols: Sequence[str] = (),
    text_cols: Sequence[str] | str | None = (),
    attribute_cols: Sequence[str] | str | None = (),
    embedding_col: str | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    base_predictions: pd.DataFrame | None = None,
    kappa: float | None = None,
) -> ColdStartForecastResult:
    """Produce a cold-start NPI forecast and calibrated interval fan for one series."""
    h = max(1, int(horizon))
    if isinstance(donor_pool, pd.DataFrame):
        donor_pool = build_cold_start_donor_pool(
            donor_pool,
            freq=freq,
            horizon=h,
            min_obs=int(min_obs),
            ts_id_col=ts_id_col,
            date_col=date_col,
            target_col=target_col,
            static_covariates=static_covariates,
            hierarchy_cols=hierarchy_cols,
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            embedding_col=embedding_col,
        )
    eff_freq = donor_pool.freq if donor_pool is not None else freq
    m_per = (
        donor_pool.seasonal_period if donor_pool is not None else max(1, seasonal_period(eff_freq))
    )
    eff_min_obs = int(min_obs)
    eff_matching: ColdStartMatching = (
        cold_start_matching if cold_start_matching is not None else matching
    )
    q_levels = tuple(sorted({round(float(q), 6) for q in quantiles} | {0.1, 0.5, 0.9}))

    tid = str(ts_id) if ts_id is not None else "series_0"
    if (
        ts_id is None
        and series_df is not None
        and not series_df.empty
        and ts_id_col in series_df.columns
    ):
        non_na_id = series_df[ts_id_col].dropna()
        if not non_na_id.empty:
            tid = str(non_na_id.iloc[0])

    sub = (
        series_df.sort_values(date_col).reset_index(drop=True)
        if (series_df is not None and not series_df.empty and date_col in series_df.columns)
        else (series_df.copy() if series_df is not None else pd.DataFrame())
    )
    y_raw = (
        pd.to_numeric(sub[target_col], errors="coerce").to_numpy(dtype=float)
        if (not sub.empty and target_col in sub.columns)
        else np.empty(0, dtype=float)
    )
    y_valid = y_raw[np.isfinite(y_raw)]
    y_obs = np.maximum(0.0, y_valid)
    n_obs = int(y_obs.size)
    n_nz = int(np.count_nonzero(y_obs > _EPS))
    is_cs = bool(n_obs < eff_min_obs)

    # Extract static/hierarchy/text/attribute metadata from series_df
    target_attrs: dict[str, Any] = {}
    if not sub.empty:
        cand_cols = set(sub.columns) - {ts_id_col, date_col, target_col}
        for col in (*static_covariates, *hierarchy_cols, *sorted(cand_cols)):
            if col in sub.columns:
                vals = sub[col].dropna()
                if not vals.empty:
                    target_attrs[str(col)] = vals.iloc[0]

    has_multimodal_args = bool(text_cols or attribute_cols or embedding_col)
    eff_attr_cols: list[str] = []
    for c_src in (attribute_cols, static_covariates):
        if isinstance(c_src, str):
            if c_src and c_src not in eff_attr_cols:
                eff_attr_cols.append(c_src)
        elif c_src is not None:
            for c_item in c_src:
                if str(c_item) and str(c_item) not in eff_attr_cols:
                    eff_attr_cols.append(str(c_item))

    prior_scale_col_val: float | None = None
    for scale_col in ("launch_scale", "prior_scale", "expected_scale"):
        if scale_col in target_attrs:
            try:
                val = float(target_attrs[scale_col])
                if math.isfinite(val) and val > 0.0:
                    prior_scale_col_val = val
                    break
            except (TypeError, ValueError) as exc:
                _log.debug("Ignoring non-numeric cold-start scale column %r: %s", scale_col, exc)

    donor_sim_map: dict[str, float] | None = None
    matching_mode_used = "attribute"
    if donor_pool is not None and donor_pool.donors:
        all_candidates = [d for d in donor_pool.donors if d.ts_id != tid]
        fallback_levels = (
            donor_pool.attribute_Hierarchy
            if donor_pool.attribute_Hierarchy
            else _build_fallback_levels(static_covariates, hierarchy_cols, set(target_attrs.keys()))
        )
        attr_donors, attr_group, attr_level = _match_donors_hierarchical(
            target_attrs,
            donor_pool.donors,
            fallback_levels,
            exclude_ts_id=tid,
        )
        if eff_matching == "similarity":
            matched_donors = all_candidates
            donor_group = f"similarity:{similarity_method}"
            matched_level = "similarity_launch_prefix"
            matching_mode_used = "similarity"
            if (n_obs >= 2 or has_multimodal_args) and matched_donors:
                donor_sim_map = _compute_donor_launch_similarity(
                    y_obs,
                    matched_donors,
                    similarity_method=similarity_method,
                    target_attrs=target_attrs,
                    text_cols=text_cols,
                    attribute_cols=eff_attr_cols if has_multimodal_args else None,
                    hierarchy_cols=hierarchy_cols if has_multimodal_args else None,
                    embedding_col=embedding_col,
                    min_obs=eff_min_obs,
                )
        elif eff_matching == "hybrid":
            matched_donors = (
                all_candidates
                if has_multimodal_args
                else (attr_donors if attr_donors else all_candidates)
            )
            donor_group = (
                attr_group if attr_group != "__global__" else f"hybrid:{similarity_method}"
            )
            matched_level = (
                f"{attr_level}+similarity"
                if attr_level != "__global__"
                else "similarity_launch_prefix"
            )
            matching_mode_used = "hybrid"
            if (n_obs >= 2 or has_multimodal_args or bool(hierarchy_cols)) and matched_donors:
                donor_sim_map = _compute_donor_launch_similarity(
                    y_obs,
                    matched_donors,
                    similarity_method=similarity_method,
                    target_attrs=target_attrs,
                    text_cols=text_cols,
                    attribute_cols=eff_attr_cols or None,
                    hierarchy_cols=hierarchy_cols or None,
                    embedding_col=embedding_col,
                    min_obs=eff_min_obs,
                )
        elif eff_matching == "attribute":
            matched_donors = attr_donors
            donor_group = attr_group
            matched_level = attr_level
            matching_mode_used = "attribute"
        else:
            # eff_matching == "auto":
            if has_multimodal_args and all_candidates:
                matched_donors = all_candidates
                donor_group = (
                    attr_group if attr_group != "__global__" else f"hybrid:{similarity_method}"
                )
                matched_level = (
                    f"{attr_level}+similarity"
                    if attr_level != "__global__"
                    else "similarity_launch_prefix"
                )
                matching_mode_used = "hybrid"
                donor_sim_map = _compute_donor_launch_similarity(
                    y_obs,
                    matched_donors,
                    similarity_method=similarity_method,
                    target_attrs=target_attrs,
                    text_cols=text_cols,
                    attribute_cols=eff_attr_cols or None,
                    hierarchy_cols=hierarchy_cols or None,
                    embedding_col=embedding_col,
                    min_obs=eff_min_obs,
                )
            elif attr_level == "__global__" and n_obs >= 2 and all_candidates:
                matched_donors = all_candidates
                donor_group = f"similarity:{similarity_method}"
                matched_level = "similarity_launch_prefix"
                matching_mode_used = "similarity"
                donor_sim_map = _compute_donor_launch_similarity(
                    y_obs, matched_donors, similarity_method=similarity_method
                )
            elif attr_level != "__global__" and n_obs >= 3 and attr_donors:
                matched_donors = attr_donors
                donor_group = attr_group
                matched_level = attr_level
                matching_mode_used = "hybrid"
                donor_sim_map = _compute_donor_launch_similarity(
                    y_obs, matched_donors, similarity_method=similarity_method
                )
            else:
                matched_donors = attr_donors
                donor_group = attr_group
                matched_level = attr_level
                matching_mode_used = "attribute"

        eff_kappa = (
            float(kappa)
            if kappa is not None
            else donor_pool.group_kappa.get(
                attr_group, donor_pool.group_kappa.get("__global__", _DEFAULT_KAPPA)
            )
        )
        oof_lift = donor_pool.group_oof_lift_pct.get(
            attr_group, donor_pool.group_oof_lift_pct.get("__global__", 0.0)
        )
    else:
        matched_donors = []
        donor_group = "__global__"
        matched_level = "__global__"
        eff_kappa = float(kappa) if kappa is not None else _DEFAULT_KAPPA
        oof_lift = 0.0
        matching_mode_used = (
            eff_matching if eff_matching in ("attribute", "similarity", "hybrid") else "attribute"
        )

    # Resolve archetype method
    if method == "auto":
        if donor_pool is not None and donor_group in donor_pool.group_archetypes:
            resolved_arch: Literal[
                "analog_transfer", "bass_diffusion", "weibull_lifecycle", "arps_decline"
            ] = donor_pool.group_archetypes[donor_group]  # type: ignore[assignment]
        elif donor_pool is not None and "__global__" in donor_pool.group_archetypes:
            resolved_arch = donor_pool.group_archetypes["__global__"]  # type: ignore[assignment]
        elif matched_donors:
            resolved_arch = "analog_transfer"
        else:
            resolved_arch = "bass_diffusion"
    else:
        resolved_arch = method

    default_seasonal = _extract_seasonal_indices(y_obs, m_per)
    yhat_cs, donor_ids, sim_weights, eb_w, prior_scale, blended_scale, lc_params = (
        _project_archetype_from_donors(
            resolved_arch,
            y_obs,
            h,
            matched_donors,
            seasonal_indices=default_seasonal,
            top_k_donors=top_k_donors,
            kappa=eff_kappa,
            prior_scale_override=prior_scale_col_val,
            donor_similarity_map=donor_sim_map,
        )
    )

    if donor_ids and donor_sim_map:
        best_sim_score = float(
            np.sum(
                np.asarray(sim_weights, dtype=float)
                * np.array([donor_sim_map.get(did, 0.5) for did in donor_ids], dtype=float)
            )
        )
    elif sim_weights:
        best_sim_score = float(np.max(sim_weights))
    else:
        best_sim_score = 0.0

    # Blend with base_predictions if provided and n_obs > 0
    if (
        base_predictions is not None
        and not base_predictions.empty
        and "yhat" in base_predictions.columns
    ):
        base_yhat = pd.to_numeric(base_predictions["yhat"], errors="coerce").to_numpy(dtype=float)
        if base_yhat.size == h and np.all(np.isfinite(base_yhat)):
            if is_cs:
                # Empirical-Bayes blend between donor trajectory and base model prediction
                yhat_final = eb_w * np.maximum(0.0, base_yhat) + (1.0 - eb_w) * yhat_cs
            else:
                yhat_final = base_yhat
        else:
            yhat_final = yhat_cs
    else:
        yhat_final = yhat_cs

    # Pooled conformal residual quantiles from matched donors
    if matched_donors:
        pooled_q_map: dict[float, float] = {}
        for q in q_levels:
            q_vals = [d.residual_quantiles.get(q, (q - 0.5) * 0.4) for d in matched_donors]
            pooled_q_map[q] = float(np.median(q_vals))
    elif donor_pool is not None and donor_pool.pooled_residual_quantiles:
        pooled_q_map = dict(donor_pool.pooled_residual_quantiles)
    else:
        pooled_q_map = {q: (q - 0.5) * 0.4 for q in q_levels}

    # Construct future dates
    if (
        base_predictions is not None
        and not base_predictions.empty
        and "ds" in base_predictions.columns
        and len(base_predictions) == h
    ):
        fut_dates = pd.to_datetime(base_predictions["ds"]).to_numpy()
    elif not sub.empty and date_col in sub.columns and sub[date_col].notna().any():
        last_dt = pd.Timestamp(sub[date_col].dropna().iloc[-1])
        fut_dates = pd.date_range(start=last_dt, periods=h + 1, freq=eff_freq)[1:].to_numpy()
    else:
        fut_dates = pd.date_range(start="2025-01-01", periods=h, freq=eff_freq).to_numpy()

    lo_q, hi_q = q_levels[0], q_levels[-1]
    lower_list: list[float] = []
    upper_list: list[float] = []
    qjson_list: list[str] = []
    for step_idx in range(h):
        center = float(yhat_final[step_idx])
        # Widening factor across horizon steps: sqrt(1 + 0.05 * step)
        step_widen = math.sqrt(1.0 + 0.05 * step_idx)
        scale_ref = max(center, blended_scale, 1.0)
        raw_qmap: dict[float, float] = {}
        for q in q_levels:
            rel_offset = pooled_q_map.get(q, (q - 0.5) * 0.4)
            val = center + scale_ref * rel_offset * step_widen
            if q == 0.5:
                val = center
            raw_qmap[q] = max(0.0, float(val))
        mono_qmap = rearrange_monotone_quantiles(raw_qmap)
        lower_list.append(float(mono_qmap.get(lo_q, center * 0.8)))
        upper_list.append(float(mono_qmap.get(hi_q, center * 1.2)))
        qjson_list.append(json.dumps({str(q): round(v, 6) for q, v in mono_qmap.items()}))

    pred_df = pd.DataFrame(
        {
            "ds": fut_dates,
            "yhat": yhat_final.astype(float),
            "yhat_raw": (
                base_predictions["yhat_raw"].to_numpy(dtype=float)
                if (
                    base_predictions is not None
                    and "yhat_raw" in base_predictions.columns
                    and len(base_predictions) == h
                )
                else yhat_final.astype(float)
            ),
            "yhat_adjusted": yhat_final.astype(float),
            "yhat_lower": np.asarray(lower_list, dtype=float),
            "yhat_upper": np.asarray(upper_list, dtype=float),
            "quantiles": pd.array(qjson_list, dtype="string"),
        },
        columns=list(CALIBRATED_COLUMNS),
    )

    pooled_rel_q80 = float(
        (pooled_q_map.get(0.9, 0.2) - pooled_q_map.get(0.1, -0.2)) * max(blended_scale, 1.0)
    )
    if n_obs == 0:
        stage = "pre_launch"
    elif is_cs:
        stage = "early_launch"
    else:
        stage = "mature"

    arch_used: ColdStartArchetype = resolved_arch if is_cs else "none"
    diag: dict[str, Any] = {
        "is_cold_start": is_cs,
        "cold_start_flag": is_cs,
        "n_obs": n_obs,
        "donor_group": donor_group,
        "matched_level": matched_level,
        "matching_mode_used": matching_mode_used,
        "donor_similarity_score": round(best_sim_score, 6),
        "top_donor_ts_ids": list(donor_ids),
        "selected_archetype": resolved_arch,
        "method_used": arch_used,
        "donor_ts_ids": list(donor_ids),
        "similarity_weights": list(sim_weights),
        "empirical_bayes_weight": round(eb_w, 6),
        "prior_scale": round(prior_scale, 4),
        "blended_scale": round(blended_scale, 4),
        "lifecycle_stage": stage,
        "lifecycle_params": lc_params,
        "pooled_q80_width": round(pooled_rel_q80, 4),
        "oof_wape_lift_pct": round(float(oof_lift), 4),
    }

    return ColdStartForecastResult(
        ts_id=tid,
        cold_start_flag=is_cs,
        is_cold_start=is_cs,
        n_obs=n_obs,
        n_nonzero=n_nz,
        donor_group=donor_group,
        matched_level=matched_level,
        selected_archetype=resolved_arch,
        method_used=arch_used,
        donor_ts_ids=donor_ids,
        similarity_weights=sim_weights,
        empirical_bayes_weight=round(eb_w, 6),
        prior_scale=round(prior_scale, 4),
        scale_factor=round(blended_scale / max(prior_scale, _EPS), 6),
        lifecycle_stage=stage,
        lifecycle_params=lc_params,
        pooled_q80_width=round(pooled_rel_q80, 4),
        oof_wape_lift_pct=round(float(oof_lift), 4),
        predictions=pred_df,
        diagnostics=diag,
        matching_mode_used=matching_mode_used,
        donor_similarity_score=round(best_sim_score, 6),
        top_donor_ts_ids=donor_ids,
    )


def cold_start_transfer_df(
    panel_df: pd.DataFrame,
    *,
    cfg: RunConfig | None = None,
    forecasts_df: pd.DataFrame | None = None,
    predictions_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    min_obs: int | None = None,
    method: ColdStartMethod | None = None,
    matching: ColdStartMatching | None = None,
    cold_start_matching: ColdStartMatching | None = None,
    similarity_method: SimilarityMethod | None = None,
    supersession_map: Mapping[str, str] | None = None,
    static_covariates: Sequence[str] = (),
    hierarchy_cols: Sequence[str] = (),
    text_cols: Sequence[str] | str | None = (),
    attribute_cols: Sequence[str] | str | None = (),
    embedding_col: str | None = None,
    top_k_donors: int = 3,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    horizon: int | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
    """Return a per-series cold-start NPI transfer diagnostic and archetype summary DataFrame."""
    eff_ts_col = cfg.data.ts_id_col if cfg is not None else ts_id_col
    eff_date_col = cfg.data.date_col if cfg is not None else date_col
    eff_target_col = cfg.data.target_col if cfg is not None else target_col
    eff_freq = freq if freq is not None else (cfg.data.freq if cfg is not None else "D")
    eff_horizon = (
        int(horizon) if horizon is not None else (int(cfg.data.horizon) if cfg is not None else 14)
    )
    eff_min_obs = (
        int(min_obs)
        if min_obs is not None
        else (int(cfg.features.cold_start_min_obs) if cfg is not None else 14)
    )
    eff_method: ColdStartMethod = (
        method
        if method is not None
        else (cfg.features.cold_start_method if cfg is not None else "auto")
    )
    req_matching = cold_start_matching if cold_start_matching is not None else matching
    eff_matching: ColdStartMatching = (
        req_matching
        if req_matching is not None
        else (cfg.features.cold_start_matching if cfg is not None else "auto")
    )
    eff_sim_method: SimilarityMethod = (
        similarity_method
        if similarity_method is not None
        else (cfg.features.similarity_method if cfg is not None else "hybrid")
    )
    eff_sup_map = (
        supersession_map
        if supersession_map is not None
        else (cfg.features.supersession_map if cfg is not None else None)
    )
    if eff_sup_map and panel_df is not None and not panel_df.empty:
        panel_df = apply_supersession_map(
            panel_df,
            eff_sup_map,
            ts_id_col=eff_ts_col,
            date_col=eff_date_col,
            target_col=eff_target_col,
        )

    preds_src = forecasts_df if forecasts_df is not None else predictions_df
    donor_pool = build_cold_start_donor_pool(
        panel_df,
        cfg,
        ts_id_col=eff_ts_col,
        date_col=eff_date_col,
        target_col=eff_target_col,
        freq=eff_freq,
        min_obs=eff_min_obs,
        static_covariates=static_covariates,
        hierarchy_cols=hierarchy_cols,
        text_cols=text_cols,
        attribute_cols=attribute_cols,
        embedding_col=embedding_col,
        oof_df=oof_df,
    )

    columns = [
        "ts_id",
        "n_obs",
        "n_nonzero",
        "is_cold_start",
        "cold_start_flag",
        "donor_group",
        "matched_level",
        "matching_mode_used",
        "selected_archetype",
        "method_used",
        "donor_ts_ids",
        "top_donor_ts_ids",
        "donor_similarity_score",
        "similarity_scores",
        "empirical_bayes_weight",
        "prior_scale",
        "scale_factor",
        "lifecycle_stage",
        "bass_p",
        "bass_q",
        "bass_m",
        "weibull_k",
        "weibull_lambda",
        "arps_qi",
        "arps_b",
        "arps_di",
        "eur_total",
        "eur_remaining",
        "pooled_q80_width",
        "yhat_mean",
        "mean_forecast",
        "interval_width_80",
        "oof_wape_lift_pct",
    ]
    if panel_df is None or panel_df.empty or eff_ts_col not in panel_df.columns:
        return pd.DataFrame(columns=columns)

    preds_by_id: dict[str, pd.DataFrame] = {}
    if preds_src is not None and not preds_src.empty and eff_ts_col in preds_src.columns:
        for raw_id, p_grp in preds_src.groupby(eff_ts_col, sort=False):
            if "model_type" in p_grp.columns and p_grp["model_type"].nunique() > 1:
                first_m = p_grp["model_type"].iloc[0]
                p_grp = p_grp[p_grp["model_type"] == first_m]
            preds_by_id[str(raw_id)] = p_grp

    rows: list[dict[str, Any]] = []
    for raw_id, grp in panel_df.groupby(eff_ts_col, sort=True):
        tid = str(raw_id)
        cs_res = forecast_cold_start_series(
            grp,
            donor_pool,
            horizon=eff_horizon,
            freq=eff_freq,
            method=eff_method,
            matching=eff_matching,
            similarity_method=eff_sim_method,
            min_obs=eff_min_obs,
            top_k_donors=top_k_donors,
            ts_id_col=eff_ts_col,
            date_col=eff_date_col,
            target_col=eff_target_col,
            static_covariates=static_covariates,
            hierarchy_cols=hierarchy_cols,
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            embedding_col=embedding_col,
            base_predictions=preds_by_id.get(tid),
        )
        yhat_mean_val = float(cs_res.predictions["yhat"].mean())
        iw80_val = float(
            (cs_res.predictions["yhat_upper"] - cs_res.predictions["yhat_lower"]).mean()
        )
        lcp = cs_res.lifecycle_params
        rows.append(
            {
                "ts_id": tid,
                "n_obs": cs_res.n_obs,
                "n_nonzero": cs_res.n_nonzero,
                "is_cold_start": cs_res.is_cold_start,
                "cold_start_flag": cs_res.cold_start_flag,
                "donor_group": cs_res.donor_group,
                "matched_level": cs_res.matched_level,
                "matching_mode_used": cs_res.matching_mode_used,
                "selected_archetype": cs_res.selected_archetype,
                "method_used": cs_res.method_used,
                "donor_ts_ids": ",".join(cs_res.donor_ts_ids),
                "top_donor_ts_ids": ",".join(cs_res.top_donor_ts_ids),
                "donor_similarity_score": cs_res.donor_similarity_score,
                "similarity_scores": ",".join(f"{w:.3f}" for w in cs_res.similarity_weights),
                "empirical_bayes_weight": cs_res.empirical_bayes_weight,
                "prior_scale": cs_res.prior_scale,
                "scale_factor": cs_res.scale_factor,
                "lifecycle_stage": cs_res.lifecycle_stage,
                "bass_p": lcp.get("p", float("nan")),
                "bass_q": lcp.get("q", float("nan")),
                "bass_m": lcp.get("m", float("nan")),
                "weibull_k": lcp.get("k", float("nan")),
                "weibull_lambda": lcp.get("lambda", float("nan")),
                "arps_qi": lcp.get("q_i", float("nan")),
                "arps_b": lcp.get("b", float("nan")),
                "arps_di": lcp.get("d_i", float("nan")),
                "eur_total": lcp.get("eur_total", float("nan")),
                "eur_remaining": lcp.get("eur_remaining", float("nan")),
                "pooled_q80_width": cs_res.pooled_q80_width,
                "yhat_mean": round(yhat_mean_val, 4),
                "mean_forecast": round(yhat_mean_val, 4),
                "interval_width_80": round(iw80_val, 4),
                "oof_wape_lift_pct": cs_res.oof_wape_lift_pct,
            }
        )

    out_df = pd.DataFrame(rows, columns=columns)
    out_df.attrs["donor_pool"] = donor_pool
    return out_df


cold_start_diagnostic_df = cold_start_transfer_df
cold_start_report_df = cold_start_transfer_df


def plot_cold_start_transfer(
    panel_or_transfer_df: pd.DataFrame,
    panel_df: pd.DataFrame | None = None,
    *,
    ts_id: str | None = None,
    donor_pool: ColdStartDonorPool | None = None,
    horizon: int = 14,
    method: ColdStartMethod = "auto",
    min_obs: int = 14,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    ax: Any | None = None,
) -> Any:
    """Plot a cold-start SKU's observed prefix, matched donor trajectories, and transferred fan."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    work_df = panel_df if panel_df is not None else panel_or_transfer_df
    if work_df is None or work_df.empty:
        ax.set_title("Cold-Start Transfer (Empty DataFrame)")
        return ax

    if "empirical_bayes_weight" in work_df.columns and target_col not in work_df.columns:
        # Summary table passed directly: plot bar chart of Empirical-Bayes weights & mean forecasts
        df_sub = work_df.head(15)
        ax.bar(df_sub[ts_id_col].astype(str), df_sub["yhat_mean"].to_numpy(dtype=float), alpha=0.8)
        ax.set_title("Cold-Start Transferred Mean Forecast by Series")
        ax.set_xlabel("Series ID")
        ax.set_ylabel("Mean Forecast")
        return ax

    pool = donor_pool or build_cold_start_donor_pool(
        work_df,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        min_obs=min_obs,
        horizon=horizon,
    )
    target_id = ts_id
    if target_id is None:
        for raw_id, grp in work_df.groupby(ts_id_col, sort=True):
            n_valid = int(pd.to_numeric(grp[target_col], errors="coerce").notna().sum())
            if n_valid < min_obs:
                target_id = str(raw_id)
                break
        if target_id is None:
            target_id = str(work_df[ts_id_col].iloc[0])

    series_sub = work_df[work_df[ts_id_col].astype(str) == str(target_id)]
    res = forecast_cold_start_series(
        series_sub,
        pool,
        horizon=horizon,
        method=method,
        min_obs=min_obs,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
    )

    y_obs = (
        pd.to_numeric(series_sub[target_col], errors="coerce").dropna().to_numpy(dtype=float)
        if not series_sub.empty and target_col in series_sub.columns
        else np.empty(0, dtype=float)
    )
    n_obs = y_obs.size
    if n_obs > 0:
        ax.plot(np.arange(1, n_obs + 1), y_obs, "o-", label=f"{target_id} observed (n={n_obs})")

    fut_x = np.arange(n_obs + 1, n_obs + len(res.predictions) + 1)
    yhat = res.predictions["yhat"].to_numpy(dtype=float)
    lo = res.predictions["yhat_lower"].to_numpy(dtype=float)
    hi = res.predictions["yhat_upper"].to_numpy(dtype=float)
    ax.plot(
        fut_x,
        yhat,
        "--",
        linewidth=2.0,
        label=f"Transferred ({res.selected_archetype}, w_EB={res.empirical_bayes_weight:.2f})",
    )
    ax.fill_between(fut_x, lo, hi, alpha=0.25, label="Pooled 80% Conformal Interval")
    ax.set_title(f"Cold-Start NPI Transfer — {target_id} ({res.donor_group})")
    ax.set_xlabel("Launch Period (t)")
    ax.set_ylabel("Demand")
    ax.legend(loc="best")
    return ax
