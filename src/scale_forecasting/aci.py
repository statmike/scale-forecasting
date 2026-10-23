"""Online Adaptive Conformal Inference (`ACI` & Volatility-Normalized `ACI`) — pure.

Standard split/out-of-fold conformal prediction intervals assume exchangeability between historical
backtest folds and future periods. Under mid-series volatility shocks, price regime shifts, or
structural breaks, static conformal intervals can suffer severe miscoverage.

This module implements:
1. **Gibbs & Candès (2021) Online Quantile Adaptation (`"aci"`):**
   $$\\alpha_{t+1} = \\text{clip}(\\alpha_t + \\gamma (\\alpha_{\\text{target}} - \\text{err}_t),
   \\; \\epsilon,\\; 1 - \\epsilon)$$
   where $\\text{err}_t = \\mathbb{I}\\{y_t \\notin [\\hat{L}_t, \\hat{U}_t]\\}$.
2. **Scale-Normalized / Volatility-Weighted ACI (`"aci_normalized"`):**
   Couples online $\\alpha_t$ adaptation with an Exponentially Weighted Moving Average (EWMA) local
   residual volatility multiplier $s_{i,t} = \\hat{\\sigma}_{i,t} / \\hat{\\sigma}_{i,0}$
   so prediction intervals expand immediately during volatility bursts and contract smoothly when
   stability returns.
3. **Leak-Free Multi-Step Horizon Feedback Delay ($h$-Step Lag):**
   For an $h$-step-ahead forecast at index $t$, online state updates $(\\alpha_t, s_{i,t})$ depend
   strictly on realized errors at indices $\\tau \\le t - h$ (i.e., available at forecast origin
   $t - h + 1$), guaranteeing zero lookahead leakage across multi-step horizons.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from .calibration import (
    MIN_RESIDUALS_TOTAL,
    Calibration,
    StepCalibration,
    empirical_quantile,
    rearrange_monotone_quantiles,
    window_residuals,
)
from .models.base_model import DEFAULT_QUANTILES

if TYPE_CHECKING:
    from .config import CalibrationMethod

_ALPHA_MIN = 0.005
_ALPHA_MAX = 0.995
_EPS = 1e-9
_EWMA_BETA = 0.35


@dataclass(frozen=True)
class AciCalibrationState:
    """Summary of online ACI state and empirical coverage across an OOF or streaming trajectory."""

    method: Literal["conformal", "aci", "aci_normalized"]
    target_alpha: float
    gamma: float
    terminal_alpha: float
    empirical_coverage_split: float
    empirical_coverage_aci: float
    mean_width_split: float
    mean_width_aci: float
    volatility_scale: float
    drift_detected: bool
    alpha_trajectory: tuple[float, ...]
    err_trajectory: tuple[int, ...]
    q80_per_step: dict[int, float]
    q80_flat: float
    quantile_offsets_per_step: dict[int, dict[float, float]]
    quantile_offsets_flat: dict[float, float]

    @property
    def n_steps(self) -> int:
        """Number of online trajectory steps processed."""
        return len(self.alpha_trajectory)


def _normal_quantile_ratio(alpha_eff: float, alpha_target: float) -> float:
    """Return $\\Phi^{-1}(1 - \\alpha_{\\text{eff}}/2) / \\Phi^{-1}(1 - \\alpha_0/2)$."""
    a_eff = float(np.clip(alpha_eff, _ALPHA_MIN, _ALPHA_MAX))
    a_tar = float(np.clip(alpha_target, _ALPHA_MIN, _ALPHA_MAX))
    z_eff = float(norm.ppf(1.0 - 0.5 * a_eff))
    z_tar = max(float(norm.ppf(1.0 - 0.5 * a_tar)), 0.05)
    return max(0.1, z_eff / z_tar)


def _run_single_series_aci(
    sub_df: pd.DataFrame,
    *,
    gamma: float,
    target_coverage: float,
    normalized: bool,
    horizon_delay: bool,
    ts_id_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, AciCalibrationState]:
    """Run leak-free online ACI on a single `(ts_id, model_type)` slice ordered chronologically."""
    df = sub_df.copy()
    if date_col in df.columns:
        df = df.sort_values(
            [c for c in (date_col, "fold_id", "horizon_step") if c in df.columns],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    target_cov = float(np.clip(target_coverage, 0.05, 0.99))
    alpha_target = round(1.0 - target_cov, 6)
    gamma_val = float(np.clip(gamma, 1e-4, 0.5))
    method_name: Literal["conformal", "aci", "aci_normalized"] = (
        "aci_normalized" if normalized else "aci"
    )

    tid = str(df[ts_id_col].iloc[0]) if (ts_id_col in df.columns and not df.empty) else "series_0"
    mtype = (
        str(df["model_type"].iloc[0]) if ("model_type" in df.columns and not df.empty) else "model"
    )

    if n == 0:
        empty_state = AciCalibrationState(
            method=method_name,
            target_alpha=alpha_target,
            gamma=gamma_val,
            terminal_alpha=alpha_target,
            empirical_coverage_split=target_cov,
            empirical_coverage_aci=target_cov,
            mean_width_split=0.0,
            mean_width_aci=0.0,
            volatility_scale=1.0,
            drift_detected=False,
            alpha_trajectory=(),
            err_trajectory=(),
            q80_per_step={},
            q80_flat=0.0,
            quantile_offsets_per_step={},
            quantile_offsets_flat={},
        )
        return pd.DataFrame(), empty_state

    y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    pred_col = "yhat" if "yhat" in df.columns else "yhat_raw"
    yhat = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    if "horizon_step" in df.columns:
        h_steps = pd.to_numeric(df["horizon_step"], errors="coerce").fillna(1).to_numpy(dtype=int)
        h_steps = np.maximum(1, h_steps)
    else:
        h_steps = np.ones(n, dtype=int)

    abs_errors = np.abs(y_true - yhat)
    signed_resids = y_true - yhat

    # Baseline calibration window: initial warmup prefix (first max(5, min(12, n // 3)) points)
    warmup = max(3, min(10, n // 3)) if n >= 6 else max(1, n // 2)
    init_errors = abs_errors[:warmup]
    init_errors = init_errors[np.isfinite(init_errors)]
    if init_errors.size > 0:
        base_half_width = float(np.quantile(init_errors, target_cov))
        base_sigma = max(float(np.mean(init_errors)), float(np.median(init_errors)), _EPS)
    else:
        base_half_width = 1.0
        base_sigma = 1.0
    base_half_width = max(base_half_width, base_sigma * 0.5, 1e-3)

    # Static split-conformal half-width (fixed from initial calibration split so static vs ACI is a
    # fair, leak-free comparison on the same stream).
    static_lower = np.zeros(n, dtype=float)
    static_upper = np.zeros(n, dtype=float)
    aci_lower = np.zeros(n, dtype=float)
    aci_upper = np.zeros(n, dtype=float)
    alpha_traj = np.zeros(n, dtype=float)
    err_traj = np.zeros(n, dtype=int)
    vol_scale_arr = np.ones(n, dtype=float)
    static_cov_Point = np.zeros(n, dtype=int)
    aci_cov_point = np.zeros(n, dtype=int)
    drift_flags = np.zeros(n, dtype=bool)

    # Online state variables updated strictly as delayed realizations arrive
    alpha_curr = alpha_target
    ewma_sigma = base_sigma
    applied_idx = -1  # highest index tau whose realization has been incorporated

    for t in range(n):
        h_t = int(h_steps[t]) if horizon_delay else 1
        max_avail_tau = t - h_t

        # Incorporate newly realized observations tau in (applied_idx + 1 .. max_avail_tau)
        while applied_idx + 1 <= max_avail_tau:
            tau = applied_idx + 1
            applied_idx = tau
            if math.isfinite(abs_errors[tau]):
                err_tau = int(err_traj[tau])
                # Gibbs & Candes (2021) online update:
                # err_tau == 1 (miscovered) -> alpha_curr decreases -> wider interval
                alpha_curr = float(
                    np.clip(
                        alpha_curr + gamma_val * (alpha_target - float(err_tau)),
                        _ALPHA_MIN,
                        _ALPHA_MAX,
                    )
                )
                ewma_sigma = _EWMA_BETA * float(abs_errors[tau]) + (1.0 - _EWMA_BETA) * ewma_sigma

        alpha_traj[t] = alpha_curr
        vol_mult = (ewma_sigma / max(base_sigma, _EPS)) if normalized else 1.0
        vol_mult = float(np.clip(vol_mult, 0.25, 25.0))
        vol_scale_arr[t] = vol_mult

        # Available realized residuals up to max_avail_tau
        if max_avail_tau >= 0:
            avail_errs = abs_errors[: max_avail_tau + 1]
            avail_errs = avail_errs[np.isfinite(avail_errs)]
        else:
            avail_errs = np.empty(0, dtype=float)

        # Static split-conformal uses warmup split without online adaptation
        s_hw = base_half_width
        static_lower[t] = yhat[t] - s_hw
        static_upper[t] = yhat[t] + s_hw

        # Online ACI half-width at step t using only realized data <= t - h_t
        q_level_eff = float(np.clip(1.0 - alpha_curr, 0.01, 0.995))
        z_ratio = _normal_quantile_ratio(alpha_curr, alpha_target)

        if avail_errs.size >= 3:
            if normalized:
                # Scale-normalized recent nonconformity scores + EWMA local volatility scale
                recent_window = avail_errs[-min(avail_errs.size, 20) :]
                emp_q = float(np.quantile(recent_window, min(q_level_eff, 0.98)))
                aci_hw = max(
                    emp_q * z_ratio,
                    base_half_width * vol_mult * z_ratio,
                    ewma_sigma * float(norm.ppf(1.0 - 0.5 * alpha_curr)),
                )
            else:
                # Unnormalized online ACI: recent sliding window empirical quantile + tail expansion
                recent_window = avail_errs[-min(avail_errs.size, 15) :]
                emp_q = float(np.quantile(recent_window, min(q_level_eff, 0.98)))
                recent_med = float(np.mean(recent_window[-min(recent_window.size, 5) :]))
                aci_hw = max(
                    emp_q * z_ratio,
                    recent_med * float(norm.ppf(1.0 - 0.5 * alpha_curr)),
                )
        else:
            aci_hw = base_half_width * vol_mult * z_ratio

        aci_lower[t] = yhat[t] - aci_hw
        aci_upper[t] = yhat[t] + aci_hw

        # Evaluate coverage at step t (used only when tau = t is realized at future step >= t + h)
        if math.isfinite(y_true[t]):
            s_cov = int(static_lower[t] <= y_true[t] <= static_upper[t])
            a_cov = int(aci_lower[t] <= y_true[t] <= aci_upper[t])
        else:
            s_cov = 1
            a_cov = 1
        static_cov_Point[t] = s_cov
        aci_cov_point[t] = a_cov
        err_traj[t] = 1 - a_cov

        # Drift flag when local volatility jumps > 1.75x or alpha_t drops < 0.65 * alpha_target
        drift_flags[t] = bool(
            vol_mult > 1.75
            or alpha_curr < 0.65 * alpha_target
            or (
                max_avail_tau >= 4
                and float(np.mean(abs_errors[max(0, max_avail_tau - 3) : max_avail_tau + 1]))
                > 1.8 * base_sigma
            )
        )

    # Final update incorporating terminal observations for forward horizon calibration
    terminal_alpha = alpha_curr
    terminal_ewma = ewma_sigma
    for tau in range(applied_idx + 1, n):
        if math.isfinite(abs_errors[tau]):
            terminal_alpha = float(
                np.clip(
                    terminal_alpha + gamma_val * (alpha_target - float(err_traj[tau])),
                    _ALPHA_MIN,
                    _ALPHA_MAX,
                )
            )
            terminal_ewma = _EWMA_BETA * float(abs_errors[tau]) + (1.0 - _EWMA_BETA) * terminal_ewma
    terminal_vol_scale = (
        float(np.clip(terminal_ewma / max(base_sigma, _EPS), 0.25, 25.0)) if normalized else 1.0
    )

    # Cumulative / rolling coverage
    rolling_static = (
        pd.Series(static_cov_Point, dtype=float)
        .expanding(min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    rolling_aci = (
        pd.Series(aci_cov_point, dtype=float).expanding(min_periods=1).mean().to_numpy(dtype=float)
    )
    static_width = static_upper - static_lower
    aci_width = aci_upper - aci_lower
    width_ratio = aci_width / np.maximum(static_width, _EPS)

    ds_vals = df[date_col].to_numpy() if date_col in df.columns else np.arange(n, dtype=int)
    fold_ids = df["fold_id"].to_numpy() if "fold_id" in df.columns else np.zeros(n, dtype=int)

    out_df = pd.DataFrame(
        {
            "ts_id": tid,
            "model_type": mtype,
            "ds": ds_vals,
            "step_idx": np.arange(n, dtype=int),
            "fold_id": fold_ids,
            "horizon_step": h_steps,
            "y_true": y_true,
            "yhat": yhat,
            "abs_error": abs_errors,
            "volatility_scale": vol_scale_arr,
            "local_scale": vol_scale_arr,
            "alpha_t": alpha_traj,
            "err_t": err_traj,
            "static_lower": static_lower,
            "static_upper": static_upper,
            "aci_lower": aci_lower,
            "aci_upper": aci_upper,
            "lower_split": static_lower,
            "upper_split": static_upper,
            "lower_aci": aci_lower,
            "upper_aci": aci_upper,
            "static_covered": static_cov_Point.astype(bool),
            "aci_covered": aci_cov_point.astype(bool),
            "covered_split": static_cov_Point.astype(bool),
            "covered_aci": aci_cov_point.astype(bool),
            "static_coverage": rolling_static,
            "aci_coverage": rolling_aci,
            "rolling_coverage_split": rolling_static,
            "rolling_coverage_aci": rolling_aci,
            "target_coverage": np.full(n, target_cov, dtype=float),
            "static_width": static_width,
            "aci_width": aci_width,
            "interval_width_ratio": width_ratio,
            "drift_detected": drift_flags,
        }
    )

    valid_res = signed_resids[np.isfinite(signed_resids)]
    terminal_scale_mult = _normal_quantile_ratio(terminal_alpha, alpha_target) * terminal_vol_scale
    q80_flat = (
        float(np.quantile(np.abs(valid_res), target_cov)) * terminal_scale_mult
        if valid_res.size > 0
        else float(np.mean(aci_width) * 0.5)
    )

    state = AciCalibrationState(
        method=method_name,
        target_alpha=alpha_target,
        gamma=gamma_val,
        terminal_alpha=round(float(terminal_alpha), 6),
        empirical_coverage_split=round(float(np.mean(static_cov_Point)), 6),
        empirical_coverage_aci=round(float(np.mean(aci_cov_point)), 6),
        mean_width_split=round(float(np.mean(static_width)), 6),
        mean_width_aci=round(float(np.mean(aci_width)), 6),
        volatility_scale=round(float(terminal_vol_scale), 6),
        drift_detected=bool(np.any(drift_flags)),
        alpha_trajectory=tuple(round(float(x), 6) for x in alpha_traj),
        err_trajectory=tuple(int(x) for x in err_traj),
        q80_per_step={1: round(q80_flat, 6)},
        q80_flat=round(q80_flat, 6),
        quantile_offsets_per_step={},
        quantile_offsets_flat={},
    )
    return out_df, state


def compute_aci_df(
    oof_or_actuals_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    *,
    gamma: float = 0.05,
    target_coverage: float = 0.80,
    normalized: bool = True,
    method: CalibrationMethod | None = None,
    horizon_delay: bool = True,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
) -> pd.DataFrame:
    """Compute online Adaptive Conformal Inference (`ACI`) trajectories and coverage diagnostics.

    Accepts either:
    - An out-of-fold DataFrame containing ``y_true`` and ``yhat`` (or ``yhat_raw``), or
    - Newly arrived actuals ``oof_or_actuals_df`` (with ``y`` or ``y_true``) paired with
      ``predictions_df`` (with ``yhat``).
    """
    use_normalized = (method == "aci_normalized") if method is not None else bool(normalized)

    columns = [
        "ts_id",
        "model_type",
        "ds",
        "step_idx",
        "fold_id",
        "horizon_step",
        "y_true",
        "yhat",
        "abs_error",
        "volatility_scale",
        "local_scale",
        "alpha_t",
        "err_t",
        "static_lower",
        "static_upper",
        "aci_lower",
        "aci_upper",
        "lower_split",
        "upper_split",
        "lower_aci",
        "upper_aci",
        "static_covered",
        "aci_covered",
        "covered_split",
        "covered_aci",
        "static_coverage",
        "aci_coverage",
        "rolling_coverage_split",
        "rolling_coverage_aci",
        "target_coverage",
        "static_width",
        "aci_width",
        "interval_width_ratio",
        "drift_detected",
    ]
    if oof_or_actuals_df is None or oof_or_actuals_df.empty:
        return pd.DataFrame(columns=columns)

    df = oof_or_actuals_df.copy()
    dcol = (
        date_col
        if date_col in df.columns
        else ("forecast_date" if "forecast_date" in df.columns else "ds")
    )

    # If predictions_df is provided separately, join actuals + predictions
    if predictions_df is not None and not predictions_df.empty and "yhat" not in df.columns:
        p_df = predictions_df.copy()
        p_dcol = (
            dcol
            if dcol in p_df.columns
            else ("forecast_date" if "forecast_date" in p_df.columns else "ds")
        )
        if p_dcol != dcol and p_dcol in p_df.columns:
            p_df = p_df.rename(columns={p_dcol: dcol})
        join_cols = [c for c in (ts_id_col, dcol) if c in df.columns and c in p_df.columns]
        if join_cols:
            df = pd.merge(df, p_df, on=join_cols, how="inner", suffixes=("_actual", ""))
            if "y_true" not in df.columns and "y" in df.columns:
                df["y_true"] = df["y"]

    if "y_true" not in df.columns and "y" in df.columns:
        df["y_true"] = df["y"]
    if "yhat" not in df.columns and "yhat_raw" in df.columns:
        df["yhat"] = df["yhat_raw"]

    if "y_true" not in df.columns or "yhat" not in df.columns:
        return pd.DataFrame(columns=columns)

    group_cols = [c for c in (ts_id_col, "model_type") if c in df.columns]
    parts: list[pd.DataFrame] = []
    states: dict[str, AciCalibrationState] = {}

    if group_cols:
        for g_key, grp in df.groupby(group_cols, sort=False):
            part_df, st = _run_single_series_aci(
                grp,
                gamma=gamma,
                target_coverage=target_coverage,
                normalized=use_normalized,
                horizon_delay=horizon_delay,
                ts_id_col=ts_id_col,
                date_col=dcol,
            )
            if not part_df.empty:
                parts.append(part_df)
                key_str = (
                    "|".join(str(x) for x in g_key) if isinstance(g_key, tuple) else str(g_key)
                )
                states[key_str] = st
    else:
        part_df, st = _run_single_series_aci(
            df,
            gamma=gamma,
            target_coverage=target_coverage,
            normalized=use_normalized,
            horizon_delay=horizon_delay,
            ts_id_col=ts_id_col,
            date_col=dcol,
        )
        if not part_df.empty:
            parts.append(part_df)
            states["series_0"] = st

    if not parts:
        return pd.DataFrame(columns=columns)

    res_df = pd.concat(parts, ignore_index=True)
    res_df.attrs["aci_states"] = states
    return res_df


aci_df = compute_aci_df
aci_trajectory_df = compute_aci_df
update_aci_online = compute_aci_df
aci_report_df = compute_aci_df


def _scale_step_calibration(
    step_cal: StepCalibration,
    scale_mult: float,
    quantiles: Sequence[float],
) -> StepCalibration:
    """Scale residual quantile offsets around the median by ``scale_mult``."""
    med = step_cal.quantiles.get(0.5, 0.0)
    raw_qmap: dict[float, float] = {}
    for q in quantiles:
        qf = float(q)
        base_q = step_cal.quantiles.get(qf, med)
        if qf == 0.5:
            raw_qmap[qf] = med
        else:
            raw_qmap[qf] = med + (base_q - med) * scale_mult
    mono_qmap = rearrange_monotone_quantiles(raw_qmap)
    return StepCalibration(
        quantiles=mono_qmap,
        mean=step_cal.mean,
        n=step_cal.n,
    )


def calibrate_aci_from_oof(
    oof: pd.DataFrame,
    *,
    method: CalibrationMethod = "aci",
    gamma: float = 0.05,
    target_coverage: float = 0.80,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    residual_column: str = "yhat_raw",
) -> tuple[Calibration | None, AciCalibrationState | None]:
    """Learn an online ACI-adjusted `Calibration` and `AciCalibrationState` from an OOF frame."""
    if oof is None or oof.empty:
        return None, None
    needed = {"y_true", residual_column}
    if not needed.issubset(oof.columns):
        return None, None

    y_true = pd.to_numeric(oof["y_true"], errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(oof[residual_column], errors="coerce").to_numpy(dtype=float)
    resid = y_true - y_raw
    ok = np.isfinite(resid)
    if int(np.count_nonzero(ok)) < MIN_RESIDUALS_TOTAL:
        return None, None

    # Run online ACI across OOF to obtain terminal alpha_T and volatility scale
    oof_for_aci = oof.copy()
    if "yhat" not in oof_for_aci.columns:
        oof_for_aci["yhat"] = oof_for_aci[residual_column]
    _, aci_state = _run_single_series_aci(
        oof_for_aci,
        gamma=gamma,
        target_coverage=target_coverage,
        normalized=(method == "aci_normalized"),
        horizon_delay=True,
        ts_id_col="ts_id" if "ts_id" in oof_for_aci.columns else "fold_id",
        date_col="ds" if "ds" in oof_for_aci.columns else "cutoff_date",
    )

    # Compute effective scale multiplier from terminal_alpha and volatility_scale
    z_ratio = _normal_quantile_ratio(aci_state.terminal_alpha, aci_state.target_alpha)
    scale_mult = float(np.clip(z_ratio * aci_state.volatility_scale, 0.35, 15.0))

    q_seq = tuple(sorted({round(float(q), 6) for q in quantiles} | {0.1, 0.5, 0.9}))
    ordered_all = np.sort(resid[ok])
    base_pooled = StepCalibration(
        quantiles={float(q): empirical_quantile(ordered_all, float(q)) for q in q_seq},
        mean=float(np.mean(resid[ok])),
        n=int(ordered_all.size),
    )
    scaled_pooled = _scale_step_calibration(base_pooled, scale_mult, q_seq)

    steps_raw = (
        pd.to_numeric(oof["horizon_step"], errors="coerce").to_numpy(dtype=float)
        if "horizon_step" in oof.columns
        else None
    )
    src_prefix = "aci-normalized" if method == "aci_normalized" else "aci"

    if steps_raw is None or not bool(np.any(ok & np.isfinite(steps_raw))):
        cal = Calibration(by_step={}, pooled=scaled_pooled, source=f"{src_prefix}-flat")
        return cal, aci_state

    have_steps = ok & np.isfinite(steps_raw)
    valid_resid = resid[have_steps]
    valid_steps = steps_raw[have_steps].astype(int)
    unique_steps = np.unique(valid_steps)
    grouped: dict[int, np.ndarray] = {int(s): valid_resid[valid_steps == s] for s in unique_steps}
    steps = [int(s) for s in unique_steps]
    by_step: dict[int, StepCalibration] = {}
    for step in steps:
        sample = window_residuals(grouped, step, steps)
        if sample.size:
            ord_s = np.sort(sample)
            base_s = StepCalibration(
                quantiles={float(q): empirical_quantile(ord_s, float(q)) for q in q_seq},
                mean=float(np.mean(sample)),
                n=int(ord_s.size),
            )
            by_step[step] = _scale_step_calibration(base_s, scale_mult, q_seq)

    distinct = {(*sorted(s.quantiles.items()), s.mean) for s in by_step.values()}
    if not by_step or len(distinct) == 1:
        cal = Calibration(by_step={}, pooled=scaled_pooled, source=f"{src_prefix}-flat")
    else:
        cal = Calibration(by_step=by_step, pooled=scaled_pooled, source=f"{src_prefix}-per-step")
    return cal, aci_state


def plot_aci(
    aci_df_in: pd.DataFrame,
    *,
    ts_id: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot static conformal vs. online ACI intervals, rolling coverage, and drift events."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    if aci_df_in is None or aci_df_in.empty:
        ax.set_title("Online Adaptive Conformal Inference (Empty)")
        return ax

    df = aci_df_in
    if "aci_lower" not in df.columns:
        df = compute_aci_df(df)

    if ts_id is not None and "ts_id" in df.columns:
        df = df[df["ts_id"].astype(str) == str(ts_id)]
    elif "ts_id" in df.columns and df["ts_id"].nunique() > 1:
        first_id = df["ts_id"].iloc[0]
        df = df[df["ts_id"] == first_id]

    if df.empty:
        ax.set_title("Online Adaptive Conformal Inference (No Matching Series)")
        return ax

    x = np.arange(len(df))
    y_true = df["y_true"].to_numpy(dtype=float)
    yhat = df["yhat"].to_numpy(dtype=float)
    s_lo = df["static_lower"].to_numpy(dtype=float)
    s_hi = df["static_upper"].to_numpy(dtype=float)
    a_lo = df["aci_lower"].to_numpy(dtype=float)
    a_hi = df["aci_upper"].to_numpy(dtype=float)

    ax.plot(x, y_true, "k.", markersize=4, label="Actual (y_true)")
    ax.plot(x, yhat, "b-", linewidth=1.3, label="Forecast (yhat)")
    ax.fill_between(x, s_lo, s_hi, color="gray", alpha=0.25, label="Static Conformal Band")
    ax.fill_between(x, a_lo, a_hi, color="tab:orange", alpha=0.25, label="Online ACI Band")

    s_cov = float(df["static_covered"].mean()) if "static_covered" in df.columns else float("nan")
    a_cov = float(df["aci_covered"].mean()) if "aci_covered" in df.columns else float("nan")
    tid_label = str(df["ts_id"].iloc[0]) if "ts_id" in df.columns else "series"
    ax.set_title(
        f"Online ACI vs. Static Conformal — {tid_label} "
        f"(Static Cov={s_cov:.1%}, ACI Cov={a_cov:.1%})"
    )
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    return ax
