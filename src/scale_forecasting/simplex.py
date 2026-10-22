"""Compositional and market-share simplex transforms (`clr`, `ilr`) and panel closure.

Implements Aitchison (1986) Centered Log-Ratio (`clr`) and Egozcue et al. (2003) Isometric
Log-Ratio (`ilr` via orthonormal Helmert contrast matrices) transforms with Martín-Fernández et al.
(2003) multiplicative zero replacement (`multiplicative_replacement`), single-series $D=1$ bounded
simplex fallbacks (`apply_series_simplex_transform`, `invert_series_simplex_transform`), and panel
closure (`enforce_simplex_closure`, `close_simplex_cells`, `forecast_simplex_panel`,
`verify_simplex_coherence`).

Pure offline module — zero Google Cloud imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import pandas as pd

from .errors import ConfigError, DataError

if TYPE_CHECKING:
    from .config import RunConfig
    from .worker import CellResult

SIMPLEX_TRANSFORMS: Final[tuple[str, ...]] = ("clr", "ilr")
DEFAULT_SIMPLEX_EPS: Final[float] = 1e-6
_LOGIT_CLIP: Final[float] = 250.0


@dataclass(frozen=True)
class SimplexSpec:
    """Metadata and orthonormal Helmert basis for a $D$-part compositional panel transform.

    Attributes:
        transform: Compositional transform type (``"clr"`` or ``"ilr"``).
        ts_ids: Ordered tuple of the $D$ component series identifiers.
        simplex_sum: Target cross-sectional simplex sum $S > 0$ (e.g. ``1.0`` or ``100.0``).
        eps: Multiplicative zero-replacement floor $\\epsilon \\in (0, 0.5)$.
        helmert_basis: Orthonormal Helmert contrast matrix $V \\in \\mathbb{R}^{D \\times (D-1)}$
            when $D \\ge 2$, else ``None``.
    """

    transform: Literal["clr", "ilr"]
    ts_ids: tuple[str, ...]
    simplex_sum: float = 1.0
    eps: float = DEFAULT_SIMPLEX_EPS
    helmert_basis: np.ndarray | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimplexSpec):
            return False
        if (
            self.transform != other.transform
            or self.ts_ids != other.ts_ids
            or self.simplex_sum != other.simplex_sum
            or self.eps != other.eps
        ):
            return False
        if self.helmert_basis is None and other.helmert_basis is None:
            return True
        if self.helmert_basis is None or other.helmert_basis is None:
            return False
        return np.array_equal(self.helmert_basis, other.helmert_basis)


def detect_simplex_sum(
    values: np.ndarray | pd.Series | pd.DataFrame | Sequence[float],
    simplex_sum: float | None = None,
) -> float:
    """Validate an explicit ``simplex_sum`` or auto-detect ``1.0`` vs ``100.0`` from ``values``."""
    if simplex_sum is not None:
        val = float(simplex_sum)
        if not np.isfinite(val) or val <= 0.0:
            raise ConfigError(f"simplex_sum must be finite and > 0, got {simplex_sum!r}")
        return val

    arr = (
        values.to_numpy(dtype=float)
        if isinstance(values, (pd.Series, pd.DataFrame))
        else np.asarray(values, dtype=float)
    )
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    if arr.ndim == 2 and arr.shape[1] >= 2:
        row_sums = np.nansum(arr, axis=1)
        pos_sums = row_sums[np.isfinite(row_sums) & (row_sums > 0.0)]
        if pos_sums.size > 0:
            med = float(np.median(pos_sums))
            if abs(med - 100.0) < abs(med - 1.0) or float(np.max(finite)) > 1.0 + 1e-6:
                return 100.0
            return 1.0
    return 100.0 if float(np.max(finite)) > 1.0 + 1e-6 else 1.0


def helmert_contrast_matrix(d: int) -> np.ndarray:
    """Construct the orthonormal Helmert contrast matrix $V \\in \\mathbb{R}^{d \\times (d - 1)}$.

    For $d \\ge 2$, column $j \\in \\{0, \\dots, d - 2\\}$ has entries:

    - $V_{0:j+1, j} = \\frac{1}{\\sqrt{(j+1)(j+2)}}$
    - $V_{j+1, j} = -\\frac{j+1}{\\sqrt{(j+1)(j+2)}}$
    - $V_{j+2:, j} = 0$

    Satisfies $V^\\top V = I_{d-1}$,
    $V V^\\top = I_d - \\frac{1}{d}\\mathbf{1}_d\\mathbf{1}_d^\\top$,
    and $\\mathbf{1}_d^\\top V = \\mathbf{0}_{d-1}^\\top$.
    """
    if int(d) < 2:
        raise ConfigError(f"helmert_contrast_matrix requires d >= 2, got {d}")
    d_int = int(d)
    v_mat = np.zeros((d_int, d_int - 1), dtype=float)
    for j in range(d_int - 1):
        scale = 1.0 / np.sqrt(float((j + 1) * (j + 2)))
        v_mat[: j + 1, j] = scale
        v_mat[j + 1, j] = -float(j + 1) * scale
    return v_mat


def _validate_nonnegative(arr: np.ndarray, context: str = "simplex transform") -> None:
    finite = arr[np.isfinite(arr)]
    if finite.size > 0 and float(np.min(finite)) < 0.0:
        min_v = float(np.min(finite))
        raise ConfigError(
            f"{context} requires non-negative target values (y >= 0); got min={min_v:.6g}"
        )


def multiplicative_replacement(
    values: np.ndarray | pd.Series | Sequence[float],
    *,
    eps: float = DEFAULT_SIMPLEX_EPS,
    simplex_sum: float = 1.0,
) -> np.ndarray:
    """Apply Martín-Fernández et al. (2003) multiplicative zero replacement on the simplex.

    Replaces zero (or sub-``eps * simplex_sum``) components with $\\delta = \\text{eps} \\cdot S$
    and rescales positive components proportionally so every component is strictly positive and
    every composition sums exactly to ``simplex_sum``.
    """
    s_val = detect_simplex_sum(values, simplex_sum=simplex_sum)
    eps_f = float(eps)
    if not np.isfinite(eps_f) or eps_f <= 0.0 or eps_f >= 0.5:
        raise ConfigError(f"eps must lie in (0, 0.5), got {eps!r}")

    arr = (
        values.to_numpy(dtype=float)
        if isinstance(values, pd.Series)
        else np.asarray(values, dtype=float)
    )
    _validate_nonnegative(arr, "multiplicative_replacement")

    was_1d = arr.ndim == 1
    if was_1d:
        if arr.size == 0:
            return arr.copy()
        if arr.size == 1:
            v0 = float(arr[0]) if np.isfinite(arr[0]) else 0.5 * s_val
            return np.array([np.clip(v0, eps_f * s_val, (1.0 - eps_f) * s_val)], dtype=float)
        mat = arr[np.newaxis, :].copy()
    elif arr.ndim == 2:
        mat = arr.copy()
    else:
        raise ConfigError(
            f"multiplicative_replacement expects a 1-D or 2-D array, got shape {arr.shape}"
        )

    n_rows, d_cols = mat.shape
    if d_cols == 1:
        clean = np.where(np.isfinite(mat), mat, 0.5 * s_val)
        out = np.clip(clean, eps_f * s_val, (1.0 - eps_f) * s_val)
        return out[0] if was_1d else out

    out = np.empty_like(mat, dtype=float)
    delta = min(eps_f * s_val, s_val / float(2 * d_cols))
    for r_idx in range(n_rows):
        row = mat[r_idx]
        row = np.where(np.isfinite(row), np.maximum(row, 0.0), 0.0)
        row_total = float(np.sum(row))
        if row_total <= 0.0:
            out[r_idx] = np.full(d_cols, s_val / float(d_cols), dtype=float)
            continue
        # Close raw positive values to simplex_sum first.
        closed = row * (s_val / row_total)
        zero_mask = closed < delta
        n_zeros = int(np.sum(zero_mask))
        if n_zeros == 0:
            out[r_idx] = closed
        elif n_zeros >= d_cols:
            out[r_idx] = np.full(d_cols, s_val / float(d_cols), dtype=float)
        else:
            pos_mask = ~zero_mask
            pos_sum = float(np.sum(closed[pos_mask]))
            rem_mass = s_val - float(n_zeros) * delta
            replaced = np.empty(d_cols, dtype=float)
            replaced[zero_mask] = delta
            replaced[pos_mask] = closed[pos_mask] * (rem_mass / pos_sum)
            # Final exact closure guard.
            replaced *= s_val / float(np.sum(replaced))
            out[r_idx] = replaced

    return out[0] if was_1d else out


multiplicative_zero_replacement = multiplicative_replacement


def clr_transform(
    values: np.ndarray | pd.Series | Sequence[float],
    *,
    simplex_sum: float | None = None,
    eps: float = DEFAULT_SIMPLEX_EPS,
) -> np.ndarray:
    """Compute the Aitchison Centered Log-Ratio (`clr`) forward transform.

    For a $D$-part composition ($D \\ge 2$) after multiplicative zero replacement:

    $$\\text{clr}(p)_{t, i} = \\ln p_{t, i} - \\frac{1}{D}\\sum_{j=1}^D \\ln p_{t, j}$$

    Satisfies $\\sum_{i=1}^D \\text{clr}(p)_{t, i} = 0$. When $D = 1$ (a `(T, 1)` single-series
    column), computes the 2-part bounded simplex log-ratio $\\frac{1}{2}\\ln(p_t / (1 - p_t))$.
    """
    arr = (
        values.to_numpy(dtype=float)
        if isinstance(values, pd.Series)
        else np.asarray(values, dtype=float)
    )
    _validate_nonnegative(arr, "clr_transform")
    s_val = detect_simplex_sum(arr, simplex_sum=simplex_sum)

    if arr.ndim == 2 and arr.shape[1] == 1:
        col_1d = apply_series_simplex_transform(
            pd.Series(arr[:, 0]), "clr", simplex_sum=s_val, eps=eps
        )
        return col_1d.to_numpy(dtype=float)[:, np.newaxis]

    reg = multiplicative_replacement(arr, eps=eps, simplex_sum=s_val)
    if reg.ndim == 1 and reg.size == 1:
        p = float(np.clip(reg[0] / s_val, eps, 1.0 - eps))
        return np.array([0.5 * np.log(p / (1.0 - p))], dtype=float)

    log_p = np.log(reg / s_val)
    mean_log = np.mean(log_p, axis=-1, keepdims=True)
    return np.asarray(log_p - mean_log, dtype=float)


clr_forward = clr_transform


def clr_inverse(
    z: np.ndarray | pd.Series | Sequence[float],
    *,
    simplex_sum: float = 1.0,
) -> np.ndarray:
    """Invert Centered Log-Ratio (`clr`) coordinates back to the simplex $\\mathcal{S}^D(S)$.

    Applies the numerically stabilized softmax closure:
    $\\hat{y}_{t, i} = S \\cdot \\exp(z_{t, i} - m_t) / \\sum_{j=1}^D \\exp(z_{t, j} - m_t)$
    where $m_t = \\max_k z_{t, k}$.
    """
    s_val = detect_simplex_sum(np.array([1.0]), simplex_sum=simplex_sum)
    arr = z.to_numpy(dtype=float) if isinstance(z, pd.Series) else np.asarray(z, dtype=float)
    clean = np.where(np.isfinite(arr), arr, 0.0)

    if (clean.ndim == 2 and clean.shape[1] == 1) or (clean.ndim == 1 and clean.size == 1):
        inv_1d = invert_series_simplex_transform(clean.reshape(-1), "clr", simplex_sum=s_val)
        return inv_1d.reshape(clean.shape)

    shifted = np.clip(clean - np.max(clean, axis=-1, keepdims=True), -_LOGIT_CLIP, 0.0)
    exp_z = np.exp(shifted)
    denom = np.sum(exp_z, axis=-1, keepdims=True)
    return np.asarray(s_val * (exp_z / denom), dtype=float)


def ilr_transform(
    values: np.ndarray | pd.Series | Sequence[float],
    *,
    simplex_sum: float | None = None,
    eps: float = DEFAULT_SIMPLEX_EPS,
    return_basis: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the Isometric Log-Ratio (`ilr`) forward transform via orthonormal Helmert basis.

    Maps a $D$-part composition matrix of shape ``(T, D)`` (or vector of shape ``(D,)``) onto
    unconstrained Euclidean coordinates
    $\\Psi = \\text{clr}(P) V \\in \\mathbb{R}^{T \\times (D-1)}$
    (or ``(D - 1,)``). When ``values`` has shape ``(T, 1)`` ($D = 1$ single-series column), falls
    back to the 2-part Helmert coordinate $\\frac{1}{\\sqrt{2}}\\ln(p_t / (1 - p_t))$ of shape
    ``(T, 1)``.
    """
    arr = (
        values.to_numpy(dtype=float)
        if isinstance(values, pd.Series)
        else np.asarray(values, dtype=float)
    )
    _validate_nonnegative(arr, "ilr_transform")
    s_val = detect_simplex_sum(arr, simplex_sum=simplex_sum)

    if arr.ndim == 2 and arr.shape[1] == 1:
        col_1d = apply_series_simplex_transform(
            pd.Series(arr[:, 0]), "ilr", simplex_sum=s_val, eps=eps
        )
        out_1d = col_1d.to_numpy(dtype=float)[:, np.newaxis]
        return (out_1d, np.ones((1, 1), dtype=float)) if return_basis else out_1d
    if arr.ndim == 1 and arr.size == 1:
        col_1d = apply_series_simplex_transform(pd.Series(arr), "ilr", simplex_sum=s_val, eps=eps)
        out_s = col_1d.to_numpy(dtype=float)
        return (out_s, np.ones((1, 1), dtype=float)) if return_basis else out_s

    z = clr_transform(arr, simplex_sum=s_val, eps=eps)
    d_cols = int(z.shape[-1])
    v_mat = helmert_contrast_matrix(d_cols)
    psi = np.asarray(z @ v_mat, dtype=float)
    return (psi, v_mat) if return_basis else psi


ilr_forward = ilr_transform


def ilr_inverse(
    psi: np.ndarray | pd.Series | Sequence[float],
    *,
    simplex_sum: float = 1.0,
    d: int | None = None,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    """Invert Isometric Log-Ratio (`ilr`) coordinates $\\Psi$ back to $\\mathcal{S}^D(S)$.

    Maps $\\Psi \\in \\mathbb{R}^{\\dots \\times (D-1)}$ to
    $Z = \\Psi V^\\top \\in \\mathbb{R}^{\\dots \\times D}$
    and closes onto the simplex via `clr_inverse`. When ``d=1`` is specified explicitly, inverts
    the single-series 2-part Helmert coordinate.
    """
    s_val = detect_simplex_sum(np.array([1.0]), simplex_sum=simplex_sum)
    arr = psi.to_numpy(dtype=float) if isinstance(psi, pd.Series) else np.asarray(psi, dtype=float)
    clean = np.where(np.isfinite(arr), arr, 0.0)

    if d == 1:
        inv_1d = invert_series_simplex_transform(clean.reshape(-1), "ilr", simplex_sum=s_val)
        return inv_1d.reshape(clean.shape)

    if basis is not None:
        v_mat = np.asarray(basis, dtype=float)
        d_orig = int(v_mat.shape[0])
    else:
        d_orig = int(d) if d is not None else (int(clean.shape[-1]) + 1)
        if d_orig < 2:
            inv_1d = invert_series_simplex_transform(clean.reshape(-1), "ilr", simplex_sum=s_val)
            return inv_1d.reshape(clean.shape)
        v_mat = helmert_contrast_matrix(d_orig)

    if clean.shape[-1] != d_orig - 1:
        raise ConfigError(
            f"ilr_inverse expected last dimension {d_orig - 1} for d={d_orig}, got {clean.shape}"
        )
    z = clean @ v_mat.T
    return clr_inverse(z, simplex_sum=s_val)


def apply_series_simplex_transform(
    y: pd.Series | np.ndarray,
    transform: str,
    *,
    simplex_sum: float | None = 1.0,
    eps: float = DEFAULT_SIMPLEX_EPS,
) -> pd.Series:
    """Apply 1-D single-series ``clr`` or ``ilr`` transform for per-cell execution.

    Models a single series $y_t \\in [0, S]$ as a 2-part composition $(p_t, 1 - p_t)$ where
    $p_t = \\text{clip}(y_t / S, \\epsilon, 1 - \\epsilon)$:

    - ``"clr"``: $z_t = \\frac{1}{2}\\ln\\left(\\frac{p_t}{1 - p_t}\\right)$
    - ``"ilr"``: $\\psi_t = \\frac{1}{\\sqrt{2}}\\ln\\left(\\frac{p_t}{1 - p_t}\\right)$
    """
    if transform not in SIMPLEX_TRANSFORMS:
        raise ConfigError(f"Unknown simplex transform {transform!r}; expected 'clr' or 'ilr'")
    series_in = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float))
    arr = series_in.to_numpy(dtype=float)
    _validate_nonnegative(arr, f"{transform} transform")

    s_val = detect_simplex_sum(arr, simplex_sum=simplex_sum)
    eps_f = float(eps)
    p = np.where(np.isfinite(arr), arr / s_val, 0.5)
    p_clipped = np.clip(p, eps_f, 1.0 - eps_f)
    logit = np.log(p_clipped / (1.0 - p_clipped))
    scale = 0.5 if transform == "clr" else (1.0 / np.sqrt(2.0))
    return pd.Series(scale * logit, index=series_in.index, name=series_in.name)


def invert_series_simplex_transform(
    values: np.ndarray | pd.Series | Sequence[float],
    transform: str,
    *,
    simplex_sum: float | None = 1.0,
) -> np.ndarray:
    """Invert 1-D single-series ``clr`` or ``ilr`` coordinates back into $(0, S)$."""
    if transform not in SIMPLEX_TRANSFORMS:
        raise ConfigError(f"Unknown simplex transform {transform!r}; expected 'clr' or 'ilr'")
    s_val = detect_simplex_sum(np.array([1.0]), simplex_sum=simplex_sum)
    arr = (
        values.to_numpy(dtype=float)
        if isinstance(values, pd.Series)
        else np.asarray(values, dtype=float)
    )
    inv_scale = 2.0 if transform == "clr" else float(np.sqrt(2.0))
    logit = np.clip(np.where(np.isfinite(arr), arr * inv_scale, 0.0), -_LOGIT_CLIP, _LOGIT_CLIP)
    p = 1.0 / (1.0 + np.exp(-logit))
    return np.asarray(s_val * p, dtype=float)


def enforce_simplex_closure(
    predictions_df: pd.DataFrame,
    *,
    simplex_sum: float | None = None,
    target_sum: float | None = None,
    value_col: str | None = None,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    eps: float = DEFAULT_SIMPLEX_EPS,
) -> pd.DataFrame:
    """Enforce exact cross-series simplex closure on a forecast or OOF DataFrame.

    For each ``(model_type, date)`` (or ``(scenario, model_type, date)`` / ``(fold_id, date)``)
    cross-section:
    - Guarantees strictly positive point forecasts $\\hat{y}_{i, t} > 0$.
    - Normalizes $\\sum_{i=1}^D \\hat{y}_{i, t} = \\text{simplex\\_sum}$ (when $D \\ge 2$) or bounds
      $\\hat{y}_{1, t} \\in (0, \\text{simplex\\_sum})$ (when $D = 1$).
    - Clips and orders quantile intervals so
      $0 \\le \\hat{y}_{\\text{lower}} \\le \\hat{y} \\le \\hat{y}_{\\text{upper}} \\le S$.
    """
    if predictions_df is None or predictions_df.empty:
        return predictions_df.copy() if predictions_df is not None else pd.DataFrame()

    df = predictions_df.copy()
    dcol = (
        date_col
        if date_col in df.columns
        else ("forecast_date" if "forecast_date" in df.columns else "ds")
    )
    val_col = (
        value_col
        if (value_col is not None and value_col in df.columns)
        else ("yhat" if "yhat" in df.columns else ("y" if "y" in df.columns else None))
    )
    if val_col is None or dcol not in df.columns:
        return df

    eff_sum = simplex_sum if simplex_sum is not None else target_sum
    s_val = detect_simplex_sum(df[val_col].to_numpy(dtype=float), simplex_sum=eff_sum)
    eps_floor = max(float(eps) * s_val, 1e-12 * s_val)

    group_cols: list[str] = []
    for cand in ("scenario", "model_type", "fold_id", "cutoff_date", dcol):
        if cand in df.columns and cand not in group_cols:
            group_cols.append(cand)

    parts: list[pd.DataFrame] = []
    for _key, grp in df.groupby(group_cols, sort=False, dropna=False):
        sub = grp.copy()
        yhat_orig = sub[val_col].to_numpy(dtype=float)
        yhat_pos = np.where(np.isfinite(yhat_orig), np.maximum(yhat_orig, eps_floor), eps_floor)
        n_rows = len(sub)

        if n_rows >= 2:
            total = float(np.sum(yhat_pos))
            if total <= 0.0 or not np.isfinite(total):
                yhat_closed = np.full(n_rows, s_val / float(n_rows), dtype=float)
                scale_factors = np.ones(n_rows, dtype=float)
            else:
                yhat_closed = yhat_pos * (s_val / total)
                # Ensure strict positivity after scaling and re-close.
                if np.any(yhat_closed <= 0.0):
                    yhat_closed = np.maximum(yhat_closed, eps_floor)
                    yhat_closed *= s_val / float(np.sum(yhat_closed))
                scale_factors = np.where(
                    np.abs(yhat_orig) > 1e-12,
                    yhat_closed / np.maximum(yhat_orig, 1e-12),
                    s_val / total,
                )
        else:
            yhat_closed = np.clip(yhat_pos, eps_floor, s_val - eps_floor)
            scale_factors = np.where(
                np.abs(yhat_orig) > 1e-12,
                yhat_closed / np.maximum(yhat_orig, 1e-12),
                1.0,
            )

        sub[val_col] = yhat_closed
        for aux_col in ("yhat_raw", "yhat_adjusted"):
            if aux_col in sub.columns and sub[aux_col].notna().any():
                aux_vals = sub[aux_col].to_numpy(dtype=float)
                aux_pos = np.where(
                    np.isfinite(aux_vals), np.maximum(aux_vals, eps_floor), yhat_closed
                )
                if n_rows >= 2:
                    aux_tot = float(np.sum(aux_pos))
                    sub[aux_col] = (
                        aux_pos * (s_val / aux_tot) if aux_tot > 0.0 else yhat_closed.copy()
                    )
                else:
                    sub[aux_col] = np.clip(aux_pos, eps_floor, s_val - eps_floor)

        if "yhat_lower" in sub.columns and "yhat_upper" in sub.columns:
            lo_raw = sub["yhat_lower"].to_numpy(dtype=float)
            hi_raw = sub["yhat_upper"].to_numpy(dtype=float)
            lo_scaled = np.where(np.isfinite(lo_raw), lo_raw * scale_factors, yhat_closed)
            hi_scaled = np.where(np.isfinite(hi_raw), hi_raw * scale_factors, yhat_closed)
            lo_clamped = np.clip(np.minimum(lo_scaled, yhat_closed), 0.0, s_val)
            hi_clamped = np.clip(np.maximum(hi_scaled, yhat_closed), 0.0, s_val)
            sub["yhat_lower"] = np.minimum(lo_clamped, yhat_closed)
            sub["yhat_upper"] = np.maximum(hi_clamped, yhat_closed)

        # Also bound and order any explicit quantile columns (`yhat_p10`, `yhat_lower_80`, etc.)
        # or per-row `quantiles` dicts.
        q_cols = [
            c
            for c in sub.columns
            if c.startswith(("yhat_p", "yhat_lower_", "yhat_upper_"))
            and pd.api.types.is_numeric_dtype(sub[c])
        ]
        for qc in q_cols:
            q_vals = sub[qc].to_numpy(dtype=float) * scale_factors
            q_clamped = np.clip(q_vals, 0.0, s_val)
            if "lower" in qc:
                q_clamped = np.minimum(q_clamped, yhat_closed)
            elif "upper" in qc:
                q_clamped = np.maximum(q_clamped, yhat_closed)
            sub[qc] = q_clamped

        if "quantiles" in sub.columns:
            new_q_dicts: list[dict[str, float] | None] = []
            for idx_r, q_item in enumerate(sub["quantiles"].tolist()):
                if not isinstance(q_item, Mapping) or not q_item:
                    new_q_dicts.append(q_item if isinstance(q_item, dict) else None)
                    continue
                sf = float(scale_factors[idx_r])
                yh_r = float(yhat_closed[idx_r])
                parsed_pairs: list[tuple[float, str, float]] = []
                for k_str, v_num in q_item.items():
                    try:
                        q_lvl = float(k_str)
                        v_scaled = float(np.clip(float(v_num) * sf, 0.0, s_val))
                        parsed_pairs.append((q_lvl, str(k_str), v_scaled))
                    except (TypeError, ValueError):
                        continue
                parsed_pairs.sort(key=lambda t: t[0])
                # Enforce monotone non-decreasing quantiles anchored around yhat at 0.5.
                vals_sorted = np.maximum.accumulate([p[2] for p in parsed_pairs])
                vals_sorted = np.clip(vals_sorted, 0.0, s_val)
                rebuilt: dict[str, float] = {}
                for (q_lvl, k_orig, _), v_mon in zip(parsed_pairs, vals_sorted, strict=True):
                    if q_lvl < 0.5:
                        rebuilt[k_orig] = float(min(v_mon, yh_r))
                    elif q_lvl > 0.5:
                        rebuilt[k_orig] = float(max(v_mon, yh_r))
                    else:
                        rebuilt[k_orig] = yh_r
                new_q_dicts.append(rebuilt)
            sub["quantiles"] = pd.Series(new_q_dicts, index=sub.index, dtype=object)

        parts.append(sub)

    return pd.concat(parts, ignore_index=True) if parts else df


close_simplex_forecasts = enforce_simplex_closure


def has_simplex_models(
    cfg: RunConfig,
    models: Sequence[str] | None = None,
) -> bool:
    """Return True when ``cfg`` or any model in ``models`` enables compositional simplex closure."""
    if cfg.features.transform in SIMPLEX_TRANSFORMS or cfg.features.simplex_sum is not None:
        return True
    model_list = list(models) if models is not None else list(cfg.models)
    for mname in model_list:
        m_low = str(mname).lower()
        if any(
            tok in m_low
            for tok in ("@transform_clr", "@transform_ilr", "@clr", "@ilr", "simplex_sum=")
        ):
            return True
        raw_params = cfg.model_params.get(mname, {})
        if isinstance(raw_params, Mapping):
            t_val = raw_params.get("transform")
            if isinstance(t_val, str) and t_val.lower() in SIMPLEX_TRANSFORMS:
                return True
            if raw_params.get("simplex_sum") is not None:
                return True
    return False


def close_simplex_cells(
    results: Sequence[CellResult],
    cfg: RunConfig,
    *,
    history_df: pd.DataFrame | None = None,
) -> list[CellResult]:
    """Apply cross-series simplex closure to `CellResult`s (`clr`/`ilr` or `simplex_sum`)."""
    from .features import resolve_model_variant
    from .metrics import compute_metrics
    from .seasonality import seasonal_period

    if not results:
        return []

    id_col = cfg.data.ts_id_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col
    period = seasonal_period(cfg.data.freq)

    y_train_by_id: dict[str, np.ndarray] = {}
    if history_df is not None and not history_df.empty and id_col in history_df.columns:
        for tid, grp in history_df.groupby(id_col, sort=False):
            y_train_by_id[str(tid)] = grp.sort_values(date_col)[target_col].to_numpy(dtype=float)

    by_model: dict[str, list[tuple[int, CellResult]]] = {}
    for idx, res in enumerate(results):
        if res.status == "ok":
            by_model.setdefault(res.model_type, []).append((idx, res))

    updated = list(results)
    for model_name, indexed_cells in by_model.items():
        try:
            spec = resolve_model_variant(cfg, model_name)
            eff_f = spec.effective_cfg.features
        except Exception:  # noqa: BLE001 - reconciled or ensemble model names may not resolve as base variants
            eff_f = cfg.features

        if eff_f.transform not in SIMPLEX_TRANSFORMS and eff_f.simplex_sum is None:
            continue

        pred_frames: list[pd.DataFrame] = []
        oof_frames: list[pd.DataFrame] = []
        for _, cell in indexed_cells:
            if not cell.predictions.empty:
                pf = cell.predictions.copy()
                pf[id_col] = cell.ts_id
                pf["model_type"] = model_name
                pred_frames.append(pf)
            if cell.oof is not None and not cell.oof.empty:
                of = cell.oof.copy()
                of[id_col] = cell.ts_id
                of["model_type"] = model_name
                oof_frames.append(of)

        if not pred_frames:
            continue

        all_preds = pd.concat(pred_frames, ignore_index=True)
        s_val = detect_simplex_sum(
            all_preds["yhat"].to_numpy(dtype=float),
            simplex_sum=eff_f.simplex_sum,
        )
        closed_preds = enforce_simplex_closure(
            all_preds, simplex_sum=s_val, ts_id_col=id_col, date_col="ds"
        )
        preds_by_id = {
            str(tid): grp.drop(columns=[id_col, "model_type"], errors="ignore").reset_index(
                drop=True
            )
            for tid, grp in closed_preds.groupby(id_col, sort=False)
        }

        oof_by_id: dict[str, pd.DataFrame] = {}
        if oof_frames:
            all_oof = pd.concat(oof_frames, ignore_index=True)
            closed_oof = enforce_simplex_closure(
                all_oof, simplex_sum=s_val, ts_id_col=id_col, date_col="ds"
            )
            oof_by_id = {
                str(tid): grp.drop(columns=[id_col, "model_type"], errors="ignore").reset_index(
                    drop=True
                )
                for tid, grp in closed_oof.groupby(id_col, sort=False)
            }

        for orig_idx, cell in indexed_cells:
            new_pred = preds_by_id.get(cell.ts_id, cell.predictions)
            new_oof = oof_by_id.get(cell.ts_id, cell.oof)
            new_metrics = dict(cell.metrics)
            if (
                new_oof is not None
                and not new_oof.empty
                and {"y_true", "yhat"}.issubset(new_oof.columns)
            ):
                y_true = new_oof["y_true"].to_numpy(dtype=float)
                yhat = new_oof["yhat"].to_numpy(dtype=float)
                valid = np.isfinite(y_true) & np.isfinite(yhat)
                if np.any(valid):
                    lower = (
                        new_oof["yhat_lower"].to_numpy(dtype=float)[valid]
                        if "yhat_lower" in new_oof.columns
                        else None
                    )
                    upper = (
                        new_oof["yhat_upper"].to_numpy(dtype=float)[valid]
                        if "yhat_upper" in new_oof.columns
                        else None
                    )
                    new_metrics = compute_metrics(
                        y_true[valid],
                        yhat[valid],
                        lower=lower,
                        upper=upper,
                        y_train=y_train_by_id.get(cell.ts_id),
                        seasonal_period=period,
                    )
            updated[orig_idx] = replace(
                cell,
                predictions=new_pred,
                oof=new_oof,
                metrics=new_metrics,
            )

    return updated


def forecast_simplex_panel(
    panel_df: pd.DataFrame,
    cfg: RunConfig | None = None,
    *,
    model_name: str | None = None,
    transform: Literal["clr", "ilr"] = "ilr",
    simplex_sum: float | None = None,
    horizon: int | None = None,
    freq: str | None = None,
    ts_id_col: str | None = None,
    date_col: str | None = None,
    target_col: str | None = None,
    eps: float = DEFAULT_SIMPLEX_EPS,
    return_spec: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, SimplexSpec]:
    """Directly forecast a $D$-part compositional panel in ``clr`` or ``ilr`` coordinate space.

    Pivots the $D$ component series into a $T \\times D$ composition matrix, applies
    `multiplicative_replacement` and `clr_transform` ($D$ coordinates) or `ilr_transform`
    ($D - 1$ orthonormal Helmert coordinates), fits the base forecasting model to each unconstrained
    coordinate series, forecasts $H$ steps ahead, and maps back to $\\mathcal{S}^D(S)$ via
    `clr_inverse` / `ilr_inverse` with ordered prediction intervals in $[0, S]$.
    """
    from .models import get_model
    from .models.base_model import ModelContext

    if transform not in SIMPLEX_TRANSFORMS:
        raise ConfigError(f"transform must be 'clr' or 'ilr', got {transform!r}")
    if panel_df is None or panel_df.empty:
        raise DataError("Cannot forecast an empty compositional simplex panel")

    id_c = ts_id_col or (cfg.data.ts_id_col if cfg is not None else "ts_id")
    dt_c = date_col or (cfg.data.date_col if cfg is not None else "ds")
    tg_c = target_col or (cfg.data.target_col if cfg is not None else "y")
    h_steps = int(horizon if horizon is not None else (cfg.data.horizon if cfg is not None else 14))
    freq_str = str(freq if freq is not None else (cfg.data.freq if cfg is not None else "D"))
    m_name = str(
        model_name
        if model_name is not None
        else (cfg.models[0] if cfg is not None and cfg.models else "theta")
    )
    base_name = m_name.split("@", 1)[0].strip()

    missing = sorted({id_c, dt_c, tg_c} - set(panel_df.columns))
    if missing:
        raise DataError(f"Panel is missing required columns: {missing}")

    df = panel_df.copy()
    df[dt_c] = pd.to_datetime(df[dt_c])
    wide = df.pivot_table(index=dt_c, columns=id_c, values=tg_c, aggfunc="sum").sort_index()
    if (
        isinstance(wide.index, pd.DatetimeIndex)
        and wide.index.freq is None
        and len(wide.index) >= 3
    ):
        inferred = pd.infer_freq(wide.index)
        if inferred is not None:
            wide.index = pd.DatetimeIndex(wide.index, freq=inferred)
    ts_ids = [str(c) for c in wide.columns]
    d_parts = len(ts_ids)
    if d_parts == 0:
        raise DataError("Panel contains zero series")

    raw_mat = wide.ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    _validate_nonnegative(raw_mat, f"forecast_simplex_panel({transform})")
    s_val = detect_simplex_sum(raw_mat, simplex_sum=simplex_sum)

    model_cls = get_model(base_name)
    ctx = ModelContext(freq=freq_str, horizon=h_steps, seed=0)
    future_dates = pd.date_range(start=wide.index[-1], periods=h_steps + 1, freq=freq_str)[
        1:
    ].as_unit("ns")

    if d_parts == 1:
        y_1d = apply_series_simplex_transform(
            pd.Series(raw_mat[:, 0], index=wide.index),
            transform,
            simplex_sum=s_val,
            eps=eps,
        )
        est = model_cls({}, ctx)
        est.fit(y_1d, None)
        pred_1d = est.predict(h_steps, None)
        yhat = invert_series_simplex_transform(
            pred_1d["yhat"].to_numpy(dtype=float), transform, simplex_sum=s_val
        )
        lo = invert_series_simplex_transform(
            pred_1d["yhat_lower"].to_numpy(dtype=float), transform, simplex_sum=s_val
        )
        hi = invert_series_simplex_transform(
            pred_1d["yhat_upper"].to_numpy(dtype=float), transform, simplex_sum=s_val
        )
        out_1d = pd.DataFrame(
            {
                id_c: ts_ids[0],
                "model_type": m_name,
                "ds": future_dates,
                "yhat": yhat,
                "yhat_lower": np.minimum(lo, yhat),
                "yhat_upper": np.maximum(hi, yhat),
            }
        )
        spec_1d = SimplexSpec(
            transform=transform,
            ts_ids=tuple(ts_ids),
            simplex_sum=s_val,
            eps=eps,
            helmert_basis=None,
        )
        out_1d.attrs["simplex_spec"] = spec_1d
        return (out_1d, spec_1d) if return_spec else out_1d

    if transform == "clr":
        coords = clr_transform(raw_mat, simplex_sum=s_val, eps=eps)
        v_basis = None
    else:
        coords_res = ilr_transform(raw_mat, simplex_sum=s_val, eps=eps, return_basis=True)
        assert isinstance(coords_res, tuple)
        coords, v_basis = coords_res

    n_coords = int(coords.shape[1])
    pred_coords = np.zeros((h_steps, n_coords), dtype=float)
    lo_coords = np.zeros((h_steps, n_coords), dtype=float)
    hi_coords = np.zeros((h_steps, n_coords), dtype=float)

    for k in range(n_coords):
        y_coord = pd.Series(coords[:, k], index=wide.index, name="y")
        est = model_cls({}, ctx)
        est.fit(y_coord, None)
        p_df = est.predict(h_steps, None)
        pred_coords[:, k] = p_df["yhat"].to_numpy(dtype=float)[:h_steps]
        lo_coords[:, k] = p_df["yhat_lower"].to_numpy(dtype=float)[:h_steps]
        hi_coords[:, k] = p_df["yhat_upper"].to_numpy(dtype=float)[:h_steps]

    if transform == "clr":
        yhat_mat = clr_inverse(pred_coords, simplex_sum=s_val)
        z_half = 0.5 * np.abs(hi_coords - lo_coords)
    else:
        yhat_mat = ilr_inverse(pred_coords, simplex_sum=s_val, d=d_parts, basis=v_basis)
        v_mat = v_basis if v_basis is not None else helmert_contrast_matrix(d_parts)
        coord_var = (0.5 * (hi_coords - lo_coords)) ** 2
        z_half = np.sqrt(np.maximum(0.0, coord_var @ (v_mat.T**2)))

    # Project interval bounds around yhat_mat via delta-method / logistic scaling on [0, s_val].
    p_share = np.clip(yhat_mat / s_val, 1e-9, 1.0 - 1e-9)
    half_share = p_share * (1.0 - p_share) * np.minimum(z_half, 10.0)
    lo_mat = np.clip(yhat_mat - s_val * half_share, 0.0, yhat_mat)
    hi_mat = np.clip(yhat_mat + s_val * half_share, yhat_mat, s_val)

    rows: list[pd.DataFrame] = []
    for j, tid in enumerate(ts_ids):
        rows.append(
            pd.DataFrame(
                {
                    id_c: tid,
                    "model_type": m_name,
                    "ds": future_dates,
                    "yhat": yhat_mat[:, j],
                    "yhat_lower": lo_mat[:, j],
                    "yhat_upper": hi_mat[:, j],
                }
            )
        )
    out_df = pd.concat(rows, ignore_index=True)
    spec = SimplexSpec(
        transform=transform,
        ts_ids=tuple(ts_ids),
        simplex_sum=s_val,
        eps=eps,
        helmert_basis=v_basis,
    )
    out_df.attrs["simplex_spec"] = spec
    return (out_df, spec) if return_spec else out_df


def verify_simplex_coherence(
    predictions_df: pd.DataFrame,
    *,
    simplex_sum: float | None = None,
    value_col: str = "yhat",
    target_col: str | None = None,
    date_col: str | None = "ds",
    ts_id_col: str = "ts_id",
    atol: float = 1e-6,
    raise_on_error: bool = False,
) -> float:
    """Compute maximum simplex discrepancy across timestamps and check positivity/interval ordering.

    Returns:
        Maximum absolute discrepancy
        $$\\max\\left(\\max_{t, m}\\left|\\sum_i \\hat{y}_{i, t, m} - S\\right|,\\;
        \\max(0, -\\hat{y}),\\; \\max(0, \\hat{y}_{\\text{lower}} - \\hat{y}),\\;
        \\max(0, \\hat{y} - \\hat{y}_{\\text{upper}})\\right)$$
    """
    _ = ts_id_col
    if predictions_df is None or predictions_df.empty:
        return 0.0

    df = predictions_df
    if "segment" in df.columns:
        df = df[df["segment"] == "forecast"]
        if df.empty:
            return 0.0

    dcol = date_col if (date_col is not None and date_col in df.columns) else None
    if dcol is None:
        for cand in ("forecast_date", "ds", "date"):
            if cand in df.columns:
                dcol = cand
                break
    vcol = target_col if target_col is not None else value_col
    if vcol not in df.columns:
        for cand in ("yhat", "yhat_sample", "y"):
            if cand in df.columns:
                vcol = cand
                break
    if dcol is None or vcol not in df.columns or dcol not in df.columns:
        return 0.0

    vals_all = df[vcol].to_numpy(dtype=float)
    s_val = detect_simplex_sum(vals_all, simplex_sum=simplex_sum)

    group_cols: list[str] = []
    for cand in ("path_id", "scenario", "model_type", "fold_id", "cutoff_date", dcol):
        if cand in df.columns and cand not in group_cols:
            group_cols.append(cand)

    max_err = 0.0
    for _key, grp in df.groupby(group_cols, sort=False, dropna=False):
        v = grp[vcol].to_numpy(dtype=float)
        if not np.all(np.isfinite(v)):
            max_err = float("inf")
            break
        neg_err = float(np.max(np.maximum(0.0, -v)))
        over_err = float(np.max(np.maximum(0.0, v - s_val)))
        sum_err = abs(float(np.sum(v)) - s_val) if len(v) >= 2 else 0.0
        order_err = 0.0
        if "yhat_lower" in grp.columns and "yhat_upper" in grp.columns:
            lo = grp["yhat_lower"].to_numpy(dtype=float)
            hi = grp["yhat_upper"].to_numpy(dtype=float)
            order_err = float(
                max(
                    np.max(np.maximum(0.0, -lo)),
                    np.max(np.maximum(0.0, lo - v)),
                    np.max(np.maximum(0.0, v - hi)),
                    np.max(np.maximum(0.0, hi - s_val)),
                )
            )
        max_err = max(max_err, neg_err, over_err, sum_err, order_err)

    if raise_on_error and (not np.isfinite(max_err) or max_err > atol):
        raise DataError(
            f"Simplex coherence check failed: max discrepancy {max_err:.3e} > {atol:.3e}"
        )
    return float(max_err)


def verify_simplex(
    df: pd.DataFrame,
    target_col: str = "yhat",
    date_col: str | None = None,
    simplex_sum: float | None = None,
    *,
    value_col: str | None = None,
    ts_id_col: str = "ts_id",
    atol: float = 1e-6,
    raise_on_error: bool = False,
) -> float:
    """Verify compositional simplex closure across timestamps
    (alias for `verify_simplex_coherence`).
    """
    eff_val_col = value_col if value_col is not None else target_col
    return verify_simplex_coherence(
        df,
        simplex_sum=simplex_sum,
        value_col=eff_val_col,
        date_col=date_col,
        ts_id_col=ts_id_col,
        atol=atol,
        raise_on_error=raise_on_error,
    )


verify_simplex_closure = verify_simplex_coherence
replace_compositional_zeros = multiplicative_replacement
multiplicative_zero_replacement = multiplicative_replacement
inverse_clr_transform = clr_inverse
inverse_ilr_transform = ilr_inverse
apply_simplex_series_transform = apply_series_simplex_transform
inverse_simplex_series_transform = invert_series_simplex_transform
transform_simplex_series = apply_series_simplex_transform
inverse_transform_simplex_series = invert_series_simplex_transform
helmert_sub_matrix = helmert_contrast_matrix
should_close_simplex = has_simplex_models
close_simplex = enforce_simplex_closure
close_simplex_predictions = enforce_simplex_closure
DEFAULT_ZERO_DELTA = DEFAULT_SIMPLEX_EPS
SIMPLICIAL_TRANSFORMS: frozenset[str] = frozenset(SIMPLEX_TRANSFORMS)
