"""Top-Down Goal-Seek and Constrained Minimum-Trace Target Disaggregation (`allocate_target`).

Implements constrained quadratic programming (QP) target allocation across hierarchical and flat
time-series panels:

1. **Constrained Minimum-Trace QP (`allocate_target`)** — Given a top-level, regional, or
   per-period target override (in units or revenue dollars), solves the equality- and
   box-constrained Minimum-Trace quadratic program

   ``min_b  0.5 * sum_t (b_t - b_hat_t)^T W_t^{-1} (b_t - b_hat_t)``
   ``s.t.   A_eq b = c_target,   l <= b <= u``

   to distribute adjustments across bottom-level leaf series and horizon timestamps proportional
   to forecast error covariance ``W_t`` while respecting hard capacity/floor bounds ``[l, u]``
   and user-locked nodes (``locked_nodes`` / ``locks``).
2. **Pre- and Post-Solve Feasibility Verification (`GoalSeekInfeasibleError`)** — Analytically
   verifies box bounds, locked-node equality compatibility, and subset complement bounds before
   and after solving the QP, raising `GoalSeekInfeasibleError` (`ConfigError`, `ValueError`)
   with actionable diagnostics whenever constraints are mathematically infeasible.
3. **Scenario, Gross-Margin & Inventory Policy Chaining** — Accepts either standard forecast
   frames (``yhat``) or `simulate_scenarios` frames (``scenario_yhat``), computes revenue and
   gross-margin deltas when ``price_col`` / ``prices`` / ``unit_cost`` are supplied, and
   propagates allocated demand into `(s, S)` inventory policy deltas (``delta_reorder_point``,
   ``delta_safety_stock``, ``delta_order_qty``) when ``include_inventory_impact=True``.
4. **Horizon Summary & Visualization (`summarize_goal_seek`, `plot_goal_seek`)** — Rolls up
   per-series baseline vs. allocated units, revenue, gross margin, and constraint status across
   the horizon and renders a publication-grade horizontal bar/interval comparison chart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import RunConfig
from .errors import ConfigError, DataError, GoalSeekInfeasibleError
from .reconciliation import (
    TOTAL_NODE_ID,
    HierarchySpec,
    build_hierarchy,
    residual_matrix,
    schafer_strimmer_shrinkage,
)

_Z_80: Final[float] = 1.2815515655446004
_VAR_FLOOR: Final[float] = 1e-8
_FEAS_TOL: Final[float] = 1e-6

GOAL_SEEK_COLUMNS: Final[tuple[str, ...]] = (
    "ts_id",
    "model_type",
    "ds",
    "forecast_date",
    "horizon_step",
    "baseline_yhat",
    "allocated_yhat",
    "yhat",
    "yhat_lower",
    "yhat_upper",
    "delta_yhat",
    "delta_units",
    "delta_pct",
    "lower_bound",
    "upper_bound",
    "is_locked",
    "allocation_weight",
    "constraint_status",
)


@dataclass(frozen=True)
class GoalSeekResult:
    """Container for constrained top-down target allocation results and diagnostics."""

    allocations: pd.DataFrame
    summary: pd.DataFrame
    target_node: str
    target_total: float
    baseline_total: float
    allocated_total: float
    coherence_max_abs_error: float
    active_bound_count: int = 0
    locked_count: int = 0
    hierarchy_spec: HierarchySpec | None = None

    @property
    def max_coherence_error(self) -> float:
        """Alias for ``coherence_max_abs_error``."""
        return float(self.coherence_max_abs_error)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Alias for ``allocations`` DataFrame."""
        return self.allocations

    @property
    def max_target_discrepancy(self) -> float:
        """Absolute discrepancy between ``allocated_total`` and ``target_total``."""
        return abs(float(self.allocated_total) - float(self.target_total))

    @property
    def max_coherence_discrepancy(self) -> float:
        """Alias for ``coherence_max_abs_error``."""
        return float(self.coherence_max_abs_error)


def _build_flat_hierarchy_spec(
    bottom_ids: Sequence[str], total_id: str = TOTAL_NODE_ID
) -> HierarchySpec:
    b_tuple = tuple(str(x) for x in bottom_ids)
    n_b = len(b_tuple)
    s_mat = np.vstack([np.ones((1, n_b), dtype=float), np.eye(n_b, dtype=float)])
    return HierarchySpec(
        node_ids=(total_id, *b_tuple),
        bottom_ids=b_tuple,
        level_nodes={total_id: (total_id,), "__bottom__": b_tuple},
        summing_matrix=s_mat,
    )


def _infer_hierarchy_spec_from_frame(
    df: pd.DataFrame,
    *,
    hier_spec: HierarchySpec | None,
    cfg: RunConfig | Mapping[str, Any] | None,
    history_df: pd.DataFrame | None,
    ts_id_col: str,
    date_col: str,
) -> tuple[HierarchySpec, bool]:
    """Resolve `(HierarchySpec, include_agg_nodes_in_output)` from spec, config, or frame IDs."""
    if hier_spec is not None:
        return hier_spec, True

    parsed_cfg: RunConfig | None = None
    if isinstance(cfg, RunConfig):
        parsed_cfg = cfg
    elif isinstance(cfg, Mapping):
        try:
            parsed_cfg = RunConfig.model_validate(cfg)
        except Exception:  # noqa: BLE001 - fallback when partial dict is passed
            parsed_cfg = None

    if (
        parsed_cfg is not None
        and parsed_cfg.hierarchy.levels
        and history_df is not None
        and not history_df.empty
    ):
        needed = {c for lvl in parsed_cfg.hierarchy.levels for c in lvl}
        if needed.issubset(history_df.columns):
            _, built_spec = build_hierarchy(history_df, parsed_cfg)
            return built_spec, True

    unique_ids = [str(x) for x in df[ts_id_col].drop_duplicates().tolist()]
    has_total = TOTAL_NODE_ID in unique_ids
    inter_ids = [nid for nid in unique_ids if nid != TOTAL_NODE_ID and "=" in nid]
    bottom_ids = sorted([nid for nid in unique_ids if nid != TOTAL_NODE_ID and "=" not in nid])

    if not bottom_ids:
        # All IDs have '=' or only __total__ exists; treat non-total IDs as bottom leaves.
        bottom_ids = sorted([nid for nid in unique_ids if nid != TOTAL_NODE_ID])
        inter_ids = []
        if not bottom_ids and has_total:
            bottom_ids = [TOTAL_NODE_ID]
            has_total = False

    if not inter_ids:
        return _build_flat_hierarchy_spec(bottom_ids), has_total

    # Reconstruct S for intermediate nodes ("col=val") from attribute columns or baseline sums.
    n_b = len(bottom_ids)
    b_idx = {bid: i for i, bid in enumerate(bottom_ids)}
    s_rows: list[np.ndarray] = [np.ones(n_b, dtype=float)]
    node_ids: list[str] = [TOTAL_NODE_ID]
    level_nodes: dict[str, list[str]] = {TOTAL_NODE_ID: [TOTAL_NODE_ID]}

    # Check if attribute columns exist in history_df or df.
    ref_df = history_df if (history_df is not None and not history_df.empty) else df
    piv_yhat = (
        df.pivot_table(index=date_col, columns=ts_id_col, values="yhat", aggfunc="first")
        if "yhat" in df.columns and date_col in df.columns
        else None
    )

    for nid in inter_ids:
        row = np.zeros(n_b, dtype=float)
        parts = [p.split("=", 1) for p in nid.split("/") if "=" in p]
        lkey = "/".join(p[0] for p in parts) if parts else "level"
        matched = False
        if parts and all(col in ref_df.columns for col, _ in parts):
            mask = pd.Series(True, index=ref_df.index)
            for col, val in parts:
                mask = mask & (ref_df[col].astype(str) == val)
            members = ref_df.loc[mask, ts_id_col].astype(str).unique().tolist()
            for m in members:
                if m in b_idx:
                    row[b_idx[m]] = 1.0
                    matched = True
        if not matched and parts:
            # Match by substring token (e.g., "North" in "North_SKU_1") or exact sum match.
            tokens = [val for _, val in parts]
            for bid, idx in b_idx.items():
                if all(tok in bid for tok in tokens):
                    row[idx] = 1.0
                    matched = True
        if not matched and piv_yhat is not None and nid in piv_yhat.columns and n_b <= 16:
            target_vec = piv_yhat[nid].to_numpy(dtype=float)
            b_mat = piv_yhat.reindex(columns=bottom_ids).to_numpy(dtype=float)
            best_err = float("inf")
            best_mask = np.ones(n_b, dtype=float)
            for mask_int in range(1, 1 << n_b):
                cand = np.array([(mask_int >> j) & 1 for j in range(n_b)], dtype=float)
                err = float(np.max(np.abs(b_mat @ cand - target_vec)))
                if err < best_err:
                    best_err = err
                    best_mask = cand
            if best_err <= 1e-3 * max(1.0, float(np.max(np.abs(target_vec)))):
                row = best_mask
                matched = True
        if not matched:
            row[:] = 1.0

        s_rows.append(row)
        node_ids.append(nid)
        level_nodes.setdefault(lkey, []).append(nid)

    s_mat = np.vstack([*s_rows, np.eye(n_b, dtype=float)])
    node_ids.extend(bottom_ids)
    level_nodes_tuple = {k: tuple(v) for k, v in level_nodes.items()}
    level_nodes_tuple["__bottom__"] = tuple(bottom_ids)
    return (
        HierarchySpec(
            node_ids=tuple(node_ids),
            bottom_ids=tuple(bottom_ids),
            level_nodes=level_nodes_tuple,
            summing_matrix=s_mat,
        ),
        True,
    )


def _resolve_bottom_covariance(
    spec: HierarchySpec,
    method: str,
    weights: str | Mapping[str, float] | Sequence[float] | np.ndarray | None,
    *,
    b_hat: np.ndarray,
    b_var_from_intervals: np.ndarray | None,
    oof_df: pd.DataFrame | None,
    ts_id_col: str,
) -> np.ndarray:
    """Build the `(n_bottom, n_bottom)` covariance matrix `W_cs` for the bottom leaves."""
    n_b = spec.n_bottom
    eff_mode = str(weights if isinstance(weights, str) else method).strip().lower()

    if isinstance(weights, Mapping):
        w_vec = np.array(
            [max(float(weights.get(bid, 1.0)), _VAR_FLOOR) for bid in spec.bottom_ids],
            dtype=float,
        )
        return np.diag(w_vec)
    if isinstance(weights, (Sequence, np.ndarray)) and not isinstance(weights, str):
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if len(arr) != n_b:
            raise ConfigError(
                f"weights array length ({len(arr)}) must match number of bottom series ({n_b})"
            )
        return np.diag(np.maximum(arr, _VAR_FLOOR))

    if eff_mode in {"ols", "wls_struct", "struc", "uniform", "equal"}:
        return np.eye(n_b, dtype=float)

    if eff_mode in {"proportional", "baseline"}:
        scale = np.maximum(np.mean(np.abs(b_hat), axis=1), 1e-3)
        return np.diag(scale)

    res_mat = residual_matrix(oof_df, spec.bottom_ids, ts_id_col=ts_id_col)
    if res_mat is not None and res_mat.shape[0] >= 1:
        if eff_mode in {"wls_var", "oof_variance", "variance"} or res_mat.shape[0] < 2 or n_b == 1:
            var_diag = np.maximum(np.mean(res_mat**2, axis=0), _VAR_FLOOR)
            return np.diag(var_diag)
        w_shrink, _ = schafer_strimmer_shrinkage(res_mat)
        return w_shrink

    if b_var_from_intervals is not None:
        mean_var = np.maximum(np.mean(b_var_from_intervals, axis=1), _VAR_FLOOR)
        return np.diag(mean_var)

    scale = np.maximum(np.mean(np.abs(b_hat), axis=1), 1.0) ** 2
    return np.diag(scale)


def _solve_bounded_simplex_projection(
    b_unconstrained: np.ndarray,
    weights_diag: np.ndarray,
    a_row: np.ndarray,
    target_val: float,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Exact water-filling projection for `a^T b = target` under `lo <= b <= hi`."""
    x = np.clip(b_unconstrained.copy(), lo, hi)
    w = np.maximum(weights_diag, _VAR_FLOOR)
    a = np.asarray(a_row, dtype=float)
    active = np.abs(a) > 1e-12
    if not np.any(active):
        return x

    for _ in range(100):
        curr = float(a @ x)
        residual = target_val - curr
        if abs(residual) <= 1e-11 * max(1.0, abs(target_val)):
            break
        if residual > 0:
            free = active & (((a > 0) & (x < hi - 1e-12)) | ((a < 0) & (x > lo + 1e-12)))
        else:
            free = active & (((a > 0) & (x > lo + 1e-12)) | ((a < 0) & (x < hi - 1e-12)))
        if not np.any(free):
            break
        denom = float(np.sum((a[free] ** 2) * w[free]))
        if denom <= 1e-18:
            break
        lam = residual / denom
        x_new = x.copy()
        x_new[free] = np.clip(x[free] + lam * w[free] * a[free], lo[free], hi[free])
        x = x_new
    return x


def _solve_constrained_qp(
    b_hat_flat: np.ndarray,
    w_blocks: list[np.ndarray],
    a_eq: np.ndarray,
    c_eq: np.ndarray,
    lo_flat: np.ndarray,
    hi_flat: np.ndarray,
    *,
    smoothness_weight: float = 0.0,
) -> np.ndarray:
    """Solve `min_b 0.5 (b - b_hat)^T W^{-1} (b - b_hat)` s.t. `A_eq b = c_eq`, `lo <= b <= hi`."""
    n_total = len(b_hat_flat)
    n_b = w_blocks[0].shape[0]
    h_steps = len(w_blocks)
    smooth_w = float(smoothness_weight)
    use_smooth = smooth_w > 0.0 and h_steps >= 2

    # Pre-check per-variable box bounds.
    bad_box = np.flatnonzero(lo_flat > hi_flat + 1e-7)
    if len(bad_box) > 0:
        idx0 = int(bad_box[0])
        raise GoalSeekInfeasibleError(
            f"Infeasible goal-seek bounds at flat index {idx0}: "
            f"lower_bound ({lo_flat[idx0]:.4f}) > upper_bound ({hi_flat[idx0]:.4f})"
        )

    # Pre-check each equality constraint against box bounds [lo_flat, hi_flat].
    n_eq = a_eq.shape[0]
    for r in range(n_eq):
        row = a_eq[r]
        target_r = float(c_eq[r])
        pos = row > 0
        neg = row < 0
        min_possible = float(np.sum(row[pos] * lo_flat[pos]) + np.sum(row[neg] * hi_flat[neg]))
        max_possible = float(np.sum(row[pos] * hi_flat[pos]) + np.sum(row[neg] * lo_flat[neg]))
        tol = _FEAS_TOL * max(1.0, abs(target_r))
        if target_r < min_possible - tol or target_r > max_possible + tol:
            raise GoalSeekInfeasibleError(
                f"Target {target_r:.4f} is mathematically infeasible under box bounds and locked "
                f"nodes; feasible range is [{min_possible:.4f}, {max_possible:.4f}]."
            )

    # Pairwise subset/difference feasibility check (e.g., locked child nodes under a parent target).
    if n_eq >= 2:
        for r1 in range(n_eq):
            for r2 in range(n_eq):
                if r1 == r2:
                    continue
                diff_row = a_eq[r1] - a_eq[r2]
                if np.all(diff_row >= -1e-12) and np.any(diff_row > 1e-12):
                    pos_d = diff_row > 1e-12
                    diff_target = float(c_eq[r1] - c_eq[r2])
                    min_d = float(np.sum(diff_row[pos_d] * lo_flat[pos_d]))
                    max_d = float(np.sum(diff_row[pos_d] * hi_flat[pos_d]))
                    tol = _FEAS_TOL * max(1.0, abs(diff_target))
                    if diff_target < min_d - tol or diff_target > max_d + tol:
                        raise GoalSeekInfeasibleError(
                            f"Conflicting equality / locked-node constraints: residual target "
                            f"{diff_target:.4f} lies outside unlocked feasible range "
                            f"[{min_d:.4f}, {max_d:.4f}]."
                        )
                elif np.all(np.abs(diff_row) <= 1e-12):
                    if abs(float(c_eq[r1] - c_eq[r2])) > _FEAS_TOL * max(1.0, abs(float(c_eq[r1]))):
                        raise GoalSeekInfeasibleError(
                            f"Contradictory equality constraints on identical node set: "
                            f"{c_eq[r1]:.4f} != {c_eq[r2]:.4f}."
                        )

    # Build block-diagonal W and W_inv across the H horizon steps.
    w_inv_blocks: list[np.ndarray] = []
    w_diag_flat = np.empty(n_total, dtype=float)
    for t_idx, w_t in enumerate(w_blocks):
        reg = 1e-8 * max(float(np.trace(w_t)) / max(1, n_b), 1.0)
        w_reg = w_t + reg * np.eye(n_b, dtype=float)
        w_inv_blocks.append(np.linalg.inv(w_reg))
        w_diag_flat[t_idx * n_b : (t_idx + 1) * n_b] = np.maximum(np.diag(w_reg), _VAR_FLOOR)

    # Fast closed-form KKT + water-filling path when there is a single linear equality constraint
    # and diagonal W (or as an exact warm-start for SLSQP).
    def _apply_w(vec: np.ndarray) -> np.ndarray:
        out = np.empty_like(vec)
        for t_idx, w_t in enumerate(w_blocks):
            sl = slice(t_idx * n_b, (t_idx + 1) * n_b)
            out[sl] = w_t @ vec[sl]
        return out

    def _apply_winv(vec: np.ndarray) -> np.ndarray:
        out = np.empty_like(vec)
        for t_idx, winv_t in enumerate(w_inv_blocks):
            sl = slice(t_idx * n_b, (t_idx + 1) * n_b)
            out[sl] = winv_t @ vec[sl]
        return out

    # Compute unconstrained KKT solution on free variables.
    free_mask = (hi_flat - lo_flat) > 1e-11
    x0 = np.clip(b_hat_flat.copy(), lo_flat, hi_flat)
    if np.any(free_mask) and n_eq > 0:
        # Project out locked variables from c_eq.
        locked_mask = ~free_mask
        c_rem = c_eq - a_eq[:, locked_mask] @ x0[locked_mask]
        a_free = a_eq[:, free_mask]
        w_a_t = np.column_stack([_apply_w(a_eq[r]) for r in range(n_eq)])
        w_a_free = w_a_t.copy()
        w_a_free[locked_mask, :] = 0.0
        gram = a_eq @ w_a_free
        res_rhs = c_rem - a_free @ x0[free_mask]
        try:
            lam_vec = np.linalg.lstsq(gram + 1e-12 * np.eye(n_eq), res_rhs, rcond=None)[0]
            x_kkt = x0 + w_a_free @ lam_vec
        except np.linalg.LinAlgError:
            x_kkt = x0.copy()
    else:
        x_kkt = x0.copy()

    if (
        not use_smooth
        and n_eq > 0
        and np.all(x_kkt >= lo_flat - 1e-9)
        and np.all(x_kkt <= hi_flat + 1e-9)
        and float(np.max(np.abs(a_eq @ x_kkt - c_eq)))
        <= 1e-9 * max(1.0, float(np.max(np.abs(c_eq))))
    ):
        return np.clip(x_kkt, lo_flat, hi_flat)

    # Iterative active-set covariance-weighted KKT projection.
    x_act = np.clip(x_kkt.copy(), lo_flat, hi_flat)
    act_free = free_mask.copy()
    for _ in range(12):
        if not np.any(act_free) or n_eq == 0:
            break
        locked_now = ~act_free
        c_rem_act = c_eq - a_eq[:, locked_now] @ x_act[locked_now]
        w_a_act = np.column_stack([_apply_w(a_eq[r]) for r in range(n_eq)])
        w_a_act[locked_now, :] = 0.0
        gram_act = a_eq @ w_a_act
        res_act = c_rem_act - a_eq[:, act_free] @ b_hat_flat[act_free]
        try:
            lam_act = np.linalg.lstsq(gram_act + 1e-12 * np.eye(n_eq), res_act, rcond=None)[0]
            x_cand = x_act.copy()
            x_cand[act_free] = b_hat_flat[act_free] + (w_a_act @ lam_act)[act_free]
        except np.linalg.LinAlgError:
            break
        viol_lo = act_free & (x_cand < lo_flat - 1e-9)
        viol_hi = act_free & (x_cand > hi_flat + 1e-9)
        if not np.any(viol_lo) and not np.any(viol_hi):
            x_act = np.clip(x_cand, lo_flat, hi_flat)
            break
        x_act[viol_lo] = lo_flat[viol_lo]
        x_act[viol_hi] = hi_flat[viol_hi]
        act_free[viol_lo | viol_hi] = False

    if (
        not use_smooth
        and n_eq > 0
        and float(np.max(np.abs(a_eq @ x_act - c_eq)))
        <= 1e-9 * max(1.0, float(np.max(np.abs(c_eq))))
    ):
        return x_act

    if n_eq == 1:
        x_init = _solve_bounded_simplex_projection(
            x_kkt, w_diag_flat, a_eq[0], float(c_eq[0]), lo_flat, hi_flat
        )
    else:
        x_init = np.clip(x_kkt, lo_flat, hi_flat)
        for _ in range(25):
            for r in range(n_eq):
                x_init = _solve_bounded_simplex_projection(
                    x_init, w_diag_flat, a_eq[r], float(c_eq[r]), lo_flat, hi_flat
                )

    # Check if x_init already satisfies all equality and box constraints to machine precision
    # when W is diagonal and smoothness regularization is not active.
    is_all_diag = all(np.allclose(w_t, np.diag(np.diag(w_t)), atol=1e-10) for w_t in w_blocks)
    if (
        not use_smooth
        and is_all_diag
        and n_eq == 1
        and float(np.max(np.abs(a_eq @ x_init - c_eq)))
        <= 1e-9 * max(1.0, float(np.max(np.abs(c_eq))))
    ):
        return x_init

    # Solve full covariance-weighted QP via SLSQP on free variables.
    free_idx = np.flatnonzero(free_mask)
    if len(free_idx) == 0:
        err = float(np.max(np.abs(a_eq @ x_init - c_eq))) if n_eq > 0 else 0.0
        if err > _FEAS_TOL * max(1.0, float(np.max(np.abs(c_eq)) if n_eq > 0 else 1.0)):
            raise GoalSeekInfeasibleError(
                f"All nodes are locked, leaving an unresolvable discrepancy of {err:.4f} "
                "against target constraints."
            )
        return x_init

    # Reduce linear equality rows to linearly independent rows over free_idx.
    a_sub = a_eq[:, free_idx]
    c_sub = c_eq - a_eq[:, ~free_mask] @ x_init[~free_mask]
    u_svd, s_svd, vt_svd = np.linalg.svd(a_sub, full_matrices=False)
    rank = int(np.sum(s_svd > 1e-9))
    if rank < len(s_svd):
        # Check consistency in nullspace of u_svd[:, rank:].
        null_proj = u_svd[:, rank:].T @ c_sub
        if float(np.max(np.abs(null_proj))) > _FEAS_TOL * max(1.0, float(np.max(np.abs(c_sub)))):
            raise GoalSeekInfeasibleError(
                "Target and locked-node equality constraints are linearly inconsistent."
            )
        a_sub_ind = np.diag(s_svd[:rank]) @ vt_svd[:rank, :]
        c_sub_ind = u_svd[:, :rank].T @ c_sub
    else:
        a_sub_ind = a_sub
        c_sub_ind = c_sub

    scale_factor = max(1.0, float(np.mean(w_diag_flat[free_idx])))
    mean_winv_scaled = (
        float(np.mean([np.mean(np.diag(w_inv)) for w_inv in w_inv_blocks])) * scale_factor
    )
    # When the unpenalized lift (x_init - b_hat_flat) already has zero first-difference across
    # time (e.g. uniform time-invariant W without active per-step bounds), penalize temporal
    # roughness of the allocated trajectory b_{i,t} - b_{i,t-1} as well as lift roughness.
    init_lift_mat = (x_init - b_hat_flat).reshape(h_steps, n_b)
    penalize_level = use_smooth and float(np.max(np.abs(np.diff(init_lift_mat, axis=0)))) <= 1e-7
    ref_flat = np.zeros_like(b_hat_flat) if penalize_level else b_hat_flat

    def _obj(z_free: np.ndarray) -> float:
        full_vec = x_init.copy()
        full_vec[free_idx] = z_free
        diff = full_vec - b_hat_flat
        winv_diff = _apply_winv(diff)
        val = 0.5 * float(diff @ winv_diff) * scale_factor
        if use_smooth:
            s_diff = (full_vec - ref_flat).reshape(h_steps, n_b)
            d_step = s_diff[1:, :] - s_diff[:-1, :]
            val += 0.5 * smooth_w * mean_winv_scaled * float(np.sum(d_step * d_step))
        return val

    def _jac(z_free: np.ndarray) -> np.ndarray:
        full_vec = x_init.copy()
        full_vec[free_idx] = z_free
        diff = full_vec - b_hat_flat
        winv_diff = _apply_winv(diff)
        grad = winv_diff * scale_factor
        if use_smooth:
            s_diff = (full_vec - ref_flat).reshape(h_steps, n_b)
            d_step = s_diff[1:, :] - s_diff[:-1, :]
            g_smooth = np.zeros((h_steps, n_b), dtype=float)
            g_smooth[:-1, :] -= d_step
            g_smooth[1:, :] += d_step
            grad = grad + (smooth_w * mean_winv_scaled) * g_smooth.reshape(-1)
        return grad[free_idx]

    bounds_list = [
        (
            float(lo_flat[i]) if np.isfinite(lo_flat[i]) else None,
            float(hi_flat[i]) if np.isfinite(hi_flat[i]) else None,
        )
        for i in free_idx
    ]
    constraints = {
        "type": "eq",
        "fun": lambda z: a_sub_ind @ z - c_sub_ind,
        "jac": lambda _z: a_sub_ind,
    }
    res = minimize(
        _obj,
        x_init[free_idx],
        method="SLSQP",
        jac=_jac,
        bounds=bounds_list,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 250 if len(free_idx) > 80 else 500},
    )

    x_sol = x_init.copy()
    x_sol[free_idx] = np.clip(res.x, lo_flat[free_idx], hi_flat[free_idx])

    # Active-set water-filling polish to enforce exact equality to machine precision.
    for _ in range(40):
        for r in range(n_eq):
            x_sol = _solve_bounded_simplex_projection(
                x_sol, w_diag_flat, a_eq[r], float(c_eq[r]), lo_flat, hi_flat
            )

    max_eq_err = float(np.max(np.abs(a_eq @ x_sol - c_eq))) if n_eq > 0 else 0.0
    max_scale = max(1.0, float(np.max(np.abs(c_eq))) if n_eq > 0 else 1.0)
    if max_eq_err > 1e-4 * max_scale:
        raise GoalSeekInfeasibleError(
            f"Goal-seek QP could not satisfy equality constraints within bounds "
            f"(max discrepancy {max_eq_err:.4f}); check locked_nodes and "
            "[lower_bounds, upper_bounds]."
        )
    return x_sol


def allocate_target(
    forecasts_df: pd.DataFrame | None = None,
    target_total: float | Sequence[float] | Mapping[Any, float] | pd.DataFrame | None = None,
    *,
    predictions_df: pd.DataFrame | None = None,
    target: float | Sequence[float] | Mapping[Any, float] | pd.DataFrame | None = None,
    targets: float | Sequence[float] | Mapping[Any, float] | pd.DataFrame | None = None,
    target_sum: float | Sequence[float] | Mapping[Any, float] | None = None,
    horizon_total: float | None = None,
    node_id: str | None = None,
    target_node: str = TOTAL_NODE_ID,
    period: Sequence[Any] | str | None = None,
    target_dates: Sequence[Any] | str | None = None,
    target_mode: Literal["auto", "per_step", "horizon_sum", "proportional"] = "auto",
    target_metric: Literal["auto", "units", "revenue"] = "auto",
    locked_nodes: Sequence[str] | Mapping[str, float] | None = None,
    locks: Sequence[str] | Mapping[str, float] | None = None,
    lower_bounds: Mapping[str, float] | Sequence[float] | np.ndarray | float | None = None,
    upper_bounds: Mapping[str, float] | Sequence[float] | np.ndarray | float | None = None,
    bounds: (
        Mapping[str, tuple[float | None, float | None]]
        | tuple[float | None, float | None]
        | pd.DataFrame
        | None
    ) = None,
    floor: float | None = 0.0,
    cap: float | None = None,
    ceiling: float | None = None,
    nonneg: bool = True,
    method: str = "mint_shrink",
    weights: str | Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    smoothness_weight: float = 0.0,
    hier_spec: HierarchySpec | None = None,
    spec: HierarchySpec | None = None,
    cfg: RunConfig | Mapping[str, Any] | None = None,
    history_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    price_col: str | Mapping[str, float] | float | None = None,
    prices: Mapping[str, float] | float | pd.DataFrame | None = None,
    unit_cost: float | Mapping[str, float] | str | None = None,
    include_inventory_impact: bool = False,
    lead_time: float = 7.0,
    lead_time_std: float = 0.0,
    service_level: float | None = 0.95,
    inventory_params: Mapping[str, Any] | None = None,
    model_type: str | None = None,
    scenario: str | None = None,
    date_col: str = "ds",
    ts_id_col: str = "ts_id",
    target_col: str = "y",
    return_result: bool = False,
) -> pd.DataFrame | GoalSeekResult:
    """Optimally disaggregate a top-down or regional target across leaf series and dates (pure).

    Solves the equality- and box-constrained Minimum-Trace quadratic program over the bottom-level
    forecast trajectory while respecting ``locked_nodes`` / ``locks`` and
    ``[lower_bounds, upper_bounds]``, then rolls up the coherent hierarchy
    ``y_tilde = S @ b_tilde`` and optionally propagates the allocated forecast into revenue,
    gross margin, and `(s, S)` inventory policy deltas.
    """
    raw_df = forecasts_df if forecasts_df is not None else predictions_df
    if raw_df is None or not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        raise DataError(
            "allocate_target requires a non-empty forecasts_df / predictions_df DataFrame"
        )

    df = raw_df.copy()

    # Support scenario simulation outputs directly (`scenario_yhat` -> `yhat`).
    if "scenario" in df.columns:
        if scenario is not None:
            df = df[df["scenario"].astype(str) == str(scenario)].copy()
        else:
            non_base = df[df["scenario"].astype(str) != "baseline"]
            if not non_base.empty:
                first_scen = str(non_base["scenario"].iloc[0])
                df = non_base[non_base["scenario"].astype(str) == first_scen].copy()
        if df.empty:
            raise DataError(f"No rows found for scenario='{scenario}' in forecasts_df")

    if "yhat" not in df.columns and "scenario_yhat" in df.columns:
        df["yhat"] = df["scenario_yhat"].astype(float)
    if "yhat" not in df.columns:
        raise DataError("forecasts_df must contain a 'yhat' or 'scenario_yhat' column")

    if ts_id_col not in df.columns:
        df[ts_id_col] = "series_0"
    dcol = (
        date_col
        if date_col in df.columns
        else ("forecast_date" if "forecast_date" in df.columns else "ds")
    )
    if dcol not in df.columns:
        df[dcol] = pd.date_range("2026-01-01", periods=len(df), freq="D")

    if model_type is not None and "model_type" in df.columns:
        df = df[df["model_type"].astype(str) == str(model_type)].copy()
        if df.empty:
            raise DataError(f"No forecast rows found for model_type='{model_type}'")
    elif "model_type" in df.columns and df["model_type"].nunique() > 1:
        first_m = str(df["model_type"].iloc[0])
        df = df[df["model_type"].astype(str) == first_m].copy()
    elif "model_type" not in df.columns:
        df["model_type"] = model_type or "default"

    eff_model_type = str(df["model_type"].iloc[0])

    # Resolve target specification.
    raw_target = (
        target_total
        if target_total is not None
        else (
            target
            if target is not None
            else (
                targets
                if targets is not None
                else (target_sum if target_sum is not None else horizon_total)
            )
        )
    )
    if raw_target is None:
        raise ConfigError(
            "allocate_target requires target_total "
            "(or target / targets / target_sum / horizon_total)"
        )
    if (
        isinstance(smoothness_weight, bool)
        or not isinstance(smoothness_weight, (int, float, np.number))
        or not np.isfinite(float(smoothness_weight))
        or float(smoothness_weight) < 0.0
    ):
        raise ConfigError(
            f"smoothness_weight must be a finite float >= 0.0; got {smoothness_weight!r}"
        )
    smooth_w = float(smoothness_weight)

    eff_cap = cap if cap is not None else ceiling
    eff_hier_spec = hier_spec if hier_spec is not None else spec
    eff_target_node = str(node_id if node_id is not None else target_node)
    resolved_spec, include_agg_in_out = _infer_hierarchy_spec_from_frame(
        df,
        hier_spec=eff_hier_spec,
        cfg=cfg,
        history_df=history_df,
        ts_id_col=ts_id_col,
        date_col=dcol,
    )
    bottom_ids = list(resolved_spec.bottom_ids)
    n_b = len(bottom_ids)
    b_to_idx = {bid: i for i, bid in enumerate(bottom_ids)}
    node_to_idx = {nid: i for i, nid in enumerate(resolved_spec.node_ids)}

    all_dates = sorted(df[dcol].drop_duplicates().tolist())
    h_total = len(all_dates)
    date_to_t = {dt: t for t, dt in enumerate(all_dates)}

    # Resolve active period / target_dates subset.
    raw_period = period if period is not None else target_dates
    if raw_period is not None:
        if isinstance(raw_period, (str, pd.Timestamp, int)):
            raw_period_seq: Sequence[Any] = [raw_period]
        else:
            raw_period_seq = list(raw_period)
        active_t_indices: list[int] = []
        for item in raw_period_seq:
            if isinstance(item, int) and 0 <= item < h_total and item not in date_to_t:
                active_t_indices.append(item)
            elif item in date_to_t:
                active_t_indices.append(date_to_t[item])
            else:
                ts_item = pd.Timestamp(item)
                for dt_cand, t_cand in date_to_t.items():
                    if pd.Timestamp(dt_cand) == ts_item:
                        active_t_indices.append(t_cand)
                        break
        if not active_t_indices:
            raise ConfigError(
                f"None of the requested target_dates / period {raw_period_seq} "
                "matched forecast dates"
            )
        active_t_indices = sorted(set(active_t_indices))
    else:
        active_t_indices = list(range(h_total))

    # Pivot bottom-level baseline yhat (n_b, H) and intervals.
    piv_yhat = (
        df.pivot_table(index=ts_id_col, columns=dcol, values="yhat", aggfunc="first")
        .reindex(index=bottom_ids, columns=all_dates)
        .fillna(0.0)
    )
    b_hat = piv_yhat.to_numpy(dtype=float)

    b_var_intervals: np.ndarray | None = None
    lo_col_name = (
        "yhat_lower"
        if "yhat_lower" in df.columns
        else ("yhat_lower_80" if "yhat_lower_80" in df.columns else None)
    )
    hi_col_name = (
        "yhat_upper"
        if "yhat_upper" in df.columns
        else ("yhat_upper_80" if "yhat_upper_80" in df.columns else None)
    )
    if lo_col_name and hi_col_name:
        piv_lo = (
            df.pivot_table(index=ts_id_col, columns=dcol, values=lo_col_name, aggfunc="first")
            .reindex(index=bottom_ids, columns=all_dates)
            .fillna(piv_yhat)
            .to_numpy(dtype=float)
        )
        piv_hi = (
            df.pivot_table(index=ts_id_col, columns=dcol, values=hi_col_name, aggfunc="first")
            .reindex(index=bottom_ids, columns=all_dates)
            .fillna(piv_yhat)
            .to_numpy(dtype=float)
        )
        hw_mat = np.maximum(0.0, 0.5 * (piv_hi - piv_lo))
        b_var_intervals = np.maximum((hw_mat / _Z_80) ** 2, _VAR_FLOOR)

    # Resolve prices and unit_cost per (bottom_i, t).
    eff_price_source = prices if prices is not None else price_col
    has_price = eff_price_source is not None or (
        "price" in df.columns and target_metric == "revenue"
    )
    price_mat = np.ones((n_b, h_total), dtype=float)
    if isinstance(eff_price_source, str):
        if eff_price_source in df.columns:
            p_piv = (
                df.pivot_table(
                    index=ts_id_col, columns=dcol, values=eff_price_source, aggfunc="first"
                )
                .reindex(index=bottom_ids, columns=all_dates)
                .fillna(1.0)
            )
            price_mat = p_piv.to_numpy(dtype=float)
        elif history_df is not None and eff_price_source in history_df.columns:
            last_p = (
                history_df.sort_values(dcol if dcol in history_df.columns else "ds")
                .groupby(ts_id_col)[eff_price_source]
                .last()
                .to_dict()
            )
            for i, bid in enumerate(bottom_ids):
                price_mat[i, :] = float(last_p.get(bid, 1.0))
        else:
            raise ConfigError(
                f"price_col '{eff_price_source}' not found in forecasts_df or history_df"
            )
    elif isinstance(eff_price_source, (int, float)):
        price_mat.fill(float(eff_price_source))
    elif isinstance(eff_price_source, Mapping):
        for i, bid in enumerate(bottom_ids):
            if bid in eff_price_source:
                price_mat[i, :] = float(eff_price_source[bid])
    elif isinstance(eff_price_source, pd.DataFrame):
        p_col = "price" if "price" in eff_price_source.columns else eff_price_source.columns[-1]
        if ts_id_col in eff_price_source.columns:
            p_map = eff_price_source.groupby(ts_id_col)[p_col].last().to_dict()
            for i, bid in enumerate(bottom_ids):
                price_mat[i, :] = float(p_map.get(bid, 1.0))
    elif "price" in df.columns:
        has_price = True
        p_piv = (
            df.pivot_table(index=ts_id_col, columns=dcol, values="price", aggfunc="first")
            .reindex(index=bottom_ids, columns=all_dates)
            .fillna(1.0)
        )
        price_mat = p_piv.to_numpy(dtype=float)

    cost_mat = np.zeros((n_b, h_total), dtype=float)
    has_cost = unit_cost is not None or "unit_cost" in df.columns
    if isinstance(unit_cost, (int, float)):
        cost_mat.fill(float(unit_cost))
    elif isinstance(unit_cost, str):
        if unit_cost in df.columns:
            c_piv = (
                df.pivot_table(index=ts_id_col, columns=dcol, values=unit_cost, aggfunc="first")
                .reindex(index=bottom_ids, columns=all_dates)
                .fillna(0.0)
            )
            cost_mat = c_piv.to_numpy(dtype=float)
        elif history_df is not None and unit_cost in history_df.columns:
            last_c = (
                history_df.sort_values(dcol if dcol in history_df.columns else "ds")
                .groupby(ts_id_col)[unit_cost]
                .last()
                .to_dict()
            )
            for i, bid in enumerate(bottom_ids):
                cost_mat[i, :] = float(last_c.get(bid, 0.0))
        else:
            raise ConfigError(
                f"unit_cost column '{unit_cost}' not found in forecasts_df or history_df"
            )
    elif isinstance(unit_cost, Mapping):
        for i, bid in enumerate(bottom_ids):
            if bid in unit_cost:
                cost_mat[i, :] = float(unit_cost[bid])
    elif "unit_cost" in df.columns:
        c_piv = (
            df.pivot_table(index=ts_id_col, columns=dcol, values="unit_cost", aggfunc="first")
            .reindex(index=bottom_ids, columns=all_dates)
            .fillna(0.0)
        )
        cost_mat = c_piv.to_numpy(dtype=float)

    use_revenue_weights = target_metric == "revenue"

    # Build box bounds [lo_mat, hi_mat] of shape (n_b, H).
    default_lo = float(floor) if floor is not None else (0.0 if nonneg else -np.inf)
    default_hi = float(eff_cap) if eff_cap is not None else np.inf
    lo_mat = np.full((n_b, h_total), default_lo, dtype=float)
    hi_mat = np.full((n_b, h_total), default_hi, dtype=float)
    locked_mat = np.zeros((n_b, h_total), dtype=bool)

    if isinstance(bounds, tuple) and len(bounds) == 2:
        if bounds[0] is not None:
            lo_mat = np.maximum(lo_mat, float(bounds[0]))
        if bounds[1] is not None:
            hi_mat = np.minimum(hi_mat, float(bounds[1]))
    elif isinstance(bounds, Mapping):
        for k_id, b_pair in bounds.items():
            k_str = str(k_id)
            if k_str in b_to_idx and isinstance(b_pair, tuple) and len(b_pair) == 2:
                idx = b_to_idx[k_str]
                if b_pair[0] is not None:
                    lo_mat[idx, :] = float(b_pair[0])
                if b_pair[1] is not None:
                    hi_mat[idx, :] = float(b_pair[1])
    elif isinstance(bounds, pd.DataFrame) and not bounds.empty:
        for brow in bounds.to_dict(orient="records"):
            bid_s = str(brow.get(ts_id_col, ""))
            if bid_s in b_to_idx:
                idx = b_to_idx[bid_s]
                l_v = brow.get("lower_bound", brow.get("lower", brow.get("min")))
                u_v = brow.get("upper_bound", brow.get("upper", brow.get("max")))
                if dcol in brow and brow[dcol] in date_to_t:
                    t_i = date_to_t[brow[dcol]]
                    if l_v is not None and pd.notna(l_v):
                        lo_mat[idx, t_i] = float(l_v)
                    if u_v is not None and pd.notna(u_v):
                        hi_mat[idx, t_i] = float(u_v)
                else:
                    if l_v is not None and pd.notna(l_v):
                        lo_mat[idx, :] = float(l_v)
                    if u_v is not None and pd.notna(u_v):
                        hi_mat[idx, :] = float(u_v)

    if lower_bounds is not None:
        if isinstance(lower_bounds, (int, float)):
            lo_mat = np.full((n_b, h_total), float(lower_bounds), dtype=float)
        elif isinstance(lower_bounds, Mapping):
            for k_id, l_val in lower_bounds.items():
                if str(k_id) in b_to_idx and l_val is not None:
                    lo_mat[b_to_idx[str(k_id)], :] = float(l_val)
        else:
            arr_l = np.asarray(lower_bounds, dtype=float)
            if arr_l.shape == (n_b, h_total):
                lo_mat = arr_l.copy()
            elif arr_l.size == n_b:
                lo_mat = np.repeat(arr_l.reshape(n_b, 1), h_total, axis=1)

    if upper_bounds is not None:
        if isinstance(upper_bounds, (int, float)):
            hi_mat = np.full((n_b, h_total), float(upper_bounds), dtype=float)
        elif isinstance(upper_bounds, Mapping):
            for k_id, u_val in upper_bounds.items():
                if str(k_id) in b_to_idx and u_val is not None:
                    hi_mat[b_to_idx[str(k_id)], :] = float(u_val)
        else:
            arr_u = np.asarray(upper_bounds, dtype=float)
            if arr_u.shape == (n_b, h_total):
                hi_mat = arr_u.copy()
            elif arr_u.size == n_b:
                hi_mat = np.repeat(arr_u.reshape(n_b, 1), h_total, axis=1)

    # Inactive dates (outside `period` / `target_dates`) stay locked to their baseline forecasts.
    active_set = set(active_t_indices)
    for t_idx in range(h_total):
        if t_idx not in active_set:
            lo_mat[:, t_idx] = b_hat[:, t_idx]
            hi_mat[:, t_idx] = b_hat[:, t_idx]
            locked_mat[:, t_idx] = True

    # Build linear equality constraints A_eq @ b_flat = c_eq
    # (where b_flat is ordered by t then i: k = t * n_b + i).
    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []

    def _node_s_row(nid_raw: str) -> np.ndarray:
        nid = str(nid_raw)
        if nid in node_to_idx:
            return resolved_spec.summing_matrix[node_to_idx[nid]].copy()
        # Allow shorthand like "North" matching "region=North".
        for cand_nid, r_idx in node_to_idx.items():
            if cand_nid.endswith(f"={nid}") or cand_nid == nid:
                return resolved_spec.summing_matrix[r_idx].copy()
        if nid in b_to_idx:
            r = np.zeros(n_b, dtype=float)
            r[b_to_idx[nid]] = 1.0
            return r
        raise ConfigError(
            f"Unknown hierarchy node '{nid}'; available nodes: {list(resolved_spec.node_ids)}"
        )

    # Process locked_nodes and locks.
    for lock_arg in (locked_nodes, locks):
        if lock_arg is None:
            continue
        if isinstance(lock_arg, Mapping):
            for k_lock, val_lock in lock_arg.items():
                s_r = _node_s_row(str(k_lock))
                member_indices = np.flatnonzero(s_r > 0.5).tolist()
                if len(member_indices) == 1 and (
                    target_mode == "per_step" or len(active_t_indices) == 1
                ):
                    idx_b = member_indices[0]
                    for t_idx in active_t_indices:
                        lo_mat[idx_b, t_idx] = float(val_lock)
                        hi_mat[idx_b, t_idx] = float(val_lock)
                        locked_mat[idx_b, t_idx] = True
                elif len(member_indices) == 1:
                    # Single leaf locked to explicit horizon total val_lock across active_t_indices.
                    idx_b = member_indices[0]
                    base_slice = np.maximum(b_hat[idx_b, active_t_indices], 0.0)
                    base_sum = float(np.sum(base_slice))
                    shares = (
                        base_slice / base_sum
                        if base_sum > 1e-12
                        else np.full(
                            len(active_t_indices), 1.0 / len(active_t_indices), dtype=float
                        )
                    )
                    for pos_t, t_idx in enumerate(active_t_indices):
                        v_step = float(val_lock) * float(shares[pos_t])
                        if (
                            v_step < lo_mat[idx_b, t_idx] - _FEAS_TOL
                            or v_step > hi_mat[idx_b, t_idx] + _FEAS_TOL
                        ):
                            raise GoalSeekInfeasibleError(
                                f"Locked value {val_lock} for '{k_lock}' violates its box bounds "
                                f"[{lo_mat[idx_b, t_idx]}, {hi_mat[idx_b, t_idx]}] at step {t_idx}."
                            )
                        lo_mat[idx_b, t_idx] = v_step
                        hi_mat[idx_b, t_idx] = v_step
                        locked_mat[idx_b, t_idx] = True
                else:
                    # Aggregate node locked to explicit value val_lock across active_t_indices.
                    row_flat = np.zeros(n_b * h_total, dtype=float)
                    for t_idx in active_t_indices:
                        row_flat[t_idx * n_b : (t_idx + 1) * n_b] = s_r
                    eq_rows.append(row_flat)
                    eq_rhs.append(float(val_lock))
        else:
            for k_lock in lock_arg:
                s_r = _node_s_row(str(k_lock))
                member_indices = np.flatnonzero(s_r > 0.5).tolist()
                if len(member_indices) == 1:
                    idx_b = member_indices[0]
                    for t_idx in active_t_indices:
                        val_b = float(b_hat[idx_b, t_idx])
                        if (
                            val_b < lo_mat[idx_b, t_idx] - _FEAS_TOL
                            or val_b > hi_mat[idx_b, t_idx] + _FEAS_TOL
                        ):
                            raise GoalSeekInfeasibleError(
                                f"Locked baseline value {val_b:.4f} for '{k_lock}' violates "
                                f"its box bounds [{lo_mat[idx_b, t_idx]:.4f}, "
                                f"{hi_mat[idx_b, t_idx]:.4f}]."
                            )
                        lo_mat[idx_b, t_idx] = val_b
                        hi_mat[idx_b, t_idx] = val_b
                        locked_mat[idx_b, t_idx] = True
                else:
                    # Aggregate node locked to its baseline forecast at each active timestamp.
                    for t_idx in active_t_indices:
                        row_flat = np.zeros(n_b * h_total, dtype=float)
                        row_flat[t_idx * n_b : (t_idx + 1) * n_b] = s_r
                        eq_rows.append(row_flat)
                        eq_rhs.append(float(s_r @ b_hat[:, t_idx]))

    # Add target equality constraint(s).
    if isinstance(raw_target, pd.DataFrame):
        if raw_target.empty:
            raise ConfigError("target_total DataFrame cannot be empty")
        val_col = (
            "target"
            if "target" in raw_target.columns
            else ("yhat" if "yhat" in raw_target.columns else raw_target.columns[-1])
        )
        for trow in raw_target.to_dict(orient="records"):
            nid_t = str(trow.get(ts_id_col, eff_target_node))
            s_r = _node_s_row(nid_t)
            if dcol in trow and trow[dcol] in date_to_t:
                t_idx = date_to_t[trow[dcol]]
                row_flat = np.zeros(n_b * h_total, dtype=float)
                w_m = price_mat[:, t_idx] if use_revenue_weights else np.ones(n_b, dtype=float)
                row_flat[t_idx * n_b : (t_idx + 1) * n_b] = s_r * w_m
                eq_rows.append(row_flat)
                eq_rhs.append(float(trow[val_col]))
            else:
                row_flat = np.zeros(n_b * h_total, dtype=float)
                for t_idx in active_t_indices:
                    w_m = price_mat[:, t_idx] if use_revenue_weights else np.ones(n_b, dtype=float)
                    row_flat[t_idx * n_b : (t_idx + 1) * n_b] = s_r * w_m
                eq_rows.append(row_flat)
                eq_rhs.append(float(trow[val_col]))
    elif isinstance(raw_target, Mapping):
        if not raw_target:
            raise ConfigError("target_total mapping cannot be empty")
        # Determine whether keys are hierarchy node IDs or dates/steps.
        sample_keys = list(raw_target.keys())
        is_node_mapping = all(
            str(k) in node_to_idx
            or str(k) in b_to_idx
            or any(cand.endswith(f"={k}") for cand in node_to_idx)
            for k in sample_keys
        )
        if is_node_mapping:
            for k_node, v_targ in raw_target.items():
                s_r = _node_s_row(str(k_node))
                if target_mode == "per_step":
                    for t_idx in active_t_indices:
                        row_flat = np.zeros(n_b * h_total, dtype=float)
                        w_m = (
                            price_mat[:, t_idx]
                            if use_revenue_weights
                            else np.ones(n_b, dtype=float)
                        )
                        row_flat[t_idx * n_b : (t_idx + 1) * n_b] = s_r * w_m
                        eq_rows.append(row_flat)
                        eq_rhs.append(float(v_targ))
                else:
                    row_flat = np.zeros(n_b * h_total, dtype=float)
                    for t_idx in active_t_indices:
                        w_m = (
                            price_mat[:, t_idx]
                            if use_revenue_weights
                            else np.ones(n_b, dtype=float)
                        )
                        row_flat[t_idx * n_b : (t_idx + 1) * n_b] = s_r * w_m
                    eq_rows.append(row_flat)
                    eq_rhs.append(float(v_targ))
        else:
            s_r = _node_s_row(eff_target_node)
            for k_dt, v_targ in raw_target.items():
                matched_t: int | None = None
                if isinstance(k_dt, int) and 0 <= k_dt < h_total and k_dt not in date_to_t:
                    matched_t = k_dt
                elif k_dt in date_to_t:
                    matched_t = date_to_t[k_dt]
                else:
                    ts_k = pd.Timestamp(k_dt)
                    for dt_cand, t_cand in date_to_t.items():
                        if pd.Timestamp(dt_cand) == ts_k:
                            matched_t = t_cand
                            break
                if matched_t is None:
                    raise ConfigError(
                        f"Target key '{k_dt}' did not match any hierarchy node or forecast date"
                    )
                row_flat = np.zeros(n_b * h_total, dtype=float)
                w_m = price_mat[:, matched_t] if use_revenue_weights else np.ones(n_b, dtype=float)
                row_flat[matched_t * n_b : (matched_t + 1) * n_b] = s_r * w_m
                eq_rows.append(row_flat)
                eq_rhs.append(float(v_targ))
    elif isinstance(raw_target, (Sequence, np.ndarray)) and not isinstance(
        raw_target, (str, bytes)
    ):
        seq_vals = [float(x) for x in raw_target]
        if len(seq_vals) != len(active_t_indices):
            raise ConfigError(
                f"Target sequence length ({len(seq_vals)}) must match "
                f"active horizon length ({len(active_t_indices)})."
            )
        s_r = _node_s_row(eff_target_node)
        for pos_t, t_idx in enumerate(active_t_indices):
            row_flat = np.zeros(n_b * h_total, dtype=float)
            w_m = price_mat[:, t_idx] if use_revenue_weights else np.ones(n_b, dtype=float)
            row_flat[t_idx * n_b : (t_idx + 1) * n_b] = s_r * w_m
            eq_rows.append(row_flat)
            eq_rhs.append(seq_vals[pos_t])
    else:
        scalar_target = float(raw_target)
        if not np.isfinite(scalar_target):
            raise ConfigError(f"target_total must be finite; got {raw_target!r}")
        s_r = _node_s_row(eff_target_node)
        if target_mode == "per_step":
            for t_idx in active_t_indices:
                row_flat = np.zeros(n_b * h_total, dtype=float)
                w_m = price_mat[:, t_idx] if use_revenue_weights else np.ones(n_b, dtype=float)
                row_flat[t_idx * n_b : (t_idx + 1) * n_b] = s_r * w_m
                eq_rows.append(row_flat)
                eq_rhs.append(scalar_target)
        else:
            row_flat = np.zeros(n_b * h_total, dtype=float)
            for t_idx in active_t_indices:
                w_m = price_mat[:, t_idx] if use_revenue_weights else np.ones(n_b, dtype=float)
                row_flat[t_idx * n_b : (t_idx + 1) * n_b] = s_r * w_m
            eq_rows.append(row_flat)
            eq_rhs.append(scalar_target)

    a_eq = np.vstack(eq_rows)
    c_eq = np.asarray(eq_rhs, dtype=float)

    # Build covariance blocks W_t across horizon steps t = 0..H-1.
    w_cs = _resolve_bottom_covariance(
        resolved_spec,
        method,
        weights,
        b_hat=b_hat,
        b_var_from_intervals=b_var_intervals,
        oof_df=oof_df,
        ts_id_col=ts_id_col,
    )
    eff_mode_str = str(weights if isinstance(weights, str) else method).strip().lower()
    w_blocks: list[np.ndarray] = []
    for t_idx in range(h_total):
        if eff_mode_str in {"proportional", "baseline"}:
            w_blocks.append(np.diag(np.maximum(np.abs(b_hat[:, t_idx]), 1e-4)))
        elif (
            b_var_intervals is not None
            and oof_df is None
            and not isinstance(weights, (Mapping, Sequence, np.ndarray))
        ):
            w_blocks.append(np.diag(np.maximum(b_var_intervals[:, t_idx], _VAR_FLOOR)))
        elif (
            oof_df is None
            and b_var_intervals is None
            and eff_mode_str not in {"ols", "wls_struct", "struc", "uniform", "equal"}
            and not isinstance(weights, (Mapping, Sequence, np.ndarray))
        ):
            w_blocks.append(np.diag(np.maximum(np.abs(b_hat[:, t_idx]), 1.0) ** 2))
        else:
            w_blocks.append(w_cs)

    # Flatten column-major by timestamp t: flat index k = t * n_b + i.
    b_hat_flat = b_hat.T.reshape(-1)
    lo_flat = lo_mat.T.reshape(-1)
    hi_flat = hi_mat.T.reshape(-1)

    b_alloc_flat = _solve_constrained_qp(
        b_hat_flat,
        w_blocks,
        a_eq,
        c_eq,
        lo_flat,
        hi_flat,
        smoothness_weight=smooth_w,
    )
    b_alloc = b_alloc_flat.reshape(h_total, n_b).T  # shape (n_b, H)

    # Full hierarchy rollup Y_tilde = S @ B_alloc and Y_hat = S @ B_hat of shape (n_nodes, H).
    s_mat = resolved_spec.summing_matrix
    y_base_all = s_mat @ b_hat
    y_alloc_all = s_mat @ b_alloc
    lo_all = s_mat @ lo_mat
    hi_all = np.where(
        (s_mat @ np.where(np.isfinite(hi_mat), 0.0, 1.0)) > 0.0,
        np.inf,
        s_mat @ np.where(np.isfinite(hi_mat), hi_mat, 0.0),
    )

    # Normalized bottom-series allocation weights for transparency.
    raw_series_w = np.diag(w_cs).copy()
    w_sum = float(np.sum(raw_series_w))
    norm_b_weights = raw_series_w / w_sum if w_sum > 1e-12 else np.full(n_b, 1.0 / n_b, dtype=float)
    node_weights = s_mat @ norm_b_weights

    # Determine which nodes to include in output DataFrame.
    output_node_ids = (
        list(resolved_spec.node_ids) if include_agg_in_out else list(resolved_spec.bottom_ids)
    )

    # Index original rows for interval extraction.
    orig_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    for r in df.to_dict(orient="records"):
        orig_by_key[(str(r[ts_id_col]), r[dcol])] = {str(k): v for k, v in r.items()}

    target_node_idx = node_to_idx.get(eff_target_node, 0)
    records: list[dict[str, Any]] = []
    for nid in output_node_ids:
        n_idx = node_to_idx[nid]
        is_bottom = nid in b_to_idx
        b_idx_opt = b_to_idx.get(nid)
        for t_idx, dt_val in enumerate(all_dates):
            base_v = float(y_base_all[n_idx, t_idx])
            alloc_v = float(y_alloc_all[n_idx, t_idx])
            d_v = alloc_v - base_v
            d_pct = (100.0 * d_v / abs(base_v)) if abs(base_v) > 1e-12 else 0.0
            lo_v = (
                float(lo_mat[b_idx_opt, t_idx])
                if b_idx_opt is not None
                else float(lo_all[n_idx, t_idx])
            )
            hi_v = (
                float(hi_mat[b_idx_opt, t_idx])
                if b_idx_opt is not None
                else float(hi_all[n_idx, t_idx])
            )
            is_lck = bool(locked_mat[b_idx_opt, t_idx]) if b_idx_opt is not None else False

            if is_lck or abs(hi_v - lo_v) <= 1e-9:
                c_status = "LOCKED"
            elif np.isfinite(lo_v) and abs(alloc_v - lo_v) <= 1e-6 * max(1.0, abs(lo_v)):
                c_status = "AT_FLOOR"
            elif np.isfinite(hi_v) and abs(alloc_v - hi_v) <= 1e-6 * max(1.0, abs(hi_v)):
                c_status = "AT_CAP"
            else:
                c_status = "FREE"

            orig_r = orig_by_key.get((nid, dt_val), {})
            base_lo = float(
                orig_r.get(
                    "yhat_lower", orig_r.get("yhat_lower_80", base_v - 0.1 * max(abs(base_v), 1.0))
                )
            )
            base_hi = float(
                orig_r.get(
                    "yhat_upper", orig_r.get("yhat_upper_80", base_v + 0.1 * max(abs(base_v), 1.0))
                )
            )
            hw = max(0.0, 0.5 * (base_hi - base_lo))
            alloc_lo = alloc_v - hw
            alloc_hi = alloc_v + hw
            if np.isfinite(lo_v):
                alloc_lo = max(lo_v, alloc_lo)
            if np.isfinite(hi_v):
                alloc_hi = min(hi_v, alloc_hi)
            alloc_lo = min(alloc_lo, alloc_v)
            alloc_hi = max(alloc_hi, alloc_v)

            root_tot_t = float(y_alloc_all[target_node_idx, t_idx])
            share_t = (
                (alloc_v / root_tot_t)
                if abs(root_tot_t) > 1e-12
                else (
                    float(norm_b_weights[b_idx_opt])
                    if b_idx_opt is not None
                    else float(node_weights[n_idx])
                )
            )

            rec: dict[str, Any] = {
                ts_id_col: nid,
                "model_type": eff_model_type,
                "ds": pd.Timestamp(dt_val) if not isinstance(dt_val, (int, np.integer)) else dt_val,
                "forecast_date": pd.Timestamp(dt_val)
                if not isinstance(dt_val, (int, np.integer))
                else dt_val,
                "horizon_step": int(orig_r.get("horizon_step", t_idx + 1)),
                "baseline_yhat": base_v,
                "allocated_yhat": alloc_v,
                "yhat": alloc_v,
                "yhat_lower": alloc_lo,
                "yhat_upper": alloc_hi,
                "delta_yhat": d_v,
                "delta_units": d_v,
                "delta_pct": d_pct,
                "lower_bound": lo_v,
                "upper_bound": hi_v,
                "is_locked": is_lck,
                "allocation_weight": float(norm_b_weights[b_idx_opt])
                if b_idx_opt is not None
                else float(node_weights[n_idx]),
                "constraint_status": c_status,
                "yhat_baseline": base_v,
                "yhat_allocated": alloc_v,
                "y_allocated": alloc_v,
                "bound_active": c_status != "FREE",
                "allocation_share": share_t,
            }

            # Shift any explicit quantile columns (`yhat_lower_80`, `yhat_upper_95`, etc.).
            for col_k, col_v in orig_r.items():
                if col_k.startswith("yhat_lower_") and col_v is not None and pd.notna(col_v):
                    q_suf = col_k[len("yhat_lower_") :]
                    u_k = f"yhat_upper_{q_suf}"
                    u_v = orig_r.get(u_k)
                    if u_v is not None and pd.notna(u_v):
                        q_hw = max(0.0, 0.5 * (float(u_v) - float(col_v)))
                        q_lo = min(
                            alloc_v, max(lo_v if np.isfinite(lo_v) else -np.inf, alloc_v - q_hw)
                        )
                        q_hi = max(
                            alloc_v, min(hi_v if np.isfinite(hi_v) else np.inf, alloc_v + q_hw)
                        )
                        rec[col_k] = q_lo
                        rec[u_k] = q_hi
                        rec[f"allocated_{col_k}"] = q_lo
                        rec[f"allocated_{u_k}"] = q_hi

            if has_price or has_cost:
                s_row_n = s_mat[n_idx]
                b_rev_vec = price_mat[:, t_idx] * b_hat[:, t_idx]
                a_rev_vec = price_mat[:, t_idx] * b_alloc[:, t_idx]
                b_rev = (
                    float(b_rev_vec[b_idx_opt])
                    if is_bottom and b_idx_opt is not None
                    else float(s_row_n @ b_rev_vec)
                )
                a_rev = (
                    float(a_rev_vec[b_idx_opt])
                    if is_bottom and b_idx_opt is not None
                    else float(s_row_n @ a_rev_vec)
                )
                rev_d = a_rev - b_rev
                rev_pct = (100.0 * rev_d / abs(b_rev)) if abs(b_rev) > 1e-12 else 0.0
                eff_p = (
                    float(price_mat[b_idx_opt, t_idx])
                    if is_bottom and b_idx_opt is not None
                    else ((a_rev / alloc_v) if abs(alloc_v) > 1e-12 else 1.0)
                )
                rec.update(
                    {
                        "price": eff_p,
                        "baseline_revenue": b_rev,
                        "allocated_revenue": a_rev,
                        "scenario_revenue": a_rev,
                        "revenue_delta": rev_d,
                        "delta_revenue": rev_d,
                        "revenue_delta_pct": rev_pct,
                    }
                )
                if has_cost:
                    b_gm_vec = (price_mat[:, t_idx] - cost_mat[:, t_idx]) * b_hat[:, t_idx]
                    a_gm_vec = (price_mat[:, t_idx] - cost_mat[:, t_idx]) * b_alloc[:, t_idx]
                    b_gm = (
                        float(b_gm_vec[b_idx_opt])
                        if is_bottom and b_idx_opt is not None
                        else float(s_row_n @ b_gm_vec)
                    )
                    a_gm = (
                        float(a_gm_vec[b_idx_opt])
                        if is_bottom and b_idx_opt is not None
                        else float(s_row_n @ a_gm_vec)
                    )
                    gm_d = a_gm - b_gm
                    gm_pct = (100.0 * gm_d / abs(b_gm)) if abs(b_gm) > 1e-12 else 0.0
                    eff_c = (
                        float(cost_mat[b_idx_opt, t_idx])
                        if is_bottom and b_idx_opt is not None
                        else (((a_rev - a_gm) / alloc_v) if abs(alloc_v) > 1e-12 else 0.0)
                    )
                    rec.update(
                        {
                            "unit_cost": eff_c,
                            "baseline_gross_margin": b_gm,
                            "allocated_gross_margin": a_gm,
                            "scenario_gross_margin": a_gm,
                            "gross_margin_delta": gm_d,
                            "delta_gross_margin": gm_d,
                            "gross_margin_delta_pct": gm_pct,
                        }
                    )

            records.append(rec)

    out_df = pd.DataFrame.from_records(records)

    # Optional inventory replenishment policy delta via diagnostics.compute_inventory_policy.
    if (include_inventory_impact or inventory_params is not None) and not out_df.empty:
        from .diagnostics import compute_inventory_policy

        inv_kw: dict[str, Any] = {
            "lead_time": float(lead_time),
            "lead_time_std": float(lead_time_std),
            "service_level": service_level,
        }
        if inventory_params:
            inv_kw.update(dict(inventory_params))

        base_for_inv = out_df.copy()
        base_for_inv["yhat"] = base_for_inv["baseline_yhat"]
        hw_curr = 0.5 * (
            out_df["yhat_upper"].to_numpy(dtype=float) - out_df["yhat_lower"].to_numpy(dtype=float)
        )
        base_for_inv["yhat_lower"] = base_for_inv["baseline_yhat"] - hw_curr
        base_for_inv["yhat_upper"] = base_for_inv["baseline_yhat"] + hw_curr

        base_pol = compute_inventory_policy(base_for_inv, oof_df=oof_df, **inv_kw)
        alloc_pol = compute_inventory_policy(out_df, oof_df=oof_df, **inv_kw)

        base_inv_map = {str(r["ts_id"]): r for r in base_pol.to_dict(orient="records")}
        alloc_inv_map = {str(r["ts_id"]): r for r in alloc_pol.to_dict(orient="records")}

        b_rop: list[float] = []
        a_rop: list[float] = []
        d_rop: list[float] = []
        b_ss: list[float] = []
        a_ss: list[float] = []
        d_ss: list[float] = []
        b_q: list[float] = []
        a_q: list[float] = []
        d_q: list[float] = []

        for tid_val in out_df[ts_id_col].astype(str):
            br = base_inv_map.get(tid_val, {})
            ar = alloc_inv_map.get(tid_val, {})
            rop_0 = float(br.get("reorder_point", 0.0))
            rop_1 = float(ar.get("reorder_point", rop_0))
            ss_0 = float(br.get("safety_stock", 0.0))
            ss_1 = float(ar.get("safety_stock", ss_0))
            q_0 = float(br.get("recommended_order_qty", 0.0))
            q_1 = float(ar.get("recommended_order_qty", q_0))
            b_rop.append(rop_0)
            a_rop.append(rop_1)
            d_rop.append(rop_1 - rop_0)
            b_ss.append(ss_0)
            a_ss.append(ss_1)
            d_ss.append(ss_1 - ss_0)
            b_q.append(q_0)
            a_q.append(q_1)
            d_q.append(q_1 - q_0)

        out_df["baseline_reorder_point"] = b_rop
        out_df["allocated_reorder_point"] = a_rop
        out_df["scenario_reorder_point"] = a_rop
        out_df["delta_reorder_point"] = d_rop
        out_df["baseline_safety_stock"] = b_ss
        out_df["allocated_safety_stock"] = a_ss
        out_df["scenario_safety_stock"] = a_ss
        out_df["delta_safety_stock"] = d_ss
        out_df["baseline_order_qty"] = b_q
        out_df["allocated_order_qty"] = a_q
        out_df["scenario_order_qty"] = a_q
        out_df["delta_order_qty"] = d_q

    coh_err = float(np.max(np.abs(a_eq @ b_alloc_flat - c_eq)))
    out_df.attrs["hierarchy_spec"] = resolved_spec
    out_df.attrs["target_node"] = eff_target_node
    out_df.attrs["target_total"] = float(np.sum(c_eq))
    out_df.attrs["coherence_max_abs_error"] = coh_err
    out_df.attrs["smoothness_weight"] = smooth_w
    if return_result:
        sum_df = summarize_goal_seek(out_df, ts_id_col=ts_id_col)
        return GoalSeekResult(
            allocations=out_df,
            summary=sum_df,
            target_node=eff_target_node,
            target_total=float(np.sum(c_eq)),
            baseline_total=float(np.sum(y_base_all[target_node_idx, active_t_indices])),
            allocated_total=float(np.sum(y_alloc_all[target_node_idx, active_t_indices])),
            coherence_max_abs_error=coh_err,
            active_bound_count=int(np.sum(out_df["bound_active"])),
            locked_count=int(np.sum(out_df["is_locked"])),
        )
    return out_df


def summarize_goal_seek(
    goal_seek_df: pd.DataFrame | GoalSeekResult,
    *,
    ts_id_col: str = "ts_id",
) -> pd.DataFrame:
    """Aggregate per-series goal-seek allocations across the forecast horizon (pure)."""
    if isinstance(goal_seek_df, GoalSeekResult):
        goal_seek_df = goal_seek_df.allocations
    if goal_seek_df is None or goal_seek_df.empty:
        return pd.DataFrame(
            columns=[
                ts_id_col,
                "model_type",
                "baseline_total_units",
                "allocated_total_units",
                "y_allocated_sum",
                "delta_units",
                "delta_pct",
                "allocation_weight",
                "constraint_status",
            ]
        )

    id_col = ts_id_col if ts_id_col in goal_seek_df.columns else "ts_id"
    rows: list[dict[str, Any]] = []
    for tid, grp in goal_seek_df.groupby(id_col, sort=False):
        b_tot = float(grp["baseline_yhat"].sum())
        a_tot = float(grp["allocated_yhat"].sum())
        d_u = a_tot - b_tot
        d_pct = (100.0 * d_u / abs(b_tot)) if abs(b_tot) > 1e-12 else 0.0
        statuses = (
            set(grp["constraint_status"].astype(str).tolist())
            if "constraint_status" in grp.columns
            else {"FREE"}
        )
        if "LOCKED" in statuses:
            dom_status = "LOCKED"
        elif "AT_CAP" in statuses:
            dom_status = "AT_CAP"
        elif "AT_FLOOR" in statuses:
            dom_status = "AT_FLOOR"
        else:
            dom_status = "FREE"

        rec: dict[str, Any] = {
            id_col: str(tid),
            "model_type": str(grp["model_type"].iloc[0])
            if "model_type" in grp.columns
            else "default",
            "baseline_total_units": b_tot,
            "allocated_total_units": a_tot,
            "y_allocated_sum": a_tot,
            "baseline_yhat": b_tot,
            "allocated_yhat": a_tot,
            "delta_units": d_u,
            "delta_yhat": d_u,
            "delta_pct": d_pct,
            "allocation_weight": float(grp["allocation_weight"].mean())
            if "allocation_weight" in grp.columns
            else 0.0,
            "constraint_status": dom_status,
        }
        if "baseline_revenue" in grp.columns and "allocated_revenue" in grp.columns:
            b_rev = float(grp["baseline_revenue"].sum())
            a_rev = float(grp["allocated_revenue"].sum())
            rec["baseline_revenue"] = b_rev
            rec["allocated_revenue"] = a_rev
            rec["revenue_delta"] = a_rev - b_rev
            rec["delta_revenue"] = a_rev - b_rev
        if "baseline_gross_margin" in grp.columns and "allocated_gross_margin" in grp.columns:
            b_gm = float(grp["baseline_gross_margin"].sum())
            a_gm = float(grp["allocated_gross_margin"].sum())
            rec["baseline_gross_margin"] = b_gm
            rec["allocated_gross_margin"] = a_gm
            rec["gross_margin_delta"] = a_gm - b_gm
            rec["delta_gross_margin"] = a_gm - b_gm
        if "delta_reorder_point" in grp.columns:
            rec["baseline_reorder_point"] = float(grp["baseline_reorder_point"].iloc[0])
            rec["allocated_reorder_point"] = float(grp["allocated_reorder_point"].iloc[0])
            rec["delta_reorder_point"] = float(grp["delta_reorder_point"].iloc[0])
            rec["baseline_safety_stock"] = float(grp["baseline_safety_stock"].iloc[0])
            rec["allocated_safety_stock"] = float(grp["allocated_safety_stock"].iloc[0])
            rec["delta_safety_stock"] = float(grp["delta_safety_stock"].iloc[0])
            rec["baseline_order_qty"] = float(grp["baseline_order_qty"].iloc[0])
            rec["allocated_order_qty"] = float(grp["allocated_order_qty"].iloc[0])
            rec["delta_order_qty"] = float(grp["delta_order_qty"].iloc[0])
        rows.append(rec)

    return pd.DataFrame.from_records(rows)


def plot_goal_seek(
    goal_seek_df: pd.DataFrame | GoalSeekResult,
    *,
    ts_id: str | None = None,
    top_n: int = 10,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot baseline vs. goal-seek allocated forecasts (bar summary or single-series trajectory)."""
    import matplotlib.pyplot as plt

    if isinstance(goal_seek_df, GoalSeekResult):
        goal_seek_df = goal_seek_df.allocations
    if goal_seek_df is None or goal_seek_df.empty:
        raise DataError("plot_goal_seek requires a non-empty goal_seek_df DataFrame")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        created_fig = True
    else:
        fig = ax.figure

    id_col = "ts_id" if "ts_id" in goal_seek_df.columns else goal_seek_df.columns[0]
    dcol = (
        "ds"
        if "ds" in goal_seek_df.columns
        else ("forecast_date" if "forecast_date" in goal_seek_df.columns else None)
    )

    unique_ids = goal_seek_df[id_col].astype(str).unique().tolist()
    if ts_id is not None or len(unique_ids) == 1:
        target_id = str(ts_id) if ts_id is not None else unique_ids[0]
        sub = goal_seek_df[goal_seek_df[id_col].astype(str) == target_id].copy()
        if sub.empty:
            raise DataError(f"ts_id='{target_id}' not found in goal_seek_df")
        x_vals: Any
        if dcol is not None:
            sub = sub.sort_values(dcol, kind="mergesort")
            x_vals = sub[dcol]
        else:
            x_vals = np.arange(len(sub))

        ax.plot(
            x_vals,
            sub["baseline_yhat"],
            label="Baseline Forecast",
            linestyle="--",
            color="#64748b",
            linewidth=1.8,
        )
        ax.plot(
            x_vals,
            sub["allocated_yhat"],
            label="Goal-Seek Allocated",
            color="#2563eb",
            linewidth=2.2,
        )
        if "yhat_lower" in sub.columns and "yhat_upper" in sub.columns:
            ax.fill_between(
                x_vals,
                sub["yhat_lower"].to_numpy(dtype=float),
                sub["yhat_upper"].to_numpy(dtype=float),
                color="#2563eb",
                alpha=0.15,
                label="Allocated 80% PI",
            )
        ax.set_title(title or f"Top-Down Goal-Seek Allocation — {target_id}")
        ax.set_ylabel("Forecast Units")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    else:
        summary = summarize_goal_seek(goal_seek_df, ts_id_col=id_col)
        non_total = summary[summary[id_col] != TOTAL_NODE_ID]
        if not non_total.empty:
            summary = non_total
        summary = summary.reindex(
            summary["delta_units"].abs().sort_values(ascending=False).index
        ).head(max(1, int(top_n)))

        y_pos = np.arange(len(summary))
        height = 0.36
        ax.barh(
            y_pos - height / 2,
            summary["baseline_total_units"],
            height=height,
            label="Baseline Total",
            color="#94a3b8",
        )
        ax.barh(
            y_pos + height / 2,
            summary["allocated_total_units"],
            height=height,
            label="Allocated Total",
            color="#2563eb",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary[id_col].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Horizon Total Units")
        ax.set_title(title or "Top-Down Goal-Seek Disaggregation across Series")
        ax.legend(loc="best")
        ax.grid(True, axis="x", alpha=0.25)

    if created_fig:
        fig.tight_layout()
    return fig


allocate_forecast_target = allocate_target
