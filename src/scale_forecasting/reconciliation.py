"""Hierarchical and grouped time-series aggregation and coherent forecast reconciliation.

Implements the Hyndman & Athanasopoulos (*Forecasting: Principles and Practice*, 3rd ed., Ch. 11)
and Wickramasuriya, Athanasopoulos & Hyndman (2019) optimal reconciliation framework in pure
pandas/NumPy:

1. **Hierarchy aggregation (`build_hierarchy`)** — Given a bottom-level panel and
   `config.HierarchyConfig`, constructs the aggregated series for the total node (``"__total__"``)
   and every cross-sectional level in ``cfg.hierarchy.levels`` (both strictly nested hierarchies
   and cross-classified grouped structures), together with the binary summing matrix
   ``S`` of shape ``(n_total, n_bottom)`` satisfying ``y_t = S @ b_t``.
2. **Reconciliation projection (`reconcile_matrix`)** — Computes the mapping matrix ``G`` of shape
   ``(n_bottom, n_total)`` such that ``y_tilde = S @ G @ y_hat`` is coherent across the entire
   hierarchy for all seven supported methods:

   * ``bottom_up`` — sum bottom-level base forecasts upward (``G = [0 | I]``).
   * ``top_down`` — disaggregate the top-level base forecast by average historical proportions
     (FPP3 §11.2).
   * ``middle_out`` — anchor on an intermediate ``middle_level``, summing upward above it and
     disaggregating downward to the bottom series via historical proportions within each
     middle-level node.
   * ``ols`` — ordinary least squares identity-weighted MinT (``W_h = I``).
   * ``wls_struct`` — structural scaling WLS (``W_h = diag(S @ 1)``), requiring only ``S``.
   * ``wls_var`` — variance scaling WLS (``W_h = diag(W_hat_1)``) from 1-step/OOF residual
     variances.
   * ``mint_shrink`` — Minimum Trace with analytical Schäfer-Strimmer shrinkage covariance of
     residuals (``W_h = lambda_D * diag(W_hat_1) + (1 - lambda_D) * W_hat_1``), guaranteed
     symmetric positive-definite even when ``n_series >> T_obs``.
3. **Forecast, OOF, and cell reconciliation (`reconcile_forecasts`, `reconcile_oof`,
   `reconcile_cells`, `verify_coherence`)** — Projects point forecasts (`yhat`, `yhat_raw`) via
   ``P = S @ G`` and prediction intervals (`yhat_lower`, `yhat_upper`) via the projected error
   variance ``diag(P @ W_h @ P.T)``, then re-scores reconciled cells on the backtest OOF window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .backtest import holdout_fold_id
from .config import RECONCILIATION_METHODS, ReconciliationMethod, RunConfig
from .errors import ConfigError, DataError

if TYPE_CHECKING:
    from .worker import CellResult

TOTAL_NODE_ID = "__total__"
# 80% normal quantile (z_{0.90}) matching the suite's 80% prediction interval convention.
_Z_80 = 1.2815515655446004
_VAR_FLOOR = 1e-8


@dataclass(frozen=True)
class HierarchySpec:
    """The structural definition of a hierarchical or grouped time-series panel.

    Attributes:
        node_ids: Ordered tuple of all ``n`` series identifiers in the hierarchy
            (``n_aggregated`` upper nodes followed by ``n_bottom`` bottom-level series).
        bottom_ids: Ordered tuple of the ``n_bottom`` leaf series identifiers.
        level_nodes: Mapping from level key (``"__total__"``, ``"region"``,
            ``"region/category"``, ``"__bottom__"``) to the ordered node identifiers at that level.
        summing_matrix: Binary matrix ``S`` of shape ``(n_nodes, n_bottom)`` with ``S[i, j] = 1.0``
            iff bottom series ``j`` aggregates into node ``i``. The bottom ``n_bottom`` rows form
            the identity matrix ``I_{n_bottom}``.
    """

    node_ids: tuple[str, ...]
    bottom_ids: tuple[str, ...]
    level_nodes: dict[str, tuple[str, ...]]
    summing_matrix: np.ndarray

    @property
    def n_nodes(self) -> int:
        """Total number of series in the hierarchy (`n_aggregated + n_bottom`)."""
        return len(self.node_ids)

    @property
    def n_bottom(self) -> int:
        """Number of bottom-level leaf series (`n_b`)."""
        return len(self.bottom_ids)

    @property
    def n_aggregated(self) -> int:
        """Number of upper-level aggregated series (`n_a = n_nodes - n_bottom`)."""
        return len(self.node_ids) - len(self.bottom_ids)

    @property
    def structural_weights(self) -> np.ndarray:
        """Number of bottom-level series summed into each node (`w = S @ 1_{n_b}`)."""
        return self.summing_matrix.sum(axis=1)


def _level_key(cols: Sequence[str]) -> str:
    return "/".join(cols)


def _format_node_id(cols: Sequence[str], values: tuple[object, ...]) -> str:
    return "/".join(f"{col}={val}" for col, val in zip(cols, values, strict=True))


def build_hierarchy(
    panel: pd.DataFrame,
    cfg: RunConfig,
) -> tuple[pd.DataFrame, HierarchySpec]:
    """Aggregate a bottom-level panel across ``cfg.hierarchy.levels`` and build ``S`` (pure).

    Given a bottom-level DataFrame containing ``cfg.data.ts_id_col``, ``cfg.data.date_col``,
    ``cfg.data.target_col``, the hierarchy attribute columns named in ``cfg.hierarchy.levels``, and
    any declared covariates:

    1. Prepends the top-level total series (``ts_id == "__total__"``).
    2. Builds one aggregated series per distinct group tuple at each level in
       ``cfg.hierarchy.levels`` (skipping any level whose node IDs would duplicate an already-added
       level). Target values (``y``) are summed across constituent bottom series at each date;
       numeric covariate columns are averaged across constituent bottom series so rates and
       bounded indicators stay on scale.
    3. Appends the bottom-level leaf series and constructs the binary summing matrix
       ``S`` of shape ``(n_total, n_bottom)``.

    Returns ``(hierarchical_panel, HierarchySpec)``.
    """
    if not cfg.hierarchy.levels:
        raise ConfigError(
            "build_hierarchy requires at least one level in cfg.hierarchy.levels "
            "(e.g. [['region'], ['region', 'category']])"
        )

    id_col = cfg.data.ts_id_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col

    required_cols = {id_col, date_col, target_col}
    for level in cfg.hierarchy.levels:
        required_cols.update(level)
    missing = sorted(required_cols - set(panel.columns))
    if missing:
        raise DataError(f"Panel is missing required hierarchy/data columns: {missing}")

    # Filter out any pre-existing aggregate nodes if build_hierarchy is called idempotently.
    bottom_df = panel[~panel[id_col].astype(str).str.startswith(TOTAL_NODE_ID)].copy()
    bottom_df[id_col] = bottom_df[id_col].astype(str)
    bottom_ids = tuple(sorted(bottom_df[id_col].unique()))
    if not bottom_ids:
        raise DataError("Cannot build hierarchy from an empty bottom-level panel")

    bottom_idx = {ts_id: idx for idx, ts_id in enumerate(bottom_ids)}
    n_b = len(bottom_ids)

    # Map each bottom series to its hierarchy attributes (one row per bottom series).
    attr_cols = sorted({col for level in cfg.hierarchy.levels for col in level})
    series_attrs = (
        bottom_df[[id_col, *attr_cols]]
        .drop_duplicates(subset=[id_col], keep="first")
        .set_index(id_col)
    )
    if len(series_attrs) != n_b:
        raise DataError("Each bottom-level series must have unambiguous hierarchy attributes")

    cov_cols = [
        c
        for c in cfg.features.all_covariates
        if c in bottom_df.columns and c not in {id_col, date_col, target_col}
    ]

    agg_frames: list[pd.DataFrame] = []
    s_rows: list[np.ndarray] = []
    node_ids: list[str] = []
    level_nodes: dict[str, tuple[str, ...]] = {}

    def _aggregate_group(group_df: pd.DataFrame, node_id: str) -> pd.DataFrame:
        grouped = group_df.groupby(date_col, as_index=False, sort=True)
        agg_spec: dict[str, str] = {target_col: "sum"}
        non_num_vals: dict[str, object] = {}
        for cov in cov_cols:
            if pd.api.types.is_numeric_dtype(group_df[cov]):
                agg_spec[cov] = "mean"
            else:
                uniq = group_df[cov].dropna().unique()
                non_num_vals[cov] = uniq[0] if len(uniq) == 1 else TOTAL_NODE_ID
        out = grouped.agg(agg_spec)
        for cov, val in non_num_vals.items():
            out[cov] = val
        out[id_col] = node_id
        return out[[id_col, date_col, target_col, *[c for c in cov_cols if c in out.columns]]]

    # 1. Total node ("__total__").
    agg_frames.append(_aggregate_group(bottom_df, TOTAL_NODE_ID))
    s_rows.append(np.ones(n_b, dtype=float))
    node_ids.append(TOTAL_NODE_ID)
    level_nodes[TOTAL_NODE_ID] = (TOTAL_NODE_ID,)

    # 2. Intermediate levels in cfg.hierarchy.levels.
    seen_nodes: set[str] = {TOTAL_NODE_ID, *bottom_ids}
    for level in cfg.hierarchy.levels:
        lkey = _level_key(level)
        current_level_nodes: list[str] = []
        for raw_vals, grp in bottom_df.groupby(list(level), sort=True):
            vals = raw_vals if isinstance(raw_vals, tuple) else (raw_vals,)
            nid = _format_node_id(level, vals)
            if nid in seen_nodes:
                raise DataError(
                    f"Hierarchy node ID collision on '{nid}' at level {level}; "
                    "ensure level column names and values produce distinct node IDs."
                )
            seen_nodes.add(nid)
            current_level_nodes.append(nid)
            node_ids.append(nid)
            agg_frames.append(_aggregate_group(grp, nid))

            member_ids = grp[id_col].unique()
            row = np.zeros(n_b, dtype=float)
            for mid in member_ids:
                row[bottom_idx[str(mid)]] = 1.0
            s_rows.append(row)
        level_nodes[lkey] = tuple(current_level_nodes)

    # 3. Bottom level.
    keep_cols = [id_col, date_col, target_col, *cov_cols]
    bottom_clean = (
        bottom_df[keep_cols]
        .sort_values([id_col, date_col], kind="mergesort")
        .reset_index(drop=True)
    )
    node_ids.extend(bottom_ids)
    level_nodes["__bottom__"] = bottom_ids
    s_bottom = np.eye(n_b, dtype=float)
    s_matrix = np.vstack([*s_rows, s_bottom]) if s_rows else s_bottom

    full_panel = pd.concat([*agg_frames, bottom_clean], ignore_index=True)
    spec = HierarchySpec(
        node_ids=tuple(node_ids),
        bottom_ids=bottom_ids,
        level_nodes=level_nodes,
        summing_matrix=s_matrix,
    )
    return full_panel, spec


def schafer_strimmer_shrinkage(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the Schäfer-Strimmer (2005) shrinkage covariance matrix and shrinkage intensity.

    Given a residual matrix ``E`` of shape ``(T, n)`` (rows = time points, columns = series),
    shrinks the sample covariance ``W_hat = (E.T @ E) / T`` toward its diagonal variance target
    ``W_diag = diag(diag(W_hat))``:

        ``W_shrink = lambda_D * W_diag + (1 - lambda_D) * W_hat``

    where the analytical shrinkage intensity ``lambda_D in [0, 1]`` is estimated from the sample
    correlations ``r_ij`` of the standardized residuals (Wickramasuriya et al. 2019, §2.4).
    """
    e = np.asarray(residuals, dtype=float)
    if e.ndim != 2:
        raise ValueError(f"residuals must be a 2-D array of shape (T, n), got shape {e.shape}")
    n_obs, n_series = e.shape
    if n_obs == 0:
        return np.eye(n_series, dtype=float), 1.0

    # Replace any non-finite residual entries with 0.0 (unobserved dates in ragged panels).
    e_clean = np.where(np.isfinite(e), e, 0.0)
    w_sample = (e_clean.T @ e_clean) / max(1, n_obs)
    diag_var = np.maximum(np.diag(w_sample), _VAR_FLOOR)
    w_diag = np.diag(diag_var)

    if n_obs <= 2 or n_series <= 1:
        return w_diag, 1.0

    std = np.sqrt(diag_var)
    z = e_clean / std[np.newaxis, :]
    # Sample correlation matrix R = (Z.T @ Z) / T and pairwise products w_k = z_{k,i} * z_{k,j}.
    corr = (z.T @ z) / n_obs
    np.fill_diagonal(corr, 1.0)

    # Analytical sum_{i != j} Var(r_ij) without materializing a (T, n, n) tensor:
    # sum_{t} (z_{t,i} z_{t,j} - r_ij)^2 = sum_t z_{t,i}^2 z_{t,j}^2 - T * r_ij^2.
    z2 = z**2
    sum_z2_prod = z2.T @ z2
    sum_sq_dev = np.maximum(0.0, sum_z2_prod - n_obs * (corr**2))
    var_r = (n_obs / ((n_obs - 1.0) ** 3)) * sum_sq_dev

    off_diag = ~np.eye(n_series, dtype=bool)
    num = float(var_r[off_diag].sum())
    den = float((corr[off_diag] ** 2).sum())
    lam = 1.0 if den <= 1e-12 else float(np.clip(num / den, 0.0, 1.0))

    w_shrink = lam * w_diag + (1.0 - lam) * w_sample
    # Guarantee strict diagonal positivity.
    idx = np.arange(n_series)
    w_shrink[idx, idx] = np.maximum(w_shrink[idx, idx], _VAR_FLOOR)
    return w_shrink, lam


def _historical_proportions(
    history_df: pd.DataFrame | None,
    bottom_ids: Sequence[str],
    parent_member_indices: Sequence[int],
    *,
    date_col: str,
    ts_id_col: str,
    target_col: str,
) -> np.ndarray:
    """Average historical proportions ``p_j = mean_t(y_{j,t} / sum_{k in parent} y_{k,t})``.

    Follows Hyndman & Athanasopoulos (FPP3 §11.2, Eq. 11.2). Falls back to uniform proportions
    ``1 / len(parent_member_indices)`` when history is unavailable or the parent sum is zero.
    """
    m = len(parent_member_indices)
    if m == 0:
        return np.empty(0, dtype=float)
    uniform = np.full(m, 1.0 / m, dtype=float)
    if history_df is None or history_df.empty:
        return uniform

    member_ids = [bottom_ids[i] for i in parent_member_indices]
    sub = history_df[history_df[ts_id_col].astype(str).isin(member_ids)]
    if sub.empty:
        return uniform

    wide = (
        sub.pivot_table(index=date_col, columns=ts_id_col, values=target_col, aggfunc="sum")
        .reindex(columns=member_ids)
        .fillna(0.0)
    )
    vals = wide.to_numpy(dtype=float)
    totals = vals.sum(axis=1)
    valid = np.isfinite(totals) & (np.abs(totals) > 1e-12)
    if not valid.any():
        return uniform

    ratios = vals[valid] / totals[valid, np.newaxis]
    props = ratios.mean(axis=0)
    props = np.maximum(props, 0.0)
    total_prop = float(props.sum())
    if not np.isfinite(total_prop) or total_prop <= 1e-12:
        return uniform
    return props / total_prop


def _residual_matrix(
    residuals_df: pd.DataFrame | None,
    node_ids: Sequence[str],
    *,
    date_col: str = "forecast_date",
    ts_id_col: str = "ts_id",
) -> np.ndarray | None:
    """Pivot a residual/OOF DataFrame into a ``(T, n_nodes)`` matrix aligned to ``node_ids``."""
    if residuals_df is None or residuals_df.empty:
        return None
    df = residuals_df.copy()
    if "residual" not in df.columns:
        if {"y_true", "yhat"}.issubset(df.columns):
            df["residual"] = df["y_true"].astype(float) - df["yhat"].astype(float)
        else:
            return None
    dcol = date_col if date_col in df.columns else ("ds" if "ds" in df.columns else None)
    if dcol is None:
        return None
    index_cols = ["cutoff_date", dcol] if "cutoff_date" in df.columns else [dcol]
    wide = (
        df.pivot_table(index=index_cols, columns=ts_id_col, values="residual", aggfunc="mean")
        .reindex(columns=list(node_ids))
        .dropna(how="all")
    )
    if wide.empty:
        return None
    return wide.fillna(0.0).to_numpy(dtype=float)


def reconcile_matrix(
    spec: HierarchySpec,
    method: ReconciliationMethod,
    *,
    history_df: pd.DataFrame | None = None,
    residuals_df: pd.DataFrame | None = None,
    middle_level: Sequence[str] | None = None,
    date_col: str = "ds",
    ts_id_col: str = "ts_id",
    target_col: str = "y",
) -> np.ndarray:
    """Compute the ``(n_bottom, n_nodes)`` reconciliation matrix ``G`` for ``method`` (pure).

    Every reconciled forecast vector is formed as ``y_tilde = S @ G @ y_hat``. Because ``S`` maps
    the ``n_bottom`` leaf forecasts ``b_tilde = G @ y_hat`` to all ``n_nodes`` series, ``y_tilde``
    is coherent across the hierarchy by construction.
    """
    if method not in RECONCILIATION_METHODS:
        raise ConfigError(
            f"Unknown reconciliation method '{method}'; available: {sorted(RECONCILIATION_METHODS)}"
        )

    s_mat = spec.summing_matrix
    n_nodes, n_bottom = s_mat.shape
    n_agg = n_nodes - n_bottom

    if method == "bottom_up":
        return np.hstack([np.zeros((n_bottom, n_agg), dtype=float), np.eye(n_bottom, dtype=float)])

    if method == "top_down":
        g_mat = np.zeros((n_bottom, n_nodes), dtype=float)
        props = _historical_proportions(
            history_df,
            spec.bottom_ids,
            range(n_bottom),
            date_col=date_col,
            ts_id_col=ts_id_col,
            target_col=target_col,
        )
        g_mat[:, 0] = props
        return g_mat

    if method == "middle_out":
        candidate_levels = [k for k in spec.level_nodes if k not in {TOTAL_NODE_ID, "__bottom__"}]
        if not candidate_levels:
            raise ConfigError("middle_out reconciliation requires at least one intermediate level")
        if middle_level is not None:
            lkey = _level_key(middle_level)
            if lkey not in spec.level_nodes:
                raise ConfigError(
                    f"middle_level '{lkey}' is not in hierarchy levels {candidate_levels}"
                )
        else:
            lkey = candidate_levels[len(candidate_levels) // 2]

        mid_nodes = spec.level_nodes[lkey]
        node_to_idx = {nid: i for i, nid in enumerate(spec.node_ids)}
        g_mat = np.zeros((n_bottom, n_nodes), dtype=float)
        for mid_node in mid_nodes:
            col_idx = node_to_idx[mid_node]
            members = np.flatnonzero(s_mat[col_idx] > 0.5).tolist()
            props = _historical_proportions(
                history_df,
                spec.bottom_ids,
                members,
                date_col=date_col,
                ts_id_col=ts_id_col,
                target_col=target_col,
            )
            for member_idx, prop in zip(members, props, strict=True):
                g_mat[member_idx, col_idx] = prop
        return g_mat

    # Optimal MinT family: G = (S.T @ W^{-1} @ S)^{-1} @ S.T @ W^{-1}
    if method == "ols":
        # W = I_n  =>  G = (S.T @ S)^{-1} @ S.T
        sts = s_mat.T @ s_mat
        return np.linalg.solve(sts, s_mat.T)

    if method == "wls_struct":
        inv_w = 1.0 / np.maximum(spec.structural_weights, _VAR_FLOOR)
        st_winv = s_mat.T * inv_w[np.newaxis, :]
        return np.linalg.solve(st_winv @ s_mat, st_winv)

    res_mat = _residual_matrix(residuals_df, spec.node_ids, ts_id_col=ts_id_col)
    if method == "wls_var":
        if res_mat is None or res_mat.shape[0] == 0:
            inv_w = 1.0 / np.maximum(spec.structural_weights, _VAR_FLOOR)
        else:
            var_diag = np.maximum((res_mat**2).mean(axis=0), _VAR_FLOOR)
            inv_w = 1.0 / var_diag
        st_winv = s_mat.T * inv_w[np.newaxis, :]
        return np.linalg.solve(st_winv @ s_mat, st_winv)

    # method == "mint_shrink"
    if res_mat is None or res_mat.shape[0] == 0:
        inv_w = 1.0 / np.maximum(spec.structural_weights, _VAR_FLOOR)
        st_winv = s_mat.T * inv_w[np.newaxis, :]
        return np.linalg.solve(st_winv @ s_mat, st_winv)

    w_shrink, _lam = schafer_strimmer_shrinkage(res_mat)
    # Solve W_shrink @ X = S for X = W_shrink^{-1} @ S, then G = (S.T @ X)^{-1} @ X.T.
    winv_s = np.linalg.solve(w_shrink, s_mat)
    st_winv_s = s_mat.T @ winv_s
    return np.linalg.solve(st_winv_s, winv_s.T)


def _project_slice(
    slice_df: pd.DataFrame,
    spec: HierarchySpec,
    p_mat: np.ndarray,
    *,
    ts_id_col: str,
) -> pd.DataFrame:
    """Apply the coherent projection ``P = S @ G`` to one ``(model, date)`` cross-section."""
    indexed = slice_df.drop_duplicates(subset=[ts_id_col], keep="first").set_index(ts_id_col)
    aligned = indexed.reindex(list(spec.node_ids))
    if aligned["yhat"].isna().any():
        missing = aligned.index[aligned["yhat"].isna()].tolist()
        raise DataError(
            f"Cannot reconcile incomplete cross-section; missing base forecasts for {missing[:5]}"
        )

    out = aligned.reset_index()
    yhat = out["yhat"].to_numpy(dtype=float)
    yhat_reconciled = p_mat @ yhat
    out["yhat"] = yhat_reconciled

    if "yhat_raw" in out.columns and out["yhat_raw"].notna().all():
        out["yhat_raw"] = p_mat @ out["yhat_raw"].to_numpy(dtype=float)
    if "yhat_adjusted" in out.columns and out["yhat_adjusted"].notna().all():
        out["yhat_adjusted"] = yhat_reconciled

    # Reconcile prediction intervals coherently via Var(y_tilde) = P @ W_h @ P.T.
    if (
        "yhat_lower" in out.columns
        and "yhat_upper" in out.columns
        and out["yhat_lower"].notna().all()
        and out["yhat_upper"].notna().all()
    ):
        half_width = np.maximum(
            0.0,
            0.5
            * (out["yhat_upper"].to_numpy(dtype=float) - out["yhat_lower"].to_numpy(dtype=float)),
        )
        base_var = (half_width / _Z_80) ** 2
        # diag(P @ diag(base_var) @ P.T) == (P**2) @ base_var
        reconciled_var = np.maximum(0.0, (p_mat**2) @ base_var)
        reconciled_hw = _Z_80 * np.sqrt(reconciled_var)
        out["yhat_lower"] = yhat_reconciled - reconciled_hw
        out["yhat_upper"] = yhat_reconciled + reconciled_hw

    return out


def reconcile_forecasts(
    predictions_df: pd.DataFrame,
    spec: HierarchySpec,
    method: ReconciliationMethod,
    *,
    history_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    middle_level: Sequence[str] | None = None,
    date_col: str = "forecast_date",
    ts_id_col: str = "ts_id",
    target_col: str = "y",
    stamp_method: bool = True,
) -> pd.DataFrame:
    """Reconcile base forecasts across ``spec`` using ``method`` (pure).

    Applies ``y_tilde = S @ G @ y_hat`` at every forecast date (and per ``model_type`` when the
     column is present). When ``stamp_method=True`` (default), stamps ``model_type`` as
    ``f"{base_model}_{method}"`` so reconciled forecasts can sit alongside base forecasts in the
    registry and leaderboard.
    """
    if predictions_df.empty:
        return predictions_df.copy()

    dcol = date_col if date_col in predictions_df.columns else "ds"
    has_model_col = "model_type" in predictions_df.columns
    group_Models = (
        predictions_df["model_type"].drop_duplicates().tolist() if has_model_col else [None]
    )

    reconciled_parts: list[pd.DataFrame] = []
    for model_name in group_Models:
        model_preds = (
            predictions_df[predictions_df["model_type"] == model_name]
            if model_name is not None
            else predictions_df
        )
        model_oof = (
            oof_df[oof_df["model_type"] == model_name]
            if (
                oof_df is not None
                and not oof_df.empty
                and "model_type" in oof_df.columns
                and model_name is not None
            )
            else oof_df
        )
        g_mat = reconcile_matrix(
            spec,
            method,
            history_df=history_df,
            residuals_df=model_oof,
            middle_level=middle_level,
            date_col="ds" if (history_df is not None and "ds" in history_df.columns) else dcol,
            ts_id_col=ts_id_col,
            target_col=target_col,
        )
        p_mat = spec.summing_matrix @ g_mat

        for _dt, slice_df in model_preds.groupby(dcol, sort=True):
            rec_slice = _project_slice(slice_df, spec, p_mat, ts_id_col=ts_id_col)
            if stamp_method and model_name is not None:
                rec_slice["model_type"] = f"{model_name}_{method}"
            reconciled_parts.append(rec_slice)

    return (
        pd.concat(reconciled_parts, ignore_index=True)
        if reconciled_parts
        else predictions_df.copy()
    )


def reconcile_oof(
    oof_df: pd.DataFrame,
    spec: HierarchySpec,
    method: ReconciliationMethod,
    *,
    cfg: RunConfig | None = None,
    history_df: pd.DataFrame | None = None,
    middle_level: Sequence[str] | None = None,
    ts_id_col: str = "ts_id",
    target_col: str = "y",
    stamp_method: bool = True,
) -> pd.DataFrame:
    """Reconcile backtest OOF predictions across ``spec`` using ``method`` (pure).

    When ``cfg`` is provided and more than one backtest fold exists, estimates residual-covariance
    weights (`wls_var`, `mint_shrink`) strictly from inner folds (`fold_id != holdout_fold_id(cfg)`)
    so the holdout fold is never leaked into the reconciliation matrix ``G``.
    """
    if oof_df.empty:
        return oof_df.copy()

    dcol = "forecast_date" if "forecast_date" in oof_df.columns else "ds"
    fold_col = "cutoff_date" if "cutoff_date" in oof_df.columns else "fold_id"
    has_model_col = "model_type" in oof_df.columns
    models = oof_df["model_type"].drop_duplicates().tolist() if has_model_col else [None]

    parts: list[pd.DataFrame] = []
    for model_name in models:
        m_oof = oof_df[oof_df["model_type"] == model_name] if model_name is not None else oof_df
        fit_oof = m_oof
        if cfg is not None and "fold_id" in m_oof.columns:
            holdout = holdout_fold_id(cfg)
            inner = m_oof[m_oof["fold_id"] != holdout]
            if not inner.empty:
                fit_oof = inner

        g_mat = reconcile_matrix(
            spec,
            method,
            history_df=history_df,
            residuals_df=fit_oof,
            middle_level=middle_level,
            date_col="ds",
            ts_id_col=ts_id_col,
            target_col=target_col,
        )
        p_mat = spec.summing_matrix @ g_mat

        group_keys = [fold_col, dcol] if fold_col in m_oof.columns else [dcol]
        for _key, slice_df in m_oof.groupby(group_keys, sort=True):
            rec_slice = _project_slice(slice_df, spec, p_mat, ts_id_col=ts_id_col)
            if stamp_method and model_name is not None:
                rec_slice["model_type"] = f"{model_name}_{method}"
            parts.append(rec_slice)

    return pd.concat(parts, ignore_index=True) if parts else oof_df.copy()


def reconcile_cells(
    results: Sequence[CellResult],
    cfg: RunConfig,
    spec: HierarchySpec,
    *,
    history_df: pd.DataFrame | None = None,
) -> list[CellResult]:
    """Reconcile a full hierarchy's ``CellResult``s across ``cfg.hierarchy.reconciliation_methods``.

    Takes the base ``CellResult``s for all ``n_nodes`` series in ``spec`` and returns new
    ``CellResult``s for every ``(base_model, reconciliation_method)`` pair, with ``model_type``
    stamped as ``f"{base_model}_{method}"``, coherent ``predictions`` and ``oof`` frames, and
    freshly computed backtest ``metrics`` (`metrics.compute_metrics`).
    """
    from .metrics import compute_metrics
    from .registry.ids import make_model_hash
    from .seasonality import seasonal_period

    if not results or not cfg.hierarchy.enabled:
        return []

    id_col = cfg.data.ts_id_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col
    period = seasonal_period(cfg.data.freq)

    # Index history per ts_id for MASE/RMSSE/MSIS scaling when scoring reconciled OOF.
    y_train_by_id: dict[str, np.ndarray] = {}
    if history_df is not None and not history_df.empty:
        for tid, grp in history_df.groupby(id_col, sort=False):
            y_train_by_id[str(tid)] = grp.sort_values(date_col)[target_col].to_numpy(dtype=float)

    by_model: dict[str, dict[str, CellResult]] = {}
    for res in results:
        if res.status == "ok":
            by_model.setdefault(res.model_type, {})[res.ts_id] = res

    reconciled_results: list[CellResult] = []
    for base_model, cell_map in by_model.items():
        if set(spec.node_ids) - set(cell_map):
            continue

        pred_frames: list[pd.DataFrame] = []
        oof_frames: list[pd.DataFrame] = []
        for nid in spec.node_ids:
            cell = cell_map[nid]
            pf = cell.predictions.copy()
            pf[id_col] = nid
            pf["model_type"] = base_model
            pred_frames.append(pf)
            if cell.oof is not None and not cell.oof.empty:
                of = cell.oof.copy()
                of[id_col] = nid
                of["model_type"] = base_model
                oof_frames.append(of)

        all_preds = pd.concat(pred_frames, ignore_index=True)
        all_oof = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()

        for method in cfg.hierarchy.reconciliation_methods:
            rec_preds = reconcile_forecasts(
                all_preds,
                spec,
                method,
                history_df=history_df,
                oof_df=all_oof if not all_oof.empty else None,
                middle_level=cfg.hierarchy.middle_level,
                date_col="ds",
                ts_id_col=id_col,
                target_col=target_col,
                stamp_method=True,
            )
            rec_oof = (
                reconcile_oof(
                    all_oof,
                    spec,
                    method,
                    cfg=cfg,
                    history_df=history_df,
                    middle_level=cfg.hierarchy.middle_level,
                    ts_id_col=id_col,
                    target_col=target_col,
                    stamp_method=True,
                )
                if not all_oof.empty
                else pd.DataFrame()
            )

            rec_model_name = f"{base_model}_{method}"
            preds_by_id = {
                str(tid): grp.drop(columns=[id_col, "model_type"], errors="ignore").reset_index(
                    drop=True
                )
                for tid, grp in rec_preds.groupby(id_col, sort=False)
            }
            oof_by_id = (
                {
                    str(tid): grp.drop(columns=[id_col, "model_type"], errors="ignore").reset_index(
                        drop=True
                    )
                    for tid, grp in rec_oof.groupby(id_col, sort=False)
                }
                if not rec_oof.empty
                else {}
            )

            for nid in spec.node_ids:
                base_cell = cell_map[nid]
                node_oof = oof_by_id.get(nid)
                if (
                    node_oof is not None
                    and not node_oof.empty
                    and {"y_true", "yhat"}.issubset(node_oof.columns)
                ):
                    lower = (
                        node_oof["yhat_lower"].to_numpy(dtype=float)
                        if "yhat_lower" in node_oof.columns
                        else None
                    )
                    upper = (
                        node_oof["yhat_upper"].to_numpy(dtype=float)
                        if "yhat_upper" in node_oof.columns
                        else None
                    )
                    node_metrics = compute_metrics(
                        node_oof["y_true"].to_numpy(dtype=float),
                        node_oof["yhat"].to_numpy(dtype=float),
                        lower=lower,
                        upper=upper,
                        y_train=y_train_by_id.get(nid),
                        seasonal_period=period,
                    )
                else:
                    node_metrics = dict(base_cell.metrics)

                reconciled_results.append(
                    replace(
                        base_cell,
                        model_type=rec_model_name,
                        model_hash=make_model_hash(base_cell.run_id, nid, rec_model_name, cfg),
                        predictions=preds_by_id[nid],
                        oof=node_oof if node_oof is not None else base_cell.oof,
                        metrics=node_metrics,
                    )
                )

    return reconciled_results


def verify_coherence(
    df: pd.DataFrame,
    spec: HierarchySpec,
    *,
    value_col: str = "yhat",
    date_col: str = "forecast_date",
    ts_id_col: str = "ts_id",
    atol: float = 1e-6,
) -> float:
    """Verify exact additive coherence ``y_t == S @ b_t`` across all timestamps and models.

    Returns the maximum absolute discrepancy ``max_{m, t} ||y_{m, t} - S @ b_{m, t}||_inf`` and
    raises `DataError` if it exceeds ``atol``.
    """
    if df.empty:
        return 0.0
    dcol = date_col if date_col in df.columns else "ds"
    group_cols = ["model_type", dcol] if "model_type" in df.columns else [dcol]
    s_mat = spec.summing_matrix
    n_agg = spec.n_aggregated

    max_err = 0.0
    for _key, grp in df.groupby(group_cols, sort=False):
        indexed = grp.drop_duplicates(subset=[ts_id_col], keep="first").set_index(ts_id_col)
        vals = indexed.reindex(list(spec.node_ids))[value_col].to_numpy(dtype=float)
        bottom_vals = vals[n_agg:]
        expected = s_mat @ bottom_vals
        err = float(np.max(np.abs(vals - expected)))
        if not np.isfinite(err) or err > atol:
            raise DataError(
                "Forecasts are not coherent across hierarchy "
                f"(max discrepancy {err:.3e} > {atol:.3e})"
            )
        max_err = max(max_err, err)
    return max_err
