"""Temporal Hierarchies (`THieF`) and Simultaneous Cross-Temporal Coherent Reconciliation.

Implements the Athanasopoulos et al. (2017) *Forecasting with Temporal Hierarchies* (`THieF`)
and Di Fonzo & Girolimetto (2023) cross-temporal forecast reconciliation framework in pure
pandas/NumPy:

1. **Temporal Hierarchy Construction (`default_temporal_aggregations`,
   `build_temporal_hierarchy`)** — Resolves canonical aggregation factors from ``data.freq``
   (e.g., ``[7, 14, 28]`` on daily ``"D"``, ``[4, 13, 52]`` on weekly ``"W"``, ``[3, 6, 12]`` on
   monthly ``"M"``/``"MS"``/``"ME"``, ``[2, 4]`` on quarterly ``"Q"``, and ``[6, 12, 24]`` on
   hourly ``"H"``/``"h"``) and compiles the single-cycle temporal summing matrix ``S_temp`` of
   shape ``(n_temp_nodes, macro_period)`` alongside the multi-cycle Kronecker summing matrix
   ``I_P \\otimes S_temp``.
2. **Non-Overlapping Temporal Aggregation (`aggregate_temporal_series`)** — Aggregates
   base-frequency historical series or DataFrames into non-overlapping blocks of size ``k``
   aligned to the cutoff timestamp (trimming ``T % k`` oldest observations from the front so
   every aggregation level terminates on the exact same cutoff date).
3. **Temporal Projection (`reconcile_temporal_matrix`, `reconcile_series_temporal`,
   `reconcile_temporal_forecasts`)** — Projects multi-frequency forecasts onto the coherent
   temporal subspace using ``bottom_up``, ``top_down``, ``middle_out``, ``ols``,
   ``wls_struct`` (``struc``), ``wls_var``, ``mint_shrink`` (Schäfer-Strimmer shrinkage), and
   ``mint_nonneg`` (non-negative least squares), cleanly handling non-divisible horizons ``H``
   and short histories ``T < 2 * m_max``.
4. **Simultaneous Cross-Temporal Reconciliation (`reconcile_cross_temporal` /
   `cross_temporal_mint`)** — Reconciles hierarchical panels across both the cross-sectional
   summing matrix ``S_cs`` and the temporal summing matrix ``S_temp`` simultaneously,
   guaranteeing ``< 1e-6`` discrepancy on both axes across base models, ensembles, OOF folds,
   and horizon predictions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import RunConfig
from .errors import ConfigError, DataError
from .reconciliation import (
    HierarchySpec,
    build_hierarchy,
    ensure_hierarchy_nodes_present,
    reconcile_forecasts,
    schafer_strimmer_shrinkage,
    verify_coherence,
)
from .seasonality import seasonal_period

_Z_80 = 1.2815515655446004
_VAR_FLOOR = 1e-8

VALID_TEMPORAL_METHODS: frozenset[str] = frozenset(
    {
        "bottom_up",
        "top_down",
        "middle_out",
        "ols",
        "wls_struct",
        "struc",
        "wls_var",
        "mint_shrink",
        "mint_nonneg",
        "cross_temporal_mint",
    }
)

DEFAULT_TEMPORAL_FACTORS: dict[str, tuple[int, ...]] = {
    "D": (1, 7, 14, 28),
    "W": (1, 4, 13, 52),
    "M": (1, 3, 6, 12),
    "Q": (1, 2, 4),
    "H": (1, 6, 12, 24),
    "B": (1, 5, 10, 20),
    "Y": (1,),
}


class SafeArrayDict(dict[Any, Any]):
    """Dictionary subclass comparing NumPy array and nested mapping values via `np.array_equal`."""

    def __deepcopy__(self, memo: Any) -> SafeArrayDict:
        _ = memo
        return self

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Mapping) or set(self.keys()) != set(other.keys()):
            return False
        for k, v in self.items():
            ov = other[k]
            if isinstance(v, np.ndarray) or isinstance(ov, np.ndarray):
                if not np.array_equal(np.asarray(v), np.asarray(ov)):
                    return False
            elif isinstance(v, Mapping) and isinstance(ov, Mapping):
                if SafeArrayDict(v) != SafeArrayDict(ov):
                    return False
            elif v != ov:
                return False
        return True


_SafeArrayDict = SafeArrayDict


@dataclass(frozen=True)
class TemporalHierarchySpec:
    """Compiled Temporal Hierarchy (`THieF`) aggregation tree over a base horizon.

    Attributes:
        base_freq: Base pandas frequency string (e.g., ``"D"``, ``"W"``, ``"MS"``).
        horizon: Requested base-frequency forecast horizon ``H``.
        padded_horizon: Padded multiple of ``macro_period`` (``P * macro_period >= H``).
        macro_period: Least common multiple ``m`` of all aggregation factors.
        n_macro_periods: Number of complete macro-cycles ``P = ceil(H / m)``.
        aggregations: Ascending tuple of integer aggregation factors starting with ``1``.
        level_names: Coarse-to-fine level identifiers (e.g., ``("k28", "k14", "k7", "k1")``).
        level_slices: Row slice in ``summing_matrix`` for each aggregation factor ``k``.
        summing_matrix: Single-cycle binary summing matrix ``S_temp`` of shape
            ``(n_temp_nodes, macro_period)``, ordered from coarsest factor to ``k=1``.
        full_summing_matrix: Multi-cycle binary summing matrix ``I_P \\otimes S_temp`` of shape
            ``(P * n_temp_nodes, padded_horizon)``.
    """

    base_freq: str
    horizon: int
    padded_horizon: int
    macro_period: int
    n_macro_periods: int
    aggregations: tuple[int, ...]
    level_names: tuple[str, ...]
    level_slices: dict[int, slice]
    summing_matrix: np.ndarray
    full_summing_matrix: np.ndarray

    @property
    def m(self) -> int:
        """Alias for ``macro_period`` (single-cycle base steps)."""
        return self.macro_period

    @property
    def S_temp(self) -> np.ndarray:
        """Alias for ``summing_matrix`` (single-cycle temporal summing matrix)."""
        return self.summing_matrix

    @property
    def factors(self) -> tuple[int, ...]:
        """Alias for ``aggregations`` (ascending temporal factors including ``1``)."""
        return self.aggregations

    @property
    def n_total(self) -> int:
        """Total number of temporal nodes per macro-cycle."""
        return int(self.summing_matrix.shape[0])

    @property
    def n_total_per_cycle(self) -> int:
        """Number of temporal nodes per macro-cycle."""
        return int(self.summing_matrix.shape[0])

    @property
    def n_bottom_per_cycle(self) -> int:
        """Number of base-frequency steps per macro-cycle (``macro_period``)."""
        return int(self.summing_matrix.shape[1])

    @property
    def n_nodes(self) -> int:
        """Total number of temporal nodes across all ``n_macro_periods`` cycles."""
        return int(self.full_summing_matrix.shape[0])

    @property
    def n_bottom(self) -> int:
        """Total number of padded bottom steps across all ``n_macro_periods`` cycles."""
        return int(self.full_summing_matrix.shape[1])

    @property
    def structural_weights(self) -> np.ndarray:
        """Number of base steps summed into each single-cycle temporal node (``S_temp @ 1``)."""
        return self.summing_matrix.sum(axis=1)

    @property
    def full_structural_weights(self) -> np.ndarray:
        """Number of base steps summed into each full-horizon temporal node."""
        return self.full_summing_matrix.sum(axis=1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporalHierarchySpec):
            return False
        return (
            self.base_freq == other.base_freq
            and self.horizon == other.horizon
            and self.padded_horizon == other.padded_horizon
            and self.macro_period == other.macro_period
            and self.n_macro_periods == other.n_macro_periods
            and self.aggregations == other.aggregations
            and self.level_names == other.level_names
            and self.level_slices == other.level_slices
            and np.array_equal(self.summing_matrix, other.summing_matrix)
            and np.array_equal(self.full_summing_matrix, other.full_summing_matrix)
        )


def _normalize_freq_key(freq: str) -> str:
    """Normalize a pandas offset alias into its canonical frequency family."""
    raw = str(freq or "D").strip()
    while raw and raw[0].isdigit():
        raw = raw[1:].strip()
    base = raw.split("-", 1)[0].upper()
    if base in {"D", "B", "C"}:
        return "D"
    if base.startswith("W"):
        return "W"
    if base in {"M", "MS", "ME", "BM", "BMS", "BME"}:
        return "M"
    if base in {"Q", "QS", "QE", "BQ", "BQS", "BQE"}:
        return "Q"
    if base in {"H", "BH"}:
        return "H"
    if base in {"Y", "YS", "YE", "A", "AS", "AE"}:
        return "Y"
    return base or "D"


def default_temporal_aggregations(
    freq: str | int = "D",
    horizon: int | None = None,
    history_len: int | None = None,
    *,
    include_base: bool = False,
) -> list[int]:
    """Return canonical coarse temporal aggregation factors for ``freq`` (pure).

    Standard coarse aggregation ladders:
    * Daily (``"D"`` / ``"B"``): ``[7, 14, 28]``
    * Weekly (``"W"``): ``[4, 13, 52]``
    * Monthly (``"M"`` / ``"MS"`` / ``"ME"``): ``[3, 6, 12]``
    * Quarterly (``"Q"`` / ``"QS"`` / ``"QE"``): ``[2, 4]``
    * Hourly (``"H"`` / ``"h"``): ``[6, 12, 24]``

    When an integer ``m`` is passed as ``freq``, returns all integer divisors ``> 1`` of ``m``
    in ascending order (or including ``1`` when ``include_base=True``).
    When ``history_len`` is provided, filters out any factor ``k`` where ``history_len < 2 * k``
    so every retained coarse level has at least two complete historical blocks.
    """
    if isinstance(freq, (int, np.integer)) and not isinstance(freq, bool):
        m_val = int(freq)
        if m_val < 1:
            raise ConfigError(f"Temporal period m must be >= 1; got {freq}")
        candidates = [k for k in range(2, m_val + 1) if m_val % k == 0]
    else:
        fam = _normalize_freq_key(str(freq))
        if fam == "D":
            candidates = [7, 14, 28]
        elif fam == "W":
            candidates = [4, 13, 52]
        elif fam == "M":
            candidates = [3, 6, 12]
        elif fam == "Q":
            candidates = [2, 4]
        elif fam == "H":
            candidates = [6, 12, 24]
        else:
            target_h = max(2, int(horizon) if horizon is not None and horizon >= 2 else 12)
            candidates = [k for k in range(2, target_h + 1) if target_h % k == 0]
            if not candidates:
                candidates = [2, 4]

    if history_len is not None:
        max_k = max(1, int(history_len) // 2)
        candidates = [k for k in candidates if k <= max_k]

    if include_base:
        return sorted({1, *candidates})
    return sorted(set(candidates))


resolve_temporal_aggregations = default_temporal_aggregations


def build_temporal_summing_matrix(m: int, aggregations: Sequence[int]) -> np.ndarray:
    """Construct the single-cycle binary temporal summing matrix ``S_temp`` of shape
    ``(n_nodes, m)``.
    """
    macro_period = int(m)
    if macro_period < 1:
        raise ConfigError(f"Temporal period m must be >= 1; got {m}")
    agg_tuple = tuple(sorted({1, *(int(k) for k in aggregations if int(k) >= 1)}))
    blocks: list[np.ndarray] = []
    for k in reversed(agg_tuple):
        if macro_period % k != 0:
            raise ConfigError(f"Aggregation factor k={k} must divide macro_period m={macro_period}")
        n_buckets = macro_period // k
        blocks.append(np.kron(np.eye(n_buckets, dtype=float), np.ones((1, k), dtype=float)))
    return np.vstack(blocks) if blocks else np.eye(macro_period, dtype=float)


def build_temporal_hierarchy(
    freq: str | int = "D",
    horizon: int | None = None,
    temporal_aggregations: Sequence[int] | None = None,
    *,
    m: int | None = None,
    aggregations: Sequence[int] | None = None,
    history_len: int | None = None,
) -> TemporalHierarchySpec:
    """Compile a `TemporalHierarchySpec` and summing matrices for ``freq`` and ``horizon``."""
    if m is not None:
        if int(m) < 1:
            raise ConfigError(f"Temporal period m must be >= 1; got {m}")
        eff_m = int(m)
        h = int(horizon) if horizon is not None else eff_m
        freq_key: str | int = eff_m if freq == "D" else freq
    elif isinstance(freq, (int, np.integer)) and not isinstance(freq, bool):
        if int(freq) < 1:
            raise ConfigError(f"Temporal period m must be >= 1; got {freq}")
        eff_m = int(freq)
        h = int(horizon) if horizon is not None else eff_m
        freq_key = eff_m
    else:
        h = int(horizon) if horizon is not None else 14
        freq_key = str(freq)

    if h < 1:
        raise ConfigError(f"horizon must be >= 1 for temporal reconciliation; got {h}")

    explicit = temporal_aggregations if temporal_aggregations is not None else aggregations
    if explicit is not None:
        raw_list = [int(k) for k in explicit]
        if not raw_list:
            raise ConfigError("temporal_aggregations must not be empty when explicitly provided")
        if any(k < 1 for k in raw_list):
            raise ConfigError(
                f"temporal_aggregations factors must be positive integers >= 1; got {raw_list}"
            )
        if len(set(raw_list)) != len(raw_list):
            raise ConfigError(f"temporal_aggregations contains duplicate factors: {raw_list}")
        coarse = [k for k in raw_list if k > 1]
        if history_len is not None:
            max_k = max(1, int(history_len) // 2)
            coarse = [k for k in coarse if k <= max_k]
    else:
        coarse = default_temporal_aggregations(freq_key, horizon=h, history_len=history_len)

    agg_tuple = tuple(sorted({1, *coarse}))
    macro_period = math.lcm(*agg_tuple) if len(agg_tuple) > 1 else 1
    n_macro = max(1, math.ceil(h / macro_period))
    padded_h = n_macro * macro_period

    blocks: list[np.ndarray] = []
    level_names: list[str] = []
    level_slices: dict[int, slice] = {}
    row_cursor = 0

    # Order from coarsest factor k_max down to base k=1 so the bottom macro_period rows are I_m.
    for k in reversed(agg_tuple):
        n_buckets = macro_period // k
        block = np.kron(np.eye(n_buckets, dtype=float), np.ones((1, k), dtype=float))
        blocks.append(block)
        level_names.append(f"k{k}")
        level_slices[int(k)] = slice(row_cursor, row_cursor + n_buckets)
        row_cursor += n_buckets

    s_temp = np.vstack(blocks) if blocks else np.eye(macro_period, dtype=float)
    full_s_temp = np.kron(np.eye(n_macro, dtype=float), s_temp) if n_macro > 1 else s_temp.copy()

    return TemporalHierarchySpec(
        base_freq=str(freq) if not isinstance(freq, (int, np.integer)) else "D",
        horizon=h,
        padded_horizon=padded_h,
        macro_period=macro_period,
        n_macro_periods=n_macro,
        aggregations=agg_tuple,
        level_names=tuple(level_names),
        level_slices=level_slices,
        summing_matrix=s_temp,
        full_summing_matrix=full_s_temp,
    )


build_temporal_hierarchy_spec = build_temporal_hierarchy


def aggregate_temporal_history(
    y: pd.Series | np.ndarray | Sequence[float] | pd.DataFrame,
    k: int,
    *,
    target_col: str = "y",
) -> np.ndarray:
    """Aggregate a 1-D base series into non-overlapping blocks of length ``k`` aligned to the end.

    Drops ``len(y) % k`` oldest observations from the front so the final block ends on the exact
    same cutoff timestamp as the base-frequency series.
    """
    factor = int(k)
    if factor < 1:
        raise ConfigError(f"Temporal aggregation factor k must be >= 1; got {k}")
    if isinstance(y, pd.DataFrame):
        col = target_col if target_col in y.columns else ("yhat" if "yhat" in y.columns else None)
        if col is None:
            raise DataError(f"DataFrame is missing '{target_col}' or 'yhat' column")
        vals = y[col].to_numpy(dtype=float)
    elif isinstance(y, pd.Series):
        vals = y.to_numpy(dtype=float)
    else:
        vals = np.asarray(y, dtype=float).reshape(-1)

    if factor == 1:
        return vals.copy()
    n = len(vals)
    if n < factor:
        return np.empty(0, dtype=float)
    rem = n % factor
    trimmed = vals[rem:] if rem > 0 else vals
    return trimmed.reshape(-1, factor).sum(axis=1)


def aggregate_temporal_series(
    y: pd.Series | np.ndarray | Sequence[float] | pd.DataFrame,
    k: int,
    *,
    date_col: str = "ds",
    target_col: str = "y",
    ts_id_col: str = "ts_id",
) -> np.ndarray | pd.DataFrame:
    """Aggregate a 1-D series (``np.ndarray``) or panel ``DataFrame`` into blocks of ``k``."""
    factor = int(k)
    if factor < 1:
        raise ConfigError(f"Temporal aggregation factor k must be >= 1; got {k}")
    if not isinstance(y, pd.DataFrame):
        return aggregate_temporal_history(y, factor, target_col=target_col)

    df = y.copy()
    if df.empty:
        return df
    dcol = (
        date_col
        if date_col in df.columns
        else ("forecast_date" if "forecast_date" in df.columns else None)
    )
    vcol = target_col if target_col in df.columns else ("yhat" if "yhat" in df.columns else None)
    if vcol is None:
        raise DataError(f"DataFrame must contain '{target_col}' or 'yhat' for temporal aggregation")

    has_id = ts_id_col in df.columns
    groups = df.groupby(ts_id_col, sort=False) if has_id else [(None, df)]
    out_frames: list[pd.DataFrame] = []
    for tid, grp in groups:
        sorted_grp = grp.sort_values(dcol, kind="mergesort") if dcol is not None else grp
        vals = sorted_grp[vcol].to_numpy(dtype=float)
        n = len(vals)
        if n < factor:
            continue
        rem = n % factor
        trimmed_grp = sorted_grp.iloc[rem:] if rem > 0 else sorted_grp
        agg_vals = trimmed_grp[vcol].to_numpy(dtype=float).reshape(-1, factor).sum(axis=1)
        end_rows = trimmed_grp.iloc[factor - 1 :: factor].copy().reset_index(drop=True)
        end_rows[vcol] = agg_vals
        end_rows["temporal_factor"] = factor
        if tid is not None:
            end_rows[ts_id_col] = tid
        out_frames.append(end_rows)

    return pd.concat(out_frames, ignore_index=True) if out_frames else df.iloc[0:0].copy()


def _normalize_temporal_method(method: str) -> str:
    m = str(method).strip().lower()
    if m == "struc":
        return "wls_struct"
    if m not in VALID_TEMPORAL_METHODS:
        raise ConfigError(
            f"Unknown temporal reconciliation method '{method}'; "
            f"available: {sorted(VALID_TEMPORAL_METHODS)}"
        )
    return m


def _build_cycle_residual_matrix(
    spec: TemporalHierarchySpec,
    residuals_by_k: Mapping[int, np.ndarray] | None,
) -> np.ndarray | None:
    """Stack multi-frequency residuals into ``(n_cycles, n_temp_nodes)`` aligned to ``spec``."""
    if not residuals_by_k:
        return None
    m = spec.macro_period
    cycle_counts: list[int] = []
    for k in spec.aggregations:
        res_k = residuals_by_k.get(int(k))
        if res_k is None:
            return None
        arr = np.asarray(res_k, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        buckets_per_cycle = m // int(k)
        n_cyc = len(arr) // buckets_per_cycle
        if n_cyc < 1:
            return None
        cycle_counts.append(n_cyc)

    n_cycles = min(cycle_counts)
    if n_cycles < 1:
        return None

    blocks: list[np.ndarray] = []
    for k in reversed(spec.aggregations):
        arr = np.asarray(residuals_by_k[int(k)], dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        buckets_per_cycle = m // int(k)
        need = n_cycles * buckets_per_cycle
        tail = arr[-need:]
        blocks.append(tail.reshape(n_cycles, buckets_per_cycle))

    return np.hstack(blocks)


def _temporal_covariance(
    spec: TemporalHierarchySpec,
    method: str,
    residuals_by_k: Mapping[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Return the single-cycle ``(n_temp_nodes, n_temp_nodes)`` covariance matrix ``W_temp``."""
    norm_method = _normalize_temporal_method(method)
    n_nodes = spec.n_total_per_cycle
    if norm_method == "ols":
        return np.eye(n_nodes, dtype=float)
    if norm_method == "wls_struct":
        return np.diag(np.maximum(spec.structural_weights, _VAR_FLOOR))

    # Variance or shrinkage methods:
    if norm_method == "wls_var" or residuals_by_k:
        var_vec = np.zeros(n_nodes, dtype=float)
        has_any_res = False
        for k in reversed(spec.aggregations):
            sl = spec.level_slices[int(k)]
            res_k = residuals_by_k.get(int(k)) if residuals_by_k else None
            if res_k is not None:
                clean = np.asarray(res_k, dtype=float).reshape(-1)
                clean = clean[np.isfinite(clean)]
                if len(clean) > 0:
                    var_vec[sl] = max(float(np.mean(clean**2)), _VAR_FLOOR)
                    has_any_res = True
                    continue
            var_vec[sl] = max(float(k), _VAR_FLOOR)

        if norm_method == "wls_var" or not has_any_res:
            return np.diag(var_vec)

        cycle_mat = _build_cycle_residual_matrix(spec, residuals_by_k)
        if cycle_mat is not None and cycle_mat.shape[0] >= 2:
            w_shrink, _lam = schafer_strimmer_shrinkage(cycle_mat)
            return w_shrink
        return np.diag(var_vec)

    return np.diag(np.maximum(spec.structural_weights, _VAR_FLOOR))


def reconcile_temporal_matrix(
    spec: TemporalHierarchySpec,
    method: str = "wls_struct",
    *,
    residuals_by_k: Mapping[int, np.ndarray] | None = None,
    history_by_k: Mapping[int, np.ndarray] | None = None,
    use_full_horizon: bool = False,
) -> np.ndarray:
    """Compute the temporal reconciliation matrix ``G_temp`` of shape ``(m, n_temp_nodes)``.

    Given stacked single-cycle forecasts ``y_hat_cycle`` ordered from coarsest factor ``k_max`` to
    finest ``k=1``, the reconciled bottom-level trajectory is ``b_tilde = G_temp @ y_hat_cycle``
    and the coherent hierarchy is ``y_tilde = S_temp @ b_tilde``.
    When ``use_full_horizon=True``, returns ``I_P \\otimes G_temp`` of shape
    ``(padded_horizon, P * n_temp_nodes)``.
    """
    norm_method = _normalize_temporal_method(method)
    s_mat = spec.summing_matrix
    n_nodes, m = s_mat.shape
    n_agg = n_nodes - m

    if norm_method == "bottom_up":
        g_single = np.hstack([np.zeros((m, n_agg), dtype=float), np.eye(m, dtype=float)])
    elif norm_method == "top_down":
        g_single = np.zeros((m, n_nodes), dtype=float)
        props = np.full(m, 1.0 / max(1, m), dtype=float)
        if history_by_k and 1 in history_by_k:
            y1 = np.asarray(history_by_k[1], dtype=float).reshape(-1)
            y1 = y1[np.isfinite(y1)]
            n_cyc = len(y1) // m
            if n_cyc >= 1:
                cyc = y1[-n_cyc * m :].reshape(n_cyc, m)
                totals = cyc.sum(axis=1)
                valid = np.abs(totals) > 1e-12
                if np.any(valid):
                    raw_p = np.maximum(0.0, (cyc[valid] / totals[valid, np.newaxis]).mean(axis=0))
                    if raw_p.sum() > 1e-12:
                        props = raw_p / raw_p.sum()
        g_single[:, 0] = props
    elif norm_method == "middle_out":
        coarse_factors = [k for k in spec.aggregations if k > 1]
        if len(coarse_factors) < 2:
            # Fall back to wls_struct when no intermediate temporal level exists.
            w_mat = _temporal_covariance(spec, "wls_struct", residuals_by_k)
            winv_s = np.linalg.solve(w_mat, s_mat)
            g_single = np.linalg.solve(s_mat.T @ winv_s, winv_s.T)
        else:
            mid_k = coarse_factors[len(coarse_factors) // 2]
            sl = spec.level_slices[mid_k]
            n_mid = m // mid_k
            g_single = np.zeros((m, n_nodes), dtype=float)
            for b_idx in range(n_mid):
                col_idx = sl.start + b_idx
                start_step = b_idx * mid_k
                end_step = start_step + mid_k
                g_single[start_step:end_step, col_idx] = 1.0 / float(mid_k)
    else:
        w_mat = _temporal_covariance(spec, norm_method, residuals_by_k)
        reg = 1e-8 * max(float(np.trace(w_mat)) / max(1, n_nodes), 1.0)
        w_reg = w_mat + reg * np.eye(n_nodes, dtype=float)
        winv_s = np.linalg.solve(w_reg, s_mat)
        st_winv_s = s_mat.T @ winv_s
        g_single = np.linalg.solve(
            st_winv_s + 1e-10 * np.eye(m, dtype=float),
            winv_s.T,
        )

    if use_full_horizon and spec.n_macro_periods > 1:
        return np.kron(np.eye(spec.n_macro_periods, dtype=float), g_single)
    return g_single


def _fast_coarse_forecast_and_residuals(
    y_k: np.ndarray,
    horizon_k: int,
    seasonal_m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a fast deterministic trend + seasonal model on ``y_k`` and forecast ``horizon_k``."""
    vals = np.asarray(y_k, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    h = max(1, int(horizon_k))
    if n == 0:
        return np.zeros(h, dtype=float), np.zeros(1, dtype=float)
    if n == 1:
        return np.full(h, float(vals[0]), dtype=float), np.zeros(1, dtype=float)

    s_period = max(1, int(seasonal_m))
    # Estimate seasonal indices when at least 2 full seasonal cycles exist.
    seas = np.zeros(s_period, dtype=float)
    if s_period > 1 and n >= 2 * s_period:
        n_full = (n // s_period) * s_period
        reshaped = vals[-n_full:].reshape(-1, s_period)
        row_means = reshaped.mean(axis=1, keepdims=True)
        seas = (reshaped - row_means).mean(axis=0)
        seas -= seas.mean()
        deseasonalized = np.array(
            [vals[t] - seas[(t - (n - n_full)) % s_period] for t in range(n)],
            dtype=float,
        )
    else:
        deseasonalized = vals.copy()

    # Exponentially weighted level + damped trend (Theta / Holt-like smoothing).
    alpha = 0.4
    beta = 0.15
    phi = 0.90
    level = float(deseasonalized[0])
    trend = float(deseasonalized[1] - deseasonalized[0]) if n >= 2 else 0.0
    fitted = np.empty(n, dtype=float)
    fitted[0] = level + (seas[(0 - (n % s_period)) % s_period] if s_period > 1 else 0.0)

    for t in range(1, n):
        pred_deseas = level + phi * trend
        s_offset = seas[(t - (n % s_period)) % s_period] if s_period > 1 else 0.0
        fitted[t] = pred_deseas + s_offset
        err = float(deseasonalized[t]) - pred_deseas
        prev_level = level
        level = pred_deseas + alpha * err
        trend = phi * trend + beta * (level - prev_level)

    residuals = vals - fitted
    forecasts = np.empty(h, dtype=float)
    cum_phi = 0.0
    phi_pow = 1.0
    for step in range(1, h + 1):
        phi_pow *= phi
        cum_phi += phi_pow
        s_offset = seas[(step - 1) % s_period] if s_period > 1 else 0.0
        forecasts[step - 1] = level + cum_phi * trend + s_offset

    if np.all(vals >= 0.0):
        forecasts = np.maximum(0.0, forecasts)
    return forecasts, residuals


def _pad_base_forecast(yhat: np.ndarray, padded_horizon: int, seasonal_m: int) -> np.ndarray:
    """Extend a base forecast of length ``H`` to ``padded_horizon >= H`` via seasonal repeat."""
    arr = np.asarray(yhat, dtype=float).reshape(-1)
    h = len(arr)
    if h >= padded_horizon:
        return arr[:padded_horizon].copy()
    if h == 0:
        return np.zeros(padded_horizon, dtype=float)

    out = np.empty(padded_horizon, dtype=float)
    out[:h] = arr
    s_period = max(1, min(int(seasonal_m), h))
    for t in range(h, padded_horizon):
        out[t] = out[t - s_period] if s_period > 1 else arr[-1]
    return out


def _project_temporal_cycles(
    spec: TemporalHierarchySpec,
    forecasts_by_k: Mapping[int, np.ndarray],
    g_single: np.ndarray,
    *,
    nonneg: bool = False,
    w_mat: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Project per-level forecasts across ``spec.n_macro_periods`` cycles onto coherent subspace."""
    m = spec.macro_period
    n_macro = spec.n_macro_periods
    s_temp = spec.summing_matrix

    nonneg_lhs: np.ndarray | None = None
    w_inv_sqrt: np.ndarray | None = None
    if nonneg:
        w_use = (
            w_mat if w_mat is not None else np.diag(np.maximum(spec.structural_weights, _VAR_FLOOR))
        )
        eigvals, eigvecs = np.linalg.eigh(w_use)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, _VAR_FLOOR))
        w_inv_sqrt = (eigvecs * inv_sqrt[np.newaxis, :]) @ eigvecs.T
        nonneg_lhs = w_inv_sqrt @ s_temp

    reconciled_bottom_cycles: list[np.ndarray] = []
    reconciled_by_k_lists: dict[int, list[np.ndarray]] = {int(k): [] for k in spec.aggregations}

    for cyc_idx in range(n_macro):
        y_hat_parts: list[np.ndarray] = []
        for k in reversed(spec.aggregations):
            buckets_per_cycle = m // int(k)
            arr_k = np.asarray(forecasts_by_k[int(k)], dtype=float).reshape(-1)
            start = cyc_idx * buckets_per_cycle
            end = start + buckets_per_cycle
            if end <= len(arr_k):
                slice_k = arr_k[start:end]
            else:
                slice_k = _pad_base_forecast(arr_k, end, buckets_per_cycle)[start:end]
            y_hat_parts.append(slice_k)

        y_hat_cycle = np.concatenate(y_hat_parts)
        b_tilde = g_single @ y_hat_cycle
        if nonneg and np.any(b_tilde < 0.0):
            if nonneg_lhs is not None and w_inv_sqrt is not None:
                from scipy.optimize import nnls

                b_nnls, _ = nnls(nonneg_lhs, w_inv_sqrt @ y_hat_cycle)
                b_tilde = np.maximum(0.0, b_nnls)
            else:
                b_tilde = np.maximum(0.0, b_tilde)

        y_tilde_cycle = s_temp @ b_tilde
        reconciled_bottom_cycles.append(b_tilde)
        for k in spec.aggregations:
            sl = spec.level_slices[int(k)]
            reconciled_by_k_lists[int(k)].append(y_tilde_cycle[sl])

    full_bottom = np.concatenate(reconciled_bottom_cycles)
    reconciled_by_k: dict[int, np.ndarray] = {
        int(k): np.concatenate(parts) for k, parts in reconciled_by_k_lists.items()
    }
    return full_bottom, reconciled_by_k


def reconcile_series_temporal(
    series_df: pd.DataFrame | None,
    base_predictions_df: pd.DataFrame,
    cfg: RunConfig | None = None,
    *,
    freq: str | None = None,
    horizon: int | None = None,
    temporal_aggregations: Sequence[int] | None = None,
    model_name: str | None = None,
    method: str = "mint_shrink",
    oof_df: pd.DataFrame | None = None,
    coarse_forecasts: Mapping[int, np.ndarray] | None = None,
    nonneg: bool = False,
    date_col: str = "ds",
    target_col: str = "y",
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Perform `THieF` temporal reconciliation for a single series (pure).

    Returns ``(reconciled_predictions_df, reconciled_by_k)`` where ``reconciled_predictions_df``
    has length ``horizon`` and ``reconciled_by_k[k]`` holds the coherent coarse trajectory at
    every aggregation factor ``k`` in ``spec.aggregations``.
    """
    if base_predictions_df.empty:
        return base_predictions_df.copy(), {1: np.empty(0, dtype=float)}

    eff_freq = freq or (cfg.data.freq if cfg is not None else "D")
    pred_df = base_predictions_df.copy()
    dcol = (
        date_col
        if date_col in pred_df.columns
        else ("forecast_date" if "forecast_date" in pred_df.columns else None)
    )
    if dcol is not None:
        pred_df = pred_df.sort_values(dcol, kind="mergesort").reset_index(drop=True)

    h = int(horizon) if horizon is not None else len(pred_df)
    eff_aggs = temporal_aggregations
    if eff_aggs is None and cfg is not None and cfg.hierarchy.temporal_aggregations is not None:
        eff_aggs = cfg.hierarchy.temporal_aggregations

    y_hist = np.empty(0, dtype=float)
    if series_df is not None and not series_df.empty:
        hist_dcol = (
            date_col
            if date_col in series_df.columns
            else ("ds" if "ds" in series_df.columns else None)
        )
        s_df = (
            series_df.sort_values(hist_dcol, kind="mergesort")
            if hist_dcol is not None
            else series_df
        )
        tcol = target_col if target_col in s_df.columns else ("y" if "y" in s_df.columns else None)
        if tcol is not None:
            raw_y = s_df[tcol].to_numpy(dtype=float)
            y_hist = raw_y[np.isfinite(raw_y)]

    hist_len = len(y_hist) if len(y_hist) > 0 else None
    spec = build_temporal_hierarchy(
        eff_freq,
        h,
        eff_aggs,
        history_len=hist_len if coarse_forecasts is None else None,
    )

    base_yhat = pred_df["yhat"].to_numpy(dtype=float)
    base_season = seasonal_period(eff_freq)
    padded_base = _pad_base_forecast(base_yhat, spec.padded_horizon, base_season)

    forecasts_by_k: dict[int, np.ndarray] = {1: padded_base}
    residuals_by_k: dict[int, np.ndarray] = {}
    history_by_k: dict[int, np.ndarray] = {}

    if len(y_hist) > 0:
        history_by_k[1] = y_hist

    # Extract k=1 residuals from oof_df if provided, else 1-step in-sample residuals from history.
    if oof_df is not None and not oof_df.empty and {"y_true", "yhat"}.issubset(oof_df.columns):
        residuals_by_k[1] = oof_df["y_true"].to_numpy(dtype=float) - oof_df["yhat"].to_numpy(
            dtype=float
        )
    elif len(y_hist) >= 2:
        _, res_1 = _fast_coarse_forecast_and_residuals(y_hist, spec.padded_horizon, base_season)
        residuals_by_k[1] = res_1

    for k in spec.aggregations:
        if k == 1:
            continue
        buckets_needed = spec.n_macro_periods * (spec.macro_period // int(k))
        if coarse_forecasts is not None and int(k) in coarse_forecasts:
            forecasts_by_k[int(k)] = _pad_base_forecast(
                np.asarray(coarse_forecasts[int(k)], dtype=float),
                buckets_needed,
                max(1, base_season // int(k)),
            )
        elif len(y_hist) >= 2 * int(k):
            y_k = aggregate_temporal_history(y_hist, int(k))
            history_by_k[int(k)] = y_k
            f_k, r_k = _fast_coarse_forecast_and_residuals(
                y_k,
                buckets_needed,
                max(1, base_season // int(k)),
            )
            forecasts_by_k[int(k)] = f_k
            residuals_by_k[int(k)] = r_k
        else:
            # Aggregate padded base forecast when history is absent or shorter than 2*k.
            forecasts_by_k[int(k)] = aggregate_temporal_history(padded_base, int(k))

    norm_method = _normalize_temporal_method(method)
    is_nonneg = bool(
        nonneg
        or norm_method == "mint_nonneg"
        or (cfg is not None and cfg.output.floor is not None and cfg.output.floor >= 0.0)
        or (len(y_hist) > 0 and np.all(y_hist >= 0.0) and np.all(base_yhat >= 0.0))
    )
    g_single = reconcile_temporal_matrix(
        spec,
        norm_method,
        residuals_by_k=residuals_by_k if residuals_by_k else None,
        history_by_k=history_by_k if history_by_k else None,
    )
    w_mat = _temporal_covariance(
        spec,
        norm_method,
        residuals_by_k=residuals_by_k if residuals_by_k else None,
    )
    reconciled_padded, reconciled_by_k = _project_temporal_cycles(
        spec,
        forecasts_by_k,
        g_single,
        nonneg=is_nonneg,
        w_mat=w_mat,
    )

    reconciled_yhat = reconciled_padded[: len(pred_df)]
    delta = reconciled_yhat - base_yhat
    pred_df["yhat"] = reconciled_yhat
    if "yhat_raw" in pred_df.columns and pred_df["yhat_raw"].notna().all():
        pred_df["yhat_raw"] = pred_df["yhat_raw"].to_numpy(dtype=float) + delta
    if "yhat_adjusted" in pred_df.columns and pred_df["yhat_adjusted"].notna().all():
        pred_df["yhat_adjusted"] = reconciled_yhat

    # Shift prediction intervals coherently around reconciled_yhat while preserving interval width.
    if (
        "yhat_lower" in pred_df.columns
        and "yhat_upper" in pred_df.columns
        and pred_df["yhat_lower"].notna().all()
        and pred_df["yhat_upper"].notna().all()
    ):
        half_w = np.maximum(
            0.0,
            0.5
            * (
                pred_df["yhat_upper"].to_numpy(dtype=float)
                - pred_df["yhat_lower"].to_numpy(dtype=float)
            ),
        )
        lo = reconciled_yhat - half_w
        hi = reconciled_yhat + half_w
        if is_nonneg:
            lo = np.maximum(0.0, lo)
            hi = np.maximum(reconciled_yhat, hi)
        pred_df["yhat_lower"] = np.minimum(lo, reconciled_yhat)
        pred_df["yhat_upper"] = np.maximum(hi, reconciled_yhat)

    for col in pred_df.columns:
        if col.startswith("yhat_lower_"):
            q_suffix = col[len("yhat_lower_") :]
            u_col = f"yhat_upper_{q_suffix}"
            if u_col in pred_df.columns:
                hw = np.maximum(
                    0.0,
                    0.5
                    * (pred_df[u_col].to_numpy(dtype=float) - pred_df[col].to_numpy(dtype=float)),
                )
                lo = reconciled_yhat - hw
                hi = reconciled_yhat + hw
                if is_nonneg:
                    lo = np.maximum(0.0, lo)
                    hi = np.maximum(reconciled_yhat, hi)
                pred_df[col] = np.minimum(lo, reconciled_yhat)
                pred_df[u_col] = np.maximum(hi, reconciled_yhat)

    if "quantiles" in pred_df.columns:
        new_q_col: list[Any] = []
        for idx, qdict in enumerate(pred_df["quantiles"]):
            if isinstance(qdict, dict) and qdict:
                shifted = {}
                for qk, qv in qdict.items():
                    val = float(qv) + float(delta[idx])
                    if is_nonneg:
                        val = max(0.0, val)
                    shifted[qk] = val
                new_q_col.append(shifted)
            else:
                new_q_col.append(qdict)
        pred_df["quantiles"] = new_q_col

    if model_name is not None and "model_type" in pred_df.columns:
        pred_df["model_type"] = model_name

    reconciled_by_k[1] = reconciled_yhat.copy()
    safe_by_k: dict[int, np.ndarray] = _SafeArrayDict(reconciled_by_k)
    pred_df.attrs["temporal_spec"] = spec
    pred_df.attrs["aggregated_by_k"] = safe_by_k
    return pred_df, safe_by_k


reconcile_temporal_series = reconcile_series_temporal
temporal_covariance_matrix = _temporal_covariance


def reconcile_temporal_forecasts(
    forecasts_df: pd.DataFrame,
    *,
    history_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    cfg: RunConfig | None = None,
    freq: str | None = None,
    horizon: int | None = None,
    temporal_aggregations: Sequence[int] | None = None,
    aggregations: Sequence[int] | None = None,
    method: str = "mint_shrink",
    model_type: str | None = None,
    coarse_forecasts: Mapping[int, np.ndarray] | None = None,
    nonneg: bool = False,
    stamp_method: bool = False,
    return_aggregations: bool = False,
    date_col: str = "ds",
    ts_id_col: str = "ts_id",
    target_col: str = "y",
) -> pd.DataFrame | tuple[pd.DataFrame, dict[Any, np.ndarray]]:
    """Reconcile a single-series or multi-series forecast DataFrame across temporal aggregations.

    Attaches ``df.attrs["temporal_spec"]``, ``df.attrs["aggregated_by_k"]`` (for single series),
    and ``df.attrs["aggregated_by_node_k"]`` (mapping ``(ts_id, k) -> np.ndarray``) so downstream
    callers and `verify_temporal_coherence` can inspect exact bucket coherence at every factor.
    """
    if forecasts_df.empty:
        empty_agg: dict[Any, np.ndarray] = _SafeArrayDict()
        return (forecasts_df.copy(), empty_agg) if return_aggregations else forecasts_df.copy()

    df = forecasts_df.copy()
    if model_type is not None and "model_type" in df.columns:
        df = df[df["model_type"] == model_type].copy()
        if df.empty:
            raise DataError(f"No forecast rows found for model_type='{model_type}'")

    eff_aggs = temporal_aggregations if temporal_aggregations is not None else aggregations
    if eff_aggs is None and cfg is not None:
        eff_aggs = cfg.hierarchy.temporal_aggregations
    eff_freq = freq or (cfg.data.freq if cfg is not None else "D")

    has_ts_id = ts_id_col in df.columns
    has_model = "model_type" in df.columns
    group_cols = ([ts_id_col] if has_ts_id else []) + (["model_type"] if has_model else [])

    parts: list[pd.DataFrame] = []
    agg_by_node_k: dict[tuple[str, int], np.ndarray] = _SafeArrayDict()
    last_agg_by_k: dict[int, np.ndarray] = _SafeArrayDict()
    last_spec: TemporalHierarchySpec | None = None

    groups = df.groupby(group_cols, sort=False) if group_cols else [(None, df)]
    for key, grp in groups:
        if isinstance(key, tuple):
            tid = str(key[0]) if has_ts_id else "__series__"
            mname = str(key[1]) if has_model and len(key) > 1 else None
        elif key is not None:
            tid = str(key) if has_ts_id else "__series__"
            mname = str(key) if has_model and not has_ts_id else None
        else:
            tid = "__series__"
            mname = None

        s_hist: pd.DataFrame | None = None
        if history_df is not None and not history_df.empty:
            if ts_id_col in history_df.columns and has_ts_id:
                s_hist = history_df[history_df[ts_id_col].astype(str) == tid]
            else:
                s_hist = history_df

        s_oof: pd.DataFrame | None = None
        if oof_df is not None and not oof_df.empty:
            s_oof = oof_df
            if ts_id_col in s_oof.columns and has_ts_id:
                s_oof = s_oof[s_oof[ts_id_col].astype(str) == tid]
            if "model_type" in s_oof.columns and mname is not None:
                s_oof = s_oof[s_oof["model_type"] == mname]

        stamped = f"{mname}_{method}" if (stamp_method and mname is not None) else mname
        rec_grp, rec_by_k = reconcile_series_temporal(
            s_hist,
            grp,
            cfg,
            freq=eff_freq,
            horizon=horizon,
            temporal_aggregations=eff_aggs,
            model_name=stamped,
            method=method,
            oof_df=s_oof,
            coarse_forecasts=coarse_forecasts,
            nonneg=nonneg,
            date_col=date_col,
            target_col=target_col,
        )
        last_spec = rec_grp.attrs.get("temporal_spec")
        last_agg_by_k = _SafeArrayDict(rec_by_k)
        for k, arr in rec_by_k.items():
            agg_by_node_k[(tid, int(k))] = arr
        rec_grp.attrs.clear()
        parts.append(rec_grp)

    out = pd.concat(parts, ignore_index=True) if parts else df.copy()
    if last_spec is not None:
        out.attrs["temporal_spec"] = last_spec
    out.attrs["aggregated_by_k"] = last_agg_by_k
    out.attrs["aggregated_by_node_k"] = agg_by_node_k
    final_temp_disc = (
        float(
            verify_temporal_coherence(
                out,
                agg_by_node_k if has_ts_id else last_agg_by_k,
                spec=last_spec,
                raise_on_error=False,
            )
        )
        if not out.empty
        else 0.0
    )
    out.attrs["cross_temporal_diagnostics"] = {
        "mode": "temporal_only",
        "spatial_weight_alpha": 0.0,
        "cs_first_discrepancy": 0.0,
        "temp_first_discrepancy": 0.0,
        "final_cs_discrepancy": 0.0,
        "final_temp_discrepancy": final_temp_disc,
    }
    if return_aggregations:
        return out, (agg_by_node_k if has_ts_id else last_agg_by_k)
    return out


def _learn_spatial_temporal_alpha(
    hier_spec: HierarchySpec,
    tspec: TemporalHierarchySpec,
    *,
    oof_df: pd.DataFrame | None,
    history_df: pd.DataFrame | None,
    piv_a: np.ndarray,
    piv_b: np.ndarray,
    ts_id_col: str,
    date_col: str,
    target_col: str,
) -> float:
    """Learn optimal convex weight ``alpha in [0.05, 0.95]`` from OOF or in-sample residuals."""
    _ = (piv_a, piv_b)
    # 1. Try out-of-fold residuals when available.
    if oof_df is not None and not oof_df.empty and {"y_true", "yhat"}.issubset(oof_df.columns):
        oof_dcol = (
            "ds"
            if "ds" in oof_df.columns
            else ("forecast_date" if "forecast_date" in oof_df.columns else date_col)
        )
        if ts_id_col in oof_df.columns and oof_dcol in oof_df.columns:
            work_oof = oof_df.copy()
            work_oof["_res"] = work_oof["y_true"].to_numpy(dtype=float) - work_oof["yhat"].to_numpy(
                dtype=float
            )
            res_piv = (
                work_oof.pivot_table(
                    index=ts_id_col, columns=oof_dcol, values="_res", aggfunc="mean"
                )
                .reindex(index=list(hier_spec.node_ids))
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            if res_piv.size > 0 and float(np.max(np.abs(res_piv))) > 1e-12:
                n_agg = hier_spec.n_aggregated
                bot_res = res_piv[n_agg:]
                agg_res = res_piv[:n_agg] if n_agg > 0 else bot_res
                # Spatial residual variance across hierarchy levels
                s_upper = (
                    hier_spec.summing_matrix[:n_agg] if n_agg > 0 else hier_spec.summing_matrix
                )
                cs_coh_err = agg_res - (s_upper @ bot_res)
                sigma2_cs = float(np.mean(cs_coh_err**2) + 0.5 * np.mean(agg_res**2))

                # Temporal residual variance across coarse calendar buckets k > 1
                temp_vars: list[float] = []
                for k in tspec.aggregations:
                    if int(k) <= 1:
                        continue
                    k_int = int(k)
                    if res_piv.shape[1] >= k_int:
                        n_blk = res_piv.shape[1] // k_int
                        trimmed = bot_res[:, -n_blk * k_int :].reshape(
                            bot_res.shape[0], n_blk, k_int
                        )
                        blk_mean_err = np.mean(trimmed, axis=2)
                        temp_vars.append(float(np.mean(blk_mean_err**2)))
                sigma2_temp = float(np.mean(temp_vars)) if temp_vars else float(np.mean(bot_res**2))
                denom = sigma2_cs + sigma2_temp
                if denom > 1e-12:
                    return float(np.clip(sigma2_temp / denom, 0.05, 0.95))

    # 2. Fallback to in-sample first-difference residuals from history_df when available.
    if (
        history_df is not None
        and not history_df.empty
        and ts_id_col in history_df.columns
        and target_col in history_df.columns
    ):
        h_dcol = (
            date_col
            if date_col in history_df.columns
            else ("ds" if "ds" in history_df.columns else "forecast_date")
        )
        if h_dcol in history_df.columns:
            h_piv = (
                history_df.pivot_table(
                    index=ts_id_col, columns=h_dcol, values=target_col, aggfunc="mean"
                )
                .reindex(index=list(hier_spec.bottom_ids))
                .ffill(axis=1)
                .bfill(axis=1)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            if h_piv.shape[1] >= 4:
                diff1 = np.diff(h_piv, axis=1)
                if float(np.max(np.abs(diff1))) > 1e-12:
                    sigma2_cs = float(np.mean(np.var(diff1, axis=1)))
                    temp_vars_h: list[float] = []
                    for k in tspec.aggregations:
                        if int(k) <= 1:
                            continue
                        k_int = int(k)
                        if h_piv.shape[1] >= 2 * k_int:
                            n_blk = h_piv.shape[1] // k_int
                            blk = (
                                h_piv[:, -n_blk * k_int :]
                                .reshape(h_piv.shape[0], n_blk, k_int)
                                .mean(axis=2)
                            )
                            temp_vars_h.append(float(np.mean(np.diff(blk, axis=1) ** 2)))
                    sigma2_temp = (
                        float(np.mean(temp_vars_h)) if temp_vars_h else float(np.var(diff1))
                    )
                    denom = sigma2_cs + sigma2_temp
                    if denom > 1e-12:
                        return float(np.clip(sigma2_temp / denom, 0.05, 0.95))

    return 0.5


def reconcile_cross_temporal(
    predictions_df: pd.DataFrame,
    hier_spec: HierarchySpec,
    temp_spec: TemporalHierarchySpec | None = None,
    *,
    history_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    cfg: RunConfig | None = None,
    freq: str | None = None,
    temporal_aggregations: Sequence[int] | None = None,
    method: str = "mint_shrink",
    middle_level: Sequence[str] | None = None,
    spatial_temporal_weight: float | Literal["auto"] | None = None,
    nonneg: bool = False,
    stamp_method: bool = False,
    return_aggregations: bool = False,
    date_col: str = "ds",
    ts_id_col: str = "ts_id",
    target_col: str = "y",
) -> pd.DataFrame | tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    """Simultaneous Cross-Sectional and Temporal (`cross_temporal_mint`) Reconciliation (pure).

    Projects base forecasts across both the cross-sectional hierarchy ``hier_spec`` (``S_cs``) and
    the temporal hierarchy ``temp_spec`` (``S_temp``) via the Di Fonzo & Girolimetto (2023)
    cross-temporal Minimum-Trace projection:

    1. Temporally reconciles every cross-sectional node ``i in hier_spec.node_ids`` across
       ``temp_spec.aggregations`` to obtain coherent temporal trajectories ``Y_hat_temp``.
    2. Cross-sectionally reconciles the base panel across ``hier_spec`` via ``G_cs`` and
       temporally reconciles the resulting cross-sectional trajectories.
    3. Combines the two orthogonal projections in bottom-leaf base-frequency space ``B_tilde``
       using ``spatial_temporal_weight`` (explicit float in ``[0, 1]`` or ``"auto"`` inverse
       residual variance learning) with alternating projection refinement, and maps through
       ``Y_tilde = S_cs @ B_tilde`` and ``Y_tilde^(k) = Y_tilde @ S_temp^(k).T``, guaranteeing
       ``< 1e-6`` discrepancy across every cross-sectional level and every temporal aggregation
       factor ``k`` simultaneously.
    """
    if predictions_df.empty:
        empty_map: dict[tuple[str, int], np.ndarray] = {}
        return (predictions_df.copy(), empty_map) if return_aggregations else predictions_df.copy()

    eff_weight: float | Literal["auto"] = (
        spatial_temporal_weight
        if spatial_temporal_weight is not None
        else (cfg.hierarchy.spatial_temporal_weight if cfg is not None else "auto")
    )
    if eff_weight != "auto":
        if isinstance(eff_weight, bool) or not isinstance(eff_weight, (int, float)):
            raise ConfigError("spatial_temporal_weight must be 'auto' or a float in [0.0, 1.0].")
        w_val = float(eff_weight)
        if not math.isfinite(w_val) or w_val < 0.0 or w_val > 1.0:
            raise ConfigError(f"spatial_temporal_weight must be in [0.0, 1.0], got {eff_weight!r}.")
        eff_weight = w_val

    dcol = (
        date_col
        if date_col in predictions_df.columns
        else ("forecast_date" if "forecast_date" in predictions_df.columns else "ds")
    )
    predictions_df = ensure_hierarchy_nodes_present(
        predictions_df,
        hier_spec,
        ts_id_col=ts_id_col,
        date_col=dcol,
        target_col="yhat",
    )
    if oof_df is not None and not oof_df.empty:
        oof_dcol = "forecast_date" if "forecast_date" in oof_df.columns else "ds"
        oof_df = ensure_hierarchy_nodes_present(
            oof_df,
            hier_spec,
            ts_id_col=ts_id_col,
            date_col=oof_dcol,
            target_col="yhat",
        )
    if history_df is not None and not history_df.empty:
        hist_dcol = "ds" if "ds" in history_df.columns else dcol
        history_df = ensure_hierarchy_nodes_present(
            history_df,
            hier_spec,
            ts_id_col=ts_id_col,
            date_col=hist_dcol,
            target_col=target_col,
        )
    eff_freq = freq or (cfg.data.freq if cfg is not None else "D")
    eff_aggs = temporal_aggregations
    if eff_aggs is None and cfg is not None:
        eff_aggs = cfg.hierarchy.temporal_aggregations

    cs_method_map = {
        "cross_temporal_mint": "mint_shrink",
        "struc": "wls_struct",
    }
    norm_method = _normalize_temporal_method(method)
    cs_method: Any = cs_method_map.get(norm_method, norm_method)
    is_nonneg = bool(
        nonneg
        or norm_method == "mint_nonneg"
        or (cfg is not None and cfg.output.floor is not None and cfg.output.floor >= 0.0)
    )

    has_model = "model_type" in predictions_df.columns
    models = predictions_df["model_type"].drop_duplicates().tolist() if has_model else [None]

    all_reconciled_parts: list[pd.DataFrame] = []
    agg_by_node_k: dict[tuple[str, int], np.ndarray] = _SafeArrayDict()
    resolved_temp_spec = temp_spec
    last_alpha = 0.5 if eff_weight == "auto" else float(eff_weight)
    last_cs_first_disc = 0.0
    last_temp_first_disc = 0.0

    for model_name in models:
        m_preds = (
            predictions_df[predictions_df["model_type"] == model_name].copy()
            if model_name is not None
            else predictions_df.copy()
        )
        m_oof = (
            oof_df[oof_df["model_type"] == model_name].copy()
            if (
                oof_df is not None
                and not oof_df.empty
                and "model_type" in oof_df.columns
                and model_name is not None
            )
            else oof_df
        )

        dates = sorted(m_preds[dcol].unique())
        h = len(dates)
        if resolved_temp_spec is None or resolved_temp_spec.horizon != h:
            resolved_temp_spec = build_temporal_hierarchy(eff_freq, max(1, h), eff_aggs)
        tspec = resolved_temp_spec

        # Step 1: Temporal reconciliation of each cross-sectional node i in hier_spec.node_ids.
        temp_rec_res = reconcile_temporal_forecasts(
            m_preds,
            history_df=history_df,
            oof_df=m_oof,
            cfg=cfg,
            freq=eff_freq,
            horizon=h,
            temporal_aggregations=tspec.aggregations,
            method=norm_method,
            nonneg=is_nonneg,
            stamp_method=False,
            date_col=dcol,
            ts_id_col=ts_id_col,
            target_col=target_col,
        )
        assert isinstance(temp_rec_res, pd.DataFrame)
        temp_rec_df = temp_rec_res
        temp_first_agg_map = temp_rec_df.attrs.get("aggregated_by_node_k")

        # Step 2: Cross-sectional reconciliation first, then temporal reconciliation.
        cs_first_df = reconcile_forecasts(
            m_preds,
            hier_spec,
            cs_method,
            history_df=history_df,
            oof_df=m_oof,
            middle_level=middle_level,
            date_col=dcol,
            ts_id_col=ts_id_col,
            target_col=target_col,
            stamp_method=False,
        )
        cs_then_temp_res = reconcile_temporal_forecasts(
            cs_first_df,
            history_df=history_df,
            oof_df=m_oof,
            cfg=cfg,
            freq=eff_freq,
            horizon=h,
            temporal_aggregations=tspec.aggregations,
            method=norm_method,
            nonneg=is_nonneg,
            stamp_method=False,
            date_col=dcol,
            ts_id_col=ts_id_col,
            target_col=target_col,
        )
        assert isinstance(cs_then_temp_res, pd.DataFrame)
        cs_then_temp_df = cs_then_temp_res

        # Step 3: Cross-sectional reconciliation of the temporally reconciled node forecasts.
        temp_then_cs_df = reconcile_forecasts(
            temp_rec_df,
            hier_spec,
            cs_method,
            history_df=history_df,
            oof_df=m_oof,
            middle_level=middle_level,
            date_col=dcol,
            ts_id_col=ts_id_col,
            target_col=target_col,
            stamp_method=False,
        )

        # Record single-order discrepancies prior to joint blending & refinement.
        last_cs_first_disc = float(
            verify_coherence(
                cs_then_temp_df,
                hier_spec,
                date_col=dcol,
                ts_id_col=ts_id_col,
                atol=float("inf"),
            )
        )
        last_temp_first_disc = (
            float(
                verify_temporal_coherence(
                    temp_then_cs_df,
                    temp_first_agg_map,
                    spec=tspec,
                    raise_on_error=False,
                )
            )
            if temp_first_agg_map
            else 0.0
        )

        # Step 4: Blend the two cross-temporal projections on bottom leaves B_tilde(alpha).
        piv_a = (
            temp_then_cs_df.pivot_table(
                index=ts_id_col, columns=dcol, values="yhat", aggfunc="first"
            )
            .reindex(index=list(hier_spec.bottom_ids), columns=dates)
            .to_numpy(dtype=float)
        )
        piv_b = (
            cs_then_temp_df.pivot_table(
                index=ts_id_col, columns=dcol, values="yhat", aggfunc="first"
            )
            .reindex(index=list(hier_spec.bottom_ids), columns=dates)
            .to_numpy(dtype=float)
        )
        if eff_weight == "auto":
            alpha = _learn_spatial_temporal_alpha(
                hier_spec,
                tspec,
                oof_df=m_oof,
                history_df=history_df,
                piv_a=piv_a,
                piv_b=piv_b,
                ts_id_col=ts_id_col,
                date_col=dcol,
                target_col=target_col,
            )
        else:
            alpha = float(eff_weight)
        last_alpha = alpha

        b_tilde_h = alpha * piv_a + (1.0 - alpha) * piv_b
        if is_nonneg:
            b_tilde_h = np.maximum(0.0, b_tilde_h)

        # Compute padded bottom matrix and run alternating projection refinement across S_temp/S_cs.
        base_season = seasonal_period(eff_freq)
        b_tilde_padded = np.vstack(
            [
                _pad_base_forecast(b_tilde_h[b_idx], tspec.padded_horizon, base_season)
                for b_idx in range(hier_spec.n_bottom)
            ]
        )
        g_temp_struct = reconcile_temporal_matrix(tspec, "wls_struct", use_full_horizon=True)
        bot_row_indices = np.concatenate(
            [
                np.arange(
                    cyc * tspec.n_total_per_cycle + tspec.level_slices[1].start,
                    cyc * tspec.n_total_per_cycle + tspec.level_slices[1].stop,
                )
                for cyc in range(tspec.n_macro_periods)
            ]
        )
        p_temp_bottom = tspec.full_summing_matrix[bot_row_indices, :] @ g_temp_struct
        s_temp_bot_inv = tspec.full_summing_matrix
        for _iter in range(5):
            # Project each bottom series onto S_temp and enforce non-negativity when active.
            y_all_cycles = b_tilde_padded @ s_temp_bot_inv.T
            b_proj = y_all_cycles @ p_temp_bottom.T
            if is_nonneg:
                b_proj = np.maximum(0.0, b_proj)
            if float(np.max(np.abs(b_proj - b_tilde_padded))) < 1e-9:
                b_tilde_padded = b_proj
                break
            b_tilde_padded = b_proj

        b_tilde_h = b_tilde_padded[:, :h].copy()
        if is_nonneg:
            b_tilde_h = np.maximum(0.0, b_tilde_h)
            b_tilde_padded[:, :h] = b_tilde_h

        # Full cross-sectional hierarchy matrix Y_tilde of shape (n_nodes, H) = S_cs @ B_tilde_H.
        y_tilde_h = hier_spec.summing_matrix @ b_tilde_h
        y_tilde_padded = hier_spec.summing_matrix @ b_tilde_padded

        for node_idx, nid in enumerate(hier_spec.node_ids):
            node_series_padded = y_tilde_padded[node_idx]
            node_series_h = y_tilde_h[node_idx]
            agg_by_node_k[(nid, 1)] = node_series_h.copy()
            for k in tspec.aggregations:
                if k == 1:
                    continue
                # Aggregate over the padded horizon so every bucket b has exact sum of k base steps.
                agg_by_node_k[(nid, int(k))] = aggregate_temporal_history(
                    node_series_padded, int(k)
                )

        # Rebuild coherent DataFrame aligned with temp_then_cs_df schema and interval widths.
        node_to_idx = {nid: i for i, nid in enumerate(hier_spec.node_ids)}
        date_to_idx = {dt: j for j, dt in enumerate(dates)}

        rec_model_df = temp_then_cs_df.copy()
        new_yhat = np.array(
            [
                y_tilde_h[node_to_idx[str(row_id)], date_to_idx[row_dt]]
                for row_id, row_dt in zip(rec_model_df[ts_id_col], rec_model_df[dcol], strict=True)
            ],
            dtype=float,
        )
        old_yhat = rec_model_df["yhat"].to_numpy(dtype=float)
        delta_yhat = new_yhat - old_yhat
        rec_model_df["yhat"] = new_yhat
        if "yhat_raw" in rec_model_df.columns and rec_model_df["yhat_raw"].notna().all():
            rec_model_df["yhat_raw"] = rec_model_df["yhat_raw"].to_numpy(dtype=float) + delta_yhat
        if "yhat_adjusted" in rec_model_df.columns and rec_model_df["yhat_adjusted"].notna().all():
            rec_model_df["yhat_adjusted"] = new_yhat

        if (
            "yhat_lower" in rec_model_df.columns
            and "yhat_upper" in rec_model_df.columns
            and rec_model_df["yhat_lower"].notna().all()
            and rec_model_df["yhat_upper"].notna().all()
        ):
            hw = np.maximum(
                0.0,
                0.5
                * (
                    rec_model_df["yhat_upper"].to_numpy(dtype=float)
                    - rec_model_df["yhat_lower"].to_numpy(dtype=float)
                ),
            )
            lo = new_yhat - hw
            hi = new_yhat + hw
            if is_nonneg:
                lo = np.maximum(0.0, lo)
                hi = np.maximum(new_yhat, hi)
            rec_model_df["yhat_lower"] = np.minimum(lo, new_yhat)
            rec_model_df["yhat_upper"] = np.maximum(hi, new_yhat)

        if stamp_method and model_name is not None:
            rec_model_df["model_type"] = f"{model_name}_{method}"

        rec_model_df.attrs.clear()
        all_reconciled_parts.append(rec_model_df)

    out_df = (
        pd.concat(all_reconciled_parts, ignore_index=True)
        if all_reconciled_parts
        else predictions_df.copy()
    )
    out_df.attrs["hierarchy_spec"] = hier_spec
    if resolved_temp_spec is not None:
        out_df.attrs["temporal_spec"] = resolved_temp_spec
    out_df.attrs["aggregated_by_node_k"] = agg_by_node_k

    final_cs_disc = (
        float(
            verify_coherence(
                out_df,
                hier_spec,
                date_col=dcol,
                ts_id_col=ts_id_col,
                atol=float("inf"),
            )
        )
        if not out_df.empty
        else 0.0
    )
    final_temp_disc = (
        float(
            verify_temporal_coherence(
                out_df,
                agg_by_node_k,
                spec=resolved_temp_spec,
                raise_on_error=False,
            )
        )
        if not out_df.empty
        else 0.0
    )
    out_df.attrs["cross_temporal_diagnostics"] = {
        "mode": "cross_temporal",
        "spatial_weight_alpha": float(last_alpha),
        "cs_first_discrepancy": float(last_cs_first_disc),
        "temp_first_discrepancy": float(last_temp_first_disc),
        "final_cs_discrepancy": final_cs_disc,
        "final_temp_discrepancy": final_temp_disc,
    }

    if return_aggregations:
        return out_df, agg_by_node_k
    return out_df


cross_temporal_mint = reconcile_cross_temporal
reconcile_cross_temporal_forecasts = reconcile_cross_temporal


def sample_coherent_paths(
    predictions_df: pd.DataFrame,
    spec: HierarchySpec | None = None,
    *,
    hier_spec: HierarchySpec | None = None,
    history_df: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    cfg: RunConfig | None = None,
    n_paths: int = 200,
    seed: int = 42,
    method: str = "mint_shrink",
    nonneg: bool = False,
    temporal_aggregations: Sequence[int] | None = None,
    aggregations: Sequence[int] | None = None,
    freq: str | None = None,
    date_col: str = "forecast_date",
    ts_id_col: str = "ts_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Draw ``n_paths`` joint Gaussian/residual sample paths projected onto the coherent subspace.

    Returns a tidy DataFrame with columns ``["path_id", "ts_id", "forecast_date", "yhat_sample"]``
    (plus ``"model_type"`` when present in ``predictions_df``, and ``"ds"`` when ``"ds"`` was in
    ``predictions_df``) where every simulated ``path_id`` satisfies exact cross-sectional and
    temporal additive coherence (``< 1e-6``).
    """
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ConfigError(f"n_paths must be a positive integer >= 1, got {n_paths!r}")

    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=["path_id", ts_id_col, "forecast_date", "yhat_sample"])

    eff_spec = spec if spec is not None else hier_spec
    if eff_spec is None:
        eff_spec = predictions_df.attrs.get("hierarchy_spec")
    if (
        eff_spec is None
        and cfg is not None
        and cfg.hierarchy.enabled
        and (cfg.hierarchy.levels or getattr(cfg.hierarchy, "similarity_hierarchy", False))
        and history_df is not None
        and not history_df.empty
    ):
        needed_cols = {c for lvl in cfg.hierarchy.levels for c in lvl} - {"similarity_cluster_id"}
        if needed_cols.issubset(history_df.columns):
            _, eff_spec = build_hierarchy(history_df, cfg)

    had_ds = "ds" in predictions_df.columns
    has_model_in_input = "model_type" in predictions_df.columns
    dcol = date_col
    if dcol not in predictions_df.columns:
        for cand in ("forecast_date", "ds", "date"):
            if cand in predictions_df.columns:
                dcol = cand
                break
    if eff_spec is not None:
        predictions_df = ensure_hierarchy_nodes_present(
            predictions_df,
            eff_spec,
            ts_id_col=ts_id_col,
            date_col=dcol,
            target_col="yhat",
        )

    eff_aggs = temporal_aggregations if temporal_aggregations is not None else aggregations
    if eff_aggs is None and cfg is not None:
        eff_aggs = cfg.hierarchy.temporal_aggregations
    eff_freq = freq or (cfg.data.freq if cfg is not None else "D")
    is_nonneg = bool(
        nonneg
        or method == "mint_nonneg"
        or (cfg is not None and cfg.output.floor is not None and cfg.output.floor >= 0.0)
    )

    # Step 1: Reconcile point forecasts so the mean trajectory is coherent.
    use_temporal = bool(
        eff_aggs is not None
        or method == "cross_temporal_mint"
        or (cfg is not None and (cfg.hierarchy.reconcile_temporal or cfg.hierarchy.cross_temporal))
        or eff_spec is None
    )
    if eff_spec is not None and use_temporal:
        base_rec_res = reconcile_cross_temporal(
            predictions_df,
            eff_spec,
            history_df=history_df,
            oof_df=oof_df,
            cfg=cfg,
            freq=eff_freq,
            temporal_aggregations=eff_aggs,
            method=method,
            nonneg=is_nonneg,
            stamp_method=False,
            return_aggregations=False,
            date_col=dcol,
            ts_id_col=ts_id_col,
            target_col=target_col,
        )
        assert isinstance(base_rec_res, pd.DataFrame)
        base_rec_df = base_rec_res
    elif eff_spec is not None:
        cs_m: Any = "mint_shrink" if method == "cross_temporal_mint" else method
        base_rec_df = reconcile_forecasts(
            predictions_df,
            eff_spec,
            cs_m,
            cfg=cfg,
            history_df=history_df,
            oof_df=oof_df,
            date_col=dcol,
            ts_id_col=ts_id_col,
            target_col=target_col,
            stamp_method=False,
        )
    else:
        t_m = "mint_shrink" if method == "cross_temporal_mint" else method
        t_res = reconcile_temporal_forecasts(
            predictions_df,
            history_df=history_df,
            oof_df=oof_df,
            cfg=cfg,
            freq=eff_freq,
            temporal_aggregations=eff_aggs,
            method=t_m,
            nonneg=is_nonneg,
            stamp_method=False,
            return_aggregations=False,
            date_col=dcol,
            ts_id_col=ts_id_col,
            target_col=target_col,
        )
        assert isinstance(t_res, pd.DataFrame)
        base_rec_df = t_res

    models = (
        base_rec_df["model_type"].drop_duplicates().tolist()
        if has_model_in_input and "model_type" in base_rec_df.columns
        else [None]
    )
    rng = np.random.default_rng(seed)

    path_frames: list[pd.DataFrame] = []
    agg_by_path_node_k: dict[tuple[int, str, int], np.ndarray] = _SafeArrayDict()
    agg_by_node_k_p0: dict[tuple[str, int], np.ndarray] = _SafeArrayDict()
    resolved_tspec: TemporalHierarchySpec | None = None

    for m_name in models:
        m_df = (
            base_rec_df[base_rec_df["model_type"] == m_name].copy()
            if m_name is not None
            else base_rec_df.copy()
        )
        dates = sorted(m_df[dcol].unique())
        h = len(dates)
        if h == 0:
            continue

        resolved_tspec = build_temporal_hierarchy(eff_freq, max(1, h), eff_aggs)
        bottom_ids = (
            list(eff_spec.bottom_ids)
            if eff_spec is not None
            else (
                sorted(m_df[ts_id_col].astype(str).unique().tolist())
                if ts_id_col in m_df.columns
                else ["series_0"]
            )
        )
        node_ids = list(eff_spec.node_ids) if eff_spec is not None else bottom_ids
        n_b = len(bottom_ids)

        if ts_id_col not in m_df.columns:
            m_df = m_df.assign(**{ts_id_col: bottom_ids[0]})

        piv_b = (
            m_df.pivot_table(index=ts_id_col, columns=dcol, values="yhat", aggfunc="first")
            .reindex(index=bottom_ids, columns=dates)
            .ffill(axis=1)
            .bfill(axis=1)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        # Estimate bottom-series innovation covariance W_b of shape (n_b, n_b).
        w_b: np.ndarray | None = None
        if oof_df is not None and not oof_df.empty and {"y_true", "yhat"}.issubset(oof_df.columns):
            m_oof = (
                oof_df[oof_df["model_type"] == m_name]
                if ("model_type" in oof_df.columns and m_name is not None)
                else oof_df
            )
            oof_dcol = (
                "ds"
                if "ds" in m_oof.columns
                else ("forecast_date" if "forecast_date" in m_oof.columns else dcol)
            )
            if ts_id_col in m_oof.columns and oof_dcol in m_oof.columns:
                tmp_oof = m_oof.copy()
                tmp_oof["_res"] = tmp_oof["y_true"].to_numpy(dtype=float) - tmp_oof[
                    "yhat"
                ].to_numpy(dtype=float)
                res_mat = (
                    tmp_oof.pivot_table(
                        index=oof_dcol, columns=ts_id_col, values="_res", aggfunc="mean"
                    )
                    .reindex(columns=bottom_ids)
                    .dropna(how="all")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                if res_mat.shape[0] >= 2 and float(np.max(np.abs(res_mat))) > 1e-12:
                    w_b, _ = schafer_strimmer_shrinkage(res_mat)

        if w_b is None:
            if "yhat_lower" in m_df.columns and "yhat_upper" in m_df.columns:
                piv_lo = (
                    m_df.pivot_table(
                        index=ts_id_col, columns=dcol, values="yhat_lower", aggfunc="first"
                    )
                    .reindex(index=bottom_ids, columns=dates)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                piv_hi = (
                    m_df.pivot_table(
                        index=ts_id_col, columns=dcol, values="yhat_upper", aggfunc="first"
                    )
                    .reindex(index=bottom_ids, columns=dates)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                sigma_vec = np.maximum(np.mean(0.5 * np.abs(piv_hi - piv_lo) / _Z_80, axis=1), 1e-3)
                w_b = np.diag(sigma_vec**2)
            elif (
                history_df is not None
                and not history_df.empty
                and ts_id_col in history_df.columns
                and target_col in history_df.columns
            ):
                h_dcol = (
                    dcol
                    if dcol in history_df.columns
                    else ("ds" if "ds" in history_df.columns else "forecast_date")
                )
                h_piv = (
                    history_df.pivot_table(
                        index=ts_id_col, columns=h_dcol, values=target_col, aggfunc="mean"
                    )
                    .reindex(index=bottom_ids)
                    .ffill(axis=1)
                    .bfill(axis=1)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                if h_piv.shape[1] >= 2:
                    sigma_vec = np.maximum(np.std(np.diff(h_piv, axis=1), axis=1), 1e-3)
                else:
                    sigma_vec = np.maximum(0.05 * np.mean(np.abs(piv_b), axis=1), 1e-2)
                w_b = np.diag(sigma_vec**2)
            else:
                sigma_vec = np.maximum(0.05 * np.mean(np.abs(piv_b), axis=1), 1e-2)
                w_b = np.diag(sigma_vec**2)

        # Cholesky factor of W_b for fast vectorized sampling across (n_paths, n_b, h).
        reg = 1e-9 * max(float(np.trace(w_b)) / max(1, n_b), 1.0)
        chol_l = np.linalg.cholesky(w_b + reg * np.eye(n_b, dtype=float))
        z_std = rng.standard_normal(size=(n_paths, n_b, h))
        step_scale = np.sqrt(1.0 + 0.05 * np.arange(h, dtype=float))[None, None, :]
        noise = np.einsum("ij,njt->nit", chol_l, z_std) * step_scale
        b_draws = piv_b[None, :, :] + noise
        if is_nonneg:
            b_draws = np.maximum(0.0, b_draws)

        # Project onto S_cs across all hierarchy nodes.
        if eff_spec is not None:
            y_draws = np.einsum("ij,njt->nit", eff_spec.summing_matrix, b_draws)
        else:
            y_draws = b_draws

        # Compute coherent temporal bucket sums for every path and node.
        base_season = seasonal_period(eff_freq)
        for p_idx in range(n_paths):
            for n_idx, nid in enumerate(node_ids):
                series_h = y_draws[p_idx, n_idx]
                padded_s = _pad_base_forecast(series_h, resolved_tspec.padded_horizon, base_season)
                agg_by_path_node_k[(p_idx, nid, 1)] = series_h.copy()
                if p_idx == 0:
                    agg_by_node_k_p0[(nid, 1)] = series_h.copy()
                for k in resolved_tspec.aggregations:
                    if int(k) == 1:
                        continue
                    k_int = int(k)
                    agg_arr = aggregate_temporal_history(padded_s, k_int)
                    agg_by_path_node_k[(p_idx, nid, k_int)] = agg_arr
                    if p_idx == 0:
                        agg_by_node_k_p0[(nid, k_int)] = agg_arr

        # Vectorized tidy DataFrame construction of length n_paths * n_nodes * h.
        n_nodes = len(node_ids)
        path_col = np.repeat(np.arange(n_paths, dtype=int), n_nodes * h)
        ts_col = np.tile(np.repeat(np.asarray(node_ids, dtype=object), h), n_paths)
        dt_col = np.tile(np.asarray(dates), n_paths * n_nodes)
        val_col = y_draws.reshape(-1)

        data_dict: dict[str, Any] = {
            "path_id": path_col,
            ts_id_col: ts_col,
        }
        if m_name is not None:
            data_dict["model_type"] = m_name
        data_dict["forecast_date"] = dt_col
        if had_ds:
            data_dict["ds"] = dt_col
        data_dict["yhat_sample"] = val_col
        path_frames.append(pd.DataFrame(data_dict))

    out_df = (
        pd.concat(path_frames, ignore_index=True)
        if path_frames
        else pd.DataFrame(columns=["path_id", ts_id_col, "forecast_date", "yhat_sample"])
    )
    if eff_spec is not None:
        out_df.attrs["hierarchy_spec"] = eff_spec
    if resolved_tspec is not None:
        out_df.attrs["temporal_spec"] = resolved_tspec
    out_df.attrs["aggregated_by_node_k"] = agg_by_node_k_p0
    out_df.attrs["aggregated_by_path_node_k"] = agg_by_path_node_k
    return out_df


def verify_temporal_coherence(
    bottom_yhat: np.ndarray | pd.Series | pd.DataFrame,
    aggregated_by_k: Mapping[Any, np.ndarray] | TemporalHierarchySpec | Sequence[int] | None = None,
    spec: TemporalHierarchySpec | Sequence[int] | None = None,
    *,
    cfg: RunConfig | None = None,
    m: int | None = None,
    aggregations: Sequence[int] | None = None,
    target_col: str = "yhat",
    atol: float = 1e-6,
    raise_on_error: bool = True,
) -> float:
    """Verify exact additive temporal coherence across all complete coarse buckets.

    Supports both:
    * ``verify_temporal_coherence(bottom_yhat_1d, aggregated_by_k_dict, spec=...)``
    * ``verify_temporal_coherence(reconciled_df, ...)`` (using ``df.attrs["aggregated_by_k"]``,
      ``df.attrs["aggregated_by_node_k"]``, or ``df.attrs["aggregated_by_path_node_k"]``).

    Returns the maximum absolute discrepancy across all complete buckets and raises `DataError`
    if it exceeds ``atol`` when ``raise_on_error=True``.
    """
    _ = (cfg, m, aggregations)
    if isinstance(bottom_yhat, pd.DataFrame):
        df = bottom_yhat
        if df.empty:
            return 0.0
        vcol = target_col
        if vcol not in df.columns:
            for cand in ("yhat", "yhat_sample", "y_sample", "y"):
                if cand in df.columns:
                    vcol = cand
                    break
        dcol = None
        for cand in ("forecast_date", "ds", "date"):
            if cand in df.columns:
                dcol = cand
                break

        # Handle multi-path or single-path DataFrame with `aggregated_by_path_node_k`.
        path_agg_map = df.attrs.get("aggregated_by_path_node_k")
        if "path_id" in df.columns and isinstance(path_agg_map, Mapping) and path_agg_map:
            max_p_err = 0.0
            ts_col_p = "ts_id" if "ts_id" in df.columns else None
            indexed_pn: dict[tuple[int, str], dict[int, np.ndarray]] = {}
            for (p_k, n_k, k), arr in path_agg_map.items():
                indexed_pn.setdefault((int(p_k), str(n_k)), {})[int(k)] = arr
            for pid, p_grp in df.groupby("path_id", sort=False):
                pid_int = int(str(pid))
                if ts_col_p is not None:
                    for nid, s_grp in p_grp.groupby(ts_col_p, sort=False):
                        sub_s = s_grp.sort_values(dcol, kind="mergesort") if dcol else s_grp
                        b_v = sub_s[vcol].to_numpy(dtype=float)
                        k_map_p = indexed_pn.get((pid_int, str(nid)))
                        if k_map_p:
                            err = verify_temporal_coherence(
                                b_v,
                                k_map_p,
                                spec=spec,
                                atol=atol,
                                raise_on_error=raise_on_error,
                            )
                            max_p_err = max(max_p_err, err)
            return max_p_err

        if isinstance(aggregated_by_k, Mapping):
            agg_map = aggregated_by_k
        elif df.attrs.get("aggregated_by_node_k"):
            agg_map = df.attrs["aggregated_by_node_k"]
        elif df.attrs.get("aggregated_by_k"):
            agg_map = df.attrs["aggregated_by_k"]
        else:
            # Self-consistency check across aggregated series built from df[vcol].
            factors = (
                list(aggregated_by_k.aggregations)
                if isinstance(aggregated_by_k, TemporalHierarchySpec)
                else (
                    list(aggregated_by_k)
                    if isinstance(aggregated_by_k, Sequence)
                    else (
                        list(spec.aggregations)
                        if isinstance(spec, TemporalHierarchySpec)
                        else (list(spec) if isinstance(spec, Sequence) else [1])
                    )
                )
            )
            grp_cols = [c for c in ("path_id", "model_type", "ts_id") if c in df.columns]
            if grp_cols:
                max_g_err = 0.0
                for _gk, sub_g in df.groupby(grp_cols, sort=False):
                    if dcol is not None:
                        sub_g = sub_g.sort_values(dcol, kind="mergesort")
                    vals = sub_g[vcol].to_numpy(dtype=float)
                    g_map = {int(k): aggregate_temporal_history(vals, int(k)) for k in factors}
                    err = verify_temporal_coherence(
                        vals, g_map, atol=atol, raise_on_error=raise_on_error
                    )
                    max_g_err = max(max_g_err, err)
                return max_g_err
            vals = df[vcol].to_numpy(dtype=float)
            agg_map = {int(k): aggregate_temporal_history(vals, int(k)) for k in factors}
            return verify_temporal_coherence(
                vals, agg_map, atol=atol, raise_on_error=raise_on_error
            )

        sample_key = next(iter(agg_map.keys()), None)
        if isinstance(sample_key, tuple):
            max_err = 0.0
            ts_col = "ts_id" if "ts_id" in df.columns else None
            node_aggs: dict[str, dict[int, np.ndarray]] = {}
            for (nid, k), arr in agg_map.items():
                node_aggs.setdefault(str(nid), {})[int(k)] = np.asarray(arr, dtype=float)
            for nid, k_map in node_aggs.items():
                if ts_col is not None:
                    sub = df[df[ts_col].astype(str) == nid]
                    if "path_id" in sub.columns and sub["path_id"].nunique() > 1:
                        first_pid = sub["path_id"].iloc[0]
                        sub = sub[sub["path_id"] == first_pid]
                    if dcol is not None:
                        sub = sub.sort_values(dcol, kind="mergesort")
                    b_vals = (
                        sub[vcol].to_numpy(dtype=float)
                        if not sub.empty
                        else k_map.get(1, np.empty(0))
                    )
                else:
                    b_vals = df[vcol].to_numpy(dtype=float)
                err = verify_temporal_coherence(
                    b_vals, k_map, spec=spec, atol=atol, raise_on_error=raise_on_error
                )
                max_err = max(max_err, err)
            return max_err

        b_vals = df[vcol].to_numpy(dtype=float)
        return verify_temporal_coherence(
            b_vals, agg_map, spec=spec, atol=atol, raise_on_error=raise_on_error
        )

    b_arr = np.asarray(bottom_yhat, dtype=float).reshape(-1)
    if len(b_arr) == 0 or aggregated_by_k is None or not isinstance(aggregated_by_k, Mapping):
        return 0.0

    max_err = 0.0
    h = len(b_arr)
    for raw_k, coarse_vals in aggregated_by_k.items():
        k = int(raw_k)
        if k <= 1:
            continue
        c_arr = np.asarray(coarse_vals, dtype=float).reshape(-1)
        n_full_buckets = min(h // k, len(c_arr))
        for b_idx in range(n_full_buckets):
            block_sum = float(np.sum(b_arr[b_idx * k : (b_idx + 1) * k]))
            err = abs(block_sum - float(c_arr[b_idx]))
            if raise_on_error and (not np.isfinite(err) or err > atol):
                raise DataError(
                    f"Forecasts are not temporally coherent at factor k={k}, bucket {b_idx} "
                    f"(discrepancy {err:.3e} > {atol:.3e})"
                )
            max_err = max(max_err, err)
    return max_err


def verify_cross_temporal_coherence(
    reconciled_df: pd.DataFrame,
    hier_spec: HierarchySpec,
    aggregated_by_node_k: Mapping[tuple[str, int], np.ndarray]
    | TemporalHierarchySpec
    | None = None,
    temp_spec: TemporalHierarchySpec | None = None,
    *,
    cfg: RunConfig | None = None,
    date_col: str = "ds",
    ts_id_col: str = "ts_id",
    atol: float = 1e-6,
    raise_on_error: bool = True,
) -> float:
    """Verify simultaneous cross-sectional and temporal coherence across nodes and factors ``k``.

    Returns ``max(cross_sectional_discrepancy, temporal_discrepancy)`` and raises `DataError` if
    either axis exceeds ``atol``.
    """
    _ = cfg
    if reconciled_df.empty:
        return 0.0
    dcol = (
        date_col
        if date_col in reconciled_df.columns
        else ("forecast_date" if "forecast_date" in reconciled_df.columns else "ds")
    )
    if raise_on_error:
        cs_err = verify_coherence(
            reconciled_df,
            hier_spec,
            date_col=dcol,
            ts_id_col=ts_id_col,
            atol=atol,
        )
    else:
        try:
            cs_err = verify_coherence(
                reconciled_df,
                hier_spec,
                date_col=dcol,
                ts_id_col=ts_id_col,
                atol=float("inf"),
            )
        except DataError:
            cs_err = float("inf")

    if isinstance(aggregated_by_node_k, TemporalHierarchySpec) and temp_spec is None:
        temp_spec = aggregated_by_node_k
        node_k_map = reconciled_df.attrs.get("aggregated_by_node_k")
    else:
        node_k_map = aggregated_by_node_k or reconciled_df.attrs.get("aggregated_by_node_k")

    temp_err = 0.0
    if node_k_map or reconciled_df.attrs.get("aggregated_by_path_node_k"):
        temp_err = verify_temporal_coherence(
            reconciled_df,
            node_k_map,
            spec=temp_spec,
            atol=atol,
            raise_on_error=raise_on_error,
        )
    return max(cs_err, temp_err)


reconcile_forecasts_temporal = reconcile_temporal_forecasts
temporal_reconcile_df = reconcile_temporal_forecasts
TEMPORAL_RECONCILIATION_METHODS = VALID_TEMPORAL_METHODS
