"""Time-series similarity, dynamical-systems phase portraits, multiscale/foundation embeddings,
clustering, and digital-twin analog search.

This single-capability module implements six specialized similarity lenses plus a default
``"hybrid"`` consensus positive-semidefinite (PSD) kernel:

1. ``"physics_attractor"`` — Takens delay-coordinate state-space reconstruction, discrete
   kinematic phase portraits :math:`(y_t, \\dot{y}_t, \\ddot{y}_t)`, Sliced Wasserstein attractor
   cloud distance, Largest Lyapunov divergence exponent (:math:`\\lambda_{\\max}`), Permutation
   Entropy (:math:`H_{\\text{perm}}`), Sample Entropy, and Ornstein-Uhlenbeck mean-reversion
   damping rate (:math:`\\kappa_{\\text{OU}}`).
2. ``"spectral_phase"`` — FFT Hilbert analytic signal :math:`z_t = y_t + i\\,\\mathcal{H}(y_t) =
   A_t e^{i\\theta_t}`, pairwise Kuramoto phase-locking order parameter :math:`R_{i,j}`, Welch
   power spectral density Jensen-Shannon divergence, and optimal cross-correlation lead-lag shift
   :math:`\\tau_{i \\to j}^*`.
3. ``"elastic_dtw"`` — Scale-normalized Sakoe-Chiba banded Dynamic Time Warping / Time-Warp Edit
   Distance (``TWED``) with physical elastic spring stiffness penalty :math:`\\nu > 0`.
4. ``"minirocket"`` — Deterministic pure-NumPy bank of 84 ternary dilated convolutional kernels
   (:math:`\\mathbf{w} \\in \\{-1, 2\\}^9`, :math:`\\sum w_k = 0`) across exponential dilations
   with Proportion of Positive Values (``PPV``) pooling.
5. ``"structural_signature"`` — 24-D standardized econometric feature vector (trend/seasonal
   strength, peak phase, Hurst exponent :math:`H` via rescaled-range :math:`R/S` analysis, ACF
   lags 1..7, PACF lags 1..3, spectral entropy, Syntetos-Boylan :math:`(\\log\\text{ADI},
   \\log\\text{CV}^2)`, skewness, kurtosis, ARCH(1) volatility clustering, level-shift ratio,
   zero-inflation fraction, velocity ratio, curvature ratio).
6. ``"foundation_latent"`` — Pooled context encoder embeddings from pre-trained time-series
   foundation models (``chronos_bolt``, ``chronos2``, ``timesfm_torch``) with automatic
   deterministic fallback to L2-normalized ``"minirocket"`` + ``"structural_signature"``
   embeddings when offline on pure core.
7. ``"hybrid"`` (default) — Consensus fusion of rank-normalized RBF similarity kernels from
   ``"physics_attractor"`` (0.30), ``"spectral_phase"`` (0.25), ``"minirocket"`` / ``"elastic_dtw"``
   shape (0.25), and ``"structural_signature"`` (0.20), projected onto the nearest symmetric PSD
   correlation matrix with unit diagonal (:math:`S_{i,i} = 1.0, S_{i,j} \\in [0, 1]`).
"""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import cluster as sp_cluster
from scipy import signal as sp_signal
from scipy.spatial import distance as sp_distance

from .config import SIMILARITY_METHODS, RunConfig, SimilarityMethod

VALID_SIMILARITY_METHODS: frozenset[str] = frozenset(SIMILARITY_METHODS)

_METHOD_ALIASES: dict[str, SimilarityMethod] = {
    "hybrid": "hybrid",
    "physics_attractor": "physics_attractor",
    "physics": "physics_attractor",
    "attractor": "physics_attractor",
    "takens": "physics_attractor",
    "spectral_phase": "spectral_phase",
    "spectral": "spectral_phase",
    "kuramoto": "spectral_phase",
    "elastic_dtw": "elastic_dtw",
    "shape_dtw": "elastic_dtw",
    "dtw": "elastic_dtw",
    "twed": "elastic_dtw",
    "minirocket": "minirocket",
    "rocket": "minirocket",
    "structural_signature": "structural_signature",
    "profile": "structural_signature",
    "econometric": "structural_signature",
    "structural": "structural_signature",
    "foundation_latent": "foundation_latent",
    "foundation": "foundation_latent",
    "chronos": "foundation_latent",
}

SimilarityMatchingMode = Literal["full_history", "launch_prefix", "recent_window"]
_MODE_ALIASES: dict[str, SimilarityMatchingMode] = {
    "full_history": "full_history",
    "full": "full_history",
    "launch_prefix": "launch_prefix",
    "partial_prefix": "launch_prefix",
    "prefix": "launch_prefix",
    "cold_start": "launch_prefix",
    "recent_window": "recent_window",
    "recent": "recent_window",
    "window": "recent_window",
}

# Precomputed 84 deterministic ternary kernels of length 9: three +2 weights and six -1 weights.
_MINIROCKET_WEIGHT_INDICES: tuple[tuple[int, int, int], ...] = tuple(
    itertools.combinations(range(9), 3)
)
_MINIROCKET_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)


def _resolve_similarity_method(method: str | None) -> SimilarityMethod:
    """Normalize a similarity method name or alias into a canonical ``SimilarityMethod``."""
    if method is None:
        return "hybrid"
    key = str(method).strip().lower()
    if key in _METHOD_ALIASES:
        return _METHOD_ALIASES[key]
    raise ValueError(
        f"Unknown similarity method {method!r}. Expected one of {sorted(VALID_SIMILARITY_METHODS)}."
    )


def _resolve_matching_mode(mode: str | None) -> SimilarityMatchingMode:
    """Normalize a digital-twin search mode or alias into a canonical mode."""
    if mode is None:
        return "full_history"
    key = str(mode).strip().lower()
    if key in _MODE_ALIASES:
        return _MODE_ALIASES[key]
    raise ValueError(
        f"Unknown similarity matching mode {mode!r}. "
        "Expected one of ('full_history', 'launch_prefix', 'recent_window')."
    )


@dataclass(frozen=True)
class SimilarityMatrixResult:
    """Result container returned by :func:`compute_similarity_matrix`.

    Attributes:
        matrix_df: Symmetric positive-semidefinite :math:`(N \\times N)` DataFrame in :math:`[0, 1]`
            with unit diagonal (:math:`S_{i,i} = 1.0`), indexed and columned by ``ts_id``.
        pairs_df: Ranked top-:math:`K` neighbor pairs DataFrame with per-lens similarity
            attributions (``physics_sim``, ``spectral_sim``, ``shape_sim``, ``structural_sim``) and
            lead-lag diagnostics (``optimal_lag``, ``lag_correlation``).
        embeddings_df: Per-series physical and econometric invariants plus 2D embedding coordinates.
        method: Canonical similarity method used.
        diagnostics: Summary diagnostics dictionary (mean similarity, PSD minimum eigenvalue,
            embedding backend used, series count).
    """

    matrix_df: pd.DataFrame
    pairs_df: pd.DataFrame
    embeddings_df: pd.DataFrame
    method: str = "hybrid"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Primary tabular view (``pairs_df``) for ergonomic DataFrame chaining."""
        return self.pairs_df

    def to_dict(self) -> dict[str, Any]:
        """Serialize the similarity result into a JSON-compatible dictionary."""
        return {
            "n_series": len(self.matrix_df),
            "method": self.method,
            "matrix": self.matrix_df.to_dict(orient="index"),
            "pairs": self.pairs_df.to_dict(orient="records"),
            "embeddings": self.embeddings_df.to_dict(orient="records"),
            "diagnostics": dict(self.diagnostics),
        }

    def __len__(self) -> int:
        return len(self.matrix_df)

    def __getitem__(self, key: Any) -> Any:
        if (
            isinstance(key, str)
            and key in self.pairs_df.columns
            and key not in self.matrix_df.columns
        ):
            return self.pairs_df[key]
        return self.matrix_df[key]


# ---------------------------------------------------------------------------
# Panel extraction & mathematical utilities
# ---------------------------------------------------------------------------


def _resolve_cols(
    panel_df: pd.DataFrame,
    cfg: RunConfig | None,
    ts_id_col: str,
    date_col: str,
    target_col: str,
) -> tuple[str, str, str]:
    if cfg is not None:
        if ts_id_col == "ts_id" and cfg.data.ts_id_col in panel_df.columns:
            ts_id_col = cfg.data.ts_id_col
        if date_col == "ds" and cfg.data.date_col in panel_df.columns:
            date_col = cfg.data.date_col
        if target_col == "y" and cfg.data.target_col in panel_df.columns:
            target_col = cfg.data.target_col
    return ts_id_col, date_col, target_col


def _clean_series_values(raw: np.ndarray) -> np.ndarray:
    """Return a finite 1D float64 array with NaNs/infinities interpolated cleanly."""
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros(arr.size, dtype=np.float64)
    if not np.all(finite_mask):
        idx = np.arange(arr.size, dtype=np.float64)
        arr = np.interp(idx, idx[finite_mask], arr[finite_mask])
    return arr


def _extract_panel_series(
    panel_df: pd.DataFrame | Mapping[str, Any],
    *,
    cfg: RunConfig | None = None,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Extract sorted series IDs and cleaned 1D float64 target arrays from a panel DataFrame."""
    if panel_df is None:
        return [], {}
    if isinstance(panel_df, Mapping) and not isinstance(panel_df, pd.DataFrame):
        if not panel_df:
            return [], {}
        s_map = {
            str(k): _clean_series_values(np.asarray(v, dtype=np.float64))
            for k, v in panel_df.items()
        }
        return sorted(s_map.keys()), s_map
    if panel_df.empty:
        return [], {}
    ts_id_col, date_col, target_col = _resolve_cols(panel_df, cfg, ts_id_col, date_col, target_col)
    if target_col not in panel_df.columns:
        for fallback in ("y", "yhat", "value", "target"):
            if fallback in panel_df.columns:
                target_col = fallback
                break
        else:
            raise ValueError(f"Target column {target_col!r} not found in DataFrame.")

    if ts_id_col not in panel_df.columns:
        vals = _clean_series_values(panel_df[target_col].to_numpy())
        return ["series_0"], {"series_0": vals}

    work = panel_df
    if date_col in work.columns:
        work = work.sort_values([ts_id_col, date_col], kind="mergesort")
    else:
        work = work.sort_values([ts_id_col], kind="mergesort")

    series_map: dict[str, np.ndarray] = {}
    for raw_id, grp in work.groupby(ts_id_col, sort=True):
        sid = str(raw_id)
        if sid == "__total__" or "=" in sid:
            continue
        series_map[sid] = _clean_series_values(grp[target_col].to_numpy())

    if not series_map:
        for raw_id, grp in work.groupby(ts_id_col, sort=True):
            sid = str(raw_id)
            series_map[sid] = _clean_series_values(grp[target_col].to_numpy())

    series_ids = sorted(series_map.keys())
    return series_ids, series_map


def _zscore(y: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    """Return ``(z_normalized, mean, std, is_flat)`` safely handling constant/zero series."""
    if y.size == 0:
        return np.zeros(1, dtype=np.float64), 0.0, 0.0, True
    mu = float(np.mean(y))
    sd = float(np.std(y))
    if not math.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(y, dtype=np.float64), mu, 0.0, True
    return (y - mu) / sd, mu, sd, False


def _project_to_psd_similarity(mat: np.ndarray) -> np.ndarray:
    """Project an :math:`(N \\times N)` matrix onto the nearest symmetric PSD matrix in
    :math:`[0, 1]` with unit diagonal."""
    n = mat.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n == 1:
        return np.ones((1, 1), dtype=np.float64)

    sym = 0.5 * (np.asarray(mat, dtype=np.float64) + np.asarray(mat, dtype=np.float64).T)
    sym = np.nan_to_num(sym, nan=0.0, posinf=1.0, neginf=0.0)
    sym = np.clip(sym, 0.0, 1.0)
    np.fill_diagonal(sym, 1.0)

    try:
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals_clipped = np.maximum(eigvals, 1e-7)
        psd = (eigvecs * eigvals_clipped) @ eigvecs.T
        d = np.sqrt(np.maximum(np.diag(psd), 1e-12))
        psd = psd / np.outer(d, d)
        psd = 0.5 * (psd + psd.T)
        psd = np.clip(psd, 0.0, 1.0)
        np.fill_diagonal(psd, 1.0)
        # Final eigenvalue guard after [0, 1] clipping.
        min_ev = float(np.min(np.linalg.eigvalsh(psd)))
        if min_ev < 0.0:
            alpha = min(1.0, max(0.0, -min_ev / (1.0 - min_ev) + 1e-6))
            psd = (1.0 - alpha) * psd + alpha * np.eye(n, dtype=np.float64)
            np.fill_diagonal(psd, 1.0)
        return psd
    except np.linalg.LinAlgError:
        return np.eye(n, dtype=np.float64)


def _distances_to_rbf_similarity(dist_mat: np.ndarray) -> np.ndarray:
    """Convert a symmetric non-negative distance matrix into an RBF similarity kernel in
    :math:`[0, 1]`."""
    n = dist_mat.shape[0]
    if n <= 1:
        return np.ones((n, n), dtype=np.float64)
    d = np.nan_to_num(np.asarray(dist_mat, dtype=np.float64), nan=0.0, posinf=1e6, neginf=0.0)
    d = np.maximum(0.5 * (d + d.T), 0.0)
    np.fill_diagonal(d, 0.0)
    triu = d[np.triu_indices(n, k=1)]
    pos = triu[triu > 1e-12]
    if pos.size == 0:
        return np.ones((n, n), dtype=np.float64)
    scale = float(np.median(pos))
    if not math.isfinite(scale) or scale < 1e-9:
        scale = float(np.mean(pos)) if pos.size > 0 else 1.0
    scale = max(scale, 1e-6)
    sim = np.exp(-d / (scale * 1.25))
    np.fill_diagonal(sim, 1.0)
    return _project_to_psd_similarity(sim)


# ---------------------------------------------------------------------------
# Multi-Modal Semantic Text, Product Attribute & Hierarchy LCA Proximity
# ---------------------------------------------------------------------------

_DEFAULT_TEXT_CANDIDATE_COLS: tuple[str, ...] = (
    "item_description",
    "product_title",
    "description",
    "title",
    "product_name",
    "item_name",
    "tags",
)
_DEFAULT_ATTR_CANDIDATE_COLS: tuple[str, ...] = (
    "brand",
    "pack_size",
    "size",
    "price_tier",
    "perishable",
    "flavor",
    "color",
    "material",
    "segment",
)
_DEFAULT_HIER_CANDIDATE_COLS: tuple[str, ...] = (
    "department",
    "category",
    "subcategory",
    "class",
    "subclass",
    "aisle",
)
_WORD_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize_col_list(
    cols: Sequence[str] | str | None,
    available_cols: Sequence[str],
) -> list[str]:
    """Normalize a column name or sequence into a deduplicated list in ``available_cols``."""
    if cols is None:
        return []
    raw_list = [cols] if isinstance(cols, str) else [str(c) for c in cols]
    avail_set = set(available_cols)
    out: list[str] = []
    for c in raw_list:
        if c in avail_set and c not in out:
            out.append(c)
    return out


def _extract_series_metadata(
    panel_df: pd.DataFrame | Mapping[str, Any] | None,
    series_ids: Sequence[str],
    *,
    cfg: RunConfig | None = None,
    ts_id_col: str = "ts_id",
) -> pd.DataFrame:
    """Extract a single metadata row per ``ts_id`` indexed in ``series_ids`` order."""
    if (
        panel_df is None
        or not isinstance(panel_df, pd.DataFrame)
        or panel_df.empty
        or len(series_ids) == 0
    ):
        return pd.DataFrame(index=pd.Index(list(series_ids), name="ts_id"))

    eff_ts_col = ts_id_col
    if cfg is not None and ts_id_col == "ts_id" and cfg.data.ts_id_col in panel_df.columns:
        eff_ts_col = cfg.data.ts_id_col
    if eff_ts_col not in panel_df.columns:
        return pd.DataFrame(index=pd.Index(list(series_ids), name="ts_id"))

    meta_cols = [c for c in panel_df.columns if c != eff_ts_col]
    if not meta_cols:
        return pd.DataFrame(index=pd.Index(list(series_ids), name="ts_id"))

    rows_by_id: dict[str, dict[str, Any]] = {}
    for raw_id, grp in panel_df.groupby(eff_ts_col, sort=False):
        sid = str(raw_id)
        row_dict: dict[str, Any] = {}
        for col in meta_cols:
            non_na = grp[col].dropna()
            row_dict[col] = non_na.iloc[0] if not non_na.empty else None
        rows_by_id[sid] = row_dict

    ordered_rows = [rows_by_id.get(sid, dict.fromkeys(meta_cols)) for sid in series_ids]
    return pd.DataFrame(ordered_rows, index=pd.Index(list(series_ids), name="ts_id"))


def _count_valid_panel_obs(
    panel_df: pd.DataFrame | Mapping[str, Any] | None,
    series_ids: Sequence[str],
    *,
    cfg: RunConfig | None = None,
    ts_id_col: str = "ts_id",
    target_col: str = "y",
) -> dict[str, int]:
    """Count the number of finite target observations per series in the raw panel."""
    counts = {str(sid): 2 for sid in series_ids}
    if panel_df is None:
        return counts
    if isinstance(panel_df, Mapping) and not isinstance(panel_df, pd.DataFrame):
        for k, v in panel_df.items():
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            counts[str(k)] = int(np.count_nonzero(np.isfinite(arr)))
        return counts
    if panel_df.empty:
        return counts

    eff_ts_col = ts_id_col
    eff_target_col = target_col
    if cfg is not None:
        if ts_id_col == "ts_id" and cfg.data.ts_id_col in panel_df.columns:
            eff_ts_col = cfg.data.ts_id_col
        if target_col == "y" and cfg.data.target_col in panel_df.columns:
            eff_target_col = cfg.data.target_col
    if eff_target_col not in panel_df.columns:
        for fallback in ("y", "yhat", "value", "target"):
            if fallback in panel_df.columns:
                eff_target_col = fallback
                break
        else:
            return counts
    if eff_ts_col not in panel_df.columns:
        vals = pd.to_numeric(panel_df[eff_target_col], errors="coerce").to_numpy(dtype=np.float64)
        for sid in series_ids:
            counts[str(sid)] = int(np.count_nonzero(np.isfinite(vals)))
        return counts

    for raw_id, grp in panel_df.groupby(eff_ts_col, sort=False):
        sid = str(raw_id)
        if sid in counts:
            vals = pd.to_numeric(grp[eff_target_col], errors="coerce").to_numpy(dtype=np.float64)
            counts[sid] = int(np.count_nonzero(np.isfinite(vals)))
    return counts


def _compute_text_ngram_lsa_similarity(texts: Sequence[str]) -> np.ndarray:
    """Compute word 1-2 gram + character 3-4 gram sublinear TF-IDF + truncated SVD LSA cosine
    similarity in :math:`[0, 1]`."""
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n == 1:
        return np.ones((1, 1), dtype=np.float64)

    doc_tokens: list[dict[str, float]] = []
    vocab_index: dict[str, int] = {}

    for raw_text in texts:
        words = _WORD_TOKEN_PATTERN.findall(str(raw_text or "").lower())
        counts: dict[str, float] = {}
        for w in words:
            tok_w1 = f"w1:{w}"
            counts[tok_w1] = counts.get(tok_w1, 0.0) + 1.0
            padded = f"#{w}#"
            plen = len(padded)
            for k in range(max(0, plen - 2)):
                c3 = f"c3:{padded[k : k + 3]}"
                counts[c3] = counts.get(c3, 0.0) + 0.35
            for k in range(max(0, plen - 3)):
                c4 = f"c4:{padded[k : k + 4]}"
                counts[c4] = counts.get(c4, 0.0) + 0.50
        for idx in range(len(words) - 1):
            tok_w2 = f"w2:{words[idx]}_{words[idx + 1]}"
            counts[tok_w2] = counts.get(tok_w2, 0.0) + 1.25
        for tok in counts:
            if tok not in vocab_index:
                vocab_index[tok] = len(vocab_index)
        doc_tokens.append(counts)

    v_size = len(vocab_index)
    if v_size == 0:
        return np.eye(n, dtype=np.float64)

    tf_mat = np.zeros((n, v_size), dtype=np.float64)
    for i, counts in enumerate(doc_tokens):
        for tok, cnt in counts.items():
            if cnt > 0.0:
                tf_mat[i, vocab_index[tok]] = 1.0 + math.log(1.0 + cnt)

    df_vec = np.count_nonzero(tf_mat > 0.0, axis=0).astype(np.float64)
    idf_vec = np.log((1.0 + float(n)) / (1.0 + df_vec)) + 1.0
    tfidf = tf_mat * idf_vec[None, :]
    row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf_norm = np.where(row_norms > 1e-12, tfidf / np.maximum(row_norms, 1e-12), 0.0)

    sim_tfidf = np.clip(tfidf_norm @ tfidf_norm.T, 0.0, 1.0)
    np.fill_diagonal(sim_tfidf, 1.0)

    if min(n, v_size) >= 3:
        try:
            u_mat, s_vals, _ = np.linalg.svd(tfidf_norm, full_matrices=False)
            k_dim = min(16, max(2, n - 1), s_vals.size)
            lsa_proj = u_mat[:, :k_dim] * s_vals[:k_dim]
            lsa_norms = np.linalg.norm(lsa_proj, axis=1, keepdims=True)
            lsa_unit = np.where(lsa_norms > 1e-12, lsa_proj / np.maximum(lsa_norms, 1e-12), 0.0)
            sim_lsa = np.clip(lsa_unit @ lsa_unit.T, 0.0, 1.0)
            # Gate LSA by lexical/subword overlap so completely disjoint texts stay near 0.0
            gate = np.clip(sim_tfidf / 0.12, 0.0, 1.0)
            sim_text = 0.70 * sim_tfidf + 0.30 * (sim_lsa * gate)
        except np.linalg.LinAlgError:
            sim_text = sim_tfidf
    else:
        sim_text = sim_tfidf

    sim_text = np.clip(0.5 * (sim_text + sim_text.T), 0.0, 1.0)
    np.fill_diagonal(sim_text, 1.0)
    return sim_text


def _compute_embedding_col_similarity(
    meta_df: pd.DataFrame,
    embedding_col: str | None,
) -> tuple[np.ndarray, bool]:
    """Parse a precomputed/LLM vector column and return its cosine similarity matrix."""
    n = len(meta_df)
    if embedding_col is None or embedding_col not in meta_df.columns or n == 0:
        return np.eye(n, dtype=np.float64), False

    parsed_vecs: list[np.ndarray | None] = []
    max_dim = 0
    for val in meta_df[embedding_col].tolist():
        if val is None or (isinstance(val, float) and math.isnan(val)):
            parsed_vecs.append(None)
            continue
        try:
            if isinstance(val, str):
                s_val = val.strip()
                arr_raw = (
                    json.loads(s_val)
                    if s_val.startswith("[")
                    else [float(x) for x in s_val.replace(",", " ").split() if x.strip()]
                )
                vec = np.asarray(arr_raw, dtype=np.float64).reshape(-1)
            else:
                vec = np.asarray(val, dtype=np.float64).reshape(-1)
            if vec.size > 0 and np.all(np.isfinite(vec)):
                parsed_vecs.append(vec)
                max_dim = max(max_dim, int(vec.size))
            else:
                parsed_vecs.append(None)
        except (ValueError, TypeError, json.JSONDecodeError):
            parsed_vecs.append(None)

    if max_dim == 0 or all(v is None for v in parsed_vecs):
        return np.eye(n, dtype=np.float64), False

    emb_mat = np.zeros((n, max_dim), dtype=np.float64)
    for idx, parsed_v in enumerate(parsed_vecs):
        if parsed_v is not None:
            emb_mat[idx, : parsed_v.size] = parsed_v
    norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
    unit_emb = np.where(norms > 1e-12, emb_mat / np.maximum(norms, 1e-12), 0.0)
    sim_emb = np.clip(unit_emb @ unit_emb.T, 0.0, 1.0)
    np.fill_diagonal(sim_emb, 1.0)
    return sim_emb, True


def _compute_attribute_gower_similarity(
    meta_df: pd.DataFrame,
    attr_cols: Sequence[str],
) -> tuple[np.ndarray, bool]:
    """Compute mixed numeric range-normalized + categorical/multi-token Jaccard Gower similarity."""
    n = len(meta_df)
    eff_cols = [c for c in attr_cols if c in meta_df.columns]
    if not eff_cols or n == 0:
        return np.eye(n, dtype=np.float64), False

    col_sims: list[np.ndarray] = []
    for col in eff_cols:
        series_col = meta_df[col]
        non_na = series_col.dropna()
        if non_na.empty:
            continue

        is_bool = pd.api.types.is_bool_dtype(series_col) or all(
            isinstance(x, (bool, np.bool_)) for x in non_na.tolist()
        )
        num_conv = pd.to_numeric(series_col, errors="coerce")
        is_numeric = (not is_bool) and int(num_conv.notna().sum()) == len(non_na)

        c_mat = np.zeros((n, n), dtype=np.float64)
        if is_numeric:
            vals = num_conv.to_numpy(dtype=np.float64)
            finite = np.isfinite(vals)
            v_min = float(np.nanmin(vals)) if np.any(finite) else 0.0
            v_max = float(np.nanmax(vals)) if np.any(finite) else 0.0
            v_range = v_max - v_min
            for i in range(n):
                c_mat[i, i] = 1.0
                for j in range(i + 1, n):
                    if not (finite[i] and finite[j]):
                        s_ij = 0.0
                    elif v_range > 1e-12:
                        s_ij = float(np.clip(1.0 - abs(vals[i] - vals[j]) / v_range, 0.0, 1.0))
                    else:
                        s_ij = 1.0
                    c_mat[i, j] = c_mat[j, i] = s_ij
        else:
            raw_list = series_col.tolist()
            for i in range(n):
                c_mat[i, i] = 1.0
                vi = raw_list[i]
                for j in range(i + 1, n):
                    vj = raw_list[j]
                    if vi is None or vj is None or pd.isna(vi) or pd.isna(vj):
                        s_ij = 0.0
                    else:
                        si = str(vi).strip().lower()
                        sj = str(vj).strip().lower()
                        if si == sj:
                            s_ij = 1.0
                        else:
                            ti = set(_WORD_TOKEN_PATTERN.findall(si))
                            tj = set(_WORD_TOKEN_PATTERN.findall(sj))
                            if len(ti) > 1 or len(tj) > 1:
                                union_len = len(ti | tj)
                                s_ij = (
                                    float(len(ti & tj)) / float(union_len) if union_len > 0 else 0.0
                                )
                            else:
                                s_ij = 0.0
                    c_mat[i, j] = c_mat[j, i] = s_ij
        col_sims.append(c_mat)

    if not col_sims:
        return np.eye(n, dtype=np.float64), False

    avg_sim = np.mean(np.stack(col_sims, axis=0), axis=0)
    avg_sim = np.clip(0.5 * (avg_sim + avg_sim.T), 0.0, 1.0)
    np.fill_diagonal(avg_sim, 1.0)
    return avg_sim, True


def _compute_semantic_similarity_matrix(
    series_ids: Sequence[str],
    meta_df: pd.DataFrame,
    *,
    text_cols: Sequence[str] | str | None = None,
    embedding_col: str | None = None,
    attribute_cols: Sequence[str] | str | None = None,
) -> tuple[np.ndarray, bool]:
    """Compute fused semantic similarity :math:`\\mathbf{S}_{\\text{sem}} \\in [0, 1]^{N \\times N}`
    across ``text_cols``, ``embedding_col``, and ``attribute_cols``."""
    n = len(series_ids)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), False
    if meta_df.empty:
        return np.eye(n, dtype=np.float64), False

    eff_text_cols = _normalize_col_list(text_cols, list(meta_df.columns))
    eff_attr_cols = _normalize_col_list(attribute_cols, list(meta_df.columns))

    weighted_mats: list[tuple[float, np.ndarray]] = []
    if eff_text_cols:
        texts: list[str] = []
        for sid in series_ids:
            parts: list[str] = []
            for col in eff_text_cols:
                val = meta_df.loc[sid, col] if sid in meta_df.index else None
                if val is not None and not pd.isna(val) and str(val).strip():
                    parts.append(str(val).strip())
            texts.append(" ".join(parts))
        if any(bool(t) for t in texts):
            s_text = _compute_text_ngram_lsa_similarity(texts)
            weighted_mats.append((0.65, s_text))

    s_emb, has_emb = _compute_embedding_col_similarity(meta_df, embedding_col)
    if has_emb:
        weighted_mats.append((0.50, s_emb))

    if eff_attr_cols:
        s_attr, has_attr = _compute_attribute_gower_similarity(meta_df, eff_attr_cols)
        if has_attr:
            weighted_mats.append((0.35, s_attr))

    if not weighted_mats:
        return np.eye(n, dtype=np.float64), False

    w_total = sum(w for w, _ in weighted_mats)
    fused_sem = np.zeros((n, n), dtype=np.float64)
    for w, mat in weighted_mats:
        fused_sem = fused_sem + w * mat
    fused_sem = fused_sem / max(w_total, 1e-9)
    return _project_to_psd_similarity(fused_sem), True


def _compute_hierarchy_lca_matrix(
    series_ids: Sequence[str],
    meta_df: pd.DataFrame,
    *,
    hierarchy_cols: Sequence[str] | str | None = None,
    cfg: RunConfig | None = None,
) -> tuple[np.ndarray, bool]:
    """Compute Wu-Palmer Lowest Common Ancestor (``LCA``) tree proximity
    :math:`S_{\\text{hier}}(i, j) = 2 \\cdot \\text{depth}(\\text{LCA}) /`
    :math:`(\\text{depth}(i) + \\text{depth}(j))`."""
    n = len(series_ids)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), False
    if meta_df.empty:
        return np.eye(n, dtype=np.float64), False

    raw_hier: list[str] = []
    if hierarchy_cols is not None:
        raw_hier = (
            [hierarchy_cols]
            if isinstance(hierarchy_cols, str)
            else [str(c) for c in hierarchy_cols]
        )
    elif cfg is not None and cfg.hierarchy.levels:
        for lvl in cfg.hierarchy.levels:
            for c in lvl:
                if c not in raw_hier:
                    raw_hier.append(str(c))

    eff_hier_cols = _normalize_col_list(raw_hier, list(meta_df.columns))
    if not eff_hier_cols:
        return np.eye(n, dtype=np.float64), False

    paths: list[list[str]] = []
    for sid in series_ids:
        path_i: list[str] = []
        for col in eff_hier_cols:
            val = meta_df.loc[sid, col] if sid in meta_df.index else None
            if val is not None and not pd.isna(val) and str(val).strip():
                path_i.append(str(val).strip().lower())
            else:
                break
        paths.append(path_i)

    if all(len(p) == 0 for p in paths):
        return np.eye(n, dtype=np.float64), False

    hier_mat = np.eye(n, dtype=np.float64)
    for i in range(n):
        pi = paths[i]
        di = len(pi)
        for j in range(i + 1, n):
            pj = paths[j]
            dj = len(pj)
            if di == 0 or dj == 0:
                s_ij = 0.0
            else:
                lca_depth = 0
                for va, vb in zip(pi, pj, strict=False):
                    if va == vb:
                        lca_depth += 1
                    else:
                        break
                s_ij = (2.0 * float(lca_depth)) / float(di + dj)
            hier_mat[i, j] = hier_mat[j, i] = s_ij

    return _project_to_psd_similarity(hier_mat), True


# ---------------------------------------------------------------------------
# Lens 1A: Physics & Dynamical Systems ("physics_attractor")
# ---------------------------------------------------------------------------


def _estimate_takens_delay(z: np.ndarray, is_flat: bool) -> int:
    """Select embedding delay :math:`\\tau` via first zero-crossing or :math:`1/e` decay of ACF."""
    n = z.size
    if is_flat or n < 6:
        return 1
    max_tau = min(7, max(1, n // 4))
    z_centered = z - np.mean(z)
    denom = float(np.dot(z_centered, z_centered))
    if denom < 1e-12:
        return 1
    threshold = 1.0 / math.e
    for tau in range(1, max_tau + 1):
        num = float(np.dot(z_centered[:-tau], z_centered[tau:]))
        acf = num / denom
        if acf <= threshold:
            return tau
    return max(1, max_tau // 2)


def _compute_permutation_entropy(
    z: np.ndarray, is_flat: bool, order: int = 3, delay: int = 1
) -> float:
    """Compute normalized Bandt-Pompe Permutation Entropy :math:`H_{\\text{perm}} \\in [0, 1]`."""
    n = z.size
    if is_flat or n < (order - 1) * delay + 2:
        return 0.0
    m = n - (order - 1) * delay
    idx = np.arange(order) * delay
    windows = z[np.arange(m)[:, None] + idx[None, :]]
    # Deterministic tie-breaking via small index jitter
    jitter = np.arange(order, dtype=np.float64)[None, :] * 1e-12
    ranks = np.argsort(windows + jitter, axis=1)
    # Encode permutation of length 3 into unique integer key
    powers = (order ** np.arange(order, dtype=np.int64))[None, :]
    keys = np.sum(ranks * powers, axis=1)
    _, counts = np.unique(keys, return_counts=True)
    probs = counts.astype(np.float64) / float(m)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-15))))
    max_entropy = math.log(float(math.factorial(order)))
    return float(np.clip(entropy / max_entropy, 0.0, 1.0)) if max_entropy > 0 else 0.0


def _compute_sample_entropy(z: np.ndarray, is_flat: bool) -> float:
    """Compute bounded Sample Entropy :math:`\\text{SampEn}(m=2, r=0.25\\sigma)`."""
    n = z.size
    if is_flat or n < 8:
        return 0.0
    work = z[-min(n, 96) :]
    w_len = work.size
    r_tol = 0.25
    m = 2
    n_templates = w_len - m
    if n_templates < 3:
        return 0.0
    xm = np.column_stack([work[k : k + n_templates] for k in range(m)])
    xm1 = np.column_stack([work[k : k + n_templates] for k in range(m + 1)])
    dist_m = sp_distance.cdist(xm, xm, metric="chebyshev")
    dist_m1 = sp_distance.cdist(xm1, xm1, metric="chebyshev")
    np.fill_diagonal(dist_m, np.inf)
    np.fill_diagonal(dist_m1, np.inf)
    b_count = float(np.sum(dist_m <= r_tol))
    a_count = float(np.sum(dist_m1 <= r_tol))
    if b_count <= 0.0:
        return 0.0
    if a_count <= 0.0:
        return float(math.log(max(b_count, 1.0)))
    return float(np.clip(-math.log(a_count / b_count), 0.0, 4.0))


def _compute_lyapunov_exponent(z: np.ndarray, is_flat: bool, tau: int = 1) -> float:
    """Estimate the Largest Lyapunov Divergence Exponent :math:`\\lambda_{\\max}` via Rosenstein
    nearest-neighbor divergence in Takens delay-coordinate space."""
    n = z.size
    if is_flat or n < 8:
        return 0.0
    m = 3
    delay = max(1, min(tau, (n - 4) // max(1, m - 1)))
    n_vec = n - (m - 1) * delay
    max_steps = min(5, max(2, n_vec // 4))
    n_usable = n_vec - max_steps
    if n_usable < 4:
        # Short-series fallback: local step expansion rate
        dz = np.abs(np.diff(z))
        if dz.size < 2 or np.mean(dz) < 1e-12:
            return 0.0
        return float(np.clip(np.std(dz) / (np.mean(dz) + 1e-9) - 0.5, -0.5, 2.0))

    cloud = np.column_stack([z[k * delay : k * delay + n_vec] for k in range(m)])
    usable = cloud[:n_usable]
    dists = sp_distance.cdist(usable, usable, metric="euclidean")
    # Enforce temporal separation (Theiler window >= max(2, delay)) to exclude autocorrelation
    theiler = max(2, delay)
    idx_grid = np.arange(n_usable)
    temp_mask = np.abs(idx_grid[:, None] - idx_grid[None, :]) <= theiler
    dists[temp_mask] = np.inf
    dists[dists < 1e-9] = np.inf

    nn_idx = np.argmin(dists, axis=1)
    nn_d0 = dists[idx_grid, nn_idx]
    valid = np.isfinite(nn_d0) & (nn_d0 > 1e-9)
    if np.sum(valid) < 3:
        return 0.0

    src = idx_grid[valid]
    nbr = nn_idx[valid]
    d0 = nn_d0[valid]
    log_ratios: list[float] = []
    for k in range(1, max_steps + 1):
        dk = np.linalg.norm(cloud[src + k] - cloud[nbr + k], axis=1)
        ratio = np.maximum(dk, 1e-9) / np.maximum(d0, 1e-9)
        log_ratios.append(float(np.mean(np.log(ratio))))

    steps = np.arange(1, max_steps + 1, dtype=np.float64)
    y_div = np.array([0.0, *log_ratios], dtype=np.float64)
    x_steps = np.array([0.0, *steps], dtype=np.float64)
    slope = float(np.polyfit(x_steps, y_div, 1)[0])

    # Incorporate amplitude envelope decay/expansion so damped OU has strictly smaller lambda than
    # sustained/chaotic trajectories.
    half = n // 2
    first_std = float(np.std(z[:half])) if half >= 2 else 1.0
    second_std = float(np.std(z[half:])) if (n - half) >= 2 else 1.0
    envelope_rate = math.log((second_std + 1e-4) / (first_std + 1e-4)) / max(float(half), 1.0)
    combined = 0.75 * slope + 0.25 * envelope_rate
    return float(np.clip(combined, -1.5, 2.5))


def _compute_ou_damping_rate(z: np.ndarray, is_flat: bool) -> float:
    """Estimate the Ornstein-Uhlenbeck mean-reversion damping rate
    :math:`\\kappa_{\\text{OU}} \\ge 0`."""
    if is_flat or z.size < 4:
        return 0.0
    x_prev = z[:-1] - np.mean(z[:-1])
    x_next = z[1:] - np.mean(z[1:])
    denom = float(np.dot(x_prev, x_prev))
    if denom < 1e-12:
        return 0.0
    phi1 = float(np.dot(x_prev, x_next) / denom)
    if phi1 >= 0.999:
        return 0.0
    phi_clamped = float(np.clip(phi1, 0.02, 0.999))
    return float(np.clip(-math.log(phi_clamped), 0.0, 4.0))


def _physics_phase_cloud_quantiles(
    z: np.ndarray, is_flat: bool, n_quantiles: int = 16
) -> np.ndarray:
    """Summarize the 3D kinematic phase portrait :math:`(z_t, \\dot{z}_t, \\ddot{z}_t)` and Takens
    attractor cloud via 1D marginal and sliced quantile projections (for Sliced Wasserstein-1)."""
    q_grid = np.linspace(0.05, 0.95, n_quantiles)
    if is_flat or z.size < 3:
        return np.zeros(5 * n_quantiles, dtype=np.float64)
    dz = np.diff(z, prepend=z[0])
    d2z = np.diff(dz, prepend=dz[0])
    tau = _estimate_takens_delay(z, is_flat)
    z_tau = np.roll(z, tau)
    z_tau[:tau] = z[0]
    diag_proj = (z + dz + z_tau) / math.sqrt(3.0)
    cross_proj = (z - dz) / math.sqrt(2.0)
    feats = [
        np.quantile(z, q_grid),
        np.quantile(dz, q_grid),
        np.quantile(d2z, q_grid),
        np.quantile(diag_proj, q_grid),
        np.quantile(cross_proj, q_grid),
    ]
    return np.concatenate(feats).astype(np.float64)


def _compute_physics_attractor_matrix(
    series_ids: list[str],
    series_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Compute the ``"physics_attractor"`` similarity matrix and per-series dynamical invariants."""
    n = len(series_ids)
    invariants: dict[str, dict[str, float]] = {}
    cloud_vecs: list[np.ndarray] = []
    inv_vecs: list[np.ndarray] = []

    for sid in series_ids:
        y = series_map[sid]
        z, _, _, is_flat = _zscore(y)
        tau = _estimate_takens_delay(z, is_flat)
        lyap = _compute_lyapunov_exponent(z, is_flat, tau=tau)
        perm_ent = _compute_permutation_entropy(z, is_flat, order=3, delay=1)
        samp_ent = _compute_sample_entropy(z, is_flat)
        ou_kappa = _compute_ou_damping_rate(z, is_flat)
        dz = np.diff(z) if z.size >= 2 else np.zeros(1, dtype=np.float64)
        d2z = np.diff(dz) if dz.size >= 2 else np.zeros(1, dtype=np.float64)
        vel_std = 0.0 if is_flat else float(np.std(dz))
        acc_std = 0.0 if is_flat else float(np.std(d2z))
        invariants[sid] = {
            "lyapunov_exp": lyap,
            "perm_entropy": perm_ent,
            "sample_entropy": samp_ent,
            "ou_damping_rate": ou_kappa,
            "velocity_std": vel_std,
            "acceleration_std": acc_std,
        }
        cloud_vecs.append(_physics_phase_cloud_quantiles(z, is_flat))
        inv_vecs.append(
            np.array(
                [lyap, perm_ent, samp_ent, ou_kappa, vel_std, acc_std, 1.0 if is_flat else 0.0],
                dtype=np.float64,
            )
        )

    if n <= 1:
        return np.ones((n, n), dtype=np.float64), invariants

    cloud_mat = np.vstack(cloud_vecs)
    inv_mat = np.vstack(inv_vecs)
    inv_std = np.std(inv_mat, axis=0)
    inv_std[inv_std < 1e-9] = 1.0
    inv_norm = (inv_mat - np.mean(inv_mat, axis=0)) / inv_std

    # Sliced Wasserstein-1 (mean L1 over quantile projections) + normalized dynamical invariants
    w1_dist = sp_distance.squareform(sp_distance.pdist(cloud_mat, metric="cityblock")) / float(
        cloud_mat.shape[1]
    )
    inv_dist = sp_distance.squareform(sp_distance.pdist(inv_norm, metric="euclidean")) / math.sqrt(
        float(inv_norm.shape[1])
    )
    total_dist = 0.65 * w1_dist + 0.35 * inv_dist
    return _distances_to_rbf_similarity(total_dist), invariants


# ---------------------------------------------------------------------------
# Lens 1B: Spectral Phase & Kuramoto Synchronization ("spectral_phase")
# ---------------------------------------------------------------------------


def _spectral_profile(
    z: np.ndarray,
    is_flat: bool,
    n_freq_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(unit_phasor, psd_prob, dominant_period, spectral_entropy)``."""
    if is_flat or z.size < 4:
        uniform = np.full(n_freq_bins, 1.0 / float(n_freq_bins), dtype=np.float64)
        return np.ones(max(z.size, 1), dtype=np.complex128), uniform, 1.0, 0.0

    # Detrend linear slope before Hilbert transform so instantaneous phase tracks oscillations
    t_idx = np.arange(z.size, dtype=np.float64)
    slope, intercept = np.polyfit(t_idx, z, 1)
    detrended = z - (slope * t_idx + intercept)
    if float(np.std(detrended)) < 1e-9:
        detrended = z

    analytic = sp_signal.hilbert(detrended)
    amp = np.abs(analytic)
    phasor = np.where(amp > 1e-12, analytic / np.maximum(amp, 1e-12), 1.0 + 0.0j)

    # Power spectral density via FFT periodogram interpolated onto common normalized frequency grid
    fft_vals = np.fft.rfft(detrended * np.hanning(detrended.size))
    power = np.abs(fft_vals) ** 2
    if power.size > 1:
        power[0] = 0.0  # exclude DC offset
    freqs = np.linspace(0.0, 0.5, power.size)
    grid_freqs = np.linspace(0.01, 0.5, n_freq_bins)
    interp_psd = np.interp(grid_freqs, freqs, power)
    psd_sum = float(np.sum(interp_psd))
    if psd_sum <= 1e-12:
        psd_prob = np.full(n_freq_bins, 1.0 / float(n_freq_bins), dtype=np.float64)
        dom_period = 1.0
        spec_ent = 0.0
    else:
        psd_prob = np.maximum(interp_psd / psd_sum, 1e-12)
        psd_prob = psd_prob / float(np.sum(psd_prob))
        peak_idx = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
        peak_freq = freqs[peak_idx] if peak_idx > 0 and freqs[peak_idx] > 1e-6 else 1.0
        dom_period = float(np.clip(round(1.0 / peak_freq, 2), 1.0, float(z.size)))
        raw_ent = -float(np.sum(psd_prob * np.log(psd_prob)))
        spec_ent = float(np.clip(raw_ent / math.log(float(n_freq_bins)), 0.0, 1.0))

    return phasor, psd_prob, dom_period, spec_ent


def _compute_optimal_lead_lag(
    za: np.ndarray,
    zb: np.ndarray,
    flat_a: bool,
    flat_b: bool,
    max_lag: int,
) -> tuple[int, float]:
    """Find the optimal cross-correlation lag :math:`\\tau \\in [-\\tau_{\\max}, +\\tau_{\\max}]`.

    Sign convention: ``optimal_lag = +L`` (:math:`L > 0`) means series ``a`` leads series ``b`` by
    :math:`L` steps (i.e., ``zb[t]`` matches ``za[t - L]`` so ``corr(za[:-L], zb[L:])`` is
    maximized).
    """
    if flat_a and flat_b:
        return 0, 1.0
    if flat_a or flat_b:
        return 0, 0.0

    m = min(za.size, zb.size)
    if m < 3:
        return 0, 0.0
    xa = za[-m:]
    xb = zb[-m:]

    def _safe_corr(u: np.ndarray, v: np.ndarray) -> float:
        if u.size < 3:
            return 0.0
        u0 = u - np.mean(u)
        v0 = v - np.mean(v)
        den = float(np.linalg.norm(u0) * np.linalg.norm(v0))
        if den < 1e-12:
            return 0.0
        return float(np.clip(np.dot(u0, v0) / den, -1.0, 1.0))

    lag0_corr = _safe_corr(xa, xb)
    if max_lag <= 0:
        return 0, lag0_corr

    eff_max_lag = min(int(max_lag), max(0, m // 3))
    best_lag = 0
    best_corr = lag0_corr
    for lag in range(-eff_max_lag, eff_max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            c = _safe_corr(xa[:-lag], xb[lag:])
        else:
            k = -lag
            c = _safe_corr(xa[k:], xb[:-k])
        # Require a small margin over lag=0 to avoid noise jitter on synchronous series
        if c > best_corr + 1e-4:
            best_corr = c
            best_lag = lag
    return best_lag, best_corr


def _compute_spectral_phase_matrix(
    series_ids: list[str],
    series_map: dict[str, np.ndarray],
    *,
    max_lag: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Return ``(spectral_sim_matrix, optimal_lag_matrix, lag_corr_matrix, spectral_meta)``."""
    n = len(series_ids)
    phasors: list[np.ndarray] = []
    psds: list[np.ndarray] = []
    z_list: list[np.ndarray] = []
    flat_flags: list[bool] = []
    meta: dict[str, dict[str, float]] = {}

    for sid in series_ids:
        y = series_map[sid]
        z, _, _, is_flat = _zscore(y)
        phasor, psd_prob, dom_p, spec_ent = _spectral_profile(z, is_flat)
        phasors.append(phasor)
        psds.append(psd_prob)
        z_list.append(z)
        flat_flags.append(is_flat)
        meta[sid] = {
            "dominant_period": dom_p,
            "spectral_entropy": spec_ent,
            "kuramoto_sync_mean": 1.0,
        }

    sim_mat = np.eye(n, dtype=np.float64)
    lag_mat = np.zeros((n, n), dtype=np.int64)
    corr_mat = np.eye(n, dtype=np.float64)
    kuramoto_mat = np.eye(n, dtype=np.float64)

    if n <= 1:
        return sim_mat, lag_mat, corr_mat, meta

    ln2 = math.log(2.0)
    for i in range(n):
        for j in range(i + 1, n):
            if flat_flags[i] and flat_flags[j]:
                r_ij = 1.0
                psd_sim = 1.0
                opt_lag, lag_corr = 0, 1.0
            elif flat_flags[i] or flat_flags[j]:
                r_ij = 0.0
                psd_sim = 0.0
                opt_lag, lag_corr = 0, 0.0
            else:
                m_len = min(phasors[i].size, phasors[j].size)
                ph_i = phasors[i][-m_len:]
                ph_j = phasors[j][-m_len:]
                # Kuramoto order parameter R_ij = |1/T sum exp(i (theta_i - theta_j))|
                r_ij = float(np.clip(np.abs(np.mean(ph_i * np.conj(ph_j))), 0.0, 1.0))

                # Welch/FFT Power Spectral Density Jensen-Shannon Divergence
                pi = psds[i]
                pj = psds[j]
                pm = 0.5 * (pi + pj)
                kl_i = float(np.sum(pi * np.log(np.maximum(pi / pm, 1e-15))))
                kl_j = float(np.sum(pj * np.log(np.maximum(pj / pm, 1e-15))))
                jsd = max(0.0, 0.5 * (kl_i + kl_j))
                psd_sim = float(np.clip(1.0 - (jsd / ln2), 0.0, 1.0))

                opt_lag, lag_corr = _compute_optimal_lead_lag(
                    z_list[i], z_list[j], flat_flags[i], flat_flags[j], max_lag=max_lag
                )

            kuramoto_mat[i, j] = kuramoto_mat[j, i] = r_ij
            lag_mat[i, j] = opt_lag
            lag_mat[j, i] = -opt_lag
            corr_mat[i, j] = corr_mat[j, i] = lag_corr

            # Combine Kuramoto synchronization, spectral PSD similarity, and lag-aligned correlation
            pos_corr = max(0.0, lag_corr)
            pair_sim = 0.45 * r_ij + 0.35 * psd_sim + 0.20 * pos_corr
            sim_mat[i, j] = sim_mat[j, i] = float(np.clip(pair_sim, 0.0, 1.0))

    for i, sid in enumerate(series_ids):
        if n > 1:
            others = [kuramoto_mat[i, j] for j in range(n) if j != i]
            meta[sid]["kuramoto_sync_mean"] = float(np.mean(others))

    return _project_to_psd_similarity(sim_mat), lag_mat, corr_mat, meta


# ---------------------------------------------------------------------------
# Lens 1C: Physical Elastic Spring DTW ("elastic_dtw")
# ---------------------------------------------------------------------------


def _elastic_dtw_distance(
    za: np.ndarray,
    zb: np.ndarray,
    flat_a: bool,
    flat_b: bool,
    *,
    spring_stiffness: float = 0.05,
    window_frac: float = 0.15,
) -> float:
    """Compute scale-normalized Sakoe-Chiba banded Dynamic Time Warping with elastic spring
    stiffness penalty :math:`\\nu \\cdot |t_a - t_b|`."""
    if flat_a and flat_b:
        return 0.0
    if flat_a or flat_b:
        return 2.5

    max_len = 64
    xa = (
        za
        if za.size <= max_len
        else np.interp(np.linspace(0.0, 1.0, max_len), np.linspace(0.0, 1.0, za.size), za)
    )
    xb = (
        zb
        if zb.size <= max_len
        else np.interp(np.linspace(0.0, 1.0, max_len), np.linspace(0.0, 1.0, zb.size), zb)
    )
    if xa.size != xb.size:
        target_n = min(max_len, max(xa.size, xb.size))
        xa = np.interp(np.linspace(0.0, 1.0, target_n), np.linspace(0.0, 1.0, xa.size), xa)
        xb = np.interp(np.linspace(0.0, 1.0, target_n), np.linspace(0.0, 1.0, xb.size), xb)

    m = xa.size
    band = max(2, math.ceil(float(window_frac) * m))
    nu = max(0.0, float(spring_stiffness))

    # Vectorized diagonal/row banded DP
    dp = np.full((m + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, m + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        xi = xa[i - 1]
        for j in range(j_start, j_end + 1):
            diff = abs(xi - xb[j - 1])
            spring_cost = nu * abs(i - j) / float(m)
            cost = diff + spring_cost
            prev = dp[i - 1, j - 1]
            if dp[i - 1, j] < prev:
                prev = dp[i - 1, j]
            if dp[i, j - 1] < prev:
                prev = dp[i, j - 1]
            dp[i, j] = cost + prev

    return float(dp[m, m] / float(m))


def _compute_elastic_dtw_matrix(
    series_ids: list[str],
    series_map: dict[str, np.ndarray],
    *,
    spring_stiffness: float = 0.05,
    window_frac: float = 0.15,
) -> np.ndarray:
    """Compute the ``"elastic_dtw"`` similarity matrix across all series pairs."""
    n = len(series_ids)
    if n <= 1:
        return np.ones((n, n), dtype=np.float64)

    z_list: list[np.ndarray] = []
    flat_flags: list[bool] = []
    for sid in series_ids:
        z, _, _, is_flat = _zscore(series_map[sid])
        z_list.append(z)
        flat_flags.append(is_flat)

    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _elastic_dtw_distance(
                z_list[i],
                z_list[j],
                flat_flags[i],
                flat_flags[j],
                spring_stiffness=spring_stiffness,
                window_frac=window_frac,
            )
            dist_mat[i, j] = dist_mat[j, i] = d
    return _distances_to_rbf_similarity(dist_mat)


# ---------------------------------------------------------------------------
# Lens 2A: Multiscale Convolutional Kernel Bank ("minirocket")
# ---------------------------------------------------------------------------


def _minirocket_embed_single(z: np.ndarray, is_flat: bool) -> np.ndarray:
    """Compute a deterministic 84-kernel multiscale dilated convolutional PPV embedding."""
    n_kernels = len(_MINIROCKET_WEIGHT_INDICES)
    n_dilations = len(_MINIROCKET_DILATIONS)
    out_dim = n_kernels * n_dilations
    if is_flat or z.size < 2:
        return np.zeros(out_dim, dtype=np.float64)

    features = np.empty(out_dim, dtype=np.float64)
    offset = 0
    for d in _MINIROCKET_DILATIONS:
        span = 8 * d + 1
        if z.size < span:
            pad_total = span - z.size + 4
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            work = np.pad(z, (pad_left, pad_right), mode="edge")
        else:
            work = z
        m = work.size - 8 * d
        # Build (9, m) shifted views for dilation d
        patches = np.vstack([work[k * d : k * d + m] for k in range(9)])
        neg_sum = -np.sum(patches, axis=0)  # baseline if all 9 weights were -1
        for i0, i1, i2 in _MINIROCKET_WEIGHT_INDICES:
            # Adding 3 * patch at the three +2 positions converts -1 to +2!
            conv = neg_sum + 3.0 * (patches[i0] + patches[i1] + patches[i2])
            # PPV against zero and unit-scale threshold
            ppv_zero = float(np.mean(conv > 0.0))
            ppv_pos = float(np.mean(conv > 0.75))
            features[offset] = 0.65 * ppv_zero + 0.35 * ppv_pos
            offset += 1
    return features


def _compute_minirocket_embeddings_and_matrix(
    series_ids: list[str],
    series_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(embeddings_matrix, similarity_matrix)`` for ``"minirocket"``."""
    n = len(series_ids)
    if n == 0:
        return np.zeros((0, 336), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    rows: list[np.ndarray] = []
    for sid in series_ids:
        z, _, _, is_flat = _zscore(series_map[sid])
        rows.append(_minirocket_embed_single(z, is_flat))
    emb = np.vstack(rows)
    if n == 1:
        return emb, np.ones((1, 1), dtype=np.float64)
    dist_mat = sp_distance.squareform(sp_distance.pdist(emb, metric="euclidean")) / math.sqrt(
        float(emb.shape[1])
    )
    return emb, _distances_to_rbf_similarity(dist_mat)


# ---------------------------------------------------------------------------
# Lens 2B: 24-D Econometric Structural Signature ("structural_signature")
# ---------------------------------------------------------------------------


def _compute_hurst_exponent(y: np.ndarray, is_flat: bool) -> float:
    """Estimate the Hurst exponent :math:`H \\in (0, 1)` via rescaled-range (:math:`R/S`) analysis
    combined with multi-lag variogram scaling."""
    n = y.size
    if is_flat or n < 8:
        return 0.5
    z = (y - np.mean(y)) / (np.std(y) + 1e-12)

    # 1. Rescaled-range (R/S) across dyadic window sizes on increments/levels
    window_sizes = [w for w in (8, 12, 16, 24, 32, 48, 64) if w <= n]
    rs_points: list[tuple[float, float]] = []
    for w in window_sizes:
        n_blocks = n // w
        if n_blocks < 1:
            continue
        rs_vals: list[float] = []
        for b in range(n_blocks):
            seg = z[b * w : (b + 1) * w]
            seg_dev = np.cumsum(seg - np.mean(seg))
            r_val = float(np.max(seg_dev) - np.min(seg_dev))
            s_val = float(np.std(seg))
            if s_val > 1e-9 and r_val > 1e-9:
                rs_vals.append(r_val / s_val)
        if rs_vals:
            rs_points.append((math.log(float(w)), math.log(float(np.mean(rs_vals)))))

    h_rs = 0.5
    if len(rs_points) >= 2:
        xs = np.array([p[0] for p in rs_points], dtype=np.float64)
        ys = np.array([p[1] for p in rs_points], dtype=np.float64)
        h_rs = float(np.polyfit(xs, ys, 1)[0])

    # 2. Lag-1 autocorrelation of levels to sharpen anti-persistent vs persistent regimes
    z0 = z - np.mean(z)
    acf1 = float(np.dot(z0[:-1], z0[1:]) / (np.dot(z0, z0) + 1e-12))
    h_acf = 0.5 + 0.35 * acf1
    h_est = 0.60 * h_rs + 0.40 * h_acf
    return float(np.clip(h_est, 0.05, 0.98))


def _structural_signature_single(
    y: np.ndarray,
    *,
    spectral_meta: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract the 24-D econometric structural signature vector and named summary metrics."""
    z, _mu, _sd, is_flat = _zscore(y)
    n = y.size
    zero_frac = float(np.mean(np.abs(y) <= 1e-9)) if n > 0 else 1.0

    # Syntetos-Boylan intermittency coordinates (log ADI and log(1 + CV^2))
    nz_idx = np.flatnonzero(np.abs(y) > 1e-9)
    if nz_idx.size == 0:
        adi = float(max(n, 1))
        cv2 = 0.0
    else:
        intervals = np.diff(nz_idx, prepend=-1)
        adi = float(np.mean(intervals)) if intervals.size > 0 else 1.0
        nz_vals = y[nz_idx]
        nz_mean = float(np.mean(np.abs(nz_vals)))
        nz_std = float(np.std(nz_vals))
        cv2 = (nz_std / max(nz_mean, 1e-9)) ** 2
    adi_log = float(math.log(max(adi, 1.0)))
    cv2_log = float(math.log1p(max(cv2, 0.0)))

    if is_flat or n < 4:
        vec = np.zeros(24, dtype=np.float64)
        vec[3] = 0.5  # Hurst H = 0.5 neutral
        vec[15] = adi_log
        vec[16] = cv2_log
        vec[21] = zero_frac
        return vec, {
            "hurst_exp": 0.5,
            "trend_strength": 0.0,
            "seasonal_strength": 0.0,
            "spectral_entropy": 0.0,
            "adi_log": adi_log,
            "cv2_log": cv2_log,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "arch_volatility": 0.0,
            "zero_fraction": zero_frac,
        }

    # Trend & seasonal decomposition via smooth moving average + seasonal folding
    win = max(3, min(7, n // 3))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    trend = np.convolve(z, kernel, mode="same")
    detrended = z - trend
    var_z = float(np.var(z)) + 1e-12
    var_detrend = float(np.var(detrended))
    trend_strength = float(np.clip(1.0 - var_detrend / var_z, 0.0, 1.0))

    if spectral_meta is not None:
        dom_p = spectral_meta.get("dominant_period", 7.0)
        spec_ent = spectral_meta.get("spectral_entropy", 0.5)
    else:
        _, _, dom_p, spec_ent = _spectral_profile(z, is_flat)

    period_int = max(2, min(round(dom_p), max(2, n // 2)))
    if n >= 2 * period_int:
        n_full = (n // period_int) * period_int
        folded = detrended[:n_full].reshape(-1, period_int)
        seas_pattern = np.mean(folded, axis=0)
        seas_tiled = np.tile(seas_pattern, n // period_int + 1)[:n]
        resid = detrended - seas_tiled
        seasonal_strength = float(
            np.clip(1.0 - float(np.var(resid)) / max(var_detrend, 1e-12), 0.0, 1.0)
        )
        peak_phase = float(np.argmax(seas_pattern)) / float(period_int)
    else:
        resid = detrended
        seasonal_strength = 0.0
        peak_phase = 0.0

    hurst = _compute_hurst_exponent(y, is_flat)

    # ACF lags 1..7
    z0 = z - np.mean(z)
    denom = float(np.dot(z0, z0)) + 1e-12
    acfs: list[float] = []
    for k in range(1, 8):
        if k < n:
            acfs.append(float(np.clip(np.dot(z0[:-k], z0[k:]) / denom, -1.0, 1.0)))
        else:
            acfs.append(0.0)

    # PACF lags 1..3 via Yule-Walker / Durbin-Levinson recursion
    r1, r2, r3 = acfs[0], acfs[1], acfs[2]
    pacf1 = r1
    denom2 = max(1.0 - r1 * r1, 1e-6)
    pacf2 = float(np.clip((r2 - r1 * r1) / denom2, -1.0, 1.0))
    phi21 = r1 * (1.0 - pacf2)
    phi22 = pacf2
    denom3 = max(1.0 - phi21 * r1 - phi22 * r2, 1e-6)
    pacf3 = float(np.clip((r3 - phi21 * r2 - phi22 * r1) / denom3, -1.0, 1.0))

    # Higher moments & ARCH(1) conditional volatility clustering
    skewness = float(np.clip(np.mean(z**3), -5.0, 5.0))
    kurtosis = float(np.clip(np.mean(z**4) - 3.0, -3.0, 15.0))
    sq_resid = resid**2
    sq0 = sq_resid - np.mean(sq_resid)
    sq_denom = float(np.dot(sq0, sq0))
    arch_vol = (
        float(np.clip(np.dot(sq0[:-1], sq0[1:]) / sq_denom, -1.0, 1.0))
        if sq_denom > 1e-12 and sq0.size >= 2
        else 0.0
    )

    # Level-shift ratio, velocity ratio, curvature ratio
    half = max(1, n // 2)
    level_shift = float(abs(np.mean(z[half:]) - np.mean(z[:half])))
    dz = np.diff(z)
    d2z = np.diff(dz) if dz.size >= 2 else np.zeros(1, dtype=np.float64)
    vel_ratio = float(np.std(dz))
    curv_ratio = float(np.std(d2z))

    vec = np.array(
        [
            trend_strength,
            seasonal_strength,
            peak_phase,
            hurst,
            *acfs,
            pacf1,
            pacf2,
            pacf3,
            spec_ent,
            adi_log,
            cv2_log,
            skewness,
            kurtosis,
            arch_vol,
            level_shift,
            zero_frac,
            vel_ratio,
            curv_ratio,
        ],
        dtype=np.float64,
    )
    summary = {
        "hurst_exp": hurst,
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "spectral_entropy": spec_ent,
        "adi_log": adi_log,
        "cv2_log": cv2_log,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "arch_volatility": arch_vol,
        "zero_fraction": zero_frac,
    }
    return vec, summary


def _compute_structural_signature_matrix(
    series_ids: list[str],
    series_map: dict[str, np.ndarray],
    *,
    spectral_meta_map: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Return ``(structural_vectors, similarity_matrix, structural_meta)``."""
    n = len(series_ids)
    if n == 0:
        return np.zeros((0, 24), dtype=np.float64), np.zeros((0, 0), dtype=np.float64), {}

    rows: list[np.ndarray] = []
    meta: dict[str, dict[str, float]] = {}
    for sid in series_ids:
        sp_m = spectral_meta_map.get(sid) if spectral_meta_map is not None else None
        vec, summ = _structural_signature_single(series_map[sid], spectral_meta=sp_m)
        rows.append(vec)
        meta[sid] = summ

    mat = np.vstack(rows)
    if n == 1:
        return mat, np.ones((1, 1), dtype=np.float64), meta

    col_std = np.std(mat, axis=0)
    col_std[col_std < 1e-9] = 1.0
    norm_mat = (mat - np.mean(mat, axis=0)) / col_std
    dist_mat = sp_distance.squareform(sp_distance.pdist(norm_mat, metric="euclidean")) / math.sqrt(
        float(norm_mat.shape[1])
    )
    return mat, _distances_to_rbf_similarity(dist_mat), meta


# ---------------------------------------------------------------------------
# Lens 2C: Foundation Model Latent Embeddings ("foundation_latent")
# ---------------------------------------------------------------------------


def extract_foundation_embeddings(
    panel_df: pd.DataFrame,
    *,
    model_name: str = "chronos_bolt",
    cfg: RunConfig | None = None,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    allow_fallback: bool = True,
    encoder_fn: Callable[[list[np.ndarray]], np.ndarray] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, list[str], str]:
    """Extract pooled encoder embeddings for each series in ``panel_df``.

    When ``encoder_fn`` is provided or a foundation model pipeline (``chronos_bolt``,
    ``chronos2``, ``timesfm_torch``) is explicitly loaded/configured, extracts latent encoder
    states. When running offline on pure core (default ``allow_fallback=True``), deterministically
    falls back to the L2-normalized concatenation of ``"minirocket"`` and
    ``"structural_signature"`` embeddings and records
    ``embedding_backend_used = "minirocket_structural_fallback"``.
    """
    series_ids, series_map = _extract_panel_series(
        panel_df, cfg=cfg, ts_id_col=ts_id_col, date_col=date_col, target_col=target_col
    )
    n = len(series_ids)
    if n == 0:
        return np.zeros((0, 16), dtype=np.float64), [], "minirocket_structural_fallback"

    if encoder_fn is not None:
        raw_list = [series_map[sid] for sid in series_ids]
        emb = np.asarray(encoder_fn(raw_list), dtype=np.float64)
        if emb.ndim == 1:
            emb = emb.reshape(n, -1)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return emb / norms, series_ids, str(model_name)

    if kwargs.get("use_pretrained_pipeline", False):
        try:
            import torch
            from chronos import ChronosBoltPipeline

            repo_id = str(kwargs.get("repo_id", "amazon/chronos-bolt-small"))
            pipeline = ChronosBoltPipeline.from_pretrained(repo_id, device_map="cpu")
            if hasattr(pipeline, "embed"):
                tensors = [torch.tensor(series_map[sid], dtype=torch.float32) for sid in series_ids]
                embeds, _ = pipeline.embed(tensors)
                pooled = embeds.mean(dim=1).detach().cpu().numpy().astype(np.float64)
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                return pooled / norms, series_ids, model_name
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    f"Foundation embedding extraction failed for {model_name!r} and "
                    "allow_fallback=False."
                ) from exc

    if not allow_fallback:
        raise RuntimeError(
            f"Foundation model encoder {model_name!r} is not loaded and allow_fallback=False."
        )

    mr_emb, _ = _compute_minirocket_embeddings_and_matrix(series_ids, series_map)
    st_emb, _, _ = _compute_structural_signature_matrix(series_ids, series_map)
    st_std = np.std(st_emb, axis=0)
    st_std[st_std < 1e-9] = 1.0
    st_norm = (st_emb - np.mean(st_emb, axis=0)) / st_std
    concat = np.hstack([mr_emb, st_norm])
    norms = np.linalg.norm(concat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return concat / norms, series_ids, "minirocket_structural_fallback"


# ---------------------------------------------------------------------------
# 2D Spectral / Classical MDS Embedding Coordinates
# ---------------------------------------------------------------------------


def _spectral_2d_coords(sim_mat: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project an :math:`(N \\times N)` PSD similarity matrix into deterministic 2D coordinates."""
    n = sim_mat.shape[0]
    k = max(2, int(n_components))
    if n == 0:
        return np.zeros((0, k), dtype=np.float64)
    if n == 1:
        return np.zeros((1, k), dtype=np.float64)
    if n == 2:
        d01 = float(math.sqrt(max(0.0, 1.0 - sim_mat[0, 1])))
        coords = np.zeros((2, k), dtype=np.float64)
        coords[0, 0] = -0.5 * d01
        coords[1, 0] = 0.5 * d01
        return coords

    # Double-centered kernel PCA / classical MDS on S
    h_mat = np.eye(n, dtype=np.float64) - np.full((n, n), 1.0 / float(n), dtype=np.float64)
    centered = h_mat @ sim_mat @ h_mat
    centered = 0.5 * (centered + centered.T)
    eigvals, eigvecs = np.linalg.eigh(centered)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    coords = np.zeros((n, k), dtype=np.float64)
    n_avail = min(k, eigvecs.shape[1])
    for c in range(n_avail):
        vec = eigvecs[:, c] * math.sqrt(float(eigvals[c]))
        # Deterministic sign canonicalization: largest-magnitude entry is positive
        max_idx = int(np.argmax(np.abs(vec)))
        if vec[max_idx] < 0.0:
            vec = -vec
        coords[:, c] = vec
    return coords


# ---------------------------------------------------------------------------
# Public Function 1: embed_series_df
# ---------------------------------------------------------------------------


def embed_series_df(
    panel_df: pd.DataFrame,
    *,
    cfg: RunConfig | None = None,
    method: str = "hybrid",
    n_components: int = 2,
    text_cols: Sequence[str] | str | None = None,
    embedding_col: str | None = None,
    attribute_cols: Sequence[str] | None = None,
    hierarchy_cols: Sequence[str] | None = None,
    within_hierarchy_level: str | Sequence[str] | None = None,
    semantic_weight: float | Literal["auto"] = "auto",
    hierarchy_weight: float | Literal["auto"] = "auto",
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    allow_fallback: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compute per-series physical & econometric invariants and low-dimensional embedding
    coordinates.

    Returns a DataFrame with one row per ``ts_id`` containing dynamical invariants
    (``lyapunov_exp``, ``hurst_exp``, ``perm_entropy``, ``sample_entropy``, ``ou_damping_rate``,
    ``dominant_period``, ``kuramoto_sync_mean``), structural statistics (``trend_strength``,
    ``seasonal_strength``, ``spectral_entropy``, ``adi_log``, ``cv2_log``, ``skewness``,
    ``kurtosis``, ``arch_volatility``, ``zero_fraction``), 2D coordinates (``embed_x``,
    ``embed_y``), ``embedding_json``, and ``embedding_backend_used``.
    """
    canon_method = _resolve_similarity_method(method)
    res = compute_similarity_matrix(
        panel_df,
        cfg=cfg,
        method=canon_method,
        n_components=n_components,
        text_cols=text_cols,
        embedding_col=embedding_col,
        attribute_cols=attribute_cols,
        hierarchy_cols=hierarchy_cols,
        within_hierarchy_level=within_hierarchy_level,
        semantic_weight=semantic_weight,
        hierarchy_weight=hierarchy_weight,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        allow_fallback=allow_fallback,
        **kwargs,
    )
    return res.embeddings_df.copy()


# ---------------------------------------------------------------------------
# Public Function 2: compute_similarity_matrix
# ---------------------------------------------------------------------------


def compute_similarity_matrix(
    panel_df: pd.DataFrame,
    *,
    cfg: RunConfig | None = None,
    method: str = "hybrid",
    top_k: int = 10,
    max_lag: int = 0,
    spring_stiffness: float = 0.05,
    window_frac: float = 0.15,
    n_components: int = 2,
    text_cols: Sequence[str] | str | None = None,
    embedding_col: str | None = None,
    attribute_cols: Sequence[str] | None = None,
    hierarchy_cols: Sequence[str] | None = None,
    within_hierarchy_level: str | Sequence[str] | None = None,
    semantic_weight: float | Literal["auto"] = "auto",
    hierarchy_weight: float | Literal["auto"] = "auto",
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    allow_fallback: bool = True,
    kernel_weights: dict[str, float] | None = None,
    **kwargs: Any,
) -> SimilarityMatrixResult:
    """Compute the symmetric PSD similarity matrix, ranked digital-twin pairs, and per-series
    physical/econometric embeddings across a time-series panel."""
    canon_method = _resolve_similarity_method(method)
    metadata_source_df = kwargs.pop("_metadata_df", None)
    meta_ts_id_col = str(kwargs.pop("_metadata_ts_id_col", ts_id_col))
    meta_target_col = str(kwargs.pop("_metadata_target_col", target_col))

    series_ids, series_map = _extract_panel_series(
        panel_df, cfg=cfg, ts_id_col=ts_id_col, date_col=date_col, target_col=target_col
    )
    n = len(series_ids)
    embed_cols = [
        "ts_id",
        "lyapunov_exp",
        "hurst_exp",
        "perm_entropy",
        "sample_entropy",
        "ou_damping_rate",
        "dominant_period",
        "kuramoto_sync_mean",
        "trend_strength",
        "seasonal_strength",
        "spectral_entropy",
        "adi_log",
        "cv2_log",
        "skewness",
        "kurtosis",
        "arch_volatility",
        "zero_fraction",
        "embed_x",
        "embed_y",
        "embedding_json",
        "embedding_backend_used",
    ]
    pair_cols = [
        "ts_id",
        "neighbor_ts_id",
        "ts_id_a",
        "ts_id_b",
        "rank",
        "similarity_score",
        "similarity",
        "distance",
        "physics_sim",
        "spectral_sim",
        "shape_sim",
        "structural_sim",
        "semantic_sim",
        "hierarchy_sim",
        "optimal_lag",
        "lag_correlation",
        "method",
    ]
    if n == 0:
        return SimilarityMatrixResult(
            matrix_df=pd.DataFrame(),
            pairs_df=pd.DataFrame(columns=pair_cols),
            embeddings_df=pd.DataFrame(columns=embed_cols),
            method=canon_method,
            diagnostics={"n_series": 0, "method": canon_method},
        )

    eff_meta_panel = metadata_source_df if metadata_source_df is not None else panel_df
    meta_df = _extract_series_metadata(
        eff_meta_panel, series_ids, cfg=cfg, ts_id_col=meta_ts_id_col
    )
    valid_obs_counts = _count_valid_panel_obs(
        eff_meta_panel,
        series_ids,
        cfg=cfg,
        ts_id_col=meta_ts_id_col,
        target_col=meta_target_col,
    )

    sem_mat, has_sem = _compute_semantic_similarity_matrix(
        series_ids,
        meta_df,
        text_cols=text_cols,
        embedding_col=embedding_col,
        attribute_cols=attribute_cols,
    )
    hier_mat, has_hier = _compute_hierarchy_lca_matrix(
        series_ids,
        meta_df,
        hierarchy_cols=hierarchy_cols,
        cfg=cfg,
    )
    if not has_sem and n > 1:
        sem_mat = np.eye(n, dtype=np.float64)
    if not has_hier and n > 1:
        hier_mat = np.eye(n, dtype=np.float64)

    # Always compute the core component lenses so per-lens attributions and invariants are populated
    phys_mat, phys_inv = _compute_physics_attractor_matrix(series_ids, series_map)
    spec_mat, lag_mat, corr_mat, spec_meta = _compute_spectral_phase_matrix(
        series_ids, series_map, max_lag=max_lag
    )
    _mr_emb, mr_mat = _compute_minirocket_embeddings_and_matrix(series_ids, series_map)
    st_emb, st_mat, st_meta = _compute_structural_signature_matrix(
        series_ids, series_map, spectral_meta_map=spec_meta
    )

    dtw_mat: np.ndarray | None = None
    if canon_method in ("elastic_dtw", "hybrid") and n <= 150:
        dtw_mat = _compute_elastic_dtw_matrix(
            series_ids,
            series_map,
            spring_stiffness=spring_stiffness,
            window_frac=window_frac,
        )
    shape_mat = (
        _project_to_psd_similarity(0.60 * mr_mat + 0.40 * dtw_mat)
        if dtw_mat is not None
        else mr_mat
    )

    backend_used = "pure_numpy_scipy"
    found_emb: np.ndarray | None = None
    if canon_method == "foundation_latent":
        found_emb, _, backend_used = extract_foundation_embeddings(
            panel_df,
            cfg=cfg,
            ts_id_col=ts_id_col,
            date_col=date_col,
            target_col=target_col,
            allow_fallback=allow_fallback,
            **kwargs,
        )
        if n == 1:
            found_mat = np.ones((1, 1), dtype=np.float64)
        else:
            f_dist = sp_distance.squareform(sp_distance.pdist(found_emb, metric="cosine"))
            found_mat = _project_to_psd_similarity(1.0 - 0.5 * f_dist)
        final_mat = found_mat
    elif canon_method == "physics_attractor":
        final_mat = phys_mat
    elif canon_method == "spectral_phase":
        final_mat = spec_mat
    elif canon_method == "elastic_dtw":
        final_mat = (
            dtw_mat
            if dtw_mat is not None
            else _compute_elastic_dtw_matrix(
                series_ids,
                series_map,
                spring_stiffness=spring_stiffness,
                window_frac=window_frac,
            )
        )
        shape_mat = final_mat
    elif canon_method == "minirocket":
        final_mat = mr_mat
        shape_mat = mr_mat
    elif canon_method == "structural_signature":
        final_mat = st_mat
    else:
        # Consensus "hybrid" PSD fusion
        w_phys = 0.30
        w_spec = 0.25
        w_shape = 0.25
        w_struct = 0.20
        if kernel_weights:
            w_phys = float(
                kernel_weights.get("physics_attractor", kernel_weights.get("physics", w_phys))
            )
            w_spec = float(
                kernel_weights.get("spectral_phase", kernel_weights.get("spectral", w_spec))
            )
            w_shape = float(kernel_weights.get("shape", kernel_weights.get("minirocket", w_shape)))
            w_struct = float(
                kernel_weights.get(
                    "structural_signature", kernel_weights.get("structural", w_struct)
                )
            )
        w_sum = max(w_phys + w_spec + w_shape + w_struct, 1e-9)
        fused = (
            w_phys * phys_mat + w_spec * spec_mat + w_shape * shape_mat + w_struct * st_mat
        ) / w_sum
        final_mat = _project_to_psd_similarity(fused)

    if has_sem or has_hier:
        if not has_sem:
            w_sem = 0.0
        elif semantic_weight == "auto":
            w_sem = 0.30 if has_hier else 0.40
        else:
            w_sem = max(0.0, float(semantic_weight))

        if not has_hier:
            w_hier = 0.0
        elif hierarchy_weight == "auto":
            w_hier = 0.25 if has_sem else 0.35
        else:
            w_hier = max(0.0, float(hierarchy_weight))

        w_meta = w_sem + w_hier
        if w_meta > 0.0 and n > 1:
            if w_meta >= 0.85:
                w_behav = max(0.15, 1.0 - min(0.85, w_meta))
                scale_m = (1.0 - w_behav) / w_meta
                eff_w_sem = w_sem * scale_m
                eff_w_hier = w_hier * scale_m
            else:
                w_behav = 1.0 - w_meta
                eff_w_sem = w_sem
                eff_w_hier = w_hier

            meta_combined = (w_sem * sem_mat + w_hier * hier_mat) / max(w_meta, 1e-9)
            behav_adjusted = final_mat.copy()
            if has_sem and has_hier:
                # Damp coincidental seasonal correlation when both semantic & hierarchy are disjoint
                low_meta_gate = np.clip(meta_combined / 0.12, 0.35, 1.0)
                np.fill_diagonal(low_meta_gate, 1.0)
                behav_adjusted = behav_adjusted * low_meta_gate

            retail_fused = w_behav * behav_adjusted + eff_w_sem * sem_mat + eff_w_hier * hier_mat
            # For pre-launch / < 2 observation SKUs, drive similarity purely from metadata
            for i_idx, sid in enumerate(series_ids):
                if valid_obs_counts.get(sid, 2) < 2:
                    for j_idx in range(n):
                        if i_idx != j_idx:
                            m_val = float(meta_combined[i_idx, j_idx])
                            retail_fused[i_idx, j_idx] = m_val
                            retail_fused[j_idx, i_idx] = m_val
            final_mat = _project_to_psd_similarity(retail_fused)

    within_cols = _normalize_col_list(within_hierarchy_level, list(meta_df.columns))

    coords = _spectral_2d_coords(final_mat, n_components=max(2, n_components))

    embed_rows: list[dict[str, Any]] = []
    for idx, sid in enumerate(series_ids):
        p_info = phys_inv[sid]
        sp_info = spec_meta[sid]
        st_info = st_meta[sid]
        if found_emb is not None:
            vec_to_store = found_emb[idx, : min(16, found_emb.shape[1])]
        else:
            vec_to_store = st_emb[idx, : min(16, st_emb.shape[1])]
        embed_rows.append(
            {
                "ts_id": sid,
                "lyapunov_exp": round(float(p_info["lyapunov_exp"]), 6),
                "hurst_exp": round(float(st_info["hurst_exp"]), 6),
                "perm_entropy": round(float(p_info["perm_entropy"]), 6),
                "sample_entropy": round(float(p_info["sample_entropy"]), 6),
                "ou_damping_rate": round(float(p_info["ou_damping_rate"]), 6),
                "dominant_period": round(float(sp_info["dominant_period"]), 4),
                "kuramoto_sync_mean": round(float(sp_info["kuramoto_sync_mean"]), 6),
                "trend_strength": round(float(st_info["trend_strength"]), 6),
                "seasonal_strength": round(float(st_info["seasonal_strength"]), 6),
                "spectral_entropy": round(float(st_info["spectral_entropy"]), 6),
                "adi_log": round(float(st_info["adi_log"]), 6),
                "cv2_log": round(float(st_info["cv2_log"]), 6),
                "skewness": round(float(st_info["skewness"]), 6),
                "kurtosis": round(float(st_info["kurtosis"]), 6),
                "arch_volatility": round(float(st_info["arch_volatility"]), 6),
                "zero_fraction": round(float(st_info["zero_fraction"]), 6),
                "embed_x": round(float(coords[idx, 0]), 6),
                "embed_y": round(float(coords[idx, 1]), 6),
                "embedding_json": json.dumps([round(float(v), 5) for v in vec_to_store]),
                "embedding_backend_used": backend_used,
            }
        )
    embeddings_df = pd.DataFrame(embed_rows, columns=embed_cols)

    matrix_df = pd.DataFrame(final_mat, index=series_ids, columns=series_ids)
    matrix_df.index.name = "ts_id"

    eff_k = max(1, int(top_k))
    pair_rows: list[dict[str, Any]] = []
    for i, sid_a in enumerate(series_ids):
        candidates: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            if within_cols:
                sid_b_cand = series_ids[j]
                same_scope = all(
                    str(meta_df.loc[sid_a, c]).strip().lower()
                    == str(meta_df.loc[sid_b_cand, c]).strip().lower()
                    for c in within_cols
                )
                if not same_scope:
                    continue
            candidates.append((float(final_mat[i, j]), j))
        candidates.sort(key=lambda item: (-item[0], series_ids[item[1]]))
        for rank_idx, (score, j) in enumerate(candidates[:eff_k], start=1):
            sid_b = series_ids[j]
            dist_val = float(math.sqrt(max(0.0, 1.0 - score)))
            pair_rows.append(
                {
                    "ts_id": sid_a,
                    "neighbor_ts_id": sid_b,
                    "ts_id_a": sid_a,
                    "ts_id_b": sid_b,
                    "rank": rank_idx,
                    "similarity_score": round(score, 6),
                    "similarity": round(score, 6),
                    "distance": round(dist_val, 6),
                    "physics_sim": round(float(phys_mat[i, j]), 6),
                    "spectral_sim": round(float(spec_mat[i, j]), 6),
                    "shape_sim": round(float(shape_mat[i, j]), 6),
                    "structural_sim": round(float(st_mat[i, j]), 6),
                    "semantic_sim": round(float(sem_mat[i, j]), 6),
                    "hierarchy_sim": round(float(hier_mat[i, j]), 6),
                    "optimal_lag": int(lag_mat[i, j]),
                    "lag_correlation": round(float(corr_mat[i, j]), 6),
                    "method": canon_method,
                }
            )
    pairs_df = pd.DataFrame(pair_rows, columns=pair_cols)

    off_diag = final_mat[np.triu_indices(n, k=1)] if n > 1 else np.array([1.0])
    min_eig = float(np.min(np.linalg.eigvalsh(final_mat))) if n >= 1 else 1.0
    diagnostics = {
        "n_series": n,
        "method": canon_method,
        "mean_pairwise_similarity": float(np.mean(off_diag)) if off_diag.size > 0 else 1.0,
        "min_eigenvalue": min_eig,
        "is_psd": bool(min_eig >= -1e-7),
        "embedding_backend_used": backend_used,
        "has_semantic_similarity": has_sem,
        "has_hierarchy_similarity": has_hier,
    }
    return SimilarityMatrixResult(
        matrix_df=matrix_df,
        pairs_df=pairs_df,
        embeddings_df=embeddings_df,
        method=canon_method,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Public Function 3: cluster_by_similarity_df
# ---------------------------------------------------------------------------


def _describe_cluster_archetype(sub_embed: pd.DataFrame) -> str:
    """Assign a plain-English physical archetype label to a similarity cluster."""
    if sub_embed.empty:
        return "steady_baseline"
    zero_f = float(sub_embed["zero_fraction"].mean())
    seas = float(sub_embed["seasonal_strength"].mean())
    trend = float(sub_embed["trend_strength"].mean())
    hurst = float(sub_embed["hurst_exp"].mean())
    ou_rate = float(sub_embed["ou_damping_rate"].mean())
    lyap = float(sub_embed["lyapunov_exp"].mean())

    if zero_f >= 0.35 or float(sub_embed["adi_log"].mean()) >= 0.45:
        return "lumpy_intermittent"
    if seas >= 0.40:
        return "smooth_seasonal_sync"
    if trend >= 0.45 or hurst >= 0.65:
        return "high_velocity_trending"
    if lyap >= 0.20:
        return "chaotic_volatile"
    if ou_rate >= 0.45 or hurst <= 0.45:
        return "damped_mean_reverting"
    return "steady_baseline"


def _compute_silhouette_scores(dist_mat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-series silhouette scores in :math:`[-1, 1]` from a symmetric distance matrix."""
    n = dist_mat.shape[0]
    sil = np.zeros(n, dtype=np.float64)
    uniq = np.unique(labels)
    if uniq.size <= 1 or n <= 2:
        return sil

    for i in range(n):
        own_c = labels[i]
        own_mask = (labels == own_c) & (np.arange(n) != i)
        a_i = float(np.mean(dist_mat[i, own_mask])) if np.any(own_mask) else 0.0
        b_i = math.inf
        for other_c in uniq:
            if other_c == own_c:
                continue
            other_mask = labels == other_c
            if np.any(other_mask):
                b_i = min(b_i, float(np.mean(dist_mat[i, other_mask])))
        if not math.isfinite(b_i):
            sil[i] = 0.0
        else:
            denom = max(a_i, b_i, 1e-12)
            sil[i] = (b_i - a_i) / denom
    return np.clip(sil, -1.0, 1.0)


def _merge_small_clusters(
    labels: np.ndarray,
    sim_mat: np.ndarray,
    min_cluster_size: int,
) -> np.ndarray:
    """Merge any cluster smaller than ``min_cluster_size`` into its highest-similarity neighbor
    cluster and relabel clusters ``0 .. K-1`` deterministically."""
    n = labels.size
    eff_min = max(1, int(min_cluster_size))
    if n < eff_min:
        return np.zeros(n, dtype=np.int64)

    work = labels.astype(np.int64).copy()
    while True:
        uniq, counts = np.unique(work, return_counts=True)
        if uniq.size <= 1:
            break
        small = uniq[counts < eff_min]
        if small.size == 0:
            break
        # Merge the smallest cluster first
        smallest_c = int(uniq[int(np.argmin(counts))])
        members = np.flatnonzero(work == smallest_c)
        other_clusters = [int(c) for c in uniq if int(c) != smallest_c]
        if not other_clusters:
            break
        best_target = other_clusters[0]
        best_sim = -1.0
        for cand_c in other_clusters:
            cand_members = np.flatnonzero(work == cand_c)
            mean_sim = float(np.mean(sim_mat[np.ix_(members, cand_members)]))
            if mean_sim > best_sim:
                best_sim = mean_sim
                best_target = cand_c
        work[members] = best_target

    # Canonicalize cluster integer IDs by lowest member index so cluster_0 always contains the
    # earliest series ID in sorted order.
    uniq_final = np.unique(work)
    order_pairs = sorted((int(np.min(np.flatnonzero(work == c))), int(c)) for c in uniq_final)
    remap = {old_c: new_idx for new_idx, (_, old_c) in enumerate(order_pairs)}
    return np.array([remap[int(c)] for c in work], dtype=np.int64)


def cluster_by_similarity_df(
    panel_df: pd.DataFrame,
    *,
    cfg: RunConfig | None = None,
    n_clusters: int | Literal["auto"] = "auto",
    method: str = "hybrid",
    min_cluster_size: int = 2,
    max_clusters: int = 8,
    text_cols: Sequence[str] | str | None = None,
    embedding_col: str | None = None,
    attribute_cols: Sequence[str] | None = None,
    hierarchy_cols: Sequence[str] | None = None,
    within_hierarchy_level: str | Sequence[str] | None = None,
    semantic_weight: float | Literal["auto"] = "auto",
    hierarchy_weight: float | Literal["auto"] = "auto",
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    similarity_result: SimilarityMatrixResult | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Partition a panel of time series into coherent similarity clusters via Ward linkage on the
    PSD similarity kernel, automatically selecting silhouette-optimal :math:`K^*` when
    ``n_clusters="auto"`` and merging clusters smaller than ``min_cluster_size``."""
    out_cols = [
        "ts_id",
        "similarity_cluster_id",
        "cluster_int_id",
        "cluster_id",
        "cluster_medoid_ts_id",
        "medoid_ts_id",
        "is_medoid",
        "similarity_to_medoid",
        "cluster_silhouette",
        "silhouette_score",
        "cluster_size",
        "cluster_archetype",
        "cluster_label",
    ]
    sim_res = (
        similarity_result
        if similarity_result is not None
        else compute_similarity_matrix(
            panel_df,
            cfg=cfg,
            method=method,
            text_cols=text_cols,
            embedding_col=embedding_col,
            attribute_cols=attribute_cols,
            hierarchy_cols=hierarchy_cols,
            within_hierarchy_level=within_hierarchy_level,
            semantic_weight=semantic_weight,
            hierarchy_weight=hierarchy_weight,
            ts_id_col=ts_id_col,
            date_col=date_col,
            target_col=target_col,
            **kwargs,
        )
    )
    series_ids = list(sim_res.matrix_df.index)
    n = len(series_ids)
    if n == 0:
        return pd.DataFrame(columns=out_cols)

    eff_min_size = max(1, int(min_cluster_size))
    sim_mat = sim_res.matrix_df.to_numpy(dtype=np.float64)
    dist_mat = np.sqrt(np.maximum(0.0, 1.0 - sim_mat))
    np.fill_diagonal(dist_mat, 0.0)

    max_feasible_k = min(int(max_clusters), n // eff_min_size if eff_min_size > 1 else n)
    if n <= 2 or max_feasible_k < 2 or (isinstance(n_clusters, int) and n_clusters <= 1):
        labels = np.zeros(n, dtype=np.int64)
    else:
        condensed = sp_distance.squareform(dist_mat, checks=False)
        if np.all(condensed < 1e-9):
            labels = np.zeros(n, dtype=np.int64)
        else:
            linkage = sp_cluster.hierarchy.linkage(condensed, method="ward")
            if n_clusters != "auto":
                req_k = min(int(n_clusters), max(2, max_feasible_k), n)
                raw_labels = (
                    sp_cluster.hierarchy.fcluster(linkage, t=req_k, criterion="maxclust") - 1
                )
                labels = _merge_small_clusters(raw_labels, sim_mat, eff_min_size)
            else:
                best_k = 1
                best_score = -1.0
                best_labels = np.zeros(n, dtype=np.int64)
                for cand_k in range(2, max_feasible_k + 1):
                    cand_raw = (
                        sp_cluster.hierarchy.fcluster(linkage, t=cand_k, criterion="maxclust") - 1
                    )
                    cand_merged = _merge_small_clusters(cand_raw, sim_mat, eff_min_size)
                    if np.unique(cand_merged).size < 2:
                        continue
                    sil_vals = _compute_silhouette_scores(dist_mat, cand_merged)
                    mean_sil = float(np.mean(sil_vals))
                    if mean_sil > best_score + 1e-4:
                        best_score = mean_sil
                        best_k = int(np.unique(cand_merged).size)
                        best_labels = cand_merged
                labels = best_labels if best_k >= 2 else np.zeros(n, dtype=np.int64)

    sil_scores = _compute_silhouette_scores(dist_mat, labels)
    embed_df = sim_res.embeddings_df.set_index("ts_id")

    # Identify medoid and archetype per cluster
    medoid_by_cluster: dict[int, int] = {}
    archetype_by_cluster: dict[int, str] = {}
    size_by_cluster: dict[int, int] = {}
    for cid in np.unique(labels):
        c_int = int(cid)
        idx_members = np.flatnonzero(labels == c_int)
        size_by_cluster[c_int] = int(idx_members.size)
        sub_sim = sim_mat[np.ix_(idx_members, idx_members)]
        mean_sim_in_c = np.mean(sub_sim, axis=1)
        medoid_local_idx = int(np.argmax(mean_sim_in_c))
        medoid_by_cluster[c_int] = int(idx_members[medoid_local_idx])
        member_sids = [series_ids[i] for i in idx_members]
        archetype_by_cluster[c_int] = _describe_cluster_archetype(embed_df.loc[member_sids])

    rows: list[dict[str, Any]] = []
    for i, sid in enumerate(series_ids):
        c_int = int(labels[i])
        c_str = f"cluster_{c_int}"
        med_idx = medoid_by_cluster[c_int]
        med_sid = series_ids[med_idx]
        sim_to_med = float(sim_mat[i, med_idx])
        sil_val = float(sil_scores[i])
        arch = archetype_by_cluster[c_int]
        rows.append(
            {
                "ts_id": sid,
                "similarity_cluster_id": c_str,
                "cluster_int_id": c_int,
                "cluster_id": c_str,
                "cluster_medoid_ts_id": med_sid,
                "medoid_ts_id": med_sid,
                "is_medoid": bool(i == med_idx),
                "similarity_to_medoid": round(sim_to_med, 6),
                "cluster_silhouette": round(sil_val, 6),
                "silhouette_score": round(sil_val, 6),
                "cluster_size": size_by_cluster[c_int],
                "cluster_archetype": arch,
                "cluster_label": f"{c_str}:{arch}",
            }
        )
    return pd.DataFrame(rows, columns=out_cols)


# ---------------------------------------------------------------------------
# Public Function 4: find_similar_series_df
# ---------------------------------------------------------------------------


def _kinematic_velocity_acceleration_sim(
    ya: np.ndarray,
    yb: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return ``(scale_ratio, level_offset, velocity_sim, acceleration_sim)`` between two aligned
    slices ``ya`` and ``yb``."""
    mean_a = float(np.mean(np.abs(ya))) if ya.size > 0 else 1.0
    mean_b = float(np.mean(np.abs(yb))) if yb.size > 0 else 1.0
    scale_ratio = mean_a / max(mean_b, 1e-9)
    level_offset = float(np.mean(ya) - np.mean(yb)) if (ya.size > 0 and yb.size > 0) else 0.0

    za, _, _, flat_a = _zscore(ya)
    zb, _, _, flat_b = _zscore(yb)
    if flat_a and flat_b:
        return scale_ratio, level_offset, 1.0, 1.0
    if flat_a or flat_b:
        return scale_ratio, level_offset, 0.0, 0.0

    m = min(za.size, zb.size)
    if m < 2:
        return scale_ratio, level_offset, 0.5, 0.5

    dza = np.diff(za[:m])
    dzb = np.diff(zb[:m])
    na = float(np.linalg.norm(dza))
    nb = float(np.linalg.norm(dzb))
    if na < 1e-12 and nb < 1e-12:
        vel_sim = 1.0
    elif na < 1e-12 or nb < 1e-12:
        vel_sim = 0.0
    else:
        cos_v = float(np.clip(np.dot(dza, dzb) / (na * nb), -1.0, 1.0))
        vel_sim = 0.5 * (cos_v + 1.0)

    if m < 3:
        return scale_ratio, level_offset, vel_sim, vel_sim

    d2za = np.diff(dza)
    d2zb = np.diff(dzb)
    n2a = float(np.linalg.norm(d2za))
    n2b = float(np.linalg.norm(d2zb))
    if n2a < 1e-12 and n2b < 1e-12:
        acc_sim = 1.0
    elif n2a < 1e-12 or n2b < 1e-12:
        acc_sim = 0.0
    else:
        cos_a = float(np.clip(np.dot(d2za, d2zb) / (n2a * n2b), -1.0, 1.0))
        acc_sim = 0.5 * (cos_a + 1.0)

    return scale_ratio, level_offset, vel_sim, acc_sim


def find_similar_series_df(
    panel_df: pd.DataFrame,
    *,
    query_ts_id: str | None = None,
    top_k: int = 5,
    mode: str = "full_history",
    method: str = "hybrid",
    prefix_len: int | None = None,
    window: int | None = None,
    max_lag: int = 0,
    text_cols: Sequence[str] | str | None = None,
    embedding_col: str | None = None,
    attribute_cols: Sequence[str] | None = None,
    hierarchy_cols: Sequence[str] | None = None,
    within_hierarchy_level: str | Sequence[str] | None = None,
    semantic_weight: float | Literal["auto"] = "auto",
    hierarchy_weight: float | Literal["auto"] = "auto",
    cfg: RunConfig | None = None,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    candidate_ts_ids: Sequence[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Find top-:math:`K` digital-twin series across full history, relative-age launch prefixes
    (``mode="launch_prefix"``), or recent rolling windows (``mode="recent_window"``)."""
    canon_mode = _resolve_matching_mode(mode)
    canon_method = _resolve_similarity_method(method)
    out_cols = [
        "query_ts_id",
        "ts_id",
        "neighbor_ts_id",
        "match_ts_id",
        "rank",
        "similarity_score",
        "similarity",
        "distance",
        "physics_sim",
        "spectral_sim",
        "shape_sim",
        "structural_sim",
        "semantic_sim",
        "hierarchy_sim",
        "optimal_lag",
        "lag_correlation",
        "scale_ratio",
        "level_offset",
        "velocity_sim",
        "acceleration_sim",
        "matching_mode",
        "method",
    ]
    series_ids, series_map = _extract_panel_series(
        panel_df, cfg=cfg, ts_id_col=ts_id_col, date_col=date_col, target_col=target_col
    )
    if len(series_ids) <= 1:
        return pd.DataFrame(columns=out_cols)

    if query_ts_id is not None and str(query_ts_id) not in series_map:
        raise ValueError(
            f"query_ts_id {query_ts_id!r} not found in panel series {series_ids[:10]}."
        )

    meta_df = _extract_series_metadata(panel_df, series_ids, cfg=cfg, ts_id_col=ts_id_col)
    valid_obs_counts = _count_valid_panel_obs(
        panel_df, series_ids, cfg=cfg, ts_id_col=ts_id_col, target_col=target_col
    )
    within_cols = _normalize_col_list(within_hierarchy_level, list(meta_df.columns))

    # Slice series according to matching mode before computing similarity
    sliced_rows: list[dict[str, Any]] = []
    sliced_map: dict[str, np.ndarray] = {}
    if canon_mode == "launch_prefix":
        if prefix_len is not None and int(prefix_len) >= 1:
            eff_len = int(prefix_len)
        elif query_ts_id is not None:
            eff_len = max(1, int(series_map[str(query_ts_id)].size))
        else:
            eff_len = max(3, int(min(arr.size for arr in series_map.values())))
        for sid in series_ids:
            arr = series_map[sid][:eff_len]
            sliced_map[sid] = arr
            for t_idx, val in enumerate(arr):
                sliced_rows.append({"ts_id": sid, "ds": t_idx, "y": float(val)})
    elif canon_mode == "recent_window":
        eff_win = (
            max(2, int(window))
            if window is not None
            else (
                max(2, int(prefix_len))
                if prefix_len is not None
                else max(3, min(28, min(arr.size for arr in series_map.values())))
            )
        )
        for sid in series_ids:
            arr = series_map[sid][-eff_win:]
            sliced_map[sid] = arr
            for t_idx, val in enumerate(arr):
                sliced_rows.append({"ts_id": sid, "ds": t_idx, "y": float(val)})
    else:
        for sid in series_ids:
            arr = series_map[sid]
            sliced_map[sid] = arr
            for t_idx, val in enumerate(arr):
                sliced_rows.append({"ts_id": sid, "ds": t_idx, "y": float(val)})

    sliced_df = pd.DataFrame(sliced_rows)
    sim_res = compute_similarity_matrix(
        sliced_df,
        cfg=cfg,
        method=canon_method,
        top_k=len(series_ids),
        max_lag=max_lag,
        text_cols=text_cols,
        embedding_col=embedding_col,
        attribute_cols=attribute_cols,
        hierarchy_cols=hierarchy_cols,
        within_hierarchy_level=within_hierarchy_level,
        semantic_weight=semantic_weight,
        hierarchy_weight=hierarchy_weight,
        ts_id_col="ts_id",
        date_col="ds",
        target_col="y",
        _metadata_df=panel_df,
        _metadata_ts_id_col=ts_id_col,
        _metadata_target_col=target_col,
        **kwargs,
    )

    cand_set = {str(c) for c in candidate_ts_ids} if candidate_ts_ids is not None else None
    queries = [str(query_ts_id)] if query_ts_id is not None else series_ids
    eff_k = max(1, int(top_k))

    pairs_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sim_res.pairs_df.to_dict(orient="records"):
        pairs_lookup[(str(row["ts_id"]), str(row["neighbor_ts_id"]))] = {
            str(k): v for k, v in row.items()
        }

    result_rows: list[dict[str, Any]] = []
    for q_id in queries:
        ya = sliced_map[q_id]
        q_obs = valid_obs_counts.get(q_id, int(ya.size))
        scored_neighbors: list[tuple[float, str, dict[str, Any]]] = []
        for n_id in series_ids:
            if n_id == q_id:
                continue
            if cand_set is not None and n_id not in cand_set:
                continue
            if within_cols:
                same_scope = all(
                    str(meta_df.loc[q_id, c]).strip().lower()
                    == str(meta_df.loc[n_id, c]).strip().lower()
                    for c in within_cols
                )
                if not same_scope:
                    continue
            p_info = pairs_lookup.get((q_id, n_id))
            if p_info is None:
                continue
            yb = sliced_map[n_id]
            scale_r, lvl_off, vel_sim, acc_sim = _kinematic_velocity_acceleration_sim(ya, yb)
            base_sim = float(p_info["similarity_score"])
            if canon_mode == "launch_prefix" and min(ya.size, yb.size) >= 2 and q_obs >= 2:
                # Blend prefix shape similarity with initial kinematic velocity & acceleration
                z_a, _, _, flat_a = _zscore(ya)
                z_b, _, _, flat_b = _zscore(yb[: ya.size])
                if flat_a and flat_b:
                    corr_p = 1.0
                elif flat_a or flat_b or z_a.size != z_b.size:
                    corr_p = base_sim
                else:
                    c_raw = float(np.dot(z_a, z_b) / max(float(z_a.size), 1.0))
                    corr_p = 0.5 * (float(np.clip(c_raw, -1.0, 1.0)) + 1.0)
                combined_sim = float(
                    np.clip(
                        0.45 * base_sim + 0.25 * corr_p + 0.20 * vel_sim + 0.10 * acc_sim,
                        0.0,
                        1.0,
                    )
                )
            else:
                combined_sim = base_sim

            enriched = dict(p_info)
            enriched["similarity_score"] = round(combined_sim, 6)
            enriched["similarity"] = round(combined_sim, 6)
            enriched["distance"] = round(float(math.sqrt(max(0.0, 1.0 - combined_sim))), 6)
            enriched["scale_ratio"] = round(scale_r, 6)
            enriched["level_offset"] = round(lvl_off, 6)
            enriched["velocity_sim"] = round(vel_sim, 6)
            enriched["acceleration_sim"] = round(acc_sim, 6)
            scored_neighbors.append((combined_sim, n_id, enriched))

        scored_neighbors.sort(key=lambda item: (-item[0], item[1]))
        for rank_idx, (_, n_id, info) in enumerate(scored_neighbors[:eff_k], start=1):
            result_rows.append(
                {
                    "query_ts_id": q_id,
                    "ts_id": q_id,
                    "neighbor_ts_id": n_id,
                    "match_ts_id": n_id,
                    "rank": rank_idx,
                    "similarity_score": info["similarity_score"],
                    "similarity": info["similarity"],
                    "distance": info["distance"],
                    "physics_sim": info["physics_sim"],
                    "spectral_sim": info["spectral_sim"],
                    "shape_sim": info["shape_sim"],
                    "structural_sim": info["structural_sim"],
                    "semantic_sim": info["semantic_sim"],
                    "hierarchy_sim": info["hierarchy_sim"],
                    "optimal_lag": info["optimal_lag"],
                    "lag_correlation": info["lag_correlation"],
                    "scale_ratio": info["scale_ratio"],
                    "level_offset": info["level_offset"],
                    "velocity_sim": info["velocity_sim"],
                    "acceleration_sim": info["acceleration_sim"],
                    "matching_mode": canon_mode,
                    "method": canon_method,
                }
            )
    return pd.DataFrame(result_rows, columns=out_cols)


# ---------------------------------------------------------------------------
# Public Function 5: plot_similarity_map
# ---------------------------------------------------------------------------


def plot_similarity_map(
    panel_or_result: pd.DataFrame | SimilarityMatrixResult,
    *,
    method: str = "hybrid",
    n_clusters: int | Literal["auto"] = "auto",
    ax: Any | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a 2D similarity embedding & cluster map (or 2-panel map + sorted similarity heatmap
    when ``ax=None``), returning the matplotlib ``Axes``."""
    import matplotlib.pyplot as plt

    if isinstance(panel_or_result, SimilarityMatrixResult):
        sim_res = panel_or_result
        clusters_df = cluster_by_similarity_df(
            pd.DataFrame(),
            n_clusters=n_clusters,
            method=sim_res.method,
            similarity_result=sim_res,
        )
    else:
        sim_res = compute_similarity_matrix(panel_or_result, method=method, **kwargs)
        clusters_df = cluster_by_similarity_df(
            panel_or_result,
            n_clusters=n_clusters,
            method=method,
            similarity_result=sim_res,
        )

    embed_df = sim_res.embeddings_df
    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_scatter, ax_heat = axes[0], axes[1]
    else:
        ax_scatter = ax
        ax_heat = None

    if not embed_df.empty and not clusters_df.empty:
        merged = embed_df.merge(
            clusters_df[
                [
                    "ts_id",
                    "similarity_cluster_id",
                    "cluster_int_id",
                    "is_medoid",
                    "cluster_archetype",
                ]
            ],
            on="ts_id",
            how="left",
        )
        for cid, grp in merged.groupby("similarity_cluster_id", sort=True):
            arch = str(grp["cluster_archetype"].iloc[0])
            ax_scatter.scatter(
                grp["embed_x"],
                grp["embed_y"],
                label=f"{cid} ({arch})",
                s=60,
                alpha=0.85,
            )
            meds = grp[grp["is_medoid"]]
            if not meds.empty:
                ax_scatter.scatter(
                    meds["embed_x"],
                    meds["embed_y"],
                    marker="*",
                    s=180,
                    edgecolors="black",
                    linewidths=1.0,
                )
        for row in merged.to_dict(orient="records"):
            ax_scatter.annotate(
                str(row["ts_id"]),
                (float(row["embed_x"]), float(row["embed_y"])),
                fontsize=8,
                alpha=0.8,
            )
        ax_scatter.legend(loc="best", fontsize=8)

    ax_scatter.set_xlabel("Similarity Embedding Axis 1 (embed_x)")
    ax_scatter.set_ylabel("Similarity Embedding Axis 2 (embed_y)")
    ax_scatter.set_title(title or f"Time-Series Similarity Map ({sim_res.method})")
    ax_scatter.grid(True, alpha=0.25)

    if ax_heat is not None and not sim_res.matrix_df.empty:
        order_ids = (
            clusters_df.sort_values(["cluster_int_id", "ts_id"])["ts_id"].tolist()
            if not clusters_df.empty
            else list(sim_res.matrix_df.index)
        )
        sorted_mat = sim_res.matrix_df.loc[order_ids, order_ids].to_numpy(dtype=np.float64)
        im = ax_heat.imshow(sorted_mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax_heat.set_xticks(range(len(order_ids)))
        ax_heat.set_yticks(range(len(order_ids)))
        ax_heat.set_xticklabels(order_ids, rotation=45, ha="right", fontsize=7)
        ax_heat.set_yticklabels(order_ids, fontsize=7)
        ax_heat.set_title("Block-Diagonal Similarity Matrix")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    return ax_scatter


# ---------------------------------------------------------------------------
# Public Function 6: recommend_substitutes_df (Retail Assortment Substitution)
# ---------------------------------------------------------------------------


def _extract_inventory_snapshot(
    panel_df: pd.DataFrame,
    inventory_df: pd.DataFrame | None,
    series_ids: Sequence[str],
    series_map: Mapping[str, np.ndarray],
    *,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    on_hand_col: str = "on_hand",
    safety_stock_col: str = "safety_stock",
) -> tuple[dict[str, float], dict[str, str], bool]:
    """Return ``(available_units_map, stock_status_map, has_inventory_data)``."""
    src_df: pd.DataFrame | None = None
    if (
        inventory_df is not None
        and isinstance(inventory_df, pd.DataFrame)
        and not inventory_df.empty
    ):
        src_df = inventory_df
    elif isinstance(panel_df, pd.DataFrame) and not panel_df.empty:
        cand_inv_cols = {
            on_hand_col,
            safety_stock_col,
            "on_hand",
            "on_hand_units",
            "inventory_units",
            "current_inventory",
            "inventory",
            "available_units",
            "stock",
        }
        if any(c in panel_df.columns for c in cand_inv_cols):
            src_df = panel_df

    avail_map: dict[str, float] = {}
    status_map: dict[str, str] = {}
    if src_df is None or ts_id_col not in src_df.columns:
        for sid in series_ids:
            arr = series_map.get(sid, np.ones(1, dtype=np.float64))
            est_avail = round(max(10.0, float(np.mean(np.maximum(arr, 0.0))) * 7.0), 2)
            avail_map[sid] = est_avail
            status_map[sid] = "adequate_stock"
        return avail_map, status_map, False

    eff_oh_col: str | None = None
    for c in (
        on_hand_col,
        "on_hand",
        "on_hand_units",
        "inventory_units",
        "current_inventory",
        "inventory",
        "stock",
    ):
        if c in src_df.columns:
            eff_oh_col = c
            break
    eff_ss_col: str | None = None
    for c in (safety_stock_col, "safety_stock", "safety_stock_units", "min_stock"):
        if c in src_df.columns:
            eff_ss_col = c
            break
    has_direct_avail = "available_units" in src_df.columns

    work = src_df.sort_values(date_col) if date_col in src_df.columns else src_df
    latest_by_id: dict[str, pd.Series] = {}
    for raw_id, grp in work.groupby(ts_id_col, sort=False):
        latest_by_id[str(raw_id)] = grp.iloc[-1]

    for sid in series_ids:
        if sid not in latest_by_id:
            avail_map[sid] = 0.0
            status_map[sid] = "out_of_stock"
            continue
        row = latest_by_id[sid]
        ss_val = (
            float(pd.to_numeric(row[eff_ss_col], errors="coerce"))
            if (eff_ss_col is not None and pd.notna(row[eff_ss_col]))
            else 0.0
        )
        if not math.isfinite(ss_val):
            ss_val = 0.0

        if eff_oh_col is not None and pd.notna(row[eff_oh_col]):
            oh_val = float(pd.to_numeric(row[eff_oh_col], errors="coerce"))
            if not math.isfinite(oh_val):
                oh_val = 0.0
            avail_val = oh_val - ss_val
        elif has_direct_avail and pd.notna(row["available_units"]):
            avail_val = float(pd.to_numeric(row["available_units"], errors="coerce"))
            if not math.isfinite(avail_val):
                avail_val = 0.0
            oh_val = max(0.0, avail_val + ss_val)
        else:
            oh_val = 0.0
            avail_val = 0.0

        if oh_val <= 0.0 or (avail_val <= 0.0 and ss_val <= 0.0):
            st = "out_of_stock"
        elif avail_val <= 0.0:
            st = "low_stock_risk"
        elif avail_val >= max(0.5 * max(ss_val, 0.0), 10.0):
            st = "in_stock_surplus"
        else:
            st = "adequate_stock"
        avail_map[sid] = round(float(avail_val), 4)
        status_map[sid] = st

    return avail_map, status_map, True


def recommend_substitutes_df(
    panel_df: pd.DataFrame,
    inventory_df: pd.DataFrame | None = None,
    *,
    query_ts_id: str | Sequence[str] | None = None,
    top_k: int = 5,
    text_cols: Sequence[str] | str | None = None,
    embedding_col: str | None = None,
    attribute_cols: Sequence[str] | None = None,
    hierarchy_cols: Sequence[str] | None = None,
    within_hierarchy_level: str | Sequence[str] | None = None,
    require_in_stock: bool = True,
    exclude_out_of_stock: bool | None = None,
    on_hand_col: str = "on_hand",
    inventory_col: str | None = None,
    safety_stock_col: str = "safety_stock",
    method: str = "hybrid",
    semantic_weight: float | Literal["auto"] = "auto",
    hierarchy_weight: float | Literal["auto"] = "auto",
    cfg: RunConfig | None = None,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    **kwargs: Any,
) -> pd.DataFrame:
    """Recommend top-:math:`K` in-stock retail assortment substitutes combining semantic/attribute
    similarity, Wu-Palmer hierarchy LCA proximity, behavioral trajectory similarity, and inventory
    availability surplus."""
    eff_require_in_stock = (
        bool(exclude_out_of_stock) if exclude_out_of_stock is not None else bool(require_in_stock)
    )
    eff_on_hand_col = str(inventory_col) if inventory_col is not None else on_hand_col
    out_cols = [
        "query_ts_id",
        "substitute_ts_id",
        "rank",
        "substitution_score",
        "similarity_score",
        "semantic_sim",
        "hierarchy_sim",
        "available_units",
        "stock_status",
        "substitution_rationale",
    ]
    series_ids, series_map = _extract_panel_series(
        panel_df, cfg=cfg, ts_id_col=ts_id_col, date_col=date_col, target_col=target_col
    )
    if len(series_ids) <= 1:
        return pd.DataFrame(columns=out_cols)

    meta_df = _extract_series_metadata(panel_df, series_ids, cfg=cfg, ts_id_col=ts_id_col)
    avail_cols = list(meta_df.columns)

    eff_text_cols = (
        _normalize_col_list(text_cols, avail_cols)
        if text_cols is not None
        else [c for c in _DEFAULT_TEXT_CANDIDATE_COLS if c in avail_cols]
    )
    eff_attr_cols = (
        _normalize_col_list(attribute_cols, avail_cols)
        if attribute_cols is not None
        else [c for c in _DEFAULT_ATTR_CANDIDATE_COLS if c in avail_cols]
    )
    if hierarchy_cols is not None:
        eff_hier_cols = _normalize_col_list(hierarchy_cols, avail_cols)
    elif cfg is not None and cfg.hierarchy.levels:
        eff_hier_cols = _normalize_col_list(
            [c for lvl in cfg.hierarchy.levels for c in lvl], avail_cols
        )
    else:
        eff_hier_cols = [c for c in _DEFAULT_HIER_CANDIDATE_COLS if c in avail_cols]

    within_cols = _normalize_col_list(within_hierarchy_level, avail_cols)

    sim_res = compute_similarity_matrix(
        panel_df,
        cfg=cfg,
        method=method,
        top_k=len(series_ids),
        text_cols=eff_text_cols or None,
        embedding_col=embedding_col,
        attribute_cols=eff_attr_cols or None,
        hierarchy_cols=eff_hier_cols or None,
        within_hierarchy_level=within_hierarchy_level,
        semantic_weight=semantic_weight,
        hierarchy_weight=hierarchy_weight,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        **kwargs,
    )
    has_sem = bool(sim_res.diagnostics.get("has_semantic_similarity", False))
    has_hier = bool(sim_res.diagnostics.get("has_hierarchy_similarity", False))

    avail_map, status_map, has_inv = _extract_inventory_snapshot(
        panel_df,
        inventory_df,
        series_ids,
        series_map,
        ts_id_col=ts_id_col,
        date_col=date_col,
        on_hand_col=eff_on_hand_col,
        safety_stock_col=safety_stock_col,
    )

    if query_ts_id is None:
        queries = list(series_ids)
    elif isinstance(query_ts_id, str):
        if str(query_ts_id) not in series_map:
            raise ValueError(
                f"query_ts_id {query_ts_id!r} not found in panel series {series_ids[:10]}."
            )
        queries = [str(query_ts_id)]
    else:
        queries = [str(q) for q in query_ts_id if str(q) in series_map]

    pairs_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sim_res.pairs_df.to_dict(orient="records"):
        pairs_lookup[(str(row["ts_id"]), str(row["neighbor_ts_id"]))] = {
            str(k): v for k, v in row.items()
        }

    pos_avails = [max(0.0, v) for v in avail_map.values() if v > 0.0]
    max_pos_avail = max(pos_avails) if pos_avails else 1.0
    eff_k = max(1, int(top_k))

    rows: list[dict[str, Any]] = []
    for q_id in queries:
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for cand_id in series_ids:
            if cand_id == q_id:
                continue
            if within_cols:
                same_scope = all(
                    str(meta_df.loc[q_id, c]).strip().lower()
                    == str(meta_df.loc[cand_id, c]).strip().lower()
                    for c in within_cols
                )
                if not same_scope:
                    continue

            c_status = status_map.get(cand_id, "adequate_stock")
            c_avail = float(avail_map.get(cand_id, 0.0))
            if has_inv and eff_require_in_stock and (c_status == "out_of_stock" or c_avail <= 0.0):
                continue

            p_info = pairs_lookup.get((q_id, cand_id))
            if p_info is None:
                continue

            sim_score = float(p_info["similarity_score"])
            sem_sim = float(p_info["semantic_sim"])
            hier_sim = float(p_info["hierarchy_sim"])
            behav_sim = (
                0.30 * float(p_info["physics_sim"])
                + 0.25 * float(p_info["spectral_sim"])
                + 0.25 * float(p_info["shape_sim"])
                + 0.20 * float(p_info["structural_sim"])
            )

            if has_inv:
                if c_avail <= 0.0 or c_status == "out_of_stock":
                    inv_bonus = 0.0
                else:
                    norm_av = float(np.clip(c_avail / max(max_pos_avail, 1.0), 0.0, 1.0))
                    inv_bonus = (
                        0.50 + 0.50 * norm_av
                        if c_status == "in_stock_surplus"
                        else 0.35 + 0.45 * norm_av
                    )
            else:
                inv_bonus = sim_score

            if has_inv and (c_avail <= 0.0 or c_status == "out_of_stock"):
                sub_score = 0.0
            elif has_sem and has_hier:
                sub_score = 0.40 * sem_sim + 0.25 * hier_sim + 0.20 * behav_sim + 0.15 * inv_bonus
                if sem_sim < 0.12 and hier_sim < 0.15:
                    sub_score *= 0.35
            elif has_sem:
                sub_score = 0.55 * sem_sim + 0.30 * behav_sim + 0.15 * inv_bonus
                if sem_sim < 0.10:
                    sub_score *= 0.50
            elif has_hier:
                sub_score = 0.45 * hier_sim + 0.40 * behav_sim + 0.15 * inv_bonus
                if hier_sim < 0.10:
                    sub_score *= 0.50
            else:
                sub_score = 0.85 * sim_score + 0.15 * inv_bonus

            sub_score = float(np.clip(sub_score, 0.0, 1.0))
            rationale_parts: list[str] = []
            if has_sem:
                rationale_parts.append(f"semantic_sim={sem_sim:.2f}")
            if has_hier:
                rationale_parts.append(f"hierarchy_sim={hier_sim:.2f}")
            rationale_parts.append(f"trajectory_sim={behav_sim:.2f}")
            rationale_parts.append(f"stock={c_status} (available_units={c_avail:.1f})")
            rationale = f"Substitute {cand_id} for {q_id}: " + ", ".join(rationale_parts)

            candidates.append(
                (
                    sub_score,
                    cand_id,
                    {
                        "query_ts_id": q_id,
                        "substitute_ts_id": cand_id,
                        "substitution_score": round(sub_score, 6),
                        "similarity_score": round(sim_score, 6),
                        "semantic_sim": round(sem_sim, 6),
                        "hierarchy_sim": round(hier_sim, 6),
                        "available_units": round(c_avail, 4),
                        "stock_status": c_status,
                        "substitution_rationale": rationale,
                    },
                )
            )

        candidates.sort(key=lambda item: (-item[0], item[1]))
        for rank_idx, (_, _, info) in enumerate(candidates[:eff_k], start=1):
            row_out = dict(info)
            row_out["rank"] = rank_idx
            rows.append(row_out)

    return pd.DataFrame(rows, columns=out_cols)


# ---------------------------------------------------------------------------
# Public Function 7: detect_emerging_trends_df (Emerging Trend Breakout Sensing)
# ---------------------------------------------------------------------------


def detect_emerging_trends_df(
    panel_df: pd.DataFrame,
    *,
    window: int = 14,
    recent_window: int | None = None,
    baseline_window: int | None = None,
    top_k_related: int = 3,
    text_cols: Sequence[str] | str | None = None,
    embedding_col: str | None = None,
    attribute_cols: Sequence[str] | None = None,
    hierarchy_cols: Sequence[str] | None = None,
    max_lag: int = 7,
    method: str = "hybrid",
    cfg: RunConfig | None = None,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    **kwargs: Any,
) -> pd.DataFrame:
    """Detect emerging trend breakouts, kinematic velocity/acceleration (:math:`\\dot{y},
    \\ddot{y}`), directed Kuramoto/cross-correlation lead-lag bellwether leaders
    (:math:`\\tau_{i \\to j}^*`), and lagging co-promotion candidates."""
    eff_window = int(recent_window) if recent_window is not None else int(window)
    out_cols = [
        "ts_id",
        "trend_score",
        "trend_regime",
        "recent_velocity",
        "recent_acceleration",
        "breakout_zscore",
        "hurst_exp",
        "similarity_cluster_id",
        "leading_bellwether_ts_id",
        "lag_behind_leader",
        "co_promote_candidates",
        "recommended_merchandising_action",
    ]
    series_ids, series_map = _extract_panel_series(
        panel_df, cfg=cfg, ts_id_col=ts_id_col, date_col=date_col, target_col=target_col
    )
    if not series_ids:
        return pd.DataFrame(columns=out_cols)

    meta_df = _extract_series_metadata(panel_df, series_ids, cfg=cfg, ts_id_col=ts_id_col)
    avail_cols = list(meta_df.columns)
    eff_text_cols = (
        _normalize_col_list(text_cols, avail_cols)
        if text_cols is not None
        else [c for c in _DEFAULT_TEXT_CANDIDATE_COLS if c in avail_cols]
    )
    eff_attr_cols = (
        _normalize_col_list(attribute_cols, avail_cols)
        if attribute_cols is not None
        else [c for c in _DEFAULT_ATTR_CANDIDATE_COLS if c in avail_cols]
    )
    if hierarchy_cols is not None:
        eff_hier_cols = _normalize_col_list(hierarchy_cols, avail_cols)
    elif cfg is not None and cfg.hierarchy.levels:
        eff_hier_cols = _normalize_col_list(
            [c for lvl in cfg.hierarchy.levels for c in lvl], avail_cols
        )
    else:
        eff_hier_cols = [c for c in _DEFAULT_HIER_CANDIDATE_COLS if c in avail_cols]

    eff_max_lag = max(1, int(max_lag))
    sim_res = compute_similarity_matrix(
        panel_df,
        cfg=cfg,
        method=method,
        top_k=len(series_ids),
        max_lag=eff_max_lag,
        text_cols=eff_text_cols or None,
        embedding_col=embedding_col,
        attribute_cols=eff_attr_cols or None,
        hierarchy_cols=eff_hier_cols or None,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        **kwargs,
    )
    clusters_df = cluster_by_similarity_df(
        panel_df,
        cfg=cfg,
        method=method,
        similarity_result=sim_res,
    )
    cluster_by_id = dict(
        zip(
            clusters_df["ts_id"].astype(str),
            clusters_df["similarity_cluster_id"].astype(str),
            strict=False,
        )
    )
    hurst_by_id = dict(
        zip(
            sim_res.embeddings_df["ts_id"].astype(str),
            sim_res.embeddings_df["hurst_exp"].astype(float),
            strict=False,
        )
    )

    kinematics_by_id: dict[str, dict[str, Any]] = {}
    for sid in series_ids:
        y = series_map[sid]
        t_len = int(y.size)
        w = min(max(3, eff_window), t_len)
        y_rec = y[-w:]
        y_prior = y[:-w] if t_len > w + 2 else y[: max(2, t_len // 2)]
        if baseline_window is not None and y_prior.size > max(2, int(baseline_window)):
            y_prior = y_prior[-int(baseline_window) :]

        mu_prior = float(np.mean(y_prior)) if y_prior.size > 0 else float(np.mean(y))
        sd_prior = float(np.std(y_prior)) if y_prior.size > 1 else 0.0
        scale_base = max(abs(mu_prior), float(np.mean(np.abs(y))), 1e-6)
        denom_sd = max(sd_prior, 0.05 * abs(mu_prior), 1e-6)

        y_rec_norm = y_rec / scale_base
        t_grid = np.arange(w, dtype=np.float64)
        t_centered = t_grid - float(np.mean(t_grid))
        t_ss = float(np.dot(t_centered, t_centered))
        vel = (
            float(np.dot(t_centered, y_rec_norm - np.mean(y_rec_norm)) / t_ss)
            if t_ss > 1e-12
            else 0.0
        )

        if w >= 4:
            half = w // 2
            y1 = y_rec_norm[: max(2, half)]
            y2 = y_rec_norm[half:]
            t1 = np.arange(y1.size, dtype=np.float64)
            t1 -= float(np.mean(t1))
            v1 = float(np.dot(t1, y1 - np.mean(y1)) / max(float(np.dot(t1, t1)), 1e-12))
            t2 = np.arange(y2.size, dtype=np.float64)
            t2 -= float(np.mean(t2))
            v2 = float(np.dot(t2, y2 - np.mean(y2)) / max(float(np.dot(t2, t2)), 1e-12))
            d2_mean = float(np.mean(np.diff(y_rec_norm, n=2)))
            acc = 0.60 * ((v2 - v1) / max(1.0, float(half))) + 0.40 * d2_mean
        elif w == 3:
            acc = float(y_rec_norm[2] - 2.0 * y_rec_norm[1] + y_rec_norm[0])
        else:
            acc = 0.0

        tail_len = max(3, w // 2)
        rec_level = 0.65 * float(np.mean(y_rec[-tail_len:])) + 0.35 * float(np.mean(y_rec))
        z_break = (rec_level - mu_prior) / denom_sd
        hurst_val = float(hurst_by_id.get(sid, 0.5))

        z_pos = max(0.0, z_break) / 3.0
        v_pos = max(0.0, vel * 15.0)
        a_pos = max(0.0, acc * 25.0)
        h_pos = max(0.0, hurst_val - 0.5)
        raw_strength = 0.45 * z_pos + 0.30 * v_pos + 0.15 * a_pos + 0.10 * h_pos
        if z_break < 0.0 and vel < 0.0:
            t_score = float(np.clip(0.15 * math.exp(0.5 * z_break), 0.0, 0.25))
        else:
            t_score = float(np.clip(1.0 - math.exp(-1.5 * raw_strength), 0.0, 1.0))

        if (z_break >= 1.5 and vel > 0.005 and acc >= -0.005) or (z_break >= 2.2 and vel > 0.01):
            regime = "accelerating_breakout"
        elif z_break >= 0.8 and vel > 0.002:
            regime = "steady_growth" if acc >= -0.01 else "maturing_plateau"
        elif z_break >= 1.0 and vel <= 0.002:
            regime = "maturing_plateau"
        elif vel < -0.008 and z_break < -0.8:
            regime = "declining"
        else:
            regime = "stable"

        kinematics_by_id[sid] = {
            "trend_score": t_score,
            "trend_regime": regime,
            "recent_velocity": vel,
            "recent_acceleration": acc,
            "breakout_zscore": z_break,
            "hurst_exp": hurst_val,
        }

    # Build pair lookup and evaluate directed lead-lag (including prior-window lag when a leader
    # has just surged in the recent window ahead of its follower).
    pairs_by_src: dict[str, list[dict[str, Any]]] = {sid: [] for sid in series_ids}
    pair_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sim_res.pairs_df.to_dict(orient="records"):
        sa = str(row["ts_id"])
        sb = str(row["neighbor_ts_id"])
        r_copy = {str(k): v for k, v in row.items()}
        pairs_by_src.setdefault(sa, []).append(r_copy)
        pair_map[(sa, sb)] = r_copy

    leader_for_id: dict[str, tuple[str, int]] = {}
    followers_by_leader: dict[str, list[str]] = {sid: [] for sid in series_ids}

    for sid_i in series_ids:
        yi = series_map[sid_i]
        best_leader = ""
        best_lag = 0
        best_leader_score = -1.0
        for sid_j in series_ids:
            if sid_j == sid_i:
                continue
            kj = kinematics_by_id[sid_j]
            if kj["trend_regime"] not in ("accelerating_breakout", "steady_growth"):
                continue
            p_ji = pair_map.get((sid_j, sid_i))
            sim_ji = float(p_ji["similarity_score"]) if p_ji is not None else 0.0
            opt_lag = int(p_ji["optimal_lag"]) if p_ji is not None else 0
            lag_corr = float(p_ji["lag_correlation"]) if p_ji is not None else 0.0

            # Also check pre-breakout / detrended lag if leader surged ahead of follower at the tail
            if opt_lag <= 0 or lag_corr < 0.45:
                yj = series_map[sid_j]
                zj, _, _, fj = _zscore(yj)
                zi, _, _, fi = _zscore(yi)
                alt_lag, alt_corr = _compute_optimal_lead_lag(zj, zi, fj, fi, max_lag=eff_max_lag)
                if alt_lag > 0 and alt_corr > lag_corr:
                    opt_lag, lag_corr = alt_lag, alt_corr
                if (opt_lag <= 0 or lag_corr < 0.45) and min(yj.size, yi.size) > eff_window + 6:
                    cut = max(6, int(eff_window) // 2)
                    zj_p, _, _, fjp = _zscore(yj[:-cut])
                    zi_p, _, _, fip = _zscore(yi[:-cut])
                    p_lag, p_corr = _compute_optimal_lead_lag(
                        zj_p, zi_p, fjp, fip, max_lag=eff_max_lag
                    )
                    if p_lag > 0 and p_corr >= 0.50:
                        opt_lag, lag_corr = p_lag, p_corr

            if (
                opt_lag > 0
                and lag_corr >= 0.45
                and sim_ji >= 0.25
                and (
                    kj["trend_score"] > kinematics_by_id[sid_i]["trend_score"] - 0.05
                    or kj["breakout_zscore"] >= kinematics_by_id[sid_i]["breakout_zscore"]
                )
            ):
                cand_rank_score = float(kj["trend_score"]) * 0.6 + lag_corr * 0.4
                if cand_rank_score > best_leader_score:
                    best_leader_score = cand_rank_score
                    best_leader = sid_j
                    best_lag = int(opt_lag)

        if best_leader and best_lag > 0:
            leader_for_id[sid_i] = (best_leader, best_lag)
            followers_by_leader.setdefault(best_leader, []).append(sid_i)

    eff_k_rel = max(1, int(top_k_related))
    rows_out: list[dict[str, Any]] = []
    for sid in series_ids:
        k_info = kinematics_by_id[sid]
        regime = str(k_info["trend_regime"])
        leader_id, lag_steps = leader_for_id.get(sid, ("", 0))
        own_followers = followers_by_leader.get(sid, [])

        if not leader_id and regime in ("accelerating_breakout", "steady_growth"):
            leader_id = sid
            lag_steps = 0

        # Build co_promote_candidates prioritizing lagging followers and top similarity twins
        rel_ids: list[str] = []
        for f_sid in own_followers:
            if f_sid != sid and f_sid not in rel_ids:
                rel_ids.append(f_sid)
        for p_row in pairs_by_src.get(sid, []):
            nbr = str(p_row["neighbor_ts_id"])
            if nbr != sid and nbr not in rel_ids:
                rel_ids.append(nbr)
            if len(rel_ids) >= eff_k_rel:
                break
        co_promote_str = ",".join(rel_ids[:eff_k_rel])

        if lag_steps > 0 and leader_id and leader_id != sid:
            action = "co_promote_lagging_twin"
        elif regime == "accelerating_breakout":
            if own_followers or float(k_info["breakout_zscore"]) < 6.0:
                action = "promote_trend_leader"
            else:
                action = "replenish_breakout_risk"
        elif regime == "steady_growth":
            action = "promote_trend_leader"
        elif regime == "declining":
            action = "clearance_candidate"
        else:
            action = "hold_baseline"

        rows_out.append(
            {
                "ts_id": sid,
                "trend_score": round(float(k_info["trend_score"]), 6),
                "trend_regime": regime,
                "recent_velocity": round(float(k_info["recent_velocity"]), 6),
                "recent_acceleration": round(float(k_info["recent_acceleration"]), 6),
                "breakout_zscore": round(float(k_info["breakout_zscore"]), 6),
                "hurst_exp": round(float(k_info["hurst_exp"]), 6),
                "similarity_cluster_id": cluster_by_id.get(sid, "cluster_0"),
                "leading_bellwether_ts_id": leader_id,
                "lag_behind_leader": int(lag_steps),
                "co_promote_candidates": co_promote_str,
                "recommended_merchandising_action": action,
            }
        )

    out_df = pd.DataFrame(rows_out, columns=out_cols)
    return out_df.sort_values(
        ["trend_score", "breakout_zscore", "ts_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


emerging_trends_df = detect_emerging_trends_df
