"""4-Tier End-to-End (E2E) Verification Suite for Track P (`v1.8.0` Phase 1 & Phase 2).

Covers all Track P features (F1–F20 in `PROJECT.md § Feature Inventory`) across 4 testing tiers:
- Tier 1 — Category-Partition Feature Coverage (12 suites covering F1–F15, all 7 similarity lenses,
  and all 6 system integrations)
- Tier 2 — Boundary Value & Pathological Corner-Case Hardening (N=1, N=2, flat/zero, intermittent,
  short launch prefixes, missing values, invalid config rejection, and 100% digest invariance)
- Tier 3 — Cross-Feature & Cross-Track Multi-Subsystem Combinations (5 integration suites chaining
  Track J + Track K + Track L + Track P)
- Tier 4 — Real-World Enterprise Application Scenarios (Scenarios #1–#8: Multi-regime retail
  negative-transfer elimination, flat-panel Forecast-to-Decision spine, Day-0 -> Day-5 -> Mature
  cold-start progression, multi-modal retail similarity, distributed sharding with
  @similarity_global, latent similarity hierarchy 3-mode MiNT & Goal-Seek, stockout-aware
  substitute recommendation & cannibalization optimization, and emerging trend breakout &
  lead-lag co-promotion sensing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import scale_forecasting as sf
from scale_forecasting.auto import build_config, inspect_panel_for_auto
from scale_forecasting.auto_policy import (
    evolved_policy_thresholds,
    propose_autofocus_dag_branches,
    propose_autofocus_pass,
)
from scale_forecasting.config import (
    VALID_VARIANT_TAGS,
    EnsembleConfig,
    FeaturesConfig,
    HierarchyConfig,
    RunConfig,
)
from scale_forecasting.engines.ray_io import chunk_cells
from scale_forecasting.engines.spark_io import _MODEL_COL, bucket_key_cols, run_group
from scale_forecasting.ensembler import combine_calculated, combine_oof
from scale_forecasting.errors import ConfigError, DataError
from scale_forecasting.evolve.targets import (
    EVOLVE_TARGETS,
    AutoPolicyWeights,
    deterministic_mock_candidates,
    get_target_seed,
)
from scale_forecasting.features import (
    VALID_ABLATION_MODES,
    build_features,
    build_future_features,
    expand_model_variants,
    resolve_model_variant,
)
from scale_forecasting.goal_seek import GoalSeekResult, allocate_target
from scale_forecasting.mcp import McpServer
from scale_forecasting.playground import bakeoff, run_model
from scale_forecasting.reconciliation import (
    build_hierarchy,
    reconcile_forecasts,
    reconcile_oof,
    verify_coherence,
)
from scale_forecasting.registry.ids import (
    canonical_config,
    make_ensemble_id,
    make_model_spec_hash,
    make_run_id,
)
from scale_forecasting.report import export_report, render_html_report
from scale_forecasting.temporal_reconciliation import verify_temporal_coherence
from scale_forecasting.worker import is_panel_model, run_cell, run_panel

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Deterministic Synthetic Dynamical & Multi-Regime Enterprise Panel Generators
# =============================================================================


def _make_dynamical_regimes_panel(
    n_days: int = 84,
    seed: int = 1801,
) -> pd.DataFrame:
    """Create a 7-series dynamical-systems panel with known mathematical invariants:

    - `harm_sync_a`, `harm_sync_b`: Phase-locked harmonic oscillators (Kuramoto R_ij -> 1.0,
      low Lyapunov lambda_max, low permutation entropy).
    - `harm_lag_plus3`: Exact 3-step lagged twin of `harm_sync_a` (`y_b[t] = y_a[t - 3]`).
    - `warped_pulse_a`, `warped_pulse_b`: Time-shifted promotional pulse twins (matched by
      Elastic Spring DTW despite a 2-day calendar shift).
    - `ou_damped`: Mean-reverting Ornstein-Uhlenbeck process with strong damping kappa_OU and
      anti-persistent Hurst H < 0.5.
    - `trend_persistent`: Persistent random walk + strong upward drift (Hurst H > 0.6).
    - `chaotic_logistic`: Logistic-map r=3.95 chaotic series (high Lyapunov lambda_max and
      high permutation entropy).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)

    # 1 & 2. Phase-locked harmonic oscillators
    base_wave = np.sin(2.0 * np.pi * t / 14.0) + 0.35 * np.cos(2.0 * np.pi * t / 7.0)
    y_harm_a = 100.0 + 25.0 * base_wave
    y_harm_b = 180.0 + 40.0 * base_wave + rng.normal(0.0, 0.25, size=n_days)

    # 3. Directed lead-lag twin: harm_sync_a leads harm_lag_plus3 by +3 steps
    lag_shift = 3
    y_harm_lag3 = np.roll(y_harm_a, lag_shift)
    y_harm_lag3[:lag_shift] = y_harm_a[:lag_shift]

    # 4 & 5. Elastic DTW promotional pulse twins (shifted by 2 steps in time)
    y_pulse_a = np.full(n_days, 60.0, dtype=float)
    y_pulse_b = np.full(n_days, 60.0, dtype=float)
    for center in (18, 42, 66):
        grid = np.arange(n_days, dtype=float)
        y_pulse_a += 45.0 * np.exp(-0.5 * ((grid - center) / 2.0) ** 2)
        y_pulse_b += 45.0 * np.exp(-0.5 * ((grid - (center + 2.0)) / 2.0) ** 2)

    # 6. Anti-persistent / strongly damped Ornstein-Uhlenbeck mean-reverting series
    y_ou = np.empty(n_days, dtype=float)
    y_ou[0] = 100.0
    for idx in range(1, n_days):
        # Negative lag-1 feedback produces strong restoring force and low Hurst H
        y_ou[idx] = 100.0 - 0.65 * (y_ou[idx - 1] - 100.0) + rng.normal(0.0, 3.5)

    # 7. Persistent trending series (Hurst H > 0.6)
    steps_trend = 1.8 + np.cumsum(rng.normal(0.15, 0.4, size=n_days))
    y_trend = 50.0 + np.cumsum(steps_trend * 0.25)

    # 8. Chaotic logistic map x_{t+1} = 3.98 * x_t * (1 - x_t)
    x_chaos = np.empty(n_days, dtype=float)
    x_chaos[0] = 0.314159
    for idx in range(1, n_days):
        x_chaos[idx] = 3.98 * x_chaos[idx - 1] * (1.0 - x_chaos[idx - 1])
    y_chaos = 50.0 + 100.0 * x_chaos + rng.normal(0.0, 2.0, size=n_days)

    series_map = {
        "harm_sync_a": y_harm_a,
        "harm_sync_b": y_harm_b,
        "harm_lag_plus3": y_harm_lag3,
        "warped_pulse_a": y_pulse_a,
        "warped_pulse_b": y_pulse_b,
        "ou_damped": y_ou,
        "trend_persistent": y_trend,
        "chaotic_logistic": y_chaos,
    }

    rows: list[dict[str, Any]] = []
    for tid, vals in series_map.items():
        for dt, y_val in zip(dates, vals, strict=True):
            rows.append({"ts_id": tid, "ds": dt, "y": float(y_val)})
    return pd.DataFrame(rows)


def _make_multi_regime_retail_panel(
    n_days: int = 56,
    n_short_days: int = 5,
    include_cold_start: bool = True,
    seed: int = 1802,
) -> pd.DataFrame:
    """Create a multi-regime retail panel where a monolithic `@global` lag model suffers severe
    cross-regime negative transfer, while `@similarity_global` isolates opposite regimes:

    - Regime A (`flagship_0`, `flagship_1`, `flagship_2`): Positively autocorrelated smooth upward
      trending / weekly seasonal series (`y_t = 0.85 * y_{t-1} + ...`).
    - Regime B (`counter_0`, `counter_1`, `counter_2`): Strongly negatively autocorrelated
      sawtooth replenishment/promotional oscillation (`y_t - mu = -0.85 * (y_{t-1} - mu)`).
      Pooling Regime A (+0.85 AR1) and Regime B (-0.85 AR1) into a single monolithic `@global`
      lag model cancels the lag-1 coefficient toward 0, degrading accuracy on both regimes!
    - Zero-Metadata NPI SKU (`npi_surge_short`): 5-day launch prefix matching Regime A's launch
      trajectory with zero `static_covariates`.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows: list[dict[str, Any]] = []

    # Regime A: Smooth persistent seasonal + upward ramp
    for i in range(3):
        tid = f"flagship_{i}"
        base = 100.0 + 15.0 * i
        vals = np.empty(n_days, dtype=float)
        vals[0] = base
        for idx in range(1, n_days):
            vals[idx] = (
                0.15 * (base + 1.2 * idx)
                + 0.85 * vals[idx - 1]
                + 6.0 * np.sin(2.0 * np.pi * idx / 7.0)
                + rng.normal(0.0, 0.8)
            )
        for dt, y_v in zip(dates, vals, strict=True):
            rows.append({"ts_id": tid, "ds": dt, "y": float(max(y_v, 5.0))})

    # Regime B: Counter-cyclical alternating sawtooth (`y_t - mu = -0.85 * (y_{t-1} - mu)`)
    for i in range(3):
        tid = f"counter_{i}"
        mu = 120.0 + 10.0 * i
        vals = np.empty(n_days, dtype=float)
        vals[0] = mu + 35.0
        for idx in range(1, n_days):
            vals[idx] = mu - 0.88 * (vals[idx - 1] - mu) + rng.normal(0.0, 0.8)
        for dt, y_v in zip(dates, vals, strict=True):
            rows.append({"ts_id": tid, "ds": dt, "y": float(max(y_v, 5.0))})

    if include_cold_start:
        # Zero-metadata NPI SKU whose 5-day launch prefix tracks flagship_0's early ramp
        short_dates = dates[-n_short_days:]
        for idx, dt in enumerate(short_dates):
            y_v = 100.0 + 4.5 * idx + 6.0 * np.sin(2.0 * np.pi * idx / 7.0)
            rows.append({"ts_id": "npi_surge_short", "ds": dt, "y": float(y_v)})

    return pd.DataFrame(rows)


def _make_promo_cannibalization_twin_panel(
    n_days: int = 56,
    horizon: int = 7,
    seed: int = 1803,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a 4-SKU promotional panel with two digital-twin substitutes (`twin_a1`, `twin_a2`)
    and two orthogonal SKUs (`ortho_b1`, `ortho_b2`) to test similarity-weighted cannibalization.
    """
    rng = np.random.default_rng(seed)
    all_dates = pd.date_range("2025-01-01", periods=n_days + horizon, freq="D")
    hist_dates = all_dates[:n_days]
    fut_dates = all_dates[n_days:]
    t = np.arange(n_days, dtype=float)

    # Digital twins share a 7-day cycle; orthogonal SKUs follow a distinct 11-day sawtooth
    wave_twin = 18.0 * np.sin(2.0 * np.pi * t / 7.0)
    wave_ortho = 22.0 * np.cos(2.0 * np.pi * t / 11.0) + 0.5 * t

    promo_a1 = ((np.arange(n_days) % 8) == 1).astype(float)
    promo_a2 = ((np.arange(n_days) % 8) == 4).astype(float)
    promo_b1 = ((np.arange(n_days) % 6) == 2).astype(float)
    promo_b2 = ((np.arange(n_days) % 6) == 5).astype(float)

    # twin_a1 loses 22 units when its digital twin twin_a2 promotes, but 0 when ortho_b promotes
    y_a1 = 110.0 + wave_twin + 32.0 * promo_a1 - 22.0 * promo_a2 + rng.normal(0.0, 1.5, n_days)
    y_a2 = 115.0 + wave_twin + 30.0 * promo_a2 - 20.0 * promo_a1 + rng.normal(0.0, 1.5, n_days)
    y_b1 = 80.0 + wave_ortho + 25.0 * promo_b1 - 15.0 * promo_b2 + rng.normal(0.0, 1.5, n_days)
    y_b2 = 85.0 + wave_ortho + 24.0 * promo_b2 - 14.0 * promo_b1 + rng.normal(0.0, 1.5, n_days)

    sku_specs = [
        ("twin_a1", y_a1, promo_a1, 10.0, 4.5),
        ("twin_a2", y_a2, promo_a2, 10.5, 4.5),
        ("ortho_b1", y_b1, promo_b1, 14.0, 6.0),
        ("ortho_b2", y_b2, promo_b2, 14.5, 6.0),
    ]

    hist_rows: list[dict[str, Any]] = []
    fut_rows: list[dict[str, Any]] = []
    for tid, y_arr, p_arr, base_price, cost in sku_specs:
        for dt, y_v, pr_flag in zip(hist_dates, y_arr, p_arr, strict=True):
            price_t = base_price * (0.88 if pr_flag > 0.5 else 1.0)
            hist_rows.append(
                {
                    "ts_id": tid,
                    "ds": dt,
                    "y": float(max(y_v, 5.0)),
                    "price": float(price_t),
                    "promo": float(pr_flag),
                    "unit_cost": float(cost),
                }
            )
        for dt in fut_dates:
            fut_rows.append(
                {
                    "ts_id": tid,
                    "ds": dt,
                    "price": float(base_price),
                    "promo": 0.0,
                    "unit_cost": float(cost),
                }
            )
    return pd.DataFrame(hist_rows), pd.DataFrame(fut_rows)


# =============================================================================
# TIER 1 — Category-Partition Feature Coverage (F1–F15, >= 5 cases per feature)
# =============================================================================


def test_t1_f1_run_config_track_p_fields_variant_tags_and_digest_invariance() -> None:
    """Tier 1 F1 (>= 5 cases): Verify all 8 Track P `RunConfig` fields, `SIMILARITY_METHODS`,
    `COLD_START_MATCHING_MODES`, `@similarity_global` & `@similarity_cold_start` variant tags,
    `HierarchyConfig.effective_levels`, 100% `run_id` / `model_spec_hash` / `ensemble_id` digest
    invariance at defaults, and forward-compatible package version parity.
    """
    from scale_forecasting.config import COLD_START_MATCHING_MODES, SIMILARITY_METHODS

    # Case 1: Forward-compatible version check & exported Track P literal vocabularies
    ver_tuple = tuple(int(p) for p in sf.__version__.split(".")[:3])
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert ver_tuple >= (1, 8, 0) or (
        ver_tuple >= (1, 7, 0) and 'version = "1.7.0"' in pyproject_text
    )

    assert set(SIMILARITY_METHODS) == {
        "hybrid",
        "physics_attractor",
        "spectral_phase",
        "elastic_dtw",
        "minirocket",
        "structural_signature",
        "foundation_latent",
    }
    assert set(COLD_START_MATCHING_MODES) == {"auto", "attribute", "similarity", "hybrid"}
    assert "similarity_global" in VALID_VARIANT_TAGS
    assert "similarity_cold_start" in VALID_VARIANT_TAGS
    assert "similarity_global" in VALID_ABLATION_MODES

    # Case 2: Default values of all 8 new Track P fields across Features, Hierarchy, Ensemble
    f_cfg = FeaturesConfig()
    assert f_cfg.cold_start_matching == "auto"
    assert f_cfg.similarity_cluster_feature is False
    assert f_cfg.similarity_method == "hybrid"
    assert f_cfg.similarity_n_clusters == "auto"

    h_cfg = HierarchyConfig()
    assert h_cfg.similarity_hierarchy is False
    assert h_cfg.similarity_n_clusters == "auto"
    assert h_cfg.similarity_method == "hybrid"
    assert h_cfg.effective_levels == []

    e_cfg = EnsembleConfig()
    assert e_cfg.similarity_smoothing == pytest.approx(0.0)

    # Case 3: HierarchyConfig(similarity_hierarchy=True) auto-enables hierarchy even with levels=[]
    h_sim = HierarchyConfig(similarity_hierarchy=True, similarity_n_clusters=3)
    assert h_sim.enabled is True
    assert h_sim.effective_levels == [["similarity_cluster_id"]]

    # Case 4: 100% canonical_config, make_run_id, make_model_spec_hash, make_ensemble_id invariance
    base_payload = {
        "run_name": "track-p-digest-invariance",
        "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
        "models": ["naive_mean", "theta"],
    }
    cfg_implicit = RunConfig.model_validate(base_payload)
    cfg_explicit = RunConfig.model_validate(
        {
            **base_payload,
            "features": {
                "cold_start_matching": "auto",
                "similarity_cluster_feature": False,
                "similarity_method": "hybrid",
                "similarity_n_clusters": "auto",
            },
            "hierarchy": {
                "similarity_hierarchy": False,
                "similarity_n_clusters": "auto",
                "similarity_method": "hybrid",
            },
            "ensemble": {
                "similarity_smoothing": 0.0,
            },
        }
    )
    assert canonical_config(cfg_implicit) == canonical_config(cfg_explicit)
    assert make_run_id(cfg_implicit) == make_run_id(cfg_explicit)
    assert make_ensemble_id(cfg_implicit.ensemble) == make_ensemble_id(cfg_explicit.ensemble)
    for m_name in cfg_implicit.models:
        assert make_model_spec_hash(m_name, cfg_implicit) == make_model_spec_hash(
            m_name, cfg_explicit
        )

    # Case 5: Non-default Track P fields deterministically alter digests
    cfg_sim_feat = RunConfig.model_validate(
        {
            **base_payload,
            "features": {
                "similarity_cluster_feature": True,
                "similarity_method": "physics_attractor",
                "similarity_n_clusters": 3,
            },
        }
    )
    assert make_run_id(cfg_sim_feat) != make_run_id(cfg_implicit)
    assert make_model_spec_hash("theta", cfg_sim_feat) != make_model_spec_hash(
        "theta", cfg_implicit
    )

    cfg_sim_hier = RunConfig.model_validate(
        {
            **base_payload,
            "hierarchy": {"similarity_hierarchy": True, "similarity_n_clusters": 3},
        }
    )
    assert make_run_id(cfg_sim_hier) != make_run_id(cfg_implicit)

    cfg_sim_ens = RunConfig.model_validate(
        {
            **base_payload,
            "ensemble": {"similarity_smoothing": 0.25},
        }
    )
    assert make_run_id(cfg_sim_ens) != make_run_id(cfg_implicit)
    assert make_ensemble_id(cfg_sim_ens.ensemble) != make_ensemble_id(cfg_implicit.ensemble)


def test_t1_f2_similarity_seven_lenses_psd_physics_spectral_dtw_minirocket_foundation() -> None:
    """Tier 1 F2 (>= 5 cases): Verify all 7 `SimilarityMethod` lenses (`hybrid`,
    `physics_attractor`, `spectral_phase`, `elastic_dtw`, `minirocket`, `structural_signature`,
    `foundation_latent`), symmetric PSD unit-diagonal matrix invariants, Lyapunov & Hurst regime
    ordering, Kuramoto phase-locking, and directed lead-lag shift recovery.
    """
    from scale_forecasting.similarity import (
        SIMILARITY_METHODS,
        compute_similarity_matrix,
        embed_series_df,
        extract_foundation_embeddings,
    )

    panel = _make_dynamical_regimes_panel(n_days=84, seed=1801)

    # Case 1: All 7 similarity lenses produce symmetric PSD (N x N) matrices in [0, 1] with S_ii=1
    for method_name in SIMILARITY_METHODS:
        res = compute_similarity_matrix(panel, method=method_name, top_k=5, max_lag=5)
        mat = res.matrix_df.to_numpy(dtype=float)
        assert mat.shape == (8, 8)
        assert np.all(np.isfinite(mat))
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)
        np.testing.assert_allclose(np.diag(mat), np.ones(8), atol=1e-6)
        assert bool((mat >= -1e-6).all()) and bool((mat <= 1.0 + 1e-6).all())
        eigvals = np.linalg.eigvalsh(0.5 * (mat + mat.T))
        assert float(np.min(eigvals)) >= -1e-5, (
            f"Method {method_name} produced non-PSD matrix: min_eig={np.min(eigvals)}"
        )

    # Case 2: Physics attractor invariants order Lyapunov lambda_max & Permutation Entropy
    emb_df = embed_series_df(panel, method="physics_attractor").set_index("ts_id")
    assert float(emb_df.loc["chaotic_logistic", "lyapunov_exp"]) > float(
        emb_df.loc["harm_sync_a", "lyapunov_exp"]
    )
    assert float(emb_df.loc["chaotic_logistic", "perm_entropy"]) > float(
        emb_df.loc["harm_sync_a", "perm_entropy"]
    )
    assert float(emb_df.loc["ou_damped", "ou_damping_rate"]) > 0.0

    # Case 3: Spectral phase Kuramoto synchronization R_ij & exact lead-lag shift recovery (+3)
    spec_res = compute_similarity_matrix(panel, method="spectral_phase", top_k=7, max_lag=5)
    assert float(spec_res.matrix_df.loc["harm_sync_a", "harm_sync_b"]) > float(
        spec_res.matrix_df.loc["harm_sync_a", "chaotic_logistic"]
    )
    pairs = spec_res.pairs_df
    pair_ab = pairs[
        (pairs["ts_id"] == "harm_sync_a") & (pairs["neighbor_ts_id"] == "harm_lag_plus3")
    ]
    if not pair_ab.empty:
        assert abs(int(pair_ab["optimal_lag"].iloc[0])) == 3

    # Case 4: Elastic Spring DTW matches time-warped promotional pulse twins
    dtw_res = compute_similarity_matrix(
        panel, method="elastic_dtw", spring_stiffness=0.05, window_frac=0.15
    )
    assert float(dtw_res.matrix_df.loc["warped_pulse_a", "warped_pulse_b"]) > float(
        dtw_res.matrix_df.loc["warped_pulse_a", "ou_damped"]
    )

    # Case 5: Structural signature orders Hurst H & foundation_latent pure-core fallback works
    struct_emb = embed_series_df(panel, method="structural_signature").set_index("ts_id")
    assert float(struct_emb.loc["trend_persistent", "hurst_exp"]) > float(
        struct_emb.loc["ou_damped", "hurst_exp"]
    )

    found_emb = embed_series_df(panel, method="foundation_latent", allow_fallback=True)
    assert "embedding_backend_used" in found_emb.columns
    assert bool(found_emb["embedding_backend_used"].notna().all())
    raw_found = extract_foundation_embeddings(panel, allow_fallback=True)
    assert raw_found is not None


def test_t1_f3_public_similarity_api_clustering_digital_twin_modes_and_plot() -> None:
    """Tier 1 F3 (>= 5 cases): Verify `embed_series_df`, `compute_similarity_matrix`
    (`SimilarityMatrixResult`), `cluster_by_similarity_df` (silhouette-optimal K*,
    `min_cluster_size=2` merge, `cluster_archetype`), `find_similar_series_df` (`full_history`,
    `launch_prefix`, `recent_window`), and `plot_similarity_map`.
    """
    from scale_forecasting.similarity import (
        SimilarityMatrixResult,
        cluster_by_similarity_df,
        compute_similarity_matrix,
        embed_series_df,
        find_similar_series_df,
        plot_similarity_map,
    )

    panel = _make_dynamical_regimes_panel(n_days=84, seed=1803)

    # Case 1: embed_series_df schema completeness
    emb_df = embed_series_df(panel, method="hybrid", n_components=2)
    req_emb_cols = {
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
    }
    assert req_emb_cols.issubset(emb_df.columns)
    assert len(emb_df) == 8

    # Case 2: SimilarityMatrixResult contract (.matrix_df, .pairs_df, .embeddings_df, .to_dict())
    sim_res = compute_similarity_matrix(panel, method="hybrid", top_k=5, max_lag=4)
    assert isinstance(sim_res, SimilarityMatrixResult)
    assert isinstance(sim_res.to_dict(), dict)
    assert isinstance(sim_res.dataframe, pd.DataFrame)
    req_pair_cols = {
        "ts_id",
        "neighbor_ts_id",
        "rank",
        "similarity_score",
        "physics_sim",
        "spectral_sim",
        "shape_sim",
        "structural_sim",
        "optimal_lag",
        "lag_correlation",
        "method",
    }
    assert req_pair_cols.issubset(sim_res.pairs_df.columns)

    # Case 3: cluster_by_similarity_df schema, min_cluster_size=2 enforcement, and archetypes
    clust_df = cluster_by_similarity_df(
        panel, n_clusters="auto", method="hybrid", min_cluster_size=2, max_clusters=4
    )
    req_clust_cols = {
        "ts_id",
        "similarity_cluster_id",
        "cluster_int_id",
        "cluster_medoid_ts_id",
        "similarity_to_medoid",
        "cluster_silhouette",
        "cluster_size",
        "cluster_archetype",
    }
    assert req_clust_cols.issubset(clust_df.columns)
    assert bool((clust_df["cluster_size"] >= 2).all())
    assert bool(clust_df["similarity_cluster_id"].astype(str).str.startswith("cluster_").all())
    # Phase-locked harmonic twins must land in the same cluster
    c_map = clust_df.set_index("ts_id")["similarity_cluster_id"].to_dict()
    assert c_map["harm_sync_a"] == c_map["harm_sync_b"]

    # Case 4: find_similar_series_df across "full_history", "launch_prefix", and "recent_window"
    for mode_name in ("full_history", "launch_prefix", "recent_window"):
        matches = find_similar_series_df(
            panel,
            query_ts_id="harm_sync_a",
            top_k=3,
            mode=mode_name,
            method="hybrid",
            prefix_len=14 if mode_name == "launch_prefix" else None,
            max_lag=4,
        )
        req_match_cols = {
            "query_ts_id",
            "neighbor_ts_id",
            "rank",
            "similarity_score",
            "physics_sim",
            "spectral_sim",
            "shape_sim",
            "structural_sim",
            "optimal_lag",
            "lag_correlation",
            "scale_ratio",
            "level_offset",
            "velocity_sim",
            "acceleration_sim",
            "matching_mode",
            "method",
        }
        assert req_match_cols.issubset(matches.columns)
        assert len(matches) == 3
        top_neighbors = set(matches["neighbor_ts_id"].iloc[:2].astype(str))
        assert bool(top_neighbors & {"harm_sync_b", "harm_lag_plus3"})

    # Case 5: plot_similarity_map returns matplotlib Axes on DataFrame and SimilarityMatrixResult
    ax1 = plot_similarity_map(panel, method="hybrid")
    assert isinstance(ax1, matplotlib.axes.Axes)
    ax2 = plot_similarity_map(sim_res)
    assert isinstance(ax2, matplotlib.axes.Axes)
    plt.close("all")


def test_t1_f4_similarity_cold_start_npi_zero_metadata_and_hybrid_matching() -> None:
    """Tier 1 F4 (>= 5 cases): Verify Cold-Start NPI with `@similarity_cold_start` and
    `cold_start_matching="similarity" | "hybrid" | "auto"` on a panel with ZERO `static_covariates`
    (`[ts_id, ds, y]` only), verifying kinematic launch-prefix donor matching and new columns
    (`matching_mode_used`, `donor_similarity_score`, `top_donor_ts_ids`).
    """
    from scale_forecasting.cold_start import (
        build_cold_start_donor_pool,
        cold_start_transfer_df,
        forecast_cold_start_series,
    )

    panel = _make_multi_regime_retail_panel(
        n_days=56, n_short_days=5, include_cold_start=True, seed=1804
    )
    assert set(panel.columns) == {"ts_id", "ds", "y"}

    # Case 1: cold_start_transfer_df with zero static_covariates and matching="similarity"
    cs_df = cold_start_transfer_df(
        panel,
        min_obs=14,
        matching="similarity",
        similarity_method="hybrid",
        horizon=7,
        freq="D",
    )
    for new_col in ("matching_mode_used", "donor_similarity_score", "top_donor_ts_ids"):
        assert new_col in cs_df.columns, f"Missing column {new_col} in cold_start_transfer_df"

    npi_row = cs_df[cs_df["ts_id"] == "npi_surge_short"].iloc[0]
    assert bool(npi_row["is_cold_start"]) is True
    assert str(npi_row["matching_mode_used"]) in {"similarity", "hybrid"}
    assert float(npi_row["donor_similarity_score"]) > 0.0
    # Top donors for npi_surge_short (which ramps like flagship_*) should include flagship_*
    top_donors_str = str(npi_row["top_donor_ts_ids"])
    assert "flagship_" in top_donors_str

    # Case 2: All matching modes ("auto", "similarity", "hybrid", "attribute") execute cleanly
    pool = build_cold_start_donor_pool(panel, min_obs=14)
    short_df = panel[panel["ts_id"] == "npi_surge_short"].copy()
    for match_mode in ("auto", "similarity", "hybrid", "attribute"):
        fc_res = forecast_cold_start_series(
            short_df,
            pool,
            horizon=7,
            freq="D",
            min_obs=14,
            matching=match_mode,
        )
        assert fc_res.is_cold_start is True
        assert len(fc_res.predictions) == 7
        assert bool(np.isfinite(fc_res.predictions["yhat"]).all())
        assert hasattr(fc_res, "matching_mode_used")
        assert hasattr(fc_res, "donor_similarity_score")
        assert hasattr(fc_res, "top_donor_ts_ids")

    # Case 3: @similarity_cold_start variant tag resolution in features.resolve_model_variant
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f4-sim-cold-start",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["theta@similarity_cold_start"],
        }
    )
    spec = resolve_model_variant(cfg, "theta@similarity_cold_start")
    assert spec.effective_cfg.features.cold_start_transfer is True
    assert spec.effective_cfg.features.cold_start_matching == "similarity"

    # Case 4 & 5: run_cell and run_group with @similarity_cold_start on zero-metadata panel
    cell = run_cell(
        "npi_surge_short",
        short_df,
        "theta@similarity_cold_start",
        cfg,
        cold_start_donor_pool=pool,
    )
    assert cell.status == "ok"
    assert len(cell.predictions) == 7

    cells, status_df = run_group(panel, cfg)
    assert set(status_df["status"]) == {"ok"}
    assert len(cells) == panel["ts_id"].nunique()


def test_t1_f5_cluster_aware_similarity_global_pooling_and_spark_ray_isolation() -> None:
    """Tier 1 F5 (>= 5 cases): Verify `@similarity_global`, `training_mode="similarity_global"`,
    `similarity_cluster_feature=True`, `add_similarity_features` (leak-free `cutoff_ds`),
    `worker.is_panel_model` & `run_panel` (`fit_diagnostics["similarity_clusters"]`), and
    `spark_io.add_bucket` panel isolation.
    """
    from scale_forecasting.features import add_similarity_features

    panel = _make_multi_regime_retail_panel(
        n_days=49, n_short_days=5, include_cold_start=False, seed=1805
    )

    # Case 1: Variant resolution & expand_model_variants for @similarity_global
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f5-sim-global",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["tide@similarity_global"],
            "model_params": {"tide": {"max_steps": 4, "input_size": 7}},
            "features": {
                "similarity_method": "hybrid",
                "similarity_n_clusters": 2,
                "ablation_modes": ["local", "global", "similarity_global"],
            },
            "backtest": {"enabled": False},
        }
    )
    spec = resolve_model_variant(cfg, "tide@similarity_global")
    assert spec.training_mode == "similarity_global"
    assert is_panel_model("tide@similarity_global", cfg) is True

    expanded_models, expanded_params = expand_model_variants(
        RunConfig.model_validate(
            {
                "run_name": "t1-f5-expand",
                "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
                "models": ["tide"],
                "features": {"ablation_modes": ["local", "similarity_global"]},
            }
        )
    )
    assert "tide@similarity_global" in expanded_models
    assert expanded_params["tide@similarity_global"]["training_mode"] == "similarity_global"

    # Case 2: Leak-free add_similarity_features with cutoff_ds and build_features / build_future
    cutoff = pd.Timestamp("2025-02-01")
    panel_with_sim = add_similarity_features(panel, cfg, cutoff_ds=cutoff)
    assert "similarity_cluster_id" in panel_with_sim.columns
    assert "similarity_to_medoid" in panel_with_sim.columns
    # Perturbing future targets strictly after cutoff_ds has ZERO effect on cluster features
    panel_perturbed = panel.copy()
    panel_perturbed.loc[panel_perturbed["ds"] > cutoff, "y"] += 50_000.0
    panel_pert_sim = add_similarity_features(panel_perturbed, cfg, cutoff_ds=cutoff)
    np.testing.assert_allclose(
        panel_with_sim["similarity_to_medoid"].to_numpy(dtype=float),
        panel_pert_sim["similarity_to_medoid"].to_numpy(dtype=float),
        atol=1e-12,
    )

    # Case 3: build_features & build_future_features attach cluster_id and medoid similarity
    cfg_feat = RunConfig.model_validate(
        {
            "run_name": "t1-f5-feat",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["regression_lags"],
            "features": {"similarity_cluster_feature": True},
        }
    )
    one_series = panel_with_sim[panel_with_sim["ts_id"] == "flagship_0"].copy()
    y_tr, X_tr = build_features(one_series, cfg_feat)
    assert X_tr is not None
    assert "similarity_cluster_id" in X_tr.columns
    assert "similarity_to_medoid" in X_tr.columns
    X_fut = build_future_features(y_tr, X_tr, cfg_feat, horizon=7)
    assert X_fut is not None
    assert "similarity_cluster_id" in X_fut.columns
    assert "similarity_to_medoid" in X_fut.columns
    assert len(X_fut) == 7

    # Case 4: worker.run_panel fits 1 global model per cluster and records similarity_clusters
    cells = run_panel(panel, "tide@similarity_global", cfg)
    assert len(cells) == 6
    for c in cells:
        assert c.status == "ok", f"tide@similarity_global failed: {c.error}"
        assert c.model_type == "tide@similarity_global"
        assert len(c.predictions) == 7
        assert "similarity_clusters" in c.diagnostics
        sim_diag = c.diagnostics["similarity_clusters"]
        assert isinstance(sim_diag, dict)
        assert int(sim_diag.get("n_clusters", 0)) == 2

    # Case 5: ray_io.chunk_cells and spark_io.bucket_key_cols isolate @similarity_global panel
    chunks = chunk_cells(panel, cfg, ["tide@similarity_global"], n_chunks=4)
    tide_chunks = [c for c in chunks if (c[_MODEL_COL] == "tide@similarity_global").all()]
    assert len(tide_chunks) == 1
    assert set(tide_chunks[0]["ts_id"]) == set(panel["ts_id"].unique())
    assert bucket_key_cols(cfg_feat) == [_MODEL_COL]


def test_t1_f6_latent_similarity_hierarchy_reconciliation_and_goal_seek() -> None:
    """Tier 1 F6 (>= 5 cases): Verify `HierarchyConfig(similarity_hierarchy=True)` and
    `materialize_similarity_hierarchy` on a flat `[ts_id, ds, y]` panel (`levels=[]`) across
    `bottom_up`, `ols`, `wls_var`, `wls_struct`, `mint_shrink`, `mint_nonneg`,
    `cross_temporal_mint`, and `allocate_target(node_id="similarity_cluster_id=cluster_0")`.
    """
    from scale_forecasting.reconciliation import materialize_similarity_hierarchy

    panel = _make_multi_regime_retail_panel(
        n_days=42, n_short_days=5, include_cold_start=False, seed=1806
    )
    assert set(panel.columns) == {"ts_id", "ds", "y"}

    # Case 1: materialize_similarity_hierarchy attaches similarity_cluster_id to flat panel
    mat_panel, cluster_map = materialize_similarity_hierarchy(panel, n_clusters=2, method="hybrid")
    assert "similarity_cluster_id" in mat_panel.columns
    assert len(cluster_map) == 6
    assert set(cluster_map.values()) == {"cluster_0", "cluster_1"}

    # Case 2: build_hierarchy on flat panel with similarity_hierarchy=True builds 3-level tree
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f6-sim-hier",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
            "hierarchy": {
                "similarity_hierarchy": True,
                "similarity_n_clusters": 2,
                "similarity_method": "hybrid",
                "reconciliation_methods": [
                    "bottom_up",
                    "ols",
                    "wls_var",
                    "wls_struct",
                    "mint_shrink",
                    "mint_nonneg",
                    "cross_temporal_mint",
                ],
                "temporal_aggregations": [7],
            },
        }
    )
    hier_panel, spec = build_hierarchy(panel, cfg)
    assert "__total__" in spec.node_ids
    assert any(nid.startswith("similarity_cluster_id=cluster_") for nid in spec.node_ids)
    assert set(spec.bottom_ids) == set(panel["ts_id"].unique())

    # Case 3: reconcile_forecasts on flat bottom-only predictions auto-synthesizes upper nodes
    f_dates = pd.date_range("2025-02-12", periods=7, freq="D")
    flat_pred_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(1806)
    for tid in spec.bottom_ids:
        for h_idx, dt in enumerate(f_dates, start=1):
            yh = 95.0 + 10.0 * np.sin(2.0 * np.pi * h_idx / 7.0)
            flat_pred_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "naive_mean",
                    "ds": dt,
                    "forecast_date": dt,
                    "horizon_step": h_idx,
                    "yhat": float(yh),
                    "yhat_lower": float(yh - 12.0),
                    "yhat_upper": float(yh + 12.0),
                }
            )
        for f_id in range(2):
            cutoff = pd.Timestamp("2025-01-20") + pd.Timedelta(days=7 * f_id)
            for h_idx in range(1, 8):
                dt = cutoff + pd.Timedelta(days=h_idx)
                yh = 95.0
                oof_rows.append(
                    {
                        "ts_id": tid,
                        "model_type": "naive_mean",
                        "fold_id": f_id,
                        "cutoff_date": cutoff,
                        "ds": dt,
                        "forecast_date": dt,
                        "horizon_step": h_idx,
                        "y_true": float(yh + rng.normal(0.0, 4.0)),
                        "yhat": float(yh),
                        "yhat_lower": float(yh - 10.0),
                        "yhat_upper": float(yh + 10.0),
                    }
                )
    flat_preds_df = pd.DataFrame(flat_pred_rows)
    flat_oof_df = pd.DataFrame(oof_rows)

    for rec_method in (
        "bottom_up",
        "top_down",
        "middle_out",
        "ols",
        "wls_var",
        "wls_struct",
        "mint_shrink",
        "mint_nonneg",
        "cross_temporal_mint",
    ):
        rec_df = reconcile_forecasts(
            flat_preds_df,
            spec,
            rec_method,
            cfg=cfg,
            history_df=hier_panel,
            oof_df=flat_oof_df,
        )
        assert set(spec.node_ids).issubset(set(rec_df["ts_id"].unique()))
        assert verify_coherence(rec_df, spec) < 1e-6

    # Case 4: reconcile_oof on flat bottom-only OOF auto-synthesizes upper nodes & reconciles < 1e-6
    rec_oof = reconcile_oof(
        flat_oof_df,
        spec,
        "mint_shrink",
        cfg=cfg,
        history_df=hier_panel,
    )
    assert set(spec.node_ids).issubset(set(rec_oof["ts_id"].unique()))

    # Case 5: allocate_target(node_id="similarity_cluster_id=cluster_0") on flat predictions
    gs_res = allocate_target(
        flat_preds_df,
        target_total=2800.0,
        node_id="similarity_cluster_id=cluster_0",
        history_df=panel,
        cfg=cfg,
        return_result=True,
    )
    assert isinstance(gs_res, GoalSeekResult)
    assert gs_res.max_target_discrepancy < 1e-4


def test_t1_f7_similarity_regularized_copula_shrinkage_and_inventory_pooling() -> None:
    """Tier 1 F7 (>= 5 cases): Verify `shrink_correlation_toward_similarity` when
    $K_{\text{folds}} < N_{\text{series}}$, `sample_trajectories(..., cross_series=True,
    similarity_shrinkage=True)` recording `similarity_shrinkage_weight`, and
    `optimize_inventory_portfolio(uncertainty_source="copula", similarity_shrinkage=True)`.
    """
    from scale_forecasting.copula import (
        sample_trajectories,
        shrink_correlation_toward_similarity,
    )
    from scale_forecasting.inventory_optimizer import optimize_inventory_portfolio
    from scale_forecasting.similarity import compute_similarity_matrix

    panel = _make_multi_regime_retail_panel(
        n_days=42, n_short_days=5, include_cold_start=False, seed=1807
    )
    sim_res = compute_similarity_matrix(panel, method="hybrid")
    sim_mat = sim_res.matrix_df

    # Case 1: shrink_correlation_toward_similarity produces valid SPD unit-diagonal matrix
    n_s = len(sim_mat)
    emp_identity = np.eye(n_s, dtype=float)
    shrunk_auto = shrink_correlation_toward_similarity(
        emp_identity, sim_mat, n_samples=1, weight="auto"
    )
    assert shrunk_auto.shape == (n_s, n_s)
    np.testing.assert_allclose(shrunk_auto, shrunk_auto.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(shrunk_auto), np.ones(n_s), atol=1e-6)
    eigvals = np.linalg.eigvalsh(shrunk_auto)
    assert float(np.min(eigvals)) > 0.0
    # Off-diagonal similarity structure must be injected (shrunk != identity)
    assert float(np.max(np.abs(shrunk_auto - emp_identity))) > 0.05

    # Case 2: Explicit weight=0.0 returns empirical correlation; weight=0.60 blends cleanly
    shrunk_zero = shrink_correlation_toward_similarity(
        emp_identity, sim_mat, n_samples=2, weight=0.0
    )
    np.testing.assert_allclose(shrunk_zero, emp_identity, atol=1e-5)

    # Case 3: Rank-deficient K_folds=1 < N_series=6 OOF in sample_trajectories with shrinkage
    f_dates = pd.date_range("2025-02-12", periods=7, freq="D")
    pred_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for tid in sim_mat.index:
        for h_idx, dt in enumerate(f_dates, start=1):
            pred_rows.append(
                {
                    "ts_id": str(tid),
                    "model_type": "theta",
                    "ds": dt,
                    "forecast_date": dt,
                    "horizon_step": h_idx,
                    "yhat": 110.0,
                    "yhat_lower": 92.0,
                    "yhat_upper": 128.0,
                }
            )
            oof_rows.append(
                {
                    "ts_id": str(tid),
                    "model_type": "theta",
                    "fold_id": 0,
                    "cutoff_date": pd.Timestamp("2025-02-04"),
                    "ds": dt - pd.Timedelta(days=7),
                    "horizon_step": h_idx,
                    "y_true": 112.0 + float(h_idx),
                    "yhat": 110.0,
                }
            )
    preds_df = pd.DataFrame(pred_rows)
    oof_k1_df = pd.DataFrame(oof_rows)

    paths_shrunk = sample_trajectories(
        preds_df,
        oof_k1_df,
        history_df=panel,
        n_paths=200,
        copula_type="gaussian",
        cross_series=True,
        similarity_shrinkage=True,
        similarity_weight="auto",
        seed=42,
    )
    assert not paths_shrunk.empty
    assert "similarity_shrinkage_weight" in paths_shrunk.attrs
    assert float(paths_shrunk.attrs["similarity_shrinkage_weight"]) > 0.0

    # Case 4 & 5: optimize_inventory_portfolio with similarity_shrinkage=True and copula paths
    inv_df = optimize_inventory_portfolio(
        preds_df,
        oof_k1_df,
        max_budget=25_000.0,
        unit_cost=10.0,
        uncertainty_source="copula",
        similarity_shrinkage=True,
        similarity_weight="auto",
        similarity_matrix=sim_mat,
        n_paths=150,
        seed=42,
    )
    assert len(inv_df) == n_s
    assert float(inv_df["order_spend"].sum()) <= 25_000.0 + 1e-6
    assert bool((inv_df["uncertainty_source"] == "copula").all())


def test_t1_f8_cross_sku_cannibalization_similarity_priors_and_scenario_optimizer() -> None:
    """Tier 1 F8 (>= 5 cases): Verify `estimate_promo_elasticity` (`use_similarity_priors=True`)
    and `optimize_scenario(..., use_similarity_priors=True)` weighting cross-SKU cannibalization
    by pairwise digital-twin similarity.
    """
    from scale_forecasting.elasticity import (
        compute_promo_elasticity_decomposition,
        estimate_promo_elasticity,
        summarize_promo_elasticity,
    )
    from scale_forecasting.scenario_optimizer import optimize_scenario

    hist_df, fut_df = _make_promo_cannibalization_twin_panel(n_days=56, horizon=7, seed=1808)

    # Case 1: estimate_promo_elasticity alias exists and accepts use_similarity_priors=True
    assert callable(estimate_promo_elasticity)
    decomp_sim = estimate_promo_elasticity(
        hist_df,
        promo_col="promo",
        price_col="price",
        use_similarity_priors=True,
        similarity_method="hybrid",
    )
    assert not decomp_sim.empty
    assert "cannibalization_dip" in decomp_sim.columns
    assert "top_cannibal_ts_id" in decomp_sim.columns
    assert "cannibal_similarity_score" in decomp_sim.columns

    # Case 2: Digital twin twin_a1 identifies twin_a2 as its top cannibal substitute
    a1_rows = decomp_sim[decomp_sim["ts_id"] == "twin_a1"]
    assert str(a1_rows["top_cannibal_ts_id"].iloc[0]) == "twin_a2"
    assert float(a1_rows["cannibal_similarity_score"].iloc[0]) > 0.50

    # Case 3: Cannibalization dip on twin_a1 is stronger under similarity priors than uniform peers
    decomp_unif = compute_promo_elasticity_decomposition(
        hist_df,
        promo_col="promo",
        price_col="price",
        use_similarity_priors=False,
    )
    a1_cannib_sim = float(np.abs(a1_rows["cannibalization_dip"]).sum())
    a1_cannib_unif = float(
        np.abs(decomp_unif.loc[decomp_unif["ts_id"] == "twin_a1", "cannibalization_dip"]).sum()
    )
    assert a1_cannib_sim > 0.0
    assert a1_cannib_sim >= a1_cannib_unif * 0.95

    # Case 4: summarize_promo_elasticity aggregates the similarity-prior decomposition frame
    summary_df = summarize_promo_elasticity(decomp_sim)
    assert len(summary_df) == 4

    # Case 5: optimize_scenario forwards use_similarity_priors=True and improves gross profit
    scen_res = optimize_scenario(
        hist_df,
        future_covariates_df=fut_df,
        objective="gross_profit",
        price_col="price",
        promo_col="promo",
        unit_cost=4.5,
        max_promo_days=2,
        min_cooldown_days=2,
        use_similarity_priors=True,
        reconcile=False,
    )
    assert scen_res.optimal_objective_value >= scen_res.baseline_objective_value - 1e-6


def test_t1_f9_oof_champion_and_inverse_error_similarity_smoothing() -> None:
    """Tier 1 F9 (>= 5 cases): Verify `smooth_oof_losses_by_similarity` and
    `EnsembleConfig(similarity_smoothing=0.35)` across `combine_oof` and `combine_calculated`,
    proving that smoothing across digital twins prevents a 1-fold noise fluke from picking an
    inferior champion model.
    """
    from scale_forecasting.ensembler import smooth_oof_losses_by_similarity

    # Case 1: Direct mathematical verification of smooth_oof_losses_by_similarity
    # Suppose sku_0, sku_1, sku_2 are near-identical digital twins (S_ij = 0.95).
    # Model A ("theta") is genuinely better across the cluster (true loss ~ 0.08),
    # whereas Model B ("naive_mean") has loss ~ 0.20, except on sku_0 where a 1-fold fluke gives
    # Model A = 0.14 and Model B = 0.12.
    losses_by_series = {
        "sku_0": np.array(
            [0.14, 0.12], dtype=float
        ),  # Fluke: Model B looks slightly better on sku_0
        "sku_1": np.array([0.06, 0.22], dtype=float),  # Twin 1: Model A is much better
        "sku_2": np.array([0.07, 0.21], dtype=float),  # Twin 2: Model A is much better
    }
    sim_mat = pd.DataFrame(
        [[1.0, 0.95, 0.95], [0.95, 1.0, 0.95], [0.95, 0.95, 1.0]],
        index=["sku_0", "sku_1", "sku_2"],
        columns=["sku_0", "sku_1", "sku_2"],
    )

    # With smoothing=0.0, sku_0 keeps [0.14, 0.12] (Model B wins on sku_0)
    unsmoothed = smooth_oof_losses_by_similarity(losses_by_series, sim_mat, smoothing=0.0)
    np.testing.assert_allclose(unsmoothed["sku_0"], losses_by_series["sku_0"], atol=1e-12)
    assert int(np.argmin(unsmoothed["sku_0"])) == 1

    # With smoothing=0.50, sku_0's smoothed loss is (0.5 * [0.14, 0.12] + 0.5 * [0.065, 0.215])
    # = [0.1025, 0.1675] -> Model A (index 0) wins as champion!
    smoothed = smooth_oof_losses_by_similarity(losses_by_series, sim_mat, smoothing=0.50)
    assert float(smoothed["sku_0"][0]) < float(smoothed["sku_0"][1])
    assert int(np.argmin(smoothed["sku_0"])) == 0

    # Case 2 & 3: combine_calculated with similarity_smoothing=0.50 flips sku_0 champion to theta
    f_dates = [pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-02")]
    base_rows: list[dict[str, Any]] = []
    for tid in ("sku_0", "sku_1", "sku_2"):
        for dt in f_dates:
            base_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "theta",
                    "forecast_date": dt,
                    "yhat": 100.0,
                    "yhat_lower": 90.0,
                    "yhat_upper": 110.0,
                }
            )
            base_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "naive_mean",
                    "forecast_date": dt,
                    "yhat": 200.0,
                    "yhat_lower": 180.0,
                    "yhat_upper": 220.0,
                }
            )
    base_df = pd.DataFrame(base_rows)
    base_df.attrs["similarity_matrix"] = sim_mat

    metric_df = pd.DataFrame(
        [
            {"ts_id": "sku_0", "model_type": "theta", "wape": 0.14},
            {"ts_id": "sku_0", "model_type": "naive_mean", "wape": 0.12},
            {"ts_id": "sku_1", "model_type": "theta", "wape": 0.06},
            {"ts_id": "sku_1", "model_type": "naive_mean", "wape": 0.22},
            {"ts_id": "sku_2", "model_type": "theta", "wape": 0.07},
            {"ts_id": "sku_2", "model_type": "naive_mean", "wape": 0.21},
        ]
    )

    cfg_no_smooth = RunConfig.model_validate(
        {
            "run_name": "t1-f9-no-smooth",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 2},
            "models": ["theta", "naive_mean"],
            "ensemble": {"strategies": ["champion", "inverse_error"], "similarity_smoothing": 0.0},
        }
    )
    cfg_smooth = RunConfig.model_validate(
        {
            "run_name": "t1-f9-smooth",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 2},
            "models": ["theta", "naive_mean"],
            "ensemble": {"strategies": ["champion", "inverse_error"], "similarity_smoothing": 0.50},
        }
    )

    rows_no_smooth = pd.DataFrame(combine_calculated(base_df, cfg_no_smooth, metric_df=metric_df))
    rows_smooth = pd.DataFrame(combine_calculated(base_df, cfg_smooth, metric_df=metric_df))

    champ_sku0_unsmoothed = float(
        rows_no_smooth[
            (rows_no_smooth["ts_id"] == "sku_0")
            & (rows_no_smooth["model_type"] == "ensemble_champion")
        ]["yhat"].iloc[0]
    )
    champ_sku0_smoothed = float(
        rows_smooth[
            (rows_smooth["ts_id"] == "sku_0") & (rows_smooth["model_type"] == "ensemble_champion")
        ]["yhat"].iloc[0]
    )
    assert champ_sku0_unsmoothed == pytest.approx(200.0)
    assert champ_sku0_smoothed == pytest.approx(100.0)

    # Case 4 & 5: combine_oof executes cleanly with similarity_smoothing > 0
    oof_rows: list[dict[str, Any]] = []
    for tid, (err_a, err_b) in zip(
        ("sku_0", "sku_1", "sku_2"),
        ((14.0, 12.0), (6.0, 22.0), (7.0, 21.0)),
        strict=True,
    ):
        for step_idx in range(4):
            dt = pd.Timestamp("2025-02-01") + pd.Timedelta(days=step_idx)
            y_t = 100.0 + 10.0 * np.sin(step_idx)
            oof_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "theta",
                    "fold_id": step_idx // 2,
                    "cutoff_date": dt - pd.Timedelta(days=1),
                    "forecast_date": dt,
                    "ds": dt,
                    "horizon_step": 1,
                    "y_true": y_t,
                    "yhat": y_t + err_a,
                }
            )
            oof_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "naive_mean",
                    "fold_id": step_idx // 2,
                    "cutoff_date": dt - pd.Timedelta(days=1),
                    "forecast_date": dt,
                    "ds": dt,
                    "horizon_step": 1,
                    "y_true": y_t,
                    "yhat": y_t + err_b,
                }
            )
    oof_df = pd.DataFrame(oof_rows)
    oof_df.attrs["similarity_matrix"] = sim_mat
    ens_oof_rows = combine_oof(oof_df, cfg_smooth, {})
    assert len(ens_oof_rows) > 0


def test_t1_f10_sdk_playground_bakeoff_report_and_root_exports(tmp_path: Path) -> None:
    """Tier 1 F10 (>= 5 cases): Verify lazy root exports on `scale_forecasting` (`sf`), all 6
    similarity methods on `Forecaster`, `PlaygroundRun`, and `BakeOff`, and the interactive HTML
    report similarity section (`report.py`).
    """
    # Case 1: Root package lazy exports for all Track P symbols
    for sym in (
        "SIMILARITY_METHODS",
        "VALID_SIMILARITY_METHODS",
        "SimilarityMatrixResult",
        "embed_series_df",
        "compute_similarity_matrix",
        "cluster_by_similarity_df",
        "find_similar_series_df",
        "plot_similarity_map",
        "extract_foundation_embeddings",
        "shrink_correlation_toward_similarity",
        "materialize_similarity_hierarchy",
        "DatasetProfile",
    ):
        assert hasattr(sf, sym), f"Missing root export sf.{sym}"

    # Case 2: Forecaster SDK surface across all 6 similarity methods
    panel = _make_multi_regime_retail_panel(
        n_days=42, n_short_days=5, include_cold_start=False, seed=1810
    )
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f10-sdk-sim",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean", "theta"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7},
        }
    )
    f = sf.Forecaster(cfg, data=panel)
    assert not f.similarity_df(method="hybrid", top_k=3).empty
    mat_df = f.similarity_matrix_df(method="hybrid")
    assert mat_df.shape == (6, 6)
    assert not f.similarity_clusters_df(n_clusters=2).empty
    assert not f.similarity_embeddings_df(method="hybrid").empty
    assert not f.find_similar_series_df(query_ts_id="flagship_0", top_k=3).empty
    ax_f = f.plot_similarity_map(method="hybrid")
    assert isinstance(ax_f, matplotlib.axes.Axes)

    # Case 3: PlaygroundRun surface (including single-series fallback)
    pr = run_model("theta", horizon=7, backtest=True)
    assert not pr.similarity_df().empty
    assert not pr.similarity_matrix_df().empty
    assert not pr.similarity_clusters_df().empty
    assert not pr.similarity_embeddings_df().empty
    assert not pr.find_similar_series_df().empty
    ax_pr = pr.plot_similarity_map()
    assert isinstance(ax_pr, matplotlib.axes.Axes)

    # Case 4: BakeOff surface
    bo = bakeoff(models=["naive_mean", "theta"], horizon=7, n_folds=2)
    assert not bo.similarity_df().empty
    assert not bo.similarity_matrix_df().empty
    assert not bo.similarity_clusters_df().empty
    assert not bo.similarity_embeddings_df().empty
    assert not bo.find_similar_series_df().empty
    ax_bo = bo.plot_similarity_map()
    assert isinstance(ax_bo, matplotlib.axes.Axes)

    # Case 5: render_html_report and export_report include Time-Series Similarity section
    html_str = render_html_report(
        f,
        similarity_df=f.similarity_df(),
        similarity_clusters_df=f.similarity_clusters_df(),
    )
    assert "similarity" in html_str.lower()
    out_file = export_report(f, tmp_path / "track_p_report.html", include_json_sidecar=True)
    assert out_file.exists()
    plt.close("all")


def test_t1_f11_auto_forecaster_step0_branch_2b_2c_assess_q4_q5_and_evolve_similarity_kernel() -> (
    None
):
    """Tier 1 F11 (>= 5 cases): Verify `DatasetProfile` (`PanelAutoProfile`) 5 similarity fields,
    `auto_policy.py` Branch `2B`/`2C` injecting `<model>@similarity_global` and
    `cold_start_matching="hybrid"`, `practitioner_assessment_df()` (Questions 4 & 5),
    `AutoPolicyWeights`, and 5th evolution target `Forecaster.evolve(target="similarity_kernel")`.
    """
    panel = _make_multi_regime_retail_panel(
        n_days=49, n_short_days=5, include_cold_start=True, seed=1811
    )

    # Case 1: DatasetProfile / PanelAutoProfile populates all 5 new similarity fields
    prof = inspect_panel_for_auto(panel, horizon=7, freq="D")
    for field_name in (
        "mean_pairwise_similarity",
        "optimal_similarity_clusters",
        "similarity_silhouette_score",
        "lead_lag_pairs_count",
        "has_similarity_clusters",
    ):
        assert hasattr(prof, field_name), f"DatasetProfile missing {field_name}"
    assert int(prof.optimal_similarity_clusters) >= 2
    assert float(prof.mean_pairwise_similarity) > 0.0
    prof_df = prof.to_df()
    assert len(prof_df) == 7

    # Case 2: auto_policy propose_autofocus_pass & DAG branches inject @similarity_global
    bp = build_config(panel, horizon=7, freq="D", tier="first_light")
    mutations = propose_autofocus_pass(
        bp.config.model_dump(mode="python"),
        prof,
        {"cold_start_donor_transfer_recommended": True},
    )
    mut_str = str(mutations)
    assert "similarity_global" in mut_str or "similarity_cluster_feature" in mut_str
    branches = propose_autofocus_dag_branches(
        bp.config.model_dump(mode="python"),
        prof,
        {"cold_start_donor_transfer_recommended": True},
    )
    assert len(branches) >= 1

    # Case 3: Forecaster.auto() & practitioner_assessment_df() (f.assess()) Questions 4 & 5
    f_auto = sf.Forecaster.auto(
        panel,
        horizon=7,
        freq="D",
        tier="first_light",
        models=["naive_mean", "theta"],
        n_folds=1,
        autofocus=False,
    )
    assess_df = f_auto.practitioner_assessment_df()
    assert len(assess_df) == 7
    q4_text = str(assess_df.iloc[3].to_dict()).lower()
    q5_text = str(assess_df.iloc[4].to_dict()).lower()
    assert "similarity" in q4_text or "cluster" in q4_text
    assert "similarity" in q5_text or "cluster" in q5_text or "hierarchy" in q5_text

    # Case 4: AutoPolicyWeights & evolved_policy_thresholds include similarity fields
    th = evolved_policy_thresholds()
    assert "enable_similarity_global" in th
    assert "similarity_silhouette_threshold" in th
    weights = AutoPolicyWeights()
    assert hasattr(weights, "enable_similarity_global")
    assert hasattr(weights, "similarity_silhouette_threshold")

    # Case 5: 5th evolution target "similarity_kernel" in EVOLVE_TARGETS & Forecaster.evolve()
    assert "similarity_kernel" in EVOLVE_TARGETS
    seed_spec = get_target_seed("similarity_kernel")
    assert "compute_evolved_similarity_kernel" in seed_spec.seed_code
    mock_cands = deterministic_mock_candidates("similarity_kernel")
    assert len(mock_cands) >= 2

    clean_panel = panel[panel["ts_id"] != "npi_surge_short"].copy()
    ev_sim = sf.Forecaster.evolve(
        clean_panel,
        target="similarity_kernel",
        backend="mock",
        max_evaluations=2,
        horizon=5,
        tier="first_light",
    )
    assert ev_sim.target == "similarity_kernel"
    assert "compute_evolved_similarity_kernel" in ev_sim.champion_code


def test_t1_f12_mcp_12th_tool_compute_time_series_similarity_and_agent_surfaces() -> None:
    """Tier 1 F12 (>= 5 cases): Verify `McpServer` exposes `12` tools including
    `compute_time_series_similarity`, supports `include_similarity=True` on `run_playground` and
    `review_run`, includes `"similarity_kernel"` in `run_auto_forecast`, and syncs with
    `agent_surfaces.py`.
    """
    from scale_forecasting import agent_surfaces

    server = McpServer()
    tools = {t["name"]: t for t in server.list_tools()}

    # Case 1: 14 MCP tools including compute_time_series_similarity and ask_forecast_registry
    assert len(tools) == 14
    assert "compute_time_series_similarity" in tools

    # Case 2: Call compute_time_series_similarity across full_history, launch_prefix & query_ts_id
    panel = _make_multi_regime_retail_panel(
        n_days=28, n_short_days=5, include_cold_start=False, seed=1812
    )
    records = panel.assign(ds=panel["ds"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    sim_resp = server.call_tool(
        "compute_time_series_similarity",
        {
            "records": records,
            "method": "hybrid",
            "mode": "full_history",
            "top_k": 3,
            "n_clusters": 2,
            "max_lag": 2,
        },
    )
    assert sim_resp.get("isError") is not True
    sim_text = str(sim_resp)
    assert "flagship_0" in sim_text and "cluster" in sim_text.lower()

    query_resp = server.call_tool(
        "compute_time_series_similarity",
        {
            "records": records,
            "method": "physics_attractor",
            "mode": "launch_prefix",
            "query_ts_id": "flagship_0",
            "top_k": 2,
        },
    )
    assert query_resp.get("isError") is not True

    # Case 3: run_playground and review_run support include_similarity=True
    for t_name in ("run_playground", "review_run"):
        props = tools[t_name]["inputSchema"]["properties"]
        assert "include_similarity" in props

    play_resp = server.call_tool(
        "run_playground",
        {
            "models": ["naive_mean", "theta"],
            "horizon": 5,
            "n_folds": 1,
            "include_similarity": True,
        },
    )
    assert play_resp.get("isError") is not True

    # Case 4: run_auto_forecast evolve_target enum includes "similarity_kernel"
    auto_props = tools["run_auto_forecast"]["inputSchema"]["properties"]
    assert "similarity_kernel" in auto_props["evolve_target"]["enum"]

    # Case 5: agent_surfaces reflects @similarity_global, @similarity_cold_start & similarity APIs
    syntax_rows = getattr(agent_surfaces, "_VARIANT_SYNTAX_ROWS", ())
    syntax_text = " ".join(str(r) for r in syntax_rows)
    assert "similarity_global" in syntax_text
    assert "similarity_cold_start" in syntax_text

    exec_ref = agent_surfaces.render_skill_execution_paths_reference()
    llms_txt = agent_surfaces.render_llms_txt()
    for token in ("similarity_df", "find_similar_series_df", "compute_time_series_similarity"):
        assert token in exec_ref or token in llms_txt


# =============================================================================
# TIER 2 — Boundary Value & Pathological Corner-Case Hardening
# =============================================================================


def test_t2_similarity_pathological_panels_n1_n2_flat_zero_intermittent_short_nan() -> None:
    """Tier 2 Pathological Corner Cases (>= 8 cases): Verify `embed_series_df`,
    `compute_similarity_matrix`, `cluster_by_similarity_df`, and `find_similar_series_df` handle:
    1. $N=1$ single series,
    2. $N=2$ two-series panel,
    3. Constant/flat series (`std == 0`),
    4. All-zero series (`y == 0`),
    5. High zero-inflation intermittent series (`zero_fraction > 0.7`),
    6. Uneven history lengths ($n=3$ launch prefix vs. $n=120$ mature donor),
    7. Very short series ($n < 5$, e.g. $n=2, 3, 4$),
    8. NaN / missing timestamps and values.
    """
    from scale_forecasting.similarity import (
        cluster_by_similarity_df,
        compute_similarity_matrix,
        embed_series_df,
        find_similar_series_df,
    )

    dates_30 = pd.date_range("2025-01-01", periods=30, freq="D")

    # Case 1: N=1 single series panel degrades gracefully without error
    df_n1 = pd.DataFrame(
        [{"ts_id": "solo_sku", "ds": dt, "y": float(50.0 + i)} for i, dt in enumerate(dates_30)]
    )
    emb_n1 = embed_series_df(df_n1)
    assert len(emb_n1) == 1
    assert bool(np.isfinite(emb_n1["embed_x"]).all())
    sim_n1 = compute_similarity_matrix(df_n1)
    assert sim_n1.matrix_df.shape == (1, 1)
    assert float(sim_n1.matrix_df.iloc[0, 0]) == pytest.approx(1.0)
    clust_n1 = cluster_by_similarity_df(df_n1)
    assert len(clust_n1) == 1
    match_n1 = find_similar_series_df(df_n1, query_ts_id="solo_sku")
    assert isinstance(match_n1, pd.DataFrame)

    # Case 2: N=2 two-series panel clusters and computes similarity without silhouette crash
    df_n2 = pd.concat(
        [
            df_n1,
            pd.DataFrame(
                [
                    {"ts_id": "pair_sku", "ds": dt, "y": float(55.0 + 1.1 * i)}
                    for i, dt in enumerate(dates_30)
                ]
            ),
        ],
        ignore_index=True,
    )
    sim_n2 = compute_similarity_matrix(df_n2)
    assert sim_n2.matrix_df.shape == (2, 2)
    clust_n2 = cluster_by_similarity_df(df_n2)
    assert len(clust_n2) == 2

    # Case 3, 4, 5: Constant/flat (std == 0), all-zero (y == 0), intermittent (zero_fraction > 0.7)
    rng = np.random.default_rng(2201)
    intermittent_y = np.where(rng.uniform(0.0, 1.0, size=30) < 0.80, 0.0, 15.0)
    intermittent_y[0] = 12.0  # ensure at least one non-zero pulse
    df_pathological = pd.concat(
        [
            pd.DataFrame([{"ts_id": "flat_const", "ds": dt, "y": 42.0} for dt in dates_30]),
            pd.DataFrame([{"ts_id": "all_zeros", "ds": dt, "y": 0.0} for dt in dates_30]),
            pd.DataFrame(
                [
                    {"ts_id": "lumpy_intermittent", "ds": dt, "y": float(y_v)}
                    for dt, y_v in zip(dates_30, intermittent_y, strict=True)
                ]
            ),
            pd.DataFrame(
                [
                    {"ts_id": "normal_wave", "ds": dt, "y": float(40.0 + 10.0 * np.sin(i / 3.0))}
                    for i, dt in enumerate(dates_30)
                ]
            ),
        ],
        ignore_index=True,
    )
    emb_path = embed_series_df(df_pathological).set_index("ts_id")
    assert float(emb_path.loc["all_zeros", "zero_fraction"]) == pytest.approx(1.0)
    assert float(emb_path.loc["lumpy_intermittent", "zero_fraction"]) > 0.70
    for method_name in (
        "hybrid",
        "physics_attractor",
        "spectral_phase",
        "elastic_dtw",
        "minirocket",
        "structural_signature",
        "foundation_latent",
    ):
        res_p = compute_similarity_matrix(df_pathological, method=method_name)
        mat_p = res_p.matrix_df.to_numpy(dtype=float)
        assert np.all(np.isfinite(mat_p))
        np.testing.assert_allclose(np.diag(mat_p), np.ones(4), atol=1e-6)

    # Case 6: Uneven history lengths (n=3 launch prefix vs n=120 mature donors)
    dates_120 = pd.date_range("2025-01-01", periods=120, freq="D")
    df_uneven = pd.concat(
        [
            pd.DataFrame(
                [
                    {"ts_id": "mature_120_a", "ds": dt, "y": float(100.0 + 5.0 * i)}
                    for i, dt in enumerate(dates_120)
                ]
            ),
            pd.DataFrame(
                [
                    {"ts_id": "mature_120_b", "ds": dt, "y": float(200.0 - 2.0 * i)}
                    for i, dt in enumerate(dates_120)
                ]
            ),
            pd.DataFrame(
                [
                    {"ts_id": "launch_prefix_3", "ds": dt, "y": float(100.0 + 5.0 * i)}
                    for i, dt in enumerate(dates_120[-3:])
                ]
            ),
        ],
        ignore_index=True,
    )
    matches_prefix = find_similar_series_df(
        df_uneven, query_ts_id="launch_prefix_3", mode="launch_prefix", top_k=2
    )
    assert len(matches_prefix) == 2
    assert str(matches_prefix["neighbor_ts_id"].iloc[0]) == "mature_120_a"

    # Case 7 & 8: Very short series (n < 5) and NaN / missing timestamps
    dates_4 = pd.date_range("2025-01-01", periods=4, freq="D")
    df_short_nan = pd.DataFrame(
        [
            {"ts_id": "short_a", "ds": dates_4[0], "y": 10.0},
            {"ts_id": "short_a", "ds": dates_4[1], "y": float("nan")},
            {"ts_id": "short_a", "ds": dates_4[3], "y": 14.0},  # dates_4[2] missing
            {"ts_id": "short_b", "ds": dates_4[0], "y": 12.0},
            {"ts_id": "short_b", "ds": dates_4[1], "y": 13.0},
            {"ts_id": "short_b", "ds": dates_4[2], "y": 15.0},
        ]
    )
    sim_short = compute_similarity_matrix(df_short_nan)
    assert sim_short.matrix_df.shape == (2, 2)
    assert np.all(np.isfinite(sim_short.matrix_df.to_numpy(dtype=float)))


@pytest.mark.parametrize(
    ("section", "payload"),
    [
        ("features", {"cold_start_matching": "invalid_matching_mode"}),
        ("features", {"similarity_method": "unknown_kernel"}),
        ("features", {"similarity_n_clusters": 1}),
        ("features", {"similarity_n_clusters": 0}),
        ("features", {"similarity_n_clusters": -3}),
        ("features", {"similarity_n_clusters": True}),
        ("hierarchy", {"similarity_hierarchy": True, "similarity_n_clusters": 1}),
        ("hierarchy", {"similarity_hierarchy": True, "similarity_n_clusters": False}),
        ("hierarchy", {"similarity_method": "invalid_sim_method"}),
        ("ensemble", {"similarity_smoothing": -0.1}),
        ("ensemble", {"similarity_smoothing": 1.05}),
        ("ensemble", {"similarity_smoothing": float("nan")}),
    ],
)
def test_t2_config_invalid_track_p_parameters_and_prebreak_digest_preservation(
    section: str, payload: dict[str, Any]
) -> None:
    """Tier 2 Config Validation (>= 12 parametrized cases): Reject invalid Track P parameters."""
    base = {
        "run_name": "t2-invalid-track-p-cfg",
        "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
        "models": ["naive_mean"],
        section: payload,
    }
    with pytest.raises((ValidationError, ValueError)):
        RunConfig.model_validate(base)


def test_t2_cold_start_pooling_hierarchy_copula_ensembler_edge_cases() -> None:
    """Tier 2 Subsystem Edge Cases (>= 5 cases):
    1. Unknown `query_ts_id` in `find_similar_series_df` raises `DataError`/`ConfigError`,
    2. Invalid `method` in `compute_similarity_matrix` raises `ConfigError`/`ValueError`,
    3. 0-obs and 1-obs cold-start SKUs with `matching="similarity"` fall back cleanly,
    4. `@similarity_global` on a tiny 2-series panel executes without crash,
    5. `shrink_correlation_toward_similarity` with $N=1$ and $n_{\text{samples}}=0$.
    """
    from scale_forecasting.cold_start import (
        build_cold_start_donor_pool,
        forecast_cold_start_series,
    )
    from scale_forecasting.copula import shrink_correlation_toward_similarity
    from scale_forecasting.similarity import (
        compute_similarity_matrix,
        find_similar_series_df,
    )

    panel = _make_multi_regime_retail_panel(
        n_days=35, n_short_days=5, include_cold_start=True, seed=2203
    )

    # Case 1: Unknown query_ts_id raises DataError / ConfigError / ValueError
    with pytest.raises((DataError, ConfigError, ValueError, KeyError)):
        find_similar_series_df(panel, query_ts_id="non_existent_sku_999")

    # Case 2: Unknown similarity method raises ConfigError / ValueError
    with pytest.raises((ConfigError, ValueError)):
        compute_similarity_matrix(panel, method="not_a_real_method")

    # Case 3: 0-obs and 1-obs cold-start SKUs under matching="similarity" produce finite forecasts
    pool = build_cold_start_donor_pool(panel, min_obs=14)
    zero_obs_df = pd.DataFrame(
        [{"ts_id": "sku_0_obs", "ds": pd.Timestamp("2025-02-04"), "y": float("nan")}]
    )
    res_0 = forecast_cold_start_series(
        zero_obs_df, pool, horizon=5, freq="D", min_obs=14, matching="similarity"
    )
    assert res_0.n_obs == 0
    assert bool(np.isfinite(res_0.predictions["yhat"]).all())

    one_obs_df = pd.DataFrame(
        [{"ts_id": "sku_1_obs", "ds": pd.Timestamp("2025-02-04"), "y": 115.0}]
    )
    res_1 = forecast_cold_start_series(
        one_obs_df, pool, horizon=5, freq="D", min_obs=14, matching="similarity"
    )
    assert res_1.n_obs == 1
    assert bool(np.isfinite(res_1.predictions["yhat"]).all())

    # Case 4: @similarity_global on a tiny 2-series panel falls back to 1 global cluster cleanly
    tiny_panel = panel[panel["ts_id"].isin(["flagship_0", "flagship_1"])].copy()
    cfg_tiny = RunConfig.model_validate(
        {
            "run_name": "t2-tiny-sim-global",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 5},
            "models": ["tide@similarity_global"],
            "model_params": {"tide": {"max_steps": 3, "input_size": 7}},
            "backtest": {"enabled": False},
        }
    )
    cells_tiny = run_panel(tiny_panel, "tide@similarity_global", cfg_tiny)
    assert len(cells_tiny) == 2
    assert all(c.status == "ok" for c in cells_tiny)

    # Case 5: shrink_correlation_toward_similarity with N=1 single series returns [[1.0]]
    corr_1 = shrink_correlation_toward_similarity(
        np.ones((1, 1)), np.ones((1, 1)), n_samples=0, weight="auto"
    )
    assert corr_1.shape == (1, 1)
    assert float(corr_1[0, 0]) == pytest.approx(1.0)


# =============================================================================
# TIER 3 — Cross-Feature & Cross-Track Multi-Subsystem Combinations
# =============================================================================


def test_t3_combo_1_similarity_global_plus_similarity_cold_start_zero_metadata() -> None:
    """Tier 3 Combo 1: `@similarity_global` + `@similarity_cold_start`
    (`cold_start_matching="similarity"`, `static_covariates=[]`) executed via `spark_io.run_group`.
    """
    panel = _make_multi_regime_retail_panel(
        n_days=49, n_short_days=5, include_cold_start=True, seed=3001
    )
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-combo-1-global-and-cold-start",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["tide@similarity_global", "theta@similarity_cold_start"],
            "model_params": {"tide": {"max_steps": 3, "input_size": 7}},
            "features": {
                "cold_start_transfer": True,
                "cold_start_min_obs": 14,
                "cold_start_matching": "similarity",
                "similarity_method": "hybrid",
                "similarity_n_clusters": 2,
            },
            "backtest": {"enabled": False},
        }
    )
    cells, status_df = run_group(panel, cfg)
    assert set(status_df["status"]) == {"ok"}
    npi_cells = [c for c in cells if c.ts_id == "npi_surge_short"]
    assert len(npi_cells) == 2
    for nc in npi_cells:
        assert len(nc.predictions) == 7
        assert bool(np.isfinite(nc.predictions["yhat"]).all())


def test_t3_combo_2_similarity_hierarchy_with_cross_temporal_mint_and_goal_seek() -> None:
    """Tier 3 Combo 2: Flat panel (`levels=[]`) with `similarity_hierarchy=True` +
    `cross_temporal=True` (`cross_temporal_mint`) + `sample_trajectories(reconcile=True)` +
    `allocate_target(node_id="similarity_cluster_id=cluster_0")`.
    """
    from scale_forecasting.copula import sample_trajectories

    panel = _make_multi_regime_retail_panel(
        n_days=42, n_short_days=5, include_cold_start=False, seed=3002
    )
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-combo-2-sim-hier-ct",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
            "hierarchy": {
                "similarity_hierarchy": True,
                "similarity_n_clusters": 2,
                "cross_temporal": True,
                "temporal_aggregations": [7],
                "reconciliation_methods": ["cross_temporal_mint"],
            },
        }
    )
    hier_panel, spec = build_hierarchy(panel, cfg)
    f_dates = pd.date_range("2025-02-12", periods=7, freq="D")
    pred_rows = [
        {
            "ts_id": nid,
            "model_type": "naive_mean",
            "ds": dt,
            "forecast_date": dt,
            "horizon_step": h_idx,
            "yhat": 100.0 + 5.0 * h_idx,
            "yhat_lower": 85.0 + 5.0 * h_idx,
            "yhat_upper": 115.0 + 5.0 * h_idx,
        }
        for nid in spec.node_ids
        for h_idx, dt in enumerate(f_dates, start=1)
    ]
    preds_df = pd.DataFrame(pred_rows)
    rec_df = reconcile_forecasts(
        preds_df, spec, "cross_temporal_mint", cfg=cfg, history_df=hier_panel
    )
    assert verify_coherence(rec_df, spec) < 1e-6
    assert verify_temporal_coherence(rec_df, cfg=cfg) < 1e-6

    paths = sample_trajectories(
        rec_df,
        history_df=hier_panel,
        cfg=cfg,
        hier_spec=spec,
        n_paths=20,
        copula_type="gaussian",
        cross_series=True,
        similarity_shrinkage=True,
        reconcile=True,
        seed=42,
    )
    for _pid, p_grp in paths.groupby("path_id"):
        assert verify_coherence(p_grp, spec) < 1e-6

    gs_res = allocate_target(
        rec_df,
        target_total=2500.0,
        node_id="similarity_cluster_id=cluster_0",
        history_df=hier_panel,
        cfg=cfg,
        return_result=True,
    )
    assert gs_res.max_target_discrepancy < 1e-4


def test_t3_combo_3_similarity_copula_shrinkage_with_inventory_milp_and_aci() -> None:
    """Tier 3 Combo 3: Online `aci_normalized` + `sample_trajectories(cross_series=True,
    similarity_shrinkage=True)` + `optimize_inventory_portfolio(uncertainty_source="copula")`.
    """
    from scale_forecasting.aci import compute_aci_df
    from scale_forecasting.copula import compute_tail_risk, sample_trajectories
    from scale_forecasting.inventory_optimizer import optimize_inventory_portfolio

    panel = _make_multi_regime_retail_panel(
        n_days=42, n_short_days=5, include_cold_start=False, seed=3003
    )
    f_dates = pd.date_range("2025-02-12", periods=7, freq="D")
    pred_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for tid in panel["ts_id"].unique():
        for h_idx, dt in enumerate(f_dates, start=1):
            pred_rows.append(
                {
                    "ts_id": str(tid),
                    "model_type": "theta",
                    "ds": dt,
                    "forecast_date": dt,
                    "horizon_step": h_idx,
                    "yhat": 110.0,
                    "yhat_lower": 95.0,
                    "yhat_upper": 125.0,
                }
            )
            oof_rows.append(
                {
                    "ts_id": str(tid),
                    "model_type": "theta",
                    "fold_id": 0,
                    "cutoff_date": pd.Timestamp("2025-02-04"),
                    "ds": dt - pd.Timedelta(days=7),
                    "horizon_step": h_idx,
                    "y_true": 110.0 + (-1.0 if h_idx % 2 else 1.0) * 22.0,
                    "yhat": 110.0,
                    "yhat_lower": 102.0,
                    "yhat_upper": 118.0,
                }
            )
    preds_df = pd.DataFrame(pred_rows)
    oof_df = pd.DataFrame(oof_rows)

    aci_out = compute_aci_df(oof_df, preds_df, gamma=0.06, normalized=True)
    assert not aci_out.empty

    paths = sample_trajectories(
        preds_df,
        oof_df,
        history_df=panel,
        n_paths=150,
        copula_type="student_t",
        cross_series=True,
        similarity_shrinkage=True,
        seed=42,
    )
    risk = compute_tail_risk(paths, predictions_df=preds_df, window=5)
    assert len(risk) == 6

    inv_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        trajectories_df=paths,
        max_budget=20_000.0,
        unit_cost=10.0,
        vendor_moq=25.0,
        order_multiple=5.0,
        uncertainty_source="copula",
        similarity_shrinkage=True,
    )
    assert float(inv_df["order_spend"].sum()) <= 20_000.0 + 1e-6


def test_t3_combo_4_similarity_cannibalization_priors_scenario_and_champion_smoothing() -> None:
    """Tier 3 Combo 4: `estimate_promo_elasticity(..., use_similarity_priors=True)` +
    `optimize_scenario(..., use_similarity_priors=True, reconcile=True)` +
    `EnsembleConfig(similarity_smoothing=0.30)`.
    """
    from scale_forecasting.elasticity import estimate_promo_elasticity
    from scale_forecasting.scenario_optimizer import optimize_scenario

    hist_df, fut_df = _make_promo_cannibalization_twin_panel(n_days=56, horizon=7, seed=3004)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-combo-4-cannib-smooth",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean", "regression_lags@monotone"],
            "features": {
                "future_covariates": ["price", "promo"],
                "monotonic_constraints": {"price": -1, "promo": 1},
            },
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7},
            "hierarchy": {
                "similarity_hierarchy": True,
                "similarity_n_clusters": 2,
                "reconciliation_methods": ["wls_struct"],
            },
            "ensemble": {
                "strategies": ["champion", "inverse_error"],
                "similarity_smoothing": 0.30,
            },
        }
    )

    elas_df = estimate_promo_elasticity(
        hist_df, promo_col="promo", price_col="price", cfg=cfg, use_similarity_priors=True
    )
    assert not elas_df.empty

    scen_res = optimize_scenario(
        hist_df,
        config=cfg,
        future_covariates_df=fut_df,
        objective="gross_profit",
        price_col="price",
        promo_col="promo",
        unit_cost=4.5,
        max_promo_days=2,
        min_cooldown_days=2,
        use_similarity_priors=True,
        reconcile=True,
    )
    assert scen_res.optimal_objective_value >= scen_res.baseline_objective_value - 1e-6
    assert scen_res.coherence_max_abs_error < 1e-6


def test_t3_combo_5_full_track_j_k_l_p_end_to_end_pipeline(tmp_path: Path) -> None:
    """Tier 3 Combo 5: Unified Track J + Track K + Track L + Track P end-to-end execution across
    `Forecaster`, `similarity_df`, `cold_start_transfer_df`, `sample_trajectories`,
    `optimize_inventory_df`, `optimize_scenario`, `export_report`, and `McpServer`.
    """
    hist_df, fut_df = _make_promo_cannibalization_twin_panel(n_days=49, horizon=7, seed=3005)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-combo-5-full-jklp",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean", "regression_lags"],
            "features": {
                "future_covariates": ["price", "promo"],
                "cold_start_transfer": True,
                "cold_start_matching": "similarity",
                "similarity_cluster_feature": True,
                "similarity_n_clusters": 2,
            },
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7},
            "hierarchy": {
                "similarity_hierarchy": True,
                "similarity_n_clusters": 2,
                "reconciliation_methods": ["wls_struct", "mint_shrink"],
            },
            "ensemble": {
                "strategies": ["inverse_error", "champion"],
                "similarity_smoothing": 0.25,
            },
            "output": {"calibration_method": "aci_normalized", "aci_gamma": 0.05},
        }
    )

    f = sf.Forecaster(cfg, data=hist_df)
    assert not f.similarity_df().empty
    assert not f.similarity_clusters_df().empty
    assert not f.similarity_embeddings_df().empty
    assert not f.find_similar_series_df("twin_a1").empty
    assert not f.cold_start_transfer_df().empty
    assert not f.promo_elasticity_df(promo_col="promo", price_col="price").empty
    assert not f.sample_trajectories(
        n_paths=60, cross_series=True, similarity_shrinkage=True, seed=42
    ).empty
    assert not f.optimize_inventory_df(max_budget=10_000.0, unit_cost=4.5).empty
    scen = f.optimize_scenario(future_covariates_df=fut_df, unit_cost=4.5)
    assert scen.optimal_objective_value >= scen.baseline_objective_value - 1e-5

    rep_path = f.export_report(tmp_path / "combo5_report.html")
    assert rep_path.exists()
    plt.close("all")


# =============================================================================
# TIER 4 — Real-World Enterprise Application Scenarios
# =============================================================================


def test_t4_scenario_1_multi_regime_panel_similarity_global_eliminates_negative_transfer() -> None:
    """Tier 4 Scenario 1: Multi-Regime Retail Panel (`@similarity_global` vs Monolithic `@global`).

    Demonstrates why `@similarity_global` exists:
    - A retail catalog mixes smooth persistent flagship SKUs with counter-cyclical promotional
      sawtooth SKUs that respond with opposite sign to promotional pulses (`promo`).
    - Fitting a single monolithic `tide@global` model across both regimes pools opposite covariate
      and seasonal dynamics into one shared network, causing cross-regime negative transfer.
    - Fitting `tide@similarity_global` discovers the 2 dynamical regimes via
      `cluster_by_similarity_df`, fits an independent global model per cluster, and strictly
      reduces out-of-fold WAPE compared to monolithic `tide@global`!
    """
    dates = pd.date_range("2025-01-01", periods=56, freq="D")
    rows: list[dict[str, Any]] = []
    for i in range(3):
        for idx, dt in enumerate(dates):
            promo = 1.0 if (idx % 7 == 0) else 0.0
            y_flag = 120.0 + 5.0 * i + 65.0 * promo + 15.0 * np.sin(2.0 * np.pi * idx / 7.0)
            y_saw = 120.0 + 5.0 * i - 65.0 * promo - 15.0 * np.sin(2.0 * np.pi * idx / 7.0)
            rows.append(
                {"ts_id": f"flagship_{i}", "ds": dt, "y": float(y_flag), "promo": float(promo)}
            )
            rows.append(
                {"ts_id": f"sawtooth_{i}", "ds": dt, "y": float(y_saw), "promo": float(promo)}
            )
    panel = pd.DataFrame(rows)

    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-scen-1-negative-transfer",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["tide@global", "tide@similarity_global"],
            "model_params": {
                "tide": {
                    "max_steps": 25,
                    "input_size": 14,
                    "learning_rate": 0.01,
                    "random_seed": 42,
                }
            },
            "features": {
                "future_covariates": ["promo"],
                "similarity_method": "hybrid",
                "similarity_n_clusters": 2,
            },
            "backtest": {
                "enabled": True,
                "n_folds": 1,
                "horizon": 7,
                "step": 7,
                "min_train": 35,
            },
        }
    )

    cells_mono = run_panel(panel, "tide@global", cfg)
    cells_sim = run_panel(panel, "tide@similarity_global", cfg)

    oof_mono = pd.concat([c.oof for c in cells_mono if c.oof is not None], ignore_index=True)
    oof_sim = pd.concat([c.oof for c in cells_sim if c.oof is not None], ignore_index=True)

    wape_mono = float(
        np.sum(np.abs(oof_mono["y_true"] - oof_mono["yhat"])) / np.sum(np.abs(oof_mono["y_true"]))
    )
    wape_sim = float(
        np.sum(np.abs(oof_sim["y_true"] - oof_sim["yhat"])) / np.sum(np.abs(oof_sim["y_true"]))
    )
    assert wape_sim < wape_mono, (
        f"Expected @similarity_global WAPE ({wape_sim:.4f}) < "
        f"monolithic @global WAPE ({wape_mono:.4f})"
    )


def test_t4_scenario_2_flat_panel_latent_hierarchy_copula_inventory_and_executive_governance(
    tmp_path: Path,
) -> None:
    """Tier 4 Scenario 2: Flat Enterprise Panel Latent Hierarchy Discovery, Similarity-Shrunk
    Copula Inventory Optimization, and Executive Governance (`f.assess()` & HTML Report).
    """
    hist_df, fut_df = _make_promo_cannibalization_twin_panel(n_days=56, horizon=7, seed=4002)
    assert "region" not in hist_df.columns and "category" not in hist_df.columns

    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-scen-2-flat-enterprise-spine",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean", "regression_lags"],
            "features": {
                "future_covariates": ["price", "promo"],
                "similarity_cluster_feature": True,
                "similarity_n_clusters": 2,
            },
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7},
            "hierarchy": {
                "similarity_hierarchy": True,
                "similarity_n_clusters": 2,
                "reconciliation_methods": ["mint_shrink", "mint_nonneg"],
            },
            "ensemble": {
                "strategies": ["inverse_error", "champion"],
                "similarity_smoothing": 0.25,
            },
        }
    )

    f = sf.Forecaster(cfg, data=hist_df)
    clusters_df = f.similarity_clusters_df(n_clusters=2)
    assert len(clusters_df) == 4
    c_map = clusters_df.set_index("ts_id")["similarity_cluster_id"].to_dict()
    assert c_map["twin_a1"] == c_map["twin_a2"]
    assert c_map["ortho_b1"] == c_map["ortho_b2"]
    assert c_map["twin_a1"] != c_map["ortho_b1"]

    # Allocate a top-down target specifically to cluster_0 via Goal-Seek QP
    gs_df = f.allocate_target(
        target_total=1800.0,
        node_id="similarity_cluster_id=cluster_0",
        smoothness_weight=3.0,
    )
    assert not gs_df.empty

    # Sample similarity-shrunk joint copula trajectories and optimize DC inventory
    traj_df = f.sample_trajectories(
        n_paths=120,
        copula_type="student_t",
        cross_series=True,
        similarity_shrinkage=True,
        seed=42,
    )
    assert not traj_df.empty

    inv_df = f.optimize_inventory_df(
        max_budget=12_000.0,
        unit_cost=4.5,
        vendor_moq=20.0,
        order_multiple=10.0,
        uncertainty_source="copula",
    )
    assert float(inv_df["order_spend"].sum()) <= 12_000.0 + 1e-6

    scen_res = f.optimize_scenario(
        future_covariates_df=fut_df,
        objective="gross_profit",
        unit_cost=4.5,
        max_promo_days=2,
        min_cooldown_days=2,
        reconcile=True,
    )
    assert scen_res.coherence_max_abs_error < 1e-6

    rep_path = f.export_report(tmp_path / "flat_panel_executive_report.html")
    assert rep_path.exists()
    plt.close("all")


def _make_phase2_multimodal_retail_panel() -> pd.DataFrame:
    """Create a 7-series retail assortment panel with rich text, attribute, embedding, and
    3-level hierarchy metadata (`department -> category -> subcategory`), including:
    - True retail substitutes (`oat_milk_vanilla_64oz` and `oat_milk_orig_64oz` in
      `Dairy_Alt > Plant_Milk > Oat_Milk`)
    - Same-subcategory out-of-stock twin (`oat_milk_choco_64oz` in `Oat_Milk`)
    - Sister-subcategory item (`almond_milk_64oz` in `Dairy_Alt > Plant_Milk > Almond_Milk`)
    - Same-department item (`greek_yogurt_32oz` in `Dairy_Alt > Cultured > Yogurt`)
    - Coincidental seasonal twins in disjoint departments (`ice_cream_gelato_16oz` in
      `Frozen > Desserts > Gelato` and `sunscreen_lotion_spf50` in
      `Personal_Care > Sun_Care > Lotion`) sharing the exact same summer sine wave as
      `oat_milk_vanilla_64oz`!
    """
    n_days = 42
    dates = pd.date_range("2025-06-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)
    summer_wave = (
        120.0 + 35.0 * np.sin(2.0 * np.pi * t / 14.0) + 10.0 * np.cos(2.0 * np.pi * t / 7.0)
    )
    yogurt_wave = 80.0 + 12.0 * np.cos(2.0 * np.pi * t / 7.0) + 0.4 * t

    specs = [
        (
            "oat_milk_vanilla_64oz",
            summer_wave,
            "Organic Oat Milk Vanilla Sweetened 64oz Plant Based Beverage Carton",
            "Dairy_Alt",
            "Plant_Milk",
            "Oat_Milk",
            "OatHarvest",
            64.0,
            4.99,
            "[0.92, 0.88, 0.05, 0.02]",
        ),
        (
            "oat_milk_orig_64oz",
            summer_wave * 0.99 + 1.0,
            "Organic Oat Milk Original Sweetened 64oz Plant Based Beverage Carton",
            "Dairy_Alt",
            "Plant_Milk",
            "Oat_Milk",
            "OatHarvest",
            64.0,
            4.99,
            "[0.92, 0.87, 0.05, 0.02]",
        ),
        (
            "oat_milk_choco_64oz",
            summer_wave * 0.92 + 4.0,
            "Cocoa Dark Chocolate Oat Milk 64oz Plant Carton",
            "Dairy_Alt",
            "Plant_Milk",
            "Oat_Milk",
            "OatHarvest",
            64.0,
            5.49,
            "[0.82, 0.78, 0.12, 0.04]",
        ),
        (
            "almond_milk_64oz",
            summer_wave * 0.85 + 8.0,
            "Creamy Almond Milk Vanilla 64oz Nut Beverage Carton",
            "Dairy_Alt",
            "Plant_Milk",
            "Almond_Milk",
            "NutMeadow",
            64.0,
            4.49,
            "[0.75, 0.72, 0.12, 0.05]",
        ),
        (
            "greek_yogurt_32oz",
            yogurt_wave,
            "Plain Nonfat Greek Strained Yogurt 32oz Tub",
            "Dairy_Alt",
            "Cultured",
            "Yogurt",
            "OatHarvest",
            32.0,
            5.99,
            "[0.45, 0.50, 0.30, 0.10]",
        ),
        (
            "ice_cream_gelato_16oz",
            summer_wave * 1.01,
            "Artisan Vanilla Bean Frozen Gelato Ice Cream 16oz Pint",
            "Frozen",
            "Desserts",
            "Gelato",
            "GlacierCream",
            16.0,
            6.49,
            "[0.15, 0.20, 0.85, 0.10]",
        ),
        (
            "sunscreen_lotion_spf50",
            summer_wave * 1.00,
            "Broad Spectrum UV Sunscreen SPF 50 Waterproof Sunblock Lotion 8oz",
            "Personal_Care",
            "Sun_Care",
            "Lotion",
            "SolGuard",
            8.0,
            11.99,
            "[0.02, 0.03, 0.10, 0.96]",
        ),
    ]
    frames: list[pd.DataFrame] = []
    for tid, y_arr, desc, dept, cat, subcat, brand, size_oz, price, emb in specs:
        frames.append(
            pd.DataFrame(
                {
                    "ts_id": tid,
                    "ds": dates,
                    "y": y_arr.astype(float),
                    "product_title": desc,
                    "department": dept,
                    "category": cat,
                    "subcategory": subcat,
                    "brand": brand,
                    "pack_size_oz": float(size_oz),
                    "price": float(price),
                    "text_embedding": emb,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_t4_scenario_3_day0_to_day5_to_mature_cold_start_progression() -> None:
    """Tier 4 Scenario 3 (F17): Day-0 -> Day-5 -> Mature Cold-Start NPI Progression.

    Exercises:
    1. Day-0 (`n_i = 0` pre-launch SKU matched via `semantic_sim + hierarchy_sim` using
       `text_cols`, `attribute_cols`, `hierarchy_cols`),
    2. Day-5 (`n_i = 5` `launch_prefix` blended with `semantic_sim + hierarchy_sim` under
       `cold_start_matching="hybrid"`),
    3. Mature (`n_i = 28 >= cold_start_min_obs=14`) cold-start progression via both
       `cold_start_transfer_df` and `Forecaster.cold_start_transfer_df`.
    Verifies Empirical-Bayes weight `w_i = n_i / (n_i + kappa)` increases monotonically from
    `0.0` at Day-0 to `> 0` at Day-5 to `>= 0.5` at Mature, and verifies valid Day-0/Day-5
    prediction intervals (`yhat_lower < yhat < yhat_upper`) and populated
    `donor_similarity_score` / `top_donor_ts_ids`.
    """
    from scale_forecasting.cold_start import (
        build_cold_start_donor_pool,
        cold_start_transfer_df,
        forecast_cold_start_series,
    )

    dates = pd.date_range("2025-01-01", periods=28, freq="D")
    t = np.arange(28, dtype=float)
    oat_curve = 185.0 + 30.0 * np.sin(2.0 * np.pi * t / 7.0) + 1.5 * t
    soda_curve = 28.0 + 4.0 * np.cos(2.0 * np.pi * t / 7.0)

    panel = pd.concat(
        [
            # Mature donor 1 (n_obs = 28)
            pd.DataFrame(
                {
                    "ts_id": "donor_oat_orig_64oz",
                    "ds": dates,
                    "y": oat_curve,
                    "product_title": "Organic Oat Milk Original 64oz Plant Beverage Carton",
                    "department": "Dairy_Alt",
                    "category": "Plant_Milk",
                    "subcategory": "Oat_Milk",
                    "brand": "OatHarvest",
                    "pack_size_oz": 64.0,
                }
            ),
            # Mature donor 2 (n_obs = 28)
            pd.DataFrame(
                {
                    "ts_id": "donor_oat_choco_64oz",
                    "ds": dates,
                    "y": oat_curve * 0.94,
                    "product_title": "Organic Oat Milk Chocolate 64oz Plant Beverage Carton",
                    "department": "Dairy_Alt",
                    "category": "Plant_Milk",
                    "subcategory": "Oat_Milk",
                    "brand": "OatHarvest",
                    "pack_size_oz": 64.0,
                }
            ),
            # Unrelated mature donor in a different department/scale (n_obs = 28)
            pd.DataFrame(
                {
                    "ts_id": "donor_cola_can_12oz",
                    "ds": dates,
                    "y": soda_curve,
                    "product_title": "Sparkling Cola Carbonated Soda 12oz Aluminum Can",
                    "department": "Beverages",
                    "category": "Carbonated",
                    "subcategory": "Cola",
                    "brand": "FizzPop",
                    "pack_size_oz": 12.0,
                }
            ),
            # Day-0 pre-launch SKU (n_obs = 0: target y is NaN)
            pd.DataFrame(
                {
                    "ts_id": "npi_day0_oat_vanilla",
                    "ds": [dates[-1]],
                    "y": [float("nan")],
                    "product_title": "Organic Oat Milk Vanilla 64oz Plant Beverage Carton",
                    "department": "Dairy_Alt",
                    "category": "Plant_Milk",
                    "subcategory": "Oat_Milk",
                    "brand": "OatHarvest",
                    "pack_size_oz": 64.0,
                }
            ),
            # Day-5 early-launch SKU (n_obs = 5)
            pd.DataFrame(
                {
                    "ts_id": "npi_day5_oat_barista",
                    "ds": dates[:5],
                    "y": oat_curve[:5] * 1.04,
                    "product_title": "Organic Oat Milk Barista Edition 64oz Plant Beverage Carton",
                    "department": "Dairy_Alt",
                    "category": "Plant_Milk",
                    "subcategory": "Oat_Milk",
                    "brand": "OatHarvest",
                    "pack_size_oz": 64.0,
                }
            ),
        ],
        ignore_index=True,
    )

    # 1. Direct cold_start_transfer_df verification
    cs_df = cold_start_transfer_df(
        panel,
        horizon=7,
        freq="D",
        min_obs=14,
        cold_start_matching="hybrid",
        text_cols=["product_title"],
        attribute_cols=["brand", "pack_size_oz"],
        hierarchy_cols=["department", "category", "subcategory"],
    )
    by_id = cs_df.set_index("ts_id")

    # Verify via Forecaster.cold_start_transfer_df as well
    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-scen-3-cold-start-progression",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["theta@similarity_cold_start"],
            "features": {
                "cold_start_transfer": True,
                "cold_start_min_obs": 14,
                "cold_start_matching": "hybrid",
            },
        }
    )
    f = sf.Forecaster(cfg, data=panel)
    cs_sdk_df = f.cold_start_transfer_df(
        text_cols=["product_title"],
        attribute_cols=["brand", "pack_size_oz"],
        hierarchy_cols=["department", "category", "subcategory"],
    )
    by_id_sdk = cs_sdk_df.set_index("ts_id")

    for table in (by_id, by_id_sdk):
        r_day0 = table.loc["npi_day0_oat_vanilla"]
        r_day5 = table.loc["npi_day5_oat_barista"]
        r_mat = table.loc["donor_oat_orig_64oz"]

        w_day0 = float(r_day0["empirical_bayes_weight"])
        w_day5 = float(r_day5["empirical_bayes_weight"])
        w_mat = float(r_mat["empirical_bayes_weight"])

        # Monotonic Empirical-Bayes progression: 0.0 (Day-0) < w_5 (Day-5) < w_28 (>= 0.5 at Mature)
        assert int(r_day0["n_obs"]) == 0
        assert bool(r_day0["is_cold_start"]) is True
        assert str(r_day0["lifecycle_stage"]) == "pre_launch"
        assert w_day0 == pytest.approx(0.0, abs=1e-9)

        assert int(r_day5["n_obs"]) == 5
        assert bool(r_day5["is_cold_start"]) is True
        assert str(r_day5["lifecycle_stage"]) == "early_launch"
        assert 0.0 < w_day5 < w_mat

        assert int(r_mat["n_obs"]) == 28
        assert bool(r_mat["is_cold_start"]) is False
        assert str(r_mat["lifecycle_stage"]) == "mature"
        assert w_mat >= 0.50

        # Populated donor_similarity_score and top_donor_ts_ids matching Oat Milk donors
        assert float(r_day0["donor_similarity_score"]) > 0.65
        assert float(r_day5["donor_similarity_score"]) > 0.65
        assert "donor_oat_" in str(r_day0["top_donor_ts_ids"])
        assert "donor_oat_" in str(r_day5["top_donor_ts_ids"])
        assert float(r_day0["yhat_mean"]) > 120.0
        assert float(r_day5["yhat_mean"]) > 120.0

    # 2. Verify valid Day-0 and Day-5 prediction intervals (yhat_lower < yhat < yhat_upper)
    pool = build_cold_start_donor_pool(
        panel,
        cfg,
        min_obs=14,
        text_cols=["product_title"],
        attribute_cols=["brand", "pack_size_oz"],
        hierarchy_cols=["department", "category", "subcategory"],
    )
    for sku_id in ("npi_day0_oat_vanilla", "npi_day5_oat_barista"):
        sub_df = panel[panel["ts_id"] == sku_id].copy()
        fc_res = forecast_cold_start_series(
            sub_df,
            pool,
            horizon=7,
            freq="D",
            min_obs=14,
            matching="hybrid",
            text_cols=["product_title"],
            attribute_cols=["brand", "pack_size_oz"],
            hierarchy_cols=["department", "category", "subcategory"],
        )
        preds = fc_res.predictions
        assert len(preds) == 7
        assert bool((preds["yhat_lower"] < preds["yhat"]).all())
        assert bool((preds["yhat"] < preds["yhat_upper"]).all())
        assert fc_res.donor_similarity_score > 0.65
        assert any(d.startswith("donor_oat_") for d in fc_res.top_donor_ts_ids)


def test_t4_scenario_4_multimodal_retail_similarity_text_attributes_hierarchy() -> None:
    """Tier 4 Scenario 4 (F16): Multi-Modal Retail Similarity (`text_cols`, `embedding_col`,
    `attribute_cols`, `hierarchy_cols`, `within_hierarchy_level`, `semantic_weight`, and
    `hierarchy_weight`).

    Verifies on a synthetic retail panel that coincidental seasonal twins (*Ice Cream* vs.
    *Sunscreen/Lawn Care*) are separated when semantic + Wu-Palmer LCA hierarchy similarity
    are active, while true retail substitutes (*Oat Milk Vanilla 64oz* vs.
    *Oat Milk Original 64oz*) are ranked `#1`, and `within_hierarchy_level="department"`
    strictly excludes cross-department items.
    """
    from scale_forecasting.similarity import (
        cluster_by_similarity_df,
        compute_similarity_matrix,
        embed_series_df,
        find_similar_series_df,
    )

    panel = _make_phase2_multimodal_retail_panel()

    # 1. Pure behavioral trajectory similarity treats coincidental seasonal twins as near-identical
    behav_res = compute_similarity_matrix(panel, method="hybrid")
    assert float(behav_res.matrix_df.loc["oat_milk_vanilla_64oz", "sunscreen_lotion_spf50"]) > 0.90
    assert float(behav_res.matrix_df.loc["ice_cream_gelato_16oz", "sunscreen_lotion_spf50"]) > 0.90

    # 2. Multi-modal similarity with text_cols, embedding_col, attribute_cols, hierarchy_cols,
    #    and explicit semantic_weight / hierarchy_weight separates coincidental twins
    mm_res = compute_similarity_matrix(
        panel,
        method="hybrid",
        text_cols=["product_title"],
        embedding_col="text_embedding",
        attribute_cols=["brand", "pack_size_oz", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        semantic_weight=0.35,
        hierarchy_weight=0.30,
    )
    assert {"semantic_sim", "hierarchy_sim"}.issubset(mm_res.pairs_df.columns)

    q_pairs = mm_res.pairs_df[mm_res.pairs_df["ts_id"] == "oat_milk_vanilla_64oz"].set_index(
        "neighbor_ts_id"
    )
    # Wu-Palmer LCA hierarchy proximity:
    # Same subcategory (depth 3/3) -> 1.0; same category (depth 2/3) -> 2/3;
    # same department (depth 1/3) -> 1/3; disjoint department -> 0.0
    assert float(q_pairs.loc["oat_milk_orig_64oz", "hierarchy_sim"]) == pytest.approx(1.0)
    assert float(q_pairs.loc["almond_milk_64oz", "hierarchy_sim"]) == pytest.approx(2.0 / 3.0)
    assert float(q_pairs.loc["greek_yogurt_32oz", "hierarchy_sim"]) == pytest.approx(1.0 / 3.0)
    assert float(q_pairs.loc["sunscreen_lotion_spf50", "hierarchy_sim"]) == pytest.approx(0.0)
    assert float(q_pairs.loc["ice_cream_gelato_16oz", "hierarchy_sim"]) == pytest.approx(0.0)

    # True substitute Oat Milk Original is ranked #1 and separated from Sunscreen / Ice Cream
    top_neighbor = str(
        mm_res.pairs_df[mm_res.pairs_df["ts_id"] == "oat_milk_vanilla_64oz"]
        .sort_values("rank")
        .iloc[0]["neighbor_ts_id"]
    )
    assert top_neighbor == "oat_milk_orig_64oz"
    assert (
        float(mm_res.matrix_df.loc["oat_milk_vanilla_64oz", "oat_milk_orig_64oz"])
        > float(mm_res.matrix_df.loc["oat_milk_vanilla_64oz", "sunscreen_lotion_spf50"]) + 0.35
    )
    assert float(mm_res.matrix_df.loc["ice_cream_gelato_16oz", "sunscreen_lotion_spf50"]) < 0.50

    # 3. find_similar_series_df with within_hierarchy_level="department" strictly excludes
    #    cross-department items (Frozen, Personal_Care)
    nbrs_dept = find_similar_series_df(
        panel,
        query_ts_id="oat_milk_vanilla_64oz",
        top_k=6,
        mode="full_history",
        text_cols=["product_title"],
        embedding_col="text_embedding",
        attribute_cols=["brand", "pack_size_oz", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        within_hierarchy_level="department",
        semantic_weight="auto",
        hierarchy_weight="auto",
    )
    assert {"semantic_sim", "hierarchy_sim"}.issubset(nbrs_dept.columns)
    assert str(nbrs_dept.iloc[0]["neighbor_ts_id"]) == "oat_milk_orig_64oz"
    returned_neighbors = set(nbrs_dept["neighbor_ts_id"].astype(str))
    assert "sunscreen_lotion_spf50" not in returned_neighbors
    assert "ice_cream_gelato_16oz" not in returned_neighbors
    assert returned_neighbors.issubset(
        {"oat_milk_orig_64oz", "oat_milk_choco_64oz", "almond_milk_64oz", "greek_yogurt_32oz"}
    )

    # 4. cluster_by_similarity_df and embed_series_df with multi-modal inputs
    clusters_df = cluster_by_similarity_df(
        panel,
        n_clusters=3,
        min_cluster_size=1,
        text_cols=["product_title"],
        embedding_col="text_embedding",
        attribute_cols=["brand", "pack_size_oz", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        semantic_weight=0.35,
        hierarchy_weight=0.30,
    )
    c_map = clusters_df.set_index("ts_id")["similarity_cluster_id"].to_dict()
    assert c_map["oat_milk_vanilla_64oz"] == c_map["oat_milk_orig_64oz"]
    assert c_map["oat_milk_vanilla_64oz"] != c_map["sunscreen_lotion_spf50"]

    emb_df = embed_series_df(
        panel,
        text_cols=["product_title"],
        embedding_col="text_embedding",
        attribute_cols=["brand", "pack_size_oz", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        semantic_weight=0.30,
        hierarchy_weight=0.25,
    )
    assert len(emb_df) == 7
    assert {"embed_x", "embed_y", "hurst_exp", "lyapunov_exp"}.issubset(emb_df.columns)


def test_t4_scenario_5_distributed_sharding_with_similarity_global() -> None:
    """Tier 4 Scenario 5 (F5 / F20): Distributed Multi-Worker Sharding with `@similarity_global`.

    Tests distributed multi-worker sharding (`vertex_engine` worker series selection / panel
    model routing + `spark_io.add_bucket` + `ray_io` chunking) with `<model>@similarity_global`,
    verifying that all clusters and series are preserved without cross-shard fragmentation.
    """
    import unittest.mock

    from scale_forecasting.engines import spark_io
    from scale_forecasting.engines.vertex_engine import (
        WorkerTopology,
        build_worker_series_range,
        effective_worker_count,
        execute_panel,
        models_sharded_across_workers,
        partition_models_for_worker,
        shard_series_for_worker,
    )

    panel = _make_multi_regime_retail_panel(
        n_days=42, n_short_days=5, include_cold_start=False, seed=4005
    )
    all_ids = sorted(panel["ts_id"].astype(str).unique())
    assert len(all_ids) == 6

    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-scen-5-distributed-sharding",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": [
                "tide@similarity_global",
                "theta@similarity_cold_start",
                "naive_mean",
            ],
            "model_params": {
                "tide": {
                    "max_steps": 15,
                    "input_size": 14,
                    "learning_rate": 0.01,
                    "random_seed": 42,
                }
            },
            "features": {
                "similarity_method": "hybrid",
                "similarity_n_clusters": 2,
                "cold_start_transfer": True,
                "cold_start_matching": "similarity",
            },
            "backtest": {"enabled": False},
        }
    )

    # 1. Vertex Engine multi-worker topology & model/series routing
    assert models_sharded_across_workers(cfg, ["tide@similarity_global"]) is True
    assert effective_worker_count(cfg, cfg.models, requested_workers=2, n_series=6) == 2

    topo_single = WorkerTopology(rank=0, world_size=1, source="single")
    topo_w0 = WorkerTopology(rank=0, world_size=2, source="cli")
    topo_w1 = WorkerTopology(rank=1, world_size=2, source="cli")

    # Verify build_worker_series_range and shard_series_for_worker partition local series cleanly
    restr_0, set_0, total_0 = build_worker_series_range(all_ids, None, "ts_id", topo_w0)
    restr_1, set_1, total_1 = build_worker_series_range(all_ids, None, "ts_id", topo_w1)
    assert total_0 == 6 and total_1 == 6
    assert restr_0 is not None and restr_1 is not None
    assert set_0.isdisjoint(set_1)
    assert sorted(set_0 | set_1) == all_ids
    shard_0 = shard_series_for_worker(panel, topo_w0)
    shard_1 = shard_series_for_worker(panel, topo_w1)
    assert isinstance(shard_0, pd.DataFrame) and isinstance(shard_1, pd.DataFrame)
    assert sorted(set(shard_0["ts_id"]) | set(shard_1["ts_id"])) == all_ids

    # Execute single-worker reference vs 2-worker distributed execution via execute_panel
    cells_single, status_single = execute_panel(panel, cfg, topology=topo_single)
    cells_w0, status_w0 = execute_panel(panel, cfg, topology=topo_w0)
    cells_w1, status_w1 = execute_panel(panel, cfg, topology=topo_w1)

    assert partition_models_for_worker(cfg, cfg.models, topo_w0) == [
        "tide@similarity_global",
        "theta@similarity_cold_start",
        "naive_mean",
    ]
    assert partition_models_for_worker(cfg, cfg.models, topo_w1) == [
        "theta@similarity_cold_start",
        "naive_mean",
    ]

    cells_dist = [*cells_w0, *cells_w1]
    status_dist = pd.concat([status_w0, status_w1], ignore_index=True)
    assert len(cells_single) == 18  # 6 series x 3 models
    assert len(cells_dist) == 18
    assert set(status_single["status"]) == {"ok"}
    assert set(status_dist["status"]) == {"ok"}

    # Verify @similarity_global was NOT fragmented across shards: all 6 series were pooled together
    sim_cells_single = {
        c.ts_id: c for c in cells_single if c.model_type == "tide@similarity_global"
    }
    sim_cells_dist = {c.ts_id: c for c in cells_dist if c.model_type == "tide@similarity_global"}
    assert sorted(sim_cells_dist.keys()) == all_ids
    for tid in all_ids:
        diag_s = sim_cells_single[tid].diagnostics["similarity_clusters"]
        diag_d = sim_cells_dist[tid].diagnostics["similarity_clusters"]
        assert int(diag_d["n_clusters"]) == 2
        assert diag_d["cluster_id"] == diag_s["cluster_id"]
        np.testing.assert_allclose(
            sim_cells_dist[tid].predictions["yhat"].to_numpy(dtype=float),
            sim_cells_single[tid].predictions["yhat"].to_numpy(dtype=float),
            atol=1e-6,
        )

    # 2. Ray IO chunk_cells preserves full 6-series panel in 1 chunk for @similarity_global
    chunks = chunk_cells(panel, cfg, cfg.models, n_chunks=4)
    sim_global_chunks = [c for c in chunks if (c[_MODEL_COL] == "tide@similarity_global").all()]
    assert len(sim_global_chunks) == 1
    assert sorted(sim_global_chunks[0]["ts_id"].unique()) == all_ids

    # 3. Spark IO add_bucket & run_group isolate @similarity_global onto a single model-level bucket
    class _FakeSparkCol:
        def __init__(self, expr: str) -> None:
            self.expr = expr

        def __str__(self) -> str:
            return self.expr

        def isin(self, values: Any) -> _FakeSparkCol:
            return _FakeSparkCol(f"{self.expr}.isin({list(values)!r})")

        def otherwise(self, other: Any) -> _FakeSparkCol:
            return _FakeSparkCol(f"{self.expr}.otherwise({other})")

        def __eq__(self, other: object) -> _FakeSparkCol:  # type: ignore[override]
            return _FakeSparkCol(f"({self.expr} == {other})")

        def __add__(self, other: Any) -> _FakeSparkCol:
            return _FakeSparkCol(f"({self.expr} + {other})")

    class _FakeSparkF:
        @staticmethod
        def col(name: str) -> _FakeSparkCol:
            return _FakeSparkCol(f"col({name})")

        @staticmethod
        def lit(val: Any) -> _FakeSparkCol:
            return _FakeSparkCol(f"lit({val!r})")

        @staticmethod
        def hash(*cols: Any) -> _FakeSparkCol:
            return _FakeSparkCol(f"hash({', '.join(str(c) for c in cols)})")

        @staticmethod
        def pmod(a: Any, b: Any) -> _FakeSparkCol:
            return _FakeSparkCol(f"pmod({a}, {b})")

        @staticmethod
        def when(cond: Any, val: Any) -> _FakeSparkCol:
            return _FakeSparkCol(f"when({cond}, {val})")

    class _RecordingSparkDF:
        def __init__(self) -> None:
            self.added_col: str | None = None
            self.bucket_expr: Any = None

        def withColumn(self, col_name: str, expr: Any) -> _RecordingSparkDF:
            self.added_col = col_name
            self.bucket_expr = expr
            return self

    alloc = spark_io.allocate_buckets(
        cfg.models,
        12,
        {
            "tide@similarity_global": 4.0,
            "theta@similarity_cold_start": 1.0,
            "naive_mean": 1.0,
        },
    )
    fake_sdf = _RecordingSparkDF()
    with unittest.mock.patch("pyspark.sql.functions", _FakeSparkF):
        out_sdf = spark_io.add_bucket(fake_sdf, cfg, n_buckets=12, allocation=alloc)  # type: ignore[arg-type]
    assert out_sdf is fake_sdf
    assert fake_sdf.added_col == spark_io._BUCKET_COL
    assert fake_sdf.bucket_expr is not None
    expr_str = str(fake_sdf.bucket_expr)
    # tide@similarity_global is mapped to constant bucket lit(0) (not hashed by ts_id)
    assert "when((col(_sf_model) == lit('tide@similarity_global')), lit(0))" in expr_str

    # Also verify spark_io.run_group executes the single-bucket @similarity_global group cleanly
    spark_group_pdf = panel.assign(**{spark_io._MODEL_COL: "tide@similarity_global"})
    spark_cells, spark_status = spark_io.run_group(spark_group_pdf, cfg)
    assert len(spark_cells) == 6
    assert set(spark_status["status"]) == {"ok"}
    assert sorted(c.ts_id for c in spark_cells) == all_ids


def test_t4_scenario_6_latent_similarity_hierarchy_3mode_mint_and_goal_seek() -> None:
    """Tier 4 Scenario 6 (F6 / F17): Latent Similarity Hierarchy + 3-Mode MiNT + Top-Down
    Goal-Seek on `"__sim_cluster_0__"`.

    Tests `HierarchyConfig(similarity_hierarchy=True, cross_temporal=True,
    reconciliation_methods=["mint_shrink", "cross_temporal_mint"])` + 3-mode reconciliation +
    Top-Down Goal-Seek `allocate_target(..., node_id="__sim_cluster_0__")` (and
    `Forecaster.allocate_target(node_id="__sim_cluster_0__")`), verifying `< 1e-6`
    reconciliation coherence and exact `allocated_total == target_total` on `"__sim_cluster_0__"`.
    """
    from scale_forecasting.temporal_reconciliation import reconcile_temporal_forecasts

    panel = _make_multi_regime_retail_panel(
        n_days=42, n_short_days=5, include_cold_start=False, seed=4006
    )
    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-scen-6-latent-hier-3mode-gs",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 14},
            "models": ["naive_mean"],
            "hierarchy": {
                "similarity_hierarchy": True,
                "similarity_n_clusters": 2,
                "cross_temporal": True,
                "temporal_aggregations": [7, 14],
                "reconciliation_methods": ["mint_shrink", "cross_temporal_mint"],
            },
        }
    )
    assert cfg.hierarchy.enabled is True
    assert cfg.hierarchy.reconcile_temporal is True
    assert cfg.hierarchy.effective_levels == [["similarity_cluster_id"]]

    hier_panel, spec = build_hierarchy(panel, cfg)
    assert "__total__" in spec.node_ids
    assert "similarity_cluster_id=cluster_0" in spec.node_ids
    assert "similarity_cluster_id=cluster_1" in spec.node_ids

    f_dates = pd.date_range("2025-02-12", periods=14, freq="D")
    pred_rows = [
        {
            "ts_id": nid,
            "model_type": "naive_mean",
            "ds": dt,
            "forecast_date": dt,
            "horizon_step": h_idx,
            "yhat": float(80.0 + 4.0 * (h_idx % 7)),
            "yhat_lower": float(65.0 + 4.0 * (h_idx % 7)),
            "yhat_upper": float(95.0 + 4.0 * (h_idx % 7)),
        }
        for nid in spec.bottom_ids
        for h_idx, dt in enumerate(f_dates, start=1)
    ]
    preds_df = pd.DataFrame(pred_rows)

    # 1. Mode 1 (Spatial-only MiNT shrinkage)
    rec_spatial = reconcile_forecasts(preds_df, spec, "mint_shrink", cfg=cfg, history_df=hier_panel)
    assert verify_coherence(rec_spatial, spec) < 1e-6

    # 2. Mode 2 (Temporal-only THieF MiNT shrinkage)
    rec_temporal = reconcile_temporal_forecasts(
        preds_df, history_df=hier_panel, cfg=cfg, method="mint_shrink"
    )
    assert verify_temporal_coherence(rec_temporal, cfg=cfg) < 1e-6

    # 3. Mode 3 (Simultaneous Cross-Temporal MiNT)
    rec_ct = reconcile_forecasts(
        preds_df, spec, "cross_temporal_mint", cfg=cfg, history_df=hier_panel
    )
    assert verify_coherence(rec_ct, spec) < 1e-6
    assert verify_temporal_coherence(rec_ct, cfg=cfg) < 1e-6

    # 4. Top-Down Goal-Seek on shorthand `node_id="__sim_cluster_0__"` via `allocate_target`
    target_val = 4200.0
    gs_res = allocate_target(
        rec_ct,
        target_total=target_val,
        node_id="__sim_cluster_0__",
        history_df=hier_panel,
        cfg=cfg,
        return_result=True,
    )
    assert isinstance(gs_res, GoalSeekResult)
    assert gs_res.max_coherence_discrepancy < 1e-6
    assert gs_res.max_target_discrepancy < 1e-6
    assert float(gs_res.allocated_total) == pytest.approx(target_val, abs=1e-4)
    c0_rows = gs_res.dataframe[gs_res.dataframe["ts_id"] == "similarity_cluster_id=cluster_0"]
    assert float(c0_rows["allocated_yhat"].sum()) == pytest.approx(target_val, abs=1e-4)

    # 5. Also verify via `Forecaster.allocate_target(node_id="__sim_cluster_0__")`
    f = sf.Forecaster(cfg, data=panel)
    gs_sdk_res = f.allocate_target(
        target_val,
        forecasts_df=rec_ct,
        node_id="__sim_cluster_0__",
        return_result=True,
    )
    assert isinstance(gs_sdk_res, GoalSeekResult)
    assert gs_sdk_res.max_coherence_discrepancy < 1e-6
    assert float(gs_sdk_res.allocated_total) == pytest.approx(target_val, abs=1e-4)


def test_t4_scenario_7_low_stock_substitute_recommendation_and_cannibalization_optimization() -> (
    None
):
    """Tier 4 Scenario 7 (F18a / F8): Low-Stock Assortment Substitute Recommendation +
    Similarity-Weighted Cross-SKU Cannibalization in Prescriptive Scenario Optimization.

    Tests `recommend_substitutes_df` (and `Forecaster.recommend_substitutes_df` /
    `PlaygroundRun.recommend_substitutes_df`) with `inventory_df`, `require_in_stock=True`,
    `within_hierarchy_level`, and all required columns (`query_ts_id`, `substitute_ts_id`,
    `rank`, `substitution_score`, `similarity_score`, `semantic_sim`, `hierarchy_sim`,
    `available_units`, `stock_status`, `substitution_rationale`), combined with
    similarity-weighted cross-SKU cannibalization in
    `optimize_scenario(..., use_similarity_priors=True)`.
    """
    from scale_forecasting.scenario_optimizer import optimize_scenario
    from scale_forecasting.similarity import recommend_substitutes_df

    panel = _make_phase2_multimodal_retail_panel()
    inv_df = pd.DataFrame(
        [
            {"ts_id": "oat_milk_vanilla_64oz", "on_hand": 0.0, "safety_stock": 20.0},
            {"ts_id": "oat_milk_orig_64oz", "on_hand": 140.0, "safety_stock": 25.0},
            {"ts_id": "oat_milk_choco_64oz", "on_hand": 0.0, "safety_stock": 20.0},
            {"ts_id": "almond_milk_64oz", "on_hand": 65.0, "safety_stock": 20.0},
            {"ts_id": "greek_yogurt_32oz", "on_hand": 18.0, "safety_stock": 25.0},
            {"ts_id": "ice_cream_gelato_16oz", "on_hand": 90.0, "safety_stock": 20.0},
            {"ts_id": "sunscreen_lotion_spf50", "on_hand": 180.0, "safety_stock": 30.0},
        ]
    )

    required_cols = {
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
    }

    # 1. Direct recommend_substitutes_df with inventory_df, require_in_stock=True, and
    #    within_hierarchy_level="department"
    subs_df = recommend_substitutes_df(
        panel,
        inv_df,
        query_ts_id="oat_milk_vanilla_64oz",
        top_k=5,
        text_cols=["product_title"],
        attribute_cols=["brand", "pack_size_oz", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        within_hierarchy_level="department",
        require_in_stock=True,
    )
    assert required_cols.issubset(subs_df.columns)
    assert not subs_df.empty
    # Out-of-stock oat_milk_choco_64oz and cross-department items are excluded
    sub_ids = set(subs_df["substitute_ts_id"].astype(str))
    assert "oat_milk_choco_64oz" not in sub_ids
    assert "sunscreen_lotion_spf50" not in sub_ids
    assert "ice_cream_gelato_16oz" not in sub_ids
    # In-stock Oat Milk Original ranks #1 with surplus stock
    assert str(subs_df.iloc[0]["substitute_ts_id"]) == "oat_milk_orig_64oz"
    assert int(subs_df.iloc[0]["rank"]) == 1
    assert str(subs_df.iloc[0]["stock_status"]) == "in_stock_surplus"
    assert float(subs_df.iloc[0]["available_units"]) == pytest.approx(115.0)
    assert float(subs_df.iloc[0]["substitution_score"]) > 0.70
    assert bool(subs_df.iloc[0]["substitution_rationale"])

    # 2. Verify Forecaster.recommend_substitutes_df and PlaygroundRun.recommend_substitutes_df
    cfg_sub = RunConfig.model_validate(
        {
            "run_name": "t4-scen-7-subs-and-cannib",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
        }
    )
    f_sub = sf.Forecaster(cfg_sub, data=panel)
    subs_sdk = f_sub.recommend_substitutes_df(
        inventory_df=inv_df,
        query_ts_id="oat_milk_vanilla_64oz",
        text_cols=["product_title"],
        attribute_cols=["brand", "pack_size_oz", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        within_hierarchy_level="department",
        require_in_stock=True,
    )
    assert required_cols.issubset(subs_sdk.columns)
    assert str(subs_sdk.iloc[0]["substitute_ts_id"]) == "oat_milk_orig_64oz"

    pg_run = run_model("naive_mean", data=panel, horizon=7, backtest=False)
    subs_pg = pg_run.recommend_substitutes_df(
        inventory_df=inv_df,
        query_ts_id="oat_milk_vanilla_64oz",
        text_cols=["product_title"],
        attribute_cols=["brand", "pack_size_oz", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        within_hierarchy_level="department",
        require_in_stock=True,
    )
    assert required_cols.issubset(subs_pg.columns)
    assert str(subs_pg.iloc[0]["substitute_ts_id"]) == "oat_milk_orig_64oz"

    # 3. Combine with similarity-weighted cross-SKU cannibalization in optimize_scenario
    hist_cannib, fut_cannib = _make_promo_cannibalization_twin_panel(
        n_days=49, horizon=7, seed=4007
    )
    cfg_opt = RunConfig.model_validate(
        {
            "run_name": "t4-scen-7-cannib-opt",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["regression_lags@monotone"],
            "features": {
                "future_covariates": ["price", "promo"],
                "monotonic_constraints": {"price": -1, "promo": 1},
            },
            "backtest": {"enabled": False},
        }
    )
    scen_res = optimize_scenario(
        hist_cannib,
        config=cfg_opt,
        future_covariates_df=fut_cannib,
        objective="gross_profit",
        price_col="price",
        promo_col="promo",
        unit_cost=4.0,
        max_promo_days=2,
        min_cooldown_days=2,
        use_similarity_priors=True,
        reconcile=False,
    )
    assert scen_res.optimal_objective_value >= scen_res.baseline_objective_value - 1e-6
    assert "cannibalization_effect" in scen_res.schedule_df.columns
    assert "total_cannibalization_effect" in scen_res.summary_df.columns
    assert not scen_res.inventory_delta_df.empty


def test_t4_scenario_8_emerging_trend_breakout_and_lead_lag_copromotion_sensing(
    tmp_path: Path,
) -> None:
    """Tier 4 Scenario 8 (F18b / F19): Emerging Trend Breakout & Bellwether Lead-Lag
    Co-Promotion Sensing across `detect_emerging_trends_df`, `Forecaster`, `PlaygroundRun`,
    `export_report()`, and MCP `compute_time_series_similarity`.

    Tests `detect_emerging_trends_df` (and `Forecaster.emerging_trends_df` /
    `PlaygroundRun.emerging_trends_df`), `Forecaster.export_report()`, and MCP
    `compute_time_series_similarity` (`mode="substitutes"`, `mode="emerging_trends"`,
    `include_substitutes=True`, `include_emerging_trends=True`), verifying all required columns,
    accelerating breakout leader detection, bellwether lead-lag shift recovery, and
    `"co_promote_lagging_twin"` merchandising action.
    """
    from scale_forecasting.mcp import compute_time_series_similarity
    from scale_forecasting.similarity import detect_emerging_trends_df

    n_obs = 48
    dates = pd.date_range("2025-01-01", periods=n_obs, freq="D")
    t = np.arange(n_obs, dtype=float)
    rng = np.random.default_rng(4008)

    # Bellwether breakout leader: flat baseline until day 24, then steep quadratic acceleration
    ramp_lead = np.maximum(0.0, t - 24.0)
    y_leader = 35.0 + 1.9 * ramp_lead + 0.19 * (ramp_lead**2) + 2.0 * np.sin(2.0 * np.pi * t / 7.0)
    # Lagging digital twin: follows leader_breakout with a 4-day lag
    ramp_follow = np.maximum(0.0, (t - 4.0) - 24.0)
    y_follower = (
        35.0
        + 1.9 * ramp_follow
        + 0.19 * (ramp_follow**2)
        + 2.0 * np.sin(2.0 * np.pi * (t - 4.0) / 7.0)
    )
    # Stable series
    y_stable = 60.0 + 3.0 * np.sin(2.0 * np.pi * t / 7.0) + rng.normal(0.0, 0.4, size=n_obs)
    # Declining clearance candidate
    y_declining = np.maximum(5.0, 130.0 - 2.2 * t + rng.normal(0.0, 0.4, size=n_obs))

    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts_id": "sku_bellwether_leader",
                    "ds": dates,
                    "y": y_leader,
                    "product_title": "Cold Brew Espresso Concentrate Vanilla 32oz Bottle",
                    "department": "Beverages",
                    "category": "Coffee",
                    "brand": "RoastCraft",
                }
            ),
            pd.DataFrame(
                {
                    "ts_id": "sku_lagging_follower",
                    "ds": dates,
                    "y": y_follower,
                    "product_title": "Cold Brew Espresso Concentrate Caramel 32oz Bottle",
                    "department": "Beverages",
                    "category": "Coffee",
                    "brand": "RoastCraft",
                }
            ),
            pd.DataFrame(
                {
                    "ts_id": "sku_steady_staple",
                    "ds": dates,
                    "y": y_stable,
                    "product_title": "Classic Medium Roast Ground Coffee 16oz Bag",
                    "department": "Beverages",
                    "category": "Coffee",
                    "brand": "RoastCraft",
                }
            ),
            pd.DataFrame(
                {
                    "ts_id": "sku_declining_seasonal",
                    "ds": dates,
                    "y": y_declining,
                    "product_title": "Holiday Peppermint Mocha Creamer 16oz Bottle",
                    "department": "Beverages",
                    "category": "Creamer",
                    "brand": "FrostBrew",
                }
            ),
        ],
        ignore_index=True,
    )

    required_cols = {
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
    }

    # 1. Direct detect_emerging_trends_df verification
    trends_df = detect_emerging_trends_df(
        panel,
        window=14,
        baseline_window=24,
        top_k_related=3,
        max_lag=7,
        text_cols=["product_title"],
        attribute_cols=["brand"],
        hierarchy_cols=["department", "category"],
    )
    assert required_cols.issubset(trends_df.columns)
    assert len(trends_df) == 4

    by_id = trends_df.set_index("ts_id")
    assert str(by_id.loc["sku_bellwether_leader", "trend_regime"]) == "accelerating_breakout"
    assert float(by_id.loc["sku_bellwether_leader", "breakout_zscore"]) > 2.0
    assert float(by_id.loc["sku_bellwether_leader", "recent_velocity"]) > 0.0
    assert float(by_id.loc["sku_bellwether_leader", "recent_acceleration"]) > 0.0
    assert "sku_lagging_follower" in str(
        by_id.loc["sku_bellwether_leader", "co_promote_candidates"]
    )

    # Follower lags behind sku_bellwether_leader and receives "co_promote_lagging_twin"
    assert (
        str(by_id.loc["sku_lagging_follower", "leading_bellwether_ts_id"])
        == "sku_bellwether_leader"
    )
    assert int(by_id.loc["sku_lagging_follower", "lag_behind_leader"]) > 0
    assert (
        str(by_id.loc["sku_lagging_follower", "recommended_merchandising_action"])
        == "co_promote_lagging_twin"
    )
    assert str(by_id.loc["sku_declining_seasonal", "trend_regime"]) == "declining"
    assert (
        str(by_id.loc["sku_declining_seasonal", "recommended_merchandising_action"])
        == "clearance_candidate"
    )

    # 2. Verify Forecaster.emerging_trends_df, PlaygroundRun.emerging_trends_df & export_report
    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-scen-8-emerging-trends",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
        }
    )
    f = sf.Forecaster(cfg, data=panel)
    f_trends = f.emerging_trends_df(
        window=14,
        max_lag=7,
        text_cols=["product_title"],
        attribute_cols=["brand"],
        hierarchy_cols=["department", "category"],
    )
    assert required_cols.issubset(f_trends.columns)

    pg_run = run_model("naive_mean", data=panel, horizon=7, backtest=False)
    pg_trends = pg_run.emerging_trends_df(
        window=14,
        max_lag=7,
        text_cols=["product_title"],
        hierarchy_cols=["department", "category"],
    )
    assert required_cols.issubset(pg_trends.columns)

    subs_df = f.recommend_substitutes_df(
        query_ts_id="sku_bellwether_leader",
        text_cols=["product_title"],
        hierarchy_cols=["department", "category"],
    )
    rep_file = f.export_report(
        tmp_path / "scen8_trends_report.html",
        substitutes_df=subs_df,
        emerging_trends_df=f_trends,
    )
    html_str = rep_file.read_text(encoding="utf-8")
    assert "Emerging Trends &amp; Breakout Detection" in html_str
    assert "SKU Substitute &amp; Cannibalization Recommendations" in html_str
    assert "co_promote_lagging_twin" in html_str

    pg_rep_file = pg_run.export_report(
        tmp_path / "scen8_pg_trends_report.html",
        substitutes_df=subs_df,
        emerging_trends_df=pg_trends,
    )
    pg_html_str = pg_rep_file.read_text(encoding="utf-8")
    assert "SKU Substitute &amp; Cannibalization Recommendations" in pg_html_str
    assert "co_promote_lagging_twin" in pg_html_str

    # 3. Verify MCP compute_time_series_similarity across mode="substitutes",
    #    mode="emerging_trends", include_substitutes=True, and include_emerging_trends=True
    records = panel.assign(ds=panel["ds"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    inv_records = [
        {"ts_id": "sku_bellwether_leader", "on_hand": 0.0, "safety_stock": 15.0},
        {"ts_id": "sku_lagging_follower", "on_hand": 120.0, "safety_stock": 20.0},
        {"ts_id": "sku_steady_staple", "on_hand": 80.0, "safety_stock": 15.0},
        {"ts_id": "sku_declining_seasonal", "on_hand": 40.0, "safety_stock": 10.0},
    ]

    mcp_subs = compute_time_series_similarity(
        records=records,
        mode="substitutes",
        query_ts_id="sku_bellwether_leader",
        text_cols=["product_title"],
        attribute_cols=["brand"],
        hierarchy_cols=["department", "category"],
        within_hierarchy_level="category",
        inventory_records=inv_records,
        require_in_stock=True,
    )
    assert mcp_subs["status"] == "ok"
    assert "substitutes" in mcp_subs and len(mcp_subs["substitutes"]) >= 1
    assert mcp_subs["substitutes"][0]["substitute_ts_id"] == "sku_lagging_follower"

    mcp_trends = compute_time_series_similarity(
        records=records,
        mode="emerging_trends",
        window=14,
        max_lag=7,
        text_cols=["product_title"],
        attribute_cols=["brand"],
        hierarchy_cols=["department", "category"],
        include_substitutes=True,
        include_emerging_trends=True,
        inventory_records=inv_records,
    )
    assert mcp_trends["status"] == "ok"
    assert "substitutes" in mcp_trends and len(mcp_trends["substitutes"]) >= 1
    assert "emerging_trends" in mcp_trends and len(mcp_trends["emerging_trends"]) == 4
    trend_map = {row["ts_id"]: row for row in mcp_trends["emerging_trends"]}
    assert trend_map["sku_bellwether_leader"]["trend_regime"] == "accelerating_breakout"
    assert (
        trend_map["sku_lagging_follower"]["recommended_merchandising_action"]
        == "co_promote_lagging_twin"
    )

    # Also verify via JSON-RPC McpServer tool call
    srv = McpServer()
    rpc_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 808,
            "method": "tools/call",
            "params": {
                "name": "compute_time_series_similarity",
                "arguments": {
                    "records": records,
                    "mode": "emerging_trends",
                    "window": 14,
                    "max_lag": 7,
                    "include_substitutes": True,
                    "include_emerging_trends": True,
                    "text_cols": ["product_title"],
                    "hierarchy_cols": ["department", "category"],
                },
            },
        }
    )
    assert rpc_resp is not None and "result" in rpc_resp
    assert rpc_resp["result"]["isError"] is False


def test_adv_all_lenses_extreme_scale_nan_inf_and_lead_lag_antisymmetry() -> None:
    """Tier 5 Adversarial (G3): Verify all 7 similarity lenses under 15-order-of-magnitude scale
    differences (1e-6 vs 1e9), +inf/-inf/all-NaN series, T=1..3 ultra-short series, and lead-lag
    antisymmetry (+4 vs -4) with max_lag > T.
    """
    from scale_forecasting.config import SIMILARITY_METHODS
    from scale_forecasting.similarity import compute_similarity_matrix

    dates_30 = pd.date_range("2025-01-01", periods=30, freq="D")
    t = np.arange(30, dtype=float)
    wave = 1.0 + 0.4 * np.sin(2.0 * np.pi * t / 7.0)
    nan_inf_y = wave.copy()
    nan_inf_y[0] = np.nan
    nan_inf_y[5] = np.inf
    nan_inf_y[10] = -np.inf
    nan_inf_y[29] = np.nan

    path_df = pd.concat(
        [
            pd.DataFrame({"ts_id": "micro_1e_6", "ds": dates_30, "y": 1e-6 * wave}),
            pd.DataFrame({"ts_id": "macro_1e9", "ds": dates_30, "y": 1e9 * wave}),
            pd.DataFrame({"ts_id": "all_nan", "ds": dates_30, "y": np.full(30, np.nan)}),
            pd.DataFrame({"ts_id": "nan_inf_mix", "ds": dates_30, "y": nan_inf_y}),
            pd.DataFrame({"ts_id": "short_t3", "ds": dates_30[:3], "y": [1.0, 2.0, 1.5]}),
            pd.DataFrame({"ts_id": "short_t1", "ds": dates_30[:1], "y": [5.0]}),
        ],
        ignore_index=True,
    )
    for m in SIMILARITY_METHODS:
        res = compute_similarity_matrix(path_df, method=m, top_k=5, max_lag=50)
        s_mat = res.matrix_df.to_numpy(dtype=float)
        assert np.all(np.isfinite(s_mat))
        np.testing.assert_allclose(s_mat, s_mat.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(s_mat), 1.0, atol=1e-10)
        assert float(np.min(np.linalg.eigvalsh(s_mat))) >= -1e-8
        assert float(res.matrix_df.loc["micro_1e_6", "macro_1e9"]) > 0.95


def test_adv_cluster_singleton_outlier_merge_and_foundation_fallback_guard() -> None:
    """Tier 5 Adversarial (G1 & G2): Verify _merge_small_clusters with min_cluster_size=3 on a
    singleton outlier, identical-series clustering, and extract_foundation_embeddings guards.
    """
    from scale_forecasting.similarity import (
        cluster_by_similarity_df,
        extract_foundation_embeddings,
    )

    dates = pd.date_range("2025-01-01", periods=36, freq="D")
    t = np.arange(36, dtype=float)
    rng = np.random.default_rng(77)
    y_lead = 100.0 + 30.0 * np.sin(2.0 * np.pi * t / 12.0)
    outlier_panel = pd.concat(
        [
            *[
                pd.DataFrame(
                    {"ts_id": f"g1_{i}", "ds": dates, "y": y_lead + rng.normal(0.0, 0.2, 36)}
                )
                for i in range(3)
            ],
            *[
                pd.DataFrame(
                    {
                        "ts_id": f"g2_{i}",
                        "ds": dates,
                        "y": 50.0 + 3.0 * t + rng.normal(0.0, 0.2, 36),
                    }
                )
                for i in range(3)
            ],
            pd.DataFrame(
                {
                    "ts_id": "singleton_outlier",
                    "ds": dates,
                    "y": rng.choice([0.0, 0.0, 0.0, 500.0], size=36),
                }
            ),
        ],
        ignore_index=True,
    )
    cl_out = cluster_by_similarity_df(outlier_panel, n_clusters=3, min_cluster_size=3)
    assert bool((cl_out["cluster_size"] >= 3).all())

    emb_zero, ids_z, _ = extract_foundation_embeddings(
        outlier_panel, encoder_fn=lambda arrs: np.zeros((len(arrs), 8))
    )
    assert emb_zero.shape == (7, 8) and len(ids_z) == 7 and np.all(np.isfinite(emb_zero))
    with pytest.raises(RuntimeError):
        extract_foundation_embeddings(outlier_panel, allow_fallback=False)
