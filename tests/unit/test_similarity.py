"""Unit tests for Track P (`v1.8.0`) Time-Series Similarity & Regime Clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.cold_start import cold_start_transfer_df, forecast_cold_start_series
from scale_forecasting.config import (
    COLD_START_MATCHING_MODES,
    SIMILARITY_METHODS,
    FeaturesConfig,
    HierarchyConfig,
    RunConfig,
)
from scale_forecasting.features import (
    add_similarity_features,
    build_features,
    build_future_features,
    expand_model_variants,
    resolve_model_variant,
)
from scale_forecasting.similarity import (
    VALID_SIMILARITY_METHODS,
    SimilarityMatrixResult,
    cluster_by_similarity_df,
    compute_similarity_matrix,
    detect_emerging_trends_df,
    embed_series_df,
    emerging_trends_df,
    extract_foundation_embeddings,
    find_similar_series_df,
    plot_similarity_map,
    recommend_substitutes_df,
)
from scale_forecasting.worker import is_panel_model, run_panel


def _make_multi_regime_panel(n_obs: int = 56, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic 6-series panel with 3 distinct dynamical regimes (2 series per regime)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_obs, freq="D")
    t = np.arange(n_obs, dtype=float)
    rows: list[pd.DataFrame] = []

    # Regime A: Smooth sinusoidal seasonal series (period 7) at two scales
    for idx, scale in enumerate((100.0, 250.0)):
        y = scale * (1.0 + 0.35 * np.sin(2.0 * np.pi * t / 7.0) + rng.normal(0.0, 0.02, size=n_obs))
        rows.append(pd.DataFrame({"ts_id": f"seasonal_{idx}", "ds": dates, "y": y}))

    # Regime B: Strong upward exponential/linear trend with counter-phase cycle
    for idx, slope in enumerate((2.5, 4.0)):
        y = (
            20.0
            + slope * t
            + 8.0 * np.cos(2.0 * np.pi * t / 14.0)
            + rng.normal(0.0, 0.5, size=n_obs)
        )
        rows.append(pd.DataFrame({"ts_id": f"trend_{idx}", "ds": dates, "y": y}))

    # Regime C: Intermittent / lumpy demand with high zero fraction
    for idx in range(2):
        spikes = rng.choice([0.0, 0.0, 0.0, 0.0, 45.0, 90.0], size=n_obs)
        rows.append(pd.DataFrame({"ts_id": f"lumpy_{idx}", "ds": dates, "y": spikes}))

    return pd.concat(rows, ignore_index=True)


def test_similarity_config_options_and_hierarchy_resolution() -> None:
    """Exercise all new Track P RunConfig fields and Literal values for config coverage parity."""
    for matching in ("auto", "attribute", "similarity", "hybrid"):
        assert matching in COLD_START_MATCHING_MODES
        f_cfg = FeaturesConfig(
            cold_start_transfer=True,
            cold_start_matching=matching,  # type: ignore[arg-type]
            similarity_cluster_feature=True,
            similarity_n_clusters=3,
        )
        assert f_cfg.cold_start_matching == matching
        assert f_cfg.similarity_cluster_feature is True

    for method in (
        "hybrid",
        "physics_attractor",
        "spectral_phase",
        "elastic_dtw",
        "minirocket",
        "structural_signature",
        "foundation_latent",
    ):
        assert method in SIMILARITY_METHODS
        f_m = FeaturesConfig(similarity_method=method, similarity_n_clusters="auto")  # type: ignore[arg-type]
        assert f_m.similarity_method == method
        h_m = HierarchyConfig(
            similarity_hierarchy=True,
            similarity_n_clusters=3,
            similarity_method=method,  # type: ignore[arg-type]
        )
        assert h_m.enabled is True
        assert h_m.similarity_hierarchy is True
        assert h_m.similarity_method == method
        assert h_m.effective_levels == [["similarity_cluster_id"]]


def test_all_seven_similarity_lenses_and_psd_invariants() -> None:
    """Every SimilarityMethod lens produces a symmetric PSD [0, 1] matrix with unit diagonal."""
    panel = _make_multi_regime_panel(n_obs=48)
    for method in (
        *VALID_SIMILARITY_METHODS,
        "profile",
        "shape_dtw",
        "dtw",
        "spectral",
    ):
        res = compute_similarity_matrix(panel, method=method, top_k=5, max_lag=3)
        assert isinstance(res, SimilarityMatrixResult)
        S = res.matrix_df.to_numpy(dtype=float)
        n = S.shape[0]
        assert S.shape == (6, 6)
        assert np.allclose(S, S.T, atol=1e-10)
        assert np.allclose(np.diag(S), np.ones(n), atol=1e-8)
        assert np.all((S >= -1e-9) & (S <= 1.0 + 1e-9))
        eigvals = np.linalg.eigvalsh(0.5 * (S + S.T))
        assert float(np.min(eigvals)) >= -1e-7

        required_pair_cols = {
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
            "optimal_lag",
            "lag_correlation",
            "method",
        }
        assert required_pair_cols.issubset(res.pairs_df.columns)
        assert not res.pairs_df.empty
        assert "ts_id" in res.embeddings_df.columns
        assert "embed_x" in res.embeddings_df.columns
        assert "embed_y" in res.embeddings_df.columns
        assert "embedding_backend_used" in res.embeddings_df.columns
        payload = res.to_dict()
        assert payload["n_series"] == 6


def test_physics_attractor_lyapunov_hurst_and_permutation_entropy() -> None:
    """Physics attractor invariants distinguish smooth periodic dynamics from noise and trends."""
    n = 80
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(123)

    sine = 50.0 + 15.0 * np.sin(2.0 * np.pi * t / 7.0)
    noise = 50.0 + rng.normal(0.0, 15.0, size=n)
    trend = 10.0 + 2.0 * t

    panel = pd.concat(
        [
            pd.DataFrame({"ts_id": "sine", "ds": dates, "y": sine}),
            pd.DataFrame({"ts_id": "noise", "ds": dates, "y": noise}),
            pd.DataFrame({"ts_id": "trend", "ds": dates, "y": trend}),
        ],
        ignore_index=True,
    )
    emb = embed_series_df(panel, method="physics_attractor").set_index("ts_id")
    assert float(emb.loc["sine", "perm_entropy"]) < float(emb.loc["noise", "perm_entropy"])
    assert float(emb.loc["trend", "hurst_exp"]) > 0.5
    assert float(emb.loc["sine", "trend_strength"]) < float(emb.loc["trend", "trend_strength"])


def test_spectral_phase_kuramoto_synchronization_and_lead_lag_recovery() -> None:
    """Kuramoto phase locking separates in-phase from orthogonal waves and recovers lead-lag."""
    n = 64
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    base_wave = 100.0 + 25.0 * np.sin(2.0 * np.pi * t / 8.0)
    in_phase = 200.0 + 50.0 * np.sin(2.0 * np.pi * t / 8.0)
    lagged_wave = 100.0 + 25.0 * np.sin(2.0 * np.pi * (t - 2.0) / 8.0)
    ortho_wave = 100.0 + 25.0 * np.sin(2.0 * np.pi * t / 21.0)

    panel = pd.concat(
        [
            pd.DataFrame({"ts_id": "wave_a", "ds": dates, "y": base_wave}),
            pd.DataFrame({"ts_id": "wave_in_phase", "ds": dates, "y": in_phase}),
            pd.DataFrame({"ts_id": "wave_lag2", "ds": dates, "y": lagged_wave}),
            pd.DataFrame({"ts_id": "wave_ortho", "ds": dates, "y": ortho_wave}),
        ],
        ignore_index=True,
    )
    res = compute_similarity_matrix(panel, method="spectral_phase", max_lag=4)
    assert float(res.matrix_df.loc["wave_a", "wave_in_phase"]) > float(
        res.matrix_df.loc["wave_a", "wave_ortho"]
    )
    pair_lag = res.pairs_df[
        (res.pairs_df["ts_id"] == "wave_a") & (res.pairs_df["neighbor_ts_id"] == "wave_lag2")
    ].iloc[0]
    assert abs(int(pair_lag["optimal_lag"])) > 0
    assert float(pair_lag["lag_correlation"]) > 0.85


def test_elastic_dtw_time_warping_and_scale_invariance() -> None:
    """Scale-normalized elastic DTW matches locally time-warped and rescaled trajectories."""
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    pulse_a = np.exp(-0.5 * ((t - 20.0) / 3.0) ** 2) * 100.0
    pulse_b_shifted = np.exp(-0.5 * ((t - 22.0) / 3.0) ** 2) * 500.0
    step_down = np.where(t < 25, 100.0, 10.0)

    panel = pd.concat(
        [
            pd.DataFrame({"ts_id": "pulse_a", "ds": dates, "y": pulse_a}),
            pd.DataFrame({"ts_id": "pulse_b", "ds": dates, "y": pulse_b_shifted}),
            pd.DataFrame({"ts_id": "step_down", "ds": dates, "y": step_down}),
        ],
        ignore_index=True,
    )
    res = compute_similarity_matrix(
        panel, method="elastic_dtw", spring_stiffness=0.02, window_frac=0.20
    )
    assert float(res.matrix_df.loc["pulse_a", "pulse_b"]) > float(
        res.matrix_df.loc["pulse_a", "step_down"]
    )


def test_cluster_by_similarity_df_archetypes_medoids_and_silhouette() -> None:
    """Hierarchical clustering discovers true dynamical regimes, medoids, and archetypes."""
    panel = _make_multi_regime_panel(n_obs=56)
    clusters = cluster_by_similarity_df(panel, n_clusters=3, method="hybrid", min_cluster_size=2)
    required_cols = {
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
    }
    assert required_cols.issubset(clusters.columns)
    assert len(clusters) == 6
    by_id = clusters.set_index("ts_id")
    assert (
        by_id.loc["seasonal_0", "similarity_cluster_id"]
        == by_id.loc["seasonal_1", "similarity_cluster_id"]
    )
    assert (
        by_id.loc["trend_0", "similarity_cluster_id"]
        == by_id.loc["trend_1", "similarity_cluster_id"]
    )
    assert int(clusters["is_medoid"].sum()) == clusters["similarity_cluster_id"].nunique()
    assert (clusters["cluster_size"] >= 2).all()


def test_find_similar_series_df_full_history_launch_prefix_and_recent_window() -> None:
    """Digital-twin retrieval works across full_history, launch_prefix, and recent_window modes."""
    panel = _make_multi_regime_panel(n_obs=48)
    # Add a 6-observation cold-start series that follows seasonal_0's launch prefix at 2x scale
    s0 = panel[panel["ts_id"] == "seasonal_0"].sort_values("ds").iloc[:6].copy()
    s0["ts_id"] = "npi_launch"
    s0["y"] = s0["y"] * 2.0
    panel_with_npi = pd.concat([panel, s0], ignore_index=True)

    prefix_matches = find_similar_series_df(
        panel_with_npi,
        query_ts_id="npi_launch",
        mode="launch_prefix",
        top_k=3,
    )
    assert not prefix_matches.empty
    top_match = str(prefix_matches.iloc[0]["neighbor_ts_id"])
    assert top_match.startswith("seasonal_")
    assert float(prefix_matches.iloc[0]["scale_ratio"]) > 0.5
    assert "velocity_sim" in prefix_matches.columns
    assert "acceleration_sim" in prefix_matches.columns

    recent_matches = find_similar_series_df(
        panel,
        query_ts_id="trend_0",
        mode="recent_window",
        window=14,
        top_k=3,
    )
    assert str(recent_matches.iloc[0]["neighbor_ts_id"]) == "trend_1"

    full_matches = find_similar_series_df(
        panel,
        query_ts_id="seasonal_0",
        mode="full_history",
        top_k=3,
    )
    assert str(full_matches.iloc[0]["neighbor_ts_id"]) == "seasonal_1"


def test_similarity_global_cluster_pooled_training_and_cluster_features() -> None:
    """@similarity_global and similarity_cluster_feature=True execute fold-safely end-to-end."""
    panel = _make_multi_regime_panel(n_obs=42)
    cfg = RunConfig.model_validate(
        {
            "run_name": "test_sim_global",
            "data": {"source_table": "p.d.t", "horizon": 7, "freq": "D"},
            "models": ["tide@similarity_global", "naive_mean@similarity_cold_start"],
            "model_params": {"tide": {"max_steps": 4, "input_size": 7}},
            "features": {
                "similarity_cluster_feature": True,
                "similarity_method": "hybrid",
                "similarity_n_clusters": 2,
                "ablation_modes": ["similarity_global"],
            },
            "backtest": {"enabled": False},
        }
    )
    expanded, _ = expand_model_variants(cfg)
    assert "tide@similarity_global" in expanded
    spec_sg = resolve_model_variant(cfg, "tide@similarity_global")
    assert spec_sg.training_mode == "similarity_global"
    assert is_panel_model("tide@similarity_global", cfg) is True

    spec_cs = resolve_model_variant(cfg, "naive_mean@similarity_cold_start")
    assert spec_cs.effective_cfg.features.cold_start_transfer is True
    assert spec_cs.effective_cfg.features.cold_start_matching == "similarity"

    # Verify fold-safe add_similarity_features and build_features / build_future_features
    cutoff = pd.Timestamp("2025-01-25")
    feat_panel = add_similarity_features(panel, cfg, cutoff_ds=cutoff)
    assert "similarity_cluster_id" in feat_panel.columns
    assert "similarity_to_medoid" in feat_panel.columns
    one_series = feat_panel[feat_panel["ts_id"] == "seasonal_0"].reset_index(drop=True)
    y_s, X_s = build_features(one_series, cfg)
    assert X_s is not None
    assert "similarity_cluster_id" in X_s.columns
    assert "similarity_to_medoid" in X_s.columns
    X_fut = build_future_features(y_s, X_s, cfg, horizon=7)
    assert X_fut is not None
    assert len(X_fut) == 7
    assert "similarity_cluster_id" in X_fut.columns

    # Run cluster-pooled panel model via run_panel
    cells = run_panel(panel, "tide@similarity_global", cfg)
    assert len(cells) == 6
    assert all(c.status == "ok" for c in cells)
    for cell in cells:
        assert "similarity_clusters" in cell.fit_diagnostics
        assert cell.fit_diagnostics["similarity_clusters"]["n_clusters"] >= 1


def test_zero_metadata_similarity_cold_start_transfer() -> None:
    """Zero-metadata cold-start NPI transfer matches donors via launch_prefix similarity."""
    panel = _make_multi_regime_panel(n_obs=42)
    s0 = panel[panel["ts_id"] == "seasonal_0"].sort_values("ds").iloc[:5].copy()
    s0["ts_id"] = "npi_zero_meta"
    s0["y"] = s0["y"] * 1.5
    full_panel = pd.concat([panel, s0], ignore_index=True)

    cs_res = forecast_cold_start_series(
        s0,
        full_panel,
        horizon=7,
        freq="D",
        method="analog_transfer",
        matching="similarity",
        similarity_method="hybrid",
        min_obs=14,
        static_covariates=[],
        hierarchy_cols=[],
    )
    assert cs_res.is_cold_start is True
    assert cs_res.matching_mode_used == "similarity"
    assert cs_res.donor_similarity_score > 0.0
    assert len(cs_res.top_donor_ts_ids) >= 1
    assert len(cs_res.predictions) == 7

    report = cold_start_transfer_df(
        full_panel,
        horizon=7,
        freq="D",
        min_obs=14,
        matching="hybrid",
        similarity_method="hybrid",
    )
    assert "matching_mode_used" in report.columns
    assert "donor_similarity_score" in report.columns
    assert "top_donor_ts_ids" in report.columns
    npi_row = report[report["ts_id"] == "npi_zero_meta"].iloc[0]
    assert bool(npi_row["is_cold_start"]) is True
    assert str(npi_row["matching_mode_used"]) == "hybrid"


def test_pathological_edge_cases_and_plot_similarity_map() -> None:
    """Verify N=1, N=2, flat/constant, all-zero, short n<5, NaNs, and plot_similarity_map."""
    dates_short = pd.date_range("2025-01-01", periods=3, freq="D")
    dates_long = pd.date_range("2025-01-01", periods=30, freq="D")

    # N=1 series
    single_df = pd.DataFrame({"ts_id": ["only_one"] * 3, "ds": dates_short, "y": [1.0, 2.0, 3.0]})
    res_1 = compute_similarity_matrix(single_df)
    assert res_1.matrix_df.shape == (1, 1)
    assert float(res_1.matrix_df.iloc[0, 0]) == pytest.approx(1.0)
    cl_1 = cluster_by_similarity_df(single_df)
    assert len(cl_1) == 1

    # Pathological panel: flat constant, all-zero, NaNs, short n=3 vs n=30
    y_with_nan = np.linspace(10.0, 40.0, 30)
    y_with_nan[5] = np.nan
    edge_panel = pd.concat(
        [
            pd.DataFrame({"ts_id": "flat_const", "ds": dates_long, "y": np.full(30, 42.0)}),
            pd.DataFrame({"ts_id": "all_zero", "ds": dates_long, "y": np.zeros(30)}),
            pd.DataFrame({"ts_id": "with_nan", "ds": dates_long, "y": y_with_nan}),
            pd.DataFrame({"ts_id": "short_3", "ds": dates_short, "y": [5.0, 10.0, 15.0]}),
        ],
        ignore_index=True,
    )
    res_edge = compute_similarity_matrix(edge_panel, method="hybrid")
    assert res_edge.matrix_df.shape == (4, 4)
    assert not res_edge.matrix_df.isna().any().any()
    cl_edge = cluster_by_similarity_df(edge_panel, min_cluster_size=2)
    assert len(cl_edge) == 4

    emb_mat, emb_ids, backend = extract_foundation_embeddings(
        {"flat_const": np.full(30, 42.0), "all_zero": np.zeros(30)}, allow_fallback=True
    )
    assert emb_mat.shape[0] == 2
    assert len(emb_ids) == 2
    assert backend != ""

    ax = plot_similarity_map(res_edge)
    assert ax is not None


def _make_multimodal_retail_panel(n_obs: int = 36) -> pd.DataFrame:
    """Create a retail panel with text descriptions, product attributes, hierarchy levels,
    and inventory stock to test coincidental seasonal twin separation vs. true substitutes."""
    dates = pd.date_range("2025-06-01", periods=n_obs, freq="D")
    t = np.arange(n_obs, dtype=float)
    summer_wave = 120.0 + 35.0 * np.sin(2.0 * np.pi * t / 7.0)
    alt_wave = 90.0 + 15.0 * np.cos(2.0 * np.pi * t / 7.0)

    skus = [
        (
            "oat_milk_vanilla_64oz",
            summer_wave,
            "Organic Oat Milk Vanilla 64oz Refrigerated Carton",
            "Grocery",
            "Dairy",
            "Plant Milk",
            "OatCo",
            4.99,
            "[0.95, 0.85, 0.05]",
            0.0,  # out of stock query item
        ),
        (
            "oat_milk_orig_64oz",
            summer_wave * 0.995,
            "Organic Oat Milk Original 64oz Refrigerated Carton",
            "Grocery",
            "Dairy",
            "Plant Milk",
            "OatCo",
            4.99,
            "[0.95, 0.86, 0.04]",
            180.0,
        ),
        (
            "almond_milk_64oz",
            summer_wave * 0.85 + 3.0,
            "Unsweetened Almond Milk 64oz Refrigerated Carton",
            "Grocery",
            "Dairy",
            "Plant Milk",
            "NutCo",
            3.99,
            "[0.82, 0.78, 0.10]",
            45.0,
        ),
        (
            "oat_milk_choco_oos",
            summer_wave * 0.90 + 3.0 * np.cos(2.0 * np.pi * t / 7.0),
            "Sweetened Chocolate Oat Milk Beverage 64oz Carton",
            "Grocery",
            "Dairy",
            "Plant Milk",
            "OatCo",
            5.49,
            "[0.88, 0.75, 0.12]",
            0.0,  # out of stock candidate
        ),
        (
            "greek_yogurt_32oz",
            alt_wave,
            "Plain Strained Greek Yogurt 32oz Family Tub",
            "Grocery",
            "Dairy",
            "Yogurt",
            "DairyFarm",
            5.99,
            "[0.45, 0.55, 0.30]",
            90.0,
        ),
        (
            "sunscreen_spf50",
            summer_wave,  # identical seasonal curve to oat_milk_vanilla_64oz!
            "Broad Spectrum Sport Sunscreen Lotion SPF 50 Waterproof",
            "Personal Care",
            "Suncare",
            "Lotion",
            "SunGuard",
            11.99,
            "[0.02, 0.05, 0.98]",
            220.0,
        ),
    ]
    frames: list[pd.DataFrame] = []
    for (
        tid,
        y_vals,
        desc,
        dept,
        cat,
        subcat,
        brand,
        price,
        emb_str,
        stock,
    ) in skus:
        frames.append(
            pd.DataFrame(
                {
                    "ts_id": tid,
                    "ds": dates,
                    "y": y_vals,
                    "description": desc,
                    "department": dept,
                    "category": cat,
                    "subcategory": subcat,
                    "brand": brand,
                    "price": price,
                    "text_embedding": emb_str,
                    "inventory_units": stock,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_multimodal_semantic_attribute_and_hierarchy_lca_similarity() -> None:
    """F16: Multi-modal text TF-IDF/LSA, Gower attributes, and Wu-Palmer LCA tree similarity
    separate coincidental seasonal twins (Oat Milk vs. Sunscreen) while ranking true substitutes #1.
    """
    panel = _make_multimodal_retail_panel()

    # Pure behavioral similarity treats oat_milk_vanilla_64oz and sunscreen_spf50 as identical twins
    behav_res = compute_similarity_matrix(panel, method="hybrid")
    assert float(behav_res.matrix_df.loc["oat_milk_vanilla_64oz", "sunscreen_spf50"]) > 0.90

    # Multi-modal similarity with text, attributes, and hierarchy LCA
    mm_res = compute_similarity_matrix(
        panel,
        method="hybrid",
        text_cols=["description"],
        attribute_cols=["brand", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
    )
    assert {"semantic_sim", "hierarchy_sim"}.issubset(mm_res.pairs_df.columns)

    pairs_q = mm_res.pairs_df[mm_res.pairs_df["ts_id"] == "oat_milk_vanilla_64oz"].set_index(
        "neighbor_ts_id"
    )
    # Wu-Palmer LCA hierarchy similarity:
    # Same leaf subcategory (depth 3 / 3) -> 1.0
    assert float(pairs_q.loc["oat_milk_orig_64oz", "hierarchy_sim"]) == pytest.approx(1.0)
    # Shared department + category (2 * 2 / (3 + 3) = 2/3)
    assert float(pairs_q.loc["greek_yogurt_32oz", "hierarchy_sim"]) == pytest.approx(2.0 / 3.0)
    # Disjoint department -> 0.0
    assert float(pairs_q.loc["sunscreen_spf50", "hierarchy_sim"]) == pytest.approx(0.0)

    # Semantic similarity separates Oat Milk from Sunscreen and ranks Oat Milk Original highest
    assert float(pairs_q.loc["oat_milk_orig_64oz", "semantic_sim"]) > 0.75
    assert float(pairs_q.loc["sunscreen_spf50", "semantic_sim"]) < 0.15

    # Composite multi-modal similarity ranks Oat Milk Original #1 and pushes Sunscreen far down
    sim_orig = float(mm_res.matrix_df.loc["oat_milk_vanilla_64oz", "oat_milk_orig_64oz"])
    sim_sun = float(mm_res.matrix_df.loc["oat_milk_vanilla_64oz", "sunscreen_spf50"])
    assert sim_orig > 0.80
    assert sim_sun < 0.35

    # Verify embedding_col and within_hierarchy_level hard constraint in find_similar_series_df
    nbrs = find_similar_series_df(
        panel,
        query_ts_id="oat_milk_vanilla_64oz",
        mode="recent_window",
        window=14,
        top_k=5,
        text_cols=["description"],
        embedding_col="text_embedding",
        attribute_cols=["brand", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        within_hierarchy_level="category",
    )
    assert {"semantic_sim", "hierarchy_sim"}.issubset(nbrs.columns)
    assert "sunscreen_spf50" not in set(nbrs["neighbor_ts_id"])
    assert str(nbrs.iloc[0]["neighbor_ts_id"]) == "oat_milk_orig_64oz"

    # Verify embed_series_df and cluster_by_similarity_df accept multi-modal parameters
    emb_df = embed_series_df(
        panel,
        text_cols=["description"],
        attribute_cols=["brand", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
    )
    assert len(emb_df) == 6
    clusters_df = cluster_by_similarity_df(
        panel,
        n_clusters=2,
        min_cluster_size=1,
        text_cols=["description"],
        attribute_cols=["brand", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
    )
    by_id = clusters_df.set_index("ts_id")
    assert (
        by_id.loc["oat_milk_vanilla_64oz", "similarity_cluster_id"]
        == by_id.loc["oat_milk_orig_64oz", "similarity_cluster_id"]
    )
    assert (
        by_id.loc["oat_milk_vanilla_64oz", "similarity_cluster_id"]
        != by_id.loc["sunscreen_spf50", "similarity_cluster_id"]
    )


def test_recommend_substitutes_df_stockout_aware_ranking() -> None:
    """F18a: recommend_substitutes_df ranks in-stock substitutes #1 and filters out-of-stock."""
    panel = _make_multimodal_retail_panel()

    subs_in_stock = recommend_substitutes_df(
        panel,
        query_ts_id="oat_milk_vanilla_64oz",
        top_k=5,
        text_cols=["description"],
        attribute_cols=["brand", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        within_hierarchy_level="subcategory",
        inventory_col="inventory_units",
        exclude_out_of_stock=True,
    )
    expected_cols = {
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
    assert expected_cols.issubset(subs_in_stock.columns)
    assert not subs_in_stock.empty
    # Out-of-stock oat_milk_choco_oos and cross-subcategory items are excluded
    assert "oat_milk_choco_oos" not in set(subs_in_stock["substitute_ts_id"])
    assert "sunscreen_spf50" not in set(subs_in_stock["substitute_ts_id"])
    assert str(subs_in_stock.iloc[0]["substitute_ts_id"]) == "oat_milk_orig_64oz"
    assert int(subs_in_stock.iloc[0]["rank"]) == 1
    assert str(subs_in_stock.iloc[0]["stock_status"]) in {"in_stock_surplus", "adequate_stock"}
    assert float(subs_in_stock.iloc[0]["substitution_score"]) > float(
        subs_in_stock.iloc[1]["substitution_score"]
    )

    # Including out-of-stock and safety_stock items tests all 4 stock_status enum literals
    inv_df = pd.DataFrame(
        [
            {"ts_id": "oat_milk_vanilla_64oz", "on_hand": 100.0, "safety_stock": 20.0},
            {"ts_id": "oat_milk_orig_64oz", "on_hand": 120.0, "safety_stock": 25.0},
            {"ts_id": "almond_milk_64oz", "on_hand": 18.0, "safety_stock": 25.0},
            {"ts_id": "oat_milk_choco_oos", "on_hand": 0.0, "safety_stock": 15.0},
            {"ts_id": "dairy_milk_whole_64oz", "on_hand": 24.0, "safety_stock": 20.0},
            {"ts_id": "sunscreen_spf50", "on_hand": 200.0, "safety_stock": 30.0},
        ]
    )
    subs_all = recommend_substitutes_df(
        panel,
        inv_df,
        query_ts_id=["oat_milk_vanilla_64oz"],
        top_k=5,
        text_cols=["description"],
        attribute_cols=["brand", "price"],
        hierarchy_cols=["department", "category", "subcategory"],
        within_hierarchy_level="subcategory",
        require_in_stock=False,
    )
    valid_stock_statuses = {
        "in_stock_surplus",
        "adequate_stock",
        "low_stock_risk",
        "out_of_stock",
    }
    assert set(subs_all["stock_status"]).issubset(valid_stock_statuses)
    by_sub = subs_all.set_index("substitute_ts_id")
    assert str(by_sub.loc["oat_milk_choco_oos", "stock_status"]) == "out_of_stock"
    assert float(by_sub.loc["oat_milk_choco_oos", "substitution_score"]) == pytest.approx(0.0)
    assert str(by_sub.loc["almond_milk_64oz", "stock_status"]) == "low_stock_risk"


def test_detect_emerging_trends_df_breakout_and_bellwether_lead_lag() -> None:
    """F18b: detect_emerging_trends_df identifies breakouts and bellwether lead-lag followers."""
    assert emerging_trends_df is detect_emerging_trends_df

    n_obs = 48
    dates = pd.date_range("2025-01-01", periods=n_obs, freq="D")
    t = np.arange(n_obs, dtype=float)
    rng = np.random.default_rng(7)

    # Bellwether leader: steady baseline until day 24, then steep accelerating breakout
    ramp = np.maximum(0.0, t - 24.0)
    y_leader = 30.0 + 1.8 * ramp + 0.18 * (ramp**2) + 2.0 * np.sin(2.0 * np.pi * t / 7.0)
    # Follower lags leader by 4 days
    ramp_lag = np.maximum(0.0, (t - 4.0) - 24.0)
    y_follower = (
        30.0 + 1.8 * ramp_lag + 0.18 * (ramp_lag**2) + 2.0 * np.sin(2.0 * np.pi * (t - 4.0) / 7.0)
    )
    # Steady stationary series
    y_steady = 50.0 + 3.0 * np.sin(2.0 * np.pi * t / 7.0) + rng.normal(0.0, 0.5, size=n_obs)
    # Declining series
    y_decline = np.maximum(5.0, 120.0 - 2.0 * t + rng.normal(0.0, 0.5, size=n_obs))

    panel = pd.concat(
        [
            pd.DataFrame({"ts_id": "leader_viral", "ds": dates, "y": y_leader}),
            pd.DataFrame({"ts_id": "follower_lag4", "ds": dates, "y": y_follower}),
            pd.DataFrame({"ts_id": "steady_mature", "ds": dates, "y": y_steady}),
            pd.DataFrame({"ts_id": "declining_sku", "ds": dates, "y": y_decline}),
        ],
        ignore_index=True,
    )

    trends = detect_emerging_trends_df(panel, window=14, baseline_window=24, max_lag=7)
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
    assert required_cols.issubset(trends.columns)
    assert len(trends) == 4

    valid_regimes = {
        "accelerating_breakout",
        "steady_growth",
        "maturing_plateau",
        "declining",
        "stable",
    }
    valid_actions = {
        "promote_trend_leader",
        "co_promote_lagging_twin",
        "replenish_breakout_risk",
        "clearance_candidate",
        "hold_baseline",
    }
    assert set(trends["trend_regime"]).issubset(valid_regimes)
    assert set(trends["recommended_merchandising_action"]).issubset(valid_actions)

    by_id = trends.set_index("ts_id")
    assert str(trends.iloc[0]["ts_id"]) in {"leader_viral", "follower_lag4"}
    assert str(by_id.loc["leader_viral", "trend_regime"]) == "accelerating_breakout"
    assert float(by_id.loc["leader_viral", "breakout_zscore"]) > 2.0
    assert float(by_id.loc["leader_viral", "recent_velocity"]) > 0.0
    assert str(by_id.loc["declining_sku", "trend_regime"]) == "declining"
    assert (
        str(by_id.loc["declining_sku", "recommended_merchandising_action"]) == "clearance_candidate"
    )
    # Follower lags behind leader_viral and receives co-promotion recommendation
    assert str(by_id.loc["follower_lag4", "leading_bellwether_ts_id"]) == "leader_viral"
    assert int(by_id.loc["follower_lag4", "lag_behind_leader"]) > 0
    assert (
        str(by_id.loc["follower_lag4", "recommended_merchandising_action"])
        == "co_promote_lagging_twin"
    )
    assert "follower_lag4" in str(by_id.loc["leader_viral", "co_promote_candidates"])


def test_sdk_playground_and_report_phase2_surfaces(tmp_path) -> None:
    """Verify top-level exports, Forecaster, Playground, and export_report Phase 2 surfaces."""
    import scale_forecasting as sf
    from scale_forecasting.playground import bakeoff, run_model
    from scale_forecasting.report import export_report

    assert sf.recommend_substitutes_df is recommend_substitutes_df
    assert sf.detect_emerging_trends_df is detect_emerging_trends_df
    assert sf.emerging_trends_df is detect_emerging_trends_df

    panel = _make_multimodal_retail_panel()
    f = sf.Forecaster.from_dict(
        {
            "run_name": "sim_p2_sdk_test",
            "data": {"source_table": "source_series", "horizon": 7, "freq": "D"},
            "models": ["naive_mean"],
        }
    )
    subs_f = f.recommend_substitutes_df(
        data=panel,
        query_ts_id="oat_milk_vanilla_64oz",
        text_cols=["description"],
        hierarchy_cols=["department", "category", "subcategory"],
    )
    assert not subs_f.empty
    trends_f = f.emerging_trends_df(data=panel, window=14)
    assert not trends_f.empty
    assert len(f.detect_emerging_trends_df(data=panel, window=14)) == len(trends_f)
    assert not f.cluster_by_similarity_df(data=panel).empty
    assert not f.embed_series_df(data=panel).empty

    bench = bakeoff(models=["naive_mean"], horizon=7)
    subs_b = bench.recommend_substitutes_df(
        data=panel,
        query_ts_id="oat_milk_vanilla_64oz",
        text_cols=["description"],
    )
    assert not subs_b.empty
    trends_b = bench.emerging_trends_df(window=14)
    assert not trends_b.empty
    assert len(bench.detect_emerging_trends_df(window=14)) == len(trends_b)
    assert not bench.cluster_by_similarity_df().empty
    assert not bench.embed_series_df().empty

    pg_run = run_model("naive_mean", horizon=7)
    assert not pg_run.recommend_substitutes_df(
        data=panel, query_ts_id="oat_milk_vanilla_64oz"
    ).empty
    assert not pg_run.emerging_trends_df(data=panel, window=14).empty
    assert not pg_run.detect_emerging_trends_df(data=panel, window=14).empty
    assert not pg_run.cluster_by_similarity_df(data=panel).empty
    assert not pg_run.embed_series_df(data=panel).empty

    out_path = export_report(
        bench,
        tmp_path / "report.html",
        substitutes_df=subs_b,
        emerging_trends_df=trends_b,
    )
    html_text = out_path.read_text(encoding="utf-8")
    assert "SKU Substitute" in html_text
    assert "Emerging Trends" in html_text
