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
    embed_series_df,
    extract_foundation_embeddings,
    find_similar_series_df,
    plot_similarity_map,
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
