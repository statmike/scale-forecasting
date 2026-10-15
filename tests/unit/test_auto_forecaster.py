"""Unit and contract tests for Track G:
- Synthetic DGP Benchmark Suite (`scale_forecasting.benchmarks`)
- 7-Runtime Parametric Wall-Clock Duration Predictor (`scale_forecasting.profiling.duration`)
- Glass-Box Auto-Forecaster, Astronomy Exposure Tiers & Closed-Loop `autofocus`
  (`scale_forecasting.auto`, `scale_forecasting.auto_policy`)
- Human-first release notes generator (`scale_forecasting.agent_surfaces.render_release_notes`)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scale_forecasting import (
    BENCHMARK_NAMES,
    EXPOSURE_TIERS,
    SCALE_REGIMES,
    Forecaster,
    RunConfig,
    auto_forecast,
    benchmark_catalog,
    build_config,
    calibrate_duration_from_frames,
    classify_scale_regime,
    inspect_panel_for_auto,
    list_benchmarks,
    load_benchmark,
    predict_run_duration,
    select_default_baselines,
    select_hierarchy_reconciliation_methods,
)
from scale_forecasting.agent_surfaces import render_release_notes
from scale_forecasting.errors import ConfigError


def test_benchmark_suite_all_18_dgp_panels() -> None:
    names = list_benchmarks()
    assert len(names) == 18
    assert tuple(names) == BENCHMARK_NAMES

    cat = benchmark_catalog()
    assert len(cat) == 18
    assert {"name", "title", "expected_regime", "true_covariates", "decoy_covariates"}.issubset(
        cat.columns
    )

    for b_name in BENCHMARK_NAMES:
        ds = load_benchmark(b_name, n_series=3)
        assert ds.spec.name == b_name
        assert not ds.df.empty
        assert not ds.panel_df.empty
        assert {"ts_id", "ds", "y"}.issubset(ds.df.columns)
        summary = ds.summary_dict()
        assert summary["name"] == b_name
        assert summary["n_series"] >= 3

    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_benchmark("dgp_99_nonexistent")


def test_predict_run_duration_all_7_runtimes_and_calibration() -> None:
    runtimes_and_modes = [
        ("spark", {}),
        ("ray", {"ray_mode": "vertex"}),
        ("vertex", {}),
        ("gce", {"workers": 1}),
        ("gke", {"gke_mode": "job"}),
        ("vertex_automl", {"automl_mode": "training_job"}),
    ]
    for rt, extra in runtimes_and_modes:
        models = ["vertex_tide"] if rt == "vertex_automl" else ["theta", "lightgbm"]
        fam_block = (
            {"automl": {"runtime": rt, **extra}}
            if rt == "vertex_automl"
            else {
                "statistical": {"runtime": rt, **extra},
                "ml": {"runtime": rt, **extra},
            }
        )
        cfg = RunConfig.model_validate(
            {
                "run_name": f"dur-{rt}",
                "data": {"source_table": "source_series", "horizon": 14},
                "models": models,
                "compute": {
                    "families": fam_block,
                },
            }
        )
        est = predict_run_duration(cfg, n_series=100, median_n_obs=365.0)
        assert est.total_p50_seconds > 0.0
        assert est.total_p90_seconds >= est.total_p50_seconds
        assert est.critical_path_family in {b.family for b in est.family_breakdowns}
        d = est.to_dict()
        assert d["n_series"] == 100
        assert "family_breakdowns" in d

    # Native BigQuery runtime
    bq_cfg = RunConfig.model_validate(
        {
            "run_name": "dur-bq",
            "data": {"source_table": "source_series", "horizon": 14},
            "models": ["arima_plus", "timesfm"],
        }
    )
    bq_est = predict_run_duration(bq_cfg, n_series=200)
    assert bq_est.critical_path_family == "native"
    assert bq_est.total_p50_seconds > 10.0

    # Local in-process execution with borrowed models discount
    local_cfg = RunConfig.model_validate(
        {
            "run_name": "dur-local",
            "data": {"source_table": "dataframe_panel", "horizon": 14},
            "models": ["naive_seasonal", "theta", "lightgbm"],
        }
    )
    est_full = predict_run_duration(local_cfg, n_series=10, where="local")
    est_borrowed = predict_run_duration(
        local_cfg,
        n_series=10,
        where="local",
        borrowed_models=["naive_seasonal", "theta"],
    )
    assert est_borrowed.execute_p50_seconds < est_full.execute_p50_seconds

    # Empirical calibration from telemetry frames
    jobs_df = pd.DataFrame(
        [
            {"runtime": "vertex", "provision_seconds": 175.0},
            {"runtime": "vertex", "provision_seconds": 195.0},
            {"runtime": "spark", "provision_seconds": 70.0},
        ]
    )
    cells_df = pd.DataFrame(
        [
            {"model_type": "theta", "fit_seconds": 0.05, "n_obs": 365},
            {"model_type": "theta", "fit_seconds": 0.06, "n_obs": 365},
            {"model_type": "lightgbm", "fit_seconds": 0.25, "n_obs": 365},
        ]
    )
    cal = calibrate_duration_from_frames(jobs_df, cells_df, project_id="demo-proj")
    assert cal.n_jobs_analyzed == 3
    assert cal.n_cells_analyzed == 3
    assert "vertex" in cal.provisioning_priors
    assert "theta" in cal.model_fit_priors


def test_build_config_tiers_sandboxes_goals_and_budgets(tmp_path: Path) -> None:
    assert tuple(EXPOSURE_TIERS.keys()) == ("first_light", "wide_field", "deep_sky", "deep_field")

    ds = load_benchmark("dgp_04_true_vs_decoy_covs", n_series=4)
    profile = inspect_panel_for_auto(ds.panel_df)
    assert profile.n_series == 4
    assert "promo_depth" in profile.future_covariates

    for tier in ("first_light", "wide_field", "deep_sky", "deep_field"):
        bp = Forecaster.build_config(
            ds.panel_df,
            tier=tier,
            where="local",
            goal="accuracy",
        )
        assert bp.tier == tier
        assert bp.where == "local"
        assert len(bp.config.models) >= 2
        assert not bp.rationale_df.empty
        assert bp.estimated_minutes_p50 > 0.0
        assert "Glass-Box Blueprint" in bp.explain()

    # Service-level goal + cost/time budget clamping + file serialization
    bp_sl = build_config(
        ds.panel_df,
        tier="deep_sky",
        where="cloud_cpu",
        goal="service_level",
        max_minutes=0.01,
        max_usd=0.01,
        only_available_models=False,
    )
    assert bp_sl.config.backtest.decision_metric == "pinball"
    assert 0.95 in bp_sl.config.output.quantiles
    assert any(bp_sl.rationale_df["category"] == "budget_guardrail")

    cfg_path = tmp_path / "compiled_auto.json"
    bp_sl.to_file(cfg_path)
    loaded = RunConfig.model_validate_json(cfg_path.read_text(encoding="utf-8"))
    assert loaded.run_name == bp_sl.config.run_name
    assert json.loads(bp_sl.to_json())["run_name"] == bp_sl.config.run_name
    assert bp_sl.to_dict()["tier"] == "deep_sky"

    with pytest.raises(ConfigError, match="Unknown exposure tier"):
        build_config(ds.panel_df, tier="invalid_tier")
    with pytest.raises(ConfigError, match="Unknown where"):
        build_config(ds.panel_df, where="invalid_where")
    with pytest.raises(ConfigError, match="Unknown goal"):
        build_config(ds.panel_df, goal="invalid_goal")


def test_forecaster_auto_closed_loop_autofocus_and_early_stopping() -> None:
    # 1. White-noise panel stops cleanly after Pass 1 (zero wasted passes)
    ds_wn = load_benchmark("dgp_12_already_optimal_white_noise", n_series=3)
    f_wn = Forecaster.auto(ds_wn.panel_df, tier="first_light", autofocus=True, max_passes=2)
    assert f_wn.auto_session is not None
    assert len(f_wn.auto_session.passes) == 1
    prev_wn = f_wn.preview_next_loop()
    assert prev_wn.should_continue is False

    # 2. Multiplicative log panel triggers @transform_log1p in Pass 2 and improves WAPE
    ds_log = load_benchmark("dgp_02_multiplicative_log", n_series=3)
    f_log = Forecaster.auto(ds_log.panel_df, tier="first_light", autofocus=True, max_passes=2)
    assert f_log.auto_session is not None
    assert len(f_log.auto_session.passes) == 2
    p1_log, p2_log = f_log.auto_session.passes
    assert p2_log.best_score < p1_log.best_score
    assert p2_log.models_borrowed >= 2
    assert any("inject_transform_log1p" in m for m in p2_log.mutations_applied)

    # 3. Decoy covariate panel prunes noise covariates in Pass 2 and improves WAPE
    ds_cov = load_benchmark("dgp_04_true_vs_decoy_covs", n_series=3)
    f_cov = auto_forecast(
        ds_cov.panel_df,
        tier="first_light",
        autofocus=True,
        autofocus_mode="linear",
        max_passes=2,
    )
    assert f_cov.auto_session is not None
    assert len(f_cov.auto_session.passes) == 2
    p1_cov, p2_cov = f_cov.auto_session.passes
    assert p2_cov.best_score < p1_cov.best_score
    assert set(p2_cov.config.features.future_covariates) == {"promo_depth", "price_index"}

    ledger_df = f_cov.autofocus_ledger_df()
    assert len(ledger_df) == 2
    assert list(ledger_df["pass_index"]) == [1, 2]
    assert f_cov.blueprint is not None
    assert not f_cov.leaderboard_df().empty
    assert not f_cov.predictions_df().empty

    # 4. Step-by-step interactive continuation (`step_by_step=True` + `step()`)
    f_step = Forecaster.auto(
        ds_log.panel_df,
        tier="first_light",
        autofocus=True,
        step_by_step=True,
        max_passes=2,
    )
    assert len(f_step.autofocus_ledger_df()) == 1
    preview = f_step.preview_next_loop()
    assert preview.should_continue is True
    assert preview.confidence >= 0.35
    f_step2 = f_step.step(steps=1)
    assert len(f_step2.autofocus_ledger_df()) == 2
    assert not f_step2.autofocus_diff_df().empty
    assert not f_step2.explain_auto_decisions_df().empty


def test_user_constraint_pinning_and_blueprint_overrides() -> None:
    ds = load_benchmark("dgp_04_true_vs_decoy_covs", n_series=3)

    # Pin models, decision_metric, n_folds, hpo, ensemble, calibration, explainability
    bp = Forecaster.build_config(
        ds.panel_df,
        tier="wide_field",
        where="local",
        models=["theta", "lightgbm"],
        include_models=["regression_lags"],
        exclude_models=["xgboost"],
        decision_metric="mase",
        n_folds=3,
        hpo=True,
        ensemble=["inverse_error", "nnls"],
        explainability=False,
        calibration=False,
        overrides={"run_name": "pinned-custom-run"},
    )
    assert bp.config.run_name == "pinned-custom-run"
    assert bp.config.models == ["theta", "lightgbm", "regression_lags"]
    assert bp.config.backtest.decision_metric == "mase"
    assert bp.config.backtest.n_folds == 3
    assert bp.config.hpo.enabled is True
    assert bp.config.ensemble.strategies == ["inverse_error", "nnls"]
    assert bp.config.output.point_forecast == "raw"
    assert "user_constraint" in set(bp.rationale_df["category"])

    # Fluent ConfigBlueprint.with_overrides()
    bp2 = bp.with_overrides(run_name="overridden-run", backtest={"n_folds": 2})
    assert bp2.config.run_name == "overridden-run"
    assert bp2.config.backtest.n_folds == 2

    # Pinned models respected across autofocus passes
    f_pinned = Forecaster.auto(
        ds.panel_df,
        tier="first_light",
        models=["theta", "lightgbm"],
        decision_metric="mase",
        autofocus=True,
        autofocus_mode="linear",
        max_passes=2,
    )
    assert f_pinned.config.backtest.decision_metric == "mase"
    for m in f_pinned.config.models:
        base_m = m.split("@")[0]
        assert base_m in {"naive_seasonal", "theta", "lightgbm"}


def test_multi_branch_dag_autofocus_and_all_4_specialist_regimes() -> None:
    from scale_forecasting.reconciliation import build_hierarchy, verify_coherence

    # 1. Multi-branch DAG (`1 -> 2A -> 2B -> 3`) on `dgp_04_true_vs_decoy_covs`
    ds_04 = load_benchmark("dgp_04_true_vs_decoy_covs", n_series=3)
    f_04 = Forecaster.auto(
        ds_04.panel_df,
        tier="first_light",
        autofocus=True,
        autofocus_mode="dag",
        max_passes=3,
    )
    assert f_04.auto_session is not None
    pass_ids_04 = [p.pass_id for p in f_04.auto_session.passes]
    assert pass_ids_04 == ["1", "2A", "2B", "3"]
    champ_04 = f_04.auto_session.champion_record()
    assert champ_04.is_session_champion is True
    assert champ_04.best_score < f_04.auto_session.passes[0].best_score
    diff_df = f_04.autofocus_diff_df()
    assert not diff_df.empty
    assert {
        "pass_id",
        "parent_pass_ids",
        "branch_hypothesis",
        "action",
        "parameter",
        "old_value",
        "new_value",
        "delta_vs_pass1_pct",
        "is_session_champion",
    }.issubset(diff_df.columns)

    # 2. Intermittent lumpy (`dgp_03`) escalates in Pass 2 and improves MASE
    ds_03 = load_benchmark("dgp_03_intermittent_lumpy", n_series=4)
    f_03 = Forecaster.auto(ds_03.panel_df, tier="first_light", autofocus=True, max_passes=2)
    assert f_03.auto_session is not None
    assert len(f_03.auto_session.passes) >= 2
    assert f_03.auto_session.champion_record().best_score < f_03.auto_session.passes[0].best_score

    # 3. Hierarchical non-negative (`dgp_08`) preserves exact additive coherence == 0.0
    ds_08 = load_benchmark("dgp_08_hierarchical_nonneg", n_series=6)
    f_08 = Forecaster.auto(ds_08.panel_df, tier="first_light", autofocus=True, max_passes=3)
    assert f_08.auto_session is not None
    champ_08 = f_08.auto_session.champion_record()
    assert champ_08.best_score < f_08.auto_session.passes[0].best_score
    _, hspec = build_hierarchy(ds_08.df, champ_08.config)
    champ_preds = f_08.auto_session.raw_predictions_df
    assert champ_preds is not None
    champ_preds = champ_preds[champ_preds["model_type"] == champ_08.best_model]
    assert float(verify_coherence(champ_preds, hspec)) < 1e-6

    # 4. Newsvendor service-level (`dgp_10`) populates pinball metric and improves score
    ds_10 = load_benchmark("dgp_10_newsvendor_service", n_series=4)
    f_10 = Forecaster.auto(
        ds_10.panel_df,
        tier="first_light",
        goal="service_level",
        autofocus=True,
        max_passes=2,
    )
    assert f_10.auto_session is not None
    champ_10 = f_10.auto_session.champion_record()
    assert champ_10.decision_metric == "pinball"
    assert champ_10.best_score < f_10.auto_session.passes[0].best_score
    assert champ_10.fva_vs_naive_pct > 0.0

    # 5. Cold-start mixed (`dgp_11`) escalates to cross-series pooling and improves WAPE
    ds_11 = load_benchmark("dgp_11_cold_start_mixed", n_series=4)
    f_11 = Forecaster.auto(ds_11.panel_df, tier="first_light", autofocus=True, max_passes=2)
    assert f_11.auto_session is not None
    assert f_11.auto_session.champion_record().best_score < f_11.auto_session.passes[0].best_score


def test_round3_scale_regimes_baselines_hierarchy_and_compound_variants() -> None:
    from dataclasses import replace

    from scale_forecasting.features import resolve_model_variant

    # 1. Scale regimes across 1 .. 50,000 series
    assert tuple(SCALE_REGIMES.keys()) == ("micro", "small", "medium", "large", "hyperscale")
    assert classify_scale_regime(5) == "micro"
    assert classify_scale_regime(50) == "small"
    assert classify_scale_regime(500) == "medium"
    assert classify_scale_regime(5000) == "large"
    assert classify_scale_regime(50000) == "hyperscale"

    ds_15 = load_benchmark("dgp_15_scale_crossover_panel", n_series=16)
    prof_15 = inspect_panel_for_auto(ds_15.panel_df)
    assert prof_15.scale_regime == "small"
    assert prof_15.cross_series_corr > 0.35

    # Simulate large and hyperscale profiles to verify O(T^2) pruning, worker sharding, and top_k
    prof_large = replace(prof_15, n_series=3500, source_table="proj.ds.tbl")
    bp_large = Forecaster.build_config(
        prof_large, tier="deep_sky", where="cloud_cpu", only_available_models=False
    )
    assert bp_large.config.ensemble.top_k == 5
    assert "autoets" not in bp_large.config.models
    assert "stl_bagging" not in bp_large.config.models
    assert any(bp_large.rationale_df["category"] == "scale_guardrail")

    prof_hyper = replace(prof_15, n_series=30000, source_table="proj.ds.tbl")
    bp_hyper = Forecaster.build_config(
        prof_hyper, tier="deep_field", where="cloud_all", only_available_models=False
    )
    assert bp_hyper.config.ensemble.top_k == 3
    assert bp_hyper.config.backtest.n_folds <= 2
    assert bp_hyper.config.hpo.enabled is False

    # Micro panel on deep_field selects per_series HPO granularity
    prof_micro = replace(prof_15, n_series=4)
    bp_micro = Forecaster.build_config(
        prof_micro, tier="deep_field", where="local", only_available_models=False
    )
    assert bp_micro.config.hpo.granularity == "per_series"
    assert bp_micro.config.features.fourier is True

    # 2. Composable compound @<tag> syntax for dual-mode foundation & ML models
    base_cfg = RunConfig.model_validate(
        {
            "run_name": "compound-test",
            "data": {"source_table": "source_series", "horizon": 14},
            "models": [
                "timesfm_torch@v2.5_univariate",
                "timesfm_torch@global_v2.5_cov_all",
                "chronos_bolt@tiny_global",
                "chronos2@global_univariate",
                "lightgbm@direct_transform_log1p",
                "arima_plus@univariate",
                "timesfm@univariate",
            ],
            "features": {
                "future_covariates": ["promo_depth", "price_index"],
                "static_covariates": ["category"],
            },
        }
    )
    v_tfm_uni = resolve_model_variant(base_cfg, "timesfm_torch@v2.5_univariate")
    assert v_tfm_uni.base_model == "timesfm_torch"
    assert v_tfm_uni.clean_params["version"] == "2.5"
    assert v_tfm_uni.is_univariate is True
    assert list(v_tfm_uni.effective_cfg.features.future_covariates) == []

    v_tfm_multi = resolve_model_variant(base_cfg, "timesfm_torch@global_v2.5_cov_all")
    assert v_tfm_multi.training_mode == "global"
    assert v_tfm_multi.clean_params["version"] == "2.5"
    assert v_tfm_multi.is_univariate is False
    assert list(v_tfm_multi.effective_cfg.features.future_covariates) == [
        "promo_depth",
        "price_index",
    ]

    v_cb = resolve_model_variant(base_cfg, "chronos_bolt@tiny_global")
    assert v_cb.clean_params["size"] == "tiny"
    assert v_cb.training_mode == "global"

    v_lgb = resolve_model_variant(base_cfg, "lightgbm@direct_transform_log1p")
    assert v_lgb.clean_params["strategy"] == "direct"
    assert v_lgb.effective_cfg.features.transform == "log1p"

    # 3. All-Decoy Covariates -> Fallback to Plain Univariate & Beaten Challenger Highlighting
    ds_13 = load_benchmark("dgp_13_all_decoy_univariate_wins", n_series=3)
    f_13 = Forecaster.auto(
        ds_13.panel_df,
        tier="first_light",
        baselines=["naive_seasonal", "theta"],
        autofocus=True,
        autofocus_mode="linear",
        max_passes=2,
    )
    assert f_13.auto_session is not None
    assert len(f_13.auto_session.passes) == 2
    p2_13 = f_13.auto_session.passes[-1]
    assert list(p2_13.config.features.future_covariates) == []
    assert "fallback_to_plain_univariate" in p2_13.mutations_applied
    bval_13 = f_13.baseline_value_df()
    assert not bval_13.empty
    assert {
        "baseline_model",
        "baseline_role",
        "champion_model",
        "champion_is_univariate",
        "fva_vs_baseline_pct",
    }.issubset(bval_13.columns)
    assert any(bval_13["baseline_role"] == "beaten_complex_challenger")

    # 4. Scale-aware hierarchy reconciliation methods & hierarchy_value_df()
    ds_14 = load_benchmark("dgp_14_large_hierarchy_intermittent_leaves", n_series=8)
    prof_14 = inspect_panel_for_auto(ds_14.panel_df)
    assert prof_14.hierarchy_node_count > 8
    rec_14 = select_hierarchy_reconciliation_methods(prof_14, tier="wide_field")
    assert "top_down" in rec_14
    rec_large = select_hierarchy_reconciliation_methods(
        replace(prof_14, hierarchy_node_count=800), tier="wide_field"
    )
    assert rec_large == ["wls_var", "wls_struct", "bottom_up", "top_down"]

    f_14 = Forecaster.auto(ds_14.panel_df, tier="first_light", autofocus=False)
    hval_14 = f_14.hierarchy_value_df()
    assert not hval_14.empty
    assert "hierarchy_level" in hval_14.columns
    assert "All Hierarchy Nodes" in set(hval_14["hierarchy_level"])
    assert not f_14.hierarchy_df().empty

    # 5. Monthly trading days, holidays, leading-zero trimming, grid_policy, and exog_lags
    ds_16 = load_benchmark("dgp_16_monthly_trading_days", n_series=3)
    bp_16 = Forecaster.build_config(ds_16.panel_df, tier="wide_field", holidays=["US"])
    assert bp_16.config.features.trading_days is True
    assert bp_16.config.features.holidays == ["US"]

    ds_17 = load_benchmark("dgp_17_missing_gaps_and_leading_zeros", n_series=4)
    bp_17 = Forecaster.build_config(ds_17.panel_df, tier="wide_field")
    assert bp_17.config.data.trim_leading_zeros is True
    assert bp_17.config.data.grid_policy in ("linear", "fill_zero")

    ds_18 = load_benchmark("dgp_18_promo_lag_pullforward", n_series=3)
    f_18 = Forecaster.auto(
        ds_18.panel_df,
        tier="first_light",
        autofocus=True,
        autofocus_mode="linear",
        max_passes=2,
    )
    assert f_18.auto_session is not None
    p2_18 = f_18.auto_session.passes[-1]
    assert "synthesize_exog_lags" in p2_18.mutations_applied
    assert "promo_flag" in p2_18.config.features.exog_lags

    # Verify default baseline ladder helper directly
    ladder = select_default_baselines(
        inspect_panel_for_auto(load_benchmark("dgp_04_true_vs_decoy_covs", n_series=3).panel_df),
        where="local",
        available_models=["naive_seasonal", "theta", "chronos_bolt", "lightgbm"],
        selected_models=["naive_seasonal", "theta", "lightgbm"],
    )
    roles = [b["baseline_role"] for b in ladder]
    assert "naive_anchor" in roles
    assert "plain_univariate_stat" in roles
    assert "zero_shot_foundation" in roles
    assert "same_model_univariate" in roles


def test_release_notes_generator() -> None:
    notes = render_release_notes("1.2.1")
    assert "## Installation & Upgrade" in notes
    assert "pip install --upgrade" in notes
    assert "## Highlights at a Glance" in notes
    assert "## Detailed Release Notes (`v1.2.1`)" in notes


def test_auto_forecaster_all_run_paths_cli_airflow_sdk_mcp(tmp_path) -> None:
    import ast

    from scale_forecasting import main as main_mod
    from scale_forecasting.airflow_emit import emit_autofocus_airflow_dag
    from scale_forecasting.mcp import McpServer

    ds = load_benchmark("dgp_01_smooth_seasonal", n_series=3)
    bp = Forecaster.build_config(ds.panel_df, tier="first_light")

    # 1. Blueprint & Forecaster Airflow DAG emission (single-pass tier & multi-branch autofocus)
    dag_single = bp.emit_airflow_dag(autofocus=False)
    ast.parse(dag_single)
    assert "begin_run" in dag_single

    dag_af = bp.emit_airflow_dag(autofocus=True, autofocus_mode="dag", max_passes=3)
    ast.parse(dag_af)
    assert "autofocus_pass_1" in dag_af
    assert "autofocus_branch_2a" in dag_af
    assert "autofocus_pass_3_synthesis" in dag_af

    dag_linear = emit_autofocus_airflow_dag(
        bp.config,
        "gs://bkt/staging/cfg.json",
        tier="wide_field",
        autofocus_mode="linear",
        max_passes=3,
    )
    ast.parse(dag_linear)
    assert "autofocus_pass_1 >> autofocus_pass_2 >> autofocus_pass_3_synthesis" in dag_linear

    # 2. CLI `--auto --benchmark ... --init-config` and `--auto --autofocus --emit-airflow`
    cfg_out = tmp_path / "auto_cfg.json"
    dag_out = tmp_path / "dag_autofocus.py"
    main_mod._main(
        [
            "--auto",
            "--benchmark",
            "dgp_01_smooth_seasonal",
            "--tier",
            "first_light",
            "--autofocus",
            "--autofocus-mode",
            "dag",
            "--max-passes",
            "3",
            "--init-config",
            str(cfg_out),
            "--emit-out",
            str(dag_out),
        ]
    )
    assert cfg_out.is_file()
    assert dag_out.is_file()
    assert "autofocus_branch_2a" in dag_out.read_text(encoding="utf-8")

    # 3. Forecaster.emit_airflow_dag() on an executed autofocus session
    f_auto = Forecaster.auto(ds.panel_df, tier="first_light", autofocus=True, max_passes=2)
    f_dag_src = f_auto.emit_airflow_dag()
    ast.parse(f_dag_src)
    assert "autofocus_pass_1" in f_dag_src

    # 4. MCP Server `build_auto_config` (with emit_airflow_path) and `run_auto_forecast`
    mcp = McpServer(allow_launch=False)
    mcp_dag_out = tmp_path / "mcp_autofocus_dag.py"
    res_bp = mcp._dispatch_tool(
        "build_auto_config",
        {
            "benchmark": "dgp_01_smooth_seasonal",
            "tier": "first_light",
            "baselines": ["naive_seasonal", "theta"],
            "emit_airflow_path": str(mcp_dag_out),
            "autofocus": True,
        },
    )
    assert res_bp["emitted_airflow_path"] == str(mcp_dag_out)
    assert mcp_dag_out.is_file()

    res_run = mcp._dispatch_tool(
        "run_auto_forecast",
        {
            "benchmark": "dgp_01_smooth_seasonal",
            "tier": "first_light",
            "autofocus": True,
            "max_passes": 2,
            "n_series": 3,
        },
    )
    assert "baseline_value" in res_run
    assert len(res_run["baseline_value"]) >= 1
    assert "autofocus_ledger" in res_run
    assert "profile_table" in res_run
    assert len(res_run["profile_table"]) == 7
    assert "practitioner_assessment" in res_run
    assert len(res_run["practitioner_assessment"]) == 7

    # 5. Step 0 Data Profile & 7-Question Practitioner Assessment on SDK & Blueprint
    prof_df = Forecaster.profile_data(ds.panel_df, horizon=14)
    assert len(prof_df) == 7
    assert list(prof_df["dimension"]) == [
        "scale_and_shape",
        "demand_regime",
        "seasonality_and_scaling",
        "breaks_and_outliers",
        "hygiene_and_calendar",
        "covariates",
        "hierarchy_topology",
    ]
    assert len(bp.profile_df()) == 7
    assert len(f_auto.profile_df()) == 7

    assess_df = f_auto.practitioner_assessment_df()
    assert len(assess_df) == 7
    assert list(assess_df["question_id"]) == [
        "model_vs_baseline",
        "ensemble_vs_single",
        "covariates_vs_overfitting",
        "global_vs_local",
        "hierarchy_reconciliation",
        "horizon_stability",
        "forecast_trust",
    ]
    assess_str = f_auto.assess()
    assert "Practitioner Trust & Evidence Assessment" in assess_str
    assert "Step 0 Data Profile" in assess_str

    cmp_df = f_auto.compare_models()
    assert not cmp_df.empty
    assert "lift_pct" in cmp_df.columns
    assert "verdict" in cmp_df.columns
    cmp_pair = f_auto.compare_models("theta", "naive_seasonal")
    assert len(cmp_pair) == 1
    assert cmp_pair.iloc[0]["model_a"] == "theta"
    assert cmp_pair.iloc[0]["model_b"] == "naive_seasonal"
