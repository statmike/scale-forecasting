"""Unit and offline integration tests for Phase G.5:
Model Variants (`<base_model>@<tag>`), Orthogonal `<scope> x <cov_profile>` Taxonomy,
Config-Driven Ablation Ladders, Ensemble Variant Pool Policy, Multi-Baseline FVA Attribution,
and Snapshot-Verified Cross-Run Cell Borrowing (`reuse_prior_runs`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scale_forecasting.config import FeaturesConfig, RunConfig
from scale_forecasting.dag import check_covariate_support, check_model_params, plan_dag
from scale_forecasting.diagnostics import compute_fva
from scale_forecasting.ensembler import filter_variants_for_ensemble
from scale_forecasting.errors import ConfigError
from scale_forecasting.features import (
    VALID_ABLATION_MODES,
    compute_cov_profile,
    effective_config_for_model,
    resolve_model_variant,
    should_stamp_variant_diagnostics,
)
from scale_forecasting.launch_plan import plan_run
from scale_forecasting.models import base_model_name, get_model, list_models
from scale_forecasting.playground import bakeoff, sample_data
from scale_forecasting.registry.ids import make_ensemble_id, make_model_spec_hash, make_run_id
from scale_forecasting.registry.ops import (
    BorrowedModelSource,
    match_reusable_models,
    render_borrow_model_sql,
)
from scale_forecasting.review import _assemble_review, build_leaderboard_frame


def test_base_model_name_and_get_model_variant_resolution() -> None:
    assert base_model_name("xgboost") == "xgboost"
    assert base_model_name("xgboost@univariate") == "xgboost"
    assert base_model_name("tft@local_cov_future_only") == "tft"

    cls = get_model("xgboost@univariate")
    assert cls is get_model("xgboost")
    assert cls.name == "xgboost"


def test_scope_capability_and_training_modes_across_all_models() -> None:
    for name in list_models():
        cls = get_model(name)
        cap = cls.scope_capability()
        modes = cls.supported_training_modes()
        assert cap in ("local_only", "multi_scope", "global_only")
        if cap == "local_only":
            assert modes == ("local",)
        elif cap == "global_only":
            assert modes == ("global",)
        else:
            assert "local" in modes and "global" in modes

    assert get_model("theta").scope_capability() == "local_only"
    assert get_model("xgboost").scope_capability() == "local_only"
    assert get_model("tft").scope_capability() == "multi_scope"
    assert get_model("tide").scope_capability() == "multi_scope"
    assert get_model("tsmixer").scope_capability() == "multi_scope"
    assert get_model("patchtst").scope_capability() == "multi_scope"
    assert get_model("neuralprophet").scope_capability() == "multi_scope"
    assert get_model("neuralprophet").supported_training_modes() == ("local", "global", "hybrid")
    assert get_model("vertex_l2l").scope_capability() == "global_only"
    assert get_model("vertex_tft").scope_capability() == "global_only"


def test_compute_cov_profile_all_eight_profiles() -> None:
    assert compute_cov_profile(FeaturesConfig()) == "cov_none"
    assert compute_cov_profile(FeaturesConfig(exog=["p"])) == "cov_future_only"
    assert compute_cov_profile(FeaturesConfig(past_covariates=["sensor"])) == "cov_past_only"
    assert compute_cov_profile(FeaturesConfig(static_covariates=["region"])) == "cov_static_only"
    assert (
        compute_cov_profile(FeaturesConfig(exog=["p"], past_covariates=["sensor"])) == "cov_dynamic"
    )
    assert (
        compute_cov_profile(FeaturesConfig(exog=["p"], static_covariates=["region"]))
        == "cov_future_static"
    )
    assert (
        compute_cov_profile(
            FeaturesConfig(past_covariates=["sensor"], static_covariates=["region"])
        )
        == "cov_past_static"
    )
    assert (
        compute_cov_profile(
            FeaturesConfig(exog=["p"], past_covariates=["sensor"], static_covariates=["region"])
        )
        == "cov_all"
    )


def test_dynamic_cov_profile_auto_refinement_when_tiers_absent() -> None:
    # Requesting covariate_mode='cov_all' on a config with only future exog and no
    # static_covariates or past_covariates must dynamically refine cov_profile to 'cov_future_only'.
    cfg = RunConfig.model_validate(
        {
            "run_name": "dynamic_cov_test",
            "data": {"source_table": "t", "horizon": 7},
            "features": {"exog": ["promo"], "holidays": ["US"], "static_covariates": []},
            "models": ["xgboost@cov_all", "xgboost@univariate"],
            "model_params": {
                "xgboost@cov_all": {"covariate_mode": "cov_all"},
                "xgboost@univariate": {"covariate_mode": "univariate"},
            },
        }
    )
    spec_all = resolve_model_variant(cfg, "xgboost@cov_all")
    assert spec_all.cov_profile == "cov_future_only"
    assert spec_all.is_univariate is False
    assert spec_all.covariates_used == ("promo",)

    spec_uni = resolve_model_variant(cfg, "xgboost@univariate")
    assert spec_uni.cov_profile == "cov_none"
    assert spec_uni.is_univariate is True
    assert spec_uni.covariates_used == ()
    # Constructor params handed to BaseModel must be stripped of variant feature meta keys
    assert "covariate_mode" not in spec_uni.clean_params


def test_per_variant_feature_overrides_and_effective_config() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "variant_override_test",
            "data": {"source_table": "t", "horizon": 7},
            "features": {
                "exog": ["promo", "price"],
                "past_covariates": ["foot_traffic"],
                "static_covariates": ["store_size"],
                "holidays": ["US"],
            },
            "models": [
                "tft@global_all",
                "tft@local_promo_only",
                "tft@past_only",
                "tft@static_only",
            ],
            "model_params": {
                "tft@global_all": {"training_mode": "global", "covariate_mode": "cov_all"},
                "tft@local_promo_only": {
                    "training_mode": "local",
                    "exog": ["promo"],
                    "past_covariates": [],
                    "static_covariates": [],
                    "holidays": [],
                },
                "tft@past_only": {"covariate_mode": "cov_past_only"},
                "tft@static_only": {"covariate_mode": "cov_static_only"},
            },
        }
    )
    s_all = resolve_model_variant(cfg, "tft@global_all")
    assert s_all.training_mode == "global"
    assert s_all.cov_profile == "cov_all"
    assert set(s_all.covariates_used) == {"foot_traffic", "price", "promo", "store_size"}

    s_promo = resolve_model_variant(cfg, "tft@local_promo_only")
    assert s_promo.training_mode == "local"
    assert s_promo.cov_profile == "cov_future_only"
    assert s_promo.covariates_used == ("promo",)

    s_past = resolve_model_variant(cfg, "tft@past_only")
    assert s_past.cov_profile == "cov_past_only"
    assert s_past.covariates_used == ("foot_traffic",)

    s_static = resolve_model_variant(cfg, "tft@static_only")
    assert s_static.cov_profile == "cov_static_only"
    assert s_static.covariates_used == ("store_size",)

    eff_static = effective_config_for_model(cfg, get_model("tft@static_only"), "tft@static_only")
    assert eff_static.features.exog == []
    assert eff_static.features.past_covariates == []
    assert eff_static.features.static_covariates == ["store_size"]


def test_include_univariate_baselines_skips_already_univariate_models() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "uni_baseline_test",
            "data": {"source_table": "t", "horizon": 7},
            "features": {
                "exog": ["promo"],
                "holidays": ["US"],
                "include_univariate_baselines": True,
            },
            "models": ["theta", "xgboost", "prophet"],
        }
    )
    # theta is strictly univariate -> no theta@univariate added!
    # xgboost and prophet consume covariates -> @univariate added for both!
    assert cfg.models == [
        "theta",
        "xgboost",
        "prophet",
        "xgboost@univariate",
        "prophet@univariate",
    ]
    assert should_stamp_variant_diagnostics("xgboost", cfg) is True


def test_ablation_modes_and_ablation_models_expansion() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "ablation_ladder_test",
            "data": {"source_table": "t", "horizon": 7},
            "features": {
                "exog": ["promo"],
                "past_covariates": ["sensor"],
                "static_covariates": ["region"],
                "holidays": ["US"],
                "ablation_modes": [
                    "local_univariate",
                    "global_univariate",
                    "cov_future_only",
                    "cov_past_only",
                    "cov_static_only",
                    "cov_all",
                ],
                "ablation_models": ["theta", "tft"],
            },
            "models": ["theta", "tft", "xgboost"],
            "model_params": {"tft": {"training_mode": "global"}},
        }
    )
    # theta is local_only and supports no covariates -> all ablation modes deduplicate to 'theta'
    assert [m for m in cfg.models if base_model_name(m) == "theta"] == ["theta"]
    # xgboost is not in ablation_models -> only primary 'xgboost'
    assert [m for m in cfg.models if base_model_name(m) == "xgboost"] == ["xgboost"]
    # tft is multi_scope with all 3 covariate tiers -> gets local_univariate, global_univariate,
    # cov_future_only, cov_past_only, cov_static_only (cov_all deduplicates with primary 'tft'!)
    tft_models = [m for m in cfg.models if base_model_name(m) == "tft"]
    assert tft_models == [
        "tft",
        "tft@local_univariate",
        "tft@global_univariate",
        "tft@cov_future_only",
        "tft@cov_past_only",
        "tft@cov_static_only",
    ]


def test_all_valid_ablation_modes_accepted() -> None:
    for mode in sorted(VALID_ABLATION_MODES):
        fc = FeaturesConfig.model_validate({"ablation_modes": [mode]})
        assert fc.ablation_modes == [mode]


def test_model_spec_hash_cross_run_equivalence_and_run_id_invariance() -> None:
    # Run A: covariate run with xgboost@univariate
    cfg_a = RunConfig.model_validate(
        {
            "run_name": "run_a",
            "data": {"source_table": "t", "horizon": 7},
            "features": {"exog": ["promo"], "include_univariate_baselines": True},
            "models": ["xgboost"],
        }
    )
    # Run B: pure univariate run with plain xgboost
    cfg_b = RunConfig.model_validate(
        {
            "run_name": "run_b",
            "data": {"source_table": "t", "horizon": 7},
            "features": {"exog": []},
            "models": ["xgboost"],
        }
    )
    assert make_model_spec_hash("xgboost@univariate", cfg_a) == make_model_spec_hash(
        "xgboost", cfg_b
    )

    # compute.reuse_prior_runs and compute.reuse_from_run_id are operational knobs
    cfg_b_no_reuse = cfg_b.model_copy(
        update={
            "compute": cfg_b.compute.model_copy(
                update={"reuse_prior_runs": True, "reuse_from_run_id": "prior-123456abcdef"}
            )
        }
    )
    assert make_run_id(cfg_b) == make_run_id(cfg_b_no_reuse)


def test_dag_validation_for_variants() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "dag_variant_test",
            "data": {"source_table": "t", "horizon": 7},
            "features": {
                "exog": ["promo"],
                "holidays": ["US"],
                "include_univariate_baselines": True,
            },
            "models": ["theta", "xgboost", "arima_plus", "arima_plus@no_holidays"],
            "model_params": {"arima_plus@no_holidays": {"holidays": []}},
        }
    )
    check_model_params(cfg)
    check_covariate_support(cfg)
    dag = plan_dag(cfg)
    assert dag.native_job is not None
    assert set(dag.native_job.models) == {
        "arima_plus",
        "arima_plus@no_holidays",
        "arima_plus@univariate",
    }

    # Invalid training_mode on a local_only model is rejected by check_model_params
    bad_cfg = RunConfig.model_validate(
        {
            "run_name": "bad_mode",
            "data": {"source_table": "t", "horizon": 7},
            "models": ["theta@global"],
            "model_params": {"theta@global": {"training_mode": "global"}},
        }
    )
    with pytest.raises((ConfigError, ValueError), match="training_mode"):
        check_model_params(bad_cfg)


def test_ensemble_variant_policy_filtering() -> None:
    cfg_all = RunConfig.model_validate(
        {
            "run_name": "ens_test",
            "data": {"source_table": "t", "horizon": 4},
            "models": ["theta", "xgboost", "xgboost@univariate"],
            "ensemble": {"enabled": True, "strategies": ["mean"], "variant_policy": "all"},
        }
    )
    cfg_primary = cfg_all.model_copy(
        update={"ensemble": cfg_all.ensemble.model_copy(update={"variant_policy": "primary_only"})}
    )
    cfg_best = cfg_all.model_copy(
        update={
            "ensemble": cfg_all.ensemble.model_copy(
                update={"variant_policy": "best_per_base_model"}
            )
        }
    )
    assert make_ensemble_id(cfg_all.ensemble) != make_ensemble_id(cfg_primary.ensemble)
    assert make_ensemble_id(cfg_primary.ensemble) != make_ensemble_id(cfg_best.ensemble)

    models = ["theta", "xgboost", "xgboost@univariate"]
    assert filter_variants_for_ensemble(models, cfg_all) == models
    assert filter_variants_for_ensemble(models, cfg_primary) == ["theta", "xgboost"]

    # When xgboost@univariate has lower error than xgboost in oof_df, best_per_base_model picks it
    oof_df = pd.DataFrame(
        [
            {
                "ts_id": "s1",
                "model_type": "theta",
                "fold_id": 0,
                "forecast_date": "2025-01-01",
                "y_true": 10.0,
                "yhat": 11.0,
            },
            {
                "ts_id": "s1",
                "model_type": "xgboost",
                "fold_id": 0,
                "forecast_date": "2025-01-01",
                "y_true": 10.0,
                "yhat": 15.0,
            },
            {
                "ts_id": "s1",
                "model_type": "xgboost@univariate",
                "fold_id": 0,
                "forecast_date": "2025-01-01",
                "y_true": 10.0,
                "yhat": 10.5,
            },
        ]
    )
    assert filter_variants_for_ensemble(models, cfg_best, oof_df=oof_df) == [
        "theta",
        "xgboost@univariate",
    ]


def test_multi_baseline_fva_covariate_and_pooling_attribution() -> None:
    rows = []
    for step, dt in enumerate(pd.date_range("2025-01-01", periods=10, freq="D")):
        y = 100.0 + step
        # naive_seasonal: error 10
        rows.append(
            {
                "ts_id": "s1",
                "model_type": "naive_seasonal",
                "fold_id": 0,
                "forecast_date": dt,
                "y_true": y,
                "yhat": y + 10.0,
            }
        )
        # tft@local_univariate: error 6
        rows.append(
            {
                "ts_id": "s1",
                "model_type": "tft@local_univariate",
                "fold_id": 0,
                "forecast_date": dt,
                "y_true": y,
                "yhat": y + 6.0,
            }
        )
        # tft@global_univariate: error 4 (pooling lift over local_univariate = (6-4)/6 = 33.3%)
        rows.append(
            {
                "ts_id": "s1",
                "model_type": "tft@global_univariate",
                "fold_id": 0,
                "forecast_date": dt,
                "y_true": y,
                "yhat": y + 4.0,
            }
        )
        # tft (global + covariates): error 2 (covariate lift over global_univariate = (4-2)/4 = 50%)
        rows.append(
            {
                "ts_id": "s1",
                "model_type": "tft",
                "fold_id": 0,
                "forecast_date": dt,
                "y_true": y,
                "yhat": y + 2.0,
            }
        )

    fva = compute_fva(pd.DataFrame(rows), baseline="naive_seasonal", metric="mae")
    by_model = fva.set_index("model_type")

    tft_row = by_model.loc["tft"]
    assert tft_row["base_model"] == "tft"
    assert tft_row["univariate_baseline_model"] == "tft@global_univariate"
    assert round(float(tft_row["covariate_fva_pct"]), 4) == 0.5
    assert tft_row["local_baseline_model"] == "tft@local_univariate"
    assert round(float(tft_row["pooling_fva_pct"]), 4) == round((6.0 - 2.0) / 6.0, 4)

    g_uni_row = by_model.loc["tft@global_univariate"]
    assert g_uni_row["local_baseline_model"] == "tft@local_univariate"
    assert round(float(g_uni_row["pooling_fva_pct"]), 4) == round((6.0 - 4.0) / 6.0, 4)


def test_playground_bakeoff_with_variants_and_ablation() -> None:
    df = sample_data(n_series=2, freq="D", with_exog=True)
    bo = bakeoff(
        df,
        models=["theta", "regression_lags"],
        horizon=7,
        n_folds=2,
        with_exog=True,
        include_univariate_baselines=True,
        variant_policy="best_per_base_model",
    )
    lb = bo.leaderboard
    assert "theta" in lb["model"].values
    assert "theta@univariate" not in lb["model"].values
    assert "regression_lags" in lb["model"].values
    assert "regression_lags@univariate" in lb["model"].values
    for col in (
        "base_model",
        "scope_capability",
        "training_mode",
        "cov_profile",
        "is_univariate",
        "covariates_used",
        "model_spec_hash",
    ):
        assert col in lb.columns

    ridge_row = lb[lb["model"] == "regression_lags"].iloc[0]
    assert ridge_row["base_model"] == "regression_lags"
    assert ridge_row["cov_profile"] == "cov_future_only"
    assert bool(ridge_row["is_univariate"]) is False

    ridge_uni_row = lb[lb["model"] == "regression_lags@univariate"].iloc[0]
    assert ridge_uni_row["base_model"] == "regression_lags"
    assert ridge_uni_row["cov_profile"] == "cov_none"
    assert bool(ridge_uni_row["is_univariate"]) is True

    fva = bo.fva_df()
    ridge_fva = fva[fva["model_type"] == "regression_lags"].iloc[0]
    assert ridge_fva["univariate_baseline_model"] == "regression_lags@univariate"


def test_match_reusable_models_and_render_borrow_sql() -> None:
    prior_cfg = RunConfig.model_validate(
        {
            "run_name": "prior_run",
            "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
            "features": {"exog": []},
            "models": ["theta", "xgboost"],
        }
    )
    target_cfg = RunConfig.model_validate(
        {
            "run_name": "target_run",
            "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
            "features": {
                "exog": ["is_holiday"],
                "include_univariate_baselines": True,
            },
            "models": ["theta", "xgboost"],
            "compute": {"reuse_prior_runs": True},
        }
    )
    # target_cfg expands to ['theta', 'xgboost', 'xgboost@univariate'].
    # Both 'theta' and 'xgboost@univariate' match prior_cfg's 'theta' and 'xgboost' spec hash!
    prior_id = "prior-run-123456abcdef"
    prior_headers = [
        {
            "run_id": prior_id,
            "status": "COMPLETED",
            "snapshot_millis": 1700000000000,
            "raw_config": prior_cfg.model_dump(mode="json"),
        }
    ]
    avail = {prior_id: {"theta", "xgboost"}}

    matched = match_reusable_models(
        target_cfg,
        "target-run-abcdef123456",
        1700000000000,
        prior_headers,
        avail,
    )
    assert set(matched.keys()) == {"theta", "xgboost@univariate"}
    assert matched["theta"] == BorrowedModelSource(
        target_model_type="theta",
        source_run_id=prior_id,
        source_model_type="theta",
        model_spec_hash=make_model_spec_hash("theta", target_cfg),
        snapshot_millis=1700000000000,
    )
    assert matched["xgboost@univariate"] == BorrowedModelSource(
        target_model_type="xgboost@univariate",
        source_run_id=prior_id,
        source_model_type="xgboost",
        model_spec_hash=make_model_spec_hash("xgboost@univariate", target_cfg),
        snapshot_millis=1700000000000,
    )

    # Snapshot mismatch prevents automatic reuse
    assert (
        match_reusable_models(
            target_cfg, "target-run-abcdef123456", 1700000999999, prior_headers, avail
        )
        == {}
    )

    # Live unpinned verification (snapshot_millis=None): succeeds when cand_snap >= last_modified
    assert set(
        match_reusable_models(
            target_cfg,
            "target-run-abcdef123456",
            None,
            prior_headers,
            avail,
            source_last_modified_millis=1699999000000,
        ).keys()
    ) == {"theta", "xgboost@univariate"}
    # Fails when source table was modified after cand_snap
    assert (
        match_reusable_models(
            target_cfg,
            "target-run-abcdef123456",
            None,
            prior_headers,
            avail,
            source_last_modified_millis=1700000500000,
        )
        == {}
    )

    # reuse_prior_runs=False disables automatic reuse
    no_reuse_cfg = target_cfg.model_copy(
        update={"compute": target_cfg.compute.model_copy(update={"reuse_prior_runs": False})}
    )
    assert (
        match_reusable_models(
            no_reuse_cfg, "target-run-abcdef123456", 1700000000000, prior_headers, avail
        )
        == {}
    )

    # Explicit reuse_from_run_id borrows even if snapshot_millis differs
    explicit_cfg = target_cfg.model_copy(
        update={
            "compute": target_cfg.compute.model_copy(
                update={"reuse_prior_runs": False, "reuse_from_run_id": prior_id}
            )
        }
    )
    assert set(
        match_reusable_models(
            explicit_cfg, "target-run-abcdef123456", 1700000999999, prior_headers, avail
        ).keys()
    ) == {"theta", "xgboost@univariate"}

    sql_map = render_borrow_model_sql("proj.ds")
    assert set(sql_map.keys()) == {"forecast_predictions", "backtest_oof", "forecast_metadata"}
    assert "$.borrowed_from_run_id" in sql_map["forecast_metadata"]


def test_launch_plan_and_review_leaderboard_frame_variant_columns() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "plan_variant_test",
            "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
            "features": {"exog": ["is_holiday"], "include_univariate_baselines": True},
            "models": ["xgboost"],
        }
    )
    lp = plan_run(cfg)
    assert len(lp.variant_plan) == 2
    assert lp.variant_plan[0]["model_type"] == "xgboost"
    assert lp.variant_plan[1]["model_type"] == "xgboost@univariate"

    rev = _assemble_review(
        "run-1",
        {"status": "COMPLETED", "n_series": 5},
        "wape",
        5,
        leaderboard_rows=[
            {
                "model_type": "xgboost@univariate",
                "base_model": "xgboost",
                "scope_capability": "local_only",
                "training_mode": "local",
                "cov_profile": "cov_none",
                "is_univariate": True,
                "covariates_used": "",
                "model_spec_hash": "abc123def456",
                "borrowed_from_run_id": "prior-001",
                "n_borrowed": 5,
                "compute_engine": "spark",
                "n_cells": 5,
                "mean_wape": 0.12,
                "median_fit_seconds": 0.5,
                "no_artifact_rate": 0.0,
            }
        ],
        aggregate_rows=[],
        prediction_counts={"xgboost@univariate": 35},
    )
    df = build_leaderboard_frame(rev)
    row = df.iloc[0]
    assert row["base_model"] == "xgboost"
    assert row["scope_capability"] == "local_only"
    assert row["training_mode"] == "local"
    assert row["cov_profile"] == "cov_none"
    assert bool(row["is_univariate"]) is True
    assert row["borrowed_from_run_id"] == "prior-001"
    assert row["n_borrowed"] == 5


def test_forecaster_variant_matrix_and_combined_uni_ablation_scoping() -> None:
    from scale_forecasting import Forecaster

    f = Forecaster(
        RunConfig.model_validate(
            {
                "run_name": "combined_uni_ablation_test",
                "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
                "models": ["theta", "sarimax", "xgboost", "tide"],
                "model_params": {"tide": {"training_mode": "global"}},
                "features": {
                    "static_covariates": ["region"],
                    "future_covariates": ["promo_flag"],
                    "past_covariates": ["temperature"],
                    "exog_lags": {"temperature": [7]},
                    "include_univariate_baselines": True,
                    "ablation_modes": ["cov_future_only", "local_univariate"],
                    "ablation_models": ["xgboost", "tide"],
                },
            }
        )
    )
    vm = f.variant_matrix()
    assert list(vm["model_type"]) == [
        "theta",
        "sarimax",
        "xgboost",
        "tide",
        "sarimax@univariate",
        "xgboost@univariate",
        "tide@univariate",
        "xgboost@cov_future_only",
        "tide@cov_future_only",
        "tide@local_univariate",
    ]
    assert set(vm.columns) == {
        "model_type",
        "base_model",
        "variant_tag",
        "family",
        "scope_capability",
        "training_mode",
        "cov_profile",
        "is_univariate",
        "covariates_used",
        "model_spec_hash",
    }


def test_vertex_engine_multi_worker_model_sharded_local_variant_reads_full_panel(
    monkeypatch,
) -> None:
    from scale_forecasting.engines import vertex_engine
    from scale_forecasting.settings import Settings

    cfg = RunConfig.model_validate(
        {
            "run_name": "dl_model_shard_local_variant",
            "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 8},
            "models": [
                "tide",
                "tide@univariate",
                "tide@cov_future_only",
                "tide@local_univariate",
            ],
            "model_params": {"tide": {"training_mode": "global"}},
            "features": {
                "future_covariates": ["promo_flag"],
                "include_univariate_baselines": True,
                "ablation_modes": ["cov_future_only", "local_univariate"],
                "ablation_models": ["tide"],
            },
            "compute": {
                "families": {"deep_learning": {"runtime": "vertex", "workers": 4}},
            },
        }
    )
    job_models = ["tide", "tide@univariate", "tide@cov_future_only", "tide@local_univariate"]
    assert vertex_engine.models_sharded_across_workers(cfg, job_models) is True

    calls: list[int] = []

    def _fake_read_source_series(_cfg, _settings):
        calls.append(1)
        return pd.DataFrame(
            {
                "ts_id": [f"s{i}" for i in range(8)],
                "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
                "y": [float(i) for i in range(8)],
            }
        )

    monkeypatch.setattr(vertex_engine, "_read_source_series", _fake_read_source_series)
    monkeypatch.setattr(vertex_engine, "_RAY_READ_SOURCE_SERIES", _fake_read_source_series)

    settings = Settings(
        project_id="demo-proj",
        connection="demo-proj.us-central1.conn",
        warehouse_uri="gs://demo-bkt/warehouse",
    )
    for rank in range(4):
        topology = vertex_engine.WorkerTopology(rank=rank, world_size=4)
        assigned = vertex_engine.partition_models_for_worker(cfg, job_models, topology)
        assert assigned == [job_models[rank]]
        _df, worker_ids, n_total = vertex_engine._read_worker_source_series(
            cfg,
            settings,
            topology,
            assigned,
            all_models=job_models,
        )
        assert worker_ids is None
        assert n_total == 8
    assert len(calls) == 4
