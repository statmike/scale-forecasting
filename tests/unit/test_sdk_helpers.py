"""Offline unit tests for SDK explanatory helpers, review frames/plots, and cross-run ensembling."""

from __future__ import annotations

from typing import Any

import matplotlib
import pandas as pd
import pytest

import scale_forecasting as sf
from scale_forecasting.config import RunConfig
from scale_forecasting.ensemble_run import cross_run_copy_sql, merge_configs_for_ensemble
from scale_forecasting.errors import ConfigError
from scale_forecasting.review import (
    EnsembleLift,
    ModelReview,
    RunReview,
    build_hierarchy_frame,
    build_leaderboard_frame,
    build_predictions_frame,
    plot_forecasts_frame,
    plot_hierarchy_frame,
)
from scale_forecasting.sdk import Forecaster, build_best_params_frame

matplotlib.use("Agg")


def _sample_config(**overrides: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "sdk-helper-unit-test",
        "data": {
            "source_table": "source_series_native",
            "freq": "W",
            "horizon": 8,
            "series_limit": 20,
        },
        "features": {
            "future_covariates": ["promo_flag"],
            "past_covariates": ["spot_index"],
            "static_covariates": ["store_tier"],
        },
        "python_runtime": "vertex",
        "models": ["autoets", "lightgbm", "tide", "timesfm"],
        "model_params": {
            "tide": {"training_mode": "global", "max_epochs": 10},
        },
        "compute": {
            "families": {
                "statistical": {"runtime": "gce", "machine_type": "n2-standard-8"},
                "deep_learning": {
                    "runtime": "vertex",
                    "hardware": "gpu",
                    "gpu_type": "L4",
                },
            },
        },
        "backtest": {"enabled": True, "n_folds": 2, "horizon": 8, "step": 8},
        "ensemble": {
            "enabled": True,
            "strategies": ["mean", "inverse_error", "ridge"],
        },
    }
    base.update(overrides)
    return RunConfig.model_validate(base)


def test_explain_frame_captures_families_runtimes_and_covariates() -> None:
    cfg = _sample_config()
    forecaster = Forecaster(cfg)
    df = forecaster.explain()
    assert isinstance(df, pd.DataFrame)
    assert list(df["family"]) == [
        "statistical",
        "ml",
        "deep_learning",
        "native",
        "ensemble",
    ]
    by_fam = df.set_index("family")
    assert by_fam.loc["statistical", "runtime"] == "gce"
    assert by_fam.loc["statistical", "machine_type"] == "n2-standard-8"
    assert by_fam.loc["ml", "runtime"] == "vertex"
    assert by_fam.loc["deep_learning", "hardware"] == "gpu"
    assert by_fam.loc["deep_learning", "gpu_type"] == "L4"
    assert by_fam.loc["deep_learning", "training_modes"] == "tide:global"
    assert by_fam.loc["native", "runtime"] == "bigquery"
    assert by_fam.loc["ensemble", "models"] == (
        "ensemble_mean, ensemble_inverse_error, ensemble_ridge"
    )
    assert "future(1)" in str(by_fam.loc["statistical", "covariates"])
    assert "past(1)" in str(by_fam.loc["statistical", "covariates"])
    assert "static(1)" in str(by_fam.loc["statistical", "covariates"])

    gke_cfg = _sample_config(
        python_runtime="gke",
        models=["autoets", "lightgbm", "tide", "tsmixer", "timesfm"],
        compute={
            "gke_mode": "job",
            "workers": 2,
            "families": {
                "deep_learning": {
                    "runtime": "gke",
                    "gke_mode": "job",
                    "hardware": "gpu",
                    "gpu_type": "L4",
                    "workers": 1,
                },
            },
        },
    )
    gke_by_fam = Forecaster(gke_cfg).explain().set_index("family")
    assert gke_by_fam.loc["statistical", "runtime"] == "gke"
    assert gke_by_fam.loc["statistical", "workers"] == 2
    assert gke_by_fam.loc["deep_learning", "runtime"] == "gke"
    assert (
        gke_by_fam.loc["deep_learning", "workers"] == 2
    )  # auto-expanded 1 -> len(["tide", "tsmixer"])


def test_build_leaderboard_frame_and_all_metrics() -> None:
    m1 = ModelReview(
        model_type="ensemble_ridge",
        family="ensemble",
        ensemble_id="ens123",
        is_ensemble=True,
        compute_engine="ensemble",
        n_series=20,
        score=0.042,
        metric_means={"wape": 0.042, "smape": 0.05, "mase": 0.72, "mae": 4.1, "rmse": 5.3},
        metric_p50={"wape": 0.039, "smape": 0.048, "mase": 0.70, "mae": 3.9, "rmse": 5.0},
        mean_fit_seconds=0.2,
        median_fit_seconds=0.18,
        no_artifact_rate=0.0,
        n_predictions=160,
        pooled_wape=0.041,
        n_comparable_series=20,
    )
    m2 = ModelReview(
        model_type="lightgbm",
        family="ml",
        ensemble_id=None,
        is_ensemble=False,
        compute_engine="vertex",
        n_series=20,
        score=0.048,
        metric_means={"wape": 0.048, "smape": 0.056, "mase": 0.80, "mae": 4.8, "rmse": 6.1},
        metric_p50={"wape": 0.045, "smape": 0.052, "mase": 0.78, "mae": 4.5, "rmse": 5.8},
        mean_fit_seconds=1.1,
        median_fit_seconds=1.0,
        no_artifact_rate=0.0,
        n_predictions=160,
        pooled_wape=0.047,
        n_comparable_series=20,
    )
    review = RunReview(
        run_id="demo-123456789abc",
        status="COMPLETED",
        decision_metric="wape",
        n_series=20,
        models=(m1, m2),
        best_per_family={"ml": m2},
        best_overall=m2,
        ensembles=(m1,),
        ensemble_lift=(
            EnsembleLift(
                model_type="ensemble_ridge",
                score=0.042,
                best_base_model="lightgbm",
                best_base_score=0.048,
                lift=0.006,
                lift_pct=0.125,
            ),
        ),
    )
    df = build_leaderboard_frame(review, all_metrics=False)
    assert len(df) == 2
    assert list(df["model_type"]) == ["ensemble_ridge", "lightgbm"]
    assert df.loc[0, "lift_vs_best_base"] == pytest.approx(0.006)
    assert pd.isna(df.loc[1, "lift_vs_best_base"])

    df_all = build_leaderboard_frame(review, all_metrics=True)
    assert "p50_wape" in df_all.columns
    assert "mean_msis" in df_all.columns


def test_build_predictions_frame_and_plot_forecasts() -> None:
    import matplotlib.pyplot as plt

    hist_rows = [
        {"ts_id": "s1", "ds": "2026-01-04", "y": 100.0},
        {"ts_id": "s1", "ds": "2026-01-11", "y": 105.0},
    ]
    oof_rows = [
        {
            "ts_id": "s1",
            "model_type": "autoets",
            "fold_id": 0,
            "forecast_date": "2026-01-11",
            "y_true": 105.0,
            "yhat": 103.5,
            "yhat_lower": 95.0,
            "yhat_upper": 112.0,
        }
    ]
    pred_rows = [
        {
            "ts_id": "s1",
            "model_type": "autoets",
            "forecast_date": "2026-01-18",
            "yhat": 108.0,
            "yhat_lower": 99.0,
            "yhat_upper": 117.0,
        },
        {
            "ts_id": "s1",
            "model_type": "autoets",
            "forecast_date": "2026-01-25",
            "yhat": 110.0,
            "yhat_lower": 100.0,
            "yhat_upper": 120.0,
        },
    ]
    frame = build_predictions_frame(pred_rows, oof_rows=oof_rows, history_rows=hist_rows)
    assert set(frame["segment"]) == {"history", "oof", "forecast"}
    ax = plot_forecasts_frame(frame, max_series=1, title="unit-test")
    assert ax is not None
    plt.close("all")


def test_build_hierarchy_frame_and_plot_hierarchy() -> None:
    import matplotlib.pyplot as plt

    hier_rows = [
        {
            "ts_id": "__total__",
            "model_type": "autoets",
            "forecast_date": "2026-02-01",
            "yhat": 30.0,
        },
        {
            "ts_id": "region=East/store=A",
            "model_type": "autoets",
            "forecast_date": "2026-02-01",
            "yhat": 30.0,
        },
        {"ts_id": "leaf_1", "model_type": "autoets", "forecast_date": "2026-02-01", "yhat": 12.0},
        {"ts_id": "leaf_2", "model_type": "autoets", "forecast_date": "2026-02-01", "yhat": 18.0},
    ]
    df = build_hierarchy_frame(hier_rows)
    assert set(df["level"]) == {"total", "aggregate", "bottom"}
    assert df["max_coherence_residual"].iloc[0] == pytest.approx(0.0)
    ax = plot_hierarchy_frame(hier_rows, model_type="autoets")
    assert ax is not None
    plt.close("all")


def test_build_best_params_frame_unpacks_hpo_and_weights() -> None:
    rows = [
        {
            "ts_id": "s1",
            "model_type": "lightgbm",
            "compute_engine": "vertex",
            "ensemble_id": None,
            "best_params": '{"learning_rate": 0.05, "num_leaves": 31}',
            "fit_seconds": 1.2,
            "wape": 0.04,
            "mae": 3.1,
            "rmse": 4.0,
            "mase": 0.75,
        },
        {
            "ts_id": "s1",
            "model_type": "ensemble_ridge",
            "compute_engine": "ensemble",
            "ensemble_id": "ens123",
            "best_params": '{"autoets": 0.35, "lightgbm": 0.65}',
            "fit_seconds": None,
            "wape": 0.036,
            "mae": 2.8,
            "rmse": 3.6,
            "mase": 0.69,
        },
    ]
    df = build_best_params_frame(rows)
    assert len(df) == 2
    assert bool(df.loc[0, "is_ensemble"]) is False
    assert bool(df.loc[1, "is_ensemble"]) is True
    assert df.loc[0, "best_params"] == {"learning_rate": 0.05, "num_leaves": 31}
    assert df.loc[1, "best_params"] == {"autoets": 0.35, "lightgbm": 0.65}


def test_merge_configs_for_ensemble_and_copy_sql() -> None:
    c1 = _sample_config(models=["arima_plus", "timesfm"], model_params={})
    c2 = _sample_config(models=["autoets", "lightgbm", "tide"])
    merged = merge_configs_for_ensemble([c1, c2], strategies=["mean", "nnls", "ridge"])
    assert merged.models == ["arima_plus", "timesfm", "autoets", "lightgbm", "tide"]
    assert merged.ensemble.enabled is True
    assert merged.ensemble.strategies == ["mean", "nnls", "ridge"]

    sql = cross_run_copy_sql(
        "proj.ds",
        "forecast_predictions",
        "'arima_plus', 'autoets'",
        "src.ts_id, src.model_type, src.forecast_date",
    )
    assert "INSERT INTO `proj.ds.forecast_predictions`" in sql
    assert "SELECT * REPLACE (@target_run_id AS run_id, @created_at AS created_at)" in sql
    assert "WHERE src.run_id IN UNNEST(@source_run_ids)" in sql

    c_bad = _sample_config(
        data={
            "source_table": "source_series_native",
            "freq": "W",
            "horizon": 12,
            "series_limit": 20,
        }
    )
    with pytest.raises(ConfigError, match="horizon"):
        merge_configs_for_ensemble([c1, c_bad])


def test_run_live_executes_and_propagates_result(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _sample_config()
    f = Forecaster(cfg)
    monkeypatch.setattr(
        f,
        "run",
        lambda **kwargs: sf.RunResult(run_id=f.run_id, dataset_ref="proj.ds", views=()),
    )
    monkeypatch.setattr(
        f,
        "monitor",
        lambda *args, **kwargs: sf.RunProgress(
            run_id=f.run_id,
            status="COMPLETED",
            n_series=20,
            families=(),
            n_done=80,
            n_expected=80,
            fraction=1.0,
        ),
    )
    res = f.run_live(poll_seconds=0.01, plot=False)
    assert res.run_id == f.run_id


def test_build_cohorts_frame_and_calibration_frames_and_plot() -> None:
    import matplotlib.pyplot as plt

    from scale_forecasting.review import (
        ArmComparison,
        BacktestCohort,
        CalibrationReport,
        CoveragePoint,
        build_calibration_frames,
        build_cohorts_frame,
        plot_calibration,
    )

    m1 = ModelReview(
        model_type="lightgbm",
        family="ml",
        ensemble_id=None,
        is_ensemble=False,
        compute_engine="vertex",
        n_series=10,
        score=0.045,
        pooled_wape=0.043,
        n_comparable_series=10,
        cohort=BacktestCohort(
            n_series=10,
            n_full=8,
            n_reduced=2,
            fold_histogram={1: 2, 2: 8},
            refit_modes={"per_fold": 10},
            staleness_gap=0.004,
        ),
    )
    rev = RunReview(
        run_id="demo-cohorts-123456789abc",
        status="COMPLETED",
        decision_metric="wape",
        n_series=10,
        models=(m1,),
        best_per_family={"ml": m1},
        best_overall=m1,
        ensembles=(),
        ensemble_lift=(),
    )
    cdf = build_cohorts_frame(rev)
    assert len(cdf) == 1
    assert cdf.loc[0, "n_full"] == 8
    assert cdf.loc[0, "fold_histogram"] == "1f:2, 2f:8"
    assert cdf.loc[0, "refit_modes"] == "per_fold:10"
    assert cdf.loc[0, "staleness_gap"] == pytest.approx(0.004)

    cal = CalibrationReport(
        run_id="demo-cal-123456789abc",
        decision_metric="wape",
        nominal_coverage=0.80,
        arms=(
            ArmComparison(
                model_type="lightgbm",
                compute_engine="vertex",
                interval_calibration="conformal",
                n_series=10,
                n_raw_arm=4,
                n_auto_decided=10,
                n_compared=10,
                n_corrected_wins=6,
                mean_margin=0.03,
                median_margin=0.025,
            ),
        ),
        coverage=(
            CoveragePoint("lightgbm", 1, 20, 0.85, 12.0),
            CoveragePoint("lightgbm", 2, 20, 0.80, 15.0),
        ),
    )
    arms_df, cov_df = build_calibration_frames(cal)
    assert len(arms_df) == 1
    assert arms_df.loc[0, "win_rate"] == pytest.approx(0.6)
    assert len(cov_df) == 2
    assert cov_df.loc[0, "coverage_error"] == pytest.approx(0.05)

    axes = plot_calibration(cal)
    assert axes is not None
    plt.close("all")


def test_build_and_plot_ensemble_weights() -> None:
    import matplotlib.pyplot as plt

    from scale_forecasting.review import build_ensemble_weights_frame, plot_ensemble_weights

    rows = [
        {
            "ts_id": "s1",
            "model_type": "ensemble_nnls",
            "ensemble_id": "ens1",
            "best_params": '{"autoets": 0.4, "lightgbm": 0.6}',
            "wape": 0.035,
        },
        {
            "ts_id": "s1",
            "model_type": "ensemble_ridge",
            "ensemble_id": "ens1",
            "best_params": '{"autoets": 0.3, "lightgbm": 0.7}',
            "wape": 0.034,
        },
    ]
    wdf = build_ensemble_weights_frame(rows)
    assert len(wdf) == 4
    assert set(wdf["base_model"]) == {"autoets", "lightgbm"}
    ax = plot_ensemble_weights(wdf)
    assert ax is not None
    plt.close("all")


def test_explain_forecast_frame_and_plot_with_covariates() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from scale_forecasting.review import explain_forecast_frame, plot_forecast_explanation

    dates = pd.date_range("2026-01-01", periods=35, freq="D")
    hist_dates = dates[:28]
    fc_dates = dates[28:]

    # Construct history with a level shift at day 14, weekly seasonality, and promo effect
    step = np.where(np.arange(35) >= 14, 15.0, 0.0)
    seas = 5.0 * np.sin(2.0 * np.pi * np.arange(35) / 7.0)
    promo = np.where(np.arange(35) % 10 == 0, 1.0, 0.0)
    signal = 100.0 + step + seas + 12.0 * promo

    hist_rows = [
        {"ts_id": "s1", "ds": d.strftime("%Y-%m-%d"), "y": float(signal[i])}
        for i, d in enumerate(hist_dates)
    ]
    oof_rows = [
        {
            "ts_id": "s1",
            "model_type": "lightgbm",
            "fold_id": 0,
            "forecast_date": d.strftime("%Y-%m-%d"),
            "y_true": float(signal[21 + i]),
            "yhat": float(signal[21 + i] - 0.8),
            "yhat_lower": float(signal[21 + i] - 5.0),
            "yhat_upper": float(signal[21 + i] + 4.0),
        }
        for i, d in enumerate(hist_dates[21:])
    ]
    pred_rows = [
        {
            "ts_id": "s1",
            "model_type": "lightgbm",
            "forecast_date": d.strftime("%Y-%m-%d"),
            "yhat": float(signal[28 + i]),
            "yhat_lower": float(signal[28 + i] - (4.0 + i * 0.5)),
            "yhat_upper": float(signal[28 + i] + (4.0 + i * 0.5)),
        }
        for i, d in enumerate(fc_dates)
    ]
    cov_df = pd.DataFrame(
        {
            "ts_id": "s1",
            "ds": dates,
            "promo_flag": promo,
        }
    )
    pred_frame = build_predictions_frame(pred_rows, oof_rows=oof_rows, history_rows=hist_rows)
    exp_df = explain_forecast_frame(
        pred_frame,
        ts_id="s1",
        model_type="lightgbm",
        seasonal_period=7,
        covariate_df=cov_df,
    )
    assert len(exp_df) == 35
    assert "cov_promo_flag" in exp_df.columns
    assert exp_df["level_shift"].abs().max() > 0
    assert exp_df["oof_residual"].notna().sum() == 7
    assert exp_df["interval_width"].notna().sum() == 14  # 7 OOF + 7 forecast

    axes = plot_forecast_explanation(exp_df)
    assert len(axes) == 4
    plt.close("all")


def test_forecaster_jobs_df_and_registry_runs_df(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting.registry import reads

    cfg = _sample_config()
    f = Forecaster(cfg)
    monkeypatch.setattr(
        f,
        "jobs",
        lambda run_id=None: [
            sf.JobTrace(
                family="native",
                job_key="job-native",
                system_job_id="bq-1",
                runtime="bigquery",
                status="COMPLETED",
                attempt=1,
                hardware="sql",
                gpu_type=None,
                spark_mode=None,
                runtime_seconds=14.2,
            )
        ],
    )
    jdf = f.jobs_df()
    assert len(jdf) == 1
    assert jdf.loc[0, "family"] == "native"

    monkeypatch.setattr(
        reads,
        "read_recent_runs",
        lambda **kwargs: [
            {
                "run_id": "run-1",
                "created_at": "2026-10-05T00:00:00Z",
                "status": "COMPLETED",
                "python_runtime": "vertex",
                "n_series": 10,
                "n_models": 4,
                "backtest_on": True,
                "runtime_seconds": 42.0,
                "total_wall_s": 55.0,
                "overhead_seconds": 13.0,
                "overhead_fraction": 0.236,
                "dcu_milli_seconds": None,
            }
        ],
    )
    rdf = f.registry().runs_df(limit=5)
    assert len(rdf) == 1
    assert rdf.loc[0, "run_id"] == "run-1"
