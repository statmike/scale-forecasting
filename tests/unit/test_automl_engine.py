"""Unit and contract tests for Vertex AI AutoML / Tabular Workflows and Two-Tier Explainability."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.automl_submit import plan_automl_job, submit_automl
from scale_forecasting.config import RunConfig
from scale_forecasting.engines.automl_engine import (
    AutoMLModelPlan,
    build_automl_column_specs,
    compile_tabular_workflow_spec,
    execute_automl_model_cells,
    extract_pipeline_artifacts,
    parse_batch_predictions_and_explanations,
    plan_automl_model,
    prepare_inference_panel,
    prepare_training_panel,
)
from scale_forecasting.errors import ConfigError
from scale_forecasting.models import get_model
from scale_forecasting.models._vertex_automl_base import (
    VertexAutoMLBaseModel,
    VertexAutoMLExecutionError,
)
from scale_forecasting.models.base_model import ModelContext
from scale_forecasting.probes.runtimes import VertexAutoMLProbe, get_probe
from scale_forecasting.probes.vocabulary import NATIVE_SUCCEEDED, ProbeHandle
from scale_forecasting.registry.rows import assemble_metadata_row, assemble_prediction_rows
from scale_forecasting.review import build_attributions_frame, plot_attributions
from scale_forecasting.settings import Settings
from scale_forecasting.worker import run_cell


def _make_panel(n_series: int = 2, n_obs: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_obs, freq="D")
    future_dates = pd.date_range("2024-01-01", periods=n_obs + 7, freq="D")
    hist_rows = []
    fut_rows = []
    for i in range(n_series):
        ts_id = f"S{i:03d}"
        store_size = float(100 + i * 25)
        region = f"R{i}"
        for d_idx, ds in enumerate(dates):
            promo = float(d_idx % 7 == 0)
            temp = float(60.0 + 10.0 * np.sin(d_idx / 7.0) + rng.normal(0, 1))
            y = float(50.0 + 15.0 * promo + 0.5 * temp + i * 10.0 + rng.normal(0, 1))
            hist_rows.append(
                {
                    "ts_id": ts_id,
                    "ds": ds,
                    "y": y,
                    "promo": promo,
                    "temperature": temp,
                    "store_size": store_size,
                    "region": region,
                }
            )
        for d_idx, ds in enumerate(future_dates):
            fut_rows.append(
                {
                    "ts_id": ts_id,
                    "ds": ds,
                    "promo": float(d_idx % 7 == 0),
                }
            )
    return pd.DataFrame(hist_rows), pd.DataFrame(fut_rows)


def _make_automl_config(
    models: list[str] | None = None,
    *,
    backtest_enabled: bool = False,
    model_params: dict | None = None,
) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run_name": "unit-automl",
            "data": {
                "source_table": "proj.ds.train",
                "ts_id_col": "ts_id",
                "date_col": "ds",
                "target_col": "y",
                "freq": "D",
                "horizon": 7,
            },
            "features": {
                "future_covariates": ["promo"],
                "past_covariates": ["temperature"],
                "static_covariates": ["store_size", "region"],
            },
            "models": models or ["vertex_tide"],
            "model_params": model_params
            or {
                "vertex_tide": {
                    "context_window": 14,
                    "train_budget_milli_node_hours": 1000,
                    "optimization_objective": "minimize-rmse",
                }
            },
            "backtest": {
                "enabled": backtest_enabled,
                "n_folds": 1,
                "step": 7,
                "horizon": 7,
                "min_train": 28,
            },
            "compute": {
                "automl_mode": "tabular_workflow",
                "families": {
                    "automl": {
                        "runtime": "vertex_automl",
                        "automl_mode": "tabular_workflow",
                    }
                },
            },
        }
    )


def _make_settings() -> Settings:
    return Settings(
        project_id="test-proj",
        region="us-central1",
        dataset_id="forecasting_test",
        connection="us-central1.bq-conn",
        warehouse_uri="gs://test-bucket/warehouse",
    )


class TestVertexAutoMLModels:
    @pytest.mark.parametrize(
        ("model_name", "expected_arch"),
        [
            ("vertex_l2l", "l2l"),
            ("vertex_tide", "tide"),
            ("vertex_tft", "tft"),
            ("vertex_seq2seq", "seq2seq"),
        ],
    )
    def test_model_metadata_and_contract(self, model_name: str, expected_arch: str) -> None:
        cls = get_model(model_name)
        assert issubclass(cls, VertexAutoMLBaseModel)
        assert cls.name == model_name
        assert cls.family == "automl"
        assert cls.runtime == "vertex_automl"
        assert cls.vertex_architecture_key == expected_arch
        assert cls.supports_global is True
        assert cls.supports_future_covariates is True
        assert cls.supports_past_covariates is True
        assert cls.supports_static_covariates is True
        assert cls.package_url.startswith(
            "https://docs.cloud.google.com/gemini-enterprise-agent-platform/"
        )

        m = cls({}, ModelContext(freq="D", horizon=7))
        s = pd.Series(np.arange(10.0))
        with pytest.raises(VertexAutoMLExecutionError, match="vertex_automl"):
            m.fit(s)
        with pytest.raises(VertexAutoMLExecutionError, match="vertex_automl"):
            m.predict(7)

    def test_validate_params_enforces_constraints(self) -> None:
        tide = get_model("vertex_tide")
        tft = get_model("vertex_tft")

        tide.validate_params(
            {
                "context_window": 14,
                "optimization_objective": "minimize-quantile-loss",
                "quantiles": [0.1, 0.5, 0.9],
                "hierarchy_group_columns": ["region"],
                "hierarchy_group_total_weight": 1.0,
            },
            max_horizon=7,
        )

        with pytest.raises(ConfigError, match="hierarchy group loss"):
            tft.validate_params({"hierarchy_group_columns": ["region"]}, max_horizon=7)

        with pytest.raises(ConfigError, match="specify at most one warm-start source"):
            tide.validate_params(
                {
                    "stage_1_tuning_result_artifact_uri": "gs://b/tuning",
                    "reuse_tuning_from_run_id": "prior-run-123",
                },
                max_horizon=7,
            )


class TestAutoMLEngineColumnSpecsAndPanels:
    def test_build_automl_column_specs_classifies_features(self) -> None:
        hist_df, fut_df = _make_panel()
        cfg = _make_automl_config()
        train_df = prepare_training_panel(hist_df, cfg)
        assert "__sf_split__" in train_df.columns
        assert set(train_df["__sf_split__"]).issubset({"TRAIN", "VALIDATE", "TEST"})

        col_specs, attr_cols, avail_cols, unavail_cols = build_automl_column_specs(cfg, train_df)
        assert set(attr_cols) == {"store_size", "region"}
        assert "promo" in avail_cols
        assert "ds" in avail_cols
        assert "temperature" in unavail_cols
        assert "y" in unavail_cols
        spec_map = {s["column_name"]: s["data_type"] for s in col_specs}
        assert spec_map["ds"] == "timestamp"
        assert spec_map["y"] == "numeric"
        assert spec_map["region"] == "categorical"

        inf_df = prepare_inference_panel(
            hist_df,
            cfg,
            context_window=14,
            horizon=7,
            future_df=fut_df,
        )
        assert len(inf_df) == 2 * (14 + 7)
        future_part = inf_df[inf_df["y"].isna()]
        assert len(future_part) == 2 * 7
        assert future_part["promo"].notna().all()

    @pytest.mark.parametrize(
        "model_name", ["vertex_l2l", "vertex_tide", "vertex_tft", "vertex_seq2seq"]
    )
    def test_plan_and_compile_tabular_workflow_spec(self, model_name: str, tmp_path: Path) -> None:
        hist_df, _ = _make_panel()
        cfg = _make_automl_config(
            models=[model_name],
            model_params={
                model_name: {"context_window": 14, "train_budget_milli_node_hours": 1000}
            },
        )
        train_df = prepare_training_panel(hist_df, cfg)
        plan = plan_automl_model(
            cfg,
            model_name,
            train_df,
            run_id="unit-automl-000000000001",
            artifact_root="gs://test-bucket/artifacts",
        )
        assert isinstance(plan, AutoMLModelPlan)
        assert plan.model_type == model_name
        assert plan.context_window == 14
        assert plan.horizon == 7

        template_path, parameter_values = compile_tabular_workflow_spec(
            plan,
            project_id="test-proj",
            region="us-central1",
            train_bq_uri="bq://test-proj.ds.train_tbl",
            ts_id_col="ts_id",
            date_col="ds",
            target_col="y",
            output_dir=tmp_path,
        )
        assert Path(template_path).is_file()
        assert parameter_values["project"] == "test-proj"
        assert parameter_values["location"] == "us-central1"
        assert parameter_values["target_column"] == "y"
        assert parameter_values["forecast_horizon"] == 7
        assert parameter_values["context_window"] == 14


class TestAutoMLEngineArtifactsAndExecution:
    def test_extract_pipeline_artifacts(self) -> None:
        mock_tuning_artifact = SimpleNamespace(
            uri="gs://test-bucket/artifacts/tuning_result_output",
            metadata={},
        )
        mock_model_artifact = SimpleNamespace(
            uri="https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/models/12345",
            metadata={"resourceName": "projects/p/locations/us-central1/models/12345"},
        )
        t1 = SimpleNamespace(
            task_name="automl-forecasting-stage-1-tuner",
            outputs={"tuning_result_output": SimpleNamespace(artifacts=[mock_tuning_artifact])},
        )
        t2 = SimpleNamespace(
            task_name="model-upload",
            outputs={"model": SimpleNamespace(artifacts=[mock_model_artifact])},
        )
        fake_pipeline_job = SimpleNamespace(
            resource_name="projects/p/locations/us-central1/pipelineJobs/pipe-1",
            gca_resource=SimpleNamespace(job_detail=SimpleNamespace(task_details=[t1, t2])),
        )
        extracted = extract_pipeline_artifacts(fake_pipeline_job)
        assert (
            extracted["stage_1_tuning_result_artifact_uri"]
            == "gs://test-bucket/artifacts/tuning_result_output"
        )
        assert (
            extracted["vertex_model_resource_name"]
            == "projects/p/locations/us-central1/models/12345"
        )

    def test_find_reusable_pipeline_artifacts_and_batch_predict_fallback(self) -> None:
        from scale_forecasting.engines.automl_engine import (
            _batch_predict_with_fallback,
            _find_reusable_pipeline_artifacts,
        )

        mock_tuning_artifact = SimpleNamespace(
            uri="gs://test-bucket/artifacts/tuning_result_output",
            metadata={},
        )
        mock_model_artifact = SimpleNamespace(
            uri="",
            metadata={"resourceName": "projects/p/locations/us-central1/models/999"},
        )
        t1 = SimpleNamespace(
            task_name="automl-forecasting-stage-1-tuner",
            outputs={"tuning_result_output": SimpleNamespace(artifacts=[mock_tuning_artifact])},
        )
        t2 = SimpleNamespace(
            task_name="model-upload",
            outputs={"model": SimpleNamespace(artifacts=[mock_model_artifact])},
        )
        succeeded_pjob = SimpleNamespace(
            name="projects/p/locations/us-central1/pipelineJobs/sf-run-automl-a1-tide-f1",
            state=SimpleNamespace(name="PIPELINE_STATE_SUCCEEDED"),
            job_detail=SimpleNamespace(task_details=[t1, t2]),
        )

        def _fake_get(name: str):
            if name.endswith("sf-run-automl-a1-tide-f1"):
                return succeeded_pjob
            raise RuntimeError("404 Not Found")

        fake_pclient = SimpleNamespace(get_pipeline_job=_fake_get)
        reused = _find_reusable_pipeline_artifacts(
            project_id="p",
            region="us-central1",
            job_prefix="sf-run-automl-a2-tide-f1",
            pipeline_client=fake_pclient,
        )
        assert reused is not None
        assert reused["pipeline_job_id"] == "sf-run-automl-a1-tide-f1"
        assert reused["vertex_model_resource_name"] == "projects/p/locations/us-central1/models/999"
        assert (
            reused["stage_1_tuning_result_artifact_uri"]
            == "gs://test-bucket/artifacts/tuning_result_output"
        )

        # Verify BatchPrediction fallback retries on transient GCE machine stockout
        tried_machines: list[str] = []

        def _fake_bp(**kwargs):
            mt = kwargs["machine_type"]
            tried_machines.append(mt)
            if len(tried_machines) == 1:
                raise RuntimeError(
                    'Job failed with: code: 14, message: "Machine type temporarily unavailable"'
                )
            return SimpleNamespace(machine_type=mt)

        fake_model = SimpleNamespace(batch_predict=_fake_bp)
        bp_res = _batch_predict_with_fallback(
            fake_model,
            job_display_name="sf-bp-test",
            bigquery_source="bq://p.ds.infer",
            bigquery_destination_prefix="bq://p.ds",
            preferred_machine_type="n1-standard-4",
            starting_replica_count=1,
            max_replica_count=2,
            generate_explanation=True,
        )
        assert tried_machines == ["n1-standard-4", "n1-highmem-8"]
        assert bp_res.machine_type == "n1-highmem-8"

    def test_parse_batch_predictions_and_explanations(self) -> None:
        raw_df = pd.DataFrame(
            [
                {
                    "ts_id": "S000",
                    "ds": "2024-02-10",
                    "predicted_y": {
                        "value": 105.5,
                        "quantile_values": [95.0, 105.5, 116.0],
                        "quantiles": [0.1, 0.5, 0.9],
                    },
                    "explanation": {
                        "attributions": [
                            {
                                "baselineOutputValue": 90.0,
                                "featureAttributions": {
                                    "promo": 10.0,
                                    "temperature": 3.5,
                                    "store_size": 2.0,
                                },
                            }
                        ]
                    },
                },
                {
                    "ts_id": "S000",
                    "ds": "2024-02-11",
                    "predicted_y": {
                        "value": 98.0,
                        "quantile_values": [88.0, 98.0, 108.0],
                        "quantiles": [0.1, 0.5, 0.9],
                    },
                    "explanation": {
                        "attributions": [
                            {
                                "baselineOutputValue": 90.0,
                                "featureAttributions": {
                                    "promo": 2.0,
                                    "temperature": 4.0,
                                    "store_size": 2.0,
                                },
                            }
                        ]
                    },
                },
            ]
        )
        preds_by_s, exps_by_s, global_attrs_by_s = parse_batch_predictions_and_explanations(
            raw_df,
            ts_id_col="ts_id",
            date_col="ds",
            target_col="y",
        )
        assert "S000" in preds_by_s
        pred_df = preds_by_s["S000"]
        tier2 = exps_by_s["S000"]
        tier1 = global_attrs_by_s["S000"]
        assert len(pred_df) == 2
        assert list(pred_df["yhat"]) == [105.5, 98.0]
        assert list(pred_df["yhat_lower"]) == [95.0, 88.0]
        assert list(pred_df["yhat_upper"]) == [116.0, 108.0]
        assert set(tier1.keys()) == {"promo", "temperature", "store_size"}
        assert tier1["promo"] == 6.0
        assert len(tier2) == 2
        assert tier2[0] is not None
        assert tier2[0]["baseline_score"] == 90.0
        assert tier2[0]["attributions"]["promo"] == 10.0

    def test_parse_batch_predictions_bq_ndarray_schema(self) -> None:
        raw_df = pd.DataFrame(
            [
                {
                    "ts_id": "S000",
                    "ds": "2024-02-10",
                    "predicted_y": {
                        "value": np.float64(105.5),
                        "quantile_values": np.array([0.1, 0.5, 0.9]),
                        "quantile_predictions": np.array([95.0, 105.5, 116.0]),
                    },
                    "explanation": {
                        "attributions": np.array(
                            [
                                {
                                    "baseline_score": np.float64(90.0),
                                    "feature_attributions": {
                                        "promo": np.array([4.0, 6.0]),
                                        "temperature": np.float64(3.5),
                                        "store_size": np.float64(2.0),
                                    },
                                }
                            ],
                            dtype=object,
                        )
                    },
                }
            ]
        )
        preds_by_s, exps_by_s, global_attrs_by_s = parse_batch_predictions_and_explanations(
            raw_df,
            ts_id_col="ts_id",
            date_col="ds",
            target_col="y",
        )
        pred_df = preds_by_s["S000"]
        assert list(pred_df["yhat"]) == [105.5]
        assert list(pred_df["yhat_lower"]) == [95.0]
        assert list(pred_df["yhat_upper"]) == [116.0]
        assert exps_by_s["S000"][0] == {
            "baseline_score": 90.0,
            "attributions": {"promo": 10.0, "temperature": 3.5, "store_size": 2.0},
        }
        assert global_attrs_by_s["S000"]["promo"] == 10.0

    def test_execute_automl_model_cells_with_backtest_and_explanations(self) -> None:
        hist_df, fut_df = _make_panel(n_series=2, n_obs=120)
        cfg = _make_automl_config(models=["vertex_tide"], backtest_enabled=True)
        settings = _make_settings()

        def fake_fit_and_predict(
            plan: AutoMLModelPlan,
            train_df: pd.DataFrame,
            infer_df: pd.DataFrame,
            cfg_in: RunConfig,
            *,
            settings: Settings,
            job_prefix: str,
            bq_client: object = None,
        ):
            future_rows = infer_df[infer_df["y"].isna()].copy()
            out_rows = []
            for _, row in future_rows.iterrows():
                val = 100.0 + 10.0 * float(row.get("promo", 0.0))
                out_rows.append(
                    {
                        "ts_id": row["ts_id"],
                        "ds": row["ds"],
                        "predicted_y": {
                            "value": val,
                            "quantile_values": [val - 5.0, val, val + 5.0],
                            "quantiles": [0.1, 0.5, 0.9],
                        },
                        "explanation": {
                            "attributions": [
                                {
                                    "baselineOutputValue": 95.0,
                                    "featureAttributions": {
                                        "promo": 8.0,
                                        "temperature": 2.0,
                                    },
                                }
                            ]
                        },
                    }
                )
            preds_by_s, exps_by_s, global_attrs_by_s = parse_batch_predictions_and_explanations(
                pd.DataFrame(out_rows),
                ts_id_col="ts_id",
                date_col="ds",
                target_col="y",
            )
            return (
                preds_by_s,
                exps_by_s,
                global_attrs_by_s,
                {
                    "stage_1_tuning_result_artifact_uri": "gs://bucket/run-1/tuning_result",
                    "vertex_model_resource_name": "projects/p/locations/us-central1/models/tide-1",
                    "pipeline_job_resource_name": (
                        "projects/p/locations/us-central1/pipelineJobs/job-1"
                    ),
                },
            )

        cells, telemetry = execute_automl_model_cells(
            hist_df,
            "vertex_tide",
            cfg,
            run_id="unit-automl-000000000001",
            settings=settings,
            future_df=fut_df,
            fit_and_predict_fn=fake_fit_and_predict,
        )
        assert len(cells) == 2
        assert telemetry["stage_1_tuning_result_artifact_uri"] == "gs://bucket/run-1/tuning_result"
        for cell in cells:
            assert cell.status == "ok"
            assert len(cell.predictions) == 7
            assert cell.oof is not None
            assert len(cell.oof) == 7
            assert "rmse" in cell.metrics
            assert cell.model_artifact_uri == "gs://bucket/run-1/tuning_result"
            assert "feature_attributions" in cell.diagnostics
            assert cell.explanations is not None
            assert len(cell.explanations) == 7
            assert cell.explanations[0]["baseline_score"] == 95.0


class TestTwoTierExplainabilityPythonMLAndSDK:
    @pytest.mark.parametrize(
        "model_name",
        ["xgboost", "lightgbm", "catboost", "random_forest", "regression_lags"],
    )
    def test_python_ml_models_emit_tier1_and_tier2_explanations(self, model_name: str) -> None:
        hist_df, _ = _make_panel(n_series=1, n_obs=80)
        cfg = RunConfig.model_validate(
            {
                "run_name": f"unit-explain-{model_name}",
                "data": {
                    "source_table": "proj.ds.train",
                    "ts_id_col": "ts_id",
                    "date_col": "ds",
                    "target_col": "y",
                    "freq": "D",
                    "horizon": 5,
                },
                "features": {"future_covariates": ["promo"]},
                "models": [model_name],
                "backtest": {"enabled": False},
            }
        )
        series_df = hist_df[hist_df["ts_id"] == "S000"].copy()
        cell = run_cell(
            series_df,
            model_name,
            cfg,
        )
        assert cell.status == "ok", f"{model_name} failed: {cell.error}"
        # Tier 1: series/model-level attributions in fit_diagnostics
        assert cell.diagnostics is not None
        assert "feature_attributions" in cell.diagnostics
        attributions = cell.diagnostics["feature_attributions"]
        assert isinstance(attributions, dict)
        assert len(attributions) > 0
        assert pytest.approx(sum(attributions.values()), rel=1e-3) == 1.0

        # Tier 2: per-horizon-step local attributions in cell.explanations
        assert cell.explanations is not None
        assert len(cell.explanations) == 5
        for step_exp in cell.explanations:
            assert step_exp is not None
            assert "baseline_score" in step_exp
            assert isinstance(step_exp["baseline_score"], float)
            assert "attributions" in step_exp
            assert isinstance(step_exp["attributions"], dict)
            assert len(step_exp["attributions"]) > 0

        # Verify serialization into BigQuery row format
        pred_rows = assemble_prediction_rows(cell)
        assert len(pred_rows) == 5
        for r in pred_rows:
            assert r["explanations"] is not None
            parsed_exp = json.loads(r["explanations"])
            assert "baseline_score" in parsed_exp
            assert "attributions" in parsed_exp

        meta = assemble_metadata_row(cell, created_at=datetime.now(UTC))
        parsed_diag = json.loads(meta["fit_diagnostics"])
        assert "feature_attributions" in parsed_diag

    def test_build_attributions_frame_and_plot(self) -> None:
        import matplotlib.pyplot as plt

        global_rows = [
            {
                "ts_id": "S000",
                "model_type": "vertex_tide",
                "compute_engine": "vertex_automl",
                "fit_diagnostics": json.dumps(
                    {"feature_attributions": {"promo": 0.6, "temperature": 0.3, "store_size": 0.1}}
                ),
            },
            {
                "ts_id": "S001",
                "model_type": "vertex_tide",
                "compute_engine": "vertex_automl",
                "fit_diagnostics": json.dumps(
                    {
                        "feature_attributions": {
                            "promo": 0.5,
                            "temperature": 0.35,
                            "store_size": 0.15,
                        }
                    }
                ),
            },
        ]
        df_global = build_attributions_frame(global_rows, level="global")
        assert not df_global.empty
        assert list(df_global["feature"].unique()[:2]) == ["promo", "temperature"]

        ax1 = plot_attributions(df_global, top_k=3)
        assert ax1 is not None
        plt.close("all")

        horizon_rows = [
            {
                "ts_id": "S000",
                "model_type": "vertex_tide",
                "compute_engine": "vertex_automl",
                "forecast_date": "2024-02-01",
                "yhat": 110.0,
                "explanations": json.dumps(
                    {"baseline_score": 95.0, "attributions": {"promo": 10.0, "temperature": 5.0}}
                ),
            },
            {
                "ts_id": "S000",
                "model_type": "vertex_tide",
                "compute_engine": "vertex_automl",
                "forecast_date": "2024-02-02",
                "yhat": 98.0,
                "explanations": json.dumps(
                    {"baseline_score": 95.0, "attributions": {"promo": 1.0, "temperature": 2.0}}
                ),
            },
        ]
        df_horizon = build_attributions_frame(horizon_rows, level="local")
        assert len(df_horizon) == 4
        assert set(df_horizon["feature"]) == {"promo", "temperature"}

        ax2 = plot_attributions(df_horizon, ts_id="S000", model_type="vertex_tide")
        assert ax2 is not None
        plt.close("all")


class TestVertexAutoMLProbeAndSubmitter:
    def test_vertex_automl_probe_maps_pipeline_states(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probe = get_probe("vertex_automl")
        assert isinstance(probe, VertexAutoMLProbe)
        assert probe.name == "vertex_automl"

        fake_job = SimpleNamespace(
            name="projects/test-proj/locations/us-central1/pipelineJobs/pipe-123",
            state=SimpleNamespace(name="PIPELINE_STATE_SUCCEEDED"),
            error=None,
        )
        fake_client = SimpleNamespace(
            get_pipeline_job=MagicMock(return_value=fake_job),
            cancel_pipeline_job=MagicMock(),
        )
        monkeypatch.setattr(
            "scale_forecasting.automl_submit._pipeline_client",
            lambda _region: fake_client,
        )
        settings = _make_settings()
        handle = ProbeHandle(
            runtime="vertex_automl",
            native_id="pipe-123",
            region="us-central1",
            resource_name="projects/test-proj/locations/us-central1/pipelineJobs/pipe-123",
        )
        res = probe.check(handle, settings=settings)
        assert res.native_state == NATIVE_SUCCEEDED
        assert res.exists is True

    def test_plan_and_submit_automl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_automl_config(models=["vertex_tide"], backtest_enabled=False)
        settings = _make_settings()
        plan = plan_automl_job(cfg, ["vertex_tide"], run_id="unit-automl-000000000001")
        assert plan.automl_mode == "tabular_workflow"
        assert plan.models == ("vertex_tide",)

        monkeypatch.setattr(
            "scale_forecasting.automl_submit.staging.stage_config",
            lambda *a, **kw: "gs://test-bucket/staged/cfg.json",
        )
        monkeypatch.setattr(
            "scale_forecasting.automl_submit.automl_engine.run",
            lambda *a, **kw: {
                "run_id": "unit-automl-000000000001",
                "n_succeeded": 2,
                "n_failed": 0,
                "models": {
                    "vertex_tide": {
                        "pipeline_job_id": "sf-unit-automl-tide-final",
                        "pipeline_resource_name": (
                            "projects/test-proj/locations/us-central1/pipelineJobs/"
                            "sf-unit-automl-tide-final"
                        ),
                    }
                },
            },
        )
        run_id, native_id, handle = submit_automl(
            cfg,
            settings=settings,
            models=["vertex_tide"],
            manage_header=False,
        )
        assert native_id == "sf-unit-automl-tide-final"
        assert handle.runtime == "vertex_automl"
        assert (
            handle.resource_name
            == "projects/test-proj/locations/us-central1/pipelineJobs/sf-unit-automl-tide-final"
        )

    @pytest.mark.parametrize("automl_mode", ["tabular_workflow", "training_job"])
    def test_automl_mode_and_family_runtime_resolution(self, automl_mode: str) -> None:
        for model_name in ("vertex_l2l", "vertex_tide", "vertex_tft", "vertex_seq2seq"):
            cfg = RunConfig.model_validate(
                {
                    "run_name": f"unit-mode-{model_name}",
                    "data": {"source_table": "proj.ds.train", "horizon": 7},
                    "models": [model_name],
                    "python_runtime": "vertex_automl",
                    "backtest": {"enabled": False},
                    "compute": {
                        "automl_mode": automl_mode,
                        "families": {
                            "automl": {
                                "runtime": "vertex_automl",
                                "automl_mode": automl_mode,
                            }
                        },
                    },
                }
            )
            assert cfg.python_runtime == "vertex_automl"
            fc = cfg.resolve_family_compute("automl")
            assert fc.runtime == "vertex_automl"
            assert fc.automl_mode == automl_mode
            assert cfg.compute.automl_mode == automl_mode
            plan = plan_automl_job(cfg, [model_name], run_id="unit-mode-000000000001")
            assert plan.automl_mode == automl_mode
            assert plan.models == (model_name,)
