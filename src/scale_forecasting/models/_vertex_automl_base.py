"""Shared base class and parameter validation for Vertex AI AutoML / Tabular Workflow models.

The four managed global forecasting architectures (`vertex_l2l`, `vertex_tide`, `vertex_tft`,
`vertex_seq2seq`) register through the same model factory (`runtime="vertex_automl"`,
`family="automl"`) so DAG planning, routing, covariate support checks, and the BigQuery registry
treat them uniformly alongside local Python and BigQuery-native models.

Execution is orchestrated by `engines.automl_engine` via either:
- **Tabular Workflow for Forecasting (`automl_mode="tabular_workflow"`, default):** Vertex AI
  Pipelines (`google-cloud-pipeline-components` preview forecasting pipelines) with step-level
  visibility, hardware overrides (`stage_1_tuner_worker_pool_specs_override`,
  `stage_2_trainer_worker_pool_specs_override`), Dataflow right-sizing, and hyperparameter
  warm-start (`stage_1_tuning_result_artifact_uri` / `reuse_tuning_from_run_id`).
- **Managed AutoML TrainingJob (`automl_mode="training_job"`):** Managed `google.cloud.aiplatform`
  `*ForecastingTrainingJob` API followed by `Model.batch_predict` with `generate_explanation=True`.

Calling `fit` or `predict` in-process raises `VertexAutoMLExecutionError` (a subclass of
`NotImplementedError`) because these models execute as managed Vertex AI jobs against BigQuery
staging tables.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from ..errors import ConfigError
from .base_model import DEFAULT_QUANTILES, BaseModel

_EXECUTED_IN_VERTEX_AUTOML = (
    "Vertex AI AutoML / Tabular Workflow models execute via engines/automl_engine.py on managed "
    "Vertex AI Pipelines or TrainingJobs, not in the per-series Python worker. Route 'automl' "
    "family models through the 'vertex_automl' runtime."
)

VALID_OPTIMIZATION_OBJECTIVES: frozenset[str] = frozenset(
    {
        "minimize-rmse",
        "minimize-mae",
        "minimize-mape",
        "minimize-wape-mae",
        "minimize-quantile-loss",
    }
)

VALID_AUTOML_MODES: frozenset[str] = frozenset({"tabular_workflow", "training_job"})

ALLOWED_VERTEX_AUTOML_PARAMS: frozenset[str] = frozenset(
    {
        "training_mode",
        "automl_mode",
        "optimization_objective",
        "train_budget_milli_node_hours",
        "context_window",
        "max_num_trials",
        "max_parallel_trial_count",
        "stage_1_num_parallel_trials",
        "stage_2_num_parallel_trials",
        "num_selected_trials",
        "stage_1_tuning_result_artifact_uri",
        "reuse_tuning_from_run_id",
        "enable_explainability",
        "generate_explanation",
        "quantiles",
        "enable_probabilistic_inference",
        "holiday_regions",
        "hierarchy_group_columns",
        "hierarchy_group_total_weight",
        "hierarchy_temporal_total_weight",
        "hierarchy_group_temporal_total_weight",
        "window_stride_length",
        "window_max_count",
        "additional_experiments",
        "data_granularity_unit",
        "data_granularity_count",
        "dataflow_machine_type",
        "dataflow_max_num_workers",
        "dataflow_disk_size_gb",
        "dataflow_service_account",
        "dataflow_subnetwork",
        "dataflow_use_public_ips",
        "evaluation_dataflow_machine_type",
        "evaluation_dataflow_starting_num_workers",
        "evaluation_dataflow_max_num_workers",
        "evaluation_batch_predict_machine_type",
        "evaluation_batch_predict_starting_replica_count",
        "evaluation_batch_predict_max_replica_count",
        "trainer_machine_type",
        "trainer_replica_count",
        "trainer_service_account",
        "run_evaluation",
        "model_display_name",
    }
)

_HIERARCHY_LOSS_KEYS: frozenset[str] = frozenset(
    {
        "hierarchy_group_columns",
        "hierarchy_group_total_weight",
        "hierarchy_temporal_total_weight",
        "hierarchy_group_temporal_total_weight",
    }
)


class VertexAutoMLExecutionError(NotImplementedError):
    """Raised if a Vertex AutoML model's in-process `fit` or `predict` is called directly."""


class VertexAutoMLBaseModel(BaseModel):
    """Base class for managed Vertex AI AutoML / Tabular Workflow global forecasting models."""

    runtime = "vertex_automl"
    family = "automl"
    supports_exog = True
    supports_future_covariates = True
    supports_past_covariates = True
    supports_static_covariates = True
    supports_explainability = True
    supports_native_intervals = True
    supports_global = True
    supports_hybrid = False
    gpu_capable = True
    package = "google-cloud-pipeline-components"
    package_url = (
        "https://docs.cloud.google.com/gemini-enterprise-agent-platform/"
        "machine-learning/tabular-data/tabular-workflows/forecasting"
    )
    # Vertex AI architectural capability flags (differ between L2L/TiDE vs TFT/Seq2Seq+):
    supports_hierarchy_group_loss: ClassVar[bool] = True
    supports_custom_quantiles_in_workflow: ClassVar[bool] = True
    vertex_architecture_key: ClassVar[str] = "l2l"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        raise VertexAutoMLExecutionError(_EXECUTED_IN_VERTEX_AUTOML)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        raise VertexAutoMLExecutionError(_EXECUTED_IN_VERTEX_AUTOML)

    @classmethod
    def gpu_useful(cls, params: Any) -> bool:
        """Stage-1 neural architecture search and Stage-2 ensemble training benefit from GPUs."""
        return True

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        """Validate `model_params.<vertex_model>` at plan time before cloud jobs are submitted."""
        unknown = sorted(set(params) - ALLOWED_VERTEX_AUTOML_PARAMS)
        if unknown:
            raise ConfigError(
                f"model_params.{cls.name} contains unsupported keys {unknown}; "
                f"supported keys: {sorted(ALLOWED_VERTEX_AUTOML_PARAMS)}."
            )

        mode = params.get("training_mode")
        if mode is not None and str(mode) != "global":
            raise ConfigError(
                f"model_params.{cls.name}.training_mode={mode!r} is invalid; "
                f"'{cls.name}' is a managed global cross-series model and only supports "
                "training_mode='global'."
            )

        automl_mode = params.get("automl_mode")
        if automl_mode is not None and str(automl_mode) not in VALID_AUTOML_MODES:
            raise ConfigError(
                f"model_params.{cls.name}.automl_mode={automl_mode!r} is invalid; "
                f"must be one of {sorted(VALID_AUTOML_MODES)}."
            )

        obj = params.get("optimization_objective")
        if obj is not None and str(obj) not in VALID_OPTIMIZATION_OBJECTIVES:
            raise ConfigError(
                f"model_params.{cls.name}.optimization_objective={obj!r} is invalid; "
                f"must be one of {sorted(VALID_OPTIMIZATION_OBJECTIVES)}."
            )

        for int_key in (
            "train_budget_milli_node_hours",
            "max_num_trials",
            "max_parallel_trial_count",
            "stage_1_num_parallel_trials",
            "stage_2_num_parallel_trials",
            "num_selected_trials",
            "dataflow_max_num_workers",
            "dataflow_disk_size_gb",
            "evaluation_dataflow_starting_num_workers",
            "evaluation_dataflow_max_num_workers",
            "evaluation_batch_predict_starting_replica_count",
            "evaluation_batch_predict_max_replica_count",
            "trainer_replica_count",
        ):
            val = params.get(int_key)
            if val is not None and (isinstance(val, bool) or not isinstance(val, int) or val < 1):
                raise ConfigError(
                    f"model_params.{cls.name}.{int_key}={val!r} must be a positive integer (>= 1)."
                )

        ctx_win = params.get("context_window")
        if ctx_win is not None and (
            isinstance(ctx_win, bool) or not isinstance(ctx_win, int) or ctx_win < 0
        ):
            raise ConfigError(
                f"model_params.{cls.name}.context_window={ctx_win!r} "
                "must be a non-negative integer."
            )

        uri = params.get("stage_1_tuning_result_artifact_uri")
        if uri is not None and (not isinstance(uri, str) or not uri.startswith("gs://")):
            raise ConfigError(
                f"model_params.{cls.name}.stage_1_tuning_result_artifact_uri={uri!r} "
                "must be a GCS URI starting with 'gs://'."
            )

        reuse_run = params.get("reuse_tuning_from_run_id")
        if reuse_run is not None and (not isinstance(reuse_run, str) or not reuse_run.strip()):
            raise ConfigError(
                f"model_params.{cls.name}.reuse_tuning_from_run_id "
                "must be a non-empty run_id string."
            )

        if uri is not None and reuse_run is not None:
            raise ConfigError(
                f"model_params.{cls.name} specifies both 'stage_1_tuning_result_artifact_uri' and "
                "'reuse_tuning_from_run_id'; specify at most one warm-start source."
            )

        quantiles = params.get("quantiles")
        if quantiles is not None:
            if not isinstance(quantiles, (list, tuple)) or not quantiles:
                raise ConfigError(
                    f"model_params.{cls.name}.quantiles "
                    "must be a non-empty list of floats in (0, 1)."
                )
            for q in quantiles:
                if (
                    isinstance(q, bool)
                    or not isinstance(q, (int, float))
                    or not (0.0 < float(q) < 1.0)
                ):
                    raise ConfigError(
                        f"model_params.{cls.name}.quantiles contains {q!r}; every quantile must be "
                        "strictly between 0.0 and 1.0."
                    )

        if not cls.supports_hierarchy_group_loss:
            used_hier = sorted(set(params) & _HIERARCHY_LOSS_KEYS)
            if used_hier:
                raise ConfigError(
                    f"model_params.{cls.name} specifies hierarchy group loss parameters "
                    f"{used_hier}, which Vertex AI only supports on 'vertex_l2l' and 'vertex_tide'."
                )

        if max_horizon > 1000:
            raise ConfigError(
                f"Vertex AI Forecasting ('{cls.name}') supports a maximum forecast horizon of "
                f"1000, but this run requires max_horizon={max_horizon}."
            )
