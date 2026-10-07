"""Vertex AI AutoML & Tabular Workflow for Forecasting execution engine.

Orchestrates global panel forecasting models on Vertex AI (`runtime == "vertex_automl"`,
`family == "automl"`):

- ``vertex_l2l``: Vertex AI AutoML Forecasting (Learn-to-Learn NAS + Ensemble)
- ``vertex_tide``: Time-series Dense Encoder (TiDE)
- ``vertex_tft``: Temporal Fusion Transformer (TFT)
- ``vertex_seq2seq``: Sequence-to-Sequence (Seq2Seq+)

Supports two execution modes configured via ``compute.automl_mode`` (or
``compute.families.automl.automl_mode``):

1. ``"tabular_workflow"`` (default): Compiles and submits the transparent Kubeflow Pipelines
   (KFP v2) Tabular Workflow for Forecasting via ``google-cloud-pipeline-components`` and
   ``google.cloud.aiplatform.PipelineJob``, with right-sized Dataflow evaluation workers, custom
   worker-pool machine/GPU overrides, automatic GCS extraction of
   ``stage_1_tuning_result_artifact_uri`` for zero-search warm-starting across runs
   (``reuse_tuning_from_run_id``), and ``BatchPredictionJob`` inference with explanations.
2. ``"training_job"``: Managed ``AutoMLForecastingTrainingJob`` /
   ``TimeSeriesDenseEncoderForecastingTrainingJob`` /
   ``TemporalFusionTransformerForecastingTrainingJob`` /
   ``Seq2SeqPlusForecastingTrainingJob`` followed by ``BatchPredictionJob``.

Every completed run writes standard ``CellResult`` records to ``forecast_metadata``,
``forecast_predictions`` (including Tier 2 per-horizon-step ``explanations`` JSON), and
``backtest_oof``, so AutoML models participate in the leaderboard, backtest coverage views, and
downstream ensembling alongside Statistical, ML, Deep Learning, and BigQuery-native models.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..backtest import (
    OOF_COLUMNS,
    make_folds,
    resolve_geometry,
)
from ..calibration import apply_calibration, calibrate_from_oof, compare_arms, select_arm
from ..config import corrected_arm_for
from ..errors import EngineError, get_logger
from ..metrics import METRIC_NAMES, compute_metrics
from ..models import get_model
from ..models.base_model import DEFAULT_QUANTILES, PREDICTION_COLUMNS
from ..registry.ids import make_model_hash, make_run_id, vertex_automl_job_id
from ..seasonality import seasonal_period
from ..worker import (
    CellResult,
    _backtest_outcome,
    _rollup_metrics,
    _worker_id,
)

if TYPE_CHECKING:
    from ..config import RunConfig
    from ..settings import Settings

_log = get_logger(__name__)

_SPLIT_COL = "__sf_split__"


def _future_dates(last_date: pd.Timestamp, horizon: int, freq: str) -> pd.DatetimeIndex:
    """Return ``horizon`` future dates after ``last_date`` at ``freq`` (pure)."""
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:].as_unit("ns")


# Map scale-forecasting pandas/DataConfig frequencies to Vertex AI data_granularity_(unit, count).
_GRANULARITY_MAP: dict[str, tuple[str, int]] = {
    "D": ("day", 1),
    "W": ("week", 1),
    "W-SUN": ("week", 1),
    "W-MON": ("week", 1),
    "M": ("month", 1),
    "ME": ("month", 1),
    "MS": ("month", 1),
    "Q": ("month", 3),
    "QE": ("month", 3),
    "QS": ("month", 3),
    "Y": ("year", 1),
    "YE": ("year", 1),
    "YS": ("year", 1),
    "H": ("hour", 1),
    "h": ("hour", 1),
    "T": ("minute", 1),
    "min": ("minute", 1),
}

_VERTEX_ACCELERATOR_MAP: dict[str, str] = {
    "T4": "NVIDIA_TESLA_T4",
    "L4": "NVIDIA_L4",
    "A100": "NVIDIA_TESLA_A100",
    "A100_80GB": "NVIDIA_A100_80GB",
}


@dataclass(frozen=True)
class AutoMLModelPlan:
    """Resolved execution plan for one Vertex AI AutoML / Tabular Workflow model."""

    model_type: str
    automl_mode: str
    root_dir: str
    context_window: int
    horizon: int
    data_granularity_unit: str
    data_granularity_count: int
    optimization_objective: str
    train_budget_milli_node_hours: int
    quantiles: list[float]
    enable_explainability: bool
    stage_1_tuning_result_artifact_uri: str | None
    reuse_tuning_from_run_id: str | None
    max_num_trials: int
    trainer_service_account: str | None
    evaluation_dataflow_machine_type: str
    evaluation_dataflow_starting_num_workers: int
    evaluation_dataflow_max_num_workers: int
    evaluation_batch_predict_machine_type: str
    evaluation_batch_predict_starting_replica_count: int
    evaluation_batch_predict_max_replica_count: int
    stage_1_tuner_worker_pool_specs_override: list[dict[str, Any]] | None
    stage_2_trainer_worker_pool_specs_override: list[dict[str, Any]] | None
    column_specs: list[dict[str, str]]
    time_series_attribute_columns: list[str]
    available_at_forecast_columns: list[str]
    unavailable_at_forecast_columns: list[str]
    extra_params: dict[str, Any] = field(default_factory=dict)


def map_freq_to_granularity(freq: str) -> tuple[str, int]:
    """Map a pandas frequency string (`cfg.data.freq`) to Vertex AI `(unit, count)` (pure)."""
    if freq in _GRANULARITY_MAP:
        return _GRANULARITY_MAP[freq]
    upper = freq.upper()
    if upper in _GRANULARITY_MAP:
        return _GRANULARITY_MAP[upper]
    if upper.startswith("W"):
        return ("week", 1)
    if upper.startswith(("M", "BM")):
        return ("month", 1)
    if upper.startswith("Q"):
        return ("month", 3)
    if upper.startswith(("Y", "A")):
        return ("year", 1)
    if upper.startswith("H"):
        return ("hour", 1)
    return ("day", 1)


def resolve_quantiles(intervals: list[float], authored_quantiles: Any = None) -> list[float]:
    """Resolve the quantile list (max 5, including 0.5) for quantile objectives (pure)."""
    if isinstance(authored_quantiles, list) and authored_quantiles:
        qs = sorted({round(float(q), 4) for q in authored_quantiles} | {0.5})
        return qs[:5]
    q_set: set[float] = {0.5}
    for level in intervals:
        alpha = (1.0 - float(level)) / 2.0
        q_set.add(round(alpha, 4))
        q_set.add(round(1.0 - alpha, 4))
    return sorted(q_set)[:5]


def build_automl_column_specs(
    cfg: RunConfig, panel_df: pd.DataFrame
) -> tuple[list[dict[str, str]], list[str], list[str], list[str]]:
    """Classify panel columns into Vertex AI Forecasting covariate lists and transform specs (pure).

    Returns ``(column_specs, time_series_attribute_columns, available_at_forecast_columns,
    unavailable_at_forecast_columns)``.
    """
    ts_id_col = cfg.data.ts_id_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col

    present_cols = set(panel_df.columns)
    static_cols = [c for c in cfg.features.static_covariates if c in present_cols]
    future_cols = [c for c in cfg.features.future_covariates if c in present_cols]
    past_cols = [c for c in cfg.features.past_covariates if c in present_cols]

    # Engineered calendar/Fourier features are deterministic functions of timestamp and therefore
    # known in advance at forecast time.
    reserved = {ts_id_col, date_col, target_col, _SPLIT_COL} | set(static_cols) | set(past_cols)
    engineered_future = [
        c
        for c in panel_df.columns
        if c not in reserved and c not in future_cols and (c.startswith(("cal_", "sin_", "cos_")))
    ]

    time_series_attribute_columns = list(dict.fromkeys(static_cols))
    available_at_forecast_columns = list(
        dict.fromkeys([date_col, *future_cols, *engineered_future])
    )
    unavailable_at_forecast_columns = list(dict.fromkeys([target_col, *past_cols]))

    all_feature_cols = list(
        dict.fromkeys(
            [
                *time_series_attribute_columns,
                *available_at_forecast_columns,
                *unavailable_at_forecast_columns,
            ]
        )
    )
    column_specs: list[dict[str, str]] = []
    for col in all_feature_cols:
        if col == date_col:
            dtype = "timestamp"
        elif col == target_col or pd.api.types.is_numeric_dtype(panel_df[col]):
            dtype = "numeric"
        else:
            dtype = "categorical"
        column_specs.append({"column_name": col, "data_type": dtype})

    return (
        column_specs,
        time_series_attribute_columns,
        available_at_forecast_columns,
        unavailable_at_forecast_columns,
    )


def prepare_training_panel(
    panel_df: pd.DataFrame,
    cfg: RunConfig,
    *,
    horizon: int | None = None,
    train_end_by_series: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Prepare the BigQuery training table DataFrame with deterministic ``__sf_split__`` (pure).

    Slices each series to ``<= train_end_by_series[ts_id]`` when provided (backtest fold), sorts
    chronologically, and assigns ``TRAIN``, ``VALIDATE``, and ``TEST`` splits per series so Vertex
    AI Tabular Workflow's ``predefined_split_key`` receives a leak-free temporal split.
    """
    ts_id_col = cfg.data.ts_id_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col
    eff_horizon = horizon or cfg.data.horizon

    df = panel_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    df = df.sort_values([ts_id_col, date_col], kind="mergesort").reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for ts_id, grp in df.groupby(ts_id_col, sort=False):
        sub = grp
        if train_end_by_series is not None and str(ts_id) in train_end_by_series:
            cutoff = pd.Timestamp(train_end_by_series[str(ts_id)])
            sub = sub[sub[date_col] <= cutoff]
        sub = sub.dropna(subset=[target_col]).copy()
        n = len(sub)
        if n == 0:
            continue
        # Ensure VALIDATE and TEST each hold at least `min(eff_horizon, max(1, n // 5))` rows while
        # TRAIN holds at least 60% of the series history.
        eval_win = min(eff_horizon, max(1, n // 5))
        if n >= 3:
            test_start = n - eval_win
            val_start = max(1, test_start - eval_win)
            splits = (
                ["TRAIN"] * val_start
                + ["VALIDATE"] * (test_start - val_start)
                + ["TEST"] * (n - test_start)
            )
        else:
            splits = ["TRAIN"] * n
        sub[_SPLIT_COL] = splits
        frames.append(sub)

    if not frames:
        raise EngineError("prepare_training_panel produced 0 rows across all series")
    out = pd.concat(frames, ignore_index=True)
    out[ts_id_col] = out[ts_id_col].astype(str)
    for col in out.columns:
        if col not in (ts_id_col, date_col, _SPLIT_COL) and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out


def prepare_inference_panel(
    panel_df: pd.DataFrame,
    cfg: RunConfig,
    *,
    context_window: int,
    horizon: int,
    origin_by_series: dict[str, pd.Timestamp] | None = None,
    future_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the Vertex AI ``BatchPredictionJob`` input table (context + future NULL-target rows).

    Vertex AI Forecasting ``BatchPredictionJob`` requires each series to include its historical
    context rows leading up to the forecast origin followed immediately by ``horizon`` future rows
    where ``target_col`` is ``NULL`` (`NaN`) and ``available_at_forecast_columns`` (timestamp +
    future covariates) are populated.
    """
    ts_id_col = cfg.data.ts_id_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col

    df = panel_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    df = df.sort_values([ts_id_col, date_col], kind="mergesort").reset_index(drop=True)

    future_lookup: dict[str, pd.DataFrame] = {}
    if future_df is not None and not future_df.empty:
        fdf = future_df.copy()
        fdf[date_col] = pd.to_datetime(fdf[date_col])
        for tid, fgrp in fdf.groupby(ts_id_col, sort=False):
            future_lookup[str(tid)] = fgrp.sort_values(date_col, kind="mergesort")

    static_cols = [c for c in cfg.features.static_covariates if c in df.columns]
    future_cols = [c for c in cfg.features.future_covariates if c in df.columns]
    past_cols = [c for c in cfg.features.past_covariates if c in df.columns]

    keep_context = max(context_window, horizon * 2, 10)
    frames: list[pd.DataFrame] = []

    for ts_id, grp in df.groupby(ts_id_col, sort=False):
        tid = str(ts_id)
        if origin_by_series is not None and tid in origin_by_series:
            origin = pd.Timestamp(origin_by_series[tid])
            hist = (
                grp[grp[date_col] <= origin].dropna(subset=[target_col]).tail(keep_context).copy()
            )
            held_out = grp[grp[date_col] > origin].head(horizon).copy()
        else:
            hist = grp.dropna(subset=[target_col]).tail(keep_context).copy()
            held_out = pd.DataFrame()

        if hist.empty:
            continue

        last_date = pd.Timestamp(hist[date_col].iloc[-1])
        fut_dates = _future_dates(last_date, horizon, cfg.data.freq)

        fut = pd.DataFrame({ts_id_col: [tid] * horizon, date_col: fut_dates})
        fut[target_col] = np.nan

        # Populate static covariates from the last historical row.
        for scol in static_cols:
            fut[scol] = hist[scol].iloc[-1]

        # Populate future covariates from future_df, held-out fold rows, or forward-fill.
        for fcol in future_cols:
            if tid in future_lookup and fcol in future_lookup[tid].columns:
                merged = pd.merge(
                    fut[[date_col]],
                    future_lookup[tid][[date_col, fcol]].drop_duplicates(subset=[date_col]),
                    on=date_col,
                    how="left",
                )
                fut[fcol] = merged[fcol].ffill().bfill().fillna(hist[fcol].iloc[-1]).to_numpy()
            elif not held_out.empty and fcol in held_out.columns:
                merged = pd.merge(
                    fut[[date_col]],
                    held_out[[date_col, fcol]].drop_duplicates(subset=[date_col]),
                    on=date_col,
                    how="left",
                )
                fut[fcol] = merged[fcol].ffill().bfill().fillna(hist[fcol].iloc[-1]).to_numpy()
            else:
                fut[fcol] = hist[fcol].iloc[-1]

        # Past covariates are unavailable in the future horizon -> NULL.
        for pcol in past_cols:
            fut[pcol] = np.nan

        combined = pd.concat([hist, fut], ignore_index=True)
        if _SPLIT_COL in combined.columns:
            combined = combined.drop(columns=[_SPLIT_COL])
        frames.append(combined)

    if not frames:
        raise EngineError("prepare_inference_panel produced 0 rows across all series")
    out = pd.concat(frames, ignore_index=True)
    out[ts_id_col] = out[ts_id_col].astype(str)
    for col in out.columns:
        if col not in (ts_id_col, date_col, _SPLIT_COL) and pd.api.types.is_numeric_dtype(df[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out


def resolve_prior_tuning_artifact_uri(
    prior_run_id: str,
    model_type: str,
    *,
    settings: Settings,
    bq_client: Any = None,
) -> str | None:
    """Look up a previously saved ``stage_1_tuning_result_artifact_uri`` in BigQuery by ``run_id``.

    Checks ``forecast_metadata`` (``model_artifact`` and
    ``best_params.stage_1_tuning_result_artifact_uri``) and ``run_jobs.job_telemetry`` so users can
    warm-start a new Tabular Workflow run simply by setting
    ``"reuse_tuning_from_run_id": "<prior_run_id>"`` in ``model_params``.
    """
    from google.cloud import bigquery

    client = bq_client or bigquery.Client(project=settings.project_id)
    meta_table = settings.registry_table_ref("forecast_metadata")
    sql = (
        f"SELECT model_artifact, TO_JSON_STRING(best_params) AS best_params_json "
        f"FROM `{meta_table}` "
        f"WHERE run_id = @run_id AND model_type = @model_type "
        f"LIMIT 1"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", prior_run_id),
            bigquery.ScalarQueryParameter("model_type", "STRING", model_type),
        ]
    )
    try:
        rows = list(client.query(sql, job_config=job_config).result())
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "Could not query forecast_metadata for reuse_tuning_from_run_id=%r (%s)",
            prior_run_id,
            exc,
        )
        return None

    for row in rows:
        if row.get("best_params_json"):
            try:
                bp = json.loads(row["best_params_json"])
                uri = bp.get("stage_1_tuning_result_artifact_uri")
                if isinstance(uri, str) and uri.startswith("gs://"):
                    return uri
            except (ValueError, TypeError):
                pass
        art = row.get("model_artifact")
        if isinstance(art, str) and art.startswith("gs://") and "tuning" in art:
            return art
    return None


def plan_automl_model(
    cfg: RunConfig,
    model_type: str,
    panel_df: pd.DataFrame,
    *,
    run_id: str,
    horizon: int | None = None,
    artifact_root: str = "gs://scale-forecasting-artifacts",
    service_account: str | None = None,
    subnetwork_uri: str | None = None,
    prior_tuning_uri: str | None = None,
) -> AutoMLModelPlan:
    """Resolve the complete execution plan for one ``automl`` model (pure)."""
    fc = cfg.resolve_family_compute("automl")
    automl_mode = fc.automl_mode or cfg.compute.automl_mode
    authored = dict(cfg.model_params.get(model_type, {}))
    eff_horizon = int(horizon or cfg.data.horizon)

    default_ctx = max(eff_horizon, min(eff_horizon * 2, 60))
    context_window = int(authored.get("context_window", default_ctx))
    budget = int(authored.get("train_budget_milli_node_hours", 1000))
    objective = str(authored.get("optimization_objective", "minimize-rmse"))
    quantiles = resolve_quantiles([0.8, 0.95], authored.get("quantiles"))
    enable_explainability = bool(
        authored.get("enable_explainability", authored.get("generate_explanation", True))
    )
    max_num_trials = int(authored.get("max_num_trials", 15))

    tuning_uri = (
        str(authored["stage_1_tuning_result_artifact_uri"])
        if authored.get("stage_1_tuning_result_artifact_uri")
        else prior_tuning_uri
    )
    reuse_run_id = (
        str(authored["reuse_tuning_from_run_id"])
        if authored.get("reuse_tuning_from_run_id")
        else None
    )

    unit, count = map_freq_to_granularity(cfg.data.freq)
    if "data_granularity_unit" in authored:
        unit = str(authored["data_granularity_unit"])
    if "data_granularity_count" in authored:
        count = int(authored["data_granularity_count"])

    # Right-size Dataflow evaluation and BatchPrediction replica bounds so small/medium panels do
    # not default to 22-25 workers and never violate starting <= max.
    max_df_workers = int(
        authored.get("evaluation_dataflow_max_num_workers")
        or authored.get("dataflow_max_num_workers")
        or fc.max_workers
        or fc.workers
        or 2
    )
    start_df_workers = min(
        int(authored.get("evaluation_dataflow_starting_num_workers") or fc.min_workers or 1),
        max_df_workers,
    )
    df_machine = str(
        authored.get("evaluation_dataflow_machine_type")
        or authored.get("dataflow_machine_type")
        or "n1-standard-4"
    )

    max_bp_replicas = int(
        authored.get("evaluation_batch_predict_max_replica_count")
        or fc.max_workers
        or fc.workers
        or 2
    )
    start_bp_replicas = min(
        int(authored.get("evaluation_batch_predict_starting_replica_count") or fc.min_workers or 1),
        max_bp_replicas,
    )
    bp_machine = str(authored.get("evaluation_batch_predict_machine_type", "n1-highmem-8"))

    # Worker pool overrides for Stage 1 Tuner and Stage 2 Trainer when hardware=="gpu" or explicit
    # machine_type is configured.
    tuner_override: list[dict[str, Any]] | None = None
    trainer_override: list[dict[str, Any]] | None = None
    if (
        fc.hardware == "gpu"
        or fc.machine_type is not None
        or "trainer_machine_type" in authored
        or "trainer_replica_count" in authored
    ):
        mt = str(authored.get("trainer_machine_type") or fc.machine_type or "n1-standard-8")
        # Vertex AutoML Stage 1/2 custom training containers support n1-* / g2-* / a2-* shapes; map
        # n2-standard-* to n1-standard-* when overriding Tabular Workflow worker pools.
        if mt.startswith("n2-standard-"):
            mt = mt.replace("n2-standard-", "n1-standard-", 1)
        machine_spec: dict[str, Any] = {"machine_type": mt}
        if fc.hardware == "gpu" and fc.gpu_type:
            machine_spec["accelerator_type"] = _VERTEX_ACCELERATOR_MAP.get(
                fc.gpu_type, "NVIDIA_TESLA_T4"
            )
            machine_spec["accelerator_count"] = max(1, fc.accelerator_count)
        replicas = max(1, int(authored.get("trainer_replica_count") or fc.workers or 1))
        pool_spec = [{"machine_spec": machine_spec, "replica_count": replicas}]
        tuner_override = pool_spec
        trainer_override = pool_spec

    (
        col_specs,
        attr_cols,
        avail_cols,
        unavail_cols,
    ) = build_automl_column_specs(cfg, panel_df)

    extra: dict[str, Any] = {}
    for k in (
        "holiday_regions",
        "hierarchy_group_columns",
        "hierarchy_group_total_weight",
        "hierarchy_temporal_total_weight",
        "hierarchy_group_temporal_total_weight",
        "window_stride_length",
        "window_max_count",
        "additional_experiments",
        "run_evaluation",
        "enable_probabilistic_inference",
        "max_parallel_trial_count",
        "stage_1_num_parallel_trials",
        "stage_2_num_parallel_trials",
        "num_selected_trials",
        "dataflow_disk_size_gb",
    ):
        if k in authored:
            extra[k] = authored[k]

    df_subnetwork = authored.get("dataflow_subnetwork") or subnetwork_uri
    if df_subnetwork:
        extra["dataflow_subnetwork"] = str(df_subnetwork)
        extra["dataflow_use_public_ips"] = bool(authored.get("dataflow_use_public_ips", False))
    elif "dataflow_use_public_ips" in authored:
        extra["dataflow_use_public_ips"] = bool(authored["dataflow_use_public_ips"])

    resolved_sa = (
        str(authored.get("trainer_service_account") or authored.get("dataflow_service_account"))
        if (authored.get("trainer_service_account") or authored.get("dataflow_service_account"))
        else service_account
    )

    clean_root = artifact_root.rstrip("/")
    root_dir = f"{clean_root}/{run_id}/tabular_workflow/{model_type}"

    return AutoMLModelPlan(
        model_type=model_type,
        automl_mode=automl_mode,
        root_dir=root_dir,
        context_window=context_window,
        horizon=eff_horizon,
        data_granularity_unit=unit,
        data_granularity_count=count,
        optimization_objective=objective,
        train_budget_milli_node_hours=budget,
        quantiles=quantiles,
        enable_explainability=enable_explainability,
        stage_1_tuning_result_artifact_uri=tuning_uri,
        reuse_tuning_from_run_id=reuse_run_id,
        max_num_trials=max_num_trials,
        trainer_service_account=resolved_sa,
        evaluation_dataflow_machine_type=df_machine,
        evaluation_dataflow_starting_num_workers=start_df_workers,
        evaluation_dataflow_max_num_workers=max_df_workers,
        evaluation_batch_predict_machine_type=bp_machine,
        evaluation_batch_predict_starting_replica_count=start_bp_replicas,
        evaluation_batch_predict_max_replica_count=max_bp_replicas,
        stage_1_tuner_worker_pool_specs_override=tuner_override,
        stage_2_trainer_worker_pool_specs_override=trainer_override,
        column_specs=col_specs,
        time_series_attribute_columns=attr_cols,
        available_at_forecast_columns=avail_cols,
        unavailable_at_forecast_columns=unavail_cols,
        extra_params=extra,
    )


def compile_tabular_workflow_spec(
    plan: AutoMLModelPlan,
    *,
    project_id: str,
    region: str,
    train_bq_uri: str,
    ts_id_col: str,
    date_col: str,
    target_col: str,
    output_dir: str | Path,
) -> tuple[str, dict[str, Any]]:
    """Compile the GCPC Tabular Workflow for Forecasting YAML and parameter dict.

    Uses ``google_cloud_pipeline_components.preview.automl.forecasting`` pipeline builders:
    - ``vertex_l2l``: ``get_learn_to_learn_forecasting_pipeline_and_parameters``
    - ``vertex_tide``: ``get_time_series_dense_encoder_forecasting_pipeline_and_parameters``
    - ``vertex_tft``: ``get_temporal_fusion_transformer_forecasting_pipeline_and_parameters``
    - ``vertex_seq2seq``: ``get_sequence_to_sequence_forecasting_pipeline_and_parameters``
    """
    import shutil

    from google_cloud_pipeline_components.preview.automl import forecasting as gcpc_fc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass {"auto": [...]} so Vertex AI Feature Transform Engine resolves BigQuery native column
    # types (TIMESTAMP, DATETIME, DATE, FLOAT64, STRING) directly from BigQuery table stats.
    transformations: dict[str, list[str]] = {
        "auto": [spec["column_name"] for spec in plan.column_specs]
    }
    run_eval = bool(plan.extra_params.get("run_evaluation", False))

    common_kwargs: dict[str, Any] = {
        "project": project_id,
        "location": region,
        "root_dir": plan.root_dir,
        "time_column": date_col,
        "time_series_identifier_columns": [ts_id_col],
        "target_column": target_col,
        "forecast_horizon": plan.horizon,
        "context_window": plan.context_window,
        "optimization_objective": plan.optimization_objective,
        "transformations": transformations,
        "train_budget_milli_node_hours": float(plan.train_budget_milli_node_hours),
        "time_series_attribute_columns": plan.time_series_attribute_columns,
        "available_at_forecast_columns": plan.available_at_forecast_columns,
        "unavailable_at_forecast_columns": plan.unavailable_at_forecast_columns,
        "data_source_bigquery_table_path": train_bq_uri,
        "predefined_split_key": _SPLIT_COL,
        "feature_transform_engine_dataflow_machine_type": plan.evaluation_dataflow_machine_type,
        "feature_transform_engine_dataflow_max_num_workers": (
            plan.evaluation_dataflow_max_num_workers
        ),
        "evaluation_dataflow_machine_type": plan.evaluation_dataflow_machine_type,
        "evaluation_dataflow_starting_num_workers": plan.evaluation_dataflow_starting_num_workers,
        "evaluation_dataflow_max_num_workers": plan.evaluation_dataflow_max_num_workers,
        "evaluation_batch_predict_machine_type": plan.evaluation_batch_predict_machine_type,
        "evaluation_batch_predict_starting_replica_count": (
            plan.evaluation_batch_predict_starting_replica_count
        ),
        "evaluation_batch_predict_max_replica_count": (
            plan.evaluation_batch_predict_max_replica_count
        ),
        "run_evaluation": run_eval,
    }
    if run_eval:
        bq_parts = train_bq_uri.removeprefix("bq://").split(".")
        if len(bq_parts) >= 2:
            common_kwargs["evaluated_examples_bigquery_path"] = f"bq://{bq_parts[0]}.{bq_parts[1]}"

    if plan.stage_1_tuning_result_artifact_uri:
        common_kwargs["stage_1_tuning_result_artifact_uri"] = (
            plan.stage_1_tuning_result_artifact_uri
        )
    if plan.stage_1_tuner_worker_pool_specs_override:
        common_kwargs["stage_1_tuner_worker_pool_specs_override"] = (
            plan.stage_1_tuner_worker_pool_specs_override
        )
    if plan.stage_2_trainer_worker_pool_specs_override:
        common_kwargs["stage_2_trainer_worker_pool_specs_override"] = (
            plan.stage_2_trainer_worker_pool_specs_override
        )
    if plan.trainer_service_account:
        common_kwargs["dataflow_service_account"] = plan.trainer_service_account
    if "dataflow_subnetwork" in plan.extra_params:
        common_kwargs["dataflow_subnetwork"] = str(plan.extra_params["dataflow_subnetwork"])
    if "dataflow_use_public_ips" in plan.extra_params:
        common_kwargs["dataflow_use_public_ips"] = bool(
            plan.extra_params["dataflow_use_public_ips"]
        )

    if "holiday_regions" in plan.extra_params:
        common_kwargs["holiday_regions"] = plan.extra_params["holiday_regions"]
    if "window_stride_length" in plan.extra_params:
        common_kwargs["window_stride_length"] = plan.extra_params["window_stride_length"]
    if "window_max_count" in plan.extra_params:
        common_kwargs["window_max_count"] = plan.extra_params["window_max_count"]
    if "stage_1_num_parallel_trials" in plan.extra_params:
        common_kwargs["stage_1_num_parallel_trials"] = int(
            plan.extra_params["stage_1_num_parallel_trials"]
        )
    elif "max_parallel_trial_count" in plan.extra_params:
        common_kwargs["stage_1_num_parallel_trials"] = int(
            plan.extra_params["max_parallel_trial_count"]
        )
    if "stage_2_num_parallel_trials" in plan.extra_params:
        common_kwargs["stage_2_num_parallel_trials"] = int(
            plan.extra_params["stage_2_num_parallel_trials"]
        )
    elif "stage_1_num_parallel_trials" in common_kwargs:
        common_kwargs["stage_2_num_parallel_trials"] = common_kwargs["stage_1_num_parallel_trials"]

    if plan.model_type != "vertex_tft":
        if "num_selected_trials" in plan.extra_params:
            common_kwargs["num_selected_trials"] = int(plan.extra_params["num_selected_trials"])
        elif plan.max_num_trials < 10:
            common_kwargs["num_selected_trials"] = max(1, int(plan.max_num_trials))

    if plan.model_type in ("vertex_l2l", "vertex_tide"):
        # Per Vertex AI Forecasting docs, enable_probabilistic_inference is incompatible with
        # minimize-quantile-loss; pass quantiles directly when using minimize-quantile-loss.
        if plan.optimization_objective == "minimize-quantile-loss":
            common_kwargs["enable_probabilistic_inference"] = False
            common_kwargs["quantiles"] = plan.quantiles
        elif plan.extra_params.get("enable_probabilistic_inference"):
            common_kwargs["enable_probabilistic_inference"] = True
        if "hierarchy_group_columns" in plan.extra_params:
            common_kwargs["group_columns"] = plan.extra_params["hierarchy_group_columns"]
        if "hierarchy_group_total_weight" in plan.extra_params:
            common_kwargs["group_total_weight"] = float(
                plan.extra_params["hierarchy_group_total_weight"]
            )
        if "hierarchy_temporal_total_weight" in plan.extra_params:
            common_kwargs["temporal_total_weight"] = float(
                plan.extra_params["hierarchy_temporal_total_weight"]
            )
        if "hierarchy_group_temporal_total_weight" in plan.extra_params:
            common_kwargs["group_temporal_total_weight"] = float(
                plan.extra_params["hierarchy_group_temporal_total_weight"]
            )

    if plan.model_type == "vertex_tide":
        src_path, params = (
            gcpc_fc.get_time_series_dense_encoder_forecasting_pipeline_and_parameters(
                **common_kwargs
            )
        )
    elif plan.model_type == "vertex_seq2seq":
        src_path, params = gcpc_fc.get_sequence_to_sequence_forecasting_pipeline_and_parameters(
            **common_kwargs
        )
    elif plan.model_type == "vertex_tft":
        src_path, params = (
            gcpc_fc.get_temporal_fusion_transformer_forecasting_pipeline_and_parameters(
                **common_kwargs
            )
        )
    else:
        src_path, params = gcpc_fc.get_learn_to_learn_forecasting_pipeline_and_parameters(
            **common_kwargs
        )

    dest_path = out_dir / f"{plan.model_type}_pipeline.yaml"
    if Path(src_path).is_file() and Path(src_path).resolve() != dest_path.resolve():
        shutil.copyfile(src_path, dest_path)
        return str(dest_path), params
    return str(src_path), params


def extract_pipeline_artifacts(pipeline_job: Any) -> dict[str, Any]:
    """Extract tuning result GCS URI, uploaded Model resource name, and task telemetry (pure).

    Inspects ``pipeline_job.gca_resource.job_detail.task_details`` from a completed Vertex AI
    Tabular Workflow ``PipelineJob`` to capture:
    - ``stage_1_tuning_result_artifact_uri`` (for warm-starting future runs)
    - ``vertex_model_resource_name`` (or ``unmanaged_container_model_uri`` for BatchPrediction)
    - ``evaluation_metrics_uri``
    """
    info: dict[str, Any] = {
        "pipeline_job_resource_name": getattr(pipeline_job, "resource_name", None),
        "stage_1_tuning_result_artifact_uri": None,
        "vertex_model_resource_name": None,
        "unmanaged_container_model_uri": None,
        "evaluation_metrics_uri": None,
    }
    gca = getattr(pipeline_job, "gca_resource", None)
    job_detail = getattr(gca, "job_detail", None)
    task_details = getattr(job_detail, "task_details", None) or []

    for task in task_details:
        outputs = getattr(task, "outputs", None) or {}
        for out_key, out_val in outputs.items():
            artifacts = getattr(out_val, "artifacts", None) or []
            for art in artifacts:
                uri = getattr(art, "uri", "") or ""
                metadata = dict(getattr(art, "metadata", None) or {})
                if out_key == "tuning_result_output" and uri:
                    info["stage_1_tuning_result_artifact_uri"] = uri
                elif out_key == "model" and (uri or metadata.get("resourceName")):
                    info["vertex_model_resource_name"] = metadata.get("resourceName") or uri
                elif out_key == "unmanaged_container_model" and uri:
                    info["unmanaged_container_model_uri"] = uri
                elif out_key == "evaluation_metrics" and uri:
                    info["evaluation_metrics_uri"] = uri

    return info


def _to_seq(val: Any) -> list[Any] | None:
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return None


def _parse_json_or_dict(val: Any) -> dict[str, Any] | list[Any] | None:
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    seq = _to_seq(val)
    if seq is not None:
        return seq
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (ValueError, TypeError):
            return None
    return None


def _extract_prediction_value_and_bounds(
    pred_obj: Any,
    *,
    quantiles: list[float],
    intervals: list[float],
) -> tuple[float, float, float, float, float]:
    """Extract `(yhat, lower_80, upper_80, lower_95, upper_95)` from a Vertex prediction cell."""
    yhat = float("nan")
    q_map: dict[float, float] = {}

    if isinstance(pred_obj, (int, float, np.number)) and not isinstance(pred_obj, bool):
        yhat = float(pred_obj)
    else:
        parsed = _parse_json_or_dict(pred_obj)
        if isinstance(parsed, dict):
            raw_val = parsed.get("value")
            if raw_val is not None:
                try:
                    yhat = float(raw_val)
                except (ValueError, TypeError):
                    pass
            # In Vertex AI Forecasting BigQuery BatchPrediction output:
            # - `quantile_values` is ARRAY<FLOAT64> of quantile probabilities ([0.1, 0.5, 0.9])
            # - `quantile_predictions` is ARRAY<FLOAT64> of predicted values ([95.0, 105.5, 116.0])
            # In JSONL/test dicts, `quantiles` holds probabilities and `quantile_values` holds
            # predicted values. Support both schemas and numpy array materializations.
            raw_q_preds = parsed.get("quantile_predictions")
            q_preds_seq = _to_seq(raw_q_preds)
            if q_preds_seq is None and isinstance(raw_q_preds, dict):
                q_preds_seq = _to_seq(raw_q_preds.get("values"))
            raw_q_values = _to_seq(parsed.get("quantile_values"))
            raw_quantiles = _to_seq(parsed.get("quantiles"))

            if q_preds_seq is not None and raw_q_values is not None:
                q_keys = raw_q_values
                q_vals = q_preds_seq
            else:
                q_keys = raw_quantiles or quantiles
                q_vals = raw_q_values or _to_seq(parsed.get("values")) or q_preds_seq

            if isinstance(q_keys, list) and isinstance(q_vals, list) and len(q_keys) == len(q_vals):
                for qk, qv in zip(q_keys, q_vals, strict=False):
                    try:
                        q_map[round(float(qk), 4)] = float(qv)
                    except (ValueError, TypeError):
                        pass
            if math.isnan(yhat) and 0.5 in q_map:
                yhat = q_map[0.5]

    def _bound(level: float, is_upper: bool) -> float:
        if level not in intervals or not q_map:
            return float("nan")
        target_q = round(
            (1.0 + level) / 2.0 if is_upper else (1.0 - level) / 2.0,
            4,
        )
        if target_q in q_map:
            return q_map[target_q]
        # Nearest available quantile on the requested side of 0.5.
        candidates = [
            (abs(q - target_q), v) for q, v in q_map.items() if (q > 0.5 if is_upper else q < 0.5)
        ]
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
        return float("nan")

    return (
        yhat,
        _bound(0.8, False),
        _bound(0.8, True),
        _bound(0.95, False),
        _bound(0.95, True),
    )


def _extract_step_explanation(exp_obj: Any) -> dict[str, Any] | None:
    """Extract normalized `{"baseline_score": float, "attributions": {feature: float}}` (pure)."""
    parsed = _parse_json_or_dict(exp_obj)
    if not isinstance(parsed, dict):
        return None

    # Already normalized format
    if "attributions" in parsed and isinstance(parsed["attributions"], dict):
        attrs: dict[str, float] = {}
        for k, v in parsed["attributions"].items():
            if isinstance(v, (int, float, np.number)) and math.isfinite(float(v)):
                attrs[str(k)] = round(float(v), 6)
        base = parsed.get("baseline_score", 0.0)
        return {
            "baseline_score": round(float(base), 6)
            if isinstance(base, (int, float, np.number))
            else 0.0,
            "attributions": attrs,
        }

    # Vertex AI BatchPrediction explanation format:
    # {"attributions": [{"baseline_score": ..., "feature_attributions": {"col": val_or_list}}]}
    attributions_list = _to_seq(parsed.get("attributions"))
    if attributions_list:
        first = attributions_list[0]
        if isinstance(first, dict):
            baseline = first.get("baseline_score")
            if baseline is None:
                baseline = first.get("baselineScore")
            if baseline is None:
                baseline = first.get("baselineOutputValue", 0.0)
            feat_attrs = (
                first.get("feature_attributions")
                or first.get("featureAttributions")
                or _parse_json_or_dict(first.get("feature_attributions"))
                or {}
            )
            flat_attrs: dict[str, float] = {}
            if isinstance(feat_attrs, dict):
                for feat, raw in feat_attrs.items():
                    if isinstance(raw, (int, float, np.number)) and math.isfinite(float(raw)):
                        flat_attrs[str(feat)] = round(float(raw), 6)
                    else:
                        seq_raw = _to_seq(raw)
                        if seq_raw is not None:
                            numeric_vals = [
                                float(x)
                                for x in seq_raw
                                if isinstance(x, (int, float, np.number))
                                and math.isfinite(float(x))
                            ]
                            if numeric_vals:
                                # Sum attribution across temporal context lags for this feature.
                                flat_attrs[str(feat)] = round(float(sum(numeric_vals)), 6)
            if flat_attrs:
                return {
                    "baseline_score": round(float(baseline), 6)
                    if isinstance(baseline, (int, float, np.number))
                    and math.isfinite(float(baseline))
                    else 0.0,
                    "attributions": flat_attrs,
                }

    # Fallback: TFT native feature importance (`tft_feature_importance`)
    tft_imp = parsed.get("tft_feature_importance")
    if isinstance(tft_imp, dict):
        cols = _to_seq(tft_imp.get("attribute_columns")) or []
        weights = _to_seq(tft_imp.get("attribute_weights")) or []
        ctx_cols = _to_seq(tft_imp.get("context_columns")) or []
        ctx_weights = _to_seq(tft_imp.get("context_weights")) or []
        horizon_cols = _to_seq(tft_imp.get("horizon_columns")) or []
        horizon_weights = _to_seq(tft_imp.get("horizon_weights")) or []
        tft_attrs: dict[str, float] = {}
        for c_list, w_list in (
            (cols, weights),
            (ctx_cols, ctx_weights),
            (horizon_cols, horizon_weights),
        ):
            for c, w in zip(c_list, w_list, strict=False):
                if isinstance(w, (int, float, np.number)) and math.isfinite(float(w)):
                    tft_attrs[str(c)] = round(tft_attrs.get(str(c), 0.0) + float(w), 6)
        if tft_attrs:
            return {"baseline_score": 0.0, "attributions": tft_attrs}

    return None


def parse_batch_predictions_and_explanations(
    bq_pred_df: pd.DataFrame,
    *,
    ts_id_col: str,
    date_col: str,
    target_col: str,
    quantiles: list[float] | None = None,
    intervals: list[float] | None = None,
    expected_dates_by_series: dict[str, pd.DatetimeIndex] | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, list[dict[str, Any] | None]],
    dict[str, dict[str, float]],
]:
    """Parse Vertex AI ``BatchPredictionJob`` output table into predictions and explanations.

    Returns:
    1. ``predictions_by_series[ts_id]``: DataFrame with canonical ``PREDICTION_COLUMNS``
       (``ds``, ``yhat``, ``yhat_raw``, ``yhat_lower``, ``yhat_upper``, ``quantiles``).
    2. ``explanations_by_series[ts_id]``: Tier 2 per-horizon-step local explanation dicts.
    3. ``global_attributions_by_series[ts_id]``: Tier 1 series-level mean ``|attribution|`` per
       feature across the forecast horizon.
    """
    eff_quantiles = quantiles or [0.1, 0.5, 0.9]
    eff_intervals = intervals or [0.8, 0.95]

    pred_col_candidates = [
        f"predicted_{target_col}",
        "predicted_on_target",
        "prediction",
        "yhat",
    ]
    pred_col = next((c for c in pred_col_candidates if c in bq_pred_df.columns), None)
    if pred_col is None:
        pred_col = next(
            (c for c in bq_pred_df.columns if c.startswith("predicted_")),
            target_col if target_col in bq_pred_df.columns else None,
        )
    if pred_col is None:
        raise EngineError(
            f"Vertex BatchPrediction output is missing a prediction column "
            f"(columns: {list(bq_pred_df.columns)})"
        )

    exp_col = next(
        (c for c in ("explanation", "explanations") if c in bq_pred_df.columns),
        None,
    )

    df = bq_pred_df.copy()
    df[ts_id_col] = df[ts_id_col].astype(str)
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    if target_col in df.columns and target_col != pred_col:
        null_mask = df[target_col].isna()
        if null_mask.any():
            df = df[null_mask].copy()

    df = df.sort_values([ts_id_col, date_col], kind="mergesort").reset_index(drop=True)

    predictions_by_series: dict[str, pd.DataFrame] = {}
    explanations_by_series: dict[str, list[dict[str, Any] | None]] = {}
    global_attributions_by_series: dict[str, dict[str, float]] = {}

    for ts_id, grp in df.groupby(ts_id_col, sort=False):
        tid = str(ts_id)
        sub = grp
        if expected_dates_by_series is not None and tid in expected_dates_by_series:
            exp_dates = pd.to_datetime(expected_dates_by_series[tid]).tz_localize(None)
            sub = sub[sub[date_col].isin(exp_dates)]
        if sub.empty:
            continue

        ds_list: list[pd.Timestamp] = []
        yhat_list: list[float] = []
        lower_list: list[float] = []
        upper_list: list[float] = []
        quantiles_json_list: list[str] = []
        step_exps: list[dict[str, Any] | None] = []
        attr_accum: dict[str, list[float]] = {}

        for _, row in sub.iterrows():
            yhat, l80, u80, l95, u95 = _extract_prediction_value_and_bounds(
                row[pred_col], quantiles=eff_quantiles, intervals=eff_intervals
            )
            ds_list.append(pd.Timestamp(row[date_col]))
            yhat_list.append(yhat)
            lower_list.append(l80)
            upper_list.append(u80)
            q_dict = {
                k: v
                for k, v in (
                    ("0.025", l95),
                    ("0.1", l80),
                    ("0.5", yhat),
                    ("0.9", u80),
                    ("0.975", u95),
                )
                if math.isfinite(v)
            }
            quantiles_json_list.append(json.dumps(q_dict))

            step_exp = _extract_step_explanation(row[exp_col]) if exp_col else None
            step_exps.append(step_exp)
            if step_exp and isinstance(step_exp.get("attributions"), dict):
                for feat, val in step_exp["attributions"].items():
                    attr_accum.setdefault(feat, []).append(abs(float(val)))

        ds_ns = pd.DatetimeIndex(ds_list).as_unit("ns")
        pred_df = pd.DataFrame(
            {
                "ds": ds_ns,
                "yhat": np.asarray(yhat_list, dtype=float),
                "yhat_raw": np.asarray(yhat_list, dtype=float),
                "yhat_lower": np.asarray(lower_list, dtype=float),
                "yhat_upper": np.asarray(upper_list, dtype=float),
                "quantiles": pd.array(quantiles_json_list, dtype="string"),
            },
            columns=list(PREDICTION_COLUMNS),
        )
        predictions_by_series[tid] = pred_df
        if any(e is not None for e in step_exps):
            explanations_by_series[tid] = step_exps
        if attr_accum:
            global_attributions_by_series[tid] = {
                feat: round(float(np.mean(vals)), 6) for feat, vals in attr_accum.items()
            }

    return predictions_by_series, explanations_by_series, global_attributions_by_series


def _stage_dataframe_to_bq(
    df: pd.DataFrame,
    table_id: str,
    *,
    project_id: str,
    bq_client: Any = None,
) -> str:
    """Upload a staging DataFrame to BigQuery (`WRITE_TRUNCATE`) and return `bq://...` URI."""
    from google.cloud import bigquery

    client = bq_client or bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    load_job.result()
    return f"bq://{table_id}"


def _drop_bq_tables(table_ids: list[str], *, project_id: str, bq_client: Any = None) -> None:
    """Best-effort cleanup of temporary BigQuery staging and batch-prediction output tables."""
    from google.cloud import bigquery

    client = bq_client or bigquery.Client(project=project_id)
    for tid in table_ids:
        clean_id = tid.removeprefix("bq://")
        try:
            client.delete_table(clean_id, not_found_ok=True)
        except Exception as exc:  # noqa: BLE001
            _log.debug("Could not delete temporary BigQuery table %s: %s", clean_id, exc)


def _start_stage1_trial_cap_watcher(
    *,
    project_id: str,
    region: str,
    pipeline_job_id: str,
    max_num_trials: int,
    num_selected_trials: int = 1,
    poll_interval_s: float = 20.0,
) -> threading.Event:
    """Monitor child Stage-1 HyperparameterTuningJob and cancel once `max_num_trials` succeed.

    Vertex AI's precompiled Tabular Workflow KFP template hardcodes `max_trial_count = 300` and
    `MIN_COMPLETED_TRIALS_COUNTS_STAGE_ONE_TUNER = 45`, only exiting early if the child
    `HyperparameterTuningJob` transitions to `JOB_STATE_CANCELLED` / `JOB_STATE_SUCCEEDED`. When a
    user specifies `max_num_trials < 45`, this daemon thread polls the child tuning job linked via
    `vertex-ai-pipelines-run-billing-id` and requests cancellation once `max(max_num_trials,
    num_selected_trials)` trials reach `SUCCEEDED`, allowing `automl-forecasting-stage-1-tuner`
    to immediately write `tuning_result_output` and advance to `automl-forecasting-ensemble-2`.
    """
    stop_event = threading.Event()
    target_trials = max(1, int(max_num_trials), int(num_selected_trials))

    def _watch() -> None:
        try:
            from google.cloud import aiplatform_v1

            api_endpoint = f"{region}-aiplatform.googleapis.com"
            pclient = aiplatform_v1.PipelineServiceClient(
                client_options={"api_endpoint": api_endpoint}
            )
            jclient = aiplatform_v1.JobServiceClient(client_options={"api_endpoint": api_endpoint})
            parent = f"projects/{project_id}/locations/{region}"
            p_name = f"{parent}/pipelineJobs/{pipeline_job_id}"
            billing_id: str | None = None

            while not stop_event.wait(poll_interval_s):
                try:
                    if not billing_id:
                        p = pclient.get_pipeline_job(name=p_name)
                        billing_id = dict(getattr(p, "labels", None) or {}).get(
                            "vertex-ai-pipelines-run-billing-id"
                        )
                        if not billing_id:
                            continue
                    for h in jclient.list_hyperparameter_tuning_jobs(parent=parent):
                        h_labels = dict(getattr(h, "labels", None) or {})
                        if h_labels.get("vertex-ai-pipelines-run-billing-id") != billing_id:
                            continue
                        state_name = getattr(getattr(h, "state", None), "name", "")
                        if state_name in (
                            "JOB_STATE_SUCCEEDED",
                            "JOB_STATE_CANCELLED",
                            "JOB_STATE_CANCELLING",
                            "JOB_STATE_FAILED",
                        ):
                            return
                        if state_name == "JOB_STATE_RUNNING":
                            succeeded = sum(
                                1
                                for t in (getattr(h, "trials", None) or [])
                                if getattr(getattr(t, "state", None), "name", "") == "SUCCEEDED"
                            )
                            if succeeded >= target_trials:
                                _log.info(
                                    "Stage-1 HyperparameterTuningJob %s reached %d/%d succeeded "
                                    "trials; requesting cancellation to advance pipeline %s.",
                                    h.name,
                                    succeeded,
                                    target_trials,
                                    pipeline_job_id,
                                )
                                jclient.cancel_hyperparameter_tuning_job(name=h.name)
                                return
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            return

    thread = threading.Thread(
        target=_watch,
        name=f"sf-stage1-cap-{pipeline_job_id[:24]}",
        daemon=True,
    )
    thread.start()
    return stop_event


def _find_reusable_pipeline_artifacts(
    *,
    project_id: str,
    region: str,
    job_prefix: str,
    pipeline_client: Any = None,
) -> dict[str, Any] | None:
    """Return extracted artifacts if any attempt of ``(run_id, model, fold)`` already succeeded.

    When a multi-stage Tabular Workflow run is retried or resumed after a downstream
    ``BatchPredictionJob`` stockout, checking ``PipelineServiceClient`` for an existing
    ``PIPELINE_STATE_SUCCEEDED`` ``PipelineJob`` across attempt numbers (``-aN-`` down to ``-a1-``)
    avoids re-running a multi-hour training pipeline whose model is already uploaded to Vertex AI
    Model Registry.
    """
    import re
    from types import SimpleNamespace

    candidates: list[str] = [vertex_automl_job_id(job_prefix)]
    match = re.search(r"-a(\d+)-", job_prefix)
    if match:
        current_attempt = int(match.group(1))
        for prev in range(current_attempt - 1, 0, -1):
            alt_prefix = re.sub(r"-a\d+-", f"-a{prev}-", job_prefix, count=1)
            alt_id = vertex_automl_job_id(alt_prefix)
            if alt_id not in candidates:
                candidates.append(alt_id)

    try:
        if pipeline_client is None:
            from google.cloud.aiplatform_v1.services.pipeline_service import (
                PipelineServiceClient,
            )

            pipeline_client = PipelineServiceClient(
                client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
            )
    except Exception:  # noqa: BLE001
        return None

    for candidate_id in candidates:
        name = f"projects/{project_id}/locations/{region}/pipelineJobs/{candidate_id}"
        try:
            raw_pjob = pipeline_client.get_pipeline_job(name=name)
        except Exception:  # noqa: BLE001
            continue
        state_name = getattr(getattr(raw_pjob, "state", None), "name", "")
        if state_name != "PIPELINE_STATE_SUCCEEDED":
            continue
        proxy = SimpleNamespace(
            resource_name=getattr(raw_pjob, "name", name),
            gca_resource=raw_pjob,
        )
        info = extract_pipeline_artifacts(proxy)
        info["pipeline_job_id"] = candidate_id
        info["pipeline_resource_name"] = getattr(raw_pjob, "name", name)
        if info.get("vertex_model_resource_name") or info.get("unmanaged_container_model_uri"):
            _log.info(
                "Reusing completed Tabular Workflow PipelineJob %s (model=%s, tuning_uri=%s).",
                candidate_id,
                info.get("vertex_model_resource_name") or info.get("unmanaged_container_model_uri"),
                info.get("stage_1_tuning_result_artifact_uri"),
            )
            return info
    return None


_BATCH_PREDICT_FALLBACK_SHAPES: tuple[str, ...] = (
    "n1-highmem-8",
    "n1-standard-8",
    "n1-standard-16",
    "n1-highmem-4",
    "n1-standard-4",
)


def _is_batch_predict_stockout(exc: BaseException) -> bool:
    """Return True if ``exc`` indicates a transient GCE machine-type stockout."""
    msg = str(exc).upper()
    return any(
        token in msg
        for token in (
            "MACHINE TYPE TEMPORARILY UNAVAILABLE",
            "ZONE_RESOURCE_POOL_EXHAUSTED",
            "STOCKOUT",
            "CODE: 14",
            "RESOURCE_EXHAUSTED",
        )
    )


def _batch_predict_with_fallback(
    model_resource: Any,
    *,
    job_display_name: str,
    bigquery_source: str,
    bigquery_destination_prefix: str,
    preferred_machine_type: str,
    starting_replica_count: int,
    max_replica_count: int,
    generate_explanation: bool,
) -> Any:
    """Run ``model_resource.batch_predict`` with automatic machine-shape fallback on stockout."""
    candidates: list[str] = []
    for mt in (preferred_machine_type, *_BATCH_PREDICT_FALLBACK_SHAPES):
        if mt and mt not in candidates:
            candidates.append(mt)

    last_exc: BaseException | None = None
    for idx, machine_type in enumerate(candidates):
        display = job_display_name if idx == 0 else f"{job_display_name}-r{idx}"
        try:
            return model_resource.batch_predict(
                job_display_name=display,
                bigquery_source=bigquery_source,
                instances_format="bigquery",
                predictions_format="bigquery",
                bigquery_destination_prefix=bigquery_destination_prefix,
                machine_type=machine_type,
                starting_replica_count=starting_replica_count,
                max_replica_count=max_replica_count,
                generate_explanation=generate_explanation,
                sync=True,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_batch_predict_stockout(exc) or idx == len(candidates) - 1:
                raise
            next_mt = candidates[idx + 1]
            _log.warning(
                "BatchPredictionJob %s encountered GCE machine stockout on %s (%s); "
                "retrying with fallback machine_type=%s.",
                display,
                machine_type,
                exc,
                next_mt,
            )
    if last_exc is not None:  # pragma: no cover
        raise last_exc
    raise RuntimeError("BatchPredictionJob fallback exhausted without running")  # pragma: no cover


def _execute_automl_fit_and_predict(
    plan: AutoMLModelPlan,
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    cfg: RunConfig,
    *,
    settings: Settings,
    job_prefix: str,
    bq_client: Any = None,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, list[dict[str, Any] | None]],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    """Run one Vertex AI AutoML / Tabular Workflow fit + BatchPredictionJob and parse outputs."""
    from google.cloud import aiplatform, bigquery

    client = bq_client or bigquery.Client(project=settings.project_id)
    aiplatform.init(
        project=settings.project_id,
        location=settings.region,
        staging_bucket=plan.root_dir,
    )

    safe_prefix = job_prefix.replace("-", "_")
    train_table_id = f"{settings.project_id}.{settings.dataset_id}._sf_automl_train_{safe_prefix}"
    infer_table_id = f"{settings.project_id}.{settings.dataset_id}._sf_automl_infer_{safe_prefix}"
    temp_tables = [train_table_id, infer_table_id]

    try:
        infer_bq_uri = _stage_dataframe_to_bq(
            infer_df, infer_table_id, project_id=settings.project_id, bq_client=client
        )

        artifacts_info: dict[str, Any] = {}
        model_resource: Any = None

        if plan.automl_mode == "tabular_workflow":
            reused_info: dict[str, Any] | None = None
            if getattr(aiplatform.PipelineJob, "__module__", "").startswith("google.cloud"):
                reused_info = _find_reusable_pipeline_artifacts(
                    project_id=settings.project_id,
                    region=settings.region,
                    job_prefix=job_prefix,
                )
            if reused_info is not None:
                artifacts_info = reused_info
            else:
                train_bq_uri = _stage_dataframe_to_bq(
                    train_df, train_table_id, project_id=settings.project_id, bq_client=client
                )
                with tempfile.TemporaryDirectory(prefix="sf_automl_") as tmp_dir:
                    template_path, parameter_values = compile_tabular_workflow_spec(
                        plan,
                        project_id=settings.project_id,
                        region=settings.region,
                        train_bq_uri=train_bq_uri,
                        ts_id_col=cfg.data.ts_id_col,
                        date_col=cfg.data.date_col,
                        target_col=cfg.data.target_col,
                        output_dir=tmp_dir,
                    )
                    pipeline_job_id = vertex_automl_job_id(job_prefix)
                    pjob = aiplatform.PipelineJob(
                        display_name=pipeline_job_id,
                        job_id=pipeline_job_id,
                        template_path=template_path,
                        parameter_values=parameter_values,
                        pipeline_root=plan.root_dir,
                        enable_caching=False,
                        project=settings.project_id,
                        location=settings.region,
                    )
                    watcher_stop: threading.Event | None = None
                    if (
                        plan.max_num_trials < 45
                        and not plan.stage_1_tuning_result_artifact_uri
                        and type(pjob).__module__.startswith("google.cloud")
                    ):
                        selected_trials = int(parameter_values.get("num_selected_trials") or 1)
                        watcher_stop = _start_stage1_trial_cap_watcher(
                            project_id=settings.project_id,
                            region=settings.region,
                            pipeline_job_id=pipeline_job_id,
                            max_num_trials=plan.max_num_trials,
                            num_selected_trials=selected_trials,
                        )
                    try:
                        pjob.run(
                            service_account=plan.trainer_service_account,
                            sync=True,
                        )
                    finally:
                        if watcher_stop is not None:
                            watcher_stop.set()
                    artifacts_info = extract_pipeline_artifacts(pjob)
                    artifacts_info["pipeline_job_id"] = pipeline_job_id
                    artifacts_info["pipeline_resource_name"] = getattr(pjob, "resource_name", None)
                    if (
                        not artifacts_info.get("stage_1_tuning_result_artifact_uri")
                        and plan.stage_1_tuning_result_artifact_uri
                    ):
                        artifacts_info["stage_1_tuning_result_artifact_uri"] = (
                            plan.stage_1_tuning_result_artifact_uri
                        )

            if artifacts_info.get("vertex_model_resource_name"):
                model_resource = aiplatform.Model(
                    model_name=artifacts_info["vertex_model_resource_name"]
                )
            elif artifacts_info.get("unmanaged_container_model_uri"):
                unmanaged_uri = artifacts_info["unmanaged_container_model_uri"]
                model_resource = aiplatform.Model.upload(
                    display_name=str(
                        artifacts_info.get("pipeline_job_id") or vertex_automl_job_id(job_prefix)
                    ),
                    artifact_uri=unmanaged_uri,
                    serving_container_image_uri=(
                        "us-docker.pkg.dev/vertex-ai/automl-tabular/prediction-server:prod"
                    ),
                    sync=True,
                )
                artifacts_info["vertex_model_resource_name"] = model_resource.resource_name
        else:
            train_bq_uri = _stage_dataframe_to_bq(
                train_df, train_table_id, project_id=settings.project_id, bq_client=client
            )
            ds = aiplatform.TimeSeriesDataset.create(
                display_name=f"sf-ds-{job_prefix}",
                bq_source=train_bq_uri,
                sync=True,
            )
            col_types = {spec["column_name"]: spec["data_type"] for spec in plan.column_specs}
            job_cls_map = {
                "vertex_l2l": aiplatform.AutoMLForecastingTrainingJob,
                "vertex_tide": getattr(
                    aiplatform,
                    "TimeSeriesDenseEncoderForecastingTrainingJob",
                    aiplatform.AutoMLForecastingTrainingJob,
                ),
                "vertex_tft": getattr(
                    aiplatform,
                    "TemporalFusionTransformerForecastingTrainingJob",
                    aiplatform.AutoMLForecastingTrainingJob,
                ),
                "vertex_seq2seq": getattr(
                    aiplatform,
                    "Seq2SeqPlusForecastingTrainingJob",
                    aiplatform.AutoMLForecastingTrainingJob,
                ),
            }
            trainer_cls = job_cls_map[plan.model_type]
            training_job = trainer_cls(
                display_name=f"sf-train-{job_prefix}",
                optimization_objective=plan.optimization_objective,
                column_types=col_types,
            )
            model_resource = training_job.run(
                dataset=ds,
                target_column=cfg.data.target_col,
                time_column=cfg.data.date_col,
                time_series_identifier_column=cfg.data.ts_id_col,
                unavailable_at_forecast_columns=plan.unavailable_at_forecast_columns,
                available_at_forecast_columns=plan.available_at_forecast_columns,
                time_series_attribute_columns=plan.time_series_attribute_columns,
                forecast_horizon=plan.horizon,
                context_window=plan.context_window,
                data_granularity_unit=plan.data_granularity_unit,
                data_granularity_count=plan.data_granularity_count,
                predefined_split_column_name=_SPLIT_COL,
                budget_milli_node_hours=plan.train_budget_milli_node_hours,
                quantiles=plan.quantiles
                if plan.optimization_objective == "minimize-quantile-loss"
                else None,
                sync=True,
            )
            artifacts_info = {
                "vertex_model_resource_name": getattr(model_resource, "resource_name", None),
                "stage_1_tuning_result_artifact_uri": plan.stage_1_tuning_result_artifact_uri,
            }

        if model_resource is None:
            raise EngineError(
                f"Vertex AI AutoML pipeline for '{plan.model_type}' completed without producing a "
                f"Model resource (artifacts: {artifacts_info})"
            )

        bp_dest_prefix = f"bq://{settings.project_id}.{settings.dataset_id}"
        bp_job = _batch_predict_with_fallback(
            model_resource,
            job_display_name=f"sf-bp-{job_prefix}",
            bigquery_source=infer_bq_uri,
            bigquery_destination_prefix=bp_dest_prefix,
            preferred_machine_type=plan.evaluation_batch_predict_machine_type,
            starting_replica_count=plan.evaluation_batch_predict_starting_replica_count,
            max_replica_count=plan.evaluation_batch_predict_max_replica_count,
            generate_explanation=plan.enable_explainability,
        )
        output_info = getattr(bp_job, "output_info", None)
        bq_out_table = getattr(output_info, "bigquery_output_table", "") or ""
        bq_out_dataset = (
            getattr(output_info, "bigquery_output_dataset", "")
            or f"bq://{settings.project_id}.{settings.dataset_id}"
        ).removeprefix("bq://")
        clean_dataset = bq_out_dataset.replace(":", ".")
        full_pred_table = f"{clean_dataset}.{bq_out_table}" if bq_out_table else ""
        if full_pred_table:
            temp_tables.append(full_pred_table)
            if bq_out_table.startswith("predictions_"):
                ts_suffix = bq_out_table.removeprefix("predictions_")
                temp_tables.append(f"{clean_dataset}.errors_{ts_suffix}")
                temp_tables.append(f"{clean_dataset}.errors_validation_{ts_suffix}")

        bq_pred_df = client.query(f"SELECT * FROM `{full_pred_table}`").to_dataframe()

        expected_dates: dict[str, pd.DatetimeIndex] = {}
        null_infer = infer_df[infer_df[cfg.data.target_col].isna()]
        for tid, grp in null_infer.groupby(cfg.data.ts_id_col, sort=False):
            expected_dates[str(tid)] = pd.DatetimeIndex(pd.to_datetime(grp[cfg.data.date_col]))

        preds_by_s, exps_by_s, global_attrs_by_s = parse_batch_predictions_and_explanations(
            bq_pred_df,
            ts_id_col=cfg.data.ts_id_col,
            date_col=cfg.data.date_col,
            target_col=cfg.data.target_col,
            quantiles=plan.quantiles,
            intervals=[0.8, 0.95],
            expected_dates_by_series=expected_dates,
        )
        return preds_by_s, exps_by_s, global_attrs_by_s, artifacts_info
    finally:
        _drop_bq_tables(temp_tables, project_id=settings.project_id, bq_client=client)


def execute_automl_model_cells(
    panel_df: pd.DataFrame,
    model_type: str,
    cfg: RunConfig,
    *,
    run_id: str,
    settings: Settings,
    future_df: pd.DataFrame | None = None,
    bq_client: Any = None,
    fit_and_predict_fn: Any = None,
    job_id: str | None = None,
) -> tuple[list[CellResult], dict[str, Any]]:
    """Execute backtest folds (if enabled) and final forward forecast for one ``automl`` model.

    Returns ``(cell_results, model_telemetry)`` where ``cell_results`` has one `CellResult` per
    series in ``panel_df`` with populated metrics, calibrated predictions, ``explanations``,
    ``diagnostics["feature_attributions"]``, and ``best_params``.
    """
    t0 = time.perf_counter()
    cell_started_at = datetime.now(UTC)
    worker_id = _worker_id()
    dispatch = fit_and_predict_fn or _execute_automl_fit_and_predict
    ts_id_col = cfg.data.ts_id_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col
    base_prefix = vertex_automl_job_id(job_id) if job_id else f"sf-{run_id[-8:]}"
    short_model = model_type.removeprefix("vertex_")
    m_period = seasonal_period(cfg.data.freq)

    df = panel_df.copy()
    df[ts_id_col] = df[ts_id_col].astype(str)
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    df = df.sort_values([ts_id_col, date_col], kind="mergesort").reset_index(drop=True)

    series_groups: dict[str, pd.DataFrame] = {
        str(tid): grp.reset_index(drop=True) for tid, grp in df.groupby(ts_id_col, sort=False)
    }
    uids = sorted(series_groups.keys())

    authored = dict(cfg.model_params.get(model_type, {}))
    prior_tuning_uri: str | None = None
    if not authored.get("stage_1_tuning_result_artifact_uri") and authored.get(
        "reuse_tuning_from_run_id"
    ):
        prior_tuning_uri = resolve_prior_tuning_artifact_uri(
            str(authored["reuse_tuning_from_run_id"]),
            model_type,
            settings=settings,
            bq_client=bq_client,
        )

    artifact_root = settings.artifact_root
    default_sa = (
        getattr(settings, "compute_service_account", None)
        or os.environ.get("SF_COMPUTE_SA")
        or None
    )
    default_subnet = (
        getattr(settings, "subnetwork_uri", None) or os.environ.get("SF_SUBNETWORK_URI") or None
    )
    plan = plan_automl_model(
        cfg,
        model_type,
        df,
        run_id=run_id,
        horizon=cfg.data.horizon,
        artifact_root=artifact_root,
        service_account=default_sa,
        subnetwork_uri=default_subnet,
        prior_tuning_uri=prior_tuning_uri,
    )

    oof_by_uid: dict[str, pd.DataFrame | None] = dict.fromkeys(uids, None)
    metrics_by_uid: dict[str, dict[str, float]] = {
        uid: {name: float("nan") for name in METRIC_NAMES} for uid in uids
    }
    bt_status_by_uid: dict[str, str | None] = dict.fromkeys(uids, None)
    bt_note_by_uid: dict[str, str | None] = dict.fromkeys(uids, None)
    n_folds_by_uid: dict[str, int | None] = dict.fromkeys(uids, None)
    bt_refit_by_uid: dict[str, str | None] = dict.fromkeys(uids, None)
    ach_step_by_uid: dict[str, int | None] = dict.fromkeys(uids, None)
    ach_min_by_uid: dict[str, int | None] = dict.fromkeys(uids, None)
    first_val_by_uid: dict[str, date | None] = dict.fromkeys(uids, None)
    last_val_by_uid: dict[str, date | None] = dict.fromkeys(uids, None)
    n_fits = 0
    train_rows_total = 0

    # 1. Backtest folds (when cfg.backtest.enabled)
    if cfg.backtest.enabled:
        folds_by_uid = {uid: make_folds(len(series_groups[uid]), cfg) for uid in uids}
        max_folds = max((len(fl) for fl in folds_by_uid.values()), default=0)
        oof_chunks_by_uid: dict[str, list[pd.DataFrame]] = {u: [] for u in uids}
        fmetrics_by_uid: dict[str, list[dict[str, float]]] = {u: [] for u in uids}
        bt_horizon = cfg.backtest.gap + cfg.backtest.horizon

        for fold_idx in range(max_folds):
            train_end_map: dict[str, pd.Timestamp] = {}
            fold_lookup: dict[str, Any] = {}
            for uid in uids:
                folds = folds_by_uid[uid]
                if fold_idx < len(folds):
                    f = folds[fold_idx]
                    sub = series_groups[uid]
                    train_slice = sub.iloc[: f.train_end]
                    train_end_map[uid] = pd.Timestamp(train_slice[date_col].iloc[-1])
                    fold_lookup[uid] = f
                    train_rows_total += len(train_slice)

            if not train_end_map:
                continue

            active_df = df[df[ts_id_col].isin(train_end_map)].copy()
            fold_plan = plan_automl_model(
                cfg,
                model_type,
                active_df,
                run_id=run_id,
                horizon=bt_horizon,
                artifact_root=artifact_root,
                service_account=default_sa,
                subnetwork_uri=default_subnet,
                prior_tuning_uri=prior_tuning_uri,
            )
            fold_train = prepare_training_panel(
                active_df, cfg, horizon=bt_horizon, train_end_by_series=train_end_map
            )
            fold_infer = prepare_inference_panel(
                active_df,
                cfg,
                context_window=fold_plan.context_window,
                horizon=bt_horizon,
                origin_by_series=train_end_map,
            )
            fold_preds, _, _, fold_artifacts = dispatch(
                fold_plan,
                fold_train,
                fold_infer,
                cfg,
                settings=settings,
                job_prefix=f"{base_prefix}-{short_model}-f{fold_idx + 1}",
                bq_client=bq_client,
            )
            n_fits += 1
            if prior_tuning_uri is None and fold_artifacts.get(
                "stage_1_tuning_result_artifact_uri"
            ):
                prior_tuning_uri = str(fold_artifacts["stage_1_tuning_result_artifact_uri"])

            for uid, f in fold_lookup.items():
                pred_df = fold_preds.get(uid)
                if pred_df is None or pred_df.empty:
                    continue
                sub = series_groups[uid]
                train_slice = sub.iloc[: f.train_end]
                val_slice = sub.iloc[f.val_start : f.val_end]
                val_dates = pd.to_datetime(val_slice[date_col]).dt.tz_localize(None)
                merged = pd.merge(
                    val_slice[[date_col, target_col]].assign(ds=val_dates),
                    pred_df,
                    on="ds",
                    how="inner",
                )
                if merged.empty:
                    continue
                y_train_raw = train_slice[target_col].to_numpy(dtype=float)
                y_val = merged[target_col].to_numpy(dtype=float)
                yhat_raw = merged["yhat_raw"].to_numpy(dtype=float)
                yhat_adjusted = merged["yhat"].to_numpy(dtype=float)
                yhat = yhat_raw if cfg.output.point_forecast == "raw" else yhat_adjusted
                lower = merged["yhat_lower"].to_numpy(dtype=float)
                upper = merged["yhat_upper"].to_numpy(dtype=float)
                fmetrics_by_uid[uid].append(
                    compute_metrics(
                        y_val,
                        yhat,
                        y_train=y_train_raw,
                        lower=lower,
                        upper=upper,
                        seasonal_period=m_period,
                    )
                )
                cutoff = pd.Timestamp(train_end_map[uid]).normalize()
                oof_chunks_by_uid[uid].append(
                    pd.DataFrame(
                        {
                            "ds": pd.to_datetime(merged["ds"]).to_numpy(),
                            "fold_id": f.fold_id,
                            "y_true": y_val,
                            "yhat": yhat,
                            "yhat_raw": yhat_raw,
                            "yhat_adjusted": yhat_adjusted,
                            "yhat_lower": lower,
                            "yhat_upper": upper,
                            "cutoff_date": cutoff,
                            "horizon_step": np.arange(1, len(y_val) + 1),
                            "yhat_stale": np.full(len(y_val), np.nan),
                        },
                        columns=list(OOF_COLUMNS),
                    )
                )

        for uid in uids:
            sub = series_groups[uid]
            flist = folds_by_uid[uid]
            n_ach = len(fmetrics_by_uid[uid])
            n_folds_by_uid[uid] = n_ach
            metrics_by_uid[uid] = _rollup_metrics(fmetrics_by_uid[uid])
            bt_status_by_uid[uid], bt_note_by_uid[uid] = _backtest_outcome(n_ach, cfg, sub)
            if n_ach > 0:
                oof_by_uid[uid] = pd.concat(oof_chunks_by_uid[uid], ignore_index=True)
                bt_refit_by_uid[uid] = "per_fold"
                geom = resolve_geometry(len(sub), cfg)
                ach_step_by_uid[uid] = geom.step
                ach_min_by_uid[uid] = geom.min_train
                first_val_by_uid[uid] = pd.Timestamp(sub[date_col].iloc[flist[0].val_start]).date()
                last_val_by_uid[uid] = pd.Timestamp(
                    sub[date_col].iloc[flist[-1].val_end - 1]
                ).date()

    # 2. Final full-history fit + forward forecast
    if prior_tuning_uri and not plan.stage_1_tuning_result_artifact_uri:
        plan = plan_automl_model(
            cfg,
            model_type,
            df,
            run_id=run_id,
            horizon=cfg.data.horizon,
            artifact_root=artifact_root,
            service_account=default_sa,
            subnetwork_uri=default_subnet,
            prior_tuning_uri=prior_tuning_uri,
        )

    final_train = prepare_training_panel(df, cfg, horizon=cfg.data.horizon)
    train_rows_total += len(final_train)
    final_infer = prepare_inference_panel(
        df,
        cfg,
        context_window=plan.context_window,
        horizon=cfg.data.horizon,
        future_df=future_df,
    )
    preds_by_s, exps_by_s, global_attrs_by_s, artifacts_info = dispatch(
        plan,
        final_train,
        final_infer,
        cfg,
        settings=settings,
        job_prefix=f"{base_prefix}-{short_model}-final",
        bq_client=bq_client,
    )
    n_fits += 1

    elapsed = max(0.0, time.perf_counter() - t0)
    per_series_fit_s = elapsed / max(1, len(uids))
    ended_at = datetime.now(UTC)

    best_params: dict[str, Any] = {
        "training_mode": "global",
        "automl_mode": plan.automl_mode,
        "context_window": plan.context_window,
        "optimization_objective": plan.optimization_objective,
        "train_budget_milli_node_hours": plan.train_budget_milli_node_hours,
    }
    if artifacts_info.get("stage_1_tuning_result_artifact_uri"):
        best_params["stage_1_tuning_result_artifact_uri"] = artifacts_info[
            "stage_1_tuning_result_artifact_uri"
        ]
    if artifacts_info.get("vertex_model_resource_name"):
        best_params["vertex_model_resource_name"] = artifacts_info["vertex_model_resource_name"]

    model_artifact_uri = (
        artifacts_info.get("stage_1_tuning_result_artifact_uri")
        or artifacts_info.get("vertex_model_resource_name")
        or plan.root_dir
    )

    metric = cfg.backtest.decision_metric
    corrected = corrected_arm_for(metric)

    results: list[CellResult] = []
    for idx, uid in enumerate(uids):
        s_df = series_groups[uid]
        raw_pred = preds_by_s.get(uid)
        if raw_pred is None or raw_pred.empty:
            fut_dates = _future_dates(
                pd.Timestamp(s_df[date_col].iloc[-1]), cfg.data.horizon, cfg.data.freq
            )
            raw_pred = pd.DataFrame(
                {
                    "ds": fut_dates,
                    "yhat": np.nan,
                    "yhat_raw": np.nan,
                    "yhat_lower": np.nan,
                    "yhat_upper": np.nan,
                    "quantiles": pd.array(["{}"] * len(fut_dates), dtype="string"),
                },
                columns=list(PREDICTION_COLUMNS),
            )

        oof = oof_by_uid[uid]
        arm, arm_decision = cfg.output.point_forecast or "median", "configured"
        if arm == "auto":
            arm, arm_decision = select_arm(oof, metric, corrected)
        cal = calibrate_from_oof(oof, DEFAULT_QUANTILES) if oof is not None else None
        calibrated, interval_calibration = apply_calibration(raw_pred, cal, arm)
        arm_comparison = compare_arms(oof, metric, corrected) if oof is not None else {}

        diag: dict[str, Any] = {
            "automl_mode": plan.automl_mode,
            "pipeline_root": plan.root_dir,
        }
        if uid in global_attrs_by_s:
            diag["feature_attributions"] = global_attrs_by_s[uid]
        if artifacts_info.get("evaluation_metrics_uri"):
            diag["evaluation_metrics_uri"] = artifacts_info["evaluation_metrics_uri"]

        results.append(
            CellResult(
                run_id=run_id,
                ts_id=uid,
                model_type=model_type,
                compute_engine="vertex_automl",
                model_hash=make_model_hash(run_id, uid, model_type, cfg),
                status="ok",
                error=None,
                predictions=calibrated,
                oof=oof,
                metrics=metrics_by_uid[uid],
                best_params=best_params,
                fit_seconds=per_series_fit_s,
                worker_id=worker_id,
                cell_started_at=cell_started_at,
                cell_ended_at=ended_at,
                model_artifact_uri=str(model_artifact_uri) if model_artifact_uri else None,
                backtest_status=bt_status_by_uid[uid],
                n_folds_achieved=n_folds_by_uid[uid],
                backtest_note=bt_note_by_uid[uid],
                backtest_refit=bt_refit_by_uid[uid],
                staleness_gap=None,
                achieved_step=ach_step_by_uid[uid],
                achieved_min_train=ach_min_by_uid[uid],
                first_val_date=first_val_by_uid[uid],
                last_val_date=last_val_by_uid[uid],
                interval_source="native",
                point_forecast_source=arm,
                point_forecast_decision=arm_decision,
                interval_calibration=interval_calibration,
                point_forecast_margin=arm_comparison.get("margin"),
                diagnostics=diag,
                explanations=exps_by_s.get(uid),
                n_fits=n_fits if idx == 0 else 0,
                train_rows_total=train_rows_total if idx == 0 else 0,
                n_hpo_fits=0,
            )
        )

    telemetry = {
        "model_type": model_type,
        "automl_mode": plan.automl_mode,
        "root_dir": plan.root_dir,
        "context_window": plan.context_window,
        "horizon": plan.horizon,
        "train_budget_milli_node_hours": plan.train_budget_milli_node_hours,
        **artifacts_info,
    }
    return results, telemetry


def run(
    cfg: RunConfig,
    *,
    models: list[str] | None = None,
    manage_header: bool = True,
    settings: Settings | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run all selected ``automl`` models on Vertex AI and write results to BigQuery."""
    from ..registry.cells import write_cells
    from ..registry.lifecycle import run_header
    from ..settings import Settings as _Settings
    from .ray_engine import _assert_source_supports_folds, _read_source_series

    settings = settings or _Settings.resolve()
    run_id = make_run_id(cfg)
    executed_models = [
        m for m in (models if models is not None else cfg.models) if get_model(m).family == "automl"
    ]
    if not executed_models:
        return {"run_id": run_id, "n_succeeded": 0, "n_failed": 0, "models": {}}

    with run_header(cfg, run_id, settings=settings, manage=manage_header) as hdr:
        panel_df = _read_source_series(cfg, settings)
        _assert_source_supports_folds(panel_df, cfg)

        future_df: pd.DataFrame | None = None
        fut_table = getattr(cfg.data, "future_covariates_table", None)
        if fut_table:
            from google.cloud import bigquery

            bq_client = bigquery.Client(project=settings.project_id)
            fut_tbl = fut_table if "." in fut_table else settings.table_ref(fut_table)
            future_df = bq_client.query(f"SELECT * FROM `{fut_tbl}`").to_dataframe()

        all_cells: list[CellResult] = []
        model_telemetry: dict[str, Any] = {}
        for model_type in executed_models:
            cells, telem = execute_automl_model_cells(
                panel_df,
                model_type,
                cfg,
                run_id=run_id,
                settings=settings,
                future_df=future_df,
                job_id=job_id,
            )
            write_cells(cells, settings=settings)
            all_cells.extend(cells)
            model_telemetry[model_type] = telem

        n_ok = sum(1 for c in all_cells if c.status == "ok")
        n_fail = len(all_cells) - n_ok
        status = "COMPLETED" if n_fail == 0 else ("PARTIAL" if n_ok > 0 else "FAILED")
        n_series = int(panel_df[cfg.data.ts_id_col].nunique()) if not panel_df.empty else 0
        hdr.finalize(status=status, n_series=n_series)

    return {
        "run_id": run_id,
        "n_succeeded": n_ok,
        "n_failed": n_fail,
        "models": model_telemetry,
    }
