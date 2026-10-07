"""Submit a run (or ``automl`` family slice) to Vertex AI AutoML / Tabular Workflows.

Dispatches the four managed Vertex AI AutoML Forecasting model plugins (``vertex_l2l``,
``vertex_tide``, ``vertex_tft``, ``vertex_seq2seq``) through `engines.automl_engine`. Supports both:

1. **Tabular Workflow for Forecasting** (``automl_mode = "tabular_workflow"``, default):
   Compiles and submits a Vertex AI ``PipelineJob`` via ``google_cloud_pipeline_components``,
   exposing per-stage hardware overrides (``feature_transform_engine_machine_type``,
   ``trainer_machine_type``, ``trainer_replica_count``, ``trainer_accelerator_type``,
   ``trainer_accelerator_count``, ``max_num_trials``, ``max_parallel_trial_count``) and Stage-1
   tuning artifact warm-start (``stage_1_tuning_result_artifact_uri`` or
   ``reuse_tuning_from_run_id``).
2. **Managed AutoML TrainingJob** (``automl_mode = "training_job"``):
   Submits a fully managed ``AutoMLForecastingTrainingJob`` /
   ``TimeSeriesDenseEncoderForecastingTrainingJob`` /
   ``TemporalFusionTransformerForecastingTrainingJob`` /
   ``SequenceToSequencePlusForecastingTrainingJob`` via the ``google-cloud-aiplatform`` SDK.

After training completes, executes a ``BatchPredictionJob`` with ``generate_explanation=True`` and
writes calibrated forecasts, backtest OOF evaluations, Tier 1 series-level feature attributions
(``forecast_metadata.fit_diagnostics["feature_attributions"]``), Tier 2 per-horizon-step local
attributions (``forecast_predictions.explanations``), and GCS tuning/model artifact URIs
(``forecast_metadata.model_artifact`` and ``best_params``) back to the BigQuery registry.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import staging
from .batch_infra import BatchInfra
from .config import RunConfig
from .engines import automl_engine
from .errors import ConfigError, configure_cli_logging, get_logger
from .models import get_model
from .probes.vocabulary import ProbeHandle
from .profiling.source import profile_for_run
from .registry.header import merge_header_telemetry
from .registry.ids import make_run_id, vertex_automl_job_id
from .registry.jobs import update_job
from .settings import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = get_logger(__name__)


def _pipeline_client(region: str) -> Any:
    """Create a regional Vertex AI ``PipelineServiceClient``."""
    from google.cloud import aiplatform_v1

    return aiplatform_v1.PipelineServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )


@dataclass(frozen=True)
class VertexAutoMLJobPlan:
    """Resolved execution summary for a ``vertex_automl`` family job (pure)."""

    display_name: str
    automl_mode: str
    models: tuple[str, ...]
    hardware: str
    gpu_type: str | None
    machine_type: str
    max_workers: int
    accelerator_count: int
    config_uri: str = ""
    service_account: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary for ``run_jobs.job_telemetry.$.vertex_automl``."""
        return {
            "display_name": self.display_name,
            "automl_mode": self.automl_mode,
            "models": list(self.models),
            "hardware": self.hardware,
            "gpu_type": self.gpu_type,
            "machine_type": self.machine_type,
            "max_workers": self.max_workers,
            "accelerator_count": self.accelerator_count,
            "config_uri": self.config_uri,
            "service_account": self.service_account,
        }


def plan_automl_job(
    cfg: RunConfig,
    models: Sequence[str] | None = None,
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    automl_mode: str | None = None,
    hardware: str | None = None,
    gpu_type: str | None = None,
    machine_type: str | None = None,
    max_workers: int | None = None,
    accelerator_count: int | None = None,
    config_uri: str = "",
    service_account: str = "",
) -> VertexAutoMLJobPlan:
    """Resolve a `VertexAutoMLJobPlan` from `cfg` and per-family overrides (pure)."""
    rid = run_id or make_run_id(cfg)
    selected = [
        m for m in (models if models is not None else cfg.models) if get_model(m).family == "automl"
    ]
    fc = cfg.resolve_family_compute("automl")
    eff_mode = automl_mode or fc.automl_mode or cfg.compute.automl_mode
    eff_hw = hardware or fc.hardware or "cpu"
    eff_gpu = (gpu_type or fc.gpu_type) if eff_hw == "gpu" else None
    eff_machine = (
        machine_type or fc.machine_type or ("g2-standard-8" if eff_hw == "gpu" else "n1-standard-8")
    )
    eff_workers = max_workers or fc.max_workers or fc.workers or 10
    eff_accel = (
        (accelerator_count if accelerator_count is not None else fc.accelerator_count)
        if eff_hw == "gpu"
        else 0
    )
    display_name = (
        vertex_automl_job_id(job_id) if job_id else vertex_automl_job_id(f"sf-{rid}-automl")
    )
    return VertexAutoMLJobPlan(
        display_name=display_name,
        automl_mode=eff_mode,
        models=tuple(selected),
        hardware=eff_hw,
        gpu_type=eff_gpu,
        machine_type=eff_machine,
        max_workers=int(eff_workers),
        accelerator_count=int(eff_accel),
        config_uri=config_uri,
        service_account=service_account,
    )


def submit_automl(
    cfg: RunConfig,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    *,
    wait: bool = True,
    models: list[str] | None = None,
    job_id: str | None = None,
    manage_header: bool = True,
    automl_mode: str | None = None,
    hardware: str | None = None,
    gpu_type: str | None = None,
    machine_type: str | None = None,
    max_workers: int | None = None,
    accelerator_count: int | None = None,
) -> tuple[str, str, ProbeHandle]:
    """Execute ``automl`` models on Vertex AI and return ``(run_id, native_id, probe_handle)``."""
    settings = settings or Settings.resolve()
    if infra is None:
        with contextlib.suppress(Exception):
            infra = BatchInfra.resolve()

    run_id = make_run_id(cfg)
    automl_models = [
        m for m in (models if models is not None else cfg.models) if get_model(m).family == "automl"
    ]
    if not automl_models:
        raise ConfigError(
            "submit_automl called with a config that has no 'automl' family models to execute."
        )

    code_bucket = ""
    service_account = ""
    if infra is not None:
        code_bucket = infra.code_bucket
        service_account = infra.compute_sa
    elif settings.warehouse_uri.startswith("gs://"):
        code_bucket = settings.warehouse_uri.removeprefix("gs://").split("/", 1)[0]

    config_uri = ""
    if code_bucket:
        with contextlib.suppress(Exception):
            config_uri = staging.stage_config(cfg, run_id, code_bucket)

    plan = plan_automl_job(
        cfg,
        automl_models,
        run_id=run_id,
        job_id=job_id,
        automl_mode=automl_mode,
        hardware=hardware,
        gpu_type=gpu_type,
        machine_type=machine_type,
        max_workers=max_workers,
        accelerator_count=accelerator_count,
        config_uri=config_uri,
        service_account=service_account,
    )

    profile = None
    with contextlib.suppress(Exception):
        profile = profile_for_run(cfg, settings=settings)

    sizing_entry = {
        "family": "automl",
        "runtime": "vertex_automl",
        "automl_mode": plan.automl_mode,
        "hardware": plan.hardware,
        "gpu_type": plan.gpu_type,
        "machine_type": plan.machine_type,
        "max_workers": plan.max_workers,
        "models": list(plan.models),
        "profile_source": getattr(profile, "source", "static") if profile is not None else "static",
    }
    with contextlib.suppress(Exception):
        merge_header_telemetry(
            run_id,
            {
                "sizing": {"automl": sizing_entry},
                "sizing_executed": {"automl": sizing_entry},
            },
            settings=settings,
        )

    handle = ProbeHandle(
        runtime="vertex_automl",
        native_id=plan.display_name,
        region=settings.region,
    )
    if job_id is not None:
        with contextlib.suppress(Exception):
            update_job(
                job_id,
                settings=settings,
                merge_telemetry={
                    "probe_handle": handle.to_blob(),
                    "vertex_automl": plan.to_dict(),
                },
            )

    _ = wait
    outcome = automl_engine.run(
        cfg,
        models=automl_models,
        manage_header=manage_header,
        settings=settings,
        job_id=plan.display_name,
    )

    # If the pipeline produced a concrete resource_name or pipeline_job_id, update the handle.
    models_telem = outcome.get("models") or {}
    first_telem: dict[str, Any] = (
        next(iter(models_telem.values()), {}) if isinstance(models_telem, dict) else {}
    )
    resource_name = first_telem.get("pipeline_resource_name") or first_telem.get(
        "vertex_model_resource_name"
    )
    native_id = first_telem.get("pipeline_job_id") or plan.display_name
    final_handle = ProbeHandle(
        runtime="vertex_automl",
        native_id=str(native_id),
        region=settings.region,
        resource_name=str(resource_name) if resource_name else None,
    )
    if job_id is not None:
        with contextlib.suppress(Exception):
            update_job(
                job_id,
                settings=settings,
                merge_telemetry={
                    "probe_handle": final_handle.to_blob(),
                    "vertex_automl": {
                        **plan.to_dict(),
                        "models_telemetry": models_telem,
                    },
                },
            )

    return run_id, final_handle.native_id, final_handle


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scale_forecasting.automl_submit",
        description=(
            "Stage config and submit a Vertex AI AutoML / Tabular Workflow forecasting job."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="Path to the run config JSON.")
    src.add_argument("--config-uri", help="gs:// URI of a staged config (portable source).")
    p.add_argument(
        "--n-series",
        type=int,
        default=None,
        help="Override data.series_limit.",
    )
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated model subset for per-family DAG jobs.",
    )
    p.add_argument(
        "--job-id",
        default=None,
        help="Deterministic job_key (sf-<run_id>-automl-a<n>).",
    )
    p.add_argument(
        "--automl-mode",
        choices=("tabular_workflow", "training_job"),
        default=None,
        help="Execution mode override ('tabular_workflow' or 'training_job').",
    )
    p.add_argument(
        "--no-manage-header",
        action="store_true",
        help="Do not start/complete the run_registry header (caller owns it).",
    )
    p.add_argument(
        "--hardware",
        choices=("cpu", "gpu"),
        default=None,
        help="Per-family hardware override ('cpu' or 'gpu').",
    )
    p.add_argument(
        "--gpu-type",
        choices=("T4", "L4", "A100", "A100_80GB"),
        default=None,
        help="Per-family GPU type override ('T4', 'L4', 'A100', or 'A100_80GB').",
    )
    p.add_argument(
        "--machine-type",
        default=None,
        help="Trainer machine type override (e.g. 'n1-standard-8', 'g2-standard-8').",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel trial workers override.",
    )
    p.add_argument(
        "--accelerator-count",
        type=int,
        default=None,
        help="Trainer GPU count per replica override.",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Return as soon as the job is launched.",
    )
    return p


def main(argv: list[str] | None = None) -> str:
    from .config import load_config_uri

    configure_cli_logging()
    args = _build_parser().parse_args(argv)
    cfg = load_config_uri(args.config or args.config_uri).with_series_limit(args.n_series)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    run_id, native_id, _ = submit_automl(
        cfg,
        wait=not args.no_wait,
        models=models,
        job_id=args.job_id,
        manage_header=not args.no_manage_header,
        automl_mode=args.automl_mode,
        hardware=args.hardware,
        gpu_type=args.gpu_type,
        machine_type=args.machine_type,
        max_workers=args.max_workers,
        accelerator_count=args.accelerator_count,
    )
    _log.info("Vertex AutoML job complete: run_id=%s native_id=%s", run_id, native_id)
    return run_id


if __name__ == "__main__":  # pragma: no cover
    main()
