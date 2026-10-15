"""Parametric 7-Runtime Wall-Clock & Cost Predictor — pure offline math + registry calibration.

Predicts end-to-end wall-clock duration (`p50` and `p90` seconds/minutes) and approximate cloud
compute cost (`USD`) for any :class:`~scale_forecasting.config.RunConfig` before a single cloud
resource is provisioned.

Why a parametric model instead of `N *models * cell_time`?
-----------------------------------------------------------
In distributed cloud forecasting, total wall-clock time is governed by **three distinct regimes**:

1. **Cold-Start & Cluster Provisioning Overhead ($T_{\\text{provision}}$):**
   Serverless Spark batches (~65s), Single-VM GCE instances (~215s), Vertex CustomJobs (~240s),
   and ephemeral Vertex AI Ray or GKE clusters (~400s) pay a fixed startup tax before the first
   series is read. Conversely, `local` execution (0s), BigQuery SQL (~3s), Spark Connect (~2s),
   and warm attached Ray/GKE clusters (~15s) start almost immediately.
2. **Storage Read / Write I/O ($T_{\\text{io}}$):**
   Reading the panel via the BigQuery Storage Read API and streaming predictions/OOF/diagnostics
   back via the Storage Write API scales with $N \\times (T + H \\cdot F)$.
3. **Parallel Per-Family Model Execution ($T_{\\text{exec}}$):**
   `main.run` dispatches one independent job per active model family (`statistical`, `ml`,
   `deep_learning`, `automl`, `native`) **in parallel**, and runs the downstream `ensemble` node
   once base families complete:
   `T_wall(cfg) = max_{f in F_active} T_f(N, T, H) + T_ensemble(N, |M|)`.

When ``reuse_prior_runs=True`` (as in Pass $k \\ge 2$ of ``autofocus=True``), any model variant in
``borrowed_models`` is copied via BigQuery metadata in ~1.5s rather than re-fitted, and any family
whose models are 100% borrowed skips cluster provisioning entirely ($T_f \\approx 1.5\\text{s}$).

Empirical Provenance
--------------------
The default coefficients in :data:`EMPIRICAL_PROVISIONING_PRIORS` and
:data:`EMPIRICAL_MODEL_FIT_PRIORS` were calibrated from **368 completed multi-runtime runs** and
**5,495,569 fitted `(ts_id, model_type)` cells** in the reference dataset
(`statmike-scale-forecasting.scale_forecasting`). Enterprise operators can re-calibrate these
coefficients against their own BigQuery registry via :func:`calibrate_duration_from_registry` or
``Registry.calibrate_duration_model()``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..models import base_model_name, get_model

if TYPE_CHECKING:
    from ..config import RunConfig
    from ..settings import Settings

__all__ = [
    "EMPIRICAL_MODEL_FIT_PRIORS",
    "EMPIRICAL_PROVISIONING_PRIORS",
    "DurationCalibration",
    "DurationEstimate",
    "FamilyDurationBreakdown",
    "calibrate_duration_from_frames",
    "calibrate_duration_from_registry",
    "predict_run_duration",
]

# Reference series history and horizon at which EMPIRICAL_MODEL_FIT_PRIORS are normalized
_REF_N_OBS = 365.0
_REF_HORIZON = 28.0

# Empirical provisioning priors (p50_seconds, p90_seconds) keyed by (runtime, mode, hardware).
# Calibrated from 423 completed jobs across all 7 runtimes in the reference BigQuery registry.
EMPIRICAL_PROVISIONING_PRIORS: dict[tuple[str, str, str], tuple[float, float]] = {
    ("local", "in_process", "cpu"): (0.0, 0.0),
    ("local", "in_process", "gpu"): (0.0, 0.0),
    ("bigquery", "sql", "cpu"): (14.0, 67.3),
    ("spark", "serverless", "cpu"): (64.7, 72.0),
    ("spark", "serverless", "gpu"): (78.0, 96.0),
    ("spark", "cluster", "cpu"): (145.0, 185.0),
    ("spark", "cluster", "gpu"): (175.0, 220.0),
    ("spark", "connect", "cpu"): (2.0, 5.0),
    ("gce", "single_vm", "cpu"): (238.0, 280.0),
    ("gce", "single_vm", "gpu"): (238.0, 295.0),
    ("vertex", "custom_job", "cpu"): (240.0, 409.6),
    ("vertex", "custom_job", "gpu"): (249.1, 308.7),
    ("gke", "job", "cpu"): (79.8, 212.6),
    ("gke", "job", "gpu"): (227.3, 418.0),
    ("gke", "ephemeral_cluster", "cpu"): (380.0, 440.0),
    ("gke", "ephemeral_cluster", "gpu"): (418.0, 480.0),
    ("ray", "vertex", "cpu"): (245.2, 832.0),
    ("ray", "vertex", "gpu"): (404.1, 880.0),
    ("ray", "warm_cluster", "cpu"): (12.0, 25.0),
    ("ray", "warm_cluster", "gpu"): (15.0, 30.0),
    ("ray", "gke", "cpu"): (22.0, 48.0),
    ("ray", "gke", "gpu"): (35.0, 75.0),
    ("vertex_automl", "tabular_workflow", "cpu"): (1930.0, 3600.0),
    ("vertex_automl", "tabular_workflow", "gpu"): (1930.0, 3600.0),
    ("vertex_automl", "training_job", "cpu"): (1930.0, 3600.0),
    ("vertex_automl", "training_job", "gpu"): (1930.0, 3600.0),
}

# Fixed distributed worker bootstrap, connector initialization & Write-API commit overhead
# (p50_s, p90_s) observed inside execute_seconds on cloud jobs (even for 10-series smoke runs).
_RUNTIME_EXEC_OVERHEAD_S: dict[tuple[str, str], tuple[float, float]] = {
    ("local", "in_process"): (0.0, 0.0),
    ("bigquery", "sql"): (28.0, 45.0),
    ("spark", "serverless"): (165.0, 225.0),
    ("spark", "cluster"): (35.0, 60.0),
    ("spark", "connect"): (8.0, 15.0),
    ("gce", "single_vm"): (45.0, 75.0),
    ("vertex", "custom_job"): (45.0, 75.0),
    ("gke", "job"): (38.0, 65.0),
    ("gke", "ephemeral_cluster"): (45.0, 75.0),
    ("ray", "vertex"): (85.0, 140.0),
    ("ray", "warm_cluster"): (18.0, 32.0),
    ("ray", "gke"): (25.0, 45.0),
    ("vertex_automl", "tabular_workflow"): (1800.0, 3200.0),
    ("vertex_automl", "training_job"): (1800.0, 3200.0),
}

# Per-model single-fit wall seconds (p50_s, p90_s) normalized to a single fold at T=365, H=28.
# Global/panel models (scope="global") record per-series amortized fit seconds across a panel.
EMPIRICAL_MODEL_FIT_PRIORS: dict[str, tuple[float, float]] = {
    # Statistical family (18 models)
    "naive_mean": (0.018, 0.035),
    "naive_last": (0.018, 0.035),
    "naive_drift": (0.020, 0.038),
    "naive_seasonal": (0.028, 0.052),
    "naive_moving_average": (0.022, 0.042),
    "snaive": (0.028, 0.052),
    "croston": (0.065, 0.120),
    "fft": (0.038, 0.075),
    "regression_lags": (0.042, 0.085),
    "holtwinters": (0.120, 0.240),
    "theta": (0.140, 0.260),
    "stl_bagging": (0.330, 0.580),
    "sarimax": (0.620, 1.150),
    "prophet": (0.420, 0.780),
    "autoets": (1.050, 1.950),
    "auto_ces": (1.120, 2.100),
    "ucm": (1.650, 3.100),
    "kalman": (4.200, 7.800),
    "auto_arima": (7.500, 14.500),
    "tbats": (8.100, 15.800),
    "auto_theta": (9.800, 18.500),
    # ML family (5 models)
    "lightgbm": (0.380, 0.680),
    "xgboost": (0.280, 0.520),
    "catboost": (1.250, 2.200),
    "random_forest": (2.600, 4.500),
    "linear_lags": (0.038, 0.072),
    # Deep Learning & Foundation family (8 models)
    "chronos_bolt": (0.240, 0.450),
    "chronos2": (0.950, 1.800),
    "timesfm_torch": (1.850, 3.200),
    "tide": (0.220, 0.420),
    "tsmixer": (0.240, 0.460),
    "patchtst": (0.350, 0.680),
    "tft": (0.920, 1.750),
    "neuralprophet": (18.500, 32.000),
    # Vertex AI AutoML family (4 models — amortized per series in cloud workflow)
    "vertex_AutoML": (12.000, 22.000),
    "vertex_seq2seq": (8.500, 15.000),
    "vertex_tft": (10.500, 18.000),
    "vertex_tide": (9.500, 16.500),
    # BigQuery ML Native family (2 models — SQL query execution amortized per series)
    "bqml_arima_plus": (0.085, 0.160),
    "bqml_timesfm": (0.065, 0.120),
}

# Approximate hourly USD rate per worker slot / node for transparent cost estimation
_HOURLY_USD_BY_RUNTIME: dict[tuple[str, str], float] = {
    ("local", "cpu"): 0.0,
    ("local", "gpu"): 0.0,
    ("bigquery", "cpu"): 0.60,
    ("spark", "cpu"): 0.42,
    ("spark", "gpu"): 1.35,
    ("ray", "cpu"): 0.45,
    ("ray", "gpu"): 1.40,
    ("vertex", "cpu"): 0.38,
    ("vertex", "gpu"): 1.25,
    ("gce", "cpu"): 0.34,
    ("gce", "gpu"): 1.10,
    ("gke", "cpu"): 0.36,
    ("gke", "gpu"): 1.18,
    ("vertex_automl", "cpu"): 3.50,
    ("vertex_automl", "gpu"): 4.80,
}


@dataclass(frozen=True)
class DurationCalibration:
    """Calibrated provisioning and per-model execution priors from a BigQuery registry."""

    provisioning_priors: dict[tuple[str, str, str], tuple[float, float]]
    model_fit_priors: dict[str, tuple[float, float]]
    n_runs_analyzed: int
    n_cells_analyzed: int
    source_dataset: str = "reference"

    @property
    def n_jobs_analyzed(self) -> int:
        """Alias for ``n_runs_analyzed``."""
        return self.n_runs_analyzed

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration priors to a JSON-compatible dictionary."""
        return {
            "source_dataset": self.source_dataset,
            "n_runs_analyzed": self.n_runs_analyzed,
            "n_jobs_analyzed": self.n_runs_analyzed,
            "n_cells_analyzed": self.n_cells_analyzed,
            "provisioning_priors": {
                (f"{k[0]}:{k[1]}:{k[2]}" if isinstance(k, tuple) else str(k)): [
                    round(p50, 3),
                    round(p90, 3),
                ]
                for k, (p50, p90) in sorted(
                    self.provisioning_priors.items(), key=lambda kv: str(kv[0])
                )
            },
            "model_fit_priors": {
                k: [round(p50, 4), round(p90, 4)]
                for k, (p50, p90) in sorted(self.model_fit_priors.items())
            },
        }


@dataclass(frozen=True)
class FamilyDurationBreakdown:
    """Estimated wall-clock and cost breakdown for a single model family job."""

    family: str
    runtime: str
    mode: str
    hardware: str
    models: tuple[str, ...]
    borrowed_models: tuple[str, ...]
    executed_models: tuple[str, ...]
    n_series: int
    parallel_slots: int
    provision_p50_s: float
    provision_p90_s: float
    io_s: float
    exec_p50_s: float
    exec_p90_s: float
    total_p50_s: float
    total_p90_s: float
    estimated_usd: float


@dataclass(frozen=True)
class DurationEstimate:
    """Complete parametric wall-clock and cost prediction for a :class:`RunConfig`."""

    total_p50_s: float
    total_p90_s: float
    total_p50_minutes: float
    total_p90_minutes: float
    bottleneck_family: str
    ensemble_s: float
    estimated_usd: float
    families: tuple[FamilyDurationBreakdown, ...]
    basis: str = "reference_5.49M_cells"

    @property
    def total_p50_seconds(self) -> float:
        """Alias for ``total_p50_s``."""
        return self.total_p50_s

    @property
    def total_p90_seconds(self) -> float:
        """Alias for ``total_p90_s``."""
        return self.total_p90_s

    @property
    def provision_p50_seconds(self) -> float:
        """Critical-path provisioning P50 duration in seconds."""
        return max((f.provision_p50_s for f in self.families), default=0.0)

    @property
    def execute_p50_seconds(self) -> float:
        """Total model execution P50 duration in seconds across families."""
        return sum((f.exec_p50_s for f in self.families), 0.0)

    @property
    def critical_path_family(self) -> str:
        """Alias for ``bottleneck_family``."""
        return self.bottleneck_family

    @property
    def family_breakdowns(self) -> tuple[FamilyDurationBreakdown, ...]:
        """Alias for ``families``."""
        return self.families

    @property
    def n_series(self) -> int:
        """Series count used for the duration estimate."""
        return self.families[0].n_series if self.families else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert estimate to a JSON-friendly dictionary."""
        fams = [
            {
                **asdict(f),
                "models": list(f.models),
                "borrowed_models": list(f.borrowed_models),
                "executed_models": list(f.executed_models),
            }
            for f in self.families
        ]
        return {
            "n_series": self.n_series,
            "total_p50_s": round(self.total_p50_s, 2),
            "total_p90_s": round(self.total_p90_s, 2),
            "total_p50_minutes": round(self.total_p50_minutes, 2),
            "total_p90_minutes": round(self.total_p90_minutes, 2),
            "bottleneck_family": self.bottleneck_family,
            "critical_path_family": self.bottleneck_family,
            "ensemble_s": round(self.ensemble_s, 2),
            "estimated_usd": round(self.estimated_usd, 4),
            "basis": self.basis,
            "families": fams,
            "family_breakdowns": fams,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a per-family breakdown table suitable for notebook display."""
        rows = [
            {
                "family": f.family,
                "runtime": f.runtime,
                "mode": f.mode,
                "hardware": f.hardware,
                "n_models": len(f.models),
                "n_borrowed": len(f.borrowed_models),
                "n_executed": len(f.executed_models),
                "parallel_slots": f.parallel_slots,
                "provision_p50_s": round(f.provision_p50_s, 1),
                "io_s": round(f.io_s, 1),
                "exec_p50_s": round(f.exec_p50_s, 1),
                "total_p50_s": round(f.total_p50_s, 1),
                "total_p90_s": round(f.total_p90_s, 1),
                "estimated_usd": round(f.estimated_usd, 3),
            }
            for f in self.families
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Compact one-line human summary of the predicted wall-clock and cost."""
        return (
            f"p50={self.total_p50_s:.1f}s ({self.total_p50_minutes:.2f}m) | "
            f"p90={self.total_p90_s:.1f}s ({self.total_p90_minutes:.2f}m) | "
            f"bottleneck={self.bottleneck_family} | est_cost=${self.estimated_usd:.2f}"
        )


def _resolve_runtime_mode(
    cfg: RunConfig,
    runtime: str,
    spark_mode: str | None,
    ray_mode: str | None,
    gke_mode: str | None,
    automl_mode: str | None,
    *,
    where_override: str | None = None,
) -> tuple[str, str]:
    """Map resolved family compute fields to ``(effective_runtime, effective_mode)``."""
    if where_override == "local":
        return "local", "in_process"
    if runtime == "bigquery":
        return "bigquery", "sql"
    if runtime == "spark":
        sm = spark_mode or "serverless"
        return "spark", sm
    if runtime == "gce":
        return "gce", "single_vm"
    if runtime == "vertex":
        return "vertex", "custom_job"
    if runtime == "gke":
        if gke_mode == "ray":
            return "ray", "gke"
        if cfg.compute.gke_cluster_name:
            return "gke", "job"
        return "gke", "job"
    if runtime == "ray":
        if ray_mode == "gke":
            return "ray", "gke"
        if cfg.compute.ray_cluster_name:
            return "ray", "warm_cluster"
        return "ray", "vertex"
    if runtime == "vertex_automl":
        return "vertex_automl", automl_mode or "tabular_workflow"
    return runtime, "in_process"


def _estimate_parallel_slots(
    cfg: RunConfig,
    runtime: str,
    hardware: str,
    workers: int | None,
    n_series: int,
    n_exec_models: int,
) -> int:
    """Estimate effective concurrent series-cell slots for a family job."""
    total_cells = max(1, n_series * max(1, n_exec_models))
    if runtime == "local":
        return min(total_cells, 4)
    if runtime == "bigquery":
        return max(1, min(total_cells, 32))
    if runtime == "gce":
        return min(total_cells, 8 if hardware == "cpu" else 4)
    if runtime in ("vertex", "gke"):
        w = max(1, workers or cfg.compute.workers or 1)
        slots_per_worker = 8 if hardware == "cpu" else 2
        return min(total_cells, w * slots_per_worker)
    if runtime == "spark":
        max_exec = cfg.compute.max_executors or cfg.compute.max_workers or 16
        slots_per_exec = 4 if hardware == "cpu" else 2
        return min(total_cells, max(4, min(max_exec * slots_per_exec, total_cells // 2 + 4)))
    if runtime == "ray":
        max_nodes = cfg.compute.ray_max_nodes or 8
        slots_per_node = 7 if hardware == "cpu" else 2
        return min(total_cells, max(4, min(max_nodes * slots_per_node, total_cells // 2 + 4)))
    if runtime == "vertex_automl":
        return max(1, n_series)
    return 4


def _model_single_cell_seconds(
    model_spec: str,
    cfg: RunConfig,
    *,
    n_series: int,
    median_n_obs: float,
    hardware: str,
    model_priors: Mapping[str, tuple[float, float]],
) -> tuple[float, float]:
    """Compute `(p50_s, p90_s)` for one `(ts_id, model_spec)` cell including folds and HPO."""
    bname = base_model_name(model_spec)
    base_p50, base_p90 = model_priors.get(bname, (0.50, 0.95))

    # Scale with observation length T and horizon H (sub-linear power law)
    length_factor = max(0.35, min(4.0, (median_n_obs / _REF_N_OBS) ** 0.65))
    horizon_factor = max(0.50, min(3.0, (float(cfg.data.horizon) / _REF_HORIZON) ** 0.40))

    # Covariate width multiplier
    n_covs = len(cfg.features.all_covariates)
    cov_factor = 1.0 + 0.06 * min(10, n_covs)
    if "@univariate" in model_spec or "@cov_none" in model_spec:
        cov_factor = 1.0

    # Direct multi-horizon strategy fits H separate estimators for tree/lag models
    direct_factor = 1.0
    is_direct = (
        "@direct" in model_spec
        or cfg.model_params.get(model_spec, {}).get("strategy") == "direct"
        or cfg.model_params.get(bname, {}).get("strategy") == "direct"
    )
    if is_direct:
        try:
            fam = get_model(bname).family
        except Exception:  # noqa: BLE001 - fallback when custom model is not registered
            fam = "statistical"
        if fam == "ml" or bname in ("regression_lags", "linear_lags", "ridge_lags"):
            direct_factor = min(6.0, 1.0 + 0.22 * cfg.data.horizon)

    # Fold count and HPO trials
    n_folds = (1 + cfg.backtest.n_folds) if cfg.backtest.enabled else 1
    hpo_mult = 1.0
    if cfg.hpo.enabled and cfg.hpo.n_trials > 0:
        if cfg.hpo.granularity == "per_series":
            hpo_mult = 1.0 + 0.55 * cfg.hpo.n_trials
        else:
            # Fleetwide HPO tunes on a small sample of series before the main fan-out
            sample_frac = min(1.0, float(cfg.hpo.sample_size) / max(1.0, float(n_series)))
            hpo_mult = 1.0 + sample_frac * 0.45 * cfg.hpo.n_trials

    # GPU acceleration speedup for GPU-capable deep learning / foundation models
    hw_factor = 1.0
    if hardware == "gpu":
        try:
            if get_model(bname).gpu_capable:
                hw_factor = 0.38
        except Exception:  # noqa: BLE001 - ignore unknown model
            pass

    mult = (
        length_factor * horizon_factor * cov_factor * direct_factor * n_folds * hpo_mult * hw_factor
    )
    return base_p50 * mult, base_p90 * mult


def predict_run_duration(
    cfg: RunConfig,
    *,
    n_series: int | None = None,
    median_n_obs: float | None = None,
    where: str | None = None,
    borrowed_models: Iterable[str] | None = None,
    calibration: DurationCalibration | None = None,
) -> DurationEstimate:
    """Predict wall-clock duration (`p50` / `p90`) and cloud cost (`USD`) for ``cfg`` (pure).

    Args:
        cfg: Validated :class:`~scale_forecasting.config.RunConfig`.
        n_series: Number of time series in the panel (falls back to ``cfg.data.series_limit``
            or ``100`` when unspecified).
        median_n_obs: Median historical observations per series (defaults to ``365.0``).
        where: Optional execution surface override (e.g. ``"local"`` for in-memory DataFrame
            execution where cloud provisioning is ``0.0s``).
        borrowed_models: Optional iterable of model variant names that will be borrowed via
            ``reuse_prior_runs`` instead of re-fitted.
        calibration: Optional custom :class:`DurationCalibration` from
            :func:`calibrate_duration_from_registry`.

    Returns:
        :class:`DurationEstimate` with per-family breakdowns, bottleneck family, and cost.
    """
    from ..dag import plan_dag

    eff_n_series = int(n_series or cfg.data.series_limit or 100)
    eff_n_obs = float(median_n_obs if median_n_obs is not None else _REF_N_OBS)
    borrowed_set = frozenset(borrowed_models or ())

    prov_priors = (
        calibration.provisioning_priors
        if calibration is not None
        else EMPIRICAL_PROVISIONING_PRIORS
    )
    model_priors = (
        calibration.model_fit_priors if calibration is not None else EMPIRICAL_MODEL_FIT_PRIORS
    )
    basis = calibration.source_dataset if calibration is not None else "reference_5.49M_cells"

    run_dag = plan_dag(cfg)
    family_breakdowns: list[FamilyDurationBreakdown] = []

    for job in run_dag.jobs:
        fam_models = tuple(job.models)
        fam_borrowed = tuple(m for m in fam_models if m in borrowed_set)
        fam_exec = tuple(m for m in fam_models if m not in borrowed_set)

        comp = job.compute
        raw_runtime = job.runtime
        hardware = "cpu" if comp is None else comp.hardware
        eff_runtime, eff_mode = _resolve_runtime_mode(
            cfg,
            raw_runtime,
            None if comp is None else comp.spark_mode,
            None if comp is None else comp.ray_mode,
            None if comp is None else comp.gke_mode,
            None if comp is None else comp.automl_mode,
            where_override=where,
        )

        # If every model in this family is borrowed from a prior run, no cluster is launched!
        if not fam_exec:
            family_breakdowns.append(
                FamilyDurationBreakdown(
                    family=job.family,
                    runtime=eff_runtime,
                    mode="borrowed_reuse",
                    hardware=hardware,
                    models=fam_models,
                    borrowed_models=fam_borrowed,
                    executed_models=(),
                    n_series=eff_n_series,
                    parallel_slots=1,
                    provision_p50_s=0.0,
                    provision_p90_s=0.0,
                    io_s=1.5 if eff_runtime != "local" else 0.05,
                    exec_p50_s=0.0,
                    exec_p90_s=0.0,
                    total_p50_s=1.5 if eff_runtime != "local" else 0.05,
                    total_p90_s=2.5 if eff_runtime != "local" else 0.10,
                    estimated_usd=0.0,
                )
            )
            continue

        prov_p50, prov_p90 = prov_priors.get(
            (eff_runtime, eff_mode, hardware),
            prov_priors.get((eff_runtime, eff_mode, "cpu"), (60.0, 90.0)),
        )
        slots = _estimate_parallel_slots(
            cfg,
            eff_runtime,
            hardware,
            None if comp is None else comp.workers,
            eff_n_series,
            len(fam_exec),
        )

        # Storage Read + Write I/O time
        total_rows = eff_n_series * (eff_n_obs + cfg.data.horizon * len(fam_exec))
        io_s = 0.05 + 1.2e-6 * total_rows if eff_runtime == "local" else 2.2 + 8.0e-6 * total_rows

        # Sum execution work across executed models in this family
        per_series_p50 = 0.0
        per_series_p90 = 0.0
        for m in fam_exec:
            m_p50, m_p90 = _model_single_cell_seconds(
                m,
                cfg,
                n_series=eff_n_series,
                median_n_obs=eff_n_obs,
                hardware=hardware,
                model_priors=model_priors,
            )
            per_series_p50 += m_p50
            per_series_p90 += m_p90

        ovh_p50, ovh_p90 = _RUNTIME_EXEC_OVERHEAD_S.get((eff_runtime, eff_mode), (0.0, 0.0))
        exec_p50_s = ovh_p50 + (eff_n_series * per_series_p50) / max(1, slots)
        exec_p90_s = ovh_p90 + (eff_n_series * per_series_p90) / max(1, slots)

        # BigQuery SQL models have a query compilation + slot floor (~35s on live BQ)
        if eff_runtime == "bigquery":
            exec_p50_s = max(35.0, exec_p50_s)
            exec_p90_s = max(55.0, exec_p90_s)

        tot_p50 = prov_p50 + io_s + exec_p50_s
        tot_p90 = prov_p90 + io_s * 1.25 + exec_p90_s

        hourly_rate = _HOURLY_USD_BY_RUNTIME.get((eff_runtime, hardware), 0.40)
        node_equiv = max(1.0, float(slots) / (8.0 if hardware == "cpu" else 2.0))
        if eff_runtime in ("local", "bigquery"):
            node_equiv = 1.0
        est_usd = (tot_p50 / 3600.0) * hourly_rate * node_equiv

        family_breakdowns.append(
            FamilyDurationBreakdown(
                family=job.family,
                runtime=eff_runtime,
                mode=eff_mode,
                hardware=hardware,
                models=fam_models,
                borrowed_models=fam_borrowed,
                executed_models=fam_exec,
                n_series=eff_n_series,
                parallel_slots=slots,
                provision_p50_s=prov_p50,
                provision_p90_s=prov_p90,
                io_s=io_s,
                exec_p50_s=exec_p50_s,
                exec_p90_s=exec_p90_s,
                total_p50_s=tot_p50,
                total_p90_s=tot_p90,
                estimated_usd=est_usd,
            )
        )

    if family_breakdowns:
        bottleneck = max(family_breakdowns, key=lambda f: f.total_p50_s)
        base_p50 = bottleneck.total_p50_s
        base_p90 = max(f.total_p90_s for f in family_breakdowns)
        bottleneck_name = bottleneck.family
    else:
        base_p50 = 0.0
        base_p90 = 0.0
        bottleneck_name = "none"

    ensemble_s = 0.0
    if cfg.ensemble.enabled and len(cfg.models) >= 2:
        n_strats = len(cfg.ensemble.strategies)
        if where == "local":
            ensemble_s = 0.08 * n_strats + 0.002 * eff_n_series * n_strats
        else:
            ensemble_s = 3.5 + 0.45 * n_strats + 0.004 * eff_n_series * n_strats

    total_p50_s = base_p50 + ensemble_s
    total_p90_s = base_p90 + ensemble_s * 1.35
    total_usd = sum(f.estimated_usd for f in family_breakdowns)

    return DurationEstimate(
        total_p50_s=total_p50_s,
        total_p90_s=total_p90_s,
        total_p50_minutes=total_p50_s / 60.0,
        total_p90_minutes=total_p90_s / 60.0,
        bottleneck_family=bottleneck_name,
        ensemble_s=ensemble_s,
        estimated_usd=total_usd,
        families=tuple(family_breakdowns),
        basis=basis,
    )


def calibrate_duration_from_frames(
    jobs_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    source_dataset: str = "custom_registry",
    project_id: str | None = None,
) -> DurationCalibration:
    """Fit empirical provisioning and per-model fit priors from registry DataFrames (pure).

    Starts from the shipped :data:`EMPIRICAL_PROVISIONING_PRIORS` and
    :data:`EMPIRICAL_MODEL_FIT_PRIORS` and updates any `(runtime, mode, hardware)` or `base_model`
    with completed observations in ``jobs_df`` / ``metadata_df``.
    """
    eff_source = f"{project_id}.{source_dataset}" if project_id else source_dataset
    prov: dict[Any, Any] = dict(EMPIRICAL_PROVISIONING_PRIORS)
    models = dict(EMPIRICAL_MODEL_FIT_PRIORS)

    n_runs = 0
    if not jobs_df.empty and "runtime" in jobs_df.columns:
        n_runs = int(jobs_df["run_id"].nunique()) if "run_id" in jobs_df.columns else len(jobs_df)
        completed = (
            jobs_df[jobs_df["status"].isin(["COMPLETED", "SUCCEEDED"])]
            if "status" in jobs_df.columns
            else jobs_df
        )
        for (rt, hw), grp in completed.groupby(
            [
                completed["runtime"].fillna("spark"),
                completed.get("hardware", pd.Series("cpu", index=completed.index)).fillna("cpu"),
            ]
        ):
            if "provision_seconds" in grp.columns:
                vals = pd.to_numeric(grp["provision_seconds"], errors="coerce").dropna()
                vals = vals[vals > 0]
            else:
                vals = pd.Series(dtype=float)
            if vals.empty and "runtime_seconds" in grp.columns:
                rt_vals = pd.to_numeric(grp["runtime_seconds"], errors="coerce").dropna()
                vals = rt_vals[rt_vals > 0] * 0.25
            if len(vals) >= 1:
                mode_col = grp.get("spark_mode", pd.Series(dtype=str)).fillna("")
                default_mode = (
                    "serverless"
                    if rt == "spark"
                    else "custom_job"
                    if rt == "vertex"
                    else "single_vm"
                    if rt == "gce"
                    else "vertex"
                    if rt == "ray"
                    else "sql"
                    if rt == "bigquery"
                    else "job"
                )
                mode_val = (
                    str(mode_col.iloc[0])
                    if not mode_col.empty and mode_col.iloc[0]
                    else default_mode
                )
                pair = (
                    float(np.percentile(vals, 50)),
                    float(np.percentile(vals, 90)),
                )
                prov[(str(rt), mode_val, str(hw))] = pair
                prov[str(rt)] = pair

    n_cells = 0
    if (
        not metadata_df.empty
        and "model_type" in metadata_df.columns
        and "fit_seconds" in metadata_df.columns
    ):
        valid = metadata_df.copy()
        valid["fit_seconds"] = pd.to_numeric(valid["fit_seconds"], errors="coerce")
        valid = valid[valid["fit_seconds"] > 0]
        n_cells = len(valid)
        if not valid.empty:
            valid["base_model"] = valid["model_type"].astype(str).map(base_model_name)
            for bname, grp in valid.groupby("base_model"):
                if str(bname).startswith("ensemble_") or len(grp) < 1:
                    continue
                fit_vals = grp["fit_seconds"].to_numpy(dtype=np.float64)
                # Normalize typical 3-fold cell fit_seconds to single-fit base unit
                p50 = float(np.percentile(fit_vals, 50)) / 3.0
                p90 = float(np.percentile(fit_vals, 90)) / 3.0
                models[str(bname)] = (max(0.005, p50), max(0.010, p90))

    return DurationCalibration(
        provisioning_priors=prov,
        model_fit_priors=models,
        n_runs_analyzed=n_runs,
        n_cells_analyzed=n_cells,
        source_dataset=eff_source,
    )


def calibrate_duration_from_registry(
    *,
    settings: Settings | None = None,
    limit_runs: int = 500,
) -> DurationCalibration:
    """Query ``v_run_jobs`` and ``forecast_metadata`` in BigQuery to calibrate duration priors."""
    from ..errors import require_extra
    from ..registry.tables import bigquery_client
    from ..settings import Settings

    require_extra("gcp", purpose="Calibrating duration model from BigQuery registry")
    eff_settings = settings or Settings.resolve()
    client = bigquery_client(eff_settings.project_id, location=eff_settings.region)
    ds_ref = eff_settings.registry_dataset_ref

    jobs_sql = f"""
    SELECT
      run_id,
      runtime,
      IFNULL(hardware, 'cpu') AS hardware,
      spark_mode,
      status,
      provision_seconds,
      runtime_seconds
    FROM `{ds_ref}.v_run_jobs`
    WHERE status IN ('COMPLETED', 'SUCCEEDED')
    LIMIT {int(limit_runs) * 5}
    """
    meta_sql = f"""
    SELECT
      model_type,
      fit_seconds
    FROM `{ds_ref}.forecast_metadata`
    WHERE fit_seconds IS NOT NULL AND fit_seconds > 0
    LIMIT 100000
    """
    jobs_df = client.query(jobs_sql).to_dataframe()
    meta_df = client.query(meta_sql).to_dataframe()
    return calibrate_duration_from_frames(
        jobs_df,
        meta_df,
        source_dataset=ds_ref,
    )
