"""Load, validate, and freeze the run config — the single source of run behavior.

A run is one JSON file. It is validated here *before* anything executes,
and the frozen, normalized object is what gets logged verbatim to
``run_registry.raw_config`` — so the config *is* the experiment record.

Public surface:
- ``RunConfig`` — the frozen pydantic model.
- ``load_config(path) -> RunConfig`` — read + validate a JSON file.
- ``estimate_fanout(cfg) -> Fanout`` — the dry-run cell-count estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import ConfigError, get_logger

_log = get_logger(__name__)

# --- shared vocabularies -------------------------------------------------------

# The full metric panel. Kept here so the config's decision-metric field is
# self-contained; the metrics module must match this set.
DecisionMetric = Literal[
    "mae", "rmse", "mse", "mape", "smape", "wape", "mase", "rmsse", "bias", "coverage", "pinball"
]

# Ensemble strategies. "Learned" strategies train on backtest OOF and
# therefore require backtesting to be ON; "calculated" ones work either way.
CALCULATED_STRATEGIES = frozenset({"mean", "median", "inverse_error"})
LEARNED_STRATEGIES = frozenset({"nnls", "ridge", "xgb"})
Strategy = Literal["mean", "median", "inverse_error", "nnls", "ridge", "xgb"]

# Per-family compute vocabulary (kept as Literals to match python_runtime/spark_deps idiom).
# ``ComputeFamily`` mirrors ``models.base_model.Family`` *minus* "native": native models always run
# in BigQuery (their natural engine), so they are never given a per-family runtime choice.
ComputeFamily = Literal["statistical", "ml", "deep_learning"]
# ``JobFamily`` is the identity vocabulary of a *job* in the run DAG: every model family that can
# launch a job (``ComputeFamily`` + "native", which runs in BigQuery) plus the downstream "ensemble"
# node. It is the ``family`` component of a job's deterministic id (see ``registry.ids``), one step
# broader than ``ComputeFamily`` since native and ensemble produce jobs but take no runtime choice.
JobFamily = Literal["statistical", "ml", "deep_learning", "native", "ensemble"]
Runtime = Literal["spark", "ray"]
SparkMode = Literal["serverless", "cluster"]
Hardware = Literal["cpu", "gpu"]
GpuType = Literal["T4", "L4"]
EnsembleMode = Literal["barrier", "microbatch"]


# --- nested config blocks ------------------------------------------------------


class DataConfig(BaseModel):
    """Where the series come from and their shape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_table: str
    ts_id_col: str = "ts_id"
    date_col: str = "ds"
    target_col: str = "y"
    freq: str = "D"
    horizon: int = Field(default=28, gt=0)
    # None = use every series; an int subsets the shipped data to demo small→large
    # on the *same* series. Must be positive when set.
    series_limit: int | None = Field(default=None, gt=0)


class FeaturesConfig(BaseModel):
    """Optional feature engineering for the Python models.

    Defaults are conservative/generic (no transform, no holidays); the shipped
    ``example_config.json`` turns on holidays + log1p.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    holidays: list[str] = Field(default_factory=list)
    transform: Literal["none", "log1p", "boxcox"] = "none"
    exog: list[str] = Field(default_factory=list)
    lags: list[int] = Field(default_factory=list)
    fourier: bool = False
    level_shift: bool = False


class BacktestConfig(BaseModel):
    """Time-series cross-validation. Off by default (cheapest first run)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    scheme: Literal["expanding", "sliding"] = "expanding"
    n_folds: int = Field(default=3, ge=1)
    horizon: int = Field(default=28, gt=0)
    step: int = Field(default=28, gt=0)
    min_train: int = Field(default=180, gt=0)
    decision_metric: DecisionMetric = "wape"


class HpoConfig(BaseModel):
    """Hyperparameter optimization on the aligned backtest (optional).

    Off by default. When ``enabled``, an Optuna study tunes each model's ``search_space`` on the
    backtest folds and the winning params are stamped to ``forecast_metadata.best_params`` (see
    `scale_forecasting.hpo`). HPO therefore requires ``backtest.enabled``.

    ``granularity`` is the DS-facing cost knob:

    * ``fleetwide`` (default) — tune each model **once** on a ``sample_size`` sample of series and
      apply the winner across *all* series. The only granularity affordable at the 100k hero scale:
      the study runs on the driver before fan-out (a handful of series × ``n_trials`` fits), not per
      cell.
    * ``per_series`` — tune inside every cell (``n_trials`` fits *per series*). Accurate for the
      tail of hard series but multiplies fit cost by ``n_trials`` fleet-wide; an explicit opt-in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    engine: Literal["optuna"] = "optuna"
    n_trials: int = Field(default=20, gt=0)
    granularity: Literal["fleetwide", "per_series"] = "fleetwide"
    # Fleetwide sample: how many series to tune on before applying the winner across the fleet.
    sample_size: int = Field(default=20, gt=0)


class EnsembleConfig(BaseModel):
    """Consensus across base models.

    ``strategies`` is a list so several ensembles can run at once. The singular
    ``strategy`` string is accepted as shorthand for a one-element list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    # Pydantic v2 deep-copies mutable defaults, so a literal default is safe here.
    strategies: list[Strategy] = ["median"]
    prune_threshold: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_singular_strategy(cls, data: Any) -> Any:
        # "strategy": "nnls" is shorthand for "strategies": ["nnls"].
        if isinstance(data, dict) and "strategy" in data:
            data = dict(data)
            singular = data.pop("strategy")
            data.setdefault("strategies", [singular] if isinstance(singular, str) else singular)
        return data


class FamilyCompute(BaseModel):
    """A sparse per-family compute override, layered over the flat `ComputeConfig` defaults.

    Every field is optional: an unset field inherits the run-level default (``python_runtime``,
    Spark ``serverless``, CPU, the flat ``gpu_type``), so a config sets only what a family needs to
    differ on. There is no ``native`` family here — native models always run in BigQuery. See
    `RunConfig.resolve_family_compute` for how these layer onto the defaults; the block is inert
    until the DAG orchestrator consumes it.

    Hardware constraints (validated): only the ``deep_learning`` family may request a GPU (enforced
    where the family key is known, in `ComputeConfig`); Dataproc Serverless offers **L4 only** (no
    T4 — use ``spark_mode="cluster"`` or ``runtime="ray"`` for T4); Spark-only fields
    (``spark_mode``/``spark_cluster_name``) are rejected on ``runtime="ray"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: Runtime | None = None
    spark_mode: SparkMode | None = None
    # Reuse an existing Dataproc cluster by name (requires spark_mode="cluster"). None = ephemeral
    # per-run cluster (create → submit → delete), mirroring the Ray cluster lifecycle.
    spark_cluster_name: str | None = None
    hardware: Hardware | None = None
    gpu_type: GpuType | None = None

    @model_validator(mode="after")
    def _check(self) -> FamilyCompute:
        if self.runtime == "ray" and (
            self.spark_mode is not None or self.spark_cluster_name is not None
        ):
            raise ValueError(
                "spark_mode/spark_cluster_name are only valid when runtime is 'spark'"
            )
        if self.spark_cluster_name is not None and self.spark_mode not in (None, "cluster"):
            raise ValueError("spark_cluster_name requires spark_mode='cluster'")
        if self.spark_mode == "serverless" and self.gpu_type == "T4":
            raise ValueError(
                "Dataproc Serverless supports L4 only, not T4; "
                "use spark_mode='cluster' or runtime='ray' for T4"
            )
        if self.hardware == "cpu" and self.gpu_type is not None:
            raise ValueError(
                "gpu_type is set but hardware='cpu'; drop gpu_type or set hardware='gpu'"
            )
        return self


class EnsembleCompute(BaseModel):
    """Compute for the ensemble DAG node — its own runtime and trigger mode.

    Distinct from `EnsembleConfig` (which selects the ensemble *strategies*): this picks *where* and
    *when* the ensemble runs. ``mode="barrier"`` ensembles once after every base model finishes;
    ``mode="microbatch"`` ensembles each series as soon as its upstream base models complete. Inert
    until the DAG orchestrator consumes it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: Runtime = "spark"
    mode: EnsembleMode = "barrier"
    spark_mode: SparkMode | None = None
    spark_cluster_name: str | None = None
    # Seconds between readiness polls in ``mode="microbatch"`` (how often the ensemble drains the
    # series whose base models have all landed). Inert in ``barrier`` mode. Part of the run_id.
    microbatch_interval_s: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> EnsembleCompute:
        if self.runtime == "ray" and (
            self.spark_mode is not None or self.spark_cluster_name is not None
        ):
            raise ValueError(
                "spark_mode/spark_cluster_name are only valid when runtime is 'spark'"
            )
        if self.spark_cluster_name is not None and self.spark_mode not in (None, "cluster"):
            raise ValueError("spark_cluster_name requires spark_mode='cluster'")
        return self


class ComputeConfig(BaseModel):
    """Runtime scale, dependency delivery, and cost guardrails."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_parallelism: int = Field(default=1000, gt=0)
    # Target cells per Spark bucket (applyInPandas frame). Buckets = ceil(cells / this), so each
    # task materializes ~this many series-histories — the knob that keeps per-task memory bounded as
    # scale grows. This is a *shuffle-partition* count, distinct from executor concurrency (capped
    # separately by spark.dynamicAllocation.maxExecutors). Small keeps frames tiny; large amortizes
    # write_cells over fatter batches. See engines/spark_io.default_bucket_count.
    bucket_target_cells: int = Field(default=8, gt=0)
    machine_family: str = "auto"
    spark_deps: Literal["packed_venv", "container"] = "packed_venv"
    # Persist each fitted model as a GCS artifact (ObjectRef in forecast_metadata.model_artifact,
    # model-artifact lineage). Off by default: at 100k×N cells the object count + write cost is
    # material, so a run opts in explicitly (demos do; the hero scale run need not). See
    # BaseModel.serialize.
    persist_models: bool = False
    use_gpu: bool = False
    gpu_type: str = "T4"
    # "auto" = profile-driven calibration, or a fixed fraction in (0, 1].
    gpu_fraction: Literal["auto"] | float = "auto"
    budget_usd: float = Field(default=50.0, ge=0.0)

    # --- Ray on Vertex ---------------------------------------------------------
    # The Ray runtime sizes an *autoscaling* cluster to the run's fan-out (default) and packs
    # GPU-benefiting models (NeuralProphet) onto fractional T4 slots while stats/ML run on CPU.
    # These knobs feed engines/ray_io.plan_cluster + calibrate_gpu_fraction; they are inert unless
    # python_runtime == "ray".
    #
    # Autoscaling. Autoscaling is the default (ray_autoscale): each pool scales in [min, max] driven
    # by Ray's pending-task demand, so a pool can grow to chew a deep task queue and shrink the
    # expensive T4 pool when idle — the right default for a bursty, embarrassingly-parallel fleet
    # where a fixed pool can do neither. Determinism is preserved because the whole spec (the flag,
    # the per-pool min/max, and the fixed-size-equivalent the fan-out implies) is a pure function of
    # the config, snapshotted into run_id + job_telemetry. ray_autoscale=False selects a fixed-size
    # cluster instead. NOTE: under autoscaling the Vertex SDK ignores a pool's node_count (it starts
    # at min_replica_count and scales to max); the derived per-pool node count is therefore the
    # *initial* size only for the fixed path, and telemetry otherwise.
    #
    # Reuse opt-in: target an existing cluster by name (skip create + skip teardown). None (default)
    # = ephemeral per-run cluster (create → submit → delete-in-finally).
    ray_cluster_name: str | None = None
    # Priority-ordered candidate regions for the ephemeral cluster. GPU capacity is regional and can
    # stock out transiently (a create is accepted, then fails to reach RUNNING with "Resources are
    # insufficient in region: <r>") even when quota is fine — so the launcher tries these in order,
    # tearing down each stocked-out attempt first. None (default) = just the [settings.region] list.
    # Only the *cluster* hops; the data plane (dataset/buckets/connection, hence config staging and
    # registry writes) stays in settings.region, so a cross-region list means cross-region reads.
    ray_regions: list[str] | None = None
    # Machine types for the two fixed worker pools. GPU workers must be N1 for T4 attachment.
    # The head node runs no cells (the driver only), so the worker pools stay independently sized —
    # but it must be big enough to serve the Ray dashboard/proxy leg. Vertex has a hard >18GB RAM
    # floor (n1-standard-4 = 15GB is rejected at create), but the *operational* floor is higher: a
    # 30GB/8-vCPU head (n1-standard-8) boots and reaches RUNNING yet its managed dashboard proxy
    # never comes up, so the JobSubmissionClient `/api/version` handshake 524s (30s timeout, 0
    # bytes). n1-standard-16 (60GB/16-vCPU) serves the handshake in <7s. So the
    # head default is n1-standard-16; do not drop it below that or Ray job submission will hang.
    ray_head_machine_type: str = "n1-standard-16"
    ray_cpu_machine_type: str = "n1-standard-8"
    ray_gpu_machine_type: str = "n1-standard-8"
    # GPUs per GPU worker node. T4 permits 1, 2, or 4 per node (not 3) — validated below.
    accelerator_count: int = Field(default=1, gt=0)
    # Fixed-pool sizing: how many cells one worker slot should chew through before we add another
    # node (amortizes per-node warm-up), plus a hard ceiling so a huge fan-out can't request an
    # unbounded cluster. n_gpu_nodes/n_cpu_nodes are derived, then clamped to [1, ray_max_nodes].
    ray_target_cells_per_slot: int = Field(default=8, gt=0)
    ray_max_nodes: int = Field(default=16, gt=0)
    # Autoscaling (default-on). When True each worker pool is created with a Vertex
    # AutoscalingSpec(min, max) and grows/shrinks with Ray's task demand; when False both pools are
    # fixed at their derived node_count. The per-pool min/max are
    # resolved offline in plan_cluster and snapshotted into run_id + job_telemetry, so an autoscaled
    # run stays as reproducible/auditable as a fixed one.
    ray_autoscale: bool = True
    # Per-pool autoscaling floor. Vertex Ray keeps at least one node allocated per pool (an
    # effective min of 0 is not honored), so the floor is 1; raise it to pre-warm a pool and skip
    # the cold 1→N ramp. Inert when ray_autoscale is False.
    ray_cpu_min_nodes: int = Field(default=1, gt=0)
    ray_gpu_min_nodes: int = Field(default=1, gt=0)
    # Per-pool autoscaling ceiling. None falls back to the shared ray_max_nodes, so a run can give
    # the (cheap) CPU pool a high ceiling while capping the (expensive) T4 GPU pool independently.
    # Inert when ray_autoscale is False (both pools are then fixed at their derived node_count).
    ray_cpu_max_nodes: int | None = Field(default=None, gt=0)
    ray_gpu_max_nodes: int | None = Field(default=None, gt=0)
    # Auto-fraction calibration (gpu_fraction == "auto"): how many series to profile and the
    # headroom multiplier applied to measured peak GPU memory before dividing by device memory.
    gpu_calibration_samples: int = Field(default=3, gt=0)
    gpu_safety_margin: float = Field(default=1.3, gt=1.0)
    # How the Ray driver reads the source panel. Both paths hit the SAME BigQuery Storage Read API
    # (no query slots, matching Spark) and yield the SAME driver-side pandas panel, so the
    # downstream fan-out is byte-identical either way — this knob only chooses the client:
    #   driver_collect (default) : google-cloud-bigquery-storage BigQueryReadClient, assembling the
    #                              Arrow streams. The default, known-good path.
    #   ray_data                 : ray.data.read_bigquery(project_id=, dataset=), the Ray-native
    #                              reader (same Storage Read API underneath), then .to_pandas().
    #                              Opt-in — the Ray-native ingest path, kept off by default so the
    #                              known-good reader stays the default until a live Ray run vets it.
    ray_read_mode: Literal["driver_collect", "ray_data"] = "driver_collect"

    # Storage Read API parallelism: the max number of read streams to request when collecting the
    # source panel. Shared across engines that read through the Storage Read API — the Spark
    # connector (its ``maxParallelism`` option) and Ray's driver_collect reader (the
    # ``create_read_session`` ``max_stream_count``). 0 (default) lets the server pick the stream
    # count from the table size — the known-good default; set a positive cap to bound read
    # parallelism (e.g. to stay inside a slot/quota budget). Inert for the ray_data path (Ray sizes
    # its own blocks) and for BigQuery-native models (they read via the query API, not the Storage
    # Read API). Part of the config, so changing it yields a new run_id.
    read_max_streams: int = Field(default=0, ge=0)

    # --- per-family compute (the multi-runtime job DAG) ------------------------
    # Sparse overrides layered over the flat defaults above: each family (statistical/ml/
    # deep_learning) may pick its own runtime + hardware; an unset family inherits the run-level
    # python_runtime / Spark-serverless / CPU defaults. Native models are never here — they always
    # run in BigQuery. ``ensemble`` runs the ensemble DAG node on its own runtime with a
    # barrier|microbatch trigger. Both are inert until the DAG orchestrator consumes them; a config
    # that omits them behaves exactly as before. See RunConfig.resolve_family_compute.
    families: dict[ComputeFamily, FamilyCompute] = Field(default_factory=dict)
    ensemble: EnsembleCompute = Field(default_factory=EnsembleCompute)

    @model_validator(mode="after")
    def _check_families(self) -> ComputeConfig:
        # GPU is a deep_learning-only capability; statistical/ml are CPU work. The family key is
        # known here (unlike inside FamilyCompute), so this is where that constraint is enforced.
        for fam, fc in self.families.items():
            if fam != "deep_learning" and (fc.hardware == "gpu" or fc.gpu_type is not None):
                raise ValueError(
                    f"family '{fam}' cannot use a GPU (hardware='gpu'/gpu_type set); "
                    "only the deep_learning family supports GPU"
                )
        return self

    @model_validator(mode="after")
    def _check_gpu_fraction(self) -> ComputeConfig:
        if isinstance(self.gpu_fraction, float) and not (0.0 < self.gpu_fraction <= 1.0):
            raise ValueError("gpu_fraction must be 'auto' or a float in (0, 1]")
        # T4 attaches in counts of 1, 2, or 4 per node (3 is not a valid GPU count on Vertex/GCE).
        if self.gpu_type == "T4" and self.accelerator_count not in (1, 2, 4):
            raise ValueError(
                f"accelerator_count for T4 must be 1, 2, or 4 (got {self.accelerator_count})"
            )
        # Per-pool autoscaling bounds must be coherent: an explicit max cannot fall below its min
        # (an unset max defers to ray_max_nodes, checked against the pool min too). Fail at load
        # rather than at cluster-create, where a bad spec would waste a provision attempt.
        for pool, min_nodes, max_nodes in (
            ("cpu", self.ray_cpu_min_nodes, self.ray_cpu_max_nodes),
            ("gpu", self.ray_gpu_min_nodes, self.ray_gpu_max_nodes),
        ):
            resolved_max = max_nodes if max_nodes is not None else self.ray_max_nodes
            if min_nodes > resolved_max:
                raise ValueError(
                    f"ray_{pool}_min_nodes ({min_nodes}) exceeds the {pool} pool max "
                    f"({resolved_max}); lower the min or raise ray_{pool}_max_nodes/ray_max_nodes"
                )
        return self


# --- resolved per-family compute ----------------------------------------------


@dataclass(frozen=True)
class ResolvedFamilyCompute:
    """One family's effective compute after layering its override on the flat defaults (pure).

    The fully-resolved plan the DAG orchestrator acts on: ``runtime`` is where the family's job
    runs; ``spark_mode``/``spark_cluster_name`` are ``None`` unless ``runtime == "spark"``;
    ``gpu_type`` is ``None`` unless ``hardware == "gpu"``. Produced by
    `RunConfig.resolve_family_compute`.
    """

    family: str
    runtime: str
    spark_mode: str | None
    spark_cluster_name: str | None
    hardware: str
    gpu_type: str | None


# --- top-level config ----------------------------------------------------------


class RunConfig(BaseModel):
    """A complete, validated, frozen run specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_name: str
    data: DataConfig
    python_runtime: Literal["spark", "ray"] = "spark"
    models: list[str] = Field(min_length=1)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    hpo: HpoConfig = Field(default_factory=HpoConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    @model_validator(mode="after")
    def _normalize(self) -> RunConfig:
        # A frozen model is mutated in place here via object.__setattr__ (the supported
        # pydantic-v2 pattern) so the validator returns `self`, not a copy.

        # 1. Duplicate models are almost certainly a mistake — fail clearly.
        if len(set(self.models)) != len(self.models):
            raise ValueError(f"models contains duplicates: {self.models}")

        # 2. HPO tunes on the backtest folds (decision_metric), so it needs backtesting ON.
        #    Fail fast at load rather than deep in an engine (a run with nothing to optimize).
        if self.hpo.enabled and not self.backtest.enabled:
            raise ValueError(
                "hpo.enabled requires backtest.enabled: HPO optimizes the backtest "
                "decision_metric, so there is nothing to tune with backtesting off."
            )

        # 3. Learned ensembles need backtest OOF. If backtest is OFF, drop them with a
        #    warning rather than failing the whole run — so the normalized
        #    config that lands in the registry honestly reflects what will run.
        if self.ensemble.enabled and not self.backtest.enabled:
            learned = [s for s in self.ensemble.strategies if s in LEARNED_STRATEGIES]
            if learned:
                kept = [s for s in self.ensemble.strategies if s not in LEARNED_STRATEGIES]
                _log.warning(
                    "Dropping learned ensemble strategies %s: they require backtest.enabled=true. "
                    "Keeping %s.",
                    learned,
                    kept,
                )
                object.__setattr__(
                    self, "ensemble", self.ensemble.model_copy(update={"strategies": kept})
                )

        # 4. Harden every per-family compute override by resolving it now, so an incoherent
        #    combination the per-block validator can't see (e.g. a T4 GPU inherited onto Dataproc
        #    Serverless) fails at load rather than at submit. Resolution is pure; the DAG
        #    orchestrator reuses the same resolver, so a validated config needs no re-check.
        for fam in self.compute.families:
            self.resolve_family_compute(fam)

        return self

    def resolve_family_compute(self, family: str) -> ResolvedFamilyCompute:
        """Resolve one family's effective compute by layering its override on the flat defaults.

        Pure and deterministic — the single resolver the DAG orchestrator also uses, so a config
        validated at load needs no re-check at submit. ``family`` is a compute family
        (``statistical``/``ml``/``deep_learning``); ``native`` has no compute choice (it always runs
        in BigQuery) and raises. Resolution rules for unset override fields:

        * ``runtime`` → ``python_runtime``.
        * ``hardware`` → ``gpu`` only for ``deep_learning`` (when ``compute.use_gpu`` or an explicit
          override); every other family is ``cpu``.
        * Spark: ``spark_mode`` → ``serverless``; ``spark_cluster_name`` applies only under
          ``cluster``. On ``ray`` both are ``None``.
        * ``gpu_type`` (when ``hardware == "gpu"``) → the flat ``compute.gpu_type``, but **forced to
          L4** on Dataproc Serverless (no T4 there). A T4 inherited onto Serverless raises.
        """
        if family == "native":
            raise ValueError(
                "native models always run in BigQuery; they have no per-family compute"
            )
        ov = self.compute.families.get(family) or FamilyCompute()
        runtime = ov.runtime or self.python_runtime

        if family == "deep_learning":
            hardware = ov.hardware or ("gpu" if self.compute.use_gpu else "cpu")
        else:
            hardware = "cpu"

        if runtime == "spark":
            spark_mode = ov.spark_mode or "serverless"
            spark_cluster_name = ov.spark_cluster_name if spark_mode == "cluster" else None
        else:
            spark_mode = None
            spark_cluster_name = None

        gpu_type: str | None
        if hardware == "gpu":
            if runtime == "spark" and spark_mode == "serverless":
                if ov.gpu_type == "T4":
                    raise ValueError(
                        f"family '{family}': Dataproc Serverless supports L4 only, not T4; "
                        "use spark_mode='cluster' or runtime='ray' for T4"
                    )
                gpu_type = "L4"
            else:
                gpu_type = ov.gpu_type or self.compute.gpu_type
        else:
            gpu_type = None

        return ResolvedFamilyCompute(
            family=family,
            runtime=runtime,
            spark_mode=spark_mode,
            spark_cluster_name=spark_cluster_name,
            hardware=hardware,
            gpu_type=gpu_type,
        )

    def with_series_limit(self, n_series: int | None) -> RunConfig:
        """Return a copy with ``data.series_limit`` overridden (``self`` if ``n_series`` is None).

        The scale knob every submit path shares (the 10 → 100 → 1k → 100k story). Because it
        changes the config, the copy yields a distinct ``run_id``, so each scale is its own
        queryable run.
        """
        if n_series is None:
            return self
        return self.model_copy(
            update={"data": self.data.model_copy(update={"series_limit": n_series})}
        )


# --- fanout estimate -----------------------------------------------------------


@dataclass(frozen=True)
class Fanout:
    """Dry-run estimate of the work a run will schedule."""

    n_series: int | None  # None = unlimited (unknown until the data is read)
    n_models: int
    n_folds: int  # backtest folds, or 1 when backtesting is off
    n_cells: int | None  # n_series × n_models × n_folds; None when n_series unknown


def estimate_fanout(cfg: RunConfig) -> Fanout:
    """Compute the cell-count estimate (n_series × n_models × folds).

    When ``data.series_limit`` is unset, series count isn't known offline, so
    ``n_series`` and ``n_cells`` are ``None`` (the CLI reports "all series").
    """
    n_series = cfg.data.series_limit
    n_models = len(cfg.models)
    n_folds = cfg.backtest.n_folds if cfg.backtest.enabled else 1
    n_cells = None if n_series is None else n_series * n_models * n_folds
    return Fanout(n_series=n_series, n_models=n_models, n_folds=n_folds, n_cells=n_cells)


# --- loading -------------------------------------------------------------------


def load_config(path: str | Path) -> RunConfig:
    """Read a JSON config file and return a validated, frozen ``RunConfig``.

    All failure modes (missing file, bad JSON, invalid schema) surface as a single
    ``ConfigError`` with a clear message, so callers fail fast and never log an
    invalid run.
    """
    p = Path(path)
    try:
        raw = p.read_text()
    except OSError as e:
        raise ConfigError(f"cannot read config file '{p}': {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(f"config file '{p}' is not valid JSON: {e}") from e
    try:
        return RunConfig(**data)
    except ValidationError as e:
        raise ConfigError(f"invalid config '{p}':\n{e}") from e


def load_config_uri(uri: str) -> RunConfig:
    """Load a validated config from a local path **or** a ``gs://`` URI (the portable source).

    A ``gs://`` URI is the staged config an emitted launch command references, so any
    ADC-authenticated machine can re-run from it without a local file. Anything else is treated as a
    filesystem path (delegates to `load_config`). Failure modes surface as a single ``ConfigError``.
    """
    if not uri.startswith("gs://"):
        return load_config(uri)
    bucket, _, blob = uri[len("gs://") :].partition("/")
    if not bucket or not blob:
        raise ConfigError(f"malformed config URI '{uri}' (expected gs://bucket/path.json)")
    from google.cloud import storage

    try:
        raw = storage.Client().bucket(bucket).blob(blob).download_as_text()
    except Exception as e:  # noqa: BLE001 - surface any fetch failure as one ConfigError
        raise ConfigError(f"cannot read config URI '{uri}': {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(f"config URI '{uri}' is not valid JSON: {e}") from e
    try:
        return RunConfig(**data)
    except ValidationError as e:
        raise ConfigError(f"invalid config '{uri}':\n{e}") from e
