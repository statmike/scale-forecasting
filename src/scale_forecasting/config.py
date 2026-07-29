"""Load, validate, and freeze the run config — the single source of run behavior.

A run is one JSON file (DESIGN §9). It is validated here *before* anything executes,
and the frozen, normalized object is what gets logged verbatim to
``run_registry.raw_config`` — so the config *is* the experiment record (G2/G3).

Public surface (CONTRACTS §6):
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

# The full metric panel (CONTRACTS §2.3 / DESIGN §5.1). Kept here so the config's
# decision-metric field is self-contained; metrics.py (BUILD 2.1) must match this set.
DecisionMetric = Literal[
    "mae", "rmse", "mse", "mape", "smape", "wape", "mase", "rmsse", "bias", "coverage", "pinball"
]

# Ensemble strategies (DESIGN §5.2). "Learned" strategies train on backtest OOF and
# therefore require backtesting to be ON; "calculated" ones work either way.
CALCULATED_STRATEGIES = frozenset({"mean", "median", "inverse_error"})
LEARNED_STRATEGIES = frozenset({"nnls", "ridge", "xgb"})
Strategy = Literal["mean", "median", "inverse_error", "nnls", "ridge", "xgb"]


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
    # on the *same* series (DESIGN §13.1). Must be positive when set.
    series_limit: int | None = Field(default=None, gt=0)


class FeaturesConfig(BaseModel):
    """Optional feature engineering for the Python models (DESIGN §4).

    Defaults are conservative/generic (no transform, no holidays); the shipped
    ``example_config.json`` turns on holidays + log1p (D2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    holidays: list[str] = Field(default_factory=list)
    transform: Literal["none", "log1p", "boxcox"] = "none"
    exog: list[str] = Field(default_factory=list)
    lags: list[int] = Field(default_factory=list)
    fourier: bool = False
    level_shift: bool = False


class BacktestConfig(BaseModel):
    """Time-series cross-validation (DESIGN §5.1). Off by default (cheapest first run)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    scheme: Literal["expanding", "sliding"] = "expanding"
    n_folds: int = Field(default=3, ge=1)
    horizon: int = Field(default=28, gt=0)
    step: int = Field(default=28, gt=0)
    min_train: int = Field(default=180, gt=0)
    decision_metric: DecisionMetric = "wape"


class HpoConfig(BaseModel):
    """Hyperparameter optimization inside the node (optional)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    engine: Literal["optuna"] = "optuna"
    n_trials: int = Field(default=20, gt=0)


class EnsembleConfig(BaseModel):
    """Consensus across base models (DESIGN §5.2).

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
    # G3 lineage). Off by default: at 100k×N cells the object count + write cost is material, so a
    # run opts in explicitly (demos do; the hero scale run need not). See BaseModel.serialize.
    persist_models: bool = False
    use_gpu: bool = False
    gpu_type: str = "T4"
    # "auto" = profile-driven calibration (DESIGN §11.1), or a fixed fraction in (0, 1].
    gpu_fraction: Literal["auto"] | float = "auto"
    budget_usd: float = Field(default=50.0, ge=0.0)

    # --- Ray on Vertex (B4) ----------------------------------------------------
    # The Ray runtime sizes a *fixed* (non-autoscaling) cluster to the run's fan-out and packs
    # GPU-benefiting models (NeuralProphet) onto fractional T4 slots while stats/ML run on CPU
    # (DESIGN §11.1, D17). These knobs feed engines/ray_io.plan_cluster + calibrate_gpu_fraction;
    # they are inert unless python_runtime == "ray".
    #
    # Reuse opt-in: target an existing cluster by name (skip create + skip teardown). None (default)
    # = ephemeral per-run cluster (create → submit → delete-in-finally).
    ray_cluster_name: str | None = None
    # Machine types for the two fixed worker pools. GPU workers must be N1 for T4 attachment.
    ray_head_machine_type: str = "n1-standard-4"
    ray_cpu_machine_type: str = "n1-standard-8"
    ray_gpu_machine_type: str = "n1-standard-8"
    # GPUs per GPU worker node. T4 permits 1, 2, or 4 per node (not 3) — validated below.
    accelerator_count: int = Field(default=1, gt=0)
    # Fixed-pool sizing: how many cells one worker slot should chew through before we add another
    # node (amortizes per-node warm-up), plus a hard ceiling so a huge fan-out can't request an
    # unbounded cluster. n_gpu_nodes/n_cpu_nodes are derived, then clamped to [1, ray_max_nodes].
    ray_target_cells_per_slot: int = Field(default=8, gt=0)
    ray_max_nodes: int = Field(default=16, gt=0)
    # Auto-fraction calibration (gpu_fraction == "auto"): how many series to profile and the
    # headroom multiplier applied to measured peak GPU memory before dividing by device memory.
    gpu_calibration_samples: int = Field(default=3, gt=0)
    gpu_safety_margin: float = Field(default=1.3, gt=1.0)

    @model_validator(mode="after")
    def _check_gpu_fraction(self) -> ComputeConfig:
        if isinstance(self.gpu_fraction, float) and not (0.0 < self.gpu_fraction <= 1.0):
            raise ValueError("gpu_fraction must be 'auto' or a float in (0, 1]")
        # T4 attaches in counts of 1, 2, or 4 per node (3 is not a valid GPU count on Vertex/GCE).
        if self.gpu_type == "T4" and self.accelerator_count not in (1, 2, 4):
            raise ValueError(
                f"accelerator_count for T4 must be 1, 2, or 4 (got {self.accelerator_count})"
            )
        return self


# --- top-level config ----------------------------------------------------------


class RunConfig(BaseModel):
    """A complete, validated, frozen run specification (DESIGN §9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_name: str
    data: DataConfig
    python_runtime: Literal["spark", "ray"] = "spark"
    # explode | multi | naive — meaningful only when python_runtime == "spark".
    #   explode : cross-join series × model, key = (ts_id, model_type); independent cells (hero).
    #   multi   : one serverless batch per model family (submitted by the CLI submit helper).
    #   naive   : group by ts_id only, sequential model loop — the straggler anti-pattern, for
    #             demonstrating why explode's per-cell fan-out matters (DESIGN §2.1). Small scales.
    spark_method: Literal["explode", "multi", "naive"] | None = None
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

        # 1. Reconcile spark_method with the chosen Python runtime.
        if self.python_runtime == "spark":
            if self.spark_method is None:
                object.__setattr__(self, "spark_method", "explode")  # sensible default
        else:  # ray
            if self.spark_method is not None:
                raise ValueError(
                    "spark_method is only valid when python_runtime is 'spark' "
                    f"(got python_runtime='ray', spark_method='{self.spark_method}')"
                )

        # 2. Duplicate models are almost certainly a mistake — fail clearly.
        if len(set(self.models)) != len(self.models):
            raise ValueError(f"models contains duplicates: {self.models}")

        # 3. Learned ensembles need backtest OOF. If backtest is OFF, drop them with a
        #    warning rather than failing the whole run (DESIGN §5.2) — so the normalized
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

        return self


# --- fanout estimate -----------------------------------------------------------


@dataclass(frozen=True)
class Fanout:
    """Dry-run estimate of the work a run will schedule (DESIGN §11)."""

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
    invalid run (DESIGN §9).
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
