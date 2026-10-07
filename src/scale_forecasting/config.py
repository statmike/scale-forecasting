"""Load, validate, and freeze the run config — the single source of run behavior.

A run is one JSON file. It is validated here *before* anything executes,
and the frozen, normalized object is what gets logged verbatim to
``run_registry.raw_config`` — so the config *is* the experiment record.

Public surface:
- ``RunConfig`` — the frozen pydantic model.
- ``load_config(path) -> RunConfig`` — read + validate a JSON file.
- ``estimate_workload(cfg, obs_counts=...) -> Workload`` — the dry-run work estimate: cells always,
  fits and fold cohorts when the caller supplies per-series observation counts.
- ``estimate_fanout(cfg) -> Fanout`` — the same estimate narrowed to its four count fields.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .capacity import DEFAULT_POLICIES, CapacityPolicy
from .errors import ConfigError, get_logger

_log = get_logger(__name__)

# --- shared vocabularies -------------------------------------------------------


def _registered_metric(value: str) -> str:
    """Accept any metric name the metric registry knows, and no other."""
    from .metrics import METRIC_NAMES

    if value not in METRIC_NAMES:
        raise ValueError(f"unknown decision_metric '{value}'; available: {', '.join(METRIC_NAMES)}")
    return value


# The decision metric: any metric the `metrics/` registry has, checked at config-validation time.
#
# This used to be a hand-written `Literal` of the fifteen built-ins, which meant that adding a
# metric to a deployment meant editing this module as well as writing the metric. The registry is
# the one source of truth now — `metrics.METRIC_NAMES` — and `registry.rows.METRIC_COLUMNS`, the
# `forecast_metadata` DDL and the Storage Write API spec are all generated from the same tuple.
#
# **Widening it costs no `run_id`.** The digest hashes dumped *values*, never the schema, and
# `"wape"` dumps as `"wape"` whether the field is a `Literal` or a validated `str` — so this change
# moved no id, and neither does adding a metric. Re-ordering `METRIC_NAMES` would still move every
# metric column, which is why additions go at the tail.
DecisionMetric = Annotated[str, AfterValidator(_registered_metric)]


def corrected_arm_for(decision_metric: str) -> str:
    """Which *corrected* point-forecast arm a decision metric implies — `"mean"` or `"median"`.

    A metric whose loss is quadratic in the error is minimised by the **mean**; everything that is
    absolute-error-shaped, and every proper interval score, is minimised by the **median**. The
    pairing is a theorem, not a preference, so it is what a run falls back to whenever no
    measurement is available to do better.

    One rule, two callers. `calibration.select_arm` uses it under `point_forecast="auto"` to know
    which arm it is weighing `raw` against; `RunConfig._normalize`'s warning about an explicit
    `median` under a squared-error metric is the same rule read the other way. Splitting it across
    those two would let a fleetwide judgement and a per-series selection disagree about what
    "corrected" means, which is the kind of drift nobody notices until a leaderboard reads oddly.

    **The answer is declared on the metric** (`BaseMetric.mean_optimal`), not held here as a list of
    names. It used to be a frozenset of four, which was correct for the fifteen shipped metrics and
    unreachable for a sixteenth: a deployment that adds a squared-error metric would have been given
    the median, silently and wrongly, with nowhere to say otherwise. An unregistered name still
    answers `"median"` — the same answer the frozenset gave it — so this reads identically for every
    caller that was ever correct.

    No longer what an unset `output.point_forecast` resolves to: with a backtest present the
    default is `auto`, which measures the choice per series rather than deducing it fleetwide. This
    function is what `auto` falls back to when a cell has nothing to measure.

    Says nothing about whether a backtest exists — that is the caller's guard.
    """
    from .metrics.base_metric import _REGISTRY

    metric_cls = _REGISTRY.get(decision_metric)
    return "mean" if metric_cls is not None and metric_cls.mean_optimal else "median"


# Ensemble strategies. "Learned" strategies train on backtest OOF and
# therefore require backtesting to be ON; "calculated" ones work either way.
CALCULATED_STRATEGIES = frozenset({"mean", "median", "inverse_error"})
LEARNED_STRATEGIES = frozenset({"nnls", "ridge", "xgb"})
Strategy = Literal["mean", "median", "inverse_error", "nnls", "ridge", "xgb"]

# Per-family compute vocabulary (kept as Literals to match python_runtime/spark_deps idiom).
# ``ComputeFamily`` mirrors ``models.base_model.Family`` *minus* "native": native models always run
# in BigQuery (their natural engine), so they are never given a per-family runtime choice.
ComputeFamily = Literal["statistical", "ml", "deep_learning", "automl"]
# The same vocabulary as a runtime tuple, so a ``str`` family can be checked against it.
COMPUTE_FAMILIES: tuple[ComputeFamily, ...] = ("statistical", "ml", "deep_learning", "automl")
# ``JobFamily`` is the identity vocabulary of a *job* in the run DAG: every model family that can
# launch a job (``ComputeFamily`` + "native", which runs in BigQuery) plus the downstream "ensemble"
# node. It is the ``family`` component of a job's deterministic id (see ``registry.ids``), one step
# broader than ``ComputeFamily`` since native and ensemble produce jobs but take no runtime choice.
JobFamily = Literal["statistical", "ml", "deep_learning", "automl", "native", "ensemble"]
Runtime = Literal["spark", "ray", "vertex", "gce", "gke", "vertex_automl"]
SparkMode = Literal["serverless", "cluster"]
GkeMode = Literal["job", "ray"]
RayMode = Literal["vertex", "gke"]
AutomlMode = Literal["tabular_workflow", "training_job"]
Hardware = Literal["cpu", "gpu"]

# What a `model_params` value may be. Deliberately narrow: the canonical config string is
# ``json.dumps(model_dump(mode="json"), sort_keys=True)`` and that string *is* the ``run_id``, so a
# value that does not round-trip through JSON either raises at digest time or hashes differently
# depending on who serialises it. Scalars and flat lists cover every per-model knob we have
# (``n_lags``, ``batch_size``, a SARIMAX ``order`` triple); a nested structure would fit the digest
# fine but is not needed, and leaving it out keeps the error message useful.
ModelParam = bool | int | float | str | None | list[bool | int | float | str | None]


def _is_non_finite(value: Any) -> bool:
    """True for NaN and ±inf. Bools are ints to Python, so they are excluded explicitly."""
    return isinstance(value, float) and not math.isfinite(value)


def _as_compute_family(family: str) -> ComputeFamily:
    """``family`` as the `ComputeFamily` literal ``compute.families`` is keyed by.

    Callers hold a model's family as a plain ``str``; a name outside the vocabulary is a caller
    bug, and naming it beats resolving the flat defaults for a family that does not exist.
    """
    for known in COMPUTE_FAMILIES:
        if family == known:
            return known
    raise ValueError(f"unknown compute family {family!r}; expected one of {COMPUTE_FAMILIES}")


GpuType = Literal["T4", "L4", "A100", "A100_80GB"]
# What a series too short for the requested fold grid gets. Ordered by what each one gives up:
# folds, fold independence, training history, the series, the run.
ShortSeriesPolicy = Literal["adapt", "overlap", "shrink_train", "skip", "error"]
EnsembleMode = Literal["barrier", "microbatch"]
ProfileMode = Literal["off", "auto", "always"]
ProfileMeasure = Literal["off", "harvest", "controlled"]
# `compute.profile.source` is not a closed set: besides the three keywords it accepts any run_id,
# which is the whole point ("size this run like run X"). The keywords are named here so the
# validator and the resolver agree on them in one place.
PROFILE_SOURCE_KEYWORDS = ("none", "auto", "baseline")
_RUN_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{12}$")


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

    **There is no ``lags`` field, and the reason is a design rule rather than an omission.**
    "Lag" means two different things, and only one of them can be built out here:

    - A lag of the **target** is only honest at predict time if the model rolls its own
      forecasts forward one step at a time. That is recursion, it belongs to the model, and
      the models that need it already own it — `_lag_forecaster` does it for ``lightgbm`` /
      ``xgboost`` / ``regression_lags``, and NeuralProphet does it behind ``n_lags``. A
      config cannot supply one, because nobody knows the future target; the removed field
      filled the horizon with a flat line at the last observation and handed the result to
      models that had fitted coefficients against real history.
    - A lag of a **covariate** is honest, because a covariate's future is as knowable as the
      covariate itself. No model in the suite lags one internally, so this is the gap that
      ``exog_lags`` fills.

    ``exog_lags`` maps an ``exog`` column to the lags to build from it: ``{"promo": [1, 7]}``
    adds ``promo_lag_1`` and ``promo_lag_7``. They are ordinary columns by the time any model
    sees them, so every ``supports_exog`` model can use them.

    Precedence, when the two rules ever meet: **the model wins.** A model that lags covariates
    internally sets ``lags_covariates_internally`` and is handed the unlagged columns only, so
    a covariate is never lagged twice. Every shipped model leaves that flag False today.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    holidays: list[str] = Field(default_factory=list)
    transform: Literal["none", "log1p", "boxcox"] = "none"
    exog: list[str] = Field(default_factory=list)
    static_covariates: list[str] = Field(default_factory=list)
    future_covariates: list[str] = Field(default_factory=list)
    past_covariates: list[str] = Field(default_factory=list)
    exog_lags: dict[str, list[int]] = Field(default_factory=dict)
    on_unsupported_covariates: Literal["fallback", "error"] = "fallback"
    fourier: bool = False
    level_shift: bool = False

    @property
    def dynamic_covariates(self) -> list[str]:
        """Time-varying numeric covariates (`exog` + `future_covariates` + `past_covariates`),
        order-preserving and deduplicated."""
        return list(dict.fromkeys([*self.exog, *self.future_covariates, *self.past_covariates]))

    @property
    def known_future_covariates(self) -> list[str]:
        """Time-varying covariates known across the forecast horizon (`exog` + `future_covariates`),
        order-preserving and deduplicated."""
        return list(dict.fromkeys([*self.exog, *self.future_covariates]))

    @property
    def all_covariates(self) -> list[str]:
        """Every declared covariate column (`dynamic_covariates` + `static_covariates`),
        order-preserving and deduplicated."""
        return list(dict.fromkeys([*self.dynamic_covariates, *self.static_covariates]))

    @model_validator(mode="after")
    def _check_exog_lags(self) -> FeaturesConfig:
        """Reject a bad covariate declaration or ``exog_lags`` at load, not at the first cell.

        Every failure here is one a reader can fix by looking at their own config: overlapping
        covariate tiers, a column that was never declared, a lag that cannot be built, or a
        generated name that would quietly overwrite a column they asked for. Raising at load turns
        all of them into one message before a single worker starts.
        """
        future_set = set(self.future_covariates)
        past_set = set(self.past_covariates)
        if overlap_fp := sorted(future_set & past_set):
            raise ValueError(
                f"features.future_covariates and features.past_covariates overlap on "
                f"{overlap_fp} — a covariate is either known in the future or observed only in "
                f"history, not both"
            )
        declared = set(self.dynamic_covariates)
        static_set = set(self.static_covariates)
        if overlap_sd := sorted(static_set & declared):
            raise ValueError(
                f"features.static_covariates overlaps with dynamic covariates on {overlap_sd} — "
                f"a series-constant attribute cannot also be declared as a time-varying covariate"
            )
        all_declared = declared | static_set
        for name, lags in self.exog_lags.items():
            if name not in declared:
                raise ValueError(
                    f"features.exog_lags names '{name}', which is not in features.exog / "
                    f"future_covariates / past_covariates "
                    f"{sorted(declared)} — a lag can only be built from a declared covariate"
                )
            if not lags:
                raise ValueError(f"features.exog_lags['{name}'] is empty; drop the key instead")
            if len(set(lags)) != len(lags):
                raise ValueError(f"features.exog_lags['{name}'] repeats a lag: {lags}")
            for lag in lags:
                if lag <= 0:
                    raise ValueError(
                        f"features.exog_lags['{name}'] must be positive, got {lag} — "
                        f"a zero lag is the column itself and a negative one reads the future"
                    )
                if (built := f"{name}_lag_{lag}") in all_declared:
                    raise ValueError(
                        f"features.exog_lags would build '{built}', which is already a "
                        f"declared exog column; rename one of them"
                    )
        return self


class BacktestConfig(BaseModel):
    """Time-series cross-validation. Off by default (cheapest first run).

    Every field here is now honoured. The five that arrived ahead of the code that reads them —
    ``short_series``, ``min_folds``, ``min_train_floor``, ``gap`` and ``window`` — were declared in
    one commit because ``run_id`` is a digest of the whole config: adding a field moves every
    identity ever recorded, so landing them together cost one identity break instead of five.
    ``test_declared_ahead_fields.py`` holds the other half of that bargain: each field reaches
    the digest.

    ``short_series`` decides what happens to a series too short for the requested fold grid, and
    every branch of it is a different trade rather than a different amount of the same thing.
    ``adapt`` (the default) holds the geometry exactly and drops the **oldest** folds, so a short
    series is scored on the most recent window it can reach. ``overlap`` holds the fold *count* and
    shrinks the step to buy it, which means validation windows share observations and the per-fold
    scores stop being independent evidence. ``shrink_train`` holds the count and the step and lowers
    the training requirement instead, never past ``min_train_floor`` — which is why that field is
    required with this mode rather than optional. ``skip`` leaves any series short of the full grid
    unscored, so every series on the leaderboard was measured identically. ``error`` refuses the
    run outright; it is checked before any cell runs and never raises inside one, because a scoring
    shortfall must never cost a forecast. ``min_folds`` is the give-up floor the first three
    respect: a series that cannot reach it is left unscored rather than weakly scored.

    ``gap`` and ``window`` are also honoured. ``gap`` is an **embargo**: training stops ``gap``
    observations before the validation window starts, so ``train_end + gap == val_start`` and the
    fold measures a forecast issued with a reporting lag. It moves the training end, never the
    validation window — the folds of a ``gap=14`` run cover the same dates as the folds of a
    ``gap=0`` run, so the two are comparable — and it costs history, so a series may achieve fewer
    folds under it. ``window`` is the ``sliding`` scheme's training width, defaulting to
    ``min_train``, which frees ``min_train`` to mean only the feasibility floor. Both are mirrored
    into the BigQuery-native fold SQL, so the two engines score the same windows.

    ``scheme`` is the one that is fully honoured, and it decides *what a fold's score is a score
    of*. ``expanding`` and ``sliding`` refit at every origin, so a score is about a freshly-trained
    model. ``expanding_frozen`` fits once and then feeds the model the observations that arrive
    between origins with its parameters held fixed, which measures what refitting less often costs.
    ``expanding_stale`` fits once and never tells the model what happened next, which measures how
    fast it decays untouched — the only one of the three every model can answer identically, and so
    the one where a cross-model leaderboard is comparing like with like. Widening this Literal moves
    no existing ``run_id`` — the digest hashes dumped values, not the schema.

    ``control_arm`` is what lets a refit scheme answer the frozen schemes' question without becoming
    one. Both frozen schemes score a blind arm — one fit, walked forward untouched — beside their
    primary arm, and ``expanding`` and ``sliding`` did not, so the run most people actually make
    could not say what refitting was buying it. The arm costs one extra fit per *cell*, not per
    fold, plus a forecast, which is why it is affordable on the default path. On ``expanding_stale``
    it is refused rather than ignored: that scheme's primary arm already is the blind arm.
    See `backtest.BacktestOutcome` for what the two arms produce.

    One asymmetry worth knowing: ``short_series`` is a **Python-path** policy. The BigQuery-native
    models count their folds back from one global ``MAX(ds)`` rather than from each series' own last
    observation, so they have no per-series grid to adapt; a short series there simply contributes
    fewer scored rows. ``error`` is the exception, because it is checked against the panel before
    either engine starts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    scheme: Literal["expanding", "sliding", "expanding_frozen", "expanding_stale"] = "expanding"
    n_folds: int = Field(default=3, ge=1)
    horizon: int = Field(default=28, gt=0)
    step: int = Field(default=28, gt=0)
    min_train: int = Field(default=180, gt=0)
    decision_metric: DecisionMetric = "wape"

    # What a series too short for the full fold grid gets. Resolved by `backtest.resolve_geometry`;
    # see the class docstring for what each branch trades away.
    short_series: ShortSeriesPolicy = "adapt"
    # The give-up floor: fewer achievable folds than this and the series is left unscored rather
    # than scored on evidence too thin to rank it. The default of 1 is today's behaviour exactly.
    min_folds: int = Field(default=1, ge=1)
    # The hard training-length minimum `short_series="shrink_train"` may not cross. Required with
    # that mode (and only meaningful there) — without it, shrinking has no stopping rule.
    min_train_floor: int | None = Field(default=None, gt=0)

    # The embargo: observations discarded between train_end and val_start, for a forecast issued
    # with a known reporting lag. `backtest.make_folds` and `engines.bigquery_sql.fold_plan`.
    gap: int = Field(default=0, ge=0)
    # A fixed training width for `sliding`, decoupled from `min_train`'s role as a data floor.
    # Resolved by `backtest.training_width`, which both engines call.
    window: int | None = Field(default=None, gt=0)

    # Also score a blind arm — one fit, never refreshed — alongside the primary one, on the schemes
    # that refit. The frozen schemes run it regardless; this is what lets the *default* scheme
    # answer "what would never refitting have cost me?" without becoming a different measurement.
    # `backtest._walk_folds`.
    control_arm: bool = False

    @model_validator(mode="after")
    def _check(self) -> BacktestConfig:
        if self.min_folds > self.n_folds:
            raise ValueError(
                f"min_folds={self.min_folds} exceeds n_folds={self.n_folds}, so no series could "
                "ever clear the floor and nothing would be scored"
            )
        if self.short_series == "shrink_train" and self.min_train_floor is None:
            raise ValueError(
                "short_series='shrink_train' needs min_train_floor: shrinking the training "
                "requirement without a floor has no stopping rule and would fit on almost nothing"
            )
        if self.short_series != "shrink_train" and self.min_train_floor is not None:
            raise ValueError(
                f"min_train_floor is only read by short_series='shrink_train', not "
                f"'{self.short_series}'; drop it or switch the policy"
            )
        if self.control_arm and self.scheme == "expanding_stale":
            # Refused rather than ignored. Under this scheme the primary arm already *is* the blind
            # arm, so honouring the flag would write a `yhat_stale` column comparing a thing to
            # itself and a `staleness_gap` of zero by construction — a number that reads like a
            # measurement and is an artefact. Silently dropping it would be worse still: the run
            # would look like it had answered the question the flag asks.
            raise ValueError(
                "backtest.control_arm=true with scheme='expanding_stale' has nothing to compare: "
                "that scheme's primary arm is already the blind, never-refreshed model, so the "
                "control arm would be the same model twice. Drop the flag, or set it on "
                "'expanding' or 'sliding' to measure what refitting is buying you there."
            )
        return self


class OutputConfig(BaseModel):
    """What the numbers we ship *mean* — the functional behind ``yhat``.

    Its own section rather than a field on `BacktestConfig`, even though both of its rules involve
    the backtest, because a reader asking "how do I control the point forecast?" will not look
    under `backtest`, and the docs are a product surface here.

    Every model emits one number per future date, so something decides what that number is. For a
    long time this project decided by accident: ten of sixteen models built their band from
    residual quantiles, the frame assembler took the 0.5 quantile as ``yhat``, and the shipped
    forecast was silently ``prediction + median(in-sample residual)`` — un-named, un-configurable,
    and applied to some models and not others. Measurement said the correction was worth keeping
    (it moved fleet WAPE by 5.7%), so it is the default. This field is what turns it from an
    opinion into a default: the alternative is now sayable.

    ``yhat_raw`` is written alongside ``yhat`` whatever this is set to, so the choice is never
    destructive and the two arms can always be compared after the fact. See `calibration.py`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # None means "derive from backtest.decision_metric" — resolved in `RunConfig._normalize`, so
    # the serialized config (and therefore the run_id) always carries the concrete arm rather than
    # a placeholder whose meaning would depend on the code that read it.
    #
    #   raw    — the model's own output, untouched.
    #   median — plus the median residual. Minimises absolute error; the default for the
    #            absolute-error metrics, which is most of them.
    #   mean   — plus the mean residual. Minimises squared error, and drives `bias` to zero by
    #            construction. **Requires a backtest**: the mean shift is estimated from
    #            out-of-fold residuals and there is no in-sample equivalent for a model that
    #            builds its band from quantiles.
    #   auto   — decide per series+model from that cell's own held-out folds, defaulting to the
    #            corrected arm and dropping to `raw` only on evidence. **Requires a backtest** —
    #            there is nothing to decide from otherwise. Unlike the other three this stays
    #            unresolved in the serialized config, because the resolution is per cell; the
    #            run_id records that selection was asked for, and `forecast_metadata` records what
    #            each cell chose. See `calibration.select_arm`.
    point_forecast: Literal["raw", "median", "mean", "auto"] | None = None


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


ReconciliationMethod = Literal[
    "bottom_up",
    "top_down",
    "middle_out",
    "ols",
    "wls_struct",
    "wls_var",
    "mint_shrink",
]
RECONCILIATION_METHODS = frozenset(
    {
        "bottom_up",
        "top_down",
        "middle_out",
        "ols",
        "wls_struct",
        "wls_var",
        "mint_shrink",
    }
)


class HierarchyConfig(BaseModel):
    """Hierarchical and grouped time-series aggregation and coherent forecast reconciliation.

    Off by default (``enabled: false``). When enabled, ``levels`` defines the cross-sectional
    aggregation hierarchy from top to bottom (for example ``[["region"], ["region", "category"]]``),
    with the total aggregate (``"__total__"``) prepended automatically and the bottom-level series
    (``data.ts_id_col``) at the leaves.

    Implements the Hyndman & Athanasopoulos (FPP3 Ch. 11) reconciliation projection
    ``y_tilde = S @ G @ y_hat`` across seven methods:

    * ``bottom_up`` — sum bottom-level forecasts upward via the summing matrix ``S``.
    * ``top_down`` — disaggregate the top-level forecast downward by average historical proportions
      (FPP3 §11.2).
    * ``middle_out`` — anchor on ``middle_level`` (sum upward above it, disaggregate downward below
      it by historical proportions within each middle-level node).
    * ``ols`` — ordinary least squares MinT reconciliation (``W_h = I``).
    * ``wls_struct`` — structural scaling WLS (``W_h = diag(S @ 1)``), requiring only the hierarchy
      structure ``S``.
    * ``wls_var`` — variance scaling WLS (``W_h = diag(W_hat_1)``) weighted by inverse residual
      variances.
    * ``mint_shrink`` — Wickramasuriya et al. (2019) Minimum Trace with Schäfer-Strimmer shrinkage
      covariance of residuals, positive-definite even when ``n_series >> T_obs``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    levels: list[list[str]] = Field(default_factory=list)
    reconciliation_methods: list[ReconciliationMethod] = [
        "bottom_up",
        "wls_struct",
        "mint_shrink",
    ]
    middle_level: list[str] | None = None

    @model_validator(mode="after")
    def _check_hierarchy(self) -> HierarchyConfig:
        if len(set(self.reconciliation_methods)) != len(self.reconciliation_methods):
            raise ValueError(
                f"hierarchy.reconciliation_methods contains duplicates: "
                f"{self.reconciliation_methods}"
            )
        for idx, level in enumerate(self.levels):
            if not level:
                raise ValueError(f"hierarchy.levels[{idx}] is empty; each level must name columns")
            if len(set(level)) != len(level):
                raise ValueError(f"hierarchy.levels[{idx}] repeats a column: {level}")
        if self.middle_level is not None and self.levels and self.middle_level not in self.levels:
            raise ValueError(
                f"hierarchy.middle_level {self.middle_level} must be one of hierarchy.levels "
                f"{self.levels}"
            )
        if self.enabled:
            if not self.levels:
                raise ValueError(
                    "hierarchy.enabled=true requires at least one aggregation level in "
                    "hierarchy.levels (e.g. [['region'], ['region', 'category']])"
                )
            if not self.reconciliation_methods:
                raise ValueError(
                    "hierarchy.enabled=true requires at least one method in "
                    "hierarchy.reconciliation_methods"
                )
        return self


class FamilyCompute(BaseModel):
    """A sparse per-family compute override, layered over the flat `ComputeConfig` defaults.

    Every field is optional: an unset field inherits the run-level default (``python_runtime``,
    Spark ``serverless``, CPU, the flat ``gpu_type``), so a config sets only what a family needs to
    differ on. There is no ``native`` family here — native models always run in BigQuery. See
    `RunConfig.resolve_family_compute` for how these layer onto the defaults; the block is inert
    until the DAG orchestrator consumes it.

    Hardware constraints (validated): only the ``deep_learning`` family may request a GPU (enforced
    where the family key is known, in `ComputeConfig`); Dataproc Serverless offers **L4 only** (no
    T4/A100 — use ``spark_mode="cluster"`` or ``runtime="ray"``/``"vertex"``/``"gce"``); Spark-only
    fields (``spark_mode``/``spark_cluster_name``) are rejected on ``runtime in ("ray", "vertex",
    "gce")``; VM fields (``machine_type``/``workers``) are rejected on ``runtime in ("spark",
    "ray")``; and ``runtime="gce"`` is strictly single-VM (``workers`` cannot exceed 1).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: Runtime | None = None
    spark_mode: SparkMode | None = None
    # Reuse an existing Dataproc cluster by name (requires spark_mode="cluster"). None = ephemeral
    # per-run cluster (create → submit → delete), mirroring the Ray cluster lifecycle.
    spark_cluster_name: str | None = None
    gke_mode: GkeMode | None = None
    gke_cluster_name: str | None = None
    ray_mode: RayMode | None = None
    automl_mode: AutomlMode | None = None
    hardware: Hardware | None = None
    gpu_type: GpuType | None = None
    accelerator_count: int | None = Field(default=None, gt=0)
    machine_type: str | None = None
    workers: int | None = Field(default=None, gt=0)
    min_workers: int | None = Field(default=None, gt=0)
    max_workers: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_vertex_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        legacy_machine = out.pop("vertex_machine_type", None)
        legacy_workers = out.pop("vertex_workers", None)
        if legacy_machine is not None:
            out.setdefault("machine_type", legacy_machine)
        if legacy_workers is not None:
            out.setdefault("workers", legacy_workers)
        return out

    @property
    def vertex_machine_type(self) -> str | None:
        return self.machine_type

    @property
    def vertex_workers(self) -> int | None:
        return self.workers

    @model_validator(mode="after")
    def _check(self) -> FamilyCompute:
        if self.runtime in ("ray", "vertex", "gce", "gke", "vertex_automl") and (
            self.spark_mode is not None or self.spark_cluster_name is not None
        ):
            raise ValueError("spark_mode/spark_cluster_name are only valid when runtime is 'spark'")
        if self.runtime in ("spark", "ray", "vertex", "gce", "vertex_automl") and (
            self.gke_mode is not None
        ):
            raise ValueError("gke_mode is only valid when runtime is 'gke'")
        if self.runtime in ("spark", "vertex", "gce", "gke", "vertex_automl") and (
            self.ray_mode is not None
        ):
            raise ValueError("ray_mode is only valid when runtime is 'ray'")
        if self.runtime in ("spark", "ray", "vertex", "gce", "gke") and (
            self.automl_mode is not None
        ):
            raise ValueError("automl_mode is only valid when runtime is 'vertex_automl'")
        if self.gke_cluster_name is not None and (
            self.runtime in ("spark", "vertex", "gce", "vertex_automl")
            or (self.runtime == "ray" and self.ray_mode == "vertex")
        ):
            raise ValueError(
                "gke_cluster_name is only valid when runtime is 'gke' or ray_mode is 'gke'"
            )
        if self.min_workers is not None and self.max_workers is not None:
            if self.min_workers > self.max_workers:
                raise ValueError(
                    f"min_workers ({self.min_workers}) cannot exceed "
                    f"max_workers ({self.max_workers})"
                )
        if (
            self.runtime in ("vertex", "gce") or (self.runtime == "gke" and self.gke_mode == "job")
        ) and (self.min_workers is not None or self.max_workers is not None):
            raise ValueError(
                "min_workers/max_workers are only valid on autoscaling runtimes "
                "('ray', 'spark', 'vertex_automl', or 'gke' with gke_mode='ray'); "
                "use 'workers' for fixed-pool 'vertex', 'gce', or 'gke' job mode"
            )
        if self.spark_mode == "serverless" and self.machine_type is not None:
            raise ValueError(
                "Dataproc Serverless does not use VM machine_type; "
                "use spark_mode='cluster' or runtime='vertex'/'gce'/'gke'/'ray'"
            )
        if self.runtime == "gce" and self.workers is not None and self.workers > 1:
            raise ValueError(
                "runtime='gce' supports single-VM execution only (workers=1); "
                "use runtime='vertex' or 'gke' for multi-worker pools"
            )
        if self.spark_cluster_name is not None and self.spark_mode not in (None, "cluster"):
            raise ValueError("spark_cluster_name requires spark_mode='cluster'")
        if self.spark_mode == "serverless" and self.gpu_type is not None and self.gpu_type != "L4":
            raise ValueError(
                f"Dataproc Serverless supports L4 only, not {self.gpu_type}; "
                "use spark_mode='cluster' or runtime='ray'/'vertex'/'gce'/'gke' "
                f"for {self.gpu_type}"
            )
        if self.hardware == "cpu" and (
            self.gpu_type is not None or self.accelerator_count is not None
        ):
            raise ValueError(
                "gpu_type/accelerator_count is set but hardware='cpu'; "
                "drop it or set hardware='gpu'"
            )
        return self


class EnsembleCompute(BaseModel):
    """*When* the ensemble DAG node runs. Not *where* — there is only one where.

    Distinct from `EnsembleConfig`, which selects the ensemble *strategies*. ``mode="barrier"``
    ensembles once after every base model finishes; ``mode="microbatch"`` ensembles each series as
    soon as its upstream base models complete, so the ensemble overlaps the families instead of
    queueing behind the slowest one.

    Both modes are live. The microbatch shape was measured on Airflow in smoke 15 (2026-09-20): the
    ensemble task started in the same second as the four family tasks, ran 2,879 s alongside them,
    and finished 20 s after the last member — which is not a shape barrier mode can produce.

    **This model used to carry ``runtime``, ``spark_mode`` and ``spark_cluster_name`` too.** They
    were accepted by the loader, documented as inert, and read by nothing: the ensemble node is
    hard-wired to the driver (`dag.build_dag_nodes` stamps it ``runtime="bigquery"``, and
    `job_launch.run_ensemble` blends in driver pandas, taking no cluster of its own). Because they
    were still in the config digest, setting one started a new run and changed nothing about it.
    They are gone rather than documented-as-inert so the surface stops offering a choice that does
    not exist. Setting one now fails at load, which beats quietly starting a differently-keyed run
    that behaves identically.

    **No ``run_id`` moved.** Identity hashes ``model_dump()``, defaults included, so dropping three
    defaulted fields would ordinarily re-key every config in existence and stale the whole
    validation ledger to buy a tidier surface. ``registry.ids._REMOVED_DEFAULTS`` pins the three
    keys back into the digest at the values they used to carry, which is legitimate here precisely
    because nothing ever read them: no id on record ever meant anything different from what it
    means now.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: EnsembleMode = "barrier"
    # Seconds between readiness polls in ``mode="microbatch"`` (how often the ensemble drains the
    # series whose base models have all landed). Inert in ``barrier`` mode. Part of the run_id.
    microbatch_interval_s: float = Field(default=60.0, gt=0)


class ProfileConfig(BaseModel):
    """Whether to *measure* what a run costs before sizing it, and with how much headroom.

    Sizing today is a pure cell **count**: ``n_series x n_models x n_folds``, turned into nodes by a
    flat cells-per-slot constant. Nothing in that arithmetic knows that a deep-learning fit and a
    naive mean differ by orders of magnitude, so a fleet is provisioned for the count and not for
    the work. Turning this on replaces the guess with a short instrumented pre-pass
    (``scale_forecasting.profiling``): fit a stratified sample of series, measure what they actually
    consumed, and size each family's slot from the measurement.

    The two margins are deliberately different, and the asymmetry is the point: over-estimating
    time buys extra slots, which costs money, while under-estimating memory OOM-kills the task,
    which costs the run. So memory carries the larger headroom, and the two are applied to
    different tails — ``memory_margin`` to the sample **max** (a slot must hold the worst case that
    lands in it), ``time_margin`` to the **median** (a fleet is sized for typical work, and sizing
    it for the worst case over-provisions every run).

    Part of the ``run_id`` digest — unlike the two fields that are not (its own ``source``, and
    ``compute.capacity``, both of which are resolved or operational rather than authored). It
    changes the resource
    shape rather than the numbers a run produces, so it is arguable — but the config *is* the
    experiment record, and a run whose fleet was sized differently is not the same run for
    performance purposes. Silently varying the shape under a stable id would be the worse trade.

    **A profile is produced by one run and consumed by later ones**, which is why ``mode`` is
    split into a *source* (what evidence to consume) and a *measure* (what evidence to produce)
    below. Every runtime consumes: ``submit.sizing_properties``, ``dataproc_cluster.cluster_sizing``
    and the Ray sizing paths all resolve one through `profiling.source.profile_for_run`.

    What differs per runtime is **when the measurement can be taken**, and only Ray gets a choice.
    ``spark.executor.cores`` and ``spark.task.cpus`` are fixed at submit (Serverless) or at create
    (cluster), before any of our code runs out there, so on Spark the only evidence that can reach
    a sizing decision is *prior* evidence — a harvest, or the shipped baseline. ``ray_engine`` can
    additionally call `profiling.source.resolve_profile` and measure in-run, because a Ray task's
    ``num_cpus``/``num_gpus`` is a request made against a pool that already exists. So ``"auto"``
    and ``"always"`` still differ only on the Ray path; ``"off"`` is the setting that is
    load-bearing everywhere, where it also suppresses the static-arithmetic overlay.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # off    — no measurement; size from static config exactly as before. The escape hatch, and
    #          the setting to reach for if a pre-pass ever misbehaves in production.
    # auto   — measure only when the fan-out is big enough to repay the pre-pass (see min_cells).
    # always — measure unconditionally. What the smokes use, so the path stays exercised cheaply
    #          on runs far too small for `auto` to trigger on.
    mode: ProfileMode = "auto"
    # Series to fit in the pre-pass, spread across length/complexity strata (see
    # `profiling.sampling.select_profile_sample`). The floor is set by wanting more than one point
    # per stratum; the ceiling by the pre-pass being pure overhead that every run pays.
    samples: int = Field(default=8, gt=0)
    # `mode="auto"` profiles only at or above this many cells. Below it the pre-pass costs a
    # meaningful fraction of the run it is sizing, and a small run's mis-sizing is cheap anyway.
    min_cells: int = Field(default=1000, gt=0)
    # Headroom on measured peaks (memory) and medians (time). Must exceed 1.0: a margin of exactly
    # 1.0 sizes a slot at the largest value that was *observed to fit*, with nothing left for the
    # series that was not sampled. Kept in step with `profiling.cost._DEFAULT_MEMORY_MARGIN` /
    # `_DEFAULT_TIME_MARGIN` by a unit test rather than by an import, so this module stays free of
    # pandas — see `test_config_profile_defaults_match_profiling`.
    memory_margin: float = Field(default=1.3, gt=1.0)
    time_margin: float = Field(default=1.2, gt=1.0)
    # What evidence this run *produces*, as distinct from what it consumes. The two are separate
    # questions and the pre-pass framing conflated them; see the class docstring.
    #
    # harvest (default) — record what each cell's fit actually cost (CPU seconds, the worker's
    #           absolute RSS high-water, peak device bytes, the thread cap in force) onto its
    #           `forecast_metadata` row. Every run already performs these fits, so the marginal
    #           cost is three cheap probes per cell and four scalars per row — no sample, no
    #           pre-pass, no extra infrastructure. A completed run is then itself a profile:
    #           `profiling.cost.harvest_profile` aggregates those rows into the same
    #           `ComputeProfile` the translators already consume, which is what makes "size this run
    #           like run X" a query rather than an artifact store.
    # controlled — harvest, and additionally do not pin the native thread pools, so
    #           `effective_cores` measures what a model's threading actually wants instead of
    #           reading back the pin the fleet imposed. This *changes how the run executes* and
    #           will usually make it slower, so it is for a small deliberate sizing run, never
    #           for production work. It is the only way to measure that axis at all.
    # off     — record nothing. Also implied by ``mode="off"``, so one setting turns the whole
    #           profiler off in an incident rather than two.
    measure: ProfileMeasure = "harvest"
    # What evidence this run *consumes*, the mirror of `measure`. Four values, three of them
    # keywords:
    #
    # auto (default) — resolve at **plan** time to the newest profile matching this run's data
    #           signature, falling back to the shipped baseline and then to static arithmetic. The
    #           resolved reference is written into the staged config before the digest is taken, so
    #           a user who never thinks about any of this still gets evidence, and re-running a
    #           staged config still reproduces exactly. Two `auto` runs a week apart may land on
    #           different ids — correct, not a bug: different evidence is a different fleet, and a
    #           different fleet is a different run.
    # <run_id> — consume that run's harvest. The explicit, reproducible form, and what `auto`
    #           resolves itself into.
    # baseline — consume the version shipped with the product and nothing else. The cold-start
    #           answer, and the one axis (`effective_cores`) a user should never have to measure.
    # none    — consume nothing; size from static arithmetic. Distinct from `mode="off"`, which
    #           additionally turns off the arithmetic itself. This one still sizes; it just does
    #           not read anyone's measurements.
    source: str = "auto"

    @field_validator("source")
    @classmethod
    def _source_is_a_keyword_or_a_run_id(cls, value: str) -> str:
        """Reject a source that is neither keyword nor run_id, rather than silently finding nothing.

        A typo here is invisible at runtime — an unresolvable reference degrades to static sizing,
        which is exactly what the run would have done anyway, so the operator would believe they
        pinned a profile and never learn otherwise.
        """
        if value in PROFILE_SOURCE_KEYWORDS or _RUN_ID_RE.match(value):
            return value
        raise ValueError(
            f"compute.profile.source must be one of {PROFILE_SOURCE_KEYWORDS} or a run_id "
            f"(<name-slug>-<12 hex>); got {value!r}"
        )

    @property
    def consumes_evidence(self) -> bool:
        """Should this run try to size itself from measurements? (``mode="off"`` vetoes it.)"""
        return self.mode != "off" and self.source != "none"

    @property
    def needs_source_resolution(self) -> bool:
        """Is the source still a *question* rather than an answer? (i.e. must plan time lock it.)"""
        return self.consumes_evidence and self.source == "auto"

    @property
    def records_measurements(self) -> bool:
        """Should this run write per-cell measurements? (``mode="off"`` vetoes ``measure``.)"""
        return self.mode != "off" and self.measure != "off"

    @property
    def unpins_threads(self) -> bool:
        """Should the fleet leave native thread pools uncapped so `effective_cores` is real?"""
        return self.records_measurements and self.measure == "controlled"


class CapacityServicePolicy(BaseModel):
    """A partial override of one service's shipped retry policy — every field optional.

    Partial on purpose. A user who wants to wait longer for a GPU should be able to say only
    ``{"max_wall_seconds": 7200}`` and inherit the rest; requiring the whole policy would mean
    copying four numbers they have no opinion about and silently freezing them against future
    default changes. Unset fields fall through to `capacity.DEFAULT_POLICIES[service]`.

    Bounds are validated here as well as in `capacity.CapacityPolicy`, so a bad config fails at load
    with a pydantic error naming the field rather than at provisioning time with a ValueError.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 0 disables a bound; see `capacity.CapacityPolicy` for what each one counts.
    max_attempts: int | None = Field(default=None, ge=0)
    max_wall_seconds: float | None = Field(default=None, ge=0)
    max_passes: int | None = Field(default=None, ge=0)
    backoff_seconds: float | None = Field(default=None, ge=0)
    backoff_multiplier: float | None = Field(default=None, ge=1.0)
    backoff_max_seconds: float | None = Field(default=None, ge=0)


class RetryResources(BaseModel):
    """What a **repair** may claim differently from the attempt it repairs.

    A repair re-submits a handful of cells out of a run that was sized for all of them. The fleet
    arithmetic does not know that: it sizes from the config's fan-out, so a forty-cell repair of a
    hundred-thousand-cell family asks for the hundred-thousand-cell fleet and pays for it. This is
    where a run says how wide its repairs should be instead.

    It lives here, under ``compute.capacity``, for the reason the parent's docstring already gives:
    everything under ``capacity`` is excluded from the ``run_id`` digest, and a repair that resized
    itself must stay the *same run*. A sibling of ``compute.max_executors`` would fork the id, which
    would mean the repair wrote its rows under a run nobody was looking at.

    It is a config block rather than only a CLI flag for the same reason ``compute.max_executors``
    is: the unattended repair paths (`airflow_tasks.retry_families`, `main`'s ``--retry``) launch
    from a config and nothing else, so a ceiling reachable only through an argument is a ceiling an
    orchestrated repair can never set. An explicit ``max_executors=`` argument still wins over it
    (`job_launch.submit_retry`) — the operator at the terminal overrides the file.

    **Why only this one knob, and what may join it.** A repair reuses the config its run staged, so
    the only settings it can vary are the ones the *driver* reads at launch: anything the worker
    reads out of the staged config (``bucket_target_cells``, ``max_parallelism``, the model list,
    the data block) is the same bytes for the repair as for the attempt, by construction. Of the
    driver-side knobs, ``max_executors`` is the one the launchers already take as an argument
    (`job_launch.launch_family_job`). Note its reach: it caps a Spark **batch's**
    ``spark.dynamicAllocation.maxExecutors`` and is ignored by the Ray and in-process paths, so a
    Ray family's repair still provisions from the fan-out.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_executors: int | None = Field(default=None, gt=0)


class CapacityConfig(BaseModel):
    """How hard to look for room when a service says it has none — per service (G2).

    "Resources are not available" is a **state**, not an exception: a run walks its candidate
    places, and if none has room it waits and walks them again, until an attempt budget or a clock
    runs out. `scale_forecasting.capacity` implements that; this is where a run tunes it.

    **Not part of the ``run_id`` digest** — excluded in `registry.ids._NOT_IDENTITY`, and there is a
    test that says so. Same rule as ``compute.profile.source``, for the same reason: a run's
    identity is *what was asked for*, and patience is an operational knob. If it moved the digest,
    waiting longer for a GPU would fork your run id and break dedupe-on-read.

    ``enabled: false`` restores the pre-retry behaviour exactly — one pass over the candidates, no
    back-off — which is the escape hatch if a retry loop ever misbehaves in production. It does not
    disable *classification*: a config fault still stops immediately and a quota ceiling is still
    named as one in the ledger, because those were improvements to the diagnosis, not to the
    patience.

    BigQuery has no entry and that omission is deliberate (`capacity.UNMANAGED_SERVICES`): slot
    contention is resolved BigQuery-side and surfaces as latency, not as a create that failed
    somewhere and could be retried elsewhere. There is no candidate list to walk.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # Read the region's quota *before* attempting a create (`scale_forecasting.quota`): drop regions
    # that cannot host the fleet at all, lower a ceiling the allowance will not grant, and report
    # what the ceiling is costing in wall clock. On by default because its failure mode is benign —
    # an unreadable meter changes nothing — and its success mode saves a ~12-minute create attempt.
    # `false` skips the read entirely, for a deployment whose runner SA is not granted
    # ``serviceusage.services.get`` and would rather not log the resulting miss on every launch.
    preflight: bool = True
    # Vertex Ray cluster creation — walks `compute.ray_regions`. The most expensive attempt
    # (~12 min for a GPU provision), so the shipped default is fewest tries and longest wait.
    ray: CapacityServicePolicy = Field(default_factory=CapacityServicePolicy)
    # Vertex AI CustomJob creation — walks `compute.ray_regions` without a Ray head-node bootstrap.
    vertex: CapacityServicePolicy = Field(default_factory=CapacityServicePolicy)
    # Vertex AI AutoML Tabular Workflow PipelineJob / ForecastingTrainingJob + BatchPredictionJob.
    vertex_automl: CapacityServicePolicy = Field(default_factory=CapacityServicePolicy)
    # Compute Engine single-VM creation — walks the zone/region candidates from `compute_fallback`.
    gce: CapacityServicePolicy = Field(default_factory=CapacityServicePolicy)
    # Google Kubernetes Engine cluster / node-pool creation and job dispatch.
    gke: CapacityServicePolicy = Field(default_factory=CapacityServicePolicy)
    # Dataproc cluster creation — walks the zone/region candidates from `compute_fallback`.
    dataproc_cluster: CapacityServicePolicy = Field(default_factory=CapacityServicePolicy)
    # Dataproc Serverless batch submission — region only, and rejections come back in seconds.
    dataproc_serverless: CapacityServicePolicy = Field(default_factory=CapacityServicePolicy)
    # How wide a *repair* runs (see `RetryResources`). Not a capacity *policy* — it does not tune
    # the candidate walk — but it belongs to the same digest-excluded operational surface, and
    # giving it its own top-level block would mean a second exclusion entry saying the same thing.
    retry: RetryResources = Field(default_factory=RetryResources)

    def policy_for(self, service: str) -> CapacityPolicy:
        """Resolve this config into the runtime policy for ``service`` (pure).

        The shipped per-service default with any authored field laid over it, plus the
        ``enabled: false`` collapse to a single pass. Raises `KeyError` for a service with no
        candidate walk (BigQuery), because asking for its retry policy is a programming error rather
        than a configuration one.
        """
        base = DEFAULT_POLICIES[service]
        override: CapacityServicePolicy = getattr(self, service)
        fields = {
            key: value
            for key, value in override.model_dump().items()
            if value is not None and key != "max_passes"
        }
        max_passes = 1 if not self.enabled else (override.max_passes or base.max_passes)
        return replace(base, max_passes=max_passes, **fields)


_L4_SINGLE_GPU_MACHINES: frozenset[str] = frozenset(
    {
        "g2-standard-4",
        "g2-standard-8",
        "g2-standard-12",
        "g2-standard-16",
        "g2-standard-32",
    }
)
_L4_MULTI_GPU_MACHINES: dict[int, str] = {
    2: "g2-standard-24",
    4: "g2-standard-48",
    8: "g2-standard-96",
}
_A100_MACHINES: dict[int, str] = {
    1: "a2-highgpu-1g",
    2: "a2-highgpu-2g",
    4: "a2-highgpu-4g",
    8: "a2-highgpu-8g",
    16: "a2-megagpu-16g",
}
_A100_80GB_MACHINES: dict[int, str] = {
    1: "a2-ultragpu-1g",
    2: "a2-ultragpu-2g",
    4: "a2-ultragpu-4g",
    8: "a2-ultragpu-8g",
}


def _default_gpu_machine_type(gpu_type: str | None, accelerator_count: int) -> str:
    eff_gpu = gpu_type or "T4"
    if eff_gpu == "T4":
        if accelerator_count not in (1, 2, 4):
            raise ValueError(
                f"accelerator_count for T4 must be 1, 2, or 4 (got {accelerator_count})"
            )
        return "n1-standard-16" if accelerator_count == 4 else "n1-standard-8"
    if eff_gpu == "L4":
        if accelerator_count == 1:
            return "g2-standard-8"
        if accelerator_count in _L4_MULTI_GPU_MACHINES:
            return _L4_MULTI_GPU_MACHINES[accelerator_count]
        raise ValueError(
            f"accelerator_count for L4 must be 1, 2, 4, or 8 (got {accelerator_count})"
        )
    if eff_gpu == "A100":
        if accelerator_count in _A100_MACHINES:
            return _A100_MACHINES[accelerator_count]
        raise ValueError(
            f"accelerator_count for A100 must be 1, 2, 4, 8, or 16 (got {accelerator_count})"
        )
    if eff_gpu == "A100_80GB":
        if accelerator_count in _A100_80GB_MACHINES:
            return _A100_80GB_MACHINES[accelerator_count]
        raise ValueError(
            f"accelerator_count for A100_80GB must be 1, 2, 4, or 8 (got {accelerator_count})"
        )
    raise ValueError(f"Unsupported gpu_type {eff_gpu!r}")


def resolve_vm_machine_type(
    hardware: str,
    gpu_type: str | None = None,
    default_machine_type: str = "auto",
    override_machine_type: str | None = None,
    *,
    accelerator_count: int = 1,
) -> str:
    """Resolve and validate the GCE/Vertex VM machine type for a family's hardware (pure).

    Enforces GCP's three structural GPU-to-VM compatibility patterns at config load time:
    * ``T4`` (``NVIDIA_TESLA_T4``): attaches to ``n1-*`` in counts of ``{1, 2, 4}`` (1-2 GPUs cap
      at 48 vCPUs; default ``n1-standard-8`` for 1-2 GPUs, ``n1-standard-16`` for 4 GPUs).
    * ``L4`` (``NVIDIA_L4``): bundled into ``g2-standard-*`` in counts of ``{1, 2, 4, 8}`` (1 GPU
      allows ``g2-standard-{4,8,12,16,32}``, default ``g2-standard-8``; 2/4/8 GPUs require fixed
      shapes ``g2-standard-{24,48,96}``).
    * ``A100`` (``NVIDIA_TESLA_A100``, 40 GiB) & ``A100_80GB`` (``NVIDIA_A100_80GB``, 80 GiB):
      strict 1:1 mapping from ``accelerator_count`` to ``a2-highgpu-{1,2,4,8}g`` /
      ``a2-megagpu-16g`` or ``a2-ultragpu-{1,2,4,8}g``.
    """
    from .resources.catalog import machine_cores

    if hardware == "gpu":
        auto_gpu = _default_gpu_machine_type(gpu_type, accelerator_count)
        if override_machine_type is not None and override_machine_type != "auto":
            chosen = override_machine_type
        elif default_machine_type in ("auto", "n2-standard-8"):
            chosen = auto_gpu
        else:
            chosen = default_machine_type

        eff_gpu = gpu_type or "T4"
        if eff_gpu == "T4":
            if not chosen.startswith("n1-"):
                raise ValueError(
                    f"Vertex AI / GCE T4 GPUs require an n1-* machine type (got {chosen!r})"
                )
            cores = machine_cores(chosen) or 0
            if accelerator_count in (1, 2) and cores > 48:
                raise ValueError(
                    f"T4 with accelerator_count={accelerator_count} allows at most 48 vCPUs "
                    f"on n1-* (got {chosen!r} with {cores} vCPUs)"
                )
        elif eff_gpu == "L4":
            if not chosen.startswith("g2-"):
                raise ValueError(
                    f"Vertex AI / GCE L4 GPUs require a g2-* machine type (got {chosen!r})"
                )
            if accelerator_count == 1 and chosen not in _L4_SINGLE_GPU_MACHINES:
                raise ValueError(
                    f"1x L4 GPU requires one of {sorted(_L4_SINGLE_GPU_MACHINES)} (got {chosen!r})"
                )
            if accelerator_count > 1 and chosen != _L4_MULTI_GPU_MACHINES.get(accelerator_count):
                expected = _L4_MULTI_GPU_MACHINES[accelerator_count]
                raise ValueError(
                    f"{accelerator_count}x L4 GPUs require machine_type={expected!r} "
                    f"(got {chosen!r})"
                )
        elif eff_gpu == "A100":
            expected = _A100_MACHINES[accelerator_count]
            if chosen != expected:
                raise ValueError(
                    f"{accelerator_count}x A100 GPUs require machine_type={expected!r} "
                    f"(got {chosen!r})"
                )
        elif eff_gpu == "A100_80GB":
            expected = _A100_80GB_MACHINES[accelerator_count]
            if chosen != expected:
                raise ValueError(
                    f"{accelerator_count}x A100_80GB GPUs require machine_type={expected!r} "
                    f"(got {chosen!r})"
                )
        return chosen

    if override_machine_type is not None and override_machine_type != "auto":
        chosen = override_machine_type
    elif default_machine_type == "auto" or default_machine_type.startswith(("g2-", "a2-")):
        chosen = "n2-standard-8"
    else:
        chosen = default_machine_type

    if chosen.startswith(("g2-", "a2-")):
        raise ValueError(
            f"CPU workloads cannot use GPU-attached machine family {chosen!r}; "
            "set hardware='gpu' or choose a CPU machine family (e.g. 'n2-standard-8')"
        )
    return chosen


def resolve_vertex_machine_type(
    hardware: str,
    gpu_type: str | None,
    cpu_machine_type: str = "auto",
    gpu_machine_type: str = "auto",
    override_machine_type: str | None = None,
    *,
    accelerator_count: int = 1,
) -> str:
    """Backward-compatible wrapper around `resolve_vm_machine_type`."""
    default_mt = (
        gpu_machine_type if hardware == "gpu" and gpu_machine_type != "auto" else cpu_machine_type
    )
    return resolve_vm_machine_type(
        hardware,
        gpu_type,
        default_mt,
        override_machine_type,
        accelerator_count=accelerator_count,
    )


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
    # The operator's *infrastructure* ceiling on the Spark fleet: the most executors a batch may
    # scale to (spark.dynamicAllocation.maxExecutors), or the most workers a cluster may hold.
    # None (default) = no operator ceiling, and the fleet arithmetic sizes to the fan-out alone.
    #
    # This exists because the derived sizing has no idea what the project can actually be given.
    # It answers "how wide would this run like to be", which at 100k series is hundreds of
    # executors; a regional CPU quota answers "how wide may it be", and the two had no way to meet.
    # Live, that gap is not a slow run but a dead one — Dataproc rejects the batch outright
    # ("Insufficient 'CPUS' quota. Requested 380.0, available 200.0") before any work starts.
    #
    # It lives on the config rather than only on `submit`/`--max-executors` because every job the
    # DAG launches is launched *from a config*: a ceiling reachable only through a CLI flag is a
    # ceiling an orchestrated run can never set. Budget for concurrency when picking one — the
    # families of a single run submit at the same time, so a two-family run at N executors of C
    # cores wants roughly 2 x N x C cores of headroom, plus a driver each.
    max_executors: int | None = Field(default=None, gt=0)
    # GCE machine family for a **Dataproc cluster's** master and CPU workers (`"auto"` = n1, the
    # shipped default). Only families `resources.catalog._MEMORY_PER_CORE_GIB` can price are
    # offered, so the profiler's executor sizing stays honest for whatever is picked. Deliberately
    # narrow in two directions: the *size* is not a knob (the profiler derives cores from the
    # fan-out), and it does not reach the GPU worker, whose machine type the accelerator dictates (a
    # T4 rides an n1, an L4 is bundled into a g2). No-op on Serverless, which has no machine concept
    # at all — its shape is executor cores/memory properties. See
    # dataproc_cluster.worker_machine_type.
    machine_family: Literal["auto", "n1", "n2", "n2d", "e2", "c2"] = "auto"
    spark_deps: Literal["packed_venv", "container"] = "packed_venv"
    # Persist each fitted model as a GCS artifact (ObjectRef in forecast_metadata.model_artifact,
    # model-artifact lineage). Off by default: at 100k×N cells the object count + write cost is
    # material, so a run opts in explicitly (demos do; the hero scale run need not). See
    # BaseModel.serialize.
    persist_models: bool = False
    use_gpu: bool = False
    gpu_type: GpuType = "T4"
    # "auto" = profile-driven calibration, or a fixed fraction in (0, 1].
    gpu_fraction: Literal["auto"] | float = "auto"
    budget_usd: float = Field(default=50.0, ge=0.0)

    # --- Unified VM & Worker Pool Sizing --------------------------------------
    # Universal compute knobs across runtimes:
    # * `machine_type = "auto"` resolves from `(hardware, gpu_type, accelerator_count)`:
    #     - CPU -> `n2-standard-8`
    #     - T4  -> `n1-standard-8` (1-2 GPUs) or `n1-standard-16` (4 GPUs)
    #     - L4  -> `g2-standard-8` (1 GPU) or `g2-standard-{24,48,96}` (2/4/8 GPUs)
    #     - A100 -> `a2-highgpu-{1,2,4,8}g` / `a2-megagpu-16g`
    #     - A100_80GB -> `a2-ultragpu-{1,2,4,8}g`
    # * `workers` sets fixed worker VM count on `runtime == "vertex"` (or caps workers/executors
    #   when set on `spark`/`ray`); `runtime == "gce"` is strictly single-VM (`workers == 1`).
    # * `min_workers` / `max_workers` set autoscaling bounds on `ray` and `spark`.
    machine_type: str = "auto"
    workers: int = Field(default=1, gt=0)
    min_workers: int | None = Field(default=None, gt=0)
    max_workers: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_vertex_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            v_mt = out.pop("vertex_machine_type", None)
            v_gpu_mt = out.pop("vertex_gpu_machine_type", None)
            if "machine_type" not in out:
                if v_gpu_mt is not None and v_gpu_mt != "auto":
                    out["machine_type"] = v_gpu_mt
                elif v_mt is not None:
                    out["machine_type"] = v_mt
            if "vertex_workers" in out:
                vw = out.pop("vertex_workers")
                if "workers" not in out and vw is not None:
                    out["workers"] = vw
            return out
        return data

    @property
    def vertex_machine_type(self) -> str:
        return "n2-standard-8" if self.machine_type == "auto" else self.machine_type

    @property
    def vertex_gpu_machine_type(self) -> str:
        return self.machine_type if self.machine_type.startswith(("n1-", "g2-", "a2-")) else "auto"

    @property
    def vertex_workers(self) -> int:
        return self.workers

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
    # Where the Ray cluster executes when runtime == "ray":
    #   "vertex" (default) : Vertex AI Managed Ray cluster.
    #   "gke"              : Ray on Google Kubernetes Engine (KubeRay / native K8s Ray cluster).
    ray_mode: RayMode = "vertex"
    # --- Google Kubernetes Engine (GKE) ----------------------------------------
    # Execution mode when runtime == "gke":
    #   "job" (default) : Kubernetes batch/v1 Indexed Job (`completionMode: Indexed`) running
    #                     `vertex_engine.py` (`JOB_COMPLETION_INDEX` -> WorkerTopology).
    #   "ray"           : Ray on GKE (`ray_engine.py` over KubeRay / native K8s Ray pods).
    gke_mode: GkeMode = "job"
    # Target an existing GKE cluster by name (or via `SF_GKE_CLUSTER`). None (default) = ephemeral
    # per-run GKE Standard cluster when neither `gke_cluster_name` nor `SF_GKE_CLUSTER` is set.
    gke_cluster_name: str | None = None
    # Kubernetes namespace for GKE Indexed Jobs and Ray pods.
    gke_namespace: str = "default"
    # --- Vertex AI AutoML / Tabular Workflows ----------------------------------
    # Execution mode when runtime == "vertex_automl":
    #   "tabular_workflow" (default) : Vertex AI Pipelines (KFP v2) Tabular Workflow for Forecasting
    #                                  with worker pool overrides, stage_1 tuning artifact export &
    #                                  warm-start reuse, and BatchPredictionJob explanations.
    #   "training_job"               : Managed Vertex AI *ForecastingTrainingJob +
    #                                  BatchPredictionJob.
    automl_mode: AutomlMode = "tabular_workflow"
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
    # GPUs per GPU worker node. Validated against gpu_type below.
    accelerator_count: int = Field(default=1, gt=0)
    # Pool sizing: how many cells one worker slot should chew through before we add another node
    # (amortizes per-node warm-up), plus a hard ceiling so a huge fan-out can't request an unbounded
    # cluster. n_gpu_nodes/n_cpu_nodes are derived, then clamped to [1, ray_max_nodes] — and under
    # autoscaling that derived count is also what sets each pool's ceiling (see the max_nodes
    # fields), so ray_max_nodes is the guardrail, not the operating point.
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
    # Per-pool autoscaling ceiling — an explicit *pin*. None (default) means the ceiling is DERIVED
    # from the run's own fan-out (the pool's derived node count, floored at 2 and capped by the hard
    # ceiling ray_max_nodes), so a small run scales to a small pool and a large one is not stuck at
    # a constant. Set it to pin a pool instead: e.g. cap the expensive GPU pool while leaving the
    # cheap CPU pool free to derive. A pin below the pool's min_nodes is rejected at plan time.
    # Inert when ray_autoscale is False (both pools are then fixed at their derived node_count).
    ray_cpu_max_nodes: int | None = Field(default=None, gt=0)
    ray_gpu_max_nodes: int | None = Field(default=None, gt=0)
    # Auto-fraction calibration (gpu_fraction == "auto"): how many series to profile and the
    # headroom multiplier applied to measured peak GPU memory before dividing by device memory.
    gpu_calibration_samples: int = Field(default=3, gt=0)
    gpu_safety_margin: float = Field(default=1.3, gt=1.0)
    # Measured compute profiling — the general form of the two knobs above. Auto-fraction
    # calibration profiles one axis (GPU bytes) for one model on one runtime; this profiles every
    # axis for every family on all three. The two coexist deliberately: auto-fraction refines
    # on-cluster after creation, because the GPU axis needs a GPU; this one sizes the fleet, which
    # has to happen before the fleet exists. That ordering is the whole constraint — a fleet is
    # fixed at submit (Serverless) or at create (cluster, Ray pool), and the submit host is kept
    # deliberately lean (no model stack, a 2-vCPU Composer worker), so there is nowhere in *this*
    # run to take a measurement that could resize it. Ray is the partial exception: per-task
    # num_cpus/num_gpus is requested in-run, so `ray_engine` really does profile and repack — but
    # within a pool that is already provisioned. Measurements that size a fleet therefore have to
    # come from an earlier run. See `ProfileConfig`.
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    # How hard to look for room when a service says it has none. The second field under `compute`
    # that is NOT part of the run_id digest (see `CapacityConfig`) — patience is an operational
    # knob, not a description of the experiment.
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
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
    # deep_learning/automl) may pick its own runtime + hardware; an unset family inherits the
    # run-level python_runtime / Spark-serverless / CPU defaults (or vertex_automl for automl).
    # Native models are never here — they always run in BigQuery. ``ensemble`` runs the ensemble
    # DAG node on its own runtime with a barrier|microbatch trigger. See
    # RunConfig.resolve_family_compute.
    families: dict[ComputeFamily, FamilyCompute] = Field(default_factory=dict)
    ensemble: EnsembleCompute = Field(default_factory=EnsembleCompute)

    @model_validator(mode="after")
    def _check_families(self) -> ComputeConfig:
        # GPU is supported on deep_learning and automl; statistical/ml are CPU work. The family key
        # is known here (unlike inside FamilyCompute), so this is where that constraint is enforced.
        for fam, fc in self.families.items():
            if fam not in ("deep_learning", "automl") and (
                fc.hardware == "gpu" or fc.gpu_type is not None
            ):
                raise ValueError(
                    f"family '{fam}' cannot use a GPU (hardware='gpu'/gpu_type set); "
                    "only the deep_learning and automl families support GPU"
                )
            if fam == "automl" and fc.runtime is not None and fc.runtime != "vertex_automl":
                raise ValueError(
                    f"family 'automl' models require runtime='vertex_automl' (got {fc.runtime!r})"
                )
            if fam != "automl" and (fc.runtime == "vertex_automl" or fc.automl_mode is not None):
                raise ValueError(
                    f"family '{fam}': runtime='vertex_automl'/automl_mode is only valid for the "
                    "'automl' family"
                )
        return self

    @model_validator(mode="after")
    def _check_gpu_fraction(self) -> ComputeConfig:
        if isinstance(self.gpu_fraction, float) and not (0.0 < self.gpu_fraction <= 1.0):
            raise ValueError("gpu_fraction must be 'auto' or a float in (0, 1]")
        if self.min_workers is not None and self.max_workers is not None:
            if self.min_workers > self.max_workers:
                raise ValueError(
                    f"min_workers ({self.min_workers}) cannot exceed "
                    f"max_workers ({self.max_workers})"
                )
        _default_gpu_machine_type(self.gpu_type, self.accelerator_count)
        if self.use_gpu and self.machine_type not in ("auto", "n2-standard-8"):
            resolve_vm_machine_type(
                "gpu",
                self.gpu_type,
                self.machine_type,
                accelerator_count=self.accelerator_count,
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
    ``gke_mode``/``gke_cluster_name`` apply when ``runtime == "gke"`` (or ``runtime == "ray"`` with
    ``ray_mode == "gke"``); ``ray_mode`` applies when ``runtime == "ray"``; ``automl_mode`` applies
    when ``runtime == "vertex_automl"``; ``gpu_type`` is ``None`` unless ``hardware == "gpu"``.
    Produced by `RunConfig.resolve_family_compute`.
    """

    family: str
    runtime: str
    spark_mode: str | None
    spark_cluster_name: str | None
    hardware: str
    gpu_type: str | None
    machine_type: str | None = None
    workers: int | None = None
    min_workers: int | None = None
    max_workers: int | None = None
    accelerator_count: int = 0
    gke_mode: str | None = None
    gke_cluster_name: str | None = None
    ray_mode: str | None = None
    automl_mode: str | None = None

    @property
    def vertex_machine_type(self) -> str | None:
        return self.machine_type

    @property
    def vertex_workers(self) -> int | None:
        return self.workers


# --- top-level config ----------------------------------------------------------


class RunConfig(BaseModel):
    """A complete, validated, frozen run specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_name: str
    data: DataConfig
    python_runtime: Runtime = "spark"
    models: list[str] = Field(min_length=1)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    hpo: HpoConfig = Field(default_factory=HpoConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    # Per-model hyperparameters, keyed by model name: {"neuralprophet": {"n_lags": 28}}. The
    # declared home for *every* per-model knob, which is what keeps the next identity break small —
    # a new hyperparameter becomes a dict key rather than a schema field, and a dict key only moves
    # the ids of configs that actually set it.
    #
    # Read at all three places a model's params resolve — the cell (`worker._resolve_params`) and
    # both halves of HPO (each trial's objective and the returned winner, in `hpo.tune_model`) — so
    # a study tunes the same model the cell will fit. HPO wins on the keys its search space names;
    # an authored key the search does not touch survives untouched.
    #
    # Unknown model names are accepted *here* on purpose: validating them means importing the model
    # registry from this module, and eager model-stack imports on the submit path have broken a live
    # run before. `dag.check_model_params` does that check instead, from the paths that are about to
    # spend, where the registry is already loaded — that is also where a model gets to refuse a
    # block it cannot honour (`models.base_model.BaseModel.validate_params`).
    model_params: dict[str, dict[str, ModelParam]] = Field(default_factory=dict)

    @field_validator("model_params")
    @classmethod
    def _model_params_survive_json(
        cls, value: dict[str, dict[str, ModelParam]]
    ) -> dict[str, dict[str, ModelParam]]:
        """Reject non-finite floats, which the type alias cannot exclude.

        ``float`` admits NaN and infinity, and ``json.dumps`` emits them as the bare tokens ``NaN``
        and ``Infinity``. Both are invalid JSON, so the digest string stops being something another
        reader can reproduce — and this string is the ``run_id``.
        """
        for model, params in value.items():
            for key, val in params.items():
                bad = [v for v in (val if isinstance(val, list) else [val]) if _is_non_finite(v)]
                if bad:
                    raise ValueError(
                        f"model_params.{model}.{key} contains {bad[0]!r}, which is not JSON. "
                        f"run_id is a digest of the serialized config, so a value that cannot "
                        f"round-trip through JSON cannot be part of one."
                    )
        return value

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

        # 3b. Resolve the point-forecast arm from the decision metric, so `yhat` and the metric it
        #     is judged on cannot disagree. The median minimises absolute error and the mean
        #     minimises squared error; shipping a median point forecast to a user scored on RMSE
        #     is a mismatch nothing used to mention. Resolved here rather than read lazily so the
        #     serialized config carries the concrete arm — the run_id then records what was
        #     actually computed, not an instruction to go and decide later.
        #
        #     `auto` is the exception, and deliberately so: its resolution is per series+model, so
        #     there is no single arm to write down here. It stays in the serialized config, which
        #     means the run_id records "selection was requested" — the right thing to record, since
        #     two runs that both selected per series are the same run even if the cells chose
        #     differently.
        if self.output.point_forecast is None:
            # A backtest earns `auto`: every series has held-out folds, so the arm can be chosen
            # per series+model from what actually scored better rather than assigned fleetwide from
            # the metric. Measured at 3 folds over ten models, fleet RMSE 28.77 against 30.54 for
            # the fleetwide rule — and the fleetwide rule's `mean` was worse there than applying no
            # correction at all (30.05), which is the case that decided this. MAE moved the same
            # way, 18.75 against 19.32. `corrected_arm_for` still resolves an explicit request; it
            # is no longer what an unset field falls back to.
            #
            # Without a backtest there is nothing to select from, and the mean shift does not exist
            # off-fold, so the honest resolution stays the model's own correction.
            resolved = "auto" if self.backtest.enabled else "median"
            object.__setattr__(
                self, "output", self.output.model_copy(update={"point_forecast": resolved})
            )
        elif self.output.point_forecast == "mean" and not self.backtest.enabled:
            raise ValueError(
                "output.point_forecast='mean' requires backtest.enabled: the mean residual shift "
                "is estimated from out-of-fold residuals, and there is no in-sample equivalent. "
                "Use 'median' (the model's own correction) or 'raw' (no correction)."
            )
        elif self.output.point_forecast == "auto" and not self.backtest.enabled:
            raise ValueError(
                "output.point_forecast='auto' requires backtest.enabled: the arm is chosen from "
                "each series' own held-out folds, and without a backtest there are none to choose "
                "from. Use 'median' (the model's own correction) or 'raw' (no correction)."
            )
        elif self.output.point_forecast == "median" and (
            corrected_arm_for(self.backtest.decision_metric) == "mean"
        ):
            _log.warning(
                "output.point_forecast='median' with decision_metric=%r: the median minimises "
                "absolute error while %r penalises squared error, so the shipped point forecast "
                "is not the one the leaderboard rewards. 'mean' is the coherent pair.",
                self.backtest.decision_metric,
                self.backtest.decision_metric,
            )

        # 4. Harden every per-family compute override by resolving it now, so an incoherent
        #    combination the per-block validator can't see (e.g. a T4 GPU inherited onto Dataproc
        #    Serverless) fails at load rather than at submit. Resolution is pure; the DAG
        #    orchestrator reuses the same resolver, so a validated config needs no re-check.
        for fam in self.compute.families:
            self.resolve_family_compute(fam)

        return self

    @model_validator(mode="after")
    def _check_horizon_linkage(self) -> RunConfig:
        """Two horizons live in one config, and nothing used to say when they may differ.

        ``data.horizon`` is how far the shipped forecast reaches. ``backtest.horizon`` is how far
        each fold predicts before it is scored. They are independent fields answering different
        questions, so a config can set them apart on purpose — but almost every config that does
        so did it by editing one and forgetting the other, and being ranked on a horizon you do
        not ship is worth saying out loud.

        **This warns and never raises**, which is the second answer to this question rather than
        the first. The plan called for refusing a run whose folds out-asked a BigQuery-native
        model's trained horizon, because ``ML.FORECAST`` cannot exceed the horizon baked in at
        ``CREATE MODEL`` time. Refusing would have been validating around a bug: the fold model was
        being trained at ``data.horizon`` while its own fold asked for ``gap + backtest.horizon``,
        so an embargo broke a native run even with the two horizons matched exactly. That is fixed
        where it belonged, in `engines.bigquery_sql.trained_horizon` — every model is now created
        for precisely what it will be asked — and with the error impossible by construction, a
        raise here would only forbid configs that work.

        This validator never rewrites either field. The run_id is a digest taken after
        ``_normalize``, so a config that silently repaired itself here would move its own identity
        and land in the registry describing a run nobody asked for.
        """
        if not self.backtest.enabled or self.backtest.horizon == self.data.horizon:
            return self

        _log.warning(
            "backtest.horizon=%d and data.horizon=%d differ: every fold is scored over %d steps "
            "while the forecast this run ships reaches %d, so the leaderboard ranks models on a "
            "horizon the run never delivers. Set the two equal unless the difference is "
            "deliberate.",
            self.backtest.horizon,
            self.data.horizon,
            self.backtest.horizon,
            self.data.horizon,
        )
        return self

    @model_validator(mode="after")
    def _check_decision_metric_is_computable(self) -> RunConfig:
        """Say at plan time when the chosen ``decision_metric`` will read NaN, and why.

        Every metric is NaN-safe by design — an undefined metric is a NaN cell, never an error —
        and that is the right behaviour for one odd series. It is the wrong *discovery* mechanism
        for a whole run: a metric that is undefined for every cell produces an empty leaderboard
        column and no explanation, and the operator's next move is to re-run something expensive
        while they work out why.

        So this reads the metric's own ``needs_*`` declarations against the config and warns on the
        two combinations that are undefined for structural reasons rather than data ones. **It
        warns and never raises**, because both are legal configs that someone may want for the rest
        of the panel — the metric is still computed for every other row, and only the ranking column
        is affected.

        It rewrites nothing. The ``run_id`` is a digest taken after ``_normalize``, so a config that
        repaired itself here would land in the registry describing a run nobody asked for.
        """
        from .metrics import get_metric
        from .seasonality import seasonal_period

        # Nothing is scored at all without a backtest, so no single metric is the thing to name;
        # `output.point_forecast` and `hpo.enabled` already refuse the settings that depend on one.
        if not self.backtest.enabled:
            return self

        name = self.backtest.decision_metric
        metric = get_metric(name)

        # 1. Ensembles are scored on blended out-of-fold predictions, which carry a point forecast
        #    and no band (`ensemble_run` passes no bounds). Base models keep their intervals, so
        #    `inverse_error` weighting and pruning are unaffected — what goes missing is the
        #    ensemble's own score, i.e. the number that answers "did blending help?".
        if metric.needs_intervals and self.ensemble.enabled:
            _log.warning(
                "decision_metric=%r needs prediction intervals and the ensemble's out-of-fold "
                "predictions carry none, so every ensemble row will score NaN on it and the "
                "ensemble cannot be ranked against the base models. The base models are scored "
                "normally; pick a point-forecast metric to rank the whole leaderboard on one "
                "number.",
                name,
            )

        # 2. A seasonal-naive denominator needs a training window longer than one full cycle. Under
        #    `sliding` that window is the same width at every origin, so this is every fold of every
        #    series; under the expanding schemes the early folds are NaN and the later ones are not,
        #    which is arguably worse to discover from the data.
        if metric.needs_seasonal_period:
            period = seasonal_period(self.data.freq)
            width = self.backtest.window or self.backtest.min_train
            if width <= period:
                _log.warning(
                    "decision_metric=%r scores against a seasonal naive of %d steps (freq=%r) but "
                    "the shortest training window this backtest allows is %d observations, which "
                    "is not longer than one cycle — the metric is undefined there. Raise "
                    "backtest.%s above %d.",
                    name,
                    period,
                    self.data.freq,
                    width,
                    "window" if self.backtest.window else "min_train",
                    period,
                )

        return self

    def resolve_family_compute(self, family: str) -> ResolvedFamilyCompute:
        """Resolve one family's effective compute by layering its override on the flat defaults.

        Pure and deterministic — the single resolver the DAG orchestrator also uses, so a config
        validated at load needs no re-check at submit. ``family`` is a compute family
        (``statistical``/``ml``/``deep_learning``/``automl``); ``native`` has no compute choice (it
        always runs in BigQuery) and raises. Resolution rules for unset override fields:

        * ``runtime`` → ``python_runtime`` (or ``"vertex_automl"`` for ``automl``).
        * ``hardware`` → ``gpu`` only for ``deep_learning`` and ``automl`` (when
          ``compute.use_gpu`` or an explicit override); every other family is ``cpu``.
        * Spark: ``spark_mode`` → ``serverless``; ``spark_cluster_name`` applies only under
          ``cluster``. On ``ray``, ``vertex``, ``gce``, ``gke``, and ``vertex_automl`` both are
          ``None``.
        * GKE & Ray modes: ``gke_mode`` → ``compute.gke_mode`` (``"job"`` | ``"ray"``) on ``gke``;
          ``ray_mode`` → ``compute.ray_mode`` (``"vertex"`` | ``"gke"``) on ``ray`` (or ``"gke"``
          when ``runtime == "gke"`` and ``gke_mode == "ray"``).
        * AutoML mode: ``automl_mode`` → ``compute.automl_mode`` (``"tabular_workflow"`` |
          ``"training_job"``) on ``vertex_automl``.
        * ``gpu_type`` (when ``hardware == "gpu"``) → the flat ``compute.gpu_type``, but **forced to
          L4** on Dataproc Serverless (no T4/A100 there). Inheriting T4/A100 on Serverless raises.
        * ``machine_type`` → ``compute.machine_type`` (auto-resolved via `resolve_vm_machine_type`
          from ``(hardware, gpu_type, accelerator_count)`` on ``vertex``/``gce``/``gke`` ``job``, or
          when explicitly set on ``ray``/``spark`` ``cluster``/``gke`` ``ray``/``vertex_automl``);
          ``workers`` → ``compute.workers`` on ``vertex`` and ``gke`` ``job``, ``1`` on ``gce``, or
          the family override when set on ``ray``/``spark``/``vertex_automl``; ``min_workers`` /
          ``max_workers`` apply on autoscaling runtimes (``ray``/``spark``/``gke`` ``ray``/
          ``vertex_automl``).
        """
        if family == "native":
            raise ValueError(
                "native models always run in BigQuery; they have no per-family compute"
            )
        ov = self.compute.families.get(_as_compute_family(family)) or FamilyCompute()
        if family == "automl":
            runtime = ov.runtime or "vertex_automl"
            if runtime != "vertex_automl":
                raise ValueError(
                    f"family 'automl' models require runtime='vertex_automl' (got {runtime!r})"
                )
        else:
            runtime = ov.runtime or self.python_runtime
            if runtime == "vertex_automl":
                raise ValueError(
                    f"family '{family}' cannot run on runtime='vertex_automl'; "
                    f"set compute.families.{family}.runtime or use "
                    "python_runtime='spark'/'ray'/'vertex'/'gce'/'gke'"
                )

        if family in ("deep_learning", "automl"):
            hardware = ov.hardware or ("gpu" if self.compute.use_gpu else "cpu")
        else:
            hardware = "cpu"

        if runtime == "spark":
            spark_mode = ov.spark_mode or "serverless"
            spark_cluster_name = ov.spark_cluster_name if spark_mode == "cluster" else None
        else:
            if ov.spark_mode is not None or ov.spark_cluster_name is not None:
                raise ValueError(
                    f"family '{family}': spark_mode/spark_cluster_name require runtime='spark' "
                    f"(got {runtime!r})"
                )
            spark_mode = None
            spark_cluster_name = None

        gke_mode: str | None = None
        gke_cluster_name: str | None = None
        ray_mode: str | None = None
        automl_mode: str | None = None
        if runtime == "gke":
            if ov.ray_mode is not None:
                raise ValueError(
                    f"family '{family}': ray_mode requires runtime='ray' (got {runtime!r}); "
                    "use gke_mode='ray' instead"
                )
            if ov.automl_mode is not None:
                raise ValueError(
                    f"family '{family}': automl_mode requires runtime='vertex_automl' "
                    f"(got {runtime!r})"
                )
            gke_mode = ov.gke_mode or self.compute.gke_mode
            gke_cluster_name = ov.gke_cluster_name or self.compute.gke_cluster_name
            ray_mode = "gke" if gke_mode == "ray" else None
        elif runtime == "ray":
            if ov.gke_mode is not None:
                raise ValueError(
                    f"family '{family}': gke_mode requires runtime='gke' (got {runtime!r}); "
                    "use ray_mode='gke' instead"
                )
            if ov.automl_mode is not None:
                raise ValueError(
                    f"family '{family}': automl_mode requires runtime='vertex_automl' "
                    f"(got {runtime!r})"
                )
            ray_mode = ov.ray_mode or self.compute.ray_mode
            if ray_mode == "gke":
                gke_mode = "ray"
                gke_cluster_name = (
                    ov.gke_cluster_name
                    or self.compute.gke_cluster_name
                    or self.compute.ray_cluster_name
                )
            elif ov.gke_cluster_name is not None:
                raise ValueError(
                    f"family '{family}': gke_cluster_name requires runtime='gke' or ray_mode='gke'"
                )
        elif runtime == "vertex_automl":
            if ov.gke_mode is not None or ov.gke_cluster_name is not None:
                raise ValueError(
                    f"family '{family}': gke_mode/gke_cluster_name require runtime='gke' "
                    f"(got {runtime!r})"
                )
            if ov.ray_mode is not None:
                raise ValueError(
                    f"family '{family}': ray_mode requires runtime='ray' (got {runtime!r})"
                )
            automl_mode = ov.automl_mode or self.compute.automl_mode
        else:
            if ov.gke_mode is not None or ov.gke_cluster_name is not None:
                raise ValueError(
                    f"family '{family}': gke_mode/gke_cluster_name require runtime='gke' "
                    f"(got {runtime!r})"
                )
            if ov.ray_mode is not None:
                raise ValueError(
                    f"family '{family}': ray_mode requires runtime='ray' (got {runtime!r})"
                )
            if ov.automl_mode is not None:
                raise ValueError(
                    f"family '{family}': automl_mode requires runtime='vertex_automl' "
                    f"(got {runtime!r})"
                )

        gpu_type: str | None
        accelerator_count: int
        if hardware == "gpu":
            if runtime == "spark" and spark_mode == "serverless":
                if ov.gpu_type in ("T4", "A100", "A100_80GB"):
                    raise ValueError(
                        f"family '{family}': Dataproc Serverless supports L4 only, not "
                        f"{ov.gpu_type}; use spark_mode='cluster' or "
                        "runtime='ray'/'vertex'/'gce'/'gke'"
                    )
                gpu_type = "L4"
            else:
                gpu_type = ov.gpu_type or self.compute.gpu_type
            accelerator_count = ov.accelerator_count or self.compute.accelerator_count
            _default_gpu_machine_type(gpu_type, accelerator_count)
        else:
            if ov.accelerator_count is not None:
                raise ValueError(
                    f"family '{family}': accelerator_count requires hardware='gpu' "
                    f"(got hardware={hardware!r})"
                )
            gpu_type = None
            accelerator_count = 0

        # ``None`` on the autoscaling runtimes means "no fixed pool" — the ceiling is max_workers.
        workers: int | None
        min_workers: int | None = None
        max_workers: int | None = None
        if runtime in ("vertex", "gce") or (runtime == "gke" and gke_mode == "job"):
            if (
                ov.min_workers is not None
                or ov.max_workers is not None
                or (
                    ov.runtime is None
                    and (
                        self.compute.min_workers is not None or self.compute.max_workers is not None
                    )
                )
            ):
                raise ValueError(
                    f"family '{family}': min_workers/max_workers are only supported on "
                    f"autoscaling runtimes ('ray' or 'spark'), not {runtime!r}; use 'workers'"
                )
            machine_type = resolve_vm_machine_type(
                hardware,
                gpu_type,
                self.compute.machine_type,
                ov.machine_type,
                accelerator_count=accelerator_count if hardware == "gpu" else 1,
            )
            if runtime == "gce":
                if (ov.workers is not None and ov.workers > 1) or (
                    ov.runtime is None and self.compute.workers > 1
                ):
                    raise ValueError(
                        f"family '{family}': runtime='gce' supports single-VM execution only "
                        "(workers=1); use runtime='vertex' or runtime='gke' for multi-worker pools"
                    )
                workers = 1
            else:
                workers = ov.workers or self.compute.workers
        else:
            if runtime == "spark" and spark_mode == "serverless" and ov.machine_type is not None:
                raise ValueError(
                    f"family '{family}': Dataproc Serverless does not use VM machine_type; "
                    "use spark_mode='cluster' or runtime='vertex'/'gce'/'gke'/'ray'"
                )
            if ov.machine_type is not None or (
                self.compute.machine_type != "auto"
                and not (runtime == "spark" and spark_mode == "serverless")
            ):
                machine_type = resolve_vm_machine_type(
                    hardware,
                    gpu_type,
                    self.compute.machine_type,
                    ov.machine_type,
                    accelerator_count=accelerator_count if hardware == "gpu" else 1,
                )
            else:
                machine_type = None
            workers = ov.workers
            min_workers = ov.min_workers if ov.min_workers is not None else self.compute.min_workers
            max_workers = ov.max_workers if ov.max_workers is not None else self.compute.max_workers
            if min_workers is not None and max_workers is not None and min_workers > max_workers:
                raise ValueError(
                    f"family '{family}': min_workers ({min_workers}) cannot exceed "
                    f"max_workers ({max_workers})"
                )

        return ResolvedFamilyCompute(
            family=family,
            runtime=runtime,
            spark_mode=spark_mode,
            spark_cluster_name=spark_cluster_name,
            hardware=hardware,
            gpu_type=gpu_type,
            machine_type=machine_type,
            workers=workers,
            min_workers=min_workers,
            max_workers=max_workers,
            accelerator_count=accelerator_count,
            gke_mode=gke_mode,
            gke_cluster_name=gke_cluster_name,
            ray_mode=ray_mode,
            automl_mode=automl_mode,
        )

    @property
    def max_horizon(self) -> int:
        """The largest horizon any ``predict`` call in this run will be asked for.

        Two different horizons exist in a config and it is easy to reach for the wrong one. The
        forward forecast uses ``data.horizon``; every backtest fold predicts ``backtest.horizon``,
        which may be larger. Anything sizing itself against "the horizon" — a params validator
        refusing a model that cannot emit enough steps, a context handed to a model — has to mean
        the larger of the two, because the run will ask for both. A property rather than a field:
        it is derived, so it stays out of ``model_dump`` and no ``run_id`` moves.

        **The embargo counts, and leaving it out was a bug.** A fold's model stops at ``train_end``
        while its validation window starts ``gap`` observations later, so the model has to forecast
        across the embargo before it reaches anything that gets scored: the Python fold asks for
        ``gap + val_size`` steps and throws the first ``gap`` away, and the BigQuery fold asks its
        ``ML.FORECAST`` for ``gap + backtest.horizon``. Both engines were
        already right; this number, which is supposed to describe them, was short by exactly
        ``gap``. What that cost: NeuralProphet emits exactly ``n_forecasts`` direct steps and does
        not recurse, so a ``gap=3`` run passed its params validator and then came up three steps
        short at the tail of every fold.
        """
        if not self.backtest.enabled:
            return self.data.horizon
        return max(self.data.horizon, self.backtest.gap + self.backtest.horizon)

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

    def with_available_models(self) -> RunConfig:
        """Return a copy with ``models`` filtered to those whose upstream packages are installed.

        Raises `ConfigError` if none of the requested models have their required packages
        installed in the current environment.
        """
        from .models import filter_available_models

        kept = filter_available_models(self.models)
        if not kept:
            raise ConfigError(
                f"none of the configured models ({list(self.models)}) have their required "
                f"packages installed in this environment"
            )
        if list(kept) == list(self.models):
            return self
        return self.model_copy(update={"models": kept})


# --- workload estimate ---------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    """Dry-run estimate of the work a run will schedule.

    Two halves, and the split is the point. The **count** half (``n_series`` … ``n_cells``) is a
    function of the config alone, so a plain ``--dry-run`` fills it with no environment and no data
    read. The **geometry** half (``n_fits`` onward) depends on how long each series actually is,
    which only the panel knows — so it is ``None``/empty unless the caller supplies ``obs_counts``
    (what ``plan --feasibility`` reads). Reporting a guess there would be worse than reporting
    nothing: on a ragged panel the guess is wrong in the direction that flatters the plan.

    ``n_cells`` is ``n_series × n_models`` — one cell is one (series, model) pair, which is what
    ``run_cell`` runs, what `engines.ray_io.plan_cluster` sizes against, and what the leaderboard's
    ``n_cells`` counts. It deliberately does **not** multiply by folds: folds happen *inside* a
    cell, and the old fold-multiplied number made a backtested run look like it scheduled three
    times the work when it schedules the same work three times as deep.

    **This is the estimate; the run records the measurement.** ``forecast_metadata.n_fits`` and
    ``train_rows_total`` are the same two quantities counted at the fit site by `worker.run_cell`.
    The two disagree on four of the six refit paths, because the estimate below assumes a fresh fit
    per fold and only ``expanding``/``sliding`` do that — see `backtest.FitTally`. That gap is not
    an error in either number. It is what the scheme cost or saved, and a completed run can be
    asked for it.
    """

    n_series: int | None  # None = unlimited (unknown until the data is read)
    n_models: int
    n_folds: int  # requested backtest folds, or 1 when backtesting is off
    n_cells: int | None  # n_series × n_models; None when n_series unknown
    n_fits: int | None  # Σ over series of n_models × (achieved folds + 1)
    full_fit_equivalents: float | None  # training rows / one whole-history fit per cell
    train_rows_total: int | None  # observations handed to a `.fit()` across the whole run
    fold_histogram: dict[int, int]  # achieved folds → series count; {} when unknown
    n_unscored: int | None  # series achieving zero folds; None when unknown


def estimate_workload(cfg: RunConfig, *, obs_counts: Sequence[int] | None = None) -> Workload:
    """Estimate the work a run will schedule, in cells and in fits (pure).

    ``obs_counts`` is the observation count of each series that will be forecast — one entry per
    series, in any order. Supply it (from ``SELECT ts_id, COUNT(*) … GROUP BY ts_id``) and the fold
    geometry is resolved exactly, per series, through `backtest.fit_rows`. Omit it and only the
    cell counts come back; ``n_series`` then falls back to ``data.series_limit``, which is ``None``
    for an unbounded run, and ``n_cells`` follows it.

    **Why fits and not just cells.** A cell with two backtest folds does three fits, and the two
    fold fits train on shorter windows than the final one — so the honest cost multiplier for
    backtesting is ``full_fit_equivalents`` (2.94 for four years of daily history, two folds, a
    28-day horizon), not ``n_folds + 1`` (3). ``full_fit_equivalents`` is dimensionless: it is what
    one cell costs relative to the same cell with backtesting off, so multiply it by ``n_cells`` to
    compare two plans. ``n_fits`` and ``train_rows_total`` are absolute and both scale with models.

    With backtesting off, every cell does exactly one full-history fit, so ``n_fits == n_cells`` and
    the multiplier is ``1.0`` without needing to know a single series length.
    """
    from .backtest import fit_rows

    n_models = len(cfg.models)
    n_series = len(obs_counts) if obs_counts is not None else cfg.data.series_limit
    n_folds = cfg.backtest.n_folds if cfg.backtest.enabled else 1
    n_cells = None if n_series is None else n_series * n_models

    n_fits: int | None = None
    equivalents: float | None = None
    rows_total: int | None = None
    histogram: dict[int, int] = {}
    n_unscored: int | None = None

    if obs_counts is None:
        # The one geometry fact that needs no data: no backtest means one fit per cell.
        if not cfg.backtest.enabled:
            n_fits, equivalents = n_cells, 1.0
    else:
        per_series = [fit_rows(int(n), cfg) for n in obs_counts]
        achieved = [len(rows) - 1 for rows in per_series]
        rows_total = n_models * sum(sum(rows) for rows in per_series)
        n_fits = n_models * sum(len(rows) for rows in per_series)
        baseline = n_models * sum(int(n) for n in obs_counts)
        equivalents = rows_total / baseline if baseline else None
        histogram = {k: achieved.count(k) for k in sorted(set(achieved))}
        n_unscored = histogram.get(0, 0) if cfg.backtest.enabled else None

    return Workload(
        n_series=n_series,
        n_models=n_models,
        n_folds=n_folds,
        n_cells=n_cells,
        n_fits=n_fits,
        full_fit_equivalents=equivalents,
        train_rows_total=rows_total,
        fold_histogram=histogram,
        n_unscored=n_unscored,
    )


@dataclass(frozen=True)
class Fanout:
    """The count half of a `Workload`, kept for callers that predate it.

    ``estimate_fanout`` is still in the public ``__init__`` and still returns this, so nothing
    downstream had to move. One number did change meaning: ``n_cells`` is now ``n_series ×
    n_models``, matching every other ``n_cells`` in the system (the leaderboard's, the Ray
    planner's, the quota estimator's). It used to multiply by folds, which is why a two-fold dry
    run reported three times the cells it would ever write a row for.
    """

    n_series: int | None
    n_models: int
    n_folds: int
    n_cells: int | None


def estimate_fanout(cfg: RunConfig) -> Fanout:
    """The config-only cell-count estimate — `estimate_workload` narrowed to its four count fields.

    When ``data.series_limit`` is unset, series count isn't known offline, so
    ``n_series`` and ``n_cells`` are ``None`` (the CLI reports "all series").
    """
    w = estimate_workload(cfg)
    return Fanout(n_series=w.n_series, n_models=w.n_models, n_folds=w.n_folds, n_cells=w.n_cells)


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
