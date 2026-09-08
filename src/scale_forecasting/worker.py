"""The cell runner — the unit of work that runs identically local / Spark / Ray.

``run_cell`` fits, optionally backtests, and predicts ONE
``(ts_id, model)`` cell and returns a `CellResult`. Engines differ only in how
they *call* it and *collect* its results — that symmetry is what makes "same code
everywhere" real.

`CellResult` is defined here because it is the worker's output type; the registry
writers (``registry/``) consume it. It carries plain data (frames + scalars), no
behavior, so it is the clean seam between compute and lineage.
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .backtest import achievable_folds, backtest_cell
from .calibration import apply_calibration, calibrate_from_oof, compare_arms, select_arm
from .config import corrected_arm_for
from .errors import ConfigError, get_logger
from .features import (
    build_features,
    build_future_features,
    fit_transform_lambda,
    holiday_frame,
)
from .hardware import provisioned_hardware, visible_device
from .metrics import METRIC_NAMES
from .models import get_model
from .models.base_model import (
    DEFAULT_QUANTILES,
    PREDICTION_COLUMNS,
    BaseModel,
    ModelContext,
)
from .registry.ids import make_model_hash, make_run_id
from .resources.catalog import _INTRAOP_ENV_VARS

if TYPE_CHECKING:
    from .config import RunConfig

_log = get_logger(__name__)


@dataclass(frozen=True)
class CellResult:
    """The result of one ``(ts_id, model)`` cell.

    A failing cell sets ``status="error"`` with ``error`` populated and empty
    ``predictions`` — it never raises out of ``run_cell``, so one bad
    cell can't sink a 100k-series batch.
    """

    run_id: str
    ts_id: str
    model_type: str
    compute_engine: str  # "spark" | "ray" | "bigquery"
    model_hash: str
    status: str  # "ok" | "error"
    error: str | None
    predictions: pd.DataFrame  # canonical prediction frame
    oof: pd.DataFrame | None  # canonical OOF frame, or None if backtest off
    metrics: dict[str, float]  # full-fit metrics
    best_params: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0
    # Per-cell wall-clock bracket + worker identity for the run trace (SDK trace()). fit_seconds is
    # the precise (monotonic) fit duration; these are absolute wall-clock stamps that position the
    # cell on a Gantt/waterfall lane, and worker_id (hostname:pid) attributes it to a worker.
    worker_id: str | None = None
    cell_started_at: datetime | None = None
    cell_ended_at: datetime | None = None
    # Serialized fitted model (from BaseModel.serialize), or None when persistence is off / the
    # model opts out. Carried as bytes rather than a temp-file path so it crosses the executor
    # boundary as plain data with no local-fs lifecycle; the registry writer uploads it to GCS and
    # stamps the ObjectRef onto forecast_metadata.model_artifact for model-artifact lineage.
    artifact_bytes: bytes | None = None
    # --- harvested compute measurement (compute.profile.measure) -------------------------------
    # What this cell cost, recorded so a completed run can size a later one. All None/0 when
    # measurement is off, which is also how a row written before these columns existed reads.
    # `fit_seconds` above is the wall-clock half of the same measurement, so it is not repeated.
    cpu_seconds: float | None = None  # time.process_time delta — sums across threads
    # The worker process's ABSOLUTE RSS high-water, not this cell's increment. Deliberate: a slot
    # must hold the interpreter, the libraries and the fit together, and the increment swings 17x
    # on the order cells happened to run in (see `profiling.measure.MeasuredFit`). Monotone within
    # a worker, so MAX across a family's cells is exactly the slot size that family needs.
    process_rss_bytes: int | None = None
    peak_gpu_bytes: int | None = None  # torch.cuda high-water; None == NOT MEASURED, never zero
    # The native-thread cap in force while this cell ran (OMP_NUM_THREADS). Without it
    # cpu_seconds/fit_seconds is uninterpretable: under a cap the ratio reports the cap back.
    intraop_threads: int | None = None
    n_obs: int | None = None  # rows fed to the fit — the data signature a later run matches on
    # --- device evidence (the GPU contract's Layer 4) ------------------------------------------
    # Recorded on every cell, never inferred from `peak_gpu_bytes` — that column's None is
    # overloaded across four causes (no torch, no CUDA build, no device, profiling off), so it
    # cannot distinguish "the accelerator never attached" from "nobody looked". These four can.
    device_requested: str | None = None  # what the cell was told: "auto" | "cpu" | "gpu"
    device_available: str | None = None  # what the worker can see: "cuda" | "cpu" | "unknown"
    device_used: str | None = None  # where the weights landed; None = the model cannot say
    device_name: str | None = None  # e.g. "Tesla T4", when a device is visible
    # --- backtest outcome, separate from the cell's own outcome --------------------------------
    # A cell can forecast perfectly well and still be unscorable, so scoring gets its own status.
    # All three are None when backtesting was never asked for — the one case where a NULL metric
    # panel is not a shortfall. "full" | "reduced" | "unscored" | "failed": `reduced` scored fewer
    # folds than requested, `unscored` could not score any, `failed` raised.
    backtest_status: str | None = None
    n_folds_achieved: int | None = None  # folds actually scored; 0 on unscored/failed
    backtest_note: str | None = None  # why it was not full — the arithmetic, or the exception
    # `error` says what went wrong in the words of whatever raised; this says what *kind* of thing
    # it was, from a fixed vocabulary (`ERROR_CLASSES`). One is for reading, the other for grouping
    # and for deciding whether a retry could possibly help. None on an ok cell.
    error_class: str | None = None
    # Where this cell's prediction bounds came from: "native" if the model computes its own
    # intervals, "residual" if `BaseModel.residual_intervals` built them from the empirical spread
    # of in-sample residuals. Both are legitimate; they are not the same claim, and the interval
    # metrics (coverage, pinball, interval_score) mean different things under each. None on an
    # error cell, where there are no bounds to describe.
    interval_source: str | None = None
    # Which functional `yhat` carries: "raw" (the model's own output), "median" or "mean" (plus the
    # corresponding residual statistic). Distinct from `interval_source`, which describes the
    # *band* — a run can ship the raw point forecast inside an out-of-fold calibrated interval.
    # This exists because for a long time the answer was "median", nobody had chosen it, and
    # nothing recorded it.
    point_forecast_source: str | None = None
    # *How* that arm was picked, which `point_forecast_source` alone cannot say: a cell shipping
    # "median" may have been told to, or may have weighed both arms on its own folds and kept the
    # default. Same two-column shape as `interval_source`/`interval_calibration` next door — one
    # column for what the number is, one for where it came from. See `calibration.ARM_DECISIONS`.
    point_forecast_decision: str | None = None
    # Where the band came from: "oof-per-step" (held-out residuals, resolved by horizon distance),
    # "oof-flat" (held-out, pooled — too few residuals per step to say more), or "in-sample" (the
    # model's own band; no backtest ran). The three are ranked, and the distinction is the whole
    # difference between a coverage number that means something and one that flatters itself.
    interval_calibration: str | None = None
    # Relative improvement of the corrected arm over the raw one on this cell's folds, in units of
    # the decision metric's loss: positive means the correction helped. NaN/None when there was no
    # backtest to measure it on. This is the per-series evidence that a single global setting
    # cannot express, and it is what 2.5b's automatic per-series selection will read.
    point_forecast_margin: float | None = None
    # Whether the hyperparameter search that produced `best_params` was scored with the newest
    # fold held out of it. "holdout" = yes, "in_sample" = there was no inner fold to search on so
    # it used every fold (and the cell's own metrics therefore include a window the search
    # optimised against). None = no search ran for this cell, which is the common case. The
    # ensemble's counterpart lives on the run's ensemble rows as `ensemble_scoring`.
    hpo_scoring: str | None = None


def _worker_id() -> str:
    """A runtime-agnostic worker identity (``hostname:pid``).

    The same call on the driver, a Spark executor, or a Ray worker, so the trace attributes each
    cell to the physical worker that ran it without any engine-specific API — keeping ``run_cell``
    identical everywhere.
    """
    return f"{socket.gethostname()}:{os.getpid()}"


def _intraop_threads() -> int | None:
    """The native-thread cap this process is running under, or None when nothing caps it.

    Read from the environment rather than inferred, because that is where the fleet actually
    sets it: `resources` exports all five of `catalog._INTRAOP_ENV_VARS` to ``spark.task.cpus``
    on every Spark job, and Ray exports ``OMP_NUM_THREADS`` to a task's ``num_cpus``. A cell that
    records the cap it ran under is a cell whose `effective_cores` can be read honestly later.

    **The widest cap wins, and reading only OMP was reporting the pin back to itself.** These
    five variables cap different thread pools, and the fit uses whichever library is underneath:
    ``OMP_NUM_THREADS=1`` beside an unset ``OPENBLAS_NUM_THREADS`` is not a one-thread process,
    it is a process where the OpenBLAS matrix ops still take the whole node. Recording 1 there
    made ``cpu_seconds / fit_seconds`` look like clean single-threaded occupancy when the run
    was oversubscribed — the measurement agreeing with the assumption instead of testing it. So
    the honest answer is the *largest* cap in force; an unset variable is no cap at all and
    yields ``None`` for the whole process, because one uncapped pool is enough to uncap the fit.
    """
    caps: list[int] = []
    for name in _INTRAOP_ENV_VARS:
        raw = os.environ.get(name)
        if not raw:
            return None  # this pool is uncapped, so the process is
        try:
            caps.append(int(raw))
        except ValueError:
            return None
    return max(caps) if caps else None


def _process_rss_bytes() -> int | None:
    """This worker process's absolute RSS high-water in bytes, or None where unmeasurable.

    Delegates to the probe `profiling` already owns, imported lazily because `profiling` imports
    *this* module (``measure_fit`` drives ``run_cell``) — a module-level import would be a cycle.
    The lazy call is a ``sys.modules`` hit after the first cell.
    """
    from .profiling.measure import _rss_bytes

    return _rss_bytes()


_gpu_probe_useful: bool | None = None  # None = not yet asked; False = no accelerator here


def _peak_gpu_bytes() -> int | None:
    """Peak CUDA bytes this process has allocated, or None when NOT MEASURED.

    ``None`` on every no-accelerator path, never ``0`` — a consumer that read a missing device
    as zero would compute a minimum GPU fraction and pack ten tasks onto a device that fits two.

    The "is there a GPU" half of the answer is cached per process because the probe's cheap case
    is not cheap at cell scale: a *failed* ``import torch`` is not memoized in ``sys.modules``, so
    on a CPU-only Spark worker every one of a hundred thousand cells would re-walk ``sys.path``.
    Neither torch's presence nor a device's appears mid-process, so one ask settles it; only the
    high-water *value* is re-read, and only where a device actually exists.
    """
    global _gpu_probe_useful
    if _gpu_probe_useful is False:
        return None
    from .profiling.measure import _peak_gpu_bytes as probe

    peak = probe()
    _gpu_probe_useful = peak is not None
    return peak


def _compute_engine(model_cls: type[BaseModel], cfg: RunConfig) -> str:
    """The engine that will execute this cell: the Python runtime, or BigQuery for native
    models (which run as SQL regardless of the run's Python runtime)."""
    return "bigquery" if model_cls.runtime == "bigquery" else cfg.python_runtime


def _model_context(
    cfg: RunConfig,
    transform_lambda: float | None = None,
    *,
    family: str | None = None,
) -> ModelContext:
    """Build the per-cell `ModelContext` from the run config.

    ``transform_lambda`` is the cell's fitted Box-Cox λ (None for stateless transforms), fit
    once in `run_cell` and shared by the backtest folds and the final fit.

    ``family`` is the cell's model family, and it is what turns the job-level "this job has
    devices" into a per-cell device. See `_resolve_device`.
    """
    holidays = holiday_frame(cfg) if cfg.features.holidays else None
    return ModelContext(
        freq=cfg.data.freq,
        # The LARGEST horizon this cell will be asked for, not the forward one. The same context
        # object is handed to the backtest folds, which predict `backtest.horizon`, and to the final
        # fit, which predicts `data.horizon`. Carrying only the forward horizon made the field a
        # trap: a model sizing anything off it — a head count, a buffer — would be right on the
        # final fit and short on every fold, in a run where both numbers are legal and different.
        horizon=cfg.max_horizon,
        seed=0,
        holidays=holidays,
        transform=cfg.features.transform,
        transform_lambda=transform_lambda,
        device=_resolve_device(cfg, family),
    )


def _resolve_device(cfg: RunConfig, family: str | None) -> str:
    """Which device this cell's model should fit on: ``"auto"``, ``"cpu"`` or ``"gpu"``.

    Two facts have to agree before a cell is told to use a device. The **job** must have been
    provisioned onto GPU hardware, which only the submitter knows and which arrives through the
    environment (`hardware.provisioned_hardware`); and the **family** must resolve to ``gpu`` in
    this config, which is the same `RunConfig.resolve_family_compute` the submitter provisioned
    from and the DAG planned from.

    Neither alone is enough, and the failure each one prevents is different. Without the job half,
    a GPU config run on a laptop would ask Lightning for a device that is not there and crash a
    local run that works today. Without the family half, a mixed-hardware Dataproc cluster would
    tell a statistical cell to use the card its executor happens to expose — the case
    ``hardware: "cpu"`` was supposed to cover and, under ``accelerator="auto"``, never did.

    Anything else is ``"auto"``: the library chooses, which can never fail and is exactly what
    every run made before this field existed did.
    """
    if family is None or provisioned_hardware() != "gpu":
        return "auto"
    return "gpu" if cfg.resolve_family_compute(family).hardware == "gpu" else "cpu"


def _require_device(device: str, family: str, engine: str) -> None:
    """Fail a cell fast when it was told to use a device that is not actually here.

    Post-Layer-3 this should be unreachable — the submitter provisions and the engine routes off
    one resolver — so it is a regression detector, and it has to be *fast*: the alternative is
    discovering a missing device after the fleet-hour is spent, with forecast rows orphaned under a
    job that failed at the end. The probe is `_peak_gpu_bytes`, which memoizes "no accelerator" per
    process, so the check costs one failed import per worker and nothing thereafter.

    Only ``device == "gpu"`` is checked. ``"cpu"`` and ``"auto"`` cannot be short of hardware.

    The message reports what the worker actually saw, because the two ways to get here need
    different fixes and look identical from the outside: torch present and reporting no device
    means the accelerator did not attach (or was hidden), while no importable torch at all means
    this worker never had a tensor library to ask.
    """
    if device != "gpu" or _peak_gpu_bytes() is not None:
        return
    available, _name = visible_device()
    saw = (
        "torch is installed here and reports no CUDA device, so the accelerator did not attach to "
        "this worker"
        if available == "cpu"
        else "torch could not be imported on this worker, so it has no tensor library to run on a "
        "device at all"
    )
    raise ConfigError(
        f"family '{family}' is set to hardware='gpu' and this {engine} job was provisioned onto "
        f"GPU hardware, but {saw}. Set compute.families.{family}.hardware to 'cpu' to run "
        f"without one."
    )


def _resolve_params(
    series: pd.DataFrame,
    model_name: str,
    cfg: RunConfig,
    ctx: ModelContext,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the hyperparameters this cell builds its model with (see `run_cell`).

    Three sources, layered in a fixed order: the model's own defaults (whatever it reads out of an
    absent key), then ``cfg.model_params[model_name]`` — what the config author wrote — then
    anything HPO tuned, which wins. Pre-resolved ``params`` are the fleetwide driver pre-pass;
    per-series HPO tunes on *this* series when enabled at that granularity; with HPO off there is
    nothing above the authored layer.

    **HPO beats an authored value on the keys it searches.** Pinning ``epochs`` while a model's
    search space also searches ``epochs`` means the trial's value is used and the pin is ignored,
    because a study that scored one value and shipped another would publish a metric that does not
    belong to the fitted model. Keys the search space does not name are unaffected — that is the
    common case, and it is how an authored ``n_lags`` survives a tuned ``learning_rate``.

    Kept tiny and separate so the resolution policy is one readable place and the HPO import stays
    lazy (Optuna loads only when a run actually tunes).
    """
    authored: dict[str, Any] = dict(cfg.model_params.get(model_name, {}))
    if params is not None:
        return {**authored, **params}
    if cfg.hpo.enabled and cfg.hpo.granularity == "per_series":
        from .hpo import tune_model

        return {**authored, **tune_model(model_name, [series], cfg, ctx)}
    return authored


def hpo_scoring_basis(n_obs: int, cfg: RunConfig, params: dict[str, Any] | None) -> str | None:
    """Whether the search behind this cell's params reserved the newest fold (pure).

    ``None`` when no search ran, which is most cells. Otherwise ``"holdout"`` when an inner fold
    existed for the search to score on, or ``"in_sample"`` when it did not and the search fell
    back to every fold — a distinction that decides whether the cell's own metrics are honest,
    since the search and the leaderboard would otherwise be reading the same window.

    The two granularities are asked different questions. Per-series tuning searched *this* series,
    so the split exists only if this series is long enough for two folds. Fleetwide tuning searched
    a sample the cell never saw, so the only thing the cell can honestly report is the run's fold
    geometry; a short series inside that sample fell back on its own, and it is the run header's
    ``scoring`` block, not this column, that would show it.
    """
    if not cfg.hpo.enabled:
        return None
    if params is not None:
        return "holdout" if cfg.backtest.n_folds >= 2 else "in_sample"
    if cfg.hpo.granularity != "per_series":
        return None
    return "holdout" if achievable_folds(n_obs, cfg) >= 2 else "in_sample"


def _rollup_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average the per-fold metric panels into one panel.

    NaNs are ignored per metric (a metric undefined on one fold shouldn't sink the mean);
    a metric NaN on every fold stays NaN. Always returns the full panel.
    """
    out: dict[str, float] = {}
    for name in METRIC_NAMES:
        vals = np.array([fm.get(name, np.nan) for fm in fold_metrics], dtype=float)
        finite = vals[~np.isnan(vals)]
        out[name] = float(finite.mean()) if finite.size else float("nan")
    return out


def _empty_predictions() -> pd.DataFrame:
    """An empty canonical prediction frame (for error cells)."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in PREDICTION_COLUMNS})


def run_cell(
    series: pd.DataFrame,
    model_name: str,
    cfg: RunConfig,
    params: dict[str, Any] | None = None,
) -> CellResult:
    """Fit + (optional backtest) + predict ONE ``(ts_id, model)`` cell.

    Pure-ish and deterministic: reads nothing global, writes nothing (no BQ), returns a
    `CellResult` carrying plain data. A failing cell returns ``status="error"`` and
    never raises, so one bad series can't sink a batch.

    ``params`` are the hyperparameters this cell's model is built with. Resolution order:

    * ``params`` given (not None) → use them. This is the **fleetwide** path: the driver tuned the
      model once on a sample (`resolve_fleetwide`) and threads the
      winning params here — never through ``cfg`` (the config is the run_id identity key).
    * else if ``cfg.hpo.enabled`` and ``granularity == "per_series"`` → tune on *this* series now.
    * else → ``{}`` (the default: today's untuned behavior).

    The resolved params drive **both** the backtest folds and the final fit, so
    ``best_params = model.get_params()`` reflects what actually ran.
    """
    ts_id = _ts_id(series, cfg)
    run_id = make_run_id(cfg)
    model_hash = make_model_hash(run_id, ts_id, model_name, cfg)
    # Wall-clock lane + worker identity for the trace, captured for every return path (ok or error).
    cell_started_at = datetime.now(UTC)
    worker_id = _worker_id()
    # What this worker can see, memoized per process. Read up here so an error cell carries it too:
    # a cell that failed *because* the accelerator was missing is the row most worth the evidence.
    available, device_name = visible_device()

    def _error(exc: BaseException, engine: str) -> CellResult:
        return CellResult(
            run_id=run_id,
            ts_id=ts_id,
            model_type=model_name,
            compute_engine=engine,
            model_hash=model_hash,
            status="error",
            error=repr(exc),
            # Classified here rather than by a reader after the fact, because this is the only
            # place the exception object still exists: `error` is a string by the time it leaves
            # the worker, and the type is the strongest signal the table has.
            error_class=classify_error(exc),
            predictions=_empty_predictions(),
            oof=None,
            metrics={name: float("nan") for name in METRIC_NAMES},
            worker_id=worker_id,
            cell_started_at=cell_started_at,
            cell_ended_at=datetime.now(UTC),
            device_available=available,
            device_name=device_name,
        )

    try:
        model_cls = get_model(model_name)
    except Exception as e:  # unknown model name → error cell, engine unknown
        return _error(e, cfg.python_runtime)

    engine = _compute_engine(model_cls, cfg)
    # Harvest: record what this fit costs so a later run can be sized from it. Three cheap probes
    # around work the run was doing anyway — no sample, no pre-pass. Neither RSS nor the CUDA
    # high-water is reset first: the absolute peak is the number that sizes a slot (see
    # `CellResult.process_rss_bytes`), and resetting would also perturb whatever else shares this
    # worker. `intraop_threads` is captured before the fit because that is when it is in force.
    measuring = cfg.compute.profile.records_measurements
    intraop_threads = _intraop_threads() if measuring else None
    cpu_started = time.process_time()
    started = time.perf_counter()
    try:
        # Fit the transform's stateful λ once per cell (None for none/log1p), on the raw target.
        # It lives on ctx so the backtest folds and the final fit share one λ — never refit at
        # predict (the whole point of carrying it on the cell).
        lam = fit_transform_lambda(_target(series, cfg), cfg.features.transform)
        ctx = _model_context(cfg, transform_lambda=lam, family=model_cls.family)
        _require_device(ctx.device, model_cls.family, engine)
        resolved = _resolve_params(series, model_name, cfg, ctx, params)

        # Optional backtest first (fresh model per fold) → OOF frame + rolled-up metrics.
        #
        # Its own try/except, and this is the point of the block. Backtesting *scores* a model; it
        # does not produce the forecast. A scoring failure that propagated turned the whole cell
        # into an error and threw away a forecast that had not even been attempted yet — which is
        # how short history became the largest error class in the registry. Whatever happens here,
        # the final fit below still runs, and the outcome is recorded rather than inferred from a
        # NULL metric (which is also what "backtesting was off" looks like).
        oof: pd.DataFrame | None = None
        metrics = {name: float("nan") for name in METRIC_NAMES}
        backtest_status: str | None = None
        n_folds_achieved: int | None = None
        backtest_note: str | None = None
        if cfg.backtest.enabled:
            try:
                oof, fold_metrics = backtest_cell(
                    series, lambda: model_cls(resolved, ctx), cfg, lam
                )
                metrics = _rollup_metrics(fold_metrics)
                n_folds_achieved = len(fold_metrics)
                backtest_status, backtest_note = _backtest_outcome(n_folds_achieved, cfg, series)
                if n_folds_achieved == 0:
                    oof = None  # an empty frame would write zero rows and read as "not asked"
            except Exception as e:  # noqa: BLE001 - scoring is not the forecast; degrade, don't fail
                _log.warning(
                    "backtest failed for %s/%s (forecast unaffected): %r", ts_id, model_name, e
                )
                oof = None
                metrics = {name: float("nan") for name in METRIC_NAMES}
                backtest_status, n_folds_achieved, backtest_note = "failed", 0, repr(e)

        # Final fit on the full history, then forecast the horizon.
        y, X = build_features(series, cfg, lam)
        model = model_cls(resolved, ctx)
        model.fit(y, X)
        # The design frame for the horizon, indexed by the *future* dates: holiday flags and
        # Fourier phase are recomputed there (exact — they are functions of the date), the
        # level-shift step is carried forward, and only user-supplied exog falls back to a
        # recency stand-in because it is genuinely unknown. Tree models ignore any lag_*
        # columns here (see _lag_forecaster).
        future_exog = build_future_features(y, X, cfg)
        predictions = model.predict(cfg.data.horizon, future_exog)

        # Recalibrate the forward forecast against held-out error.
        #
        # The model's own band comes from *in-sample* residuals, which are one-step-ahead by
        # construction and are systematically too small for any model that fits its training data
        # tightly. Measured consequence: fleet coverage of 0.601 against a nominal 0.8. The OOF
        # frame above is a genuine held-out sample that this run has already paid for, and it
        # carries `horizon_step`, so it can also say how the error *grows* — which in-sample
        # residuals structurally cannot. Both the band and the bias correction are re-derived from
        # it when it is available, per horizon step where there are enough residuals to support
        # one, and the provenance is recorded either way rather than left to be inferred from the
        # fold count. `cal is None` leaves the model's own band untouched.
        #
        # Which arm ships is a separate question from how the band was built, and under
        # `point_forecast="auto"` it is answered per series from this cell's own folds rather than
        # once for the whole fleet. The margin is always reported corrected-minus-raw whatever the
        # cell chose, so a fleet-wide average stays comparable across cells that chose differently.
        metric = cfg.backtest.decision_metric
        corrected = corrected_arm_for(metric)
        arm, arm_decision = cfg.output.point_forecast or "median", "configured"
        if arm == "auto":
            arm, arm_decision = select_arm(oof, metric, corrected)
        cal = calibrate_from_oof(oof, DEFAULT_QUANTILES) if oof is not None else None
        predictions, interval_calibration = apply_calibration(predictions, cal, arm)
        # `corrected`, not `arm`: the margin has to mean the same thing on every row for a
        # fleet-wide GROUP BY to be worth reading, so the comparison is always
        # corrected-vs-raw under the metric's own functional. Passing the selected arm would
        # have made a `raw`-configured run with `decision_metric="rmse"` report a *median*
        # comparison under a column the rest of the run reads as the mean one.
        arm_comparison = compare_arms(oof, metric, corrected) if oof is not None else {}

        # Persist the fitted model as an artifact only when the run opts in (model-artifact
        # lineage). A serialize failure must not sink an otherwise-good forecast, so it degrades to
        # no artifact rather than turning the cell into an error.
        artifact_bytes: bytes | None = None
        if cfg.compute.persist_models:
            try:
                artifact_bytes = model.serialize()
            except Exception as e:  # noqa: BLE001 - persistence is best-effort, never fatal
                _log.warning("serialize failed for %s/%s: %r", ts_id, model_name, e)

        fit_seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        return CellResult(
            run_id=run_id,
            ts_id=ts_id,
            model_type=model_name,
            compute_engine=engine,
            model_hash=model_hash,
            status="ok",
            error=None,
            predictions=predictions,
            oof=oof,
            metrics=metrics,
            best_params=model.get_params(),
            fit_seconds=fit_seconds,
            worker_id=worker_id,
            cell_started_at=cell_started_at,
            cell_ended_at=datetime.now(UTC),
            artifact_bytes=artifact_bytes,
            cpu_seconds=cpu_seconds if measuring else None,
            process_rss_bytes=_process_rss_bytes() if measuring else None,
            # Unconditional, unlike its neighbours: this is the only record of whether a device
            # the run paid for was ever visible to a cell, and gating it on profiling left 8 of
            # 21 historical GPU jobs with no evidence either way. It is also the cheapest probe
            # here — `_peak_gpu_bytes` memoizes "no accelerator" per process, so a CPU-only
            # worker pays one failed import for its whole life and every later cell short-circuits.
            peak_gpu_bytes=_peak_gpu_bytes(),
            intraop_threads=intraop_threads,
            n_obs=len(series) if measuring else None,
            # Unconditional for the same reason: this is the only per-cell record of whether the
            # device a run paid for was ever visible, and of where the fit actually landed. Both
            # probes are memoized or trivial, and `device_used` is asked of the fitted model.
            device_requested=ctx.device,
            device_available=available,
            device_used=model.device_used(),
            device_name=device_name,
            backtest_status=backtest_status,
            n_folds_achieved=n_folds_achieved,
            backtest_note=backtest_note,
            # A class attribute, so this is the model's own declaration rather than an inference
            # from the frame — a residual band on a model with no recorded residuals collapses to
            # bounds equal to `yhat`, which is indistinguishable from a native zero-width interval
            # by inspection and very distinguishable by what it means.
            interval_source="native" if model_cls.supports_native_intervals else "residual",
            # Which functional `yhat` carries, and where the band came from. Two separate facts:
            # a run can select the raw arm and still ship an OOF-calibrated interval.
            point_forecast_source=arm,
            point_forecast_decision=arm_decision,
            interval_calibration=interval_calibration,
            # The arm comparison, scored on the folds. Held-out by construction — each fold's
            # correction came from that fold's own training window and never saw the slice it is
            # measured against. This is the number that says whether the correction earned its
            # keep *for this series*, rather than asking anyone to trust a fleetwide average.
            point_forecast_margin=arm_comparison.get("margin"),
            hpo_scoring=hpo_scoring_basis(len(series), cfg, params),
        )
    except Exception as e:  # any failure → error cell, batch survives
        return _error(e, engine)


# --- why a cell failed ------------------------------------------------------------------------
#
# One flat ordered table, and it lives here because `worker.py` is the one file both Python
# runtimes share — a Spark executor and a Ray worker classify a failure identically or the column
# is worthless for grouping.
#
# Each row is ``(token, exception type names, message substrings)``. A row matches if the
# exception's MRO contains one of the type names **or** its text contains one of the substrings.
# Types are matched by *name*, not by class, so classifying a `CapacityExhausted` or a Google API
# error costs no import — this table is read on every failed cell on every executor, and pulling
# `capacity.py` or `google.api_core` into that path to identify an error we have already lost to
# would be a poor trade.
#
# **First match wins, so the order is the policy**, and two orderings below are load-bearing:
#
# * `SHORT_HISTORY` precedes `BAD_DATA` because both are `DataError`. Short history is a fact about
#   the series that a user can act on; "bad data" is a shrug.
# * `CONFIG_REPAIRABLE` precedes `MODEL_ERROR` because `models.get_model` raises `ModelError` for a
#   name that is not registered — which is a typo in the config, not a modelling failure. Filing it
#   under `MODEL_ERROR` would send a reader to debug a model that was never constructed.
#
# The point of the vocabulary is triage: `OOM`, `TRANSIENT_INFRA` and `CAPACITY` describe the
# machine and may well succeed on a retry; `BAD_DATA`, `SHORT_HISTORY` and `CONFIG_REPAIRABLE`
# describe the input and never will; `MODEL_ERROR` is the model's own answer. Retry logic is not
# written yet (Phase 7) — this is the evidence it will need, recorded now while it is free.
ERROR_CLASSES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "OOM",
        ("MemoryError", "OutOfMemoryError"),
        ("out of memory", "cannot allocate memory", "oom-kill", "oomkilled"),
    ),
    (
        "CAPACITY",
        ("CapacityExhausted", "ResourceExhausted"),
        (
            "resource_exhausted",
            "quota exceeded",
            "insufficient capacity",
            "does not have enough resources",
            "stockout",
        ),
    ),
    (
        "TRANSIENT_INFRA",
        (
            "ServiceUnavailable",
            "InternalServerError",
            "DeadlineExceeded",
            "RetryError",
            "ConnectionError",
            "ConnectionResetError",
            "BrokenPipeError",
            "TimeoutError",
        ),
        (
            "503 ",
            "502 ",
            "504 ",
            "500 internal",
            "service unavailable",
            "deadline exceeded",
            "connection reset",
            "broken pipe",
            "temporarily unavailable",
            "try again later",
        ),
    ),
    (
        "SHORT_HISTORY",
        (),
        ("not enough data", "too short", "insufficient history", "needs >= "),
    ),
    ("CONFIG_REPAIRABLE", ("ConfigError",), ("unknown model '",)),
    (
        "BAD_DATA",
        ("DataError",),
        ("duplicate timestamp", "gap at ", "not on the freq", "contains nan", "contains inf"),
    ),
    (
        "MODEL_ERROR",
        ("ModelError", "LinAlgError", "ConvergenceError"),
        ("did not converge", "convergence", "singular matrix", "optimization failed"),
    ),
)


def classify_error(exc: BaseException) -> str:
    """Which `ERROR_CLASSES` token describes ``exc`` — ``"UNKNOWN"`` if none does (pure).

    ``UNKNOWN`` is a real answer, not a failure of the table. A token invented per stack trace
    would make the column unqueryable, so the vocabulary stays fixed and the honest response to an
    unrecognised failure is to say so and leave the full text in ``error_detail``. A rising
    ``UNKNOWN`` share is the signal that this table needs a row, and it is visible in one GROUP BY.
    """
    names = {t.__name__ for t in type(exc).__mro__}
    text = f"{type(exc).__name__}: {exc}".lower()
    for token, types, needles in ERROR_CLASSES:
        if names.intersection(types) or any(needle in text for needle in needles):
            return token
    return "UNKNOWN"


def _backtest_outcome(
    achieved: int, cfg: RunConfig, series: pd.DataFrame
) -> tuple[str, str | None]:
    """Classify a completed backtest ``full``/``reduced``/``unscored``, with a reason if not full.

    Separated from the metrics so a reader can tell the three apart at a glance in SQL. They look
    identical through the metric columns — a reduced backtest and a full one both produce numbers,
    and an unscored one produces the same NULLs as a run with backtesting switched off — yet they
    support very different conclusions about a leaderboard. The note names the arithmetic rather
    than restating the status, because the actionable part is *how much* history the series would
    have needed.
    """
    bt = cfg.backtest
    if achieved >= bt.n_folds:
        return "full", None
    need = bt.min_train + bt.horizon + (bt.n_folds - 1) * bt.step
    shortfall = (
        f"{len(series)} observations support {achieved} of {bt.n_folds} folds; "
        f"{need} needed for all of them "
        f"(min_train={bt.min_train} + horizon={bt.horizon} + (n_folds-1)*step={bt.step})"
    )
    return ("reduced" if achieved > 0 else "unscored"), shortfall


def _ts_id(series: pd.DataFrame, cfg: RunConfig) -> str:
    """The series id for this cell: the ``ts_id_col`` value, or ``"unknown"`` if absent."""
    col = cfg.data.ts_id_col
    if col in series.columns and len(series):
        return str(series[col].iloc[0])
    return "unknown"


def _target(series: pd.DataFrame, cfg: RunConfig) -> pd.Series:
    """The raw target column as a float Series, for fitting the transform λ (before features).

    Raises ``ConfigError`` if the target column is absent — the same failure ``build_features``
    would raise a moment later, surfaced here so the λ fit names it.
    """
    col = cfg.data.target_col
    if col not in series:
        raise ConfigError(
            f"series missing required target column '{col}'; has {list(series.columns)}"
        )
    return series[col].astype(float)
