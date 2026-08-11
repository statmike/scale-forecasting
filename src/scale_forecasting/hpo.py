"""In-node hyperparameter optimization — an Optuna study over the aligned backtest (C5).

Gated on ``cfg.hpo.enabled``; a strict no-op otherwise (a model with no search space, or HPO
off, tunes nothing and yields ``{}`` — exactly today's behavior). Two granularities, the DS-facing
knob in :class:`~scale_forecasting.config.HpoConfig`:

* ``fleetwide`` (default): tune each model **once** on a representative sample of series and apply
  the winning params across *all* series — the only granularity affordable at 100k. Resolved on the
  driver (:func:`resolve_fleetwide`) before the engine fans out, then threaded to
  :func:`~scale_forecasting.worker.run_cell` as pre-resolved ``params`` (never via ``cfg`` — the
  config is the run_id identity key, so putting tuned params in it would shift the run_id and break
  reproducibility/idempotency; see :func:`~scale_forecasting.registry.ids.make_run_id`).
* ``per_series``: tune on each series inside ``run_cell`` (heavier; a DS opt-in for the tail of
  hard series).

The objective reuses the C2-aligned backtest: for one trial's params it runs
:func:`~scale_forecasting.backtest.backtest_cell` on each sampled series, averages the per-fold
panels, and scores on ``cfg.backtest.decision_metric``. HPO therefore *requires* backtesting
(:func:`require_backtest`) — there are no folds to tune on otherwise.

Pure + offline: no GCP, no Spark, deterministic (fixed-seed TPE sampler). This is the substance of
C5; the engines only add a tiny driver-side sample-and-resolve call in front of their existing
fan-out.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import ConfigError, get_logger
from .models import get_model
from .models.base_model import BaseModel

if TYPE_CHECKING:
    import pandas as pd

    from .config import RunConfig
    from .models.base_model import ModelContext

_log = get_logger(__name__)


def require_backtest(cfg: RunConfig) -> None:
    """Raise if HPO is enabled without backtesting — the study has no folds to score on."""
    if cfg.hpo.enabled and not cfg.backtest.enabled:
        raise ConfigError(
            "hpo.enabled requires backtest.enabled: HPO tunes on the backtest folds "
            "(decision_metric), so there is nothing to optimize with backtesting off."
        )


def _has_search_space(model_cls: type[BaseModel]) -> bool:
    """True if the model overrides :meth:`BaseModel.search_space` (i.e. has params to tune).

    A model that inherits the base (empty) space has nothing to optimize, so HPO skips it and it
    keeps its ``{}`` defaults — the additive-by-default contract. BigQuery-native models tune in
    BQML, not here, so they are excluded regardless.
    """
    if model_cls.runtime == "bigquery":
        return False
    return model_cls.search_space.__func__ is not BaseModel.search_space.__func__  # type: ignore[attr-defined]


def _minimize_scalar(metric: str, value: float) -> float:
    """Map a decision-metric value to a scalar to *minimize* (the study direction is fixed).

    ``coverage`` is better when higher (interval coverage toward its nominal target) → minimize its
    negation. ``bias`` is better near zero (systematic over/under-forecast) → minimize its
    magnitude. Every other panel metric is an error, better when smaller → minimize as-is. A NaN
    score (a trial that produced no scorable fold) becomes ``+inf`` so it can never win.
    """
    if value != value:  # NaN
        return float("inf")
    if metric == "coverage":
        return -value
    if metric == "bias":
        return abs(value)
    return value


def _score_params(
    model_name: str,
    params: dict[str, Any],
    sample: list[pd.DataFrame],
    cfg: RunConfig,
    ctx: ModelContext,
) -> float:
    """One trial's objective: mean decision-metric (as a minimize-scalar) over the sample.

    Runs the aligned backtest for ``params`` on each sampled series, averages the metric across
    folds then across series. A series that raises (too short for the fold geometry, a fit failure)
    is skipped rather than sinking the trial — the same fault-tolerance ``run_cell`` gives a cell.
    An empty sample (or all-skipped) scores ``+inf``.
    """
    from functools import partial

    from .backtest import backtest_cell
    from .features import fit_transform_lambda

    model_cls = get_model(model_name)
    metric = cfg.backtest.decision_metric
    per_series: list[float] = []
    for series in sample:
        try:
            # Box-Cox λ is per-series: fit it on this series and hand the same λ to both the
            # forward features and the folds' inverse (mirrors run_cell). None for none/log1p.
            target = series[cfg.data.target_col].astype(float)
            lam = fit_transform_lambda(target, cfg.features.transform)
            series_ctx = replace(ctx, transform_lambda=lam)
            # partial binds this iteration's series_ctx (no loop-var capture; mypy-typed).
            _, fold_metrics = backtest_cell(
                series, partial(model_cls, params, series_ctx), cfg, lam
            )
        except Exception as e:  # noqa: BLE001 - a bad series must not sink the whole trial
            _log.debug("hpo: skipping a series for %s: %r", model_name, e)
            continue
        vals = [fm.get(metric, float("nan")) for fm in fold_metrics]
        finite = [v for v in vals if v == v]  # drop NaN folds
        if finite:
            per_series.append(float(np.mean(finite)))
    if not per_series:
        return float("inf")
    return _minimize_scalar(metric, float(np.mean(per_series)))


def tune_model(
    model_name: str, sample: list[pd.DataFrame], cfg: RunConfig, ctx: ModelContext | None = None
) -> dict[str, Any]:
    """Tune one model on ``sample`` and return its winning params (``{}`` if nothing to tune).

    Builds a deterministic Optuna study (fixed-seed TPE) of ``cfg.hpo.n_trials`` trials whose
    objective is :func:`_score_params`. Returns ``{}`` immediately — creating no study — when the
    model has no search space (:func:`_has_search_space`), so an all-defaults model costs nothing.
    The returned dict is exactly what ``search_space`` proposes for the best trial, ready to hand a
    model constructor as ``model_cls(params, ctx)``.
    """
    model_cls = get_model(model_name)
    if not _has_search_space(model_cls):
        return {}

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ctx = ctx if ctx is not None else _context(cfg)
    metric = cfg.backtest.decision_metric

    sampler = optuna.samplers.TPESampler(seed=ctx.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        return _score_params(model_name, model_cls.search_space(trial), sample, cfg, ctx)

    study.optimize(objective, n_trials=cfg.hpo.n_trials)
    _log.info(
        "hpo %s: best=%s (%s→%.4g over %d trials)",
        model_name,
        study.best_params,
        metric,
        study.best_value,
        cfg.hpo.n_trials,
    )
    return dict(study.best_params)


def resolve_fleetwide(
    sample: list[pd.DataFrame], cfg: RunConfig, ctx: ModelContext | None = None
) -> dict[str, dict[str, Any]]:
    """Tune every model in ``cfg.models`` once on the shared sample → ``{model: params}``.

    The driver-side fleetwide pre-pass (the opinionated default): the winning params for each model
    are applied across *all* series in the run. Models with no search space are simply absent from
    the mapping (they keep their ``{}`` defaults in ``run_cell``). Pure — the caller supplies the
    pandas ``sample`` (a small, driver-collected set of series); this function does no I/O.
    """
    require_backtest(cfg)
    ctx = ctx if ctx is not None else _context(cfg)
    resolved: dict[str, dict[str, Any]] = {}
    for name in cfg.models:
        params = tune_model(name, sample, cfg, ctx)
        if params:
            resolved[name] = params
    return resolved


def _context(cfg: RunConfig) -> ModelContext:
    """Build the per-run :class:`ModelContext` (lazy import of the worker helper avoids a cycle)."""
    from .worker import _model_context

    return _model_context(cfg)
