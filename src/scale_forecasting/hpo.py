"""In-node hyperparameter optimization — an Optuna study over the aligned backtest.

Gated on ``cfg.hpo.enabled``; a strict no-op otherwise (a model with no search space, or HPO
off, tunes nothing and yields ``{}`` — exactly today's behavior). Two granularities, the DS-facing
knob in `HpoConfig`:

* ``fleetwide`` (default): tune each model **once** on a representative sample of series and apply
  the winning params across *all* series — the only granularity affordable at 100k. Resolved on the
  driver (`resolve_fleetwide`) before the engine fans out, then threaded to
  `run_cell` as pre-resolved ``params`` (never via ``cfg`` — the
  config is the run_id identity key, so putting tuned params in it would shift the run_id and break
  reproducibility/idempotency; see `make_run_id`).
* ``per_series``: tune on each series inside ``run_cell`` (heavier; a DS opt-in for the tail of
  hard series).

The objective reuses the aligned backtest: for one trial's params it runs
`backtest_cell` on each sampled series, averages the per-fold
panels, and scores on ``cfg.backtest.decision_metric``. HPO therefore *requires* backtesting
(`require_backtest`) — there are no folds to tune on otherwise.

Pure + offline: no GCP, no Spark, deterministic (fixed-seed TPE sampler). The engines only add a
tiny driver-side sample-and-resolve call in front of their existing fan-out.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import ConfigError, get_logger
from .metrics import loss_of
from .models import get_model
from .models.base_model import BaseModel

if TYPE_CHECKING:
    import pandas as pd

    from .backtest import FitTally
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
    """True if the model overrides `BaseModel.search_space` (i.e. has params to tune).

    A model that inherits the base (empty) space has nothing to optimize, so HPO skips it and it
    keeps its ``{}`` defaults — the additive-by-default contract. BigQuery-native models tune in
    BQML, not here, so they are excluded regardless.
    """
    if model_cls.runtime == "bigquery":
        return False
    return model_cls.search_space.__func__ is not BaseModel.search_space.__func__  # type: ignore[attr-defined]


def _minimize_scalar(metric: str, value: float) -> float:
    """Map a decision-metric value to a scalar to *minimize* (the study direction is fixed).

    Now one line, because the direction of every metric lives in `metrics.METRIC_DIRECTION` and
    the conversion in `metrics.loss_of`. This used to be the only place in the codebase that knew
    coverage is better when higher and bias is better near zero; the ensembler's weighting and
    pruner each assumed lower-is-better and were silently wrong for those two.

    One value moved when the map was centralised: ``coverage`` now scores ``1 - value`` instead of
    ``-value``. The two rank identically — the transform is monotone decreasing either way — so
    every study picks the same trial; it is the recorded objective number that shifts, and it
    shifts to something a reader can interpret (a shortfall from perfect coverage rather than a
    negative). A NaN score, meaning a trial that produced no scorable fold, is still ``+inf``.
    """
    return loss_of(metric, value)


def _score_panel_params(
    model_cls: type[BaseModel],
    params: dict[str, Any],
    sample: list[pd.DataFrame],
    cfg: RunConfig,
    ctx: ModelContext,
    tally: FitTally | None = None,
) -> float:
    """Score one trial of a ``global`` or ``hybrid`` model across ``sample`` via ``fit_panel``."""
    from .backtest import holdout_fold_id, make_folds
    from .features import (
        build_features,
        build_future_features,
        extract_static_covariates,
        fit_transform_lambda,
    )
    from .metrics import compute_metrics
    from .seasonality import seasonal_period

    if not sample:
        return float("inf")
    id_col = cfg.data.ts_id_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col
    metric = cfg.backtest.decision_metric
    panel_ctx = replace(
        ctx,
        device="auto",
        past_covariates=tuple(cfg.features.past_covariates),
    )

    series_frames: dict[str, pd.DataFrame] = {}
    for idx, s in enumerate(sample):
        if s.empty:
            continue
        uid = str(s[id_col].iloc[0]) if id_col in s.columns else f"s_{idx:04d}"
        series_frames[uid] = s.sort_values(date_col).reset_index(drop=True)
    uids = sorted(series_frames.keys())
    if not uids:
        return float("inf")

    try:
        lams = {
            uid: fit_transform_lambda(
                series_frames[uid][target_col].astype(float), cfg.features.transform
            )
            for uid in uids
        }
        static_map: dict[str, dict[str, Any]] = {}
        for uid in uids:
            sc = extract_static_covariates(series_frames[uid], cfg)
            if sc:
                static_map[uid] = sc

        folds_by_uid = {uid: make_folds(len(series_frames[uid]), cfg) for uid in uids}
        max_folds = max((len(fl) for fl in folds_by_uid.values()), default=0)
        if max_folds == 0:
            return float("inf")

        m_period = seasonal_period(cfg.data.freq)
        gap = cfg.backtest.gap
        bt_h = cfg.backtest.horizon
        fmetrics_by_uid: dict[str, list[dict[str, float]]] = {u: [] for u in uids}

        for fold_idx in range(max_folds):
            active_uids = [u for u in uids if fold_idx < len(folds_by_uid[u])]
            if not active_uids:
                continue
            fold_series_map: dict[str, tuple[pd.Series, pd.DataFrame | None]] = {}
            fold_futr_map: dict[str, pd.DataFrame | None] = {}
            fold_lams: dict[str, float | None] = {}
            fold_rows = 0
            for uid in active_uids:
                fold = folds_by_uid[uid][fold_idx]
                sub = series_frames[uid]
                train_slice = sub.iloc[: fold.train_end]
                val_future = sub.iloc[fold.train_end : fold.val_end]
                if cfg.features.past_covariates:
                    val_future = val_future.drop(
                        columns=[c for c in cfg.features.past_covariates if c in val_future.columns]
                    )
                y_tr, X_tr = build_features(
                    train_slice, cfg, lams[uid], model_cls.lags_covariates_internally
                )
                fold_series_map[uid] = (y_tr, X_tr)
                fold_futr_map[uid] = build_future_features(
                    y_tr, X_tr, cfg, horizon=gap + bt_h, future_covariates_df=val_future
                )
                fold_lams[uid] = lams[uid]
                fold_rows += len(y_tr)

            fold_model = model_cls(params, panel_ctx)
            fold_model.fit_panel(fold_series_map, static_map=static_map if static_map else None)
            if tally is not None:
                tally.record(fold_rows)
            preds_map = fold_model.predict_panel(
                gap + bt_h, fold_futr_map, transform_lambdas=fold_lams
            )
            for uid in active_uids:
                fold = folds_by_uid[uid][fold_idx]
                sub = series_frames[uid]
                train_slice = sub.iloc[: fold.train_end]
                val_slice = sub.iloc[fold.val_start : fold.val_end]
                pred = preds_map[uid].iloc[gap:].reset_index(drop=True)
                y_train_raw = train_slice[target_col].to_numpy(dtype=float)
                y_val = val_slice[target_col].to_numpy(dtype=float)
                yhat = (
                    pred["yhat_raw"].to_numpy(dtype=float)
                    if cfg.output.point_forecast == "raw"
                    else pred["yhat"].to_numpy(dtype=float)
                )
                lower = pred["yhat_lower"].to_numpy(dtype=float)
                upper = pred["yhat_upper"].to_numpy(dtype=float)
                fm = compute_metrics(
                    y_val,
                    yhat,
                    y_train=y_train_raw,
                    lower=lower,
                    upper=upper,
                    seasonal_period=m_period,
                )
                fm["fold_id"] = float(fold.fold_id)
                fmetrics_by_uid[uid].append(fm)
    except Exception as e:  # noqa: BLE001
        _log.debug("hpo: panel trial failed for %s: %r", model_cls.name, e)
        return float("inf")

    per_series: list[float] = []
    holdout = holdout_fold_id(cfg)
    for uid in uids:
        fold_metrics = fmetrics_by_uid[uid]
        inner = [fm for fm in fold_metrics if fm.get("fold_id") != holdout]
        scored = inner or fold_metrics
        vals = [fm.get(metric, float("nan")) for fm in scored]
        finite = [v for v in vals if v == v]
        if finite:
            per_series.append(float(np.mean(finite)))
    if not per_series:
        return float("inf")
    return _minimize_scalar(metric, float(np.mean(per_series)))


def _score_params(
    model_name: str,
    params: dict[str, Any],
    sample: list[pd.DataFrame],
    cfg: RunConfig,
    ctx: ModelContext,
    tally: FitTally | None = None,
) -> float:
    """One trial's objective: mean decision-metric (as a minimize-scalar) over the sample.

    Runs the aligned backtest for ``params`` on each sampled series, averages the metric across
    the **inner** folds then across series. A series that contributes no score is skipped rather
    than sinking the trial — the same fault-tolerance ``run_cell`` gives a cell. Two ways that
    happens: the fit raises, or the series is too short for the fold geometry and `backtest_cell`
    returns no folds to average (it clamps rather than raising, so the second case never reaches
    the ``except``). An empty sample, or one where every series was skipped, scores ``+inf``.

    **The newest fold is not in the objective.** A search that optimises against every fold is
    then scored on those same folds by `worker.run_cell`, so the winning trial's advantage is
    partly the search having seen the answer — and the cell's metric panel, which is what the
    leaderboard ranks, carries that advantage without saying so. Reserving the fold
    `backtest.holdout_fold_id` names leaves the search a window it never touched. A series with
    only that one fold has nothing left to search on, so it falls back to using it — recorded as
    ``hpo_scoring='in_sample'``, never assumed.
    """
    from functools import partial

    from .backtest import backtest_cell, holdout_fold_id
    from .features import (
        effective_config_for_model,
        extract_static_covariates,
        fit_transform_lambda,
    )

    model_cls = get_model(model_name)
    model_cfg = effective_config_for_model(cfg, model_cls)
    if str(params.get("training_mode", "local")) in ("global", "hybrid"):
        return _score_panel_params(model_cls, params, sample, model_cfg, ctx, tally)

    metric = model_cfg.backtest.decision_metric
    per_series: list[float] = []
    for series in sample:
        try:
            # Box-Cox λ is per-series: fit it on this series and hand the same λ to both the
            # forward features and the folds' inverse (mirrors run_cell). None for none/log1p.
            target = series[model_cfg.data.target_col].astype(float)
            lam = fit_transform_lambda(target, model_cfg.features.transform)
            static_covs = extract_static_covariates(series, model_cfg) or None
            # device="auto" is forced, not inherited. A trial may run on the driver — the Ray head
            # node, a Spark driver, an Airflow worker — none of which has an accelerator, and
            # Lightning raises rather than degrading when asked for one that is not there. Auto
            # still uses a device where one exists, so a per-series study on a GPU worker is
            # unaffected; what it gives up is forcing a search OFF a visible card, which is a
            # property of the published fit and not of the search. See `hardware.driver_fit_scope`.
            series_ctx = replace(
                ctx,
                transform_lambda=lam,
                device="auto",
                past_covariates=tuple(model_cfg.features.past_covariates),
                static_covariates=static_covs,
            )
            # partial binds this iteration's series_ctx (no loop-var capture; mypy-typed).
            # The search scores the primary arm only. Its job is to rank parameter sets under the
            # run's own scheme, and the control arm answers a question about refit cadence that no
            # choice of hyperparameter changes.
            _, fold_metrics, _ = backtest_cell(
                series,
                partial(model_cls, params, series_ctx),
                model_cfg,
                lam,
                tally,
                *((True,) if model_cls.lags_covariates_internally else ()),
            )
        except Exception as e:  # noqa: BLE001 - a bad series must not sink the whole trial
            _log.debug("hpo: skipping a series for %s: %r", model_name, e)
            continue
        inner = [fm for fm in fold_metrics if fm.get("fold_id") != holdout_fold_id(model_cfg)]
        # Only this series falls back, not the trial: one short series must not put the whole
        # search back on the fold everything else is reserving.
        scored = inner or fold_metrics
        vals = [fm.get(metric, float("nan")) for fm in scored]
        finite = [v for v in vals if v == v]  # drop NaN folds
        if finite:
            per_series.append(float(np.mean(finite)))
    if not per_series:
        return float("inf")
    return _minimize_scalar(metric, float(np.mean(per_series)))


def tune_model(
    model_name: str,
    sample: list[pd.DataFrame],
    cfg: RunConfig,
    ctx: ModelContext | None = None,
    tally: FitTally | None = None,
) -> dict[str, Any]:
    """Tune one model on ``sample`` and return its winning params (``{}`` if nothing to tune).

    Builds a deterministic Optuna study (fixed-seed TPE) of ``cfg.hpo.n_trials`` trials whose
    objective is `_score_params`. Returns ``{}`` immediately — creating no study — when the
    model has no search space (`_has_search_space`), so an all-defaults model costs nothing.

    **``cfg.model_params[model_name]`` sits underneath every trial.** A model tuned without the
    hyperparameters its author pinned is a different model from the one that will be fitted, so the
    study would optimize the wrong thing — an authored ``n_lags`` changes what ``learning_rate`` is
    best for. Where the two name the same key the trial wins, and the winner is returned with the
    authored layer still beneath it, so the returned dict is exactly the params the cell will build
    with and exactly what lands in ``forecast_metadata.best_params``.

    A model with no search space returns ``{}`` rather than the authored params: nothing was tuned,
    and `worker._resolve_params` applies the authored layer at the cell either way.

    ``tally`` counts the fits the *search* paid for, and only the ``per_series`` granularity passes
    one. A search costs ``n_trials × sample × folds`` fits and none of them produce a shipped
    forecast, so they are counted separately from the cell's own — `worker.run_cell` keeps two
    tallies and writes them to ``n_hpo_fits`` and ``n_fits``. The fleetwide pre-pass passes nothing:
    it runs on the driver before any cell exists, so its fits belong to the run rather than to any
    one row, and there is nowhere honest to put them.
    """
    model_cls = get_model(model_name)
    authored: dict[str, Any] = dict(cfg.model_params.get(model_name, {}))
    if not _has_search_space(model_cls):
        return {}

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ctx = ctx if ctx is not None else _context(cfg)
    metric = cfg.backtest.decision_metric

    sampler = optuna.samplers.TPESampler(seed=ctx.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        return _score_params(
            model_name, {**authored, **model_cls.search_space(trial)}, sample, cfg, ctx, tally
        )

    study.optimize(objective, n_trials=cfg.hpo.n_trials)
    _log.info(
        "hpo %s: best=%s (%s→%.4g over %d trials)",
        model_name,
        study.best_params,
        metric,
        study.best_value,
        cfg.hpo.n_trials,
    )
    return {**authored, **study.best_params}


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
    """Build the per-run `ModelContext` (lazy import of the worker helper avoids a cycle)."""
    from .worker import _model_context

    return _model_context(cfg)
