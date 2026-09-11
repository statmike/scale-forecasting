"""Execute + score the run's ensembles into the registry (the orchestration seam).

`ensembler` is **pure** — it blends the base forecasts (calculated) and fits the learned
meta-learners, but touches no GCP. This module is the thin I/O orchestrator that makes those
consensuses *real*: it reads the base rows, computes both families **in pandas**, appends the
``ensemble_<s>`` prediction rows via the Storage Write API, and — the missing leaderboard link —
**scores every ensemble pseudo-model into ``forecast_metadata``** so it appears on
``v_model_leaderboard`` beside the base models. It is engine-agnostic (it only reads the shared
registry) and runs either inline from `main.run` (after both engines join, under the one
shared ``run_id``) or standalone via ``python -m scale_forecasting.ensemble_run`` (re-ensembling an
already-completed run — see `_main`).

**Config-keyed ensembles.** Every ensemble row carries an ``ensemble_id =
make_ensemble_id(cfg.ensemble)`` — a digest of the ensemble configuration alone — so *several*
ensemble configs can be scored under one ``run_id`` without their ``ensemble_<strategy>``
pseudo-models colliding. Re-running the *same* ensemble config lands the same ``ensemble_id`` (and
so the same cells, deduped newest-first on read); a *different* config lands a different
``ensemble_id`` and sits beside the first on the leaderboard, distinctly keyed. The view groups by
``(run_id, model_type, ensemble_id)`` so the two never merge.

Three responsibilities, in order:

1. **Calculated** (``mean`` / ``median`` / ``inverse_error``) — read the base
   ``forecast_predictions`` (+ ``forecast_metadata`` for the inverse-error weights / pruning) and
   blend them **in pandas** (`ensembler.combine_calculated`), then append the ``ensemble_<s>``
   rows via the **Storage Write API** — the same append path the learned strategies use (no
   ``INSERT…SELECT`` DML). ``compute_engine='ensemble'``.
2. **Learned** (``nnls`` / ``ridge`` / ``xgb``) — read the base predictions + ``backtest_oof``,
   ``fit_learned`` on the OOF, then apply the weights **in pandas** (``yhat = Σ wₘ·yhatₘ``,
   renormalized over whichever base models are present per ``(ts_id, forecast_date)`` — robust when
   the Spark future window and the native held-out window don't overlap), and append the resulting
   ``ensemble_<s>`` prediction rows via the Storage Write API. Each fitted meta-learner is uploaded
   as a GCS artifact and linked from its scored metadata row. These prediction rows are a **true
   beyond-data forecast** (the base predictions they blend are, too), so — like the base
   models — they carry no ground truth of their own.
3. **Score** — blend the base ``backtest_oof`` into an **ensemble OOF** with the same consensus
   rules (`ensembler.combine_oof`), **append those blended rows back into ``backtest_oof``** (same
   table as the base rows, keyed apart by ``ensemble_id``, so anything computed from that
   table — the pooled comparable leaderboard, a per-horizon coverage read — sees the consensuses
   and the base models on identical footing), and run the shared `metrics.compute_metrics` per
   ``(model, ts_id)`` → ``forecast_metadata`` rows with ``fold_id=NULL`` and
   ``compute_engine='ensemble'``. Scoring lives on the OOF window because the base
   predictions (and therefore every ensemble prediction) are a true beyond-data forecast with no
   actuals to join — so an ensemble earns its leaderboard metric on **exactly the window the base
   models are scored on** (``backtest_oof``, where ``y_true`` lives). OOF carries no interval
   bounds, so ensemble coverage/pinball are unscored — computed as NaN and written as NULL, like
   every other metric this registry stores, consistent with the base models' OOF metrics.
   Learned consensuses are scored on the folds their meta-learner trained on (mildly optimistic,
   the price of stacking having no held-out-of-held-out window). Once these ``fold_id IS NULL`` rows
   land, the leaderboard shows the ensembles automatically — **no view change** beyond the
   ``ensemble_id`` group key.

**Idempotency (append-only + dedupe-on-read).** Every ensemble row is now written through the
Write API and is keyed in ``(run_id, ensemble_id, ts_id, model_type)``, so a re-run of the same
ensemble config re-appends the same cells — correct-but-wasteful when the numbers repeat, and a
genuine conflict when they don't (``xgb`` is a stochastic meta-learner, and a repair re-fits). Both
cases resolve the same way: every ensemble row carries a ``created_at``, and each read dedupes to
one row per cell by ``ORDER BY created_at DESC NULLS LAST`` — see `base_read_sql`, and
`registry.rows.cell_dedup_key` for the same rule on the base tables. That sentence was aspirational
until 2026-09-11: the OOF and metadata rows carried the stamp, the *prediction* rows did not, so the
one table a reader takes the forecast from was the one table with no tiebreak. Every ensemble
prediction row written before that date is NULL and loses to any later one, which is what
``NULLS LAST`` is for. No pre-delete — a ``DELETE``
matching rows still in the ~90-min Write API streaming buffer is rejected for the whole window
(the constraint every cell writer already lives under).
A *different* ensemble config keys distinctly (different ``ensemble_id``), so it never overwrites
and never collides — both coexist.

Public surface: `run_ensembles`, ``python -m scale_forecasting.ensemble_run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .ensembler import combine_calculated, fit_learned

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .config import RunConfig
    from .settings import Settings


# The `backtest_oof` columns the ensemble reads back before blending. `cutoff_date` earns its place
# here: it is the key `ensembler._fold_key` joins models on, and leaving it out of the SELECT does
# not fail anything — it silently drops the ensemble back to the fold ordinal, which on a ragged
# panel pairs nothing across engines and emits a one-model blend under an ensemble's name. A column
# list that has to be right is a column list worth naming once and asserting on.
OOF_READ_COLUMNS: tuple[str, ...] = (
    "ts_id",
    "model_type",
    "fold_id",
    "cutoff_date",
    "forecast_date",
    # Read only so it can be carried back out: the blended rows are written into `backtest_oof`
    # beside the base rows, and a per-horizon read that finds it NULL on the ensembles reports on
    # the base models only. Nothing in the blend itself joins or aggregates on it.
    "horizon_step",
    "y_true",
    "yhat",
)


def base_read_sql(
    dataset: str,
    table: str,
    columns: str,
    model_list: str,
    ts_filter: str,
    extra: str = "",
    dedupe_by: str = "",
) -> str:
    """One run-, model- and series-scoped SELECT over a registry table (pure).

    All three reads this module makes share a shape, and the ``model_type IN (...)`` clause is the
    part that has to be in every one of them. Two of the tables read here — ``forecast_predictions``
    and now ``backtest_oof`` — hold *the ensemble's own rows* beside the base models', so a read
    that omits the filter blends consensuses of consensuses on any second pass over a run: a
    microbatch drain, a ``--force`` re-ensemble, a standalone re-score. Building the clause in one
    place is what stops that from being a thing to remember at each call site.

    ``extra`` is for a clause only one read needs (``forecast_metadata`` wants the full-fit rows
    only); ``ts_filter`` is the microbatch's ``AND ts_id IN UNNEST(@ts_ids)``, empty for a barrier.

    ``dedupe_by`` names the cell grain of the table, and turns the read into "one row per cell,
    newest write wins" via ``QUALIFY ROW_NUMBER() … ORDER BY created_at DESC NULLS LAST``. Three
    things downstream need this and none of them fail loudly without it:
    `ensembler.combine_calculated` pivots with ``pivot_table``, whose default ``aggfunc`` is
    ``mean``, so a duplicated base row is silently *averaged* with itself — harmless for two
    identical copies, wrong the moment a repair re-fit one of them. ``_inverse_error_run_weights``
    takes an ungrouped mean of the decision metric, so a model with duplicate metadata rows is
    weighted by an average of its attempts. And the blended OOF is appended back into
    ``backtest_oof``, so a duplicate on the way in becomes a duplicate on the way out.

    ``NULLS LAST`` is the part that matters for old data: ``created_at`` has only had a writer on
    these two tables since P10, so every row from before it is NULL, and a NULL must lose to any
    real timestamp rather than sorting first and winning permanently.
    """
    qualify = (
        ""
        if not dedupe_by
        else (
            f"\nQUALIFY ROW_NUMBER() OVER ("
            f"\n  PARTITION BY {dedupe_by}"
            f"\n  ORDER BY created_at DESC NULLS LAST"
            f"\n) = 1"
        )
    )
    return (
        f"SELECT {columns}\n"
        f"FROM `{dataset}.{table}`\n"
        f"WHERE run_id = @run_id AND model_type IN ({model_list}){extra}{ts_filter}{qualify}"
    )


def run_ensembles(
    cfg: RunConfig, run_id: str, *, settings: Settings, job_id_prefix: str | None = None
) -> None:  # pragma: no cover - GCP I/O, @gcp ensemble smoke
    """Execute + score every requested ensemble for ``run_id`` into the shared registry (barrier).

    A no-op when ``cfg.ensemble.enabled`` is false. Otherwise blends the calculated + learned
    consensuses in pandas, appends their ``ensemble_<s>`` prediction rows via the Storage Write API,
    and scores every ensemble pseudo-model into ``forecast_metadata`` (see the module docstring),
    all stamped with ``ensemble_id = make_ensemble_id(cfg.ensemble)`` so multiple ensemble configs
    coexist under one ``run_id``. Raises on any failure so `main.run` can finalize the shared
    header FAILED — mirroring how an engine error is surfaced. ``settings`` is the orchestrator's
    already-resolved infra (never re-resolved here, so one identity governs the whole run).

    This is the **barrier** trigger: it blends every series in one pass, after all base jobs have
    joined. The **microbatch** counterpart (`run_ensembles_microbatch`) drains series incrementally
    as each one's base set completes; both call the shared per-batch core (`_ensemble_batch`). They
    land identical rows for the *calculated* strategies; the learned ones are re-fit per batch under
    microbatch and so can differ — see `_ensemble_batch`.
    """
    from datetime import UTC, datetime

    from google.cloud import bigquery

    from .errors import get_logger
    from .registry.ids import make_ensemble_id

    if not cfg.ensemble.enabled:
        return

    log = get_logger(__name__)
    ensemble_id = make_ensemble_id(cfg.ensemble)
    client = bigquery.Client(project=settings.project_id)
    created_at = datetime.now(UTC)
    log.info(
        "ensemble run start (barrier): run_id=%s ensemble_id=%s strategies=%s",
        run_id,
        ensemble_id,
        cfg.ensemble.strategies,
    )
    _ensemble_batch(
        cfg,
        run_id,
        settings=settings,
        client=client,
        ensemble_id=ensemble_id,
        created_at=created_at,
        log=log,
        ts_ids=None,
        job_id_prefix=job_id_prefix,
    )


def run_ensembles_microbatch(
    cfg: RunConfig,
    run_id: str,
    *,
    settings: Settings,
    poll_interval_s: float = 15.0,
    max_polls: int = 960,
    upstream_done: Callable[[], bool] | None = None,
    job_id_prefix: str | None = None,
) -> None:  # pragma: no cover - GCP I/O, @gcp ensemble smoke
    """Execute + score ensembles for ``run_id`` **incrementally, per series as it completes**.

    The microbatch trigger: rather than waiting for every base model on every series (the
    `run_ensembles` barrier), this drains series in ready-batches — a series is *ready* once all
    ``cfg.models`` base models have landed a prediction row for it (`_ready_series`), and each
    ready-batch is blended + scored by the same core as the barrier (`_ensemble_batch` over that
    ts_id subset), so partial consensus output lands progressively and one slow family never blocks
    the series that are already complete. The loop (`_drain_ready`) polls every ``poll_interval_s``
    seconds until ``upstream_done`` reports the base jobs finished *and* no ready series remain
    unprocessed (bounded by ``max_polls`` as a safety stop).

    v1 default: the orchestrator calls this **after** the family join, so ``upstream_done`` defaults
    to "always done" and the loop drains every ready series in a single pass — the streaming
    *semantics* (per-series readiness, incremental writes) with a post-join trigger. Passing an
    ``upstream_done`` that tracks live base-job completion turns it into a consumer that overlaps
    base computation (the concurrent-trigger extension). Raises on any batch failure so `main.run`
    finalizes the header FAILED.
    """
    import time
    from datetime import UTC, datetime

    from google.cloud import bigquery

    from .errors import get_logger
    from .registry.ids import make_ensemble_id

    if not cfg.ensemble.enabled:
        return

    log = get_logger(__name__)
    ensemble_id = make_ensemble_id(cfg.ensemble)
    client = bigquery.Client(project=settings.project_id)
    created_at = datetime.now(UTC)
    done = upstream_done or (lambda: True)
    log.info(
        "ensemble run start (microbatch): run_id=%s ensemble_id=%s strategies=%s",
        run_id,
        ensemble_id,
        cfg.ensemble.strategies,
    )

    def _process(ts_ids: list[str]) -> None:
        _ensemble_batch(
            cfg,
            run_id,
            settings=settings,
            client=client,
            ensemble_id=ensemble_id,
            created_at=created_at,
            log=log,
            ts_ids=ts_ids,
            job_id_prefix=job_id_prefix,
        )

    processed = _drain_ready(
        ready_fn=lambda: _ready_series(
            cfg, run_id, settings=settings, client=client, job_id_prefix=job_id_prefix
        ),
        process_fn=_process,
        done_fn=done,
        sleep_fn=lambda: time.sleep(poll_interval_s),
        max_polls=max_polls,
    )
    log.info(
        "ensemble run done (microbatch): run_id=%s ensemble_id=%s series_drained=%d",
        run_id,
        ensemble_id,
        len(processed),
    )


def _drain_ready(
    *,
    ready_fn: Callable[[], set[str]],
    process_fn: Callable[[list[str]], None],
    done_fn: Callable[[], bool],
    sleep_fn: Callable[[], None],
    max_polls: int,
) -> set[str]:
    """Drive incremental per-series draining until upstream is done and nothing new is ready (pure).

    Each poll takes the currently-ready series (`ready_fn`) minus those already processed; if any
    are new, they're handed to ``process_fn`` (in sorted order, for determinism) and marked done. If
    none are new *and* ``done_fn`` reports upstream finished, the drain is complete; if none are new
    but upstream is still running, it waits (``sleep_fn``) and polls again. ``max_polls`` bounds the
    loop so a stuck upstream can't spin forever. All I/O is injected, so this is unit-testable with
    plain callables. Returns the set of series processed.
    """
    processed: set[str] = set()
    for _ in range(max_polls):
        pending = sorted(ready_fn() - processed)
        if pending:
            process_fn(pending)
            processed.update(pending)
        elif done_fn():
            break
        else:
            sleep_fn()
    return processed


def _ready_series(
    cfg: RunConfig,
    run_id: str,
    *,
    settings: Settings,
    client: Any,
    job_id_prefix: str | None = None,
) -> set[str]:  # pragma: no cover - GCP I/O, @gcp ensemble smoke
    """The ``ts_id``s whose full base-model set has landed a prediction for ``run_id``.

    A series is ready to ensemble once every ``cfg.models`` base model has appended at least one
    ``forecast_predictions`` row for it — counted with ``COUNT(DISTINCT model_type)`` against the
    configured model count. This is the microbatch readiness signal (the registry is the completion
    source of truth; no separate event bus).
    """
    from google.cloud import bigquery

    dataset = settings.registry_dataset_ref
    models = list(cfg.models)
    model_list = ", ".join(f"'{m}'" for m in models)
    sql = (
        "SELECT ts_id\n"
        f"FROM `{dataset}.forecast_predictions`\n"
        f"WHERE run_id = @run_id AND model_type IN ({model_list})\n"
        "GROUP BY ts_id\n"
        f"HAVING COUNT(DISTINCT model_type) = {len(models)}"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    )
    rows = client.query(sql, job_config=job_config, job_id_prefix=job_id_prefix).result()
    return {str(r["ts_id"]) for r in rows}


def _ensemble_batch(
    cfg: RunConfig,
    run_id: str,
    *,
    settings: Settings,
    client: Any,
    ensemble_id: str,
    created_at: Any,
    log: Any,
    ts_ids: list[str] | None,
    job_id_prefix: str | None = None,
) -> None:  # pragma: no cover - GCP I/O, @gcp ensemble smoke
    """Blend + score the ensembles for one batch of series (``ts_ids=None`` = every series).

    The shared core behind both triggers: reads the batch's base predictions / OOF / decision-metric
    rows (filtered to ``ts_ids`` when given), blends the calculated + learned consensuses in pandas,
    appends the ``ensemble_<s>`` prediction rows via the Storage Write API, and scores each
    pseudo-model into ``forecast_metadata`` — identical logic whether the caller passes one ready
    series (microbatch) or all of them (barrier).

    **Identical logic is not identical numbers, and the difference is confined to the learned
    strategies.** The calculated ones (``mean``, ``median``, ``inverse_error``) are per-series, so
    partitioning the series changes nothing. The learned ones are not: `fit_learned` below trains on
    whatever OOF this call was handed, so microbatch fits ``nnls``/``ridge``/``xgb`` **once per
    ready-batch on that batch's series**, while barrier fits once over all of them. Different
    training sample, different weights. Confirmed live on 2026-09-02 by running smokes 11 and 12
    back to back on the same data: the three calculated strategies agreed to float noise and
    ``ensemble_nnls`` differed in the fourth decimal (see `docs/validation.md`).

    That first sentence was only two-thirds true until 2026-09-11, and re-running the same pair
    caught it. ``mean`` and ``median`` were per-series and came back bit-identical, but
    ``inverse_error``'s *future* blend took its weights from a run-wide
    ``groupby("model_type").mean()`` over this call's metric rows — which microbatch filters to the
    batch's series — so it moved on every one of 2,800 rows while its OOF-scored counterpart, which
    really is per-series, matched exactly. The leaderboard therefore looked clean while the shipped
    forecasts disagreed. `ensembler._inverse_error_weight_matrix` now estimates those weights per
    series, and the claim above holds for all three.

    Neither answer is wrong, but the microbatch one depends on how series happened to batch, which
    depends on job timing — so it is not reproducible the way the rest of a run is. Deciding whether
    learned strategies should defer to a final global fit is a design question, deliberately left
    open here rather than changed mid-campaign; this docstring's job is to stop the next reader
    assuming the two triggers are interchangeable for stacking.
    """

    from google.cloud import bigquery

    from .backtest import training_window
    from .engines import bigquery_engine
    from .engines.bigquery_sql import build_history_query
    from .ensembler import combine_oof
    from .metrics import compute_metrics
    from .registry.artifacts import upload_artifact_bytes
    from .registry.header import merge_header_telemetry
    from .registry.rows import assemble_ensemble_oof_rows, stamp_ensemble_prediction_rows
    from .registry.write_api import _META_SPEC, _OOF_SPEC, _PRED_SPEC
    from .seasonality import seasonal_period
    from .worker import _rollup_metrics

    # Every table read here (forecast_predictions / backtest_oof / forecast_metadata) is a registry
    # table — the ensemble reads run outputs, never the source panel.
    dataset = settings.registry_dataset_ref

    ts_filter = "" if ts_ids is None else " AND ts_id IN UNNEST(@ts_ids)"

    def _query(sql: str) -> Any:
        params: list[Any] = [bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
        if ts_ids is not None:
            params.append(bigquery.ArrayQueryParameter("ts_ids", "STRING", ts_ids))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        return client.query(sql, job_config=job_config, job_id_prefix=job_id_prefix).result()

    models = list(cfg.models)
    model_list = ", ".join(f"'{m}'" for m in models)
    metric = cfg.backtest.decision_metric
    base_pred_sql = base_read_sql(
        dataset,
        "forecast_predictions",
        "ts_id, model_type, forecast_date, yhat, yhat_lower, yhat_upper",
        model_list,
        ts_filter,
        # The base models' rows carry a NULL `ensemble_id`, and the `model_type IN (...)` filter
        # already excludes the ensemble's own; the grain is still partitioned on it so a future
        # caller that widens the model list cannot collapse two ensembles into one row.
        dedupe_by="run_id, ts_id, model_type, ensemble_id, forecast_date",
    )
    oof_sql = base_read_sql(
        dataset,
        "backtest_oof",
        ", ".join(OOF_READ_COLUMNS),
        model_list,
        ts_filter,
        dedupe_by="run_id, ts_id, model_type, ensemble_id, fold_id, forecast_date",
    )
    metric_sql = base_read_sql(
        dataset,
        "forecast_metadata",
        # `backtest_refit` rides along on the read the decision metric already needs, so the
        # ensemble can say how its members were scored without a fourth query — see
        # `ensemble_refit_mode`.
        f"ts_id, model_type, backtest_refit, {metric}",
        model_list,
        ts_filter,
        extra=" AND fold_id IS NULL",
        # `fold_id` stays in the grain even though `extra` pins it to NULL: the partition describes
        # the table's cell identity, and a read that changes its filter should not silently change
        # what counts as a duplicate.
        dedupe_by="run_id, ts_id, model_type, ensemble_id, fold_id",
    )
    base_df = _query(base_pred_sql).to_dataframe()
    oof_df = _query(oof_sql).to_dataframe()
    metric_df = _query(metric_sql).to_dataframe()

    # A single Write-API append of every ensemble prediction row (calculated + learned). Both
    # blenders are pure and config-only, so every run-scoped column is stamped afterwards, in the
    # one call below — see `stamp_ensemble_prediction_rows` for why that is not two inline loops.
    pred_rows: list[dict[str, Any]] = []

    # 1. Calculated ensembles — blend the base predictions in pandas (Write API, not DML).
    pred_rows.extend(
        {**row, "quantiles": None} for row in combine_calculated(base_df, cfg, metric_df)
    )

    # 2. Learned ensembles — fit on the OOF, apply the weights in pandas, append prediction rows.
    artifact_uris: dict[str, str] = {}
    learned_weights, artifacts, learned_basis = fit_learned(oof_df, cfg)
    for strategy, wmap in learned_weights.items():
        pred_rows.extend(_apply_weights(base_df, wmap, run_id, strategy))
        artifact_uris[strategy] = upload_artifact_bytes(
            artifacts[strategy],
            f"ensemble_{ensemble_id}_{strategy}.pkl",
            run_id,
            settings.artifact_root,
        )

    stamp_ensemble_prediction_rows(
        pred_rows, run_id=run_id, ensemble_id=ensemble_id, created_at=created_at
    )
    if pred_rows:
        bigquery_engine._append_rows(settings, "forecast_predictions", _PRED_SPEC, pred_rows)
    log.info(
        "ensemble predictions appended: run_id=%s ensemble_id=%s rows=%d strategies=%s",
        run_id,
        ensemble_id,
        len(pred_rows),
        cfg.ensemble.strategies,
    )

    # 3. Score every ensemble_* pseudo-model into forecast_metadata (fold_id=NULL). The base
    #    predictions are a true beyond-data forecast (no actuals to join), so ensembles earn their
    #    metric on the backtest OOF window — the same window the base models are scored on. We blend
    #    the base OOF with the same consensus rules, then compute_metrics per (model, ts_id) and
    #    roll the folds up exactly as worker.py does for the base models.
    if oof_df.empty:
        log.warning("ensemble scoring: backtest_oof empty for run_id=%s — nothing to score", run_id)
        return
    ens_oof = combine_oof(oof_df, cfg, learned_weights)
    if ens_oof.empty:
        log.warning("ensemble scoring: no ensemble OOF produced for run_id=%s", run_id)
        return

    # Persist the blended OOF before scoring it. The metrics panel below is a summary; these are the
    # rows it summarizes, and they are what `v_model_leaderboard_comparable` pools over — an
    # ensemble that only ever wrote its panel is missing from every comparison computed from
    # `backtest_oof`, which is not the same as ranking badly in one.
    oof_rows = assemble_ensemble_oof_rows(ens_oof, run_id, ensemble_id, created_at)
    bigquery_engine._append_rows(settings, "backtest_oof", _OOF_SPEC, oof_rows)
    log.info(
        "ensemble OOF appended: run_id=%s ensemble_id=%s rows=%d",
        run_id,
        ensemble_id,
        len(oof_rows),
    )

    # y_train (for MASE/RMSSE scale) is per-series history; the base OOF has no in-sample rows, so
    # read the full series history once, matching the natives' history read. The dates come with it
    # because the scale denominator is the *fold's* training window, not the whole series — see
    # `backtest.training_window`. `build_history_query` orders by ts_id, ds, so groups are sorted.
    history = _query(build_history_query(cfg, dataset)).to_dataframe()
    hist_by_id = {
        tid: (pd.to_datetime(g["ds"]).to_numpy(), g["y"].to_numpy())
        for tid, g in history.groupby("ts_id")
    }

    period = seasonal_period(cfg.data.freq)
    # Per-series list of how each base model was actually backtested, for `ensemble_refit_mode`.
    refit_by_id: dict[str, list[str | None]] = {
        tid: list(g["backtest_refit"]) for tid, g in metric_df.groupby("ts_id")
    }
    meta_rows: list[dict[str, Any]] = []
    for (model_type, ts_id), g in ens_oof.groupby(["model_type", "ts_id"]):
        # Score per fold, then roll up (NaN-ignoring mean) — identical to the base-model path
        # (worker._rollup_metrics), so ensemble and base metrics are computed the same way.
        hist = hist_by_id.get(ts_id)
        fold_panels: list[dict[str, float]] = []
        for _fold, fg in g.sort_values("forecast_date").groupby("fold_id"):
            # `combine_oof` joins base models on the cutoff, so it is constant within a fold; `min`
            # only matters on the ordinal fallback, where taking the earliest cutoff is the choice
            # that cannot let a later fold's data into an earlier fold's denominator.
            cutoff = fg["cutoff_date"].min() if "cutoff_date" in fg.columns else None
            fold_panels.append(
                compute_metrics(
                    fg["y_true"].to_numpy(),
                    fg["yhat"].to_numpy(),
                    y_train=None
                    if hist is None
                    else training_window(hist[0], hist[1], cutoff, cfg),
                    seasonal_period=period,
                )
            )
        strategy = str(model_type).removeprefix("ensemble_")
        meta_rows.append(
            _ensemble_meta_row(
                run_id=run_id,
                ts_id=str(ts_id),
                model_type=str(model_type),
                panel=_rollup_metrics(fold_panels),
                ensemble_id=ensemble_id,
                weights=learned_weights.get(strategy),
                artifact_uri=artifact_uris.get(strategy),
                created_at=created_at,
                cfg=cfg,
                ensemble_scoring=ensemble_scoring_basis(strategy, oof_df, cfg, learned_basis),
                backtest_refit=ensemble_refit_mode(refit_by_id.get(str(ts_id), [])),
            )
        )
    bigquery_engine._append_rows(settings, "forecast_metadata", _META_SPEC, meta_rows)
    # The run-level counterpart of the per-row `ensemble_scoring`, next to `$.scoring.hpo` that
    # `main.run` writes. Best-effort: the rows are already written and they carry the same answer
    # per strategy, so a failed merge costs legibility, not evidence.
    claim = run_ensemble_scoring(row["ensemble_scoring"] for row in meta_rows)
    if claim is not None:
        try:
            merge_header_telemetry(run_id, {"scoring.ensemble": claim}, settings=settings)
        except Exception as exc:  # noqa: BLE001 - telemetry is never worth failing a run over
            log.warning("ensemble scoring telemetry not recorded for %s: %s", run_id, exc)
    log.info(
        "ensemble run done: run_id=%s ensemble_id=%s scored=%d models=%s",
        run_id,
        ensemble_id,
        len(meta_rows),
        sorted(ens_oof["model_type"].unique()),
    )


def run_ensemble_scoring(bases: Iterable[str | None]) -> str | None:
    """Roll the per-strategy scoring bases up into one claim for the run (pure).

    ``None`` when nothing in the run fit anything — a mean/median-only ensemble has no fold to
    reserve, so there is no claim to make and the header stays quiet rather than saying "holdout"
    about a fit that never happened. Otherwise the **weakest** answer wins: one strategy that fell
    back to ``in_sample`` makes the run's ensemble metrics partly in-sample, and a header that
    reported the best case would be exactly the reassurance this column exists to withhold.
    """
    seen = {b for b in bases if b is not None}
    if not seen:
        return None
    return "in_sample" if "in_sample" in seen else "holdout"


def ensemble_refit_mode(modes: Iterable[str | None]) -> str | None:
    """How this ensemble's *members* were scored, rolled up into one answer (pure).

    A blend inherits its members' backtest — an ``ensemble_mean`` over three frozen models is a
    frozen result, and over two frozen models and one that refit at every origin it is neither.
    So: unanimous members give the ensemble their mode, disagreement gives ``"mixed"``, and no
    member rows at all give ``None``.

    ``"mixed"`` is a fifth value that only ensemble rows can carry, and it is the point of the
    exercise. NULL already means "no backtest happened here", so reusing it for disagreement would
    hide the one case a reader most needs to see: a row on an ``expanding_frozen`` leaderboard
    whose members were not all frozen. Saying so is the same instinct as `run_ensemble_scoring`
    refusing to report the best case.
    """
    seen = {m for m in modes if m is not None}
    if not seen:
        return None
    return seen.pop() if len(seen) == 1 else "mixed"


def ensemble_scoring_basis(
    strategy: str, oof_df: pd.DataFrame, cfg: RunConfig, learned_basis: str
) -> str | None:
    """Whether this strategy's metrics were earned with the newest fold held out of its fit (pure).

    ``None`` for ``mean`` and ``median``: they fit nothing, so there is no fold to reserve and no
    honesty question to answer. Writing ``"holdout"`` there would be a claim about a split that
    played no part in the number.

    ``inverse_error`` fits per-series weights inside `ensembler.combine_oof`, so it takes the OOF
    frame's own verdict. The learned strategies take ``learned_basis``, which
    `ensembler.fit_learned` already resolved — including the case where the inner folds existed
    but produced no complete training row.
    """
    from .config import LEARNED_STRATEGIES
    from .ensembler import inner_fold_mask

    if strategy in LEARNED_STRATEGIES:
        return learned_basis
    if strategy == "inverse_error":
        return inner_fold_mask(oof_df, cfg)[1]
    return None


def _ensemble_meta_row(
    *,
    run_id: str,
    ts_id: str,
    model_type: str,
    panel: dict[str, float],
    ensemble_id: str,
    weights: dict[str, float] | None,
    artifact_uri: str | None,
    created_at: Any,
    cfg: RunConfig,
    ensemble_scoring: str | None,
    backtest_refit: str | None,
) -> dict[str, Any]:
    """Assemble one ``forecast_metadata`` row for an ``ensemble_<strategy>`` pseudo-model (pure).

    The third independent producer of ``forecast_metadata`` rows, alongside `registry.rows`'s
    Python-worker path and `engines.bigquery_engine._meta_row`. `write_api._encode_rows` walks the
    column spec rather than the row, so a key here that ``_META_SPEC`` does not name is dropped in
    silence — which is why this is a function with its own offline assertion rather than a literal
    inside the GCP body.

    ``weights`` is the fitted weight map for a *learned* strategy and ``None`` for a calculated one
    (mean/median/inverse_error have no fitted parameters); the distinction is what ``best_params``
    and ``model_artifact`` record. An empty-but-present weight map still serializes, so a strategy
    that legitimately fitted to nothing is distinguishable from one that never fitted at all.
    """
    import json

    from .metrics import METRIC_NAMES
    from .registry.ids import make_model_hash
    from .registry.rows import _as_float

    return {
        "run_id": run_id,
        "ts_id": ts_id,
        "model_type": model_type,
        "compute_engine": "ensemble",
        "model_hash": make_model_hash(run_id, ts_id, model_type, cfg),
        "ensemble_id": ensemble_id,
        "fold_id": None,
        # Through `_as_float` like every other metric write: an unscored or non-finite metric is
        # NULL, never NaN. See `bigquery_engine._meta_row` for what the raw passthrough cost.
        **{name: _as_float(panel[name]) for name in METRIC_NAMES},
        "fit_seconds": None,
        "best_params": None if weights is None else json.dumps(weights, sort_keys=True),
        "model_artifact": artifact_uri,
        "created_at": created_at,
        # Same reasoning as the native row: a blend that failed produces no row, so an ensemble row
        # that exists is an "ok" one, and saying so keeps `WHERE cell_status = 'ok'` honest across
        # every engine. The two error columns stay NULL — there is no ensemble error row to fill.
        "cell_status": "ok",
        "error_class": None,
        "error_detail": None,
        # Whether the metrics above were earned with the newest fold kept out of whatever this
        # strategy fitted. NULL where the strategy fits nothing — see `ensemble_scoring_basis`.
        "ensemble_scoring": ensemble_scoring,
        # Inherited from the members this series' blend was built from — see `ensemble_refit_mode`.
        "backtest_refit": backtest_refit,
        # No ensemble control arm exists to compare against: `ensembler.combine_oof` blends the
        # primary arm's ``yhat`` and never reads ``yhat_stale``, so there is no blind blend whose
        # loss this could be the difference from. NULL is the honest answer, not zero.
        "staleness_gap": None,
    }


def _apply_weights(
    base_df: pd.DataFrame,
    wmap: dict[str, float],
    run_id: str,
    strategy: str,
) -> list[dict[str, Any]]:
    """Blend base predictions by learned weights → ``ensemble_<strategy>`` prediction rows (pure).

    ``base_df`` is long-format ``(ts_id, model_type, forecast_date, yhat, yhat_lower, yhat_upper)``.
    For each ``(ts_id, forecast_date)`` the weighted mean is taken over **whichever base models are
    present**, with the weights renormalized to sum to 1 over that present subset — so a date where
    only some base models forecast (the disjoint Spark-future / native-held-out case) still yields a
    well-defined blend rather than a NULL. Rows where no weighted base model is present, or the
    present weights sum to zero, are dropped. Pure (no GCP) so it is unit-tested offline."""
    model_type = f"ensemble_{strategy}"
    rows: list[dict[str, Any]] = []
    if base_df.empty:
        return rows
    models = [m for m in wmap if m in set(base_df["model_type"].unique())]
    if not models:
        return rows
    weights = np.array([wmap[m] for m in models], dtype=float)

    def _blend(value_col: str) -> Any:
        wide = base_df.pivot_table(
            index=["ts_id", "forecast_date"], columns="model_type", values=value_col
        ).reindex(columns=models)
        vals = wide.to_numpy(dtype=float)
        present = ~np.isnan(vals)
        wrow = present * weights  # zero out absent models per row
        denom = wrow.sum(axis=1)
        num = np.nansum(np.where(present, vals, 0.0) * weights, axis=1)
        blended = np.where(denom > 0.0, num / np.where(denom > 0.0, denom, 1.0), np.nan)
        return wide.index, blended

    index, yhat = _blend("yhat")
    _, lower = _blend("yhat_lower")
    _, upper = _blend("yhat_upper")
    for (ts_id, forecast_date), yh, lo, up in zip(index, yhat, lower, upper, strict=True):
        if np.isnan(yh):
            continue
        rows.append(
            {
                "run_id": run_id,
                "ts_id": ts_id,
                "model_type": model_type,
                "compute_engine": "ensemble",
                "forecast_date": forecast_date,
                "yhat": float(yh),
                "yhat_lower": None if np.isnan(lo) else float(lo),
                "yhat_upper": None if np.isnan(up) else float(up),
                "quantiles": None,
            }
        )
    return rows


# --- standalone CLI (re-ensemble an already-completed run) ---------------------


def _override_ensemble(cfg: RunConfig, strategies: list[str] | None) -> RunConfig:
    """Return ``cfg`` with its ``ensemble`` block overridden to ``strategies`` (pure).

    ``None`` leaves the config's own ensemble block untouched; a list rebuilds
    `EnsembleConfig` (so the strategies are **re-validated**
    against the known calculated/learned sets and enabled), preserving the original
    ``prune_threshold``. This is what lets one base run be ensembled several ways from the CLI —
    each override is a distinct ``EnsembleConfig`` and thus a distinct ``ensemble_id`` per run.
    """
    from .config import EnsembleConfig

    if strategies is None:
        return cfg
    # Rebuild via validated model construction so CLI strings are checked against the known
    # strategy set (raises a clear pydantic error on a typo) rather than trusted blindly.
    ensemble = EnsembleConfig.model_validate(
        {
            "enabled": True,
            "strategies": strategies,
            "prune_threshold": cfg.ensemble.prune_threshold,
        }
    )
    return cfg.model_copy(update={"ensemble": ensemble})


def _main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    """``python -m scale_forecasting.ensemble_run --config c.json [--run-id …] [--strategies …]``.

    Re-runs the ensemble stage against an *already-completed* run's base predictions — the
    standalone counterpart to the inline call `main.run` makes. Loads the config, optionally
    overrides the ensemble strategies (``--strategies mean,median``), resolves the infra identity
    from the ``SF_*`` environment (``--sf-*`` promoted first), and calls `run_ensembles`.

    ``--run-id`` is the **base run whose forecasts are blended**; it defaults to the config's own
    ``make_run_id`` (re-ensembling the run that config produced), but is passed explicitly to
    ensemble a run whose base models were computed under a *different* config revision. The
    ``ensemble_id`` is always derived from the (possibly overridden) ensemble block, so re-running
    with new ``--strategies`` lands a *new* ``ensemble_id`` beside the existing one — never
    overwriting (append-only), never colliding (distinctly keyed).
    """
    import argparse

    from ._infra_args import add_infra_args, export_infra_env
    from .config import load_config
    from .errors import get_logger
    from .registry.ids import make_run_id
    from .settings import Settings

    parser = argparse.ArgumentParser(
        prog="ensemble_run", description="Re-run the ensemble stage for a completed run."
    )
    parser.add_argument("--config", required=True, help="path to the run config JSON")
    parser.add_argument(
        "--run-id",
        default=None,
        help="base run to ensemble (default: derived from --config via make_run_id)",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="comma-separated override of ensemble.strategies (default: the config's own block)",
    )
    add_infra_args(parser)
    ns = parser.parse_args(argv)
    export_infra_env(ns)

    strategies = (
        [s.strip() for s in ns.strategies.split(",") if s.strip()]
        if ns.strategies is not None
        else None
    )
    cfg = _override_ensemble(load_config(ns.config), strategies)
    run_id = ns.run_id or make_run_id(cfg)
    get_logger(__name__).info(
        "ensemble_run CLI: run_id=%s strategies=%s", run_id, cfg.ensemble.strategies
    )
    run_ensembles(cfg, run_id, settings=Settings.resolve())


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    _main()
