"""Analyst-facing SQL views over the registry.

The three-tier registry stores raw rows; a data scientist reviewing a run shouldn't have to
re-derive the same roll-ups every time. These views are the *curated read surface* — the point
where "rows in BigQuery" become "assets you review". They are pure ``CREATE OR REPLACE VIEW``
strings (no client), so they render + snapshot-test offline exactly like the table DDL, and
``registry.tables.ensure_views`` executes what this renders.

Five core operational & leaderboard views, plus six server-side diagnostic & S&OP views (11 views
total), matched to the questions a run prompts:

- ``v_run_summary`` — *how did each run go, and how efficiently?* One row per run: the scaling
  knobs (``n_series``, ``n_models``), the header's whole-run ``runtime_seconds``, and a **time
  ledger rolled up from the run's job rows**: ``n_jobs``, ``longest_job_seconds`` (the
  critical-path job), ``jobs_seconds`` (every job's wall added up — families run in parallel, so
  this exceeds the run's own wall when the DAG overlapped), ``jobs_span_seconds`` (first job start
  to last job end), and ``overhead_seconds = jobs_span_seconds − longest_job_seconds`` with its
  share ``overhead_fraction``. That overhead is the wall inside the run's job window that the
  slowest family did not account for: the barrier ensemble, the native BigQuery job, family start
  skew, and — under Composer — the scheduler's gaps between tasks. It means the same thing
  whichever tier finalized the header, which is why it is derived from the job rows rather than
  from header columns: the header's ``runtime_seconds`` is the launcher's wall under `main.run`
  but the slowest job under Composer (`airflow_tasks.finalize_run`), and the platform wall a family
  batch stamps on the header (``$.total_wall_s``) is one family's, not the run's — the earlier
  ``total_wall_s − runtime_seconds`` definition compared those two and came out ≤ 0 on nearly
  every run.

  Three rules make the ledger exact rather than approximately right. Every term is measured on the
  job rows' own ``started_at`` → ``ended_at`` timestamps (not on ``runtime_seconds``, which the
  launcher reads off a monotonic clock — mixing the two left overhead a few milliseconds negative
  under clock slew), so a span contains its longest member and ``overhead_seconds`` is ≥ 0 by
  arithmetic; the bracket exceeds the row's ``runtime_seconds`` by the row-write latency, which is
  the only difference between them. The rows are the **current attempt per family** (the same
  highest-attempt rule ``v_run_jobs`` applies) **written after the header row** — the summary
  describes the header's own launch, so a run resumed hours later is not charged the idle time
  between its launches. And **repair rows are excluded** (families ending in
  `registry.ids.REPAIR_SUFFIX`): a repair is filed days later under its own token, writes no
  header, and would otherwise stretch the span to cover the gap. A header with no job row in its
  launch (a run that failed before its first family launched, or one written by an older release)
  carries NULL in the ledger and still renders.

  The executor columns (``executor_instances`` / ``executor_cores`` / ``max_executors`` /
  ``executor_memory`` / ``executor_memory_overhead``), ``dcu_milli_seconds``, and
  ``runtime_version`` are unpacked from the header's ``job_telemetry``, and those keys are written
  by **the run's Dataproc Serverless batches, last to finish wins** — see
  `batch_telemetry._stamp_job_telemetry`, which merges by key; a Ray job or a Dataproc cluster job
  stamps other keys. So they are the echoed shape of *a* family batch: exact for a single-family
  Serverless run, one family's answer for a mixed run, and NULL for a run that submitted no
  Serverless batch.
  The per-family decision — one entry per family under ``$.sizing`` — is the whole story, left as
  raw ``JSON`` because its interesting parts are nested: read a run's shape from the scalar
  columns, read *why*, per family, with ``JSON_QUERY(sizing, '$.deep_learning')``
  (`resources.audit.sizing_telemetry`). NULL on a run submitted before this existed, or one that
  left the platform's own defaults standing. ``capacity`` is the same shape for the other run-level
  wait: one attempt ledger per *shared* cluster, keyed by service
  (`shared_clusters.shared_capacity_path`), recording every region tried before one had room. A
  per-family walk is on the family's row instead — see ``v_run_jobs``. A forced re-run of an
  unchanged config appends a second header row under the same ``run_id``; the view keeps only the
  latest (``QUALIFY ROW_NUMBER() … ORDER BY created_at DESC NULLS LAST = 1``) so one run is always
  one row.

- ``v_run_jobs`` — *what jobs ran for this run, on what runtime/hardware, and how did each fare?*
  One row per ``(run_id, family)`` = the run's DAG as executed: the deterministic ``job_id``, the
  resolved ``runtime`` / ``spark_mode`` / ``hardware`` / ``gpu_type``, the platform's own
  ``system_job_id``, the per-job ``status``, its ``started_at`` / ``ended_at`` bracket and
  ``runtime_seconds`` — the launcher's wall around the platform job, from submit to terminal,
  which is what ``v_run_summary``'s ledger is built from. ``provision_seconds`` and
  ``execute_seconds`` split that wall the way the platform saw it: how long it took to find and
  start capacity before any of our code ran, and how long the work ran once it had
  (``job_telemetry.$.platform``, stamped by `job_launch.launch_family_job` from one probe read of
  the finished job — see `batch_telemetry.platform_phases`). They are populated for Dataproc
  Serverless and Vertex ``CustomJob`` families, whose APIs expose the start transition, and NULL
  for the rest; a long ``provision_seconds`` next to a short ``execute_seconds`` is the signature
  of a family that should move runtime, region, or machine shape. ``failure_reason`` says *why* a
  FAILED row failed
  (``CAPACITY_EXHAUSTED`` is the first token) and ``capacity`` carries the whole attempt ledger —
  every candidate tried, its verdict, and the cloud's verbatim message. Both are NULL for a job
  that never had to wait, which is nearly all of them. ``device_verdict`` is the one word that says
  whether the accelerator on this row's ``hardware`` did any work (`device_audit`) — NULL for every
  CPU family, so ``WHERE device_verdict != 'ENGAGED_UTILISED'`` is the "what did I pay for and not
  use" query — and ``device_use`` beside it carries the counts and the peak byte figure the word was
  decided from. The platform's DCU and cluster-shape figures are *not* on this row: the batch
  telemetry lands on the header (above), and a column that unpacked it from here would read NULL
  on every job. A ``--force`` re-run appends a
  higher-``attempt`` job under the same ``(run_id, family)``; the view keeps only the current one
  (``QUALIFY ROW_NUMBER() … ORDER BY attempt DESC = 1``), so the forward ``run_id → current job``
  map is one row per family.

- ``v_model_leaderboard`` — *which model won, per run?* One row per ``(run_id, model_type,
  ensemble_id)``: cell counts, the error rate (a model failing every cell — the libgomp/lightgbm
  class of problem — shows as ``error_rate = 1.0``), median fit time, and the mean decision metrics
  where a backtest populated them. The entry point for "is this model worth keeping" before
  ensembling. ``ensemble_id`` is NULL for base models (so they group exactly as before) and the
  ``EnsembleConfig`` digest for ensemble pseudo-models — so two ensemble configs scored under one
  ``run_id`` keep their ``ensemble_<strategy>`` rows distinct instead of collapsing into one.
  Result writes are append-only and at-least-once (a task retry or a ``--force`` re-run can
  re-append a cell), so — like the two views above — the leaderboard first collapses to one row
  per cell (``ROW_NUMBER() … PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id ORDER BY
  created_at DESC NULLS LAST = 1``, latest write wins) before aggregating; otherwise a duplicated
  cell would double-count and skew ``mean_wape`` / ``mean_mae`` / ``n_cells``.
  ``mean_staleness_gap`` is non-NULL only under the frozen backtest schemes, and reads as "what
  this model loses, in the run's decision metric, if it is never refit" — the column that turns
  refit cadence from a guess into a number. ``refit_modes`` beside it says whether the cohort
  earned that ranking the way the run asked; see the note under
  ``v_model_leaderboard_comparable``, which carries the same column.

- ``v_backtest_coverage`` — *how much of the panel did each model actually get scored on?* The
  question ``v_model_leaderboard`` cannot answer, and the one that decides whether its ranking
  means anything. A ragged panel does not give every series the same number of folds:
  `backtest.make_folds` drops the oldest folds a short series cannot afford, so one model's
  ``mean_wape`` can be an average over ten folds of two thousand series while its neighbour's is
  an average over one fold of two hundred. One row per ``(run_id, model_type, ensemble_id,
  backtest_status, n_folds_achieved)`` with the series count and its share of that model's panel —
  long-format rather than one wide row per model, because the achieved-fold histogram has no fixed
  width and a wide shape would need a self-join that breaks on the NULL ``ensemble_id`` of every
  base model. ``backtest_status`` is
  ``full`` / ``reduced`` / ``unscored`` / ``failed``, or NULL where the run never asked for a
  backtest at all. Read it beside the leaderboard: a model whose panel is mostly ``reduced`` won on
  an easier question.

  ``backtest_refit`` is in the grouping for the same reason ``n_folds_achieved`` is — it is the
  other way two rows of the same leaderboard can be answers to different questions. Under
  ``expanding_frozen`` a model without the re-conditioning seam falls back to a fresh fit per fold
  and lands here as ``unsupported``; that row's error is a refit model's error sitting next to
  frozen ones. Under the two refit schemes every row reads ``per_fold`` and the column adds
  nothing, which is the correct amount for it to add. ``mean_staleness_gap`` rides along on the
  same rows: what that cohort loses, in the run's decision metric, when the model is never refit.
  It is populated on the frozen schemes and on any refit scheme that set
  ``backtest.control_arm``, so a ``per_fold`` row carrying a gap is not a contradiction — it is a
  refit run that also scored the never-refreshed counterfactual.

- ``v_model_leaderboard_comparable`` — *which model won, holding the question fixed?* The same
  ranking as ``v_model_leaderboard``, rebuilt so the numbers are comparable across models rather
  than merely present. Two differences, and both are the point. First, it is restricted to the
  **holdout fold** — the newest one, ``MAX(fold_id) OVER (PARTITION BY run_id)``, which every series
  that achieved any fold achieved, because `backtest.make_folds` keeps a survivor's original
  ``fold_id`` and drops from the oldest end. That is derived from the rows rather than from the
  config on purpose: the view has no config to read. Second, the error is **pooled, not averaged** —
  ``SUM(|y_true − yhat|) / SUM(|y_true|)`` over every series at once, so a fleet number is one WAPE
  of the whole panel instead of the mean of per-series WAPEs, where a single near-zero series can
  dominate. It reads ``backtest_oof`` rather than ``forecast_metadata`` because that is the only
  table holding per-fold truth; ``forecast_metadata`` stores one rolled-up row per cell and cannot
  be restricted to a fold after the fact. Ensembles appear here beside the base models because
  `ensemble_run` writes its blended OOF into the same table (keyed by ``ensemble_id``). Carry
  ``n_series`` into any comparison you draw from it — equal ``n_series`` across the rows is the
  evidence that the models answered the same question, and unequal ``n_series`` is a finding.
  Like the views above it collapses to one row per cell before aggregating, because a task
  retry can re-append rows and a *partial* duplication skews a pooled ratio (a uniform one does
  not — it doubles both sides). Every cell-table writer now stamps ``created_at``, so the
  ``ORDER BY created_at DESC NULLS LAST`` tiebreak is a real newest-wins rule rather than an
  arbitrary pick: a repaired cell re-fits the model, and a stochastic learner does not reproduce
  its old numbers to the bit, so the duplicate pair is a genuine conflict and the repair has to
  win it. ``NULLS LAST`` is what makes that hold across the seam — rows written before the
  stamp existed carry NULL and must lose to any row that carries a timestamp.

  ``refit_modes`` is on both leaderboards, as a sorted ``STRING_AGG(DISTINCT …)`` rather than in
  the grouping. It is the answer to "was every row in this ranking scored the same way?" without
  splitting a model into two rows to say so: ``recondition`` means the whole cohort was frozen,
  ``recondition,unsupported`` means some of it refit instead and this ranking is mixing two
  questions. The comparable view has to reach into ``forecast_metadata`` for it — ``backtest_oof``
  is per-row truth and carries no per-cell refit mode — and joins on ``COALESCE(ensemble_id, '')``
  because ``ensemble_id`` is NULL on every base model and NULL never equals NULL. A ``LEFT`` join
  so a model whose metadata row is missing still ranks, with a NULL ``refit_modes`` rather than
  vanishing from the leaderboard.

``JSON_VALUE`` reads scalars straight out of the native ``JSON`` ``job_telemetry`` column (the
registry is native BigQuery, so the column is the real ``JSON`` type — ``JSON_VALUE`` works on it
unchanged; see ``ddl.py``). Views tolerate a NULL ``job_telemetry`` (runs before this column,
or whose telemetry capture was skipped): the unpacked fields come back NULL, the row still renders.

Six server-side diagnostic & S&OP views extend the read surface for planners, executives,
`review.py` pushdown, and the BigQuery Conversational Analytics Agent (`bq_agent`):

- ``v_fva_waterfall`` — *did complexity pay off?* Holdout-fold Forecast Value Added (FVA) across
  six pipeline stages (`1_baseline_naive` -> `2_statistical` -> `3_univariate_ml_dl` ->
  `4_covariate_ml_dl` -> `5_ensemble` -> `6_reconciled`) vs. the run's naive anchor.
- ``v_horizon_decay`` — *how far ahead can we trust the forecast?* Step-by-step (`horizon_step`
  1..H) pooled WAPE, MAE, signed bias, empirical coverage, and interval width from `backtest_oof`.
- ``v_bias_and_exceptions`` — *which series need planner review?* Per-`(run_id, ts_id, model_type)`
  Trigg's `tracking_signal`, SBC demand regime, WAPE, MASE, `staleness_gap`, and actionable
  `exception_reason` (`CHRONIC_OVER_FORECAST`, `CHRONIC_UNDER_FORECAST`, `WORSE_THAN_NAIVE`,
  `STALE_REFIT_DRIFT`, `CELL_FIT_ERROR`, or `OK`).
- ``v_anomalies`` — *where did historical demand shock outside expectations?* Out-of-fold points
  where `|residual_zscore| >= 2.5` or `y_true` breached `[yhat_lower, yhat_upper]`.
- ``v_feature_attributions`` — *what drove the forecast?* Unpacks both global feature importances
  (`fit_diagnostics.feature_attributions`) and horizon-averaged local attributions
  (`forecast_predictions.explanations.attributions`) into one queryable long-format view.
- ``v_executive_sop_summary`` — *one-row executive S&OP scorecard per run:* winning model, FVA %
  lift over naive, forward horizon total volume and uncertainty band width, planner exception rate,
  and GPU FinOps waste count.

Every rendered ``CREATE OR REPLACE VIEW`` embeds positional column ``OPTIONS(description='...')``
and view-level ``OPTIONS(description='...')`` from `registry.metadata` so BigQuery Studio,
Knowledge Catalog, and Conversational Analytics API agents always read full schema semantics.

Public surface: ``VIEW_NAMES``, ``CORE_VIEW_NAMES``, ``DIAGNOSTIC_VIEW_NAMES``,
``view_output_columns``, ``render_create_views``.
"""

from __future__ import annotations

from .ids import REPAIR_SUFFIX
from .metadata import column_description, object_description, sql_escape_description

# View bodies. `{d}` is the dataset ref (`project.dataset` or `dataset`); the registry tables the
# view reads are qualified with the same ref so a view and its sources always share a dataset.
_VIEW_BODIES: dict[str, str] = {
    "v_run_summary": """\
CREATE OR REPLACE VIEW `{d}.v_run_summary` AS
WITH header AS (
  SELECT *
  FROM `{d}.run_registry`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC NULLS LAST) = 1
),
current_jobs AS (
  SELECT run_id, family, created_at, started_at, ended_at
  FROM `{d}.run_jobs`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, family ORDER BY attempt DESC, created_at DESC
  ) = 1
),
launch_jobs AS (
  SELECT
    j.run_id,
    j.started_at,
    j.ended_at,
    TIMESTAMP_DIFF(j.ended_at, j.started_at, MILLISECOND) / 1000 AS job_seconds
  FROM current_jobs AS j
  JOIN header AS h USING (run_id)
  WHERE j.created_at >= h.created_at
    AND NOT ENDS_WITH(j.family, '{repair_suffix}')
),
ledger AS (
  SELECT
    run_id,
    COUNT(*) AS n_jobs,
    MAX(job_seconds) AS longest_job_seconds,
    SUM(job_seconds) AS jobs_seconds,
    TIMESTAMP_DIFF(MAX(ended_at), MIN(started_at), MILLISECOND) / 1000 AS jobs_span_seconds
  FROM launch_jobs
  GROUP BY run_id
)
SELECT
  h.run_id,
  h.created_at,
  h.status,
  h.python_runtime,
  h.n_series,
  h.n_models,
  h.backtest_on,
  h.runtime_seconds,
  l.n_jobs,
  l.longest_job_seconds,
  l.jobs_seconds,
  l.jobs_span_seconds,
  l.jobs_span_seconds - l.longest_job_seconds AS overhead_seconds,
  SAFE_DIVIDE(l.jobs_span_seconds - l.longest_job_seconds, l.jobs_span_seconds)
    AS overhead_fraction,
  CAST(JSON_VALUE(h.job_telemetry, '$.executor_instances') AS INT64) AS executor_instances,
  CAST(JSON_VALUE(h.job_telemetry, '$.executor_cores') AS INT64) AS executor_cores,
  CAST(JSON_VALUE(h.job_telemetry, '$.max_executors') AS INT64) AS max_executors,
  JSON_VALUE(h.job_telemetry, '$.executor_memory') AS executor_memory,
  JSON_VALUE(h.job_telemetry, '$.executor_memory_overhead') AS executor_memory_overhead,
  CAST(JSON_VALUE(h.job_telemetry, '$.dcu_milli_seconds') AS INT64) AS dcu_milli_seconds,
  JSON_VALUE(h.job_telemetry, '$.runtime_version') AS runtime_version,
  JSON_QUERY(h.job_telemetry, '$.sizing') AS sizing,
  JSON_QUERY(h.job_telemetry, '$.capacity') AS capacity
FROM header AS h
LEFT JOIN ledger AS l USING (run_id)""",
    "v_run_jobs": """\
CREATE OR REPLACE VIEW `{d}.v_run_jobs` AS
SELECT
  run_id,
  family,
  job_id,
  attempt,
  runtime,
  spark_mode,
  hardware,
  gpu_type,
  system_job_id,
  status,
  created_at,
  started_at,
  ended_at,
  runtime_seconds,
  CAST(JSON_VALUE(job_telemetry, '$.platform.provision_s') AS FLOAT64) AS provision_seconds,
  CAST(JSON_VALUE(job_telemetry, '$.platform.execute_s') AS FLOAT64) AS execute_seconds,
  failure_reason,
  JSON_VALUE(job_telemetry, '$.device_use.verdict') AS device_verdict,
  JSON_QUERY(job_telemetry, '$.device_use') AS device_use,
  JSON_QUERY(job_telemetry, '$.probe_handle') AS probe_handle,
  JSON_QUERY(job_telemetry, '$.capacity') AS capacity
FROM `{d}.run_jobs`
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY run_id, family ORDER BY attempt DESC, created_at DESC
) = 1""",
    "v_model_leaderboard": """\
CREATE OR REPLACE VIEW `{d}.v_model_leaderboard` AS
WITH deduped AS (
  SELECT *
  FROM `{d}.forecast_metadata`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
)
SELECT
  run_id,
  model_type,
  ensemble_id,
  COALESCE(
    JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.base_model'),
    SPLIT(model_type, '@')[SAFE_OFFSET(0)]
  ) AS base_model,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.scope_capability') AS scope_capability,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.training_mode') AS training_mode,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.cov_profile') AS cov_profile,
  CAST(JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.is_univariate') AS BOOL) AS is_univariate,
  ARRAY_TO_STRING(JSON_VALUE_ARRAY(ANY_VALUE(fit_diagnostics), '$.covariates_used'), ',')
    AS covariates_used,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.model_spec_hash') AS model_spec_hash,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.borrowed_from_run_id') AS borrowed_from_run_id,
  COUNTIF(JSON_VALUE(fit_diagnostics, '$.borrowed_from_run_id') IS NOT NULL) AS n_borrowed,
  ANY_VALUE(compute_engine) AS compute_engine,
  COUNT(*) AS n_cells,
  COUNTIF(model_artifact IS NULL) AS n_no_artifact,
  SAFE_DIVIDE(COUNTIF(model_artifact IS NULL), COUNT(*)) AS no_artifact_rate,
  APPROX_QUANTILES(fit_seconds, 2)[OFFSET(1)] AS median_fit_seconds,
  AVG(wape) AS mean_wape,
  AVG(mae) AS mean_mae,
  AVG(staleness_gap) AS mean_staleness_gap,
  STRING_AGG(DISTINCT backtest_refit ORDER BY backtest_refit) AS refit_modes
FROM deduped
WHERE fold_id IS NULL
GROUP BY run_id, model_type, ensemble_id""",
    "v_backtest_coverage": """\
CREATE OR REPLACE VIEW `{d}.v_backtest_coverage` AS
WITH deduped AS (
  SELECT *
  FROM `{d}.forecast_metadata`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
)
SELECT
  run_id,
  model_type,
  ensemble_id,
  backtest_status,
  n_folds_achieved,
  backtest_refit,
  COUNT(*) AS n_series,
  AVG(staleness_gap) AS mean_staleness_gap,
  SAFE_DIVIDE(
    COUNT(*),
    SUM(COUNT(*)) OVER (PARTITION BY run_id, model_type, ensemble_id)
  ) AS series_share
FROM deduped
WHERE fold_id IS NULL
GROUP BY run_id, model_type, ensemble_id, backtest_status, n_folds_achieved, backtest_refit""",
    "v_model_leaderboard_comparable": """\
CREATE OR REPLACE VIEW `{d}.v_model_leaderboard_comparable` AS
WITH deduped AS (
  SELECT *
  FROM `{d}.backtest_oof`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
holdout AS (
  SELECT *
  FROM deduped
  QUALIFY fold_id = MAX(fold_id) OVER (PARTITION BY run_id)
),
refit AS (
  SELECT
    run_id,
    model_type,
    ensemble_id,
    STRING_AGG(DISTINCT backtest_refit ORDER BY backtest_refit) AS refit_modes
  FROM `{d}.forecast_metadata`
  WHERE fold_id IS NULL
  GROUP BY run_id, model_type, ensemble_id
)
SELECT
  h.run_id,
  h.model_type,
  h.ensemble_id,
  ANY_VALUE(h.fold_id) AS holdout_fold_id,
  COUNT(DISTINCT h.ts_id) AS n_series,
  COUNT(*) AS n_points,
  SAFE_DIVIDE(SUM(ABS(h.y_true - h.yhat)), SUM(ABS(h.y_true))) AS pooled_wape,
  AVG(ABS(h.y_true - h.yhat)) AS pooled_mae,
  MIN(h.forecast_date) AS first_forecast_date,
  MAX(h.forecast_date) AS last_forecast_date,
  ANY_VALUE(r.refit_modes) AS refit_modes
FROM holdout AS h
LEFT JOIN refit AS r
  ON h.run_id = r.run_id
  AND h.model_type = r.model_type
  AND COALESCE(h.ensemble_id, '') = COALESCE(r.ensemble_id, '')
WHERE h.y_true IS NOT NULL AND h.yhat IS NOT NULL
GROUP BY h.run_id, h.model_type, h.ensemble_id""",
    "v_fva_waterfall": """\
CREATE OR REPLACE VIEW `{d}.v_fva_waterfall` AS
WITH oof_deduped AS (
  SELECT *
  FROM `{d}.backtest_oof`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
holdout AS (
  SELECT *
  FROM oof_deduped
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
  QUALIFY fold_id = MAX(fold_id) OVER (PARTITION BY run_id)
),
meta_deduped AS (
  SELECT
    run_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    ANY_VALUE(compute_engine) AS compute_engine,
    COALESCE(
      JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.base_model'),
      SPLIT(model_type, '@')[SAFE_OFFSET(0)]
    ) AS base_model,
    COALESCE(
      CAST(JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.is_univariate') AS BOOL),
      TRUE
    ) AS is_univariate
  FROM (
    SELECT *
    FROM `{d}.forecast_metadata`
    WHERE fold_id IS NULL
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
      ORDER BY created_at DESC NULLS LAST
    ) = 1
  )
  GROUP BY run_id, model_type, ens_key
),
model_scores AS (
  SELECT
    h.run_id,
    h.model_type,
    h.ensemble_id,
    COALESCE(h.ensemble_id, '') AS ens_key,
    COUNT(DISTINCT h.ts_id) AS n_series,
    COUNT(*) AS n_points,
    SAFE_DIVIDE(SUM(ABS(h.y_true - h.yhat)), SUM(ABS(h.y_true))) AS pooled_wape,
    AVG(ABS(h.y_true - h.yhat)) AS pooled_mae
  FROM holdout AS h
  GROUP BY h.run_id, h.model_type, h.ensemble_id
),
classified AS (
  SELECT
    s.run_id,
    CASE
      WHEN REGEXP_CONTAINS(
        s.model_type,
        r'_(bottom_up|wls_struct|mint_shrink|mint_nonneg|thief|cross_temporal_mint)$'
      ) THEN 6
      WHEN STARTS_WITH(s.model_type, 'ensemble_') OR s.ensemble_id IS NOT NULL THEN 5
      WHEN STARTS_WITH(
        LOWER(COALESCE(m.base_model, SPLIT(s.model_type, '@')[SAFE_OFFSET(0)])),
        'naive'
      ) THEN 1
      WHEN LOWER(COALESCE(m.base_model, SPLIT(s.model_type, '@')[SAFE_OFFSET(0)])) IN (
        'arima_plus', 'auto_arima', 'auto_ces', 'auto_ets', 'bats', 'croston',
        'dynamic_optimized_theta', 'dynamicoptimizedtheta', 'ets', 'fft', 'holtwinters',
        'kalman', 'mstl', 'prophet', 'sarimax', 'stl_bagging', 'tbats', 'theta', 'ucm'
      ) THEN 2
      WHEN COALESCE(m.is_univariate, TRUE)
        AND COALESCE(m.compute_engine, '') != 'vertex_automl' THEN 3
      ELSE 4
    END AS stage_order,
    s.model_type,
    s.ensemble_id,
    s.n_series,
    s.n_points,
    s.pooled_wape,
    s.pooled_mae
  FROM model_scores AS s
  LEFT JOIN meta_deduped AS m
    ON s.run_id = m.run_id AND s.model_type = m.model_type AND s.ens_key = m.ens_key
),
staged AS (
  SELECT
    c.*,
    CASE c.stage_order
      WHEN 1 THEN '1_baseline_naive'
      WHEN 2 THEN '2_statistical'
      WHEN 3 THEN '3_univariate_ml_dl'
      WHEN 4 THEN '4_covariate_ml_dl'
      WHEN 5 THEN '5_ensemble'
      ELSE '6_reconciled'
    END AS stage_name
  FROM classified AS c
),
baseline AS (
  SELECT
    run_id,
    model_type AS baseline_model,
    pooled_wape AS baseline_wape
  FROM staged
  WHERE stage_order <= 4 AND pooled_wape IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id
    ORDER BY
      IF(stage_order = 1, 0, 1) ASC,
      IF(stage_order = 1, pooled_wape, -pooled_wape) ASC,
      model_type ASC
  ) = 1
)
SELECT
  s.run_id,
  s.stage_order,
  s.stage_name,
  s.model_type,
  s.ensemble_id,
  s.n_series,
  s.n_points,
  s.pooled_wape,
  s.pooled_mae,
  b.baseline_model,
  b.baseline_wape,
  b.baseline_wape - s.pooled_wape AS fva_wape_delta,
  100.0 * SAFE_DIVIDE(b.baseline_wape - s.pooled_wape, b.baseline_wape)
    AS fva_relative_lift_pct
FROM staged AS s
LEFT JOIN baseline AS b USING (run_id)""",
    "v_horizon_decay": """\
CREATE OR REPLACE VIEW `{d}.v_horizon_decay` AS
WITH deduped AS (
  SELECT *
  FROM `{d}.backtest_oof`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
stepped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    ensemble_id,
    COALESCE(
      horizon_step,
      ROW_NUMBER() OVER (
        PARTITION BY run_id, ts_id, model_type, fold_id, COALESCE(ensemble_id, '')
        ORDER BY forecast_date ASC
      )
    ) AS horizon_step,
    y_true,
    yhat,
    yhat_lower,
    yhat_upper,
    IF(
      yhat_lower IS NOT NULL AND yhat_upper IS NOT NULL,
      IF(y_true BETWEEN yhat_lower AND yhat_upper, 1.0, 0.0),
      NULL
    ) AS in_bounds
  FROM deduped
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
)
SELECT
  run_id,
  model_type,
  ensemble_id,
  horizon_step,
  COUNT(DISTINCT ts_id) AS n_series,
  COUNT(*) AS n_points,
  SAFE_DIVIDE(SUM(ABS(y_true - yhat)), SUM(ABS(y_true))) AS pooled_wape,
  AVG(ABS(y_true - yhat)) AS pooled_mae,
  AVG(yhat - y_true) AS signed_bias,
  AVG(in_bounds) AS empirical_coverage,
  AVG(yhat_upper - yhat_lower) AS mean_interval_width
FROM stepped
GROUP BY run_id, model_type, ensemble_id, horizon_step""",
    "v_bias_and_exceptions": """\
CREATE OR REPLACE VIEW `{d}.v_bias_and_exceptions` AS
WITH meta_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    ensemble_id,
    COALESCE(ensemble_id, '') AS ens_key,
    cell_status,
    wape,
    mase,
    staleness_gap,
    COALESCE(
      JSON_VALUE(fit_diagnostics, '$.demand_profile.sbc_class'),
      JSON_VALUE(fit_diagnostics, '$.sbc_class')
    ) AS sbc_class
  FROM `{d}.forecast_metadata`
  WHERE fold_id IS NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
oof_deduped AS (
  SELECT *
  FROM `{d}.backtest_oof`
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
oof_agg AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    COUNT(*) AS n_oof_points,
    AVG(y_true) AS mean_actual,
    AVG(yhat) AS mean_forecast,
    AVG(yhat - y_true) AS signed_bias,
    AVG(ABS(y_true - yhat)) AS mad,
    SAFE_DIVIDE(SUM(ABS(y_true - yhat)), SUM(ABS(y_true))) AS oof_wape,
    SAFE_DIVIDE(SUM(yhat - y_true), NULLIF(AVG(ABS(y_true - yhat)), 0.0)) AS tracking_signal
  FROM oof_deduped
  GROUP BY run_id, ts_id, model_type, ens_key
),
joined AS (
  SELECT
    m.run_id,
    m.ts_id,
    m.model_type,
    m.ensemble_id,
    m.sbc_class,
    m.cell_status,
    COALESCE(o.n_oof_points, 0) AS n_oof_points,
    o.mean_actual,
    o.mean_forecast,
    o.signed_bias,
    o.mad,
    COALESCE(m.wape, o.oof_wape) AS wape,
    m.mase,
    m.staleness_gap,
    o.tracking_signal,
    CASE
      WHEN m.cell_status = 'error' THEN 'CELL_FIT_ERROR'
      WHEN o.tracking_signal > 4.0 THEN 'CHRONIC_OVER_FORECAST'
      WHEN o.tracking_signal < -4.0 THEN 'CHRONIC_UNDER_FORECAST'
      WHEN m.mase > 1.0 THEN 'WORSE_THAN_NAIVE'
      WHEN m.staleness_gap > 0.05 THEN 'STALE_REFIT_DRIFT'
      ELSE 'OK'
    END AS exception_reason
  FROM meta_deduped AS m
  LEFT JOIN oof_agg AS o
    ON m.run_id = o.run_id
    AND m.ts_id = o.ts_id
    AND m.model_type = o.model_type
    AND m.ens_key = o.ens_key
)
SELECT
  j.run_id,
  j.ts_id,
  j.model_type,
  j.ensemble_id,
  j.sbc_class,
  j.cell_status,
  j.n_oof_points,
  j.mean_actual,
  j.mean_forecast,
  j.signed_bias,
  j.mad,
  j.wape,
  j.mase,
  j.staleness_gap,
  j.tracking_signal,
  j.exception_reason != 'OK' AS is_exception,
  j.exception_reason
FROM joined AS j""",
    "v_anomalies": """\
CREATE OR REPLACE VIEW `{d}.v_anomalies` AS
WITH deduped AS (
  SELECT *
  FROM `{d}.backtest_oof`
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
scored AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    ensemble_id,
    forecast_date,
    fold_id,
    y_true,
    yhat,
    yhat_lower,
    yhat_upper,
    y_true - yhat AS residual,
    SAFE_DIVIDE(
      (y_true - yhat) - AVG(y_true - yhat) OVER (
        PARTITION BY run_id, ts_id, model_type, COALESCE(ensemble_id, '')
      ),
      NULLIF(
        STDDEV_POP(y_true - yhat) OVER (
          PARTITION BY run_id, ts_id, model_type, COALESCE(ensemble_id, '')
        ),
        0.0
      )
    ) AS residual_zscore,
    IF(
      yhat_lower IS NOT NULL AND yhat_upper IS NOT NULL,
      y_true < yhat_lower OR y_true > yhat_upper,
      FALSE
    ) AS outside_interval,
    IF(y_true >= yhat, 'POSITIVE_SPIKE', 'NEGATIVE_DIP') AS anomaly_type
  FROM deduped
)
SELECT
  s.run_id,
  s.ts_id,
  s.model_type,
  s.ensemble_id,
  s.forecast_date,
  s.fold_id,
  s.y_true,
  s.yhat,
  s.yhat_lower,
  s.yhat_upper,
  s.residual,
  s.residual_zscore,
  s.outside_interval,
  s.anomaly_type
FROM scored AS s
WHERE ABS(s.residual_zscore) >= 2.5 OR s.outside_interval = TRUE""",
    "v_feature_attributions": """\
CREATE OR REPLACE VIEW `{d}.v_feature_attributions` AS
WITH meta_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    TO_JSON_STRING(JSON_QUERY(fit_diagnostics, '$.feature_attributions')) AS attr_json
  FROM `{d}.forecast_metadata`
  WHERE fold_id IS NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
global_attrs AS (
  SELECT
    m.run_id,
    m.ts_id,
    m.model_type,
    'global_fit' AS attribution_scope,
    REGEXP_EXTRACT(kv, r'^"([^"]+)"') AS feature_name,
    SAFE_CAST(REGEXP_EXTRACT(kv, r':\\s*(.+)$') AS FLOAT64) AS mean_attribution,
    ABS(SAFE_CAST(REGEXP_EXTRACT(kv, r':\\s*(.+)$') AS FLOAT64)) AS mean_abs_attribution,
    CAST(NULL AS INT64) AS n_horizon_steps
  FROM meta_deduped AS m,
  UNNEST(
    REGEXP_EXTRACT_ALL(
      m.attr_json,
      r'"[^"]+"\\s*:\\s*-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
    )
  ) AS kv
  WHERE m.attr_json IS NOT NULL
),
pred_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    forecast_date,
    TO_JSON_STRING(JSON_QUERY(explanations, '$.attributions')) AS attr_json
  FROM `{d}.forecast_predictions`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
local_unpacked AS (
  SELECT
    p.run_id,
    p.ts_id,
    p.model_type,
    REGEXP_EXTRACT(kv, r'^"([^"]+)"') AS feature_name,
    SAFE_CAST(REGEXP_EXTRACT(kv, r':\\s*(.+)$') AS FLOAT64) AS attr_val
  FROM pred_deduped AS p,
  UNNEST(
    REGEXP_EXTRACT_ALL(
      p.attr_json,
      r'"[^"]+"\\s*:\\s*-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
    )
  ) AS kv
  WHERE p.attr_json IS NOT NULL
),
local_attrs AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    'horizon_local' AS attribution_scope,
    feature_name,
    AVG(attr_val) AS mean_attribution,
    AVG(ABS(attr_val)) AS mean_abs_attribution,
    COUNT(*) AS n_horizon_steps
  FROM local_unpacked
  WHERE feature_name IS NOT NULL AND attr_val IS NOT NULL
  GROUP BY run_id, ts_id, model_type, feature_name
),
combined AS (
  SELECT * FROM global_attrs WHERE feature_name IS NOT NULL AND mean_attribution IS NOT NULL
  UNION ALL
  SELECT * FROM local_attrs
)
SELECT
  c.run_id,
  c.ts_id,
  c.model_type,
  c.attribution_scope,
  c.feature_name,
  c.mean_attribution,
  c.mean_abs_attribution,
  c.n_horizon_steps
FROM combined AS c""",
    "v_executive_sop_summary": """\
CREATE OR REPLACE VIEW `{d}.v_executive_sop_summary` AS
WITH header AS (
  SELECT
    run_id,
    created_at,
    status,
    JSON_VALUE(raw_config, '$.data.source_table') AS source_table,
    n_series,
    n_models,
    runtime_seconds
  FROM `{d}.run_registry`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC NULLS LAST) = 1
),
oof_deduped AS (
  SELECT *
  FROM `{d}.backtest_oof`
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
holdout_scores AS (
  SELECT
    run_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    SAFE_DIVIDE(SUM(ABS(y_true - yhat)), SUM(ABS(y_true))) AS pooled_wape,
    AVG(ABS(y_true - yhat)) AS pooled_mae
  FROM (
    SELECT *
    FROM oof_deduped
    QUALIFY fold_id = MAX(fold_id) OVER (PARTITION BY run_id)
  )
  GROUP BY run_id, model_type, ens_key
),
pred_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    forecast_date,
    yhat,
    yhat_lower,
    yhat_upper
  FROM `{d}.forecast_predictions`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
fallback_models AS (
  SELECT
    run_id,
    model_type,
    ens_key
  FROM pred_deduped
  GROUP BY run_id, model_type, ens_key
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id
    ORDER BY
      IF(model_type = 'ensemble_champion', 0, IF(STARTS_WITH(model_type, 'ensemble_'), 1, 2)),
      model_type ASC
  ) = 1
),
scored_champion AS (
  SELECT
    run_id,
    model_type,
    ens_key,
    pooled_wape,
    pooled_mae
  FROM holdout_scores
  WHERE pooled_wape IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id ORDER BY pooled_wape ASC, model_type ASC
  ) = 1
),
champion AS (
  SELECT
    h.run_id,
    COALESCE(sc.model_type, fm.model_type) AS champion_model,
    COALESCE(sc.ens_key, fm.ens_key, '') AS ens_key,
    sc.pooled_wape AS champion_pooled_wape,
    sc.pooled_mae AS champion_pooled_mae
  FROM header AS h
  LEFT JOIN scored_champion AS sc USING (run_id)
  LEFT JOIN fallback_models AS fm USING (run_id)
),
baseline AS (
  SELECT
    run_id,
    pooled_wape AS baseline_wape
  FROM holdout_scores
  WHERE NOT STARTS_WITH(model_type, 'ensemble_') AND ens_key = '' AND pooled_wape IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id
    ORDER BY
      IF(STARTS_WITH(LOWER(SPLIT(model_type, '@')[SAFE_OFFSET(0)]), 'naive'), 0, 1) ASC,
      IF(
        STARTS_WITH(LOWER(SPLIT(model_type, '@')[SAFE_OFFSET(0)]), 'naive'),
        pooled_wape,
        -pooled_wape
      ) ASC,
      model_type ASC
  ) = 1
),
forward_rollup AS (
  SELECT
    p.run_id,
    MIN(p.forecast_date) AS horizon_start_date,
    MAX(p.forecast_date) AS horizon_end_date,
    SUM(p.yhat) AS total_forecast_units,
    AVG(p.yhat_upper - p.yhat_lower) AS mean_interval_width
  FROM pred_deduped AS p
  JOIN champion AS c
    ON p.run_id = c.run_id AND p.model_type = c.champion_model AND p.ens_key = c.ens_key
  GROUP BY p.run_id
),
meta_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    cell_status,
    mase,
    staleness_gap
  FROM `{d}.forecast_metadata`
  WHERE fold_id IS NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
oof_ts AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    SAFE_DIVIDE(SUM(yhat - y_true), NULLIF(AVG(ABS(y_true - yhat)), 0.0)) AS tracking_signal
  FROM oof_deduped
  GROUP BY run_id, ts_id, model_type, ens_key
),
exceptions_rollup AS (
  SELECT
    m.run_id,
    COUNTIF(
      m.cell_status = 'error'
      OR ABS(t.tracking_signal) > 4.0
      OR m.mase > 1.0
      OR m.staleness_gap > 0.05
    ) AS n_exception_series,
    SAFE_DIVIDE(
      COUNTIF(
        m.cell_status = 'error'
        OR ABS(t.tracking_signal) > 4.0
        OR m.mase > 1.0
        OR m.staleness_gap > 0.05
      ),
      COUNT(*)
    ) AS exception_rate
  FROM meta_deduped AS m
  JOIN champion AS c
    ON m.run_id = c.run_id AND m.model_type = c.champion_model AND m.ens_key = c.ens_key
  LEFT JOIN oof_ts AS t
    ON m.run_id = t.run_id
    AND m.ts_id = t.ts_id
    AND m.model_type = t.model_type
    AND m.ens_key = t.ens_key
  GROUP BY m.run_id
),
gpu_audit AS (
  SELECT
    run_id,
    COUNTIF(
      hardware = 'gpu'
      AND JSON_VALUE(job_telemetry, '$.device_use.verdict') IS NOT NULL
      AND JSON_VALUE(job_telemetry, '$.device_use.verdict') != 'ENGAGED_UTILISED'
    ) AS n_gpu_waste_jobs
  FROM (
    SELECT *
    FROM `{d}.run_jobs`
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY run_id, family ORDER BY attempt DESC, created_at DESC
    ) = 1
  )
  GROUP BY run_id
)
SELECT
  h.run_id,
  h.created_at,
  h.status,
  h.source_table,
  h.n_series,
  h.n_models,
  h.runtime_seconds,
  c.champion_model,
  c.champion_pooled_wape,
  c.champion_pooled_mae,
  b.baseline_wape,
  100.0 * SAFE_DIVIDE(b.baseline_wape - c.champion_pooled_wape, b.baseline_wape)
    AS fva_relative_lift_pct,
  f.horizon_start_date,
  f.horizon_end_date,
  f.total_forecast_units,
  f.mean_interval_width,
  COALESCE(e.n_exception_series, 0) AS n_exception_series,
  e.exception_rate,
  COALESCE(g.n_gpu_waste_jobs, 0) AS n_gpu_waste_jobs
FROM header AS h
LEFT JOIN champion AS c USING (run_id)
LEFT JOIN baseline AS b USING (run_id)
LEFT JOIN forward_rollup AS f USING (run_id)
LEFT JOIN exceptions_rollup AS e USING (run_id)
LEFT JOIN gpu_audit AS g USING (run_id)""",
}

CORE_VIEW_NAMES: tuple[str, ...] = (
    "v_run_summary",
    "v_run_jobs",
    "v_model_leaderboard",
    "v_backtest_coverage",
    "v_model_leaderboard_comparable",
)
DIAGNOSTIC_VIEW_NAMES: tuple[str, ...] = (
    "v_fva_waterfall",
    "v_horizon_decay",
    "v_bias_and_exceptions",
    "v_anomalies",
    "v_feature_attributions",
    "v_executive_sop_summary",
)
VIEW_NAMES: tuple[str, ...] = tuple(_VIEW_BODIES)


def _split_top_level_commas(select_list: str) -> list[str]:
    """Split a SQL SELECT list on top-level commas, ignoring commas inside parentheses/quotes."""
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    in_quote = False
    i = 0
    while i < len(select_list):
        ch = select_list[i]
        if ch == "'" and (i == 0 or select_list[i - 1] != "\\"):
            in_quote = not in_quote
        elif not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif ch == "," and depth == 0:
                items.append("".join(buf).strip())
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return items


def view_output_columns(view_name: str) -> list[str]:
    """Return the ordered list of output column names produced by ``view_name``'s final SELECT."""
    body = _VIEW_BODIES[view_name]
    select_list = body.rsplit("\nSELECT\n", 1)[1].split("\nFROM ", 1)[0]
    cols: list[str] = []
    for expr in _split_top_level_commas(select_list):
        normalized = " ".join(expr.split())
        if " AS " in normalized:
            col = normalized.rsplit(" AS ", 1)[1]
        else:
            col = normalized.rsplit(".", 1)[-1]
        cols.append(col)
    return cols


def _annotated_view_header(dataset: str, view_name: str) -> str:
    """Render the ``CREATE OR REPLACE VIEW ... (...) OPTIONS (...) AS`` header for ``view_name``."""
    cols = view_output_columns(view_name)
    col_lines = []
    for idx, col in enumerate(cols):
        col_desc = sql_escape_description(column_description(view_name, col))
        comma = "," if idx < len(cols) - 1 else ""
        col_lines.append(f"  {col} OPTIONS(description='{col_desc}'){comma}")
    view_desc = sql_escape_description(object_description(view_name))
    return (
        f"CREATE OR REPLACE VIEW `{dataset}.{view_name}` (\n"
        + "\n".join(col_lines)
        + "\n)\nOPTIONS (\n"
        + f"  description = '{view_desc}'\n"
        + ") AS"
    )


def render_create_views(dataset: str) -> dict[str, str]:
    """Render ``{view_name: CREATE OR REPLACE VIEW statement}`` for all 11 analyst views.

    Args:
        dataset: dataset ref, ``project.dataset`` or ``dataset`` — substituted for both the view
            name and the registry tables it reads.

    Each statement is ``CREATE OR REPLACE`` (idempotent — safe to re-run on every ``ensure``) and
    carries positional column ``OPTIONS(description='...')`` and view ``OPTIONS(description='...')``
    derived from `registry.metadata`.
    """
    out: dict[str, str] = {}
    for name, body in _VIEW_BODIES.items():
        raw_prefix = f"CREATE OR REPLACE VIEW `{{d}}.{name}` AS"
        assert body.startswith(raw_prefix), f"{name} missing standard view header"
        rest = (
            body[len(raw_prefix) :]
            .replace("{d}", dataset)
            .replace("{repair_suffix}", REPAIR_SUFFIX)
        )
        out[name] = f"{_annotated_view_header(dataset, name)}{rest};"
    return out
