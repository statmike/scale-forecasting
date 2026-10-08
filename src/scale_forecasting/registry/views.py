"""Analyst-facing SQL views over the registry.

The three-tier registry stores raw rows; a data scientist reviewing a run shouldn't have to
re-derive the same roll-ups every time. These views are the *curated read surface* — the point
where "rows in BigQuery" become "assets you review". They are pure ``CREATE OR REPLACE VIEW``
strings (no client), so they render + snapshot-test offline exactly like the table DDL, and
``registry.tables.ensure_views`` executes what this renders.

Five views, matched to the questions a run prompts:

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
  which is what ``v_run_summary``'s ledger is built from. ``failure_reason`` says *why* a FAILED
  row failed
  (``CAPACITY_EXHAUSTED`` is the first token) and ``capacity`` carries the whole attempt ledger —
  every candidate tried, its verdict, and the cloud's verbatim message. Both are NULL for a job
  that never had to wait, which is nearly all of them. ``device_verdict`` is the one word that says
  whether the accelerator on this row's ``hardware`` did any work (`device_audit`) — NULL for every
  CPU family, so ``WHERE device_verdict != 'ENGAGED_UTILISED'`` is the "what did I pay for and not
  use" query — and ``device_use`` beside it carries the counts and the peak byte figure the word was
  decided from. The platform's own wall and DCU figures are *not* on this row: the batch telemetry
  lands on the header (above), and a column that unpacked it from here would read NULL on every
  job. A ``--force`` re-run appends a
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

Public surface: ``VIEW_NAMES``, ``render_create_views``.
"""

from __future__ import annotations

from .ids import REPAIR_SUFFIX

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
}

VIEW_NAMES: tuple[str, ...] = tuple(_VIEW_BODIES)


def render_create_views(dataset: str) -> dict[str, str]:
    """Render ``{view_name: CREATE OR REPLACE VIEW statement}`` for the analyst views.

    Args:
        dataset: dataset ref, ``project.dataset`` or ``dataset`` — substituted for both the view
            name and the registry tables it reads.

    Each statement is ``CREATE OR REPLACE`` (idempotent — safe to re-run on every ``ensure``).
    ``v_run_summary``'s ledger excludes repair rows by the same `registry.ids.REPAIR_SUFFIX` the
    id scheme files them under, substituted here rather than spelled in the SQL so the two cannot
    drift.
    """
    return {
        name: body.format(d=dataset, repair_suffix=REPAIR_SUFFIX) + ";"
        for name, body in _VIEW_BODIES.items()
    }
