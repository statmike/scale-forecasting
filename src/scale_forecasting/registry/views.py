"""Analyst-facing SQL views over the registry.

The three-tier registry stores raw rows; a data scientist reviewing a run shouldn't have to
re-derive the same roll-ups every time. These views are the *curated read surface* — the point
where "rows in BigQuery" become "assets you review". They are pure ``CREATE OR REPLACE VIEW``
strings (no client), so they render + snapshot-test offline exactly like the table DDL, and
``registry.tables.ensure_views`` executes what this renders.

Five views, matched to the questions a run prompts:

- ``v_run_summary`` — *how did each run go, and how efficiently?* One row per run: the scaling
  knobs (``n_series``, ``n_models``), the engine's own ``runtime_seconds``, and the Dataproc
  ``job_telemetry`` overlay unpacked from its JSON column — total wall-clock, the provisioning
  overhead (``total_wall_s − runtime_seconds``) and its share, cluster sizing, and DCU usage. This
  is the run-level scaling-and-efficiency story as one ``SELECT * ORDER BY n_series`` — how
  wall-clock and overhead move with scale (overhead amortizes as the series count grows); the
  per-family runtime/hardware breakdown that composes each run lives in ``v_run_jobs``.
  The executor columns are the shape the platform was *told* (echoed back off the submitted batch),
  while ``sizing`` is the whole decision behind it — one entry per family under ``$.sizing``, each
  holding the fleet plan, its translation to platform settings, and the `ComputeProfile` it was
  sized off (`resources.audit.sizing_telemetry`). Left as raw ``JSON`` rather than unpacked into
  columns because its interesting parts are per-family and nested: read a run's shape from the
  scalar columns, read *why* with ``JSON_QUERY(sizing, '$.deep_learning')``. NULL on a run submitted
  before this existed, or one that left the platform's own defaults standing. ``capacity`` is the
  same shape for the other run-level wait: one attempt ledger per *shared* cluster, keyed by service
  (`shared_clusters.shared_capacity_path`), recording every region tried before one had room. A
  per-family walk is on the family's row instead — see ``v_run_jobs``. A forced re-run of an
  unchanged config appends a second header row under the same ``run_id``; the view keeps only the
  latest (``QUALIFY ROW_NUMBER() … ORDER BY created_at DESC = 1``) so one run is always one row.

- ``v_run_jobs`` — *what jobs ran for this run, on what runtime/hardware, and how did each fare?*
  One row per ``(run_id, family)`` = the run's DAG as executed: the deterministic ``job_id``, the
  resolved ``runtime`` / ``spark_mode`` / ``hardware`` / ``gpu_type``, the platform's own
  ``system_job_id``, the per-job ``status`` and ``runtime_seconds``, and a ``dcu_milli_seconds``
  overlay from the per-job ``job_telemetry``. ``failure_reason`` says *why* a FAILED row failed
  (``CAPACITY_EXHAUSTED`` is the first token) and ``capacity`` carries the whole attempt ledger —
  every candidate tried, its verdict, and the cloud's verbatim message. Both are NULL for a job
  that never had to wait, which is nearly all of them. ``device_verdict`` is the one word that says
  whether the accelerator on this row's ``hardware`` did any work (`device_audit`) — NULL for every
  CPU family, so ``WHERE device_verdict != 'ENGAGED_UTILISED'`` is the "what did I pay for and not
  use" query — and ``device_use`` beside it carries the counts and the peak byte figure the word was
  decided from. A ``--force`` re-run appends a
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
  created_at DESC = 1``, latest write wins) before aggregating; otherwise a duplicated cell would
  double-count and skew ``mean_wape`` / ``mean_mae`` / ``n_cells``.

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
  not — it doubles both sides). ``backtest_oof.created_at`` has no writer yet, so the
  ``ORDER BY created_at DESC`` tiebreak picks arbitrarily among duplicates today; that is harmless
  while duplicates are byte-identical, which append-only + deterministic rows make them.

``JSON_VALUE`` reads scalars straight out of the native ``JSON`` ``job_telemetry`` column (the
registry is native BigQuery, so the column is the real ``JSON`` type — ``JSON_VALUE`` works on it
unchanged; see ``ddl.py``). Views tolerate a NULL ``job_telemetry`` (runs before this column,
or whose telemetry capture was skipped): the unpacked fields come back NULL, the row still renders.

Public surface: ``VIEW_NAMES``, ``render_create_views``.
"""

from __future__ import annotations

# View bodies. `{d}` is the dataset ref (`project.dataset` or `dataset`); the registry tables the
# view reads are qualified with the same ref so a view and its sources always share a dataset.
_VIEW_BODIES: dict[str, str] = {
    "v_run_summary": """\
CREATE OR REPLACE VIEW `{d}.v_run_summary` AS
SELECT
  run_id,
  created_at,
  status,
  python_runtime,
  n_series,
  n_models,
  backtest_on,
  runtime_seconds,
  CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64) AS total_wall_s,
  CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64)
    - runtime_seconds AS overhead_seconds,
  SAFE_DIVIDE(
    CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64) - runtime_seconds,
    CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64)
  ) AS overhead_fraction,
  CAST(JSON_VALUE(job_telemetry, '$.executor_instances') AS INT64) AS executor_instances,
  CAST(JSON_VALUE(job_telemetry, '$.executor_cores') AS INT64) AS executor_cores,
  CAST(JSON_VALUE(job_telemetry, '$.max_executors') AS INT64) AS max_executors,
  JSON_VALUE(job_telemetry, '$.executor_memory') AS executor_memory,
  JSON_VALUE(job_telemetry, '$.executor_memory_overhead') AS executor_memory_overhead,
  CAST(JSON_VALUE(job_telemetry, '$.dcu_milli_seconds') AS INT64) AS dcu_milli_seconds,
  JSON_VALUE(job_telemetry, '$.runtime_version') AS runtime_version,
  JSON_QUERY(job_telemetry, '$.sizing') AS sizing,
  JSON_QUERY(job_telemetry, '$.capacity') AS capacity
FROM `{d}.run_registry`
QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC) = 1""",
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
  CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64) AS total_wall_s,
  CAST(JSON_VALUE(job_telemetry, '$.dcu_milli_seconds') AS INT64) AS dcu_milli_seconds,
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
    ORDER BY created_at DESC
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
  AVG(mae) AS mean_mae
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
    ORDER BY created_at DESC
  ) = 1
)
SELECT
  run_id,
  model_type,
  ensemble_id,
  backtest_status,
  n_folds_achieved,
  COUNT(*) AS n_series,
  SAFE_DIVIDE(
    COUNT(*),
    SUM(COUNT(*)) OVER (PARTITION BY run_id, model_type, ensemble_id)
  ) AS series_share
FROM deduped
WHERE fold_id IS NULL
GROUP BY run_id, model_type, ensemble_id, backtest_status, n_folds_achieved""",
    "v_model_leaderboard_comparable": """\
CREATE OR REPLACE VIEW `{d}.v_model_leaderboard_comparable` AS
WITH deduped AS (
  SELECT *
  FROM `{d}.backtest_oof`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC
  ) = 1
),
holdout AS (
  SELECT *
  FROM deduped
  QUALIFY fold_id = MAX(fold_id) OVER (PARTITION BY run_id)
)
SELECT
  run_id,
  model_type,
  ensemble_id,
  ANY_VALUE(fold_id) AS holdout_fold_id,
  COUNT(DISTINCT ts_id) AS n_series,
  COUNT(*) AS n_points,
  SAFE_DIVIDE(SUM(ABS(y_true - yhat)), SUM(ABS(y_true))) AS pooled_wape,
  AVG(ABS(y_true - yhat)) AS pooled_mae,
  MIN(forecast_date) AS first_forecast_date,
  MAX(forecast_date) AS last_forecast_date
FROM holdout
WHERE y_true IS NOT NULL AND yhat IS NOT NULL
GROUP BY run_id, model_type, ensemble_id""",
}

VIEW_NAMES: tuple[str, ...] = tuple(_VIEW_BODIES)


def render_create_views(dataset: str) -> dict[str, str]:
    """Render ``{view_name: CREATE OR REPLACE VIEW statement}`` for the analyst views.

    Args:
        dataset: dataset ref, ``project.dataset`` or ``dataset`` — substituted for both the view
            name and the registry tables it reads.

    Each statement is ``CREATE OR REPLACE`` (idempotent — safe to re-run on every ``ensure``).
    """
    return {name: body.format(d=dataset) + ";" for name, body in _VIEW_BODIES.items()}
