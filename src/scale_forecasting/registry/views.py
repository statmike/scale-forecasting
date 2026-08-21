"""Analyst-facing SQL views over the registry.

The three-tier registry stores raw rows; a data scientist reviewing a run shouldn't have to
re-derive the same roll-ups every time. These views are the *curated read surface* — the point
where "rows in BigQuery" become "assets you review". They are pure ``CREATE OR REPLACE VIEW``
strings (no client), so they render + snapshot-test offline exactly like the table DDL, and
``registry/bq.ensure_views`` executes what this renders.

Three views, matched to the questions a run prompts:

- ``v_run_summary`` — *how did each run go, and how efficiently?* One row per run: the scaling
  knobs (``n_series``, ``n_models``), the engine's own ``runtime_seconds``, and the Dataproc
  ``job_telemetry`` overlay unpacked from its JSON column — total wall-clock, the provisioning
  overhead (``total_wall_s − runtime_seconds``) and its share, cluster sizing, and DCU usage. This
  is the run-level scaling-and-efficiency story as one ``SELECT * ORDER BY n_series`` — how
  wall-clock and overhead move with scale (overhead amortizes as the series count grows); the
  per-family runtime/hardware breakdown that composes each run lives in ``v_run_jobs``. A forced
  re-run of an unchanged config appends a second header row under the same ``run_id``; the view
  keeps only the latest (``QUALIFY ROW_NUMBER() … ORDER BY created_at DESC = 1``) so one run is
  always one row.

- ``v_run_jobs`` — *what jobs ran for this run, on what runtime/hardware, and how did each fare?*
  One row per ``(run_id, family)`` = the run's DAG as executed: the deterministic ``job_id``, the
  resolved ``runtime`` / ``spark_mode`` / ``hardware`` / ``gpu_type``, the platform's own
  ``system_job_id``, the per-job ``status`` and ``runtime_seconds``, and a ``dcu_milli_seconds``
  overlay from the per-job ``job_telemetry``. A ``--force`` re-run appends a higher-``attempt`` job
  under the same ``(run_id, family)``; the view keeps only the current one (``QUALIFY ROW_NUMBER()
  … ORDER BY attempt DESC = 1``), so the forward ``run_id → current job`` map is one row per family.

- ``v_model_leaderboard`` — *which model won, per run?* One row per ``(run_id, model_type,
  ensemble_id)``: cell counts, the error rate (a model failing every cell — the libgomp/lightgbm
  class of problem — shows as ``error_rate = 1.0``), median fit time, and the mean decision metrics
  where a backtest populated them. The entry point for "is this model worth keeping" before
  ensembling. ``ensemble_id`` is NULL for base models (so they group exactly as before) and the
  ``EnsembleConfig`` digest for ensemble pseudo-models — so two ensemble configs scored under one
  ``run_id`` keep their ``ensemble_<strategy>`` rows distinct instead of collapsing into one.

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
  CAST(JSON_VALUE(job_telemetry, '$.dcu_milli_seconds') AS INT64) AS dcu_milli_seconds,
  JSON_VALUE(job_telemetry, '$.runtime_version') AS runtime_version
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
  runtime_seconds,
  CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64) AS total_wall_s,
  CAST(JSON_VALUE(job_telemetry, '$.dcu_milli_seconds') AS INT64) AS dcu_milli_seconds
FROM `{d}.run_jobs`
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY run_id, family ORDER BY attempt DESC, created_at DESC
) = 1""",
    "v_model_leaderboard": """\
CREATE OR REPLACE VIEW `{d}.v_model_leaderboard` AS
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
FROM `{d}.forecast_metadata`
WHERE fold_id IS NULL
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
