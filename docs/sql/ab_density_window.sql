-- ============================================================================================
-- Steady-state density for a two-arm A/B: the repaired version of control 4.
--
-- WHAT THIS IS, AND WHAT IT IS NOT
--
-- This file is NOT the rule that produced any result recorded in docs/validation.md. Those were
-- produced by control 4 as pre-registered in docs/sql/neuralprophet_ab.sql and
-- docs/sql/neuralprophet_ab_cluster.sql, and those two files are deliberately unchanged. A
-- pre-registered analysis that gets edited after the numbers land stops being pre-registered, and
-- the whole value of the discipline is that the edit is impossible rather than merely discouraged.
-- This is a separate file so the next A/B can pre-register a better rule without anyone having to
-- rewrite a record of what was actually run.
--
-- THE DEFECT IN THE ORIGINAL
--
-- The original asks the right question — were the two fleets equally busy — with a denominator that
-- is also right: fit-seconds over (bucket seconds x nodes that ran a cell in this bucket). What it
-- gets wrong is which buckets count as steady state. It aligns 30-minute buckets to the wall clock
-- and then drops exactly one bucket from each end, so it assumes both that the ramp and the drain
-- each fit inside one bucket and that they line up with the top of the hour. Two failures followed:
--
--   * A cluster run whose last two cells finished 73 seconds past the hour grew a seventh bucket.
--     The seventh absorbed the "drop the last one" rule, the real drain was promoted into the
--     steady-state set, and the control returned a 10.6% gap against a 10% threshold. The fleets
--     were identical; the clock was not.
--   * On an autoscaled fleet the ramp takes two buckets, not one, because workers arrive over
--     several minutes. Both arms of the Ray A/B therefore carried a ramp bucket AND a drain bucket
--     into their steady-state sets, and the control passed only because the two arms were
--     contaminated by nearly the same amount. Its recorded agreement "within 1.0E-4" is the
--     difference between two wrong numbers, not the agreement of two right ones.
--
-- WHAT THIS FILE CHANGES, AND WHY EACH ALTERNATIVE WAS DROPPED
--
-- The window start is measured rather than assumed: it is the moment the last worker joined the
-- fleet, MAX over workers of that worker's first cell start. That is an observable fact about the
-- run, not a guess about how long a ramp takes, and it is not answer-dependent — it can be computed
-- without looking at a single density. It matters: on one autoscaled arm the fleet was still growing
-- 28.8 minutes after the first cell finished, so a fixed 30-minute ramp trim would have cleared it
-- by 1.2 minutes. That is a near miss, not a rule.
--
-- The window end is still an assumption — the last cell end minus one bucket — because the drain has
-- no equally clean observable. The symmetric definition (the first worker to stop working) is
-- useless on an autoscaled fleet, where a churned worker that ran two cells early and was reclaimed
-- reads as a drain starting three hours before the run ends. One bucket is enough on every arm
-- measured: the cluster arms drained in 22.4 and 24.5 minutes, and on all four arms exactly one
-- end-anchored bucket is below saturation.
--
-- Two alternatives were tested against the surviving data and both are worse:
--
--   * Trim by saturation — keep only buckets at the run's maximum active_nodes. On a fixed-size
--     cluster all 28 nodes are present through the drain, so the drain survives the trim. On an
--     autoscaled fleet the maximum is 108, which occurs during the ramp, so the trim keeps the ramp
--     and discards the steady state. It fails in both directions.
--   * Normalise by node coverage — charge each worker only for the seconds it was in the fleet,
--     and drop the windowing entirely. This removes every tunable, which is genuinely attractive,
--     but it over-charges autoscaled fleets: workers that churned in and out are billed from first
--     cell to run end, pulling steady-state density down to 0.78 and 0.82 on two arms that are both
--     actually saturated. Worse, the two arms then differ by 6% for no reason except that the
--     autoscaler drew 108 workers in one and 102 in the other. It trades a bias that can be
--     corrected for one that cannot.
--
-- The repaired rule was checked against all four arms that still have cells in BigQuery. Every
-- steady-state bucket comes out with the fleet at exactly its intended size — 84.0 nodes on both
-- autoscaled arms, 28.0 on both cluster arms — which is the independent confirmation that the
-- window is right rather than merely flattering. No recorded verdict changes: both pairs still pass,
-- the cluster pair by 0.05% and the autoscaled pair by 0.24%.
--
-- A KNOWN, DELIBERATELY UNCORRECTED BIAS
--
-- A cell is charged to the bucket its fit *ended* in, which is what the original does and is not the
-- part that was wrong. It does mean the first bucket is short by whatever was already in flight when
-- the window opened: a worker mid-fit at `window_start` finished that fit outside the window and
-- contributes nothing for it. The effect is one fit per worker, once, in bucket zero only — visible
-- in the live numbers as the cluster arms' min_density of 0.9747 and 0.9671 against means of 0.9921
-- and 0.9916. It is left uncorrected because it lands on both arms identically and because the
-- alternative, apportioning each fit across the buckets it spans, would change the quantity being
-- measured rather than the buckets it is measured over.
--
-- FAILURE DIRECTION
--
-- A worker that joins unusually late pushes the window start past most of the run and leaves too few
-- whole buckets. The rule below returns INCONCLUSIVE in that case rather than averaging one bucket
-- and calling it a steady state. A control that refuses to answer is doing its job; a control that
-- answers from one bucket is not.
--
-- The offline reference implementation and its tests are tests/unit/test_ab_density_window.py.
-- That file is the executable specification of the rule written below; the SQL is what runs.
--
-- Usage:
--   bq query --project_id=<project> --use_legacy_sql=false < docs/sql/ab_density_window.sql
-- Set the DECLAREs below first.
-- ============================================================================================

DECLARE gpu_run_id STRING DEFAULT 'neuralprophet-ab-cluster-gpu-273d32b553c8';
DECLARE cpu_run_id STRING DEFAULT 'neuralprophet-ab-cluster-cpu-a402414abf5e';
DECLARE model_name STRING DEFAULT 'neuralprophet';

-- Bucket width. Wide enough that a handful of long cells do not swing a bucket, narrow enough that
-- a run of a couple of hours still yields several.
DECLARE bucket_minutes INT64 DEFAULT 30;

-- How much of the tail to treat as drain. One bucket, for the reasons in the header.
DECLARE drain_minutes INT64 DEFAULT 30;

-- Fewer whole buckets than this and the arm has no steady state to average over.
DECLARE min_steady_buckets INT64 DEFAULT 2;

-- The agreement threshold between the two arms' mean densities.
DECLARE density_tolerance FLOAT64 DEFAULT 0.10;

CREATE TEMP TABLE cells AS
SELECT
  IF(run_id = gpu_run_id, 'gpu', 'cpu') AS arm,
  worker_id,
  cell_started_at,
  cell_ended_at,
  fit_seconds
FROM `scale_forecasting.forecast_metadata`
WHERE run_id IN (gpu_run_id, cpu_run_id)
  AND model_type = model_name
  AND fold_id IS NULL
  AND ensemble_id IS NULL
  AND cell_status = 'ok'
  AND cell_started_at IS NOT NULL
  AND cell_ended_at IS NOT NULL;

-- The window. `window_start` is the last worker's arrival, which is measured. `window_end` is the
-- last cell end less one drain, which is assumed.
CREATE TEMP TABLE windows AS
WITH joined AS (
  SELECT arm, worker_id, MIN(cell_started_at) AS joined_at
  FROM cells
  GROUP BY arm, worker_id
),
ends AS (
  SELECT arm, MAX(cell_ended_at) AS last_cell_end FROM cells GROUP BY arm
)
SELECT
  j.arm,
  MAX(j.joined_at)                                                          AS window_start,
  TIMESTAMP_SUB(ANY_VALUE(e.last_cell_end), INTERVAL drain_minutes MINUTE)  AS window_end
FROM joined j
JOIN ends e USING (arm)
GROUP BY j.arm;

-- Whole buckets only, tiled forward from `window_start`. A cell belongs to the bucket its fit
-- *ended* in, matching the original rule.
CREATE TEMP TABLE buckets AS
WITH placed AS (
  SELECT
    c.arm,
    c.worker_id,
    c.fit_seconds,
    DIV(
      UNIX_SECONDS(c.cell_ended_at) - UNIX_SECONDS(w.window_start), bucket_minutes * 60
    )                                                                      AS bucket_index,
    DIV(
      UNIX_SECONDS(w.window_end) - UNIX_SECONDS(w.window_start), bucket_minutes * 60
    )                                                                      AS whole_buckets
  FROM cells c
  JOIN windows w USING (arm)
  WHERE c.cell_ended_at >= w.window_start
    AND c.cell_ended_at <  w.window_end
)
SELECT
  arm,
  bucket_index,
  SUM(fit_seconds)                                                         AS fit_seconds_in_bucket,
  COUNT(DISTINCT worker_id)                                                AS active_nodes,
  COUNT(*)                                                                 AS cells_in_bucket,
  SAFE_DIVIDE(SUM(fit_seconds), (bucket_minutes * 60) * COUNT(DISTINCT worker_id)) AS density
FROM placed
WHERE bucket_index >= 0 AND bucket_index < whole_buckets
GROUP BY arm, bucket_index;

-- Per-arm detail. Read this alongside the verdict; `mean_active_nodes` landing on the fleet size the
-- arm was planned with is the sign that the window excluded the ramp and the drain.
SELECT
  'density_window_detail' AS control,
  arm,
  COUNT(*)                        AS steady_state_buckets,
  ROUND(AVG(density), 4)          AS mean_density,
  ROUND(MIN(density), 4)          AS min_density,
  ROUND(MAX(density), 4)          AS max_density,
  SUM(cells_in_bucket)            AS cells_in_steady_state,
  ROUND(AVG(active_nodes), 2)     AS mean_active_nodes
FROM buckets
GROUP BY arm
ORDER BY arm;

-- The verdict, as one row. Too few buckets is INCONCLUSIVE on its own terms, before the gap is even
-- looked at, because a mean over one bucket is that bucket.
WITH d AS (
  SELECT arm, AVG(density) AS mean_density, COUNT(*) AS n_buckets
  FROM buckets
  GROUP BY arm
),
g AS (
  SELECT
    MAX(IF(arm = 'gpu', mean_density, NULL)) AS gpu_density,
    MAX(IF(arm = 'cpu', mean_density, NULL)) AS cpu_density,
    MIN(n_buckets)                           AS min_buckets,
    COUNT(*)                                 AS arms_present
  FROM d
)
SELECT
  'density_window_verdict' AS control,
  ROUND(gpu_density, 4)                                                        AS gpu_density,
  ROUND(cpu_density, 4)                                                        AS cpu_density,
  ROUND(SAFE_DIVIDE(ABS(gpu_density - cpu_density), cpu_density), 4)           AS relative_gap,
  -- Not aliased `min_steady_buckets`: that is the name of the threshold this is compared against,
  -- and a column sharing it would make the CASE below ambiguous to read even where it resolves.
  min_buckets                                                                  AS buckets_in_arm,
  CASE
    WHEN arms_present < 2 OR min_buckets IS NULL THEN
      'INCONCLUSIVE - an arm produced no steady-state buckets at all'
    WHEN min_buckets < min_steady_buckets THEN
      'INCONCLUSIVE - too few whole buckets between the last arrival and the drain'
    WHEN SAFE_DIVIDE(ABS(gpu_density - cpu_density), cpu_density) <= density_tolerance THEN
      'PASS'
    ELSE
      'INCONCLUSIVE - the arms were not equally busy'
  END                                                                          AS verdict
FROM g;
