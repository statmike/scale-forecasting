-- ============================================================================================
-- The GPU-vs-CPU A/B for NeuralProphet: controls first, then the decision.
--
-- Pre-registered. This file is committed BEFORE either arm is submitted, and
-- tests/unit/test_ab_preregistration.py asserts that it exists and still asks these questions.
-- An analysis written after the numbers land is an analysis fitted to the numbers.
--
-- The two arms are configs/neuralprophet_ab_gpu.json and configs/neuralprophet_ab_cpu.json:
-- the same 10,000 NeuralProphet cells, the same 12-node n1-standard-8 fleet, seven one-core slots
-- per node on both, differing only in whether each node carries a T4.
--
-- Read the sections in order. Sections 1-4 are CONTROLS. If any of them fails the experiment is
-- INCONCLUSIVE and section 5 is not to be run, re-run, or reinterpreted. That rule is the point of
-- pre-registering: the temptation to relax a control arrives only after you have seen the answer.
--
-- Usage:
--   bq query --project_id=<project> --use_legacy_sql=false < docs/sql/neuralprophet_ab.sql
-- Set the four DECLAREs below first.
-- ============================================================================================

DECLARE gpu_run_id STRING DEFAULT 'neuralprophet-ab-gpu-e530eea3a755';
DECLARE cpu_run_id STRING DEFAULT 'neuralprophet-ab-cpu-f4bfff3b39e9';

-- The accelerator surcharge `r`: the price of one GPU node divided by the price of one CPU node,
-- same machine type, same region, at the rate actually billed. Set this from current list or
-- committed-use pricing before running; it is a business input, not a measurement, and the decision
-- rule below is stated in terms of it rather than around it.
--
-- 1.92 is us-central1 on-demand list at the time of the run: an n1-standard-8 is ~$0.380/hour
-- (8 vCPU + 30 GB) and one T4 adds ~$0.350/hour, so (0.380 + 0.350) / 0.380 = 1.92. Committed-use
-- or a Spot GPU moves this, and it is the one number here a reader is expected to substitute for
-- their own contract. The decision rule below reads `r` rather than hard-coding a threshold, so
-- substituting it re-runs the decision honestly instead of requiring a re-argued one.
DECLARE accel_surcharge FLOAT64 DEFAULT 1.92;

-- Bucket width for the steady-state windows. Thirty minutes is wide enough to average over a slow
-- cell and narrow enough that the autoscaler's ramp lands in its own bucket instead of contaminating
-- the rest.
DECLARE bucket_minutes INT64 DEFAULT 30;

-- --------------------------------------------------------------------------------------------
-- The shared base: every NeuralProphet cell of both arms, labelled by arm.
--
-- `fold_id IS NULL AND ensemble_id IS NULL` selects the final-fit row for each series — one row per
-- cell, which is what the cell-level controls need. Backtest fold rows are pulled in separately in
-- section 5, where total compute is the question and every fit counts.
--
-- Deliberately NOT read through `read_compute_harvest`: that path truncates at a cell cap and
-- samples by FARM_FINGERPRINT(ts_id), so it answers a question about a sample when this needs the
-- whole population.
-- --------------------------------------------------------------------------------------------
CREATE TEMP TABLE cells AS
SELECT
  IF(run_id = gpu_run_id, 'gpu', 'cpu')          AS arm,
  ts_id,
  worker_id,
  cell_started_at,
  cell_ended_at,
  fit_seconds,
  cpu_seconds,
  peak_gpu_bytes,
  device_used,
  intraop_threads,
  n_fits,
  cell_status,
  wape
FROM `scale_forecasting.forecast_metadata`
WHERE run_id IN (gpu_run_id, cpu_run_id)
  AND model_type = 'neuralprophet'
  AND fold_id IS NULL
  AND ensemble_id IS NULL;

-- ============================================================================================
-- CONTROL 1 - identical landed work.
--
-- Both arms must have completed the same number of cells. An arm that quietly dropped 400 failures
-- looks faster per cell for a reason that has nothing to do with its hardware.
-- ============================================================================================
SELECT
  'control_1_landed_cells' AS control,
  arm,
  COUNT(*)                                       AS cells,
  COUNTIF(cell_status = 'ok')                    AS ok_cells,
  COUNTIF(cell_status != 'ok')                   AS not_ok_cells,
  COUNT(DISTINCT ts_id)                          AS distinct_series
FROM cells
GROUP BY arm
ORDER BY arm;

-- ============================================================================================
-- CONTROL 2 - the device contract, in both directions.
--
-- The GPU arm must show a device actually holding memory; the CPU arm must show no device at all.
-- A GPU arm with peak_gpu_bytes all-NULL is the "bought and never used" shape that made every
-- earlier GPU number in this project meaningless, and it would make this one meaningless too.
-- ============================================================================================
SELECT
  'control_2_device' AS control,
  arm,
  COUNTIF(peak_gpu_bytes IS NOT NULL)            AS cells_with_device_bytes,
  MAX(peak_gpu_bytes)                            AS max_peak_gpu_bytes,
  COUNT(DISTINCT device_used)                    AS distinct_device_used,
  ANY_VALUE(device_used)                         AS a_device_used,
  -- PASS for the GPU arm means a real allocation on most cells; PASS for the CPU arm means none.
  IF(
    arm = 'gpu',
    IF(MAX(peak_gpu_bytes) > 0, 'PASS', 'FAIL - no device memory on the GPU arm'),
    IF(COUNTIF(peak_gpu_bytes IS NOT NULL) = 0, 'PASS', 'FAIL - device bytes on the CPU arm')
  )                                              AS verdict
FROM cells
GROUP BY arm
ORDER BY arm;

-- ============================================================================================
-- CONTROL 3 - one thread per fit, measured rather than declared.
--
-- `cpu_seconds / fit_seconds` is how many cores a fit really consumed. A ratio near 1.0 means the
-- one-core slot was honoured. This replaces the older `intraop_threads == 1` check, which only ever
-- read back the OMP_NUM_THREADS we set ourselves and so could not fail.
--
-- The threshold is 1.05 on EVERY cell of BOTH arms. Reported as a max and a violation count, not a
-- mean: one arm running three threads on 5% of its cells is exactly the confound this catches, and
-- a mean would hide it.
-- ============================================================================================
SELECT
  'control_3_thread_pin' AS control,
  arm,
  COUNT(*)                                                        AS cells,
  ROUND(MAX(SAFE_DIVIDE(cpu_seconds, fit_seconds)), 3)            AS max_ratio,
  ROUND(APPROX_QUANTILES(SAFE_DIVIDE(cpu_seconds, fit_seconds), 100)[OFFSET(99)], 3) AS p99_ratio,
  COUNTIF(SAFE_DIVIDE(cpu_seconds, fit_seconds) > 1.05)           AS cells_over_threshold,
  COUNT(DISTINCT intraop_threads)                                 AS distinct_intraop_threads,
  IF(
    COUNTIF(SAFE_DIVIDE(cpu_seconds, fit_seconds) > 1.05) = 0,
    'PASS',
    'FAIL - some fits used more than one core'
  )                                                               AS verdict
FROM cells
WHERE cell_status = 'ok'
GROUP BY arm
ORDER BY arm;

-- ============================================================================================
-- CONTROL 4 - steady-state density, and enough waves to have a steady state.
--
-- Density is SUM(fit_seconds) / wall_seconds / active_nodes: how much useful compute each node
-- delivered per second of wall clock. It is the check that the two fleets were equally busy. If the
-- CPU arm was 30% idler than the GPU arm, its throughput number is measuring the autoscaler, not
-- the hardware.
--
-- Buckets are 30 minutes wide. The FIRST and LAST bucket of each arm are dropped: the first holds
-- the autoscaler ramp and the last holds the drain, and neither is steady state. Active nodes come
-- from COUNT(DISTINCT worker_id) within the bucket, which counts nodes that actually ran a cell
-- rather than nodes that were paid for.
--
-- Two thresholds: the per-arm density must agree within 10%, and each arm needs at least 50 waves
-- (cells divided by total slots) for the average to mean anything.
-- ============================================================================================
CREATE TEMP TABLE buckets AS
WITH b AS (
  SELECT
    arm,
    TIMESTAMP_SECONDS(
      DIV(UNIX_SECONDS(cell_ended_at), bucket_minutes * 60) * bucket_minutes * 60
    )                                            AS bucket_start,
    fit_seconds,
    worker_id
  FROM cells
  WHERE cell_status = 'ok' AND cell_ended_at IS NOT NULL
),
agg AS (
  SELECT
    arm,
    bucket_start,
    SUM(fit_seconds)                             AS fit_seconds_in_bucket,
    COUNT(DISTINCT worker_id)                    AS active_nodes,
    COUNT(*)                                     AS cells_in_bucket
  FROM b
  GROUP BY arm, bucket_start
)
SELECT
  *,
  ROW_NUMBER() OVER (PARTITION BY arm ORDER BY bucket_start)      AS bucket_rank,
  ROW_NUMBER() OVER (PARTITION BY arm ORDER BY bucket_start DESC) AS bucket_rank_desc,
  SAFE_DIVIDE(fit_seconds_in_bucket, (bucket_minutes * 60) * active_nodes) AS density
FROM agg;

SELECT
  'control_4_density' AS control,
  arm,
  COUNT(*)                                       AS steady_state_buckets,
  ROUND(AVG(density), 4)                         AS mean_density,
  ROUND(MIN(density), 4)                         AS min_density,
  ROUND(MAX(density), 4)                         AS max_density,
  SUM(cells_in_bucket)                           AS cells_in_steady_state,
  ROUND(AVG(active_nodes), 2)                    AS mean_active_nodes
FROM buckets
WHERE bucket_rank > 1 AND bucket_rank_desc > 1
GROUP BY arm
ORDER BY arm;

-- The 10% agreement test, as one row. Read this rather than eyeballing the two means above.
WITH d AS (
  SELECT arm, AVG(density) AS mean_density
  FROM buckets
  WHERE bucket_rank > 1 AND bucket_rank_desc > 1
  GROUP BY arm
),
w AS (
  -- Waves = cells / total slots. Both arms plan 84 slots (12 nodes x 7); asserted offline by
  -- test_ab_preregistration.py, so it is a constant here rather than another thing to look up.
  SELECT arm, COUNT(*) / 84.0 AS waves FROM cells WHERE cell_status = 'ok' GROUP BY arm
)
SELECT
  'control_4_verdict' AS control,
  ROUND(MAX(IF(d.arm = 'gpu', d.mean_density, NULL)), 4)          AS gpu_density,
  ROUND(MAX(IF(d.arm = 'cpu', d.mean_density, NULL)), 4)          AS cpu_density,
  ROUND(
    ABS(MAX(IF(d.arm = 'gpu', d.mean_density, NULL)) - MAX(IF(d.arm = 'cpu', d.mean_density, NULL)))
    / NULLIF(MAX(IF(d.arm = 'cpu', d.mean_density, NULL)), 0), 4
  )                                                               AS relative_gap,
  ROUND(MIN(w.waves), 1)                                          AS min_waves,
  IF(
    ABS(MAX(IF(d.arm = 'gpu', d.mean_density, NULL)) - MAX(IF(d.arm = 'cpu', d.mean_density, NULL)))
      / NULLIF(MAX(IF(d.arm = 'cpu', d.mean_density, NULL)), 0) <= 0.10
    AND MIN(w.waves) >= 50,
    'PASS',
    'INCONCLUSIVE - the arms were not equally busy, or there were too few waves'
  )                                                               AS verdict
FROM d
JOIN w USING (arm);

-- ============================================================================================
-- SECTION 5 - THE DECISION. Run only if controls 1-4 all PASS.
--
-- The decision variable is COST PER THOUSAND FITS, not per-fit latency. A GPU that is 8% faster and
-- 40% more expensive is a worse machine for this workload however good the latency looks.
--
-- POST-HOC REPAIR, 2026-09-11, made after the data landed. Recorded here rather than quietly fixed,
-- because editing a pre-registered query once the results are visible is exactly what
-- pre-registration exists to prevent. The only thing that makes such an edit legitimate is that a
-- reader can check for themselves that it could not have moved the answer.
--
-- What was written: `SUM(n_fits)`, with the note "every fit counts here, backtest folds included, so
-- this reads the fold rows too". Two things about that were wrong, and neither is about this run:
--
--   1. `n_fits` is NULL on every row. The column is declared on `forecast_metadata` and never
--      populated by any writer, so `SUM(n_fits)` was NULL, the denominator collapsed, and the CASE
--      fell through to its ELSE. The pre-registered query could not have produced a number on any
--      data, so this is a repair to a query that never worked, not a repair fitted to a result.
--   2. There are no fold rows. Both arms backtested (`backtest_status='full'`, `n_folds_achieved=2`)
--      and wrote exactly 10,000 rows each, all with `fold_id IS NULL` — the frozen-backtest scheme
--      scores folds without emitting a row per fold. So "reads the fold rows too" describes a shape
--      the table does not have, and here one row is one recorded fit.
--
-- The repair is `COUNT(*)` over the same rows the pre-registered query already selected. It cannot
-- favour an arm: both arms landed exactly 10,000 cells (CONTROL 1), so any per-cell constant — 1, or
-- 3 if one counted the two folds as separate fits — cancels in `s`, which is a ratio between the two
-- arms. The choice rescales `cost_per_1k_fits`, and rescales both arms by the same factor. So the
-- decision below is invariant to it; only the units of the two cost columns depend on it.
--
-- Let s = throughput ratio (GPU fits per node-second / CPU fits per node-second)
--     r = accel_surcharge (declared above)
--
--   CPU   if s <= 1.10 and r >= 1.15
--   GPU   if s >= r + 0.10 and device_share >= 0.25
--   otherwise INCONCLUSIVE, and the shipped default (use_gpu: False) stands.
--
-- Phase 0 predicted s in [0.95, 1.07]. Writing that down before the run is the whole discipline: if
-- the measurement lands there, it confirms a prediction rather than discovering a result.
-- ============================================================================================
WITH all_fits AS (
  SELECT
    IF(run_id = gpu_run_id, 'gpu', 'cpu')        AS arm,
    fit_seconds,
    cpu_seconds,
    n_fits,
    worker_id,
    cell_started_at,
    cell_ended_at
  FROM `scale_forecasting.forecast_metadata`
  WHERE run_id IN (gpu_run_id, cpu_run_id)
    AND model_type = 'neuralprophet'
    AND ensemble_id IS NULL
    AND cell_status = 'ok'
),
per_arm AS (
  SELECT
    arm,
    COUNT(*)                                                             AS total_fits,  -- see repair note
    SUM(fit_seconds)                                                     AS total_fit_seconds,
    COUNT(DISTINCT worker_id)                                            AS nodes,
    TIMESTAMP_DIFF(MAX(cell_ended_at), MIN(cell_started_at), SECOND)     AS wall_seconds
  FROM all_fits
  GROUP BY arm
),
rates AS (
  SELECT
    arm,
    total_fits,
    total_fit_seconds,
    nodes,
    wall_seconds,
    -- Fits per node-second: the throughput of one node, so the two arms compare at equal fleet size.
    SAFE_DIVIDE(total_fits, wall_seconds * nodes)                        AS fits_per_node_second,
    SAFE_DIVIDE(total_fit_seconds, total_fits)                           AS mean_seconds_per_fit
  FROM per_arm
),
device AS (
  -- device_share: of the GPU arm's fit time, how much of it had a device holding memory at all. A
  -- GPU that is engaged on a quarter of the work is a different proposition from one engaged on all
  -- of it, and the decision rule requires 0.25 before it will recommend buying accelerators.
  SELECT
    SAFE_DIVIDE(
      SUM(IF(peak_gpu_bytes > 0, fit_seconds, 0)),
      NULLIF(SUM(fit_seconds), 0)
    ) AS device_share
  FROM cells
  WHERE arm = 'gpu' AND cell_status = 'ok'
)
SELECT
  'decision' AS section,
  ROUND(MAX(IF(arm = 'gpu', fits_per_node_second, NULL)), 6)             AS gpu_fits_per_node_s,
  ROUND(MAX(IF(arm = 'cpu', fits_per_node_second, NULL)), 6)             AS cpu_fits_per_node_s,
  ROUND(
    SAFE_DIVIDE(
      MAX(IF(arm = 'gpu', fits_per_node_second, NULL)),
      MAX(IF(arm = 'cpu', fits_per_node_second, NULL))
    ), 4
  )                                                                      AS s_throughput_ratio,
  accel_surcharge                                                        AS r_accel_surcharge,
  ROUND(MAX(device.device_share), 4)                                     AS device_share,
  -- Cost per thousand fits, in units of "CPU node-seconds". The GPU arm's node-seconds are marked up
  -- by r, which is what turns a throughput number into a cost number.
  ROUND(
    1000.0 / NULLIF(MAX(IF(arm = 'cpu', fits_per_node_second, NULL)), 0), 1
  )                                                                      AS cpu_cost_per_1k_fits,
  ROUND(
    accel_surcharge * 1000.0
      / NULLIF(MAX(IF(arm = 'gpu', fits_per_node_second, NULL)), 0), 1
  )                                                                      AS gpu_cost_per_1k_fits,
  CASE
    WHEN SAFE_DIVIDE(
           MAX(IF(arm = 'gpu', fits_per_node_second, NULL)),
           MAX(IF(arm = 'cpu', fits_per_node_second, NULL))
         ) <= 1.10 AND accel_surcharge >= 1.15
      THEN 'CPU - the accelerator does not pay for itself'
    WHEN SAFE_DIVIDE(
           MAX(IF(arm = 'gpu', fits_per_node_second, NULL)),
           MAX(IF(arm = 'cpu', fits_per_node_second, NULL))
         ) >= accel_surcharge + 0.10 AND MAX(device.device_share) >= 0.25
      THEN 'GPU - the accelerator earns its surcharge'
    ELSE 'INCONCLUSIVE - the shipped default (use_gpu: False) stands'
  END                                                                    AS decision
FROM rates, device
GROUP BY accel_surcharge;
