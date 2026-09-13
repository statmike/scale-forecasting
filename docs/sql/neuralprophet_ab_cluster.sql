-- ============================================================================================
-- The GPU-vs-CPU A/B for NeuralProphet, on a **Dataproc cluster**: controls first, then the
-- decision.
--
-- Pre-registered. This file is committed BEFORE either arm is submitted, and
-- tests/unit/test_ab_preregistration.py asserts that it exists and still asks these questions.
-- An analysis written after the numbers land is an analysis fitted to the numbers.
--
-- The two arms are configs/neuralprophet_ab_cluster_gpu.json and
-- configs/neuralprophet_ab_cluster_cpu.json: the same 3,000 NeuralProphet cells on the same
-- four-worker n1-standard-8 cluster, seven one-core slots per worker on both, differing only in
-- whether each worker carries a T4.
--
-- This is the second half of one experiment, not a separate one. The Ray pair
-- (docs/sql/neuralprophet_ab.sql) answered the same question on the other runtime and landed on
-- CPU; the decision rule in section 3 of PLAN_REFINEMENT_V2 requires both runtimes before the
-- shipped default is treated as settled, because a result on one scheduler is a result about that
-- scheduler. Everything structural below is deliberately identical to the Ray file so the two
-- analyses can be read side by side. What differs, and why, is written down at each point.
--
-- Read the sections in order. Sections 1-4 are CONTROLS. If any of them fails the experiment is
-- INCONCLUSIVE and section 5 is not to be run, re-run, or reinterpreted. That rule is the point of
-- pre-registering: the temptation to relax a control arrives only after you have seen the answer.
--
-- Usage:
--   bq query --project_id=<project> --use_legacy_sql=false < docs/sql/neuralprophet_ab_cluster.sql
-- Set the DECLAREs below first.
-- ============================================================================================

DECLARE gpu_run_id STRING DEFAULT 'neuralprophet-ab-cluster-gpu-273d32b553c8';
DECLARE cpu_run_id STRING DEFAULT 'neuralprophet-ab-cluster-cpu-a402414abf5e';

-- The accelerator surcharge `r`: the price of one GPU worker divided by the price of one CPU
-- worker, same machine type, same region, at the rate actually billed.
--
-- 1.76, and note that it is NOT the 1.92 the Ray file declares, for a reason that is about billing
-- rather than about hardware. Both files price the same n1-standard-8 (~$0.380/hour) and the same
-- T4 (~$0.350/hour). Dataproc additionally bills its own premium of ~$0.010 per vCPU-hour, which on
-- an 8-vCPU worker is ~$0.080/hour — and it lands on BOTH arms, so it enlarges the denominator
-- without touching the accelerator: (0.380 + 0.080 + 0.350) / (0.380 + 0.080) = 1.76. A managed
-- service that charges per core therefore makes an accelerator look *relatively* cheaper than the
-- same card does on raw VMs. The master node is identical on both arms and cancels, so it is not
-- in the ratio.
--
-- Substitute your own contract. The decision rule reads `r` rather than hard-coding a threshold, so
-- changing it re-runs the decision honestly instead of requiring a re-argued one.
DECLARE accel_surcharge FLOAT64 DEFAULT 1.76;

-- Total concurrent cell slots per arm: four workers x seven slots. Both arms plan exactly this;
-- tests/unit/test_ab_preregistration.py asserts it offline, so it is a constant here rather than
-- another thing to look up after the fact. Four workers because the project's Compute Engine T4
-- quota in the deployment region is four cards, which is what sizes this experiment — the CPU arm
-- is held to the same four so the fleets match.
DECLARE total_slots FLOAT64 DEFAULT 28.0;

-- Bucket width for the steady-state windows, kept at the Ray file's thirty minutes. There is an
-- argument for narrowing it here — a cluster's workers all exist from create, so there is no
-- autoscaler ramp for the first bucket to absorb — but narrowing a window after choosing an
-- experiment size is the kind of adjustment that is indistinguishable from tuning, so it stays.
-- The first bucket still has something real to absorb: cluster create, the driver's Iceberg read,
-- and the cold first fit on each worker.
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
--
-- This control carries extra weight on the cluster path, because the cluster's device is shared
-- differently from Ray's: there is no YARN GPU isolation, so nothing but `spark.task.cpus` stops
-- seven cells from each believing they own the whole card (see resources/cluster.py). If the device
-- were absent or unreachable the fits would simply run on CPU and the "GPU arm" would be a second
-- CPU arm with a T4 on the invoice.
-- ============================================================================================
SELECT
  'control_2_device' AS control,
  arm,
  COUNTIF(peak_gpu_bytes IS NOT NULL)            AS cells_with_device_bytes,
  MAX(peak_gpu_bytes)                            AS max_peak_gpu_bytes,
  COUNT(DISTINCT device_used)                    AS distinct_device_used,
  ANY_VALUE(device_used)                         AS a_device_used,
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
--
-- On this runtime the confound had a name. Left unpinned, `gpu_fraction` resolves to a half card on
-- the GPU arm, `_cluster_task_cpus` widens each task to three cores to honour two cells per device,
-- and the GPU arm runs two cells per worker against the CPU arm's seven — a 3.5x concurrency gap
-- and three threads a fit. Both configs pin `gpu_fraction: 0.125` to close it, and this control is
-- what would catch the pin being dropped.
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
-- delivered per second of wall clock. It is the check that the two fleets were equally busy.
--
-- Buckets are 30 minutes wide. The FIRST and LAST bucket of each arm are dropped: the first holds
-- the cold start and the last holds the drain, and neither is steady state. Active nodes come from
-- COUNT(DISTINCT worker_id) within the bucket, which counts nodes that actually ran a cell rather
-- than nodes that were paid for.
--
-- Two thresholds: the per-arm density must agree within 10%, and each arm needs at least 50 waves
-- (cells divided by total slots) for the average to mean anything. At 3,000 cells over 28 slots the
-- planned figure is 107 waves, so the floor has roughly a 2x margin — chosen that way because the
-- four-card quota caps concurrency here at a third of the Ray fleet's, and a wave count sized to
-- the floor would be one failed cell away from INCONCLUSIVE.
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
  SELECT arm, COUNT(*) / total_slots AS waves FROM cells WHERE cell_status = 'ok' GROUP BY arm
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
-- The fit count is derived from the backtest columns rather than read from `n_fits`. That is not a
-- refinement, it is the only form that works: `n_fits` is declared in the `forecast_metadata` row
-- spec (`registry/write_api.py`) and populated by no writer, so it is NULL on every row ever
-- written. The Ray file's version of this query used it, produced a NULL denominator, and fell
-- through to INCONCLUSIVE — a repair recorded there as a dated note. This file simply starts from
-- the working form. With `backtest_refit='per_fold'` a cell fits once per fold plus once on full
-- history, so `n_folds_achieved + 1`; with a frozen scheme the fold scores reuse one fit, so 1.
-- Both arms here are `per_fold` at 2 folds, so 3 fits per cell and 9,000 per arm — and because both
-- arms run the identical backtest block, any per-cell fit constant cancels in `s`, which is a ratio
-- between the arms.
--
-- Let s = throughput ratio (GPU fits per node-second / CPU fits per node-second)
--     r = accel_surcharge (declared above)
--
--   CPU   if s <= 1.10 and r >= 1.15
--   GPU   if s >= r + 0.10 and device_share >= 0.25
--   otherwise INCONCLUSIVE, and the shipped default (use_gpu: False) stands.
--
-- The prediction, written down before the run and stated here so the result can confirm or refute
-- it rather than be discovered: s in [0.80, 1.00]. That is narrower and lower than the Ray file's
-- [0.95, 1.07], because the Ray pair has already measured s = 0.82 on this model — a T4 making each
-- fit about 30 seconds slower, on a network that allocates 87 KB of device memory. Nothing about
-- the scheduler should change the arithmetic of a fit, so the expectation is the same answer again.
-- A cluster result far from 0.82 would be the interesting outcome, and would say the difference is
-- in how the two runtimes share a device rather than in the device.
-- ============================================================================================
WITH all_fits AS (
  SELECT
    IF(run_id = gpu_run_id, 'gpu', 'cpu')        AS arm,
    fit_seconds,
    cpu_seconds,
    IF(backtest_refit = 'per_fold', COALESCE(n_folds_achieved, 0) + 1, 1) AS fits_in_cell,
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
    SUM(fits_in_cell)                                                    AS total_fits,
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
    SAFE_DIVIDE(total_fits, wall_seconds * nodes)                        AS fits_per_node_second,
    SAFE_DIVIDE(total_fit_seconds, total_fits)                           AS mean_seconds_per_fit
  FROM per_arm
),
device AS (
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
