# Considerations

A living log of **deliberate workarounds and trade-offs** this project makes in response to
platform limitations outside our control. Each entry records *what* we did, *why*, the *impact*, and
the *evidence/references* that forced it — so any entry can be re-evaluated later (a platform fix, a
new API, or a changed requirement may make the workaround unnecessary).

This is not a bug tracker and not an architecture-decision record for our own design choices; it is
specifically the set of places where an external constraint made us deviate from what we would
otherwise have built. When you touch code that an entry covers, check whether the constraint still
holds.

Each entry: **Status** · **Context / limitation** · **Decision** · **Impact** · **References** ·
**Re-evaluation trigger**.

---

## C1 — BigQuery-native models read the source un-pinned (no snapshot time-travel)

- **Status:** Active workaround — adopted 2026-08-22; broadened to all source types 2026-08-25.
- **Area:** `src/scale_forecasting/engines/bigquery_engine.py` (`run`).

**Context / limitation.**
Every run pins a single input snapshot so all runtimes read the *identical* source state: the run
header records one instant and each reader time-travels its source read to it. The Spark and Ray
readers pin via the Storage Read API's snapshot-time field; the BigQuery-native (BQML) models would
pin via a `FOR SYSTEM_TIME AS OF` clause in their SQL.

BigQuery ML `CREATE MODEL` **cannot** time-travel to a constant snapshot instant. A `CREATE MODEL ...
AS (SELECT ... FROM <table> FOR SYSTEM_TIME AS OF <ts>)` statement is rejected with a 400,
`'FOR SYSTEM_TIME AS OF' expression ... evaluates to a TIMESTAMP value in the future`, even though
`<ts>` is a valid, already-committed instant. It is **not** a clock-skew or metadata-freshness issue
and it is **not** intermittent:

- A plain `SELECT ... FROM <table> FOR SYSTEM_TIME AS OF <ts>` at the *same instant* with the *same
  timestamp* succeeds and returns the full table.
- The identical `CREATE MODEL` with the `FOR SYSTEM_TIME AS OF` clause **removed** succeeds.
- The rejection fires for a fixed `<ts>` committed *hours* earlier, yet the **same** `CREATE MODEL`
  succeeds when the AS OF is written as a `CURRENT_TIMESTAMP()`-relative expression
  (`TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL n MINUTE)`). So `CREATE MODEL` accepts only a
  current-timestamp-relative AS OF, never a constant snapshot instant — the recency of `<ts>` is not
  the axis; being a fixed value is.
- Reproduced deterministically against both a BigLake Iceberg source **and** a plain native BigQuery
  table that had not changed in weeks. The original scoping to Iceberg-only (2026-08-22) was
  incomplete; a native table exhibits the identical rejection (confirmed 2026-08-25).

So the rejection is specific to the `CREATE MODEL` code path combined with a **constant**
`FOR SYSTEM_TIME AS OF`, independent of source format.

**Decision.**
Drop the snapshot pin for the **entire** native subset, for **any** source type — every native read
(series selection, model training, backtest history and evaluation) runs un-pinned against the live
table. We unpin the whole subset rather than only `CREATE MODEL` so the native reads stay internally
consistent (models train on the same state the backtest evaluates against). No table-metadata probe
is needed any more (the pin is simply never applied on the native path). The Spark and Ray paths are
unaffected; they continue to pin the snapshot.

**Impact.**
- The cross-runtime "every job reads the identical snapshot" guarantee is relaxed **for native
  models**: they read whatever the table holds at query time, which can differ from the Spark/Ray
  snapshot *if the source table is mutated while a run is in flight*. For the normal case — a source
  panel that is static during a run — the data read is identical, so there is no practical difference.
- No change to the Spark/Ray runtimes on any source.
- The run still records a snapshot on its header (used by the pinned runtimes); the native path
  simply does not apply it.

**References.**
- [Access historical data (`FOR SYSTEM_TIME AS OF`)](https://cloud.google.com/bigquery/docs/access-historical-data)
- [Data retention with time travel](https://cloud.google.com/bigquery/docs/time-travel)
- [BigLake Iceberg tables in BigQuery](https://cloud.google.com/bigquery/docs/biglake-iceberg-tables-in-bigquery)
- [BigQuery ML `CREATE MODEL` syntax](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create)
- The specific `CREATE MODEL` + Iceberg + time-travel rejection is **not documented** by Google as of
  2026-08-22; it was found empirically by the A/B tests above.

**Re-evaluation trigger.**
Retry a pinned `CREATE MODEL` against a BigLake Iceberg source; if BigQuery ML accepts
`FOR SYSTEM_TIME AS OF` there, remove `_source_is_iceberg`'s special-case and let the native path pin
Iceberg sources like every other runtime. Worth checking on BigQuery ML / BigLake Iceberg release
notes.

---

## C2 — On Serverless Spark, executor shape is a coarse, indirect control

- **Status:** Active constraint — documented 2026-08-30. Shapes the compute-profiler design; no
  behaviour change on its own.
- **Area:** `src/scale_forecasting/submit.py` (`_serverless_gpu_properties`, executor properties);
  `src/scale_forecasting/config.py` (`FamilyCompute` validator).

**Context / limitation.**
Every runtime we target lets us size a unit of work, but Serverless for Apache Spark exposes that
sizing through a narrower and more coupled interface than the others. Four distinct constraints:

1. **`spark.executor.cores` is a short enumeration, not a number.** Non-GPU workloads accept
   **4, 8, or 16** (default 4). GPU (L4) workloads accept a *different* set — **4, 8, 12, 16, 24,
   48, 96** — where 24/48/96 additionally attach **2/4/8** GPUs per executor rather than one. An
   arithmetically-derived core count is therefore usually illegal and must be snapped to a member of
   the applicable set.
2. **GPU concurrency is not independently settable; it is `1/cores`.** Serverless applies
   `spark.executor.resource.gpu.amount=1` and `spark.task.resource.gpu.amount=1/$spark_executor_cores`
   as service defaults, and rejects attempts to set them explicitly. Fractional GPU scheduling *is*
   available — but the fraction is the reciprocal of the CPU concurrency, so GPU packing and CPU
   packing cannot be tuned separately. On Ray (and on a Dataproc cluster) they can.
3. **On the GPU path, `spark.executor.memoryOverhead` is restricted.** The docs state memory may be
   set but overhead may not. Since PySpark charges the Python worker's footprint — which is where
   *all* of our model fitting happens — to `memoryOverhead` (defaulted to 40% for PySpark rather
   than the usual 10%), the pool that actually holds our working set is the one we cannot address
   directly on GPU workloads. It can only be moved indirectly, by choosing `spark.executor.memory`
   so the derived overhead lands where we need it.
4. **Related fixed values.** `spark.dataproc.executor.disk.size` is pinned at 375 GB for L4 (any
   other value errors); accelerator type is **L4 only** (T4 is rejected); `spark.executor.instances`
   is bounded to [2, 2000]; and GPU workloads are incompatible with the organization policy
   `constraints/compute.requireShieldedVm`.

**Decision.**
Use Serverless as-is and treat the constraint as part of the plan, not as an error path:

- Resource planning computes the *ideal* shape from measured need, then **snaps to the nearest legal
  value** — downward for GPU concurrency (fewer tasks sharing a device is the safe direction),
  upward for memory. Both the ideal and the snapped value are recorded, so an audit shows what was
  wanted and what the platform allowed.
- An illegal cores/memory pair is rejected **at plan time, offline**, rather than surfacing as a
  submit-time API error minutes into a run.
- On the GPU path, the target per-task GPU share is expressed as a choice of `spark.executor.cores`
  rather than as a fraction, because on this runtime those are the same decision.
- Where the platform's shape cannot express what we measured, we accept the nearest safe
  over-provision rather than tuning around it.

**Impact.**
- Serverless can be left less densely packed than Ray for the same measured workload, because the
  legal core counts are coarse and GPU/CPU packing share one knob. The cost is idle headroom, not
  correctness.
- A run whose family is pinned to Serverless cannot express "many small CPU tasks, few large GPU
  tasks" within one executor; splitting that intent across runtimes (per-family `runtime`) is the
  supported way to get it.
- The GPU-memory ceiling on Serverless is reached through `spark.executor.memory`, which reads
  backwards relative to every other runtime; the indirection is a documented step, not an accident.
- No impact on Ray-on-Vertex or Dataproc-cluster paths, which expose the direct controls.

**References.**
- [Run Spark workloads with GPUs (Serverless)](https://docs.cloud.google.com/managed-spark/docs/guides/gpus-serverless)
- [Spark properties (Serverless)](https://docs.cloud.google.com/dataproc-serverless/docs/concepts/properties)
- [Serverless autoscaling / dynamic allocation](https://docs.cloud.google.com/dataproc-serverless/docs/concepts/autoscaling)
- [Spark stage-level scheduling / GPU resource properties](https://spark.apache.org/docs/latest/configuration.html)

**Re-evaluation trigger.**
Check whether Serverless has begun accepting an explicit `spark.task.resource.gpu.amount` (which
would decouple GPU packing from executor cores), whether `spark.executor.memoryOverhead` becomes
settable on GPU workloads, or whether the legal `spark.executor.cores` sets widen. Any of the three
lets the planner emit the measured shape directly instead of snapping to it. Worth checking on
Serverless for Apache Spark release notes and the runtime-version release notes when the pinned
runtime version is next bumped.
