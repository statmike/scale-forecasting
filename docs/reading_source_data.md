# Reading the source data — how each runtime reads the panel

Every run reads the **same** source: one BigQuery table (native or managed-Iceberg) holding the long
panel of `(series, timestamp, target[, exog])` rows. Each runtime reads it its own way, but they all
agree on four things — **what** columns they read, **which** rows, **as of when**, and **through which
API**. This doc is the reader's-eye view: open it when you want to understand how the panel gets into
a cell, or how to bound the read's parallelism.

For the config knobs mentioned here see [configuration_reference.md](./configuration_reference.md);
for where the *results* go see [output_schemas.md](./output_schemas.md); for the end-to-end flow see
[architecture.md](./architecture.md).

---

## The four invariants (true for every reader)

1. **Column projection.** A cell needs only the id, date, and target columns plus any configured
   `features.exog`, so every reader projects to exactly those columns — never `SELECT *`. Narrow rows
   matter because the Spark/Ray fan-out cross-joins each series once per model, so an unused column is
   paid for on every cell. (The projection is order-preserving and de-duplicated.)
2. **Deterministic subset.** With `data.series_limit` set, each reader keeps the *same* first N series
   — distinct ids, ordered, first N — so "10 vs 100 vs 100k series" is a clean apples-to-apples
   runtime comparison rather than a different sample each time. Unset = the whole panel.
3. **Snapshot pinning.** A run records one input snapshot on its header (a BigQuery time-travel
   instant, taken with a small safety margin). Every family job of that run pins its read to that
   instant, so all families read **byte-identical** source data even if the table is written to
   mid-run. A missing snapshot leaves the read unpinned (best-effort). Each reader expresses the pin
   in its own dialect (below).
4. **Storage Read API, Arrow.** The Python runtimes read through the **BigQuery Storage Read API** in
   **Arrow** format — the columnar, zero-copy path into the executor-side pandas frames the cells
   consume. The Storage Read API does **not** consume query slots, so a wide fan-out doesn't compete
   with the analyst queries on the project. (The BigQuery-native family is the exception — it never
   leaves BigQuery; see below.)

---

## Per-runtime read paths

### Spark (`explode`)

`read_source_series` reads through the **spark-bigquery connector**
(`spark.read.format("bigquery")`). It sets `readDataFormat=ARROW` **explicitly** (rather than relying
on a connector default), applies the column projection with `.select(...)`, and enforces
`series_limit` with a deterministic semi-join. The snapshot pin is the connector's
`snapshotTimeMillis` time-travel option.

### Ray — `driver_collect` (default)

`_read_driver_collect` reads with the `BigQueryReadClient` (`create_read_session`) directly, in
`DataFormat.ARROW`. The snapshot pin is the Storage Read API's native `table_modifiers.snapshot_time`
field. This is the proven default path.

### Ray — `ray_data` (opt-in)

`_read_ray_data` uses the Ray-native `ray.data.read_bigquery` reader, which reads over the **same**
Storage Read API underneath. Because that reader's table-scan form doesn't expose a snapshot option,
a *pinned* read falls back to a `FOR SYSTEM_TIME AS OF TIMESTAMP_MILLIS(...)` query; an unpinned read
stays a pure table scan. Select it with `compute.ray_read_mode="ray_data"`.

### BigQuery-native (`arima_plus`, `arima_plus_xreg`, `timesfm`)

The native family never leaves BigQuery — it reads the source **via the query API** as a subquery
inside its BQML SQL, not through the Storage Read API. Its snapshot pin is a `FOR SYSTEM_TIME AS OF`
clause spliced into that subquery. (This is why `read_max_streams`, below, is inert for native
models.)

---

## Bounding read parallelism — `read_max_streams`

`compute.read_max_streams` caps the number of Storage Read streams the source read requests, shared
across the two engines that read through the Storage Read API:

| Reader | How the cap is applied |
|--------|------------------------|
| Spark connector | the connector's `maxParallelism` option |
| Ray `driver_collect` | `create_read_session`'s `max_stream_count` |

`0` (the default) lets the **server** size the stream count from the table — the known-good default;
leave it there unless you have a reason not to. Set a **positive** value to bound read parallelism —
e.g. to stay inside a slot or quota budget on a shared project. The knob is **inert** for the
`ray_data` path (Ray sizes its own blocks) and for BigQuery-native models (they read via the query
API). Because it's part of the config, changing it yields a new `run_id`.

---

## Why one table, both formats, one read path

The source table can be a **native** BigQuery table or a **managed-Iceberg** table, and every reader
above works against both unchanged — they all go through BigQuery's table interface (the Storage Read
API for the Python runtimes, the query API for native), which reads either format transparently. There
is no per-format read fork to maintain; the same code reads whichever format the deployment provisions.

### Direct Iceberg read — a deliberate non-goal

A managed-Iceberg source could in principle be read **directly** from its Parquet/metadata in object
storage (e.g. via the Iceberg Java reader or a BigLake external path), bypassing the Storage Read API
entirely. We are **not** building that, and the reason is not effort — it is that the second path
costs more than it saves:

- **It would fork the read.** Today one code path reads both formats, so there is no per-format
  branch to keep in sync across three runtimes. A direct reader is native-vs-Iceberg forever.
- **It would lose the snapshot semantics the design depends on.** Every source read is pinned to one
  BigQuery time-travel timestamp so a run is reproducible and every family in a multi-runtime DAG
  sees byte-identical input. Object-storage reads would have to re-derive that from Iceberg snapshot
  ids — a second, weaker mechanism for a guarantee we already have for free.
- **The saving is narrow.** It buys read cost on very wide scans. A forecasting run reads the panel
  **once** and then fits `series × models` models on it, so the read is amortized over the
  expensive part by construction.

Revisit only if a measured run shows the source read is a material share of wall-clock or spend —
that would be a change in the facts, not a change of mind. Until then this is closed, not pending.

---

## See also

- [configuration_reference.md](./configuration_reference.md) — `read_max_streams`, `ray_read_mode`,
  `series_limit`, and the rest of the `data`/`compute` knobs.
- [architecture.md](./architecture.md) — how a read feeds the engine fan-out and the unit of work.
- [output_schemas.md](./output_schemas.md) — the tables a run *writes*.
