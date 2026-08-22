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

## C1 — BigQuery-native models read a BigLake Iceberg source un-pinned (no snapshot time-travel)

- **Status:** Active workaround — adopted 2026-08-22.
- **Area:** `src/scale_forecasting/engines/bigquery_engine.py` (`_source_is_iceberg`, `run`).

**Context / limitation.**
Every run pins a single input snapshot so all runtimes read the *identical* source state: the run
header records one instant and each reader time-travels its source read to it. The Spark and Ray
readers pin via the Storage Read API's snapshot-time field; the BigQuery-native (BQML) models pin via
a `FOR SYSTEM_TIME AS OF` clause in their SQL.

BigQuery ML `CREATE MODEL` **cannot** time-travel a BigLake Iceberg source. A `CREATE MODEL ... AS
(SELECT ... FROM <iceberg_table> FOR SYSTEM_TIME AS OF <ts>)` statement is rejected with a 400,
`'FOR SYSTEM_TIME AS OF' expression ... evaluates to a TIMESTAMP value in the future`, even though
`<ts>` is a valid, already-committed instant several seconds in the past. It is **not** a clock-skew
or metadata-freshness issue and it is **not** intermittent:

- A plain `SELECT ... FROM <iceberg_table> FOR SYSTEM_TIME AS OF <ts>` at the *same instant* with the
  *same timestamp* succeeds and returns the full table.
- The identical `CREATE MODEL` with the `FOR SYSTEM_TIME AS OF` clause **removed** succeeds.
- Reproduced deterministically against a source table that had not changed in weeks.

So the rejection is specific to the `CREATE MODEL` code path combined with a BigLake Iceberg source.
A native (non-Iceberg) BigQuery table time-travels correctly inside `CREATE MODEL`.

**Decision.**
When the native subset's source resolves to a BigLake Iceberg table, drop the snapshot pin for the
**entire** native subset — every native read (series selection, model training, backtest history and
evaluation) runs un-pinned against the live table. We unpin the whole subset rather than only
`CREATE MODEL` so the native reads stay internally consistent (models train on the same state the
backtest evaluates against). Detection is at runtime via table metadata
(`biglakeConfiguration.tableFormat == "ICEBERG"`); a native source keeps the pin. This is a
best-effort check — if the lookup fails we keep pinning (the safe default). The Spark and Ray paths
are unaffected; they continue to pin the snapshot.

**Impact.**
- The cross-runtime "every job reads the identical snapshot" guarantee is relaxed **for native
  models when the source is Iceberg**: native models read whatever the table holds at query time,
  which can differ from the Spark/Ray snapshot *if the source table is mutated while a run is in
  flight*. For the normal case — a source panel that is static during a run — the data read is
  identical, so there is no practical difference.
- No change for native BigQuery source tables, and no change to the Spark/Ray runtimes on any source.
- The run still records a snapshot on its header (used by the pinned runtimes); the native path
  simply does not apply it for an Iceberg source.

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
