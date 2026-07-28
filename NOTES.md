# NOTES.md — build log

Running log of deviations, surprises, and decisions made during the build. Append-only,
newest at the bottom. Keep entries short: what, why, and the contract section touched.

---

- **0.1 scaffold** — created the package tree per CONTRACTS §6 / DESIGN §6. All
  later-owned files are stubs raising `NotImplementedError` with a pointer to their BUILD
  step; `errors.py` is fully implemented (it's foundational and tiny). No deviations.

- **A-END: Arc A complete** — the whole offline core is built and green (241 passed,
  5 skipped, ruff + mypy clean). Handoff notes for Arc B:
  - **Optional deps are register-but-skip.** prophet, xgboost, lightgbm, neuralprophet
    import lazily inside `fit()`, so every model registers and the factory is complete
    even when the extra is absent; the contract test skips the missing ones. `neuralprophet`
    0.9.0 is incompatible with pandas 3.x (calls the removed `Series.view`) — its model code
    is correct but unreachable until upstream fixes it; it ships as an optional extra that
    registers and skips. Don't put it in the packed cluster venv yet.
  - **Metric panel single source of truth.** `config.DecisionMetric` (the literal) is the
    canon; `metrics.METRIC_NAMES`, `registry/bq.METRIC_COLUMNS` (derived via `get_args`),
    and the DDL columns are all pinned to it by tests. Add a metric in one place → add it
    to `metrics.compute_metrics` + the DDL, and the tests will tell you if they drift.
  - **The G1 seam is done and never raises.** `worker.run_cell` returns a `CellResult`
    (status ok|error) for every input; error cells carry empty predictions + NaN metrics +
    the identity. Engines (Arc B) only fan it out and collect — no per-environment logic.
  - **Pure/IO seam in `registry/bq.py`.** All row assemblers are pure and tested now;
    `ensure_tables`/`write_header`/`update_header`/`write_cells` are the only stubs left
    there (BUILD B1) — wire them to the Storage Write API, idempotent per `model_hash`
    (`cell_dedup_key`).
  - **Data generator partition-union invariant** holds (each series seeded by its own
    index) — the Spark seed job (`data_gen/seed_spark.py`, B-stub) can partition `range(n)`
    any way and the union equals `generate_panel(n)`.
  - **Ensembler:** calculated strategies render as BigQuery SQL (`build_ensemble_sql`,
    correct `INSERT INTO (cols) WITH cte SELECT` form); learned strategies (`fit_learned`)
    train on OOF and refuse to run without backtest. Arc B just needs to execute the SQL
    and persist/apply the learned weights.
  - **Remaining stubs for Arc B:** `main.run`, `router.split_by_runtime`, all four
    `engines/*`, `data_gen/seed_spark`, the four `registry/bq` writers, and `bigquery_native`
    fit/predict (raise `NotImplementedError(_ARC_B)`).

- **Pre-deploy hardening (post A-END).** Added a shared `seasonality.py` (one freq→period
  source for the 5 seasonal models + Fourier features) and `validation.py` (pre-flight
  input-contract validator: fail-fast, names the first offending series/column/value; we
  *validate*, never *prepare* — no resampling/tz). Reconciled the exog seam on the business
  name `price_index` (source table carries business names; `features.exog` list *is* the
  role assignment; no synthetic `exog_1` column). Generator is now freq-general (seasonality
  measured in steps; daily output byte-for-byte unchanged).
  - **pandas 3 dropped the `M` and `H` freq aliases** → use `ME` (month-end) and `h`
    (hourly). `SUPPORTED_FREQS` is now `(D, W, MS, ME, h)`. This was latent: nothing ran a
    monthly/hourly panel end-to-end before, so `date_range(freq="M")` would have raised only
    in production. `seasonality.py` is the SSOT for the spellings.

- **Arc B B0.1–B0.2 (Terraform authored, not yet applied).** Two-stage state:
  `terraform/bootstrap` (local state → creates project + GCS state bucket) and
  `terraform/main` (GCS backend → everything else). Small one-capability modules mirroring
  the Python side: `apis`, `iam` (scale-forecasting-runner/-compute, no keys, least-privilege),
  `storage` (warehouse/artifacts/code buckets), `bigquery` (dataset + BigLake connection +
  warehouse grant), `budget` (50/90/100% alerts), `composer` (Composer 3, gated). Greenfield
  default + BYO toggles (`create_project`, `enable_apis`, `create_service_accounts`,
  `create_composer`). Gate: `terraform validate` green on both stages + `fmt` clean.
  - **DEVIATION from BUILD B0.2** ("registry tables via Terraform"): the five tables' DDL is
    single-sourced in `registry/ddl.py` (snapshot-tested) and created by
    `registry.bq.ensure_tables()` at run time. Terraform owns the *containers* (dataset,
    connection, bucket grants), the app owns the *tables* — one source of truth for the DDL,
    no HCL/Python drift. Table creation moves to Arc B step B1 (registry round-trip).
  - **Composer is a first-class on/off toggle** (`create_composer`, default false): the only
    at-rest cost, and many deployments run the pipeline ad-hoc (notebook/local) with no
    scheduler. Module header documents start (`=true` + apply) → run (DAG orchestrates
    fan-out) → stop (`=false` + apply destroys just the env; data/registry untouched).
  - **Not yet applied** — awaiting billing account id + human review of `terraform plan`
    before any spend. Composer image pinned `composer-3-airflow-2.10.5-build.0` (verify/bump
    to a currently-offered Composer 3 build at apply time).

- **Arc B B0 apply — bootstrap live, main reviewed (naming polish before main apply).**
  Bootstrap applied cleanly (project `statmike-scale-forecasting` + state bucket). First apply
  attempt failed on `billing.resourceAssociations.create`; resolved by granting the dev
  identity `roles/billing.user` at the **org** level (798987785246) — the billing account was
  not directly editable, org-scoped grant cascaded. Two review decisions on the main stage,
  made before applying it:
  - **Service accounts renamed** `sf-runner`/`sf-compute` → `scale-forecasting-runner`/
    `scale-forecasting-compute`. Full names read self-evidently in the IAM console; both fit
    the 30-char `account_id` limit (24/25). No applied resources to migrate (main not applied
    yet).
  - **Kept three GCS buckets** (warehouse/artifacts/code) rather than one with folder
    prefixes — reasoning captured in `modules/storage/main.tf` header. Decisive points: GCS
    applies IAM/versioning/lifecycle/force_destroy at the *bucket* level (folders are just name
    prefixes); the BigLake connection SA needs objectAdmin scoped to *warehouse only*; and
    code (derivable, GitHub-backed, wants force_destroy) vs artifacts (G3 lineage, must never
    be force_destroyed) have opposite retention postures. Cost is identical (GCS bills per byte,
    not per bucket). Also rejected merging artifacts+code for the same retention-posture reason.
  - **Two apply-time fixes** (both real GCP eventual-consistency / ADC-quota gaps, not design
    errors — the config was valid, the cloud needed coaxing):
    1. **Budget needed a quota project.** `billingbudgets.googleapis.com` refuses ADC requests
       with no quota project set. Fix: a dedicated `google.billing_quota` provider alias
       (`billing_project` + `user_project_override`) used ONLY by the budget module — NOT the
       default provider, because that override also routes the Service Usage (API-enable) calls,
       which can't be billed to a project where Service Usage isn't enabled yet (bootstrap
       deadlock). Added `serviceusage` + `billingbudgets` to the apis module. No global ADC
       mutation (DESIGN §13.0 honored).
    2. **BigLake connection service-agent race.** The connection's `bqcx-…@gcp-sa-bigquery-condel`
       agent is provisioned async; the warehouse-bucket grant referenced it before it existed
       ("service account ... does not exist"). Fix: a `time_sleep` (20s) between connection and
       grant — standard idiom.
  - **APPLIED & verified** (2026-07-27): project + 12 APIs + 2 SAs + 10 grants + 3 buckets +
    dataset + connection + connection grant + budget. Composer gated off (0 composer resources).
    `terraform output` gives dataset/connection/warehouse_uri/buckets/SA emails for the run
    config. Near-zero cost at rest. Next: B0.3 seed write-path spike.

- **Arc B B0.3 write-path spike (LIVE infra) — findings that reshape the writers.** Ran a
  throwaway spike (`spikes/`, gitignored) that reused the REAL pure code (generate_panel,
  render_create_tables, assemble_prediction_rows) to test both write routes against the live
  managed-Iceberg tables. Managed Iceberg (GA 2026) has real constraints the writers must honor:
  1. **No native JSON column type.** DDL had `raw_config`/`best_params`/`quantiles` as `JSON`;
     Iceberg rejects it. FIXED in `ddl.py` → `STRING` (the assemblers already emit JSON strings
     via `json.dumps`/`_as_json`, so STRING matches the data; read back with `PARSE_JSON`).
     Snapshot regenerated; test_ddl green.
  2. **No `WRITE_TRUNCATE` on load.** Iceberg load jobs reject truncate; documented pattern is
     `DELETE FROM t WHERE TRUE` then `WRITE_APPEND`. → seed job (B0.4) uses delete-then-append.
  3. **No legacy streaming (`tabledata.insertAll`) on partitioned BigLake managed tables.** BQ
     error explicitly says "use the Write API." So `insert_rows_json` is OUT for the registry
     writers.
  - **Route decisions (verified working):**
    - **Route 2 — example/seed data → `source_series`:** LOAD JOB (delete-then-append). PASS.
    - **Route 1 — per-series/per-forecast results → registry tables:** the Storage Write API is
      the supported streaming path (legacy streaming blocked). A **query-based `INSERT`** also
      works and is the simplest robust path for small/medium batches (PASS in spike). B1 will use
      the Storage Write API for the high-fanout worker writes (millions of rows) and can fall back
      to query-INSERT where simpler.
    - **Idempotency:** `forecast_predictions` has NO `model_hash` column (that's in
      `forecast_metadata`); predictions key on `(run_id, ts_id, model_type)`. DELETE-by-key +
      INSERT, run 2x, produced no duplicates. PASS. (Note: `cell_dedup_key` returns
      `{run_id, model_hash}` — correct for metadata/oof tables, NOT predictions. B1 needs
      per-table dedup keys.)

- **Arc B B0.3 — Storage Write API proven end-to-end (the chosen route-1 path).** Extended the
  spike with a real Storage Write API append (dynamic proto descriptor matching the table +
  default stream) — the reference the B1 writers lift. Two more findings, both baked into infra
  or the B1 design:
  1. **Connection SA needs `storage.buckets.get`, not just object access.** The Write API
     streaming path checks bucket-metadata permission on the warehouse bucket; load jobs and
     query-INSERT don't (they only touch objects), which is why those passed while streaming
     404'd… actually 403'd: "connection … does not have permissions storage.buckets.get". No
     single predefined role has both object ops AND buckets.get except `storage.admin` (too
     broad). FIX (applied to live infra): TWO scoped grants on the warehouse bucket for the
     connection SA — `roles/storage.objectUser` (object read/write/delete) +
     `roles/storage.legacyBucketReader` (the one bucket-metadata read). `modules/bigquery`
     now renders both (`conn_warehouse_objects` + `conn_warehouse_bucket`). NOTE: `objectUser`
     does NOT include `buckets.get` — I assumed it did and had to add the second grant.
  2. **The default write stream buffers rows; DELETE/UPDATE can't touch the buffer.** BigQuery
     rejects `DELETE … WHERE` over rows still in the Write API streaming buffer ("would affect
     rows in the streaming buffer, which is not supported"). So the DELETE-by-key idempotency
     pattern CANNOT be combined with a same-key Write API append while the buffer is hot. B1's
     idempotency must therefore lean on **run_id being unique per run** (each cell writes
     exactly once — no per-cell delete), and a *whole-run* retry either uses a fresh run_id or
     deletes the run partition only after the buffer flushes. Query-INSERT (step 3b/4) is
     unaffected (no buffer) and remains the simple path for small/medium batches + the header
     row.
  - **All 7 spike checks green (exit 0):** ddl×2, route2 load, legacy-stream-blocked (expected),
    query-INSERT, Storage Write API, idempotency. Spike is done; writers get built on this.
    B1 route split: **Storage Write API** for high-fanout cell writes (predictions/metadata/oof);
    **query-INSERT** for the run_registry header + updates. Then delete `spikes/`.

- **Arc B B1 — registry writers built; live gate overturned the idempotency design.** Implemented
  the four writers in `registry/bq.py` (`ensure_tables`, `write_header`, `update_header`,
  `write_cells`) + `artifacts.upload_artifact`, plus a new infra seam `settings.py` (`SF_*` env
  vars → `Settings`, so the identical code runs local↔Composer, G1). Cell tables encode rows to a
  dynamic protobuf descriptor per table (`_proto_for`) and append via the Write API default stream
  using the direct `append_rows(requests=…)` iterator (NOT the `AppendRowsStream` wrapper, which
  masks the gRPC error). New permanent `@gcp` test `tests/integration/test_registry_roundtrip.py`
  replaces the spike; `conftest.py` skips `@gcp` unless `SF_PROJECT_ID` is set.
  - **FINDING (reshaped the design): the locked "run-level clear + append" idempotency is not
    viable — a DELETE that *matches* rows in the Write API streaming buffer is rejected for the
    whole buffer window (~90 min), even when the DELETE is the first write for that run_id (it hits
    a *prior* run's still-buffered rows for the same deterministic run_id). A DELETE matching
    *nothing* succeeds — so clear-then-append fails exactly when it's needed. This was foreshadowed
    by B0.3 but the plan's DELETE step contradicted it; the live gate caught it before commit.**
  - **NEW design (user-approved): append-only + dedupe-on-read.** `write_cells` never DELETEs. Because
    `run_id` is a pure function of config (`make_run_id`), a re-run writes byte-identical rows, so
    "duplicates" are exact copies; serving views dedupe with `GROUP BY`/`DISTINCT` on `run_id` +
    cell keys (predictions: ts_id,model_type,forecast_date; oof: +fold_id; metadata: ts_id,model_type).
    No write-time delete, scales to high fanout, and `write_cells` composes whether called once
    (driver collect) or per-partition (Spark/Ray). `cell_dedup_key` docstring updated accordingly.
  - **Live gate green** against `statmike-scale-forecasting`: ensure_tables idempotent, all 3 cell
    tables land via Write API, re-run keeps the dedupe-on-read count stable while raw rows grow,
    artifact upload→readable GCS object→`model_artifact` link, error cell→PARTIAL header. Spike deleted.
  - **Dep:** added `protobuf>=4.25` (explicit — we import `google.protobuf` directly) + a mypy
    override for `google.protobuf.*` (no py.typed).
  - **NOTE (pre-existing, not B1):** `ruff format --check` flags ~12–13 files I never touched (a
    newer ruff resolves via `ruff>=0.5`). Left untouched; worth a separate `ruff format` sweep +
    pinning ruff.

- **Arc B B0.4 — 100k example dataset seeded via Dataproc Serverless Spark (both write paths
  proven).** The shipped 100,000-series dataset is live in managed-Iceberg `source_series`
  (146,000,000 rows = 100k × 1,460 daily obs, 2021-01-01→2024-12-30, 5 archetypes × 20k each).
  This step also doubled as the platform's first **Spark scale smoke** — the first Spark→
  managed-Iceberg write ever (B0.3 only proved the Python-client registry routes). Built:
  real `data_gen/seed_spark.py` (pure `_to_source_rows` transform, offline-tested; lazy pyspark),
  a shared Spark runtime **container** (`docker/` + `modules/container` AR repo), a minimal
  **VPC** (`modules/network`), and the gated **`seed`** batch module.
  - **Dep delivery = custom container.** Debian-slim base, NO Spark/Java/PySpark (runtime-mounted),
    `procps`/`tini`/`libjemalloc2`, spark user UID/GID **1099**, `pip install .` (core deps only —
    pyspark is an excluded extra). Built by `docker/cloudbuild.yaml` (amd64) → AR. Rebuild the
    image whenever `seed_spark.py` / deps change — the code is baked in at `pip install`.
  - **Infra identity is delivered as CLI args, not Spark env properties.** Dataproc Serverless
    allowlists Spark property prefixes and REJECTS driver-env (`spark.driverEnv.*` isn't a real
    property; `spark.kubernetes.driverEnv.*` → 400 "unsupported properties"). Executors accept
    `spark.executorEnv.*` but only the DRIVER needs SF_* (Settings.resolve + ensure_tables + the
    write all run driver-side). FIX: the `seed` module passes `--sf-project-id/-connection/
    -warehouse-uri/-dataset-id/-region`; `seed_spark.main()` exports them to `os.environ` before
    `Settings.resolve()`, keeping env-based resolution the single G1 seam. Local runs pass no
    `--sf-*` and use the ambient env untouched.
  - **Fresh-project gaps this org exposed:** (a) Cloud Build SA (the Compute Engine default SA,
    `<num>-compute@`) had NO roles → `modules/container` grants it `cloudbuild.builds.builder` +
    `artifactregistry.writer`. (b) No default VPC → `modules/network` (VPC + subnet, Private
    Google Access + internal-ingress firewall) is required for any serverless batch. (c) The
    compute SA's `bigquery.connectionUser` (get+use) is NOT enough to CREATE managed-Iceberg
    tables through the connection — that needs `bigquery.connections.delegate`, which among
    predefined roles ships only in `connectionAdmin` (over-broad: +setIamPolicy/+delete). FIX:
    a custom role **`sfConnectionDelegate`** (get/use/delegate only), granted to both SAs in
    place of `connectionUser` — same least-privilege reasoning as the B0.3 `legacyBucketReader`
    choice.
  - **Write paths — both proven.** Smoke (100 series) used `direct` (Storage Write API, no temp
    bucket); 100k used `indirect` (Spark→Parquet→GCS→BQ load). `indirect` is the right choice for
    a full RE-seed: `direct`'s ~90-min Write-API buffer blocks the driver-side `DELETE WHERE TRUE`
    (replace-on-reseed), whereas `indirect` has no buffer, so the DELETE clears cleanly. Both write
    APPEND (managed Iceberg rejects truncate).
  - **Cost/runtime (real):** smoke ≈ **$0.02**, ~2.5 min compute; **100k ≈ $0.11–0.15, 8m34s**
    (6.34M milliDcuSeconds ≈ 1.76 DCU-hr). Far under the pre-run $5–20 estimate — Spark fan-out +
    Serverless autoscaling makes the full seed effectively free.
  - **`google_dataproc_batch` client-wait gotcha:** the provider blocks until terminal but its
    client-side wait timed out at 10 min while the 100k batch was still finishing (succeeded at
    8m34s of compute, but provisioning pushed total past the client deadline). The apply errored
    even though the batch SUCCEEDED, leaving the batch OUT of state → `terraform import` reconciled
    it. Watch for this on any long batch; the batch state in GCP is the source of truth, not the
    apply exit code.
  - **Verified:** 146M rows, 100k distinct ts_id, 5 archetypes × 20k, 1,460 distinct days, holiday
    rows scale exactly (5,200 smoke → 5,200,000 at 100k), price_index all-NULL (no `--with-exog`),
    schema matches the DDL (ts_id/ds REQUIRED, DATE/BOOL/FLOAT), per-archetype means consistent
    smoke↔100k (parity across scale = generator determinism holds under Spark fan-out).
