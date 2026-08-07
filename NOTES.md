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
  - **Ensembler:** calculated strategies blend the base predictions in pandas
    (`combine_calculated`) and append via the Storage Write API — the same path the learned
    strategies use (C4 / Q4 fix: no `INSERT…SELECT` DML); learned strategies (`fit_learned`)
    train on OOF and refuse to run without backtest. Every ensemble row is keyed by
    `ensemble_id = make_ensemble_id(cfg.ensemble)` so several ensemble configs coexist under
    one `run_id`.
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

- **Arc B B0.4-hardening — runtime code delivery, one deps-only container, TF sync, VPC BYO.**
  Five architectural cleanups after operating B0.4, all verified with a live 100-series smoke
  (`sf-seed-smoke2-100-749aa78b`, SUCCEEDED, apply stayed in state):
  - **Code ships at RUNTIME, not in the image.** The container previously `pip install .`-baked the
    package, so a code edit forced an image rebuild+repush (it bit us mid-B0.4). Now Terraform
    `archive_file` zips `src/` every apply → uploads `seed/scale_forecasting-<md5>.zip` → the batch
    loads it via `pyspark_batch.python_file_uris` (on `sys.path`, so `seed_entry.py`'s
    `import scale_forecasting` resolves from the zip). The 8-char md5 is folded into `batch_id`
    (`sf-seed-<label>-<n>-<hash>`), so changed code ⇒ new immutable batch (batches never update in
    place). Needed the `hashicorp/archive ~> 2.4` provider in `versions.tf`.
  - **One deps-only container (core + [models] + [ray], NOT [spark], NOT the package).** Dockerfile
    now `COPY docker/requirements.txt` + `pip install -r`, no package. `requirements.txt` is
    GENERATED+committed via `uv export --frozen --no-emit-project --no-dev --no-hashes --extra
    models --extra ray` (satisfies CLAUDE.md §5 "locked requirements.txt for clusters"). pyspark
    excluded on purpose — installing it shadows the Serverless-mounted Spark. Image is now
    slow-moving: rebuild only on a dep change, never on a code edit.
  - **TF sync fix:** `timeouts { create = "60m" }` on `google_dataproc_batch` (provider default is
    10m; that's what left the 100k batch out of state). Recovery pattern documented in the seed
    module header: batch state in GCP is source of truth; `terraform import
    module.seed.google_dataproc_batch.seed[0] projects/<p>/locations/<r>/batches/<id>` if an apply
    ever errors post-submit. The smoke2 apply completed in state — no import needed.
  - **Network greenfield/brownfield toggle:** `modules/network` gained `create` (default **true**,
    greenfield) + `subnetwork_uri` (BYO, default null), mirroring iam/composer. `create = false`
    builds nothing and passes an existing subnet through to the `subnetwork_uri` output. Wired
    `create_network` + `subnetwork_uri` through root `main.tf`/`variables.tf` + README BYO table.
    It was the last module without the BYO toggle.
  - **Clarity:** fixed the stale `registry/bq.py` module docstring (idempotency is **append-only +
    dedupe-on-read**, never clear-then-append) and the `seed_spark.py` infra-identity docstring
    (`--sf-*` args exported to env by `main()`, not Spark env props). Confirmed the BQ Write-API
    code is centralized in `registry/bq.py` (`_append_via_write_api`/`_proto_for`/`_encode_rows`),
    reused by every registry writer.
  - **NOTE — smoke replaced the 100k dataset** (replace-on-reseed DELETE-then-append). Restored to
    the 100k deliverable with `seed_run_label=full seed_num_series=100000 seed_write_method=indirect`
    after the smoke verified the runtime-code path.

- **Arc B B2 — Spark forecast engines (explode · naive · multi) on Dataproc Serverless.** The
  compute track: a real run fans `run_cell` across a serverless cluster and lands full lineage in
  the three-tier registry. Architecture decisions + why (CONTRACTS §3.4/§3.5, DESIGN §2.1, D8–D12):
  - **Executor-side batched writes**, not driver-collect. `groupBy(bucket).applyInPandas` runs
    `run_cell` per cell and calls the *exact* B1 `write_cells` once per bucket; `bucket =
    abs(hash(ts_id)) % n_buckets` keeps a series' rows together. 100k×N cells can't be `collect()`-ed;
    append-only + dedupe-on-read (from B1) makes per-partition writes safe. Settings reach executors
    by `sc.broadcast` (frozen str dataclass); driver owns the header + aggregates a compact 4-col
    status frame to `update_header`.
  - **`multi` is submit-side, not on-cluster.** `google-cloud-dataproc` is `[spark]`-only (excluded
    from the container), so `spark_multi.run` is a guarded stub; the submit helper loops families and
    launches one child `explode` batch each. Code ships at runtime via `python_file_uris` (zip of
    `src/`) + a `spark_entry.py` launcher; RunConfig staged to GCS as JSON, passed as `--config-uri`.
  - **`naive` is a first-class engine** (`spark_naive.py`), registry-logged, `--max-executors 2`
    throttle → the straggler anti-pattern is real and queryable next to `explode`, small scales only.
  - **`job_telemetry` column + analyst views.** Submitter stamps the terminal Dataproc `Batch`
    (wall-clock, DCU, shuffle, cores, instances, maxExecutors, version, image, SA, subnet) into
    `run_registry.job_telemetry` (JSON-as-STRING), best-effort, even on FAILED. `v_run_summary`
    (scaling knobs + telemetry + derived `overhead_seconds`/`overhead_fraction`) and
    `v_model_leaderboard` (per model: `no_artifact_rate`, `median_fit_seconds`, mean wape/mae) are
    the reviewable read surface, created by `ensure_tables`→`ensure_views`.
  - **DEVIATION — live schema drift found + fixed two ways.** The `job_telemetry` view failed live:
    `ensure_tables` is `CREATE IF NOT EXISTS`, so the already-deployed `run_registry` never gained the
    column. Fix: (1) one-time `ALTER TABLE ADD COLUMN IF NOT EXISTS` on the live table; (2) permanent
    **additive self-migration** — `ddl.render_migrations` derives `ALTER ... ADD COLUMN IF NOT EXISTS`
    from the same DDL bodies the CREATE uses, run by `ensure_tables` after the CREATEs (NOT NULL cols
    excluded). New nullable columns now deploy without CREATE/migration drift (D12).
  - **DEVIATION — `libgomp1` missing from the image.** lightgbm hard-links OpenMP (xgboost bundles
    its own); without it every lightgbm cell degraded to an error (resilient PARTIAL, not a crash —
    CONTRACTS §3.3). Added `libgomp1` to the Dockerfile; after rebuild the same run went all-green
    (`no_artifact_rate=0.0` for all four models). Confirms the degrade-don't-die contract works.
  - **Verified live (project `…-scale-forecasting`, never the default project).** explode n=10/100
    SUCCEEDED→COMPLETED; naive n=100 `--max-executors 2` slower than explode at equal scale (the
    anti-pattern); multi → one child `explode` batch per family; explode **n=1000** COMPLETED,
    exactly 1000 distinct series × 4 models = 4000 full-fit cells, telemetry auto-stamped
    (`total_wall_s≈527`, `runtime_seconds≈321`, `overhead_fraction≈0.39`). The scaling thesis is now
    queryable: DCU-hours are nearly flat n=10→1000 (~10→12 DCU-hr) — the serverless provisioning
    floor dominates until work outgrows it, which is the whole point of `explode`. Next gate: 100k.
  - **FIX — 100k bucket-sizing OOM.** The first 100k attempt FAILED: executors OOM-killed (exit
    137) → shuffle `FetchFailedException` → stage aborted, at the `applyInPandas`/`toPandas` shuffle.
    Root cause: `default_bucket_count` clamped buckets to `compute.max_parallelism` (default 1000),
    so 400k cells forced ~400 series-histories into each per-task pandas frame — fine at 1k (4/frame),
    fatal at 100k. Buckets are *shuffle partitions*, not executor concurrency (that's
    `spark.dynamicAllocation.maxExecutors`), so decoupled them: new `compute.bucket_target_cells`
    (default 8) sizes buckets as `ceil(cells / target)`, bounding per-task memory at every scale.
    The 100k hero run uses a dedicated config (`bucket_target_cells=200` → 2000 buckets to keep the
    Storage-Write stream count sane; `persist_models=false` to avoid 400k artifact objects).

- **B3 BigQuery-native runtime (`engines/bigquery_engine.py`).** Two native models —
  `arima_plus`, `timesfm` — run as SQL *in* BigQuery and land in the **same**
  three registry tables as the Spark models, so they're directly comparable/ensemble-able.
  (`arima_plus_xreg` was dropped along with the shipped `price_index` exog column — the system is
  univariate by default now; the generic exog seam stays dormant, see the pre-deploy note above.)
  `run(cfg, models)` is a self-contained engine (own header lifecycle, `bq_models` array stamped);
  `router.split_by_runtime` partitions the model list by `runtime`, and a thin
  `python -m scale_forecasting.engines.bigquery_engine --config …` CLI runs a BQ-only run. Fanning
  it in parallel with a Python engine under one shared run_id is Arc B (`main.py`), not B3.
  - **Metric parity, not a shortcut.** Each native model does a held-out single fold
    (`cutoff = MAX(ds) − horizon`; train `ds <= cutoff`, score the last `horizon` window `ds > cutoff`).
    The engine reads the held-out `(y_true, yhat, lower, upper)` + history back and calls the **same**
    `metrics.compute_metrics` the Python models use — so the 11-metric panel can't drift by runtime.
    `forecast_predictions` + `backtest_oof` (`fold_id=0`) + `forecast_metadata` (`fold_id=NULL`) all
    land with `compute_engine='bigquery'`. Multi-fold BQ backtest is deferred.
  - **Holiday parity via a custom-holiday CTE.** `CREATE MODEL … AS (training_data AS (…),
    custom_holiday AS (SELECT … FROM UNNEST([STRUCT(region, holiday_name, primary_date,
    preholiday_days, postholiday_days), …])))` built from the *same* `features.holiday_frame(cfg)`
    calendar the Python suite uses — not `holiday_region`. Holiday names are sanitized to valid
    identifiers (BQML surfaces them as columns).
  - **Showpiece contrast.** ARIMA_PLUS with `time_series_id_col` trains **every series in one SQL
    statement** — the opposite of Spark's per-cell fan-out. "One query forecasts N series" next to
    "a serverless cluster fans N×M cells," both landing in one registry.
  - **Three bugs the offline snapshot couldn't catch, surfaced by the live `@gcp` smoke:**
    (1) `client.query(...).to_dataframe()` needs **`db-dtypes`** to map DATE/NUMERIC → pandas (added
    to core deps); (2) `bqml_options` must resolve for `timesfm` (returns the AI.FORECAST params) since
    it has no `CREATE MODEL` OPTIONS map — `run()` still stamps `best_params` for every model. TimesFM
    is serverless (`AI.FORECAST`, no training, no `model =>` arg). The smoke seeds its **own**
    univariate scratch table. (A third bug seen at the time — XREG `ML.FORECAST` argument order — is
    moot now that `arima_plus_xreg` is dropped.)

- **Arc B — `main.run`: Spark ∥ BigQuery in parallel, ONE run_id / ONE header.** The last
  orchestration stub. `python -m scale_forecasting.main --config <mixed.json>` runs the Python
  compute runtime (Spark on Dataproc) **and** the BigQuery-native models **at the same time, in the
  same run**, so they land side-by-side on `v_model_leaderboard` — the "wall-clock ≈ max(python, bq),
  not sum" thesis, made queryable. Architecture + why:
  - **The core asymmetry that drove the seam.** `make_run_id(cfg)` is a pure digest over the *whole*
    config incl. `cfg.models`, so two engines can only share a run_id if both see the full cfg. The
    B3 BQ engine already decoupled run_id (full cfg) from executed models (explicit `models` arg); the
    Spark engines did **not** — `cross_join_models` / the naive loop read `cfg.models` directly. Handing
    a mixed cfg to Spark unchanged would cross-join the BQ-native models into Spark cells →
    `worker.run_cell` → `NotImplementedError` (natives execute as SQL, not Python). So a **models
    subset** param was threaded through the Spark path (`spark_io` → `spark_explode`/`spark_naive` →
    `submit` → `spark_entry`), mirroring the BQ engine: run_id from the full cfg, executed set = the
    explicit subset.
  - **`main` is the sole header owner; engines run in contributor mode.** An opt-in
    `manage_header=False` on every engine skips `ensure_tables`/`write_header`/`update_header` and
    writes only cells. `main.run` writes the single header RUNNING up front, runs both engines as
    contributors (each with only its subset), joins, and finalizes the one header with a combined
    status. Defaults (`models=None`, `manage_header=True`) preserve standalone behavior byte-for-byte —
    every existing engine CLI + `@gcp` smoke is unchanged.
  - **Parallelism via one worker thread, no header race.** `main` launches the remote Spark batch in a
    `ThreadPoolExecutor` future (`submit_batch(wait=True)` — its telemetry stamp runs in-thread) while
    the in-process BigQuery engine runs on the main thread; the BQ work (minutes, in-process) overlaps
    the Spark provisioning floor. Combined status: COMPLETED iff both engines green, else FAILED —
    finalized **before** the error re-raises, so the run stays queryable and the CLI exits non-zero.
    The only in-window header UPDATE is `submit`'s best-effort telemetry stamp, which completes inside
    the joined future before `main`'s finalize — so nothing else ever touches the header.
  - **Metric parity needs backtest ON for the Spark path.** The BQ natives always score a held-out
    fold; the Python/Spark path only emits a metric panel from OOF when `backtest.enabled=true`. So the
    mixed demo config enables a small backtest (`n_folds=2`) — otherwise the Spark model would land
    forecasts but NULL `mean_wape` and the two runtimes wouldn't be comparable, which is the whole
    point of the single-run leaderboard.
  - **Coarsening (documented).** A remote contributor batch can't return its run-level PARTIAL (some
    cells errored) to the orchestrator, so a SUCCEEDED batch reports COMPLETED; per-model failure stays
    visible on `v_model_leaderboard` (a failed model → NULL metric AVGs).
  - **Rejected shapes, with a pointer.** `python_runtime="ray"` → clean `ConfigError` (unbuilt stub);
    `spark_method="multi"` → `ConfigError` pointing at `python -m scale_forecasting.submit --engine
    multi` (multi is inherently multi-run — each family child gets its own run_id, can't share one
    header). Both guarded only when there *are* Python models, so an all-BQ config plans regardless.
    `--dry-run` resolves run_id + `estimate_fanout` offline, touching no GCP.
  - **Verified live (project `…-scale-forecasting`, never the default project).** A mixed config (one
    Spark model + the three natives, `series_limit=10`, backtest on) through `main.run` launched a real
    Dataproc Serverless batch **and** the in-BigQuery engine under one run_id: exactly **one**
    `run_registry` header, COMPLETED, `n_models=4`, `bq_models` = the three natives; all four models on
    `v_model_leaderboard` under the same run_id, cleanly split `compute_engine='spark'` (the Spark
    model) vs `'bigquery'` (the natives), each with a non-NULL metric panel. The single-run,
    two-engine, directly-comparable leaderboard is the demo spine.

- **B4 — Ray on Vertex AI: a deterministic fixed-size T4 cluster, fractional-GPU NeuralProphet.**
  The third Python runtime, and the design's home for GPU-accelerated models (Spark can't share a
  GPU fractionally). `python_runtime="ray"` now runs the Python-runtime models on a **fixed-size**
  Vertex Ray cluster in parallel with the BigQuery-native engine under one run_id — exactly the Arc B
  contract, a different compute backend. Architecture + decisions:
  - **Deterministic sizing, not autoscaling (supersedes the original design).** The framing that
    drove B4: *don't autoscale — size the cluster to the run's fan-out, and show resizing it for
    larger/smaller scales.* A fixed-size pool is the honest model for a fixed-scale batch job, so a
    pure `ray_io.plan_cluster(cfg) -> RayClusterPlan` sizes two fixed pools from `estimate_fanout`
    (GPU-worker pool for the deep-learning models, CPU-worker pool for stats/ML), each a fixed
    `node_count` with **no** `autoscaling_spec`. "Resize for scale" is just a different `series_limit`
    → a different fixed plan (n=6 → 1+1 nodes; n=600 → clamped to `ray_max_nodes`); the whole sizing
    decision is logged + stamped to the run. Proven offline by the `plan_cluster` sizing tests.
  - **Heterogeneous fractional-GPU routing — the reason Ray exists here.** `ray_engine.run` (the
    on-cluster driver) splits the executed models into GPU models (NeuralProphet, `family ==
    "deep_learning"`) and CPU models, dispatching each chunk of cells as a `@ray.remote(num_gpus=frac)`
    or `@ray.remote(num_cpus=1)` task — both calling the **exact** B2 `run_group` + `bq.write_cells`
    executor-side (a "chunk" is Ray's "bucket"; `spark_io`/`worker`/`registry` untouched). NP cells
    pack onto a T4 at a calibrated fraction while stats cells run on CPU, all landing
    `compute_engine="ray"` (falls out free — `run_cell` already stamps `cfg.python_runtime`).
  - **Auto-fraction GPU calibration.** `gpu_fraction: "auto"` (default) → sample a few series, fit NP
    measuring peak GPU memory, solve `fraction ≈ (peak × safety_margin) / device_mem`, clamp; a fixed
    float short-circuits it. Unit-tested with injected memory numbers (no GPU needed for the offline
    gate); the chosen fraction is stamped to the run for reproducibility.
  - **Ephemeral-default / reuse-opt-in lifecycle, teardown-in-`finally`.** `ray_submit.submit_ray`
    (sibling of `submit.py`) owns the cluster: **ephemeral** (default) creates the planned cluster →
    submits the driver as a Ray Job → polls to terminal + stamps telemetry → `delete_ray_cluster`
    in a `finally` (teardown survives a failing job — no orphaned T4s billing forever); **reuse**
    (`compute.ray_cluster_name` or `--cluster-name`) targets a standing cluster and skips both
    create and delete. Both paths unit-tested with `vertex_ray` monkeypatched.
  - **No on-cluster shim (unlike Spark).** A Vertex Ray Job's entrypoint is a *shell command*
    (`python -m scale_forecasting.ray_entry …`) that runs **with** package context, and the current
    `src/` ships via the job's `runtime_env` working dir — so, unlike Dataproc's bare
    `main_python_file_uri` (which needs the `spark_main` shim), `ray_entry` *is* the entrypoint. Same
    G1 "same code local↔cluster" seam, delivered at runtime, never baked into the image.
  - **Telemetry parity, no schema change.** A Ray analog of Spark's `job_telemetry` (cluster name,
    node counts, machine/accelerator types, `sizing_gpu_fraction`, ray/python versions, wall-clock,
    job id) is stamped into the existing `run_registry.job_telemetry` STRING column via
    `update_header` — `v_run_summary`/`v_model_leaderboard` need nothing new.
  - **`main.run` dispatch.** `_plan` now accepts `python_runtime="ray"`; a one-point
    `_launch_python_runtime` picks Ray vs Spark on the worker thread (both contributor-mode, one
    shared header), so Ray ∥ BigQuery works identically to Spark ∥ BigQuery. The `multi` guard stays
    but is unreachable for ray — the config layer forbids `ray` + any `spark_method` outright.
  - **Dep note.** `vertex_ray` imports `immutabledict` at module load, but `google-cloud-aiplatform`
    only declares it under its own `[ray]` extra, which would pin `ray[default]<=2.47.1` and conflict
    with ours — so the `[ray]` extra pulls `immutabledict` directly.
  - **Test-harness note.** The `@ray` engine tests spin up a real local Ray session (`local_mode` is
    gone in Ray 2.x); under `uv run`, Ray's worker subprocesses can't re-resolve the venv and fail to
    `import ray` (60s registration timeouts). Run them via `.venv/bin/python -m pytest` directly —
    then they pass in seconds. The offline gate otherwise stays green (ruff, mypy, unit suite).
  - **Live @gpu smoke authored, run pending T4 quota.** `configs/ray_gpu_demo.json` +
    `tests/integration/test_ray_gpu_smoke.py` (`@gpu`, collect-but-skip unless `SF_ENABLE_GPU`) run a
    mixed config (NeuralProphet + a stat model + the three natives, `series_limit=6`, backtest on)
    through `main.run` against a fixed-size T4 cluster, asserting: one COMPLETED header,
    `python_runtime='ray'`; NP + the stat model `compute_engine='ray'` with non-NULL metrics beside
    the natives on one leaderboard; the calibrated fraction + fixed plan in `job_telemetry`; and the
    ephemeral cluster **gone afterward** (`list_ray_clusters`). Needs `NVIDIA_T4_GPUS` quota in-region
    (a console/gcloud request, not Terraform) before the one authorized live run.
  - **Pre-flight fixes (found preparing the live T4 run — all offline-verified before any spend):**
    - **pandas capped `<3`.** NeuralProphet 0.9.0 (its latest release) calls the `Series.view` API
      that pandas 3.0 removed, so it can't `fit` under 3.x — proven by a local NP fit, not guessed.
      `pyproject.toml` now pins `pandas>=2.2,<3` (+ `pandas-stubs<3`); re-locked to pandas 2.3.3 and
      regenerated `docker/requirements.txt`. Our own code has no pandas-3-only idioms, so the cap is
      clean product-wide. The full offline suite now runs **with** NeuralProphet (no `-k` skip) —
      **460 passed**. Two knock-ons: NP `fit` now calls `set_random_seed(ctx.seed)` (torch was
      unseeded → the determinism contract test failed once NP actually ran); a `.where(cond, None)`
      call in `data_gen/seed_spark.py` gained one `type: ignore[call-overload]` (pandas-stubs<3 lacks
      the overload, valid at runtime). `lightning_logs/` (a Lightning artifact NP writes to CWD on
      fit) is now git-ignored.
    - **Ray version skew.** `runtime_env["pip"]` no longer points at the raw `requirements.txt` (which
      pins `ray==2.56.1`); it ships a parsed package **list minus Ray** (`build_runtime_env` /
      `_requirements_packages`). Vertex Ray's cluster image provides Ray (latest supported = 2.47),
      and pip-installing a newer Ray over the running head/workers breaks the job. Everything else
      (torch, neuralprophet, …) still installs on the prebuilt image at job start.
    - **Public endpoint (no VPC peering).** `RayInfra.network` is now optional: unset → the cluster
      gets a **public endpoint** (`create_ray_cluster(network=None)`, Vertex's own default), so a
      deployment with no private-services-access connection can still run. A VPC (with PSA in place)
      still yields a private endpoint. This is a real capability add, not a workaround — the offline
      deployment here has no PSA peering, and requiring it would have blocked the runtime entirely.

- **B5: ensembling wired end-to-end.** `ensembler.py` stays pure (renders calculated INSERT SQL +
  `fit_learned` returns weights/artifacts); a new engine-agnostic `ensemble_run.run_ensembles(cfg,
  run_id, *, settings)` executes it. It runs inside `main.run` **after** the engine join and
  **before** the header finalize, gated on `cfg.ensemble.enabled`, and its failure is captured like
  an engine error (flips the shared header FAILED, re-raises) — ensembles are part of the run's
  success contract. Three steps: (1) run each calculated statement with `@run_id` bound
  (`ensemble_{mean,median,inverse_error}` → `forecast_predictions`, `compute_engine='ensemble'`);
  (2) learned apply in pandas — read base preds + `backtest_oof`, `fit_learned`, blend
  `yhat = Σ w·yhat` (weights renormalized over whichever base models are present per
  (ts_id, forecast_date)), append rows + upload artifacts; (3) **score into `forecast_metadata`**
  (`fold_id=NULL`) by joining ensemble preds to `source_series` actuals and running
  `compute_metrics`. That `fold_id IS NULL` write is the missing leaderboard link —
  `v_model_leaderboard` then shows the ensembles automatically, **no view change**.
  - **Disjoint forecast windows (the subtlety).** Spark/Python models forecast the *true future*
    (`ds > MAX(ds)`, no actuals); native BQ models forecast the *held-out fold* (`ds > cutoff`, has
    actuals). Base preds from the two engines land on disjoint dates, so scoring joins ensemble
    preds to actuals — only held-out dates with ground truth contribute, and future-dated blended
    rows drop out of the join naturally. The combo demo config uses **calculated** strategies only
    for this reason; learned strategies are exercised by the offline unit tests.

- **Spark Connect capability (injectable session).** The engines can now be driven from a notebook
  over a **Dataproc Spark Connect** endpoint, not just as a remote Dataproc batch — the *same*
  engine code both ways (G1). Two changes: (1) `spark_explode.run` / `spark_naive.run` take an
  optional `spark=` session — `owns_session = spark is None`, so a self-created session is
  `stop()`-ed in `finally` (the batch path, unchanged) while an injected caller-owned session is
  used but **not** stopped; `main.run` gains a matching `spark=` that dispatches in-process to the
  engine instead of `submit_batch` when set (and `python_runtime="spark"`). (2) `make_group_runner`
  captures the frozen `Settings` **directly** in the closure (`applyInPandas` cloudpickles it) —
  dropping `spark.sparkContext.broadcast(settings)`, which a Spark Connect session doesn't expose
  (no RDD/`sparkContext` API). `applyInPandas` and the DataFrame API (incl. the `F.broadcast` hint
  in the model cross-join) *are* Connect-supported, so nothing else moved.
  - **Client-only dep.** `dataproc-spark-connect` is in the `[spark]` extra but **never imported by
    engine modules** (they run on-cluster against a classic session) — only by the notebook/client,
    preserving the G1 seam. Pin the Connect session to **runtime 3.0** (Connect requires ≥ 3.0); the
    batch default is left untouched.
  - **Reachability + Python parity + fallback.** Spark Connect needs outbound reach to the endpoint
    *and* the driver kernel's Python **minor** must match the workers'. **Dataproc 3.0 workers run
    Python 3.12**, and Connect refuses a driver↔worker minor skew, so the `applyInPandas` fan-out
    fails with `PYTHON_VERSION_MISMATCH` from a 3.11 driver. The fixture/notebook do a scratch
    `spark.range(5).count()` (JVM-only reachability) **and** a one-row `applyInPandas` (worker-side
    Python parity) before real work, and fall back to `main.run(cfg)` (remote batch) on any failure —
    identical engine, so the fallback is a proven path, not a second code path to trust. The
    `@spark`+`@gcp` Connect smoke **skips** (not fails) when the endpoint is unreachable or the
    driver Python doesn't match; the remote-batch path exercises the same engine on 3.12 workers with
    no local driver.

- **Demo notebooks.** Four notebooks in `notebooks/` run + review against a live deployment
  (`Settings.resolve()` from `SF_*`, poll `v_model_leaderboard` / `v_run_summary` since Write-API
  rows are async-visible): `01_spark_via_connect` (Spark UDF fan-out over Connect + remote-batch
  fallback), `02_bigquery_native` (native models, no Spark thread), `03_combo_and_ensemble` (Spark ∥
  BigQuery under one `run_id` with B5 firing after the join — one leaderboard shows base Spark + base
  BigQuery + `ensemble_*` side by side, plus a mean-WAPE bar colored by compute engine),
  `04_ray_on_vertex` (Python-runtime models on a fixed-size Ray-on-Vertex cluster ∥ natives in
  BigQuery — CPU-only `ray_cpu_demo.json` by default, one config flag from the `ray_gpu_demo.json`
  fractional-T4 path). Notebooks ship with **empty outputs** (no executed identifiers to leak).

- **B5 + Connect: live verification (what actually ran on real GCP).**
  - **`ensemble_*` type bug, found + fixed live.** The calculated ensemble SQL emitted `NULL AS
    quantiles`, which BigQuery types as INT64 — but `forecast_predictions.quantiles` is STRING, so
    the `INSERT … SELECT` failed with a 400 (`type INT64 … cannot be inserted into column quantiles,
    which has type STRING`). Fixed to `CAST(NULL AS STRING) AS quantiles` in both the mean/median and
    inverse-error builders (snapshot updated). After the fix the `@gcp` ensemble smoke passes: base
    Spark + native + `ensemble_{mean,median,inverse_error}` all land on `v_model_leaderboard` with
    non-NULL `mean_wape` under one `run_id`.
  - **Orchestration smoke still green** after the injectable-session refactor — Spark ∥ BQ under one
    run_id, COMPLETED header, both engines on the leaderboard (the default no-`spark=` path still
    submits a remote Dataproc batch, unchanged).
  - **Connect: endpoint reachable, driver-Python skew is the real limiter.** From this GCP
    workstation the Connect endpoint provisions fine at runtime 3.0 (no egress block) — the smoke
    reached the cluster and launched the `applyInPandas` fan-out, which then failed on
    `PYTHON_VERSION_MISMATCH` (driver 3.11 vs Dataproc-3.0 workers 3.12). The fixture now probes
    Python parity and skips cleanly with that message; a true live Connect pass needs a 3.12 driver
    kernel. The remote-batch path already covers the identical engine on 3.12 workers.

- **Ray on Vertex: the blocker is the job-submission hop, not the model path.** A **CPU-only** live
  smoke (`test_ray_cpu_smoke`, gate `SF_ENABLE_RAY` + the `raylive` marker; config
  `configs/ray_cpu_demo.json` — `use_gpu:false`, no `neuralprophet`, so `plan_cluster` sizes the GPU
  pool to zero) deliberately removes GPU quota as a variable to isolate the lifecycle. Live result:
  the fixed cluster **creates** (PROVISIONING → RUNNING), the BigQuery natives **run in parallel**
  and score, and the cluster **tears down** cleanly in the `finally` (no orphaned nodes) — but
  submitting the Ray Job fails at `JobSubmissionClient("vertex_ray://…")` construction with a
  repeated **HTTP 524** on the dashboard's `/api/version` handshake (an upstream proxy→origin
  timeout), which exhausts the connect-retry budget. The submitter classifies 524 as transient and
  retries correctly — so the fault is isolated to Google's managed dashboard-proxy → head-node
  dashboard-port hop (`*.aiplatform-training.googleusercontent.com`), a *different* host than the
  Dataproc Connect endpoint, not a code fault. The `plan_cluster` sizing itself is proven offline+free
  by the unit tests, and Cloud Logging shows the head node is **healthy** (`Ray runtime started`, jobs
  run) — only the proxy hop to its dashboard port fails.

  The 524 proved **independent of client location, endpoint type, and org policy**: it reproduced
  identically from a local workstation, an on-VPC VM, and a Colab Enterprise runtime; on both public
  (`network=None`) and VPC-peering (`network=<vpc>`) endpoints; and with `compute.vmExternalIpAccess`
  set to both DENY and ALLOW. Every one of those still 524'd. The axis that finally mattered was the
  **cluster's connectivity mode**.

- **Ray on Vertex: SOLVED — two co-requisites, a PSC-I attachment AND a ≥16-vCPU head node.** The
  managed dashboard-proxy → head-node hop needs **both**: (1) the cluster on a **PSC-I (Private
  Service Connect Interface) network attachment** (`psc_interface_config`) — not public, not VPC
  peering; and (2) a head node of **n1-standard-16 (60GB) or larger**. Either alone still 524s.
  The head-size requirement was the confounder that made this take weeks: every failing cluster used
  a small head (n1-standard-8, 30GB — which *boots*, reaches RUNNING, and runs Ray, but whose managed
  dashboard proxy never comes up → `/api/version` 524s at a 30s timeout with 0 bytes), and the one
  cluster that ever worked happened to be the only one with a 16-vCPU head — so "PSC-I is the fix"
  looked complete until we reproduced it through our own SDK path. **Controlled proof** (identical
  PSC-I attachment `scale-forecasting-ray`, Ray 2.47.1/py 3.11, same `create_ray_cluster` code, only
  the head machine varied): n1-standard-8 → `/api/version` 524 (30s, 0 bytes); **n1-standard-16 →
  HTTP 200 in 6.7s**. The live resource confirmed PSC-I was applied identically in both cases
  (`psc_interface_config.network_attachment` set, `network` empty), so the delta was purely head
  size. The fix is **cluster-side, not client-location**: the dashboard address stays the public
  `*.aiplatform-training.googleusercontent.com` proxy host, but a PSC-I + big-head cluster serves it
  to any authed client (local, on-VPC, Colab, headless Composer) — no in-network submitter needed.
  In the product: `config.ray_head_machine_type` defaults to **n1-standard-16** (do not lower it),
  and `ray_submit` selects connectivity in precedence order PSC-I (`network_attachment` /
  `SF_RAY_NETWORK_ATTACHMENT` / TF `network_attachment_id`) → VPC peering (`network`) → public; the
  attachment is provisioned in Terraform (`modules/network`, `ACCEPT_AUTOMATIC`) with the Vertex AI
  service agent granted consume-only access (`modules/iam`). Ray-on-Vertex is now a supported compute
  track alongside Dataproc Serverless and BigQuery-native.

- **Storage split by role — native registry + dual-format source (reverses the all-Iceberg
  decision).** With Ray unblocked we removed the two frictions Iceberg forced on the run-collection
  tables. The four registry tables (`run_registry`, `forecast_metadata`, `forecast_predictions`,
  `backtest_oof`) are now **always native BigQuery**: (1) native `JSON` columns — `raw_config`,
  `job_telemetry`, `quantiles`, `best_params` are `JSON`, not `STRING`+`JSON_VALUE`/`PARSE_JSON`
  (reversing the earlier "Iceberg rejects JSON → store as STRING" accommodation); (2) `WRITE_TRUNCATE`
  reseed instead of driver-side `DELETE WHERE TRUE`; (3) no BigLake connection needed for the
  registry. **JSON serialization gotcha:** header params (`raw_config`/`job_telemetry`) are now passed
  to `ScalarQueryParameter(type="JSON")` as Python **dicts**, not `json.dumps` strings — the client's
  `_json_to_json` converter runs `json.dumps` itself, so a pre-serialized string double-encodes. Cell
  fields (`quantiles`/`best_params`) stay string-serialized in the Storage Write API proto (the Write
  API models a JSON column as a string field parsed on ingest — no proto change). `JSON_VALUE` works
  on native JSON, so `views.py` SQL is unchanged.
  - **Source table now ships in both formats.** The example input is created as
    `source_series_iceberg` (managed Iceberg) **and** `source_series_native` (native) from **one**
    seed panel, so a deployment can benchmark identical series on either storage. The canonical
    `source_series` name is retired; a run picks a variant via `cfg.data.source_table`. All three
    engines read through BigQuery's table interface, so storage format is transparent to engine code —
    zero engine logic changed. `seed_spark --variant {iceberg,native,both}` (default `both`) seeds
    them; the native reseed uses `TRUNCATE TABLE`, the Iceberg one still `DELETE WHERE TRUE`.
  - **Honesty note (unchanged):** append-only / dedupe-on-read / no-DELETE in `write_cells` is a
    Storage Write API streaming-buffer (~90 min) constraint, **not** Iceberg-specific — going native
    does *not* remove the immediate-rerun double-count, so timestamped `run_name`s stay necessary.
  - **Fresh start + reset tooling.** The Iceberg→native registry switch is drop-and-recreate, not
    `ALTER`, so we added `bq.drop_all` + a `python -m scale_forecasting.reset [--yes]` entrypoint
    (dry-run without `--yes`; drops all six tables + the two views). The Iceberg source variant still
    needs the connection + warehouse bucket, so those stay in Terraform + required on `Settings`.
