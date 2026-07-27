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
