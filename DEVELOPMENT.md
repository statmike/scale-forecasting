# Development notes

> **Temporary, living document — delete before production.**
> This file is the single home for developer-facing notes: the rationale behind
> non-obvious design choices, and the running list of outstanding / in-progress / done work.
> It is intentionally the *only* place in the repo that carries this kind of note, so it can be
> removed wholesale once the build stabilizes. Nothing here is required to run or understand the
> product — the user-facing docs under [`docs/`](./docs/) and the auto-generated
> [API reference](./docs/api/index.md) are the source of truth for behavior.

---

## Decision log

Plain-language rationale for the choices that aren't obvious from the code alone.

- **Everything is pinned to Python 3.11.** Vertex AI's managed Ray runtime tops out at Python 3.11,
  and the system runs the *same* code across Spark, Ray, and local execution. 3.11 is therefore the
  only interpreter version that satisfies all three surfaces at once, so the whole project pins to
  it. See [`docs/version_matrix.md`](./docs/version_matrix.md).

- **The Ray-on-Vertex cluster autoscales by default.** The cluster is sized to each run's fan-out
  rather than provisioned as a fixed pool: worker pools carry an autoscaling spec and grow/shrink
  with demand. A fixed-size mode remains available via config for deterministic capacity tests.

- **Native BigQuery registry + dual-format source data.** The run registry is plain BigQuery tables,
  and the example input data ships in two storage formats — managed Apache Iceberg on GCS and native
  BigQuery. This keeps the system runnable without an Iceberg catalog (e.g. against a BigQuery
  emulator) and lets a deployment choose its data layer.

- **The config *is* the experiment record.** The validated run config is stored verbatim on the
  registry (`run_registry.raw_config`) and the `run_id` is derived from it, so a run is fully
  reproducible from its config alone. Behavior changes come from editing JSON config, not code.

- **Artifact lineage via GCS object references.** When a run opts in, each fitted model is persisted
  to GCS and its object reference is stamped onto the `forecast_metadata` row, giving per-series
  model lineage without bloating the tables.

- **In-node hyperparameter optimization.** HPO runs as an Optuna study over the aligned backtest
  inside each worker cell; the engines only add a small driver-side sample-and-resolve step in front
  of their existing fan-out. Works fleet-wide or per-series.

- **Same code locally and in the cloud; code ships at submit time.** There is one code path for
  local, Spark, and Ray execution — no per-environment forks. A code edit is delivered with the next
  run (no container image rebuild). See
  [`docs/editing_code_without_rebuilding.md`](./docs/editing_code_without_rebuilding.md).

---

## Work items

### In progress
- [ ] SDK runner refinement — tighten both the high-level `Forecaster` path and the lower-level
      direct job runners (the effort this cleanup unblocks).

### Outstanding / deferred
- [ ] Trim the Vertex agent's subnet-scoped custom role down to its true floor — deferred until a
      greenfield Ray run confirms the minimum permission set.
- [ ] Lightning Engine A/B on large Spark runs — expected to show little gain because the per-series
      model fit (a Python UDF) is the bottleneck, not Spark I/O; worth a controlled measurement.

### Known limits
- Live scale has been proven to the 1k–10k series range; larger runs are designed for but not yet
  routinely exercised.
- Long Ray runs (beyond ~60 min) can outlive the submission bearer token; see
  [`docs/troubleshooting.md`](./docs/troubleshooting.md).
- `neuralprophet` is incompatible with pandas 3.0 and ships as an optional extra that registers but
  skips when unavailable.

### Recently done
- Family→runtime DAG: one run plans one job per model family (statistical / ml / deep_learning /
  native), each on its own resolved runtime, all in parallel under a shared `run_id` plus a
  downstream ensemble node. Traceable via the `v_run_jobs` view and the SDK's `Forecaster.dag()` /
  `Forecaster.jobs()`. The retired `spark_method` config knob and the `multi`/`naive` Spark methods
  are gone — `explode` is the sole Spark engine.
- Documentation & repo refactor: MkDocs Material site + auto-generated API reference published to
  GitHub Pages, slim README, single-sourced guides, all internal tokens/dev-notes corralled here.
- Ray-on-Vertex autoscaling.
- Python SDK (`Forecaster`).
- Config-level rerun guard (same config → same `run_id`, the dedupe key for idempotent re-runs).
- Cross-run ensembling (best model per engine across a group of runs).
