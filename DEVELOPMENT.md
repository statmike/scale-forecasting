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
- Run-inspection layer (`review.py`): keyed on a bare `run_id` (reads the run's own `raw_config`
  back to recover its plan), with the same pure/I-O seam as `sdk`. `monitor_run` → a `RunProgress`
  (per-family job state on its runner, `n_done / n_expected` cells, mean fit time, run-wide fraction)
  for a run in flight; `review_run` → a `RunReview` (every model best-first in the run's decision
  metric, best per family/overall, the full metric panel aggregated server-side across all series —
  mean + p10/p50/p90 — and each ensemble's lift over the best base model). Plots (`plot_progress`,
  `plot_leaderboard`, `plot_metric_distribution`) with a palette validated by the dataviz checker;
  the execution timeline reuses `sdk.build_trace_frame` + `plot_trace`. Exposed lazily off the
  package and via `Forecaster.monitor()` / `Forecaster.review_run()`; new registry readers
  (`read_run_config`, `read_progress`, `read_metric_aggregates`, `read_cell_metrics`). Demonstrated
  by `notebooks/08_job_review.ipynb` (registry-read-only, smoke tier). Pure assembly + plots
  offline-tested in `test_review.py`; the `@gcp` readers await a live acceptance run.
- Airflow/Composer DAG emitter: `airflow_emit.emit_airflow_dag` renders a run's execution DAG as a
  flat, hand-written-quality `dag_<run_id>.py` (one `PythonOperator` per family node calling the
  `airflow_tasks` callables, explicit `>>` edges, a shared-cluster create/delete bracket when several
  ephemeral Ray/Dataproc-cluster families co-locate, and the ensemble node wired `barrier` or
  `microbatch`). It resolves the same DAG as `main.run` and calls the identical run building blocks,
  so a config produces the same run on Composer as locally under one `run_id`. Exposed via
  `--emit-airflow` on the CLI and `Forecaster.emit_airflow()`; `staging.stage_dag` uploads the
  rendered file next to the staged config. The renderer is pure/offline (verified by
  compiling the emitted source, no Airflow install needed). The docstring records the two native
  operator alternatives (deferrable Dataproc operators + per-family finalize; native operators +
  single reconcile-at-finalize) and when to prefer them over the uniform-PythonOperator model.
- Airflow emitter — two-level testing beyond the offline `compile()`/`ast` checks:
  - Parse-under-Airflow (`tests/unit/test_airflow_dagbag.py`, `@airflow`): loads an emitted DAG
    through a real `airflow.models.DagBag` and asserts no import errors, so operator-kwarg / import-
    chain mistakes the string checks can't see are caught. Airflow conflicts with our torch/ray/spark
    pins, so it's deliberately **not** in `uv.lock`; the test skips cleanly when Airflow is absent
    (like `@spark`/`@ray`), and a dedicated CI job (`airflow-parse`) installs it isolated against its
    official constraints and runs it every push — a resolution problem there can't break the main
    offline gate.
  - Live Composer smoke (`configs/smokes/15_airflow_multi_engine.json` +
    `tests/smokes/airflow_smoke.py`, `@gcp`): drives the most-complex config (three engines —
    Spark + Ray GPU + BigQuery — under a microbatch ensemble) through Composer end to end (stage →
    emit → import → trigger → wait → verify), reusing the direct smoke's verifiers. The
    config-derived `run_id` proves same-code local↔Composer. Gated on `create_composer=true`; runbook
    in `docs/smoke_testing.md`. The Ray-token-expiry known limit covers the long-GPU-run caveat.
- Composer enablement (a worker = a launch point, no product-code changes): `build_package_zip`
  resolves its zip root from `__file__`, so wherever `src/` is synced becomes the code shipped to the
  jobs — the emitted DAG already calls the identical driver path, so nothing in `src/` changed. The
  wiring is environment-only: the composer module now takes `env_variables` (the `SF_*` identity,
  built in `terraform/main` from module outputs) + `pypi_packages` (the **submit-side** subset only —
  Dataproc/Vertex/BQ clients + the version-matched Ray client + `holidays` for the native track's
  worker-side holiday-feature build, *not* torch/darts/pyspark), and code is
  delivered by `make composer-sync` (rsyncs the working tree's `src/` into the env's plugins prefix,
  on `PYTHONPATH`). Image stays deps-only (the `test_code_delivery` invariant holds); GitHub is only
  the origin, nothing pulls from it at runtime. **Live validation done** (Composer 3 / Airflow
  2.10.5): a multi-runtime run (Spark statistical + ml, BigQuery native + ensemble) reached
  `COMPLETED` under the config-derived `run_id`, byte-identical to the local `plan_dag` id. Three
  launch-point defects surfaced and were fixed here: (1) the worker OOM-restarted on the family
  fan-out at 1cpu/2gb → raised to 2cpu/6gb; (2) model files eager-imported the model stack
  (`statsmodels`, `scipy`, …) at module top, crashing family tasks on the lean worker → moved every
  heavy import into `fit`/`predict`, guarded by `tests/unit/test_launch_point_lean.py`; (3) the
  native track builds holiday exog columns in Python on the worker but `holidays` wasn't in
  `pypi_packages` → added. Trigger the DAG via the Airflow REST API (`executeAirflowCommand` 500s/502s
  on a minimal env). Re-running the same `run_id` collides on the deterministic Dataproc batch id —
  delete the prior batches or use a fresh `run_id`.
- Family→runtime DAG: one run plans one job per model family (statistical / ml / deep_learning /
  native), each on its own resolved runtime, all in parallel under a shared `run_id` plus a
  downstream ensemble node. Traceable via the `v_run_jobs` view and the SDK's `Forecaster.dag()` /
  `Forecaster.jobs()`. The retired `spark_method` config knob and the `multi`/`naive` Spark methods
  are gone — the cross-join/explode strategy is the
  sole, built-in Spark engine, so the `--engine` dispatch flag was removed too (both Spark and Ray
  run their one engine directly).
- Documentation & repo refactor: MkDocs Material site + auto-generated API reference published to
  GitHub Pages, slim README, single-sourced guides, all internal tokens/dev-notes corralled here.
- Ray-on-Vertex autoscaling.
- Python SDK (`Forecaster`).
- Config-level rerun guard (same config → same `run_id`, the dedupe key for idempotent re-runs).
- Cross-run ensembling (best model per engine across a group of runs).
