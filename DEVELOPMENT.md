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
- The Ray engine materializes one driver-side pandas panel before the fan-out shards it (both
  readers do). Bounded on purpose — Ray is the GPU/modest-scale runtime here and Spark is the
  100k-series one — but it is the ceiling that keeping the panel distributed as `ray.data` blocks
  all the way into the fan-out would lift. Gated on a live Ray run at a scale where the driver
  actually binds.
- Forward `features.exog` values are unknown offline, so the horizon's exog columns fall back to
  the most recent observed rows. Everything else in the design matrix (holidays, Fourier phase, the
  level-shift step, the first `L` steps of each `lag_L`) is computed exactly at the future dates —
  see `features.build_future_features`. Supply real forward exog by extending the source table
  past the cutoff.
- Long Ray runs (beyond ~60 min) can outlive the submission bearer token; see
  [`docs/troubleshooting.md`](./docs/troubleshooting.md).
- `neuralprophet` is incompatible with pandas 3.0 and ships as an optional extra that registers but
  skips when unavailable.

### Recently done
- **Two reserved-but-inert config fields wired, and a correctness bug they surfaced.**
  `features.level_shift` and `compute.machine_family` had both been declared, documented as
  "reserved", and consumed nowhere. `level_shift` now detects a single abrupt regime change
  (O(n) binary segmentation standardized by a MAD noise estimate, 3σ acceptance) and emits it as a
  **step** dummy — carried forward as `1` across the whole horizon, which is what distinguishes a
  level shift from an outlier. That the shipped example data plants exactly this pattern
  (`data_gen.generator`'s per-archetype `level_shift_prob`) while the flag to model it was inert
  was the tell. `machine_family` now selects the GCE family for a Dataproc **cluster's** master and
  CPU workers, restricted to families `resources` can price so the sizing plan stays honest, and
  deliberately not reaching the GPU worker (the accelerator dictates that machine) or Serverless
  (no machine concept). `spark_deps` was *also* labelled reserved in the docs and had in fact been
  consumed all along — a doc bug that under-sold a working feature.
  Building `level_shift` surfaced the real find: **the horizon design matrix was the first
  `horizon` rows of history.** Deterministic columns — holiday flags, Fourier phase — are functions
  of the *date*, so an exog-aware model was being handed the seasonal phase from the start of its
  history for the dates it was forecasting. Backtesting was never affected (its "future" is
  in-sample, so it slices the real `X`), which is exactly why it hid: the folds scored on correct
  features while the shipped forecast did not. `features.build_future_features` now builds the
  horizon frame properly, with column-order parity by construction because
  `_lag_forecaster.recursive_predict` reads exog positionally.
- Registry operations (`registry/ops.py`) — the manage-only operator surface, six verbs over the one
  registry `SF_*` resolves to: `init`, `doctor` (read-only: row counts, runs stuck `RUNNING`,
  orphaned artifacts), `drop-run`, `sweep-orphans`, `snapshot`, `export`. One implementation, three
  entry points (`python -m scale_forecasting.registry.ops <verb>`, the `Registry` SDK class,
  a notebook) — G1 applies to operations too. Deliberately **not** shipped: a wipe verb (a full
  teardown is `bq rm -r -f <dataset>` — a one-liner nobody needs wrapped, and wrapping it invites the
  accident) and anything that touches the source panel. The design work was the *ordering*: a
  registry row is the only index of which GCS objects exist, so every destructive verb enumerates
  artifact prefixes → deletes objects → deletes rows, never the reverse. Doing it backwards is how
  the old reset path accumulated an unbounded orphan pile, and `sweep-orphans` is that pile's
  cleanup — correctly scopeable only because the artifact root now carries the registry key. Two
  things the registry can't tell you were built out here: BQML `sf_model_*` objects are invisible to
  it (nothing records their names), so `drop-run` lists the dataset's models and matches names back
  to runs via `model_object_matches_run`, tested as the exact inverse of `_model_ref` so the namer
  and the matcher can't drift; and a `RUNNING` header can mean a live job *or* a dead one, so
  mutating verbs refuse an in-flight run and point at `monitor(probe=True)` rather than guessing
  (`--force` overrides). Preview-by-default with exact blast radius (runs, objects, bytes), the same
  shape the probe's cancel path uses. Pure/I-O seam throughout: planners, SQL renderers, and
  formatters are offline-tested (`test_registry_ops.py`); the six verbs are `@gcp`.
  With this landed, **`reset.py` and `registry.bq.drop_all` are gone** — the destructive tier leaves
  the product entirely. What replaces them is a `bq rm` one-liner
  ([operations.md §2c](docs/operations.md)), because the only thing `reset` did that a `bq rm`
  doesn't was give a whole-registry drop the *appearance* of being a supported, safe operation while
  silently stranding every artifact it had just orphaned. The pure renderer
  `ddl.render_drop_tables` stays — it is strings only and snapshot-tested.
- A registry now has an **address**. `project.dataset` is a guaranteed-unique registry key — BigQuery
  allows exactly one `run_registry` per dataset — so `SF_REGISTRY_DATASET_ID` (optional, defaults to
  `SF_DATASET_ID`) is all it takes to put the registry somewhere other than the source panel, and the
  GCS artifact root becomes `<warehouse>/artifacts/<project>/<registry-dataset>/<run_id>/`. That path
  is what makes cleanup well-defined: an object prefix names the dataset that owns the run, so an
  orphan sweep has an unambiguous scope and can never touch another registry sharing the bucket.
  The split runs all the way down — `ddl` exposes `REGISTRY_TABLE_NAMES` / `SOURCE_TABLE_NAMES` and
  every renderer takes a `tables=` subset, BQML `sf_model_*` objects follow the registry (they are
  run outputs keyed by `run_id`, so a per-run teardown has to find them) while source reads stay on
  `SF_DATASET_ID`, and `--sf-registry-dataset-id` travels to cluster drivers so a split deployment's
  workers write to the right place. Two defects fell out of it: the then-current `reset` was dropping
  the two **source** tables along with the registry (reseeding is a Spark job over millions of rows —
  a registry clear that silently took the input panel with it was a wipe nobody asked for; scoped
  here, and the whole verb retired one item later), and the registry/source distinction is
  now made at every call site rather than being discovered later, because a miss is invisible until
  someone actually splits the datasets. Unset variable ⇒ byte-identical behaviour to before.
- Runtime-environment standardization: asked whether the three dep-delivery mechanisms could collapse
  to one, and answered no with reasons rather than preference — the four constraints now open
  [runtime_dependencies.md](docs/runtime_dependencies.md#why-more-than-one-mechanism). The decisive
  one is that `spark.archives` localizes to *executors*, not a client-mode driver (we hit this on
  clusters and fixed it with an init action; Serverless has none), and Ray takes neither an image nor
  an archive — so the ceiling is two mechanisms, not one. What *is* single is the part that matters:
  one `uv.lock`, one build, one bump. Serverless gained the archive path anyway as a tested fallback
  for a deployment with no Artifact Registry (`SF_SERVERLESS_DEPS=packed_venv` →
  `serverless_dep_properties`, shared by the submitter and the command emitter so they can't drift);
  it lives on `BatchInfra`, *not* in the run config, because both envelopes deliver the identical
  environment and folding the choice into `run_id` would make one experiment two runs. Which envelope
  ran is recorded on the header (`container_image` xor `venv_archive`). Unproven live — it is also
  the experiment that *measures* the driver-localization gap instead of inferring it.
- Monitor ⇄ probe convergence: a registry row is written *by the job*, so a job that dies without
  writing leaves its row `RUNNING` and its bar frozen — indistinguishable from a slow one. Every
  `FamilyProgress` now carries `last_signal_at` / `quiet_seconds` (derived from rows `monitor_run`
  already reads — no runtime call, safe in a poll loop) and `plot_progress` prints the age on a
  running family. `monitor_run(probe=True)` / `Forecaster.monitor(probe=True)` escalate through the
  probe's single read+reconcile pass and attach the `ProbeReport`, so a suspicious age can be turned
  into a `LOST` / `RUNNING_CONFIRMED` verdict without a second set of registry queries; the default
  stays registry-only, because a poll loop must never fan native calls. The age is a *fact* the
  monitor reports and the escalation *threshold* stays with the probe (`probes._is_stale`, which now
  reads `quiet_seconds` rather than re-parsing rows), so the two can never disagree about how quiet
  a family has been. Notebook 08's escalate-on-quiet loop lands with its next live re-execution.
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
  by a pair of notebooks: `08_run_and_monitor.ipynb` launches a multi-engine run (Spark ∥ BigQuery)
  on a background thread and drives a live-refreshing progress dashboard until it lands (batch tier —
  it submits Dataproc), and `09_review_run.ipynb` reviews any finished `run_id` read-only —
  leaderboard, metric distribution, ensemble lift, execution timeline (smoke tier). Pure assembly +
  plots offline-tested in `test_review.py`; the `@gcp` readers ran live through notebooks 08 + 09
  in the acceptance refresh at `ff1f8bf` — see the [validation ledger](docs/validation.md), which is
  the single record of what has been proven live and on which architecture.
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
