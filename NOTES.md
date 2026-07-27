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
