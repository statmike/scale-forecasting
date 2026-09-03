# Running a forecast and reviewing it

This is the end-to-end path: submit a run, watch it land, review which model won, and (optionally)
re-ensemble it — all from the config you wrote (see the
[configuration reference](./configuration_reference.md)). No image rebuild is ever part of this loop;
your `src/` ships at submit time (see [editing code without rebuilding](./editing_code_without_rebuilding.md)).

## Prerequisites

- A deployed environment (see [deploying on GCP](./deploying_on_gcp.md)) — or point at any project
  where the registry tables exist.
- The `SF_*` identity in your environment (the same identity every writer uses):

  | Variable | Required | Default | Meaning |
  |----------|----------|---------|---------|
  | `SF_PROJECT_ID` | yes | — | GCP project. |
  | `SF_CONNECTION` | yes | — | BigLake connection ref, `project.region.name`. |
  | `SF_WAREHOUSE_URI` | yes | — | GCS warehouse root, `gs://<bucket>/warehouse`. |
  | `SF_DATASET_ID` | no | `scale_forecasting` | Dataset holding the **source** panel (and, by default, the registry). |
  | `SF_REGISTRY_DATASET_ID` | no | `SF_DATASET_ID` | Dataset holding the **registry** — set it only to split the two. |
  | `SF_REGION` | no | `us-central1` | Region. |

  `SF_REGISTRY_DATASET_ID` exists because the registry and the source panel have different
  lifetimes. The registry is churn — you clear it, you keep several of them side by side, you tear
  one down after an experiment. The source panel is a Spark seed job over millions of rows that you
  rebuild rarely and never by accident. Leaving the variable unset keeps both in one dataset, which
  is what every deployment does until it wants otherwise.

  Setting it also fixes the **artifact root**: model objects live under
  `<warehouse>/artifacts/<project>/<registry-dataset>/<run_id>/`, so two registries can share a
  warehouse bucket and still be cleaned up independently. `project.dataset` is a
  guaranteed-unique registry key — BigQuery allows exactly one `run_registry` per dataset — so
  the path is self-describing: read an object's prefix and you know which registry owns the run.

  Every value comes straight from `terraform output` in `terraform/main`. After a fresh apply, wire
  them into your environment directly — no manual copy-paste:

  ```bash
  cd terraform/main
  export SF_PROJECT_ID="$(terraform output -raw project_id)"
  export SF_CONNECTION="$(terraform output -raw iceberg_connection)"
  export SF_WAREHOUSE_URI="$(terraform output -raw warehouse_uri)"
  export SF_DATASET_ID="$(terraform output -raw dataset_id)"
  export SF_REGION="us-central1"   # or your deploy region
  ```

  The five above are all **reviewing** needs. **Submitting** a run needs a little more, depending on
  which runtime the run targets — also straight from `terraform output`:

  - **Spark / Dataproc** (the CLI, and notebooks `03` / `08`) resolves its batch infra from four more
    vars (`BatchInfra.resolve()` reads them; a bare shell that skips them fails with
    `missing required environment variable SF_CODE_BUCKET`):

    ```bash
    export SF_CODE_BUCKET="$(terraform output -raw code_bucket)"
    export SF_CONTAINER_IMAGE="$(terraform output -raw runtime_image_repo):latest"
    export SF_COMPUTE_SA="$(terraform output -raw compute_sa)"
    export SF_SUBNETWORK_URI="$(terraform output -raw subnetwork_uri)"
    # optional — only the Dataproc-*cluster* path (packed-venv deps / pre-baked GPU image) reads these:
    export SF_VENV_ARCHIVE="$(terraform output -raw venv_archive_uri)"
    export SF_GPU_IMAGE="$(terraform output -raw gpu_image_uri)"
    ```

  - **Ray on Vertex** (notebook `04`) reads `compute_sa`, `code_bucket`, and (for the private path)
    `network_attachment_id` — notebook `04` calls `RayInfra.from_terraform_outputs()` itself, so on a
    machine with the Terraform dir reachable it needs no extra exports.

  The Spark/Dataproc path (`03` / `08` / the CLI) resolves its infra from the `SF_*` env only — it
  does **not** auto-read `terraform output` — so export the four `SF_*` vars above before launching
  from a bare shell (Cloud Shell, CI, a local kernel). The deployed Colab templates and Composer bake
  them in, so a one-click or Composer run never needs this step.

  > **No Terraform state handy?** These values are also **deterministic from your `project_id` +
  > `region`** — the deployment names everything by convention — so a Cloud Shell session with neither
  > the Terraform directory nor its state can still derive the full `SF_*` set (including the
  > batch-infra and Ray extras) from those two variables. The copy-paste convention block lives in the runbooks that need
  > it offline: [workshop.md Act 1](./workshop.md#act-1--populate-the-run-history-at-100k-cloud-shell-before-the-workshop)
  > (demo) and [operations.md §3](./operations.md#3-re-run-a-config-short-runs-only) (rework). Only
  > override defaults (dataset/bucket/connection/subnet names) break the convention — then read the
  > exact `terraform output` values instead.

- For submitting (not for reviewing), install the client deps. The lean, disk-light choice is the
  `[submit]` extra — `uv sync --extra submit` — the Dataproc **and** Ray submit clients with **no**
  pyspark, so it fits a thin client like Cloud Shell (pyspark's ~300MB of JARs are only needed for
  notebook 01's interactive Spark Connect path, not for launching batches). Use `[spark]` / `[ray]`
  only when you also want that interactive Connect session or a local Spark session. The runtime
  image itself is code-free — see the note above.

## Notebooks and kernels

The notebooks run from a local kernel — most are pure **orchestration** of the Dataproc / Ray /
BigQuery work (with ADC for auth), so a local clone drives cloud compute directly. Register the
project kernel once:

```bash
uv sync                                                          # core deps incl. ipykernel + matplotlib
uv run python -m ipykernel install --user --name scale-forecasting --display-name "scale-forecasting (uv)"
```

Three notebooks need **no** cluster and run fully locally: `model_playground.ipynb` (pure
`worker.run_cell`), `07_scale_review.ipynb` (compares several runs side by side), and
`09_review_run.ipynb` (review a finished run via the `review` layer) — the latter two are read-only
over the registry, needing only the `SF_*` env + ADC. The rest submit to Dataproc / Ray / BigQuery
(`08_run_and_monitor.ipynb` launches a run *and* watches it land, so it submits too).

**Python-version note.** The project pins Python **3.11** on every surface (why: Vertex Ray
client↔cluster parity and the Dataproc packed-venv — see [version_matrix.md](./version_matrix.md)).
Notebook 01's interactive Spark Connect path holds that parity too, and documents a **remote-batch**
escape hatch (`main.run(cfg)` with no injected session) — the *identical* engine on-cluster, same
`run_id`, same results. For the full per-notebook version mapping (local and on Colab Enterprise) and
the runtime template Terraform ships, see [notebook_runtimes.md](./notebook_runtimes.md).

## 1. Check the config offline first

`--dry-run` resolves the config and estimates the fan-out (series × models × folds = cells) without
touching GCP. Always cheap, always safe:

```bash
python -m scale_forecasting.main --config configs/explode_demo.json --dry-run
```

## 2. Submit the run

The one entrypoint that runs the **whole config** is `main.run` — it plans the DAG (one job per model
family) and launches every family in parallel under one `run_id`, each on its resolved runtime, then
the ensemble node:

```bash
python -m scale_forecasting.main --config configs/explode_demo.json
```

Useful flags: `--n-series N` overrides `series_limit` (scale the same config up or down without
editing it), `--max-executors N` caps a Spark batch's executors, `--force` re-runs (a fresh attempt
under the same `run_id`).

Under the hood `main.run` launches each family through its runtime's submit-side entrypoint — you can
also drive those directly for a single-runtime job. Both stage your current `src/` + the config JSON
to GCS, then submit:

```bash
# Spark family → Dataproc (explode fan-out: one task per (series, model) cell)
python -m scale_forecasting.submit --config configs/explode_demo.json

# Ray family → Vertex
python -m scale_forecasting.ray_submit --config configs/ray_cpu_demo.json
```

Ray flags: `--n-series N`, `--cluster-name NAME` (reuse a standing cluster, skipping
create/teardown), `--no-wait`.

**BigQuery-native models** need no separate submit — list them in `models` (e.g. `arima_plus`,
`timesfm`) and they run **in parallel** with the Python families under the same `run_id`; `main.run`
orchestrates all of them.

> **Scale knob.** The `configs/*_100k.json` files are demo configs at `series_limit=100000` — the same
> shapes, at scale. Submit them the same way (review spend first).

## 3. Watch it land

Runs are written through the Storage Write API and are **async-visible** — rows appear a few seconds
after the engine finishes, so poll briefly. The run header:

```sql
SELECT * FROM `PROJECT.DATASET.v_run_summary` WHERE run_id = 'YOUR_RUN_ID';
```

`v_run_summary` is one row per run: `status`, `python_runtime`, `n_series`/`n_models`,
the engine's `runtime_seconds`, plus the Dataproc telemetry overlay — `total_wall_s`,
`overhead_seconds` and `overhead_fraction` (provisioning tax, which amortizes as series grow), and
`dcu_milli_seconds` (the cost proxy).

For the **per-family** breakdown — which family ran on which runtime/hardware, its platform job id,
status, and per-job wall-clock — query `v_run_jobs` (one row per `(run_id, family)`, plus the
ensemble node):

```sql
SELECT family, runtime, hardware, system_job_id, status, runtime_seconds
FROM `PROJECT.DATASET.v_run_jobs` WHERE run_id = 'YOUR_RUN_ID';
```

Rather than poll SQL by hand, `review.monitor_run(run_id)` (also `Forecaster.monitor()`) rolls the
header, the run's own config, the per-family jobs, and the landed-cell counts into a `RunProgress`:
per-family job state on its runner, `n_done / n_expected` cells, mean fit time, and a run-wide
fraction — with `review.plot_progress` for the progress-bar readout. Progress is coarse (cells land
when a family's writer runs, often at job end), so the per-job `status` is the primary live signal.
[`08_run_and_monitor`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/08_run_and_monitor.ipynb)
launches a run on a background thread and drives this live-refreshing dashboard until it lands.

**When the bar stops moving.** A registry row is written *by the job*, so a job that dies without
writing leaves its row `RUNNING` and its bar frozen — visually identical to a slow one. Two things
tell them apart, and they cost differently:

- **`quiet_seconds`** (free) — every family carries how long it has been since its last registry
  signal, and `plot_progress` prints it (`5/100 · RUNNING · quiet 22m`). It comes off rows the
  monitor already reads, so it costs no extra call and is safe in a poll loop. It is an age, not a
  verdict: a family that writes its cells at job end is legitimately quiet for its whole run.
- **`monitor_run(run_id, probe=True)`** (a few native calls) — escalates the run's non-terminal
  jobs to their runtime and attaches a reconciled `ProbeReport`, so the bar says `LOST` or
  `RUNNING_CONFIRMED` instead of an age. It is the same reconciliation the `--probe` CLI verb (and
  `Forecaster.probe()`) performs, reached from the monitor — use it when an age has grown long
  enough to be suspicious, not on every poll. An already-terminal run short-circuits and touches no
  runtime at all.

For the full operational picture — the six verdicts and how to read them, settling a stale row from
the verdict, cancelling safely, and what a cancelled run keeps — see
[troubleshooting.md § In-flight runs](./troubleshooting.md#in-flight-runs--probe-settle-cancel).

## 4. Review — which model won

```sql
SELECT model_type, compute_engine, n_cells, no_artifact_rate,
       median_fit_seconds, mean_wape, mean_mae
FROM `PROJECT.DATASET.v_model_leaderboard`
WHERE run_id = 'YOUR_RUN_ID'
ORDER BY mean_wape;
```

`v_model_leaderboard` is one row per `(run_id, model_type, ensemble_id)`:

- `compute_engine` — `spark` / `ray` / `bigquery` / `ensemble`, so the two tracks (and ensembles)
  are distinguishable on one board.
- `n_cells` / `no_artifact_rate` — coverage and failure signal (a model failing every cell —
  e.g. a missing native lib — shows as `no_artifact_rate = 1.0`).
- `median_fit_seconds` — per-cell fit time (the per-model straggler signal).
- `mean_wape` / `mean_mae` — the decision metrics, populated where a backtest ran.

For the data-science view without hand-writing SQL, `review.review_run(run_id)` (also
`Forecaster.review_run()`) returns a `RunReview`: every model best-first in the run's own
`decision_metric`, the best model per family and overall, the full metric panel aggregated across
every series (mean + p10/p50/p90, read server-side so it holds at 100k+), and each ensemble's lift
over the best base model — with `plot_leaderboard` / `plot_metric_distribution` and, for the
execution timeline, `sdk.build_trace_frame` + `plot_trace`.

The demo notebooks ([`notebooks/`](https://github.com/statmike/scale-forecasting/tree/main/notebooks)) wrap these queries in charts:
[`07_scale_review`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/07_scale_review.ipynb) compares several runs
side by side, [`08_run_and_monitor`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/08_run_and_monitor.ipynb) launches
a run and watches it land, and [`09_review_run`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/09_review_run.ipynb)
reviews any finished run in data-science detail.

These views read from the underlying registry tables (`run_registry`, `run_jobs`, `forecast_metadata`,
`forecast_predictions`, `backtest_oof`). To query the raw values — the forecast points themselves, or
per-fold OOF truth — see [output_schemas.md](./output_schemas.md) for every column and how the tiers
join.

## 5. Re-ensemble a completed run (optional)

You don't have to re-run the base models to try a different consensus. `ensemble_run` reads an
already-completed run's base predictions and scores new `ensemble_*` pseudo-models onto the same
leaderboard:

```bash
python -m scale_forecasting.ensemble_run \
  --config configs/mixed_demo.json \
  --run-id YOUR_RUN_ID \
  --strategies mean,median,inverse_error
```

- `--run-id` is the **base run** whose forecasts get blended (defaults to the config's own
  `make_run_id`). Pass it explicitly to ensemble a run computed under a different config revision.
- `--strategies` overrides the config's ensemble block (comma-separated; a bad name fails validation
  with a clear error). Omit it to use the config's own `ensemble.strategies`.
- Each distinct ensemble config lands a distinct `ensemble_id`, so several ensembles coexist under
  one `run_id` on the leaderboard instead of overwriting each other. Re-running the *same* config is
  idempotent (append-only + dedupe-on-read).

Learned strategies (`nnls`/`ridge`/`xgb`) need the base run to have had `backtest.enabled` (they fit
on the OOF); calculated ones (`mean`/`median`/`inverse_error`) don't.

## 6. Managing the registry

Runs accumulate. `registry.ops` is the operator surface over the one registry your `SF_*`
environment points at — seven verbs, reachable identically from the CLI, the SDK (`Registry`), and a
notebook:

```bash
python -m scale_forecasting.registry.ops doctor          # read-only: what's in here, what's wrong
python -m scale_forecasting.registry.ops drop-run RUN_ID # preview — deletes nothing
python -m scale_forecasting.registry.ops drop-run RUN_ID --yes
```

```python
from scale_forecasting import Registry

reg = Registry()                 # or Forecaster(...).registry()
print(reg.doctor())
reg.drop_run("abc123", yes=True)
```

| Verb | What it does |
|------|--------------|
| `init` | Create this registry's five tables + three views (idempotent). Point `SF_REGISTRY_DATASET_ID` at a fresh dataset and this stands up a second registry. Does **not** touch the source panel. |
| `doctor` | Read-only report: per-table row counts, runs still marked `RUNNING`, and artifact prefixes with no `run_registry` row. Touches nothing. |
| `close-runs` | Finalize abandoned `RUNNING` headers to the status their own job rows already imply. Deletes nothing. Names no runs = every stuck header. |
| `drop-run` | Delete named run(s) from every tier — GCS artifacts, BQML `sf_model_*` objects, then registry rows. Takes as many ids as you like. |
| `sweep-orphans` | Delete artifact prefixes under *this* registry's root that have no `run_registry` row. |
| `snapshot` | BigQuery table snapshots of the five registry tables — `--into` another dataset, `--expiration-days` for a TTL. |
| `export` | Dump the registry to GCS as Parquet (default) or newline-delimited JSON. |

`drop-run` and `sweep-orphans` are **previews by default** — they print the exact runs, object
counts and byte totals they would touch and change nothing until you add `--yes`. They also refuse
to touch a run whose header is still `RUNNING` or `PENDING`; check with `monitor(probe=True)` first
(a `RUNNING` row can also be a dead job), then `--force` if you're sure.

**`close-runs` is the one for a header that is stuck rather than wrong.** A driver that dies after
writing its header leaves a `RUNNING` row forever, and none of the other verbs fit: `--cancel`
stamps `CANCELLED` over families that actually completed, `drop-run` destroys real predictions to
repair a status field, and `--probe` only reads. `close-runs` writes the header status its own job
rows already imply — every job `COMPLETED` ⇒ `COMPLETED`, a mix of terminals ⇒ `PARTIAL`, no job
rows at all ⇒ `FAILED` (the run died in the submit path) — and touches nothing else:

```bash
python -m scale_forecasting.registry.ops close-runs          # preview every stuck header
python -m scale_forecasting.registry.ops close-runs --yes
```

It **skips any run that still has a non-terminal job row**, with the reason printed, because only a
runtime probe can tell a live job from a stale one. Settling those rows is what unblocks it, so the
order is `monitor(probe=True)` first, then `main --settle --force` for the rows the probe can call,
then `close-runs` for the header. Settle is the same preview-by-default shape as the verbs in this
table — it writes only on confirmation, refuses anything ambiguous, and never deletes; the full
decision table is in
[troubleshooting.md § Settle a stale row](./troubleshooting.md#settle-a-stale-row).

**Order matters, and the verbs enforce it.** A registry row is the only index of which GCS objects
belong to which run, so every delete goes *artifacts first, rows last*. Dropping the rows first
would strand the artifacts permanently — which is what makes `sweep-orphans` necessary at all, for
everything stranded before this surface existed.

Two things you won't find here:

- **No wipe verb.** Deleting a whole registry is `bq rm -r -f <project>:<dataset>` (or the BigQuery
  console) — a one-liner nobody needs us to wrap, and wrapping it invites the accident. If you want
  a disposable registry, give it its own dataset via `SF_REGISTRY_DATASET_ID` and delete that.
  [operations.md §2c](./operations.md#2c-discard-the-registry-entirely--bq-rm-and-there-is-no-verb-for-it)
  has the full teardown, including the shared-dataset case (drop the objects by name so the source
  panel survives) and why you sweep the artifacts *before* the rows that index them.
- **No source-table verb.** The source panel is a separate lifetime (a Spark seed job over millions
  of rows); nothing in this surface reads or writes it.

> Just re-running a config? You usually need none of this. Runs land under a deterministic `run_id`
> and dedupe on read, so re-running the same config never double-counts.

## Quick reference — entrypoints

| Command | Purpose |
|---------|---------|
| `python -m scale_forecasting.main --config C [--dry-run]` | Orchestrate one run — a job per family in parallel (Spark/Ray ∥ BigQuery) under one `run_id`. |
| `python -m scale_forecasting.submit --config C` | Submit a single Spark family job to Dataproc. |
| `python -m scale_forecasting.ray_submit --config C` | Submit a Ray run to Vertex. |
| `python -m scale_forecasting.ensemble_run --config C [--run-id R] [--strategies …]` | Re-ensemble a completed run. |
| `python -m scale_forecasting.playground --model M [--backtest]` | Run one model on sample data, offline (no GCP). |
| `python -m scale_forecasting.registry.ops <verb>` | Manage the registry — `init` / `doctor` / `close-runs` / `drop-run` / `sweep-orphans` / `snapshot` / `export`. |
