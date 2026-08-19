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
  | `SF_DATASET_ID` | no | `scale_forecasting` | Registry dataset. |
  | `SF_REGION` | no | `us-central1` | Region. |

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

  The Ray path needs a little more, also straight from `terraform output`: `compute_sa`,
  `code_bucket`, and (for the private path) `network_attachment_id` — `RayInfra.from_terraform_outputs()`
  reads them for you (see notebook 04).

  > **No Terraform state handy?** These values are also **deterministic from your `project_id` +
  > `region`** — the deployment names everything by convention — so a Cloud Shell session with neither
  > the Terraform directory nor its state can still derive the full `SF_*` set (including the Ray
  > extras) from those two variables. The copy-paste convention block lives in the runbooks that need
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

Two notebooks need **no** cluster and run fully locally: `model_playground.ipynb` (pure
`worker.run_cell`) and `07_scale_review.ipynb` (read-only over the registry views, needs the `SF_*`
env + ADC). The rest submit to Dataproc / Ray / BigQuery.

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

Pick the entrypoint by runtime. Every one takes `--config` and stages your current `src/` + the
config JSON to GCS, then submits.

**Spark (Dataproc):** `--engine` selects the fan-out method.

```bash
# explode — the hero path: one task per (series, model) cell
python -m scale_forecasting.submit --config configs/explode_demo.json --engine explode

# naive — bucket on series (models run sequentially per series; the straggler anti-pattern)
python -m scale_forecasting.submit --config configs/naive_demo.json  --engine naive

# multi — one child explode batch per model family, all under ONE run_id
python -m scale_forecasting.submit --config configs/multi_demo.json  --engine multi
```

Useful flags: `--n-series N` overrides `series_limit` (scale the same config up or down without
editing it), `--max-executors N` caps executors, `--no-wait` returns as soon as it's submitted.

**Ray (Vertex):**

```bash
python -m scale_forecasting.ray_submit --config configs/ray_cpu_demo.json
```

Ray flags: `--n-series N`, `--cluster-name NAME` (reuse a standing cluster, skipping
create/teardown), `--no-wait`.

**BigQuery-native models** need no separate submit — list them in `models` (e.g. `arima_plus`,
`timesfm`) and they run **in parallel** with the Spark/Ray track under the same `run_id`. `main.run`
orchestrates both.

> **Scale knob.** The four `configs/*_100k.json` files are the demo configs at `series_limit=100000`
> — the same four approaches, at scale. Submit them the same way (review spend first).

## 3. Watch it land

Runs are written through the Storage Write API and are **async-visible** — rows appear a few seconds
after the engine finishes, so poll briefly. The run header:

```sql
SELECT * FROM `PROJECT.DATASET.v_run_summary` WHERE run_id = 'YOUR_RUN_ID';
```

`v_run_summary` is one row per run: `status`, `spark_method`/`python_runtime`, `n_series`/`n_models`,
the engine's `runtime_seconds`, plus the Dataproc telemetry overlay — `total_wall_s`,
`overhead_seconds` and `overhead_fraction` (provisioning tax, which amortizes as series grow), and
`dcu_milli_seconds` (the cost proxy).

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
- `median_fit_seconds` — per-cell fit time (the straggler signal under `naive`).
- `mean_wape` / `mean_mae` — the decision metrics, populated where a backtest ran.

The demo notebooks ([`notebooks/`](https://github.com/statmike/scale-forecasting/tree/main/notebooks)) wrap these two queries in a small polling helper
and a chart; [`07_scale_review`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/07_scale_review.ipynb) compares several runs
side by side.

Both views read from the underlying registry tables (`run_registry`, `forecast_metadata`,
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

## Resetting the environment (destructive)

`reset` drops the registry + source tables (for a clean reseed). It's a dry run without `--yes`:

```bash
python -m scale_forecasting.reset            # prints what WOULD be dropped, changes nothing
python -m scale_forecasting.reset --yes      # actually drops
```

## Quick reference — entrypoints

| Command | Purpose |
|---------|---------|
| `python -m scale_forecasting.main --config C [--dry-run]` | Orchestrate one run (Spark/Ray ∥ BigQuery). Rejects `multi` — use `submit`. |
| `python -m scale_forecasting.submit --config C --engine {explode,naive,multi}` | Submit a Spark run to Dataproc. |
| `python -m scale_forecasting.ray_submit --config C` | Submit a Ray run to Vertex. |
| `python -m scale_forecasting.ensemble_run --config C [--run-id R] [--strategies …]` | Re-ensemble a completed run. |
| `python -m scale_forecasting.playground --model M [--backtest]` | Run one model on sample data, offline (no GCP). |
| `python -m scale_forecasting.reset [--yes]` | Drop registry + source tables. |
