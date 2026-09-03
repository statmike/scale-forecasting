# Workshop runbook — demo the runtimes and the family DAG, then review at scale

**You've deployed the platform** (see [`terraform/README.md`](https://github.com/statmike/scale-forecasting/blob/main/terraform/README.md)) and the smoke
forecast came back `SUCCEEDED`. This is the guided path from there to a full demo: submit a few runs at
**100k series** — one per Python runtime, plus the all-families run — then walk the notebooks in
**Colab Enterprise**, ending on `07_scale_review`, which renders the cross-run comparison over the runs
you just made.

Everything here runs **in the browser** — [Cloud Shell](https://console.cloud.google.com/?cloudshell=true)
for the Act 1 submits, [Colab Enterprise](https://console.cloud.google.com/vertex-ai/colab) for the
Act 3 notebooks. No local machine or SDK install is assumed.

> **Prep before, present live.** Acts 1–2 populate the run history and (optionally) pre-render the
> tour — real compute (~minutes to hours), best done **before** the workshop. Act 3 is the live tour;
> `07_scale_review` needs Act 1's runs to have data to show. The three acts run **in order**, and the
> section numbers are strictly sequential (1 → 2 → 3).

---

## Who can run this — IAM roles

A workshop presenter and any attendees you invite need their own grants (the Terraform grants roles to
the *service accounts*, not to people). The complete, copy-paste role set — for both the **Cloud Shell
submitter** (Act 1) and the **Colab Enterprise notebook user** (Act 3) — lives in one place:

➡️ **[Human users (running jobs + notebooks)](./deploying_on_gcp.md#human-users-running-jobs--notebooks)**

For a workshop, grant a **Google group** once and add attendees to it — one binding, not one per
person. Do this before the session so nobody is blocked at the console.

---

## Act 1 — Populate the run history at 100k (Cloud Shell, before the workshop)

Three `configs/*_100k.json` files give `07_scale_review` a clean cross-run comparison — the same work on
two runtimes, plus the all-families DAG:

| Config | Runtime(s) | What it demonstrates |
|--------|-----------|----------------------|
| `explode_100k.json` | Spark | The 100k CPU workhorse — `theta`, `holtwinters`, `sarimax`, `xgboost` (statistical + ml families) on Dataproc Serverless, one Spark task per `(series, model)` cell. |
| `ray_100k.json` | Ray | **The same four models, on Ray** — the runtime-parity comparison. Same unit of work, same answers, different engine; the wall-clock/overhead difference is the story. |
| `all_families_10k.json` | Ray ∥ BigQuery | **The family DAG.** One config, seven models across **all four families** (statistical / ml / deep-learning / native) — each family runs as its own parallel job under one `run_id`, deep-learning packing NeuralProphet onto fractional T4s and the native models running in BigQuery. |

`explode_100k` and `ray_100k` are the *same models on different runtimes*; `all_families_10k` is *one
config fanned into a job per family*. Together they give notebook 07 both stories: runtime parity and the
per-family placement.

**Why the family DAG runs at 10,000 series and not 100,000.** It is the only one of the three that
includes a deep-learning model, and NeuralProphet needs a T4. A default project is allowed **four**
of them, which is enough for ~10,000 series in about ten hours and nowhere near enough for 100,000.
The CPU-only configs stay at 100k because CPU quota is not the binding constraint at that scale.
[Quota and scale](quota_and_scale.md) has the arithmetic, and the quota to request if you want to
run the family DAG larger.

Open [Cloud Shell](https://console.cloud.google.com/?cloudshell=true), then:

```bash
# Clone (skip if you still have the deploy clone) and install with uv — the repo is uv-managed
# (pyproject.toml + uv.lock). The `submit` extra is the thin client set: it pulls the Dataproc +
# Ray submit clients (both runtimes) but NOT pyspark — batch submission never imports it, and
# pyspark's ~300MB of JARs overflow Cloud Shell's home quota. (pyspark is only for notebook 01's
# interactive Spark Connect path, which runs in Colab Enterprise — Act 3 — not here.)
cd ~ && git clone https://github.com/statmike/scale-forecasting.git 2>/dev/null; cd ~/scale-forecasting
uv sync --extra submit
```

> **`uv` in Cloud Shell.** If `uv` isn't on `PATH`, install it once (it lands in `~/.local/bin`, which
> is durable): `curl -LsSf https://astral.sh/uv/install.sh | sh` then `source ~/.bashrc`. `uv sync`
> creates `.venv` from the locked `uv.lock`; prefix the run commands below with `uv run` so they use it.

**Wire the `SF_*` identity.** These values are **deterministic from your project id + region** — the
deployment names everything by convention — so you can set them in **any** Cloud Shell session without
being in the Terraform directory or having its state. Set the two variables at the top and paste the
rest as-is:

```bash
# --- set these two (the same values you deployed with) ----------------------
PROJECT=gcp-scale-forecasting   # ← your project_id (this is the reference deploy's)
REGION=us-central1              # your deploy region
# ---------------------------------------------------------------------------
export SF_PROJECT_ID="$PROJECT"
export SF_REGION="$REGION"
export SF_DATASET_ID="scale_forecasting"                                   # deploy default
export SF_CONNECTION="$PROJECT.$REGION.sf-iceberg"
export SF_WAREHOUSE_URI="gs://$PROJECT-warehouse/warehouse"
export SF_CODE_BUCKET="$PROJECT-code"
export SF_COMPUTE_SA="scale-forecasting-compute@$PROJECT.iam.gserviceaccount.com"
export SF_CONTAINER_IMAGE="$REGION-docker.pkg.dev/$PROJECT/scale-forecasting/spark-runtime:latest"
export SF_SUBNETWORK_URI="https://www.googleapis.com/compute/v1/projects/$PROJECT/regions/$REGION/subnetworks/scale-forecasting-compute"
```

> **These `export`s live only in the current shell.** They don't survive a disconnect, and a **new
> `tmux` session is a new shell** — so if Cloud Shell drops, you reconnect, or you start/attach a
> different tmux session, **re-paste this whole block before submitting** or the submit fails with
> `ConfigError: missing required environment variable SF_PROJECT_ID`. (Only files persist across a
> Cloud Shell disconnect; environment variables do not.)

> These follow the deployment's naming convention, which holds as long as you kept the defaults for
> `dataset_id`, the bucket/connection/subnet/repo names, and the region. **If you overrode any of those
> in `terraform.tfvars`,** read the exact values instead — from the same Cloud Shell you deployed in
> (`cd terraform/main && terraform init -backend-config="bucket=$PROJECT-tfstate" && terraform output`),
> or from the console (BigQuery for the dataset/connection, Cloud Storage for the buckets).
>
> The `SF_*` reference, the config reference, and the review SQL all live in
> [`docs/running_and_reviewing.md`](./running_and_reviewing.md) — this runbook just orchestrates it for
> the workshop. If a submit errors on a missing var, that doc's Prerequisites table is the checklist.

**Sanity-check a config offline first** (resolves the config + estimates the fan-out
`series × models × folds`, touches no GCP):

```bash
uv run python -m scale_forecasting.main --config configs/explode_100k.json --dry-run
```

**Submit the three.** `main.run` is the one entrypoint that runs a whole config — it plans the DAG (one
job per model family) and launches every family in parallel under one `run_id`, each on its resolved
runtime. Each command **blocks until its run finishes** (that's how it stamps the wall-clock / DCU
telemetry `07_scale_review` charts), so they run **one after another** — budget for the sum, not the max.

> **Run them under `tmux` so a lost tab doesn't sever the wait.** Cloud Shell disconnects if the
> browser tab sleeps or the network blips, which SIGHUPs a foreground process — killing the current
> wait and every command queued behind it in the line. `tmux` keeps the session alive server-side
> so you can reattach:
>
> ```bash
> tmux new -s runs        # start (reattach later with: tmux attach -t runs)
> ```
>
> Then, inside tmux, submit the three. The Dataproc/Ray jobs themselves run server-side and survive a
> disconnect regardless — but only a live wait stamps their telemetry, so `tmux` is what protects
> the `07` charts.
>
> **A new tmux session is a new shell** — it does *not* inherit the `SF_*` exports from the tab you
> ran them in. Re-paste the `SF_*` block above **inside** tmux before submitting, or the submit fails
> with `missing required environment variable SF_PROJECT_ID`. (Already inside tmux? `tmux new` will
> warn `sessions should be nested with care` and no-op — just re-export and submit in the session
> you're in.)

```bash
uv run python -m scale_forecasting.main --config configs/explode_100k.json        # Spark
uv run python -m scale_forecasting.main --config configs/ray_100k.json            # Ray
uv run python -m scale_forecasting.main --config configs/all_families_10k.json   # every family
```

> **On the wait timeout.** A 100k run runs longer than the client's old 15-minute default wait, so
> the submitter blocks up to **2 h**. That bound is **fixed on this entrypoint** — `--wait-timeout`
> is a flag on the lower-level `python -m scale_forecasting.submit` module, and passing it to
> `main` fails with `unrecognized arguments`. If your run will outlast two hours, drive it from a
> persistent VM (see the note at the end of this Act) rather than reaching for a longer wait.
> If you ever *do*
> see a client-side `TimeoutError`, the **jobs are unaffected** — they keep running server-side; only
> the local wait gave up. Check true state with
> `gcloud dataproc batches list --project $PROJECT --region $REGION` or the `v_run_summary` query
> below, and re-submit only what didn't land.

**Watch them make progress (healthy vs. stuck).** A 100k run runs for 1–2 h, so "is it working or
hung?" is the natural question. The runs write forecast rows to `forecast_metadata` **as they go**, so
the real health signal is simply: *are rows accumulating?* Run this in
[BigQuery Studio](https://console.cloud.google.com/bigquery), then **run it again in ~5 min** — if the
counts climb, the runs are healthy (not wedged):

```sql
-- swap gcp-scale-forecasting for your project_id if you deployed elsewhere
SELECT run_id, COUNT(*) AS cells_written, MAX(created_at) AS latest_write
FROM `gcp-scale-forecasting.scale_forecasting.forecast_metadata`
WHERE run_id LIKE '%100k%'
GROUP BY run_id
ORDER BY latest_write DESC;
```

Each run's target is **`n_series` × `n_models` cells** — `explode_100k` / `ray_100k` are 100,000 × 4
models = **~400,000 cells**, and `all_families_10k` is 10,000 × 7 models = **~70,000 cells** — so
`cells_written / target` is a rough % complete. Reading the numbers:

- **Counts climbing between two checks = healthy.** Leave the jobs alone; they're serverless/autoscaling
  and finish server-side regardless of your shell. **Don't kill them to "restart"** — you'd discard the
  cells already written and pay to recompute (re-submitting the same config reuses the same
  deterministic `run_id` and dedupes-on-read, so nothing already done is wasted).
- **Each family fills independently** in `all_families_10k` — the deep-learning family (NeuralProphet on
  T4s) is the slowest, so its cells trail the statistical/ml ones. Expected: a run's wall-clock is its
  *slowest* family, and the others finish and wait.
- **Genuinely stuck** looks like: count **flat** across several minutes **and** the job's UI (the
  Dataproc batch detail page → *View Spark UI*, or the Ray dashboard) shows no task progress. Only then
  investigate.

> **The metric columns (`mae`/`rmse`/`wape`/…) are NULL — by design.** Accuracy metrics need a
> held-out actual to score against, which only exists when **backtest is on**. The 100k configs run
> with **backtest off** (the fleet-scale default — at 100k you're proving *throughput*, and folds
> would multiply the compute), so every accuracy column is NULL for these runs. The forecasts and the
> scale telemetry (wall-clock, DCU) — the actual 100k showpiece — are fully populated. Accuracy parity
> across engines is the **notebook 03** story (small scale, backtest on). Same point the `07`
> expectation-setter below makes.

**Confirm they landed and capture the `run_id`s.** The `run_id` is a deterministic digest of the
config, so the shipped configs always produce the **same** ids — but confirm via the registry rather
than assume. In [BigQuery Studio](https://console.cloud.google.com/bigquery) (or `bq query`):

```sql
-- swap gcp-scale-forecasting for your project_id if you deployed elsewhere
SELECT run_id, created_at, status, python_runtime, n_series, n_models
FROM `gcp-scale-forecasting.scale_forecasting.v_run_summary`
ORDER BY created_at DESC
LIMIT 25;
```

You want three **`COMPLETED`** rows — one `explode-100k-…`, one `ray-100k-…`, one
`all-families-100k-…`. (`COMPLETED` is the registry's word for a finished run. `SUCCEEDED` is the
*platform's* word — what Dataproc and Ray call a finished job — and it never appears in this column;
the registry's vocabulary is `COMPLETED` / `PARTIAL` / `FAILED` / `CANCELLED`, with `RUNNING` and
`PENDING` as the non-terminal pair. `PARTIAL` means the run finished with some cells failed.)
Copy those three `run_id`s; Act 3's notebook 07 reads them. For the **per-family** breakdown of any run
— which family ran on which runtime/hardware, its platform job id, and per-job wall-clock — query
`v_run_jobs` (one row per `(run_id, family)`, plus the ensemble node):

```sql
SELECT run_id, family, runtime, hardware, status, runtime_seconds
FROM `gcp-scale-forecasting.scale_forecasting.v_run_jobs`
WHERE run_id LIKE 'all-families-100k-%'
ORDER BY runtime_seconds DESC;
```

> **Expectation-setter for the accuracy chart.** The 100k configs run with **backtest off** (that's
> the fleet-scale default), so `07_scale_review`'s accuracy-parity panel (`mean_wape`) is **all-NULL**
> for these runs — the notebook says so explicitly. **The scaling / efficiency panels are the 100k
> showpiece** (wall-clock, provisioning overhead, DCU) and the **per-family placement** (`v_run_jobs`).
> Accuracy parity across engines is demonstrated at small scale in notebook **03** (where backtest is
> on). Say this out loud before opening 07 and it's a feature, not a surprise.

> **Runs that outlast Cloud Shell?** The full-suite run `configs/all_families_10k_full.json`
> (100k × 7 models, backtest on, NeuralProphet on T4s) runs for **hours** — longer than Cloud Shell
> will hold the orchestrator that finalizes the run header. Drive it from a **persistent VM** instead:
> ➡️ [operations.md §4 — Long runs on a persistent
> VM](./operations.md#4-long-runs-on-a-persistent-vm-when-a-run-outlasts-cloud-shell).

---

## Act 2 — Pre-render the notebook tour (optional, the night before)

Act 3 is a **live** tour — but the expensive notebooks (`04_ray_on_vertex` stands up a Ray cluster;
`01`/`03` submit Dataproc batches) take too long to run in front of an audience. This step
**pre-executes every notebook headless** so tomorrow you walk **already-rendered** notebooks (outputs
baked in) from the Colab Enterprise **Executions** menu — and run only the cheap, fast ones (`07`,
`02`, playground) live if you want.

It reuses the same Vertex `NotebookExecutionJob` machinery as the acceptance harness, but in
**fire-and-forget** mode (`--no-wait`): it submits all the notebooks back-to-back (each submit
returns in ~1s) and returns — **the notebooks then run concurrently, server-side, decoupled from your
shell.** You can close Cloud Shell; they keep going. No `tmux` needed for this step (nothing blocks).

This runs in its **own** Cloud Shell, at a different time from Act 1 — so it's **self-contained** and
does **not** need the Terraform directory or state. Everything it needs is either a naming convention
(the runner SA and code bucket, exactly like Act 1's `SF_*` block) or looked up from the deployed
resources by its stable display name (the `sf-main` Colab template). Set the two variables at the top
and paste the rest as-is:

```bash
# --- set these two (the values you deployed with) --------------------------
PROJECT=gcp-scale-forecasting   # ← your project_id (this is the reference deploy's)
REGION=us-central1              # your deploy region
# ---------------------------------------------------------------------------
cd ~/scale-forecasting
# By convention (same names Terraform assigns) — no state lookup needed:
RUNNER_SA="scale-forecasting-runner@$PROJECT.iam.gserviceaccount.com"
CODE_BUCKET="$PROJECT-code"
# The template id is API-minted, so fetch it by its stable display name (sf-main) — works from any
# shell, no Terraform. NOTE: on some projects `--format="value(name)"` returns only the bare numeric
# id, but the CLI needs the FULL resource path, so we normalise to
# `projects/.../notebookRuntimeTemplates/<id>`:
_main_id=$(gcloud colab runtime-templates list --project "$PROJECT" --region "$REGION" \
  --filter="displayName=sf-main" --format="value(name)")
_prefix="projects/$PROJECT/locations/$REGION/notebookRuntimeTemplates"
MAIN_TEMPLATE="$_prefix/${_main_id##*/}"
echo "MAIN=[$MAIN_TEMPLATE]"   # should start with projects/

uv run python -m scale_forecasting.notebook_acceptance \
  --no-wait --tier full \
  --project "$PROJECT" --region "$REGION" \
  --main-template "$MAIN_TEMPLATE" \
  --service-account "$RUNNER_SA" \
  --gcs-output "gs://$CODE_BUCKET/notebooks" \
  --run-label "demo-$(date +%Y%m%d)" \
  --ack-out-of-org
```

> **Note:** this fan-out does *not* need the `SF_*` env block from Act 1 — the notebooks it launches
> get their `SF_*` identity from the **template env** (baked in at deploy), and the submitter takes
> everything it needs as explicit `--flags` above. The only prerequisites are `uv sync` (Act 1's
> install) and being authenticated (`gcloud auth login` — Cloud Shell already is).

> **`--ack-out-of-org`** is required when your project is a **personal or standalone project outside
> a Google-corp org** (which is the usual case). Without it, Vertex rejects the submit with
> `FAILED_PRECONDITION (NOTEBOOK_RUNTIME_OUT_OF_ORGANIZATION)`. The flag acknowledges that the
> runtime service account's credentials may be visible to the project owner — pass it only for a
> project you trust (your own deploy qualifies). Inside a corp org it's unnecessary (a harmless no-op).

It prints each notebook's **job id** and a link to the **Executions** menu — the console page where
the jobs appear with live state, and where a finished one **opens as the executed notebook with
rendered outputs**. That menu is your tour surface tomorrow.

> **`--tier` picks how much to pre-render** (same cost tiers as the acceptance harness, and each tier
> is *cumulative* — it runs its own notebooks plus every cheaper tier's):
>
> | Tier | Adds | Total |
> |------|------|-------|
> | `smoke` | `model_playground`, `02_bigquery_native`, `07_scale_review`, `09_review_run` — registry-read or BigQuery only, cheap | 4 |
> | `batch` | `01_spark_via_connect`, `03_combo_and_ensemble`, `08_run_and_monitor` — these submit Dataproc work | 7 |
> | `full` | `04_ray_on_vertex` — stands up a live Ray cluster | 8 |
>
> `full` fires the whole tour at once — the **same total spend as one full acceptance run** (8 Colab
> runtimes + the Dataproc batches + a live Ray cluster), just concurrent. Pre-workshop prep, not free
> — but it's what pre-renders everything.

> **Run this *after* Act 1's three 100k runs land.** `07_scale_review` reads those runs, so pre-render
> it only once they're `COMPLETED` — otherwise its rendered output shows missing data. (`07` reads
> its `RUN_IDS` from the shipped deterministic defaults, which match the unchanged configs — if you
> overrode a config, edit `07`'s `RUN_IDS` cell before this step or run `07` live instead.)

**Tomorrow:** open the [Executions menu](https://console.cloud.google.com/vertex-ai/colab/execution-jobs),
find each `sf-demo-…` job, and click into the pre-rendered notebook. Run `07` and `02` (and any other
cheap one) **live** in the console on `sf-main` (every notebook uses that one template — see the Act 3
table) for the interactive moments, and lean on the pre-rendered set for the expensive `01`/`03`/`04`.

---

## Act 3 — The guided notebook tour (Colab Enterprise, live)

Every notebook has a one-click **Run in Colab Enterprise** badge in its first cell. Clicking it
imports the notebook; then **pick the runtime template it names** and **Run all** — the deployed
templates already carry the `SF_*` identity in their env, so there is **no environment cell to fill
in**. The per-notebook template mapping and the one-click mechanics are documented in
[`docs/notebook_runtimes.md`](./notebook_runtimes.md).

Run them **in this order** — each builds on the story of the last:

| # | Notebook | Template | What you show | Scale |
|---|----------|----------|---------------|-------|
| 1 | [`model_playground`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/model_playground.ipynb) | `sf-main` | Pick any registered model, fit it on a small panel — the one unit of work, no cluster. | sample |
| 2 | [`01_spark_via_connect`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/01_spark_via_connect.ipynb) | `sf-main` | The Spark UDF fan-out (`applyInPandas`, one task per cell) over a live Dataproc Connect endpoint. | 100 |
| 3 | [`02_bigquery_native`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/02_bigquery_native.ipynb) | `sf-main` | The BigQuery-native family — `ARIMA_PLUS` + `TimesFM` as pure SQL, no cluster. | 100 |
| 4 | [`03_combo_and_ensemble`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/03_combo_and_ensemble.ipynb) | `sf-main` | One config mixing a Spark model **and** the BQ natives under one `run_id`, with ensembles on — **and the accuracy-parity leaderboard** (backtest is on here). | 10 |
| 5 | [`04_ray_on_vertex`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/04_ray_on_vertex.ipynb) | `sf-main` | The Python models on a Ray-on-Vertex cluster ∥ the BQ natives — job submission from any authenticated client via the PSC-I attachment. | demo |
| 6 | [`08_run_and_monitor`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/08_run_and_monitor.ipynb) | `sf-main` | The **live** half of the run/review pair — launches one config mixing Spark `theta` with the BQ natives (backtest + ensembles on) and watches both tracks land under one `run_id`. | 100 |
| 7 | [`09_review_run`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/09_review_run.ipynb) | `sf-main` | The **finished-run** half — point it at what `08` just launched and it answers *how did it do?*: leaderboard, per-series metric distribution, ensemble lift, timeline. Read-only. | reads a run |
| 8 | [`07_scale_review`](https://github.com/statmike/scale-forecasting/blob/main/notebooks/07_scale_review.ipynb) | `sf-main` | **The payoff** — the cross-run comparison over your **Act 1** 100k runs: runtime parity (Spark vs Ray) and the per-family placement (`v_run_jobs`). Runs nothing; reads the registry views. | reads 100k |

**Notebook 07 is the one notebook you configure.** Its first code cell has an edit-me `RUN_IDS` block:

```python
# === Parameters — edit me ===============================================
RUN_IDS = {
    "spark":        "explode-100k-…",       # ← paste your Act 1 run_ids
    "ray":          "ray-100k-…",           #    (same models as spark, Ray runtime)
    "all-families": "all-families-100k-…",  #    (every family, one run_id)
}
# ========================================================================
```

Paste the three `run_id`s you captured in Act 1 (set any you skipped to `None`), then **Run all**. It
renders wall-clock + provisioning overhead from `v_run_summary`, the per-family-job placement from
`v_run_jobs`, and the per-model panel from `v_model_leaderboard`, side by side across the runs. (Because
the configs are shipped unchanged, the deterministic default ids in the cell often already match your
runs — but paste yours to be sure.)

> **Runtime note for notebook 01.** It runs interactively on `sf-main` like every other notebook: the
> Spark Connect session pins **Dataproc runtime 2.3** (the Connect floor; py3.11 workers, matching the
> 3.11 kernel) and attaches the project container image for deps. A **remote-batch** escape hatch is
> documented at the bottom of the notebook — the identical engine on-cluster, same result — if you'd
> rather not open a live Connect session.

---

## Cost + timing at a glance

- **Act 1 (three 100k runs):** the Spark run is a Dataproc Serverless batch (single-digit dollars,
  minutes); the Ray runs stand up and tear down an autoscaling cluster. Run once before the workshop and
  the results persist in the registry — Act 3 just reads them.
- **Act 3 (notebooks):** the demo-scale notebooks (100 series or fewer) are cents. `07_scale_review`
  runs no compute — it only queries views.
- **Reset when you're done:** the destructive teardown is documented in
  [`docs/running_and_reviewing.md`](./running_and_reviewing.md#resetting-the-environment-destructive),
  and `terraform destroy` removes the project's infrastructure.

## See also

- [`terraform/README.md`](https://github.com/statmike/scale-forecasting/blob/main/terraform/README.md) — deploy the platform (the step before this runbook).
- [`docs/running_and_reviewing.md`](./running_and_reviewing.md) — the submit/watch/review mechanics this
  runbook orchestrates, plus re-ensembling and teardown.
- [`docs/notebook_runtimes.md`](./notebook_runtimes.md) — per-notebook template mapping, the one-click
  open path, and the headless acceptance harness.
- [`docs/operations.md`](./operations.md) — rework an already-deployed environment, and the
  [persistent-VM path](./operations.md#4-long-runs-on-a-persistent-vm-when-a-run-outlasts-cloud-shell)
  for runs that outlast Cloud Shell.
- [`docs/deploying_on_gcp.md`](./deploying_on_gcp.md#human-users-running-jobs--notebooks) — the human
  IAM roles for running jobs and notebooks.
