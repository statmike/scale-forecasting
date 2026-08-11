# Workshop runbook — demo every runtime, then review at scale

**You've deployed the platform** (see [`terraform/README.md`](../terraform/README.md)) and the smoke
forecast came back `SUCCEEDED`. This is the guided path from there to a full demo: submit one run per
approach at **100k series**, then walk the notebooks in **Colab Enterprise** — ending on
`07_scale_review`, which renders the cross-approach comparison over the runs you just made.

Everything here runs **in the browser** — [Cloud Shell](https://console.cloud.google.com/?cloudshell=true)
for the Act 1 submits, [Colab Enterprise](https://console.cloud.google.com/vertex-ai/colab) for the
Act 2 notebooks. No local machine or SDK install is assumed.

> **Two acts, two moments.** Act 1 (below) submits the four 100k runs — that's real compute
> (~minutes each) and is best done **before** the workshop so the run history is already populated.
> Act 2 is the live tour. `07_scale_review` needs Act 1's runs to have data to show.

---

## Who can run this — IAM roles

A workshop presenter and any attendees you invite need their own grants (the Terraform grants roles to
the *service accounts*, not to people). The complete, copy-paste role set — for both the **Cloud Shell
submitter** (Act 1) and the **Colab Enterprise notebook user** (Act 2) — lives in one place:

➡️ **[Human users (running jobs + notebooks)](./deploying_on_gcp.md#human-users-running-jobs--notebooks)**

For a workshop, grant a **Google group** once and add attendees to it — one binding, not one per
person. Do this before the session so nobody is blocked at the console.

---

## Act 1 — Populate the run history at 100k (Cloud Shell, before the workshop)

The four `configs/*_100k.json` files are the full-dataset runs — each forecasts **100,000 series** with
the same models (`theta`, `holtwinters`, `sarimax`, `xgboost`), differing only in **how** the work
fans out. Submitting all four gives `07_scale_review` one run per approach to compare.

Open [Cloud Shell](https://console.cloud.google.com/?cloudshell=true), then:

```bash
# Clone (skip if you still have the deploy clone) and install the submit extras:
cd ~ && git clone https://github.com/statmike/scale-forecasting.git 2>/dev/null; cd ~/scale-forecasting
pip install -e '.[spark,ray]'

# Wire the SF_* identity straight from terraform output (the submitters read these):
cd terraform/main
export SF_PROJECT_ID="$(terraform output -raw project_id)"
export SF_CONNECTION="$(terraform output -raw iceberg_connection)"
export SF_WAREHOUSE_URI="$(terraform output -raw warehouse_uri)"
export SF_DATASET_ID="$(terraform output -raw dataset_id)"
export SF_REGION=us-central1
export SF_CODE_BUCKET="$(terraform output -raw code_bucket)"
export SF_CONTAINER_IMAGE="$(terraform output -raw runtime_image_repo):latest"
export SF_COMPUTE_SA="$(terraform output -raw compute_sa)"
export SF_SUBNETWORK_URI="$(terraform output -raw subnetwork_uri)"
cd ~/scale-forecasting
```

> The `SF_*` wiring, the config reference, and the review SQL all live in
> [`docs/running_and_reviewing.md`](./running_and_reviewing.md) — this runbook just orchestrates it for
> the workshop. If a submit errors on a missing var, that doc's Prerequisites table is the checklist.

**Sanity-check a config offline first** (resolves the config + estimates the fan-out
`series × models × folds`, touches no GCP):

```bash
python -m scale_forecasting.main --config configs/explode_100k.json --dry-run
```

**Submit all four** — three Spark methods + Ray. Each returns once submitted; drop `--no-wait` to
block until it lands, or keep it and watch them run in parallel from the Dataproc / Vertex consoles:

```bash
python -m scale_forecasting.submit     --config configs/explode_100k.json --engine explode
python -m scale_forecasting.submit     --config configs/multi_100k.json   --engine multi
python -m scale_forecasting.submit     --config configs/naive_100k.json   --engine naive
python -m scale_forecasting.ray_submit --config configs/ray_100k.json
```

| Config | Runtime | What it demonstrates |
|--------|---------|----------------------|
| `explode_100k.json` | Spark (`explode`) | Max parallelism — one task per `(series, model)` cell. The 100k workhorse. |
| `multi_100k.json` | Spark (`multi`) | One child `explode` batch per model family under one `run_id`. |
| `naive_100k.json` | Spark (`naive`) | The straggler anti-pattern — one task per series, models run sequentially. |
| `ray_100k.json` | Ray on Vertex | The Python-runtime path on a Ray cluster (CPU here; GPU is the NeuralProphet demo). |

**Confirm they landed and capture the `run_id`s.** The `run_id` is a deterministic digest of the
config, so the shipped configs always produce the **same** ids — but confirm via the registry rather
than assume. In [BigQuery Studio](https://console.cloud.google.com/bigquery) (or `bq query`):

```sql
SELECT run_id, created_at, status, spark_method, python_runtime, n_series, n_models
FROM `<project>.scale_forecasting.v_run_summary`
ORDER BY created_at DESC
LIMIT 25;
```

You want four `SUCCEEDED` rows — one `explode-100k-…`, one `multi-100k-…`, one `naive-100k-…`, one
`ray-100k-…`. Copy those four `run_id`s; Act 2's notebook 07 reads them.

> **Expectation-setter for the accuracy chart.** The 100k configs run with **backtest off** (that's
> the fleet-scale default), so `07_scale_review`'s accuracy-parity panel (`mean_wape`) is **all-NULL**
> for these runs — the notebook says so explicitly. **The scaling / efficiency panels are the 100k
> showpiece** (wall-clock, provisioning overhead, DCU). Accuracy parity across engines is demonstrated
> at small scale in notebook **03** (where backtest is on). Say this out loud before opening 07 and
> it's a feature, not a surprise.

---

## Act 2 — The guided notebook tour (Colab Enterprise, live)

Every notebook has a one-click **Run in Colab Enterprise** badge in its first cell. Clicking it
imports the notebook; then **pick the runtime template it names** and **Run all** — the deployed
templates already carry the `SF_*` identity in their env, so there is **no environment cell to fill
in**. The per-notebook template mapping and the one-click mechanics are documented in
[`docs/notebook_runtimes.md`](./notebook_runtimes.md).

Run them **in this order** — each builds on the story of the last:

| # | Notebook | Template | What you show | Scale |
|---|----------|----------|---------------|-------|
| 1 | [`model_playground`](../notebooks/model_playground.ipynb) | `sf-main` | Pick any registered model, fit it on a small panel — the one unit of work, no cluster. | sample |
| 2 | [`01_spark_via_connect`](../notebooks/01_spark_via_connect.ipynb) | `sf-spark-connect` | The Spark UDF fan-out (`applyInPandas`, one task per cell) over a live Dataproc Connect endpoint. | 100 |
| 3 | [`02_bigquery_native`](../notebooks/02_bigquery_native.ipynb) | `sf-main` | The BigQuery-native track — `ARIMA_PLUS` + `TimesFM` as pure SQL, no cluster. | 100 |
| 4 | [`03_combo_and_ensemble`](../notebooks/03_combo_and_ensemble.ipynb) | `sf-main` | One config mixing a Spark model **and** the BQ natives under one `run_id`, with ensembles on — **and the accuracy-parity leaderboard** (backtest is on here). | 10 |
| 5 | [`04_ray_on_vertex`](../notebooks/04_ray_on_vertex.ipynb) | `sf-main` | The Python models on a Ray-on-Vertex cluster ∥ the BQ natives — job submission from any authenticated client via the PSC-I attachment. | demo |
| 6 | [`05_spark_naive`](../notebooks/05_spark_naive.ipynb) | `sf-main` | The `naive` straggler anti-pattern made visible — a run drags on its slowest *series*. | 100 |
| 7 | [`06_spark_multi`](../notebooks/06_spark_multi.ipynb) | `sf-main` | `multi` fanning one child `explode` batch per model family, all under one `run_id`. | 100 |
| 8 | [`07_scale_review`](../notebooks/07_scale_review.ipynb) | `sf-main` | **The payoff** — the cross-approach comparison over your **Act 1** 100k runs. Runs nothing; reads the registry views. | reads 100k |

**Notebook 07 is the one notebook you configure.** Its first code cell has an edit-me `RUN_IDS` block:

```python
# === Parameters — edit me ===============================================
RUN_IDS = {
    "spark-explode": "explode-100k-…",   # ← paste your four Act 1 run_ids
    "spark-multi":   "multi-100k-…",
    "spark-naive":   "naive-100k-…",
    "ray":           "ray-100k-…",
}
# ========================================================================
```

Paste the four `run_id`s you captured in Act 1 (set any you skipped to `None`), then **Run all**. It
renders wall-clock + provisioning overhead from `v_run_summary` and the per-model panel from
`v_model_leaderboard`, side by side across the four approaches. (Because the configs are shipped
unchanged, the deterministic default ids in the cell often already match your runs — but paste yours to
be sure.)

> **Runtime note for notebook 01.** The `sf-spark-connect` template is Python **3.12** for the
> interactive Connect path. If you open 01 on a 3.11 runtime it falls back to the identical
> **remote-batch** engine — same result, just not the interactive Connect session. Either is a fine
> demo; the header names the intended template.

---

## Cost + timing at a glance

- **Act 1 (four 100k runs):** each Spark method is a Dataproc Serverless batch (single-digit dollars,
  minutes); the Ray run stands up and tears down a fixed-size cluster. Run once before the workshop and
  the results persist in the registry — Act 2 just reads them.
- **Act 2 (notebooks):** the demo-scale notebooks (100 series or fewer) are cents. `07_scale_review`
  runs no compute — it only queries views.
- **Reset when you're done:** the destructive teardown is documented in
  [`docs/running_and_reviewing.md`](./running_and_reviewing.md#resetting-the-environment-destructive),
  and `terraform destroy` removes the project's infrastructure.

## See also

- [`terraform/README.md`](../terraform/README.md) — deploy the platform (the step before this runbook).
- [`docs/running_and_reviewing.md`](./running_and_reviewing.md) — the submit/watch/review mechanics this
  runbook orchestrates, plus re-ensembling and teardown.
- [`docs/notebook_runtimes.md`](./notebook_runtimes.md) — per-notebook template mapping, the one-click
  open path, and the headless acceptance harness.
- [`docs/deploying_on_gcp.md`](./deploying_on_gcp.md#human-users-running-jobs--notebooks) — the human
  IAM roles for running jobs and notebooks.
