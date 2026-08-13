# Workshop runbook — demo every runtime, then review at scale

**You've deployed the platform** (see [`terraform/README.md`](../terraform/README.md)) and the smoke
forecast came back `SUCCEEDED`. This is the guided path from there to a full demo: submit one run per
approach at **100k series**, then walk the notebooks in **Colab Enterprise** — ending on
`07_scale_review`, which renders the cross-approach comparison over the runs you just made.

Everything here runs **in the browser** — [Cloud Shell](https://console.cloud.google.com/?cloudshell=true)
for the Act 1 submits, [Colab Enterprise](https://console.cloud.google.com/vertex-ai/colab) for the
Act 4 notebooks. No local machine or SDK install is assumed.

> **Prep before, present live.** Acts 1–3 populate the run history and (optionally) pre-render the
> tour — real compute (~minutes to hours), best done **before** the workshop. Act 4 is the live tour;
> `07_scale_review` needs Act 1's runs to have data to show. The four acts run **in order**, and the
> section numbers are strictly sequential (1 → 2 → 3 → 4).

---

## Who can run this — IAM roles

A workshop presenter and any attendees you invite need their own grants (the Terraform grants roles to
the *service accounts*, not to people). The complete, copy-paste role set — for both the **Cloud Shell
submitter** (Act 1) and the **Colab Enterprise notebook user** (Act 4) — lives in one place:

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
# Clone (skip if you still have the deploy clone) and install with uv — the repo is uv-managed
# (pyproject.toml + uv.lock). The `submit` extra is the thin client set: it pulls the Dataproc +
# Ray submit clients (both fan-out paths) but NOT pyspark — batch submission never imports it, and
# pyspark's ~300MB of JARs overflow Cloud Shell's home quota. (pyspark is only for notebook 01's
# interactive Spark Connect path, which runs in Colab Enterprise — Act 4 — not here.)
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

**Submit all four** — three Spark methods + Ray. Each command **blocks until its batch finishes**
(that's how it stamps the wall-clock / DCU telemetry `07_scale_review` charts), so the four run
**one after another** — budget for the sum, not the max.

> **Run them under `tmux` so a lost tab doesn't sever the wait.** Cloud Shell disconnects if the
> browser tab sleeps or the network blips, which SIGHUPs a foreground process — killing the current
> `wait` and every command queued behind it in the line. `tmux` keeps the session alive server-side
> so you can reattach:
>
> ```bash
> tmux new -s runs        # start (reattach later with: tmux attach -t runs)
> ```
>
> Then, inside tmux, submit the four. The Dataproc batches themselves run server-side and survive a
> disconnect regardless — but only a live `wait` stamps their telemetry, so `tmux` is what protects
> the `07` charts.
>
> **A new tmux session is a new shell** — it does *not* inherit the `SF_*` exports from the tab you
> ran them in. Re-paste the `SF_*` block above **inside** tmux before submitting, or the submit fails
> with `missing required environment variable SF_PROJECT_ID`. (Already inside tmux? `tmux new` will
> warn `sessions should be nested with care` and no-op — just re-export and submit in the session
> you're in.)

```bash
uv run python -m scale_forecasting.submit     --config configs/explode_100k.json --engine explode
uv run python -m scale_forecasting.submit     --config configs/multi_100k.json   --engine multi
uv run python -m scale_forecasting.submit     --config configs/naive_100k.json   --engine naive
uv run python -m scale_forecasting.ray_submit --config configs/ray_100k.json
```

> **On the wait timeout.** A 100k batch runs longer than the client's old 15-minute default wait, so
> the submitter now blocks up to **2 h** (`--wait-timeout <seconds>` to change it). If you ever *do*
> see a client-side `TimeoutError`, the **batch is unaffected** — it keeps running server-side; only
> the local wait gave up. Check its true state with
> `gcloud dataproc batches list --project $PROJECT --region $REGION` or the `v_run_summary` query
> below, and re-submit only what didn't land.

| Config | Runtime | What it demonstrates |
|--------|---------|----------------------|
| `explode_100k.json` | Spark (`explode`) | Max parallelism — one task per `(series, model)` cell. The 100k workhorse. |
| `multi_100k.json` | Spark (`multi`) | One child `explode` batch per model family under one `run_id`. |
| `naive_100k.json` | Spark (`naive`) | The straggler anti-pattern — one task per series, models run sequentially. |
| `ray_100k.json` | Ray on Vertex | The Python-runtime path on a Ray cluster (CPU here; GPU is the NeuralProphet demo). |

**Watch them make progress (healthy vs. stuck).** A 100k batch runs for 1–2 h, so "is it working or
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

Each run's target is **`n_series` × `n_models` cells** — the 100k configs are 100,000 × 4 models =
**~400,000 cells** — so `cells_written / 400000` is a rough % complete. Reading the numbers:

- **Counts climbing between two checks = healthy.** Leave the batches alone; they're serverless and
  finish server-side regardless of your shell. **Don't kill them to "restart"** — you'd discard the
  cells already written and pay to recompute (re-submitting the same config reuses the same
  deterministic `run_id` and dedupes-on-read, so nothing already done is wasted).
- **`naive` fills slowest and unevenly** — that's its anti-pattern signature (one task per *series*,
  models run sequentially, so it drags on the slowest series). Expected, not a problem.
- **Genuinely stuck** looks like: count **flat** across several minutes **and** the batch's Spark UI
  (batch detail page → *View Spark UI*) shows no task progress. Only then investigate.

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
SELECT run_id, created_at, status, spark_method, python_runtime, n_series, n_models
FROM `gcp-scale-forecasting.scale_forecasting.v_run_summary`
ORDER BY created_at DESC
LIMIT 25;
```

You want four `SUCCEEDED` rows — one `explode-100k-…`, one `multi-100k-…`, one `naive-100k-…`, one
`ray-100k-…`. Copy those four `run_id`s; Act 4's notebook 07 reads them.

> **Expectation-setter for the accuracy chart.** The 100k configs run with **backtest off** (that's
> the fleet-scale default), so `07_scale_review`'s accuracy-parity panel (`mean_wape`) is **all-NULL**
> for these runs — the notebook says so explicitly. **The scaling / efficiency panels are the 100k
> showpiece** (wall-clock, provisioning overhead, DCU). Accuracy parity across engines is demonstrated
> at small scale in notebook **03** (where backtest is on). Say this out loud before opening 07 and
> it's a feature, not a surprise.

---

## Act 2 — The long Ray run on a persistent VM (when the run outlasts Cloud Shell)

Some runs are **too long to babysit from Cloud Shell** — the full-suite Ray config
`configs/all_methods_100k_full.json` (100k series × 7 models, **backtest on**, `persist_models`,
NeuralProphet on T4 GPUs) runs for **hours**. Cloud Shell isn't built for that: it idles out after
~20 min of inactivity and hard-caps a session at ~12 h, and when the tab closes it SIGHUPs your
process. The Ray *cluster* survives server-side, but the **orchestrator** (`main.py`) is what polls
the jobs and **finalizes the run-registry header** (`COMPLETED`/`FAILED`) — kill it and the header is
stranded `RUNNING` even though the work finished.

The fix is a **persistent GCE VM** that owns the orchestrator process. You drive it entirely from
Cloud Shell — create it, SSH into it, launch under `tmux`, and detach. The VM keeps running when Cloud
Shell disconnects; reattach any time to watch. The VM is a thin **driver**, not compute — the fan-out
happens on the Ray cluster — so a small `e2-standard-4` is plenty.

> **Everything below runs in [Cloud Shell](https://console.cloud.google.com/?cloudshell=true).** Set
> the same `PROJECT` / `REGION` you used in Act 1.

**1. One-time: allow IAP to SSH the VM.** The deploy's VPC has **no external IPs** (Cloud NAT for
egress only; many orgs deny external IPs by policy), so you reach the VM over
[IAP TCP forwarding](https://cloud.google.com/iap/docs/using-tcp-forwarding), which tunnels SSH from
IAP's range `35.235.240.0/20`. Add the firewall rule that permits it (idempotent — skip if it exists):

```bash
PROJECT=gcp-scale-forecasting   # ← your project_id
REGION=us-central1              # your deploy region
ZONE=$REGION-a

gcloud compute firewall-rules create scale-forecasting-allow-iap-ssh \
  --project "$PROJECT" --network scale-forecasting \
  --direction INGRESS --action ALLOW --rules tcp:22 \
  --source-ranges 35.235.240.0/20 2>/dev/null \
  || echo "rule already exists — continuing"
```

You also need `roles/iap.tunnelResourceAccessor` on the project (the deploy grants the runner SA the
data roles; this is a *human* role, like the Act 1 submitter roles — see
[Human users](./deploying_on_gcp.md#human-users-running-jobs--notebooks)).

**2. Create the VM** — on the deploy's subnet, **no external IP**, running **as the runner SA** so its
ADC already carries every data/compute permission a run needs (no keys). `cloud-platform` scope lets
the SA's roles do the gating:

```bash
gcloud compute instances create sf-runner \
  --project "$PROJECT" --zone "$ZONE" \
  --machine-type e2-standard-4 \
  --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 50GB \
  --network scale-forecasting --subnet scale-forecasting-compute \
  --no-address \
  --service-account "scale-forecasting-runner@$PROJECT.iam.gserviceaccount.com" \
  --scopes cloud-platform
```

**3. SSH in from Cloud Shell** (the `--tunnel-through-iap` flag is what makes a no-external-IP VM
reachable):

```bash
gcloud compute ssh sf-runner --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap
```

**4. On the VM: install `git` + `uv`, clone, sync.** (First login may prompt to generate an SSH key —
accept.) The minimal Debian image ships **none of `git`, `tmux` (step 8 needs it), or `uv`**, and the
`uv` installer drops its binary in `~/.local/bin` which isn't on `PATH` until you source its env — so
install all three, put `uv` on `PATH`, then clone and sync:

```bash
# the minimal image ships neither git nor tmux (step 8 needs tmux) — install both:
sudo apt-get update -qq && sudo apt-get install -y -qq git tmux

# install uv and put it on PATH for THIS shell (its installer prints this same `source` line):
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version                    # confirm uv is on PATH

# clone + sync the thin client (Dataproc + Ray submit clients, no pyspark):
cd ~ && git clone https://github.com/statmike/scale-forecasting.git
cd ~/scale-forecasting
uv sync --extra submit
```

> **Want to run Terraform from this VM too** (e.g. to update the Colab runtime template)? Two extra
> one-time steps, because the VM is authenticated as the *runner* SA, which has only data/compute
> roles — **not** the admin permissions Terraform needs:
>
> ```bash
> # a) install Terraform (not preinstalled on the VM):
> sudo apt-get update -qq && sudo apt-get install -y -qq unzip
> TF_VERSION=1.9.8; mkdir -p ~/bin && cd ~/bin
> curl -fsSL -o tf.zip "https://releases.hashicorp.com/terraform/${TF_VERSION}/terraform_${TF_VERSION}_linux_amd64.zip"
> unzip -o tf.zip && rm tf.zip
> grep -qxF 'export PATH="$HOME/bin:$PATH"' ~/.bashrc || echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
> export PATH="$HOME/bin:$PATH"; cd ~/scale-forecasting
>
> # b) authenticate Terraform AS YOU (the provider reads ADC, not the attached SA):
> gcloud auth application-default login
> ```
>
> Then follow [`terraform/README.md`](../terraform/README.md) for `init` / `plan` / `apply`. When the
> apply is done, **revoke the human ADC** so subsequent runs on this VM revert to the runner SA:
> `gcloud auth application-default revoke`.

**5. Wire the `SF_*` identity** — the same deterministic block as Act 1 (the VM runs as the runner SA,
so ADC is already in place; these vars just tell the orchestrator *what* to talk to):

```bash
PROJECT=gcp-scale-forecasting   # ← your project_id (re-set on the VM — a new shell)
REGION=us-central1
export SF_PROJECT_ID="$PROJECT"
export SF_REGION="$REGION"
export SF_DATASET_ID="scale_forecasting"
export SF_CONNECTION="$PROJECT.$REGION.sf-iceberg"
export SF_WAREHOUSE_URI="gs://$PROJECT-warehouse/warehouse"
export SF_CODE_BUCKET="$PROJECT-code"
export SF_COMPUTE_SA="scale-forecasting-compute@$PROJECT.iam.gserviceaccount.com"
export SF_CONTAINER_IMAGE="$REGION-docker.pkg.dev/$PROJECT/scale-forecasting/spark-runtime:latest"
export SF_SUBNETWORK_URI="https://www.googleapis.com/compute/v1/projects/$PROJECT/regions/$REGION/subnetworks/scale-forecasting-compute"
```

**6. Clear the OUTPUT tables only** (optional — do this to reset the registry before a clean run).
The canonical reset — the output-only `TRUNCATE` that **keeps** the seeded source data
(`source_series_iceberg`), so you don't pay to reseed 100k series — lives in the operations runbook:
➡️ **[operations.md §2a — Output-only reset](./operations.md#2a-output-only-reset-keep-the-seed--the-usual-rework)**.

> This clears **all** runs, including Act 1's four — run it only for a clean slate; otherwise skip (the
> orchestrator dedupes-on-read against the deterministic `run_id`, so re-running never double-counts).
> Do **not** touch `source_series_iceberg`. To drop the seed too (schema change), see
> [operations.md §2b](./operations.md#2b-full-reset-drop-everything-incl-the-seed--needs-a-reseed-after).

**7. Preflight offline** (resolves the config + estimates the fan-out, touches no GCP):

```bash
uv run python -m scale_forecasting.main --config configs/all_methods_100k_full.json --dry-run
```

**8. Launch under `tmux` and detach.** `tmux` keeps the orchestrator alive on the VM when you close
Cloud Shell — so it lives to poll the Ray jobs and **finalize the header**. `tee` mirrors the log to a
file you can tail later:

```bash
tmux new -s ray100k \
  'uv run python -m scale_forecasting.main --config configs/all_methods_100k_full.json 2>&1 | tee ~/ray100k.log'
```

Detach with **`Ctrl-b` then `d`** (two keystrokes: hold Ctrl + tap `b`, release both, then tap `d`) —
the run keeps going. Now you can safely close Cloud Shell.

**9. Check on it — WITHOUT `tmux attach`.** Prefer watching the **log file** or **BigQuery**, not the
live tmux viewer. `tmux attach` opens a full-screen console that's easy to get stuck in (`Ctrl-b d` is
the only clean exit, and stray keys like `:q`/`Ctrl-Z` just jam it) — and you never need it, because
`tee` already mirrors everything to `~/ray100k.log`. SSH back in from any new Cloud Shell and tail the
log (exit the tail with a plain **`Ctrl-C`** — it stops the *viewer*, not the run):

```bash
gcloud compute ssh sf-runner --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap
tmux ls                          # confirm the ray100k session is still alive
tail -f ~/ray100k.log            # follow progress; Ctrl-C to stop watching (run keeps going)
```

The best monitor needs no VM at all — **watch it from BigQuery in the browser**, exactly as in Act 1.
Re-run this every 5–10 min; climbing counts = healthy (early zeros are normal — the Ray cluster is
still provisioning, which is what the "uploading package" log line means):

```sql
-- swap gcp-scale-forecasting for your project_id; the run_id is the config's deterministic digest
SELECT run_id, COUNT(*) AS cells_written, MAX(created_at) AS latest_write
FROM `gcp-scale-forecasting.scale_forecasting.forecast_metadata`
WHERE run_id = 'all-methods-100k-full-036327523e0a'
GROUP BY run_id;
```

`v_run_summary` flips the header to `COMPLETED` when the orchestrator finalizes it. Because this config
has **backtest on**, its accuracy columns (`mean_wape`, …) *are* populated — unlike the throughput-only
Act 1 runs.

> **If the tmux viewer traps you** (you attached and can't get out): don't restart Cloud Shell — from
> a **second** SSH session run `tmux detach-client -t ray100k` to free the stuck viewer from outside,
> or just close the tab. The run is in tmux on the VM and survives disconnects, closed tabs, and Cloud
> Shell restarts regardless — none of that can kill it.

**10. Clean up the VM when the run lands** (it bills while it exists, ~cents/hr, but tidy is tidy):

```bash
gcloud compute instances delete sf-runner --project "$PROJECT" --zone "$ZONE" --quiet
```

---

## Act 3 — Pre-render the notebook tour (optional, the night before)

Act 4 is a **live** tour — but the expensive notebooks (`04_ray_on_vertex` stands up a Ray cluster;
`05`/`06` submit Dataproc batches) take too long to run in front of an audience. This step
**pre-executes every notebook headless** so tomorrow you walk **already-rendered** notebooks (outputs
baked in) from the Colab Enterprise **Executions** menu — and run only the cheap, fast ones (`07`,
`01`, `02`, `03`, playground) live if you want.

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

> **`--tier` picks how much to pre-render** (same cost tiers as the acceptance harness):
> `smoke` = the 4 BQ/local notebooks (cheap); `batch` = those + the 3 Dataproc ones (`01`,`05`,`06`);
> `full` = **all 8**, adding `04_ray_on_vertex`. `full` fires the whole tour at once — that's the
> **same total spend as one full acceptance run** (8 Colab runtimes + the Dataproc batches + a live
> Ray cluster), just concurrent. Pre-workshop prep, not free — but it's what pre-renders everything.

> **Run this *after* Act 1's four 100k runs land.** `07_scale_review` reads those runs, so pre-render
> it only once they're `SUCCEEDED` — otherwise its rendered output shows missing data. (`07` reads
> its `RUN_IDS` from the shipped deterministic defaults, which match the unchanged configs — if you
> overrode a config, edit `07`'s `RUN_IDS` cell before this step or run `07` live instead.)

**Tomorrow:** open the [Executions menu](https://console.cloud.google.com/vertex-ai/colab/execution-jobs),
find each `sf-demo-…` job, and click into the pre-rendered notebook. Run `07` and `01` (and any other
cheap one) **live** in the console on `sf-main` (every notebook uses that one template — see the Act 4
table) for the interactive moments, and lean on the pre-rendered set for the expensive `04`/`05`/`06`.

---

## Act 4 — The guided notebook tour (Colab Enterprise, live)

Every notebook has a one-click **Run in Colab Enterprise** badge in its first cell. Clicking it
imports the notebook; then **pick the runtime template it names** and **Run all** — the deployed
templates already carry the `SF_*` identity in their env, so there is **no environment cell to fill
in**. The per-notebook template mapping and the one-click mechanics are documented in
[`docs/notebook_runtimes.md`](./notebook_runtimes.md).

Run them **in this order** — each builds on the story of the last:

| # | Notebook | Template | What you show | Scale |
|---|----------|----------|---------------|-------|
| 1 | [`model_playground`](../notebooks/model_playground.ipynb) | `sf-main` | Pick any registered model, fit it on a small panel — the one unit of work, no cluster. | sample |
| 2 | [`01_spark_via_connect`](../notebooks/01_spark_via_connect.ipynb) | `sf-main` | The Spark UDF fan-out (`applyInPandas`, one task per cell) over a live Dataproc Connect endpoint. | 100 |
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

> **Runtime note for notebook 01.** It runs interactively on `sf-main` like every other notebook: the
> Spark Connect session pins **Dataproc runtime 2.3** (the Connect floor; py3.11 workers, matching the
> 3.11 kernel) and attaches the project container image for deps. A **remote-batch** escape hatch is
> documented at the bottom of the notebook — the identical engine on-cluster, same result — if you'd
> rather not open a live Connect session.

---

## Cost + timing at a glance

- **Act 1 (four 100k runs):** each Spark method is a Dataproc Serverless batch (single-digit dollars,
  minutes); the Ray run stands up and tears down a fixed-size cluster. Run once before the workshop and
  the results persist in the registry — Act 4 just reads them.
- **Act 4 (notebooks):** the demo-scale notebooks (100 series or fewer) are cents. `07_scale_review`
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
