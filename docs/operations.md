# Operations runbook — rework an already-deployed environment

The [deploy](https://github.com/statmike/scale-forecasting/blob/main/terraform/README.md) and [demo](./workshop.md) runbooks assume a **fresh** setup.
This doc is the **rework** path: you've already deployed, and now you need to **re-apply Terraform**
(pick up a code/infra change), **clear the BigQuery registry** (clean slate for a re-run), or
**re-run** a config. Everything here runs from **[Cloud Shell](https://console.cloud.google.com/?cloudshell=true)** —
short, interactive tasks that fit a thin client. Long, multi-hour runs belong on a persistent VM
instead ([§4 — Long runs on a persistent VM](#4-long-runs-on-a-persistent-vm-when-a-run-outlasts-cloud-shell)).

---

## 0. Disk hygiene FIRST (Cloud Shell has ~5 GB of home)

Cloud Shell's **home directory is only ~5 GB**, and it fills fast: `uv`'s cache + `.venv` (pyspark's
JARs alone are ~300 MB) + Terraform's cached providers can exhaust it, and when it's full **Terraform
dies with a `Bus error`** (a full-disk symptom, not a Terraform bug). Check and reclaim before you
start:

```bash
df -h $HOME                                    # want >1 GB Avail. (Ignore the "/" overlay % — that's
                                               # the system disk, not your home; Terraform writes to home.)
du -sh ~/* ~/.[!.]* 2>/dev/null | sort -h | tail -12   # what's eating it
```

If `Avail` is low, reclaim — all of these rebuild on demand, so they're safe to delete:

```bash
uv cache clean                                 # uv's global download cache (usually the biggest win, ~1 GB)
rm -rf ~/scale-forecasting/.venv               # the venv (pyspark JARs); recreate with uv sync --extra submit
rm -rf ~/.terraform.d/plugin-cache             # cached Terraform providers
rm -rf ~/scale-forecasting/terraform/*/.terraform   # per-stage provider copies (terraform init rebuilds)
df -h $HOME                                     # confirm the space came back
```

**Two rules that keep it from re-filling:**

1. **Terraform needs no Python** — don't run `uv sync` for a Terraform-only task. `terraform` +
   `gcloud` are the only tools.
2. **If you do need Python, use the thin client:** `uv sync --extra submit` (Dataproc + Ray submit
   clients, **no pyspark**). A bare `uv sync` pulls pyspark's ~300 MB of JARs and refills the disk —
   only notebook 01's interactive Connect path needs those, and that runs in Colab, not here.

> **A multi-hour re-run? Don't do it in Cloud Shell.** Cloud Shell idles out (~20 min) and hard-caps
> a session (~12 h); a dropped tab SIGHUPs the orchestrator that finalizes the run-registry header.
> For anything long (e.g. the full-suite Ray run), use the persistent VM in
> [§4 below](#4-long-runs-on-a-persistent-vm-when-a-run-outlasts-cloud-shell).

---

## 1. Re-apply Terraform (pick up an infra/template change)

Re-applying is **safe and idempotent** — the state lives in the `<project_id>-tfstate` bucket, so a
re-apply keeps everything already created and proposes **only real drift**. It never rebuilds from
scratch.

**`terraform.tfvars` is git-ignored**, so a fresh clone (or a recycled Cloud Shell disk) won't have
it — recreate it before you plan, or Terraform interactively prompts for `project_id` /
`billing_account` on every command (and Ctrl-C won't cleanly exit the prompt):

```bash
cd ~/scale-forecasting/terraform/main
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars          # set project_id + billing_account to the SAME values you deployed with

# Route the provider's API calls through the deployed project (not your Cloud Shell default):
export GOOGLE_CLOUD_QUOTA_PROJECT=<project_id>

# Re-attach to the existing remote state, then see the drift:
terraform init -backend-config="bucket=<project_id>-tfstate"
terraform plan                 # READ THIS — a rework should be a small, expected diff
terraform apply                # type `yes` only after the plan matches what you intended to change
```

> **Read the plan before `apply`.** For a rework you want to see a **small** diff — the specific
> resource you changed (e.g. the Colab runtime template), `0 to destroy` unless you meant to destroy
> something. If the plan proposes tearing down data resources (buckets, the dataset, SAs) you did
> **not** intend to touch, stop and check `GOOGLE_CLOUD_QUOTA_PROJECT` and the `-backend-config`
> bucket — a wrong quota project or a missing state backend makes Terraform think it's deploying
> greenfield.

The image build (`build_image`) and the data seed (`run_seed`) are **content-addressed** — they do
**not** re-run on a plain re-apply unless you changed the Dockerfile/`requirements.txt` or the seed
parameters. So a template-only rework is fast and cheap.

---

## 2. Clear the BigQuery registry

Three ways to get a clean registry, narrowest first. **None of them touch the 100k source seed** —
that panel has its own lifetime and no product verb reads or writes it (rebuilding it is a Spark
job via the Terraform `seed` module or `data_gen.seed_spark`).

### 2a. Truncate the output tables (keep the seed) — the usual rework

Truncates only the **four run-output tables** and **keeps `source_series_iceberg`** (the 100k seeded
panel every run reads), so you get a clean registry **without** paying to reseed 100k series. This is
the right reset between demo runs. Run in
[BigQuery Studio](https://console.cloud.google.com/bigquery) (or `bq query --use_legacy_sql=false`):

```sql
-- swap gcp-scale-forecasting for your project_id if you deployed elsewhere
TRUNCATE TABLE `gcp-scale-forecasting.scale_forecasting.run_registry`;
TRUNCATE TABLE `gcp-scale-forecasting.scale_forecasting.forecast_metadata`;
TRUNCATE TABLE `gcp-scale-forecasting.scale_forecasting.forecast_predictions`;
TRUNCATE TABLE `gcp-scale-forecasting.scale_forecasting.backtest_oof`;
```

> You rarely even need this — the orchestrator writes under a deterministic `run_id` and
> **dedupes-on-read**, so re-running the same config never double-counts. Truncate only when you want
> a visibly empty registry (e.g. a clean demo). Note what this *doesn't* do: it leaves every run's
> GCS artifacts behind with nothing left to attribute them to. Prefer §2b when you know which runs
> you want gone. **Never** truncate `source_series_iceberg` here — that's the seed, and dropping it
> forces a full reseed.

### 2b. Per-run teardown — the scoped, complete option

`registry.ops` deletes a named run from **every** tier — GCS artifacts, its BQML `sf_model_*`
objects, and its registry rows — in that order, so nothing is stranded. Preview by default, and it
refuses a run that is still in flight. This is the right tool whenever you know which runs you want
gone; §2a and §2c are the blunt instruments:

```bash
# needs the SF_* identity (section 3) so it targets the right deployment:
uv run python -m scale_forecasting.registry.ops doctor              # counts, stuck runs, orphans
uv run python -m scale_forecasting.registry.ops drop-run RUN_ID     # preview — deletes nothing
uv run python -m scale_forecasting.registry.ops drop-run RUN_ID --yes
uv run python -m scale_forecasting.registry.ops sweep-orphans --yes # artifacts nothing indexes now
```

A header stuck at `RUNNING` because its driver died is a *different* problem from a run you want
gone — `close-runs` finalizes it from its own job rows without deleting anything, and is the verb to
reach for before you consider dropping the run to clear the status:

```bash
uv run python -m scale_forecasting.registry.ops close-runs          # preview every stuck header
uv run python -m scale_forecasting.registry.ops close-runs --yes
```

The full verb set (`init` / `doctor` / `close-runs` / `drop-run` / `sweep-orphans` / `snapshot` /
`export`) and the
matching `Registry` SDK class are documented in
[running_and_reviewing.md §6](./running_and_reviewing.md#6-managing-the-registry).

### 2c. Discard the registry entirely — `bq rm`, and there is no verb for it

**You probably don't need this.** Schema updates are automatic: every write path calls
`ensure_tables`, which issues an idempotent `ALTER TABLE … ADD COLUMN IF NOT EXISTS` per table, so a
repo update that *adds* columns lands on your existing tables with no data loss and no action from
you. Dropping is for the rare non-additive change (a retyped or removed column) or for retiring a
registry you're done with.

**Delete the GCS artifacts first — the rows are the only thing that identifies them.** Once the
tables are gone, nothing says which object belonged to which run. So either scope it with §2b
(`drop-run` the runs you care about) or, if the registry is going away completely, empty it and then
sweep — an empty `run_registry` makes *every* prefix an orphan by definition, which is exactly what
you want here:

```bash
uv run python -m scale_forecasting.registry.ops sweep-orphans        # preview: objects + bytes
uv run python -m scale_forecasting.registry.ops sweep-orphans --yes
```

Then remove the BigQuery objects. **There is no product verb for this, on purpose** — a full
teardown is a `bq` one-liner nobody needs us to wrap, and wrapping it invites the accident:

```bash
# The registry has its own dataset (you set SF_REGISTRY_DATASET_ID) — delete the dataset:
bq rm -r -f <project>:<registry_dataset>

# The registry shares SF_DATASET_ID with the source panel — drop the eight objects by name,
# because deleting the dataset would take the seeded source panel with it:
DS=<project>:<dataset>
for v in v_run_summary v_run_jobs v_model_leaderboard; do bq rm -f -t "$DS.$v"; done
for t in run_registry run_jobs forecast_metadata forecast_predictions backtest_oof; do
  bq rm -f -t "$DS.$t"
done
```

Drop the views before the tables — they read the tables, so the reverse order trips a dependency.
Recreate everything with `python -m scale_forecasting.registry.ops init` (or just submit a run;
`ensure_tables` runs on the way in).

> **Want a disposable registry instead?** Give it its own dataset via `SF_REGISTRY_DATASET_ID` and
> `registry.ops init` it. Then discarding it is one `bq rm -r -f` that cannot reach the source panel
> or another registry — the artifact root carries the registry key
> (`<warehouse>/artifacts/<project>/<registry-dataset>/<run_id>/`), so its GCS side is independently
> scoped too.

---

## 3. Re-run a config (short runs only)

For a quick re-submit from Cloud Shell, wire the `SF_*` identity (deterministic from project + region)
and submit. These values and the submit/watch/review mechanics are the same as the demo path — see
[running_and_reviewing.md](./running_and_reviewing.md) for the full reference.

```bash
cd ~/scale-forecasting
uv sync --extra submit          # thin client — NO pyspark (keeps the disk under quota)

# --- set these two, then paste the rest ---
PROJECT=gcp-scale-forecasting   # ← your project_id
REGION=us-central1              # your deploy region
# ------------------------------------------
export SF_PROJECT_ID="$PROJECT"
export SF_REGION="$REGION"
export SF_DATASET_ID="scale_forecasting"
export SF_CONNECTION="$PROJECT.$REGION.sf-iceberg"
export SF_WAREHOUSE_URI="gs://$PROJECT-warehouse/warehouse"
export SF_CODE_BUCKET="$PROJECT-code"
export SF_COMPUTE_SA="scale-forecasting-compute@$PROJECT.iam.gserviceaccount.com"
export SF_CONTAINER_IMAGE="$REGION-docker.pkg.dev/$PROJECT/scale-forecasting/spark-runtime:latest"
export SF_SUBNETWORK_URI="https://www.googleapis.com/compute/v1/projects/$PROJECT/regions/$REGION/subnetworks/scale-forecasting-compute"

uv run python -m scale_forecasting.main --config configs/explode_demo.json --dry-run   # preflight
uv run python -m scale_forecasting.submit --config configs/explode_demo.json
```

> **Long run?** Anything that runs for hours (the full 100k suite, GPU NeuralProphet) must go on the
> persistent VM, not here — Cloud Shell will drop the orchestrator mid-run and strand the header.
> See [§4 — Long runs on a persistent VM](#4-long-runs-on-a-persistent-vm-when-a-run-outlasts-cloud-shell).

---

## 4. Long runs on a persistent VM (when a run outlasts Cloud Shell)

Some runs are **too long to babysit from Cloud Shell** — the full-suite Ray config
`configs/all_families_10k_full.json` (10k series × 7 models × 2 folds = 140k cells, **backtest on**, `persist_models`,
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
> the same `PROJECT` / `REGION` you deployed with.

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
data roles; this is a *human* role, like the submitter roles — see
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
> Then follow [`terraform/README.md`](https://github.com/statmike/scale-forecasting/blob/main/terraform/README.md) for `init` / `plan` / `apply`. When the
> apply is done, **revoke the human ADC** so subsequent runs on this VM revert to the runner SA:
> `gcloud auth application-default revoke`.

**5. Wire the `SF_*` identity** — re-paste the deterministic block from
[§3 above](#3-re-run-a-config-short-runs-only) (re-set `PROJECT`/`REGION` first — the VM is a new
shell). The VM runs as the runner SA, so ADC is already in place; these vars just tell the
orchestrator *what* to talk to.

**6. Clear the OUTPUT tables only** (optional — do this to reset the registry before a clean run).
The canonical reset — the output-only `TRUNCATE` that **keeps** the seeded source data
(`source_series_iceberg`), so you don't pay to reseed 100k series — is
[§2a above](#2a-truncate-the-output-tables-keep-the-seed--the-usual-rework).

> This clears **all** runs — run it only for a clean slate; otherwise skip (the orchestrator
> dedupes-on-read against the deterministic `run_id`, so re-running never double-counts). Do **not**
> touch `source_series_iceberg`. To remove specific runs completely (artifacts and models included),
> use [§2b above](#2b-per-run-teardown--the-scoped-complete-option) instead.

**7. Preflight offline** (resolves the config + estimates the fan-out, touches no GCP):

```bash
uv run python -m scale_forecasting.main --config configs/all_families_10k_full.json --dry-run
```

**8. Launch under `tmux` and detach.** `tmux` keeps the orchestrator alive on the VM when you close
Cloud Shell — so it lives to poll the Ray jobs and **finalize the header**. `tee` mirrors the log to a
file you can tail later:

```bash
tmux new -s ray100k \
  'uv run python -m scale_forecasting.main --config configs/all_families_10k_full.json 2>&1 | tee ~/ray100k.log'
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

The best monitor needs no VM at all — **watch it from BigQuery in the browser** (the `forecast_metadata`
row-count query from the demo runbook). Re-run it every 5–10 min; climbing counts = healthy (early
zeros are normal — the Ray cluster is still provisioning, which is what the "uploading package" log
line means):

```sql
-- swap gcp-scale-forecasting for your project_id; the run_id is the config's deterministic digest
SELECT run_id, COUNT(*) AS cells_written, MAX(created_at) AS latest_write
FROM `gcp-scale-forecasting.scale_forecasting.forecast_metadata`
WHERE run_id = 'all-methods-100k-full-036327523e0a'
GROUP BY run_id;
```

`v_run_summary` flips the header to `COMPLETED` when the orchestrator finalizes it. Because this config
has **backtest on**, its accuracy columns (`mean_wape`, …) *are* populated.

> **If the tmux viewer traps you** (you attached and can't get out): don't restart Cloud Shell — from
> a **second** SSH session run `tmux detach-client -t ray100k` to free the stuck viewer from outside,
> or just close the tab. The run is in tmux on the VM and survives disconnects, closed tabs, and Cloud
> Shell restarts regardless — none of that can kill it.

**10. Clean up the VM when the run lands** (it bills while it exists, ~cents/hr, but tidy is tidy):

```bash
gcloud compute instances delete sf-runner --project "$PROJECT" --zone "$ZONE" --quiet
```

---

## See also

- [`terraform/README.md`](https://github.com/statmike/scale-forecasting/blob/main/terraform/README.md) — the **fresh deploy** (zero to deployed).
- [`docs/workshop.md`](./workshop.md) — the **fresh demo** (populate runs → tour notebooks). The
  durable-VM path for long runs now lives here in [§4](#4-long-runs-on-a-persistent-vm-when-a-run-outlasts-cloud-shell).
- [`docs/running_and_reviewing.md`](./running_and_reviewing.md) — submit / watch / review mechanics and
  the `SF_*` reference.
- [`docs/deploying_on_gcp.md`](./deploying_on_gcp.md#human-users-running-jobs--notebooks) — the human
  IAM roles.
