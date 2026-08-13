# Operations runbook — rework an already-deployed environment

The [deploy](../terraform/README.md) and [demo](./workshop.md) runbooks assume a **fresh** setup.
This doc is the **rework** path: you've already deployed, and now you need to **re-apply Terraform**
(pick up a code/infra change), **reset the BigQuery output tables** (clean slate for a re-run), or
**re-run** a config. Everything here runs from **[Cloud Shell](https://console.cloud.google.com/?cloudshell=true)** —
short, interactive tasks that fit a thin client. Long, multi-hour runs belong on a persistent VM
instead ([workshop.md Act 2](./workshop.md#act-2--the-long-ray-run-on-a-persistent-vm-when-the-run-outlasts-cloud-shell)).

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
> [workshop.md Act 2](./workshop.md#act-2--the-long-ray-run-on-a-persistent-vm-when-the-run-outlasts-cloud-shell).

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

## 2. Reset the BigQuery tables

There are **two** resets. Pick deliberately — they differ in whether the 100k **source seed**
survives.

### 2a. Output-only reset (keep the seed) — the usual rework

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
> a visibly empty registry (e.g. a clean demo). **Never** truncate `source_series_iceberg` here —
> that's the seed, and dropping it forces a full reseed (2b).

### 2b. Full reset (drop everything, incl. the seed) — needs a reseed after

`reset.py` drops **all six tables** (the four outputs **and both source variants**) plus the two
analyst views, for a clean `ensure_tables` recreate. Use it only when the schema itself changed or you
want a truly empty dataset — **it forces a 100k reseed afterward** (via the Terraform `seed` module or
`data_gen.seed_spark`). It reads the `SF_*` env for its target and is a dry run without `--yes`:

```bash
# needs the SF_* identity (section 3) so it targets the right deployment:
uv run python -m scale_forecasting.reset          # DRY RUN — prints what would drop, touches nothing
uv run python -m scale_forecasting.reset --yes    # actually drops all six tables + views
```

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
uv run python -m scale_forecasting.submit --config configs/explode_demo.json --engine explode
```

> **Long run?** Anything that runs for hours (the full 100k suite, GPU NeuralProphet) must go on the
> persistent VM, not here — Cloud Shell will drop the orchestrator mid-run and strand the header.
> See [workshop.md Act 2](./workshop.md#act-2--the-long-ray-run-on-a-persistent-vm-when-the-run-outlasts-cloud-shell).

---

## See also

- [`terraform/README.md`](../terraform/README.md) — the **fresh deploy** (zero to deployed).
- [`docs/workshop.md`](./workshop.md) — the **fresh demo** (populate runs → tour notebooks); Act 2 is
  the durable-VM path for long runs.
- [`docs/running_and_reviewing.md`](./running_and_reviewing.md) — submit / watch / review mechanics and
  the `SF_*` reference.
- [`docs/deploying_on_gcp.md`](./deploying_on_gcp.md#human-users-running-jobs--notebooks) — the human
  IAM roles.
