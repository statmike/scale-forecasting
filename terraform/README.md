# Terraform — deploy scale-forecasting into a GCP project

Two stages, run in order. **Stage 1 (bootstrap)** creates the project and the bucket that
holds Terraform's own state. **Stage 2 (main)** creates everything else, storing its state
in that bucket. This split resolves the chicken-and-egg problem (you can't keep state in a
bucket that doesn't exist yet).

> **Cost:** the infrastructure is effectively free at rest — empty buckets, an empty dataset,
> service accounts, network plumbing. Three things cost money:
> - **The runtime-image build** (`build_image = true`, **on by default**) runs Cloud Build once on
>   the first apply to build + push the shared Spark/Ray image (a few minutes, cents of Cloud Build
>   time). It's content-addressed on `docker/` and rebuilds only when the Dockerfile or
>   `requirements.txt` change — never on a source-code edit. Set `build_image = false` if you build
>   the image yourself (CI / air-gapped registry).
> - **The example-data seed** (`run_seed = true`, **on by default**) runs a one-time Dataproc
>   Serverless batch on the first apply (~8.5 min, ~$0.15 measured at the 100k default) to materialize
>   the shipped dataset. It's content-addressed, so it does **not** re-run on later applies unless you
>   change the series count / label / seed code. Set `run_seed = false` to skip it, or
>   `seed_num_series = 100` to smoke-test cost first.
> - **Composer 3** (~$300–400/mo) is **off by default** (`create_composer = false`) and only starts
>   when you turn it on. See [`main/modules/composer/main.tf`](./main/modules/composer/main.tf) for
>   the start/run/stop lifecycle.

## Prerequisites

### Tools

- The `gcloud` CLI (pre-installed in [Cloud Shell](https://cloud.google.com/shell), the recommended
  place to run this) and `terraform >= 1.5`.
- **Terraform is no longer pre-installed in Cloud Shell** (Google removed the bundled CLI after
  HashiCorp's BSL license change). Install it into your Cloud Shell **home directory** so it persists
  across sessions (Cloud Shell's home is durable; `/usr/*` is reset). Step 0a below does this; the APT
  route from HashiCorp's docs also works but is wiped when the session VM recycles.
- The main stage shells out to `gcloud builds submit` to build the runtime image (unless
  `build_image = false`), so `gcloud` must be authenticated for **both** the CLI *and* ADC (the
  runbook does both).

### Identifiers you need up front

| You need | Get it with |
|----------|-------------|
| **Billing account id** (`XXXXXX-XXXXXX-XXXXXX`) | `gcloud billing accounts list` |
| **Org id** *(or a folder id)* | `gcloud organizations list` (org) / `gcloud resource-manager folders list --organization=<ORG_ID>` (folder) |
| A **globally-unique project id** to create | you choose (e.g. `my-scale-forecasting`) |

### Permissions **you** (the operator) must hold

These are the roles the *human running Terraform* needs — distinct from the two service accounts the
deployment creates for the workload (those are in
[`docs/deploying_on_gcp.md`](../docs/deploying_on_gcp.md#permissions-why-each-is-granted-and-who-uses-it)).

**At the org (or folder) level** — needed by **bootstrap**, because it creates a project and links
billing:

| Role | Why |
|------|-----|
| `roles/resourcemanager.projectCreator` | Create the new project (`google_project`). Grant on the org or the parent folder. |
| `roles/billing.user` | Link the project to your billing account. Granted on the **billing account** (`gcloud billing accounts add-iam-policy-binding`), not the org. |

> Skipping bootstrap? If your org **pre-creates projects** (`create_project = false`), you don't need
> `projectCreator` — but you still need `billing.user` on the billing account unless an admin has
> already linked billing, plus `roles/storage.admin` on the pre-made project to create the state bucket.

**At the project level** — needed by **main**, which enables APIs, creates SAs + role bindings,
buckets, network, dataset, and runs Cloud Build. The simplest correct grant is **`roles/owner`** on
the project bootstrap just created. If your org forbids `owner`, the equivalent least-privilege set is:
`roles/serviceusage.serviceUsageAdmin`, `roles/iam.serviceAccountAdmin`,
`roles/resourcemanager.projectIamAdmin`, `roles/iam.roleAdmin` (three custom roles),
`roles/storage.admin`, `roles/bigquery.admin`, `roles/compute.networkAdmin`,
`roles/dataproc.admin`, `roles/aiplatform.admin`, `roles/cloudbuild.builds.editor`,
`roles/artifactregistry.admin`, and `roles/billingbudgets.budgetsEditor`.

---

## Zero-to-deployed from Cloud Shell (copy-paste runbook)

This is the fully prescriptive path — from an empty Cloud Shell to a deployed, seeded platform.
[Open Cloud Shell](https://console.cloud.google.com/?cloudshell=true) (the terminal icon, top-right of
the Cloud Console), then run these blocks **in order**. Cloud Shell already carries your identity and
has `terraform` + `gcloud` installed.

### 0a. Install Terraform, auth, clone (and confirm your roles)

Cloud Shell no longer ships Terraform, so install it into your home directory (persists across
sessions), authenticate ADC (the Terraform provider reads it), then grab the repo. Before going
further, confirm you hold the operator roles from
[Permissions](#permissions-you-the-operator-must-hold) above — `projectCreator` on the org/folder and
`billing.user` on the billing account.

```bash
# Install a recent Terraform into ~/bin (durable in Cloud Shell; survives session recycles):
TF_VERSION=1.9.8
mkdir -p ~/bin && cd ~/bin
curl -fsSL -o terraform.zip "https://releases.hashicorp.com/terraform/${TF_VERSION}/terraform_${TF_VERSION}_linux_amd64.zip"
unzip -o terraform.zip && rm terraform.zip
grep -qxF 'export PATH="$HOME/bin:$PATH"' ~/.bashrc || echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/bin:$PATH"
terraform version          # confirm it's on PATH

# ADC for the Terraform provider (Cloud Shell has your gcloud identity, but the provider reads ADC):
gcloud auth application-default login

# Clone the repo:
cd ~
git clone https://github.com/statmike/scale-forecasting.git
cd scale-forecasting
```

### 0b. Discover ids + enable the Cloud Billing API on your active project

```bash
gcloud billing accounts list          # copy the ACCOUNT_ID (XXXXXX-XXXXXX-XXXXXX)
gcloud organizations list             # copy your ORG_ID  (a number)

# Bootstrap's billing-account permission check routes through your ADC quota project (the project
# gcloud is currently set to). That project needs the Cloud Billing API enabled, or the project-create
# step fails with "Cloud Billing API has not been used in project <X> ... SERVICE_DISABLED":
gcloud config get-value project                              # <-- your active/quota project
gcloud services enable cloudbilling.googleapis.com          # enable it there (wait ~1-2 min to propagate)
```

### 1a. Bootstrap — prepare the vars (edit before you apply)

```bash
cd terraform/bootstrap
cp terraform.tfvars.example terraform.tfvars
```

Now **edit `terraform.tfvars`** — set `project_id` (globally unique), `billing_account`, and `org_id`
(or `folder_id`) from the ids you copied in 0b. In Cloud Shell, open the built-in editor:

```bash
cloudshell edit terraform.tfvars
```

### 1b. Bootstrap — create the project + the Terraform state bucket (run once)

```bash
terraform init
terraform apply          # review the plan, type `yes`. Creates the project + <project_id>-tfstate bucket.
```

### 2a. Main — prepare the vars (edit before you apply)

```bash
cd ../main
cp terraform.tfvars.example terraform.tfvars
```

Now **edit `terraform.tfvars`** — set `project_id` + `billing_account` to the **same** values you used
in bootstrap:

```bash
cloudshell edit terraform.tfvars
```

### 2b. Main — build everything else

```bash
# Point Terraform's state at the bucket bootstrap just made (name is "<project_id>-tfstate"):
terraform init -backend-config="bucket=$(cd ../bootstrap && terraform output -raw state_bucket)"

terraform plan           # read it — nothing is created until apply
terraform apply          # type `yes`. Builds the image (Cloud Build) + seeds 100k series; ~15-20 min total.
```

> The first apply **blocks** while Cloud Build builds the runtime image and the Dataproc seed batch
> runs (~8.5 min at 100k). That's expected — it's doing real work. To smoke cheaply first, set
> `seed_num_series = 100` in `terraform.tfvars` before `apply`.

### 3. Verify — read the outputs the app consumes (from `terraform/main`)

```bash
terraform output                              # dataset, connection, warehouse, buckets, SA emails
terraform output -raw project_id
```

Those outputs feed the run config and the `SF_*` environment the notebooks resolve. The Colab
Enterprise runtime templates (`sf-main`, `sf-spark-connect`) are created on by default and already
carry that `SF_*` identity in their env — so from here you can open any notebook in Colab Enterprise
and **Run all** with no environment cell. See
[`docs/notebook_runtimes.md`](../docs/notebook_runtimes.md) for the per-notebook template mapping and
the headless acceptance harness that verifies every notebook runs green.

---

## The commands, condensed

If you don't want the narrated runbook, this is the whole sequence:

```bash
# Stage 1 — bootstrap (once)
cd terraform/bootstrap
cp terraform.tfvars.example terraform.tfvars   # edit: project_id, billing_account, org_id
terraform init
terraform apply                                # creates the project + state bucket
terraform output state_bucket                  # the bucket name for stage 2

# Stage 2 — main
cd ../main
cp terraform.tfvars.example terraform.tfvars   # edit: project_id, billing_account
terraform init -backend-config="bucket=<project_id>-tfstate"
terraform plan                                 # review — nothing is created until apply
terraform apply
```

Outputs (`terraform output`) give you the dataset, the Iceberg connection ref, the warehouse
URI, the code/artifacts buckets, and the two service-account emails — these feed the run
config and the seed job.

## The BYO toggles

Defaults create everything (the 5-minute quickstart). To fit a locked-down org, flip these
in `terraform.tfvars` and pass existing resources by variable:

| Variable | Default | Turn off when… |
|----------|---------|----------------|
| `enable_apis` | `true` | an admin already enabled the APIs |
| `create_service_accounts` | `true` | you bring your own SAs (`runner_sa_email`, `compute_sa_email`) |
| `create_network` | `true` | your org already has a network — pass an existing subnet (`subnetwork_uri`) with Private Google Access + internal-ingress |
| `create_composer` | `false` | (already off) turn **on** for scheduled DAG runs |
| `build_image` | `true` | (already **on**) turn **off** when you build/push the runtime image yourself (CI / air-gapped) |
| `run_seed` | `true` | (already **on**) turn **off** to skip the example dataset (bring your own source table) |
| `create_project` (bootstrap) | `true` | your org pre-creates projects |

## Modules (one capability each)

`apis` · `iam` · `storage` · `bigquery` · `budget` · `composer` · `container` · `network` ·
`seed` — mirroring the Python side's one-file-one-capability rule. Read each module's header
comment for what and why. `container` owns the Artifact Registry repo for the shared Spark/Ray
runtime image **and builds + pushes it on apply** (via `docker/cloudbuild.yaml`, `build_image`
toggle), so one apply fills the repo the seed/engines pull from; `network` provides the VPC + subnet (Private
Google Access) that serverless compute requires; `seed` submits the gated Dataproc Serverless
batch that materializes the example dataset (BUILD B0.4).

**Table schemas live in Python, not here.** The five registry/data tables are defined once in
`src/scale_forecasting/registry/ddl.py` and created by `registry.bq.ensure_tables()` at run
time. Terraform owns the *containers* (dataset, BigLake connection, bucket grants); the app
owns the *tables* — so there's a single source of truth for the DDL.
