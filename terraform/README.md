# Terraform — deploy scale-forecasting into a GCP project

Two stages, run in order. **Stage 1 (bootstrap)** creates the project and the bucket that
holds Terraform's own state. **Stage 2 (main)** creates everything else, storing its state
in that bucket. This split resolves the chicken-and-egg problem (you can't keep state in a
bucket that doesn't exist yet).

> **Cost:** the base deployment is effectively free at rest — empty buckets, an empty
> dataset, service accounts. The one real cost, **Composer 3** (~$300–400/mo), is **off by
> default** (`create_composer = false`) and only starts when you turn it on. See
> [`main/modules/composer/main.tf`](./main/modules/composer/main.tf) for the start/run/stop
> lifecycle.

## Prerequisites

- `terraform >= 1.5`, `gcloud` authenticated: `gcloud auth application-default login`.
- A **billing account id** (`gcloud billing accounts list`) and your **org or folder id**.
- Permission to create projects under that org/folder.

## Stage 1 — bootstrap (once)

```bash
cd terraform/bootstrap
cp terraform.tfvars.example terraform.tfvars   # edit: project_id, billing_account, org_id
terraform init
terraform apply                                # creates the project + state bucket
terraform output backend_config                # note the bucket name for stage 2
```

## Stage 2 — main

```bash
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
| `create_composer` | `false` | (already off) turn **on** for scheduled DAG runs |
| `run_seed` | `false` | (already off) turn **on** to submit the seed batch (real spend) |
| `create_project` (bootstrap) | `true` | your org pre-creates projects |

## Modules (one capability each)

`apis` · `iam` · `storage` · `bigquery` · `budget` · `composer` · `container` · `network` ·
`seed` — mirroring the Python side's one-file-one-capability rule. Read each module's header
comment for what and why. `container` owns the Artifact Registry repo for the shared Spark/Ray
runtime image (built by `docker/cloudbuild.yaml`); `network` provides the VPC + subnet (Private
Google Access) that serverless compute requires; `seed` submits the gated Dataproc Serverless
batch that materializes the example dataset (BUILD B0.4).

**Table schemas live in Python, not here.** The five registry/data tables are defined once in
`src/scale_forecasting/registry/ddl.py` and created by `registry.bq.ensure_tables()` at run
time. Terraform owns the *containers* (dataset, BigLake connection, bucket grants); the app
owns the *tables* — so there's a single source of truth for the DDL.
