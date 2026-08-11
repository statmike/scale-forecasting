# Main-stage inputs. Grouped: identity, region, the BYO toggles (DESIGN §13.0), and the
# names Terraform assigns to what it creates. Sensible defaults everywhere except project_id.

# --- identity & location -------------------------------------------------------

variable "project_id" {
  description = "Target project id (created by the bootstrap stage, or your existing project)."
  type        = string
}

variable "region" {
  description = "Region for buckets, BigQuery, Dataproc/Ray, and Composer."
  type        = string
  default     = "us-central1"
}

# --- greenfield / BYO toggles (DESIGN §13.0) -----------------------------------
# Default true = Terraform owns everything (the 5-minute quickstart). A locked-down org
# flips off what its admins already manage and passes existing resources by variable.

variable "enable_apis" {
  description = "Enable the required Google APIs. false if an admin already enabled them."
  type        = bool
  default     = true
}

variable "create_service_accounts" {
  description = "Create the runner / compute SAs. false to bring your own (set *_email vars)."
  type        = bool
  default     = true
}

variable "create_network" {
  description = <<-EOT
    Create the VPC + subnet + firewall for serverless compute. Default TRUE (greenfield). Set
    FALSE if your org already manages a network, and pass an existing subnet via subnetwork_uri
    (it must have Private Google Access + an internal-ingress firewall rule).
  EOT
  type        = bool
  default     = true
}

variable "subnetwork_uri" {
  description = "Existing subnet self-link for serverless batches; used only when create_network = false."
  type        = string
  default     = null
}

variable "create_composer" {
  description = <<-EOT
    Create the Composer 3 (Airflow) environment. Default FALSE — Composer is the only
    at-rest cost (~$300-400/mo), and many use cases run the pipeline ad-hoc (local/notebook)
    and never need a scheduler. Turn it on to develop scheduled DAG orchestration (the run DAG
    is not shipped yet — see modules/composer); turn it off again with `terraform apply` to stop
    the meter. See modules/composer for the documented start / run / stop lifecycle.
  EOT
  type        = bool
  default     = false
}

# --- Colab Enterprise runtime templates ----------------------------------------
# Blueprints for the VM a Colab runtime runs on — one per Python version the notebooks need. Free at
# rest (a template costs nothing until a runtime starts, and runtimes idle-shutdown), so ON by
# default. See modules/colab and docs/notebook_runtimes.md.

variable "create_colab_templates" {
  description = <<-EOT
    Create the two Colab Enterprise runtime templates (sf-main: Python 3.11 / [ray]; sf-spark-connect:
    Python 3.12 / [spark]). Default TRUE — templates are FREE at rest (no VM until someone starts a
    runtime, and runtimes idle-shutdown), so shipping them costs nothing and makes the notebooks
    runnable on Colab Enterprise out of the box. Set FALSE for a CLI-only / locked-down deploy.
  EOT
  type        = bool
  default     = true
}

variable "colab_attach_network" {
  description = <<-EOT
    Attach the project VPC + subnet to the Colab runtimes (VPC-private path). Default TRUE — this
    stack is greenfield (create_network = true) and has no `default` network, so a public runtime
    would 404 looking for one; attaching the custom VPC gives egress with NO external IP via the
    Cloud NAT + Private Google Access from modules/network (also compatible with a
    compute.vmExternalIpAccess = DENY org policy). Set FALSE only in a brownfield project that has a
    usable `default` network and permits external IPs.
  EOT
  type        = bool
  default     = true
}

variable "colab_main_release_name" {
  description = <<-EOT
    Colab image release for sf-main. py311 matches the project pin (Ray parity + Dataproc packed-venv)
    and is applied via a REST PATCH (the provider can't set it — #25217). Bump this to re-pin before
    a Python version reaches end-of-availability (templates auto-upgrade to Latest otherwise).
  EOT
  type        = string
  default     = "py311"
}

variable "colab_spark_release_name" {
  description = <<-EOT
    Colab image release for sf-spark-connect. py312 matches Dataproc 3.0 Connect workers and is pinned
    EXPLICITLY via a REST PATCH (the provider can't set it — #25217) so it can't drift to 3.13 when
    Colab advances Latest, which would re-break NB01 interactive with PYTHON_VERSION_MISMATCH. Bump to
    re-pin before py312 reaches end-of-availability.
  EOT
  type        = string
  default     = "py312"
}

# --- naming (what Terraform creates) -------------------------------------------

variable "dataset_id" {
  description = "BigQuery dataset for the registry + example data (DESIGN §8.1)."
  type        = string
  default     = "scale_forecasting"
}

variable "runner_sa_email" {
  description = "Existing runner SA email; used only when create_service_accounts = false."
  type        = string
  default     = null
}

variable "compute_sa_email" {
  description = "Existing compute SA email; used only when create_service_accounts = false."
  type        = string
  default     = null
}

# --- seed job (BUILD B0.4) -----------------------------------------------------
# The Dataproc Serverless batch that materializes the example dataset. ON by default (run_seed =
# true) so a fresh deploy ships with data; real (small) cloud spend — 100k measured at ~$0.11-0.15,
# ~8.5 min compute. Content-addressed, so it runs once, not per-apply. See modules/seed for the
# smoke → review → full lifecycle.

variable "run_seed" {
  description = <<-EOT
    Submit the Dataproc Serverless seed batch that materializes the example dataset (real cloud
    spend). Default TRUE — a fresh deploy comes with the shipped 100k-series dataset in both source
    tables, ready to forecast against immediately (the "solution-in-a-box" promise). The batch is
    content-addressed (batch_id embeds series count + code hash), so it runs on the FIRST apply and
    does NOT re-run on later applies unless you change seed_num_series / seed_run_label / the seed
    code — reseeds are deliberate, not per-apply. google_dataproc_batch blocks until the batch is
    terminal, so the first `terraform apply` submits and waits (~8.5 min compute, ~$0.15 measured at
    100k). Set FALSE to skip the example data (e.g. you'll bring your own source table); set
    seed_num_series = 100 first if you want to smoke-test cost/runtime before the full 100k.
  EOT
  type        = bool
  default     = true
}

variable "seed_num_series" {
  description = "Series to generate: 100 for the smoke, 100000 for the shipped example dataset."
  type        = number
  default     = 100000
}

variable "seed_master_seed" {
  description = "Master RNG seed — identical shipped data on every deploy (DESIGN §13.1)."
  type        = number
  default     = 20260726
}

variable "seed_write_method" {
  description = "spark-bigquery write path: direct (Storage Write API) or indirect (GCS→BQ load)."
  type        = string
  default     = "direct"
}

variable "seed_run_label" {
  description = "Short label distinguishing seed batches (e.g. \"smoke\", \"full\"); part of batch_id."
  type        = string
  default     = "full"
}

variable "seed_variant" {
  description = "Source storage format(s) to seed: iceberg, native, or both (one panel, D19)."
  type        = string
  default     = "both"
}

variable "seed_image_tag" {
  description = "Tag of the runtime image the seed batch runs (built by docker/cloudbuild.yaml)."
  type        = string
  default     = "latest"
}

# --- default-on smoke forecast -------------------------------------------------

variable "run_smoke" {
  description = <<-EOT
    Submit a tiny, TOLERANT smoke forecast on apply, so the first `terraform apply` also proves the
    platform forecasts: a few fast Python models on Dataproc Serverless in parallel with `arima_plus`
    in BigQuery, under one shared run_id. Default TRUE. It needs the seeded data + the runtime image,
    so it is effectively gated on run_seed (module.smoke.create = run_smoke && run_seed) and ordered
    after the seed/container/network. Unlike the seed, the smoke is NON-BLOCKING: it is submitted with
    `gcloud dataproc batches submit --async` via a null_resource with on_failure = continue, so a
    forecast failure NEVER fails the apply — inspect the surfaced smoke_batch_id / smoke_describe_hint
    to see its outcome. Content-addressed (batch id embeds code + config hash), so it runs on the
    first apply and re-submits only when the code or smoke config changes. Set FALSE to skip it.
  EOT
  type        = bool
  default     = true
}

# --- runtime image build -------------------------------------------------------

variable "build_image" {
  description = <<-EOT
    Build the shared Spark/Ray runtime image with Cloud Build on apply, so a single `terraform
    apply` creates the Artifact Registry repo AND fills it — the seed batch and forecast engines
    all pull this image, so with build_image = false and no pre-built image, a fresh deploy has
    nothing to run. Default TRUE. The build is content-addressed on docker/ (Dockerfile +
    requirements.txt): it runs on the first apply and rebuilds ONLY when those deps change, never
    on a source-code edit (code ships at runtime via python_file_uris). Requires the `gcloud` CLI
    on the machine running Terraform. If enable_apis = false, an admin must have already enabled
    cloudbuild.googleapis.com. Set FALSE to skip when you build the image yourself (CI / air-gapped
    registry) — then push it to the repo tag the seed/engines consume before running compute.
  EOT
  type        = bool
  default     = true
}

# --- budget --------------------------------------------------------------------

variable "billing_account" {
  description = "Billing account id — required to create the budget + alert."
  type        = string
}

variable "budget_amount_usd" {
  description = "Monthly budget in USD; alert thresholds fire at 50/90/100%."
  type        = number
  default     = 200
}
