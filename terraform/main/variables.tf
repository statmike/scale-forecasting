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
    and never need a scheduler. Turn it on when you want scheduled DAG runs (Phase 7); turn
    it off again with `terraform apply` to stop the meter. See modules/composer for the
    documented start / run / stop lifecycle.
  EOT
  type        = bool
  default     = false
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
