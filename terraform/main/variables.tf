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
  description = "Create sf-runner / sf-compute SAs. false to bring your own (set *_email vars)."
  type        = bool
  default     = true
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
  description = "Existing sf-runner SA email; used only when create_service_accounts = false."
  type        = string
  default     = null
}

variable "compute_sa_email" {
  description = "Existing sf-compute SA email; used only when create_service_accounts = false."
  type        = string
  default     = null
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
