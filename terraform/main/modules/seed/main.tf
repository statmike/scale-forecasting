# seed — submit the Dataproc Serverless Spark batch that seeds source_series (BUILD B0.4).
#
# This runs the platform's own core pattern (parallel Spark generation + high-throughput BigQuery
# write) to materialize the shipped example dataset, and doubles as the first Spark scale smoke.
# It is GATED (run_seed) and BLOCKING: google_dataproc_batch waits for the batch to reach a
# terminal state on apply, so `terraform apply` == submit + wait for SUCCEEDED/FAILED.
#
# ─── LIFECYCLE: smoke → review → full (real cloud spend; BUILD gate) ───────────────────────
#
#   SMOKE  (cents, minutes — verifies schema/dtypes/count/determinism AND surfaces real cost):
#     run_seed = true, seed_num_series = 100, seed_run_label = "smoke", then `terraform apply`.
#
#   REVIEW:
#     inspect the batch's real cost + runtime and query source_series before scaling up.
#
#   FULL   (the 100k dataset — the B0.4 deliverable):
#     seed_num_series = 100000, seed_run_label = "full", then `terraform apply`. The batch_id
#     embeds the label + series count, so this is a DISTINCT immutable batch (Terraform creates
#     the new one; batches are immutable and are not updated in place).
# ───────────────────────────────────────────────────────────────────────────────────────────
#
# Idempotency note: the seed job DELETEs source_series before writing (replace-on-reseed). Right
# after a `direct` (Storage Write API) run the rows buffer ~90 min and can't be DELETE-d; an
# immediate re-seed should use seed_write_method = "indirect" or wait out the buffer.

variable "create" {
  description = "Submit the seed batch. false = no batch, no spend (default)."
  type        = bool
  default     = false
}

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

# --- what to seed (the dataset shape) ---
variable "num_series" {
  description = "Number of series to generate. 100 for the smoke, 100000 for the full dataset."
  type        = number
  default     = 100000
}

variable "master_seed" {
  description = "Master RNG seed — identical data on every deploy (DESIGN §13.1 reproducibility)."
  type        = number
  default     = 20260726
}

variable "write_method" {
  description = "spark-bigquery write path: direct (Storage Write API) or indirect (GCS→load)."
  type        = string
  default     = "direct"

  validation {
    condition     = contains(["direct", "indirect"], var.write_method)
    error_message = "write_method must be \"direct\" or \"indirect\"."
  }
}

variable "run_label" {
  description = "Short label distinguishing batches (e.g. \"smoke\", \"full\"); part of batch_id."
  type        = string
  default     = "full"
}

# --- infra the batch runs against (from other modules' outputs) ---
variable "code_bucket" {
  description = "Bucket the launcher (seed_entry.py) is uploaded to and loaded from."
  type        = string
}

variable "container_image" {
  description = "Full runtime image path incl. tag (the container module's image + a tag)."
  type        = string
}

variable "compute_service_account" {
  description = "Runtime SA the batch runs as — scale-forecasting-compute."
  type        = string
}

variable "connection" {
  description = "BigLake connection ref (project.region.name) passed through as SF_CONNECTION."
  type        = string
}

variable "warehouse_uri" {
  description = "gs:// warehouse root passed through as SF_WAREHOUSE_URI."
  type        = string
}

variable "dataset_id" {
  description = "BigQuery dataset passed through as SF_DATASET_ID."
  type        = string
}

variable "runtime_version" {
  description = "Dataproc Serverless runtime version (pin to avoid surprise upgrades)."
  type        = string
  default     = "2.2"
}

variable "subnetwork_uri" {
  description = "Subnet the batch runs in (needs Private Google Access + internal-ingress fw)."
  type        = string
}

locals {
  # Batch ids: lowercase alnum + hyphens, 4-63 chars, unique per (label, count) so smoke and full
  # are distinct immutable batches.
  batch_id = "sf-seed-${var.run_label}-${var.num_series}"

  # The infra identity is passed as JOB ARGS, not Spark env properties. Dataproc Serverless
  # allowlists Spark property prefixes and rejects driver-env (spark.kubernetes.driverEnv.* →
  # "unsupported properties"), so args are the reliable delivery path to the driver — which is
  # where Settings.resolve() + ensure_tables + the write all run. seed_spark.main() exports these
  # --sf-* args into the environment so env-based resolution stays the single G1 seam.
  infra_args = [
    "--sf-project-id", var.project_id,
    "--sf-connection", var.connection,
    "--sf-warehouse-uri", var.warehouse_uri,
    "--sf-dataset-id", var.dataset_id,
    "--sf-region", var.region,
  ]
}

# The launcher must be a gs:// file for main_python_file_uri; the package itself is in the image.
resource "google_storage_bucket_object" "launcher" {
  count  = var.create ? 1 : 0
  bucket = var.code_bucket
  name   = "seed/seed_entry.py"
  source = "${path.module}/seed_entry.py"
}

resource "google_dataproc_batch" "seed" {
  count = var.create ? 1 : 0

  project  = var.project_id
  location = var.region
  batch_id = local.batch_id

  runtime_config {
    version         = var.runtime_version
    container_image = var.container_image
  }

  environment_config {
    execution_config {
      service_account = var.compute_service_account
      subnetwork_uri  = var.subnetwork_uri
    }
  }

  pyspark_batch {
    main_python_file_uri = "gs://${var.code_bucket}/${google_storage_bucket_object.launcher[0].name}"
    args = concat([
      "--n-series", tostring(var.num_series),
      "--master-seed", tostring(var.master_seed),
      "--write-method", var.write_method,
    ], local.infra_args)
  }
}

output "batch_id" {
  description = "Submitted seed batch id (null when gated off)."
  value       = var.create ? google_dataproc_batch.seed[0].batch_id : null
}

output "batch_state" {
  description = "Terminal state of the seed batch (null when gated off)."
  value       = var.create ? google_dataproc_batch.seed[0].state : null
}
