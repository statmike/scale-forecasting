# seed — submit the Dataproc Serverless Spark batch that seeds source_series (BUILD B0.4).
#
# This runs the platform's own core pattern (parallel Spark generation + high-throughput BigQuery
# write) to materialize the shipped example dataset, and doubles as the first Spark scale smoke.
# It is GATED (run_seed) and BLOCKING: google_dataproc_batch waits for the batch to reach a
# terminal state on apply, so `terraform apply` == submit + wait for SUCCEEDED/FAILED.
#
# CODE DELIVERY: the batch loads the scale_forecasting package at RUNTIME via python_file_uris, not
# from the container image. Terraform zips src/ (archive_file) every apply and uploads it; the zip's
# md5 is folded into batch_id, so any code change yields a new (immutable) batch that runs the new
# code. The runtime image carries only locked deps (docker/requirements.txt) — no package, no stale
# code. The thin launcher (seed_entry.py) is the gs:// main file; the zip supplies what it imports.
#
# SYNC / RECOVERY: google_dataproc_batch blocks until terminal, but its client-side wait has a
# timeout (we set it to 60m below; the provider default is only 10m and it bit us on the 100k run —
# the batch SUCCEEDED at ~11m wall but the apply errored and left the resource OUT of state). If an
# apply ever errors AFTER the batch was submitted, the batch state in GCP is the source of truth:
# `gcloud dataproc batches describe <id>` to check, then `terraform import
# module.seed.google_dataproc_batch.seed[0] projects/<proj>/locations/<region>/batches/<id>` to
# reconcile state. (Benign perpetual diffs on budget + terraform_labels are cosmetic — see NOTES.)
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
  # Repo src/ (contains only the scale_forecasting/ package, so it sits at the zip root and is
  # importable once the zip is on sys.path via python_file_uris).
  src_dir  = "${path.module}/../../../../src"
  zip_path = "${path.module}/.terraform-tmp/scale_forecasting.zip"

  # Batch ids: lowercase alnum + hyphens, 4-63 chars. Unique per (label, count) AND per code
  # version — the 8-char code hash means a source change yields a NEW immutable batch that runs the
  # new code (batches are never updated in place). "sf-seed-full-100000-1a2b3c4d" = 28 chars < 63.
  code_hash = var.create ? substr(data.archive_file.package[0].output_md5, 0, 8) : ""
  batch_id  = "sf-seed-${var.run_label}-${var.num_series}-${local.code_hash}"

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

# Zip the scale_forecasting package from src/ at apply time. output_md5 changes iff the source
# changes, driving both the uploaded object name and the batch_id (so new code => new batch).
data "archive_file" "package" {
  count       = var.create ? 1 : 0
  type        = "zip"
  source_dir  = local.src_dir
  output_path = local.zip_path
}

# The package zip the batch loads at runtime via python_file_uris (NOT baked into the image). The
# md5 in the name makes each code version a distinct object (no in-place overwrite races).
resource "google_storage_bucket_object" "package" {
  count  = var.create ? 1 : 0
  bucket = var.code_bucket
  name   = "seed/scale_forecasting-${local.code_hash}.zip"
  source = data.archive_file.package[0].output_path
}

# The launcher must be a gs:// file for main_python_file_uri; it just imports main() from the
# package supplied by python_file_uris (the zip above), so the batch runs current code.
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
    # The package zip: put on sys.path so seed_entry.py's `import scale_forecasting` resolves to
    # this apply's code, not anything in the image.
    python_file_uris = ["gs://${var.code_bucket}/${google_storage_bucket_object.package[0].name}"]
    args = concat([
      "--n-series", tostring(var.num_series),
      "--master-seed", tostring(var.master_seed),
      "--write-method", var.write_method,
    ], local.infra_args)
  }

  # The provider's default create-wait is 10m; a large batch (100k took ~11m wall) blows past it and
  # errors the apply even though the batch succeeds — leaving the resource out of state. 60m covers
  # the 100k seed and future B2 runs with headroom.
  timeouts {
    create = "60m"
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
