# smoke — submit a tiny, DEFAULT-ON, NON-BLOCKING smoke forecast after the data is seeded (Arc B).
#
# The first `terraform apply` builds the runtime image (module.container) and seeds the example
# dataset (module.seed); this module adds the natural next proof: that the platform *forecasts*. It
# runs a small end-to-end forecast — a few fast Python models on Dataproc Serverless IN PARALLEL
# with `arima_plus` in BigQuery, all under ONE shared run_id (the single-run, two-engine showpiece,
# in miniature) — by calling `scale_forecasting.main.run(cfg, spark=session)` with the batch's own
# SparkSession injected (the injectable-session seam). That path runs the Spark engine in-process
# and the BigQuery engine inline, importing NEITHER google-cloud-dataproc NOR the [spark] extra —
# so this is an ordinary pyspark batch on the SAME runtime image the seed uses, no image change.
#
# CODE DELIVERY: identical to modules/seed — the batch loads the scale_forecasting package at
# RUNTIME via python_file_uris (a zip of src/, content-addressed by md5), launched by a thin gs://
# shim (smoke_entry.py). No package baked into the image; a code change yields a new immutable batch.
#
# TOLERANT / NON-BLOCKING (the key difference from modules/seed): the seed uses a
# google_dataproc_batch resource, which BLOCKS the apply and FAILS it if the batch fails. The smoke
# must not — a forecast hiccup should never fail infra provisioning. So it submits via a
# null_resource + local-exec `gcloud dataproc batches submit pyspark` with `on_failure = continue`
# (mirroring the container module's build null_resource). The apply proceeds regardless; the batch
# id is deterministic (content-addressed like seed) and surfaced as an output with a describe hint
# so the operator can inspect the smoke's outcome. `--async` returns as soon as the batch is
# accepted, so the apply doesn't wait on the forecast either.

variable "create" {
  description = "Submit the smoke forecast. false = no batch, no spend (default)."
  type        = bool
  default     = false
}

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

# --- the smoke forecast shape (small + fast on purpose) ---
variable "source_table" {
  description = "Source table the smoke forecasts against (seeded by module.seed)."
  type        = string
  default     = "source_series_native"
}

variable "series_limit" {
  description = "How many series to forecast — tiny, this is a proof not a benchmark."
  type        = number
  default     = 20
}

variable "horizon" {
  description = "Forecast horizon (steps)."
  type        = number
  default     = 14
}

variable "models" {
  description = "Models to run: fast Python models on Dataproc + arima_plus in BigQuery, one run_id."
  type        = list(string)
  default     = ["theta", "holtwinters", "arima_plus"]
}

variable "run_label" {
  description = "Short label distinguishing smoke batches; part of batch_id."
  type        = string
  default     = "smoke"
}

# --- infra the batch runs against (from other modules' outputs) ---
variable "code_bucket" {
  description = "Bucket the launcher (smoke_entry.py), package zip, and config JSON are uploaded to."
  type        = string
}

variable "container_image" {
  description = "Full runtime image path incl. tag (the same image the seed batch runs)."
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
  # importable once the zip is on sys.path via --py-files).
  src_dir  = "${path.module}/../../../../src"
  zip_path = "${path.module}/.terraform-tmp/scale_forecasting.zip"

  # The run config the smoke forecasts with. UNIVARIATE (no exog): fast Python models on Dataproc
  # in parallel with arima_plus in BigQuery, backtest OFF for speed (a forecast, not a benchmark),
  # holidays on so arima_plus has its calendar. run_name is fixed so the deterministic run_id is
  # stable across re-applies (idempotent: re-running the same config lands identical rows).
  smoke_config = jsonencode({
    run_name       = "terraform smoke ${var.run_label}"
    python_runtime = "spark"
    spark_method   = "explode"
    data = {
      source_table = var.source_table
      horizon      = var.horizon
      series_limit = var.series_limit
    }
    models   = var.models
    features = { holidays = ["US"] }
    backtest = { enabled = false }
  })

  # Batch id: lowercase alnum + hyphens, 4-63 chars. Content-addressed on the code hash AND the
  # config, so a code or config change yields a NEW immutable batch (batches are never updated in
  # place). "sf-smoke-smoke-1a2b3c4d" style, well under 63 chars.
  code_hash   = var.create ? substr(data.archive_file.package[0].output_md5, 0, 8) : ""
  config_hash = var.create ? substr(md5(local.smoke_config), 0, 8) : ""
  batch_id    = "sf-smoke-${var.run_label}-${local.code_hash}-${local.config_hash}"

  # Infra identity passed as JOB ARGS, not Spark env properties (Dataproc rejects driver-env);
  # smoke_run.main() exports these --sf-* args into os.environ so env-based Settings stays the one
  # G1 seam (parity with the seed).
  infra_args = [
    "--sf-project-id=${var.project_id}",
    "--sf-connection=${var.connection}",
    "--sf-warehouse-uri=${var.warehouse_uri}",
    "--sf-dataset-id=${var.dataset_id}",
    "--sf-region=${var.region}",
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

# The package zip the batch loads at runtime via --py-files (NOT baked into the image). The md5 in
# the name makes each code version a distinct object (no in-place overwrite races).
resource "google_storage_bucket_object" "package" {
  count  = var.create ? 1 : 0
  bucket = var.code_bucket
  name   = "smoke/scale_forecasting-${local.code_hash}.zip"
  source = data.archive_file.package[0].output_path
}

# The launcher must be a gs:// file for main_python_file_uri; it just imports main() from the
# package supplied by --py-files (the zip above), so the batch runs current code.
resource "google_storage_bucket_object" "launcher" {
  count  = var.create ? 1 : 0
  bucket = var.code_bucket
  name   = "smoke/smoke_entry.py"
  source = "${path.module}/smoke_entry.py"
}

# The run config JSON, staged to GCS — the smoke's reproducibility record (logged verbatim to
# run_registry.raw_config). Content-addressed name so a config change is a distinct object.
resource "google_storage_bucket_object" "config" {
  count   = var.create ? 1 : 0
  bucket  = var.code_bucket
  name    = "smoke/run_config-${local.config_hash}.json"
  content = local.smoke_config
}

# Submit the smoke batch TOLERANTLY. Unlike modules/seed's google_dataproc_batch (which blocks and
# fails the apply on batch failure), this is a null_resource whose local-exec runs `gcloud dataproc
# batches submit` with on_failure = continue and --async — so the apply neither waits on nor fails
# with the forecast. triggers is content-addressed on code + config, so it re-submits only when
# either changes (a new immutable batch id), never per-apply.
resource "null_resource" "smoke" {
  count = var.create ? 1 : 0

  triggers = {
    code_hash   = local.code_hash
    config_hash = local.config_hash
    batch_id    = local.batch_id
  }

  provisioner "local-exec" {
    on_failure = continue
    command = join(" ", concat([
      "gcloud dataproc batches submit pyspark",
      "gs://${var.code_bucket}/${google_storage_bucket_object.launcher[0].name}",
      "--py-files=gs://${var.code_bucket}/${google_storage_bucket_object.package[0].name}",
      "--project=${var.project_id}", # explicit — never the ambient ADC project (DESIGN §13.0)
      "--region=${var.region}",
      "--batch=${local.batch_id}",
      "--version=${var.runtime_version}",
      "--container-image=${var.container_image}",
      "--service-account=${var.compute_service_account}",
      "--subnet=${var.subnetwork_uri}",
      "--async", # return once accepted; don't block the apply on the forecast
      "--",      # everything after this is the job's argv
      "--config-uri=gs://${var.code_bucket}/${google_storage_bucket_object.config[0].name}",
    ], local.infra_args))
  }

  depends_on = [
    google_storage_bucket_object.package,
    google_storage_bucket_object.launcher,
    google_storage_bucket_object.config,
  ]
}

output "batch_id" {
  description = "Submitted smoke batch id (null when gated off). Content-addressed on code + config."
  value       = var.create ? local.batch_id : null
}

output "describe_hint" {
  description = "How to inspect the tolerant smoke batch's outcome (it doesn't fail the apply)."
  value = var.create ? join(" ", [
    "gcloud dataproc batches describe ${local.batch_id}",
    "--project ${var.project_id} --region ${var.region}",
  ]) : null
}
