# iam — the two service accounts and their least-privilege roles (DESIGN §12).
#
# No keys, ever. These SAs are used via ADC + impersonation:
#   scale-forecasting-runner  — orchestration: read/write BQ, GCS, submit Dataproc/Ray jobs.
#                               Composer runs as it.
#   scale-forecasting-compute — attached to Dataproc/Ray workers: BQ data + GCS artifacts only.
#
# Names are spelled out in full (not abbreviated) so they read self-evidently in the IAM
# console — a reader sees the product they belong to without a decoder. The account_id limit
# is 30 chars; both fit (24 and 25).
#
# `create` = false lets you bring your own SAs (pass their emails in); then this module only
# resolves the emails through to outputs and grants nothing (your admin owns the grants).

variable "project_id" {
  type = string
}

variable "create" {
  description = "Create the SAs + grant roles. false = BYO (pass runner_email/compute_email)."
  type        = bool
  default     = true
}

variable "runner_email" {
  description = "Existing runner SA email when create = false."
  type        = string
  default     = null
}

variable "compute_email" {
  description = "Existing compute SA email when create = false."
  type        = string
  default     = null
}

# --- the accounts --------------------------------------------------------------

resource "google_service_account" "runner" {
  count        = var.create ? 1 : 0
  project      = var.project_id
  account_id   = "scale-forecasting-runner"
  display_name = "scale-forecasting orchestration (BQ/GCS/Dataproc/Ray submit)"
}

resource "google_service_account" "compute" {
  count        = var.create ? 1 : 0
  project      = var.project_id
  account_id   = "scale-forecasting-compute"
  display_name = "scale-forecasting workers (BQ data + GCS artifacts)"
}

locals {
  runner_email  = var.create ? google_service_account.runner[0].email : var.runner_email
  compute_email = var.create ? google_service_account.compute[0].email : var.compute_email

  # Least-privilege role sets. Kept as locals so the grants below stay a single readable loop.
  runner_roles = [
    "roles/bigquery.dataEditor",     # write registry rows + create tables in the dataset
    "roles/bigquery.jobUser",        # run queries / load jobs
    "roles/bigquery.connectionUser", # use the BigLake connection for Iceberg
    "roles/storage.objectAdmin",     # warehouse + artifacts + code buckets
    "roles/dataproc.editor",         # submit Dataproc Serverless batches
    "roles/aiplatform.user",         # submit Ray on Vertex jobs
  ]
  compute_roles = [
    "roles/bigquery.dataEditor", # read source_series, write results
    "roles/bigquery.jobUser",
    "roles/bigquery.connectionUser",
    "roles/storage.objectAdmin", # read/write model artifacts
  ]

  # Flatten (email, role) pairs into one map so a single resource block does all grants.
  grants = var.create ? merge(
    { for r in local.runner_roles : "runner:${r}" => { member = local.runner_email, role = r } },
    { for r in local.compute_roles : "compute:${r}" => { member = local.compute_email, role = r } },
  ) : {}
}

resource "google_project_iam_member" "grant" {
  for_each = local.grants

  project = var.project_id
  role    = each.value.role
  member  = "serviceAccount:${each.value.member}"
}

# Let the runner impersonate the compute SA (needed to attach it to worker jobs) — no keys.
resource "google_service_account_iam_member" "runner_impersonates_compute" {
  count              = var.create ? 1 : 0
  service_account_id = google_service_account.compute[0].name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.runner_email}"
}

output "runner_email" {
  description = "scale-forecasting-runner service account email."
  value       = local.runner_email
}

output "compute_email" {
  description = "scale-forecasting-compute service account email."
  value       = local.compute_email
}
