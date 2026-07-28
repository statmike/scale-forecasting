# storage — the three GCS buckets the platform uses (DESIGN §13.0).
#
#   warehouse — the open-format data lake: managed-Iceberg table files live here.
#   artifacts — serialized fitted models (ObjectRef lineage from the registry).
#   code      — the packaged src/ + seed_spark.py that Dataproc Serverless loads.
#
# Why three buckets and not one with folder prefixes? GCS applies IAM, versioning,
# lifecycle, and force_destroy at the BUCKET level — "folders" are just name prefixes, not
# real boundaries — so separate buckets are what let each of these get the policy it needs:
#
#   * warehouse is the decisive one: the BigLake connection SA is granted objectUser +
#     legacyBucketReader scoped to THIS bucket only (see modules/bigquery). Fold everything
#     into one bucket and that
#     least-privilege grant would also expose the code and model artifacts. Prefix-scoped IAM
#     (IAM Conditions on resource-name prefixes) exists but is brittle and not universally
#     honored — a real separate bucket is the clean boundary.
#   * artifacts vs code have OPPOSITE retention postures, and the settings are bucket-wide:
#       - code is a derivable deploy artifact (source of truth is GitHub); it WANTS
#         force_destroy = true in dev and would tolerate an aggressive lifecycle/TTL.
#       - artifacts carry G3 lineage (a forecast row points back to the exact fitted model);
#         they must NEVER be force_destroyed, and a code-oriented TTL would silently delete
#         reproducibility. One bucket can't hold both stances at once.
#
# Cost is identical either way — GCS bills per byte + operations, not per bucket, so three
# empty buckets cost exactly what one empty bucket does (nothing). The split buys policy
# isolation and clean per-bucket cost visibility for free.
#
# Bucket names are globally unique, so all three are prefixed with the project id. Uniform
# access (no legacy ACLs), and versioning on so an overwrite is recoverable.

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "force_destroy" {
  description = "Allow `terraform destroy` to delete non-empty buckets. true for throwaway dev."
  type        = bool
  default     = false
}

locals {
  buckets = {
    warehouse = "${var.project_id}-warehouse"
    artifacts = "${var.project_id}-artifacts"
    code      = "${var.project_id}-code"
  }
}

resource "google_storage_bucket" "b" {
  for_each = local.buckets

  project                     = var.project_id
  name                        = each.value
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = var.force_destroy

  versioning {
    enabled = true
  }
}

output "warehouse_bucket" {
  value = google_storage_bucket.b["warehouse"].name
}

output "warehouse_uri" {
  description = "gs:// root for the managed-Iceberg warehouse (feeds ddl.render_create_tables)."
  value       = "gs://${google_storage_bucket.b["warehouse"].name}/warehouse"
}

output "artifacts_bucket" {
  value = google_storage_bucket.b["artifacts"].name
}

output "code_bucket" {
  value = google_storage_bucket.b["code"].name
}
