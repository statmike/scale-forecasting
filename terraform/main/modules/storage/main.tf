# storage — the three GCS buckets the platform uses (DESIGN §13.0).
#
#   warehouse — the open-format data lake: managed-Iceberg table files live here.
#   artifacts — serialized fitted models (ObjectRef lineage from the registry).
#   code      — the packaged src/ + seed_spark.py that Dataproc Serverless loads.
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
