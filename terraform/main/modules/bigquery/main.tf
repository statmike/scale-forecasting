# bigquery — the dataset and the BigLake connection that managed-Iceberg tables need.
#
# NOTE on the table schemas: the five tables' DDL is single-sourced in Python
# (src/scale_forecasting/registry/ddl.py, snapshot-tested) and created by
# registry.bq.ensure_tables() at run time. We deliberately do NOT re-declare those schemas
# in HCL — two copies of the DDL would drift. Terraform owns the *containers* (dataset,
# connection, bucket grant); the app owns the *tables*. (Deviation from BUILD B0.2's literal
# "registry tables via Terraform"; recorded in NOTES.md.)

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "dataset_id" {
  type = string
}

variable "warehouse_bucket" {
  description = "Warehouse bucket name; the connection's service agent is granted access to it."
  type        = string
}

resource "google_bigquery_dataset" "ds" {
  project     = var.project_id
  dataset_id  = var.dataset_id
  location    = var.region
  description = "scale-forecasting run registry + example data (managed Iceberg)."
}

# Cloud Resource connection: managed-Iceberg tables read/write their GCS files through this
# connection's Google-managed service agent, not through the caller. We create it, then grant
# its auto-generated service account object access on the warehouse bucket.
resource "google_bigquery_connection" "iceberg" {
  project       = var.project_id
  location      = var.region
  connection_id = "sf-iceberg"
  friendly_name = "scale-forecasting managed Iceberg"
  description   = "BigLake connection for managed-Iceberg tables on the warehouse bucket."

  cloud_resource {}
}

# A cloud-resource connection's service agent is provisioned asynchronously — the agent id is
# returned immediately, but it can take a few seconds to become referenceable in IAM. Without
# this pause the grant below can fail with "service account ... does not exist". A short sleep
# is the standard idiom for this eventual-consistency gap.
resource "time_sleep" "wait_for_connection_agent" {
  depends_on      = [google_bigquery_connection.iceberg]
  create_duration = "20s"
}

# The connection's service agent needs TWO grants on the warehouse bucket, because the
# Storage Write API streaming path (the route the workers use to write per-series results)
# checks BOTH object access AND `storage.buckets.get` on the bucket — and no single predefined
# role covers both without over-granting (only storage.admin does, which also adds bucket
# delete + setIamPolicy). Verified empirically in the B0.3 spike: with object access alone,
# append_rows failed 403 "connection does not have permissions storage.buckets.get".
#   1. objectUser        — read/write/delete the Iceberg data files (storage.objects.*).
#   2. legacyBucketReader — the single bucket-metadata read (storage.buckets.get) the Write
#                           API requires; the least-privilege role that carries it.
# Load jobs and query-INSERT never hit this check, which is why they passed on objects alone.
resource "google_storage_bucket_iam_member" "conn_warehouse_objects" {
  bucket = var.warehouse_bucket
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_bigquery_connection.iceberg.cloud_resource[0].service_account_id}"

  depends_on = [time_sleep.wait_for_connection_agent]
}

resource "google_storage_bucket_iam_member" "conn_warehouse_bucket" {
  bucket = var.warehouse_bucket
  role   = "roles/storage.legacyBucketReader"
  member = "serviceAccount:${google_bigquery_connection.iceberg.cloud_resource[0].service_account_id}"

  depends_on = [time_sleep.wait_for_connection_agent]
}

output "dataset_id" {
  value = google_bigquery_dataset.ds.dataset_id
}

output "connection_id" {
  description = "Fully-qualified connection ref (project.region.name) for ddl.render_create_tables."
  value       = "${var.project_id}.${var.region}.${google_bigquery_connection.iceberg.connection_id}"
}
