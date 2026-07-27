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

# The connection's service agent must be able to read/write the warehouse bucket objects.
resource "google_storage_bucket_iam_member" "conn_warehouse" {
  bucket = var.warehouse_bucket
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_bigquery_connection.iceberg.cloud_resource[0].service_account_id}"
}

output "dataset_id" {
  value = google_bigquery_dataset.ds.dataset_id
}

output "connection_id" {
  description = "Fully-qualified connection ref (project.region.name) for ddl.render_create_tables."
  value       = "${var.project_id}.${var.region}.${google_bigquery_connection.iceberg.connection_id}"
}
