# Outputs — the handful of values the app + the operator need after apply. These feed the
# run config (dataset/connection/warehouse) and the seed job (code bucket), and show the
# operator the SAs and (if on) the Airflow UI.

output "project_id" {
  value = var.project_id
}

output "dataset_id" {
  description = "BigQuery dataset for registry + example data."
  value       = module.bigquery.dataset_id
}

output "iceberg_connection" {
  description = "BigLake connection ref (project.region.name) for ddl.render_create_tables."
  value       = module.bigquery.connection_id
}

output "warehouse_uri" {
  description = "gs:// root for managed-Iceberg table files."
  value       = module.storage.warehouse_uri
}

output "artifacts_bucket" {
  value = module.storage.artifacts_bucket
}

output "code_bucket" {
  description = "Bucket the Dataproc Serverless jobs load packaged src/ + seed_spark.py from."
  value       = module.storage.code_bucket
}

output "runner_sa" {
  value = module.iam.runner_email
}

output "compute_sa" {
  value = module.iam.compute_email
}

output "airflow_uri" {
  description = "Airflow UI (null unless create_composer = true)."
  value       = module.composer.airflow_uri
}

output "runtime_image_repo" {
  description = "Base path for the shared Spark/Ray runtime image (append :tag). Build target for Cloud Build."
  value       = module.container.image_repo_path
}

output "subnetwork_uri" {
  description = "Subnet the serverless batches run in (needed by the submit helper for forecast runs)."
  value       = module.network.subnetwork_uri
}

output "seed_batch_id" {
  description = "Submitted seed batch id (null unless run_seed = true)."
  value       = module.seed.batch_id
}

output "seed_batch_state" {
  description = "Terminal state of the seed batch (null unless run_seed = true)."
  value       = module.seed.batch_state
}
