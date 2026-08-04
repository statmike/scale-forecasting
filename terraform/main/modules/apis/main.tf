# apis — enable exactly the Google services the platform uses, and nothing more.
#
# One capability: turn APIs on. The `enable` toggle lets an org that pre-enables APIs skip
# this entirely (count = 0). disable_on_destroy = false so tearing down the deployment does
# not yank APIs that other work in the project might rely on.

variable "project_id" {
  type = string
}

variable "enable" {
  description = "Master switch; false = manage no APIs (an admin already enabled them)."
  type        = bool
  default     = true
}

locals {
  # The full set for Spark + Ray + BigQuery + Composer + lineage. Grouped by purpose in the
  # comments so a reader knows why each is here.
  services = [
    "compute.googleapis.com",            # networking substrate for Dataproc/Vertex/Composer
    "storage.googleapis.com",            # GCS: warehouse, artifacts, code buckets
    "bigquery.googleapis.com",           # registry + native models + Iceberg tables
    "bigqueryconnection.googleapis.com", # BigLake / Cloud Resource connection for Iceberg
    "dataproc.googleapis.com",           # Dataproc Serverless (Spark engines + seed job)
    "aiplatform.googleapis.com",         # Vertex AI (Ray on Vertex)
    # The Ray interactive dashboard / job-submission handshake is served through the managed
    # Inverting-Proxy fabric (*.aiplatform-training.googleusercontent.com), which is built on the
    # same Notebooks/IAP-backed path as Colab Enterprise. Without these, the proxy's backend leg
    # never answers and the JobSubmissionClient GET /api/version hangs → HTTP 524.
    "servicenetworking.googleapis.com",    # Private Services Access peering for the Ray private endpoint
    "composer.googleapis.com",             # Composer 3 (Airflow) — created only if create_composer
    "artifactregistry.googleapis.com",     # container images for engines / Ray
    "cloudbuild.googleapis.com",           # build those images
    "serviceusage.googleapis.com",         # enable/disable APIs; also the quota target for the budget
    "iam.googleapis.com",                  # service accounts + role grants
    "cloudresourcemanager.googleapis.com", # project-level IAM bindings
    "cloudbilling.googleapis.com",         # link project ↔ billing account
    "billingbudgets.googleapis.com",       # the budget + threshold alerts (distinct API)
  ]
}

resource "google_project_service" "svc" {
  for_each = var.enable ? toset(local.services) : []

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

output "enabled_services" {
  description = "The services this module turned on (empty when enable = false)."
  value       = [for s in google_project_service.svc : s.service]
}
