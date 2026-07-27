# composer — Composer 3 (Managed Service for Apache Airflow), gated and lifecycle-managed.
#
# ─── LIFECYCLE: start → run → stop (the whole point of the create_composer toggle) ───────
#
#   START  (turn scheduling on):
#     set create_composer = true, then `terraform apply`. ~25 min to build. From then on the
#     DAG (Phase 7 / B6) runs on a schedule. This starts the meter (~$300-400/mo, smallest env).
#
#   RUN:
#     the environment hosts the Airflow DAG that orchestrates prep → fan-out (Spark|Ray + BQ)
#     → fan-in → ensemble → finalize. Git-Sync pulls the DAG from the repo. The SAME run_cell
#     code runs here as locally (G1) — Composer only schedules and fans out.
#
#   STOP  (turn the meter off):
#     set create_composer = false, then `terraform apply`. Terraform destroys just this
#     environment; everything else (data, registry, buckets) is untouched, so ad-hoc/notebook
#     runs keep working. Flip it back on whenever you next need scheduled runs.
#
# This is why Composer is a first-class toggle, not a hardcoded resource: many deployments run
# the pipeline on demand and never want a scheduler at rest; others want to control exactly
# when it is on. Either is one variable + one apply.
# ─────────────────────────────────────────────────────────────────────────────────────────

variable "create" {
  description = "Create the Composer 3 environment. false = no scheduler, no at-rest cost."
  type        = bool
  default     = false
}

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "service_account" {
  description = "Runner SA the environment (and its workers) run as — scale-forecasting-runner."
  type        = string
}

variable "image_version" {
  description = "Pinned Composer 3 image (pin exactly to avoid surprise upgrades on apply)."
  type        = string
  default     = "composer-3-airflow-2.10.5-build.0"
}

# Composer 3 resources require the google-beta provider surface for some fields.
resource "google_composer_environment" "this" {
  count    = var.create ? 1 : 0
  provider = google-beta

  name    = "scale-forecasting"
  project = var.project_id
  region  = var.region

  config {
    software_config {
      image_version = var.image_version
    }

    # Smallest env (DESIGN §13.0 / D5): the platform's compute lives in Dataproc/Ray, not in
    # Airflow — Composer only schedules and fans out, so minimal workers are correct.
    environment_size = "ENVIRONMENT_SIZE_SMALL"

    workloads_config {
      scheduler {
        cpu        = 1
        memory_gb  = 2
        storage_gb = 1
        count      = 1
      }
      web_server {
        cpu        = 1
        memory_gb  = 2
        storage_gb = 1
      }
      worker {
        cpu        = 1
        memory_gb  = 2
        storage_gb = 1
        min_count  = 1
        max_count  = 3
      }
    }

    node_config {
      service_account = var.service_account
    }
  }
}

output "airflow_uri" {
  description = "Airflow web UI URI (null when Composer is gated off)."
  value       = var.create ? google_composer_environment.this[0].config[0].airflow_uri : null
}

output "dag_gcs_prefix" {
  description = "GCS prefix Git-Sync / DAG uploads target (null when gated off)."
  value       = var.create ? google_composer_environment.this[0].config[0].dag_gcs_prefix : null
}
