# composer — Composer 3 (Managed Service for Apache Airflow), gated and lifecycle-managed.
#
# ─── LIFECYCLE: start → run → stop (the whole point of the create_composer toggle) ───────
#
#   NOTE — this module provisions the Composer environment; the DAGs are emitted per-run, not
#   checked in. There is no dags/ directory: `airflow_emit.emit_airflow_dag` renders a
#   `dag_<run_id>.py` for a given config, which you import into the environment's DAG folder
#   (the dag_gcs_prefix output below). A fresh environment is idle until you import a DAG.
#
#   START  (turn scheduling on):
#     set create_composer = true, then `terraform apply`. ~25 min to build. This starts the meter
#     (~$300-400/mo, smallest env). Import an emitted DAG to give it something to run.
#
#   RUN:
#     an emitted DAG orchestrates prep → fan-out (Spark|Ray + BQ) → fan-in → ensemble → finalize.
#     Import it into dag_gcs_prefix (the Airflow smoke does this with `gcloud composer environments
#     storage dags import`; Git-Sync is the alternative for repo-hosted DAGs). The SAME run-cell
#     code runs there as locally — Composer only schedules and fans out. The Airflow smoke
#     (tests/smokes/airflow_smoke.py, config 15) exercises exactly this path end to end.
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
  default     = "composer-3-airflow-2.10.5"
}

# The SF_* run-time identity a worker needs to BE a launch point — same set the smokes/SDK read
# (Settings + BatchInfra + RayInfra). The root wires this from Terraform outputs; a worker resolves
# the run from the staged config URI + this env, exactly as a local launch does. Empty = bare env
# (idle scheduler, no runs).
variable "env_variables" {
  description = "SF_* environment for the workers (the launch-point identity), from Terraform outputs."
  type        = map(string)
  default     = {}
}

# The SUBMIT-side dependency subset — what the driver imports to talk to the services (BigQuery
# registry, Dataproc/Vertex submit, the Ray JobSubmissionClient handshake). NOT the in-service model
# stack (torch/darts/neuralprophet/pyspark): that ships to the jobs in the src/ zip + runs in the
# container/venv/Ray cluster, never on Composer. The Ray client MUST match the cluster's Ray version
# (SF_RAY_VERSION default, currently 2.47) or the dashboard handshake hangs. Validate this set against
# Composer's preinstalled packages on first apply — the classic version-conflict trap lives here.
variable "pypi_packages" {
  description = "Submit-side deps installed on the workers (map package -> version spec). NOT the model stack."
  type        = map(string)
  default = {
    "google-cloud-dataproc"   = ""
    "google-cloud-aiplatform" = ""
    "ray"                     = "[default]==2.47.1" # extras go in the VALUE — Composer rejects extras in the key (must be a bare PEP-508 name)
    "pydantic"                = ">=2"
    # Feature-engineering that runs ON the launch point (not in a job): the native/BigQuery track
    # builds holiday exog columns in Python on the worker before issuing BQML SQL, so `holidays` is a
    # submit-side dep here even though it's not part of the in-service model stack.
    "holidays"                = ">=0.50"
  }
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
      # Give the workers the launch-point identity + the submit-side deps. Model code is NOT here —
      # it arrives per-job as the src/ zip (see the composer-sync bootstrap + docs/smoke_testing.md).
      env_variables = var.env_variables
      pypi_packages = var.pypi_packages
    }

    # The platform's heavy compute lives in Dataproc/Ray/BigQuery, not in Airflow — but a family
    # task is NOT free on the worker: it imports the package and submits+polls a job, and the run
    # fans out to all families at once, so one worker forks that many `airflow tasks run` children
    # concurrently. At 2 GB a worker OOMs mid-fan-out and restarts, orphaning its tasks in `queued`.
    # Workers stay small (compute is remote) but need headroom for the concurrent submit-side imports.
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
        cpu        = 2
        memory_gb  = 6
        storage_gb = 2
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
  description = "GCS prefix emitted-DAG imports / Git-Sync target (null when gated off)."
  value       = var.create ? google_composer_environment.this[0].config[0].dag_gcs_prefix : null
}

# The sibling plugins/ prefix (on the workers' PYTHONPATH) — the composer-sync bootstrap rsyncs
# src/ here so a worker can import the driver AND re-zip that same src/ to deliver code to the jobs.
output "plugins_gcs_prefix" {
  description = "GCS prefix the composer-sync bootstrap rsyncs src/ into (null when gated off)."
  value       = var.create ? replace(google_composer_environment.this[0].config[0].dag_gcs_prefix, "/dags", "/plugins") : null
}
