# container — the Artifact Registry Docker repo that holds the shared Spark runtime image.
#
# This is the dependency-delivery mechanism for every Spark path (the seed job in B0.4 and the
# forecast engines in B2): one image = the `scale_forecasting` package + core deps, so the same
# code runs local == Managed Spark == (later) Ray. The image itself is built by Cloud Build from
# docker/Dockerfile (see docker/cloudbuild.yaml) and pushed here; this module owns only the
# *repository* (the container), mirroring how bigquery owns the dataset and the app owns the
# tables — one source of truth for the image (the Dockerfile), no HCL/Docker drift.
#
# The image is built on demand today:
#
#   gcloud builds submit --config docker/cloudbuild.yaml \
#     --substitutions=_REGION=<region>,_REPO=<repo>,_IMAGE=spark-runtime,_TAG=latest \
#     --project <project_id>
#
# A git-push-triggered rebuild (google_cloudbuild_trigger) is intentionally DEFERRED: it needs a
# GitHub repo connection, which waits on the first push to the private repo (§A-PUSH). Wiring it
# now would fail to plan (no connection to reference).

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "repository_id" {
  description = "Artifact Registry repo name that holds the Spark/Ray runtime images."
  type        = string
  default     = "scale-forecasting"
}

variable "image_name" {
  description = "Image name within the repo (the Dockerfile builds one shared runtime image)."
  type        = string
  default     = "spark-runtime"
}

resource "google_artifact_registry_repository" "images" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
  format        = "DOCKER"
  description   = "scale-forecasting shared Spark/Ray runtime images (built by Cloud Build)."
}

# Cloud Build runs as the project's Compute Engine default SA (fresh projects no longer grant it
# roles automatically). It needs to read its own source-staging bucket + push to Artifact
# Registry, so grant it the builder role + AR writer. Scoped to exactly the build path — this is
# the identity `gcloud builds submit` uses until a dedicated build SA/trigger lands (§A-PUSH).
data "google_project" "this" {
  project_id = var.project_id
}

locals {
  cloudbuild_sa = "${data.google_project.this.number}-compute@developer.gserviceaccount.com"
  cloudbuild_roles = [
    "roles/cloudbuild.builds.builder", # run builds, read the source-staging bucket, write logs
    "roles/artifactregistry.writer",   # push the built image into this repo
  ]
}

resource "google_project_iam_member" "cloudbuild" {
  for_each = toset(local.cloudbuild_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${local.cloudbuild_sa}"
}

output "repository_id" {
  description = "Artifact Registry repository id."
  value       = google_artifact_registry_repository.images.repository_id
}

# The full image path the Dataproc batch's runtime_config.container_image consumes and that the
# `gcloud builds submit` _IMAGE/_TAG substitutions extend. Tag is appended by the consumer.
output "image_repo_path" {
  description = "Base image path: <region>-docker.pkg.dev/<project>/<repo>/<image>."
  value = format(
    "%s-docker.pkg.dev/%s/%s/%s",
    var.region,
    var.project_id,
    google_artifact_registry_repository.images.repository_id,
    var.image_name,
  )
}
