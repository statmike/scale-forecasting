# container — the Artifact Registry Docker repo that holds the shared Spark runtime image.
#
# This is the dependency-delivery mechanism for every Spark and Ray path (the seed job and the
# forecast engines): one image = the `scale_forecasting` package + core deps, so the same
# code runs local == Managed Spark == Ray on Vertex. The image itself is built by Cloud Build from
# docker/Dockerfile (see docker/cloudbuild.yaml) and pushed here; this module owns only the
# *repository* (the container), mirroring how bigquery owns the dataset and the app owns the
# tables — one source of truth for the image (the Dockerfile), no HCL/Docker drift.
#
# Terraform builds the image on apply (null_resource.build below) by running that same
# docker/cloudbuild.yaml via `gcloud builds submit` — so one `terraform apply` creates the repo AND
# fills it, and the seed batch (which pulls this image) has something to pull. The build is
# content-addressed on docker/ (Dockerfile + requirements.txt), so it runs on the first apply and
# rebuilds ONLY when those deps change — never on a source-code edit (code ships at runtime via
# python_file_uris, not baked in). Set build_image = false to skip it (image pre-built in CI or an
# air-gapped registry); the equivalent manual command is:
#
#   gcloud builds submit --config docker/cloudbuild.yaml \
#     --substitutions=_REGION=<region>,_REPO=<repo>,_IMAGE=spark-runtime,_TAG=latest \
#     --project <project_id>
#
# There is deliberately NO git-push-triggered rebuild (google_cloudbuild_trigger). Source code ships
# at runtime via python_file_uris, so a push that only edits src/ must NOT rebuild the image — the
# image changes only when docker/Dockerfile or docker/requirements.txt do. The one-shot build above
# is content-addressed on exactly those two files, so a normal `terraform apply` after a dependency
# change rebuilds automatically and nothing else does. When you bump deps, re-apply (or run the
# manual `gcloud builds submit` above). A push-trigger would add a GitHub-connection dependency and
# rebuild on every commit for no benefit, so it is intentionally omitted.

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

variable "build_image" {
  description = "Build + push the runtime image via Cloud Build on apply. false = pre-built image."
  type        = bool
  default     = true
}

variable "image_tag" {
  description = "Tag to build/push (must match the tag the seed batch and engines consume)."
  type        = string
  default     = "latest"
}

variable "code_bucket" {
  description = <<-EOT
    Bucket that receives the packed-venv archive (envs/<hash>.tar.gz). The Dataproc *cluster* path
    can't use the custom container, so it attaches this venv-pack of the same locked env instead.
    The build packs it from the image's /opt/venv and uploads it here.
  EOT
  type        = string
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
# the identity `gcloud builds submit` uses for the one-shot build on apply.
data "google_project" "this" {
  project_id = var.project_id
}

locals {
  # Base image path (no tag). The build's _IMAGE/_TAG extend it and the seed batch consumes it.
  image_repo_path = format(
    "%s-docker.pkg.dev/%s/%s/%s",
    var.region,
    var.project_id,
    google_artifact_registry_repository.images.repository_id,
    var.image_name,
  )

  cloudbuild_sa = "${data.google_project.this.number}-compute@developer.gserviceaccount.com"
  cloudbuild_roles = [
    "roles/cloudbuild.builds.builder", # run builds, read the source-staging bucket, write logs
    "roles/artifactregistry.writer",   # push the built image into this repo
  ]

  # The packed-venv archive is content-addressed on requirements.txt (the same file that gates the
  # image build), so its object name changes only when deps change — matching the image lifecycle.
  venv_hash        = filemd5("${path.module}/../../../../docker/requirements.txt")
  venv_archive_uri = "gs://${var.code_bucket}/envs/${local.venv_hash}.tar.gz"
}

# Cloud Build packs the image's /opt/venv and uploads it to the code bucket, so its SA needs object
# write there (scoped to just this bucket — narrower than a project-wide storage role).
resource "google_storage_bucket_iam_member" "cloudbuild_code_bucket" {
  bucket = var.code_bucket
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${local.cloudbuild_sa}"
}

resource "google_project_iam_member" "cloudbuild" {
  for_each = toset(local.cloudbuild_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${local.cloudbuild_sa}"
}

# Build + push the runtime image via Cloud Build, reusing docker/cloudbuild.yaml (one source of
# truth for the build). triggers is content-addressed on the two files that define the image —
# Dockerfile + requirements.txt — plus the destination path/tag, so the build runs on the FIRST
# apply (no prior state) and re-runs ONLY when those change. Source code is NOT a trigger: it ships
# at runtime via python_file_uris, so editing src/ must not rebuild the slow image (mirrors the seed
# module's content-addressing). depends_on ensures the repo exists and the Cloud Build SA can push
# before the build runs; downstream consumers (the seed batch) order after this via module.container.
resource "null_resource" "build" {
  count = var.build_image ? 1 : 0

  triggers = {
    dockerfile   = filemd5("${path.module}/../../../../docker/Dockerfile")
    requirements = filemd5("${path.module}/../../../../docker/requirements.txt")
    image        = "${local.image_repo_path}:${var.image_tag}"
    venv_archive = local.venv_archive_uri
  }

  provisioner "local-exec" {
    working_dir = "${path.module}/../../../.." # repo root: build context "." + docker/cloudbuild.yaml
    command = join(" ", [
      "gcloud builds submit",
      "--config docker/cloudbuild.yaml",
      # _CODE_BUCKET/_VENV_HASH drive the pack+upload of the venv archive for the cluster path.
      "--substitutions=_REGION=${var.region},_REPO=${var.repository_id},_IMAGE=${var.image_name},_TAG=${var.image_tag},_CODE_BUCKET=${var.code_bucket},_VENV_HASH=${local.venv_hash}",
      "--project ${var.project_id}", # explicit — never the ambient ADC project
      ".",
    ])
  }

  depends_on = [
    google_artifact_registry_repository.images,
    google_project_iam_member.cloudbuild,
    google_storage_bucket_iam_member.cloudbuild_code_bucket,
  ]
}

output "repository_id" {
  description = "Artifact Registry repository id."
  value       = google_artifact_registry_repository.images.repository_id
}

output "image_built" {
  description = "Set once the build has run (or null when build_image = false); order downstream on this."
  value       = var.build_image ? null_resource.build[0].id : null
}

# The full image path the Dataproc batch's runtime_config.container_image consumes and that the
# `gcloud builds submit` _IMAGE/_TAG substitutions extend. Tag is appended by the consumer.
output "image_repo_path" {
  description = "Base image path: <region>-docker.pkg.dev/<project>/<repo>/<image>."
  value       = local.image_repo_path
}

# The packed-venv archive the Dataproc *cluster* path attaches (--archives=<uri>#env). Feeds
# SF_VENV_ARCHIVE / BatchInfra.venv_archive_uri; content-addressed on requirements.txt so it tracks
# the image. The object exists after the build has packed+uploaded it (build_image = true).
output "venv_archive_uri" {
  description = "gs:// URI of the packed-venv archive for the Dataproc-cluster dependency path."
  value       = local.venv_archive_uri
}
