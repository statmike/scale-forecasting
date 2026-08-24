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

variable "build_gpu_image" {
  description = <<-EOT
    Build a custom Dataproc VM image with the NVIDIA driver pre-baked, for GPU clusters (opt-in;
    off by default). Needs extra IAM (Compute image/instance admin) and builder-VM egress to the
    NVIDIA mirrors, so it is separate from the always-on container build. When false, GPU clusters
    fall back to installing the driver at cluster-create time.
  EOT
  type        = bool
  default     = false
}

variable "gpu_image_zone" {
  description = "Zone the GPU-image builder VM boots in. Empty = <region>-a."
  type        = string
  default     = ""
}

variable "gpu_image_subnet" {
  description = "Subnet (relative form) for the GPU-image builder VM. Empty = the tool's default network."
  type        = string
  default     = ""
}

variable "gpu_dataproc_version" {
  description = "Base Dataproc image line the GPU custom image is built from (must match the cluster image)."
  type        = string
  default     = "2.2-debian12"
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
  cloudbuild_roles = concat(
    [
      "roles/cloudbuild.builds.builder", # run builds, read the source-staging bucket, write logs
      "roles/artifactregistry.writer",   # push the built image into this repo
    ],
    # The GPU custom-image build boots + captures a temporary builder VM, so the build SA additionally
    # needs Compute instance/image admin and to actAs the builder VM's SA. Granted only when opted in.
    var.build_gpu_image ? [
      "roles/compute.instanceAdmin.v1", # create/delete the builder VM + create the captured image
      "roles/iam.serviceAccountUser",   # actAs the builder VM's service account
    ] : [],
  )

  # The packed-venv archive is content-addressed on requirements.txt (the same file that gates the
  # image build), so its object name changes only when deps change — matching the image lifecycle.
  venv_hash        = filemd5("${path.module}/../../../../docker/requirements.txt")
  venv_archive_uri = "gs://${var.code_bucket}/envs/${local.venv_hash}.tar.gz"

  # The custom GPU image is content-addressed on the customization script + the base Dataproc version,
  # so its name changes only when either does — the driver layer is independent of the Python deps, so
  # it has its own lifecycle (never rebuilds on a requirements or source change). Image names are
  # lowercase, <=63 chars: a stable prefix + an 8-char digest fits with room to spare.
  gpu_image_hash = substr(md5("${filemd5("${path.module}/../../../../docker/gpu_image_customize.sh")}:${var.gpu_dataproc_version}"), 0, 8)
  gpu_image_name = "sf-dataproc-gpu-${local.gpu_image_hash}"
  # Dataproc accepts the relative resource path for a custom image. Emitted only when opted in — when
  # off, the app gets no SF_GPU_IMAGE and GPU clusters install the driver at create (the fallback).
  gpu_image_uri  = var.build_gpu_image ? "projects/${var.project_id}/global/images/${local.gpu_image_name}" : null
  gpu_image_zone = var.gpu_image_zone != "" ? var.gpu_image_zone : "${var.region}-a"
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

# Build the custom GPU cluster image via Cloud Build, reusing docker/cloudbuild-gpu-image.yaml. Same
# content-addressing discipline as the container build, but on its OWN inputs — the customization
# script + the base Dataproc version — because the driver layer is independent of the Python deps
# (it must NOT rebuild on a requirements or source change). Opt-in (build_gpu_image); --no-source
# because this build needs no repo context beyond the customization script, which is passed by path.
resource "null_resource" "build_gpu_image" {
  count = var.build_gpu_image ? 1 : 0

  triggers = {
    customize = filemd5("${path.module}/../../../../docker/gpu_image_customize.sh")
    version   = var.gpu_dataproc_version
    image     = local.gpu_image_name
  }

  provisioner "local-exec" {
    working_dir = "${path.module}/../../../.." # repo root: customization script path is /workspace-relative
    command = join(" ", [
      "gcloud builds submit",
      "--config docker/cloudbuild-gpu-image.yaml",
      "--substitutions=_IMAGE_NAME=${local.gpu_image_name},_DATAPROC_VERSION=${var.gpu_dataproc_version},_ZONE=${local.gpu_image_zone},_CODE_BUCKET=${var.code_bucket},_SUBNET=${var.gpu_image_subnet}",
      "--project ${var.project_id}", # explicit — never the ambient ADC project
      ".",
    ])
  }

  depends_on = [
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

# The custom GPU cluster image (NVIDIA driver pre-baked). Feeds SF_GPU_IMAGE /
# BatchInfra.gpu_image_uri; null when build_gpu_image = false, in which case GPU clusters install the
# driver at create time. Content-addressed on the customization script + Dataproc version.
output "gpu_image_uri" {
  description = "Resource path of the pre-baked GPU cluster image, or null when not built."
  value       = local.gpu_image_uri
}
