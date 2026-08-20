# Bootstrap stage — create the project (optional) and the remote-state bucket.
#
# Run this ONCE per deployment, with local state. After it applies, you copy the
# generated backend config into the main stage and `terraform init` there. This stage
# is deliberately tiny: a project, a state bucket, and nothing that costs money at rest
# (an empty versioned bucket is effectively free).

locals {
  # Default the state bucket name off the project id unless the caller overrode it.
  state_bucket = coalesce(var.state_bucket_name, "${var.project_id}-tfstate")
}

# The provider here is intentionally project-less for the create-project call: creating a
# project is an org/folder-level operation. Once the project exists, the bucket resource
# targets it explicitly. We never mutate the caller's gcloud/ADC default project.
provider "google" {
  region = var.region
}

# --- the project ---------------------------------------------------------------
# Created only when create_project = true. exactly one of org_id / folder_id must be set.
resource "google_project" "this" {
  count = var.create_project ? 1 : 0

  name            = var.project_id
  project_id      = var.project_id
  billing_account = var.billing_account
  org_id          = var.folder_id == null ? var.org_id : null
  folder_id       = var.folder_id

  # Keep the default network out of a fresh project; the deployment creates only what it needs.
  auto_create_network = false

  lifecycle {
    precondition {
      condition     = (var.org_id == null) != (var.folder_id == null)
      error_message = "Set exactly one of org_id or folder_id."
    }
  }
}

# The Cloud Storage API must be on before we can create the bucket. In a brand-new project
# nothing is enabled yet, so enable just this one API here; the main stage's `apis` module
# turns on the full set.
resource "google_project_service" "storage" {
  project            = var.project_id
  service            = "storage.googleapis.com"
  disable_on_destroy = false

  # Ensure the project exists first when we're the ones creating it.
  depends_on = [google_project.this]
}

# --- the remote-state bucket ---------------------------------------------------
# Versioned so a botched apply can be rolled back; uniform access (no ACLs); force_destroy
# stays false so state can never be deleted by an accidental `terraform destroy`.
resource "google_storage_bucket" "tfstate" {
  project                     = var.project_id
  name                        = local.state_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.storage]
}
