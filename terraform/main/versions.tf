# Main stage — provider pins + the GCS remote backend.
#
# The backend bucket is created by the bootstrap stage. `terraform init` here reads state
# from it (with locking), so a fork can be shared across a team. The bucket name is supplied
# at init time via -backend-config so this file carries no environment-specific ids:
#
#   terraform init -backend-config="bucket=<project_id>-tfstate"

terraform {
  required_version = ">= 1.5"

  backend "gcs" {
    prefix = "main"
    # bucket = "..."  # provided via `terraform init -backend-config=bucket=...`
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.0"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.12"
    }
    # Zips src/ at apply time so the seed batch loads current code via python_file_uris — the
    # package is NOT baked into the runtime image (modules/seed).
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}
