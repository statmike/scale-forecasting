# What the main stage needs from bootstrap. After `terraform apply` here, run
# `terraform output backend_config` and paste the block into main/backend.tf (or pass
# the bucket via `terraform init -backend-config=...`).

output "project_id" {
  description = "The project id (created or pre-existing)."
  value       = var.project_id
}

output "state_bucket" {
  description = "GCS bucket holding the main stage's Terraform state."
  value       = google_storage_bucket.tfstate.name
}

output "backend_config" {
  description = "Copy-paste backend block for the main stage."
  value       = <<-EOT
    terraform {
      backend "gcs" {
        bucket = "${google_storage_bucket.tfstate.name}"
        prefix = "main"
      }
    }
  EOT
}
