# Bootstrap inputs. Only a handful — everything else is derived or defaulted.
#
# You supply these once (via terraform.tfvars or -var), then never again: the main
# stage reads the project id and state bucket from here via its own variables.

variable "project_id" {
  description = "Project id to create (must be globally unique). e.g. statmike-scale-forecasting"
  type        = string
}

variable "billing_account" {
  description = "Billing account id to link the new project to (format XXXXXX-XXXXXX-XXXXXX)."
  type        = string
}

variable "org_id" {
  description = "Organization id the project is created under. Mutually exclusive with folder_id."
  type        = string
  default     = null
}

variable "folder_id" {
  description = "Folder id to create the project under (use instead of org_id if you nest projects)."
  type        = string
  default     = null
}

variable "region" {
  description = "Default region for the state bucket (and the rest of the deployment)."
  type        = string
  default     = "us-central1"
}

variable "create_project" {
  description = <<-EOT
    Greenfield toggle (DESIGN §13.0 BYO posture). true (default) = Terraform creates the
    project. false = you already have the project (pass its id in project_id) and Terraform
    only creates the state bucket inside it. A locked-down org whose admins pre-create
    projects sets this false.
  EOT
  type        = bool
  default     = true
}

variable "state_bucket_name" {
  description = <<-EOT
    Name of the GCS bucket that will hold the MAIN stage's Terraform state. Defaults to
    "<project_id>-tfstate". Bucket names are globally unique, so override if that is taken.
  EOT
  type        = string
  default     = null
}
