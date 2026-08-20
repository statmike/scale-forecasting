# Providers. Both the GA and beta Google providers target the same project/region; the beta
# surface is only needed by the composer module. Project is passed explicitly — we never rely
# on (or mutate) the caller's gcloud/ADC default project.

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Quota-scoped alias — used ONLY by the budget module. The billingbudgets API refuses to run
# against ADC unless a quota project is set on the request; billing_project + user_project_override
# add the X-Goog-User-Project header pointing at THIS project (never the caller's global ADC).
#
# Why an alias and not the default provider: user_project_override routes the quota header on
# EVERY call, including the Service Usage calls that enable APIs — and Service Usage can't be
# billed to a project where it isn't enabled yet (a bootstrap deadlock). Scoping the override to
# the one resource that needs it avoids that. billingbudgets + serviceusage are enabled on the
# project by the apis module first (via the override-free default provider).
provider "google" {
  alias                 = "billing_quota"
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}
