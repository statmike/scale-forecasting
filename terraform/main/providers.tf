# Providers. Both the GA and beta Google providers target the same project/region; the beta
# surface is only needed by the composer module. Project is passed explicitly — we never rely
# on (or mutate) the caller's gcloud/ADC default project (DESIGN §13.0).

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}
