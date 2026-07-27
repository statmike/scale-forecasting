# Root — wire the one-capability modules together. Read top-to-bottom: APIs first, then the
# things that depend on them. Dependencies are expressed by passing one module's output into
# the next (Terraform infers order from that), with a few explicit depends_on for API enablement.

module "apis" {
  source     = "./modules/apis"
  project_id = var.project_id
  enable     = var.enable_apis
}

module "iam" {
  source        = "./modules/iam"
  project_id    = var.project_id
  create        = var.create_service_accounts
  runner_email  = var.runner_sa_email
  compute_email = var.compute_sa_email

  depends_on = [module.apis]
}

module "storage" {
  source     = "./modules/storage"
  project_id = var.project_id
  region     = var.region

  depends_on = [module.apis]
}

module "bigquery" {
  source           = "./modules/bigquery"
  project_id       = var.project_id
  region           = var.region
  dataset_id       = var.dataset_id
  warehouse_bucket = module.storage.warehouse_bucket # also sequences storage before bq

  depends_on = [module.apis]
}

module "budget" {
  source          = "./modules/budget"
  project_id      = var.project_id
  billing_account = var.billing_account
  amount_usd      = var.budget_amount_usd

  depends_on = [module.apis]
}

# Gated: created only when create_composer = true (see modules/composer for the lifecycle).
module "composer" {
  source          = "./modules/composer"
  create          = var.create_composer
  project_id      = var.project_id
  region          = var.region
  service_account = module.iam.runner_email

  depends_on = [module.apis]
}
