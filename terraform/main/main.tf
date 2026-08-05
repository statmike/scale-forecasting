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

  providers = {
    google.billing_quota = google.billing_quota
  }

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

# The Artifact Registry repo for the shared Spark/Ray runtime image. The image is built into it
# by Cloud Build (docker/cloudbuild.yaml); this owns only the container.
module "container" {
  source     = "./modules/container"
  project_id = var.project_id
  region     = var.region

  depends_on = [module.apis]
}

# Minimal VPC + subnet for serverless compute (Dataproc Serverless now, Ray later). Fresh projects
# have no default network, and serverless batches need a subnet with Private Google Access.
# Greenfield (create_network = true) builds it; brownfield passes an existing subnetwork_uri.
module "network" {
  source         = "./modules/network"
  project_id     = var.project_id
  region         = var.region
  create         = var.create_network
  subnetwork_uri = var.subnetwork_uri

  depends_on = [module.apis]
}

# Gated: submits the Dataproc Serverless seed batch only when run_seed = true (see modules/seed
# for the smoke → review → full lifecycle). Depends on everything the batch touches at runtime.
module "seed" {
  source     = "./modules/seed"
  create     = var.run_seed
  project_id = var.project_id
  region     = var.region

  num_series   = var.seed_num_series
  master_seed  = var.seed_master_seed
  write_method = var.seed_write_method
  run_label    = var.seed_run_label
  variant      = var.seed_variant

  code_bucket             = module.storage.code_bucket
  container_image         = "${module.container.image_repo_path}:${var.seed_image_tag}"
  compute_service_account = module.iam.compute_email
  connection              = module.bigquery.connection_id
  warehouse_uri           = module.storage.warehouse_uri
  dataset_id              = var.dataset_id
  subnetwork_uri          = module.network.subnetwork_uri

  depends_on = [module.apis, module.iam, module.storage, module.bigquery, module.container, module.network]
}
