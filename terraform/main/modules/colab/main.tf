# colab — Colab Enterprise runtime templates for running the notebooks from inside GCP.
#
# WHAT THIS OWNS ───────────────────────────────────────────────────────────────────────────
# Two durable, free-at-rest runtime TEMPLATES (blueprints for the VM a Colab runtime runs on).
# A template costs nothing until someone starts a runtime from it, and runtimes idle-shutdown, so
# both ship on by default (create = true from the root):
#
#   * sf-main          — Python 3.11, the everyday template. Matches the project's requires-python
#                        pin (>=3.11,<3.12), which is load-bearing for Vertex Ray client↔cluster
#                        parity and the Dataproc packed-venv. Drives notebooks 02–07 + the
#                        playground; the [ray] extra is what notebook 04 needs.
#   * sf-spark-connect — Python 3.12, ONLY for notebook 01's interactive Spark Connect path.
#                        Dataproc 3.0 Connect workers run Python 3.12 and Connect refuses mismatched
#                        minors (PYTHON_VERSION_MISMATCH); the [spark] extra carries
#                        dataproc-spark-connect. From a 3.11 kernel NB01 falls back to remote-batch.
#
# See docs/notebook_runtimes.md for the per-notebook Python-version mapping.
#
# THE PYTHON-VERSION PIN (why there are null_resources) ──────────────────────────────────────
# The pinned google provider (6.x) resource google_colab_runtime_template CANNOT set the Python
# version — software_config exposes only env + a deprecated post_startup_script_config, with no
# colab_image/release_name field (hashicorp/terraform-provider-google#25217, still open). The only
# lever is the REST field software_config.colab_image.release_name (py310|py311|py312; empty =
# Latest). So the TF resource owns the durable template and every spec it CAN express, and a
# tolerant REST PATCH pins the one field it can't. BOTH templates are pinned explicitly: sf-main to
# py311 and sf-spark-connect to py312. (sf-spark-connect must track Dataproc 3.0 Connect workers,
# which run 3.12; if we left it on the API default of "latest" it would silently drift to 3.13 when
# Colab advances Latest, re-breaking NB01 interactive with PYTHON_VERSION_MISMATCH — the exact
# failure this template exists to avoid.) When the provider adds the image field, delete these
# null_resources and set release_name inline on each resource.
#
# PACKAGES ──────────────────────────────────────────────────────────────────────────────────
# By default the notebooks' own bootstrap cell installs the repo + extra on first cell-run (a bit
# slower on cold start, always correct). Flip install_via_post_startup = true to pre-install via a
# post-startup script staged to code_bucket (faster cold start; the field is deprecated and some
# orgs block it on new templates, hence off by default).
#
# NETWORK (attach_network) ───────────────────────────────────────────────────────────────────
#   * attach_network = true (default): attach this project's VPC + subnet; the VM gets egress with
#     no external IP via the Cloud NAT + Private Google Access from modules/network. This is the
#     default because the greenfield stack builds a CUSTOM VPC and has no `default` network — a
#     public runtime (network unset) makes Colab fall back to `default` and 404. Also compatible
#     with a compute.vmExternalIpAccess = DENY org policy.
#   * attach_network = false: public runtime (external IP, internet egress). Only for a brownfield
#     project that has a usable `default` network and permits external IPs.

variable "create" {
  description = "Create the Colab Enterprise runtime templates. false = no templates. Free at rest either way."
  type        = bool
  default     = false
}

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "service_account" {
  description = "Runner SA the notebooks execute as (already holds roles/aiplatform.user)."
  type        = string
}

variable "code_bucket" {
  description = "Bucket for notebook-execution output + the optional post-startup install script."
  type        = string
}

variable "attach_network" {
  description = "Attach the project VPC + subnet to the runtimes (VPC-private path). false = public internet-access runtime."
  type        = bool
  default     = false
}

variable "network_id" {
  description = "VPC (project-number form) to attach when attach_network = true. From module.network.network_id."
  type        = string
  default     = null
}

variable "subnetwork_uri" {
  description = "Subnet self-link to attach when attach_network = true. From module.network.subnetwork_uri."
  type        = string
  default     = null
}

# --- run identity baked into the templates' software_config.env -------------------------------
# These are the SF_* values Settings.resolve() (and the batch/Ray submit paths) read. Baking them
# into each template's env means a headless NotebookExecutionJob's fresh, empty kernel — and a human
# who just opens the template and runs the notebook — both get the full run identity with no manual
# env cell. Values are wired from the sibling modules by the root module "colab" block. All optional
# (null/empty entries are dropped from the env map) so the module still plans when create = false or
# a BYO deploy leaves some unset.

variable "connection" {
  description = "BigLake connection ref (project.region.name) → SF_CONNECTION. From module.bigquery.connection_id."
  type        = string
  default     = null
}

variable "warehouse_uri" {
  description = "GCS managed-Iceberg warehouse root → SF_WAREHOUSE_URI. From module.storage.warehouse_uri."
  type        = string
  default     = null
}

variable "dataset_id" {
  description = "BigQuery dataset → SF_DATASET_ID. From module.bigquery.dataset_id."
  type        = string
  default     = null
}

variable "compute_sa" {
  description = "Compute SA the batch/Ray jobs run as → SF_COMPUTE_SA. From module.iam.compute_email."
  type        = string
  default     = null
}

variable "container_image" {
  description = "Full runtime image ref (repo:tag) → SF_CONTAINER_IMAGE. Built from module.container.image_repo_path."
  type        = string
  default     = null
}

variable "network_attachment_id" {
  description = "PSC-I network attachment self-link for Vertex Ray → SF_RAY_NETWORK_ATTACHMENT. From module.network.network_attachment_id."
  type        = string
  default     = null
}

variable "machine_type" {
  description = "Machine type for the runtime VM. e2-standard-4 is ample to drive Dataproc/Ray/BigQuery from a notebook."
  type        = string
  default     = "e2-standard-4"
}

variable "idle_timeout" {
  description = "Auto-shutdown a runtime after this idle period (keeps at-rest cost near zero)."
  type        = string
  default     = "1800s"
}

variable "main_release_name" {
  description = "Colab image release for sf-main. py311 matches the project pin (Ray parity + packed-venv)."
  type        = string
  default     = "py311"
}

variable "spark_release_name" {
  description = "Colab image release for sf-spark-connect. py312 pinned explicitly (required by Dataproc 3.0 Connect workers; pinned so it can't drift off Latest)."
  type        = string
  default     = "py312"
}

variable "main_extra" {
  description = "Optional-dependency extra pre-installed on sf-main when install_via_post_startup = true."
  type        = string
  default     = "ray"
}

variable "spark_extra" {
  description = "Optional-dependency extra pre-installed on sf-spark-connect when install_via_post_startup = true."
  type        = string
  default     = "spark"
}

variable "install_via_post_startup" {
  description = <<-EOT
    Pre-install the repo + extra via a post-startup script (faster cold start) instead of relying on
    each notebook's own bootstrap cell. Default FALSE — the field is deprecated and some orgs block
    it on new templates; the notebook bootstrap installs correctly on first cell-run regardless.
  EOT
  type        = bool
  default     = false
}

variable "repo_url" {
  description = "Repo the post-startup script clones (only used when install_via_post_startup = true)."
  type        = string
  default     = "https://github.com/statmike/scale-forecasting.git"
}

locals {
  # region-scoped aiplatform endpoint the runtime-template REST API lives behind.
  api_host = "https://${var.region}-aiplatform.googleapis.com/v1"

  # Post-startup install scripts (only staged/attached when install_via_post_startup = true). Each
  # clones the repo and editable-installs the package with its extra — the same thing the notebooks'
  # bootstrap cell does, just at runtime-creation time so the notebook's install cell is a fast no-op.
  post_startup = {
    main  = "#!/bin/bash\nset -e\ngit clone --depth 1 ${var.repo_url} /opt/scale-forecasting || true\npip install -e '/opt/scale-forecasting[${var.main_extra}]'\n"
    spark = "#!/bin/bash\nset -e\ngit clone --depth 1 ${var.repo_url} /opt/scale-forecasting || true\npip install -e '/opt/scale-forecasting[${var.spark_extra}]'\n"
  }
  gate = var.create && var.install_via_post_startup

  # SF_* run identity baked into each template's software_config.env. Settings.resolve() reads the
  # first five (SF_PROJECT_ID/REGION/CONNECTION/WAREHOUSE_URI/DATASET_ID); sf-main also carries the
  # batch + Ray infra vars (submit.py:BatchInfra, ray_submit.py:RayInfra) and sf-spark-connect the
  # Dataproc-Connect vars NB01 reads. Note the subnet ALIAS: submit.py wants SF_SUBNETWORK_URI in
  # ABSOLUTE form; NB01 wants SF_DATAPROC_SUBNET in RELATIVE form (same strip the network_spec uses).
  # SF_RUNTIME_VERSION / SF_RAY_VERSION / SF_RAY_NETWORK are intentionally NOT set so the code
  # defaults (submit.py 2.2, ray_submit.py 2.47; attachment beats peering in RayInfra) stay the
  # single source of truth. compact() + the null-safe values below drop any entry that isn't wired
  # (BYO deploys / create = false), so the env map only ever contains resolved values.
  identity_env = {
    SF_PROJECT_ID    = var.project_id
    SF_REGION        = var.region
    SF_CONNECTION    = var.connection
    SF_WAREHOUSE_URI = var.warehouse_uri
    SF_DATASET_ID    = var.dataset_id
  }

  main_env = merge(local.identity_env, {
    SF_CODE_BUCKET            = var.code_bucket
    SF_COMPUTE_SA             = var.compute_sa
    SF_CONTAINER_IMAGE        = var.container_image
    SF_SUBNETWORK_URI         = var.subnetwork_uri
    SF_RAY_NETWORK_ATTACHMENT = var.network_attachment_id
  })

  spark_env = merge(local.identity_env, {
    SF_DATAPROC_REGION = var.region
    SF_DATAPROC_SUBNET = var.subnetwork_uri == null ? null : replace(var.subnetwork_uri, "/^https://[^/]+/compute/v1//", "")
  })

  # Drop null/empty entries — a template env can't carry a value we don't have.
  main_env_clean  = { for k, v in local.main_env : k => v if v != null && v != "" }
  spark_env_clean = { for k, v in local.spark_env : k => v if v != null && v != "" }
}

# Colab validates the gcsOutputUri bucket with storage.buckets.get before running a notebook. The
# runner SA has object-level roles/storage.objectAdmin (objects.*) which does NOT include
# buckets.get — so an execution 403s on the bucket check. legacyBucketReader grants exactly
# buckets.get (+ objects.list), scoped to THIS bucket — least-privilege, matching the deploy.
resource "google_storage_bucket_iam_member" "runner_bucket_get" {
  count  = var.create ? 1 : 0
  bucket = var.code_bucket
  role   = "roles/storage.legacyBucketReader"
  member = "serviceAccount:${var.service_account}"
}

# The optional post-startup install scripts, staged to GCS (content-addressed so a change is a new
# object). Only created when install_via_post_startup = true.
resource "google_storage_bucket_object" "post_startup_main" {
  count   = local.gate ? 1 : 0
  bucket  = var.code_bucket
  name    = "colab/post_startup_main-${substr(md5(local.post_startup.main), 0, 8)}.sh"
  content = local.post_startup.main
}

resource "google_storage_bucket_object" "post_startup_spark" {
  count   = local.gate ? 1 : 0
  bucket  = var.code_bucket
  name    = "colab/post_startup_spark-${substr(md5(local.post_startup.spark), 0, 8)}.sh"
  content = local.post_startup.spark
}

# sf-main — the everyday template (Python 3.11, [ray]). The Python version is pinned by
# null_resource.pin_main_python below, NOT here (provider can't set it — see header).
resource "google_colab_runtime_template" "main" {
  count        = var.create ? 1 : 0
  project      = var.project_id
  location     = var.region
  display_name = "sf-main"

  machine_spec {
    machine_type = var.machine_type
  }

  data_persistent_disk_spec {
    disk_type    = "pd-standard"
    disk_size_gb = "100"
  }

  # enable_internet_access controls whether the VM gets an EXTERNAL IP. Under
  # compute.vmExternalIpAccess = DENY, true makes the VM fail to launch, so VPC-attached mode routes
  # egress with no external IP via this project's Cloud NAT + Private Google Access instead. Hence
  # enable_internet_access = !attach_network. The subnetwork field demands the RELATIVE form
  # (projects/<p>/regions/<r>/subnetworks/<n>), so strip the https://.../compute/v1/ prefix; the
  # network field accepts either form.
  network_spec {
    enable_internet_access = !var.attach_network
    network                = var.attach_network ? var.network_id : null
    subnetwork             = var.attach_network ? replace(var.subnetwork_uri, "/^https://[^/]+/compute/v1//", "") : null
  }

  idle_shutdown_config {
    idle_timeout = var.idle_timeout
  }

  # Keeps the template creatable under compute.requireShieldedVm (commonly enforced).
  shielded_vm_config {
    enable_secure_boot = true
  }

  # software_config is ALWAYS present now: it carries the SF_* run identity as env so a headless
  # NotebookExecutionJob's fresh kernel (and a human who just opens the template) get the full
  # identity with no manual env cell. post_startup_script_config stays optional (install_via_post_startup).
  software_config {
    dynamic "env" {
      for_each = local.main_env_clean
      content {
        name  = env.key
        value = env.value
      }
    }
    dynamic "post_startup_script_config" {
      for_each = local.gate ? [1] : []
      content {
        post_startup_script_url      = "gs://${var.code_bucket}/${google_storage_bucket_object.post_startup_main[0].name}"
        post_startup_script_behavior = "RUN_ONCE"
      }
    }
  }
}

# sf-spark-connect — Python 3.12 ([spark]) for notebook 01's interactive Spark Connect ONLY. The
# Python version is pinned by null_resource.pin_spark_python below, NOT here (provider can't set it).
# It's pinned EXPLICITLY (not left on the API default of "latest") so it can't drift off 3.12 when
# Colab advances Latest — see header.
resource "google_colab_runtime_template" "spark" {
  count        = var.create ? 1 : 0
  project      = var.project_id
  location     = var.region
  display_name = "sf-spark-connect"

  machine_spec {
    machine_type = var.machine_type
  }

  data_persistent_disk_spec {
    disk_type    = "pd-standard"
    disk_size_gb = "100"
  }

  network_spec {
    enable_internet_access = !var.attach_network
    network                = var.attach_network ? var.network_id : null
    subnetwork             = var.attach_network ? replace(var.subnetwork_uri, "/^https://[^/]+/compute/v1//", "") : null
  }

  idle_shutdown_config {
    idle_timeout = var.idle_timeout
  }

  shielded_vm_config {
    enable_secure_boot = true
  }

  # Always-present software_config (same rationale as sf-main): the SF_* env below is what lets a
  # headless execution / a human's fresh kernel resolve the run identity. sf-spark-connect carries the
  # slimmer identity + the Dataproc-Connect vars NB01 reads (SF_DATAPROC_REGION/SUBNET).
  software_config {
    dynamic "env" {
      for_each = local.spark_env_clean
      content {
        name  = env.key
        value = env.value
      }
    }
    dynamic "post_startup_script_config" {
      for_each = local.gate ? [1] : []
      content {
        post_startup_script_url      = "gs://${var.code_bucket}/${google_storage_bucket_object.post_startup_spark[0].name}"
        post_startup_script_behavior = "RUN_ONCE"
      }
    }
  }
}

# Pin each template's Python version via a TOLERANT REST PATCH — the one field the provider can't set
# (#25217). Mirrors modules/smoke's null_resource: on_failure = continue so a transient PATCH error
# never fails the apply (the template just stays on Latest until the next apply / a manual patch);
# triggers content-addressed on release_name + the template id so it re-patches only when either
# changes; --project explicit, never the ambient ADC project (DESIGN §13.0). BOTH templates are
# pinned — sf-spark-connect to py312 explicitly, so it can't drift to 3.13 when Colab advances Latest
# and re-break NB01 (see header). Delete these blocks once the provider adds the image field and set
# the release_name inline on each google_colab_runtime_template.
resource "null_resource" "pin_main_python" {
  count = var.create ? 1 : 0

  triggers = {
    release_name = var.main_release_name
    template_id  = google_colab_runtime_template.main[0].id
  }

  provisioner "local-exec" {
    on_failure = continue
    command = join(" ", [
      "curl -sS -X PATCH",
      "-H \"Authorization: Bearer $(gcloud auth print-access-token --project=${var.project_id})\"",
      "-H \"Content-Type: application/json\"",
      "\"${local.api_host}/${google_colab_runtime_template.main[0].id}?updateMask=software_config.colab_image.release_name\"",
      "-d '${jsonencode({ softwareConfig = { colabImage = { releaseName = var.main_release_name } } })}'",
    ])
  }

  depends_on = [google_colab_runtime_template.main]
}

resource "null_resource" "pin_spark_python" {
  count = var.create ? 1 : 0

  triggers = {
    release_name = var.spark_release_name
    template_id  = google_colab_runtime_template.spark[0].id
  }

  provisioner "local-exec" {
    on_failure = continue
    command = join(" ", [
      "curl -sS -X PATCH",
      "-H \"Authorization: Bearer $(gcloud auth print-access-token --project=${var.project_id})\"",
      "-H \"Content-Type: application/json\"",
      "\"${local.api_host}/${google_colab_runtime_template.spark[0].id}?updateMask=software_config.colab_image.release_name\"",
      "-d '${jsonencode({ softwareConfig = { colabImage = { releaseName = var.spark_release_name } } })}'",
    ])
  }

  depends_on = [google_colab_runtime_template.spark]
}

output "main_runtime_template_id" {
  description = "Full resource name of the sf-main (Python 3.11 / [ray]) runtime template. null when gated off."
  # .id is the full projects/.../notebookRuntimeTemplates/<n> form the API wants; .name is the bare <n>.
  value = var.create ? google_colab_runtime_template.main[0].id : null
}

output "spark_runtime_template_id" {
  description = "Full resource name of the sf-spark-connect (Python 3.12 / [spark]) runtime template. null when gated off."
  value       = var.create ? google_colab_runtime_template.spark[0].id : null
}
