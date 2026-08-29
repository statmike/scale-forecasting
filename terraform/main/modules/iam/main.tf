# iam — the two service accounts and their least-privilege roles.
#
# No keys, ever. These SAs are used via ADC + impersonation:
#   scale-forecasting-runner  — orchestration: read/write BQ, GCS, submit Dataproc/Ray jobs.
#                               Composer runs as it.
#   scale-forecasting-compute — attached to Dataproc/Ray workers: BQ data + GCS artifacts only.
#
# Names are spelled out in full (not abbreviated) so they read self-evidently in the IAM
# console — a reader sees the product they belong to without a decoder. The account_id limit
# is 30 chars; both fit (24 and 25).
#
# `create` = false lets you bring your own SAs (pass their emails in); then this module only
# resolves the emails through to outputs and grants nothing (your admin owns the grants).
#
# This module also DEFINES (but does not bind) two run-observability operator roles —
# sfProbeReader and sfJobCanceller (§9 of the runtime-probe design) — for the humans/responders who
# probe and cancel live runs. They are deliberately separate from the runner SA's operational grants
# so a read-only responder can diagnose a stuck run without the power to stop it. An admin binds them
# to a person or group; the module outputs their ids.

variable "project_id" {
  type = string
}

variable "create" {
  description = "Create the SAs + grant roles. false = BYO (pass runner_email/compute_email)."
  type        = bool
  default     = true
}

variable "runner_email" {
  description = "Existing runner SA email when create = false."
  type        = string
  default     = null
}

variable "compute_email" {
  description = "Existing compute SA email when create = false."
  type        = string
  default     = null
}

# --- the accounts --------------------------------------------------------------

resource "google_service_account" "runner" {
  count        = var.create ? 1 : 0
  project      = var.project_id
  account_id   = "scale-forecasting-runner"
  display_name = "scale-forecasting orchestration (BQ/GCS/Dataproc/Ray submit)"
}

resource "google_service_account" "compute" {
  count        = var.create ? 1 : 0
  project      = var.project_id
  account_id   = "scale-forecasting-compute"
  display_name = "scale-forecasting workers (BQ data + GCS artifacts)"
}

locals {
  runner_email  = var.create ? google_service_account.runner[0].email : var.runner_email
  compute_email = var.create ? google_service_account.compute[0].email : var.compute_email

  # Least-privilege role sets. Kept as locals so the grants below stay a single readable loop.
  # The connection role is our custom sfConnectionDelegate (below), not connectionUser: creating
  # managed-Iceberg tables through the BigLake connection needs bigquery.connections.delegate, and
  # among predefined roles that permission ships ONLY in connectionAdmin — which also carries
  # setIamPolicy + delete on the connection. Same reasoning as the warehouse-bucket grants (we
  # chose legacyBucketReader over storage.admin): take the exact permissions, not the broad role.
  connection_role = var.create ? google_project_iam_custom_role.connection_delegate[0].id : null

  # Ray-cluster lifecycle role — our custom sfRayClusterManager (below), added on top of
  # roles/aiplatform.user. Headless Ray (e.g. Composer) runs AS the runner SA and must create + tear
  # down its own fixed-size cluster (a Vertex PersistentResource).
  # aiplatform.user carries only persistentResources.get/list; create/delete ship exclusively in
  # roles/aiplatform.admin (440+ perms) — so, same reasoning as sfConnectionDelegate, we take the
  # four exact permissions instead of the broad role.
  ray_cluster_role = var.create ? google_project_iam_custom_role.ray_cluster_manager[0].id : null

  # (key, role) pairs. The key is a STATIC label so it can be a for_each map key even though the
  # connection role's value is only known after apply (Terraform requires known keys, apply-time
  # values). Keys read as the role's short name; `connection` is the custom sfConnectionDelegate.
  runner_roles = {
    "bq.dataEditor"    = "roles/bigquery.dataEditor"      # write registry rows + create tables
    "bq.jobUser"       = "roles/bigquery.jobUser"         # run queries / load jobs
    "bq.readSession"   = "roles/bigquery.readSessionUser" # Storage Read API: notebooks/tools read result + leaderboard tables as the runner SA (headless acceptance + human Colab-open both run AS this SA)
    "connection"       = local.connection_role            # get/use/delegate the BigLake connection
    "storage.objAdmin" = "roles/storage.objectAdmin"      # warehouse + artifacts + code buckets
    "dataproc.editor"  = "roles/dataproc.editor"          # submit Dataproc Serverless batches
    "aiplatform.user"  = "roles/aiplatform.user"          # submit Ray on Vertex jobs (get/list clusters)
    "ray.cluster"      = local.ray_cluster_role           # create/delete the Ray cluster it runs on
    "composer.worker"  = "roles/composer.worker"          # Composer runs AS this SA: env workers need logging.logEntries.create + monitoring/storage (Google's prescribed env-SA role). Inert until create_composer = true.
  }
  compute_roles = {
    "bq.dataEditor"    = "roles/bigquery.dataEditor" # read source_series, write results
    "bq.jobUser"       = "roles/bigquery.jobUser"
    "bq.readSession"   = "roles/bigquery.readSessionUser" # Storage Read API: spark-bigquery connector reads source_series
    "connection"       = local.connection_role            # get/use/delegate the BigLake connection
    "storage.objAdmin" = "roles/storage.objectAdmin"      # read/write model artifacts
    "dataproc.worker"  = "roles/dataproc.worker"          # batch RUNTIME SA: logs/metrics/staging
    "artifactreg.read" = "roles/artifactregistry.reader"  # pull the custom Spark runtime image
  }

  # Run-observability operator permission sets (§9 two-tier). job-canceller is a strict SUPERSET of
  # probe-reader — expressed with concat() so the "superset" relationship is structural, not a
  # copy-paste that can drift. probe-reader READS live runtime state (Dataproc/Ray/BigQuery) to
  # reconcile a run; job-canceller adds the STOP verbs. Reading the registry itself (the reconcile
  # query) needs BigQuery job + dataset read (roles/bigquery.jobUser + a dataViewer/readSession grant
  # on the registry dataset); that is granted where the operator is bound, not in these roles.
  probe_reader_perms = [
    "dataproc.batches.get",               # Dataproc Serverless: batch state
    "dataproc.jobs.get",                  # Dataproc cluster: job state
    "bigquery.jobs.list",                 # native family: resolve the run's jobs by id prefix
    "bigquery.jobs.get",                  # ...and read each statement's state
    "aiplatform.persistentResources.get", # Ray: reach the cluster to read job status
  ]
  job_canceller_perms = concat(local.probe_reader_perms, [
    "dataproc.batches.delete", # Serverless has no cancel — deleting a running batch stops it
    "dataproc.jobs.cancel",    # Dataproc cluster job cancel
    "bigquery.jobs.update",    # cancel a running BigQuery statement
    # Ray stop_job goes through the cluster dashboard — no IAM verb beyond persistentResources.get.
  ])

  # Flatten (email, role) pairs into one map — static keys, apply-time role values.
  grants = var.create ? merge(
    { for k, r in local.runner_roles : "runner:${k}" => { member = local.runner_email, role = r } },
    { for k, r in local.compute_roles : "compute:${k}" => { member = local.compute_email, role = r } },
  ) : {}
}

# Custom role: exactly the connection permissions the Iceberg path needs — get + use + delegate.
# `delegate` is what lets the SA create/write managed-Iceberg tables *through* the connection's
# service agent; predefined connectionUser omits it and connectionAdmin over-grants (setIamPolicy,
# delete). Least-privilege, matching this module's philosophy.
resource "google_project_iam_custom_role" "connection_delegate" {
  count       = var.create ? 1 : 0
  project     = var.project_id
  role_id     = "sfConnectionDelegate"
  title       = "scale-forecasting BigLake connection delegate"
  description = "Get, use, and delegate the BigLake connection for managed-Iceberg tables."
  permissions = [
    "bigquery.connections.get",
    "bigquery.connections.use",
    "bigquery.connections.delegate",
  ]
}

# Custom role: exactly the Vertex PersistentResource (Ray cluster) lifecycle permissions the runner
# needs — create + delete + get + list. Headless Ray creates its fixed-size cluster, submits, then
# tears it down; those first two ship only in roles/aiplatform.admin (440+ perms). Same least-privilege
# philosophy as sfConnectionDelegate: take the four exact permissions, not the broad role.
resource "google_project_iam_custom_role" "ray_cluster_manager" {
  count       = var.create ? 1 : 0
  project     = var.project_id
  role_id     = "sfRayClusterManager"
  title       = "scale-forecasting Ray cluster manager"
  description = "Create, delete, get, and list Vertex Ray clusters (PersistentResources)."
  permissions = [
    "aiplatform.persistentResources.create",
    "aiplatform.persistentResources.delete",
    "aiplatform.persistentResources.get",
    "aiplatform.persistentResources.list",
  ]
}

# Operator role, tier 1: read live runtime state to reconcile a run — but NOT stop it. Bind this to
# a read-only responder (an on-call diagnosing a stuck run) who should never be able to cancel.
resource "google_project_iam_custom_role" "probe_reader" {
  count       = var.create ? 1 : 0
  project     = var.project_id
  role_id     = "sfProbeReader"
  title       = "scale-forecasting probe reader"
  description = "Read live Dataproc/Ray/BigQuery job state to reconcile a run — no cancel."
  permissions = local.probe_reader_perms
}

# Operator role, tier 2: probe-reader + the stop verbs. Bind this to whoever may cancel in-flight
# runs. A strict superset of sfProbeReader (see local.job_canceller_perms). The cancel path
# pre-flights nothing extra — a missing permission surfaces as a clear "needs the job-canceller
# role" message (probes._cancel_failure) on the actual cancel call, not a stack trace.
resource "google_project_iam_custom_role" "job_canceller" {
  count       = var.create ? 1 : 0
  project     = var.project_id
  role_id     = "sfJobCanceller"
  title       = "scale-forecasting job canceller"
  description = "Superset of probe-reader: also STOP a run's in-flight Dataproc/Ray/BigQuery jobs."
  permissions = local.job_canceller_perms
}

resource "google_project_iam_member" "grant" {
  for_each = local.grants

  project = var.project_id
  role    = each.value.role
  member  = "serviceAccount:${each.value.member}"
}

# Ray on Vertex over a PSC-I (Private Service Connect Interface) network attachment: the managed
# Vertex tenant reaches back into this VPC through the attachment, and it does so AS the Vertex AI
# Service Agent (service-<project_number>@gcp-sa-aiplatform.iam.gserviceaccount.com).
#
# This is a SINGLE-project topology (attachment, subnet, and Vertex agent all in this project), not
# Shared VPC. Google's PSC-I setup docs prescribe roles/compute.networkAdmin for the single-project
# case; roles/compute.networkUser is the Shared-VPC (host→service) grant. We deliberately take the
# LEANER networkUser here (not networkAdmin) plus the custom sfNetworkAttachmentConsumer below —
# together they cover what the agent actually exercises: pulling an interface IP from the subnet
# (compute.subnetworks.use, carried by networkUser) and consuming the attachment
# (get/use/update/list, in the custom role). networkUser is broader than the four attachment verbs
# but is NOT redundant with the custom role: subnetworks.use ships only in networkUser/networkAdmin,
# so dropping it would 403 the interface-IP allocation.
#
# Further least-privilege trim (not applied here): to replace networkUser entirely, add
# compute.subnetworks.use to a custom role scoped to the compute subnet (a subnet-level IAM binding,
# not project-wide). networkUser is the documented-adjacent, known-good grant, and narrowing subnet
# IP allocation is the exact change that risks a silent 403 on cluster create.
data "google_project" "this" {
  project_id = var.project_id
}

locals {
  vertex_agent = "service-${data.google_project.this.number}@gcp-sa-aiplatform.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "vertex_agent_network_user" {
  count   = var.create ? 1 : 0
  project = var.project_id
  role    = "roles/compute.networkUser"
  member  = "serviceAccount:${local.vertex_agent}"
}

# On top of networkUser, consuming a PSC-I network attachment needs the attachment-specific verbs
# networkUser lacks. When Vertex attaches, it (1) reads the attachment (.get — in networkUser),
# (2) uses it (.use), and (3) patches it to auto-add its producer tenant project to the accepted
# list (.update) — the "Producer service automatically adds the producer tenant project" step. The
# console create fails 403 on each in turn without these. networkAdmin would cover them but grants
# project-wide network admin to a service agent; take the four exact permissions instead (same
# least-privilege philosophy as sfConnectionDelegate / sfRayClusterManager).
resource "google_project_iam_custom_role" "network_attachment_consumer" {
  count       = var.create ? 1 : 0
  project     = var.project_id
  role_id     = "sfNetworkAttachmentConsumer"
  title       = "scale-forecasting network attachment consumer"
  description = "Get, use, update, and list PSC-I network attachments (Vertex Ray private path)."
  permissions = [
    "compute.networkAttachments.get",
    "compute.networkAttachments.use",
    "compute.networkAttachments.update",
    "compute.networkAttachments.list",
  ]
}

resource "google_project_iam_member" "vertex_agent_attachment_consumer" {
  count   = var.create ? 1 : 0
  project = var.project_id
  role    = google_project_iam_custom_role.network_attachment_consumer[0].id
  member  = "serviceAccount:${local.vertex_agent}"
}

# Let the runner impersonate the compute SA (needed to attach it to worker jobs) — no keys.
resource "google_service_account_iam_member" "runner_impersonates_compute" {
  count              = var.create ? 1 : 0
  service_account_id = google_service_account.compute[0].name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.runner_email}"
}

# Let the runner act as ITSELF. Creating a Dataproc Spark Connect Session (notebook 01, run headless
# AS the runner) requires serviceAccountUser on the session's own service account — without this the
# Session create fails "User not authorized to act as service account <runner>". Batch/Ray jobs run
# as the compute SA (grant above); only the interactive Connect Session runs as the runner itself.
resource "google_service_account_iam_member" "runner_impersonates_self" {
  count              = var.create ? 1 : 0
  service_account_id = google_service_account.runner[0].name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.runner_email}"
}

output "runner_email" {
  description = "scale-forecasting-runner service account email."
  value       = local.runner_email
}

output "compute_email" {
  description = "scale-forecasting-compute service account email."
  value       = local.compute_email
}

output "probe_reader_role_id" {
  description = "Custom role id (sfProbeReader): read-only run reconciliation. Bind to responders. null in BYO mode."
  value       = var.create ? google_project_iam_custom_role.probe_reader[0].id : null
}

output "job_canceller_role_id" {
  description = "Custom role id (sfJobCanceller): probe-reader + stop verbs. Bind to run cancellers. null in BYO mode."
  value       = var.create ? google_project_iam_custom_role.job_canceller[0].id : null
}
