# network — a minimal VPC + subnet for the serverless compute (Dataproc Serverless, later Ray).
#
# Fresh projects in this org are created WITHOUT a `default` network, and Dataproc Serverless (and
# Ray on Vertex) require a subnet that has **Private Google Access** enabled plus a firewall rule
# allowing **internal ingress within the subnet** — the executors talk to each other and reach
# Google APIs (BigQuery Storage Write, GCS, Artifact Registry) over private access, with no
# external IPs. This module provides exactly that and nothing more (one capability).
#
# Custom subnet mode (not auto): we create a single regional subnet on purpose, so there are no
# surprise subnets in other regions and the firewall scope is exactly this CIDR.
#
# GREENFIELD vs BROWNFIELD (same BYO pattern as iam/composer/seed):
#   * create = true  (default): create the VPC + subnet + firewall. The 5-minute quickstart.
#   * create = false (brownfield): create nothing; you already have a network. You MUST pass an
#     existing subnet via `subnetwork_uri` (which must have Private Google Access + internal-ingress
#     allowed) — the module just passes it through to the `subnetwork_uri` output the seed/Ray jobs
#     consume. A locked-down org owns its own VPC; this module stays out of the way.

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "create" {
  description = "Create the VPC + subnet + firewall. false = BYO (pass an existing subnetwork_uri)."
  type        = bool
  default     = true
}

variable "subnetwork_uri" {
  description = "Existing subnet self-link, used ONLY when create = false (brownfield). Must have Private Google Access + internal-ingress firewall."
  type        = string
  default     = null
}

variable "subnet_cidr" {
  description = "Primary CIDR for the compute subnet (create = true only). /24 is ample."
  type        = string
  default     = "10.10.0.0/24"
}

resource "google_compute_network" "vpc" {
  count                   = var.create ? 1 : 0
  project                 = var.project_id
  name                    = "scale-forecasting"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "compute" {
  count         = var.create ? 1 : 0
  project       = var.project_id
  name          = "scale-forecasting-compute"
  region        = var.region
  network       = google_compute_network.vpc[0].id
  ip_cidr_range = var.subnet_cidr

  # Required by Dataproc Serverless / Ray: reach Google APIs privately, no external IPs.
  private_ip_google_access = true
}

# Dataproc Serverless requires ingress that allows ALL internal traffic within the subnet (the
# driver/executors communicate on arbitrary ports). Scope the source to the subnet CIDR only.
resource "google_compute_firewall" "internal" {
  count     = var.create ? 1 : 0
  project   = var.project_id
  name      = "scale-forecasting-allow-internal"
  network   = google_compute_network.vpc[0].id
  direction = "INGRESS"

  source_ranges = [var.subnet_cidr]

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "icmp"
  }
}

output "network_id" {
  description = "Self-link of the created VPC (null in brownfield/BYO mode)."
  value       = var.create ? google_compute_network.vpc[0].id : null
}

output "subnetwork_uri" {
  description = "Subnet URI for google_dataproc_batch subnetwork_uri — created self-link, or the BYO value when create = false."
  value       = var.create ? google_compute_subnetwork.compute[0].self_link : var.subnetwork_uri
}
