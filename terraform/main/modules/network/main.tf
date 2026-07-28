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

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "subnet_cidr" {
  description = "Primary CIDR for the compute subnet. /24 is ample for serverless batches."
  type        = string
  default     = "10.10.0.0/24"
}

resource "google_compute_network" "vpc" {
  project                 = var.project_id
  name                    = "scale-forecasting"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "compute" {
  project       = var.project_id
  name          = "scale-forecasting-compute"
  region        = var.region
  network       = google_compute_network.vpc.id
  ip_cidr_range = var.subnet_cidr

  # Required by Dataproc Serverless / Ray: reach Google APIs privately, no external IPs.
  private_ip_google_access = true
}

# Dataproc Serverless requires ingress that allows ALL internal traffic within the subnet (the
# driver/executors communicate on arbitrary ports). Scope the source to the subnet CIDR only.
resource "google_compute_firewall" "internal" {
  project   = var.project_id
  name      = "scale-forecasting-allow-internal"
  network   = google_compute_network.vpc.id
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
  value = google_compute_network.vpc.id
}

output "subnetwork_id" {
  description = "Self-link of the compute subnet — feeds Dataproc/Ray execution_config."
  value       = google_compute_subnetwork.compute.id
}

output "subnetwork_uri" {
  description = "Fully-qualified subnet URI for google_dataproc_batch subnetwork_uri."
  value       = google_compute_subnetwork.compute.self_link
}
