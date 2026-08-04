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

variable "psa_prefix_length" {
  description = "Reserved-range size for Private Services Access peering (create = true only). Google recommends /16; a smaller range risks exhaustion as peered services grow."
  type        = number
  default     = 16
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

# Vertex Managed Ray's dashboard/proxy reaches the cluster head node **over the PSA peering** — the
# ingress it originates is sourced from the reserved peering range, NOT the subnet CIDR. The
# subnet-only rule above therefore blocks the dashboard's proxy→origin hop, which surfaces as a
# 524 on the Ray Jobs `GET /api/version` handshake even though the cluster itself is RUNNING. This
# rule opens internal ingress from the reserved peering range so that hop succeeds. (The broad
# `default` network's default-allow-internal covers this implicitly; a locked-down custom VPC must
# add it explicitly.) Scoped to the peering range only.
resource "google_compute_firewall" "psa_ingress" {
  count     = var.create ? 1 : 0
  project   = var.project_id
  name      = "scale-forecasting-allow-psa"
  network   = google_compute_network.vpc[0].id
  direction = "INGRESS"

  source_ranges = ["${google_compute_global_address.psa_range[0].address}/${google_compute_global_address.psa_range[0].prefix_length}"]

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

# Private Services Access (PSA) — peers the VPC with Google's service-producer network so
# VPC-attached managed services can be reached over private IPs. Dataproc Serverless does NOT need
# this (it runs inside the subnet via Private Google Access, the subnet flag above); Vertex Managed
# Ray DOES: Vertex provisions the cluster in a Google tenant project and peers it into this VPC, and
# without PSA the cluster only gets a public dashboard endpoint whose origin is unreachable
# off-cluster (job submission times out). Two pieces: a reserved internal range, then the peering
# connection that hands that range to servicenetworking.
data "google_project" "this" {
  count      = var.create ? 1 : 0
  project_id = var.project_id
}

resource "google_compute_global_address" "psa_range" {
  count         = var.create ? 1 : 0
  project       = var.project_id
  name          = "scale-forecasting-psa"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = var.psa_prefix_length
  network       = google_compute_network.vpc[0].id
}

resource "google_service_networking_connection" "psa" {
  count                   = var.create ? 1 : 0
  network                 = google_compute_network.vpc[0].id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.psa_range[0].name]
}

# Cloud NAT — egress-only internet for VMs on this VPC that have NO external IP (many orgs deny
# external IPs org-wide via compute.vmExternalIpAccess). A VPC-attached client that sits ON this VPC
# to reach the PSA-peered managed-Ray cluster's private endpoint still needs outbound internet to
# pip-install the `google-cloud-aiplatform[ray]` SDK / dependencies. Cloud NAT provides exactly that —
# outbound only, NO inbound exposure — so the no-external-IP posture is preserved. Dataproc Serverless
# does not need this (it reaches Google APIs over Private Google Access), which is why NAT wasn't here
# before. Regional, auto-allocated IPs, all subnets in the region.
resource "google_compute_router" "nat_router" {
  count   = var.create ? 1 : 0
  project = var.project_id
  name    = "scale-forecasting-nat-router"
  region  = var.region
  network = google_compute_network.vpc[0].id
}

resource "google_compute_router_nat" "nat" {
  count                              = var.create ? 1 : 0
  project                            = var.project_id
  name                               = "scale-forecasting-nat"
  router                             = google_compute_router.nat_router[0].name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# PSC-I (Private Service Connect Interface) network attachment — the *newer* private path for Vertex
# Managed Ray, an alternative to the PSA peering above. Vertex's tenant project attaches an interface
# into this VPC through this attachment to reach the cluster head node (and, we are testing, to serve
# the Ray dashboard/`/api/version` proxy hop that 524s over the public + PSA paths).
#
# We create the attachment ourselves (declarative, least-privilege): with connection_preference =
# ACCEPT_AUTOMATIC the attachment auto-accepts the producer (Vertex) tenant project, so the service
# agent only needs to *consume* it (compute.networkAttachments.get/use) — it does NOT need to create,
# patch (.update), or poll (regionOperations.get) the attachment, which is what the console's
# create-inline flow was failing on. The consumer then selects this existing attachment by name.
resource "google_compute_network_attachment" "psc" {
  count                 = var.create ? 1 : 0
  provider              = google-beta
  project               = var.project_id
  name                  = "scale-forecasting-ray"
  region                = var.region
  description           = "Vertex Managed Ray PSC-I interface into the scale-forecasting VPC."
  connection_preference = "ACCEPT_AUTOMATIC"
  subnetworks           = [google_compute_subnetwork.compute[0].self_link]
}

output "network_attachment_id" {
  description = <<-EOT
    PSC-I network attachment for Vertex Managed Ray, in the project-NUMBER form Vertex requires
    (projects/<number>/regions/<region>/networkAttachments/<name>) — same reason network_id uses
    number form: create_ray_cluster rejects the project-ID self-link. null in brownfield/BYO mode.
  EOT
  value = var.create ? "projects/${data.google_project.this[0].number}/regions/${var.region}/networkAttachments/${google_compute_network_attachment.psc[0].name}" : null
}

output "network_id" {
  description = <<-EOT
    VPC network in the project-NUMBER form Vertex Managed Ray requires
    (projects/<number>/global/networks/<name>); the project-ID self-link form is rejected there.
    null in brownfield/BYO mode.
  EOT
  value       = var.create ? "projects/${data.google_project.this[0].number}/global/networks/${google_compute_network.vpc[0].name}" : null
}

output "subnetwork_uri" {
  description = "Subnet URI for google_dataproc_batch subnetwork_uri — created self-link, or the BYO value when create = false."
  value       = var.create ? google_compute_subnetwork.compute[0].self_link : var.subnetwork_uri
}
