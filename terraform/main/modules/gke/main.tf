# gke — Standing Google Kubernetes Engine cluster with scale-to-zero CPU & GPU node pools (opt-in).
#
# ─── LIFECYCLE: start → run → stop (the create_gke toggle) ───────────────────────────────
#
#   NOTE — `runtime = "gke"` works WITHOUT this module: when `compute.gke_cluster_name` and
#   `SF_GKE_CLUSTER` are unset, `gke_submit` provisions an ephemeral GKE cluster per run and tears
#   it down unconditionally in `finally`.
#
#   Turn `create_gke = true` on when you want a standing GKE cluster (`SF_GKE_CLUSTER`) with
#   scale-to-zero (`min_node_count = 0`) CPU and GPU worker node pools so Kubernetes Indexed Jobs
#   (`gke_mode = "job"`) and Ray-on-GKE jobs (`gke_mode = "ray"` / `ray_mode = "gke"`) skip the
#   3-5 minute control-plane creation step and scale worker nodes from 0 -> N on demand.
#
#   START:
#     set `create_gke = true`, then `terraform apply`.
#   STOP:
#     set `create_gke = false`, then `terraform apply` to destroy the standing GKE cluster.
# ─────────────────────────────────────────────────────────────────────────────────────────

variable "create" {
  description = "Create the standing GKE cluster with scale-to-zero CPU and GPU pools. Default false."
  type        = bool
  default     = false
}

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "cluster_name" {
  description = "Standing GKE cluster name (exported as SF_GKE_CLUSTER when create = true)."
  type        = string
  default     = "scale-forecasting-gke"
}

variable "network_id" {
  description = "VPC network id (optional; uses default network when null)."
  type        = string
  default     = null
}

variable "subnetwork_uri" {
  description = "Subnetwork self-link (optional)."
  type        = string
  default     = null
}

variable "compute_service_account" {
  description = "Worker node service account email (scale-forecasting-compute)."
  type        = string
}

variable "cpu_machine_type" {
  description = "Machine type for the scale-to-zero CPU worker pool."
  type        = string
  default     = "n2-standard-8"
}

variable "max_cpu_nodes" {
  description = "Autoscaling ceiling for the CPU worker pool (min is 0)."
  type        = number
  default     = 16
}

variable "gpu_machine_type" {
  description = "Machine type for the scale-to-zero GPU worker pool."
  type        = string
  default     = "n1-standard-8"
}

variable "gpu_accelerator_type" {
  description = "Accelerator type for the scale-to-zero GPU worker pool."
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_accelerator_count" {
  description = "Accelerators per GPU worker node."
  type        = number
  default     = 1
}

variable "max_gpu_nodes" {
  description = "Autoscaling ceiling for the GPU worker pool (min is 0)."
  type        = number
  default     = 8
}

resource "google_container_cluster" "this" {
  count = var.create ? 1 : 0

  name     = var.cluster_name
  project  = var.project_id
  location = "${var.region}-a"

  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false

  network    = var.network_id
  subnetwork = var.subnetwork_uri

  ip_allocation_policy {}

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
  }

  control_plane_endpoints_config {
    dns_endpoint_config {
      allow_external_traffic = true
    }
  }

  resource_labels = {
    managed-by = "scale-forecasting"
  }
}

# Minimal 1-node system pool for kube-system controllers (CoreDNS, metrics-server, konnectivity).
resource "google_container_node_pool" "system" {
  count = var.create ? 1 : 0

  name       = "sf-system-pool"
  project    = var.project_id
  location   = google_container_cluster.this[0].location
  cluster    = google_container_cluster.this[0].name
  node_count = 1

  node_config {
    machine_type    = "e2-standard-2"
    disk_size_gb    = 50
    disk_type       = "pd-balanced"
    service_account = var.compute_service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}

# Scale-to-zero CPU worker pool (0 -> max_cpu_nodes).
resource "google_container_node_pool" "cpu" {
  count = var.create ? 1 : 0

  name     = "sf-cpu-pool"
  project  = var.project_id
  location = google_container_cluster.this[0].location
  cluster  = google_container_cluster.this[0].name

  initial_node_count = 0
  autoscaling {
    min_node_count = 0
    max_node_count = var.max_cpu_nodes
  }

  node_config {
    machine_type    = var.cpu_machine_type
    disk_size_gb    = 100
    disk_type       = "pd-balanced"
    service_account = var.compute_service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    taint {
      key    = "scale-forecasting/pool"
      value  = "cpu"
      effect = "NO_SCHEDULE"
    }
  }
}

# Scale-to-zero GPU worker pool (0 -> max_gpu_nodes).
resource "google_container_node_pool" "gpu" {
  count = var.create ? 1 : 0

  name     = "sf-gpu-pool"
  project  = var.project_id
  location = google_container_cluster.this[0].location
  cluster  = google_container_cluster.this[0].name

  initial_node_count = 0
  autoscaling {
    min_node_count = 0
    max_node_count = var.max_gpu_nodes
  }

  node_config {
    machine_type    = var.gpu_machine_type
    disk_size_gb    = 100
    disk_type       = "pd-balanced"
    service_account = var.compute_service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = var.gpu_accelerator_type
      count = var.gpu_accelerator_count
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    taint {
      key    = "scale-forecasting/pool"
      value  = "gpu"
      effect = "NO_SCHEDULE"
    }
  }
}

output "cluster_name" {
  description = "Standing GKE cluster name (null when create = false)."
  value       = var.create ? google_container_cluster.this[0].name : null
}

output "cluster_location" {
  description = "Standing GKE cluster location (null when create = false)."
  value       = var.create ? google_container_cluster.this[0].location : null
}
