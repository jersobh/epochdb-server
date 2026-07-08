terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# --- Variables ---
variable "gcp_project_id" {
  type        = string
  description = "The GCP Project ID to deploy resources to"
}

variable "gcp_region" {
  type        = string
  default     = "us-central1"
  description = "GCP Region for resources"
}

variable "cluster_name" {
  type        = string
  default     = "epochdb-cluster"
  description = "GKE Cluster Name"
}

variable "node_machine_type" {
  type        = string
  default     = "e2-medium"
  description = "GCE Instance machine type for GKE nodes"
}

# --- Network Infrastructure ---
resource "google_compute_network" "vpc" {
  name                    = "epochdb-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "epochdb-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.gcp_region
  network       = google_compute_network.vpc.id

  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = "10.10.0.0/16"
  }

  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = "10.20.0.0/20"
  }
}

# --- GKE Cluster ---
resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.gcp_region

  # We create a custom node pool separately, so delete default pool on creation
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "gke-pods"
    services_secondary_range_name = "gke-services"
  }

  # Enable network policy for cluster if needed
  network_policy {
    enabled = true
  }

  # Make it a regional cluster for HA, or zonal by replacing location with zone
  # For simplicity, location is set to the region.
}

# --- GKE Custom Node Pool ---
resource "google_container_node_pool" "nodes" {
  name       = "epochdb-node-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  node_count = 1 # 1 node per zone (typically 3 zones in a region = 3 nodes total)

  node_config {
    preemptible  = false
    machine_type = var.node_machine_type

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      app = "epochdb"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}

# --- Kubernetes Storage Class for pd-ssd ---
data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.main.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
}

resource "kubernetes_storage_class" "pd_ssd" {
  metadata {
    name = "pd-ssd"
  }
  storage_class_provisions = "pd.csi.storage.gke.io"
  volume_binding_mode      = "WaitForFirstConsumer"
  parameters = {
    type             = "pd-ssd"
    replication-type = "none"
  }
}

# --- Outputs ---
output "cluster_endpoint" {
  value       = google_container_cluster.main.endpoint
  description = "GKE Master endpoint"
}

output "configure_kubectl" {
  value       = "gcloud container clusters get-credentials ${google_container_cluster.main.name} --region ${var.gcp_region}"
  description = "Command to configure local kubectl to point to GKE"
}
