terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# --- Variables ---
variable "azure_location" {
  type        = string
  default     = "East US"
  description = "Azure Region for resources"
}

variable "resource_group_name" {
  type        = string
  default     = "epochdb-resources"
  description = "Name of the Resource Group"
}

variable "cluster_name" {
  type        = string
  default     = "epochdb-cluster"
  description = "AKS Cluster Name"
}

variable "node_vm_size" {
  type        = string
  default     = "Standard_DS2_v2"
  description = "Azure VM Instance size for AKS nodes"
}

# --- Resource Group ---
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.azure_location
}

# --- Network Infrastructure ---
resource "azurerm_virtual_network" "vnet" {
  name                = "epochdb-vnet"
  address_space       = ["10.0.0.0/8"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
}

resource "azurerm_subnet" "subnet" {
  name                 = "epochdb-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.240.0.0/16"]
}

# --- AKS Cluster ---
resource "azurerm_kubernetes_cluster" "main" {
  name                = var.cluster_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "epochdb"

  default_node_pool {
    name           = "default"
    node_count     = 3
    vm_size        = var.node_vm_size
    vnet_subnet_id = azurerm_subnet.subnet.id
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
  }
}

# --- Kubernetes Storage Class for Azure Disk ---
provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.main.kube_config[0].host
  client_certificate     = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_certificate)
  client_key             = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_key)
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].cluster_ca_certificate)
}

resource "kubernetes_storage_class" "azure_disk" {
  metadata {
    name = "managed-premium-ssd"
  }
  storage_class_provisions = "disk.csi.azure.com"
  volume_binding_mode      = "WaitForFirstConsumer"
  parameters = {
    storageaccounttype = "Premium_LRS"
    kind               = "Managed"
  }
}

# --- Outputs ---
output "cluster_name" {
  value       = azurerm_kubernetes_cluster.main.name
  description = "AKS Cluster Name"
}

output "configure_kubectl" {
  value       = "az aks get-credentials --resource-group ${var.resource_group_name} --name ${azurerm_kubernetes_cluster.main.name}"
  description = "Command to configure local kubectl to point to AKS"
}
