terraform {
  required_providers {
    azurerm = {
      source = "hashicorp/azurerm"
      version = "~>2.0"
    }
  }
}

provider "azurerm" {
  features {}

  subscription_id   = "cb129d26-97d3-43a4-9782-7b2ca3e81cd4"
  tenant_id         = "9da468c6-2260-4466-9e2b-d0dbb15ef88b"
}

resource "azurerm_resource_group" "rsg" {
  name = "rsg_case2_mdl"
  location = "Central US"
}

resource "azurerm_databricks_workspace" "databricks" {
  name                = "databricks-case2"
  resource_group_name = azurerm_resource_group.rsg.name
  location            = azurerm_resource_group.rsg.location
  sku                 = "standard"
  tags = {
    Environment = "Case 2 - Streaming"
  }
}

resource "azurerm_eventhub_namespace" "namespace" {
  name                = "case2StreamNamespace"
  location            = azurerm_resource_group.rsg.location
  resource_group_name = azurerm_resource_group.rsg.name
  sku                 = "Standard"
  capacity            = 1

  tags = {
    environment = "Development"
  }
}

resource "azurerm_eventhub" "namespace" {
  name                = "dadosHeartEventHub"
  namespace_name      = azurerm_eventhub_namespace.namespace.name
  resource_group_name = azurerm_resource_group.rsg.name
  partition_count     = 1
  message_retention   = 1
}


