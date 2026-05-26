#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Azure Deployment Script — Smart Retail Analytics Engine
# Capstone Project — Left Shift Program 2026 (Data & AI T5)
#
# Prerequisites:
#   1. Azure CLI installed: https://docs.microsoft.com/cli/azure/install-azure-cli
#   2. Docker installed and running
#   3. GitHub account with this repo pushed
#
# Run: bash azure/deploy.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ── Configuration ──────────────────────────────────────────────────────────────
RESOURCE_GROUP="rg-retail-engine"
LOCATION="eastus"
APP_NAME="smart-retail-engine"
APP_PLAN="retail-engine-plan"
KEYVAULT_NAME="retail-engine-kv"
COSMOS_ACCOUNT="retail-cosmos-db"
REGISTRY="ghcr.io"
IMAGE_NAME="retail-engine"
GITHUB_ORG="YOUR_GITHUB_USERNAME"

echo "=============================================="
echo " Smart Retail Analytics Engine — Azure Deploy"
echo "=============================================="

# ── Step 1: Login to Azure ─────────────────────────────────────────────────────
echo ""
echo "[1/8] Logging in to Azure..."
az login

# ── Step 2: Create Resource Group ─────────────────────────────────────────────
echo ""
echo "[2/8] Creating Resource Group: $RESOURCE_GROUP..."
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION"

# ── Step 3: Create Azure Key Vault ────────────────────────────────────────────
echo ""
echo "[3/8] Creating Azure Key Vault: $KEYVAULT_NAME..."
az keyvault create \
  --name "$KEYVAULT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --enable-soft-delete true

echo "  Adding secrets to Key Vault..."
az keyvault secret set --vault-name "$KEYVAULT_NAME" --name "MongoConnection"     --value "mongodb://localhost:27017"
az keyvault secret set --vault-name "$KEYVAULT_NAME" --name "AzureOpenAIKey"      --value "YOUR_AZURE_OPENAI_KEY"
az keyvault secret set --vault-name "$KEYVAULT_NAME" --name "AzureOpenAIEndpoint" --value "https://YOUR_RESOURCE.openai.azure.com/"

# ── Step 4: Create Cosmos DB (MongoDB API) ────────────────────────────────────
echo ""
echo "[4/8] Creating Azure Cosmos DB (MongoDB API): $COSMOS_ACCOUNT..."
az cosmosdb create \
  --name "$COSMOS_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --kind MongoDB \
  --server-version "4.2" \
  --locations regionName="$LOCATION" failoverPriority=0

az cosmosdb mongodb database create \
  --account-name "$COSMOS_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --name "smart_retail"

COSMOS_CONN=$(az cosmosdb keys list \
  --name "$COSMOS_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --type connection-strings \
  --query "connectionStrings[0].connectionString" -o tsv)

az keyvault secret set \
  --vault-name "$KEYVAULT_NAME" \
  --name "MongoConnection" \
  --value "$COSMOS_CONN"

echo "  Cosmos DB connection string saved to Key Vault."

# ── Step 5: Build and Push Docker Image ───────────────────────────────────────
echo ""
echo "[5/8] Building Docker image..."
docker build -t "$REGISTRY/$GITHUB_ORG/$IMAGE_NAME:latest" .

echo "  Pushing to GitHub Container Registry..."
echo "  (Make sure you are logged in: docker login ghcr.io -u $GITHUB_ORG)"
docker push "$REGISTRY/$GITHUB_ORG/$IMAGE_NAME:latest"

# ── Step 6: Create App Service Plan ───────────────────────────────────────────
echo ""
echo "[6/8] Creating App Service Plan: $APP_PLAN..."
az appservice plan create \
  --name "$APP_PLAN" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --is-linux \
  --sku B2

# ── Step 7: Create Web App ────────────────────────────────────────────────────
echo ""
echo "[7/8] Creating Azure Web App: $APP_NAME..."
az webapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_PLAN" \
  --deployment-container-image-name "$REGISTRY/$GITHUB_ORG/$IMAGE_NAME:latest"

echo "  Configuring app settings..."
az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
    WEBSITES_PORT=8000 \
    MONGO_DB_NAME=smart_retail \
    USE_AZURE_OPENAI=true \
    AZURE_OPENAI_DEPLOYMENT=gpt-4o \
    AZURE_OPENAI_API_VERSION=2024-02-01 \
    MONGO_CONNECTION="@Microsoft.KeyVault(VaultName=$KEYVAULT_NAME;SecretName=MongoConnection)" \
    AZURE_OPENAI_KEY="@Microsoft.KeyVault(VaultName=$KEYVAULT_NAME;SecretName=AzureOpenAIKey)" \
    AZURE_OPENAI_ENDPOINT="@Microsoft.KeyVault(VaultName=$KEYVAULT_NAME;SecretName=AzureOpenAIEndpoint)"

echo "  Enabling HTTPS only..."
az webapp update \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --https-only true

# ── Step 8: Verify Deployment ─────────────────────────────────────────────────
echo ""
echo "[8/8] Verifying deployment..."
sleep 30

APP_URL="https://$APP_NAME.azurewebsites.net"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$APP_URL/ping")

if [ "$HTTP_STATUS" = "200" ]; then
  echo "  ✅ Deployment successful!"
  echo "  App URL: $APP_URL"
  echo "  Swagger: $APP_URL/docs"
else
  echo "  ⚠️  App returned HTTP $HTTP_STATUS — may still be starting up."
  echo "  Check: $APP_URL/ping in a few minutes."
fi

echo ""
echo "=============================================="
echo " Deployment Complete!"
echo "=============================================="
echo " App URL  : https://$APP_NAME.azurewebsites.net"
echo " Swagger  : https://$APP_NAME.azurewebsites.net/docs"
echo " Key Vault: https://$KEYVAULT_NAME.vault.azure.net"
echo "=============================================="
