# ===============================================================
# Azure Deployment Script (PowerShell) — Smart Retail Engine
# Capstone Project — Left Shift Program 2026 (Data & AI T5)
#
# Prerequisites:
#   1. Install Azure CLI: https://aka.ms/installazurecliwindows
#   2. Install Docker Desktop
#   3. Run: Set-ExecutionPolicy RemoteSigned
#
# Usage: .\azure\deploy_windows.ps1
# ===============================================================

$RESOURCE_GROUP  = "rg-retail-engine"
$LOCATION        = "eastus"
$APP_NAME        = "smart-retail-engine"
$APP_PLAN        = "retail-engine-plan"
$KEYVAULT_NAME   = "retail-engine-kv"
$COSMOS_ACCOUNT  = "retail-cosmos-db"
$GITHUB_ORG      = "YOUR_GITHUB_USERNAME"
$IMAGE            = "ghcr.io/$GITHUB_ORG/retail-engine:latest"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Smart Retail Analytics Engine — Azure Deploy  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Step 1 — Login
Write-Host "`n[1/8] Logging in to Azure..." -ForegroundColor Yellow
az login

# Step 2 — Resource Group
Write-Host "`n[2/8] Creating Resource Group..." -ForegroundColor Yellow
az group create --name $RESOURCE_GROUP --location $LOCATION

# Step 3 — Key Vault
Write-Host "`n[3/8] Creating Key Vault: $KEYVAULT_NAME..." -ForegroundColor Yellow
az keyvault create `
  --name $KEYVAULT_NAME `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --enable-soft-delete true

Write-Host "  Adding placeholder secrets (update with real values)..."
az keyvault secret set --vault-name $KEYVAULT_NAME --name "MongoConnection"     --value "mongodb://localhost:27017"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "AzureOpenAIKey"      --value "YOUR_OPENAI_KEY"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "AzureOpenAIEndpoint" --value "https://YOUR_RESOURCE.openai.azure.com/"

# Step 4 — Cosmos DB
Write-Host "`n[4/8] Creating Cosmos DB (MongoDB API)..." -ForegroundColor Yellow
az cosmosdb create `
  --name $COSMOS_ACCOUNT `
  --resource-group $RESOURCE_GROUP `
  --kind MongoDB `
  --server-version "4.2" `
  --locations regionName=$LOCATION failoverPriority=0

az cosmosdb mongodb database create `
  --account-name $COSMOS_ACCOUNT `
  --resource-group $RESOURCE_GROUP `
  --name "smart_retail"

$COSMOS_CONN = az cosmosdb keys list `
  --name $COSMOS_ACCOUNT `
  --resource-group $RESOURCE_GROUP `
  --type connection-strings `
  --query "connectionStrings[0].connectionString" -o tsv

az keyvault secret set --vault-name $KEYVAULT_NAME --name "MongoConnection" --value $COSMOS_CONN
Write-Host "  Cosmos DB connection saved to Key Vault." -ForegroundColor Green

# Step 5 — Docker Build & Push
Write-Host "`n[5/8] Building Docker image..." -ForegroundColor Yellow
docker build -t $IMAGE .
Write-Host "  Pushing to GitHub Container Registry..."
docker push $IMAGE

# Step 6 — App Service Plan
Write-Host "`n[6/8] Creating App Service Plan..." -ForegroundColor Yellow
az appservice plan create `
  --name $APP_PLAN `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --is-linux `
  --sku B2

# Step 7 — Web App
Write-Host "`n[7/8] Creating Azure Web App: $APP_NAME..." -ForegroundColor Yellow
az webapp create `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --plan $APP_PLAN `
  --deployment-container-image-name $IMAGE

az webapp config appsettings set `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --settings `
    WEBSITES_PORT=8000 `
    MONGO_DB_NAME=smart_retail `
    USE_AZURE_OPENAI=true `
    AZURE_OPENAI_DEPLOYMENT=gpt-4o `
    AZURE_OPENAI_API_VERSION=2024-02-01 `
    "MONGO_CONNECTION=@Microsoft.KeyVault(VaultName=$KEYVAULT_NAME;SecretName=MongoConnection)" `
    "AZURE_OPENAI_KEY=@Microsoft.KeyVault(VaultName=$KEYVAULT_NAME;SecretName=AzureOpenAIKey)" `
    "AZURE_OPENAI_ENDPOINT=@Microsoft.KeyVault(VaultName=$KEYVAULT_NAME;SecretName=AzureOpenAIEndpoint)"

az webapp update --name $APP_NAME --resource-group $RESOURCE_GROUP --https-only true

# Step 8 — Verify
Write-Host "`n[8/8] Waiting for app to start (30s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

$APP_URL = "https://$APP_NAME.azurewebsites.net"
try {
    $response = Invoke-WebRequest -Uri "$APP_URL/ping" -UseBasicParsing -TimeoutSec 15
    if ($response.StatusCode -eq 200) {
        Write-Host "  Deployment successful!" -ForegroundColor Green
    }
} catch {
    Write-Host "  App may still be starting. Check $APP_URL/ping in a few minutes." -ForegroundColor Yellow
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host " Deployment Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " App URL  : https://$APP_NAME.azurewebsites.net"
Write-Host " Swagger  : https://$APP_NAME.azurewebsites.net/docs"
Write-Host " Key Vault: https://$KEYVAULT_NAME.vault.azure.net"
Write-Host "================================================" -ForegroundColor Cyan
