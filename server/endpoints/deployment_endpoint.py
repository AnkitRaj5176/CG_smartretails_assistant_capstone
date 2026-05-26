"""
deployment_endpoint.py
──────────────────────
Section G: Final Deployment — REST Endpoints

Shows deployment configuration, Docker status, CI/CD pipeline,
and Azure infrastructure details.

Endpoints:
  GET /api/deployment/status    — Full deployment configuration
  GET /api/deployment/docker    — Docker configuration details
  GET /api/deployment/cicd      — CI/CD pipeline details
  GET /api/deployment/azure     — Azure infrastructure summary
"""

import logging
import os
import sys

from fastapi import APIRouter, status

logger = logging.getLogger(__name__)

deployment_router = APIRouter(tags=["G. Final Deployment"])

_project_root = os.getcwd()


def _file_exists(relative_path: str) -> bool:
    return os.path.exists(os.path.join(_project_root, relative_path))


# ── 1. Full Deployment Status ─────────────────────────────────────────────────

@deployment_router.get("/api/deployment/status", status_code=status.HTTP_200_OK)
async def get_deployment_status() -> dict:
    """
    Complete deployment status — Docker, CI/CD, Azure infrastructure.
    Section G: Final Deployment (Capstone requirement).
    """
    return {
        "project": "Smart Retail Analytics Engine",
        "version": "2.0.0",
        "capstone_section": "G. Final Deployment",

        "deployment_options": {
            "local": {
                "command": "python -m uvicorn server.startup:retail_application --host 0.0.0.0 --port 8000",
                "url": "http://localhost:8000",
                "swagger": "http://localhost:8000/docs",
                "status": "running",
            },
            "docker": {
                "command": "docker-compose up --build",
                "dockerfile": "Dockerfile",
                "compose_file": "docker-compose.yml",
                "image": "retail-engine:latest",
                "port": 8000,
                "files_exist": {
                    "Dockerfile": _file_exists("Dockerfile"),
                    "docker-compose.yml": _file_exists("docker-compose.yml"),
                },
            },
            "azure_web_app": {
                "app_name": "smart-retail-engine",
                "url": "https://smart-retail-engine.azurewebsites.net",
                "swagger": "https://smart-retail-engine.azurewebsites.net/docs",
                "config_file": "azure/azure-deploy.yml",
                "deploy_script_windows": "azure/deploy_windows.ps1",
                "deploy_script_linux": "azure/deploy.sh",
                "files_exist": {
                    "azure/azure-deploy.yml": _file_exists("azure/azure-deploy.yml"),
                    "azure/deploy_windows.ps1": _file_exists("azure/deploy_windows.ps1"),
                    "azure/deploy.sh": _file_exists("azure/deploy.sh"),
                },
            },
        },

        "cicd_pipeline": {
            "provider": "GitHub Actions",
            "workflow_file": ".github/workflows/ci-cd.yml",
            "file_exists": _file_exists(".github/workflows/ci-cd.yml"),
            "stages": [
                "1. Run pytest (65 tests)",
                "2. Build Docker image",
                "3. Push to GitHub Container Registry",
                "4. Deploy to Azure Web App",
            ],
            "triggers": ["push to main", "push to develop", "pull_request to main"],
        },

        "azure_infrastructure": {
            "web_app": "Azure Web App (Container) — B2 Linux",
            "database": "Azure Cosmos DB (MongoDB API 4.2)",
            "secrets": "Azure Key Vault",
            "ai_services": [
                "Azure OpenAI (GPT-4o)",
                "Azure Cognitive Services — Text Analytics",
                "Azure AI Search",
                "Azure ML Workspace",
            ],
            "data_engineering": [
                "Azure Data Factory",
                "Azure Databricks (PySpark)",
                "Azure Fabric (Lakehouse)",
            ],
        },

        "security": {
            "secrets_management": "Azure Key Vault",
            "https_only": True,
            "docker_user": "non-root (appuser)",
            "input_validation": "Pydantic models on all endpoints",
            "no_secrets_in_code": True,
        },
    }


# ── 2. Docker Details ─────────────────────────────────────────────────────────

@deployment_router.get("/api/deployment/docker", status_code=status.HTTP_200_OK)
async def get_docker_details() -> dict:
    """Docker configuration and container details."""

    dockerfile_content = ""
    compose_content = ""

    try:
        with open(os.path.join(_project_root, "Dockerfile"), "r") as f:
            dockerfile_content = f.read()
    except Exception:
        dockerfile_content = "Dockerfile not found"

    try:
        with open(os.path.join(_project_root, "docker-compose.yml"), "r") as f:
            compose_content = f.read()
    except Exception:
        compose_content = "docker-compose.yml not found"

    return {
        "docker_image": {
            "base_image": "python:3.11-slim",
            "build_stages": ["base", "deps", "final"],
            "exposed_port": 8000,
            "workers": 2,
            "user": "appuser (non-root)",
            "health_check": "GET /ping every 30s",
        },
        "docker_compose_services": {
            "retail_api": {
                "port": "8000:8000",
                "depends_on": "mongo",
                "volumes": ["sales_data:/app/sales_store", "model_data:/app/model_vault"],
            },
            "mongo": {
                "image": "mongo:7.0",
                "port": "27017:27017",
                "health_check": "mongosh ping",
            },
        },
        "run_commands": {
            "build_and_run": "docker-compose up --build",
            "run_only": "docker-compose up",
            "stop": "docker-compose down",
            "rebuild": "docker-compose up --build --force-recreate",
        },
        "dockerfile": dockerfile_content,
        "docker_compose": compose_content,
    }


# ── 3. CI/CD Details ──────────────────────────────────────────────────────────

@deployment_router.get("/api/deployment/cicd", status_code=status.HTTP_200_OK)
async def get_cicd_details() -> dict:
    """GitHub Actions CI/CD pipeline configuration."""

    workflow_content = ""
    try:
        with open(os.path.join(_project_root, ".github", "workflows", "ci-cd.yml"), "r") as f:
            workflow_content = f.read()
    except Exception:
        workflow_content = "Workflow file not found"

    return {
        "provider": "GitHub Actions",
        "workflow_file": ".github/workflows/ci-cd.yml",
        "jobs": {
            "test": {
                "name": "Run Tests (pytest)",
                "runs_on": "ubuntu-latest",
                "services": ["MongoDB 7.0"],
                "steps": [
                    "Checkout code",
                    "Setup Python 3.11",
                    "Install requirements",
                    "Generate sample data",
                    "Run pytest (65 tests)",
                    "Upload test results",
                ],
            },
            "build": {
                "name": "Build & Push Docker Image",
                "needs": "test",
                "trigger": "push to main only",
                "steps": [
                    "Login to GitHub Container Registry",
                    "Extract Docker metadata (tags)",
                    "Build Docker image",
                    "Push to ghcr.io",
                ],
            },
            "deploy": {
                "name": "Deploy to Azure Web App",
                "needs": "build",
                "trigger": "push to main only",
                "steps": [
                    "Deploy container to Azure Web App",
                    "Verify health check (/ping)",
                ],
            },
            "pipeline": {
                "name": "Run Data Engineering Pipeline",
                "trigger": "scheduled (daily 2 AM UTC)",
                "steps": [
                    "Run retail_pipeline.py",
                    "Upload pipeline outputs as artifacts",
                ],
            },
        },
        "required_secrets": [
            "AZURE_WEBAPP_NAME",
            "AZURE_WEBAPP_PUBLISH_PROFILE",
            "GITHUB_TOKEN (auto-provided)",
        ],
        "workflow_yaml": workflow_content,
    }


# ── 4. Azure Infrastructure ───────────────────────────────────────────────────

@deployment_router.get("/api/deployment/azure", status_code=status.HTTP_200_OK)
async def get_azure_infrastructure() -> dict:
    """Azure infrastructure summary and deployment guide."""
    return {
        "resource_group": "rg-retail-engine",
        "location": "eastus",

        "resources": {
            "azure_web_app": {
                "type": "Microsoft.Web/sites",
                "name": "smart-retail-engine",
                "sku": "B2 Linux",
                "url": "https://smart-retail-engine.azurewebsites.net",
                "swagger": "https://smart-retail-engine.azurewebsites.net/docs",
                "deployment_method": "Docker container from GitHub Container Registry",
            },
            "azure_cosmos_db": {
                "type": "Microsoft.DocumentDB/databaseAccounts",
                "name": "retail-cosmos-db",
                "api": "MongoDB 4.2",
                "database": "smart_retail",
                "collection": "retail_records",
                "indexes": ["product_id+date", "store_id", "category", "region"],
            },
            "azure_key_vault": {
                "type": "Microsoft.KeyVault/vaults",
                "name": "retail-engine-kv",
                "secrets": [
                    "MongoConnection",
                    "AzureOpenAIKey",
                    "AzureOpenAIEndpoint",
                    "TextAnalyticsKey",
                    "SearchKey",
                ],
            },
            "azure_openai": {
                "type": "Microsoft.CognitiveServices/accounts",
                "kind": "OpenAI",
                "deployment": "gpt-4o",
                "used_for": "Multi-agent natural language generation",
            },
            "azure_text_analytics": {
                "type": "Microsoft.CognitiveServices/accounts",
                "kind": "TextAnalytics",
                "used_for": "Sentiment analysis, key phrase extraction, language detection",
            },
            "azure_ai_search": {
                "type": "Microsoft.Search/searchServices",
                "index": "retail-policy-index",
                "used_for": "Semantic search over retail policy documents",
            },
            "azure_ml": {
                "type": "Microsoft.MachineLearningServices/workspaces",
                "name": "retail-ml-workspace",
                "used_for": "Model registry, training jobs, online endpoints",
            },
        },

        "deployment_steps": [
            "1. Install Azure CLI",
            "2. Run: az login",
            "3. Run: .\\azure\\deploy_windows.ps1 (Windows) or bash azure/deploy.sh (Linux/Mac)",
            "4. Push code to GitHub main branch",
            "5. GitHub Actions automatically deploys to Azure Web App",
        ],

        "config_files": {
            "azure_deploy_yml": "azure/azure-deploy.yml",
            "deploy_script_windows": "azure/deploy_windows.ps1",
            "deploy_script_linux": "azure/deploy.sh",
            "cicd_workflow": ".github/workflows/ci-cd.yml",
            "dockerfile": "Dockerfile",
            "docker_compose": "docker-compose.yml",
        },
    }
