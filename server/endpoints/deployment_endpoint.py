"""
deployment_endpoint.py
──────────────────────
Section G: Final Deployment
"""

import logging
import os

from fastapi import APIRouter, status

logger = logging.getLogger(__name__)

deployment_router = APIRouter(tags=["G. Final Deployment"])

_project_root = os.getcwd()


def _exists(rel: str) -> bool:
    return os.path.exists(os.path.join(_project_root, rel))


@deployment_router.get("/api/deployment/status", status_code=status.HTTP_200_OK)
async def get_deployment_status() -> dict:
    """Complete deployment status — Docker, CI/CD, Azure infrastructure."""
    return {
        "project": "Smart Retail Analytics Engine",
        "version": "2.0.0",
        "capstone_section": "G. Final Deployment",
        "deployment_options": {
            "local": {
                "command": "python -m uvicorn server.startup:retail_application --host 0.0.0.0 --port 8000",
                "url": "http://localhost:8000",
                "swagger": "http://localhost:8000/docs",
            },
            "docker": {
                "command": "docker-compose up --build",
                "files_exist": {
                    "Dockerfile": _exists("Dockerfile"),
                    "docker-compose.yml": _exists("docker-compose.yml"),
                },
            },
            "azure_web_app": {
                "app_name": "smart-retail-engine-2026",
                "url": "https://smart-retail-engine-2026-cbepgye8cbaqaga2.westeurope-01.azurewebsites.net",
                "swagger": "https://smart-retail-engine-2026-cbepgye8cbaqaga2.westeurope-01.azurewebsites.net/docs",
                "config_file": "azure/azure-deploy.yml",
                "files_exist": {
                    "azure/azure-deploy.yml": _exists("azure/azure-deploy.yml"),
                    "azure/deploy_windows.ps1": _exists("azure/deploy_windows.ps1"),
                },
            },
        },
        "cicd_pipeline": {
            "provider": "GitHub Actions",
            "workflow_file": ".github/workflows/ci-cd.yml",
            "file_exists": _exists(".github/workflows/ci-cd.yml"),
            "stages": [
                "1. Run pytest (65 tests)",
                "2. Build Docker image",
                "3. Push to GitHub Container Registry",
                "4. Deploy to Azure Web App",
            ],
        },
        "azure_infrastructure": {
            "web_app": "Azure Web App (Container) — B2 Linux",
            "database": "Azure Cosmos DB (MongoDB API 4.2)",
            "secrets": "Azure Key Vault",
            "ai_services": ["Azure OpenAI (GPT-4o)", "Azure Text Analytics", "Azure AI Search", "Azure ML"],
            "data_engineering": ["Azure Data Factory", "Azure Databricks (PySpark)", "Azure Fabric (Lakehouse)"],
        },
        "security": {
            "secrets_management": "Azure Key Vault",
            "https_only": True,
            "docker_user": "non-root (appuser)",
            "input_validation": "Pydantic models on all endpoints",
        },
    }
