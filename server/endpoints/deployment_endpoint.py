"""
deployment_endpoint.py — Section G: Final Deployment
"""
import logging
import os
from fastapi import APIRouter, status

logger = logging.getLogger(__name__)
deployment_router = APIRouter(tags=["G. Final Deployment"])
_project_root = os.getcwd()


@deployment_router.get("/api/deployment/status")
async def get_deployment_status() -> dict:
    """Deployment status — Docker, GitHub Actions CI/CD, Azure Web App infrastructure."""
    def e(p): return os.path.exists(os.path.join(_project_root, p))
    return {
        "project": "Smart Retail Analytics Engine",
        "version": "2.0.0",
        "live_url": "https://smart-retail-engine-2026-cbepgye8cbaqaga2.westeurope-01.azurewebsites.net",
        "swagger":  "https://smart-retail-engine-2026-cbepgye8cbaqaga2.westeurope-01.azurewebsites.net/docs",
        "deployment_files": {
            "Dockerfile":            e("Dockerfile"),
            "docker-compose.yml":    e("docker-compose.yml"),
            "ci-cd.yml":             e(".github/workflows/ci-cd.yml"),
            "azure-deploy.yml":      e("azure/azure-deploy.yml"),
            "deploy_windows.ps1":    e("azure/deploy_windows.ps1"),
        },
        "cicd": {
            "provider": "GitHub Actions",
            "stages": ["pytest (65 tests)", "Docker build", "Push to GHCR", "Deploy to Azure Web App"],
        },
        "azure": {
            "web_app":    "Azure Web App — West Europe",
            "database":   "Azure Cosmos DB (MongoDB API 4.2)",
            "secrets":    "Azure Key Vault",
            "ai_services": ["Azure OpenAI GPT-4o", "Azure Text Analytics", "Azure AI Search", "Azure ML"],
        },
    }
