"""
azure_endpoint.py
─────────────────
Section D: Azure AI & Cloud Services
"""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from server.env_config import env_settings

logger = logging.getLogger(__name__)

azure_router = APIRouter(prefix="/api/azure", tags=["D. Azure AI & Cloud"])


class TextRequest(BaseModel):
    text: str


@azure_router.get("/status", status_code=status.HTTP_200_OK)
async def azure_services_status() -> dict:
    """Return the configuration status of all Azure services."""
    return {
        "azure_components": {
            "azure_openai": {
                "enabled": env_settings.USE_AZURE_OPENAI,
                "endpoint": env_settings.AZURE_OPENAI_ENDPOINT or "not configured",
                "deployment": env_settings.AZURE_OPENAI_DEPLOYMENT,
                "description": "GPT-4o for multi-agent natural language generation",
            },
            "azure_text_analytics": {
                "enabled": env_settings.USE_AZURE_TEXT_ANALYTICS,
                "endpoint": env_settings.AZURE_TEXT_ANALYTICS_ENDPOINT or "not configured",
                "description": "Sentiment analysis, key phrase extraction, language detection",
            },
            "azure_ai_search": {
                "enabled": env_settings.USE_AZURE_SEARCH,
                "endpoint": env_settings.AZURE_SEARCH_ENDPOINT or "not configured",
                "index": env_settings.AZURE_SEARCH_INDEX,
                "description": "Semantic search over retail policy documents",
            },
            "azure_ml": {
                "enabled": env_settings.USE_AZURE_ML,
                "workspace": env_settings.AZURE_ML_WORKSPACE,
                "description": "Model registry, training jobs, online endpoints",
            },
            "azure_key_vault": {
                "enabled": env_settings.USE_AZURE_KEYVAULT,
                "vault_url": env_settings.AZURE_KEYVAULT_URL or "not configured",
                "description": "Secure secret management for all API keys",
            },
            "azure_web_app": {
                "enabled": True,
                "app_name": env_settings.AZURE_WEBAPP_NAME,
                "description": "Container-based deployment via GitHub Actions CI/CD",
            },
        },
        "active_components": sum([
            env_settings.USE_AZURE_OPENAI,
            env_settings.USE_AZURE_TEXT_ANALYTICS,
            env_settings.USE_AZURE_SEARCH,
            env_settings.USE_AZURE_ML,
            env_settings.USE_AZURE_KEYVAULT,
        ]),
        "note": "Set USE_*=true in server/.env to activate each Azure service. "
                "All services have offline fallbacks.",
    }


@azure_router.post("/text/sentiment", status_code=status.HTTP_200_OK)
async def analyze_sentiment(request_body: TextRequest) -> dict:
    """
    Analyze sentiment using Azure Cognitive Services Text Analytics.
    Falls back to local rule-based analysis when Azure not configured.
    """
    if not request_body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    try:
        from server.azure.text_analytics import analyze_sentiment as _analyze
        return _analyze(request_body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
