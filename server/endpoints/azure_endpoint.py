"""
azure_endpoint.py — Section D: Azure AI & Cloud
"""
import logging
from fastapi import APIRouter
from server.env_config import env_settings

logger = logging.getLogger(__name__)
azure_router = APIRouter(prefix="/api/azure", tags=["D. Azure AI & Cloud"])


@azure_router.get("/status")
async def azure_services_status() -> dict:
    """Status of all Azure AI components — OpenAI, Text Analytics, AI Search, ML, Key Vault."""
    return {
        "azure_components": {
            "azure_openai":         {"enabled": env_settings.USE_AZURE_OPENAI,         "deployment": env_settings.AZURE_OPENAI_DEPLOYMENT,  "description": "GPT-4o for multi-agent NLG"},
            "azure_text_analytics": {"enabled": env_settings.USE_AZURE_TEXT_ANALYTICS, "description": "Sentiment analysis, key phrases, language detection"},
            "azure_ai_search":      {"enabled": env_settings.USE_AZURE_SEARCH,          "index": env_settings.AZURE_SEARCH_INDEX,            "description": "Semantic search over policy documents"},
            "azure_ml":             {"enabled": env_settings.USE_AZURE_ML,              "workspace": env_settings.AZURE_ML_WORKSPACE,        "description": "Model registry and training jobs"},
            "azure_key_vault":      {"enabled": env_settings.USE_AZURE_KEYVAULT,        "description": "Secure secret management"},
            "azure_web_app":        {"enabled": True,                                   "app_name": env_settings.AZURE_WEBAPP_NAME,          "description": "Container deployment via GitHub Actions"},
        },
        "note": "Set USE_*=true in .env to activate. All services have offline fallbacks.",
    }
