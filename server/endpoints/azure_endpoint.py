"""
azure_endpoint.py
─────────────────
REST endpoints for Azure AI & Cloud services.

Endpoints:
  GET  /api/azure/status              — All Azure services status
  POST /api/azure/text/sentiment      — Sentiment analysis (Text Analytics)
  POST /api/azure/text/keyphrases     — Key phrase extraction (Text Analytics)
  POST /api/azure/search              — Policy search (Azure AI Search)
  POST /api/azure/search/index        — Create/update search index
  GET  /api/azure/ml/status           — Azure ML deployment status
  POST /api/azure/ml/register         — Register model to Azure ML
  GET  /api/azure/keyvault/status     — Key Vault connection status
"""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from server.env_config import env_settings

logger = logging.getLogger(__name__)

azure_router = APIRouter(prefix="/api/azure", tags=["D. Azure AI & Cloud"])


# ── Request Models ─────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class RegisterModelRequest(BaseModel):
    model_name: str = "retail-demand-rf"
    version: str = "1"


# ── Status Overview ────────────────────────────────────────────────────────────

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
                "resource_group": env_settings.AZURE_ML_RESOURCE_GROUP,
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


# ── Text Analytics ─────────────────────────────────────────────────────────────

@azure_router.post("/text/sentiment", status_code=status.HTTP_200_OK)
async def analyze_sentiment(request_body: TextRequest) -> dict:
    """
    Analyze sentiment of text using Azure Cognitive Services Text Analytics.
    Falls back to local rule-based analysis when Azure not configured.
    """
    if not request_body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    try:
        from server.azure.text_analytics import analyze_sentiment as _analyze
        return _analyze(request_body.text)
    except Exception as e:
        logger.warning("Sentiment analysis error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@azure_router.post("/text/keyphrases", status_code=status.HTTP_200_OK)
async def extract_key_phrases(request_body: TextRequest) -> dict:
    """
    Extract key phrases using Azure Cognitive Services Text Analytics.
    Falls back to local extraction when Azure not configured.
    """
    if not request_body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    try:
        from server.azure.text_analytics import extract_key_phrases as _extract
        return _extract(request_body.text)
    except Exception as e:
        logger.warning("Key phrase extraction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@azure_router.post("/text/language", status_code=status.HTTP_200_OK)
async def detect_language(request_body: TextRequest) -> dict:
    """Detect language of text using Azure Text Analytics."""
    if not request_body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    try:
        from server.azure.text_analytics import detect_language as _detect
        return _detect(request_body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Azure AI Search ────────────────────────────────────────────────────────────

@azure_router.post("/search", status_code=status.HTTP_200_OK)
async def azure_search_policy(request_body: SearchRequest) -> dict:
    """
    Search retail policy documents using Azure AI Search.
    Falls back to local VectorStore when Azure Search not configured.
    """
    if not request_body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")
    try:
        from server.azure.ai_search import search_policy
        return search_policy(request_body.query, top_k=request_body.top_k)
    except Exception as e:
        logger.warning("Azure Search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@azure_router.post("/search/index", status_code=status.HTTP_200_OK)
async def create_search_index() -> dict:
    """Create/update the Azure AI Search index and upload policy documents."""
    try:
        from server.azure.ai_search import create_policy_index, upload_policy_documents
        index_result = create_policy_index()
        upload_result = upload_policy_documents()
        return {"index": index_result, "upload": upload_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Azure ML ───────────────────────────────────────────────────────────────────

@azure_router.get("/ml/status", status_code=status.HTTP_200_OK)
async def azure_ml_status() -> dict:
    """Return Azure ML deployment and model status."""
    from server.azure.azure_ml import get_deployment_status
    return get_deployment_status()


@azure_router.post("/ml/register", status_code=status.HTTP_200_OK)
async def register_model_to_azure_ml(request_body: RegisterModelRequest) -> dict:
    """Register the trained model to Azure ML Model Registry."""
    try:
        from server.azure.azure_ml import register_model
        return register_model(
            model_name=request_body.model_name,
            version=request_body.version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Key Vault ──────────────────────────────────────────────────────────────────

@azure_router.get("/keyvault/status", status_code=status.HTTP_200_OK)
async def key_vault_status() -> dict:
    """Return Azure Key Vault connection status."""
    from server.azure.key_vault import get_vault_status
    return get_vault_status()
