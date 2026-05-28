import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from server.infra.mongo_store import setup_mongo_indexes
from server.endpoints.upload_endpoint import upload_router
from server.endpoints.ml_endpoint import ml_router
from server.endpoints.chat_endpoint import chat_router
from server.endpoints.lookup_endpoint import lookup_router
from server.endpoints.azure_endpoint import azure_router
from server.endpoints.pipeline_endpoint import pipeline_router
from server.endpoints.powerbi_endpoint import powerbi_router
from server.endpoints.deployment_endpoint import deployment_router


# ── Swagger UI tag ordering (A → B → C → D → E → F) ──────────────────────────
OPENAPI_TAGS = [
    {
        "name": "A. Data Ingestion",
        "description": "Upload and validate retail CSV data. Persists to MongoDB.",
    },
    {
        "name": "B. Machine Learning / Deep Learning",
        "description": "Train RandomForest demand model, predict units sold, detect anomalies with IsolationForest.",
    },
    {
        "name": "C. GenAI & Multi-Agent System",
        "description": "3-agent system (Analytics, Policy, Forecast) with Azure OpenAI, TF-IDF RAG, Vector Store, and MCP Protocol.",
    },
    {
        "name": "D. Azure AI & Cloud",
        "description": "Azure OpenAI, Cognitive Services (Text Analytics, AI Search), Azure ML, Key Vault integration.",
    },
    {
        "name": "E. Data Engineering Pipeline",
        "description": "Azure Data Factory, Databricks PySpark, Azure Fabric Lakehouse — RAW → STAGED → CURATED data flow with Parquet storage.",
    },
    {
        "name": "F. Analytics & Visualization (Power BI)",
        "description": "Power BI dashboard data endpoints — key metrics, revenue charts, anomaly alerts, ML model performance.",
    },
    {
        "name": "F. Analytics & Visualization (Power BI)",
        "description": "Power BI dashboard data — key metrics, model outputs, anomaly alerts, revenue trends, agent insights.",
    },
    {
        "description": "Docker, GitHub Actions CI/CD, Azure Web App deployment configuration and status.",
    },
    {
        "name": "System",
        "description": "Health check and system status.",
    },
]


@asynccontextmanager
async def application_lifespan(application: FastAPI):
    """Run startup tasks before the application begins serving requests."""
    logger.info("Starting Smart Retail Analytics Engine...")
    # Run MongoDB setup in background — don't block startup
    try:
        setup_mongo_indexes()
    except Exception as e:
        logger.warning("MongoDB setup skipped (will retry on first use): %s", e)
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down Smart Retail Analytics Engine.")


retail_application = FastAPI(
    title="Smart Retail Analytics Engine",
    description=(
        "**Capstone Project — Left Shift Program 2026 (Data & AI T5)**\n\n"
        "End-to-end Multi-Agent AI Platform for retail demand forecasting, "
        "anomaly detection, policy Q&A, and business analytics.\n\n"
        "**Stack:** FastAPI · scikit-learn · Azure OpenAI · TF-IDF RAG · "
        "MongoDB · PySpark · Docker · GitHub Actions"
    ),
    version="2.0.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=application_lifespan,
)

retail_application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers in A → B → C → D → E → F → G order
retail_application.include_router(upload_router)      # A
retail_application.include_router(ml_router)          # B
retail_application.include_router(chat_router)        # C
retail_application.include_router(lookup_router)      # C
retail_application.include_router(azure_router)       # D
retail_application.include_router(pipeline_router)    # E
retail_application.include_router(powerbi_router)     # F
retail_application.include_router(deployment_router)  # G

@retail_application.get("/ping", tags=["System"])
async def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "alive", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.startup:retail_application",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
