import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_env_file_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_file_path)


class EnvConfig:
    """Holds all environment-based configuration for the retail engine."""

    def __init__(self) -> None:
        # ── MongoDB ────────────────────────────────────────────────────────────
        self.MONGO_CONNECTION: str = os.getenv("MONGO_CONNECTION", "mongodb://localhost:27017")
        self.MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "smart_retail")

        # ── Azure OpenAI (Component 1) ─────────────────────────────────────────
        self.AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
        self.AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.USE_AZURE_OPENAI: bool = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

        # ── Azure Cognitive Services — Text Analytics (Component 2) ───────────
        self.AZURE_TEXT_ANALYTICS_KEY: str = os.getenv("AZURE_TEXT_ANALYTICS_KEY", "")
        self.AZURE_TEXT_ANALYTICS_ENDPOINT: str = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT", "")
        self.USE_AZURE_TEXT_ANALYTICS: bool = os.getenv("USE_AZURE_TEXT_ANALYTICS", "false").lower() == "true"

        # ── Azure Cognitive Services — AI Search (Component 3) ────────────────
        self.AZURE_SEARCH_KEY: str = os.getenv("AZURE_SEARCH_KEY", "")
        self.AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
        self.AZURE_SEARCH_INDEX: str = os.getenv("AZURE_SEARCH_INDEX", "retail-policy-index")
        self.USE_AZURE_SEARCH: bool = os.getenv("USE_AZURE_SEARCH", "false").lower() == "true"

        # ── Azure ML (Component 4) ─────────────────────────────────────────────
        self.AZURE_ML_SUBSCRIPTION_ID: str = os.getenv("AZURE_ML_SUBSCRIPTION_ID", "")
        self.AZURE_ML_RESOURCE_GROUP: str = os.getenv("AZURE_ML_RESOURCE_GROUP", "rg-retail-engine")
        self.AZURE_ML_WORKSPACE: str = os.getenv("AZURE_ML_WORKSPACE", "retail-ml-workspace")
        self.USE_AZURE_ML: bool = os.getenv("USE_AZURE_ML", "false").lower() == "true"

        # ── Azure Key Vault (Security) ─────────────────────────────────────────
        self.AZURE_KEYVAULT_URL: str = os.getenv("AZURE_KEYVAULT_URL", "")
        self.USE_AZURE_KEYVAULT: bool = os.getenv("USE_AZURE_KEYVAULT", "false").lower() == "true"

        # ── Azure Web App ──────────────────────────────────────────────────────
        self.AZURE_WEBAPP_NAME: str = os.getenv("AZURE_WEBAPP_NAME", "smart-retail-engine")

        logger.info(
            "EnvConfig loaded | AzureOpenAI=%s | TextAnalytics=%s | Search=%s | ML=%s | KeyVault=%s",
            self.USE_AZURE_OPENAI,
            self.USE_AZURE_TEXT_ANALYTICS,
            self.USE_AZURE_SEARCH,
            self.USE_AZURE_ML,
            self.USE_AZURE_KEYVAULT,
        )


env_settings = EnvConfig()
