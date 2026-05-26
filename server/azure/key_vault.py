"""
key_vault.py
────────────
Azure Key Vault integration — Security layer.

Loads secrets from Azure Key Vault at runtime instead of
storing them in environment variables or config files.

Secret names in Key Vault:
  - MongoConnection       → MONGO_CONNECTION
  - AzureOpenAIKey        → AZURE_OPENAI_KEY
  - AzureOpenAIEndpoint   → AZURE_OPENAI_ENDPOINT
  - TextAnalyticsKey      → AZURE_TEXT_ANALYTICS_KEY
  - SearchKey             → AZURE_SEARCH_KEY

Falls back to environment variables when Key Vault not configured.
"""

import logging
import os

from server.env_config import env_settings

logger = logging.getLogger(__name__)

_kv_client = None

# Mapping: Key Vault secret name → env var name
SECRET_MAP = {
    "MongoConnection":       "MONGO_CONNECTION",
    "AzureOpenAIKey":        "AZURE_OPENAI_KEY",
    "AzureOpenAIEndpoint":   "AZURE_OPENAI_ENDPOINT",
    "TextAnalyticsKey":      "AZURE_TEXT_ANALYTICS_KEY",
    "TextAnalyticsEndpoint": "AZURE_TEXT_ANALYTICS_ENDPOINT",
    "SearchKey":             "AZURE_SEARCH_KEY",
    "SearchEndpoint":        "AZURE_SEARCH_ENDPOINT",
}


def _get_kv_client():
    """Return cached Key Vault SecretClient or None."""
    global _kv_client
    if _kv_client is not None:
        return _kv_client
    if not env_settings.USE_AZURE_KEYVAULT:
        return None
    try:
        from azure.keyvault.secrets import SecretClient         # type: ignore
        from azure.identity import DefaultAzureCredential       # type: ignore
        _kv_client = SecretClient(
            vault_url=env_settings.AZURE_KEYVAULT_URL,
            credential=DefaultAzureCredential(),
        )
        logger.info("Azure Key Vault client initialised: %s", env_settings.AZURE_KEYVAULT_URL)
        return _kv_client
    except ImportError:
        logger.warning("azure-keyvault-secrets not installed.")
        return None
    except Exception as e:
        logger.warning("Key Vault client init failed: %s", e)
        return None


def get_secret(secret_name: str, default: str = "") -> str:
    """
    Retrieve a secret from Azure Key Vault.
    Falls back to environment variable if Key Vault unavailable.
    """
    client = _get_kv_client()
    if client:
        try:
            secret = client.get_secret(secret_name)
            logger.debug("Secret '%s' loaded from Key Vault.", secret_name)
            return secret.value or default
        except Exception as e:
            logger.warning("Key Vault get_secret('%s') failed: %s", secret_name, e)

    # Fallback: read from environment variable
    env_var = SECRET_MAP.get(secret_name, secret_name.upper().replace("-", "_"))
    value = os.getenv(env_var, default)
    logger.debug("Secret '%s' loaded from env var '%s'.", secret_name, env_var)
    return value


def load_all_secrets() -> dict:
    """
    Load all mapped secrets from Key Vault and inject into environment.
    Call this at application startup before EnvConfig is used.
    Returns dict of {secret_name: loaded_successfully}.
    """
    results = {}
    client = _get_kv_client()

    if not client:
        logger.info("Key Vault not configured — using environment variables for all secrets.")
        return {name: False for name in SECRET_MAP}

    for secret_name, env_var in SECRET_MAP.items():
        try:
            secret = client.get_secret(secret_name)
            if secret.value:
                os.environ[env_var] = secret.value
                results[secret_name] = True
                logger.info("Loaded secret '%s' from Key Vault.", secret_name)
            else:
                results[secret_name] = False
        except Exception as e:
            logger.warning("Could not load secret '%s': %s", secret_name, e)
            results[secret_name] = False

    return results


def get_vault_status() -> dict:
    """Return Key Vault connection status."""
    client = _get_kv_client()
    if not client:
        return {
            "configured": False,
            "vault_url": "N/A",
            "message": "Set USE_AZURE_KEYVAULT=true and AZURE_KEYVAULT_URL in .env",
        }
    try:
        # Try listing secrets to verify connectivity
        secrets = list(client.list_properties_of_secrets())
        return {
            "configured": True,
            "vault_url": env_settings.AZURE_KEYVAULT_URL,
            "secret_count": len(secrets),
            "status": "connected",
        }
    except Exception as e:
        return {
            "configured": True,
            "vault_url": env_settings.AZURE_KEYVAULT_URL,
            "status": "error",
            "detail": str(e),
        }
