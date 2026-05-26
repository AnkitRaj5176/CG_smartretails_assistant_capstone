"""
llm_client.py
─────────────
Thin wrapper around Azure OpenAI Chat Completions.

When USE_AZURE_OPENAI=false (default / no credentials) the client falls back
to a local rule-based response so the project runs fully offline.
"""

import logging
from typing import Optional

from server.env_config import env_settings

logger = logging.getLogger(__name__)


# ── Azure OpenAI client (lazy-loaded) ─────────────────────────────────────────

_azure_client = None


def _get_azure_client():
    """Return a cached AzureOpenAI client, or None if not configured."""
    global _azure_client
    if _azure_client is not None:
        return _azure_client

    if not env_settings.USE_AZURE_OPENAI:
        return None

    try:
        from openai import AzureOpenAI  # type: ignore

        _azure_client = AzureOpenAI(
            api_key=env_settings.AZURE_OPENAI_KEY,
            azure_endpoint=env_settings.AZURE_OPENAI_ENDPOINT,
            api_version=env_settings.AZURE_OPENAI_API_VERSION,
        )
        logger.info("Azure OpenAI client initialised.")
        return _azure_client
    except ImportError:
        logger.warning("openai package not installed — Azure OpenAI unavailable.")
        return None
    except Exception as init_error:
        logger.warning("Azure OpenAI client init failed: %s", init_error)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    user_message: str,
    context: str = "",
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """
    Send a chat completion request to Azure OpenAI.

    Falls back to a formatted local response when Azure OpenAI is disabled
    or unavailable, so the project always returns a meaningful answer.

    Parameters
    ----------
    system_prompt : str
        Role/instruction for the assistant.
    user_message : str
        The user's question or request.
    context : str
        Optional grounding context (e.g. retrieved data or policy chunks).
    max_tokens : int
        Maximum tokens in the completion.
    temperature : float
        Sampling temperature (0 = deterministic).

    Returns
    -------
    str
        The assistant's response text.
    """
    client = _get_azure_client()

    if client is not None:
        return _call_azure(client, system_prompt, user_message, context, max_tokens, temperature)

    # ── Offline fallback ───────────────────────────────────────────────────────
    logger.debug("LLM fallback: returning context-based response.")
    return _offline_response(system_prompt, user_message, context)


def _call_azure(
    client,
    system_prompt: str,
    user_message: str,
    context: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Make the actual Azure OpenAI API call."""
    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "system",
            "content": f"Use the following context to answer:\n\n{context}",
        })

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=env_settings.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = response.choices[0].message.content or ""
        logger.info(
            "Azure OpenAI responded — tokens used: %d",
            response.usage.total_tokens if response.usage else 0,
        )
        return answer.strip()
    except Exception as api_error:
        logger.warning("Azure OpenAI API call failed: %s", api_error)
        return _offline_response(system_prompt, user_message, context)


def _offline_response(system_prompt: str, user_message: str, context: str) -> str:
    """
    Rule-based fallback when Azure OpenAI is not available.
    Returns the context directly with a framing sentence.
    """
    if context:
        return (
            f"[Offline mode — Azure OpenAI not configured]\n\n"
            f"Based on available data:\n\n{context}"
        )
    return (
        f"[Offline mode — Azure OpenAI not configured]\n\n"
        f"I received your query: \"{user_message}\". "
        f"Please configure AZURE_OPENAI_KEY and set USE_AZURE_OPENAI=true in .env "
        f"to enable AI-generated responses."
    )
