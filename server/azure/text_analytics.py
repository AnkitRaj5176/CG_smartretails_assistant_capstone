"""
text_analytics.py
─────────────────
Azure Cognitive Services — Text Analytics integration.

Features used:
  - Sentiment Analysis   : detect positive/negative/neutral sentiment in user queries
  - Key Phrase Extraction: extract key topics from user messages
  - Language Detection   : detect query language

Falls back to local rule-based analysis when Azure not configured.
"""

import logging
import re
from typing import Optional

from server.env_config import env_settings

logger = logging.getLogger(__name__)

_ta_client = None


def _get_client():
    """Return cached Azure Text Analytics client or None."""
    global _ta_client
    if _ta_client is not None:
        return _ta_client
    if not env_settings.USE_AZURE_TEXT_ANALYTICS:
        return None
    try:
        from azure.ai.textanalytics import TextAnalyticsClient  # type: ignore
        from azure.core.credentials import AzureKeyCredential  # type: ignore
        _ta_client = TextAnalyticsClient(
            endpoint=env_settings.AZURE_TEXT_ANALYTICS_ENDPOINT,
            credential=AzureKeyCredential(env_settings.AZURE_TEXT_ANALYTICS_KEY),
        )
        logger.info("Azure Text Analytics client initialised.")
        return _ta_client
    except ImportError:
        logger.warning("azure-ai-textanalytics not installed.")
        return None
    except Exception as e:
        logger.warning("Azure Text Analytics init failed: %s", e)
        return None


# ── Sentiment Analysis ─────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of the given text.
    Returns: {sentiment, confidence_scores, service}
    """
    client = _get_client()
    if client:
        try:
            result = client.analyze_sentiment([text])[0]
            return {
                "sentiment": result.sentiment,
                "confidence_scores": {
                    "positive": round(result.confidence_scores.positive, 3),
                    "neutral":  round(result.confidence_scores.neutral, 3),
                    "negative": round(result.confidence_scores.negative, 3),
                },
                "service": "azure_text_analytics",
            }
        except Exception as e:
            logger.warning("Sentiment analysis failed: %s", e)

    # ── Local fallback ─────────────────────────────────────────────────────────
    return _local_sentiment(text)


def _local_sentiment(text: str) -> dict:
    """Rule-based sentiment fallback."""
    text_lower = text.lower()
    positive_words = {"good", "great", "excellent", "best", "top", "high", "increase", "growth"}
    negative_words = {"bad", "poor", "low", "decrease", "drop", "loss", "problem", "issue", "anomaly"}
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    if pos > neg:
        sentiment = "positive"
        scores = {"positive": 0.8, "neutral": 0.15, "negative": 0.05}
    elif neg > pos:
        sentiment = "negative"
        scores = {"positive": 0.05, "neutral": 0.15, "negative": 0.8}
    else:
        sentiment = "neutral"
        scores = {"positive": 0.1, "neutral": 0.8, "negative": 0.1}
    return {"sentiment": sentiment, "confidence_scores": scores, "service": "local_fallback"}


# ── Key Phrase Extraction ──────────────────────────────────────────────────────

def extract_key_phrases(text: str) -> dict:
    """
    Extract key phrases from text.
    Returns: {key_phrases, service}
    """
    client = _get_client()
    if client:
        try:
            result = client.extract_key_phrases([text])[0]
            return {
                "key_phrases": list(result.key_phrases),
                "service": "azure_text_analytics",
            }
        except Exception as e:
            logger.warning("Key phrase extraction failed: %s", e)

    # ── Local fallback ─────────────────────────────────────────────────────────
    return _local_key_phrases(text)


def _local_key_phrases(text: str) -> dict:
    """Simple noun-phrase extraction fallback."""
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "have", "has", "do", "does", "will", "would", "can", "could",
                  "me", "my", "i", "you", "we", "they", "it", "this", "that",
                  "what", "how", "why", "when", "where", "which", "who",
                  "show", "give", "tell", "get", "find", "list"}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    phrases = [w for w in words if w not in stop_words]
    # Deduplicate preserving order
    seen = set()
    unique = [p for p in phrases if not (p in seen or seen.add(p))]
    return {"key_phrases": unique[:8], "service": "local_fallback"}


# ── Language Detection ─────────────────────────────────────────────────────────

def detect_language(text: str) -> dict:
    """
    Detect the language of the text.
    Returns: {language, iso6391_name, confidence, service}
    """
    client = _get_client()
    if client:
        try:
            result = client.detect_language([text])[0]
            return {
                "language": result.primary_language.name,
                "iso6391_name": result.primary_language.iso6391_name,
                "confidence": round(result.primary_language.confidence_score, 3),
                "service": "azure_text_analytics",
            }
        except Exception as e:
            logger.warning("Language detection failed: %s", e)

    return {"language": "English", "iso6391_name": "en",
            "confidence": 1.0, "service": "local_fallback"}
