"""
query_handler.py
────────────────
Multi-agent orchestrator.

Flow:
  1. classify_query()  — keyword scoring → assign agent type
  2. Each agent runs its own pipeline:
       AnalyticsAgent  → action_registry data  → Azure OpenAI insight
       PolicyAgent     → TF-IDF RAG retrieval  → Azure OpenAI grounded answer
       ForecastAgent   → ML context            → Azure OpenAI interpretation
  3. Returns a unified structured response dict.
"""

import logging

from server.orchestration.action_registry import run_action

logger = logging.getLogger(__name__)

# ── Agent identifiers ──────────────────────────────────────────────────────────
ANALYTICS_AGENT: str = "AnalyticsAgent"
POLICY_AGENT: str = "PolicyAgent"
FORECAST_AGENT: str = "ForecastAgent"

# ── Keyword sets for classification ───────────────────────────────────────────
FORECAST_KEYWORDS: set[str] = {
    "forecast", "predict", "prediction", "demand", "future", "units",
    "how many will", "expected sales", "next week", "next month",
    "anomaly", "spike", "outlier", "unusual", "detection",
}

POLICY_KEYWORDS: set[str] = {
    "policy", "rule", "return", "refund", "promo", "promotion",
    "stock", "guideline", "procedure", "allowed", "permitted",
    "regulation", "compliance", "warranty", "exchange",
}

ANALYTICS_KEYWORDS: set[str] = {
    "revenue", "sales", "top", "category", "region", "monthly",
    "overview", "summary", "breakdown", "trend", "performance",
    "best", "highest", "earner", "product", "store", "total",
}

AGENT_ROLE_MAP: dict[str, str] = {
    ANALYTICS_AGENT: "Retail sales analytics and business intelligence specialist",
    POLICY_AGENT: "Retail policy and compliance document specialist",
    FORECAST_AGENT: "Demand forecasting and predictive analytics specialist",
}


def classify_query(message_text: str) -> str:
    """Classify the query into one of the three agent types based on keyword scoring."""
    normalized = message_text.lower()
    forecast_score = sum(1 for kw in FORECAST_KEYWORDS if kw in normalized)
    policy_score = sum(1 for kw in POLICY_KEYWORDS if kw in normalized)
    analytics_score = sum(1 for kw in ANALYTICS_KEYWORDS if kw in normalized)

    if forecast_score >= policy_score and forecast_score >= analytics_score and forecast_score > 0:
        return FORECAST_AGENT
    if policy_score >= analytics_score and policy_score > 0:
        return POLICY_AGENT
    return ANALYTICS_AGENT


def process_query(message_text: str) -> dict:
    """
    Classify the query, dispatch to the correct GenAI agent,
    and return a unified structured response.
    """
    assigned_agent = classify_query(message_text)
    logger.info("Routing query to agent: %s", assigned_agent)

    try:
        if assigned_agent == ANALYTICS_AGENT:
            from server.genai.analytics_agent import run_analytics_agent
            result = run_analytics_agent(message_text)

        elif assigned_agent == POLICY_AGENT:
            from server.genai.policy_agent import run_policy_agent
            result = run_policy_agent(message_text)

        else:  # FORECAST_AGENT
            from server.genai.forecast_agent import run_forecast_agent
            result = run_forecast_agent(message_text)

    except Exception as agent_error:
        logger.warning("Agent %s failed, falling back to action registry: %s", assigned_agent, agent_error)
        # Graceful fallback — always return something useful
        fallback = run_action(message_text)
        result = {
            "agent": assigned_agent,
            "agent_description": AGENT_ROLE_MAP[assigned_agent],
            "tool_used": fallback["action_name"],
            "raw_data": fallback["result"],
            "response": fallback["result"],
        }

    # Normalise output shape
    return {
        "message": message_text,
        "agent": result.get("agent", assigned_agent),
        "agent_description": result.get("agent_description", AGENT_ROLE_MAP[assigned_agent]),
        "tool_used": result.get("tool_used", "unknown"),
        "tool_description": result.get("tool_description", ""),
        "raw_data": result.get("raw_data", ""),
        "sources": result.get("sources", []),
        "response": result.get("response", ""),
    }
