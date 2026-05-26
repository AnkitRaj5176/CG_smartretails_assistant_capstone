"""
forecast_agent.py
─────────────────
ML Expert Agent — interprets demand forecasting and anomaly detection outputs,
then generates actionable business insights via Azure OpenAI.
Uses AgentTools for structured ML data retrieval.
"""

import logging

from server.genai.llm_client import call_llm
from server.genai.agent_tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

FORECAST_SYSTEM_PROMPT = """You are an expert retail demand forecasting and analytics AI.
Your role is to interpret ML model outputs and anomaly detection results, then provide
clear, actionable business recommendations.
Always:
- Explain what the numbers mean in plain business language.
- Identify risks (e.g. stockouts, overstock, unusual demand patterns).
- Suggest 1-2 concrete actions the retail team should take.
- Keep your response under 250 words.
- Do NOT make up predictions — only interpret the data provided."""


def run_forecast_agent(user_message: str) -> dict:
    """
    Execute the Forecast Agent pipeline:
    1. Select best ML tool based on user message.
    2. Execute tool to get ML context.
    3. Pass context + user question to Azure OpenAI for interpretation.
    4. Return structured response.
    """
    msg_lower = user_message.lower()

    # Step 1 — select tool
    if any(w in msg_lower for w in ["anomaly", "spike", "outlier", "unusual", "detection"]):
        tool_name = "get_anomaly_summary"
        tool_params = {}
    elif any(w in msg_lower for w in ["model", "algorithm", "feature", "training", "accuracy"]):
        tool_name = "get_model_info"
        tool_params = {}
    else:
        tool_name = "get_demand_forecast"
        tool_params = {
            "product_id": "PROD_001",
            "target_date": "2025-07-01",
            "price": 100.0,
            "discount": 0.0,
            "store_id": "STORE_A",
            "region": "North",
        }

    # Step 2 — execute tool
    raw_data = TOOL_REGISTRY.run(tool_name, tool_params)
    logger.info("ForecastAgent using tool: %s", tool_name)

    # Step 3 — LLM interpretation
    llm_response = call_llm(
        system_prompt=FORECAST_SYSTEM_PROMPT,
        user_message=user_message,
        context=raw_data,
        max_tokens=450,
        temperature=0.3,
    )

    return {
        "agent": "ForecastAgent",
        "agent_description": "Demand forecasting and predictive analytics specialist",
        "tool_used": tool_name,
        "raw_data": raw_data,
        "response": llm_response,
    }
