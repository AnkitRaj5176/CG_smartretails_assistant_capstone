"""
analytics_agent.py
──────────────────
Data Analyst Agent — answers questions from retail analytics data.
Uses AgentTools for structured data retrieval and Azure OpenAI
(or offline fallback) to generate natural-language insights.
"""

import logging

from server.genai.llm_client import call_llm
from server.genai.agent_tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

ANALYTICS_SYSTEM_PROMPT = """You are a senior retail data analyst AI assistant.
Your job is to analyse retail sales data and provide clear, concise, actionable insights.
Always:
- Highlight the most important numbers and trends.
- Suggest one concrete business recommendation based on the data.
- Keep your response under 200 words.
- Use bullet points for lists of metrics.
- Do NOT make up numbers — only use the data provided in the context."""

# Tool selection map: keyword → tool name
_TOOL_SELECTION = {
    "top": "get_top_products",
    "best": "get_top_products",
    "highest": "get_top_products",
    "category": "get_category_breakdown",
    "categories": "get_category_breakdown",
    "region": "get_region_breakdown",
    "area": "get_region_breakdown",
    "monthly": "get_monthly_trend",
    "month": "get_monthly_trend",
    "trend": "get_monthly_trend",
    "product": "get_product_detail",
    "item": "get_product_detail",
}


def _select_tool(user_message: str) -> tuple[str, dict]:
    """Select the best analytics tool based on message keywords."""
    msg_lower = user_message.lower()
    for keyword, tool_name in _TOOL_SELECTION.items():
        if keyword in msg_lower:
            params = {}
            if tool_name == "get_top_products":
                for token in msg_lower.split():
                    if token.isdigit():
                        params["top_n"] = int(token)
                        break
            elif tool_name == "get_product_detail":
                for token in user_message.upper().split():
                    if token.startswith("PROD") and len(token) >= 6:
                        params["product_id"] = token
                        break
            return tool_name, params
    return "get_sales_overview", {}


def run_analytics_agent(user_message: str) -> dict:
    """
    Execute the Analytics Agent pipeline:
    1. Select best tool from AgentTools registry.
    2. Execute tool to fetch structured sales data.
    3. Pass data + user question to Azure OpenAI for insight generation.
    4. Return structured response.
    """
    # Step 1 — select and run tool
    tool_name, tool_params = _select_tool(user_message)
    raw_data = TOOL_REGISTRY.run(tool_name, tool_params)

    logger.info("AnalyticsAgent using tool: %s", tool_name)

    # Step 2 — generate LLM insight
    llm_response = call_llm(
        system_prompt=ANALYTICS_SYSTEM_PROMPT,
        user_message=user_message,
        context=raw_data,
        max_tokens=400,
        temperature=0.2,
    )

    return {
        "agent": "AnalyticsAgent",
        "agent_description": "Retail sales analytics and business intelligence specialist",
        "tool_used": tool_name,
        "raw_data": raw_data,
        "response": llm_response,
    }
