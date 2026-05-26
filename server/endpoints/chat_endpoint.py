"""
chat_endpoint.py — Section C: GenAI & Multi-Agent System
"""
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from server.orchestration.query_handler import process_query
from server.orchestration.action_registry import get_all_actions

logger = logging.getLogger(__name__)
chat_router = APIRouter(tags=["C. GenAI & Multi-Agent System"])
_project_root = os.getcwd()
_RETAIL_CSV_PATH = os.path.join(_project_root, "sales_store", "retail_records.csv")


class ChatRequest(BaseModel):
    user_message: str


class MCPChatRequest(BaseModel):
    user_message: str
    session_id: Optional[str] = None


@chat_router.post("/api/assistant/chat")
async def chat_with_assistant(request_body: ChatRequest) -> dict:
    """Multi-agent chat — routes to Analytics, Policy, or Forecast agent."""
    if not request_body.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message must not be empty.")
    try:
        return process_query(request_body.user_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/api/assistant/actions")
async def list_available_actions() -> dict:
    """Return all registered agent actions."""
    return {"total_actions": len(get_all_actions()), "actions": get_all_actions()}


@chat_router.get("/api/metrics/overview")
async def get_metrics_overview() -> dict:
    """Dashboard KPIs — revenue, units, top products, category, region, monthly trend."""
    if not os.path.exists(_RETAIL_CSV_PATH):
        raise HTTPException(status_code=404, detail="No data. Upload CSV first.")
    try:
        from server.forecasting.csv_reader import read_retail_csv
        from server.forecasting.record_cleaner import clean_retail_records
        df = clean_retail_records(read_retail_csv(_RETAIL_CSV_PATH))
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
        return {
            "total_revenue": float(df["revenue"].sum()),
            "total_units_sold": float(df["units_sold"].sum()),
            "top_products": [{"product_id": k, "revenue": float(v)} for k, v in df.groupby("product_id")["revenue"].sum().sort_values(ascending=False).head(5).items()],
            "revenue_by_category": {k: float(v) for k, v in df.groupby("category")["revenue"].sum().sort_values(ascending=False).items()},
            "revenue_by_region": {k: float(v) for k, v in df.groupby("region")["revenue"].sum().sort_values(ascending=False).items()},
            "monthly_sales_trend": {k: float(v) for k, v in df.groupby("year_month")["revenue"].sum().sort_index().items()},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.post("/api/agent/mcp")
async def mcp_agent_chat(request_body: MCPChatRequest) -> dict:
    """MCP Protocol — full agent trace with tool calls and results."""
    if not request_body.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message must not be empty.")
    try:
        from server.genai.mcp_protocol import get_mcp_orchestrator
        return get_mcp_orchestrator().process(
            user_message=request_body.user_message,
            session_id=request_body.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
