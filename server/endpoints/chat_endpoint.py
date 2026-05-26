import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from server.orchestration.query_handler import process_query
from server.orchestration.action_registry import get_all_actions

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["C. GenAI & Multi-Agent System"])

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RETAIL_CSV_PATH: str = os.path.join(_project_root, "sales_store", "retail_records.csv")


class ChatRequest(BaseModel):
    """Request body for the chat assistant endpoint."""
    user_message: str


class MCPChatRequest(BaseModel):
    """Request body for the MCP agent chat endpoint."""
    user_message: str
    session_id: Optional[str] = None


@chat_router.post("/api/assistant/chat", status_code=status.HTTP_200_OK)
async def chat_with_assistant(request_body: ChatRequest) -> dict:
    """Process a user message through the multi-agent query handler."""
    if not request_body.user_message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_message must not be empty.",
        )

    try:
        query_response = process_query(request_body.user_message)
    except Exception as query_error:
        logger.warning("Query processing failed: %s", query_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {query_error}",
        )

    return query_response


@chat_router.get("/api/assistant/actions", status_code=status.HTTP_200_OK)
async def list_available_actions() -> dict:
    """Return all registered agent actions."""
    all_actions = get_all_actions()
    return {"total_actions": len(all_actions), "actions": all_actions}


@chat_router.get("/api/metrics/overview", status_code=status.HTTP_200_OK)
async def get_metrics_overview() -> dict:
    """Return a dashboard overview of key retail metrics."""
    if not os.path.exists(_RETAIL_CSV_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No retail data available. Please upload a CSV file first.",
        )

    try:
        import pandas

        from server.forecasting.csv_reader import read_retail_csv
        from server.forecasting.record_cleaner import clean_retail_records

        raw_dataframe = read_retail_csv(_RETAIL_CSV_PATH)
        retail_dataframe = clean_retail_records(raw_dataframe)

        total_revenue = float(retail_dataframe["revenue"].sum())
        total_units_sold = float(retail_dataframe["units_sold"].sum())

        top_products_series = (
            retail_dataframe.groupby("product_id")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        top_products = [
            {"product_id": product_id, "revenue": float(product_revenue)}
            for product_id, product_revenue in top_products_series.items()
        ]

        category_revenue_series = (
            retail_dataframe.groupby("category")["revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        revenue_by_category = {
            category_name: float(category_total)
            for category_name, category_total in category_revenue_series.items()
        }

        region_revenue_series = (
            retail_dataframe.groupby("region")["revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        revenue_by_region = {
            region_name: float(region_total)
            for region_name, region_total in region_revenue_series.items()
        }

        retail_dataframe["year_month"] = retail_dataframe["date"].dt.to_period("M").astype(str)
        monthly_series = (
            retail_dataframe.groupby("year_month")["revenue"]
            .sum()
            .sort_index()
        )
        monthly_sales_trend = {
            month_label: float(month_total)
            for month_label, month_total in monthly_series.items()
        }

        return {
            "total_revenue": total_revenue,
            "total_units_sold": total_units_sold,
            "top_products": top_products,
            "revenue_by_category": revenue_by_category,
            "revenue_by_region": revenue_by_region,
            "monthly_sales_trend": monthly_sales_trend,
        }

    except Exception as metrics_error:
        logger.warning("Metrics overview failed: %s", metrics_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute metrics: {metrics_error}",
        )


@chat_router.post("/api/agent/mcp", status_code=status.HTTP_200_OK)
async def mcp_agent_chat(request_body: MCPChatRequest) -> dict:
    """
    MCP (Model Context Protocol) agent endpoint.
    Full trace: user message → agent classification → tool call → tool result → LLM response.
    Returns the complete MCP context trace for transparency.
    """
    if not request_body.user_message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_message must not be empty.",
        )
    try:
        from server.genai.mcp_protocol import get_mcp_orchestrator
        orchestrator = get_mcp_orchestrator()
        result = orchestrator.process(
            user_message=request_body.user_message,
            session_id=request_body.session_id,
        )
    except Exception as mcp_error:
        logger.warning("MCP processing failed: %s", mcp_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MCP processing failed: {mcp_error}",
        )
    return result


@chat_router.get("/api/agent/tools", status_code=status.HTTP_200_OK)
async def list_agent_tools() -> dict:
    """Return all registered agent tools with descriptions and parameters."""
    from server.genai.agent_tools import TOOL_REGISTRY
    tools = TOOL_REGISTRY.list_tools()
    return {"total_tools": len(tools), "tools": tools}


@chat_router.get("/api/agent/vector-store/status", status_code=status.HTTP_200_OK)
async def vector_store_status() -> dict:
    """Return the current status of the vector store (embeddings index)."""
    from server.retrieval.vector_store import get_vector_store
    store = get_vector_store()
    return {
        "is_ready": store.is_ready,
        "chunk_count": store.chunk_count,
        "embedding_type": "TF-IDF (bigrams, L2-normalised)",
        "doc_folder": "raw_docs/",
    }


@chat_router.post("/api/agent/vector-store/rebuild", status_code=status.HTTP_200_OK)
async def rebuild_vector_store() -> dict:
    """Force a full rebuild of the vector store embeddings index."""
    try:
        from server.retrieval.vector_store import rebuild_vector_store as _rebuild
        store = _rebuild()
        return {
            "status": "Vector store rebuilt successfully.",
            "chunk_count": store.chunk_count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rebuild failed: {e}",
        )
