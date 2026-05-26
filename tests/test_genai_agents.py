"""Unit tests for GenAI agents, vector store, agent tools, and MCP protocol."""
import pytest
from server.genai.analytics_agent import run_analytics_agent
from server.genai.policy_agent import run_policy_agent
from server.genai.forecast_agent import run_forecast_agent
from server.genai.llm_client import call_llm
from server.genai.agent_tools import TOOL_REGISTRY
from server.genai.mcp_protocol import MCPContext, MCPMessage, MCPOrchestrator
from server.retrieval.vector_store import VectorStore


# ── LLM Client ────────────────────────────────────────────────────────────────

def test_llm_client_offline_returns_string():
    result = call_llm(
        system_prompt="You are a helpful assistant.",
        user_message="What is the total revenue?",
        context="Total Revenue: $500,000",
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_llm_client_no_context_returns_string():
    result = call_llm(
        system_prompt="You are helpful.",
        user_message="Hello",
    )
    assert isinstance(result, str)


# ── Agent Tools ───────────────────────────────────────────────────────────────

def test_tool_registry_has_tools():
    tools = TOOL_REGISTRY.list_tools()
    assert len(tools) >= 10


def test_tool_registry_tool_names():
    names = TOOL_REGISTRY.tool_names()
    assert "get_sales_overview" in names
    assert "search_policy" in names
    assert "get_anomaly_summary" in names
    assert "get_demand_forecast" in names


def test_tool_sales_overview_returns_string():
    result = TOOL_REGISTRY.run("get_sales_overview", {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_tool_category_breakdown_returns_string():
    result = TOOL_REGISTRY.run("get_category_breakdown", {})
    assert isinstance(result, str)


def test_tool_policy_search_returns_string():
    result = TOOL_REGISTRY.run("search_policy", {"query": "return policy"})
    assert isinstance(result, str)
    assert len(result) > 5


def test_tool_unknown_returns_error():
    result = TOOL_REGISTRY.run("nonexistent_tool", {})
    assert "Unknown tool" in result


# ── Vector Store ──────────────────────────────────────────────────────────────

def test_vector_store_builds():
    store = VectorStore()
    store.build()
    assert store.is_ready
    assert store.chunk_count > 0


def test_vector_store_search_returns_results():
    store = VectorStore()
    store.build()
    results = store.search("return policy electronics", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]
    assert "source" in results[0]


def test_vector_store_scores_between_0_and_1():
    store = VectorStore()
    store.build()
    results = store.search("discount promotion", top_k=3)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_vector_store_get_context_returns_string():
    store = VectorStore()
    store.build()
    ctx = store.get_context("inventory stock management", top_k=2)
    assert isinstance(ctx, str)
    assert len(ctx) > 10


# ── MCP Protocol ──────────────────────────────────────────────────────────────

def test_mcp_message_creation():
    msg = MCPMessage(role="user", content="Hello", agent_id="user")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.message_id is not None
    assert msg.timestamp is not None


def test_mcp_message_to_dict():
    msg = MCPMessage(role="agent", content="Response", agent_id="AnalyticsAgent")
    d = msg.to_dict()
    assert d["role"] == "agent"
    assert d["agent_id"] == "AnalyticsAgent"
    assert "message_id" in d
    assert "timestamp" in d


def test_mcp_context_tracks_agents():
    ctx = MCPContext()
    ctx.add_user_message("What is revenue?")
    ctx.add_agent_message("AnalyticsAgent", "Here is the data.")
    assert "AnalyticsAgent" in ctx.agents_involved
    assert ctx.message_count == 2


def test_mcp_context_tracks_tool_calls():
    ctx = MCPContext()
    ctx.add_agent_message(
        "AnalyticsAgent", "Using tool.",
        tool_call={"tool_name": "get_sales_overview", "params": {}}
    )
    assert "get_sales_overview" in ctx.tools_called


def test_mcp_context_summary():
    ctx = MCPContext(session_id="test-123")
    ctx.add_user_message("Hello")
    summary = ctx.summary()
    assert summary["session_id"] == "test-123"
    assert "message_count" in summary


def test_mcp_orchestrator_process():
    orch = MCPOrchestrator()
    result = orch.process("What is the total revenue?")
    assert "agent" in result
    assert "response" in result
    assert "tool_used" in result
    assert "mcp_context" in result
    assert "session_id" in result


# ── GenAI Agents ──────────────────────────────────────────────────────────────

def test_analytics_agent_returns_required_keys():
    result = run_analytics_agent("What is the total revenue?")
    assert result["agent"] == "AnalyticsAgent"
    assert "response" in result
    assert "tool_used" in result
    assert isinstance(result["response"], str)


def test_policy_agent_returns_required_keys():
    result = run_policy_agent("What is the return policy?")
    assert result["agent"] == "PolicyAgent"
    assert "response" in result
    assert "sources" in result
    assert isinstance(result["response"], str)


def test_forecast_agent_returns_required_keys():
    result = run_forecast_agent("Explain the demand forecasting model")
    assert result["agent"] == "ForecastAgent"
    assert "response" in result
    assert isinstance(result["response"], str)


def test_policy_agent_response_not_empty():
    result = run_policy_agent("What is the discount policy?")
    assert len(result["response"]) > 10


def test_analytics_agent_response_not_empty():
    result = run_analytics_agent("Show me sales overview")
    assert len(result["response"]) > 10


def test_policy_agent_has_chunks():
    result = run_policy_agent("What is the return policy?")
    assert "chunks" in result
    assert isinstance(result["chunks"], list)
