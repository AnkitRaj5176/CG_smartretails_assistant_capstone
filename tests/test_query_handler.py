"""Unit tests for orchestration/query_handler.py"""
import pytest
from server.orchestration.query_handler import (
    classify_query, process_query,
    ANALYTICS_AGENT, POLICY_AGENT, FORECAST_AGENT,
)


@pytest.mark.parametrize("message,expected_agent", [
    ("What is the total revenue this month?", ANALYTICS_AGENT),
    ("Show me the top 5 products by revenue", ANALYTICS_AGENT),
    ("What is the return policy for electronics?", POLICY_AGENT),
    ("What are the promotion rules?", POLICY_AGENT),
    ("Forecast demand for next week", FORECAST_AGENT),
    ("Predict units sold for PROD_001", FORECAST_AGENT),
    ("Show me anomaly detection results", FORECAST_AGENT),
])
def test_classify_query(message, expected_agent):
    assert classify_query(message) == expected_agent


def test_process_query_returns_required_keys():
    result = process_query("Show me the sales overview")
    for key in ("agent", "response", "tool_used", "message"):
        assert key in result


def test_process_query_agent_is_valid():
    result = process_query("What is the revenue by category?")
    assert result["agent"] in {ANALYTICS_AGENT, POLICY_AGENT, FORECAST_AGENT}


def test_process_query_response_is_string():
    result = process_query("Give me a sales summary")
    assert isinstance(result["response"], str)
