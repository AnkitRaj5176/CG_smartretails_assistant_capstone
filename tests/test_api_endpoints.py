"""Integration tests for all FastAPI endpoints using TestClient."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from server.startup import retail_application

client = TestClient(retail_application, raise_server_exceptions=False)

VALID_CSV = (
    "product_id,category,region,store_id,date,price,discount,units_sold,revenue\n"
    "PROD_001,Electronics,North,STORE_A,2024-01-10,299.99,10.0,5,1349.96\n"
    "PROD_002,Clothing,South,STORE_B,2024-02-15,49.99,5.0,12,569.89\n"
    "PROD_003,Groceries,East,STORE_C,2024-03-20,12.50,0.0,30,375.00\n"
    "PROD_001,Electronics,West,STORE_A,2024-04-05,299.99,20.0,8,1919.94\n"
    "PROD_002,Clothing,Central,STORE_B,2024-05-12,49.99,15.0,20,849.83\n"
)

# ── Health ─────────────────────────────────────────────────────────────────────

def test_ping():
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"

# ── Upload ─────────────────────────────────────────────────────────────────────

def test_upload_rejects_non_csv():
    r = client.post("/api/data/upload",
                    files={"file": ("data.txt", b"text", "text/plain")})
    assert r.status_code == 400


def test_upload_rejects_missing_columns():
    r = client.post("/api/data/upload",
                    files={"file": ("data.csv", b"col1,col2\n1,2\n", "text/csv")})
    assert r.status_code in (400, 422)


def test_upload_valid_csv():
    r = client.post("/api/data/upload",
                    files={"file": ("retail.csv", VALID_CSV.encode(), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert "cleaned_row_count" in body
    assert body["cleaned_row_count"] > 0

# ── Chat ───────────────────────────────────────────────────────────────────────

def test_chat_empty_message():
    r = client.post("/api/assistant/chat", json={"user_message": "  "})
    assert r.status_code == 400


def test_chat_valid_message():
    r = client.post("/api/assistant/chat", json={"user_message": "Give me a sales overview"})
    assert r.status_code == 200
    body = r.json()
    assert "agent" in body
    assert "response" in body


def test_chat_policy_query():
    r = client.post("/api/assistant/chat", json={"user_message": "What is the return policy?"})
    assert r.status_code == 200
    assert r.json()["agent"] == "PolicyAgent"


def test_chat_forecast_query():
    r = client.post("/api/assistant/chat", json={"user_message": "Forecast demand for next week"})
    assert r.status_code == 200
    assert r.json()["agent"] == "ForecastAgent"

# ── Lookup ─────────────────────────────────────────────────────────────────────

def test_lookup_empty_query():
    r = client.post("/api/docs/search", json={"search_query": ""})
    assert r.status_code == 400


def test_lookup_valid_query():
    r = client.post("/api/docs/search", json={"search_query": "return policy for electronics"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "query" in body

# ── Metrics ────────────────────────────────────────────────────────────────────

def test_metrics_overview_after_upload():
    client.post("/api/data/upload",
                files={"file": ("retail.csv", VALID_CSV.encode(), "text/csv")})
    r = client.get("/api/metrics/overview")
    assert r.status_code == 200
    body = r.json()
    assert "total_revenue" in body
    assert "top_products" in body

# ── ML Endpoints ───────────────────────────────────────────────────────────────

def test_train_with_data():
    client.post("/api/data/upload",
                files={"file": ("retail.csv", VALID_CSV.encode(), "text/csv")})
    r = client.post("/api/ml/train", json={"num_trees": 10, "holdout_ratio": 0.2})
    assert r.status_code == 200
    body = r.json()
    assert "mae" in body
    assert "r2" in body


def test_predict_after_training():
    r = client.post("/api/ml/predict", json={
        "product_id": "PROD_001",
        "target_date": "2025-06-01",
        "price": 299.99,
        "discount": 10.0,
        "store_id": "STORE_A",
        "region": "North",
    })
    assert r.status_code in (200, 404)


def test_train_without_data_returns_404():
    import server.endpoints.ml_endpoint as ml_mod
    original_path = ml_mod._RETAIL_CSV_PATH
    ml_mod._RETAIL_CSV_PATH = "/nonexistent/path/retail_records.csv"
    try:
        r = client.post("/api/ml/train", json={})
        assert r.status_code == 404
    finally:
        ml_mod._RETAIL_CSV_PATH = original_path

# ── Azure ──────────────────────────────────────────────────────────────────────

def test_azure_status():
    r = client.get("/api/azure/status")
    assert r.status_code == 200
    body = r.json()
    assert "azure_components" in body

# ── Pipeline ───────────────────────────────────────────────────────────────────

def test_pipeline_status():
    r = client.get("/api/pipeline/status")
    assert r.status_code == 200
    body = r.json()
    assert "pipeline_components" in body
    assert "data_flow" in body
