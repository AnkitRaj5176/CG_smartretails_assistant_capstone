"""
powerbi_endpoint.py — Section F: Analytics & Visualization (Power BI)
Connect Power BI Desktop: Get Data → Web → paste URL
"""
import logging
import os
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
powerbi_router = APIRouter(prefix="/api/powerbi", tags=["F. Analytics & Visualization (Power BI)"])
_project_root    = os.getcwd()
_RETAIL_CSV_PATH = os.path.join(_project_root, "sales_store", "retail_records.csv")
_RF_MODEL_PATH   = os.path.join(_project_root, "model_vault", "rf_demand_model.pkl")
_METRICS_PATH    = os.path.join(_project_root, "model_vault", "model_metrics.json")


@powerbi_router.get("/dashboard")
async def get_powerbi_dashboard() -> dict:
    """
    Complete Power BI dashboard data — all 5 sections in one call.
    Connect: Power BI Desktop → Get Data → Web → this URL.

    Sections:
    1. Key Metrics (KPI cards)
    2. Model Outputs (ML performance)
    3. Anomaly Alerts (IsolationForest results)
    4. Revenue Trends (charts)
    5. Agent Insights (GenAI responses)
    """
    if not os.path.exists(_RETAIL_CSV_PATH):
        raise HTTPException(status_code=404,
            detail="No data available. Upload CSV via POST /api/data/upload first.")
    try:
        from server.forecasting.csv_reader import read_retail_csv
        from server.forecasting.record_cleaner import clean_retail_records
        df = clean_retail_records(read_retail_csv(_RETAIL_CSV_PATH))
        df["year_month"] = df["date"].dt.to_period("M").astype(str)

        # ── 1. Key Metrics ─────────────────────────────────────────────────────
        key_metrics = {
            "total_revenue":    round(float(df["revenue"].sum()), 2),
            "total_units_sold": int(df["units_sold"].sum()),
            "unique_products":  int(df["product_id"].nunique()),
            "unique_stores":    int(df["store_id"].nunique()),
            "unique_categories":int(df["category"].nunique()),
            "unique_regions":   int(df["region"].nunique()),
            "avg_price":        round(float(df["price"].mean()), 2),
            "avg_discount_pct": round(float(df["discount"].mean()), 2),
            "date_range_start": str(df["date"].min().date()),
            "date_range_end":   str(df["date"].max().date()),
            "total_records":    len(df),
        }

        # ── 2. Model Outputs ───────────────────────────────────────────────────
        model_outputs = {
            "model_name":    "RandomForestRegressor",
            "model_trained": os.path.exists(_RF_MODEL_PATH),
            "feature_count": 34,
            "algorithm":     "Random Forest + IsolationForest",
            "mae":  None, "rmse": None, "r2": None,
        }
        if os.path.exists(_METRICS_PATH):
            import json
            with open(_METRICS_PATH) as f:
                saved = json.load(f)
            model_outputs.update(saved)

        # ── 3. Anomaly Alerts ──────────────────────────────────────────────────
        anomaly_alerts = {"total_anomalies": 0, "high_sales": 0, "low_sales": 0, "top_anomalies": []}
        try:
            from server.forecasting.spike_detector import detect_sales_spikes
            result = detect_sales_spikes(_RETAIL_CSV_PATH, 0.05, 2.0, 20)
            anomaly_alerts = {
                "total_anomalies": result["total_anomalies"],
                "high_sales":      result["high_sales_anomalies"],
                "low_sales":       result["low_sales_anomalies"],
                "top_anomalies":   result["anomalies"][:10],
            }
        except Exception:
            pass

        # ── 4. Revenue Trends ──────────────────────────────────────────────────
        revenue_trends = {
            "by_category": {k: round(float(v), 2) for k, v in
                df.groupby("category")["revenue"].sum().sort_values(ascending=False).items()},
            "by_region": {k: round(float(v), 2) for k, v in
                df.groupby("region")["revenue"].sum().sort_values(ascending=False).items()},
            "monthly": {k: round(float(v), 2) for k, v in
                df.groupby("year_month")["revenue"].sum().sort_index().items()},
            "top_10_products": {k: round(float(v), 2) for k, v in
                df.groupby("product_id")["revenue"].sum().sort_values(ascending=False).head(10).items()},
            "by_store": {k: round(float(v), 2) for k, v in
                df.groupby("store_id")["revenue"].sum().sort_values(ascending=False).items()},
        }

        # ── 5. Agent Insights ──────────────────────────────────────────────────
        agent_insights = []
        queries = [
            "What are the top performing products by revenue?",
            "What is the revenue trend by category?",
            "What is the return policy for products?",
        ]
        try:
            from server.orchestration.query_handler import process_query
            for q in queries:
                result = process_query(q)
                agent_insights.append({
                    "query":   q,
                    "agent":   result.get("agent", ""),
                    "insight": result.get("response", ""),
                })
        except Exception:
            pass

        return {
            "dashboard_title": "Smart Retail Analytics Engine — Power BI Dashboard",
            "capstone_section": "F. Analytics & Visualization",
            "generated_at": str(df["date"].max().date()),
            "key_metrics":    key_metrics,
            "model_outputs":  model_outputs,
            "anomaly_alerts": anomaly_alerts,
            "revenue_trends": revenue_trends,
            "agent_insights": agent_insights,
        }

    except Exception as e:
        logger.warning("Dashboard failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
