"""
powerbi_endpoint.py
───────────────────
Power BI Dashboard Data Endpoints — Section F: Analytics & Visualization

All endpoints return JSON optimized for Power BI Web connector.
Connect Power BI Desktop using: Get Data → Web → paste URL

Endpoints:
  GET /api/powerbi/key-metrics          — KPI cards
  GET /api/powerbi/revenue-by-category  — Bar/Pie chart
  GET /api/powerbi/revenue-by-region    — Map/Bar chart
  GET /api/powerbi/monthly-trend        — Line chart
  GET /api/powerbi/top-products         — Table/Bar chart
  GET /api/powerbi/anomaly-alerts       — Alert table
  GET /api/powerbi/model-performance    — ML model KPIs
  GET /api/powerbi/agent-insights       — Agent-driven insights
  GET /api/powerbi/full-dashboard       — All data in one call
"""

import logging
import os

from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger(__name__)

powerbi_router = APIRouter(prefix="/api/powerbi", tags=["F. Analytics & Visualization (Power BI)"])

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RETAIL_CSV_PATH = os.path.join(_project_root, "sales_store", "retail_records.csv")
_RF_MODEL_PATH   = os.path.join(_project_root, "model_vault", "rf_demand_model.pkl")
_PIPELINE_PATH   = os.path.join(_project_root, "model_vault", "encoding_pipeline.pkl")


def _load_df():
    """Load and clean the retail dataframe."""
    if not os.path.exists(_RETAIL_CSV_PATH):
        return None
    from server.forecasting.csv_reader import read_retail_csv
    from server.forecasting.record_cleaner import clean_retail_records
    return clean_retail_records(read_retail_csv(_RETAIL_CSV_PATH))


def _no_data_error():
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="No retail data available. Please upload a CSV via POST /api/data/upload first.",
    )


# ── 1. Key Metrics (KPI Cards) ────────────────────────────────────────────────

@powerbi_router.get("/key-metrics", status_code=200)
async def get_key_metrics() -> dict:
    """
    Key metrics for Power BI KPI cards.
    Shows: total revenue, units sold, products, stores, avg discount, avg price.
    """
    df = _load_df()
    if df is None:
        _no_data_error()

    model_trained = os.path.exists(_RF_MODEL_PATH)

    return {
        "dashboard_title": "Smart Retail Analytics Engine",
        "kpi_cards": {
            "total_revenue":        round(float(df["revenue"].sum()), 2),
            "total_units_sold":     int(df["units_sold"].sum()),
            "total_records":        len(df),
            "unique_products":      int(df["product_id"].nunique()),
            "unique_stores":        int(df["store_id"].nunique()),
            "unique_categories":    int(df["category"].nunique()),
            "unique_regions":       int(df["region"].nunique()),
            "avg_price":            round(float(df["price"].mean()), 2),
            "avg_discount_pct":     round(float(df["discount"].mean()), 2),
            "avg_revenue_per_txn":  round(float(df["revenue"].mean()), 2),
            "date_range_start":     str(df["date"].min().date()),
            "date_range_end":       str(df["date"].max().date()),
            "ml_model_trained":     model_trained,
        }
    }


# ── 2. Revenue by Category ────────────────────────────────────────────────────

@powerbi_router.get("/revenue-by-category", status_code=200)
async def get_revenue_by_category() -> dict:
    """
    Revenue breakdown by product category.
    Use for: Bar chart, Pie chart, Treemap in Power BI.
    """
    df = _load_df()
    if df is None:
        _no_data_error()

    cat = (df.groupby("category")
           .agg(
               total_revenue=("revenue", "sum"),
               total_units=("units_sold", "sum"),
               record_count=("revenue", "count"),
               avg_price=("price", "mean"),
               avg_discount=("discount", "mean"),
           )
           .round(2)
           .sort_values("total_revenue", ascending=False)
           .reset_index())

    total_rev = cat["total_revenue"].sum()
    cat["revenue_share_pct"] = (cat["total_revenue"] / total_rev * 100).round(2)

    return {
        "chart_title": "Revenue by Product Category",
        "chart_type": "Bar Chart / Pie Chart",
        "x_axis": "category",
        "y_axis": "total_revenue",
        "data": cat.to_dict(orient="records"),
    }


# ── 3. Revenue by Region ──────────────────────────────────────────────────────

@powerbi_router.get("/revenue-by-region", status_code=200)
async def get_revenue_by_region() -> dict:
    """
    Revenue breakdown by sales region.
    Use for: Map visual, Bar chart in Power BI.
    """
    df = _load_df()
    if df is None:
        _no_data_error()

    reg = (df.groupby("region")
           .agg(
               total_revenue=("revenue", "sum"),
               total_units=("units_sold", "sum"),
               store_count=("store_id", "nunique"),
               product_count=("product_id", "nunique"),
               avg_discount=("discount", "mean"),
           )
           .round(2)
           .sort_values("total_revenue", ascending=False)
           .reset_index())

    total_rev = reg["total_revenue"].sum()
    reg["revenue_share_pct"] = (reg["total_revenue"] / total_rev * 100).round(2)

    return {
        "chart_title": "Revenue by Sales Region",
        "chart_type": "Map / Bar Chart",
        "x_axis": "region",
        "y_axis": "total_revenue",
        "data": reg.to_dict(orient="records"),
    }


# ── 4. Monthly Revenue Trend ──────────────────────────────────────────────────

@powerbi_router.get("/monthly-trend", status_code=200)
async def get_monthly_trend() -> dict:
    """
    Monthly revenue and units trend over time.
    Use for: Line chart, Area chart in Power BI.
    """
    df = _load_df()
    if df is None:
        _no_data_error()

    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    monthly = (df.groupby("year_month")
               .agg(
                   monthly_revenue=("revenue", "sum"),
                   monthly_units=("units_sold", "sum"),
                   transaction_count=("revenue", "count"),
                   avg_discount=("discount", "mean"),
               )
               .round(2)
               .sort_index()
               .reset_index())

    # Add MoM growth
    monthly["prev_revenue"] = monthly["monthly_revenue"].shift(1)
    monthly["mom_growth_pct"] = (
        (monthly["monthly_revenue"] - monthly["prev_revenue"])
        / monthly["prev_revenue"] * 100
    ).round(2)
    monthly = monthly.drop(columns=["prev_revenue"])
    monthly["mom_growth_pct"] = monthly["mom_growth_pct"].fillna(0)

    return {
        "chart_title": "Monthly Revenue Trend",
        "chart_type": "Line Chart / Area Chart",
        "x_axis": "year_month",
        "y_axis": "monthly_revenue",
        "data": monthly.to_dict(orient="records"),
    }


# ── 5. Top Products ───────────────────────────────────────────────────────────

@powerbi_router.get("/top-products", status_code=200)
async def get_top_products() -> dict:
    """
    Top 15 products by revenue with full metrics.
    Use for: Bar chart, Table visual in Power BI.
    """
    df = _load_df()
    if df is None:
        _no_data_error()

    top = (df.groupby(["product_id", "category"])
           .agg(
               total_revenue=("revenue", "sum"),
               total_units=("units_sold", "sum"),
               avg_price=("price", "mean"),
               avg_discount=("discount", "mean"),
               store_count=("store_id", "nunique"),
               region_count=("region", "nunique"),
           )
           .round(2)
           .sort_values("total_revenue", ascending=False)
           .head(15)
           .reset_index())

    top["rank"] = range(1, len(top) + 1)

    return {
        "chart_title": "Top 15 Products by Revenue",
        "chart_type": "Bar Chart / Table",
        "x_axis": "product_id",
        "y_axis": "total_revenue",
        "data": top.to_dict(orient="records"),
    }


# ── 6. Anomaly Alerts ─────────────────────────────────────────────────────────

@powerbi_router.get("/anomaly-alerts", status_code=200)
async def get_anomaly_alerts() -> dict:
    """
    Anomaly detection results for Power BI alert table.
    Use for: Table visual, conditional formatting in Power BI.
    """
    if not os.path.exists(_RETAIL_CSV_PATH):
        _no_data_error()

    try:
        from server.forecasting.spike_detector import detect_sales_spikes
        result = detect_sales_spikes(
            csv_path=_RETAIL_CSV_PATH,
            outlier_fraction=0.05,
            spike_multiplier=2.0,
            max_results=100,
        )

        return {
            "chart_title": "Sales Anomaly Alerts",
            "chart_type": "Table with Conditional Formatting",
            "summary": {
                "total_anomalies":      result["total_anomalies"],
                "high_sales_anomalies": result["high_sales_anomalies"],
                "low_sales_anomalies":  result["low_sales_anomalies"],
            },
            "data": result["anomalies"],
        }
    except Exception as e:
        logger.warning("Anomaly detection failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── 7. ML Model Performance ───────────────────────────────────────────────────

@powerbi_router.get("/model-performance", status_code=200)
async def get_model_performance() -> dict:
    """
    ML model performance metrics for Power BI KPI cards.
    Use for: KPI cards, Gauge visuals in Power BI.
    """
    model_trained = os.path.exists(_RF_MODEL_PATH)
    pipeline_exists = os.path.exists(_PIPELINE_PATH)

    metrics = {
        "model_name": "RandomForestRegressor",
        "model_trained": model_trained,
        "pipeline_ready": pipeline_exists,
        "algorithm": "Random Forest",
        "feature_count": 34,
        "anomaly_model": "IsolationForest",
        "training_status": "Trained" if model_trained else "Not trained — call POST /api/ml/train",
    }

    # Load saved metrics if available
    metrics_path = os.path.join(_project_root, "model_vault", "model_metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            saved = json.load(f)
        metrics.update(saved)

    return {
        "chart_title": "ML Model Performance",
        "chart_type": "KPI Cards / Gauge",
        "data": metrics,
    }


# ── 8. Agent-Driven Insights ──────────────────────────────────────────────────

@powerbi_router.get("/agent-insights", status_code=200)
async def get_agent_insights() -> dict:
    """
    Agent-driven insights from all 3 GenAI agents.
    Use for: Text cards, insight panels in Power BI.
    """
    from server.orchestration.query_handler import process_query

    insights = []

    queries = [
        ("What are the top performing products by revenue?", "AnalyticsAgent"),
        ("What is the revenue trend by category?",           "AnalyticsAgent"),
        ("What is the return policy for products?",          "PolicyAgent"),
        ("Explain the demand forecasting model performance", "ForecastAgent"),
    ]

    for query, expected_agent in queries:
        try:
            result = process_query(query)
            insights.append({
                "query":   query,
                "agent":   result.get("agent", expected_agent),
                "insight": result.get("response", ""),
                "tool":    result.get("tool_used", ""),
            })
        except Exception as e:
            insights.append({
                "query":   query,
                "agent":   expected_agent,
                "insight": f"Agent unavailable: {e}",
                "tool":    "error",
            })

    return {
        "chart_title": "Agent-Driven Business Insights",
        "chart_type": "Text Cards / Insight Panel",
        "data": insights,
    }


# ── 9. Full Dashboard (All data in one call) ──────────────────────────────────

@powerbi_router.get("/full-dashboard", status_code=200)
async def get_full_dashboard() -> dict:
    """
    Complete dashboard data in a single API call.
    Use this endpoint in Power BI Web connector for the full report.
    Connect: Get Data → Web → http://localhost:8000/api/powerbi/full-dashboard
    """
    df = _load_df()
    if df is None:
        _no_data_error()

    # Key metrics
    model_trained = os.path.exists(_RF_MODEL_PATH)
    total_revenue = round(float(df["revenue"].sum()), 2)
    total_units   = int(df["units_sold"].sum())

    # Category
    cat = (df.groupby("category")["revenue"].sum()
           .sort_values(ascending=False).round(2).to_dict())

    # Region
    reg = (df.groupby("region")["revenue"].sum()
           .sort_values(ascending=False).round(2).to_dict())

    # Monthly
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    monthly = (df.groupby("year_month")["revenue"].sum()
               .sort_index().round(2).to_dict())

    # Top 10 products
    top10 = (df.groupby("product_id")["revenue"].sum()
             .sort_values(ascending=False).head(10).round(2).to_dict())

    # Anomalies
    anomaly_summary = {"total": 0, "high": 0, "low": 0}
    try:
        from server.forecasting.spike_detector import detect_sales_spikes
        anom = detect_sales_spikes(_RETAIL_CSV_PATH, 0.05, 2.0, 20)
        anomaly_summary = {
            "total": anom["total_anomalies"],
            "high":  anom["high_sales_anomalies"],
            "low":   anom["low_sales_anomalies"],
            "top_anomalies": anom["anomalies"][:5],
        }
    except Exception:
        pass

    return {
        "report_title":   "Smart Retail Analytics Engine — Power BI Dashboard",
        "generated_at":   str(df["date"].max().date()),
        "key_metrics": {
            "total_revenue":     total_revenue,
            "total_units_sold":  total_units,
            "unique_products":   int(df["product_id"].nunique()),
            "unique_stores":     int(df["store_id"].nunique()),
            "avg_discount_pct":  round(float(df["discount"].mean()), 2),
            "date_range":        f"{df['date'].min().date()} to {df['date'].max().date()}",
            "ml_model_trained":  model_trained,
        },
        "revenue_by_category": cat,
        "revenue_by_region":   reg,
        "monthly_trend":       monthly,
        "top_10_products":     top10,
        "anomaly_summary":     anomaly_summary,
    }
