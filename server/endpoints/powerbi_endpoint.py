"""
powerbi_endpoint.py — Section F: Analytics & Visualization (Power BI)
"""
import logging
import os
from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger(__name__)
powerbi_router = APIRouter(prefix="/api/powerbi", tags=["F. Analytics & Visualization (Power BI)"])
_project_root    = os.getcwd()
_RETAIL_CSV_PATH = os.path.join(_project_root, "sales_store", "retail_records.csv")
_RF_MODEL_PATH   = os.path.join(_project_root, "model_vault", "rf_demand_model.pkl")


@powerbi_router.get("/full-dashboard")
async def get_full_dashboard() -> dict:
    """
    Complete Power BI dashboard data in one call.
    Connect: Power BI Desktop → Get Data → Web → this URL.
    """
    if not os.path.exists(_RETAIL_CSV_PATH):
        raise HTTPException(status_code=404, detail="No data. Upload CSV via POST /api/data/upload first.")
    try:
        from server.forecasting.csv_reader import read_retail_csv
        from server.forecasting.record_cleaner import clean_retail_records
        df = clean_retail_records(read_retail_csv(_RETAIL_CSV_PATH))
        df["year_month"] = df["date"].dt.to_period("M").astype(str)

        anomaly_summary = {"total": 0, "high": 0, "low": 0}
        try:
            from server.forecasting.spike_detector import detect_sales_spikes
            a = detect_sales_spikes(_RETAIL_CSV_PATH, 0.05, 2.0, 10)
            anomaly_summary = {"total": a["total_anomalies"], "high": a["high_sales_anomalies"], "low": a["low_sales_anomalies"]}
        except Exception:
            pass

        return {
            "report_title": "Smart Retail Analytics Engine — Power BI Dashboard",
            "key_metrics": {
                "total_revenue":    round(float(df["revenue"].sum()), 2),
                "total_units_sold": int(df["units_sold"].sum()),
                "unique_products":  int(df["product_id"].nunique()),
                "unique_stores":    int(df["store_id"].nunique()),
                "avg_discount_pct": round(float(df["discount"].mean()), 2),
                "ml_model_trained": os.path.exists(_RF_MODEL_PATH),
            },
            "revenue_by_category": {k: round(float(v), 2) for k, v in df.groupby("category")["revenue"].sum().sort_values(ascending=False).items()},
            "revenue_by_region":   {k: round(float(v), 2) for k, v in df.groupby("region")["revenue"].sum().sort_values(ascending=False).items()},
            "monthly_trend":       {k: round(float(v), 2) for k, v in df.groupby("year_month")["revenue"].sum().sort_index().items()},
            "top_10_products":     {k: round(float(v), 2) for k, v in df.groupby("product_id")["revenue"].sum().sort_values(ascending=False).head(10).items()},
            "anomaly_summary":     anomaly_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
