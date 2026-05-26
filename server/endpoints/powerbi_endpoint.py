"""
powerbi_endpoint.py
───────────────────
Section F: Analytics & Visualization (Power BI)
"""

import logging
import os

from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger(__name__)

powerbi_router = APIRouter(
    prefix="/api/powerbi",
    tags=["F. Analytics & Visualization (Power BI)"]
)

_project_root = os.getcwd()
_RETAIL_CSV_PATH = os.path.join(_project_root, "sales_store", "retail_records.csv")
_RF_MODEL_PATH   = os.path.join(_project_root, "model_vault", "rf_demand_model.pkl")


def _load_df():
    if not os.path.exists(_RETAIL_CSV_PATH):
        return None
    from server.forecasting.csv_reader import read_retail_csv
    from server.forecasting.record_cleaner import clean_retail_records
    return clean_retail_records(read_retail_csv(_RETAIL_CSV_PATH))


@powerbi_router.get("/full-dashboard", status_code=status.HTTP_200_OK)
async def get_full_dashboard() -> dict:
    """
    Complete dashboard data in a single API call.
    Connect Power BI: Get Data → Web → this URL.
    """
    df = _load_df()
    if df is None:
        raise HTTPException(status_code=404,
            detail="No retail data. Upload CSV via POST /api/data/upload first.")

    model_trained = os.path.exists(_RF_MODEL_PATH)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    anomaly_summary = {"total": 0, "high": 0, "low": 0, "top_anomalies": []}
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
        "report_title":  "Smart Retail Analytics Engine — Power BI Dashboard",
        "generated_at":  str(df["date"].max().date()),
        "key_metrics": {
            "total_revenue":    round(float(df["revenue"].sum()), 2),
            "total_units_sold": int(df["units_sold"].sum()),
            "unique_products":  int(df["product_id"].nunique()),
            "unique_stores":    int(df["store_id"].nunique()),
            "avg_discount_pct": round(float(df["discount"].mean()), 2),
            "date_range":       f"{df['date'].min().date()} to {df['date'].max().date()}",
            "ml_model_trained": model_trained,
        },
        "revenue_by_category": {k: round(float(v), 2) for k, v in
            df.groupby("category")["revenue"].sum().sort_values(ascending=False).items()},
        "revenue_by_region": {k: round(float(v), 2) for k, v in
            df.groupby("region")["revenue"].sum().sort_values(ascending=False).items()},
        "monthly_trend": {k: round(float(v), 2) for k, v in
            df.groupby("year_month")["revenue"].sum().sort_index().items()},
        "top_10_products": {k: round(float(v), 2) for k, v in
            df.groupby("product_id")["revenue"].sum()
            .sort_values(ascending=False).head(10).items()},
        "anomaly_summary": anomaly_summary,
    }
