"""
ml_endpoint.py
──────────────
REST endpoints for ML operations:
  POST /api/ml/train      — train RandomForest demand model
  POST /api/ml/predict    — predict units sold for a product/date
  POST /api/ml/anomalies  — run IsolationForest anomaly detection
"""

import logging
import os

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from server.forecasting.spike_detector import detect_sales_spikes
from server.forecasting.units_predictor import forecast_product_demand
from server.forecasting.rf_trainer import run_rf_training

logger = logging.getLogger(__name__)

ml_router = APIRouter(tags=["B. Machine Learning / Deep Learning"])

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RETAIL_CSV_PATH: str = os.path.join(_project_root, "sales_store", "retail_records.csv")


# ── Request models ─────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    holdout_ratio: float = Field(default=0.2, ge=0.05, le=0.5,
        description="Fraction of data held out for evaluation.")
    num_trees: int = Field(default=100, ge=10, le=500,
        description="Number of trees in the Random Forest.")
    tree_depth: int = Field(default=0, ge=0, le=30,
        description="Max tree depth; 0 means unlimited.")
    random_seed: int = Field(default=42,
        description="Random seed for reproducibility.")


class PredictRequest(BaseModel):
    product_id: str = Field(..., description="Product identifier, e.g. PROD_001.")
    target_date: str = Field(..., description="Date for prediction in YYYY-MM-DD format.")
    price: float = Field(..., gt=0, description="Sale price of the product.")
    discount: float = Field(default=0.0, ge=0.0, le=100.0,
        description="Discount percentage (0–100).")
    store_id: str = Field(..., description="Store identifier, e.g. STORE_A.")
    region: str = Field(..., description="Sales region, e.g. North.")


class AnomalyRequest(BaseModel):
    outlier_fraction: float = Field(default=0.05, ge=0.01, le=0.5,
        description="Expected fraction of outliers.")
    spike_multiplier: float = Field(default=2.0, ge=1.0, le=10.0,
        description="Multiplier above mean to classify as high-sales anomaly.")
    max_results: int = Field(default=50, ge=1, le=500,
        description="Maximum number of anomalies to return.")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@ml_router.post("/api/ml/train", status_code=status.HTTP_200_OK)
async def train_demand_model(request_body: TrainRequest) -> dict:
    """Train the RandomForest demand forecasting model on the uploaded retail data."""
    if not os.path.exists(_RETAIL_CSV_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No retail data found. Please upload a CSV file first via /api/data/upload.",
        )
    try:
        result = run_rf_training(
            csv_path=_RETAIL_CSV_PATH,
            holdout_ratio=request_body.holdout_ratio,
            num_trees=request_body.num_trees,
            tree_depth=request_body.tree_depth,
            random_seed=request_body.random_seed,
        )
    except Exception as train_error:
        logger.warning("Model training failed: %s", train_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {train_error}",
        )
    return {
        "status": "Model trained and saved successfully.",
        "mae": round(result["mae"], 4),
        "rmse": round(result["rmse"], 4),
        "r2": round(result["r2"], 4),
        "num_trees": request_body.num_trees,
        "holdout_ratio": request_body.holdout_ratio,
    }


@ml_router.post("/api/ml/predict", status_code=status.HTTP_200_OK)
async def predict_demand(request_body: PredictRequest) -> dict:
    """Predict units sold for a given product, date, price, and store."""
    try:
        result = forecast_product_demand(
            product_code=request_body.product_id,
            target_date=request_body.target_date,
            sale_price=request_body.price,
            discount_amount=request_body.discount,
            outlet_id=request_body.store_id,
            sales_region=request_body.region,
        )
    except FileNotFoundError as model_missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(model_missing),
        )
    except Exception as predict_error:
        logger.warning("Prediction failed: %s", predict_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {predict_error}",
        )
    return result


@ml_router.post("/api/ml/anomalies", status_code=status.HTTP_200_OK)
async def detect_anomalies(request_body: AnomalyRequest) -> dict:
    """Run IsolationForest anomaly detection on the uploaded retail data."""
    if not os.path.exists(_RETAIL_CSV_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No retail data found. Please upload a CSV file first via /api/data/upload.",
        )
    try:
        result = detect_sales_spikes(
            csv_path=_RETAIL_CSV_PATH,
            outlier_fraction=request_body.outlier_fraction,
            spike_multiplier=request_body.spike_multiplier,
            max_results=request_body.max_results,
        )
    except Exception as anomaly_error:
        logger.warning("Anomaly detection failed: %s", anomaly_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {anomaly_error}",
        )
    return result
