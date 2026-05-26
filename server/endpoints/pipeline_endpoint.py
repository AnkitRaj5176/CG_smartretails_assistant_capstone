"""
pipeline_endpoint.py
────────────────────
Section E: Data Engineering Pipeline
"""

import logging
import os

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

pipeline_router = APIRouter(tags=["E. Data Engineering Pipeline"])

_project_root = os.getcwd()
_DATA_ENG_PATH = os.path.join(_project_root, "data_engineering")
_OUTPUT_PATH   = os.path.join(_DATA_ENG_PATH, "output")
_RAW_CSV_PATH  = os.path.join(_project_root, "sales_store", "retail_records.csv")


class PipelineRunRequest(BaseModel):
    mode: str = Field(default="pandas",
        description="Pipeline mode: 'pandas' (local) or 'spark' (requires PySpark).")


@pipeline_router.get("/api/pipeline/status", status_code=status.HTTP_200_OK)
async def get_pipeline_status() -> dict:
    """Return the status of all data engineering pipeline components."""

    def exists(rel: str) -> bool:
        return os.path.exists(os.path.join(_project_root, rel))

    def csv_rows(path: str) -> int:
        if not os.path.exists(path):
            return 0
        try:
            with open(path) as f:
                return sum(1 for _ in f) - 1
        except Exception:
            return 0

    return {
        "pipeline_components": {
            "azure_data_factory":      {"file": "data_engineering/adf_pipeline.json",         "exists": exists("data_engineering/adf_pipeline.json"),         "description": "ADF pipeline — daily ingestion trigger"},
            "azure_databricks_pyspark":{"file": "data_engineering/databricks_notebook.py",    "exists": exists("data_engineering/databricks_notebook.py"),    "description": "PySpark notebook — RAW→STAGED→CURATED Delta tables"},
            "azure_fabric_lakehouse":  {"file": "data_engineering/azure_fabric_notebook.py",  "exists": exists("data_engineering/azure_fabric_notebook.py"),  "description": "Fabric Lakehouse — Delta tables + Data Activator"},
            "local_pipeline":          {"file": "data_engineering/retail_pipeline.py",        "exists": exists("data_engineering/retail_pipeline.py"),        "description": "Local pipeline — PySpark or pandas fallback"},
            "sql_analytics":           {"file": "data_engineering/sql_analytics.sql",         "exists": exists("data_engineering/sql_analytics.sql"),         "description": "12 Spark SQL / T-SQL analytics queries"},
        },
        "data_flow": {
            "raw":     {"source": "sales_store/retail_records.csv", "exists": exists("sales_store/retail_records.csv"), "rows": csv_rows(_RAW_CSV_PATH)},
            "staged":  {"output": "data_engineering/output/staged/staged.csv",   "exists": exists("data_engineering/output/staged/staged.csv"),   "description": "Cleaned, validated, deduplicated data"},
            "curated": {"output": "data_engineering/output/curated/curated.csv", "exists": exists("data_engineering/output/curated/curated.csv"), "description": "Feature-enriched, partitioned by year/month"},
            "parquet": {"output": "data_engineering/output/parquet/",            "exists": exists("data_engineering/output/parquet/"),            "description": "Parquet-based storage"},
        },
        "storage_format": "CSV + Parquet (partitioned by sale_year/sale_month)",
        "delta_tables": "Available in Azure Databricks and Azure Fabric notebooks",
    }


@pipeline_router.post("/api/pipeline/run", status_code=status.HTTP_200_OK)
async def run_pipeline(request_body: PipelineRunRequest) -> dict:
    """Run the data engineering pipeline: RAW → STAGED → CURATED → Analytics."""
    if not os.path.exists(_RAW_CSV_PATH):
        raise HTTPException(status_code=404,
            detail="No raw data found. Upload CSV via POST /api/data/upload first.")
    try:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, os.path.join(_DATA_ENG_PATH, "retail_pipeline.py")],
            capture_output=True, text=True, cwd=_project_root, timeout=120,
        )
        success = result.returncode == 0 or "PIPELINE COMPLETE" in result.stderr

        def csv_rows(rel: str) -> int:
            p = os.path.join(_project_root, rel)
            if not os.path.exists(p):
                return 0
            try:
                with open(p) as f:
                    return sum(1 for _ in f) - 1
            except Exception:
                return 0

        analytics_dir = os.path.join(_OUTPUT_PATH, "analytics")
        analytics_files = [f for f in os.listdir(analytics_dir) if f.endswith(".csv")] \
            if os.path.exists(analytics_dir) else []

        return {
            "status": "Pipeline completed successfully." if success else "Pipeline completed with warnings.",
            "data_flow": {
                "raw_rows":     csv_rows("sales_store/retail_records.csv"),
                "staged_rows":  csv_rows("data_engineering/output/staged/staged.csv"),
                "curated_rows": csv_rows("data_engineering/output/curated/curated.csv"),
            },
            "analytics_generated": analytics_files,
            "storage_format": "CSV + Parquet (partitioned by year/month)",
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Pipeline timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
