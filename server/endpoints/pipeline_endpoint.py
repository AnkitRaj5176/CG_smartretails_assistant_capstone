"""
pipeline_endpoint.py — Section E: Data Engineering Pipeline
"""
import logging
import os
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
pipeline_router = APIRouter(tags=["E. Data Engineering Pipeline"])
_project_root = os.getcwd()
_DATA_ENG_PATH = os.path.join(_project_root, "data_engineering")
_RAW_CSV_PATH  = os.path.join(_project_root, "sales_store", "retail_records.csv")


class PipelineRunRequest(BaseModel):
    mode: str = Field(default="pandas", description="'pandas' (local) or 'spark' (Databricks)")


def _exists(rel: str) -> bool:
    return os.path.exists(os.path.join(_project_root, rel))

def _rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path) as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return 0


@pipeline_router.get("/api/pipeline/status")
async def get_pipeline_status() -> dict:
    """Status of all data engineering components — ADF, Databricks, Fabric, SQL."""
    return {
        "pipeline_components": {
            "azure_data_factory":       {"exists": _exists("data_engineering/adf_pipeline.json"),        "description": "ADF pipeline — daily ingestion trigger"},
            "azure_databricks_pyspark": {"exists": _exists("data_engineering/databricks_notebook.py"),   "description": "PySpark — RAW→STAGED→CURATED Delta tables"},
            "azure_fabric_lakehouse":   {"exists": _exists("data_engineering/azure_fabric_notebook.py"), "description": "Fabric Lakehouse — Delta + Data Activator"},
            "local_pipeline":           {"exists": _exists("data_engineering/retail_pipeline.py"),       "description": "Local pipeline — PySpark or pandas fallback"},
            "sql_analytics":            {"exists": _exists("data_engineering/sql_analytics.sql"),        "description": "12 Spark SQL / T-SQL analytics queries"},
        },
        "data_flow": {
            "raw":     {"exists": _exists("sales_store/retail_records.csv"),                "rows": _rows(_RAW_CSV_PATH)},
            "staged":  {"exists": _exists("data_engineering/output/staged/staged.csv"),    "description": "Cleaned, validated data"},
            "curated": {"exists": _exists("data_engineering/output/curated/curated.csv"),  "description": "Feature-enriched, partitioned by year/month"},
            "parquet": {"exists": _exists("data_engineering/output/parquet/"),             "description": "Parquet-based storage"},
        },
        "storage_format": "CSV + Parquet (partitioned by sale_year/sale_month)",
        "delta_tables": "Available in Azure Databricks and Azure Fabric notebooks",
    }


@pipeline_router.post("/api/pipeline/run")
async def run_pipeline(request_body: PipelineRunRequest) -> dict:
    """Run RAW → STAGED → CURATED data engineering pipeline locally."""
    if not os.path.exists(_RAW_CSV_PATH):
        raise HTTPException(status_code=404, detail="No data. Upload CSV via POST /api/data/upload first.")
    try:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, os.path.join(_DATA_ENG_PATH, "retail_pipeline.py")],
            capture_output=True, text=True, cwd=_project_root, timeout=120,
        )
        success = result.returncode == 0 or "PIPELINE COMPLETE" in result.stderr
        return {
            "status": "Pipeline completed." if success else "Pipeline completed with warnings.",
            "data_flow": {
                "raw_rows":     _rows(_RAW_CSV_PATH),
                "staged_rows":  _rows(os.path.join(_project_root, "data_engineering/output/staged/staged.csv")),
                "curated_rows": _rows(os.path.join(_project_root, "data_engineering/output/curated/curated.csv")),
            },
            "storage_format": "CSV + Parquet (partitioned by year/month)",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
