"""
pipeline_endpoint.py
────────────────────
Section E: Data Engineering Pipeline — REST Endpoints

Exposes the data engineering pipeline via REST API so it appears
in Swagger UI under Section E.

Endpoints:
  GET  /api/pipeline/status     — Pipeline files and output status
  POST /api/pipeline/run        — Run RAW → STAGED → CURATED pipeline
  GET  /api/pipeline/output     — View pipeline output (analytics results)
  GET  /api/pipeline/sql        — View SQL analytics queries
"""

import logging
import os

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

pipeline_router = APIRouter(tags=["E. Data Engineering Pipeline"])

_project_root = os.getcwd()  # startup.py sets cwd to project root
_DATA_ENG_PATH  = os.path.join(_project_root, "data_engineering")
_OUTPUT_PATH    = os.path.join(_DATA_ENG_PATH, "output")
_RAW_CSV_PATH   = os.path.join(_project_root, "sales_store", "retail_records.csv")


class PipelineRunRequest(BaseModel):
    """Request body for running the data engineering pipeline."""
    mode: str = Field(
        default="pandas",
        description="Pipeline mode: 'pandas' (local) or 'spark' (requires PySpark).",
    )


# ── 1. Pipeline Status ────────────────────────────────────────────────────────

@pipeline_router.get("/api/pipeline/status", status_code=status.HTTP_200_OK)
async def get_pipeline_status() -> dict:
    """
    Return the status of all data engineering pipeline components.
    Shows which files exist and what outputs have been generated.
    """
    def file_info(path: str) -> dict:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists and os.path.isfile(path) else 0
        return {"exists": exists, "size_bytes": size, "path": os.path.relpath(path, _project_root)}

    # Check output directories
    staged_csv   = os.path.join(_OUTPUT_PATH, "staged", "staged.csv")
    curated_csv  = os.path.join(_OUTPUT_PATH, "curated", "curated.csv")
    parquet_dir  = os.path.join(_OUTPUT_PATH, "parquet")
    analytics    = os.path.join(_OUTPUT_PATH, "analytics")

    analytics_files = {}
    if os.path.exists(analytics):
        for f in os.listdir(analytics):
            if f.endswith(".csv"):
                fp = os.path.join(analytics, f)
                analytics_files[f] = {"size_bytes": os.path.getsize(fp)}

    return {
        "pipeline_components": {
            "azure_data_factory": {
                "file": "data_engineering/adf_pipeline.json",
                "exists": os.path.exists(os.path.join(_DATA_ENG_PATH, "adf_pipeline.json")),
                "description": "ADF pipeline config — daily ingestion trigger, Databricks job trigger",
            },
            "azure_databricks_pyspark": {
                "file": "data_engineering/databricks_notebook.py",
                "exists": os.path.exists(os.path.join(_DATA_ENG_PATH, "databricks_notebook.py")),
                "description": "PySpark notebook — RAW → STAGED → CURATED Delta tables",
            },
            "azure_fabric_lakehouse": {
                "file": "data_engineering/azure_fabric_notebook.py",
                "exists": os.path.exists(os.path.join(_DATA_ENG_PATH, "azure_fabric_notebook.py")),
                "description": "Fabric Lakehouse notebook — Delta tables + Data Activator alerts",
            },
            "local_pipeline": {
                "file": "data_engineering/retail_pipeline.py",
                "exists": os.path.exists(os.path.join(_DATA_ENG_PATH, "retail_pipeline.py")),
                "description": "Local pipeline — PySpark or pandas fallback",
            },
            "sql_analytics": {
                "file": "data_engineering/sql_analytics.sql",
                "exists": os.path.exists(os.path.join(_DATA_ENG_PATH, "sql_analytics.sql")),
                "description": "12 Spark SQL / T-SQL analytics queries",
            },
        },
        "data_flow": {
            "raw": {
                "source": "sales_store/retail_records.csv",
                "exists": os.path.exists(_RAW_CSV_PATH),
                "rows": _count_csv_rows(_RAW_CSV_PATH),
            },
            "staged": {
                "output": "data_engineering/output/staged/staged.csv",
                "exists": os.path.exists(staged_csv),
                "description": "Cleaned, validated, deduplicated data",
            },
            "curated": {
                "output": "data_engineering/output/curated/curated.csv",
                "exists": os.path.exists(curated_csv),
                "description": "Feature-enriched data with date, price, aggregate features",
            },
            "parquet": {
                "output": "data_engineering/output/parquet/",
                "exists": os.path.exists(parquet_dir),
                "description": "Parquet-based storage partitioned by year/month",
            },
        },
        "analytics_outputs": analytics_files,
        "storage_format": "CSV + Parquet (partitioned by sale_year / sale_month)",
        "delta_tables": "Available in Azure Databricks and Azure Fabric notebooks",
    }


def _count_csv_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r") as f:
            return sum(1 for _ in f) - 1  # subtract header
    except Exception:
        return 0


# ── 2. Run Pipeline ───────────────────────────────────────────────────────────

@pipeline_router.post("/api/pipeline/run", status_code=status.HTTP_200_OK)
async def run_pipeline(request_body: PipelineRunRequest) -> dict:
    """
    Run the data engineering pipeline locally.
    Executes: RAW → STAGED → CURATED → Analytics outputs.
    On Azure: use Azure Data Factory to trigger Databricks notebook.
    """
    if not os.path.exists(_RAW_CSV_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No raw data found. Please upload CSV via POST /api/data/upload first.",
        )

    try:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, os.path.join(_DATA_ENG_PATH, "retail_pipeline.py")],
            capture_output=True,
            text=True,
            cwd=_project_root,
            timeout=120,
        )

        success = result.returncode == 0 or "PIPELINE COMPLETE" in result.stderr

        # Count output rows
        staged_rows  = _count_csv_rows(os.path.join(_OUTPUT_PATH, "staged", "staged.csv"))
        curated_rows = _count_csv_rows(os.path.join(_OUTPUT_PATH, "curated", "curated.csv"))

        analytics_files = []
        analytics_dir = os.path.join(_OUTPUT_PATH, "analytics")
        if os.path.exists(analytics_dir):
            analytics_files = [f for f in os.listdir(analytics_dir) if f.endswith(".csv")]

        return {
            "status": "Pipeline completed successfully." if success else "Pipeline completed with warnings.",
            "pipeline_mode": request_body.mode,
            "data_flow": {
                "raw_rows":     _count_csv_rows(_RAW_CSV_PATH),
                "staged_rows":  staged_rows,
                "curated_rows": curated_rows,
            },
            "analytics_generated": analytics_files,
            "storage_format": "CSV + Parquet (partitioned by year/month)",
            "azure_equivalent": {
                "adf_pipeline":        "data_engineering/adf_pipeline.json",
                "databricks_notebook": "data_engineering/databricks_notebook.py",
                "fabric_notebook":     "data_engineering/azure_fabric_notebook.py",
            },
            "logs": result.stderr[-1000:] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Pipeline timed out after 120 seconds.")
    except Exception as e:
        logger.warning("Pipeline run failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


# ── 3. Pipeline Output ────────────────────────────────────────────────────────

@pipeline_router.get("/api/pipeline/output", status_code=status.HTTP_200_OK)
async def get_pipeline_output() -> dict:
    """
    Return analytics output from the data engineering pipeline.
    Shows category revenue, monthly trend, top products, region performance.
    """
    analytics_dir = os.path.join(_OUTPUT_PATH, "analytics")

    if not os.path.exists(analytics_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pipeline output found. Run POST /api/pipeline/run first.",
        )

    try:
        import pandas as pd

        output = {}

        file_map = {
            "category_revenue":   "category_revenue.csv",
            "monthly_trend":      "monthly_trend.csv",
            "top_products":       "top_products.csv",
            "region_performance": "region_performance.csv",
        }

        for key, filename in file_map.items():
            filepath = os.path.join(analytics_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                output[key] = df.to_dict(orient="records")
            else:
                output[key] = []

        return {
            "pipeline_stage": "CURATED",
            "data_flow": "RAW → STAGED → CURATED → Analytics",
            "storage": "CSV + Parquet (partitioned by sale_year/sale_month)",
            "analytics": output,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read output: {e}")


# ── 4. SQL Queries ────────────────────────────────────────────────────────────

@pipeline_router.get("/api/pipeline/sql", status_code=status.HTTP_200_OK)
async def get_sql_queries() -> dict:
    """
    Return the SQL analytics queries used in the data engineering pipeline.
    Compatible with Spark SQL, Azure Synapse T-SQL, and Azure Fabric SQL.
    """
    sql_path = os.path.join(_DATA_ENG_PATH, "sql_analytics.sql")

    if not os.path.exists(sql_path):
        raise HTTPException(status_code=404, detail="SQL file not found.")

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # Parse individual queries
    queries = []
    current_name = ""
    current_sql = []

    for line in sql_content.split("\n"):
        if line.startswith("-- ──") and "──" in line:
            if current_name and current_sql:
                queries.append({
                    "name": current_name.strip(),
                    "sql": "\n".join(current_sql).strip(),
                })
            current_name = line.replace("--", "").replace("─", "").strip()
            current_sql = []
        else:
            current_sql.append(line)

    if current_name and current_sql:
        queries.append({
            "name": current_name.strip(),
            "sql": "\n".join(current_sql).strip(),
        })

    return {
        "description": "SQL analytics queries for the data engineering pipeline",
        "compatible_with": ["Spark SQL", "Azure Synapse T-SQL", "Azure Fabric SQL"],
        "total_queries": len(queries),
        "queries": queries,
    }
