"""
retail_pipeline.py
──────────────────
Smart Retail Analytics Engine — Data Engineering Pipeline
Section E: Azure Data Engineering

Pipeline Stages:
  RAW      → Read raw CSV from source (local / Azure Data Lake)
  STAGED   → Validate, clean, normalize → save as Parquet
  CURATED  → Feature-enrich, aggregate → save as Parquet (partitioned)
  ANALYTICS→ Spark SQL queries for business insights

Supports:
  - Azure Databricks (PySpark) — production mode
  - Local pandas fallback       — development mode

Run locally:
    python data_engineering/retail_pipeline.py

Run on Databricks:
    Use databricks_notebook.py
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Local paths (override with ADLS paths on Databricks)
RAW_CSV_PATH   = os.path.join(_PROJECT_ROOT, "sales_store", "retail_records.csv")
STAGED_PATH    = os.path.join(_PROJECT_ROOT, "data_engineering", "output", "staged")
CURATED_PATH   = os.path.join(_PROJECT_ROOT, "data_engineering", "output", "curated")
PARQUET_PATH   = os.path.join(_PROJECT_ROOT, "data_engineering", "output", "parquet")
ANALYTICS_PATH = os.path.join(_PROJECT_ROOT, "data_engineering", "output", "analytics")

# Azure Data Lake paths (used when running on Databricks)
ADLS_RAW     = "abfss://raw@retailstorage.dfs.core.windows.net/retail/retail_records.csv"
ADLS_STAGED  = "abfss://staged@retailstorage.dfs.core.windows.net/retail/"
ADLS_CURATED = "abfss://curated@retailstorage.dfs.core.windows.net/retail/"

VALID_CATEGORIES = {
    "Electronics", "Clothing", "Groceries", "Furniture", "Sports",
    "Beauty", "Toys", "Books", "Automotive", "Health",
}
VALID_REGIONS = {
    "North", "South", "East", "West", "Central",
    "Northeast", "Northwest", "Southeast", "Southwest",
}

# ── Detect environment ─────────────────────────────────────────────────────────
_IN_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ
spark = None

if not _IN_DATABRICKS:
    try:
        from pyspark.sql import SparkSession
        spark = (SparkSession.builder
                 .appName("RetailEnginePipeline")
                 .master("local[*]")
                 .config("spark.sql.shuffle.partitions", "4")
                 .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                 .config("spark.sql.catalog.spark_catalog",
                         "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("WARN")
        logger.info("Local SparkSession started.")
    except ImportError:
        logger.warning("PySpark not installed — running pandas fallback pipeline.")
        spark = None


# ══════════════════════════════════════════════════════════════════════════════
# SPARK PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_spark_pipeline() -> dict:
    """Full PySpark pipeline: RAW → STAGED → CURATED → ANALYTICS."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType,
        DoubleType, IntegerType,
    )

    results = {}

    # ── STAGE 1: RAW ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 — RAW: Reading source CSV")
    logger.info("=" * 60)

    raw_schema = StructType([
        StructField("product_id",  StringType(),  True),
        StructField("category",    StringType(),  True),
        StructField("region",      StringType(),  True),
        StructField("store_id",    StringType(),  True),
        StructField("date",        StringType(),  True),
        StructField("price",       DoubleType(),  True),
        StructField("discount",    DoubleType(),  True),
        StructField("units_sold",  IntegerType(), True),
        StructField("revenue",     DoubleType(),  True),
    ])

    raw_df = (spark.read
              .option("header", "true")
              .schema(raw_schema)
              .csv(RAW_CSV_PATH))

    raw_count = raw_df.count()
    logger.info("RAW row count: %d", raw_count)
    results["raw_count"] = raw_count

    # ── STAGE 2: STAGED ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 — STAGED: Cleaning & validating")
    logger.info("=" * 60)

    staged_df = (raw_df
        .dropDuplicates()
        .withColumn("date",       F.to_date("date", "yyyy-MM-dd"))
        .dropna(subset=["date", "price", "units_sold"])
        .withColumn("product_id", F.upper(F.trim("product_id")))
        .withColumn("store_id",   F.upper(F.trim("store_id")))
        .withColumn("category",   F.initcap(F.trim("category")))
        .withColumn("region",     F.initcap(F.trim("region")))
        .fillna({"discount": 0.0})
        .filter(F.col("price") > 0)
        .filter(F.col("units_sold") >= 0)
        .filter(F.col("discount").between(0, 100))
        .filter(F.col("category").isin(list(VALID_CATEGORIES)))
        .filter(F.col("region").isin(list(VALID_REGIONS)))
        .withColumn("revenue",
                    F.round(F.col("price") * F.col("units_sold")
                            * (1 - F.col("discount") / 100), 2))
    )

    staged_count = staged_df.count()
    logger.info("STAGED row count: %d", staged_count)
    results["staged_count"] = staged_count

    # Save as Parquet
    os.makedirs(STAGED_PATH, exist_ok=True)
    (staged_df.coalesce(1)
     .write.mode("overwrite")
     .parquet(STAGED_PATH))
    logger.info("STAGED saved as Parquet: %s", STAGED_PATH)

    # Also save as CSV for compatibility
    (staged_df.coalesce(1)
     .write.mode("overwrite")
     .option("header", "true")
     .csv(STAGED_PATH + "_csv"))

    # ── STAGE 3: CURATED ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3 — CURATED: Feature enrichment")
    logger.info("=" * 60)

    # Date features
    curated_df = (staged_df
        .withColumn("sale_day",     F.dayofmonth("date"))
        .withColumn("sale_month",   F.month("date"))
        .withColumn("sale_year",    F.year("date"))
        .withColumn("sale_quarter", F.quarter("date"))
        .withColumn("weekday_num",  F.dayofweek("date"))
        .withColumn("is_weekend",   (F.col("weekday_num") >= 6).cast("int"))
        .withColumn("year_month",   F.date_format("date", "yyyy-MM"))
        .withColumn("net_price",
                    F.round(F.col("price") * (1 - F.col("discount") / 100), 2))
        .withColumn("price_x_discount",
                    F.round(F.col("price") * F.col("discount"), 2))
    )

    # Aggregate features
    product_agg = staged_df.groupBy("product_id").agg(
        F.avg("units_sold").alias("avg_units_by_product"),
        F.stddev("units_sold").alias("std_units_by_product"),
        F.sum("revenue").alias("total_revenue_by_product"),
    )
    category_agg = staged_df.groupBy("category").agg(
        F.avg("units_sold").alias("avg_units_by_category"))
    region_agg = staged_df.groupBy("region").agg(
        F.avg("units_sold").alias("avg_units_by_region"))
    store_agg = staged_df.groupBy("store_id").agg(
        F.avg("units_sold").alias("avg_units_by_store"))

    curated_df = (curated_df
        .join(product_agg,  on="product_id", how="left")
        .join(category_agg, on="category",   how="left")
        .join(region_agg,   on="region",     how="left")
        .join(store_agg,    on="store_id",   how="left")
        .fillna(0.0))

    curated_count = curated_df.count()
    logger.info("CURATED row count: %d", curated_count)
    results["curated_count"] = curated_count

    # Save as Parquet partitioned by year and month
    os.makedirs(CURATED_PATH, exist_ok=True)
    (curated_df
     .write.mode("overwrite")
     .partitionBy("sale_year", "sale_month")
     .parquet(CURATED_PATH))
    logger.info("CURATED saved as partitioned Parquet: %s", CURATED_PATH)

    # ── STAGE 4: SPARK SQL ANALYTICS ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 — ANALYTICS: Spark SQL queries")
    logger.info("=" * 60)

    curated_df.createOrReplaceTempView("retail_curated")

    # Revenue by category
    cat_df = spark.sql("""
        SELECT category,
               ROUND(SUM(revenue), 2)  AS total_revenue,
               SUM(units_sold)         AS total_units,
               COUNT(*)                AS record_count,
               ROUND(AVG(discount), 2) AS avg_discount
        FROM retail_curated
        GROUP BY category
        ORDER BY total_revenue DESC
    """)
    logger.info("Revenue by Category:")
    cat_df.show(truncate=False)

    # Monthly revenue trend
    monthly_df = spark.sql("""
        SELECT year_month,
               ROUND(SUM(revenue), 2) AS monthly_revenue,
               SUM(units_sold)        AS monthly_units
        FROM retail_curated
        GROUP BY year_month
        ORDER BY year_month
    """)
    logger.info("Monthly Revenue Trend:")
    monthly_df.show(30, truncate=False)

    # Top 10 products
    top_df = spark.sql("""
        SELECT product_id,
               ROUND(SUM(revenue), 2) AS total_revenue,
               SUM(units_sold)        AS total_units,
               ROUND(AVG(price), 2)   AS avg_price
        FROM retail_curated
        GROUP BY product_id
        ORDER BY total_revenue DESC
        LIMIT 10
    """)
    logger.info("Top 10 Products:")
    top_df.show(truncate=False)

    # Region performance
    region_df = spark.sql("""
        SELECT region,
               ROUND(SUM(revenue), 2) AS total_revenue,
               COUNT(DISTINCT store_id) AS store_count
        FROM retail_curated
        GROUP BY region
        ORDER BY total_revenue DESC
    """)
    logger.info("Region Performance:")
    region_df.show(truncate=False)

    # Save analytics as Parquet
    os.makedirs(ANALYTICS_PATH, exist_ok=True)
    cat_df.coalesce(1).write.mode("overwrite").parquet(
        os.path.join(ANALYTICS_PATH, "category_revenue"))
    monthly_df.coalesce(1).write.mode("overwrite").parquet(
        os.path.join(ANALYTICS_PATH, "monthly_trend"))
    top_df.coalesce(1).write.mode("overwrite").parquet(
        os.path.join(ANALYTICS_PATH, "top_products"))

    logger.info("Analytics saved as Parquet: %s", ANALYTICS_PATH)
    results["analytics_saved"] = True

    logger.info("=" * 60)
    logger.info("SPARK PIPELINE COMPLETE")
    logger.info("=" * 60)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PANDAS FALLBACK PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pandas_pipeline() -> dict:
    """Pandas pipeline — mirrors Spark logic for local dev without PySpark."""
    import pandas as pd
    import numpy as np

    results = {}

    # ── STAGE 1: RAW ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 — RAW (pandas mode)")
    logger.info("=" * 60)

    raw_df = pd.read_csv(RAW_CSV_PATH)
    logger.info("RAW row count: %d", len(raw_df))
    results["raw_count"] = len(raw_df)

    # ── STAGE 2: STAGED ───────────────────────────────────────────────────────
    logger.info("STAGE 2 — STAGED: Cleaning & validating")

    df = raw_df.copy()
    df = df.drop_duplicates()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "price", "units_sold"])
    df["product_id"] = df["product_id"].astype(str).str.strip().str.upper()
    df["store_id"]   = df["store_id"].astype(str).str.strip().str.upper()
    df["category"]   = df["category"].astype(str).str.strip().str.title()
    df["region"]     = df["region"].astype(str).str.strip().str.title()
    df["discount"]   = df["discount"].fillna(0.0)
    df = df[df["price"] > 0]
    df = df[df["units_sold"] >= 0]
    df = df[df["discount"].between(0, 100)]
    df = df[df["category"].isin(VALID_CATEGORIES)]
    df = df[df["region"].isin(VALID_REGIONS)]
    df["revenue"] = (df["price"] * df["units_sold"] * (1 - df["discount"] / 100)).round(2)
    df = df.reset_index(drop=True)

    logger.info("STAGED row count: %d", len(df))
    results["staged_count"] = len(df)

    # Save staged as CSV and Parquet
    os.makedirs(STAGED_PATH, exist_ok=True)
    staged_csv = os.path.join(STAGED_PATH, "staged.csv")
    df.to_csv(staged_csv, index=False)
    logger.info("STAGED CSV saved: %s", staged_csv)

    # Save as Parquet (if pyarrow available)
    os.makedirs(PARQUET_PATH, exist_ok=True)
    staged_parquet = os.path.join(PARQUET_PATH, "staged.parquet")
    try:
        df.to_parquet(staged_parquet, index=False, engine="pyarrow")
        logger.info("STAGED Parquet saved: %s", staged_parquet)
    except ImportError:
        logger.warning("pyarrow not installed — skipping Parquet. Install: pip install pyarrow")
        staged_parquet = staged_csv

    # ── STAGE 3: CURATED ──────────────────────────────────────────────────────
    logger.info("STAGE 3 — CURATED: Feature enrichment")

    df["sale_day"]     = df["date"].dt.day
    df["sale_month"]   = df["date"].dt.month
    df["sale_year"]    = df["date"].dt.year
    df["sale_quarter"] = df["date"].dt.quarter
    df["weekday_num"]  = df["date"].dt.weekday
    df["is_weekend"]   = (df["weekday_num"] >= 5).astype(int)
    df["year_month"]   = df["date"].dt.to_period("M").astype(str)
    df["net_price"]    = (df["price"] * (1 - df["discount"] / 100)).round(2)
    df["price_x_discount"] = (df["price"] * df["discount"]).round(2)

    df["avg_units_by_product"]  = df.groupby("product_id")["units_sold"].transform("mean")
    df["std_units_by_product"]  = df.groupby("product_id")["units_sold"].transform("std").fillna(0)
    df["avg_units_by_category"] = df.groupby("category")["units_sold"].transform("mean")
    df["avg_units_by_region"]   = df.groupby("region")["units_sold"].transform("mean")
    df["avg_units_by_store"]    = df.groupby("store_id")["units_sold"].transform("mean")
    df["total_revenue_by_product"] = df.groupby("product_id")["revenue"].transform("sum")

    logger.info("CURATED row count: %d", len(df))
    results["curated_count"] = len(df)

    # Save curated as CSV
    os.makedirs(CURATED_PATH, exist_ok=True)
    curated_csv = os.path.join(CURATED_PATH, "curated.csv")
    df.to_csv(curated_csv, index=False)
    logger.info("CURATED CSV saved: %s", curated_csv)

    # Save curated as Parquet (partitioned by year/month)
    try:
        for (year, month), group in df.groupby(["sale_year", "sale_month"]):
            partition_dir = os.path.join(PARQUET_PATH, "curated",
                                         f"year={year}", f"month={month:02d}")
            os.makedirs(partition_dir, exist_ok=True)
            group.to_parquet(
                os.path.join(partition_dir, "data.parquet"),
                index=False, engine="pyarrow",
            )
        logger.info("CURATED Parquet (partitioned by year/month) saved: %s",
                    os.path.join(PARQUET_PATH, "curated"))
    except ImportError:
        logger.warning("pyarrow not installed — Parquet skipped. Install: pip install pyarrow")

    # ── STAGE 4: SQL ANALYTICS ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 — ANALYTICS: SQL-style queries")
    logger.info("=" * 60)

    os.makedirs(ANALYTICS_PATH, exist_ok=True)

    # Revenue by category
    cat_summary = (df.groupby("category")
                   .agg(total_revenue=("revenue", "sum"),
                        total_units=("units_sold", "sum"),
                        record_count=("revenue", "count"),
                        avg_discount=("discount", "mean"))
                   .round(2)
                   .sort_values("total_revenue", ascending=False)
                   .reset_index())
    cat_summary.to_csv(os.path.join(ANALYTICS_PATH, "category_revenue.csv"), index=False)
    try:
        cat_summary.to_parquet(os.path.join(ANALYTICS_PATH, "category_revenue.parquet"),
                               index=False, engine="pyarrow")
    except ImportError:
        pass
    logger.info("Revenue by Category:\n%s", cat_summary.to_string(index=False))

    # Monthly revenue trend
    monthly = (df.groupby("year_month")
               .agg(monthly_revenue=("revenue", "sum"),
                    monthly_units=("units_sold", "sum"))
               .round(2)
               .sort_index()
               .reset_index())
    monthly.to_csv(os.path.join(ANALYTICS_PATH, "monthly_trend.csv"), index=False)
    try:
        monthly.to_parquet(os.path.join(ANALYTICS_PATH, "monthly_trend.parquet"),
                           index=False, engine="pyarrow")
    except ImportError:
        pass
    logger.info("Monthly Trend:\n%s", monthly.to_string(index=False))

    # Top 10 products
    top_products = (df.groupby("product_id")
                    .agg(total_revenue=("revenue", "sum"),
                         total_units=("units_sold", "sum"),
                         avg_price=("price", "mean"))
                    .round(2)
                    .sort_values("total_revenue", ascending=False)
                    .head(10)
                    .reset_index())
    top_products.to_csv(os.path.join(ANALYTICS_PATH, "top_products.csv"), index=False)
    try:
        top_products.to_parquet(os.path.join(ANALYTICS_PATH, "top_products.parquet"),
                                index=False, engine="pyarrow")
    except ImportError:
        pass
    logger.info("Top 10 Products:\n%s", top_products.to_string(index=False))

    # Region performance
    region_perf = (df.groupby("region")
                   .agg(total_revenue=("revenue", "sum"),
                        store_count=("store_id", "nunique"))
                   .round(2)
                   .sort_values("total_revenue", ascending=False)
                   .reset_index())
    region_perf.to_csv(os.path.join(ANALYTICS_PATH, "region_performance.csv"), index=False)
    try:
        region_perf.to_parquet(os.path.join(ANALYTICS_PATH, "region_performance.parquet"),
                               index=False, engine="pyarrow")
    except ImportError:
        pass
    logger.info("Region Performance:\n%s", region_perf.to_string(index=False))

    results["analytics_saved"] = True

    logger.info("=" * 60)
    logger.info("PANDAS PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info("  Staged CSV    : %s", staged_csv)
    logger.info("  Staged Parquet: %s", staged_parquet)
    logger.info("  Curated CSV   : %s", curated_csv)
    logger.info("  Curated Parquet (partitioned): %s", os.path.join(PARQUET_PATH, "curated"))
    logger.info("  Analytics     : %s", ANALYTICS_PATH)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not os.path.exists(RAW_CSV_PATH):
        logger.error("Raw CSV not found: %s", RAW_CSV_PATH)
        logger.error("Run: python scripts/generate_sample_data.py")
        sys.exit(1)

    if spark is not None:
        results = run_spark_pipeline()
    else:
        results = run_pandas_pipeline()

    logger.info("Pipeline results: %s", results)
