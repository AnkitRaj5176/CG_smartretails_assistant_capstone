# Databricks notebook source
# Smart Retail Analytics Engine — Azure Databricks Pipeline
# Attach to a Databricks cluster with Delta Lake support
# MAGIC %md
# ## Smart Retail Analytics Engine — Data Engineering Pipeline
# ### Raw → Staged → Curated (Delta Lake)

# COMMAND ----------
# MAGIC %md ### 1. Configuration

storage_account = "<YOUR_STORAGE_ACCOUNT>"
container_raw     = "raw"
container_staged  = "staged"
container_curated = "curated"

raw_path     = f"abfss://{container_raw}@{storage_account}.dfs.core.windows.net/retail/retail_records.csv"
staged_path  = f"abfss://{container_staged}@{storage_account}.dfs.core.windows.net/retail/"
curated_path = f"abfss://{container_curated}@{storage_account}.dfs.core.windows.net/retail/"

# COMMAND ----------
# MAGIC %md ### 2. RAW — Ingest from Azure Data Lake

from pyspark.sql import functions as F
from pyspark.sql.types import *

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
          .csv(raw_path))

print(f"RAW rows: {raw_df.count()}")
raw_df.printSchema()
display(raw_df.limit(5))

# COMMAND ----------
# MAGIC %md ### 3. STAGED — Clean & Validate

VALID_CATEGORIES = ["Electronics","Clothing","Groceries","Furniture","Sports",
                    "Beauty","Toys","Books","Automotive","Health"]
VALID_REGIONS    = ["North","South","East","West","Central",
                    "Northeast","Northwest","Southeast","Southwest"]

staged_df = (raw_df
    .dropDuplicates()
    .withColumn("date",       F.to_date("date", "yyyy-MM-dd"))
    .dropna(subset=["date","price","units_sold"])
    .withColumn("product_id", F.upper(F.trim("product_id")))
    .withColumn("store_id",   F.upper(F.trim("store_id")))
    .withColumn("category",   F.initcap(F.trim("category")))
    .withColumn("region",     F.initcap(F.trim("region")))
    .fillna({"discount": 0.0})
    .filter(F.col("price") > 0)
    .filter(F.col("units_sold") >= 0)
    .filter(F.col("discount").between(0, 100))
    .filter(F.col("category").isin(VALID_CATEGORIES))
    .filter(F.col("region").isin(VALID_REGIONS))
    .withColumn("revenue",
                F.round(F.col("price") * F.col("units_sold")
                        * (1 - F.col("discount") / 100), 2))
)

print(f"STAGED rows: {staged_df.count()}")
display(staged_df.limit(5))

# Write as Delta table
(staged_df.write
 .format("delta")
 .mode("overwrite")
 .save(staged_path))
print(f"STAGED Delta table written to: {staged_path}")

# COMMAND ----------
# MAGIC %md ### 4. CURATED — Feature Enrichment

curated_df = (staged_df
    .withColumn("sale_day",     F.dayofmonth("date"))
    .withColumn("sale_month",   F.month("date"))
    .withColumn("sale_year",    F.year("date"))
    .withColumn("sale_quarter", F.quarter("date"))
    .withColumn("weekday_num",  F.dayofweek("date"))
    .withColumn("is_weekend",   (F.col("weekday_num") >= 6).cast("int"))
    .withColumn("year_month",   F.date_format("date", "yyyy-MM"))
    .withColumn("net_price",    F.round(F.col("price") * (1 - F.col("discount") / 100), 2))
    .withColumn("price_x_discount", F.round(F.col("price") * F.col("discount"), 2))
)

# Product-level aggregates
product_agg = (staged_df.groupBy("product_id").agg(
    F.avg("units_sold").alias("avg_units_by_product"),
    F.stddev("units_sold").alias("std_units_by_product"),
    F.sum("revenue").alias("total_revenue_by_product"),
))

# Category & region aggregates
category_agg = staged_df.groupBy("category").agg(
    F.avg("units_sold").alias("avg_units_by_category"))
region_agg = staged_df.groupBy("region").agg(
    F.avg("units_sold").alias("avg_units_by_region"))

curated_df = (curated_df
    .join(product_agg,  on="product_id", how="left")
    .join(category_agg, on="category",   how="left")
    .join(region_agg,   on="region",     how="left")
    .fillna(0.0))

print(f"CURATED rows: {curated_df.count()}")
display(curated_df.limit(5))

# Write as Delta table
(curated_df.write
 .format("delta")
 .mode("overwrite")
 .partitionBy("sale_year", "sale_month")
 .save(curated_path))
print(f"CURATED Delta table written to: {curated_path}")

# COMMAND ----------
# MAGIC %md ### 5. Analytics — Spark SQL Queries

curated_df.createOrReplaceTempView("retail_curated")

# Revenue by category
spark.sql("""
    SELECT category,
           ROUND(SUM(revenue), 2)   AS total_revenue,
           SUM(units_sold)          AS total_units,
           COUNT(*)                 AS record_count
    FROM retail_curated
    GROUP BY category
    ORDER BY total_revenue DESC
""").show(truncate=False)

# Monthly revenue trend
spark.sql("""
    SELECT year_month,
           ROUND(SUM(revenue), 2) AS monthly_revenue
    FROM retail_curated
    GROUP BY year_month
    ORDER BY year_month
""").show(30, truncate=False)

# Top 10 products
spark.sql("""
    SELECT product_id,
           ROUND(SUM(revenue), 2) AS total_revenue,
           SUM(units_sold)        AS total_units
    FROM retail_curated
    GROUP BY product_id
    ORDER BY total_revenue DESC
    LIMIT 10
""").show(truncate=False)

# COMMAND ----------
# MAGIC %md ### Pipeline Complete ✅
