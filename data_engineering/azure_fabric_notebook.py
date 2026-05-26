# Azure Fabric Notebook — Smart Retail Analytics Engine
# Section E: Data Engineering Pipeline
# Run this notebook in Microsoft Fabric (Lakehouse)
#
# Prerequisites:
#   1. Create a Fabric Lakehouse named "RetailLakehouse"
#   2. Upload retail_records.csv to Files/raw/retail/
#   3. Attach this notebook to the Lakehouse

# COMMAND ----------
# MAGIC %md
# ## Smart Retail Analytics Engine — Azure Fabric Pipeline
# ### Lakehouse: RAW → STAGED → CURATED (Delta Tables)

# COMMAND ----------
# MAGIC %md ### 1. Configuration

LAKEHOUSE_NAME = "RetailLakehouse"
RAW_PATH       = "Files/raw/retail/retail_records.csv"
STAGED_TABLE   = "staged_retail"
CURATED_TABLE  = "curated_retail"

VALID_CATEGORIES = [
    "Electronics", "Clothing", "Groceries", "Furniture", "Sports",
    "Beauty", "Toys", "Books", "Automotive", "Health"
]
VALID_REGIONS = [
    "North", "South", "East", "West", "Central",
    "Northeast", "Northwest", "Southeast", "Southwest"
]

print(f"Lakehouse: {LAKEHOUSE_NAME}")
print(f"Raw path : {RAW_PATH}")

# COMMAND ----------
# MAGIC %md ### 2. RAW — Read from Lakehouse Files

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
          .csv(f"abfss://{LAKEHOUSE_NAME}@onelake.dfs.fabric.microsoft.com/{RAW_PATH}"))

print(f"RAW rows: {raw_df.count()}")
display(raw_df.limit(5))

# COMMAND ----------
# MAGIC %md ### 3. STAGED — Clean & Validate → Delta Table

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
    .filter(F.col("category").isin(VALID_CATEGORIES))
    .filter(F.col("region").isin(VALID_REGIONS))
    .withColumn("revenue",
                F.round(F.col("price") * F.col("units_sold")
                        * (1 - F.col("discount") / 100), 2))
)

print(f"STAGED rows: {staged_df.count()}")
display(staged_df.limit(5))

# Write as Delta table to Lakehouse
staged_df.write.format("delta").mode("overwrite").saveAsTable(STAGED_TABLE)
print(f"Delta table '{STAGED_TABLE}' created in Lakehouse.")

# COMMAND ----------
# MAGIC %md ### 4. CURATED — Feature Enrichment → Partitioned Delta Table

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

# Aggregate features
product_agg  = staged_df.groupBy("product_id").agg(
    F.avg("units_sold").alias("avg_units_by_product"),
    F.stddev("units_sold").alias("std_units_by_product"),
    F.sum("revenue").alias("total_revenue_by_product"))
category_agg = staged_df.groupBy("category").agg(
    F.avg("units_sold").alias("avg_units_by_category"))
region_agg   = staged_df.groupBy("region").agg(
    F.avg("units_sold").alias("avg_units_by_region"))
store_agg    = staged_df.groupBy("store_id").agg(
    F.avg("units_sold").alias("avg_units_by_store"))

curated_df = (curated_df
    .join(product_agg,  on="product_id", how="left")
    .join(category_agg, on="category",   how="left")
    .join(region_agg,   on="region",     how="left")
    .join(store_agg,    on="store_id",   how="left")
    .fillna(0.0))

print(f"CURATED rows: {curated_df.count()}")
display(curated_df.limit(5))

# Write as partitioned Delta table
(curated_df.write
 .format("delta")
 .mode("overwrite")
 .partitionBy("sale_year", "sale_month")
 .saveAsTable(CURATED_TABLE))
print(f"Delta table '{CURATED_TABLE}' created (partitioned by year/month).")

# COMMAND ----------
# MAGIC %md ### 5. Spark SQL Analytics

spark.sql(f"USE {LAKEHOUSE_NAME}")

# Revenue by category
spark.sql(f"""
    SELECT category,
           ROUND(SUM(revenue), 2)   AS total_revenue,
           SUM(units_sold)          AS total_units,
           COUNT(*)                 AS record_count,
           ROUND(AVG(discount), 2)  AS avg_discount_pct
    FROM {CURATED_TABLE}
    GROUP BY category
    ORDER BY total_revenue DESC
""").show(truncate=False)

# Monthly revenue trend
spark.sql(f"""
    SELECT year_month,
           ROUND(SUM(revenue), 2) AS monthly_revenue,
           SUM(units_sold)        AS monthly_units
    FROM {CURATED_TABLE}
    GROUP BY year_month
    ORDER BY year_month
""").show(30, truncate=False)

# Top 10 products
spark.sql(f"""
    SELECT product_id,
           ROUND(SUM(revenue), 2) AS total_revenue,
           SUM(units_sold)        AS total_units
    FROM {CURATED_TABLE}
    GROUP BY product_id
    ORDER BY total_revenue DESC
    LIMIT 10
""").show(truncate=False)

# Region performance
spark.sql(f"""
    SELECT region,
           ROUND(SUM(revenue), 2)    AS total_revenue,
           COUNT(DISTINCT store_id)  AS store_count,
           ROUND(AVG(discount), 2)   AS avg_discount
    FROM {CURATED_TABLE}
    GROUP BY region
    ORDER BY total_revenue DESC
""").show(truncate=False)

# COMMAND ----------
# MAGIC %md ### 6. Data Activator Alert (Anomaly Trigger)
# MAGIC Configure Data Activator in Fabric to trigger alerts when:
# MAGIC - Daily revenue drops > 20% vs 7-day average
# MAGIC - Units sold = 0 for any product for 3+ consecutive days
# MAGIC - Discount > 40% detected (potential pricing error)

anomaly_check = spark.sql(f"""
    SELECT product_id, year_month,
           ROUND(SUM(revenue), 2) AS monthly_revenue,
           SUM(CASE WHEN units_sold = 0 THEN 1 ELSE 0 END) AS zero_sales_days,
           MAX(discount) AS max_discount
    FROM {CURATED_TABLE}
    GROUP BY product_id, year_month
    HAVING zero_sales_days > 3 OR max_discount > 40
    ORDER BY zero_sales_days DESC
""")
print(f"Anomaly alerts: {anomaly_check.count()} products flagged")
display(anomaly_check)

# COMMAND ----------
# MAGIC %md ### Pipeline Complete ✅
# MAGIC
# MAGIC Delta tables created:
# MAGIC - `staged_retail`  — cleaned data
# MAGIC - `curated_retail` — feature-enriched, partitioned by year/month
# MAGIC
# MAGIC Connect Power BI to these Delta tables for live dashboards.
