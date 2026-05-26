-- ═══════════════════════════════════════════════════════════════════════════
-- Smart Retail Analytics Engine — SQL Analytics Scripts
-- Section E: Data Engineering Pipeline
-- Compatible with: Spark SQL, Azure Synapse T-SQL, Azure Fabric SQL
-- ═══════════════════════════════════════════════════════════════════════════

-- ── 1. Revenue by Category ───────────────────────────────────────────────────
SELECT
    category,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    SUM(units_sold)                 AS total_units_sold,
    COUNT(*)                        AS record_count,
    ROUND(AVG(price), 2)            AS avg_price,
    ROUND(AVG(discount), 2)         AS avg_discount_pct,
    ROUND(SUM(revenue) * 100.0
          / SUM(SUM(revenue)) OVER (), 2) AS revenue_share_pct
FROM curated_retail
GROUP BY category
ORDER BY total_revenue DESC;

-- ── 2. Monthly Revenue Trend ─────────────────────────────────────────────────
SELECT
    year_month,
    ROUND(SUM(revenue), 2)          AS monthly_revenue,
    SUM(units_sold)                 AS monthly_units,
    COUNT(DISTINCT product_id)      AS active_products,
    COUNT(DISTINCT store_id)        AS active_stores,
    ROUND(AVG(discount), 2)         AS avg_discount_pct
FROM curated_retail
GROUP BY year_month
ORDER BY year_month;

-- ── 3. Top 15 Products by Revenue ────────────────────────────────────────────
SELECT
    product_id,
    category,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    SUM(units_sold)                 AS total_units,
    ROUND(AVG(price), 2)            AS avg_price,
    ROUND(AVG(discount), 2)         AS avg_discount_pct,
    COUNT(DISTINCT store_id)        AS stores_selling,
    COUNT(DISTINCT region)          AS regions_present
FROM curated_retail
GROUP BY product_id, category
ORDER BY total_revenue DESC
LIMIT 15;

-- ── 4. Region Performance ────────────────────────────────────────────────────
SELECT
    region,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    SUM(units_sold)                 AS total_units,
    COUNT(DISTINCT store_id)        AS store_count,
    COUNT(DISTINCT product_id)      AS product_count,
    ROUND(AVG(discount), 2)         AS avg_discount_pct,
    ROUND(SUM(revenue) / COUNT(DISTINCT store_id), 2) AS revenue_per_store
FROM curated_retail
GROUP BY region
ORDER BY total_revenue DESC;

-- ── 5. Store Performance ─────────────────────────────────────────────────────
SELECT
    store_id,
    region,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    SUM(units_sold)                 AS total_units,
    COUNT(DISTINCT product_id)      AS product_count,
    ROUND(AVG(discount), 2)         AS avg_discount_pct
FROM curated_retail
GROUP BY store_id, region
ORDER BY total_revenue DESC
LIMIT 20;

-- ── 6. Discount Impact Analysis ──────────────────────────────────────────────
SELECT
    CASE
        WHEN discount = 0          THEN '0% (No Discount)'
        WHEN discount BETWEEN 1 AND 10  THEN '1-10%'
        WHEN discount BETWEEN 11 AND 20 THEN '11-20%'
        WHEN discount BETWEEN 21 AND 30 THEN '21-30%'
        ELSE '30%+'
    END                             AS discount_band,
    COUNT(*)                        AS transaction_count,
    ROUND(AVG(units_sold), 2)       AS avg_units_sold,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    ROUND(AVG(revenue), 2)          AS avg_revenue_per_txn
FROM curated_retail
GROUP BY discount_band
ORDER BY discount_band;

-- ── 7. Seasonal Analysis (Quarter) ───────────────────────────────────────────
SELECT
    sale_year,
    sale_quarter,
    ROUND(SUM(revenue), 2)          AS quarterly_revenue,
    SUM(units_sold)                 AS quarterly_units,
    COUNT(*)                        AS transaction_count,
    ROUND(AVG(discount), 2)         AS avg_discount_pct
FROM curated_retail
GROUP BY sale_year, sale_quarter
ORDER BY sale_year, sale_quarter;

-- ── 8. Weekend vs Weekday Sales ──────────────────────────────────────────────
SELECT
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    COUNT(*)                        AS transaction_count,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    ROUND(AVG(units_sold), 2)       AS avg_units_per_txn,
    ROUND(AVG(discount), 2)         AS avg_discount_pct
FROM curated_retail
GROUP BY is_weekend
ORDER BY day_type;

-- ── 9. Product Category × Region Matrix ──────────────────────────────────────
SELECT
    category,
    region,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    SUM(units_sold)                 AS total_units
FROM curated_retail
GROUP BY category, region
ORDER BY category, total_revenue DESC;

-- ── 10. Anomaly Detection — Zero Sales Products ───────────────────────────────
SELECT
    product_id,
    category,
    store_id,
    region,
    year_month,
    SUM(CASE WHEN units_sold = 0 THEN 1 ELSE 0 END) AS zero_sales_days,
    ROUND(SUM(revenue), 2)          AS monthly_revenue,
    MAX(discount)                   AS max_discount_applied
FROM curated_retail
GROUP BY product_id, category, store_id, region, year_month
HAVING zero_sales_days > 0
ORDER BY zero_sales_days DESC
LIMIT 20;

-- ── 11. High Discount Transactions (Potential Anomalies) ─────────────────────
SELECT
    product_id,
    category,
    store_id,
    region,
    date,
    price,
    discount,
    units_sold,
    revenue
FROM curated_retail
WHERE discount >= 30
ORDER BY discount DESC, revenue DESC
LIMIT 50;

-- ── 12. Revenue Growth MoM (Month-over-Month) ────────────────────────────────
WITH monthly_revenue AS (
    SELECT
        year_month,
        ROUND(SUM(revenue), 2) AS monthly_revenue
    FROM curated_retail
    GROUP BY year_month
),
with_lag AS (
    SELECT
        year_month,
        monthly_revenue,
        LAG(monthly_revenue) OVER (ORDER BY year_month) AS prev_month_revenue
    FROM monthly_revenue
)
SELECT
    year_month,
    monthly_revenue,
    prev_month_revenue,
    ROUND((monthly_revenue - prev_month_revenue)
          / NULLIF(prev_month_revenue, 0) * 100, 2) AS mom_growth_pct
FROM with_lag
ORDER BY year_month;
