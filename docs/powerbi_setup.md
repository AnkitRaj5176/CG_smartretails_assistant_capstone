# Power BI Dashboard Setup Guide
## Smart Retail Analytics Engine — Section F

---

## Step 1 — Prerequisites

1. Download **Power BI Desktop** (free): https://powerbi.microsoft.com/desktop
2. Make sure the server is running: `python -m uvicorn server.startup:retail_application --port 8000`
3. Upload data: `POST /api/data/upload` with retail_records.csv
4. Train model: `POST /api/ml/train`

---

## Step 2 — Connect Power BI to API

### Method A — Full Dashboard (Recommended, 1 connection)

1. Open Power BI Desktop
2. Click **Get Data** → **Web**
3. Enter URL:
   ```
   http://localhost:8000/api/powerbi/full-dashboard
   ```
4. Click **OK** → **Connect**
5. In Navigator, select the data → **Transform Data**
6. Power Query Editor opens — expand nested records

### Method B — Individual endpoints (separate tables)

Connect each endpoint as a separate table:

| Table Name | URL |
|---|---|
| Key Metrics | `http://localhost:8000/api/powerbi/key-metrics` |
| Category Revenue | `http://localhost:8000/api/powerbi/revenue-by-category` |
| Region Revenue | `http://localhost:8000/api/powerbi/revenue-by-region` |
| Monthly Trend | `http://localhost:8000/api/powerbi/monthly-trend` |
| Top Products | `http://localhost:8000/api/powerbi/top-products` |
| Anomaly Alerts | `http://localhost:8000/api/powerbi/anomaly-alerts` |
| Model Performance | `http://localhost:8000/api/powerbi/model-performance` |
| Agent Insights | `http://localhost:8000/api/powerbi/agent-insights` |

---

## Step 3 — Build the Dashboard

### Page 1 — Key Metrics (KPI Cards)
- Add **Card** visual → Field: `total_revenue` → Title: "Total Revenue"
- Add **Card** visual → Field: `total_units_sold` → Title: "Units Sold"
- Add **Card** visual → Field: `unique_products` → Title: "Products"
- Add **Card** visual → Field: `avg_discount_pct` → Title: "Avg Discount %"
- Add **Card** visual → Field: `ml_model_trained` → Title: "ML Model Status"

### Page 2 — Revenue Analysis
- Add **Bar Chart** → X: `category`, Y: `total_revenue` → Title: "Revenue by Category"
- Add **Pie Chart** → Legend: `category`, Values: `revenue_share_pct`
- Add **Bar Chart** → X: `region`, Y: `total_revenue` → Title: "Revenue by Region"
- Add **Table** → Columns: `product_id`, `total_revenue`, `total_units`, `avg_price`

### Page 3 — Trends
- Add **Line Chart** → X: `year_month`, Y: `monthly_revenue` → Title: "Monthly Revenue Trend"
- Add **Line Chart** → X: `year_month`, Y: `mom_growth_pct` → Title: "Month-over-Month Growth %"
- Add **Area Chart** → X: `year_month`, Y: `monthly_units`

### Page 4 — Anomaly Alerts
- Add **Table** visual with columns:
  - `product_id`, `date`, `store_id`, `category`, `units_sold`, `anomaly_label`, `reason`
- Add **Conditional Formatting** on `anomaly_label`:
  - "High Sales Anomaly" → Red background
  - "Low Sales Anomaly" → Orange background
- Add **Card** → `total_anomalies` → Title: "Total Anomalies Detected"
- Add **Card** → `high_sales_anomalies` → Title: "High Sales Spikes"
- Add **Card** → `low_sales_anomalies` → Title: "Low Sales Issues"

### Page 5 — ML Model Performance
- Add **Card** → `mae` → Title: "MAE (Mean Absolute Error)"
- Add **Card** → `rmse` → Title: "RMSE"
- Add **Card** → `r2` → Title: "R² Score"
- Add **Card** → `feature_count` → Title: "Features Used"
- Add **Text Box** → Model description

### Page 6 — Agent Insights (Optional)
- Add **Table** → Columns: `agent`, `query`, `insight`
- Add **Text Box** for each agent insight

---

## Step 4 — Publish & Share

### Publish to Power BI Service
1. Click **Publish** in Power BI Desktop
2. Sign in with Microsoft account
3. Select workspace → **My Workspace**
4. Click **Publish**
5. Open Power BI Service: https://app.powerbi.com
6. Find your report → Click **Share**
7. Enter email addresses to share

### Schedule Refresh (for live data)
1. In Power BI Service → Dataset settings
2. **Scheduled refresh** → Turn on
3. Set frequency: Daily
4. Set time: 6:00 AM

---

## API Endpoints Summary for Power BI

| Endpoint | Data | Power BI Visual |
|---|---|---|
| `/api/powerbi/key-metrics` | KPIs | Card visuals |
| `/api/powerbi/revenue-by-category` | Category breakdown | Bar/Pie chart |
| `/api/powerbi/revenue-by-region` | Region breakdown | Map/Bar chart |
| `/api/powerbi/monthly-trend` | Time series | Line/Area chart |
| `/api/powerbi/top-products` | Product ranking | Bar chart/Table |
| `/api/powerbi/anomaly-alerts` | Anomaly data | Table + alerts |
| `/api/powerbi/model-performance` | ML metrics | KPI cards |
| `/api/powerbi/agent-insights` | AI insights | Text cards |
| `/api/powerbi/full-dashboard` | Everything | Full report |
