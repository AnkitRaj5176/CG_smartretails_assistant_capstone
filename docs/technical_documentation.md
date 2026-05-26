# Technical Documentation
## Smart Retail Analytics Engine v2.0.0

---

## 1. System Overview

The Smart Retail Analytics Engine is a production-ready Multi-Agent AI Platform built for
the Smart Retail Assistant domain. It provides:

- **Demand Forecasting** — RandomForest model predicting units sold
- **Anomaly Detection** — IsolationForest identifying unusual sales patterns
- **Policy Q&A** — TF-IDF RAG over retail policy documents
- **Multi-Agent Chat** — 3 specialized AI agents powered by Azure OpenAI
- **Business Analytics** — Revenue, category, region, and product metrics
- **Data Engineering** — PySpark pipeline: RAW → STAGED → CURATED

---

## 2. API Endpoints

### POST /api/data/upload
Accepts a CSV file, validates mandatory columns, cleans data, saves to disk and MongoDB.

**Request:** `multipart/form-data` with `file` field (CSV)

**Response:**
```json
{
  "status_message": "File uploaded and processed successfully.",
  "original_row_count": 5000,
  "cleaned_row_count": 4987
}
```

---

### POST /api/ml/train
Trains the RandomForest demand forecasting model on uploaded data.

**Request:**
```json
{
  "holdout_ratio": 0.2,
  "num_trees": 100,
  "tree_depth": 0,
  "random_seed": 42
}
```

**Response:**
```json
{
  "status": "Model trained and saved successfully.",
  "mae": 12.34,
  "rmse": 18.56,
  "r2": 0.87,
  "num_trees": 100,
  "holdout_ratio": 0.2
}
```

---

### POST /api/ml/predict
Predicts units sold for a given product, date, price, and store.

**Request:**
```json
{
  "product_id": "PROD_001",
  "target_date": "2025-06-15",
  "price": 299.99,
  "discount": 10.0,
  "store_id": "STORE_A",
  "region": "North"
}
```

**Response:**
```json
{
  "product_id": "PROD_001",
  "predicted_units_sold": 23.45,
  "model_used": "RandomForestRegressor"
}
```

---

### POST /api/ml/anomalies
Runs IsolationForest anomaly detection on the uploaded retail data.

**Request:**
```json
{
  "outlier_fraction": 0.05,
  "spike_multiplier": 2.0,
  "max_results": 50
}
```

**Response:**
```json
{
  "total_anomalies": 250,
  "high_sales_anomalies": 75,
  "low_sales_anomalies": 175,
  "anomalies": [
    {
      "product_id": "PROD_001",
      "date": "2024-11-25",
      "store_id": "STORE_A",
      "category": "Electronics",
      "region": "North",
      "price": 299.99,
      "discount": 30.0,
      "units_sold": 450,
      "revenue": 94496.85,
      "anomaly_label": "High Sales Anomaly",
      "reason": "Discount driven spike"
    }
  ]
}
```

---

### POST /api/docs/search
Searches retail policy documents using TF-IDF RAG.

**Request:**
```json
{"search_query": "return policy for electronics"}
```

**Response:**
```json
{
  "query": "return policy for electronics",
  "answer": "Electronics may be returned within 15 days of purchase...",
  "sources": ["retail_policy.txt"]
}
```

---

### POST /api/assistant/chat
Routes the message to the best-matching GenAI agent.

**Request:**
```json
{"user_message": "What is the total revenue by category?"}
```

**Response:**
```json
{
  "message": "What is the total revenue by category?",
  "agent": "AnalyticsAgent",
  "agent_description": "Retail sales analytics and business intelligence specialist",
  "tool_used": "pull_category_revenue",
  "raw_data": "Revenue by Category:\n  Electronics: $3,238,189.61\n...",
  "sources": [],
  "response": "Electronics leads with $3.2M revenue, followed by Groceries..."
}
```

---

### GET /api/metrics/overview
Returns dashboard KPIs for Power BI or frontend consumption.

**Response:**
```json
{
  "total_revenue": 17915475.45,
  "total_units_sold": 284500,
  "top_products": [
    {"product_id": "PROD_001", "revenue": 3238189.61}
  ],
  "revenue_by_category": {"Electronics": 3238189.61, "Groceries": 2384146.17},
  "revenue_by_region": {"Northwest": 2346308.62, "South": 2343318.86},
  "monthly_sales_trend": {"2023-01": 450000.00, "2023-02": 480000.00}
}
```

---

## 3. ML Model Details

### RandomForest Demand Forecasting

| Parameter | Value |
|---|---|
| Algorithm | RandomForestRegressor (scikit-learn) |
| Features | 34 engineered features |
| Outlier removal | 99th percentile filter on target |
| max_features | 0.5 (50% feature sampling) |
| bootstrap | True |
| n_jobs | -1 (all CPU cores) |
| Evaluation | MAE, RMSE, R² on holdout set |
| Persistence | joblib → model_vault/rf_demand_model.pkl |

### Feature Groups (34 total)

| Group | Features |
|---|---|
| Date (9) | sale_day, sale_month, sale_year, weekday_num, is_weekend_flag, sale_quarter, iso_week, early_month_flag, late_month_flag |
| Price (4) | net_price, squared_discount, price_x_discount, price_ratio_to_category |
| Lag (4) | units_lag_1, units_lag_7, units_lag_14, units_lag_30 |
| Rolling (7) | roll_avg_7, roll_avg_14, roll_avg_30, roll_std_7, roll_high_7, roll_low_7, momentum_ratio |
| Aggregate (6) | avg_units_by_product, std_units_by_product, avg_units_by_store, avg_units_by_category, avg_units_by_region, lag1_to_product_ratio |
| Encoded (4) | product_code, category_code, store_code, region_code |

### IsolationForest Anomaly Detection

| Parameter | Value |
|---|---|
| Algorithm | IsolationForest (scikit-learn) |
| Features | price, discount, units_sold, revenue |
| contamination | Configurable (default 0.05) |
| random_state | 42 |
| Classification | High Sales / Low Sales Anomaly |

---

## 4. Multi-Agent Architecture

### Agent Classification

```python
# Keyword scoring determines which agent handles the query
FORECAST_KEYWORDS = {"forecast", "predict", "demand", "anomaly", "spike", ...}
POLICY_KEYWORDS   = {"policy", "rule", "return", "refund", "promo", ...}
ANALYTICS_KEYWORDS = {"revenue", "sales", "top", "category", "region", ...}
```

### Agent Pipelines

**AnalyticsAgent:**
1. `run_action(message)` → fetches structured sales data
2. `call_llm(system_prompt, user_message, context)` → Azure OpenAI insight
3. Returns: agent, tool_used, raw_data, response

**PolicyAgent:**
1. `retrieve_policy_answer(query)` → TF-IDF top-3 chunks
2. `call_llm(system_prompt, user_message, context)` → grounded answer
3. Returns: agent, tool_used, raw_data, sources, response

**ForecastAgent:**
1. Builds ML context (model description + live sales snapshot)
2. `call_llm(system_prompt, user_message, context)` → interpretation
3. Returns: agent, tool_used, raw_data, response

---

## 5. Data Engineering Pipeline

### Stages

| Stage | Input | Output | Operations |
|---|---|---|---|
| RAW | CSV file | Spark DataFrame | Schema enforcement, read |
| STAGED | Raw DataFrame | Delta table / CSV | Dedup, date parse, normalize, filter, revenue recalc |
| CURATED | Staged DataFrame | Delta table / CSV (partitioned) | Date features, price features, aggregate joins |

### Spark SQL Analytics
- Revenue by category (ORDER BY DESC)
- Monthly revenue trend
- Top 10 products by revenue

---

## 6. Database Schema

### MongoDB Collection: `retail_records`

```json
{
  "product_id": "PROD_001",
  "category": "Electronics",
  "region": "North",
  "store_id": "STORE_A",
  "date": "2024-01-10T00:00:00",
  "price": 299.99,
  "discount": 10.0,
  "units_sold": 5,
  "revenue": 1349.96
}
```

### Indexes
- `idx_product_date` — (product_id ASC, date ASC) — compound
- `idx_store_id` — (store_id ASC)
- `idx_category` — (category ASC)
- `idx_region` — (region ASC)
- `idx_date` — (date ASC)

---

## 7. Security Considerations

| Concern | Implementation |
|---|---|
| API keys | Azure Key Vault (never in code) |
| Environment variables | `.env` file (gitignored) |
| Docker | Non-root `appuser` in container |
| Input validation | Pydantic models on all endpoints |
| Error handling | No stack traces exposed to clients |
| MongoDB | Connection timeout + graceful fallback |
| CORS | Configurable origins (restrict in production) |

---

## 8. Deployment

### Local
```bash
pip install -r requirements.txt
python -m uvicorn server.startup:retail_application --port 8000
```

### Docker
```bash
docker-compose up --build
```

### Azure Web App
1. Push to `main` branch
2. GitHub Actions runs tests → builds Docker image → pushes to GHCR
3. Azure Web App pulls latest image automatically
4. Secrets loaded from Azure Key Vault at runtime
