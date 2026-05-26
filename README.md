# Smart Retail Analytics Engine
### Capstone Project — Left Shift Program 2026 (Data & AI T5)

> End-to-end **Multi-Agent AI Platform** for retail demand forecasting, anomaly detection,
> policy Q&A, and business analytics — built with FastAPI, scikit-learn, Azure OpenAI,
> TF-IDF RAG, MongoDB, PySpark, Docker, and Azure.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data
python scripts/generate_sample_data.py

# 3. Run the server
python -m uvicorn server.startup:retail_application --host 0.0.0.0 --port 8000

# 4. Open Swagger UI
# http://localhost:8000/docs

# 5. Run tests
python -m pytest tests/ -v

# 6. Run data engineering pipeline
python data_engineering/retail_pipeline.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Client / Power BI / Swagger UI                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST (HTTP/JSON)
┌──────────────────────────────▼──────────────────────────────────────┐
│                  FastAPI Backend  (server/startup.py)               │
│                                                                     │
│  POST /api/data/upload       → CSV ingestion & validation           │
│  POST /api/ml/train          → Train RandomForest model             │
│  POST /api/ml/predict        → Predict units sold                   │
│  POST /api/ml/anomalies      → IsolationForest anomaly detection    │
│  POST /api/docs/search       → TF-IDF RAG policy search            │
│  POST /api/assistant/chat    → Multi-agent GenAI chat               │
│  GET  /api/assistant/actions → List all agent actions               │
│  GET  /api/metrics/overview  → Dashboard KPIs                       │
│  GET  /ping                  → Health check                         │
└──────┬───────────────────────┬─────────────────────────────────────┘
       │                       │
┌──────▼──────┐   ┌────────────▼────────────────────────────────────┐
│  MongoDB    │   │              Multi-Agent AI Layer                │
│  NoSQL DB   │   │                                                  │
│  5 indexes  │   │  AnalyticsAgent  → action_registry → Azure OAI  │
│  retail_    │   │  PolicyAgent     → TF-IDF RAG      → Azure OAI  │
│  records    │   │  ForecastAgent   → ML context      → Azure OAI  │
└─────────────┘   │                                                  │
                  │  ML Pipeline:                                    │
                  │  csv_reader → record_cleaner → feature_engineer  │
                  │  → rf_trainer (34 features, RandomForest)        │
                  │  → spike_detector (IsolationForest)              │
                  │  → units_predictor (joblib inference)            │
                  └─────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│              Data Engineering Pipeline (PySpark / Pandas)           │
│                                                                     │
│  RAW CSV → STAGED (cleaned) → CURATED (feature-enriched)           │
│  Delta tables / Parquet on Azure Data Lake                          │
│  Azure Databricks notebook included                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
retail_engine/
├── server/
│   ├── startup.py                    # FastAPI app, routers, lifespan
│   ├── env_config.py                 # Environment config (MongoDB + Azure OAI)
│   ├── .env                          # Environment variables (fill in your keys)
│   ├── endpoints/
│   │   ├── upload_endpoint.py        # POST /api/data/upload
│   │   ├── ml_endpoint.py            # POST /api/ml/train|predict|anomalies
│   │   ├── lookup_endpoint.py        # POST /api/docs/search
│   │   └── chat_endpoint.py          # POST /api/assistant/chat + GET metrics
│   ├── forecasting/
│   │   ├── csv_reader.py             # CSV ingestion + column validation
│   │   ├── record_cleaner.py         # Data cleaning & normalization
│   │   ├── feature_engineer.py       # 34-feature engineering pipeline
│   │   ├── rf_trainer.py             # RandomForest training + evaluation
│   │   ├── spike_detector.py         # IsolationForest anomaly detection
│   │   └── units_predictor.py        # Demand prediction from saved model
│   ├── genai/
│   │   ├── llm_client.py             # Azure OpenAI wrapper + offline fallback
│   │   ├── analytics_agent.py        # Data Analyst Agent
│   │   ├── policy_agent.py           # Document Assistant Agent (RAG)
│   │   └── forecast_agent.py         # ML Expert Agent
│   ├── retrieval/
│   │   └── doc_retriever.py          # TF-IDF RAG over raw_docs/*.txt
│   ├── orchestration/
│   │   ├── action_registry.py        # 10 actions + keyword routing
│   │   └── query_handler.py          # 3-agent classifier + orchestrator
│   └── infra/
│       └── mongo_store.py            # MongoDB persistence + indexes
├── data_engineering/
│   ├── retail_pipeline.py            # PySpark/Pandas: RAW→STAGED→CURATED
│   └── databricks_notebook.py        # Azure Databricks Delta Lake notebook
├── tests/
│   ├── conftest.py                   # Shared fixtures
│   ├── test_csv_reader.py            # 3 tests
│   ├── test_record_cleaner.py        # 7 tests
│   ├── test_feature_engineer.py      # 5 tests
│   ├── test_query_handler.py         # 10 tests
│   ├── test_genai_agents.py          # 6 tests
│   └── test_api_endpoints.py         # 16 tests  → Total: 47 tests
├── scripts/
│   └── generate_sample_data.py       # 5000-row synthetic retail CSV
├── raw_docs/
│   └── retail_policy.txt             # Policy knowledge base for RAG
├── sales_store/                      # Runtime: uploaded CSV stored here
├── model_vault/                      # Runtime: trained model .pkl files
├── azure/
│   └── azure-deploy.yml              # Azure Web App + Key Vault config
├── .github/
│   └── workflows/
│       └── ci-cd.yml                 # GitHub Actions: test→build→deploy
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Mandatory Components Coverage

| Component | Requirement | Implementation | Status |
|---|---|---|---|
| **A. Python Fullstack** | FastAPI backend | `server/startup.py` | ✅ |
| | Min 4 REST APIs | 9 endpoints across 4 routers | ✅ |
| | NoSQL database | MongoDB + 5 indexes | ✅ |
| | Logging & error handling | `logging` module throughout | ✅ |
| | Unit testing (pytest) | 47 tests, all passing | ✅ |
| **B. ML/DL** | ML model | RandomForestRegressor | ✅ |
| | Clean data pipeline | csv_reader → record_cleaner | ✅ |
| | Feature engineering | 34 features | ✅ |
| | Training + evaluation | MAE, RMSE, R² | ✅ |
| | Model persistence | joblib (.pkl) | ✅ |
| | Anomaly detection | IsolationForest | ✅ |
| **C. GenAI/Agents** | 3-agent system | Analytics, Policy, Forecast | ✅ |
| | Prompt engineering | System prompts per agent | ✅ |
| | RAG | TF-IDF over policy docs | ✅ |
| | Azure OpenAI | `llm_client.py` + fallback | ✅ |
| | Multi-agent orchestration | `query_handler.py` | ✅ |
| **D. Azure AI & Cloud** | Azure OpenAI | `genai/llm_client.py` | ✅ |
| | Azure Web App | `azure/azure-deploy.yml` | ✅ |
| | Key Vault | Secrets in deploy config | ✅ |
| | Deployment diagram | See below | ✅ |
| **E. Data Engineering** | PySpark pipeline | `data_engineering/retail_pipeline.py` | ✅ |
| | RAW→STAGED→CURATED | 3-stage pipeline | ✅ |
| | Databricks notebook | `databricks_notebook.py` | ✅ |
| | Delta tables | In Databricks notebook | ✅ |
| **F. Analytics** | Metrics API | `GET /api/metrics/overview` | ✅ |
| | Power BI ready | JSON output for PBI connection | ✅ |
| **G. Deployment** | Docker | `Dockerfile` + `docker-compose.yml` | ✅ |
| | CI/CD | GitHub Actions | ✅ |
| | Azure App Service | `azure-deploy.yml` | ✅ |

---

## API Reference

### Data Ingestion
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/data/upload` | Upload retail CSV (multipart/form-data) |

### Machine Learning
| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/ml/train` | `{"num_trees":100,"holdout_ratio":0.2}` | Train RandomForest |
| `POST` | `/api/ml/predict` | `{"product_id":"PROD_001","target_date":"2025-06-01","price":299.99,"discount":10,"store_id":"STORE_A","region":"North"}` | Predict demand |
| `POST` | `/api/ml/anomalies` | `{"outlier_fraction":0.05,"spike_multiplier":2.0,"max_results":50}` | Detect anomalies |

### Analytics
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/metrics/overview` | Revenue, top products, monthly trend |

### GenAI / Agents
| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/assistant/chat` | `{"user_message":"show me sales overview"}` | Multi-agent chat |
| `GET` | `/api/assistant/actions` | — | List all 10 agent actions |
| `POST` | `/api/docs/search` | `{"search_query":"return policy"}` | RAG policy search |

### System
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/ping` | Health check |
| `GET` | `/docs` | Swagger UI |

---

## Multi-Agent System

Three specialized agents, orchestrated by keyword classification:

| Agent | Keywords | Responsibility |
|---|---|---|
| **AnalyticsAgent** | revenue, sales, top, category, region, monthly | Business intelligence from retail data |
| **PolicyAgent** | policy, rule, return, refund, promo, guideline | Policy Q&A via TF-IDF RAG |
| **ForecastAgent** | forecast, predict, demand, anomaly, spike | Demand forecasting insights |

**Flow:**
```
User Message
    → classify_query()          (keyword scoring)
    → Agent selected
    → Agent fetches context     (data / RAG / ML)
    → call_llm()                (Azure OpenAI or offline fallback)
    → Structured JSON response
```

---

## ML Pipeline

```
Raw CSV
  └─► csv_reader.py         validate mandatory columns
        └─► record_cleaner.py    dedup, date parse, numeric coerce,
                                  category/region filter, revenue recalc
              └─► feature_engineer.py   34 features:
                    - Date: day, month, year, weekday, weekend, quarter, week
                    - Price: net_price, squared_discount, price×discount, ratio
                    - Lag: units_lag_1/7/14/30
                    - Rolling: avg/std/high/low (7,14,30-day windows)
                    - Aggregate: avg by product/store/category/region
                    - Encoded: product_code, category_code, store_code, region_code
                        └─► rf_trainer.py    RandomForestRegressor
                              99th-pct outlier removal
                              MAE / RMSE / R² evaluation
                              joblib persistence → model_vault/
```

---

## Data Engineering Pipeline

```
Azure Data Lake (raw CSV)
    │
    ▼ Stage 1: RAW
    Read with schema enforcement
    │
    ▼ Stage 2: STAGED (Delta table)
    - Drop duplicates
    - Parse dates
    - Normalize strings (UPPER/Title case)
    - Filter invalid categories/regions
    - Recalculate revenue
    │
    ▼ Stage 3: CURATED (Delta table, partitioned by year/month)
    - Date features (day, month, quarter, weekday, weekend)
    - Price features (net_price, price×discount)
    - Aggregate features (avg/std by product, category, region)
    │
    ▼ Spark SQL Analytics
    - Revenue by category
    - Monthly revenue trend
    - Top 10 products
```

---

## Azure Deployment Architecture

```
GitHub Repository
    │
    ▼ GitHub Actions CI/CD
    ├── pytest (47 tests)
    ├── Docker build & push → GitHub Container Registry
    └── Azure Web App deploy
            │
            ▼
    Azure Web App (Container)
    ├── FastAPI application (port 8000)
    ├── Environment variables from Azure Key Vault
    │       ├── MONGO_CONNECTION
    │       ├── AZURE_OPENAI_KEY
    │       └── AZURE_OPENAI_ENDPOINT
    │
    ├── Azure OpenAI (GPT-4o)
    │       └── 3 GenAI agents
    │
    ├── Azure Cosmos DB (MongoDB API)
    │       └── retail_records collection
    │
    └── Azure Data Lake Storage
            └── raw / staged / curated containers
                    │
                    ▼
            Azure Databricks
            └── PySpark pipeline (Delta Lake)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MONGO_CONNECTION` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `smart_retail` | Database name |
| `USE_AZURE_OPENAI` | `false` | Set `true` to enable real LLM |
| `AZURE_OPENAI_KEY` | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-02-01` | API version |

---

## CSV Format

| Column | Type | Rules |
|---|---|---|
| `product_id` | string | Any identifier |
| `category` | string | Must be one of 10 valid categories |
| `region` | string | Must be one of 9 valid regions |
| `store_id` | string | Any identifier |
| `date` | date | YYYY-MM-DD format |
| `price` | float | Must be > 0 |
| `discount` | float | 0–100 range |
| `units_sold` | int | Must be ≥ 0 |
| `revenue` | float | Auto-recalculated on upload |

**Valid categories:** Electronics, Clothing, Groceries, Furniture, Sports, Beauty, Toys, Books, Automotive, Health

**Valid regions:** North, South, East, West, Central, Northeast, Northwest, Southeast, Southwest

---

## Docker

```bash
# Build and run with MongoDB
docker-compose up --build

# API available at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

---

## Testing

```bash
pytest tests/ -v
# 47 tests across 6 test files
# test_csv_reader.py        — 3 tests
# test_record_cleaner.py    — 7 tests
# test_feature_engineer.py  — 5 tests
# test_query_handler.py     — 10 tests
# test_genai_agents.py      — 6 tests
# test_api_endpoints.py     — 16 tests
```
#   C G _ s m a r t r e t a i l s _ a s s i s t a n t _ c a p s t o n e  
 