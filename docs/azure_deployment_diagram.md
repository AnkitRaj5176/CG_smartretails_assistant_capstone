# Azure Deployment Diagram
## Smart Retail Analytics Engine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEVELOPER / CI-CD                                 │
│                                                                             │
│   GitHub Repository                                                         │
│       │                                                                     │
│       ▼  push to main                                                       │
│   GitHub Actions CI/CD (.github/workflows/ci-cd.yml)                       │
│       ├── pytest (65 tests)                                                 │
│       ├── Docker build                                                      │
│       ├── Push → GitHub Container Registry (ghcr.io)                       │
│       └── Deploy → Azure Web App                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AZURE CLOUD INFRASTRUCTURE                           │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Azure Web App (Container)                         │  │
│  │                  smart-retail-engine.azurewebsites.net               │  │
│  │                                                                      │  │
│  │   FastAPI Application (port 8000)                                    │  │
│  │   ├── POST /api/data/upload        (data ingestion)                  │  │
│  │   ├── POST /api/ml/train           (ML training)                     │  │
│  │   ├── POST /api/ml/predict         (demand forecast)                 │  │
│  │   ├── POST /api/ml/anomalies       (anomaly detection)               │  │
│  │   ├── POST /api/assistant/chat     (multi-agent chat)                │  │
│  │   ├── POST /api/agent/mcp          (MCP protocol)                    │  │
│  │   ├── POST /api/docs/search        (RAG policy search)               │  │
│  │   ├── GET  /api/metrics/overview   (dashboard KPIs)                  │  │
│  │   ├── POST /api/azure/text/*       (Text Analytics)                  │  │
│  │   ├── POST /api/azure/search       (AI Search)                       │  │
│  │   ├── GET  /api/azure/ml/status    (Azure ML)                        │  │
│  │   └── GET  /api/azure/status       (all Azure services)              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│           │              │              │              │                    │
│           ▼              ▼              ▼              ▼                    │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────────┐   │
│  │ Azure OpenAI │ │  Azure     │ │   Azure AI   │ │   Azure Cosmos   │   │
│  │              │ │  Cognitive │ │   Search     │ │   DB (MongoDB    │   │
│  │  GPT-4o      │ │  Services  │ │              │ │   API)           │   │
│  │              │ │            │ │  Policy docs │ │                  │   │
│  │  3 Agents:   │ │ Text       │ │  indexed     │ │  retail_records  │   │
│  │  Analytics   │ │ Analytics  │ │              │ │  collection      │   │
│  │  Policy      │ │            │ │  Semantic    │ │  5 indexes       │   │
│  │  Forecast    │ │ Sentiment  │ │  search      │ │                  │   │
│  │              │ │ KeyPhrase  │ │              │ │                  │   │
│  │  Offline     │ │ Language   │ │  Fallback:   │ │                  │   │
│  │  fallback ✓  │ │ Detection  │ │  VectorStore │ │                  │   │
│  └──────────────┘ └────────────┘ └──────────────┘ └──────────────────┘   │
│                                                                             │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────────────────────────────┐ │
│  │  Azure ML    │ │  Azure     │ │        Azure Data Lake Storage       │ │
│  │              │ │  Key Vault │ │                                      │ │
│  │  Model       │ │            │ │  raw/      → staged/   → curated/   │ │
│  │  Registry    │ │  Secrets:  │ │  (CSV)       (Delta)     (Delta,    │ │
│  │              │ │  MongoConn │ │                           partitioned│ │
│  │  Training    │ │  OAI Key   │ │                           by year/  │ │
│  │  Jobs        │ │  OAI Endpt │ │                           month)    │ │
│  │              │ │  Search Key│ │                                      │ │
│  │  Online      │ │  TA Key    │ │  Azure Databricks                    │ │
│  │  Endpoints   │ │            │ │  PySpark Pipeline                    │ │
│  │              │ │  No secrets│ │  (retail_pipeline.py)                │ │
│  │  Fallback:   │ │  in code ✓ │ │                                      │ │
│  │  local pkl ✓ │ │            │ │                                      │ │
│  └──────────────┘ └────────────┘ └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONSUMERS                                      │
│                                                                             │
│   Power BI Dashboard          Swagger UI              External Apps         │
│   (GET /api/metrics/overview) (localhost:8000/docs)   (REST API)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Azure Components Used

| # | Component | Purpose | Fallback |
|---|---|---|---|
| 1 | **Azure OpenAI (GPT-4o)** | Multi-agent NLG | Local keyword response |
| 2 | **Azure Cognitive Services — Text Analytics** | Sentiment, key phrases, language | Local rule-based |
| 3 | **Azure AI Search** | Semantic policy document search | Local VectorStore |
| 4 | **Azure ML** | Model registry, training jobs | Local joblib |
| 5 | **Azure Key Vault** | Secret management | Environment variables |
| 6 | **Azure Web App** | Container deployment | Docker local |
| 7 | **Azure Cosmos DB** | MongoDB API database | Local MongoDB |
| 8 | **Azure Data Lake** | Raw/staged/curated storage | Local CSV |
| 9 | **Azure Databricks** | PySpark pipeline | Pandas pipeline |

## Security Considerations

| Concern | Implementation |
|---|---|
| API Keys | Azure Key Vault — never hardcoded |
| Environment Variables | `.env` file gitignored |
| Docker Security | Non-root `appuser` in container |
| Input Validation | Pydantic models on all endpoints |
| Error Handling | No stack traces exposed to clients |
| CORS | Configurable origins |
| MongoDB | Connection timeout + graceful fallback |
| Secrets in CI/CD | GitHub Secrets for AZURE_WEBAPP_PUBLISH_PROFILE |
