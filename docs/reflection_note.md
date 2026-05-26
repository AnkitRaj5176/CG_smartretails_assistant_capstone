# Reflection Note
## Smart Retail Analytics Engine — Capstone Project
### Left Shift Program 2026 — Data & AI (T5)

---

## Project Summary

Built an end-to-end Multi-Agent AI Platform for retail analytics covering demand forecasting,
anomaly detection, policy Q&A, and business intelligence. The system integrates FastAPI,
scikit-learn ML models, Azure OpenAI GenAI agents, TF-IDF RAG, MongoDB, PySpark data
engineering, Docker containerization, and GitHub Actions CI/CD.

---

## Challenges Faced

### 1. Multi-Agent Orchestration Without LangChain
**Challenge:** The capstone required a multi-agent system but adding LangChain/CrewAI
would have introduced heavy dependencies and complexity.

**Solution:** Built a lightweight custom orchestrator (`query_handler.py`) with keyword-based
agent classification and a clean action registry pattern. Each agent (Analytics, Policy,
Forecast) has its own system prompt and context-fetching logic, then calls Azure OpenAI
for natural language generation.

**Learning:** Custom orchestration gives more control and is easier to debug than
framework-based approaches for well-defined agent boundaries.

---

### 2. Azure OpenAI Offline Fallback
**Challenge:** The project needed to run locally without Azure credentials during development
and testing, but also support real LLM responses in production.

**Solution:** Implemented a feature flag (`USE_AZURE_OPENAI=true/false`) in `llm_client.py`.
When disabled, the system returns structured data directly with an offline mode label.
This allowed 47 tests to pass without any Azure credentials.

**Learning:** Always design AI systems with graceful degradation — the core business logic
should work independently of external AI services.

---

### 3. Feature Engineering for Time-Series Data
**Challenge:** The RandomForest model needed temporal features (lag values, rolling averages)
but the data was not strictly time-ordered per product.

**Solution:** Sorted by `[product_id, date]` before computing lag and rolling features,
used `groupby().transform()` to compute per-product statistics, and applied 99th percentile
outlier removal before training.

**Learning:** Feature engineering for retail demand forecasting requires careful handling
of temporal ordering and product-level grouping to avoid data leakage.

---

### 4. PySpark Local vs Databricks
**Challenge:** PySpark requires Java and a full Spark installation locally, which is heavy
for development.

**Solution:** Built the pipeline with automatic environment detection — uses PySpark when
available (Databricks/production), falls back to pandas for local development. Both paths
produce identical output (staged CSV and curated CSV with the same schema).

**Learning:** Design data pipelines to be environment-agnostic. The business logic should
be the same whether running on a laptop or a Databricks cluster.

---

### 5. MongoDB Availability
**Challenge:** MongoDB may not be running locally during development or testing.

**Solution:** All MongoDB operations are wrapped in try/except blocks with graceful
degradation. The application starts and serves all endpoints even if MongoDB is unavailable
— it just logs a warning instead of crashing.

**Learning:** External dependencies should never be single points of failure. Always
implement circuit-breaker patterns for database connections.

---

## Key Learnings

1. **End-to-end thinking** — Building a complete platform (data → ML → GenAI → API → deployment)
   requires careful interface design between each layer.

2. **Offline-first development** — Designing systems to work without cloud services locally
   dramatically speeds up development and testing cycles.

3. **Feature engineering matters more than model choice** — The 34 engineered features
   (especially lag and rolling statistics) contributed more to model performance than
   hyperparameter tuning.

4. **RAG is powerful even without embeddings** — TF-IDF retrieval with cosine similarity
   provides surprisingly good policy document search without requiring vector databases
   or embedding models.

5. **CI/CD from day one** — Setting up GitHub Actions early caught integration issues
   before they became hard to debug.

---

## Optimizations Applied

| Area | Optimization | Impact |
|---|---|---|
| ML Training | 99th percentile outlier removal | Reduced noise in training data |
| ML Training | `n_jobs=-1` parallel training | Faster training on multi-core |
| Feature Engineering | Per-product groupby transforms | Correct temporal features |
| MongoDB | Background index creation | No blocking during startup |
| API | Graceful error handling everywhere | No 500 crashes in production |
| GenAI | Offline fallback mode | Works without Azure credentials |
| Data Pipeline | Pandas fallback for PySpark | Runs on any machine |

---

## Future Improvements

1. **Real embeddings** — Replace TF-IDF with Azure OpenAI `text-embedding-ada-002` for
   semantic search in the RAG pipeline.

2. **Vector store** — Add Azure AI Search or FAISS for scalable document retrieval.

3. **Model retraining trigger** — Automatically retrain when R² drops below 0.70 threshold
   (as defined in the policy document).

4. **Real-time streaming** — Add Azure Event Hub + Data Activator for real-time anomaly
   alerts as sales data streams in.

5. **Power BI live connection** — Connect Power BI directly to the `/api/metrics/overview`
   endpoint using the Web connector for live dashboard updates.

6. **LLM fine-tuning** — Fine-tune a smaller model on retail-specific Q&A pairs for
   faster and cheaper inference than GPT-4o.

---

## Technologies Used

| Category | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | scikit-learn (RandomForest, IsolationForest), joblib, pandas, numpy |
| GenAI | Azure OpenAI (GPT-4o), TF-IDF RAG |
| Database | MongoDB, PyMongo |
| Data Engineering | PySpark, pandas, Azure Databricks (Delta Lake) |
| Testing | pytest, httpx, FastAPI TestClient |
| Deployment | Docker, docker-compose, GitHub Actions, Azure Web App |
| Security | Azure Key Vault, environment variables, non-root Docker user |
