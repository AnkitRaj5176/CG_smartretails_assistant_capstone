"""
agent_tools.py
──────────────
LangChain-style Agent Tool definitions.

Each tool is a callable with:
  - name        : unique identifier
  - description : what the tool does (used in prompts)
  - parameters  : expected input keys
  - run(params) : executes the tool and returns a string result

The ToolRegistry manages all tools and allows agents to discover
and invoke them by name — this is the "tool use" pattern from
LangChain / OpenAI function calling.
"""

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RETAIL_CSV_PATH = os.path.join(_project_root, "sales_store", "retail_records.csv")
_RF_MODEL_PATH   = os.path.join(_project_root, "model_vault", "rf_demand_model.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Tool base class
# ══════════════════════════════════════════════════════════════════════════════

class AgentTool:
    """A single callable tool that an agent can invoke."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable[[dict], str],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._handler = handler

    def run(self, params: dict) -> str:
        """Execute the tool with the given parameters."""
        try:
            result = self._handler(params)
            logger.info("Tool '%s' executed successfully.", self.name)
            return result
        except Exception as e:
            logger.warning("Tool '%s' failed: %s", self.name, e)
            return f"Tool execution failed: {e}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Tool handlers
# ══════════════════════════════════════════════════════════════════════════════

def _load_df():
    """Load and clean the retail dataframe."""
    if not os.path.exists(_RETAIL_CSV_PATH):
        return None
    from server.forecasting.csv_reader import read_retail_csv
    from server.forecasting.record_cleaner import clean_retail_records
    return clean_retail_records(read_retail_csv(_RETAIL_CSV_PATH))


def _tool_sales_overview(params: dict) -> str:
    df = _load_df()
    if df is None:
        return "No sales data available. Please upload a CSV first."
    return (
        f"Total Records: {len(df):,} | "
        f"Total Revenue: ${df['revenue'].sum():,.2f} | "
        f"Total Units Sold: {df['units_sold'].sum():,.0f} | "
        f"Unique Products: {df['product_id'].nunique()} | "
        f"Unique Stores: {df['store_id'].nunique()} | "
        f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}"
    )


def _tool_top_products(params: dict) -> str:
    df = _load_df()
    if df is None:
        return "No sales data available."
    n = int(params.get("top_n", 5))
    top = df.groupby("product_id")["revenue"].sum().sort_values(ascending=False).head(n)
    lines = [f"Top {n} Products by Revenue:"]
    for rank, (pid, rev) in enumerate(top.items(), 1):
        lines.append(f"  {rank}. {pid}: ${rev:,.2f}")
    return "\n".join(lines)


def _tool_category_breakdown(params: dict) -> str:
    df = _load_df()
    if df is None:
        return "No sales data available."
    cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    lines = ["Revenue by Category:"]
    for name, val in cat.items():
        lines.append(f"  {name}: ${val:,.2f}")
    return "\n".join(lines)


def _tool_region_breakdown(params: dict) -> str:
    df = _load_df()
    if df is None:
        return "No sales data available."
    reg = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    lines = ["Revenue by Region:"]
    for name, val in reg.items():
        lines.append(f"  {name}: ${val:,.2f}")
    return "\n".join(lines)


def _tool_monthly_trend(params: dict) -> str:
    df = _load_df()
    if df is None:
        return "No sales data available."
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("year_month")["revenue"].sum().sort_index()
    lines = ["Monthly Revenue Trend:"]
    for period, val in monthly.items():
        lines.append(f"  {period}: ${val:,.2f}")
    return "\n".join(lines)


def _tool_product_detail(params: dict) -> str:
    df = _load_df()
    if df is None:
        return "No sales data available."
    pid = str(params.get("product_id", "")).upper()
    if not pid:
        return "Please provide product_id parameter."
    rows = df[df["product_id"] == pid]
    if rows.empty:
        return f"No data found for product: {pid}"
    return (
        f"Product {pid}: "
        f"Revenue=${rows['revenue'].sum():,.2f} | "
        f"Units={rows['units_sold'].sum():,.0f} | "
        f"Avg Price=${rows['price'].mean():.2f} | "
        f"Avg Discount={rows['discount'].mean():.1f}% | "
        f"Records={len(rows)}"
    )


def _tool_anomaly_summary(params: dict) -> str:
    if not os.path.exists(_RETAIL_CSV_PATH):
        return "No sales data available."
    try:
        from server.forecasting.spike_detector import detect_sales_spikes
        result = detect_sales_spikes(
            csv_path=_RETAIL_CSV_PATH,
            outlier_fraction=float(params.get("outlier_fraction", 0.05)),
            spike_multiplier=float(params.get("spike_multiplier", 2.0)),
            max_results=int(params.get("max_results", 5)),
        )
        top = result["anomalies"][:3]
        lines = [
            f"Anomaly Detection Summary:",
            f"  Total anomalies: {result['total_anomalies']}",
            f"  High sales: {result['high_sales_anomalies']}",
            f"  Low sales: {result['low_sales_anomalies']}",
            "  Top anomalies:",
        ]
        for a in top:
            lines.append(
                f"    - {a['product_id']} on {a['date']}: "
                f"{a['units_sold']} units ({a['anomaly_label']}) — {a['reason']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Anomaly detection failed: {e}"


def _tool_demand_forecast(params: dict) -> str:
    if not os.path.exists(_RF_MODEL_PATH):
        return "Model not trained yet. Please call /api/ml/train first."
    try:
        from server.forecasting.units_predictor import forecast_product_demand
        result = forecast_product_demand(
            product_code=str(params.get("product_id", "PROD_001")),
            target_date=str(params.get("target_date", "2025-06-01")),
            sale_price=float(params.get("price", 100.0)),
            discount_amount=float(params.get("discount", 0.0)),
            outlet_id=str(params.get("store_id", "STORE_A")),
            sales_region=str(params.get("region", "North")),
        )
        return (
            f"Demand Forecast: {result['product_id']} | "
            f"Date: {params.get('target_date')} | "
            f"Predicted Units: {result['predicted_units_sold']} | "
            f"Model: {result['model_used']}"
        )
    except Exception as e:
        return f"Forecast failed: {e}"


def _tool_policy_search(params: dict) -> str:
    query = str(params.get("query", "retail policy"))
    from server.retrieval.doc_retriever import retrieve_policy_answer
    result = retrieve_policy_answer(query)
    return result.get("full_context") or result.get("answer", "No answer found.")


def _tool_model_info(params: dict) -> str:
    model_exists = os.path.exists(_RF_MODEL_PATH)
    return (
        f"ML Model Status: {'Trained and ready' if model_exists else 'Not trained yet'}\n"
        f"Algorithm: RandomForestRegressor\n"
        f"Features: 34 engineered features (date, price, lag, rolling, aggregate)\n"
        f"Anomaly Detection: IsolationForest on price, discount, units_sold, revenue\n"
        f"Evaluation Metrics: MAE, RMSE, R²\n"
        f"Persistence: joblib (.pkl) in model_vault/"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Registry of all available agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def run(self, name: str, params: dict) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        return tool.run(params)

    def list_tools(self) -> list[dict]:
        return [t.to_dict() for t in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


# ── Build the global registry ──────────────────────────────────────────────────

TOOL_REGISTRY = ToolRegistry()

TOOL_REGISTRY.register(AgentTool(
    name="get_sales_overview",
    description="Get high-level sales summary: total revenue, units, products, stores, date range.",
    parameters={},
    handler=_tool_sales_overview,
))
TOOL_REGISTRY.register(AgentTool(
    name="get_top_products",
    description="Get top revenue-generating products. Optional: top_n (default 5).",
    parameters={"top_n": "int — number of products to return"},
    handler=_tool_top_products,
))
TOOL_REGISTRY.register(AgentTool(
    name="get_category_breakdown",
    description="Get revenue breakdown by product category.",
    parameters={},
    handler=_tool_category_breakdown,
))
TOOL_REGISTRY.register(AgentTool(
    name="get_region_breakdown",
    description="Get revenue breakdown by sales region.",
    parameters={},
    handler=_tool_region_breakdown,
))
TOOL_REGISTRY.register(AgentTool(
    name="get_monthly_trend",
    description="Get monthly revenue trend over time.",
    parameters={},
    handler=_tool_monthly_trend,
))
TOOL_REGISTRY.register(AgentTool(
    name="get_product_detail",
    description="Get detailed metrics for a specific product. Required: product_id.",
    parameters={"product_id": "str — product identifier e.g. PROD_001"},
    handler=_tool_product_detail,
))
TOOL_REGISTRY.register(AgentTool(
    name="get_anomaly_summary",
    description="Run anomaly detection and return summary of unusual sales patterns.",
    parameters={
        "outlier_fraction": "float — expected fraction of outliers (default 0.05)",
        "spike_multiplier": "float — threshold multiplier (default 2.0)",
        "max_results": "int — max anomalies to return (default 5)",
    },
    handler=_tool_anomaly_summary,
))
TOOL_REGISTRY.register(AgentTool(
    name="get_demand_forecast",
    description="Predict units sold for a product on a specific date.",
    parameters={
        "product_id": "str — product identifier",
        "target_date": "str — date in YYYY-MM-DD format",
        "price": "float — sale price",
        "discount": "float — discount percentage",
        "store_id": "str — store identifier",
        "region": "str — sales region",
    },
    handler=_tool_demand_forecast,
))
TOOL_REGISTRY.register(AgentTool(
    name="search_policy",
    description="Search retail policy documents for rules, guidelines, and procedures.",
    parameters={"query": "str — the policy question to search for"},
    handler=_tool_policy_search,
))
TOOL_REGISTRY.register(AgentTool(
    name="get_model_info",
    description="Get information about the ML models used for forecasting and anomaly detection.",
    parameters={},
    handler=_tool_model_info,
))
