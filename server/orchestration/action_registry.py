import logging
import os
from typing import Optional

import pandas

from server.retrieval.doc_retriever import retrieve_policy_answer

logger = logging.getLogger(__name__)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

RETAIL_CSV_PATH: str = os.path.join(_project_root, "sales_store", "retail_records.csv")


def _get_retail_dataframe() -> Optional[pandas.DataFrame]:
    """Load the retail CSV and return a cleaned DataFrame, or None on failure."""
    if not os.path.exists(RETAIL_CSV_PATH):
        logger.warning("Retail CSV not found at: %s", RETAIL_CSV_PATH)
        return None
    try:
        from server.forecasting.csv_reader import read_retail_csv
        from server.forecasting.record_cleaner import clean_retail_records

        raw_dataframe = read_retail_csv(RETAIL_CSV_PATH)
        cleaned_dataframe = clean_retail_records(raw_dataframe)
        return cleaned_dataframe
    except Exception as load_error:
        logger.warning("Failed to load retail dataframe: %s", load_error)
        return None


def pull_sales_overview(params: dict) -> str:
    """Return a high-level sales summary from the retail dataset."""
    retail_dataframe = _get_retail_dataframe()
    if retail_dataframe is None:
        return "Sales data is not available."
    total_revenue = retail_dataframe["revenue"].sum()
    total_units = retail_dataframe["units_sold"].sum()
    total_records = len(retail_dataframe)
    unique_products = retail_dataframe["product_id"].nunique()
    unique_stores = retail_dataframe["store_id"].nunique()
    return (
        f"Sales Overview: {total_records} records | "
        f"Total Revenue: ${total_revenue:,.2f} | "
        f"Total Units Sold: {total_units:,.0f} | "
        f"Unique Products: {unique_products} | "
        f"Unique Stores: {unique_stores}"
    )


def pull_top_earners(params: dict) -> str:
    """Return the top revenue-generating products."""
    retail_dataframe = _get_retail_dataframe()
    if retail_dataframe is None:
        return "Sales data is not available."
    top_count = int(params.get("top_n", 5))
    top_products = (
        retail_dataframe.groupby("product_id")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_count)
    )
    result_lines = [f"Top {top_count} Products by Revenue:"]
    for rank, (product_identifier, product_revenue) in enumerate(top_products.items(), start=1):
        result_lines.append(f"  {rank}. {product_identifier}: ${product_revenue:,.2f}")
    return "\n".join(result_lines)


def pull_category_revenue(params: dict) -> str:
    """Return revenue breakdown by product category."""
    retail_dataframe = _get_retail_dataframe()
    if retail_dataframe is None:
        return "Sales data is not available."
    category_revenue = (
        retail_dataframe.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    result_lines = ["Revenue by Category:"]
    for category_name, category_total in category_revenue.items():
        result_lines.append(f"  {category_name}: ${category_total:,.2f}")
    return "\n".join(result_lines)


def pull_region_revenue(params: dict) -> str:
    """Return revenue breakdown by sales region."""
    retail_dataframe = _get_retail_dataframe()
    if retail_dataframe is None:
        return "Sales data is not available."
    region_revenue = (
        retail_dataframe.groupby("region")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    result_lines = ["Revenue by Region:"]
    for region_name, region_total in region_revenue.items():
        result_lines.append(f"  {region_name}: ${region_total:,.2f}")
    return "\n".join(result_lines)


def pull_monthly_revenue(params: dict) -> str:
    """Return monthly revenue trend."""
    retail_dataframe = _get_retail_dataframe()
    if retail_dataframe is None:
        return "Sales data is not available."
    retail_dataframe["year_month"] = retail_dataframe["date"].dt.to_period("M")
    monthly_revenue = (
        retail_dataframe.groupby("year_month")["revenue"]
        .sum()
        .sort_index()
    )
    result_lines = ["Monthly Revenue Trend:"]
    for month_period, month_total in monthly_revenue.items():
        result_lines.append(f"  {month_period}: ${month_total:,.2f}")
    return "\n".join(result_lines)


def pull_product_breakdown(params: dict) -> str:
    """Return detailed breakdown for a specific product."""
    retail_dataframe = _get_retail_dataframe()
    if retail_dataframe is None:
        return "Sales data is not available."
    product_identifier = str(params.get("product_id", "")).upper()
    if not product_identifier:
        return "Please provide a product_id parameter."
    product_rows = retail_dataframe[retail_dataframe["product_id"] == product_identifier]
    if product_rows.empty:
        return f"No data found for product: {product_identifier}"
    total_revenue = product_rows["revenue"].sum()
    total_units = product_rows["units_sold"].sum()
    avg_price = product_rows["price"].mean()
    avg_discount = product_rows["discount"].mean()
    return (
        f"Product {product_identifier}: "
        f"Revenue=${total_revenue:,.2f} | "
        f"Units={total_units:,.0f} | "
        f"Avg Price=${avg_price:.2f} | "
        f"Avg Discount={avg_discount:.1f}%"
    )


def search_policy_docs(params: dict) -> str:
    """Search policy documents and return the answer."""
    search_query = str(params.get("query", "retail policy"))
    retrieval_result = retrieve_policy_answer(search_query)
    return retrieval_result.get("answer", "No answer found.")


def describe_ml_model(params: dict) -> str:
    """Describe the machine learning model used for demand forecasting."""
    return (
        "The demand forecasting model is a RandomForestRegressor trained on historical retail sales data. "
        "It uses 34 engineered features including date components, price features, lag values (1, 7, 14, 30 days), "
        "rolling statistics (7, 14, 30-day windows), and label-encoded categorical identifiers. "
        "The model is trained with 50% max_features, bootstrap sampling, and parallel execution. "
        "Performance is evaluated using MAE, RMSE, and R² on a held-out test set."
    )


def describe_anomaly_system(params: dict) -> str:
    """Describe the anomaly detection system."""
    return (
        "The anomaly detection system uses IsolationForest on four features: price, discount, units_sold, and revenue. "
        "Anomalies are classified as High Sales Anomaly or Low Sales Anomaly based on a spike multiplier threshold. "
        "Reasons are assigned: discount-driven spikes (discount >= 20%), unexpected spikes, stockouts (zero units), "
        "or low demand issues. Results are sorted and capped at a configurable maximum."
    )


def describe_forecast_system(params: dict) -> str:
    """Describe the full forecasting pipeline."""
    return (
        "The forecasting pipeline consists of: (1) CSV ingestion with mandatory column validation, "
        "(2) data cleaning with deduplication, date parsing, numeric coercion, and category normalization, "
        "(3) feature engineering with 34 features across date, price, lag, rolling, and aggregate dimensions, "
        "(4) RandomForest training with 99th percentile outlier removal and holdout evaluation, "
        "(5) demand prediction for new product/date combinations using the saved model and encoding pipeline."
    )


ACTION_REGISTRY: list[dict] = [
    {
        "name": "pull_sales_overview",
        "description": "Get a high-level summary of total sales, revenue, and product counts.",
        "parameters": {},
        "keywords": ["overview", "summary", "total", "sales", "revenue", "how many", "overall"],
        "handler": pull_sales_overview,
    },
    {
        "name": "pull_top_earners",
        "description": "Get the top revenue-generating products.",
        "parameters": {"top_n": "Number of top products to return (default 5)"},
        "keywords": ["top", "best", "highest", "earner", "product", "leading", "most revenue"],
        "handler": pull_top_earners,
    },
    {
        "name": "pull_category_revenue",
        "description": "Get revenue breakdown by product category.",
        "parameters": {},
        "keywords": ["category", "categories", "segment", "type", "department"],
        "handler": pull_category_revenue,
    },
    {
        "name": "pull_region_revenue",
        "description": "Get revenue breakdown by sales region.",
        "parameters": {},
        "keywords": ["region", "area", "location", "geography", "zone", "territory"],
        "handler": pull_region_revenue,
    },
    {
        "name": "pull_monthly_revenue",
        "description": "Get monthly revenue trend over time.",
        "parameters": {},
        "keywords": ["monthly", "month", "trend", "over time", "timeline", "period"],
        "handler": pull_monthly_revenue,
    },
    {
        "name": "pull_product_breakdown",
        "description": "Get detailed metrics for a specific product.",
        "parameters": {"product_id": "The product identifier to look up"},
        "keywords": ["product", "item", "sku", "breakdown", "detail", "specific"],
        "handler": pull_product_breakdown,
    },
    {
        "name": "search_policy_docs",
        "description": "Search retail policy documents for rules and guidelines.",
        "parameters": {"query": "The policy question to search for"},
        "keywords": ["policy", "rule", "return", "promo", "promotion", "stock", "guideline", "procedure"],
        "handler": search_policy_docs,
    },
    {
        "name": "describe_ml_model",
        "description": "Describe the machine learning demand forecasting model.",
        "parameters": {},
        "keywords": ["model", "machine learning", "ml", "random forest", "algorithm", "training"],
        "handler": describe_ml_model,
    },
    {
        "name": "describe_anomaly_system",
        "description": "Describe the anomaly detection system.",
        "parameters": {},
        "keywords": ["anomaly", "spike", "outlier", "detection", "isolation forest", "unusual"],
        "handler": describe_anomaly_system,
    },
    {
        "name": "describe_forecast_system",
        "description": "Describe the full demand forecasting pipeline.",
        "parameters": {},
        "keywords": ["forecast", "prediction", "pipeline", "predict", "demand", "future"],
        "handler": describe_forecast_system,
    },
]


def pick_best_action(message_text: str) -> dict:
    """Select the best matching action from the registry based on keyword scoring."""
    normalized_message = message_text.lower()
    best_action_definition = ACTION_REGISTRY[0]
    best_action_score = 0

    for action_definition in ACTION_REGISTRY:
        action_score = sum(
            1 for keyword in action_definition["keywords"] if keyword in normalized_message
        )
        if action_score > best_action_score:
            best_action_score = action_score
            best_action_definition = action_definition

    logger.info("Selected action: %s (score=%d)", best_action_definition["name"], best_action_score)
    return best_action_definition


def parse_action_params(message_text: str, action_def: dict) -> dict:
    """Extract parameter values from the message text for the given action."""
    extracted_params: dict = {}
    normalized_message = message_text.lower()

    if action_def["name"] == "pull_top_earners":
        for number_token in normalized_message.split():
            if number_token.isdigit():
                extracted_params["top_n"] = int(number_token)
                break

    if action_def["name"] == "pull_product_breakdown":
        for word_token in message_text.upper().split():
            if word_token.startswith("P") and len(word_token) >= 3:
                extracted_params["product_id"] = word_token
                break

    if action_def["name"] == "search_policy_docs":
        extracted_params["query"] = message_text

    return extracted_params


def run_action(message_text: str) -> dict:
    """Pick the best action, parse params, execute, and return the result."""
    selected_action = pick_best_action(message_text)
    action_params = parse_action_params(message_text, selected_action)

    try:
        action_result = selected_action["handler"](action_params)
    except Exception as action_error:
        logger.warning("Action %s failed: %s", selected_action["name"], action_error)
        action_result = f"Action failed: {action_error}"

    return {
        "action_name": selected_action["name"],
        "action_description": selected_action["description"],
        "result": action_result,
    }


def get_all_actions() -> list:
    """Return all registered actions without their handler callables."""
    return [
        {
            "name": action_definition["name"],
            "description": action_definition["description"],
            "parameters": action_definition["parameters"],
            "keywords": action_definition["keywords"],
        }
        for action_definition in ACTION_REGISTRY
    ]
