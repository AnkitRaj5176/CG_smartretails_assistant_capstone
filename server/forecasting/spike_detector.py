import logging

import pandas
from sklearn.ensemble import IsolationForest

from server.forecasting.csv_reader import read_retail_csv
from server.forecasting.record_cleaner import clean_retail_records

logger = logging.getLogger(__name__)


def _classify_anomaly_reason(row: pandas.Series) -> str:
    """Determine the human-readable reason for an anomaly."""
    if row["anomaly_label"] == "High Sales Anomaly":
        if row["discount"] >= 20:
            return "Discount driven spike"
        return "Unexpected sales spike"
    if row["units_sold"] == 0:
        return "Stockout — zero units sold"
    return "Low demand issue"


def detect_sales_spikes(
    csv_path: str,
    outlier_fraction: float,
    spike_multiplier: float,
    max_results: int,
) -> dict:
    """Run IsolationForest on sales data and return classified anomalies."""
    raw_dataframe = read_retail_csv(csv_path)
    cleaned_dataframe = clean_retail_records(raw_dataframe)

    detection_features = cleaned_dataframe[["price", "discount", "units_sold", "revenue"]].copy()

    isolation_model = IsolationForest(
        contamination=outlier_fraction,
        random_state=42,
        n_jobs=-1,
    )
    anomaly_predictions = isolation_model.fit_predict(detection_features)

    cleaned_dataframe["is_anomaly"] = anomaly_predictions == -1
    anomaly_dataframe = cleaned_dataframe[cleaned_dataframe["is_anomaly"]].copy()

    units_mean = cleaned_dataframe["units_sold"].mean()
    units_threshold = units_mean * spike_multiplier

    anomaly_dataframe["anomaly_label"] = anomaly_dataframe["units_sold"].apply(
        lambda units: "High Sales Anomaly" if units >= units_threshold else "Low Sales Anomaly"
    )

    anomaly_dataframe["reason"] = anomaly_dataframe.apply(_classify_anomaly_reason, axis=1)

    high_sales_count = int((anomaly_dataframe["anomaly_label"] == "High Sales Anomaly").sum())
    low_sales_count = int((anomaly_dataframe["anomaly_label"] == "Low Sales Anomaly").sum())

    top_anomalies = anomaly_dataframe.head(max_results)

    anomaly_list = []
    for _, anomaly_row in top_anomalies.iterrows():
        anomaly_list.append(
            {
                "product_id": anomaly_row["product_id"],
                "date": str(anomaly_row["date"].date()),
                "store_id": anomaly_row["store_id"],
                "category": anomaly_row["category"],
                "region": anomaly_row["region"],
                "price": float(anomaly_row["price"]),
                "discount": float(anomaly_row["discount"]),
                "units_sold": float(anomaly_row["units_sold"]),
                "revenue": float(anomaly_row["revenue"]),
                "anomaly_label": anomaly_row["anomaly_label"],
                "reason": anomaly_row["reason"],
            }
        )

    logger.info(
        "Anomaly detection complete — total=%d, high=%d, low=%d",
        len(anomaly_dataframe),
        high_sales_count,
        low_sales_count,
    )

    return {
        "total_anomalies": len(anomaly_dataframe),
        "high_sales_anomalies": high_sales_count,
        "low_sales_anomalies": low_sales_count,
        "anomalies": anomaly_list,
    }
