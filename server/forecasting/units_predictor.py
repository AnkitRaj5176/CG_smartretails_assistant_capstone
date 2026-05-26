import logging
import os

import joblib
import numpy
import pandas

logger = logging.getLogger(__name__)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_RF_MODEL_PATH: str = os.path.join(_project_root, "model_vault", "rf_demand_model.pkl")
_PIPELINE_PATH: str = os.path.join(_project_root, "model_vault", "encoding_pipeline.pkl")

_BASELINE_ESTIMATE: float = 20.0


def _encode_label(encoder_map: dict, column_name: str, raw_value: str) -> int:
    """Return the encoded integer for a label, defaulting to 0 if unseen."""
    label_mapping = encoder_map.get(column_name, {})
    return label_mapping.get(raw_value, 0)


def forecast_product_demand(
    product_code: str,
    target_date: str,
    sale_price: float,
    discount_amount: float,
    outlet_id: str,
    sales_region: str,
) -> dict:
    """Load the trained model and predict units sold for the given product and date."""
    if not os.path.exists(_RF_MODEL_PATH):
        raise FileNotFoundError(f"Random Forest model not found at: {_RF_MODEL_PATH}")
    if not os.path.exists(_PIPELINE_PATH):
        raise FileNotFoundError(f"Encoding pipeline not found at: {_PIPELINE_PATH}")

    demand_model = joblib.load(_RF_MODEL_PATH)
    pipeline_data = joblib.load(_PIPELINE_PATH)
    encoder_map: dict = pipeline_data["encoder_map"]

    parsed_date = pandas.to_datetime(target_date)

    sale_day = parsed_date.day
    sale_month = parsed_date.month
    sale_year = parsed_date.year
    weekday_num = parsed_date.weekday()
    is_weekend_flag = int(weekday_num >= 5)
    sale_quarter = parsed_date.quarter
    iso_week = int(parsed_date.isocalendar()[1])
    early_month_flag = int(sale_day <= 10)
    late_month_flag = int(sale_day >= 21)

    net_price = sale_price * (1 - discount_amount / 100)
    squared_discount = discount_amount ** 2
    price_x_discount = sale_price * discount_amount
    price_ratio_to_category = 1.0

    units_lag_1 = _BASELINE_ESTIMATE
    units_lag_7 = _BASELINE_ESTIMATE
    units_lag_14 = _BASELINE_ESTIMATE
    units_lag_30 = _BASELINE_ESTIMATE

    roll_avg_7 = _BASELINE_ESTIMATE
    roll_avg_14 = _BASELINE_ESTIMATE
    roll_avg_30 = _BASELINE_ESTIMATE
    roll_std_7 = 0.0
    roll_high_7 = _BASELINE_ESTIMATE
    roll_low_7 = _BASELINE_ESTIMATE
    momentum_ratio = 1.0

    avg_units_by_product = _BASELINE_ESTIMATE
    std_units_by_product = 0.0
    avg_units_by_store = _BASELINE_ESTIMATE
    avg_units_by_category = _BASELINE_ESTIMATE
    avg_units_by_region = _BASELINE_ESTIMATE
    lag1_to_product_ratio = 1.0

    encoded_product_code = _encode_label(encoder_map, "product_id", product_code.upper())
    encoded_category_code = 0
    encoded_store_code = _encode_label(encoder_map, "store_id", outlet_id.upper())
    encoded_region_code = _encode_label(encoder_map, "region", sales_region.title())

    feature_row = pandas.DataFrame(
        [
            {
                "sale_day": sale_day,
                "sale_month": sale_month,
                "sale_year": sale_year,
                "weekday_num": weekday_num,
                "is_weekend_flag": is_weekend_flag,
                "sale_quarter": sale_quarter,
                "iso_week": iso_week,
                "early_month_flag": early_month_flag,
                "late_month_flag": late_month_flag,
                "net_price": net_price,
                "squared_discount": squared_discount,
                "price_x_discount": price_x_discount,
                "price_ratio_to_category": price_ratio_to_category,
                "units_lag_1": units_lag_1,
                "units_lag_7": units_lag_7,
                "units_lag_14": units_lag_14,
                "units_lag_30": units_lag_30,
                "roll_avg_7": roll_avg_7,
                "roll_avg_14": roll_avg_14,
                "roll_avg_30": roll_avg_30,
                "roll_std_7": roll_std_7,
                "roll_high_7": roll_high_7,
                "roll_low_7": roll_low_7,
                "momentum_ratio": momentum_ratio,
                "avg_units_by_product": avg_units_by_product,
                "std_units_by_product": std_units_by_product,
                "avg_units_by_store": avg_units_by_store,
                "avg_units_by_category": avg_units_by_category,
                "avg_units_by_region": avg_units_by_region,
                "lag1_to_product_ratio": lag1_to_product_ratio,
                "product_code": encoded_product_code,
                "category_code": encoded_category_code,
                "store_code": encoded_store_code,
                "region_code": encoded_region_code,
            }
        ]
    )

    predicted_units = float(demand_model.predict(feature_row)[0])
    logger.info("Forecast for %s on %s: %.2f units", product_code, target_date, predicted_units)

    return {
        "product_id": product_code.upper(),
        "predicted_units_sold": round(predicted_units, 2),
        "model_used": "RandomForestRegressor",
    }
