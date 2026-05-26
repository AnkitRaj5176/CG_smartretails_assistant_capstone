import logging

import numpy
import pandas
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def engineer_sales_features(
    input_dataframe: DataFrame,
) -> tuple[DataFrame, Series, list[str], dict, DataFrame]:
    """Engineer all features for demand forecasting and return feature matrix, target, names, encoders, enriched df."""
    enriched_dataframe = input_dataframe.copy()
    enriched_dataframe = enriched_dataframe.sort_values(["product_id", "date"]).reset_index(drop=True)

    enriched_dataframe["sale_day"] = enriched_dataframe["date"].dt.day
    enriched_dataframe["sale_month"] = enriched_dataframe["date"].dt.month
    enriched_dataframe["sale_year"] = enriched_dataframe["date"].dt.year
    enriched_dataframe["weekday_num"] = enriched_dataframe["date"].dt.weekday
    enriched_dataframe["is_weekend_flag"] = (enriched_dataframe["weekday_num"] >= 5).astype(int)
    enriched_dataframe["sale_quarter"] = enriched_dataframe["date"].dt.quarter
    enriched_dataframe["iso_week"] = enriched_dataframe["date"].dt.isocalendar().week.astype(int)
    enriched_dataframe["early_month_flag"] = (enriched_dataframe["sale_day"] <= 10).astype(int)
    enriched_dataframe["late_month_flag"] = (enriched_dataframe["sale_day"] >= 21).astype(int)

    enriched_dataframe["net_price"] = enriched_dataframe["price"] * (1 - enriched_dataframe["discount"] / 100)
    enriched_dataframe["squared_discount"] = enriched_dataframe["discount"] ** 2
    enriched_dataframe["price_x_discount"] = enriched_dataframe["price"] * enriched_dataframe["discount"]

    category_avg_price = enriched_dataframe.groupby("category")["price"].transform("mean")
    enriched_dataframe["price_ratio_to_category"] = enriched_dataframe["price"] / category_avg_price.replace(0, numpy.nan)
    enriched_dataframe["price_ratio_to_category"] = enriched_dataframe["price_ratio_to_category"].fillna(1.0)

    for lag_window in [1, 7, 14, 30]:
        enriched_dataframe[f"units_lag_{lag_window}"] = (
            enriched_dataframe.groupby("product_id")["units_sold"]
            .shift(lag_window)
            .fillna(enriched_dataframe["units_sold"].median())
        )

    for rolling_window in [7, 14, 30]:
        rolling_group = enriched_dataframe.groupby("product_id")["units_sold"]
        enriched_dataframe[f"roll_avg_{rolling_window}"] = (
            rolling_group.transform(lambda series: series.shift(1).rolling(rolling_window, min_periods=1).mean())
            .fillna(enriched_dataframe["units_sold"].median())
        )

    enriched_dataframe["roll_std_7"] = (
        enriched_dataframe.groupby("product_id")["units_sold"]
        .transform(lambda series: series.shift(1).rolling(7, min_periods=1).std())
        .fillna(0.0)
    )
    enriched_dataframe["roll_high_7"] = (
        enriched_dataframe.groupby("product_id")["units_sold"]
        .transform(lambda series: series.shift(1).rolling(7, min_periods=1).max())
        .fillna(enriched_dataframe["units_sold"].median())
    )
    enriched_dataframe["roll_low_7"] = (
        enriched_dataframe.groupby("product_id")["units_sold"]
        .transform(lambda series: series.shift(1).rolling(7, min_periods=1).min())
        .fillna(enriched_dataframe["units_sold"].median())
    )
    enriched_dataframe["momentum_ratio"] = (
        enriched_dataframe["roll_avg_7"] / enriched_dataframe["roll_avg_30"].replace(0, numpy.nan)
    ).fillna(1.0)

    enriched_dataframe["avg_units_by_product"] = enriched_dataframe.groupby("product_id")["units_sold"].transform("mean")
    enriched_dataframe["std_units_by_product"] = enriched_dataframe.groupby("product_id")["units_sold"].transform("std").fillna(0.0)
    enriched_dataframe["avg_units_by_store"] = enriched_dataframe.groupby("store_id")["units_sold"].transform("mean")
    enriched_dataframe["avg_units_by_category"] = enriched_dataframe.groupby("category")["units_sold"].transform("mean")
    enriched_dataframe["avg_units_by_region"] = enriched_dataframe.groupby("region")["units_sold"].transform("mean")
    enriched_dataframe["lag1_to_product_ratio"] = (
        enriched_dataframe["units_lag_1"] / enriched_dataframe["avg_units_by_product"].replace(0, numpy.nan)
    ).fillna(1.0)

    encoder_map: dict = {}

    for original_column, encoded_column in [
        ("product_id", "product_code"),
        ("category", "category_code"),
        ("store_id", "store_code"),
        ("region", "region_code"),
    ]:
        unique_values = enriched_dataframe[original_column].unique()
        label_to_code = {label: index for index, label in enumerate(sorted(unique_values))}
        enriched_dataframe[encoded_column] = enriched_dataframe[original_column].map(label_to_code)
        encoder_map[original_column] = label_to_code

    feature_name_list: list[str] = [
        "sale_day", "sale_month", "sale_year", "weekday_num", "is_weekend_flag",
        "sale_quarter", "iso_week", "early_month_flag", "late_month_flag",
        "net_price", "squared_discount", "price_x_discount", "price_ratio_to_category",
        "units_lag_1", "units_lag_7", "units_lag_14", "units_lag_30",
        "roll_avg_7", "roll_avg_14", "roll_avg_30", "roll_std_7", "roll_high_7", "roll_low_7", "momentum_ratio",
        "avg_units_by_product", "std_units_by_product", "avg_units_by_store",
        "avg_units_by_category", "avg_units_by_region", "lag1_to_product_ratio",
        "product_code", "category_code", "store_code", "region_code",
    ]

    feature_matrix = enriched_dataframe[feature_name_list].copy()
    target_column = enriched_dataframe["units_sold"]

    logger.info("Feature engineering complete — %d features, %d rows.", len(feature_name_list), len(feature_matrix))
    return feature_matrix, target_column, feature_name_list, encoder_map, enriched_dataframe
