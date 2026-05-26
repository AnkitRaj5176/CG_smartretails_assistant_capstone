import logging

import numpy
import pandas

from server.forecasting.csv_reader import VALID_CATEGORIES, VALID_REGIONS

logger = logging.getLogger(__name__)


def clean_retail_records(input_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """Clean and normalize the retail records dataframe."""
    retail_dataframe = input_dataframe.copy()

    retail_dataframe = retail_dataframe.drop_duplicates()
    logger.info("After drop_duplicates: %d rows", len(retail_dataframe))

    retail_dataframe["date"] = pandas.to_datetime(retail_dataframe["date"], errors="coerce")
    retail_dataframe = retail_dataframe.dropna(subset=["date"])

    for numeric_column in ["price", "discount", "units_sold", "revenue"]:
        retail_dataframe[numeric_column] = pandas.to_numeric(retail_dataframe[numeric_column], errors="coerce")

    price_median = retail_dataframe["price"].median()
    units_median = retail_dataframe["units_sold"].median()

    retail_dataframe["price"] = retail_dataframe["price"].fillna(price_median)
    retail_dataframe["discount"] = retail_dataframe["discount"].fillna(0.0)
    retail_dataframe["units_sold"] = retail_dataframe["units_sold"].fillna(units_median)

    retail_dataframe = retail_dataframe[retail_dataframe["price"] > 0]
    retail_dataframe = retail_dataframe[retail_dataframe["units_sold"] >= 0]
    retail_dataframe = retail_dataframe[retail_dataframe["discount"].between(0, 100)]

    retail_dataframe["revenue"] = retail_dataframe["price"] * retail_dataframe["units_sold"] * (
        1 - retail_dataframe["discount"] / 100
    )

    revenue_mismatch_mask = ~numpy.isclose(
        retail_dataframe["revenue"],
        retail_dataframe["price"] * retail_dataframe["units_sold"] * (1 - retail_dataframe["discount"] / 100),
        rtol=0.01,
    )
    retail_dataframe.loc[revenue_mismatch_mask, "revenue"] = (
        retail_dataframe.loc[revenue_mismatch_mask, "price"]
        * retail_dataframe.loc[revenue_mismatch_mask, "units_sold"]
        * (1 - retail_dataframe.loc[revenue_mismatch_mask, "discount"] / 100)
    )

    retail_dataframe["category"] = retail_dataframe["category"].astype(str).str.strip().str.title()
    retail_dataframe["region"] = retail_dataframe["region"].astype(str).str.strip().str.title()
    retail_dataframe["store_id"] = retail_dataframe["store_id"].astype(str).str.strip().str.upper()
    retail_dataframe["product_id"] = retail_dataframe["product_id"].astype(str).str.strip().str.upper()

    retail_dataframe = retail_dataframe[retail_dataframe["category"].isin(VALID_CATEGORIES)]
    retail_dataframe = retail_dataframe[retail_dataframe["region"].isin(VALID_REGIONS)]

    retail_dataframe = retail_dataframe.reset_index(drop=True)
    logger.info("Cleaned dataframe has %d rows.", len(retail_dataframe))
    return retail_dataframe
