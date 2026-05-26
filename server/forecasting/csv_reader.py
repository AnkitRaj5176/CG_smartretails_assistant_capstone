import logging

import pandas

logger = logging.getLogger(__name__)

MANDATORY_COLUMNS: list[str] = [
    "product_id",
    "category",
    "region",
    "store_id",
    "date",
    "price",
    "discount",
    "units_sold",
    "revenue",
]

VALID_CATEGORIES: set[str] = {
    "Electronics",
    "Clothing",
    "Groceries",
    "Furniture",
    "Sports",
    "Beauty",
    "Toys",
    "Books",
    "Automotive",
    "Health",
}

VALID_REGIONS: set[str] = {
    "North",
    "South",
    "East",
    "West",
    "Central",
    "Northeast",
    "Northwest",
    "Southeast",
    "Southwest",
}


def read_retail_csv(file_path: str) -> pandas.DataFrame:
    """Read a retail CSV file and return a DataFrame."""
    try:
        retail_dataframe = pandas.read_csv(file_path)
        logger.info("Read %d rows from %s", len(retail_dataframe), file_path)
        return retail_dataframe
    except FileNotFoundError as file_error:
        logger.warning("CSV file not found: %s", file_error)
        raise
    except pandas.errors.ParserError as parse_error:
        logger.warning("CSV parse error: %s", parse_error)
        raise


def check_mandatory_columns(retail_dataframe: pandas.DataFrame) -> None:
    """Raise ValueError if any mandatory column is missing from the dataframe."""
    missing_columns = [col for col in MANDATORY_COLUMNS if col not in retail_dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing mandatory columns: {missing_columns}")
    logger.info("All mandatory columns present.")
