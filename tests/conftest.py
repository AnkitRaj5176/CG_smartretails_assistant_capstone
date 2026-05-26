"""Shared pytest fixtures for the retail engine test suite."""

import os
import sys
import pytest
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


@pytest.fixture(scope="session")
def sample_raw_df() -> pd.DataFrame:
    """Minimal raw retail DataFrame that passes all mandatory column checks."""
    return pd.DataFrame({
        "product_id": ["PROD_001", "PROD_002", "PROD_003", "PROD_001", "PROD_002"],
        "category":   ["Electronics", "Clothing", "Groceries", "Electronics", "Clothing"],
        "region":     ["North", "South", "East", "West", "Central"],
        "store_id":   ["STORE_A", "STORE_B", "STORE_C", "STORE_A", "STORE_B"],
        "date":       ["2024-01-10", "2024-02-15", "2024-03-20", "2024-04-05", "2024-05-12"],
        "price":      [299.99, 49.99, 12.50, 299.99, 49.99],
        "discount":   [10.0, 5.0, 0.0, 20.0, 15.0],
        "units_sold": [5, 12, 30, 8, 20],
        "revenue":    [1349.96, 569.89, 375.00, 1919.94, 849.83],
    })


@pytest.fixture(scope="session")
def cleaned_df(sample_raw_df) -> pd.DataFrame:
    from server.forecasting.record_cleaner import clean_retail_records
    return clean_retail_records(sample_raw_df)
