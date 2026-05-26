"""Unit tests for record_cleaner.py"""
import pandas as pd
import pytest
from server.forecasting.record_cleaner import clean_retail_records


def test_clean_returns_dataframe(sample_raw_df):
    assert isinstance(clean_retail_records(sample_raw_df), pd.DataFrame)


def test_clean_removes_duplicates():
    df = pd.DataFrame({
        "product_id": ["PROD_001", "PROD_001"],
        "category":   ["Electronics", "Electronics"],
        "region":     ["North", "North"],
        "store_id":   ["STORE_A", "STORE_A"],
        "date":       ["2024-01-10", "2024-01-10"],
        "price":      [299.99, 299.99],
        "discount":   [10.0, 10.0],
        "units_sold": [5, 5],
        "revenue":    [1349.96, 1349.96],
    })
    assert len(clean_retail_records(df)) == 1


def test_clean_drops_invalid_dates():
    df = pd.DataFrame({
        "product_id": ["PROD_001"], "category": ["Electronics"],
        "region": ["North"], "store_id": ["STORE_A"],
        "date": ["not-a-date"], "price": [100.0],
        "discount": [0.0], "units_sold": [5], "revenue": [500.0],
    })
    assert len(clean_retail_records(df)) == 0


def test_clean_filters_invalid_category():
    df = pd.DataFrame({
        "product_id": ["PROD_001"], "category": ["InvalidCat"],
        "region": ["North"], "store_id": ["STORE_A"],
        "date": ["2024-01-10"], "price": [100.0],
        "discount": [0.0], "units_sold": [5], "revenue": [500.0],
    })
    assert len(clean_retail_records(df)) == 0


def test_clean_recalculates_revenue(sample_raw_df):
    result = clean_retail_records(sample_raw_df)
    expected = result["price"] * result["units_sold"] * (1 - result["discount"] / 100)
    pd.testing.assert_series_equal(result["revenue"].round(2), expected.round(2), check_names=False)


def test_clean_product_id_uppercased(sample_raw_df):
    result = clean_retail_records(sample_raw_df)
    assert result["product_id"].str.isupper().all()


def test_clean_no_negative_units():
    df = pd.DataFrame({
        "product_id": ["PROD_001"], "category": ["Electronics"],
        "region": ["North"], "store_id": ["STORE_A"],
        "date": ["2024-01-10"], "price": [100.0],
        "discount": [0.0], "units_sold": [-5], "revenue": [-500.0],
    })
    assert len(clean_retail_records(df)) == 0
