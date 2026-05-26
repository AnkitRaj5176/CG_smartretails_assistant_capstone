"""Unit tests for csv_reader.py"""
import pytest
import pandas as pd
from server.forecasting.csv_reader import check_mandatory_columns, MANDATORY_COLUMNS


def test_check_mandatory_columns_passes(sample_raw_df):
    check_mandatory_columns(sample_raw_df)  # must not raise


def test_check_mandatory_columns_raises_on_missing():
    incomplete_df = pd.DataFrame({"product_id": ["P1"], "category": ["Electronics"]})
    with pytest.raises(ValueError, match="Missing mandatory columns"):
        check_mandatory_columns(incomplete_df)


def test_mandatory_columns_list_not_empty():
    assert "product_id" in MANDATORY_COLUMNS
    assert "units_sold" in MANDATORY_COLUMNS
    assert "revenue" in MANDATORY_COLUMNS
    assert len(MANDATORY_COLUMNS) >= 9
