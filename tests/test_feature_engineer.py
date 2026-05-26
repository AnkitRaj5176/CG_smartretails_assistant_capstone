"""Unit tests for feature_engineer.py"""
import pandas as pd
from server.forecasting.feature_engineer import engineer_sales_features


def test_engineer_returns_correct_types(cleaned_df):
    fm, target, names, enc_map, enriched = engineer_sales_features(cleaned_df)
    assert hasattr(fm, "shape")
    assert len(target) == len(fm)
    assert isinstance(names, list)
    assert isinstance(enc_map, dict)
    assert isinstance(enriched, pd.DataFrame)


def test_engineer_feature_count(cleaned_df):
    fm, _, names, _, _ = engineer_sales_features(cleaned_df)
    assert len(names) == 34
    assert fm.shape[1] == 34


def test_engineer_no_nulls_in_features(cleaned_df):
    fm, _, _, _, _ = engineer_sales_features(cleaned_df)
    assert fm.isnull().sum().sum() == 0


def test_engineer_encoder_map_keys(cleaned_df):
    _, _, _, enc_map, _ = engineer_sales_features(cleaned_df)
    for key in ["product_id", "category", "store_id", "region"]:
        assert key in enc_map


def test_engineer_target_non_negative(cleaned_df):
    _, target, _, _, _ = engineer_sales_features(cleaned_df)
    assert (target >= 0).all()
