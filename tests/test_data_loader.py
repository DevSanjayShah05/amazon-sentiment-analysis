"""
Tests for data loader module in Project 2.
"""
import pandas as pd
from src.data_loader import load_reviews_data


def test_load_reviews_data():
    df = load_reviews_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 100
    expected_cols = ["Id", "ProductId", "Score", "Summary", "Text", "GroundTruthSentiment"]
    for col in expected_cols:
        assert col in df.columns
    assert set(df["Score"].unique()).issubset({1, 2, 3, 4, 5})
