"""
Tests for RoBERTa analyzer in Project 2.
"""
from src.roberta_model import RobertaAnalyzer


def test_roberta_inference():
    analyzer = RobertaAnalyzer()
    res_pos = analyzer.score_text("Best snack I have ever had. Absolutely delicious!")
    assert res_pos["positive"] > res_pos["negative"]
    assert analyzer.predict_sentiment(res_pos) == "positive"

    res_neg = analyzer.score_text("Arrived broken, smells awful, worst customer experience ever.")
    assert res_neg["negative"] > res_neg["positive"]
    assert analyzer.predict_sentiment(res_neg) == "negative"
