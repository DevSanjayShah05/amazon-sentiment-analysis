"""
Tests for VADER analyzer in Project 2.
"""
from src.vader_model import VaderAnalyzer


def test_vader_positive_text():
    analyzer = VaderAnalyzer()
    res = analyzer.score_text("This product is absolutely amazing and wonderful!")
    assert res["compound"] > 0.05
    assert analyzer.predict_sentiment(res["compound"]) == "positive"


def test_vader_negative_text():
    analyzer = VaderAnalyzer()
    res = analyzer.score_text("Terrible quality. Complete waste of money and broke on day one.")
    assert res["compound"] < -0.05
    assert analyzer.predict_sentiment(res["compound"]) == "negative"
