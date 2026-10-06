"""
NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer.
Rule-based lexicon model designed for general sentiment scoring.
"""
from typing import Dict, Any
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

from src.config import VADER_POS_THRESHOLD, VADER_NEG_THRESHOLD


class VaderAnalyzer:
    """Wrapper around NLTK's SentimentIntensityAnalyzer."""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> Dict[str, float]:
        """Compute VADER polarity scores (neg, neu, pos, compound) for a text string."""
        if not isinstance(text, str) or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self.sia.polarity_scores(text)

    def predict_sentiment(self, compound_score: float) -> str:
        """Categorize compound score into positive, neutral, or negative sentiment."""
        if compound_score >= VADER_POS_THRESHOLD:
            return "positive"
        elif compound_score <= VADER_NEG_THRESHOLD:
            return "negative"
        else:
            return "neutral"

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "Text") -> pd.DataFrame:
        """Run VADER sentiment scoring across an entire DataFrame."""
        result_df = df.copy()
        
        scores_list = []
        preds_list = []

        print(f"Executing VADER Sentiment Analysis across {len(df):,} reviews...")
        for text in tqdm(df[text_column], desc="VADER Scoring"):
            scores = self.score_text(text)
            pred = self.predict_sentiment(scores["compound"])
            scores_list.append(scores)
            preds_list.append(pred)

        scores_df = pd.DataFrame(scores_list)
        result_df["vader_neg"] = scores_df["neg"]
        result_df["vader_neu"] = scores_df["neu"]
        result_df["vader_pos"] = scores_df["pos"]
        result_df["vader_compound"] = scores_df["compound"]
        result_df["vader_pred"] = preds_list

        return result_df
