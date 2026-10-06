"""
Configuration module for Amazon Reviews Sentiment Analysis project.
"""
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "amazon_reviews.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "reviews_with_sentiment.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

# Model constants
ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
ROBERTA_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

# VADER thresholds
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# Ground truth mapping from Star rating (1-5) to sentiment
def rating_to_sentiment(score: int) -> str:
    """Map 1-5 star score to categorical sentiment."""
    if score in [1, 2]:
        return "negative"
    elif score == 3:
        return "neutral"
    else:  # 4, 5
        return "positive"

RANDOM_STATE = 42
BATCH_SIZE = 32
