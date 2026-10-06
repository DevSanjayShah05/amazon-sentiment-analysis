"""
Hugging Face RoBERTa Deep Learning Transformer Sentiment Analyzer.
Pre-trained contextual model capable of understanding sarcasm, syntax, and negations.
"""
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from src.config import ROBERTA_MODEL_NAME, ROBERTA_LABELS, BATCH_SIZE


class RobertaAnalyzer:
    """Wrapper around CardiffNLP Twitter-RoBERTa sentiment classification model."""

    def __init__(self, model_name: str = ROBERTA_MODEL_NAME, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Initializing RoBERTa ({model_name}) on device: {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("RoBERTa model loaded successfully.")

    def score_text(self, text: str) -> Dict[str, float]:
        """Run inference on a single text string and return softmax probabilities."""
        if not isinstance(text, str) or not text.strip():
            return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}

        # Truncate to maximum RoBERTa context window (512 tokens)
        encoded_input = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded_input)
            scores = output[0][0].detach().cpu()
            scores = F.softmax(scores, dim=0).numpy()

        return {
            "negative": float(scores[0]),
            "neutral": float(scores[1]),
            "positive": float(scores[2]),
        }

    def predict_sentiment(self, scores: Dict[str, float]) -> str:
        """Return the class with highest softmax probability."""
        return max(scores, key=scores.get)

    def analyze_dataframe(
        self, df: pd.DataFrame, text_column: str = "Text", batch_size: int = BATCH_SIZE
    ) -> pd.DataFrame:
        """Run batched RoBERTa inference across an entire DataFrame."""
        result_df = df.copy()
        texts = df[text_column].fillna("").tolist()

        neg_scores = []
        neu_scores = []
        pos_scores = []
        predictions = []

        print(f"Executing RoBERTa Transformer Inference across {len(texts):,} reviews...")
        for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa Batches"):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()

            for p in probs:
                neg_scores.append(float(p[0]))
                neu_scores.append(float(p[1]))
                pos_scores.append(float(p[2]))
                pred_class = ROBERTA_LABELS[int(p.argmax())]
                predictions.append(pred_class)

        result_df["roberta_neg"] = neg_scores
        result_df["roberta_neu"] = neu_scores
        result_df["roberta_pos"] = pos_scores
        result_df["roberta_pred"] = predictions

        return result_df
