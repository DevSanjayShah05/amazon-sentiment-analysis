"""
Evaluation and visualization module for sentiment analysis.
Calculates classification metrics and generates diagnostic charts.
"""
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.config import ARTIFACTS_DIR


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculate accuracy and macro precision, recall, and F1."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "Accuracy": float(acc),
        "Macro Precision": float(prec),
        "Macro Recall": float(rec),
        "Macro F1": float(f1),
    }


def plot_vader_vs_stars(df: pd.DataFrame, output_path: Path = None) -> Path:
    """Plot VADER compound score across Amazon 1-5 Star Ratings."""
    if output_path is None:
        output_path = ARTIFACTS_DIR / "vader_vs_star_ratings.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("VADER Lexicon Sentiment vs. Amazon Customer Star Ratings", fontsize=15, fontweight="bold")

    # 1. Barplot of Mean Compound Score
    vader_stats = df.groupby("Score")["vader_compound"].agg(["mean", "std"]).reset_index()
    sns.barplot(data=vader_stats, x="Score", y="mean", ax=axes[0], palette="coolwarm", hue="Score", legend=False)
    axes[0].set_title("Average VADER Compound Score by Star Rating")
    axes[0].set_xlabel("Review Star Rating (1-5)")
    axes[0].set_ylabel("Mean Compound Score (-1.0 to +1.0)")
    axes[0].set_ylim(-0.6, 0.9)
    axes[0].axhline(0, color="grey", linestyle="--", alpha=0.7)

    # 2. Boxplot of distribution
    sns.boxplot(data=df, x="Score", y="vader_compound", ax=axes[1], palette="Blues", hue="Score", legend=False)
    axes[1].set_title("VADER Compound Score Distribution Across Ratings")
    axes[1].set_xlabel("Review Star Rating (1-5)")
    axes[1].set_ylabel("Compound Score")
    axes[1].axhline(0, color="grey", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_roberta_vs_stars(df: pd.DataFrame, output_path: Path = None) -> Path:
    """Plot RoBERTa probability confidence curves across Star Ratings."""
    if output_path is None:
        output_path = ARTIFACTS_DIR / "roberta_vs_star_ratings.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Hugging Face RoBERTa Sentiment Probabilities vs. Star Ratings", fontsize=15, fontweight="bold")

    # 1. Positive probability by stars
    sns.barplot(data=df, x="Score", y="roberta_pos", ax=axes[0], color="#2ca02c", errorbar=None)
    axes[0].set_title("Mean RoBERTa Positive Probability")
    axes[0].set_xlabel("Star Rating")
    axes[0].set_ylabel("Probability [0, 1]")
    axes[0].set_ylim(0, 1.05)

    # 2. Neutral probability by stars
    sns.barplot(data=df, x="Score", y="roberta_neu", ax=axes[1], color="#1f77b4", errorbar=None)
    axes[1].set_title("Mean RoBERTa Neutral Probability")
    axes[1].set_xlabel("Star Rating")
    axes[1].set_ylabel("Probability [0, 1]")
    axes[1].set_ylim(0, 1.05)

    # 3. Negative probability by stars
    sns.barplot(data=df, x="Score", y="roberta_neg", ax=axes[2], color="#d62728", errorbar=None)
    axes[2].set_title("Mean RoBERTa Negative Probability")
    axes[2].set_xlabel("Star Rating")
    axes[2].set_ylabel("Probability [0, 1]")
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Path = None) -> Path:
    """Plot head-to-head bar comparison between VADER and RoBERTa."""
    if output_path is None:
        output_path = ARTIFACTS_DIR / "model_head_to_head_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    melted = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette=["#ff7f0e", "#1f77b4"])
    plt.title("Head-to-Head Performance: NLTK VADER vs. Hugging Face RoBERTa", fontsize=14, fontweight="bold", pad=12)
    plt.ylim(0, 1.1)
    plt.ylabel("Score [0.0 - 1.0]")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_confusion_matrices(
    y_true: pd.Series, y_vader: pd.Series, y_roberta: pd.Series, output_path: Path = None
) -> Path:
    """Plot side-by-side confusion matrices for VADER and RoBERTa."""
    if output_path is None:
        output_path = ARTIFACTS_DIR / "confusion_matrices.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["negative", "neutral", "positive"]
    cm_vader = confusion_matrix(y_true, y_vader, labels=labels)
    cm_roberta = confusion_matrix(y_true, y_roberta, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Sentiment Classification Confusion Matrices", fontsize=15, fontweight="bold")

    sns.heatmap(
        cm_vader,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("NLTK VADER Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("Ground Truth Sentiment")

    sns.heatmap(
        cm_roberta,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        cbar=False,
    )
    axes[1].set_title("Hugging Face RoBERTa Confusion Matrix", fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("Ground Truth Sentiment")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
