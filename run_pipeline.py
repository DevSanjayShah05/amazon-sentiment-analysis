#!/usr/bin/env python3
"""
End-to-End Sentiment Analysis Pipeline: Amazon Reviews Classification.
Executes VADER rule-based scoring and Hugging Face RoBERTa transformer inference,
benchmarks performance, analyzes edge cases/sarcasm, and generates visualizations.
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, ARTIFACTS_DIR
from src.data_loader import load_reviews_data
from src.vader_model import VaderAnalyzer
from src.roberta_model import RobertaAnalyzer
from src.evaluate import (
    calculate_metrics,
    plot_vader_vs_stars,
    plot_roberta_vs_stars,
    plot_model_comparison,
    plot_confusion_matrices,
)


def main():
    print("=" * 70)
    print(" AMAZON REVIEWS SENTIMENT ANALYSIS: VADER vs. RoBERTa")
    print("=" * 70)

    # 1. Ingest Data
    print("\n[Step 1/5] Loading Amazon Customer Reviews dataset...")
    df = load_reviews_data(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} customer reviews across 1-5 star ratings.")
    print("Star rating breakdown:")
    print(df["Score"].value_counts().sort_index().to_string())

    # 2. Run VADER Analyzer
    print("\n[Step 2/5] Running NLTK VADER sentiment analyzer...")
    vader = VaderAnalyzer()
    df_vader = vader.analyze_dataframe(df, text_column="Text")
    print("VADER scoring completed.")

    # 3. Run RoBERTa Deep Learning Model
    print("\n[Step 3/5] Running Hugging Face RoBERTa transformer inference...")
    roberta = RobertaAnalyzer()
    df_results = roberta.analyze_dataframe(df_vader, text_column="Text", batch_size=32)
    print("RoBERTa inference completed.")

    # Save processed dataframe
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Enriched sentiment dataset saved to: {PROCESSED_DATA_PATH}")

    # 4. Comparative Metrics & Failure Case Studies
    print("\n[Step 4/5] Evaluating performance against Ground Truth sentiment...")
    y_true = df_results["GroundTruthSentiment"]
    vader_metrics = calculate_metrics(y_true, df_results["vader_pred"])
    roberta_metrics = calculate_metrics(y_true, df_results["roberta_pred"])

    comparison_records = []
    for metric_name in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]:
        comparison_records.append({
            "Metric": metric_name,
            "VADER": round(vader_metrics[metric_name], 4),
            "RoBERTa": round(roberta_metrics[metric_name], 4),
            "Improvement": f"{(roberta_metrics[metric_name] - vader_metrics[metric_name]) * 100:+.2f}%"
        })

    comp_df = pd.DataFrame(comparison_records)
    print("\n" + "=" * 60)
    print(" HEAD-TO-HEAD SENTIMENT CLASSIFICATION SCORECARD")
    print("=" * 60)
    print(comp_df.to_string(index=False))
    print("=" * 60)

    # Highlight Sarcasm & Negation Failure Modes
    print("\n--- Failure Case Studies: Where VADER Fails & RoBERTa Succeeds ---")
    sarcasm_cases = df_results[
        (df_results["Score"] == 1) &
        (df_results["vader_pred"] == "positive") &
        (df_results["roberta_pred"] == "negative")
    ]
    if not sarcasm_cases.empty:
        for idx, row in sarcasm_cases.head(3).iterrows():
            print(f"\nReview [{row['Score']} Stars]: \"{row['Text']}\"")
            print(f" -> VADER Prediction:   {row['vader_pred'].upper()} (Compound: {row['vader_compound']:.2f}) [FAILED - misled by lexical cues]")
            print(f" -> RoBERTa Prediction: {row['roberta_pred'].upper()} (Negative Prob: {row['roberta_neg']:.2%}) [CORRECT - captured context]")

    # 5. Visualizations
    print("\n[Step 5/5] Generating publication-grade visualizations...")
    vader_plot = plot_vader_vs_stars(df_results, ARTIFACTS_DIR / "vader_vs_star_ratings.png")
    roberta_plot = plot_roberta_vs_stars(df_results, ARTIFACTS_DIR / "roberta_vs_star_ratings.png")
    comp_plot = plot_model_comparison(comp_df[["Metric", "VADER", "RoBERTa"]], ARTIFACTS_DIR / "model_head_to_head_comparison.png")
    cm_plot = plot_confusion_matrices(y_true, df_results["vader_pred"], df_results["roberta_pred"], ARTIFACTS_DIR / "confusion_matrices.png")

    print(f"Saved VADER rating plot to:           {vader_plot}")
    print(f"Saved RoBERTa probability plot to:     {roberta_plot}")
    print(f"Saved head-to-head comparison plot to: {comp_plot}")
    print(f"Saved confusion matrices to:          {cm_plot}")

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" Artifacts directory: {ARTIFACTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
