
"""Analyze held-out test errors without retraining either model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
BASELINE_PATH = ROOT / "models" / "tfidf_logistic_regression.joblib"
TRANSFORMER_PATH = ROOT / "models" / "distilbert_fake_news"
REPORT_DIR = ROOT / "reports"
ERRORS_PATH = REPORT_DIR / "test_error_analysis.csv"
SUMMARY_PATH = REPORT_DIR / "error_analysis_summary.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 64
BATCH_SIZE = 16
LABEL_NAMES = {0: "Fake", 1: "Real"}


class HeadlineDataset(Dataset):
    def __init__(self, headlines, tokenizer):
        self.headlines = headlines
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, index):
        return self.tokenizer(
            self.headlines[index],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(TEST_PATH)
    headlines = frame["headline"].fillna("").astype(str).tolist()
    y_true = frame["label"].astype(int).tolist()

    # TF-IDF baseline inference.
    baseline = joblib.load(BASELINE_PATH)
    baseline_predictions = baseline.predict(headlines).astype(int)

    # DistilBERT inference. No training or weight updates.
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_PATH
    ).to(DEVICE)
    model.eval()

    dataset = HeadlineDataset(headlines, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    transformer_predictions = []
    fake_probabilities = []
    real_probabilities = []

    with torch.no_grad():
        for batch in loader:
            # DataLoader collates the tokenizer output dictionaries.
            inputs = {
                key: value.squeeze(1).to(DEVICE)
                for key, value in batch.items()
            }
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1).cpu()

            transformer_predictions.extend(
                probabilities.argmax(dim=1).tolist()
            )
            fake_probabilities.extend(probabilities[:, 0].tolist())
            real_probabilities.extend(probabilities[:, 1].tolist())

    results = pd.DataFrame({
        "headline": headlines,
        "true_label_id": y_true,
        "true_label": [LABEL_NAMES[label] for label in y_true],
        "tfidf_prediction_id": baseline_predictions,
        "tfidf_prediction": [
            LABEL_NAMES[int(label)] for label in baseline_predictions
        ],
        "distilbert_prediction_id": transformer_predictions,
        "distilbert_prediction": [
            LABEL_NAMES[int(label)] for label in transformer_predictions
        ],
        "distilbert_fake_probability": fake_probabilities,
        "distilbert_real_probability": real_probabilities,
    })

    results["tfidf_correct"] = (
        results["true_label_id"] == results["tfidf_prediction_id"]
    )
    results["distilbert_correct"] = (
        results["true_label_id"] == results["distilbert_prediction_id"]
    )
    results["both_models_correct"] = (
        results["tfidf_correct"] & results["distilbert_correct"]
    )
    results["both_models_wrong"] = (
        ~results["tfidf_correct"] & ~results["distilbert_correct"]
    )

    results.to_csv(ERRORS_PATH, index=False, encoding="utf-8-sig")

    summary = {
        "test_rows": len(results),
        "warning": (
            "Descriptive analysis only. Do not use the final test set "
            "for model selection or hyperparameter tuning."
        ),
        "tfidf_errors": int((~results["tfidf_correct"]).sum()),
        "distilbert_errors": int((~results["distilbert_correct"]).sum()),
        "both_models_correct": int(results["both_models_correct"].sum()),
        "both_models_wrong": int(results["both_models_wrong"].sum()),
        "tfidf_only_wrong": int(
            (results["tfidf_correct"] == False)
            .where(results["distilbert_correct"], False).sum()
        ),
        "distilbert_only_wrong": int(
            (results["distilbert_correct"] == False)
            .where(results["tfidf_correct"], False).sum()
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("ERROR ANALYSIS COMPLETE")
    print(f"Device: {DEVICE}")
    print(f"Test headlines: {len(results):,}")
    print(f"TF-IDF errors: {summary['tfidf_errors']:,}")
    print(f"DistilBERT errors: {summary['distilbert_errors']:,}")
    print(f"Both models wrong: {summary['both_models_wrong']:,}")
    print(f"Misclassified rows: {ERRORS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
