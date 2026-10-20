
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "distilbert_fake_news"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
BASELINE_PATH = ROOT / "models" / "tfidf_logistic_regression.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 64
BATCH_SIZE = 16
LABEL_NAMES = ["Fake", "Real"]


class HeadlineEvaluationDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.headlines = frame["headline"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.headlines[index],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index], dtype=torch.long
        )
        return item


def calculate_metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(
            precision_score(
                y_true, predictions, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, predictions, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true, predictions, average="macro", zero_division=0
            )
        ),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    test_frame = pd.read_csv(DATA_DIR / "test.csv")
    y_true = test_frame["label"].astype(int).to_numpy()

    print(f"Device: {DEVICE}")
    print(f"Test rows: {len(test_frame):,}")
    print("Evaluating the baseline and selected transformer...")

    # Baseline: the saved pipeline includes TF-IDF and logistic regression.
    baseline = joblib.load(BASELINE_PATH)
    baseline_predictions = baseline.predict(
        test_frame["headline"].astype(str).tolist()
    )
    baseline_metrics = calculate_metrics(
        y_true, baseline_predictions
    )

    # Transformer: inference only; no training or weight updates.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )
    model.to(DEVICE)
    model.eval()

    dataset = TestHeadlineDataset(test_frame, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    transformer_predictions = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=1)
            transformer_predictions.extend(
                predictions.cpu().tolist()
            )

    transformer_metrics = calculate_metrics(
        y_true, transformer_predictions
    )

    results = {
        "test_rows": len(test_frame),
        "warning": (
            "Final test evaluation. Do not use these results "
            "for further model selection or hyperparameter tuning."
        ),
        "tfidf_logistic_regression": baseline_metrics,
        "distilbert": transformer_metrics,
    }

    output_path = REPORT_DIR / "final_test_metrics.json"
    output_path.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    for model_name, predictions in [
        ("tfidf_baseline", baseline_predictions),
        ("distilbert", transformer_predictions),
    ]:
        matrix = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        )
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=LABEL_NAMES,
        )
        display.plot(cmap="viridis", values_format="d")
        plt.title(f"{model_name} — Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / f"{model_name}_test_confusion_matrix.png",
            dpi=160,
        )
        plt.close()

    print("\nFINAL TEST RESULTS")
    print("=" * 55)

    for name in ("tfidf_logistic_regression", "distilbert"):
        metrics = results[name]
        print(f"\n{name}")
        for metric in (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ):
            print(f"  {metric}: {metrics[metric]:.4f}")

        print("\nClass-level report:")
        print(
            pd.DataFrame(metrics["classification_report"])
            .T.round(4).to_string()
        )

    print(f"\nSaved final metrics: {output_path}")
    print("Saved test confusion matrices in reports/figures/")


if __name__ == "__main__":
    main()
