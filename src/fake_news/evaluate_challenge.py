
import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "challenge_headlines.csv"
BASELINE_PATH = ROOT / "models" / "tfidf_logistic_regression.joblib"
TRANSFORMER_PATH = ROOT / "models" / "distilbert_fake_news"
OUTPUT_PATH = ROOT / "reports" / "challenge_evaluation.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    df = pd.read_csv(DATA_PATH)

    required = {"headline", "true_label", "evidence_url", "notes"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    df["headline"] = df["headline"].fillna("").astype(str).str.strip()
    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")

    # Only evaluate examples with verified binary labels and evidence.
    verified = df[
        df["headline"].ne("")
        & df["true_label"].isin([0, 1])
        & df["evidence_url"].fillna("").astype(str).str.strip().ne("")
    ].copy()

    if verified.empty:
        print("No verified challenge examples yet.")
        print("Add true_label (0=fake, 1=real) and an evidence_url.")
        print(f"Template rows available: {len(df)}")
        return

    y_true = verified["true_label"].astype(int).tolist()
    headlines = verified["headline"].tolist()

    baseline = joblib.load(BASELINE_PATH)
    baseline_predictions = baseline.predict(headlines).tolist()

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_PATH
    ).to(DEVICE)
    model.eval()

    transformer_predictions = []
    transformer_probabilities = []

    with torch.no_grad():
        for headline in headlines:
            inputs = tokenizer(
                headline,
                max_length=64,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

            transformer_predictions.append(
                int(max(range(len(probabilities)), key=probabilities.__getitem__))
            )
            transformer_probabilities.append(probabilities)

    def metrics(y, predictions):
        return {
            "accuracy": accuracy_score(y, predictions),
            "precision_macro": precision_score(
                y, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y, predictions, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y, predictions, average="macro", zero_division=0
            ),
            "classification_report": classification_report(
                y, predictions, labels=[0, 1],
                target_names=["Fake", "Real"],
                output_dict=True, zero_division=0
            ),
        }

    output = {
        "verified_example_count": len(verified),
        "label_definition": {"0": "Fake", "1": "Real"},
        "tfidf_logistic_regression": metrics(
            y_true, baseline_predictions
        ),
        "distilbert": metrics(y_true, transformer_predictions),
        "individual_predictions": [],
    }

    for index, (_, row) in enumerate(verified.iterrows()):
        output["individual_predictions"].append({
            "headline": row["headline"],
            "true_label": int(row["true_label"]),
            "evidence_url": row["evidence_url"],
            "notes": row["notes"],
            "tfidf_prediction": int(baseline_predictions[index]),
            "distilbert_prediction": int(transformer_predictions[index]),
            "distilbert_fake_probability": transformer_probabilities[index][0],
            "distilbert_real_probability": transformer_probabilities[index][1],
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, default=float),
        encoding="utf-8",
    )

    print(f"Evaluated verified examples: {len(verified)}")
    print(f"Saved results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
