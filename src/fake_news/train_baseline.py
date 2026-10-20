
"""Train and evaluate a TF-IDF + Logistic Regression baseline."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42


def load_split(name):
    """Load and validate one saved dataset split."""
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run split_dataset.py first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{name} split is empty.")

    if df["headline"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{name} contains missing inputs or labels.")

    return df


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    validation_df = load_split("validation")

    X_train = train_df["headline"].astype(str)
    y_train = train_df["label"].astype(int)

    X_validation = validation_df["headline"].astype(str)
    y_validation = validation_df["label"].astype(int)

    # TF-IDF learns vocabulary and IDF weights from training data only.
    # The Pipeline ensures that validation data does not fit the vectorizer.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=100_000,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1_000,
                class_weight=None,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print("=" * 60)
    print("TRAINING TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")

    model.fit(X_train, y_train)

    # Use validation data for model assessment.
    predictions = model.predict(X_validation)

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "random_state": RANDOM_STATE,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "tfidf_parameters": {
            "ngram_range": [1, 2],
            "max_features": 100_000,
            "min_df": 2,
            "sublinear_tf": True,
        },
        "validation_metrics": {
            "accuracy": float(
                accuracy_score(y_validation, predictions)
            ),
            "precision_macro": float(
                precision_score(
                    y_validation, predictions,
                    average="macro", zero_division=0
                )
            ),
            "recall_macro": float(
                recall_score(
                    y_validation, predictions,
                    average="macro", zero_division=0
                )
            ),
            "f1_macro": float(
                f1_score(
                    y_validation, predictions,
                    average="macro", zero_division=0
                )
            ),
            "classification_report": classification_report(
                y_validation,
                predictions,
                labels=[0, 1],
                target_names=["fake", "real"],
                output_dict=True,
                zero_division=0,
            ),
        },
    }

    print("\nValidation metrics:")
    for name, value in metrics["validation_metrics"].items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=[0, 1],
            target_names=["fake", "real"],
            zero_division=0,
        )
    )

    # Save the complete fitted pipeline.
    model_path = MODEL_DIR / "tfidf_logistic_regression.joblib"
    joblib.dump(model, model_path)

    # Save validation metrics.
    report_path = REPORT_DIR / "baseline_metrics.json"
    report_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Save validation confusion matrix.
    matrix = confusion_matrix(
        y_validation, predictions, labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Fake", "Real"],
    )

    display.plot(values_format="d")
    plt.title("Baseline Validation Confusion Matrix")
    plt.tight_layout()

    figure_path = FIGURE_DIR / "baseline_confusion_matrix.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {report_path}")
    print(f"Saved confusion matrix: {figure_path}")


if __name__ == "__main__":
    main()
