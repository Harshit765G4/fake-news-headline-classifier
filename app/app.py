
"""Streamlit application for Fake News Headline Classification."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from joblib import load
from lime.lime_text import LimeTextExplainer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "distilbert_fake_news"
TFIDF_MODEL_PATH = ROOT / "models" / "tfidf_logistic_regression.joblib"

METRICS_PATH = ROOT / "reports" / "final_test_metrics.json"

TFIDF_CM_PATH = (
    ROOT / "reports" / "figures"
    / "tfidf_baseline_test_confusion_matrix.png"
)
DISTILBERT_CM_PATH = (
    ROOT / "reports" / "figures"
    / "distilbert_test_confusion_matrix.png"
)

SRC_DIR = ROOT / "src" / "fake_news"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_predict import load_headline_csv, predict_batch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 64
MAX_UPLOAD_ROWS = 5000
CLASS_NAMES = ["Fake", "Real"]


st.set_page_config(
    page_title="Fake News Headline Classifier",
    page_icon="📰",
    layout="wide",
)


# ============================================================
# Cached model loading
# ============================================================

@st.cache_resource
def load_transformer_model():
    """Load and cache the trained DistilBERT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


@st.cache_resource
def load_tfidf_model():
    """Load and cache the TF-IDF + Logistic Regression pipeline."""
    if not TFIDF_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF model not found: {TFIDF_MODEL_PATH}"
        )

    return load(TFIDF_MODEL_PATH)


@st.cache_resource
def load_lime_explainer():
    """Create and cache the LIME text explainer."""
    return LimeTextExplainer(
        class_names=CLASS_NAMES,
        random_state=42,
    )


# ============================================================
# Single-headline prediction
# ============================================================

def predict_headline(headline, tokenizer, model):
    """Predict Fake/Real probabilities using DistilBERT."""
    encoded = tokenizer(
        headline,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    prediction_id = int(np.argmax(probabilities))

    return {
        "label": CLASS_NAMES[prediction_id],
        "confidence": float(probabilities[prediction_id]),
        "fake_probability": float(probabilities[0]),
        "real_probability": float(probabilities[1]),
    }


# ============================================================
# LIME explanation for TF-IDF
# ============================================================

def explain_tfidf_headline(headline, tfidf_model, num_features=10):
    """Explain the TF-IDF model's local prediction."""
    explanation = load_lime_explainer().explain_instance(
        text_instance=headline,
        classifier_fn=tfidf_model.predict_proba,
        num_features=num_features,
        labels=[0, 1],
    )

    prediction_id = int(tfidf_model.predict([headline])[0])

    return explanation, prediction_id


def display_lime_explanation(headline, transformer_label):
    """Display the TF-IDF explanation alongside the DistilBERT result."""
    st.subheader("LIME explainability")

    st.caption(
        "LIME explains the TF-IDF baseline, not DistilBERT. "
        "Word contributions describe model behavior, not factual truth."
    )

    if not TFIDF_MODEL_PATH.exists():
        st.info(
            "The TF-IDF model was not found. "
            "Train the baseline model before using LIME."
        )
        return

    try:
        with st.spinner("Generating word-level explanation..."):
            tfidf_model = load_tfidf_model()

            explanation, prediction_id = explain_tfidf_headline(
                headline,
                tfidf_model,
            )

        tfidf_label = CLASS_NAMES[prediction_id]

        col1, col2 = st.columns(2)
        col1.metric("DistilBERT prediction", transformer_label)
        col2.metric("TF-IDF prediction", tfidf_label)

        if transformer_label != tfidf_label:
            st.warning(
                "The models disagree. Neither prediction establishes "
                "whether the headline is true."
            )

        contributions = [
            (str(word), float(weight))
            for word, weight in explanation.as_list(label=prediction_id)
        ]

        contributions.sort(key=lambda item: item[1])

        if not contributions:
            st.info("No word contributions were returned.")
            return

        words = [word for word, _ in contributions]
        weights = [weight for _, weight in contributions]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(words, weights)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME contribution")
        ax.set_ylabel("Word or token")
        ax.set_title(f"Words influencing the TF-IDF {tfidf_label} prediction")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        contribution_df = pd.DataFrame(
            [
                {
                    "Word": word,
                    "Contribution": round(weight, 5),
                    "Direction": (
                        f"Supports {tfidf_label}"
                        if weight > 0
                        else f"Opposes {tfidf_label}"
                        if weight < 0
                        else "Neutral"
                    ),
                }
                for word, weight in reversed(contributions)
            ]
        )

        st.dataframe(
            contribution_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "LIME weights are local approximation values, not probabilities "
            "or proof that a particular word causes a prediction."
        )

    except Exception:
        st.error(
            "Could not generate the LIME explanation. "
            "The DistilBERT prediction remains available."
        )
        st.exception(Exception)


# ============================================================
# Model evaluation dashboard
# ============================================================

def display_evaluation_dashboard():
    """Display saved held-out test metrics and confusion matrices."""
    st.header("3. Model Evaluation Dashboard")

    st.write(
        "Compare TF-IDF + Logistic Regression with DistilBERT "
        "using the saved held-out test evaluation."
    )

    if not METRICS_PATH.exists():
        st.warning(
            "Saved test metrics were not found. "
            "Run the final evaluation script first."
        )
        return

    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        baseline = metrics["tfidf_logistic_regression"]
        transformer = metrics["distilbert"]

        required_keys = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]

        for model_metrics in (baseline, transformer):
            for key in required_keys:
                if key not in model_metrics:
                    raise ValueError(f"Missing metric: {key}")

        st.caption(
            f"Held-out test examples: {metrics['test_rows']:,}. "
            "These are saved final-test results."
        )

        # Overall performance
        st.subheader("Overall performance")

        metric_definitions = [
            ("Accuracy", "accuracy"),
            ("Macro precision", "precision_macro"),
            ("Macro recall", "recall_macro"),
            ("Macro F1", "f1_macro"),
        ]

        columns = st.columns(4)

        for column, (label, key) in zip(columns, metric_definitions):
            difference = (
                transformer[key] - baseline[key]
            ) * 100

            column.metric(
                label,
                f"{transformer[key]:.2%}",
                delta=f"{difference:+.2f} pp",
                help=(
                    "DistilBERT score. The difference compares it "
                    "with TF-IDF in percentage points."
                ),
            )

        comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Macro precision",
                    "Macro recall",
                    "Macro F1",
                ],
                "TF-IDF + Logistic Regression": [
                    baseline["accuracy"],
                    baseline["precision_macro"],
                    baseline["recall_macro"],
                    baseline["f1_macro"],
                ],
                "DistilBERT": [
                    transformer["accuracy"],
                    transformer["precision_macro"],
                    transformer["recall_macro"],
                    transformer["f1_macro"],
                ],
            }
        )

        st.dataframe(
            comparison_df.style.format(
                {
                    "TF-IDF + Logistic Regression": "{:.2%}",
                    "DistilBERT": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            comparison_df.set_index("Metric"),
            y=[
                "TF-IDF + Logistic Regression",
                "DistilBERT",
            ],
            y_label="Score",
        )

        # Class-level metrics
        st.subheader("Performance by class")

        selected_class = st.selectbox(
            "Select a class",
            ["Fake", "Real"],
            key="evaluation_class",
        )

        baseline_report = baseline["classification_report"]
        transformer_report = transformer["classification_report"]

        if (
            selected_class not in baseline_report
            or selected_class not in transformer_report
        ):
            st.warning(
                f"Class metrics for {selected_class} are unavailable."
            )
        else:
            baseline_class = baseline_report[selected_class]
            transformer_class = transformer_report[selected_class]

            class_df = pd.DataFrame(
                {
                    "Metric": ["Precision", "Recall", "F1-score"],
                    "TF-IDF + Logistic Regression": [
                        baseline_class["precision"],
                        baseline_class["recall"],
                        baseline_class["f1-score"],
                    ],
                    "DistilBERT": [
                        transformer_class["precision"],
                        transformer_class["recall"],
                        transformer_class["f1-score"],
                    ],
                }
            )

            st.dataframe(
                class_df.style.format(
                    {
                        "TF-IDF + Logistic Regression": "{:.2%}",
                        "DistilBERT": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"Number of {selected_class} examples in the test set: "
                f"{int(transformer_class['support']):,}"
            )

        # Confusion matrices
        st.subheader("Confusion matrices")

        cm_col1, cm_col2 = st.columns(2)

        with cm_col1:
            st.markdown("**TF-IDF + Logistic Regression**")

            if TFIDF_CM_PATH.exists():
                st.image(
                    str(TFIDF_CM_PATH),
                    caption="TF-IDF held-out test confusion matrix",
                    use_container_width=True,
                )
            else:
                st.warning("TF-IDF confusion matrix not found.")

        with cm_col2:
            st.markdown("**DistilBERT**")

            if DISTILBERT_CM_PATH.exists():
                st.image(
                    str(DISTILBERT_CM_PATH),
                    caption="DistilBERT held-out test confusion matrix",
                    use_container_width=True,
                )
            else:
                st.warning("DistilBERT confusion matrix not found.")

        st.info(
            "Precision measures how often predictions for a class are "
            "correct. Recall measures how many examples of that class "
            "are detected. F1 balances precision and recall."
        )

        st.warning(
            "These results measure performance on this dataset's held-out "
            "test split. Dataset artifacts and distribution shifts can "
            "reduce real-world performance. This classifier does not "
            "fact-check claims."
        )

    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        st.error("Could not read the saved evaluation metrics.")
        st.exception(exc)


# ============================================================
# Main page
# ============================================================

st.title("📰 Fake News Headline Classifier")

st.write(
    "Classify individual headlines, explore word-level explanations, "
    "predict headlines in bulk, and compare model evaluation results."
)

st.info(
    "This is a machine-learning demonstration, not a fact-checking service. "
    "Predictions do not establish whether a claim is true."
)

if not MODEL_DIR.exists():
    st.error(
        "The trained DistilBERT model was not found. "
        "Run the transformer training script first."
    )
    st.stop()

try:
    tokenizer, model = load_transformer_model()
except Exception as exc:
    st.error("Failed to load the DistilBERT model.")
    st.exception(exc)
    st.stop()


# ============================================================
# Section 1: Single-headline prediction
# ============================================================

st.header("1. Classify a single headline")

headline = st.text_area(
    "Enter a news headline",
    placeholder="Type or paste a headline here...",
    height=120,
    max_chars=2000,
    key="single_headline_input",
)

if st.button(
    "Classify headline",
    type="primary",
    use_container_width=True,
):
    cleaned = " ".join(headline.split())

    if not cleaned:
        st.warning("Please enter a headline first.")
    else:
        with st.spinner("Analyzing headline patterns..."):
            st.session_state["single_result"] = predict_headline(
                cleaned,
                tokenizer,
                model,
            )
            st.session_state["single_text"] = cleaned


if (
    "single_result" in st.session_state
    and "single_text" in st.session_state
):
    result = st.session_state["single_result"]
    cleaned = st.session_state["single_text"]

    if result["label"] == "Fake":
        st.error(f"Model prediction: {result['label']}")
    else:
        st.success(f"Model prediction: {result['label']}")

    st.metric(
        "Model confidence (not truth probability)",
        f"{result['confidence']:.2%}",
    )

    if result["confidence"] < 0.70:
        st.warning("Low model confidence: this prediction is uncertain.")
    else:
        st.info(
            "The model is confident in its learned classification, "
            "but confidence does not guarantee correctness."
        )

    st.warning(
        "This model classifies headline patterns. It does not verify "
        "the underlying claim, a person's identity, or whether an event "
        "actually occurred."
    )

    col1, col2 = st.columns(2)
    col1.metric("Fake probability", f"{result['fake_probability']:.2%}")
    col2.metric("Real probability", f"{result['real_probability']:.2%}")

    with st.expander("Show LIME explanation", expanded=True):
        display_lime_explanation(cleaned, result["label"])


# ============================================================
# Section 2: CSV batch prediction
# ============================================================

st.divider()
st.header("2. Batch prediction from CSV")

st.write(
    "Upload a CSV containing a column named `headline`. "
    "The app predicts each non-empty headline and preserves the input rows."
)

st.caption(
    f"Maximum upload size: {MAX_UPLOAD_ROWS:,} rows. "
    "Empty headlines are marked as skipped."
)

uploaded_file = st.file_uploader(
    "Upload headlines CSV",
    type=["csv"],
    key="batch_csv_upload",
)

if uploaded_file is not None:
    try:
        original_df, headlines = load_headline_csv(
            uploaded_file,
            max_rows=MAX_UPLOAD_ROWS,
        )

        if original_df.empty:
            st.warning("The uploaded CSV contains no data rows.")

        else:
            result_df = original_df.copy()
            result_df["headline"] = headlines

            valid_mask = headlines.ne("")
            valid_headlines = headlines[valid_mask].tolist()

            if valid_headlines:
                with st.spinner(
                    f"Classifying {len(valid_headlines):,} headlines..."
                ):
                    predictions = predict_batch(
                        valid_headlines,
                        tokenizer,
                        model,
                        DEVICE,
                        max_length=MAX_LENGTH,
                        batch_size=16,
                    )

                for column in predictions.columns:
                    if column == "headline":
                        continue

                    result_df[column] = pd.Series(
                        [None] * len(result_df),
                        index=result_df.index,
                        dtype=object,
                    )

                    result_df.loc[valid_mask, column] = (
                        predictions[column].to_numpy()
                    )

            else:
                for column in [
                    "predicted_label",
                    "confidence",
                    "fake_probability",
                    "real_probability",
                ]:
                    result_df[column] = None

            result_df["status"] = np.where(
                valid_mask,
                "Predicted",
                "Skipped: empty headline",
            )

            st.success(
                f"Processed {len(result_df):,} rows: "
                f"{int(valid_mask.sum()):,} predictions, "
                f"{int((~valid_mask).sum()):,} skipped."
            )

            if valid_mask.any():
                counts = (
                    result_df.loc[valid_mask, "predicted_label"]
                    .value_counts()
                )

                col1, col2 = st.columns(2)
                col1.metric("Predicted Fake", int(counts.get("Fake", 0)))
                col2.metric("Predicted Real", int(counts.get("Real", 0)))

            st.subheader("Batch results preview")

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Download prediction results (CSV)",
                data=csv_bytes,
                file_name="fake_news_batch_predictions.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

    except (ValueError, pd.errors.EmptyDataError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Could not process this CSV file.")
        st.exception(exc)


# ============================================================
# Section 3: Model evaluation dashboard
# ============================================================

st.divider()
display_evaluation_dashboard()


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Models: DistilBERT + TF-IDF Logistic Regression | "
    "Task: binary headline classification | Labels: Fake = 0, Real = 1"
)
