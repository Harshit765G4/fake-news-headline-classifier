
"""Explain TF-IDF fake-news predictions using LIME."""

from pathlib import Path

import numpy as np
from lime.lime_text import LimeTextExplainer
from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_logistic_regression.joblib"

CLASS_NAMES = ["Fake", "Real"]


def load_model():
    """Load the saved TF-IDF + Logistic Regression pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Train the baseline first."
        )

    return load(MODEL_PATH)


def explain_headline(headline: str, num_features: int = 10):
    """Return a LIME explanation for one headline."""
    if not isinstance(headline, str) or not headline.strip():
        raise ValueError("Headline must be a non-empty string.")

    if num_features < 1:
        raise ValueError("num_features must be at least 1.")

    model = load_model()
    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        random_state=42,
    )

    def predict_probabilities(texts):
        probabilities = model.predict_proba(texts)
        return np.asarray(probabilities)

    explanation = explainer.explain_instance(
        text_instance=headline.strip(),
        classifier_fn=predict_probabilities,
        num_features=num_features,
        labels=[0, 1],
    )

    return {
        "headline": headline.strip(),
        "predicted_label": CLASS_NAMES[int(model.predict([headline.strip()])[0])],
        "probabilities": {
            CLASS_NAMES[i]: float(probability)
            for i, probability in enumerate(
                model.predict_proba([headline.strip()])[0]
            )
        },
        "word_contributions": {
            CLASS_NAMES[label]: explanation.as_list(label=label)
            for label in [0, 1]
        },
        "explanation": explanation,
    }


if __name__ == "__main__":
    sample = "BREAKING: shocking secret they don't want you to know"

    result = explain_headline(sample)

    print(f"\nHeadline: {result['headline']}")
    print(f"Prediction: {result['predicted_label']}")
    print("\nModel probabilities:")

    for label, probability in result["probabilities"].items():
        print(f"  {label}: {probability:.2%}")

    print("\nLIME word contributions:")
    for label, contributions in result["word_contributions"].items():
        print(f"\n{label}:")
        for word, weight in contributions:
            direction = "supports" if weight > 0 else "opposes"
            print(f"  {word!r}: {weight:+.4f} ({direction} this class)")
