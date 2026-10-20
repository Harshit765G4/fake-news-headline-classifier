
from pathlib import Path
import argparse

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_logistic_regression.joblib"


def predict_headline(headline: str) -> dict:
    """Predict the dataset label for one news headline."""
    headline = " ".join(headline.split())

    if not headline:
        raise ValueError("Headline cannot be empty.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run train_baseline.py first."
        )

    model = joblib.load(MODEL_PATH)

    prediction = int(model.predict([headline])[0])
    probabilities = model.predict_proba([headline])[0]

    labels = {0: "Fake", 1: "Real"}

    return {
        "headline": headline,
        "prediction": labels[prediction],
        "confidence": float(probabilities[prediction]),
        "fake_probability": float(probabilities[0]),
        "real_probability": float(probabilities[1]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify a news headline using the trained baseline model."
    )
    parser.add_argument(
        "headline",
        nargs="*",
        help="News headline to classify.",
    )
    args = parser.parse_args()

    if args.headline:
        headline = " ".join(args.headline)
    else:
        headline = input("Enter a news headline: ")

    try:
        result = predict_headline(headline)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    print("\n" + "=" * 50)
    print("FAKE NEWS HEADLINE CLASSIFIER")
    print("=" * 50)
    print(f"Headline:         {result['headline']}")
    print(f"Predicted label:  {result['prediction']}")
    print(f"Model confidence: {result['confidence']:.2%}")
    print(f"Fake probability: {result['fake_probability']:.2%}")
    print(f"Real probability: {result['real_probability']:.2%}")
    print("=" * 50)
    print("Note: This predicts dataset patterns, not verified truth.")


if __name__ == "__main__":
    main()
