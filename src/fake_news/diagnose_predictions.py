
from pathlib import Path

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "models" / "tfidf_logistic_regression.joblib"
TRANSFORMER_PATH = ROOT / "models" / "distilbert_fake_news"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

headlines = [
    "Salman Khan is dead",
    "Amitabh Bachchan is dead",
    "Government announces new economic policy",
    "BREAKING: shocking secret they don't want you to know",
]

baseline = joblib.load(BASELINE_PATH)

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    TRANSFORMER_PATH
).to(DEVICE)
model.eval()

print("\nMODEL COMPARISON")
print("=" * 80)

for headline in headlines:
    baseline_probs = baseline.predict_proba([headline])[0]
    baseline_class = int(baseline.predict([headline])[0])

    inputs = tokenizer(
        headline,
        max_length=64,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        transformer_probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    transformer_class = int(
        torch.tensor(transformer_probs).argmax()
    )

    print(f"\nHeadline: {headline}")
    print(
        f"TF-IDF:    {'Fake' if baseline_class == 0 else 'Real'} "
        f"(fake={baseline_probs[0]:.2%}, real={baseline_probs[1]:.2%})"
    )
    print(
        f"DistilBERT: {'Fake' if transformer_class == 0 else 'Real'} "
        f"(fake={transformer_probs[0]:.2%}, "
        f"real={transformer_probs[1]:.2%})"
    )
