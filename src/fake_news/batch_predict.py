
"""Batch prediction for fake-news headlines using DistilBERT."""

import pandas as pd
import torch


LABELS = {0: "Fake", 1: "Real"}


def predict_batch(
    headlines,
    tokenizer,
    model,
    device,
    max_length=64,
    batch_size=16,
):
    """Predict labels and probabilities for a sequence of headlines."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if max_length < 1:
        raise ValueError("max_length must be at least 1.")

    results = []

    # Process small batches to avoid using excessive GPU memory.
    for start in range(0, len(headlines), batch_size):
        batch = headlines[start : start + batch_size]

        encoded = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1).cpu()

        for headline, probs in zip(batch, probabilities.tolist()):
            prediction_id = int(torch.tensor(probs).argmax().item())

            results.append(
                {
                    "headline": headline,
                    "predicted_label": LABELS[prediction_id],
                    "confidence": float(probs[prediction_id]),
                    "fake_probability": float(probs[0]),
                    "real_probability": float(probs[1]),
                }
            )

    return pd.DataFrame(
        results,
        columns=[
            "headline",
            "predicted_label",
            "confidence",
            "fake_probability",
            "real_probability",
        ],
    )


def load_headline_csv(uploaded_file, max_rows=5000):
    """Validate a CSV upload and return cleaned headlines."""

    try:
        dataframe = pd.read_csv(
            uploaded_file,
            encoding="utf-8-sig",
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(
            "The uploaded file is not a valid UTF-8 CSV."
        ) from exc

    if "headline" not in dataframe.columns:
        raise ValueError(
            "The CSV must contain a column named 'headline'."
        )

    if len(dataframe) > max_rows:
        raise ValueError(
            f"The CSV contains {len(dataframe)} rows. "
            f"The maximum allowed is {max_rows}."
        )

    # Preserve every row; blank headlines receive an explicit status.
    headlines = dataframe["headline"].fillna("").astype(str).str.strip()

    return dataframe, headlines
