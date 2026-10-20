
"""Create reproducible train, validation, and test datasets."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "headlines.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "split_report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE_OF_REMAINDER = 0.15 / 0.85


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "headline",
        "label",
        "label_name",
        "normalized_headline",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    if df["headline"].isna().any() or df["label"].isna().any():
        raise ValueError("Missing headlines or labels detected.")

    # Ensure each normalized headline occurs only once.
    if df["normalized_headline"].duplicated().any():
        raise ValueError(
            "Duplicate normalized headlines found. "
            "Run prepare_dataset.py again before splitting."
        )

    # Step 1: reserve 15% for the final test set.
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Step 2: reserve 15% of the complete dataset for validation.
    train, validation = train_test_split(
        train_val,
        test_size=VALIDATION_SIZE_OF_REMAINDER,
        random_state=RANDOM_STATE,
        stratify=train_val["label"],
    )

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # Verify that no normalized headline crosses splits.
    headline_sets = {
        name: set(part["normalized_headline"])
        for name, part in splits.items()
    }

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = headline_sets[first] & headline_sets[second]

        if overlap:
            raise ValueError(
                f"Headline leakage detected between {first} "
                f"and {second}: {len(overlap)} overlaps"
            )

    # Save each split separately.
    report = {
        "random_state": RANDOM_STATE,
        "split_strategy": "stratified random split",
        "requested_proportions": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "total_rows": int(len(df)),
        "splits": {},
        "cross_split_headline_overlap": {},
    }

    for name, part in splits.items():
        output_path = OUTPUT_DIR / f"{name}.csv"
        part.to_csv(output_path, index=False)

        report["splits"][name] = {
            "rows": int(len(part)),
            "proportion": round(len(part) / len(df), 5),
            "class_counts": {
                str(label): int(count)
                for label, count in part["label_name"]
                .value_counts()
                .items()
            },
        }

        print(f"\n{name.upper()}")
        print(f"Rows: {len(part):,}")
        print(f"Saved: {output_path}")
        print(part["label_name"].value_counts().to_string())

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        report["cross_split_headline_overlap"][
            f"{first}_vs_{second}"
        ] = 0

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("DATASET SPLITTING COMPLETED")
    print("=" * 60)
    print(f"Total rows: {len(df):,}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
