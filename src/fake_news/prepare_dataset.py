
"""Prepare a deduplicated headline-level classification dataset."""

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_PATH = PROCESSED_DIR / "headlines.csv"
REPORT_PATH = REPORT_DIR / "preprocessing_report.json"
CONFLICT_PATH = REPORT_DIR / "conflicting_headlines.csv"

SOURCES = [
    ("Fake.csv", 0, "fake"),
    ("True.csv", 1, "real"),
]


def normalize_headline(value):
    """Normalize whitespace and letter case for grouping."""
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def load_source(filename, label, label_name):
    """Load headline text and attach its source label."""
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    df = pd.read_csv(path, low_memory=False)

    if "title" not in df.columns:
        raise ValueError(f"{filename} has no 'title' column")

    result = pd.DataFrame({
        "headline": df["title"].astype("string"),
        "label": label,
        "label_name": label_name,
        "source_file": filename,
    })

    result["normalized_headline"] = (
        result["headline"].map(normalize_headline)
    )

    return result


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    frames = [
        load_source(filename, label, label_name)
        for filename, label, label_name in SOURCES
    ]

    combined = pd.concat(frames, ignore_index=True)
    initial_rows = len(combined)

    # Remove missing or whitespace-only headlines.
    valid = combined[
        combined["normalized_headline"] != ""
    ].copy()

    removed_empty = initial_rows - len(valid)

    # A normalized headline should not have conflicting labels.
    label_counts = valid.groupby(
        "normalized_headline"
    )["label"].nunique()

    conflict_keys = label_counts[
        label_counts > 1
    ].index

    conflicts = valid[
        valid["normalized_headline"].isin(conflict_keys)
    ].copy()

    conflicts.to_csv(CONFLICT_PATH, index=False)

    # Exclude conflicting groups from the modelling dataset.
    eligible = valid[
        ~valid["normalized_headline"].isin(conflict_keys)
    ].copy()

    before_dedup = len(eligible)

    # Keep the first source record for each normalized headline.
    # Original headline formatting is retained for readability.
    cleaned = eligible.drop_duplicates(
        subset="normalized_headline",
        keep="first",
    ).copy()

    cleaned = cleaned[
        [
            "headline",
            "label",
            "label_name",
            "normalized_headline",
            "source_file",
        ]
    ].reset_index(drop=True)

    cleaned.to_csv(OUTPUT_PATH, index=False)

    report = {
        "project": "Fake News Headline Binary Classifier",
        "initial_rows": int(initial_rows),
        "removed_empty_headlines": int(removed_empty),
        "conflicting_headline_groups": int(len(conflict_keys)),
        "rows_in_conflicting_groups": int(len(conflicts)),
        "eligible_rows_before_deduplication": int(before_dedup),
        "unique_headlines_after_deduplication": int(len(cleaned)),
        "duplicate_rows_removed": int(before_dedup - len(cleaned)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in cleaned["label"].value_counts().items()
        },
        "class_names": {"0": "fake", "1": "real"},
        "output_file": str(OUTPUT_PATH),
        "conflict_file": str(CONFLICT_PATH),
        "notes": [
            "Raw files are not modified.",
            "Only the headline is used as model input.",
            "Duplicate grouping uses lowercase and normalized whitespace.",
            "The first record is retained for each normalized headline.",
            "Dataset labels do not establish factual truth.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("HEADLINE DATASET PREPARATION")
    print("=" * 60)
    print(f"Initial rows: {initial_rows:,}")
    print(f"Empty headlines removed: {removed_empty:,}")
    print(f"Conflicting headline groups: {len(conflict_keys):,}")
    print(f"Rows in conflicting groups: {len(conflicts):,}")
    print(f"Rows before deduplication: {before_dedup:,}")
    print(f"Duplicate rows removed: {before_dedup - len(cleaned):,}")
    print(f"Final unique headlines: {len(cleaned):,}")
    print("\nClass counts:")
    print(cleaned["label_name"].value_counts().to_string())
    print(f"\nSaved dataset: {OUTPUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved conflict review: {CONFLICT_PATH}")


if __name__ == "__main__":
    main()
