
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"

df = pd.read_csv(TRAIN_PATH)

# Inspect the training set only. Do not inspect test.csv for model tuning.
groups = {
    "celebrity names": [
        "salman khan",
        "amitabh bachchan",
    ],
    "death-related wording": [
        "is dead",
        "dead",
        "dies",
        "died",
        "death",
    ],
    "sensational wording": [
        "breaking",
        "shocking",
        "secret",
        "they don't want you to know",
    ],
}

for group_name, terms in groups.items():
    print("\n" + "=" * 75)
    print(f"GROUP: {group_name}")
    print("=" * 75)

    for term in terms:
        matches = df[
            df["headline"].astype(str).str.contains(
                term, case=False, regex=False, na=False
            )
        ]

        print(f"\nSearch term: {term!r} | Matches: {len(matches)}")

        if matches.empty:
            print("  No matching training headlines.")
            continue

        # Show a few examples for manual inspection.
        display_columns = ["headline", "label", "label_name"]
        print(
            matches[display_columns]
            .drop_duplicates()
            .head(8)
            .to_string(index=False)
        )
