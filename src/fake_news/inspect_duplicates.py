
"""Inspect repeated headlines without modifying raw data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def normalize_headline(series):
    return (
        series.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )


def inspect_file(filename, label, sample_size=10):
    path = RAW_DIR / filename
    df = pd.read_csv(path, low_memory=False)

    df["normalized_title"] = normalize_headline(df["title"])

    duplicated = df[
        df["normalized_title"].duplicated(keep=False)
    ].copy()

    print(f"\n{'=' * 65}")
    print(f"{label}: {filename}")
    print(f"Rows involved in repeated headlines: {len(duplicated):,}")
    print(f"Unique repeated headlines: {duplicated['normalized_title'].nunique():,}")

    if duplicated.empty:
        print("No repeated headlines found.")
        return

    # Show the most frequently repeated headlines.
    counts = (
        duplicated.groupby("normalized_title")
        .size()
        .sort_values(ascending=False)
        .head(sample_size)
    )

    print("\nMost frequently repeated headlines:")
    for headline, count in counts.items():
        print(f"\nOccurrences: {count}")
        print("Headline:", headline[:250])

        matching = duplicated[
            duplicated["normalized_title"] == headline
        ]

        for _, row in matching.head(3).iterrows():
            print("  Subject:", row.get("subject"))
            print("  Date:", row.get("date"))
            print("  Article excerpt:", str(row.get("text", ""))[:180])


def main():
    inspect_file("Fake.csv", "FAKE")
    inspect_file("True.csv", "REAL")


if __name__ == "__main__":
    main()
