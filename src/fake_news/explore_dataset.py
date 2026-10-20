
"""Exploratory data analysis for the cleaned headline dataset."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "headlines.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_PATH = REPORT_DIR / "eda_summary.json"


# --------------------------------------------------
# 2. Main EDA pipeline
# --------------------------------------------------

def main():
    """Load the dataset, analyze it, create plots, and save a report."""

    # Create output directories if they do not exist.
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate the input file.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}\n"
            "Run prepare_dataset.py before running this script."
        )

    # Load the processed dataset.
    df = pd.read_csv(DATA_PATH)

    required_columns = {"headline", "label", "label_name"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The processed dataset is empty.")

    # Add analysis-only features.
    # These columns are not added to the original CSV file.
    df["word_count"] = (
        df["headline"].fillna("").astype(str).str.split().str.len()
    )
    df["character_count"] = (
        df["headline"].fillna("").astype(str).str.len()
    )

    # --------------------------------------------------
    # 3. Dataset overview
    # --------------------------------------------------

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nDataset shape:", df.shape)

    print("\nClass distribution:")
    class_counts = df["label_name"].value_counts()
    print(class_counts)

    print("\nClass proportions:")
    class_proportions = (
        df["label_name"].value_counts(normalize=True) * 100
    ).round(2)

    print(class_proportions.astype(str).add("%"))

    print("\nHeadline length statistics by class:")
    length_columns = ["word_count", "character_count"]

    print(
        df.groupby("label_name")[length_columns]
        .describe()
        .round(2)
    )

    # --------------------------------------------------
    # 4. Inspect shortest headlines
    # --------------------------------------------------

    print("\nShortest headlines:")

    print(
        df.nsmallest(10, "word_count")[
            ["headline", "label_name", "word_count"]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 5. Inspect longest headlines
    # --------------------------------------------------

    print("\nLongest headlines:")

    print(
        df.nlargest(10, "word_count")[
            ["headline", "label_name", "word_count"]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 6. Plot class distribution
    # --------------------------------------------------

    ordered_counts = class_counts.reindex(
        ["fake", "real"], fill_value=0
    )

    plt.figure(figsize=(8, 5))

    ordered_counts.plot(
        kind="bar",
        edgecolor="black",
    )

    plt.title("Headline Class Distribution")
    plt.xlabel("Dataset label")
    plt.ylabel("Number of headlines")
    plt.xticks(rotation=0)
    plt.tight_layout()

    class_plot_path = FIGURE_DIR / "class_distribution.png"

    plt.savefig(class_plot_path, dpi=150)
    plt.close()

    # --------------------------------------------------
    # 7. Plot headline word-count distributions
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    for label in ["fake", "real"]:
        subset = df.loc[
            df["label_name"] == label, "word_count"
        ]

        if not subset.empty:
            subset.plot(
                kind="hist",
                bins=40,
                alpha=0.55,
                label=label,
            )

    plt.title("Headline Length Distribution")
    plt.xlabel("Words per headline")
    plt.ylabel("Number of headlines")
    plt.legend()
    plt.tight_layout()

    length_plot_path = FIGURE_DIR / "headline_length.png"

    plt.savefig(length_plot_path, dpi=150)
    plt.close()

    # --------------------------------------------------
    # 8. Calculate numerical summary statistics
    # --------------------------------------------------

    length_stats = {}

    for label, group in df.groupby("label_name"):
        length_stats[str(label)] = {}

        for column in length_columns:
            statistics = group[column].agg(
                ["mean", "median", "min", "max"]
            )

            length_stats[str(label)][column] = {
                str(statistic): float(value)
                for statistic, value in statistics.items()
            }

    # IMPORTANT:
    # Build the complete summary once, outside the loop.
    summary = {
        "dataset": "Processed fake news headline dataset",
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "class_counts": {
            str(label): int(count)
            for label, count in class_counts.items()
        },
        "class_proportions_percent": {
            str(label): float(value)
            for label, value in class_proportions.items()
        },
        "length_statistics": length_stats,
        "headline_length_percentiles": {
            str(percentile): float(
                df["word_count"].quantile(percentile)
            )
            for percentile in [0.50, 0.90, 0.95, 0.99]
        },
        "figures": {
            "class_distribution": str(class_plot_path.relative_to(
                PROJECT_ROOT
            )),
            "headline_length": str(length_plot_path.relative_to(
                PROJECT_ROOT
            )),
        },
    }

    # --------------------------------------------------
    # 9. Save the EDA report
    # --------------------------------------------------

    REPORT_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 10. Final status
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Class distribution plot: {class_plot_path}")
    print(f"Headline length plot:    {length_plot_path}")
    print(f"Numerical report:        {REPORT_PATH}")


if __name__ == "__main__":
    main()
