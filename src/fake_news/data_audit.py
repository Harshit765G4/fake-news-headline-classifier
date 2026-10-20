
"""Audit the raw ISOT fake/real news CSV files."""

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "data_audit.json"

FILES = {
    "fake": RAW_DIR / "Fake.csv",
    "real": RAW_DIR / "True.csv",
}


def normalize_headline(value):
    """Normalize whitespace and case for duplicate comparison."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def audit_file(label, path):
    """Load one CSV and calculate dataset quality statistics."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    if "title" not in df.columns:
        raise ValueError(
            f"{path.name} must contain a 'title' column. "
            f"Found columns: {df.columns.tolist()}"
        )

    normalized_titles = df["title"].map(normalize_headline)
    nonempty_titles = normalized_titles[normalized_titles != ""]

    report = {
        "file": path.name,
        "assigned_label": label,
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "missing_values": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
        },
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "missing_or_empty_titles": int(
            (normalized_titles == "").sum()
        ),
        "unique_nonempty_normalized_titles": int(
            nonempty_titles.nunique()
        ),
        "duplicate_normalized_title_rows": int(
            nonempty_titles.duplicated().sum()
        ),
    }

    return df, normalized_titles, report


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {}
    normalized = {}
    reports = {}

    for label, path in FILES.items():
        df, titles, report = audit_file(label, path)

        datasets[label] = df
        normalized[label] = titles
        reports[label] = report

    fake_titles = set(
        normalized["fake"][normalized["fake"] != ""]
    )
    real_titles = set(
        normalized["real"][normalized["real"] != ""]
    )

    overlap = fake_titles.intersection(real_titles)

    summary = {
        "dataset": "ISOT Fake and Real News Dataset",
        "task": "Headline binary classification",
        "label_mapping": {"fake": 0, "real": 1},
        "files": reports,
        "cross_class_overlap": {
            "unique_normalized_headlines_in_both_classes": len(
                overlap
            )
        },
        "total_rows": sum(
            item["rows"] for item in reports.values()
        ),
        "notes": [
            "Labels are inherited from the source files.",
            "Duplicate counts are diagnostic, not deletion instructions.",
            "Headline overlap across labels requires investigation.",
            "This audit does not establish factual truth."
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("DATASET AUDIT")
    print("=" * 60)

    for label, report in reports.items():
        print(f"\nClass: {label.upper()}")
        print(f"File: {report['file']}")
        print(f"Rows: {report['rows']:,}")
        print(f"Columns: {report['columns']}")
        print(
            "Exact duplicate rows:",
            f"{report['exact_duplicate_rows']:,}",
        )
        print(
            "Missing/empty titles:",
            f"{report['missing_or_empty_titles']:,}",
        )
        print(
            "Duplicate normalized title rows:",
            f"{report['duplicate_normalized_title_rows']:,}",
        )
        print("Missing values:", report["missing_values"])

    print("\nCross-class headline overlap:")
    print(
        "Unique headlines appearing in both classes:",
        f"{len(overlap):,}",
    )
    print(
        "\nTotal rows across both files:",
        f"{summary['total_rows']:,}",
    )
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
