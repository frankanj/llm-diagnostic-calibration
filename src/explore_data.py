"""
Step 1b: Deeper data quality checks on Symptom2Disease.

Run from repo root:
    python src/explore_data.py
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/symptom2disease.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    # Drop the unnamed index column if present
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    print("=" * 60)
    print("EXACT DUPLICATE ROWS")
    print("=" * 60)
    dupes = df[df.duplicated(subset=["text"], keep=False)]
    print(f"Found {len(dupes)} rows involved in exact text duplicates")
    if len(dupes) > 0:
        print(dupes.sort_values("text").head(10))

    print("\n" + "=" * 60)
    print("NEAR-DUPLICATE CHECK (same text, different label)")
    print("=" * 60)
    # If any identical text string maps to more than one label, that's a red flag
    label_conflicts = df.groupby("text")["label"].nunique()
    conflicting = label_conflicts[label_conflicts > 1]
    print(f"Found {len(conflicting)} text strings mapped to multiple labels")
    if len(conflicting) > 0:
        print(conflicting)

    print("\n" + "=" * 60)
    print("TEXT LENGTH BY CLASS (word count)")
    print("=" * 60)
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    length_by_class = df.groupby("label")["word_count"].agg(["mean", "min", "max"]).sort_values("mean")
    print(length_by_class)

    print("\n" + "=" * 60)
    print("SPOT-CHECK: 2 SAMPLE ROWS FROM 5 RANDOM CLASSES")
    print("=" * 60)
    sample_classes = df["label"].drop_duplicates().sample(5, random_state=42)
    for cls in sample_classes:
        print(f"\n--- {cls} ---")
        for _, row in df[df["label"] == cls].sample(2, random_state=1).iterrows():
            print(f"  {row['text']}")


if __name__ == "__main__":
    main()