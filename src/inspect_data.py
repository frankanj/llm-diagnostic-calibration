"""
Step 1: Inspect the Symptom2Disease dataset.

Run from repo root:
    python src/inspect_data.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/symptom2disease.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {DATA_PATH}. Download the dataset from Kaggle "
            "(Symptom2Disease) and place the CSV at this path."
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("COLUMNS & DTYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())

    # Try to auto-detect the label/text columns — adjust if names differ
    label_col = next((c for c in df.columns if "label" in c.lower() or "disease" in c.lower()), df.columns[0])
    text_col = next((c for c in df.columns if "text" in c.lower() or "symptom" in c.lower()), df.columns[-1])
    print(f"\nDetected label column: '{label_col}'")
    print(f"Detected text column:  '{text_col}'")

    print("\n" + "=" * 60)
    print(f"CLASS BALANCE ({label_col})")
    print("=" * 60)
    class_counts = df[label_col].value_counts()
    print(class_counts)
    print(f"\nNumber of classes: {df[label_col].nunique()}")

    print("\n" + "=" * 60)
    print(f"TEXT LENGTH STATS ({text_col}, in words)")
    print("=" * 60)
    word_counts = df[text_col].astype(str).apply(lambda x: len(x.split()))
    print(word_counts.describe())

    print("\n" + "=" * 60)
    print("SAMPLE ROWS")
    print("=" * 60)
    print(df.sample(5, random_state=42))

    # Save a class balance chart for quick reference
    fig, ax = plt.subplots(figsize=(10, max(4, len(class_counts) * 0.3)))
    class_counts.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Count")
    ax.set_title("Class balance — Symptom2Disease")
    fig.tight_layout()
    out_path = RESULTS_DIR / "class_balance.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved class balance chart to {out_path}")


if __name__ == "__main__":
    main()