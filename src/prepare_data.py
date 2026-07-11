"""
Step 0 (prep): Clean and shuffle the raw dataset once, producing a single
canonical file that baseline_model.py, build_structured_data.py, and
llm_pipeline.py all read from.

Why this exists:
  - The raw CSV is sorted by disease label (50 rows per class, in a block).
    Shuffling means any partial/interrupted run (e.g. the LLM pipeline
    stopping halfway through) still covers a representative mix of classes.
  - case_id is derived from a hash of the case text, NOT row position. This
    makes it stable regardless of row order -- so baseline predictions and
    LLM predictions can always be joined correctly on case_id, even if the
    scripts are re-run after further shuffling or the file changes shape.

Run this FIRST, once. Re-run only if you want to regenerate the cleaned file
(e.g. after fixing a data issue) -- doing so changes case_id values, so any
existing results/*.csv files should be regenerated afterward too.

Run from repo root:
    python src/prepare_data.py
"""

import hashlib
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/symptom2disease.csv")
CLEAN_PATH = Path("data/symptom2disease_clean.csv")

RANDOM_STATE = 42


def make_case_id(text: str) -> str:
    """Short, stable, deterministic ID derived from case text content."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:8]


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Couldn't find {RAW_PATH}.")

    df = pd.read_csv(RAW_PATH)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    print(f"Dropped {before - len(df)} exact duplicate rows")

    # Shuffle so classes aren't grouped in blocks
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Stable, content-derived case_id (survives any future re-shuffling)
    df["case_id"] = df["text"].apply(make_case_id)

    n_dupe_ids = df["case_id"].duplicated().sum()
    if n_dupe_ids > 0:
        raise ValueError(
            f"{n_dupe_ids} case_id collisions found -- this shouldn't happen "
            "after deduplication. Investigate before proceeding."
        )

    # Reorder columns: case_id first
    df = df[["case_id", "label", "text"]]

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved {len(df)} shuffled, deduplicated cases to {CLEAN_PATH}")
    print("\nClass distribution after shuffle (first 20 rows, sanity check):")
    print(df.head(20)["label"].tolist())


if __name__ == "__main__":
    main()
