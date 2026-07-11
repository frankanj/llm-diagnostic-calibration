"""
Step 2: Baseline classical model + difficulty proxy.

Fits a TF-IDF + Logistic Regression classifier on the Symptom2Disease text data.
This baseline serves two purposes:
  1. Sanity check on the dataset (is the task trivially separable, are labels clean?)
  2. Source of the "difficulty" proxy used later to stratify LLM calibration
     results into easy vs. hard cases. Difficulty = entropy of the predicted
     class-probability distribution for each case (flat distribution = hard/
     ambiguous, peaked distribution = easy).

Note: the project brief originally described this baseline as running on
"structured" symptom-checklist data. Only the free-text Symptom2Disease dataset
is available so far, so this baseline is fit directly on the text via TF-IDF.
If a structured dataset is added later, a second baseline can be built for the
cross-format calibration check.

Run from repo root:
    python src/baseline_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss

DATA_PATH = Path("data/symptom2disease.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Couldn't find {DATA_PATH}.")

    df = pd.read_csv(DATA_PATH)

    # Drop the unnamed index column if present
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Drop exact text duplicates (90 known from data exploration), keep first
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"Dropped {before - len(df)} exact duplicate rows (kept first occurrence)")
    print(f"Remaining rows: {len(df)}")

    # Keep a stable case ID so results can be joined with the LLM pipeline later
    df["case_id"] = df.index

    X_text = df["text"]
    y = df["label"]

    # Stratified split so all 24 classes are represented in both sets
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_test = X_text.loc[train_idx], X_text.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print("\n" + "=" * 60)
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print("=" * 60)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Baseline classifier
    clf = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, clf.predict_proba(X_test_vec), labels=clf.classes_)

    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy: {acc:.4f}")
    print(f"Log loss: {ll:.4f}")
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    # ---- Predicted probabilities + entropy-based difficulty proxy ----
    # Compute this over the FULL dataset (not just the test set) so every case
    # gets a difficulty label to attach to the LLM pipeline later.
    X_full_vec = vectorizer.transform(X_text)
    proba_full = clf.predict_proba(X_full_vec)  # shape (n_samples, n_classes)

    # Shannon entropy of each row's probability distribution.
    # Max possible entropy = log(n_classes); normalize to [0, 1] for readability.
    n_classes = len(clf.classes_)
    max_entropy = np.log(n_classes)
    row_entropy = entropy(proba_full, axis=1)
    normalized_entropy = row_entropy / max_entropy

    top1_idx = np.argmax(proba_full, axis=1)
    top1_label = clf.classes_[top1_idx]
    top1_prob = proba_full[np.arange(len(proba_full)), top1_idx]

    results = pd.DataFrame({
        "case_id": df["case_id"],
        "text": df["text"],
        "true_label": df["label"],
        "baseline_pred_label": top1_label,
        "baseline_pred_prob": top1_prob,
        "baseline_entropy": row_entropy,
        "baseline_entropy_normalized": normalized_entropy,
        "in_test_set": df.index.isin(test_idx),
    })

    # Difficulty tercile split for convenience (easy / medium / hard)
    results["difficulty_bucket"] = pd.qcut(
        results["baseline_entropy_normalized"],
        q=[0, 1/3, 2/3, 1],
        labels=["easy", "medium", "hard"],
    )

    out_path = RESULTS_DIR / "baseline_predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved per-case baseline predictions + difficulty proxy to {out_path}")

    print("\n" + "=" * 60)
    print("DIFFICULTY BUCKET COUNTS")
    print("=" * 60)
    print(results["difficulty_bucket"].value_counts().sort_index())

    print("\n" + "=" * 60)
    print("SAMPLE: 3 EASIEST AND 3 HARDEST CASES")
    print("=" * 60)
    print("\n-- Easiest (lowest entropy) --")
    print(results.nsmallest(3, "baseline_entropy")[["true_label", "baseline_pred_label", "baseline_pred_prob", "text"]])
    print("\n-- Hardest (highest entropy) --")
    print(results.nlargest(3, "baseline_entropy")[["true_label", "baseline_pred_label", "baseline_pred_prob", "text"]])


if __name__ == "__main__":
    main()