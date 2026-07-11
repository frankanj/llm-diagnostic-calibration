"""
Step 4: Calibration analysis.

Joins the baseline model's difficulty labels (results/baseline_predictions.csv)
with the LLM's diagnoses + confidence (results/llm_predictions.csv) on case_id,
then computes:
  - Expected Calibration Error (ECE)
  - Brier score
  - Reliability diagram (confidence vs. actual accuracy)
  - The same three, split by difficulty bucket (easy/medium/hard) -- this is
    the core research question: is the LLM overconfident specifically on
    cases the baseline model found ambiguous?

Important simplification, stated explicitly: because confidence is reported
per-diagnosis (not as a full probability distribution over all 24 classes),
these are BINARY calibration metrics -- "was the top-1 diagnosis correct"
vs. "how confident was the model in that top-1 diagnosis" -- not full
multi-class Brier/ECE over the whole label set. This matches what verbal
top-1 confidence can actually support. Document this in the README.

Run from repo root (after both baseline_model.py and llm_pipeline.py have
been run):
    python src/calibration_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
BASELINE_PATH = RESULTS_DIR / "baseline_predictions.csv"
LLM_PATH = RESULTS_DIR / "llm_predictions.csv"


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins_labels=None) -> float:
    """ECE using the discrete confidence values actually present (here: 3
    verbal-confidence anchors) as bins, rather than equal-width bins -- since
    confidence only takes 3 possible values, using more bins would just
    create empty ones."""
    bins = np.unique(confidence)
    total_n = len(confidence)
    ece = 0.0
    for b in bins:
        mask = confidence == b
        bin_n = mask.sum()
        bin_confidence = confidence[mask].mean()
        bin_accuracy = correct[mask].mean()
        ece += (bin_n / total_n) * abs(bin_confidence - bin_accuracy)
    return ece


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Binary Brier score: mean squared error between stated confidence and
    the binary correctness outcome. See module docstring for why this is
    binary rather than full multi-class Brier."""
    return float(np.mean((confidence - correct) ** 2))


def reliability_table(confidence: np.ndarray, correct: np.ndarray, confidence_label: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"confidence": confidence, "correct": correct, "label": confidence_label})
    table = df.groupby("label").agg(
        n=("correct", "size"),
        mean_confidence=("confidence", "mean"),
        accuracy=("correct", "mean"),
    ).reset_index()
    table["gap"] = table["mean_confidence"] - table["accuracy"]
    # Order low/medium/high sensibly rather than alphabetically
    order = {"low": 0, "medium": 1, "high": 2}
    table["_order"] = table["label"].map(order)
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return table


def plot_reliability_diagram(table: pd.DataFrame, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.scatter(table["mean_confidence"], table["accuracy"], s=table["n"] * 2, alpha=0.7, color="C0")
    for _, row in table.iterrows():
        ax.annotate(f"{row['label']} (n={row['n']})",
                     (row["mean_confidence"], row["accuracy"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean stated confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Couldn't find {BASELINE_PATH}. Run src/baseline_model.py first.")
    if not LLM_PATH.exists():
        raise FileNotFoundError(f"Couldn't find {LLM_PATH}. Run src/llm_pipeline.py first.")

    baseline = pd.read_csv(BASELINE_PATH)
    llm = pd.read_csv(LLM_PATH)

    # Drop cases where the LLM call failed to parse -- can't score these
    n_failed = llm["parse_error"].notna().sum()
    if n_failed > 0:
        print(f"Excluding {n_failed} cases with parse_error (LLM call/parse failed)")
    llm = llm[llm["parse_error"].isna()].copy()

    # Merge on case_id. Inner join: only cases present in both files.
    df = llm.merge(
        baseline[["case_id", "difficulty_bucket", "baseline_entropy_normalized"]],
        on="case_id", how="inner"
    )
    n_dropped = len(llm) - len(df)
    if n_dropped > 0:
        print(f"Warning: {n_dropped} LLM-predicted cases had no matching baseline case_id "
              f"(check both scripts ran against the same data/symptom2disease_clean.csv)")

    df["correct"] = (df["diagnosis"] == df["true_label"]).astype(int)

    confidence = df["confidence_numeric"].to_numpy()
    correct = df["correct"].to_numpy()
    confidence_label = df["confidence_label"].to_numpy()

    print("=" * 60)
    print(f"OVERALL  (n={len(df)})")
    print("=" * 60)
    overall_accuracy = correct.mean()
    overall_ece = expected_calibration_error(confidence, correct)
    overall_brier = brier_score(confidence, correct)
    print(f"Accuracy:     {overall_accuracy:.3f}")
    print(f"ECE:          {overall_ece:.3f}")
    print(f"Brier score:  {overall_brier:.3f}")

    overall_table = reliability_table(confidence, correct, confidence_label)
    print("\nReliability by stated confidence level:")
    print(overall_table.to_string(index=False))

    plot_reliability_diagram(overall_table, "Reliability diagram (overall)",
                              RESULTS_DIR / "reliability_overall.png")

    # ---- Core research question: easy vs. medium vs. hard ----
    print("\n" + "=" * 60)
    print("BY DIFFICULTY BUCKET (from baseline model entropy)")
    print("=" * 60)

    difficulty_summary = []
    for bucket in ["easy", "medium", "hard"]:
        sub = df[df["difficulty_bucket"] == bucket]
        if len(sub) == 0:
            continue
        sub_conf = sub["confidence_numeric"].to_numpy()
        sub_correct = sub["correct"].to_numpy()
        acc = sub_correct.mean()
        ece = expected_calibration_error(sub_conf, sub_correct)
        brier = brier_score(sub_conf, sub_correct)
        mean_conf = sub_conf.mean()
        gap = mean_conf - acc
        difficulty_summary.append({
            "difficulty": bucket, "n": len(sub), "accuracy": acc,
            "mean_confidence": mean_conf, "overconfidence_gap": gap,
            "ece": ece, "brier": brier,
        })
        print(f"\n-- {bucket} (n={len(sub)}) --")
        print(f"Accuracy: {acc:.3f}  |  Mean confidence: {mean_conf:.3f}  |  "
              f"Gap (confidence - accuracy): {gap:+.3f}")
        print(f"ECE: {ece:.3f}  |  Brier: {brier:.3f}")

        sub_table = reliability_table(sub_conf, sub_correct, sub["confidence_label"].to_numpy())
        plot_reliability_diagram(sub_table, f"Reliability diagram ({bucket} cases)",
                                  RESULTS_DIR / f"reliability_{bucket}.png")

    difficulty_df = pd.DataFrame(difficulty_summary)
    difficulty_df.to_csv(RESULTS_DIR / "calibration_by_difficulty.csv", index=False)

    print("\n" + "=" * 60)
    print("HEADLINE CHECK")
    print("=" * 60)
    if len(difficulty_df) == 3:
        easy_gap = difficulty_df.loc[difficulty_df["difficulty"] == "easy", "overconfidence_gap"].iloc[0]
        hard_gap = difficulty_df.loc[difficulty_df["difficulty"] == "hard", "overconfidence_gap"].iloc[0]
        print(f"Overconfidence gap on easy cases: {easy_gap:+.3f}")
        print(f"Overconfidence gap on hard cases: {hard_gap:+.3f}")
        if hard_gap > easy_gap:
            print(f"-> The LLM IS more overconfident on hard cases "
                  f"(gap grows by {hard_gap - easy_gap:.3f})")
        else:
            print(f"-> No evidence of increased overconfidence on hard cases "
                  f"in this run (gap did not grow)")

    # Save the merged, scored dataset too -- useful for further ad hoc analysis
    df.to_csv(RESULTS_DIR / "merged_scored_predictions.csv", index=False)
    print(f"\nSaved: results/calibration_by_difficulty.csv")
    print(f"Saved: results/merged_scored_predictions.csv")
    print(f"Saved: results/reliability_overall.png, reliability_easy.png, "
          f"reliability_medium.png, reliability_hard.png")


if __name__ == "__main__":
    main()