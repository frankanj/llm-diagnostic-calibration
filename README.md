# LLM Diagnostic Calibration

## Research question

Are LLMs appropriately calibrated when acting as symptom checkers? Does stated
confidence predict correctness, and are they overconfident on ambiguous/hard cases?

## Motivation

LLMs are increasingly used for informal diagnosis based on user-described symptoms.
If a model's stated confidence doesn't track its actual correctness, there is a
safety issue: a confidently wrong answer is more dangerous than a uncertain one.

## Data

**Symptom2Disease** (Kaggle): 1,200 free-text symptom descriptions across 24
disease labels, 50 per class. After removing 47 exact-duplicate descriptions,
1,153 unique cases remain. The Symptom2Disease dataset contains synthetic data, not validated
clinical data, therfore labels appear to be generated rather than drawn from
real patient records (see Limitations). Although when users input their symptoms into an LLM, 
they may get things wrong and it could look similar to the information in this dataset. 
This project is a methodologicalcontribution, a way to test calibration vs. difficulty, not a clinical claim.

## Method

1. **Prep** (`src/prepare_data.py`): remove duplicates, shuffle (so disease classes
   aren't grouped in blocks), and assign each case a stable ID to keep results
   joinable across scripts regardless of row order.
2. **Baseline model** (`src/baseline_model.py`): TF-IDF and logistic regression
   fit directly on the free text (94.1% test accuracy). Its role is to produce an objective, 
   difficulty proxy. For each case, the entropy of the baseline's predicted
   probability distribution across all 24 classes indicates how distinctive
   the symptom pattern is. Low entropy (one class dominates) = easy/unambiguous,
   high entropy (probability spread across classes) = hard/ambiguous. Cases are
   split into easy/medium/hard groupings by this score.
3. **LLM pipeline** (`src/llm_pipeline.py`): each case's free text is sent to
   an LLM (Llama 3.1 8B via Groq's free API), constrained by prompt to the same
   24 disease labels, asking for a top diagnosis plus **verbal confidence**
   (low/medium/high — a project design choice, mapped to fixed numeric anchors
   0.33/0.66/0.90) and a top-3 differential.
4. **Calibration analysis** (`src/calibration_analysis.py`): joins baseline
   difficulty labels with LLM predictions on case_id, computes Expected
   Calibration Error (ECE), Brier score, and reliability diagrams both overall,
   and split by easy/medium/hard.

## Results

*(Full run: n=1,153 cases, Llama 3.1 8B via Groq)*

**Headline finding: the LLM is substantially overconfident overall, and that
overconfidence gets worse on harder cases.**

| Difficulty | n   | Accuracy | Mean stated confidence | Overconfidence gap | ECE   | Brier |
| ---------- | --- | -------- | ---------------------- | ------------------ | ----- | ----- |
| Easy       | 385 | 0.673    | 0.850                  | +0.177             | 0.177 | 0.245 |
| Medium     | 384 | 0.659    | 0.849                  | +0.190             | 0.190 | 0.246 |
| Hard       | 384 | 0.461    | 0.792                  | **+0.331**         | 0.331 | 0.348 |

Overall accuracy: 59.8% · Overall ECE: 0.233 · Overall Brier: 0.280

Two distinct findings here, worth separating:

- **Baseline overconfidence, even on easy cases.**: Even where the model is
  most accurate (easy cases, 67.3%), stated confidence still runs 17.7 points
  higher than warranted. This shows up most prominently in the model's
  "high confidence" tier: across all 848 cases where the model claimed 90%
  confidence, actual accuracy was only 66.7% — a 23-point gap at the model's
  most confident setting.
- **The gap widens sharply on hard cases.** Easy and medium cases are close to
  each other (17.7 vs. 19.0 point gap). The real cliff is specifically at
  "hard," where the gap nearly doubles to 33.1 points, driven mostly by a
  drop in accuracy (46.1%) rather than a corresponding drop in confidence
  (still 79.2%). The model's confidence barely adjusts downward even as its
  actual reliability falls off substantially.

See `results/reliability_overall.png`, `reliability_easy.png`,
`reliability_medium.png`, and `reliability_hard.png` for the full reliability
diagrams, and `results/calibration_by_difficulty.csv` /
`results/merged_scored_predictions.csv` for the underlying numbers.

## Limitations

- **Dataset is not clinically validated.** Symptom2Disease is a practice
  dataset; several disease groups show templated or near-duplicate phrasing
  across cases, suggesting synthetic or semi-synthetic generation rather than
  real patient-reported data. Results describe calibration *behavior*, not
  real-world diagnostic safety.
- **Difficulty proxy is model-derived, not ground-truth difficulty.** "Hard"
  means "the TF-IDF/logistic regression baseline found this case's symptom
  vocabulary ambiguous across classes" — a reasonable, objective, and
  LLM-independent proxy, but not the same as clinical diagnostic difficulty.
- **Calibration metrics here are binary, not full multi-class.** Confidence
  was collected per top-1 diagnosis (verbal low/med/high) rather than as a
  full probability distribution over all 24 classes, so ECE/Brier reflect
  "was the top pick right, and how confident was the model in it" not
  full multi-class calibration.
- **Only 3 discrete confidence values.** Verbal confidence, mapped to 3 fixed
  numeric anchors, means reliability diagrams have only 3 points to work
  with which is coarse, though sufficient to reveal the pattern reported above.
  A finer-grained self-reported percentage would allow more granular
  reliability curves in future work.
- **Single model, single run.** Results reflect one model (Llama 3.1 8B) at
  one point in time, with no repeated-sampling variance estimate. Prompt
  wording, temperature, and model choice could all shift these numbers;
  no claim is made that this generalizes to other LLMs. Validating these findings across 
  various models/prompts would improve the results of this project.
- **Model was switched mid-project** from `openai/gpt-oss-120b` to
  `llama-3.1-8b-instant` after discovering the former's free-tier daily
  token quota (200K/day) couldn't cover the full dataset in a practical
  timeframe. All reported results use `llama-3.1-8b-instant` exclusively.
- **No negation or severity handling** in how symptom text is interpreted which
  affects the LLM directly (it reads raw text) rather than the analysis
  itself.

## Repo structure

```
data/         raw + cleaned/shuffled dataset (symptom2disease_clean.csv)
src/          pipeline code: prepare_data, baseline_model, llm_pipeline, calibration_analysis
results/      saved predictions, calibration tables, reliability diagram PNGs
```

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) (no
credit card required), then add it to a `.env` file in the repo root:

```
GROQ_API_KEY=your_key_here
```

Run the full pipeline in order:

```bash
python src/prepare_data.py
python src/baseline_model.py
python src/llm_pipeline.py       # supports --limit N for a quick test run
python src/calibration_analysis.py
```
