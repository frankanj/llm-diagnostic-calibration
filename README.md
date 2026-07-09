# LLM Diagnostic Calibration

## Research question

Are LLMs appropriately calibrated when acting as symptom checkers? Does stated
confidence actually predict correctness, and are they overconfident specifically on
ambiguous/hard cases?

## Motivation

LLM's are increasingly being used for informal diagnoses based on symptoms the user inputs, so if confidence doesn't track correctness than a safety issue occurs.

## Data

- **Symptom2Disease** (Kaggle) — free-text symptom descriptions paired with disease labels.
- *(Optional second dataset — structured symptom checklist, if used for the cross-format check.)*

**Caveat:** Symptom2Disease is a portfolio-appropriate dataset, not validated clinical
data. This project is a methodological contribution — a way to test calibration vs.
difficulty — not a clinical claim.

## Method

1. Fit a baseline classical model (logistic regression / random forest) on structured
   data to get predicted probabilities per diagnosis class.
2. Prompt an LLM with the same case (free text) to get a top diagnosis + confidence
   (and top-3 guesses).
3. Define an objective "difficulty" proxy from the classical model's prediction
   entropy (flat distribution = hard/ambiguous, peaked = easy).
4. Compute Expected Calibration Error (ECE), Brier score, and reliability diagrams
   for the LLM — overall, and split by easy vs. hard cases.
5. *(Stretch)* Repeat across 2-3 models of different sizes/providers.

## Results

*(Fill in once the analysis is done — lead with the headline number/finding, then
the reliability diagrams, then the easy/hard split.)*

## Limitations

- Dataset is not clinically validated.
- Difficulty proxy is model-derived, not ground-truth difficulty.
- *(Add others as they come up — e.g. prompt sensitivity, single-run variance.)*

## Repo structure

```
data/         raw + processed datasets
src/          pipeline code (prompting, evaluation)
notebooks/    exploration / analysis
results/      saved outputs, figures
```

## Setup

```bash
pip install -r requirements.txt
```

Add your API key(s) as environment variables (see `.env.example`) — never commit keys.
