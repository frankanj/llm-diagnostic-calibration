"""
Step 3: LLM prompting pipeline.

Maps symptom free text -> predicted diagnosis + verbal confidence (low/med/high),
mapped to a numeric score for later calibration analysis.

Uses Groq's free API (OpenAI-compatible chat completions) with an open model
(Llama 3.1). Free tier, no credit card required: https://console.groq.com

Design notes:
  - The model is constrained to the same 24 disease labels as the dataset via
    the prompt, so predictions are directly comparable to true labels and to
    the baseline model's classes.
  - Confidence is reported verbally (low/medium/high) rather than as a raw
    percentage, per project decision. This is mapped to fixed numeric anchors
    (see CONFIDENCE_MAP below). Tradeoff to note in the README: 3 discrete
    confidence values means reliability diagrams / ECE only ever have 3 bins
    to work with -- a coarse but interpretable measurement.
  - Also asks for top-3 differential diagnoses (each with its own verbal
    confidence) to support a richer analysis later if wanted, but the core
    calibration analysis only needs the top-1 diagnosis + confidence.

Run from repo root (after setting GROQ_API_KEY in .env):
    python src/llm_pipeline.py --limit 20      # quick test on 20 cases
    python src/llm_pipeline.py                 # full dataset run
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path("data/symptom2disease_clean.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS_DIR / "llm_predictions.csv"

MODEL_NAME = "llama-3.1-8b-instant"

# Verbal confidence -> numeric anchor. Coarse by design (see module docstring).
CONFIDENCE_MAP = {"low": 0.33, "medium": 0.66, "high": 0.90}

DISEASE_LABELS = [
    "Acne", "Arthritis", "Bronchial Asthma", "Cervical spondylosis",
    "Chicken pox", "Common Cold", "Dengue", "Dimorphic Hemorrhoids",
    "Fungal infection", "Hypertension", "Impetigo", "Jaundice", "Malaria",
    "Migraine", "Pneumonia", "Psoriasis", "Typhoid", "Varicose Veins",
    "allergy", "diabetes", "drug reaction", "gastroesophageal reflux disease",
    "peptic ulcer disease", "urinary tract infection",
]

# Case-insensitive lookup: models (especially smaller ones) often auto-capitalize
# the start of JSON string values regardless of instructions. The dataset's own
# labels are inconsistently cased (e.g. "Migraine" vs. "urinary tract infection"),
# so an exact-match check on diagnosis was rejecting otherwise-correct answers.
_LABEL_LOOKUP = {label.lower(): label for label in DISEASE_LABELS}


def normalize_diagnosis(raw_diagnosis: str) -> str:
    """Map a model's diagnosis string to the canonical dataset label,
    tolerating case differences. Returns None if no match found at all."""
    return _LABEL_LOOKUP.get(raw_diagnosis.strip().lower())

SYSTEM_PROMPT = f"""You are a diagnostic assistant. Given a patient's free-text
description of their symptoms, identify the most likely diagnosis and your
confidence in that diagnosis.

You MUST choose diagnoses only from this exact list (use the exact spelling
and capitalization shown):
{", ".join(DISEASE_LABELS)}

Respond with ONLY a JSON object in this exact format, with no other text,
no markdown formatting, and no code fences:
{{
  "diagnosis": "<one label from the list above>",
  "confidence": "<low, medium, or high>",
  "top_3": [
    {{"diagnosis": "<label>", "confidence": "<low, medium, or high>"}},
    {{"diagnosis": "<label>", "confidence": "<low, medium, or high>"}},
    {{"diagnosis": "<label>", "confidence": "<low, medium, or high>"}}
  ]
}}

The "diagnosis" field must match your top_3[0] entry. Base your confidence on
how distinctive and unambiguous the described symptoms are for that specific
condition versus other conditions on the list."""


def get_client():
    """Lazy import + client creation so the module can be imported without
    the groq package installed (e.g. for testing parse_response alone)."""
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and add it to a .env file as GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=api_key)


def parse_response(raw_text: str) -> dict:
    """Parse the model's JSON response, tolerating stray markdown fences or
    leading/trailing text some models add despite instructions."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # If there's leading/trailing prose, grab the outermost {...}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    parsed = json.loads(text)  # raises if truly malformed -- caught by caller

    diagnosis = normalize_diagnosis(parsed["diagnosis"])
    confidence_label = parsed["confidence"].strip().lower()
    if diagnosis is None:
        raise ValueError(f"Model returned an out-of-list diagnosis: {parsed['diagnosis']!r}")
    if confidence_label not in CONFIDENCE_MAP:
        raise ValueError(f"Model returned an unrecognized confidence: {confidence_label!r}")

    top_3 = parsed.get("top_3", [])
    # Normalize top_3 diagnoses too, dropping any that don't match a known label
    # rather than failing the whole case over a secondary field.
    normalized_top_3 = []
    for entry in top_3:
        norm_label = normalize_diagnosis(entry.get("diagnosis", ""))
        if norm_label is not None:
            normalized_top_3.append({
                "diagnosis": norm_label,
                "confidence": entry.get("confidence", "").strip().lower(),
            })

    return {
        "diagnosis": diagnosis,
        "confidence_label": confidence_label,
        "confidence_numeric": CONFIDENCE_MAP[confidence_label],
        "top_3_raw": json.dumps(normalized_top_3),
        "parse_error": None,
    }


def diagnose_case(client, symptom_text: str, max_retries: int = 2) -> dict:
    """Send one case to the LLM and return a parsed result dict.
    Retries on failure before giving up and recording the error. Rate-limit
    errors (429) get a longer backoff since an instant retry will just fail
    again; other errors (empty response, malformed JSON) get a short backoff
    since those are usually transient."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Patient symptoms: {symptom_text}"},
                ],
                temperature=0.2,  # low temperature: favors consistent, less erratic outputs
                max_tokens=600,   # generous headroom so top_3 JSON doesn't get truncated
            )
            raw_text = response.choices[0].message.content
            if not raw_text or not raw_text.strip():
                raise ValueError("Empty response from model")
            result = parse_response(raw_text)
            result["raw_response"] = raw_text
            return result
        except Exception as e:
            last_error = str(e)
            if "rate_limit" in last_error or "429" in last_error:
                time.sleep(15)  # rate limits need real time to clear, not a quick retry
            else:
                time.sleep(1)

    # All attempts failed -- record the failure rather than silently dropping the row
    return {
        "diagnosis": None,
        "confidence_label": None,
        "confidence_numeric": None,
        "top_3_raw": None,
        "raw_response": None,
        "parse_error": last_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N cases (for a quick/cheap test run)")
    parser.add_argument("--sleep", type=float, default=2.0,
                         help="Seconds to sleep between API calls (stay under free-tier rate limits)")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Couldn't find {DATA_PATH}.")

    df = pd.read_csv(DATA_PATH)

    if args.limit:
        df = df.head(args.limit)

    # Resume support: skip cases that already SUCCEEDED. Failed cases (parse_error
    # set) are retried -- their old failed row is dropped and replaced below.
    already_succeeded = set()
    prev = None
    if OUT_PATH.exists():
        prev = pd.read_csv(OUT_PATH)
        already_succeeded = set(prev.loc[prev["parse_error"].isna(), "case_id"])
        n_prev_failed = prev["parse_error"].notna().sum()
        print(f"Found {len(already_succeeded)} previously successful cases (will skip) "
              f"and {n_prev_failed} previously failed cases (will retry).")

    client = get_client()
    rows = []
    n_errors = 0

    for i, row in df.iterrows():
        if row["case_id"] in already_succeeded:
            continue

        result = diagnose_case(client, row["text"])
        result["case_id"] = row["case_id"]
        result["true_label"] = row["label"]
        result["text"] = row["text"]
        rows.append(result)

        if result["parse_error"]:
            n_errors += 1
            print(f"[{i+1}/{len(df)}] case {row['case_id']}: FAILED -- {result['parse_error'][:120]}")
        else:
            correct = "✓" if result["diagnosis"] == row["label"] else "✗"
            print(f"[{i+1}/{len(df)}] case {row['case_id']}: {result['diagnosis']} "
                  f"({result['confidence_label']}) {correct}")

        # Write incrementally so a crash mid-run doesn't lose progress.
        # Combine: previously succeeded rows + all new attempts this run
        # (new attempts may include retries of previously-failed case_ids,
        # so we don't carry forward any old failed rows here).
        new_df = pd.DataFrame(rows)
        if prev is not None:
            prev_success_only = prev[prev["case_id"].isin(already_succeeded)]
            out_df = pd.concat([prev_success_only, new_df], ignore_index=True)
        else:
            out_df = new_df
        out_df.to_csv(OUT_PATH, index=False)

        time.sleep(args.sleep)

    print(f"\nDone. {n_errors} cases failed to parse after retries (see parse_error column).")
    print(f"Results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()