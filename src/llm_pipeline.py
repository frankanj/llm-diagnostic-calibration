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

MODEL_NAME = "openai/gpt-oss-120b"

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

    diagnosis = parsed["diagnosis"]
    confidence_label = parsed["confidence"].strip().lower()
    if diagnosis not in DISEASE_LABELS:
        raise ValueError(f"Model returned an out-of-list diagnosis: {diagnosis!r}")
    if confidence_label not in CONFIDENCE_MAP:
        raise ValueError(f"Model returned an unrecognized confidence: {confidence_label!r}")

    top_3 = parsed.get("top_3", [])

    return {
        "diagnosis": diagnosis,
        "confidence_label": confidence_label,
        "confidence_numeric": CONFIDENCE_MAP[confidence_label],
        "top_3_raw": json.dumps(top_3),
        "parse_error": None,
    }


def diagnose_case(client, symptom_text: str, max_retries: int = 2) -> dict:
    """Send one case to the LLM and return a parsed result dict.
    Retries once on a parse failure (asking the model to strictly follow format)
    before giving up and recording the error."""
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
                max_tokens=400,
            )
            raw_text = response.choices[0].message.content
            result = parse_response(raw_text)
            result["raw_response"] = raw_text
            return result
        except Exception as e:
            last_error = str(e)
            time.sleep(1)  # brief backoff before retry

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

    # Resume support: skip cases already present in an existing output file
    already_done = set()
    if OUT_PATH.exists():
        prev = pd.read_csv(OUT_PATH)
        already_done = set(prev["case_id"])
        print(f"Found existing results for {len(already_done)} cases -- will skip those.")

    client = get_client()
    rows = []
    n_errors = 0

    for i, row in df.iterrows():
        if row["case_id"] in already_done:
            continue

        result = diagnose_case(client, row["text"])
        result["case_id"] = row["case_id"]
        result["true_label"] = row["label"]
        result["text"] = row["text"]
        rows.append(result)

        if result["parse_error"]:
            n_errors += 1
            print(f"[{i+1}/{len(df)}] case {row['case_id']}: FAILED -- {result['parse_error']}")
        else:
            correct = "✓" if result["diagnosis"] == row["label"] else "✗"
            print(f"[{i+1}/{len(df)}] case {row['case_id']}: {result['diagnosis']} "
                  f"({result['confidence_label']}) {correct}")

        # Write incrementally so a crash mid-run doesn't lose progress
        out_df = pd.DataFrame(rows)
        if OUT_PATH.exists() and already_done:
            out_df = pd.concat([prev, out_df], ignore_index=True)
        out_df.to_csv(OUT_PATH, index=False)

        time.sleep(args.sleep)

    print(f"\nDone. {n_errors} cases failed to parse after retries (see parse_error column).")
    print(f"Results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
