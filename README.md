# AI-Based Text Classification Helper

Sorts short text entries (feedback, queries, etc.) into labels you define, using GroqCloud.

## Setup

```
python3 -m pip install -r requirements.txt
export GROQ_API_KEY="your-key"        (macOS/Linux)
set GROQ_API_KEY=your-key              (Windows cmd)
$env:GROQ_API_KEY="your-key"          (PowerShell)
```

## Run

- Web app: `streamlit run app.py` (edit the labels, classify one text or a batch/CSV)
- Evaluation: `python evaluate.py` (runs `test_data.csv`, prints accuracy, confusion matrix, and every incorrect classification; saves `results.csv`)

Optional: `CLASSIFIER_MODEL` env var switches the model (default `openai/gpt-oss-20b`).

## How it works

`classifier.py` sends the text and label list in one prompt and asks for JSON
(`label`, `confidence`, `reason`). The reply is validated: any label not in your list becomes `Unknown`.

## Test set

`test_data.csv` has 32 examples across 4 labels (Bug Report, Feature Request, Praise, Billing Question).
Each label has clear cases plus deliberately tricky ones (see the `note` column): mixed intent,
questions that hide bug reports, polite wording over a real request, and sarcasm-like phrasing.

## Discussing incorrect classifications (for the report)

After running `evaluate.py`, for each row in the "Incorrect classifications" output, discuss:
1. What the text says versus the label you expected.
2. Why the model chose differently (use its `reason` field).
3. Whether the ground truth itself is arguable (a mixed-intent text can have two valid labels).
4. A fix: sharper label names, label descriptions in the prompt, a "multiple/other" label, or few-shot examples.
