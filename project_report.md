# AI-Based Text Classification Helper

## Project Report

**Course:** CSE2503  
**Project type:** LLM-powered text classification application  
**Date:** 18 November 2026

## Abstract

This project implements a small application that assigns short user messages to one of a set of user-defined categories. The example categories represent common product feedback: Bug Report, Feature Request, Praise, and Billing Question. A Streamlit interface supports single-message and batch classification, while a separate evaluation script measures performance on a labelled CSV test set. Classification is performed by a Groq-hosted large language model (LLM), which returns a label, a confidence value, and a short explanation. The supplied results show 31 correct predictions out of 32 examples (96.9%). The sole error is a message phrased as a feature wish that actually describes an application defect, illustrating the difficulty of separating intent from surface wording.

## 1. Introduction

Short feedback messages are useful but can be difficult to triage manually when they arrive in large numbers. This project explores whether an LLM can sort these messages into a small set of categories while providing a brief explanation for each decision. The categories are supplied at run time, so the same interface can be adapted to other classification tasks by changing the labels.

The project focuses on short text entries and single-label classification. It is a prototype for assisting a human reviewer; its confidence value and explanation are informative outputs, not guarantees that a decision is correct.

## 2. Objectives

- Build an interface for entering or uploading short text messages.
- Let the user provide the category labels.
- Use an LLM to select one category and return a confidence value and reason.
- Evaluate the classifier against labelled examples and inspect its errors.

## 3. System Design and Implementation

### 3.1 Architecture

The application has three main parts:

1. **User interface (`app.py`):** A Streamlit web app accepts comma-separated labels and offers single-text and batch modes. Batch mode accepts one message per line or a CSV file with a `text` column, displays results, and allows the results to be downloaded as CSV.
2. **Classification logic (`classifier.py`):** The classifier formats the allowed labels and input text into a prompt, calls the Groq chat-completions API, and requests a JSON response containing `label`, `confidence`, and `reason`. It checks the returned label against the supplied labels; an unrecognized label or unparseable response is represented as `Unknown`.
3. **Evaluation (`evaluate.py`):** The evaluation script reads a labelled CSV, classifies each row, calculates overall and per-category accuracy, prints a confusion matrix and incorrect predictions, and saves the enriched data to `results.csv`. It retries temporary rate-limit and service-unavailable errors after a 15-second wait.

### 3.2 Model and configuration

The default model is `openai/gpt-oss-20b`, accessed through Groq. The model can be changed with the `CLASSIFIER_MODEL` environment variable. The API key is read from `GROQ_API_KEY`; it is not included in the source code. The prompt asks for exactly one of the supplied labels and instructs the model to judge what the writer wants or reports rather than relying only on phrasing. Sampling temperature is set to zero.

### 3.3 Test data

`test_data.csv` contains 32 labelled examples, with eight examples for each of four categories. The examples include straightforward cases as well as challenging ones involving mixed intent, indirect bug reports, positive wording alongside a request, and understated praise. The optional `note` field records why selected examples may be ambiguous.

## 4. Evaluation and Results

The checked-in `results.csv` contains the completed evaluation run. Its summary is:

| Measure | Result |
|---|---:|
| Overall accuracy | 31/32 (96.9%) |
| Bug Report accuracy | 7/8 (87.5%) |
| Feature Request accuracy | 8/8 (100%) |
| Praise accuracy | 8/8 (100%) |
| Billing Question accuracy | 8/8 (100%) |

### Confusion matrix

Rows are the expected labels; columns are the predicted labels. Counts are derived from the saved evaluation results.

| Expected \\ Predicted | Bug Report | Feature Request | Praise | Billing Question |
|---|---:|---:|---:|---:|
| Bug Report | 7 | 1 | 0 | 0 |
| Feature Request | 0 | 8 | 0 | 0 |
| Praise | 0 | 0 | 8 | 0 |
| Billing Question | 0 | 0 | 0 | 8 |

### Incorrect classification

The only incorrect prediction was:

| Text | Expected | Predicted | Confidence |
|---|---|---|---:|
| “It would be great if the app didn't freeze when I upload large photos.” | Bug Report | Feature Request | 95% |

The model's recorded reason was: “User requests improved handling of large photo uploads.” The message uses wish-like language, which resembles a feature request, but the reported freezing describes a defect in existing behavior. The expected label is defensible because the text identifies a failure; however, the wording also expresses a desired improvement, so intent is somewhat mixed. This case shows that a high reported confidence does not ensure correctness.

## 5. Discussion

The saved run performed well on this small, balanced test set. Direct bug reports, clear requests, positive feedback, and billing questions were consistently classified. The main weakness observed is overlap between a request for improvement and a report that existing functionality is broken. The model's short reason is useful for review because it exposes which part of the message drove the decision.

The evaluation is limited by its size and scope: there are only 32 examples, all four categories have equal representation, and the data is a hand-built example set rather than a sample of real incoming messages. Results may differ for other labels, writing styles, languages, or model versions. Confidence is generated by the LLM and has not been calibrated against empirical probabilities. Also, the checked-in results describe one recorded run; the external model may produce different results in a later run.

## 6. Conclusion and Future Improvements

The project delivers an end-to-end text classification helper with editable labels, single and batch workflows, explanatory outputs, and a repeatable evaluation script. On the supplied 32-example result set it achieved 96.9% accuracy, with one error caused by ambiguous bug-versus-feature wording.

Useful next steps include adding concise definitions and examples for each label to the prompt, expanding the test set with realistic and ambiguous examples, and introducing a review path for low-confidence or mixed-intent messages. Evaluation across repeated runs and additional datasets would provide a stronger estimate of reliability. A multi-label or “Other / Needs review” option could also better represent messages that genuinely contain more than one intent.

## 7. Running the Project

Install dependencies, set the Groq API key, and start the interface:

```bash
python3 -m pip install -r requirements.txt
export GROQ_API_KEY="your-key"
streamlit run app.py
```

To evaluate the supplied examples:

```bash
python evaluate.py
```

The evaluation accepts an optional input CSV path, for example `python evaluate.py another_test_set.csv`. It writes its predictions to `results.csv` in the current working directory.

## Appendix: Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application interface |
| `classifier.py` | Groq API call, prompt, and response validation |
| `evaluate.py` | Batch evaluation, metrics, and error reporting |
| `test_data.csv` | Labelled test examples |
| `results.csv` | Saved predictions and evaluation details |
| `requirements.txt` | Python dependencies |
| `README.md` | Setup and usage notes |
