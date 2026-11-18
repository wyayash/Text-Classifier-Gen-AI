"""Core LLM classification logic."""
import json
import os

from groq import Groq

MODEL = os.getenv("CLASSIFIER_MODEL", "openai/gpt-oss-20b")
_client = None

PROMPT = """You are a text classifier. Assign the text below to exactly one of these labels:
{labels}

Rules:
- Choose the single best label, even if the text is ambiguous.
- Judge what the writer actually wants or reports, not just the wording.

Reply with ONLY a JSON object:
{{"label": "<one label from the list>", "confidence": <number 0 to 1>, "reason": "<one short sentence>"}}

Text: {text}"""


def _client_instance():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it in this terminal and try again."
            )
        _client = Groq(api_key=api_key)
    return _client


def parse_response(raw: str, labels: list[str]) -> dict:
    """Extract and validate the JSON reply. Falls back to 'Unknown' on bad output."""
    try:
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        return {"label": "Unknown", "confidence": 0.0, "reason": f"Unparseable reply: {raw[:80]}"}
    by_lower = {l.lower(): l for l in labels}
    label = by_lower.get(str(data.get("label", "")).strip().lower(), "Unknown")
    return {
        "label": label,
        "confidence": float(data.get("confidence", 0) or 0),
        "reason": str(data.get("reason", "")),
    }


def classify(text: str, labels: list[str]) -> dict:
    """Classify one short text into one of `labels`."""
    response = _client_instance().chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT.format(labels="\n".join(f"- {l}" for l in labels), text=text),
        }],
        temperature=0,
        max_tokens=200,
    )
    return parse_response(response.choices[0].message.content or "", labels)
