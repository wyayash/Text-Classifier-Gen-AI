"""Run the test set through the classifier and report accuracy + misclassifications.

Usage: python evaluate.py [test_data.csv]
"""
import sys
import time

import pandas as pd

from classifier import classify

path = sys.argv[1] if len(sys.argv) > 1 else "test_data.csv"
df = pd.read_csv(path)
labels = sorted(df["label"].unique())

def classify_with_retry(text):
    while True:
        try:
            return classify(text, labels)
        except Exception as error:
            message = str(error).lower()
            retryable = (
                "429" in message
                or "resource_exhausted" in message
                or "rate limit" in message
                or "503" in message
                or "unavailable" in message
            )
            if not retryable:
                raise
            print("The API is temporarily unavailable or rate-limited; waiting 15 seconds before retrying...")
            time.sleep(15)


results = []
for i, text in enumerate(df["text"], 1):
    print(f"Classifying example {i}/{len(df)}...", flush=True)
    results.append(classify_with_retry(text))

df["predicted"] = [r["label"] for r in results]
df["confidence"] = [r["confidence"] for r in results]
df["reason"] = [r["reason"] for r in results]
df["correct"] = df["label"] == df["predicted"]
df.to_csv("results.csv", index=False)

print(f"\nOverall accuracy: {df['correct'].mean():.1%} ({df['correct'].sum()}/{len(df)})\n")
print("Per-label accuracy:")
print(df.groupby("label")["correct"].mean().map("{:.0%}".format).to_string(), "\n")
print("Confusion matrix (rows = actual, cols = predicted):")
print(pd.crosstab(df["label"], df["predicted"]), "\n")

wrong = df[~df["correct"]]
print(f"Incorrect classifications ({len(wrong)}):")
for _, r in wrong.iterrows():
    print(f"\n- \"{r['text']}\"")
    print(f"  actual: {r['label']} | predicted: {r['predicted']} ({r['confidence']:.0%})")
    print(f"  model's reason: {r['reason']}")
    if isinstance(r.get("note"), str):
        print(f"  why it's tricky: {r['note']}")
print("\nFull results saved to results.csv")
