"""Streamlit UI: streamlit run app.py"""
import pandas as pd
import streamlit as st

from classifier import classify

st.set_page_config(page_title="AI Text Classification Helper", page_icon="🏷️")
st.title("🏷️ AI Text Classification Helper")
st.caption("Define your own labels, then let an LLM sort short texts into them.")

labels_text = st.text_input(
    "Labels (comma-separated)",
    "Bug Report, Feature Request, Praise, Billing Question",
)
labels = [l.strip() for l in labels_text.split(",") if l.strip()]
if len(labels) < 2:
    st.warning("Enter at least two labels.")
    st.stop()

single, batch = st.tabs(["Single text", "Batch"])

with single:
    text = st.text_area("Text to classify", height=100)
    if st.button("Classify", type="primary") and text.strip():
        with st.spinner("Asking the model..."):
            r = classify(text, labels)
        st.success(f"**{r['label']}**  (confidence {r['confidence']:.0%})")
        st.write(r["reason"])

with batch:
    st.write("One text per line, or upload a CSV with a `text` column.")
    lines = st.text_area("Texts", height=160, key="batch")
    up = st.file_uploader("CSV", type="csv")
    texts = [t for t in lines.splitlines() if t.strip()]
    if up is not None:
        texts = pd.read_csv(up)["text"].dropna().astype(str).tolist()
    if st.button("Classify all") and texts:
        rows = []
        bar = st.progress(0.0)
        for i, t in enumerate(texts, 1):
            rows.append({"text": t, **classify(t, labels)})
            bar.progress(i / len(texts))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "classified.csv")
