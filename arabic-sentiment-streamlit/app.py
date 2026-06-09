import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Arabic Sentiment Analyzer",
    page_icon="AR",
    layout="wide",
)

LABELS = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}

SAMPLE_POSTS = [
    ("الخدمة ممتازة والتجربة كانت رائعة جدا", "positive"),
    ("التطبيق بطيء جدا ويحتاج تحسين", "negative"),
    ("وصلني الطلب اليوم ولم أستخدمه بعد", "neutral"),
    ("الدعم الفني متعاون وسريع في الرد", "positive"),
    ("لم تعجبني جودة المنتج هذه المرة", "negative"),
    ("تم تحديث المنصة خلال نهاية الأسبوع", "neutral"),
    ("واجهة الاستخدام أصبحت أسهل بكثير", "positive"),
    ("هناك مشكلة متكررة في تسجيل الدخول", "negative"),
    ("سأجرب الميزة الجديدة لاحقا", "neutral"),
]


@st.cache_resource(show_spinner=False)
def load_pipeline():
    model_id = os.getenv("HF_MODEL_ID", "").strip()
    if not model_id:
        return None

    from transformers import pipeline

    return pipeline("text-classification", model=model_id, tokenizer=model_id)


def fallback_predict(text):
    positive_terms = ["ممتاز", "رائع", "جميل", "متعاون", "سريع", "أسهل", "احب", "جيد"]
    negative_terms = ["بطيء", "مشكلة", "سيء", "لم تعجبني", "ضعيف", "صعب", "تأخير"]

    score = 0
    for term in positive_terms:
        if term in text:
            score += 1
    for term in negative_terms:
        if term in text:
            score -= 1

    if score > 0:
        return "positive", min(0.98, 0.72 + score * 0.08)
    if score < 0:
        return "negative", min(0.98, 0.72 + abs(score) * 0.08)
    return "neutral", 0.68


def predict(text):
    pipe = load_pipeline()
    if pipe is None:
        return fallback_predict(text)

    result = pipe(text, truncation=True)[0]
    label = result["label"].lower()
    if "pos" in label or "positive" in label:
        normalized = "positive"
    elif "neg" in label or "negative" in label:
        normalized = "negative"
    else:
        normalized = "neutral"
    return normalized, float(result["score"])


def sample_dataframe():
    start = date.today() - timedelta(days=8)
    rows = []
    for idx, (text, label) in enumerate(SAMPLE_POSTS):
        rows.append(
            {
                "date": start + timedelta(days=idx),
                "post": text,
                "sentiment": LABELS[label],
                "confidence": round(0.74 + (idx % 4) * 0.06, 2),
            }
        )
    return pd.DataFrame(rows)


st.title("Arabic Sentiment Analyzer")
st.caption("AraBERT sentiment classification for Arabic social media text")

with st.sidebar:
    st.header("Model")
    configured = bool(os.getenv("HF_MODEL_ID", "").strip())
    st.metric("HuggingFace model", "Configured" if configured else "Fallback demo")
    st.write("Set `HF_MODEL_ID` to connect the fine-tuned AraBERT classifier.")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Try a sentence")
    text = st.text_area(
        "Arabic text",
        value="الخدمة ممتازة والتجربة كانت رائعة جدا",
        height=120,
        label_visibility="collapsed",
    )
    if st.button("Analyze sentiment", type="primary", use_container_width=True):
        sentiment, confidence = predict(text)
        st.session_state["last_prediction"] = (sentiment, confidence)

    sentiment, confidence = st.session_state.get(
        "last_prediction", predict("الخدمة ممتازة والتجربة كانت رائعة جدا")
    )
    st.metric("Prediction", LABELS[sentiment], f"{confidence:.0%} confidence")

with right:
    st.subheader("Class distribution")
    df = sample_dataframe()
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.donut(
        counts,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map={
            "Positive": "#2fbf71",
            "Negative": "#f05d5e",
            "Neutral": "#9aa0a6",
        },
        hole=0.55,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Sentiment trends over time")
trend = df.groupby(["date", "sentiment"]).size().reset_index(name="posts")
trend_fig = px.line(
    trend,
    x="date",
    y="posts",
    color="sentiment",
    markers=True,
    color_discrete_map={
        "Positive": "#2fbf71",
        "Negative": "#f05d5e",
        "Neutral": "#9aa0a6",
    },
)
trend_fig.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Date",
    yaxis_title="Posts",
    legend_title_text="",
)
st.plotly_chart(trend_fig, use_container_width=True)

st.subheader("Sample analyzed posts")
st.dataframe(df, use_container_width=True, hide_index=True)

