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

COLORS = {
    "Positive": "#16a34a",
    "Negative": "#dc2626",
    "Neutral": "#64748b",
}

EXAMPLES = {
    "Positive service review": "الخدمة ممتازة والتجربة كانت رائعة جدا",
    "Negative product review": "لم تعجبني جودة المنتج هذه المرة وكان هناك تأخير",
    "Neutral update": "تم تحديث المنصة خلال نهاية الأسبوع",
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

st.markdown(
    """
    <style>
      .block-container { max-width: 1160px; padding-top: 2rem; }
      textarea { direction: rtl; text-align: right; }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def infer_column(columns, candidates, fallback):
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return fallback


def parse_date(value, fallback_date):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return fallback_date
    return parsed.date()


def build_dataset(uploaded_file):
    if uploaded_file is None:
        return sample_dataframe(), "Sample social posts"

    source = pd.read_csv(uploaded_file)
    if source.empty:
        st.warning("The uploaded CSV is empty. Showing sample posts instead.")
        return sample_dataframe(), "Sample social posts"

    text_column = infer_column(
        source.columns,
        ["post", "text", "tweet", "content", "comment", "message"],
        source.columns[0],
    )
    date_column = infer_column(source.columns, ["date", "created_at", "day"], None)

    rows = []
    for idx, text in enumerate(source[text_column].fillna("").astype(str)):
        if not text.strip():
            continue
        sentiment, confidence = predict(text)
        fallback_date = date.today() - timedelta(days=max(0, len(source) - idx - 1))
        rows.append(
            {
                "date": parse_date(source[date_column].iloc[idx], fallback_date)
                if date_column
                else fallback_date,
                "post": text,
                "sentiment": LABELS[sentiment],
                "confidence": round(confidence, 2),
            }
        )

    if not rows:
        st.warning("No usable text rows were found. Showing sample posts instead.")
        return sample_dataframe(), "Sample social posts"

    return pd.DataFrame(rows), uploaded_file.name


def sentiment_summary(df):
    return (
        df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
        .sort_values("sentiment")
    )


def make_distribution_chart(df):
    counts = sentiment_summary(df)
    fig = px.bar(
        counts,
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map=COLORS,
        text="count",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="",
        yaxis_title="Posts",
        showlegend=False,
    )
    return fig


def make_trend_chart(df):
    trend = df.groupby(["date", "sentiment"]).size().reset_index(name="posts")
    fig = px.line(
        trend,
        x="date",
        y="posts",
        color="sentiment",
        markers=True,
        color_discrete_map=COLORS,
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="",
        yaxis_title="Posts",
        legend_title_text="",
    )
    return fig


def sentiment_badge(sentiment):
    if sentiment == "Positive":
        return "Positive", "The text sounds favorable or satisfied."
    if sentiment == "Negative":
        return "Negative", "The text sounds dissatisfied or critical."
    return "Neutral", "The text sounds informational or mixed."


with st.sidebar:
    st.header("Setup")
    configured = bool(os.getenv("HF_MODEL_ID", "").strip())
    if configured:
        st.success("HuggingFace model connected")
    else:
        st.info("Demo mode: keyword fallback")
    st.caption("Set `HF_MODEL_ID` in Streamlit secrets to use the fine-tuned AraBERT model.")

    uploaded_file = st.file_uploader("Upload Arabic posts CSV", type=["csv"])
    st.caption("Supported text columns: post, text, tweet, content, comment, message.")

    st.divider()
    st.markdown("**Workflow**")
    st.write("1. Enter Arabic text")
    st.write("2. Run sentiment analysis")
    st.write("3. Review trends and export results")

df, dataset_name = build_dataset(uploaded_file)

st.title("Arabic Sentiment Analyzer")
st.caption("AraBERT-style sentiment classification for Arabic social media text.")

intro_left, intro_right = st.columns([0.68, 0.32], gap="large")
with intro_left:
    st.write(
        "Analyze Arabic posts one by one, or upload a CSV to generate a dashboard "
        "with sentiment counts, confidence scores, and trends over time."
    )
with intro_right:
    st.info(f"Current dataset: {dataset_name}")

st.divider()

positive_count = int((df["sentiment"] == "Positive").sum())
neutral_count = int((df["sentiment"] == "Neutral").sum())
negative_count = int((df["sentiment"] == "Negative").sum())
average_confidence = df["confidence"].mean()

metric_cols = st.columns(5)
metric_cols[0].metric("Posts", f"{len(df):,}")
metric_cols[1].metric("Positive", positive_count)
metric_cols[2].metric("Neutral", neutral_count)
metric_cols[3].metric("Negative", negative_count)
metric_cols[4].metric("Avg confidence", f"{average_confidence:.0%}")

analyze_tab, dashboard_tab, data_tab = st.tabs(
    ["Analyze", "Dashboard", "Data"]
)

with analyze_tab:
    input_col, result_col = st.columns([0.58, 0.42], gap="large")

    with input_col:
        with st.container(border=True):
            st.subheader("Try a sentence")
            selected_example = st.selectbox("Sample text", list(EXAMPLES.keys()))
            default_text = EXAMPLES[selected_example]
            text = st.text_area("Arabic text", value=default_text, height=160)
            analyze_clicked = st.button("Analyze sentiment", type="primary", use_container_width=True)

    if analyze_clicked or "last_prediction" not in st.session_state:
        st.session_state["last_text"] = text
        st.session_state["last_prediction"] = predict(text)

    sentiment, confidence = st.session_state["last_prediction"]
    sentiment_label, explanation = sentiment_badge(LABELS[sentiment])

    with result_col:
        with st.container(border=True):
            st.subheader("Result")
            if sentiment_label == "Positive":
                st.success(f"{sentiment_label} sentiment")
            elif sentiment_label == "Negative":
                st.error(f"{sentiment_label} sentiment")
            else:
                st.warning(f"{sentiment_label} sentiment")

            st.metric("Confidence", f"{confidence:.0%}")
            st.write(explanation)
            st.text_area(
                "Analyzed text",
                value=st.session_state["last_text"],
                height=115,
                disabled=True,
            )

with dashboard_tab:
    chart_col, trend_col = st.columns(2, gap="large")
    with chart_col:
        with st.container(border=True):
            st.subheader("Sentiment distribution")
            st.plotly_chart(make_distribution_chart(df), use_container_width=True)
    with trend_col:
        with st.container(border=True):
            st.subheader("Trend over time")
            st.plotly_chart(make_trend_chart(df), use_container_width=True)

with data_tab:
    with st.container(border=True):
        st.subheader("Analyzed posts")
        st.dataframe(
            df.sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name="arabic_sentiment_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
