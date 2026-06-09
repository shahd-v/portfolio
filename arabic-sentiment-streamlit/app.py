import os
from html import escape
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
    "Positive": "#22c55e",
    "Negative": "#ef4444",
    "Neutral": "#94a3b8",
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
      .stApp {
        background: #f8fafc;
        color: #111827;
      }

      [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
      }

      [data-testid="stSidebar"],
      [data-testid="stSidebar"] *:not(button):not(svg):not(path) {
        color: #111827 !important;
      }

      .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
      }

      h1, h2, h3, p, label,
      [data-testid="stMarkdownContainer"],
      [data-testid="stCaptionContainer"] {
        color: #111827 !important;
      }

      .hero {
        background: linear-gradient(135deg, #111827 0%, #1e293b 62%, #0f766e 100%);
        border-radius: 18px;
        padding: 2.2rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 24px 50px rgba(15, 23, 42, 0.16);
      }

      .hero-kicker {
        color: #99f6e4;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
      }

      .hero-title {
        font-size: clamp(2rem, 4vw, 3.4rem);
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.8rem;
      }

      .hero-copy {
        color: #cbd5e1 !important;
        max-width: 720px;
        font-size: 1.03rem;
        line-height: 1.7;
      }

      .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
      }

      .result-card {
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid #ccfbf1;
        background: #f0fdfa;
        margin-top: 1rem;
      }

      .result-label {
        color: #0f766e;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }

      .result-value {
        color: #111827;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.2rem;
      }

      .arabic-box {
        direction: rtl;
        text-align: right;
        color: #111827;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        line-height: 1.9;
        margin-top: 0.7rem;
      }

      .small-muted {
        color: #64748b !important;
        font-size: 0.9rem;
        line-height: 1.6;
      }

      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
      }

      [data-testid="stMetricLabel"],
      [data-testid="stMetricValue"],
      [data-testid="stMetricDelta"] {
        color: #111827 !important;
      }

      textarea,
      input,
      [data-baseweb="textarea"] textarea {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: #d1d5db !important;
      }

      [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 1px dashed #cbd5e1 !important;
      }

      [data-testid="stFileUploader"] section *,
      [data-testid="stFileUploader"] small {
        color: #111827 !important;
      }

      [data-testid="stTabs"] button {
        color: #334155 !important;
      }

      [data-testid="stTabs"] button[aria-selected="true"] {
        color: #0f766e !important;
      }

      .stButton > button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
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
        rows.append(
            {
                "date": (
                    pd.to_datetime(source[date_column].iloc[idx], errors="coerce").date()
                    if date_column
                    and pd.notna(
                        pd.to_datetime(source[date_column].iloc[idx], errors="coerce")
                    )
                    else date.today() - timedelta(days=max(0, len(source) - idx - 1))
                ),
                "post": text,
                "sentiment": LABELS[sentiment],
                "confidence": round(confidence, 2),
            }
        )

    if not rows:
        st.warning("No usable text rows were found. Showing sample posts instead.")
        return sample_dataframe(), "Sample social posts"

    return pd.DataFrame(rows), uploaded_file.name


def make_pie_chart(df):
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map=COLORS,
        hole=0.58,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Date",
        yaxis_title="Posts",
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


with st.sidebar:
    st.subheader("Arabic Sentiment Analyzer")
    configured = bool(os.getenv("HF_MODEL_ID", "").strip())
    st.metric("HuggingFace model", "Configured" if configured else "Fallback demo")
    st.caption("Set `HF_MODEL_ID` to connect the fine-tuned AraBERT classifier.")
    st.divider()
    uploaded_file = st.file_uploader("Upload Arabic posts CSV", type=["csv"])
    st.caption("CSV can include columns like `post`, `text`, `tweet`, and `date`.")

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">AraBERT + HuggingFace + Plotly</div>
      <div class="hero-title">Arabic Sentiment Analyzer</div>
      <div class="hero-copy">
        Analyze Arabic social media posts, classify sentiment, and explore trend
        dashboards in a Streamlit interface styled like a focused AI demo.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

df, dataset_name = build_dataset(uploaded_file)

positive_count = int((df["sentiment"] == "Positive").sum())
neutral_count = int((df["sentiment"] == "Neutral").sum())
negative_count = int((df["sentiment"] == "Negative").sum())
average_confidence = df["confidence"].mean()

st.caption(f"Dataset: {dataset_name}")
metric_cols = st.columns(5)
metric_cols[0].metric("Posts analyzed", f"{len(df):,}")
metric_cols[1].metric("Positive", positive_count)
metric_cols[2].metric("Neutral", neutral_count)
metric_cols[3].metric("Negative", negative_count)
metric_cols[4].metric("Avg confidence", f"{average_confidence:.0%}")

analyze_tab, dashboard_tab, data_tab = st.tabs(
    ["Analyze text", "Dashboard", "Analyzed data"]
)

with analyze_tab:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("### Try an Arabic sentence")
        text = st.text_area(
            "Arabic text",
            value="الخدمة ممتازة والتجربة كانت رائعة جدا",
            height=150,
            label_visibility="collapsed",
        )
        analyze_clicked = st.button(
            "Analyze sentiment",
            type="primary",
            use_container_width=True,
        )
        if analyze_clicked or "last_prediction" not in st.session_state:
            st.session_state["last_prediction"] = predict(text)
            st.session_state["last_text"] = text

    with right:
        sentiment, confidence = st.session_state["last_prediction"]
        sentiment_label = LABELS[sentiment]
        displayed_text = escape(st.session_state.get("last_text", text))
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Prediction</div>
              <div class="result-value">{sentiment_label}</div>
              <div class="small-muted">{confidence:.0%} confidence from the active classifier.</div>
              <div class="arabic-box">{displayed_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "The live classifier uses the HuggingFace model when `HF_MODEL_ID` is set. "
            "Without it, the app uses a transparent keyword fallback for demo review."
        )

with dashboard_tab:
    st.markdown("### Sentiment overview")
    chart_left, chart_right = st.columns([0.9, 1.1], gap="large")
    with chart_left:
        st.plotly_chart(make_pie_chart(df), use_container_width=True)
    with chart_right:
        st.plotly_chart(make_trend_chart(df), use_container_width=True)

with data_tab:
    st.markdown("### Analyzed posts")
    st.dataframe(
        df.sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download analyzed CSV",
        data=csv,
        file_name="arabic_sentiment_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
