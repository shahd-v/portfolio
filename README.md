# Portfolio

Source for my personal portfolio site, plus the Arabic Sentiment Analyzer app that is embedded as a live demo on it.

**Live site:** [shahdfaisal-portfolio.vercel.app](https://shahdfaisal-portfolio.vercel.app/)

## Contents

| Path | What it is |
|---|---|
| `index.html` | The portfolio site, single page, no build step |
| `arabic-sentiment-streamlit/` | Arabic Sentiment Analyzer (Streamlit app) |
| `.streamlit/` | Streamlit theme and config |
| `requirements.txt` | Python dependencies for the Streamlit app |

## Arabic Sentiment Analyzer

A sentiment classifier for Arabic social media text, with a dashboard for batch analysis.

**▶️ Live demo:** [portfolio-arabic-sentiment-analyzer-sfs.streamlit.app](https://portfolio-arabic-sentiment-analyzer-sfs.streamlit.app/)

- **Single-text mode** - paste Arabic text and get a sentiment label with a confidence score
- **Batch mode** - upload a CSV and classify every row at once; column names are detected automatically (`post`, `text`, `tweet` and similar all work)
- **Dashboard** - sentiment distribution and trend-over-time charts built with Plotly
- **Export** - download the classified results back out as CSV
- **Graceful degradation** - if no model is configured it falls back to a keyword-based classifier, so the app still runs

Model: AraBERT via HuggingFace Transformers, loaded from an environment variable.

### Running the analyzer locally

```bash
pip install -r requirements.txt
streamlit run arabic-sentiment-streamlit/app.py
```

## Stack

HTML/CSS, Streamlit, Transformers (AraBERT), Pandas, Plotly, Vercel

---

Built by [Shahd Faisal](https://shahdfaisal-portfolio.vercel.app/)
