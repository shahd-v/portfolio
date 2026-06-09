# Arabic Sentiment Analyzer

Streamlit dashboard for Arabic social media sentiment analysis.

The app is designed for a fine-tuned AraBERT model hosted on HuggingFace. If a
model ID is not configured, it uses a lightweight local fallback so the dashboard
still works for portfolio review.

## Run locally

```bash
cd arabic-sentiment-streamlit
pip install -r requirements.txt
streamlit run app.py
```

## Configure HuggingFace

Set the model repository ID before running or deploying:

```bash
export HF_MODEL_ID="your-org/your-arabert-sentiment-model"
```

On Streamlit Community Cloud, add `HF_MODEL_ID` in app secrets or environment
variables, then deploy `arabic-sentiment-streamlit/app.py`.

