import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR

for candidate in (BASE_DIR, BASE_DIR.parent):
    if (candidate / "models").exists():
        PROJECT_ROOT = candidate
        break

MODEL_PATH = PROJECT_ROOT / "models"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "model"

HISTORY_FILE = PROJECT_ROOT / "predictions.csv"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def save_prediction(text: str, label: str, confidence: float) -> None:
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "text_preview": text[:120].replace("\n", " "),
        "prediction": label,
        "confidence": confidence,
    }

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([new_entry])

    if HISTORY_FILE.exists():
        existing_df = pd.read_csv(HISTORY_FILE)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(HISTORY_FILE, index=False)
    else:
        new_df.to_csv(HISTORY_FILE, index=False)


def predict_news(text: str):
    if not text.strip():
        return "Please enter some text", 0.0

    tokenizer, model, device = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence = float(torch.max(probs).item())
    prediction = int(torch.argmax(probs).item())
    label = "REAL NEWS" if prediction == 0 else "FAKE NEWS"

    save_prediction(text, label, confidence)
    return label, confidence * 100


def get_dashboard_data():
    if not HISTORY_FILE.exists():
        return None, None, None

    df = pd.read_csv(HISTORY_FILE)
    total = len(df)
    fake_count = int((df["prediction"] == "FAKE NEWS").sum())
    real_count = int((df["prediction"] == "REAL NEWS").sum())

    if total > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = [fake_count, real_count]
        labels = ["Fake News", "Real News"]
        colors = ["#FF4D4D", "#2ECC71"]

        ax.pie(
            counts,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            textprops={"color": "black"},
        )
        ax.set_title("Prediction Distribution")
        plt.tight_layout()
    else:
        fig = None

    summary = {
        "Total Predictions": total,
        "Fake News": fake_count,
        "Real News": real_count,
    }
    return summary, df, fig


st.set_page_config(page_title="Fake News Detection Dashboard", layout="wide")

if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

if not st.session_state.get("admin_authenticated", False):
    st.title("🔐 Admin Access")
    st.caption("Secure sign-in required to open the fake news detection prediction and analytics workspace.")

    with st.form("admin_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter Dashboard", use_container_width=True)

    if submitted:
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state["admin_authenticated"] = True
            st.success("Authentication successful. Opening dashboard...")
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")
else:
    col_title, col_logout = st.columns([8, 2])
    with col_title:
        st.title("📰 Fake News Detection System")
        st.caption("Professional Streamlit interface for model prediction and analytics review.")
    with col_logout:
        if st.button("Logout", use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.rerun()

    tab1, tab2 = st.tabs(["Prediction", "Dashboard"])

    with tab1:
        st.subheader("Analyze a news article")
        news_text = st.text_area(
            "Enter news text",
            height=220,
            placeholder="Paste article text or claim here...",
        )

        if st.button("Analyze News", width="stretch"):
            if news_text.strip():
                label, confidence = predict_news(news_text)
                st.success(f"Prediction: {label}")
                st.metric("Confidence", f"{confidence:.2f}%")
            else:
                st.warning("Please enter some text before analyzing.")

    with tab2:
        st.subheader("Admin Dashboard")
        summary, df, fig = get_dashboard_data()

        if summary is None:
            st.info("No prediction history yet. Run a prediction from the Prediction tab to populate the dashboard.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", summary["Total Predictions"])
            col2.metric("Fake News", summary["Fake News"])
            col3.metric("Real News", summary["Real News"])

            if fig is not None:
                st.pyplot(fig)

            st.subheader("Prediction history")
            st.dataframe(df, width="stretch")
