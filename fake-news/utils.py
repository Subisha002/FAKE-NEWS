from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models"
HISTORY_FILE = BASE_DIR / "predictions.csv"


@torch.inference_mode()
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

    fig = None
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

    summary = {
        "Total Predictions": total,
        "Fake News": fake_count,
        "Real News": real_count,
    }
    return summary, df, fig
