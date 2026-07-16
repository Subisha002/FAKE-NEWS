import gradio as gr
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from transformers import BertTokenizerFast, BertForSequenceClassification

# -------------------------
# LOAD MODEL
# -------------------------
# Absolute path or relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model")
HISTORY_FILE = os.path.join(SCRIPT_DIR, "predictions.csv")

print(f"Loading model from: {MODEL_PATH}")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_news(text):
    if not text.strip():
        return "Please enter some text", "0.00%"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()
    prediction = torch.argmax(probs).item()

    label = "REAL NEWS" if prediction == 0 else "FAKE NEWS"

    save_prediction(text, label, confidence)

    return label, f"{confidence*100:.2f}%"

# -------------------------
# SAVE HISTORY
# -------------------------
def save_prediction(text, label, confidence):
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "text_preview": text[:120].replace("\n", " "),
        "prediction": label,
        "confidence": confidence
    }

    df = pd.DataFrame([new_entry])

    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)

# -------------------------
# ADMIN DASHBOARD
# -------------------------
def load_dashboard():
    if not os.path.exists(HISTORY_FILE):
        return "No predictions yet.", None, None

    df = pd.read_csv(HISTORY_FILE)

    total = len(df)
    fake_count = len(df[df["prediction"] == "FAKE NEWS"])
    real_count = len(df[df["prediction"] == "REAL NEWS"])

    summary = f"""Total Predictions: {total}
Fake News: {fake_count}
Real News: {real_count}"""

    # Create pie chart
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Theme configuration for high visual quality
    colors = ['#FF4D4D', '#2ECC71'] # Sleek custom red/green instead of generic colors
    labels = ["Fake News", "Real News"]
    counts = [fake_count, real_count]
    
    # Clean zeros to avoid empty slice warning
    if sum(counts) > 0:
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            textprops=dict(color="black")
        )
        # Style layout
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
    else:
        ax.text(0.5, 0.5, "No Data Available", ha='center', va='center')
        
    ax.set_title("Prediction Distribution", fontsize=14, pad=15)
    plt.tight_layout()

    return summary, df, fig

# -------------------------
# PROFESSIONAL UI
# -------------------------
with gr.Blocks(title="Fake News Detection System") as app:

    gr.Markdown("""
    # 📰 Fake News Detection System
    ### AI-powered verification using BERT and Transformers
    """)

    with gr.Tab("User Panel"):
        news_input = gr.Textbox(
            label="Enter News Article Text",
            lines=10,
            placeholder="Paste full article or claim text here to analyze..."
        )

        predict_btn = gr.Button("Analyze News", variant="primary")

        with gr.Row():
            result_label = gr.Textbox(label="Prediction Result", interactive=False)
            confidence_label = gr.Textbox(label="Confidence Level", interactive=False)

        predict_btn.click(
            predict_news,
            inputs=news_input,
            outputs=[result_label, confidence_label]
        )

    with gr.Tab("Admin Dashboard"):
        refresh_btn = gr.Button("Load / Refresh Analytics Dashboard", variant="secondary")
        
        summary_box = gr.Textbox(label="System Statistics Summary", interactive=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                chart_output = gr.Plot(label="Distribution Chart")
            with gr.Column(scale=1):
                history_table = gr.Dataframe(label="Prediction History Logs")

        refresh_btn.click(
            load_dashboard,
            outputs=[summary_box, history_table, chart_output]
        )

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        theme=gr.themes.Soft()
    )
