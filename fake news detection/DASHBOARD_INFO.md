# 📰 Fake News Detection System - Dashboard Overview

## Project Architecture

This is a **Gradio-based web application** for fake news detection using a fine-tuned BERT transformer model.

---

## Dashboard Components

### 1. **User Panel Tab** 
Users can:
- Input news article text in a text area (10 lines)
- Click "Analyze News" button
- Get results showing:
  - **Prediction Result**: "REAL NEWS" or "FAKE NEWS"
  - **Confidence Level**: Percentage confidence (0-100%)

**Model Details:**
- Uses BERT tokenizer and fine-tuned BERT classification model
- Maximum sequence length: 256 tokens
- Softmax probability calculation for confidence
- Runs on GPU if available, otherwise CPU

---

### 2. **Admin Dashboard Tab**
Shows analytics with:
- **Refresh Button**: "Load / Refresh Analytics Dashboard"
- **System Statistics Summary**:
  - Total Predictions (count)
  - Fake News Count
  - Real News Count
- **Distribution Chart**: Pie chart showing ratio of Fake vs Real news
  - Color scheme: Red (#FF4D4D) for Fake, Green (#2ECC71) for Real
  - Shows percentages
- **Prediction History Logs**: Table displaying:
  - Timestamp
  - Text preview (first 120 characters)
  - Prediction (REAL/FAKE)
  - Confidence score

---

## Model Information
- **Framework**: HuggingFace Transformers (BERT)
- **Model Type**: BertForSequenceClassification
- **Location**: `./model/` directory
- **Model Files**: 
  - `model.safetensors` (437 MB)
  - `tokenizer.json`
  - `config.json`
  - `tokenizer_config.json`

---

## Data Storage
- **Predictions CSV**: `predictions.csv`
  - Tracks all predictions with timestamps
  - Used for dashboard analytics

---

## UI Theme
- Framework: Gradio with Soft theme
- Professional, clean interface
- Responsive layout with tabbed navigation

---

## Server Details
- **Host**: 127.0.0.1 (localhost)
- **Port**: 7860
- **URL**: http://127.0.0.1:7860

