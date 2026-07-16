#!/usr/bin/env python
import sys
import traceback

print("Starting import test...", flush=True)

try:
    print("[1/6] Importing torch...", flush=True)
    import torch
    print(f"  ✓ PyTorch loaded (device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')})", flush=True)
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    print("[2/6] Importing transformers...", flush=True)
    from transformers import BertTokenizerFast, BertForSequenceClassification
    print("  ✓ Transformers loaded", flush=True)
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    print("[3/6] Importing gradio...", flush=True)
    import gradio
    print(f"  ✓ Gradio loaded (version: {gradio.__version__})", flush=True)
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    print("[4/6] Importing pandas...", flush=True)
    import pandas
    print("  ✓ Pandas loaded", flush=True)
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    print("[5/6] Importing matplotlib...", flush=True)
    import matplotlib
    print("  ✓ Matplotlib loaded", flush=True)
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    print("[6/6] Loading model from disk...", flush=True)
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "model")
    print(f"  Model path: {MODEL_PATH}", flush=True)
    print(f"  Exists: {os.path.exists(MODEL_PATH)}", flush=True)
    
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    print("  ✓ Tokenizer loaded", flush=True)
    
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    print("  ✓ Model loaded", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  ✓ Model on device: {device}", flush=True)
    
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All imports and model loading successful!", flush=True)
