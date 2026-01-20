# Pharma Voice Orders - Deployment Guide

This guide explains how to deploy and configure the application for production use with large AI models.

---

## 🚀 Deployment Options Comparison

| Feature | Streamlit Cloud (Deploy Button) | Hugging Face Spaces |
|---------|--------------------------------|---------------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ One-click | ⭐⭐⭐⭐ Simple |
| **Free Tier** | 1GB RAM, limited | 16GB RAM (with GPU upgrade) |
| **GPU Support** | ❌ No | ✅ Yes (paid: T4, A10G) |
| **Large Models (Whisper Medium+)** | ⚠️ May timeout | ✅ Works well |
| **Privacy/Secrets** | ✅ Secrets Manager | ✅ Secrets Manager |
| **Best For** | Quick demos (tiny model) | Production + Large Models |

---

## 📱 Option 1: Streamlit Cloud (The "Deploy" Button)

The **Deploy** button in your localhost Streamlit UI deploys directly to **Streamlit Community Cloud**.

### How It Works:
1. Click **Deploy** → **Streamlit Community Cloud**
2. Connect your GitHub account
3. Select your repository and branch
4. Streamlit Cloud builds and hosts your app

### ⚠️ Limitations for Your Use Case:
- **1GB RAM limit** on free tier → Whisper Medium (3GB) will **fail**
- **No GPU** → Slow inference
- **Good for**: Demo with `whisper-tiny` only

### Setup:
```bash
# Push your code to GitHub first
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```
Then click **Deploy** in the Streamlit UI.

---

## ☁️ Option 2: Hugging Face Spaces (Recommended)

**Best for**: Large models (Whisper Medium, Large, Google Med SR) with HF Token.

### Step-by-Step Deployment:

#### 1. Create a Hugging Face Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Select:
   - **SDK**: Streamlit
   - **Hardware**: CPU Basic (free) or upgrade for GPU
   - **Visibility**: Public or Private

#### 2. Create `app.py` (Already Done ✅)

#### 3. Create `requirements.txt` for HF Spaces
Create a file **specifically for Spaces** (different from local):

```txt
streamlit
pandas
openpyxl
torch
transformers
librosa
noisereduce
soundfile
rapidfuzz
jiwer
regex
webrtcvad
numpy<2
huggingface_hub
```

#### 4. Add Your HF Token as a Secret
1. Go to your Space → **Settings** → **Repository secrets**
2. Add a new secret:
   - **Name**: `HF_TOKEN`
   - **Value**: Your Hugging Face read token (from [hf.co/settings/tokens](https://huggingface.co/settings/tokens))

#### 5. Update Code to Use Token
In `core/asr_engine.py`, the model will automatically use `HF_TOKEN`:

```python
import os
from huggingface_hub import login

# Auto-login with Space secret
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
```

#### 6. Push Code to the Space
```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/pharma-voice-orders
cd pharma-voice-orders

# Copy your files
cp -r /path/to/your/local/project/* .

# Push
git add .
git commit -m "Initial deployment"
git push
```

---

## 🔑 Using Gated Models (Google Med SR, etc.)

Some models require you to accept terms on the model page before using.

### Steps:
1. Visit the model page (e.g., `google/med-sr-model`)
2. Click **Agree and access model**
3. Add your `HF_TOKEN` to the Space secrets (as shown above)
4. Update your code to specify the model ID:

```python
# In core/asr_engine.py
model_id = "google/med-speech-recognition"  # Example
```

---

## 🎯 Recommended Strategy for Your Project

| Phase | Platform | Model | Why |
|-------|----------|-------|-----|
| **Development** | Local (`uv run start`) | `whisper-tiny` | Fast iteration |
| **University Demo** | Hugging Face Spaces (Free CPU) | `whisper-small` | Balance of quality + speed |
| **Production Demo** | HF Spaces + GPU (T4) | `whisper-medium` or Google Med SR | Best quality |

---

## 🔄 Pre-Caching Models (Avoid First-Run Download)

To make the model load instantly for visitors, add a **pre-download script** in your Space:

Create `preload.py`:
```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Pre-download during build
model_id = "openai/whisper-medium"
AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
AutoProcessor.from_pretrained(model_id)
print("Model pre-cached!")
```

Then add to your Space's `README.md`:
```yaml
---
title: Pharma Voice Orders
sdk: streamlit
sdk_version: 1.53.0
app_file: app.py
pinned: false
preload: preload.py
---
```

---

## 📁 Final File Structure for HF Spaces

```
pharma-voice-orders/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── preload.py               # Model pre-download script
├── README.md                # Space metadata (YAML frontmatter)
├── core/                    # Your modules
├── simulation/
├── evaluation/
└── data/
```

---

*Last Updated: January 2026*
