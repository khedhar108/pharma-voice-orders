---
title: Pharma Voice Orders
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# 🏥 Pharma Voice Orders

> **Accent-Aware Speech-to-Text Engine for Distributor Order Processing**

Transform voice orders into structured data using OpenAI Whisper. Designed for pharmaceutical distributors who need fast, accurate order transcription.

## 🚀 Deployment Options

### Option 1: Local (CPU Mode)
Run on your machine with optimized CPU threading. No GPU required.

```bash
# Install dependencies
pip install uv
uv sync

# Run the app
uv run start
```

**What happens:**
- Models download to `C:\Users\<you>\.cache\huggingface\` (Windows) or `~/.cache/huggingface/` (Linux/Mac)
- Uses CPU with thread limiting (4-6 threads) to prevent crashes
- Best models: `whisper-medium` or `whisper-large-v3-turbo`

---

### Option 2: Google Colab (T4 GPU)
Run on free cloud GPU for faster inference with larger models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khedhar108/pharma-voice-orders/blob/main/notebooks/colab_gpu_launcher.ipynb)

**Or manually:**
1. Open `notebooks/colab_gpu_launcher.ipynb` in Google Colab
2. Set Runtime → Change runtime type → **T4 GPU**
3. Run all cells
4. Click the `loca.lt` URL to access the app

**What happens:**
- Models download to Colab's cloud storage (not your local drive)
- Uses T4 GPU (16GB VRAM) for fast inference (~3 sec/audio)
- Best models: `whisper-large-v3` or `whisper-large-v3-turbo`

---

### Option 3: Local GPU (NVIDIA CUDA)
For users with NVIDIA GPUs (RTX 2050+). Requires one-time PyTorch CUDA setup.

```bash
# Install PyTorch with CUDA (one-time, ~2.5GB download)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Then run normally
uv sync
uv run start
```

---

## 📋 Model Recommendations

| Model | VRAM | Speed | Quality | Best For |
| :--- | :--- | :--- | :--- | :--- |
| `whisper-medium` | 1.5 GB | Fast | Good | CPU / Low VRAM |
| `whisper-large-v3-turbo` | 2 GB | Fast | Great | **RTX 2050 / Colab** |
| `whisper-large-v3` | 3.1 GB | Slower | Best | Colab T4 |

---

## 📂 Project Structure

```
pharma-voice-orders/
├── app.py                 # Main Streamlit app
├── core/
│   ├── asr_engine.py      # Whisper transcription
│   ├── preprocessor.py    # Audio preprocessing
│   ├── entity_extractor.py  # Medicine/dosage extraction
│   └── runtime_resources.py # CPU thread optimization
├── simulation/            # Order routing mock
├── data/                  # Medicine database
├── evaluation/            # WER/accuracy metrics
└── notebooks/
    └── colab_gpu_launcher.ipynb  # Colab launcher
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Model**: OpenAI Whisper (HuggingFace Transformers)
- **Matching**: RapidFuzz (Fuzzy String Matching)
- **Audio**: Librosa, SoundFile, WebRTC VAD
- **Data**: Pandas, OpenPyXL

---

## 🎓 Academic Project

Minor Project demonstrating:
- Noise Reduction & Audio Preprocessing
- Accent-Aware Speech-to-Text (Whisper)
- Entity Extraction (Medicine/Dosage/Quantity)
- Performance Evaluation (WER Report)
