# Hugging Face Spaces - Docker Setup Guide

## 📋 Fill the Form (Screenshot Reference)

| Field | Suggested Value |
|-------|-----------------|
| **Owner** | `Khedhar` (your account) ✅ |
| **Space name** | `pharma-voice-orders` |
| **Short description** | `Voice-to-Order: Speech-to-text pharmaceutical ordering system using Whisper ASR` |
| **License** | `MIT` (or leave blank for now) |
| **Select the Space SDK** | 🐳 **Docker** |
| **Space hardware** | `CPU basic` (free) or `T4 GPU` for faster inference |
| **Visibility** | `Public` (for demo) or `Private` |

---

## 🐳 Why Docker?

1. **Full control** over dependencies and environment
2. **Pre-download models** during build (instant startup for users)
3. **Consistent behavior** across local and cloud
4. **Streamlit works perfectly** with Docker on HF Spaces

---

## 📁 Files You Need in Your Space

After creating the Space, you'll push these files:

```
pharma-voice-orders/
├── Dockerfile           # Build instructions
├── requirements.txt     # Python packages
├── app.py              # Your Streamlit app
├── core/               # Your modules
├── simulation/
├── evaluation/
└── data/
```

---

## ⚙️ Adding Secrets (HF Token)

After creating the Space:
1. Go to **Settings** → **Repository secrets**
2. Add:
   - **Name**: `HF_TOKEN`
   - **Value**: Your token from https://huggingface.co/settings/tokens

The app will automatically use this token for gated models.

---

## 🚀 Next Steps

1. Fill the form with values above
2. Click **Create Space**
3. Clone the Space repo to your local machine
4. Copy all project files
5. Push to the Space

I'm now creating the `Dockerfile` and updating `app.py` with proper status indicators!
