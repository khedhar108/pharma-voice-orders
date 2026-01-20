# Hugging Face Spaces Deployment Guide

This guide explains how to deploy the **Pharma Voice Orders** application to Hugging Face Spaces using a Docker Space.

---

## Prerequisites

1.  A [Hugging Face account](https://huggingface.co/join).
2.  A Hugging Face Space created with **Docker SDK**.
3.  Git installed on your local machine.
4.  [uv](https://docs.astral.sh/uv/) installed (recommended for HF CLI).

---

## ⚠️ IMPORTANT: No Hardcoded Secrets

**Never hardcode API tokens, passwords, or secrets in your code.**

HuggingFace will automatically reject pushes containing tokens like:
```python
# ❌ WRONG - Never do this
TOKEN = "hf_abc123xyz..."
```

Instead, always use environment variables:
```python
# ✅ CORRECT - Use environment variables
import os
token = os.environ.get("HF_TOKEN", "")
```

Configure secrets in **Space Settings > Repository Secrets** after deployment.

---

## Complete Deployment Flow

Follow this exact sequence from start to finish:

### Step 1: Initialize Git Repository

```bash
cd pharma-voice-orders
git init
```

### Step 2: Rename Branch to `main`

HuggingFace Spaces uses `main` as the default branch:

```bash
git branch -M main
```

### Step 3: Login to HuggingFace (Using UVX - Recommended)

The easiest way to login is using `uvx` (no global install needed):

```bash
uvx hf auth login
```

This will prompt you to enter your token (get one from [Settings > Tokens](https://huggingface.co/settings/tokens)).

**Alternative methods:**

```bash
# Standalone installer (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
hf auth login

# Or using pip
pip install huggingface_hub
huggingface-cli login
```

### Step 4: Stage and Commit Files

```bash
git add .
git commit -m "Initial commit"
```

### Step 5: Add HuggingFace Remote

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/pharma-voice-orders
```

Replace `YOUR_USERNAME` with your actual HuggingFace username.

### Step 6: Force Push to HuggingFace

```bash
git push hf main --force
```

> **Note:** If you get `src refspec main does not match any`, you forgot to commit. Run `git commit` first.

---

## Step 7: Configure Space Secrets

After pushing, add your secrets in the HuggingFace Space:

1. Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/pharma-voice-orders`
2. Click **Settings** → **Repository secrets**
3. Add required secrets:
   - **Name:** `HF_TOKEN`
   - **Value:** Your HuggingFace token

---

## Step 8: Verify Deployment

1. Go to your Space URL
2. Wait for the Docker build to complete (check the **Logs** tab)
3. Once running, the app will be live!

---

## README.md Configuration

Your `README.md` must include YAML frontmatter for HuggingFace Spaces:

```yaml
---
title: Pharma Voice Orders
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---
```

---

## Dockerfile Notes

The `Dockerfile` in this project:
- Uses Python 3.11 slim image
- Installs system dependencies for audio processing (ffmpeg, libsndfile)
- Installs Python dependencies with `uv`
- Exposes port `7860` (HF Spaces default)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Permission denied` | Run `uvx hf auth login` again |
| `src refspec main does not match any` | You forgot to commit. Run `git commit -m "message"` first |
| `pre-receive hook declined` | You have hardcoded secrets in code. Remove them and rewrite git history |
| `Configuration error` | Add YAML frontmatter to README.md |
| `Build failed` | Check Logs tab for error details |
| `Port not accessible` | Ensure Dockerfile exposes port `7860` |

---

## Useful Commands

```bash
# Check current branch
git branch

# Check current remotes
git remote -v

# Remove HF remote
git remote remove hf

# Re-add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/pharma-voice-orders

# Check HF login status
uvx hf auth whoami
```
