# Hugging Face Spaces Deployment Guide

This guide explains how to deploy the **Pharma Voice Orders** application to Hugging Face Spaces using a Docker Space.

---

## Prerequisites

1.  A [Hugging Face account](https://huggingface.co/join).
2.  A Hugging Face Space created with **Docker SDK**.
3.  Git installed on your local machine.

---

## Step 1: Install Hugging Face CLI

Install the CLI globally:

```bash
pip install huggingface_hub
```

---

## Step 2: Login to Hugging Face

Authenticate with your HF token (get one from [Settings > Tokens](https://huggingface.co/settings/tokens)):

```bash
huggingface-cli login
```

Enter your token when prompted. This saves your credentials for Git operations.

---

## Step 3: Add HF Space as Git Remote

Navigate to your project folder and add the Space as a remote:

```bash
cd pharma-voice-orders
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/pharma-voice-orders
```

Replace `YOUR_USERNAME` with your actual HuggingFace username (e.g., `Khedhar`).

---

## Step 4: Push to Hugging Face

Force push your code to the Space. **Important:** HuggingFace Spaces uses `main` as the default branch.

If your local branch is `master`:
```bash
git push hf master:main --force
```

If your local branch is already `main`:
```bash
git push hf main --force
```

> **Tip:** To check your current branch name, run: `git branch`

---

## Step 5: Verify Deployment

1.  Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/pharma-voice-orders`
2.  Wait for the build to complete (check the **Logs** tab).
3.  Once running, the app will be live at the Space URL.

---

## Dockerfile Notes

The `Dockerfile` in this project:
-   Uses Python 3.11 slim image.
-   Installs system dependencies for audio processing.
-   Installs Python dependencies with `uv`.
-   Exposes port `7860` (HF Spaces default).

---

## Environment Variables (Optional)

If your app requires secrets (e.g., `HF_TOKEN`), configure them in Space Settings > Repository Secrets.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Permission denied` | Run `huggingface-cli login` again |
| `Build failed` | Check Logs tab for error details |
| `Port not accessible` | Ensure `Dockerfile` exposes port `7860` |

---

## Useful Commands

```bash
# Check current remotes
git remote -v

# Remove HF remote
git remote remove hf

# Re-add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/pharma-voice-orders
```
