# Pharma Voice Orders - Getting Started

This document explains how to set up and run the **Pharma Voice Orders** application.

---

## 📋 Prerequisites

- **Python** 3.12+
- **[uv](https://github.com/astral-sh/uv)** (Modern Python package manager)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd pharma-voice-orders
uv sync
```

### 2. Run the Application
```bash
uv run start
```
This will launch the Streamlit app at `http://localhost:8501`.

---

## 📦 Available Commands

```bash
# Run the app
uv run start

# Add a new dependency
uv add <package-name>

# Sync dependencies (install/update)
uv sync

# Run streamlit directly (alternative)
uv run streamlit run app.py
```

---

## 🔧 Project Structure

```
pharma-voice-orders/
├── app.py                # Main Streamlit entry point
├── main.py               # Script wrapper (for `uv run start`)
├── pyproject.toml        # Project config & dependencies
├── core/                 # Preprocessing, ASR, Entity Extraction, Export
├── simulation/           # Manufacturer DB, Order Queue
├── evaluation/           # Metrics (WER, Accuracy)
└── data/                 # CSV files for medicines & manufacturers
```

---

## ❓ Why Use `uv run`?

Using `uv run` ensures the command executes within the project's **isolated virtual environment** (`.venv`), avoiding conflicts with globally installed packages (like Anaconda). This is the recommended way to run Python projects managed by `uv`.

---

## 🧪 Testing Your Setup

After running `uv run start`:
1. Open `http://localhost:8501` in your browser.
2. Select a distributor from the sidebar.
3. Record or upload an audio file (e.g., "Send 20 strips of Augmentin").
4. Watch orders get routed to manufacturer boxes.

---

*Last Updated: January 2026*
