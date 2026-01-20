# 🏥 Pharma Voice Orders

> **Accent-Aware Speech-to-Text Engine for Distributor Order Processing**

This application helps pharmaceutical manufacturers process voice orders from primary distributors efficiently. It simulates an end-to-end pipeline:
1. **Distributor Input**: Voice recording of orders (e.g., "Send 20 strips of Augmentin 625").
2. **AI Processing**: Transcription using OpenAI Whisper and Entity Extraction.
3. **Simulation**: Routing orders to specific manufacturer boxes (Sun Pharma, GSK, etc.).
4. **Export**: Generating structured Excel sheets for ERP systems.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure

- `app.py`: Main Streamlit application entry point.
- `core/`: Contains ASR engine, Preprocessor, and Entity Extractor.
- `simulation/`: Mock database and order routing logic.
- `data/`: Sample medicine and manufacturer databases.
- `evaluation/`: Scripts to calculate WER, Accuracy, and Latency.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Model**: OpenAI Whisper (via HuggingFace Transformers)
- **Data Processing**: Pandas, OpenPyXL
- **Matching**: RapidFuzz (Fuzzy String Matching)
- **Audio**: Librosa, SoundFile

## 🎓 University Use

This project demonstrates the "Minor Project" proposal deliverables:
- Noise Reduction & Preprocessing
- Accent-Aware STT (simulated via Whisper)
- Entity Extraction (Medicine/Dosage/Quantity)
- Performance Evaluation (WER Report)
