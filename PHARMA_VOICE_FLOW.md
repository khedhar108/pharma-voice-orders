# Pharma Voice Order Flow

This document outlines the end-to-end architecture of the Pharma Voice Order system, detailing how voice inputs are transformed into standardized pharmaceutical orders and routed to manufacturers.

## 1. Taking Input (Voice Capture)
The system accepts input through two primary channels in the Streamlit UI:
- **Live Recording**: Direct microphone input using the browser's recording capabilities (`st.audio_input`).
- **File Upload**: Support for pre-recorded audio files (`.wav`, `.mp3`, `.m4a`, `.ogg`).

**Preprocessing**:
- Regardless of the input source, all audio is normalized to **16kHz mono WAV** format.
- This standardization ensures compatibility with transformer-based ASR models and removes encoding inconsistencies.

## 2. Converting Voice to Digital Text (Transcription)
"Converting Voice to Digital Text" refers to the process of abstracting the audio signal into textual data (Speech-to-Text). This is the bridge between the analogue and digital worlds.

- **Engine**: The system utilizes **Automatic Speech Recognition (ASR)** pipelines.
- **Modes**:
  - **Cloud Mode**: Offloads processing to the **HuggingFace Inference API**, allowing for high-quality transcription without local resource usage.
  - **Local Mode**: Loads models directly onto the host machine using `torch` and `transformers`.

## 3. Using Transformers
The core intelligence relies on state-of-the-art Transformer architectures:

- **Models**:
  - **OpenAI Whisper** (Tiny, Small, Medium, Large): General-purpose, robust speech recognition.
  - **Google MedASR** (via `transformers`): Specialized models fine-tuned for medical terminology.
- **Function**: These transformers handle the "seq2seq" (sequence-to-sequence) task of mapping audio spectrograms to text tokens, effectively handling accents, noise, and complex medical drug names.

## 4. Standardizing (Entity Extraction & Normalization)
Once text is obtained, raw strings are converted into structured data objects:

- **Text Cleaning**: Removal of ASR artifacts (`</s>`, `<unk>`) and filler words ("um", "ah", "please").
- **Numeric Conversion**: Spoken numbers ("two hundred") are converted to digits ("200").
- **Entity Recognition (NER)**:
  - **Medicine Name**: `rapidfuzz` is used to fuzzy-match transcribed words against a canonical `ManufacturerDB`.
  - **Form Detection**: Classifies items as *tablet, syrup, injection, cream, etc.*
  - **Quantity/Unit Parsing**: Extracts amounts (e.g., "50 strips", "10 bottles").
  - **Dosage Extraction**: Identifies strengths (e.g., "500mg", "10ml").

## 5. Connecting to Routers & Manufacturers
The final stage involves routing the structured orders to their supply chain destinations:

- **Manufacturer Database**: Acts as the routing table, linking every medicine (e.g., "Augmentin") to its manufacturer (e.g., "GSK").
- **Order Queueing**: The `OrderQueue` system aggregates individual items.
- **Visual Routing**:
  - **Mapped Orders**: Automatically routed to specific Manufacturer Nodes (e.g., *Sun Pharma Node*, *Cipla Node*).
  - **Unmapped Orders**: Items with low confidence or unknown manufacturers are routed to a **Quarantine Node** for manual review.
- **Supply Chain Connection**: Validated batches can be exported (Excel/CSV) for integration with pharmaceutical ERP systems.
