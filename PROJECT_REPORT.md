# PHARMA VOICE ORDERS

**21CSA697A**  
**Final Report**

Submitted by  
**PRADEEP KUMAR**  
*(AA.SC.P2MCAXXXXXXX)*

in partial fulfilment of the requirements for the award of the degree of

## MASTER OF COMPUTER APPLICATIONS

**February 2026**

---

## Acknowledgement

I would like to express my sincere gratitude to my project guide and the faculty of the Department of Computer Applications for their invaluable guidance and support throughout this project. I am also thankful to my family and friends for their encouragement.

---

## Abstract

The **Pharma Voice Order** system is an AI-powered conversational interface designed to streamline B2B pharmaceutical ordering between distributors and manufacturers. Traditional processes rely on manual entry or prone-to-error phone calls. This project introduces a **Voice-First Architecture** running on **Google Colab T4 GPUs**, utilizing **OpenAI Whisper** for state-of-the-art speech recognition. A core innovation is the **Dual-Layer Entity Extraction Engine**, which combines O(1) semantic lookups with a composite fuzzy logic scorer to bridge the gap between imperfect speech and rigid inventory databases. The system features a real-time, glassmorphic UI built with Streamlit, ensuring a modern user experience while accurately routing orders to manufacturers like GSK, Sun Pharma, and Cipla.

---

## List of Figures

1.  **Figure 1.1**: System Data Flow (Mermaid)
2.  **Figure 3.1**: Dual-Layer Architecture Diagram
3.  **Figure 3.2**: Audio Preprocessing Pipeline
4.  **Figure 4.1**: Streamlit User Interface
5.  **Figure 4.2**: Manufacturer Routing Grid

---

## List of Abbreviations

-   **ASR**: Automatic Speech Recognition
-   **NER**: Named Entity Recognition
-   **NLP**: Natural Language Processing
-   **GPU**: Graphics Processing Unit (Google Colab T4)
-   **VAD**: Voice Activity Detection
-   **UI/UX**: User Interface / User Experience

---

# Chapter 1

## 1. Introduction

### 1.1 Background
The pharmaceutical supply chain is a complex network where accuracy is paramount. A single error in drug name transcription can have serious safety implications. Current digital solutions often require rigid manual entry, which is inefficient for distributors who are often on the move.

### 1.2 Motivation
The motivation is to "humanize" the B2B interface. By leveraging **Transformer-based Generative AI** accelerated by cloud GPUs, we can build a system that understands natural speech—accents, fillers, and speed—and converts it into structured business data (JSON/CSV) without manual intervention.

### 1.3 Problem Statement
**Problem**: Existing ordering systems lack the robust NLU capabilities to parse unstructured voice commands and route them to specific manufacturers dynamically.
**Solution**: An intelligent "Router" system that acts as a middleware between the spoken intent and the manufacturer's ERP, utilizing fuzzy logic and semantic mapping.

---

# Chapter 2

## 2. Literature Review

The evolution of medical speech recognition has moved from statistical HMMs to End-to-End Deep Learning:

-   **Contextual Understanding**: Traditional models required specific prompts. New **Transformer models** (Vaswani et al.) like OpenAI Whisper utilize self-attention mechanisms to understand context, making them superior for recognizing drug names in sentence flows.
-   **Fuzzy Logic in Domain Specificity**: Standard NLP libraries (Spacy/NLTK) fail on proprietary drug names (e.g., "Augmentin"). Research suggests that **Morphological Matching** (Levenshtein Distance) combined with **Phonetic Encoding** (Soundex) provides the highest recall for domain-specific jargon. The project adopts this hybrid approach.

---

# Chapter 3

## 3. System Design & Architecture

### 3.1 High-Level Data Flow

The system follows a linear yet fault-tolerant pipeline, accelerated by the **NVIDIA T4 GPU** environment.

```mermaid
flowchart TD
    subgraph Client["🖥️ Client Interface"]
        A[Microphone Input] -->|WAV Data| B[Streamlit UI]
    end

    subgraph Cloud["☁️ Processing Core (T4 GPU)"]
        B -->|Raw Audio| C[Audio Preprocessor]
        C -->|16kHz Mono| D[ASR Engine (Whisper)]
        D -->|Text| E[Entity Extractor]
        E -->|Structured Data| F[Manufacturer Router]
    end

    subgraph Output["📦 Supply Chain"]
        F --> G{Confidence?}
        G -->|High| H[Manufacturer Queue]
        G -->|Low| I[Quarantine Node]
    end

    style Cloud fill:#e1f5fe,stroke:#01579b
    style Output fill:#f1f8e9,stroke:#33691e
```

### 3.2 Dual-Layer Entity Extraction (The "New Modus")
To ensure patient safety and high order accuracy, we implemented a **Dual-Layer Architecture**:

```mermaid
flowchart LR
    Input[Transcribed Text] --> Layer1
    
    subgraph Layer1["Layer 1: Semantic (O(1))"]
        Direction1[Alias Lookup]
        Result1{Match?}
    end
    
    subgraph Layer2["Layer 2: Morphological (O(N))"]
        Direction2[Composite Scorer]
        Logic[60% Token Set + 40% Partial]
    end
    
    Layer1 --> Result1
    Result1 -->|Yes| Success[✅ Mapped Entity]
    Result1 -->|No| Layer2
    
    Layer2 --> Direction2 --> Logic --> Threshold{Score > 75%?}
    Threshold -->|Yes| Success
    Threshold -->|No| Fail[❌ Quarantine]
```

-   **Layer 1 (Alias Map)**: Handles common slang (e.g., "PCM" -> "Paracetamol").
-   **Layer 2 (Fuzzy Logic)**: Handles spelling errors using a composite score of Token Set Ratio and Partial Ratio.
-   **The Trust Cliff**: A distinct threshold of **75%** is enforced. Matches below this are rejected to prevent "hallucinations" of similar-sounding drugs.

---

# Chapter 4

## 4. Implementation Details

### 4.1 Technology Stack
-   **Runtime**: Python 3.10 on Google Colab (T4 GPU).
-   **Frontend**: Streamlit (Glassmorphic Design System).
-   **AI Engine**: OpenAI Whisper (Medium/Large).
-   **Logic**: RapidFuzz (C++ accelerated string matching).

### 4.2 Routing Logic
The routing engine uses **Prominent Mapping** for top brands:
```python
# Pseudo-code for Routing
if drug_name in PROMINENT_MAP:
    manufacturer = PROMINENT_MAP[drug_name] # e.g., "Augmentin" -> "GSK"
else:
    manufacturer = fuzzy_search(drug_name, MANUFACTURER_DB)
```

---

# Chapter 5

## 5. Testing & Validation

### 5.1 Qualitative Assessment
The system was tested with diverse inputs:
-   **Fast Speech**: "Send 50 Dolo sheets quickly" -> **Correctly Parsed**.
-   **Background Noise**: "Order Metacin" (with fan noise) -> **Correctly Parsed** (due to Spectral Gating).
-   **Ambiguity**: "Give me Azithral" -> **Mapped to Alembic Pharma** (Correct).

### 5.2 Performance Metrics
On the **T4 GPU**, average inference time for a 10-second audio clip remains under **1.5 seconds**, providing a near real-time experience for the user. The Dual-Layer extraction operates in sub-millisecond timeframes.

---

# Chapter 6

## 6. Conclusion

The **Pharma Voice Order** system successfully demonstrates that high-accuracy, voice-driven B2B commerce is viable with modern Transformer architectures. By solving the specific challenges of drug name recognition through a **Dual-Layer Architecture** and leveraging **GPU Acceleration**, the system provides a robust, scalable solution for the pharmaceutical supply chain.

---

# Chapter 7

## 7. References

1.  A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
2.  OpenAI, "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision", 2022.
3.  Streamlit Documentation, https://docs.streamlit.io/
4.  Hugging Face Transformers, https://huggingface.co/

---

## Appendix: Code Sample
*(Core Entity Extraction Logic)*
```python
def extract(text):
    # 1. Alias Resolution
    if text in ALIAS_MAP: return ALIAS_MAP[text]
    
    # 2. Fuzzy Matching
    score = fuzz.token_set_ratio(text, target) * 0.6 + fuzz.partial_ratio(text, target) * 0.4
    if score > 75: return target
    return None
```
