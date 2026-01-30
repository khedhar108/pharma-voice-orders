# Evaluation Framework Documentation

## Evaluation Workflow

```mermaid
flowchart TD
    subgraph Setup["⚙️ Setup"]
        A[Load Ground Truth CSV] --> B{Has Data?}
        B -->|No| C[Show Template/Warning]
        B -->|Yes| D[Display Entry Count]
    end
    
    subgraph Selection["🎵 Audio Selection"]
        D --> E[List audioData Files]
        E --> F[Multi-select Files]
        F --> G[Run Evaluation Button]
    end
    
    subgraph Processing["🔄 Batch Processing"]
        G --> H[For Each Audio File]
        H --> I[Preprocess Audio]
        I --> J[Transcribe]
        J --> K[Extract Entities]
        K --> L{More Files?}
        L -->|Yes| H
        L -->|No| M[Collected Extractions]
    end
    
    subgraph Evaluation["📊 Metrics Calculation"]
        M --> N[EntityEvaluator.evaluate]
        N --> O[Match Expected vs Extracted]
        O --> P[Calculate TP/FP/FN]
        P --> Q[Compute Precision/Recall/F1]
    end
    
    subgraph Display["📈 Results Display"]
        Q --> R[Metrics Cards]
        R --> S[Per-Field Metrics Table]
        S --> T[Comparison Table]
        T --> U[Filter by Match Type]
        U --> V[Export CSV/JSON]
    end
    
    style A fill:#4facfe,stroke:#333,color:#fff
    style G fill:#00f260,stroke:#333,color:#fff
    style N fill:#a855f7,stroke:#333,color:#fff
    style R fill:#ff6b6b,stroke:#333,color:#fff
```

## How to Create Ground Truth

### Option A: Manual (Recommended)
1. Open `evaluation/ground_truth.csv`
2. Listen to each audio file (e.g., `R_001.m4a`).
3. Add rows matching the audio:
    - `audio_file`: Filename (e.g., `R_001.m4a`)
    - `order_index`: 1, 2, 3...
    - `medicine_name`: **Must match `medicines.csv` exactly** (canonical name).
    - `quantity`: Spoken quantity (e.g., "50 strips").

### Matching Logic (EntityEvaluator)

| Field | Match Criteria | Example |
| :--- | :--- | :--- |
| **Medicine** | Fuzzy match > 85% OR exact string match | "DOLO 650" ↔ "DOLO-650" ✅ |
| **Quantity** | Number match + unit similarity | "50 strips" ↔ "50 strip" ✅ |
| **Dosage** | Normalized text match | "500 mg" ↔ "500mg" ✅ |

### Metrics Definitions

- **Precision**: (TP / (TP + FP)) — "How many of the *system's* findings were correct?"
- **Recall**: (TP / (TP + FN)) — "How many of the *actual* medicines did we find?"
- **F1 Score**: Harmonic mean of Precision and Recall.
