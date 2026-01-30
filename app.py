"""
Pharma Voice Orders - Main Application
Streamlit UI for simulating Distributor -> Manufacturer Voice Ordering System
"""

import os

import streamlit as st
from huggingface_hub import InferenceClient

# Page Config
st.set_page_config(
    page_title="Pharma Voice Orders",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Avant-Garde Glassmorphic Design
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("assets/styles.css")

# --- Session State Initialization ---
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'orders' not in st.session_state:
    st.session_state.orders = []
if 'last_transcription' not in st.session_state:
    st.session_state.last_transcription = ""

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063167.png", width=50)
    st.title("PharmaVoice")
    st.caption("v1.0.0 | Minor Project")
    
    st.markdown("---")
    
    # Page Navigation
    page_mode = st.radio(
        "📍 Navigation",
        ["🏥 Order Processing", "📊 Evaluation"],
        index=0,
        help="Switch between order processing and model evaluation"
    )
    
    st.markdown("---")
    st.header("⚙️ Configuration")
    
    distributor = st.selectbox(
        "Select Distributor",
        ["Apollo Pharmacy", "MedPlus", "Frank Ross", "Online Pharma", "Local Chemist"]
    )
    
    asr_model = st.selectbox(
        "ASR Model",
        [
            "openai/whisper-medium",
            "openai/whisper-large-v3-turbo", # Turbo (Efficient)
            "openai/whisper-large",      # Large V1 (Original)
            "openai/whisper-large-v3",   # Large V3 (Latest)
        ],
        help="Medium: 1.5GB | Turbo: 2GB | Large V1: 3GB | Large V3: 3.1GB"
    )
    
    st.markdown("---")
    
    # Inference Mode Toggle
    st.subheader("⚡ Inference Mode")
    
    # HF Token Configuration
    # Token should be set via environment variable or entered by user
    hf_token_input = st.text_input(
        "🔑 HF Token",
        value=os.environ.get("HF_TOKEN", ""),
        type="password",
        help="Required for Cloud mode and gated models. Set via HF_TOKEN env var or enter here.",
    )
    
    # Check for token from input or environment
    hf_token = hf_token_input or os.environ.get("HF_TOKEN", "")
    
    # Mode selection based on token availability
    if hf_token:
        inference_mode = st.radio(
            "Select Mode",
            ["💻 Local (Faster)", "☁️ Cloud (No Download)"],
            index=0,
            help="Cloud uses HF servers. Local downloads model to your PC."
        )
        use_cloud = "Cloud" in inference_mode
        st.success("🔓 Token configured" + (" • Cloud Mode" if use_cloud else " • Local Mode"))
    else:
        use_cloud = False
        st.warning("⚠️ No token → Local mode only (requires download)")
        inference_mode = "💻 Local (Faster)"
    
    st.markdown("---")
    st.info("""
    **Instructions:**
    1. Select a distributor.
    2. Record your voice order.
    3. Watch orders route to manufacturers!
    """)
    
    if st.button("🔄 Clear Session", type="secondary"):
        st.session_state.clear()
        st.rerun()

# --- Cloud Inference (HuggingFace Inference API) ---
def transcribe_cloud(audio_data, model_name: str, token: str):
    """Transcribe audio using HuggingFace Inference API (no local download)."""

    
    client = InferenceClient(token=token)
    
    # Get audio bytes
    if hasattr(audio_data, 'read'):
        audio_bytes = audio_data.read()
        audio_data.seek(0)  # Reset for replay
    else:
        audio_bytes = audio_data
    
    # Call HuggingFace Inference API
    result = client.automatic_speech_recognition(
        audio=audio_bytes,
        model=model_name
    )
    
    # Result is either a string or dict with 'text' key
    if isinstance(result, str):
        return result
    else:
        return result.get("text", str(result))

# --- Local ASR Engine (Downloads Model) ---
@st.cache_resource(show_spinner=False)
def load_asr_engine(model_name: str, token: str = None):
    """Load ASR engine locally with proper status handling."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    from core.runtime_resources import configure_runtime
    
    # Configure runtime resources (thread limiting for CPU)
    configure_runtime(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Dynamic batch size: 1 for CPU (prevent OOM), 16 for GPU
    batch_size = 16 if device == "cuda" else 1
    
    # Login if token provided
    if token:
        from huggingface_hub import login
        login(token=token)
    
    # Load Model
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
    except OSError:
        # Fallback for models that might not support safetensors
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True
        )
            
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    
    return pipe

# --- Other Components (Lazy Load) ---
@st.cache_resource
def get_db():
    from simulation.manufacturer_db import ManufacturerDB
    return ManufacturerDB(data_dir="data")

@st.cache_resource
def get_preprocessor():
    from core.preprocessor import AudioPreprocessor
    return AudioPreprocessor()

@st.cache_resource
def get_extractor(_db):
    from core.entity_extractor import EntityExtractor
    return EntityExtractor(_db)

# --- Model Cache Checker ---
def check_model_status(model_name: str) -> dict:
    """Check if model is cached locally and get disk space info."""
    import shutil
    from pathlib import Path
    
    # HuggingFace cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_folder_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = cache_dir / model_folder_name
    
    # Check if model is cached
    is_cached = model_cache_path.exists() and any(model_cache_path.iterdir()) if model_cache_path.exists() else False
    
    # Check snapshots folder for actual model files
    snapshots_path = model_cache_path / "snapshots" if model_cache_path.exists() else None
    has_model_files = False
    if snapshots_path and snapshots_path.exists():
        for snapshot in snapshots_path.iterdir():
            # Check for safetensors or bin files
            if any(f.suffix in ['.safetensors', '.bin'] for f in snapshot.iterdir() if f.is_file()):
                has_model_files = True
                break
    
    # Get free disk space (C: drive on Windows)
    try:
        disk_usage = shutil.disk_usage(cache_dir if cache_dir.exists() else Path.home())
        free_gb = disk_usage.free / (1024 ** 3)
    except Exception:
        free_gb = -1
    
    # Model sizes (approximate)
    model_sizes = {
        "openai/whisper-medium": 1.5,
        "openai/whisper-large-v3-turbo": 2.0, # ~2GB VRAM
        "openai/whisper-large": 3.0,     # Large V1
        "openai/whisper-large-v3": 3.1,
    }
    required_gb = model_sizes.get(model_name, 2.0)
    
    return {
        "is_cached": is_cached and has_model_files,
        "free_gb": round(free_gb, 1),
        "required_gb": required_gb,
        "has_space": free_gb >= required_gb or is_cached,
        "cache_path": str(model_cache_path) if model_cache_path.exists() else None
    }

# Load non-blocking components
db = get_db()
preprocessor = get_preprocessor()
extractor = get_extractor(db)

# --- Page Routing ---
if "Evaluation" in page_mode:
    # ===== EVALUATION PAGE =====
    from evaluation.evaluation_ui import render_evaluation_tab
    
    # Create transcription function for evaluation
    def transcribe_local(audio_data, model_name, token):
        pipe = load_asr_engine(model_name, token)
        result = pipe(audio_data)
        return result.get("text", "") if isinstance(result, dict) else str(result)
    
    render_evaluation_tab(
        db=db,
        preprocessor=preprocessor,
        extractor=extractor,
        transcribe_fn=transcribe_local,
        use_cloud=use_cloud,
        model_name=asr_model,
        hf_token=hf_token
    )

else:
    # ===== ORDER PROCESSING PAGE =====
    # Calculate model status for UI
    model_status = check_model_status(asr_model)

    from order_processing_ui import render_order_processing_interface
    
    render_order_processing_interface(
        db=db,
        preprocessor=preprocessor,
        extractor=extractor,
        transcribe_cloud_fn=transcribe_cloud,
        load_asr_engine_fn=load_asr_engine,
        distributor=distributor,
        asr_model=asr_model,
        use_cloud=use_cloud,
        hf_token=hf_token,
        model_status=model_status
    )

