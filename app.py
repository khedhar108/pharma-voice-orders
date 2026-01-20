"""
Pharma Voice Orders - Main Application
Streamlit UI for simulating Distributor -> Manufacturer Voice Ordering System
"""

import streamlit as st
import pandas as pd
import time
import os
from pathlib import Path

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
    st.header("⚙️ Configuration")
    
    distributor = st.selectbox(
        "Select Distributor",
        ["Apollo Pharmacy", "MedPlus", "Frank Ross", "Online Pharma", "Local Chemist"]
    )
    
    asr_model = st.selectbox(
        "ASR Model",
        [
            "google/medasr",
            "openai/whisper-tiny",
            "openai/whisper-small", 
            "openai/whisper-medium",
            "openai/whisper-large-v3",
        ]
    )
    
    # Note about MedASR - now enabled!
    if "medasr" in asr_model:
        st.success("✅ MedASR enabled (transformers from GitHub installed)")
    
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
    from huggingface_hub import InferenceClient
    import io
    
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Login if token provided
    if token:
        from huggingface_hub import login
        login(token=token)
    
    # Determine model class based on model name
    if "medasr" in model_name:
        from transformers import AutoModelForCTC
        model_class = AutoModelForCTC
    else:
        model_class = AutoModelForSpeechSeq2Seq

    # Load Model with support for custom code (trust_remote_code=True)
    try:
        model = model_class.from_pretrained(
            model_name,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=True
        )
    except OSError:
        # Fallback for models that might not support safetensors or other issues
        model = model_class.from_pretrained(
            model_name,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
            
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
        trust_remote_code=True
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
    import os
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
    except:
        free_gb = -1
    
    # Model sizes (approximate)
    model_sizes = {
        "openai/whisper-tiny": 0.15,
        "openai/whisper-small": 0.5,
        "openai/whisper-medium": 1.5,
        "openai/whisper-large-v3": 3.1,
        "google/medasr": 0.3,  # ~300MB
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

# --- Main Content ---
st.markdown('<h1 class="main-header">🏥 Order Processing Center</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Reviewing orders from: <strong>{distributor}</strong></p>', unsafe_allow_html=True)

# Smart Model Status Indicator
model_status = check_model_status(asr_model)

# Use a flex container for the status badge (aligned right) to avoid empty column artifacts
status_html = ""
if use_cloud:
    status_html = '''
        <div class="status-ready" style="border-color: rgba(139, 92, 246, 0.3); background: rgba(139, 92, 246, 0.1); color: #a78bfa;">
            <span class="status-dot" style="background: #a78bfa;"></span>
            ☁️ Cloud Ready
        </div>
    '''
elif model_status["is_cached"]:
    status_html = '''
        <div class="status-ready">
            <span class="status-dot green"></span>
            ✅ Cached (Local)
        </div>
    '''
elif model_status["has_space"]:
    status_html = f'''
        <div class="status-loading" style="border-color: rgba(251, 191, 36, 0.3); background: rgba(251, 191, 36, 0.1); color: #fbbf24;">
            <span class="status-dot" style="background: #fbbf24;"></span>
            ⬇️ Download ({model_status["required_gb"]}GB)
        </div>
    '''
else:
    status_html = f'''
        <div class="status-loading" style="border-color: rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.1); color: #ef4444;">
            <span class="status-dot" style="background: #ef4444;"></span>
            ⚠️ Low Space ({model_status["free_gb"]}GB free)
        </div>
    '''
    st.warning(f"Need {model_status['required_gb']}GB, only {model_status['free_gb']}GB free. Choose a smaller model or free disk space.")

# Render Status aligned to right
if status_html:
    st.markdown(f'<div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">{status_html}</div>', unsafe_allow_html=True)

# Download confirmation state
if 'download_approved' not in st.session_state:
    st.session_state.download_approved = {}

# Show download confirmation ONLY if Local mode AND model not cached AND not yet approved
if not use_cloud and not model_status["is_cached"] and asr_model not in st.session_state.download_approved:
    with st.container():
        st.markdown("---")
        st.markdown(f"### ⬇️ Download Required")
        st.info(f"""**{asr_model}** is not cached locally.
        
📦 Size: **{model_status['required_gb']}GB**
💾 Free space: **{model_status['free_gb']}GB**
📂 Cache location: `C:\\Users\\{os.environ.get('USERNAME', 'User')}\\.cache\\huggingface\\hub\\`

💡 **Tip:** Switch to Cloud Mode to avoid downloading!
        """)
        
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ Yes, Download", type="primary", use_container_width=True):
                # UI Update: Show "Downloading" in the badge
                status_placeholder.markdown('''
                    <div class="status-loading" style="border-color: rgba(59, 130, 246, 0.3); background: rgba(59, 130, 246, 0.1); color: #60a5fa;">
                        <span class="status-dot" style="background: #3b82f6; animation: pulse 0.5s infinite;"></span>
                        ⏳ Downloading...
                    </div>
                ''', unsafe_allow_html=True)
                
                with st.spinner(f"⬇️ Downloading {asr_model}... This may take a while."):
                    try:
                        # Trigger download and load into cache
                        load_asr_engine(asr_model, hf_token)
                        st.session_state.download_approved[asr_model] = True
                        st.session_state.model_ready = True
                        st.success("✅ Download complete! Model is ready.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Download failed: {e}")
        with col_no:
            if st.button("❌ Cancel", type="secondary", use_container_width=True):
                st.info("Download cancelled. Select a cached model or use Cloud Mode.")

# Layout: Input (Left) vs Output (Right)
col1, col2 = st.columns([1, 2])

with col1:
    # Voice Container
    st.markdown('<div class="voice-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #4facfe; margin: 0 0 10px 0;">Voice Input</h3>', unsafe_allow_html=True)
    
    # Example Prompt Tagline
    example_prompt = "Send me 50 strips of Paracetamol, 20 bottles of Ascoril syrup, and also 10 tubes of Betnovate cream."
    st.markdown(f'''
        <div style="background: rgba(79, 172, 254, 0.1); border: 1px dashed rgba(79, 172, 254, 0.4); border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <span style="color: #4facfe; font-weight: 600; font-size: 0.75rem;">💡 TRY SAYING:</span>
            <p style="color: rgba(255,255,255,0.9); font-style: italic; margin: 8px 0 0 0; font-size: 0.9rem; line-height: 1.5;">"{example_prompt}"</p>
        </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="mic-icon">🎙️</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔴 Record", "📁 Upload"])
    
    audio_data = None
    
    with tab1:
        try:
            audio_val_rec = st.audio_input("Click to record", label_visibility="collapsed")
            if audio_val_rec:
                audio_data = audio_val_rec
        except AttributeError:
            st.warning("Update Streamlit to use `st.audio_input`.")
    
    with tab2:
        audio_val_up = st.file_uploader("Upload Audio", type=['wav', 'mp3'], label_visibility="collapsed")
        if audio_val_up:
            audio_data = audio_val_up
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process Audio
    if audio_data:
        st.success("✅ Audio captured!")
        st.audio(audio_data)
        
        if st.button("🚀 Process Order", type="primary", use_container_width=True):
            transcription_text = ""
            
            if use_cloud:
                # CLOUD MODE - Use HuggingFace Inference API (no download)
                with st.spinner("☁️ Transcribing via Cloud..."):
                    try:
                        transcription_text = transcribe_cloud(audio_data, asr_model, hf_token)
                        st.toast("✅ Cloud Transcription Complete!")
                    except Exception as e:
                        st.error(f"❌ Cloud API failed: {e}")
                        st.info("💡 Try Local mode or check your token/model.")
                        st.stop()
            else:
                # LOCAL MODE - Download and run model locally
                with st.spinner("🔄 Loading Local ASR Model..."):
                    try:
                        asr = load_asr_engine(asr_model, hf_token)
                        st.session_state.model_ready = True
                    except Exception as e:
                        st.error(f"❌ Model load failed: {e}")
                        st.stop()
                
                with st.spinner("🎧 Transcribing Locally..."):
                    processed_audio = preprocessor.process(audio_data)
                    result = asr(processed_audio)
                    transcription_text = result["text"].replace("</s>", "").strip()
                    st.toast("✅ Local Transcription Complete!")
            
            # Store transcription
            st.session_state.last_transcription = transcription_text
            
            with st.spinner("📦 Extracting Orders..."):
                extracted_orders = extractor.extract(transcription_text)
                
                if extracted_orders:
                    st.success(f"Found {len(extracted_orders)} items!")
                    for order in extracted_orders:
                        st.session_state.orders.append(order)
                    st.rerun()
                else:
                    st.warning("No medicines found. Try: 'Send 20 strips of Augmentin'")

    st.markdown("---")
    st.markdown("### 📝 Transcription")
    
    current_text = st.session_state.get('last_transcription', "")
    st.text_area(
        "Transcription Output",
        current_text,
        height=120,
        disabled=True,
        placeholder="Transcription will appear here...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 🏭 Manufacturer Routing")
    
    # Get grouped orders from session state
    from simulation.order_queue import OrderQueue
    queue = OrderQueue()
    grouped_orders = queue.get_grouped_orders(db)
    all_manufacturers = db.get_all_manufacturers()
    
    # Grid Layout
    import textwrap
    row1_cols = st.columns(2)
    row2_cols = st.columns(2)
    row3_cols = st.columns(2)
    
    # 6 Manufacturers -> 3 Rows of 2
    for idx, mfr in enumerate(all_manufacturers):
        if idx < 2:
            col = row1_cols[idx]
        elif idx < 4:
            col = row2_cols[idx - 2]
        elif idx < 6:
            col = row3_cols[idx - 4]
        else:
            continue
            
        with col:
            mfr_name = mfr['name']
            orders = grouped_orders.get(mfr_name, [])
            order_count = len(orders)
            
            # Determine Visual State
            is_active = order_count > 0
            active_class = "active" if is_active else ""
            badge_class = "active" if is_active else ""
            
            # Generate HTML - Single line to prevent Markdown parsing issues
            html_parts = []
            
            # 1. Header & Open Body
            html_parts.append(f'<div class="node-card {active_class}">')
            html_parts.append('<div class="node-header">')
            html_parts.append(f'<span class="node-title"><span style="opacity:0.7">🏭</span> {mfr_name}</span>')
            html_parts.append(f'<span class="node-badge {badge_class}">{order_count} Items</span>')
            html_parts.append('</div><div class="node-body">')
            
            # 2. Body Content
            if is_active:
                for order in orders:
                    # Confidence Logic
                    conf = order.get('confidence', 0)
                    conf_class = "conf-low"
                    if conf >= 90: conf_class = "conf-high"
                    elif conf >= 75: conf_class = "conf-med"
                    
                    med_name = order.get('medicine_standardized', order['medicine'])
                    dosage = order.get('dosage', '-')
                    
                    html_parts.append(f'<div class="order-chip {conf_class}">')
                    html_parts.append('<div class="chip-main">')
                    html_parts.append(f'<span class="chip-med">{med_name}</span>')
                    html_parts.append(f'<span class="chip-meta">{dosage}</span>')
                    html_parts.append('</div>')
                    html_parts.append(f'<span class="chip-qty">{order["quantity"]}</span>')
                    html_parts.append('</div>')
            else:
                html_parts.append('<div style="color: rgba(255,255,255,0.2); font-style: italic; font-size: 0.85rem; text-align: center; padding: 10px;">Waiting for data...</div>')
                
            # 3. Close Body & Card
            html_parts.append('</div></div>')
            
            st.markdown("".join(html_parts), unsafe_allow_html=True)
    
    # Unknown Orders (Quarantine Node)
    unknowns = grouped_orders.get('Unknown', [])
    if unknowns:
        html_parts = []
        html_parts.append('<div class="node-card active" style="border-color: rgba(255, 51, 102, 0.3); box-shadow: 0 0 20px rgba(255, 51, 102, 0.1);">')
        html_parts.append('<div class="node-header">')
        html_parts.append('<span class="node-title" style="color: #ff3366;"><span>⚠️</span> Quarantine / Unmapped</span>')
        html_parts.append(f'<span class="node-badge" style="background: rgba(255, 51, 102, 0.1); color: #ff3366; border: 1px solid rgba(255, 51, 102, 0.2);">{len(unknowns)} Items</span>')
        html_parts.append('</div><div class="node-body">')
        
        for order in unknowns:
            html_parts.append('<div class="order-chip conf-low">')
            html_parts.append('<div class="chip-main">')
            html_parts.append(f'<span class="chip-med" style="color: #ff3366;">{order["medicine"]} (Raw)</span>')
            html_parts.append(f'<span class="chip-meta">Confidence: {order.get("confidence", 0)}%</span>')
            html_parts.append('</div>')
            html_parts.append(f'<span class="chip-qty">{order["quantity"]}</span>')
            html_parts.append('</div>')
            
        html_parts.append('</div></div>')
        st.markdown("".join(html_parts), unsafe_allow_html=True)

    st.markdown("---")
    
    # Export Buttons
    if st.session_state.orders:
        from core.excel_exporter import ExcelExporter
        
        col_excel, col_csv = st.columns(2)
        
        with col_excel:
            excel_data = ExcelExporter.export(st.session_state.orders, db=db)
            st.download_button(
                label="📥 Export to Excel",
                data=excel_data,
                file_name="pharma_orders.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        with col_csv:
            csv_data = ExcelExporter.export_csv(st.session_state.orders, db=db)
            st.download_button(
                label="📄 Export to CSV",
                data=csv_data,
                file_name="pharma_orders.csv",
                mime="text/csv",
                use_container_width=True
            )

    # --- Informational Footer (New Section) ---
    # --- Informational Footer (New Section) ---
    footer_html = []
    footer_html.append('<div class="info-container">')
    footer_html.append('<div class="info-grid">')
    
    # 1. How to Use Section
    footer_html.append('<div class="info-section">')
    footer_html.append('<h4>💡 How to use it</h4>')
    footer_html.append('<ul class="info-list">')
    footer_html.append('<li class="info-item">')
    footer_html.append('<span>🔹 <span class="info-highlight">Mixed Manufacturers:</span></span>')
    footer_html.append('<span class="info-example">"Send Paracetamol tablet 300 strips, also Azithromycin 50 strips and Volini spray 20 pieces."</span>')
    footer_html.append('</li>')
    footer_html.append('<li class="info-item">')
    footer_html.append('<span>🔹 <span class="info-highlight">Forms & Units:</span></span>')
    footer_html.append('<span class="info-example">"Order 50 bottles of Ascoril syrup, 20 tubes of Betnovate cream, and 10 packs of Prega News."</span>')
    footer_html.append('</li>')
    footer_html.append('<li class="info-item">')
    footer_html.append('<span>🔹 <span class="info-highlight">Pronunciation/Noisy:</span></span>')
    footer_html.append('<span class="info-example">"Uh, give me some Combiflam... maybe 20 strips? And... Zinetac 150."</span>')
    footer_html.append('</li>')
    footer_html.append('</ul></div>')
    
    # 2. Medical Areas Section
    footer_html.append('<div class="info-section">')
    footer_html.append('<h4>🏥 Medical Areas Covered</h4>')
    footer_html.append('<ul class="info-list">')
    footer_html.append('<li class="info-item">')
    footer_html.append('<span>🍬 <span class="info-highlight">Syrups</span></span>')
    footer_html.append('<span class="info-example">("50 bottles of Ascoril")</span>')
    footer_html.append('</li>')
    footer_html.append('<li class="info-item">')
    footer_html.append('<span>🧴 <span class="info-highlight">Creams/Gels</span></span>')
    footer_html.append('<span class="info-example">("20 tubes of Betnovate")</span>')
    footer_html.append('</li>')
    footer_html.append('<li class="info-item">')
    footer_html.append('<span>💉 <span class="info-highlight">Injections</span></span>')
    footer_html.append('<span class="info-example">("10 vials of Amikacin")</span>')
    footer_html.append('</li>')
    footer_html.append('<li class="info-item">')
    footer_html.append('<span>💨 <span class="info-highlight">Sprays/Inhalers</span></span>')
    footer_html.append('<span class="info-example">("5 pcs of Volini spray")</span>')
    footer_html.append('</li>')
    footer_html.append('<li class="info-item">')
    footer_html.append('<span>💊 <span class="info-highlight">Tablets/Capsules</span></span>')
    footer_html.append('<span class="info-example">("100 strips of Paracetamol")</span>')
    footer_html.append('</li>')
    footer_html.append('</ul></div>')
    
    footer_html.append('</div></div>')
    
    st.markdown("".join(footer_html), unsafe_allow_html=True)
