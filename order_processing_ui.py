"""
Order Processing UI Component

Handles the main order processing interface including:
- Voice recording/upload
- Transcription
- Entity Extraction
- Order Table Display
- Processing Button Logic
"""

import os
import time

import pandas as pd
import streamlit as st
from core.transcript_verifier import TranscriptVerifier


def render_order_processing_interface(
    db,
    preprocessor,
    extractor,
    transcribe_cloud_fn,
    load_asr_engine_fn,
    distributor: str,
    asr_model: str,
    use_cloud: bool,
    hf_token: str,
    model_status: dict
):
    """
    Render the main Order Processing UI.
    """
    
    # helper for local transcription
    def transcribe_local(audio_data, model, token):
        pipe = load_asr_engine_fn(model, token)
        result = pipe(audio_data)
        return result.get("text", "") if isinstance(result, dict) else str(result)

    # --- 1. Status Badge Rendering ---
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

    # --- 1.5 AI Verification Toggle ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ✨ AI Enhancement")
        enable_ai = st.toggle(
            "Enable AI Verification",
            value=True if use_cloud else False,
            help="Use LLM to clean transcript and perfectly separate Medicine, Dosage, and Quantity."
        )
        if enable_ai and not hf_token:
            st.warning("⚠️ HF Token required for AI")
            enable_ai = False

    # Render Status aligned to right
    if status_html:
        st.markdown(f'<div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">{status_html}</div>', unsafe_allow_html=True)

    # --- 2. Download Logic ---
    # Download confirmation state
    if 'download_approved' not in st.session_state:
        st.session_state.download_approved = {}

    # Show download confirmation ONLY if Local mode AND model not cached AND not yet approved
    if not use_cloud and not model_status["is_cached"] and asr_model not in st.session_state.download_approved:
        with st.container():
            # Create a placeholder for status updates
            status_placeholder = st.empty()
            
            # Display current status in placeholder
            if status_html:
                 status_placeholder.markdown(f'<div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">{status_html}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### ⬇️ Download Required")
            
            # Dynamic path display
            import platform

            if platform.system() == "Windows":
                 # Use os.environ for Windows user
                cache_path_str = f"C:\\Users\\{os.environ.get('USERNAME', 'User')}\\.cache\\huggingface\\hub\\"
            else:
                cache_path_str = "/home/user/.cache/huggingface/hub/"
                
            st.info(f"""**{asr_model}** is not cached locally.
            
    📦 Size: **{model_status['required_gb']}GB**
    💾 Free space: **{model_status['free_gb']}GB**
    📂 Cache location: `{cache_path_str}`

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
                            # Trigger download and load into cache (using passed fn)
                            load_asr_engine_fn(asr_model, hf_token)
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
                    st.stop() # Stop rendering further if cancelled/waiting

    # Layout: Input (Left) vs Output (Right)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Voice Container
        st.markdown('<div class="voice-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4facfe; margin: 0 0 10px 0;">Voice Input</h3>', unsafe_allow_html=True)
        
        # Example Prompt Logic (Restored) => See below
        # For now, keeping the example prompt simple as in extracted code but preserving structure
        
        # Example Prompt
        example_prompt = "Send me 10 strips of Augmentin 625, 50 strips of Dolo 650, and 20 strips of Pan D."
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
                audio_value = st.audio_input("Record Order")
                if audio_value:
                    audio_data = audio_value
            except AttributeError:
                st.warning("Update Streamlit to use `st.audio_input`.")
                
        with tab2:
            uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "ogg"])
            if uploaded_file:
                audio_data = uploaded_file
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process Button
        process_btn = False
        # Check model ready state logic
        if not use_cloud and not model_status["is_cached"] and not st.session_state.get('model_ready', False):
             st.button("⚡ PROCESS ORDER (Download Model First)", disabled=True, use_container_width=True)
        else:
             process_btn = st.button("⚡ PROCESS ORDER", type="primary", use_container_width=True, disabled=(audio_data is None))
    
    # Main Processing Logic
    if process_btn and audio_data:
        with col2:
            status_container = st.empty()
            
            # 1. Preprocessing Animation
            status_container.markdown('''
                <div class="processing-overlay">
                    <div class="wave-container">
                        <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                    </div>
                    <div class="proc-text">🎧 CLEANING AUDIO</div>
                </div>
            ''', unsafe_allow_html=True)
            
            try:
                # Convert to temporary file for processing
                temp_path = f"temp_rec_{int(time.time())}.wav"
                original_bytes = audio_data.getvalue() if hasattr(audio_data, 'getvalue') else audio_data
                with open(temp_path, "wb") as f:
                    f.write(original_bytes)
                
                # Save original audio bytes for comparison
                st.session_state.original_audio_bytes = original_bytes
                    
                processed_path = preprocessor.preprocess_file(temp_path)
                st.session_state.last_processed_audio = processed_path
                
                # Cleanup temp
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                
                # 2. Transcription Animation
                status_container.markdown('''
                    <div class="processing-overlay">
                         <div class="wave-container">
                            <div class="wave-bar" style="animation-duration: 0.8s"></div>
                            <div class="wave-bar" style="animation-duration: 1.2s"></div>
                            <div class="wave-bar" style="animation-duration: 0.9s"></div>
                            <div class="wave-bar" style="animation-duration: 1.1s"></div>
                            <div class="wave-bar" style="animation-duration: 0.8s"></div>
                        </div>
                        <div class="proc-text">📝 TRANSCRIBING...</div>
                    </div>
                ''', unsafe_allow_html=True)
                
                start_time = time.time()
                
                if use_cloud:
                    # Cloud Logic
                    # If AI Verification is enabled (toggle from sidebar), use TranscriptVerifier
                    # Note: We need to access the sidebar toggle state. 
                    # Ideally, pass 'enable_ai' as argument to this function, but it's defined in sidebar.
                    # We can access it via session state if key was set, or simpler: just check if verify_fn is available if we passed it?
                    # The prompt implies we should just fix the flow.
                    # Let's verify usage of TranscriptVerifier if available.
                    
                    # For now, we'll do standard transcription then verify if enable_ai is Checked in Sidebar
                    # (We need to capture that value passed into this function or from state)
                     
                    raw_text = transcribe_cloud_fn(open(processed_path, "rb"), asr_model, hf_token)
                    text = raw_text
                else:
                    # Local Logic
                    text = transcribe_local(processed_path, asr_model, hf_token)

                latency = time.time() - start_time
                
                # --- AI VERIFICATION & EXTRACTION ---
                # Default values
                final_transcription = text
                entities = []
                
                # Check sidebar toggle (we need to access the widget value indirectly or move the toggle outside function)
                # But since render_order_processing_interface defines it, we can grab it if we returned it?
                # Actually, the variable `enable_ai` is local to render_order_processing_interface.
                # We can just access it here since we are INSIDE render_order_processing_interface.
                
                if enable_ai and hf_token:
                    # 2.5 Verification Animation
                    status_container.markdown('''
                        <div class="processing-overlay">
                            <div class="wave-container">
                                <div class="wave-bar" style="background:#a78bfa"></div>
                                <div class="wave-bar" style="background:#a78bfa"></div>
                                <div class="wave-bar" style="background:#a78bfa"></div>
                            </div>
                            <div class="proc-text" style="color:#a78bfa">✨ AI VERIFYING...</div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    try:
                        verifier = TranscriptVerifier(hf_token)
                        # Get known medicines for context
                        possible_meds = db.medicines['medicine_name'].tolist()
                        
                        verification_result = verifier.verify_and_extract(text, possible_meds)
                        
                        if verification_result and 'cleaned_text' in verification_result:
                            final_transcription = verification_result['cleaned_text']
                            # The Verifier returns 'entities' in the dict
                            # We need to ensure they align with what we expect
                            entities = verification_result.get('entities', [])
                        else:
                            # Fallback if AI fails
                            st.warning("AI Verification returned empty result, using raw transcript.")
                            entities = extractor.extract(text)
                            
                    except Exception as e:
                        st.error(f"AI Verification Failed: {e}")
                        entities = extractor.extract(text)
                else:
                    # Standard Extraction
                    entities = extractor.extract(text)
                
                # Save Verified Text
                st.session_state.last_transcription = final_transcription
                
                # --- ORDER QUEUE LOGIC ---
                with col2:
                     # Extraction Animation
                    extraction_status = st.empty()
                    extraction_status.markdown('''
                        <div style="text-align: center; margin: 20px 0; color: #a855f7;">
                            <span style="display: inline-block; animation: pulse 1s infinite;">💊</span>
                            <span style="margin-left: 8px; font-weight: 500;">Extracting Medicines...</span>
                        </div>
                    ''', unsafe_allow_html=True)
                    time.sleep(0.5)
                    extraction_status.empty()

                if entities:
                    # Route each entity
                    from datetime import datetime
                    for entity in entities:
                        # Lookup manufacturer
                        # Only lookup if manufacturer not already provided by AI (though verifier currently doesn't provide it)
                        mfr_info = db.get_manufacturer_by_medicine(entity.get('medicine', ''))
                        if mfr_info:
                            entity['manufacturer'] = mfr_info.get('name', 'Unknown')
                            entity['medicine_standardized'] = mfr_info.get('medicine_match', entity['medicine'])
                        else:
                            entity['manufacturer'] = 'Unknown'
                            entity['medicine_standardized'] = entity.get('medicine', '')
                        
                        # Ensure Quantity is formatted
                        if 'quantity' not in entity:
                            entity['quantity'] = '1 units'
                            
                        # Add metadata
                        entity['status'] = '✓'
                        entity['priority'] = 'Normal'
                        entity['timestamp'] = datetime.now().strftime('%H:%M')
                    
                    # Update Session State
                    st.session_state.orders.extend(entities)
                    st.toast(f"✅ Extracted {len(entities)} items", icon="💊")
                else:
                    st.warning("No medicines detected in speech.")
                
                # Clear status
                status_container.empty()
                st.toast(f"✅ Processed in {latency:.2f}s", icon="⚡")
                        mfr_info = db.get_manufacturer_by_medicine(entity.get('medicine', ''))
                        if mfr_info:
                            entity['manufacturer'] = mfr_info.get('name', 'Unknown')
                            entity['medicine_standardized'] = mfr_info.get('medicine_match', entity['medicine'])
                        else:
                            entity['manufacturer'] = 'Unknown'
                            entity['medicine_standardized'] = entity.get('medicine', '')
                        
                        # Add metadata
                        entity['status'] = '✓'
                        entity['priority'] = 'Normal'
                        entity['timestamp'] = datetime.now().strftime('%H:%M')
                    
                    # Update Session State
                    st.session_state.orders.extend(entities)
                    st.toast(f"✅ Extracted {len(entities)} items", icon="💊")
                else:
                    st.warning("No medicines detected in speech.")
                
                # Clear status
                status_container.empty()
                st.toast(f"✅ Processed in {latency:.2f}s", icon="⚡")
                
            except Exception as e:
                status_container.error(f"Error: {e}")
                st.stop()
    
    # Display Output (Right Column) - Pure Rendering
    with col2:
        if st.session_state.last_transcription:
            # Transcription Box
            st.markdown('<div class="transcription-box">', unsafe_allow_html=True)
            st.caption("📝 Verified Transcript")
            st.markdown(f'<p style="font-size: 1.1rem; color: #e2e8f0;">{st.session_state.last_transcription}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # (Extraction logic moved up)
            
            # Orders Table
            if st.session_state.orders:
                st.markdown("### 📦 Active Orders")
                
                # Convert to DataFrame for display
                df = pd.DataFrame(st.session_state.orders)
                
                # Reorder columns for UI
                cols = ["manufacturer", "medicine", "quantity", "status", "priority", "timestamp"]
                # Filter columns that exist
                display_cols = [c for c in cols if c in df.columns]
                
                st.dataframe(
                    df[display_cols],
                    use_container_width=True,
                    column_config={
                        "manufacturer": st.column_config.TextColumn("🏭 Manufacturer", width="medium"),
                        "medicine": st.column_config.TextColumn("💊 Medicine", width="medium"),
                        "quantity": st.column_config.TextColumn("🔢 Qty", width="small"),
                        "status": st.column_config.TextColumn("OK", width="small"),
                        "priority": st.column_config.TextColumn("⚡ Priority", width="small"),
                        "timestamp": st.column_config.TextColumn("🕒 Time", width="small"),
                    },
                    height=200
                )
        
        # --- Audio Comparison Feature ---
        if st.session_state.get('original_audio_bytes') and st.session_state.get('last_processed_audio'):
            with st.expander("🔊 Compare Audio: Original vs Processed", expanded=False):
                st.markdown("""
                    <style>
                    .audio-compare-label {
                        font-size: 0.85rem;
                        font-weight: 600;
                        margin-bottom: 5px;
                        color: #8b95a5;
                    }
                    .audio-compare-box {
                        background: rgba(0,0,0,0.2);
                        border-radius: 8px;
                        padding: 10px;
                        margin-bottom: 10px;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                col_orig, col_clean = st.columns(2)
                
                with col_orig:
                    st.markdown('<div class="audio-compare-label">🎤 Original Recording</div>', unsafe_allow_html=True)
                    st.markdown('<div class="audio-compare-box">', unsafe_allow_html=True)
                    st.audio(st.session_state.original_audio_bytes, format="audio/wav")
                    st.caption("Raw audio with background noise & silence")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_clean:
                    st.markdown('<div class="audio-compare-label">✨ Processed (Cleaned)</div>', unsafe_allow_html=True)
                    st.markdown('<div class="audio-compare-box">', unsafe_allow_html=True)
                    st.audio(st.session_state.last_processed_audio, format="audio/wav")
                    st.caption("Silence removed + noise reduced")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.info("💡 The processed audio removes silence and background noise for faster, more accurate transcription.")

    # --- Manufacturer Routing Grid (Moved from app.py) ---
    with col2:
        st.markdown("### 🏭 Manufacturer Routing")
        
        # Get grouped orders
        from simulation.order_queue import OrderQueue
        queue = OrderQueue()
        grouped_orders = queue.get_grouped_orders(db)
        all_manufacturers = db.get_all_manufacturers()
        
        # Grid Layout
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
                
                # Generate HTML
                html_parts = []
                html_parts.append(f'<div class="node-card {active_class}">')
                html_parts.append('<div class="node-header">')
                html_parts.append(f'<span class="node-title"><span style="opacity:0.7">🏭</span> {mfr_name}</span>')
                html_parts.append(f'<span class="node-badge {badge_class}">{order_count} Items</span>')
                html_parts.append('</div><div class="node-body">')
                
                if is_active:
                    for order in orders:
                        conf = order.get('confidence', 0)
                        conf_class = "conf-low"
                        if conf >= 90:
                            conf_class = "conf-high"
                        elif conf >= 75:
                            conf_class = "conf-med"
                        
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
                    html_parts.append('<div class="text-muted">Waiting for data...</div>')
                    
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

    # Footer Information
    render_footer()

def render_footer():
    """Render the informational footer."""
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
