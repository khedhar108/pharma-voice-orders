"""
Evaluation UI Component for Streamlit App

Displays metrics dashboard with confusion matrix, precision/recall/F1,
and comparison table for entity extraction evaluation.
"""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st


def render_demo_mode():
    """
    Render the Demo Mode evaluation dashboard with pre-computed metrics.
    This bypasses the actual evaluation pipeline and shows showcase data.
    """
    demo_dir = Path("evaluation/demo")
    metrics_path = demo_dir / "demo_metrics.json"
    
    if not metrics_path.exists():
        st.error("❌ Demo data not found. Please ensure `evaluation/demo/demo_metrics.json` exists.")
        return
    
    # Load pre-computed metrics
    with open(metrics_path, 'r') as f:
        demo_data = json.load(f)
    
    metrics = demo_data
    confusion = metrics['confusion_matrix']
    summary = metrics['summary']
    
    st.markdown("---")
    st.markdown("### 📈 Demo Evaluation Results")
    st.caption("🎭 **Demo Mode Active** — Displaying pre-computed metrics for demonstration purposes")
    
    # ===== 5.1 ASR Accuracy Section =====
    st.markdown("#### 🎤 5.1 ASR Accuracy")
    st.markdown("The system was tested with varying accents and background noise levels:")
    
    asr = metrics.get('asr_accuracy', {})
    col_asr1, col_asr2 = st.columns(2)
    with col_asr1:
        whisper_med = asr.get('whisper_medium', {})
        st.metric("Whisper Medium", f"{whisper_med.get('accuracy', 90)}%+")
        st.caption(whisper_med.get('description', 'High accuracy with balanced inference speed'))
    with col_asr2:
        whisper_large = asr.get('whisper_large_v3_turbo', {})
        st.metric("Whisper Large-v3-Turbo", f"{whisper_large.get('accuracy', 92)}%+")
        st.caption(whisper_large.get('description', 'Best accuracy for complex medical terminology'))
    
    # ===== 5.2 Routing Precision Section =====
    st.markdown("---")
    st.markdown("#### 🎯 5.2 Routing Precision")
    
    routing_prec = metrics.get('routing_precision', {})
    st.markdown(f"The fuzzy matching threshold was tuned to **{routing_prec.get('fuzzy_matching_threshold', 75)}%**:")
    
    col_rp1, col_rp2, col_rp3 = st.columns(3)
    with col_rp1:
        st.metric("True Positives", f"{routing_prec.get('true_positives', 92)}/100", delta="correct mappings")
    with col_rp2:
        st.metric("False Positives", f"{routing_prec.get('false_positives', 3)}/100", delta="similar sounding", delta_color="inverse")
    with col_rp3:
        st.metric("Unmapped (Quarantine)", f"{routing_prec.get('unmapped_quarantine', 5)}/100", delta="correctly routed")
    
    # ===== 5.3 Matching Logic Table =====
    st.markdown("---")
    st.markdown("#### 📐 5.3.1 Matching Logic")
    
    matching = metrics.get('matching_logic', {})
    matching_df = pd.DataFrame({
        "Field": ["Medicine", "Quantity", "Dosage"],
        "Match Criteria": [
            matching.get('medicine', {}).get('match_criteria', 'Fuzzy match (ratio or partial_ratio)'),
            matching.get('quantity', {}).get('match_criteria', 'Number match + unit similarity'),
            matching.get('dosage', {}).get('match_criteria', 'Normalized text match')
        ],
        "Threshold": [
            matching.get('medicine', {}).get('threshold', '>=85%'),
            matching.get('quantity', {}).get('threshold', '>=80%'),
            matching.get('dosage', {}).get('threshold', 'Exact')
        ],
        "Example": [
            matching.get('medicine', {}).get('example', '"DOLO 650" = "DOLO-650"'),
            matching.get('quantity', {}).get('example', '"50 strips" = "50 strip"'),
            matching.get('dosage', {}).get('example', '"500 mg" = "500mg"')
        ]
    })
    
    st.dataframe(matching_df, use_container_width=True, hide_index=True)
    
    # Top-level metrics cards
    st.markdown("---")
    st.markdown("#### 📊 Overall Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Expected", summary['total_expected'])
    with col2:
        st.metric("Total Extracted", summary['total_extracted'])
    with col3:
        medicine_f1 = confusion['medicine']['f1_score']
        st.metric("Medicine F1", f"{medicine_f1}%", 
                 delta="Excellent" if medicine_f1 >= 85 else ("Good" if medicine_f1 >= 70 else "Needs Work"))
    with col4:
        overall_acc = summary['overall_accuracy']
        st.metric("Overall Accuracy", f"{overall_acc}%",
                 delta="Excellent" if overall_acc >= 85 else ("Good" if overall_acc >= 70 else "Needs Work"))
    
    st.markdown("---")
    
    # Per-Field Metrics Table (Confusion Matrix Style)
    st.markdown("#### 📊 Per-Field Confusion Matrix")
    
    metrics_df = pd.DataFrame({
        "Field": ["Medicine Name", "Quantity", "Dosage"],
        "True Positive (✅)": [confusion['medicine']['true_positive'], confusion['quantity']['true_positive'], confusion['dosage']['true_positive']],
        "False Negative (❌)": [confusion['medicine']['false_negative'], confusion['quantity']['false_negative'], confusion['dosage']['false_negative']],
        "False Positive (⚠️)": [confusion['medicine']['false_positive'], confusion['quantity']['false_positive'], confusion['dosage']['false_positive']],
        "Precision %": [confusion['medicine']['precision'], confusion['quantity']['precision'], confusion['dosage']['precision']],
        "Recall %": [confusion['medicine']['recall'], confusion['quantity']['recall'], confusion['dosage']['recall']],
        "F1 Score %": [confusion['medicine']['f1_score'], confusion['quantity']['f1_score'], confusion['dosage']['f1_score']],
    })
    
    st.dataframe(
        metrics_df.style.background_gradient(subset=['F1 Score %'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True
    )
    
    # Manufacturer Routing Accuracy
    st.markdown("---")
    st.markdown("#### 🏭 Manufacturer Routing Accuracy")
    
    routing = metrics['manufacturer_routing_accuracy']
    col_r1, col_r2 = st.columns([1, 2])
    
    with col_r1:
        st.metric("Routing Accuracy", f"{routing['routing_accuracy']}%")
        st.metric("Correctly Routed", f"{routing['correctly_routed']} / {routing['total_medicines']}")
    
    with col_r2:
        routing_df = pd.DataFrame(routing['breakdown_by_manufacturer'])
        st.dataframe(
            routing_df.style.background_gradient(subset=['accuracy'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True,
            height=200
        )
    
    # Audio File Breakdown
    st.markdown("---")
    st.markdown("#### 🎵 Per-Audio File Performance")
    
    audio_df = pd.DataFrame(metrics['audio_file_breakdown'])
    st.dataframe(
        audio_df.style.background_gradient(subset=['accuracy'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True
    )
    
    # Error Analysis
    st.markdown("---")
    st.markdown("#### ⚠️ Error Analysis")
    
    error_data = metrics['error_analysis']['common_errors']
    for error_type in error_data:
        with st.expander(f"**{error_type['type']}** ({error_type['count']} occurrences)"):
            for example in error_type['examples']:
                st.json(example)
    
    # Detailed Comparison Table
    st.markdown("---")
    st.markdown("#### 🔍 Detailed Comparison Table")
    
    comparison_df = pd.DataFrame(metrics['detailed_comparison'])
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        match_filter = st.selectbox("Filter by match type", ["All", "TP (Correct)", "FN (Missed)", "FP (Extra)"], key="demo_match_filter")
    with filter_col2:
        file_filter = st.selectbox("Filter by file", ["All"] + list(comparison_df['audio_file'].unique()), key="demo_file_filter")
    
    # Apply filters
    filtered_df = comparison_df.copy()
    if match_filter != "All":
        match_type = match_filter.split(" ")[0]
        filtered_df = filtered_df[filtered_df['match_type'] == match_type]
    if file_filter != "All":
        filtered_df = filtered_df[filtered_df['audio_file'] == file_filter]
    
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Export buttons
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_data = comparison_df.to_csv(index=False)
        st.download_button(
            "📥 Download Comparison CSV",
            data=csv_data,
            file_name="demo_evaluation_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        report_json = json.dumps(metrics, indent=2)
        st.download_button(
            "📄 Download Full Metrics JSON",
            data=report_json,
            file_name="demo_evaluation_metrics.json",
            mime="application/json",
            use_container_width=True
        )


def render_evaluation_tab(
    db,
    preprocessor,
    extractor,
    transcribe_fn,
    use_cloud: bool = False,
    model_name: str = "openai/whisper-medium",
    hf_token: str = None
):
    """
    Render the Evaluation tab UI.
    
    Args:
        db: ManufacturerDB instance
        preprocessor: AudioPreprocessor instance
        extractor: EntityExtractor instance
        transcribe_fn: Function to transcribe audio (local or cloud)
        use_cloud: Whether using cloud inference
        model_name: Selected ASR model
        hf_token: HuggingFace token
    """
    from evaluation.entity_evaluator import EntityEvaluator
    
    st.markdown("## 📊 Evaluation Dashboard")
    st.markdown("Test your model's entity extraction accuracy against ground truth data.")
    
    # Demo Mode Toggle
    st.markdown("---")
    demo_mode = st.toggle(
        "🎭 **Demo Mode** — Use pre-computed showcase data",
        value=True,
        help="Enable to display pre-computed evaluation metrics without running the actual pipeline. Disable to run live evaluation on your audio files."
    )
    
    if demo_mode:
        render_demo_mode()
        return
    
    # Initialize evaluator
    evaluator = EntityEvaluator()
    
    # Ground truth file path
    gt_path = Path("evaluation/ground_truth.csv")
    audio_dir = Path("audioData")
    
    # Check if ground truth exists and has data
    gt_exists = gt_path.exists()
    gt_has_data = False
    gt_df = None
    
    if gt_exists:
        try:
            gt_df = pd.read_csv(gt_path, comment='#')
            gt_has_data = len(gt_df) > 0 and 'medicine_name' in gt_df.columns
        except Exception:
            gt_has_data = False
    
    # Setup Section
    # Setup Section
    with st.expander("⚙️ Setup Ground Truth", expanded=not gt_has_data):
        st.markdown("""
        **To evaluate your model, you need ground truth data.**
        
        1. Listen to each audio file in `audioData/`
        2. Add rows to the table below with expected medicines
        3. Click **Save Ground Truth**
        """)
        
        # Default Template Data
        default_data = {
            "audio_file": ["R_001.m4a", "R_001.m4a"],
            "order_index": [1, 2],
            "medicine_name": ["DOLO-650", "AZILIDE-500MG"],
            "quantity": ["50 strips", "10 packs"],
            "dosage": ["650mg", "500mg"],
            "form": ["tablet", "tablet"]
        }
        
        # Load editor data
        if gt_has_data:
            editor_df = gt_df.copy()
        else:
            # If no data, start with empty structure or default if requested
            editor_df = pd.DataFrame(columns=["audio_file", "order_index", "medicine_name", "quantity", "dosage", "form"])
            
        # Action Buttons Row
        col_act1, col_act2, col_act3 = st.columns([1, 1, 2])
        
        with col_act1:
            if st.button("📝 Load Template"):
                editor_df = pd.DataFrame(default_data)
                # We need to rerun to show this in editor, or we rely on session state?
                # st.data_editor keys off value, but doesn't update if value changes externally easily without key change.
                # Let's just save it to a temp state or rewrite file for "Load Template".
                # Actually, simplest is just to write default CSV and rerun?
                pd.DataFrame(default_data).to_csv(gt_path, index=False)
                st.rerun()

        # The Data Editor
        edited_df = st.data_editor(
            editor_df,
            num_rows="dynamic",
            use_container_width=True,
            key="gt_editor",
            column_config={
                "audio_file": st.column_config.SelectboxColumn(
                    "Audio File",
                    options=sorted([f.name for f in audio_dir.iterdir() if f.suffix in ['.m4a', '.wav', '.mp3']]) if audio_dir.exists() else [],
                    required=True
                ),
                "order_index": st.column_config.NumberColumn("Index", min_value=1, step=1, default=1),
                "medicine_name": st.column_config.TextColumn("Medicine", required=True),
                "quantity": "Quantity",
                "dosage": "Dosage",
                "form": st.column_config.SelectboxColumn(
                    "Form",
                    options=["tablet", "syrup", "cream", "injection", "spray", "drops"],
                    default="tablet"
                )
            }
        )
        
        with col_act2:
            if st.button("💾 Save Ground Truth", type="primary"):
                # Save to CSV
                try:
                    # Create directory if likely executing locally and it's missing (though evaluation folder exists)
                    gt_path.parent.mkdir(exist_ok=True, parents=True)
                    edited_df.to_csv(gt_path, index=False)
                    st.success("Saved!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")

        with col_act3:
             # Download Button
            csv_csv = edited_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV (for backup)",
                data=csv_csv,
                file_name="ground_truth.csv",
                mime="text/csv"
            )

        if not gt_has_data:
             st.warning("⚠️ No ground truth data saved yet. Load Template or add rows above.")
        else:
             st.success(f"✅ {len(gt_df)} rows loaded from disk.")
    
    # Audio files section
    st.markdown("---")
    st.markdown("### 🎵 Select Audio Files")
    
    audio_files = []
    if audio_dir.exists():
        audio_files = sorted([f.name for f in audio_dir.iterdir() if f.suffix in ['.m4a', '.wav', '.mp3', '.ogg']])
    
    if not audio_files:
        st.warning("No audio files found in `audioData/` folder.")
        return
    
    # File selection
    selected_files = st.multiselect(
        "Select files to evaluate",
        audio_files,
        default=audio_files[:5] if len(audio_files) >= 5 else audio_files,
        help="Select audio files to run through the evaluation pipeline"
    )
    
    if not selected_files:
        st.info("Please select at least one audio file.")
        return
    
    # Run Evaluation Button
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        run_eval = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)
    with col2:
        st.metric("Files", len(selected_files))
    
    # Store results in session state
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'eval_report' not in st.session_state:
        st.session_state.eval_report = None
    
    if run_eval:
        all_extracted = []
        
        # Custom Progress
        progress_placeholder = st.empty()

        
        # Process each audio file
        total_files = len(selected_files)
        for i, filename in enumerate(selected_files):
            # Update Custom Progress Bar
            progress_pct = int((i / total_files) * 100)
            progress_placeholder.markdown(f'''
                <div class="neon-progress-container">
                    <div class="neon-progress-fill" style="width: {progress_pct}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #8b95a5; margin-top: 4px;">
                    <span>Processing: <span style="color: #00f2ea;">{filename}</span></span>
                    <span>{progress_pct}%</span>
                </div>
            ''', unsafe_allow_html=True)
            
            # status_text.text(f"Processing {filename}...") # Replaced by custom UI above
            audio_path = audio_dir / filename
            
            try:
                # Preprocess audio
                processed_audio = preprocessor.preprocess_file(str(audio_path))
                
                # Transcribe
                if use_cloud:
                    from huggingface_hub import InferenceClient
                    client = InferenceClient(token=hf_token)
                    # Use preprocessed audio (silence-trimmed) for cloud inference
                    with open(processed_audio, 'rb') as f:
                        result = client.automatic_speech_recognition(audio=f.read(), model=model_name)
                    transcription = result if isinstance(result, str) else result.get("text", "")
                else:
                    # Use local transcription
                    transcription = transcribe_fn(processed_audio, model_name, hf_token)
                
                # Extract entities
                entities = extractor.extract(transcription)
                
                # Add filename to each entity
                for entity in entities:
                    entity['audio_file'] = filename
                    entity['transcription'] = transcription
                    all_extracted.append(entity)
                    
            except Exception as e:
                st.error(f"Error processing {filename}: {e}")
            
            # Update progress end of loop
            # progress_bar.progress((i + 1) / len(selected_files)) # Removed standard bar
        
        # Final 100% state
        progress_placeholder.markdown('''
            <div class="neon-progress-container">
                <div class="neon-progress-fill" style="width: 100%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #8b95a5; margin-top: 4px;">
                <span style="color: #00f260;">✅ Evaluation Complete</span>
                <span>100%</span>
            </div>
        ''', unsafe_allow_html=True)
        
        time.sleep(0.5) # Pause to show 100%
        # status_text.text("Evaluation complete!") # Handled by UI above
        progress_placeholder.empty()
        
        # If ground truth exists, compute metrics
        if gt_has_data:
            # Filter ground truth to selected files
            gt_filtered = gt_df[gt_df['audio_file'].isin(selected_files)]
            
            if len(gt_filtered) > 0:
                report = evaluator.evaluate(gt_filtered, all_extracted)
                st.session_state.eval_report = report
                st.session_state.eval_results = all_extracted
            else:
                st.warning("No ground truth entries for selected files.")
                st.session_state.eval_results = all_extracted
        else:
            st.session_state.eval_results = all_extracted
        
        st.rerun()
    
    # Display Results
    if st.session_state.eval_results is not None:
        st.markdown("---")
        st.markdown("### 📈 Evaluation Results")
        
        # Metrics Cards (if report available)
        if st.session_state.eval_report is not None:
            report = st.session_state.eval_report
            metrics = report.to_dict()
            
            # Top-level metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Expected", metrics['total_expected'])
            with col2:
                st.metric("Total Extracted", metrics['total_extracted'])
            with col3:
                medicine_f1 = metrics['medicine']['f1']
                st.metric("Medicine F1", f"{medicine_f1}%", 
                         delta="Good" if medicine_f1 >= 80 else "Needs Work")
            with col4:
                overall_acc = (metrics['medicine']['tp'] / max(metrics['total_expected'], 1)) * 100
                st.metric("Accuracy", f"{overall_acc:.1f}%")
            
            st.markdown("---")
            
            # Confusion Matrix Style Table
            st.markdown("#### 📊 Per-Field Metrics")
            
            metrics_df = pd.DataFrame({
                "Field": ["Medicine Name", "Quantity", "Dosage"],
                "True Positive (✅)": [metrics['medicine']['tp'], metrics['quantity']['tp'], metrics['dosage']['tp']],
                "False Negative (❌)": [metrics['medicine']['fn'], metrics['quantity']['fn'], metrics['dosage']['fn']],
                "False Positive (⚠️)": [metrics['medicine']['fp'], metrics['quantity']['fp'], metrics['dosage']['fp']],
                "Precision %": [metrics['medicine']['precision'], metrics['quantity']['precision'], metrics['dosage']['precision']],
                "Recall %": [metrics['medicine']['recall'], metrics['quantity']['recall'], metrics['dosage']['recall']],
                "F1 Score %": [metrics['medicine']['f1'], metrics['quantity']['f1'], metrics['dosage']['f1']],
            })
            
            st.dataframe(
                metrics_df.style.background_gradient(subset=['F1 Score %'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )
            
            # Comparison Table
            st.markdown("---")
            st.markdown("#### 🔍 Detailed Comparison")
            
            comparison_df = evaluator.generate_comparison_table(report)
            
            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                match_filter = st.selectbox("Filter by match type", ["All", "TP (Correct)", "FN (Missed)", "FP (Extra)"])
            with filter_col2:
                file_filter = st.selectbox("Filter by file", ["All"] + list(comparison_df['audio_file'].unique()))
            
            # Apply filters
            filtered_df = comparison_df.copy()
            if match_filter != "All":
                match_type = match_filter.split(" ")[0]
                filtered_df = filtered_df[filtered_df['match_type'] == match_type]
            if file_filter != "All":
                filtered_df = filtered_df[filtered_df['audio_file'] == file_filter]
            
            st.dataframe(filtered_df, use_container_width=True, height=300)
            
            # Export buttons
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv_data = comparison_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Comparison CSV",
                    data=csv_data,
                    file_name="evaluation_comparison.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_exp2:
                report_json = json.dumps(metrics, indent=2)
                st.download_button(
                    "📄 Download Metrics JSON",
                    data=report_json,
                    file_name="evaluation_metrics.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        else:
            # No ground truth, just show extracted results
            st.info("No ground truth data to compare. Showing extracted results only.")
            
            results_df = pd.DataFrame(st.session_state.eval_results)
            st.dataframe(results_df, use_container_width=True)
            
            # Export
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Extracted Results",
                data=csv_data,
                file_name="extracted_entities.csv",
                mime="text/csv"
            )
