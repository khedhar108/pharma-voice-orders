"""
Evaluation UI Component for Streamlit App

Displays metrics dashboard with confusion matrix, precision/recall/F1,
and comparison table for entity extraction evaluation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict
import json
import time


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
    from evaluation.entity_evaluator import EntityEvaluator, EvaluationReport
    
    st.markdown("## 📊 Evaluation Dashboard")
    st.markdown("Test your model's entity extraction accuracy against ground truth data.")
    
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
        except:
            gt_has_data = False
    
    # Setup Section
    with st.expander("⚙️ Setup Ground Truth", expanded=not gt_has_data):
        st.markdown("""
        **To evaluate your model, you need ground truth data.**
        
        1. Listen to each audio file in `audioData/`
        2. Add rows to `evaluation/ground_truth.csv` with expected medicines
        3. Run evaluation to compare extracted vs expected
        """)
        
        # Show template
        st.code("""# Ground Truth CSV Format
audio_file,order_index,medicine_name,quantity,dosage,form
R_001.m4a,1,DOLO-650,50 strips,650mg,tablet
R_001.m4a,2,AZILIDE-500MG,10 packs,500mg,tablet""", language="csv")
        
        # Ground truth editor
        if gt_has_data:
            st.success(f"✅ Ground truth loaded: {len(gt_df)} entries")
            st.dataframe(gt_df, use_container_width=True, height=200)
        else:
            st.warning("⚠️ No ground truth data. Add entries to `evaluation/ground_truth.csv`")
    
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
        status_text = st.empty()
        
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
                    with open(str(audio_path), 'rb') as f:
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
        progress_placeholder.markdown(f'''
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
    if st.session_state.eval_results:
        st.markdown("---")
        st.markdown("### 📈 Evaluation Results")
        
        # Metrics Cards (if report available)
        if st.session_state.eval_report:
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
