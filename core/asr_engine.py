import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import streamlit as st

class ASREngine:
    def __init__(self, model_id: str = "openai/whisper-tiny"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = model_id
        self.pipe = self._load_model()
        
    @st.cache_resource(show_spinner=False)
    def _load_model(_self):
        """Load model with caching to avoid reloading on every run."""
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            _self.model_id, 
            torch_dtype=_self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(_self.device)

        processor = AutoProcessor.from_pretrained(_self.model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=_self.torch_dtype,
            device=_self.device,
        )
        return pipe

    def transcribe(self, audio_array) -> str:
        """Transcribe audio array or path."""
        try:
            result = self.pipe(audio_array)
            return result["text"]
        except Exception as e:
            return f"Error: {str(e)}"
