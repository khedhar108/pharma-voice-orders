import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import io
import tempfile
import os

class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self._temp_dir = tempfile.mkdtemp(prefix="pharma_audio_")
        
    def process(self, audio_file) -> np.ndarray:
        """
        Process audio file (path or bytes) for ASR.
        Returns: 16kHz mono audio array.
        """
        # Load audio (handles both paths and file-like objects)
        try:
            audio, sr = librosa.load(audio_file, sr=self.target_sr, mono=True)
        except Exception as e:
            # Fallback for file-like objects if librosa fails directly
            if hasattr(audio_file, 'read'):
                audio_file.seek(0)
                audio, sr = librosa.load(audio_file, sr=self.target_sr, mono=True)
            else:
                raise e

        # 2. Silence Trimming (VAD - Voice Activity Detection)
        # Remove leading/trailing silence (top_db=30 is standard for speech)
        # This reduces processing time and improves ASR accuracy
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
            # Safety Check: If trim removes everything (empty audio), revert to original
            if len(audio_trimmed) > 0:
                audio = audio_trimmed
        except Exception:
            # Fallback if VAD fails for any reason
            pass

        # Noise Reduction (Spectral Gating)
        # Only apply if audio is long enough to have a noise profile
        if len(audio) > self.target_sr * 0.5:
            audio = nr.reduce_noise(y=audio, sr=self.target_sr, stationary=True)

        # Normalization
        audio = librosa.util.normalize(audio)
        
        return audio

    def preprocess_file(self, audio_path: str) -> str:
        """
        Process audio and save to a temporary file.
        Returns path to processed file (in temp directory to avoid pollution).
        """
        audio = self.process(audio_path)
        
        # Create output path in temp directory to avoid polluting source dir
        base_name = os.path.basename(audio_path).rsplit(".", 1)[0] + "_clean.wav"
        output_path = os.path.join(self._temp_dir, base_name)
        
        # Save
        sf.write(output_path, audio, self.target_sr)
        return output_path
    
    def cleanup(self):
        """Remove temporary files created during preprocessing."""
        import shutil
        try:
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
        except Exception:
            pass
