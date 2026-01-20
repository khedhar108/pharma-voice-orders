import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import io

class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        
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

        # Noise Reduction (Spectral Gating)
        # Only apply if audio is long enough to have a noise profile
        if len(audio) > self.target_sr * 0.5:
            audio = nr.reduce_noise(y=audio, sr=self.target_sr, stationary=True)

        # Normalization
        audio = librosa.util.normalize(audio)
        
        return audio
