# src/audio.py

import librosa
import numpy as np
from . import config  # Use a relative import to get settings from config.py

def get_mel_spectrogram(filepath: str) -> np.ndarray:
    """
    Converts an audio file into a log-scale mel spectrogram.
    """
    # Load the audio, resampling to our standard rate
    y, sr = librosa.load(filepath, sr=config.SAMPLING_RATE)
    
    # Compute the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLING_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX
    )
    
    # Convert the power spectrogram to decibels with proper dynamic range
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1] range for stable training
    # Typical mel spectrograms range from -80dB to 0dB
    mel_spec_normalized = (mel_spec_db + 80.0) / 80.0  # Map [-80, 0] to [0, 1]
    mel_spec_normalized = np.clip(mel_spec_normalized, 0.0, 1.0)  # Ensure bounds
    
    return mel_spec_normalized