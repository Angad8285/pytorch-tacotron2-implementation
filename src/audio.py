# src/audio.py

import librosa
import numpy as np
import torch 
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
    
    # Convert the power spectrogram to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return torch.FloatTensor(mel_spec_db)