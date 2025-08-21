# src/audio.py

import librosa
import numpy as np
import torch
from . import config  # relative import
from .mel_griffinlim import mel_to_audio as _griffinlim_mel_to_audio

"""Audio feature extraction utilities.

Switched to vocoder-compatible log-mel representation (natural log of mel power).
This replaces the previous dB -> [0,1] normalization which was incompatible with
pretrained HiFi-GAN expectations. New pipeline:

1. Load waveform (resampled to config.SAMPLING_RATE).
2. Compute mel power spectrogram using librosa's validated implementation.
3. Dynamic range compression: log(clamp(mel_power, min=1e-5)).

Returned tensor shape: (n_mels, T), dtype float32.

NOTE: After this change you MUST regenerate any previously processed mels; mixing
old [0,1]-normalized dB mels with new log-mels will break training/inference.
"""

_MEL_EPS = 1e-5

def get_mel_spectrogram(filepath: str) -> torch.Tensor:
    """Compute vocoder-style log-mel spectrogram.

    Returns:
        torch.Tensor: (n_mels, T) log-mel (natural log of mel power).
    """
    y, _ = librosa.load(filepath, sr=config.SAMPLING_RATE)
    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLING_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0,               # power spectrogram (|S|**2)
        center=True
    )  # (n_mels, T)
    mel_power = np.clip(mel_power, _MEL_EPS, None)
    log_mel = np.log(mel_power).astype(np.float32)  # natural log(power)
    return torch.from_numpy(log_mel)


def mel_to_audio(mel: torch.Tensor) -> torch.Tensor:
    """Convert a (n_mels, T) mel (log-power or linear) to waveform via Griffin-Lim.

    Accepts either log-power or linear mel; downstream helper detects scale.
    """
    return _griffinlim_mel_to_audio(mel)

__all__ = ["get_mel_spectrogram", "mel_to_audio"]