import torch
import numpy as np
import librosa

from . import config

def mel_to_audio(mel: torch.Tensor, n_iter: int = 60) -> torch.Tensor:
    """Invert a mel spectrogram to waveform via Griffin-Lim.

    Accepts either linear(power) mel or log-mel (natural log of power).
    Heuristic: if any value < -0.5 treat as log-mel and exponentiate first.
    Args:
        mel: (n_mels, T)
        n_iter: Griffin-Lim iterations
    Returns:
        torch.Tensor waveform (T_samples,)
    """
    if isinstance(mel, torch.Tensor):
        mel_local = mel.detach().cpu()
        mel_np = mel_local.numpy()
    else:
        mel_np = np.asarray(mel, dtype=np.float32)
    # Improved scale/orientation diagnostics
    if mel_np.shape[0] < mel_np.shape[1] and mel_np.shape[0] in (config.N_MELS, getattr(config, 'n_mels', config.N_MELS)):
        pass  # expected (n_mels, T)
    elif mel_np.shape[1] in (config.N_MELS, getattr(config, 'n_mels', config.N_MELS)) and mel_np.shape[1] < mel_np.shape[0]:
        # Transposed accidentally (T, n_mels)
        print("[mel_to_audio] Detected transposed mel, correcting orientation.")
        mel_np = mel_np.T
    else:
        # Ambiguous shapes; continue
        pass

    mn = mel_np.min(); mx = mel_np.max()
    dynamic = mx - mn
    is_log_like = (mn < -0.5) or (dynamic > 5.0)
    if is_log_like:
        mel_lin = np.exp(mel_np)  # log-power -> power
    else:
        mel_lin = np.maximum(mel_np, 0.0)  # assume already linear/power
    wav = librosa.feature.inverse.mel_to_audio(
        mel_lin,
        sr=config.SAMPLING_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        n_iter=n_iter,
        power=1.0
    )
    return torch.from_numpy(wav)

def approximate_linear01_to_log(mel_linear01: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Heuristic: convert a 0–1 linear-normalized mel back to a pseudo log scale
    (for experimentation). Assumes original log-mel was min-max scaled.
    This is ONLY a diagnostic helper and may not match real inverse.
    """
    # Stretch slightly to avoid zeros then map to [-4, 0] dB-ish band
    x = mel_linear01.clamp(0.0, 1.0)
    # Map 0..1 -> [-6, 0] (rough dynamic range guess)
    return (-6.0 + 6.0 * x)
