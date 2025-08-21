import os
import json
import argparse
import random
import torch
import soundfile as sf
import numpy as np
import pandas as pd
import librosa

from datetime import datetime

# Project imports
from src import config
from src import audio          # must expose get_mel_spectrogram (same params as training)
from src.mel_griffinlim import mel_to_audio, approximate_linear01_to_log
from src.text import text_to_sequence  # optional (just for completeness)

def _mel_stats(mel: torch.Tensor):
    flat = mel.view(-1).float()
    with torch.no_grad():
        q = torch.quantile(flat, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=flat.device)).cpu().numpy()
        return {
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "mean": float(flat.mean().item()),
            "std": float(flat.std().item()),
            "p01": float(q[0]), "p05": float(q[1]),
            "p50": float(q[2]), "p95": float(q[3]), "p99": float(q[4])
        }

def _scale_interpretation(stats):
    linear_like = stats["min"] >= -1e-4 and 0.0 <= stats["max"] <= 1.05
    narrow_dyn = (stats["max"] - stats["min"]) < 1.2
    if linear_like and narrow_dyn:
        return "LIKELY_LINEAR_0_1"
    if stats["min"] < -0.5:
        return "LIKELY_LOG"
    return "AMBIGUOUS"

def _prepare_mel_for_griffin_lim(mel: torch.Tensor, scale_guess: str) -> torch.Tensor:
    """Return a linear(power) mel suitable for librosa.feature.inverse.mel_to_audio.

    Heuristics:
    - LIKELY_LINEAR_0_1: assume values are normalized dB: mel_norm = (db + 80)/80 where db in [-80,0].
      We undo: db = mel_norm*80 - 80, then convert dB -> power.
    - LIKELY_LOG: assume natural log amplitude / power (values typically ~[-11,0]); invert with exp().
    - AMBIGUOUS: return as-is (user can inspect result).
    """
    if scale_guess == "LIKELY_LINEAR_0_1":
        # Undo normalization back to dB scale then to power
        mel_db = mel * 80.0 - 80.0  # [-80,0]
        # librosa.db_to_power expects numpy
        mel_power = librosa.db_to_power(mel_db.detach().cpu().numpy(), ref=1.0)
        return torch.from_numpy(mel_power).to(mel.device)
    if scale_guess == "LIKELY_LOG":
        # Assume natural log magnitude or power. exp restores linear magnitude/power-ish.
        mel_linear = torch.exp(mel)
        # If these are log-magnitude, exp(mel) is magnitude; Griffin-Lim expects power.
        # Squaring is risky; leave as magnitude (power=1.0 arg will interpret array as mel power).
        return mel_linear
    return mel

def _load_processed_mel(mels_dir, basename):
    pt_path = os.path.join(mels_dir, f"{basename}.pt")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Processed mel not found: {pt_path}")
    mel = torch.load(pt_path)
    if mel.dim() != 2:
        raise ValueError(f"Mel tensor expected 2-D, got shape {tuple(mel.shape)}")
    # Expect (n_mels, T)
    if mel.shape[0] != config.N_MELS and mel.shape[1] == config.N_MELS:
        mel = mel.transpose(0, 1)
    return mel

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _load_hifigan(device):
    print("Loading HiFi-GAN (NVIDIA TorchHub)...")
    hifi_gan_tuple = torch.hub.load(
        'nvidia/DeepLearningExamples:torchhub',
        'nvidia_hifigan',
        pretrained=False,
        trust_repo=True
    )
    generator = hifi_gan_tuple[0].to(device)
    ckpt_url = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz/versions/21.08.0_amp/files/hifigan_gen_checkpoint_10000_ft.pt"
    ckpt_file = "hifigan_checkpoint.pt"
    if not os.path.exists(ckpt_file):
        from torch.hub import download_url_to_file
        download_url_to_file(ckpt_url, ckpt_file)
    state_dict = torch.load(ckpt_file, map_location=device)
    generator.load_state_dict(state_dict['generator'])
    generator.eval()
    print("HiFi-GAN loaded.")
    return generator

def _write_wav(path, audio_tensor):
    audio_np = audio_tensor.detach().cpu().numpy()
    sf.write(path, audio_np, config.SAMPLING_RATE)
    print(f"Saved: {path}")

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )
    print(f"Device: {device}")

    df = pd.read_csv(args.metadata)
    if "filepath" not in df.columns or "text" not in df.columns:
        raise ValueError("Metadata must contain 'filepath' and 'text' columns.")
    total = len(df)
    if total == 0:
        raise ValueError("Empty metadata.")

    # Pick sample
    if args.index is not None:
        if args.index < 0 or args.index >= total:
            raise IndexError(f"--index out of range (0..{total-1})")
        row = df.iloc[args.index]
    else:
        row = df.iloc[random.randint(0, total - 1)]
    wav_path = row["filepath"]
    text = row["text"]
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    print(f"Selected sample: index={df.index[df['filepath']==wav_path][0]} basename={basename}")

    # Prepare output directory
    out_dir = _ensure_dir(args.output_dir)
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "wav_path": wav_path,
        "text": text,
        "basename": basename
    }

    # Load processed mel (if provided)
    processed_mel = None
    if args.processed_root:
        mels_dir = os.path.join(args.processed_root, "mels")
        if os.path.isdir(mels_dir):
            try:
                processed_mel = _load_processed_mel(mels_dir, basename)
                print("Loaded processed mel.")
            except Exception as e:
                print(f"Could not load processed mel: {e}")
        else:
            print(f"Processed mels dir not found: {mels_dir}")

    # Recompute mel from raw wav (authoritative)
    try:
        recomputed_mel = audio.get_mel_spectrogram(wav_path)
        if not isinstance(recomputed_mel, torch.Tensor):
            recomputed_mel = torch.from_numpy(recomputed_mel)
        # Ensure shape (n_mels, T)
        if recomputed_mel.shape[0] != config.N_MELS and recomputed_mel.shape[1] == config.N_MELS:
            recomputed_mel = recomputed_mel.transpose(0, 1)
    except Exception as e:
        raise RuntimeError(f"Failed to recompute mel: {e}")

    # Stats
    if processed_mel is not None:
        proc_stats = _mel_stats(processed_mel)
        proc_scale = _scale_interpretation(proc_stats)
        print(f"[PROC MEL] stats={proc_stats} scale_guess={proc_scale}")
        report["processed_mel_stats"] = proc_stats
        report["processed_mel_scale_guess"] = proc_scale

    rec_stats = _mel_stats(recomputed_mel)
    rec_scale = _scale_interpretation(rec_stats)
    print(f"[RECOMP MEL] stats={rec_stats} scale_guess={rec_scale}")
    report["recomputed_mel_stats"] = rec_stats
    report["recomputed_mel_scale_guess"] = rec_scale

    # Griffin-Lim on recomputed mel (convert if needed)
    print("Preparing mel for Griffin-Lim (scale guess: %s)" % rec_scale)
    mel_for_gl = _prepare_mel_for_griffin_lim(recomputed_mel, rec_scale)
    print("Running Griffin-Lim on prepared mel...")
    wav_gl = mel_to_audio(mel_for_gl, n_iter=args.gl_iters)
    _write_wav(os.path.join(out_dir, f"{basename}_gt_griffinlim.wav"), wav_gl)

    # If linear 0–1 and user wants experiment with heuristic pseudo-log
    if rec_scale == "LIKELY_LINEAR_0_1" and args.try_pseudo_log:
        pseudo_log = approximate_linear01_to_log(recomputed_mel)
        # Shift pseudo-log into a log-like magnitude exp() domain for Griffin-Lim test
        # This is heuristic; mostly for relative timbre comparison.
        # Convert back to "linear-ish" magnitude
        pseudo_lin = torch.exp(pseudo_log)
        print("Running Griffin-Lim on pseudo-log transformed mel...")
        wav_pseudo = mel_to_audio(pseudo_lin, n_iter=args.gl_iters)
        _write_wav(os.path.join(out_dir, f"{basename}_gt_griffinlim_pseudolog.wav"), wav_pseudo)

    # Optional HiFi-GAN
    if args.hifigan:
        try:
            hifigan = _load_hifigan(device)
            # Need shape (B, T, n_mels)
            mel_bt_tn = recomputed_mel.transpose(0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                wav_voc = hifigan(mel_bt_tn).squeeze(0).cpu()
            _write_wav(os.path.join(out_dir, f"{basename}_gt_hifigan.wav"), wav_voc)
        except Exception as e:
            print(f"HiFi-GAN synthesis failed: {e}")
            report["hifigan_error"] = str(e)

    # Save JSON report
    report_path = os.path.join(out_dir, f"{basename}_vocoder_check.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground-truth vocoder sanity check.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to processed metadata.csv (with filepath,text).")
    parser.add_argument("--processed_root", type=str, default=None, help="Root of processed dataset (expects mels/).")
    parser.add_argument("--index", type=int, default=None, help="Specific sample index (default random).")
    parser.add_argument("--output_dir", type=str, default="gt_vocoder_check", help="Directory for outputs.")
    parser.add_argument("--hifigan", action="store_true", help="Also synthesize with pretrained HiFi-GAN.")
    parser.add_argument("--gl_iters", type=int, default=60, help="Griffin-Lim iterations.")
    parser.add_argument("--try_pseudo_log", action="store_true", help="Run heuristic pseudo-log experiment if mel looks linear.")
    args = parser.parse_args()
    main(args)
