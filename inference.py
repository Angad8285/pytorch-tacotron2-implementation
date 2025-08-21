# inference.py

import torch
import argparse
import os
from scipy.io.wavfile import write as write_wav
from torch.hub import download_url_to_file
import numpy as np

from src import config
from src.model import Tacotron2
from src.text import text_to_sequence
from src.mel_griffinlim import mel_to_audio as fallback_mel_to_audio

def inference(text, checkpoint_path, output_dir, vocoder):
    """
    Main inference routine.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load Trained Tacotron 2 Model ---
    print("Loading Tacotron 2 model...")
    model = Tacotron2().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Tacotron 2 model loaded.")

    use_hifigan = (vocoder.lower() == "hifigan")

    hifi_gan = None  # type: ignore
    if use_hifigan:
        print("Loading HiFi-GAN vocoder from NVIDIA's repo...")
        hifi_gan_tuple = torch.hub.load(
            'nvidia/DeepLearningExamples:torchhub',
            'nvidia_hifigan',
            pretrained=False,
            trust_repo=True
        )
        hifi_gan = hifi_gan_tuple[0].to(device)  # type: ignore[index]
        ckpt_url = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz/versions/21.08.0_amp/files/hifigan_gen_checkpoint_10000_ft.pt"
        ckpt_file = "hifigan_checkpoint.pt"
        if not os.path.exists(ckpt_file):
            download_url_to_file(ckpt_url, ckpt_file)
        state_dict = torch.load(ckpt_file, map_location=device)
        hifi_gan.load_state_dict(state_dict['generator'])
        hifi_gan.eval()
        print("HiFi-GAN vocoder loaded.")
    else:
        print("Using Griffin-Lim fallback vocoder (no pretrained download).")

    # --- Process Input Text ---
    print("Processing input text...")
    sequence = text_to_sequence(text)
    sequence = torch.LongTensor(sequence).unsqueeze(0).to(device)

    # --- Generate Speech ---
    print("Generating mel spectrogram...")
    with torch.no_grad():
        mel_outputs_postnet, _, _, _ = model.inference(sequence)
        _print_mel_stats(mel_outputs_postnet, "Pred PostNet Mel")
        if use_hifigan and hifi_gan is not None:
            print("Synthesizing waveform with HiFi-GAN...")
            # Model returns (B, T, n_mels); vocoder expects (B, n_mels, T)
            mel_for_vocoder = mel_outputs_postnet.transpose(1, 2)
            assert mel_for_vocoder.shape[1] in (config.N_MELS, getattr(config, 'n_mels', config.N_MELS)), "Mel dim mismatch"
            waveform = hifi_gan(mel_for_vocoder)
            audio_numpy = waveform.squeeze().to('cpu').numpy()
        else:
            print("Reconstructing waveform via Griffin-Lim (auto log->linear conversion if needed)...")
            mel_nt = mel_outputs_postnet[0].transpose(0, 1)  # (n_mels,T)
            audio_tensor = fallback_mel_to_audio(mel_nt)
            audio_numpy = audio_tensor.numpy()
    
    # --- THIS IS THE NEW LOGIC ---
    # 1. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Find the next available successive filename
    counter = 1
    while True:
        output_path = os.path.join(output_dir, f"output_{counter}.wav")
        if not os.path.exists(output_path):
            break
        counter += 1
    
    # 3. Save the .wav file
    write_wav(output_path, config.SAMPLING_RATE, audio_numpy)
    
    print(f"\n✅ Audio successfully saved to: {output_path}")

def _print_mel_stats(mel_bt_tn: torch.Tensor, tag: str):
    """
    mel_bt_tn shape (B, T, n_mels)
    """
    m = mel_bt_tn.detach().float()
    vals = m.view(-1)
    mn = float(vals.min().item()); mx = float(vals.max().item())
    mean = float(vals.mean().item()); std = float(vals.std().item())
    p = torch.quantile(vals, torch.tensor([0.01,0.5,0.99], device=vals.device)).cpu().numpy()
    print(f"[MEL STATS] {tag}: min {mn:.4f} max {mx:.4f} mean {mean:.4f} std {std:.4f} p01 {p[0]:.4f} p50 {p[1]:.4f} p99 {p[2]:.4f}")
    if mn >= -1e-4 and 0.0 <= mx <= 1.05:
        print(f"[WARN] {tag}: Mel appears 0–1 linear; pretrained HiFi-GAN expects log-mel (negative values). Expect noisy / 'wind' audio.")
    else:
        print(f"[INFO] {tag}: Mel dynamic range includes negatives or >1 values; may be log-compressed already.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='Text to synthesize.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a trained model checkpoint.')
    parser.add_argument('--output_dir', type=str, default='generated_audio', help='Directory to save the output audio files.')
    parser.add_argument('--vocoder', type=str, default='hifigan', choices=['hifigan','griffinlim'], help='Select vocoder backend.')
    
    args = parser.parse_args()
    
    inference(
        text=args.text,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        vocoder=args.vocoder
    )