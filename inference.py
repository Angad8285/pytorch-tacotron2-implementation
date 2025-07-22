# inference.py

import torch
import argparse
import os
from scipy.io.wavfile import write as write_wav
from torch.hub import download_url_to_file

from src import config
from src.model import Tacotron2
from src.text import text_to_sequence

def inference(text, checkpoint_path, output_dir):
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

    # --- Load Pre-trained HiFi-GAN Vocoder ---
    print("Loading HiFi-GAN vocoder from NVIDIA's official repo...")
    hifi_gan_tuple = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_hifigan', pretrained=False, trust_repo=True)
    hifi_gan = hifi_gan_tuple[0].to(device)
    
    ckpt_url = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz/versions/21.08.0_amp/files/hifigan_gen_checkpoint_10000_ft.pt"
    ckpt_file = "hifigan_checkpoint.pt"
    if not os.path.exists(ckpt_file):
        download_url_to_file(ckpt_url, ckpt_file)

    state_dict = torch.load(ckpt_file, map_location=device)
    hifi_gan.load_state_dict(state_dict['generator'])
    hifi_gan.eval()
    print("HiFi-GAN vocoder loaded.")

    # --- Process Input Text ---
    print("Processing input text...")
    sequence = text_to_sequence(text)
    sequence = torch.LongTensor(sequence).unsqueeze(0).to(device)

    # --- Generate Speech ---
    print("Generating mel spectrogram...")
    with torch.no_grad():
        mel_outputs_postnet, _, _, _ = model.inference(sequence)
        
        print("Synthesizing waveform with vocoder...")
        waveform = hifi_gan(mel_outputs_postnet)
        
    audio_numpy = waveform.squeeze().to('cpu').numpy()
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='Text to synthesize.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a trained model checkpoint.')
    # --- ARGUMENT CHANGED HERE ---
    parser.add_argument('--output_dir', type=str, default='generated_audio', help='Directory to save the output audio files.')
    
    args = parser.parse_args()
    
    inference(
        text=args.text,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir # Argument name also changed here
    )