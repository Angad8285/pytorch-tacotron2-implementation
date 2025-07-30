# preprocess.py

import os
import torch
import pandas as pd
from tqdm import tqdm
import argparse

# Add the local NLTK data path
import nltk
nltk.data.path.append('./nltk_data')

from src import text, audio

def preprocess_data(metadata_path, output_dir):
    """
    Performs one-time pre-processing of the dataset.
    """
    print(f"Loading metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    # Create output directories
    mels_dir = os.path.join(output_dir, "mels")
    text_dir = os.path.join(output_dir, "text")
    os.makedirs(mels_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    print(f"Processing {len(metadata)} files. This will take a while...")

    # Loop through each audio file and process it
    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        # Get a unique name for each file from its original path
        basename = os.path.basename(row['filepath']).replace('.flac', '').replace('.wav', '')

        # Process and save the mel spectrogram
        try:
            mel_spectrogram = audio.get_mel_spectrogram(row['filepath'])
            mel_path = os.path.join(mels_dir, f"{basename}.pt")
            torch.save(mel_spectrogram, mel_path)

            # Process and save the text sequence
            text_sequence = text.text_to_sequence(row['text'])
            text_path = os.path.join(text_dir, f"{basename}.pt")
            torch.save(torch.LongTensor(text_sequence), text_path)
        except Exception as e:
            print(f"Skipping file {row['filepath']} due to error: {e}")

    # Copy the metadata file to the new directory for convenience
    processed_metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata.to_csv(processed_metadata_path, index=False)

    print(f"\n✅ Pre-processing complete. Processed data saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-process dataset for Tacotron 2 training.")
    parser.add_argument('metadata', type=str, help='Path to the original metadata file.')
    parser.add_argument('output_dir', type=str, help='Directory to save the processed data.')
    args = parser.parse_args()

    preprocess_data(args.metadata, args.output_dir)