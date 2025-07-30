# prepare_ljspeech.py

import pandas as pd
import os
import argparse
from tqdm import tqdm

def prepare_ljspeech_metadata(dataset_path, output_path):
    """Prepares a clean metadata file for the LJSpeech dataset."""
    metadata_file = os.path.join(dataset_path, 'metadata.csv')
    wavs_path = os.path.join(dataset_path, 'wavs')
    
    print("Reading original LJSpeech metadata...")
    # The file is pipe-separated, has no header, and needs quoting handled
    df = pd.read_csv(metadata_file, sep='|', header=None, quoting=3)
    
    # We only need the first column (basename) and the third (normalized text)
    df = df[[0, 2]]
    df.columns = ['basename', 'text']
    
    # Create the full, absolute filepath for each wav file
    df['filepath'] = df['basename'].apply(lambda x: os.path.abspath(os.path.join(wavs_path, f"{x}.wav")))
    
    # We only need the 'filepath' and 'text' columns for our pipeline
    final_df = df[['filepath', 'text']]
    
    final_df.to_csv(output_path, index=False)
    print(f"✅ Clean LJSpeech metadata successfully created at: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare LJSpeech metadata.")
    parser.add_argument('dataset_path', type=str, help='Path to the root of the LJSpeech dataset directory.')
    parser.add_argument('output_path', type=str, help='Path to save the new, clean metadata CSV file.')
    args = parser.parse_args()
    
    prepare_ljspeech_metadata(args.dataset_path, args.output_path)