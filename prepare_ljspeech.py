# prepare_ljspeech.py

import pandas as pd
import os
import argparse
from tqdm import tqdm

def prepare_ljspeech_metadata(dataset_path, output_path, debug=False):
    """Prepares a clean metadata file for the LJSpeech dataset."""
    metadata_file = os.path.join(dataset_path, 'metadata.csv')
    wavs_path = os.path.join(dataset_path, 'wavs')
    
    if debug:
        print(f"[DEBUG] dataset_path={dataset_path}")
        print(f"[DEBUG] metadata_file exists? {os.path.isfile(metadata_file)}")
        print(f"[DEBUG] wavs_path exists? {os.path.isdir(wavs_path)}")

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

    if debug:
        print("[DEBUG] First 5 rows:")
        print(final_df.head())
        total_rows = len(final_df)
        print(f"[DEBUG] Total rows: {total_rows}")
        # Check missing wav files
        missing = []
        for fp in final_df['filepath'].head(200):  # limit for speed
            if not os.path.isfile(fp):
                missing.append(fp)
        if missing:
            print(f"[DEBUG][WARN] Missing {len(missing)} wav files (showing up to 5):")
            for m in missing[:5]:
                print("   ", m)
        else:
            print("[DEBUG] No missing wav files detected in first 200 entries.")
        # Basic text sanity
        empty_text = final_df[final_df['text'].str.strip() == '']
        if not empty_text.empty:
            print(f"[DEBUG][WARN] Empty text rows: {len(empty_text)}")
        # Character distribution sample
        import collections
        chars = ''.join(final_df['text'].head(50).tolist()).lower()
        most_common = collections.Counter(chars).most_common(10)
        print(f"[DEBUG] Common chars (sample first 50 rows): {most_common}")

    final_df.to_csv(output_path, index=False)
    print(f"✅ Clean LJSpeech metadata successfully created at: {output_path}")
    if debug:
        print(f"[DEBUG] Wrote file size: {os.path.getsize(output_path)} bytes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare LJSpeech metadata.")
    parser.add_argument('dataset_path', type=str, help='Path to the root of the LJSpeech dataset directory.')
    parser.add_argument('output_path', type=str, help='Path to save the new, clean metadata CSV file.')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug output and validations.')
    args = parser.parse_args()
    
    prepare_ljspeech_metadata(args.dataset_path, args.output_path, debug=args.debug)