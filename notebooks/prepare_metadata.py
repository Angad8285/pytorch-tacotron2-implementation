import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def create_metadata(librispeech_root_str: str, output_csv_path: str):
    """
    Parses the LibriSpeech directory to create a metadata CSV file.

    This function walks through the LibriSpeech dataset directory, finds all
    transcription files (*.trans.txt), and creates a CSV file mapping each
    audio file to its transcription and speaker ID.

    Args:
        librispeech_root_str (str): The root path of the LibriSpeech dataset.
        output_csv_path (str): The path to save the output metadata CSV file.
    """
    librispeech_root = Path(librispeech_root_str)

    if not librispeech_root.exists():
        print(f"Error: LibriSpeech root directory not found at '{librispeech_root}'")
        return

    # Find all *.trans.txt files recursively
    trans_files = list(librispeech_root.rglob("*.trans.txt"))

    if not trans_files:
        print("Error: No '*.trans.txt' files found. Did you specify the correct LibriSpeech root directory?")
        return

    metadata = []
    for trans_file in tqdm(trans_files, desc="Parsing files"):
        chapter_path = trans_file.parent
        speaker_id = chapter_path.parent.name

        with open(trans_file, 'r') as f:
            for line in f:
                # Line format: 19-198-0000 THE TEXT OF THE UTTERANCE
                # We split only at the first space to keep the rest of the text intact.
                parts = line.strip().split(" ", 1)
                utterance_id = parts[0]
                text = parts[1]

                audio_path = chapter_path / f"{utterance_id}.flac"

                if audio_path.exists():
                    metadata.append({
                        "filepath": str(audio_path.resolve()),
                        "text": text,
                        "speaker_id": int(speaker_id)
                    })

    df = pd.DataFrame(metadata)
    
    # Save to CSV without the pandas DataFrame index
    df.to_csv(output_csv_path, index=False)

    print(f"\n✅ Successfully created metadata file at: {output_csv_path}")
    print(f"Total utterances found: {len(df)}")
    print("\nHere's a sample of your metadata:")
    print(df.head())


# This block runs when the script is executed directly from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech metadata.")
    parser.add_argument("librispeech_path", type=str, help="Path to the root of the LibriSpeech dataset.")
    args = parser.parse_args()

    create_metadata(args.librispeech_path, "metadata.csv")