# In src/data_utils.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from . import config

class TextMelDataset(Dataset):
    """
    A custom PyTorch Dataset for loading pre-computed (text, mel_spectrogram) pairs.
    """
    def __init__(self, metadata_path):
        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = os.path.dirname(metadata_path)
        
    def __getitem__(self, index):
        """
        Returns a single pre-processed data point.
        """
        row = self.metadata.iloc[index]
        basename = os.path.basename(row['filepath']).replace('.flac', '').replace('.wav', '')
        
        text_path = os.path.join(self.data_dir, "text", f"{basename}.pt")
        text_sequence = torch.load(text_path, weights_only=False)
        
        mel_path = os.path.join(self.data_dir, "mels", f"{basename}.pt")
        # --- THIS IS THE FIX ---
        # Load the file and ensure it's converted to a PyTorch FloatTensor.
        # This handles cases where the saved file is a NumPy array.
        mel_spectrogram_numpy = torch.load(mel_path, weights_only=False)
        mel_spectrogram = torch.FloatTensor(mel_spectrogram_numpy)
        # --- END OF FIX ---
        
        return text_sequence, mel_spectrogram

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.metadata)

# In src/data_utils.py

class TextMelCollate:
    """
    A collate function to pad a batch of (text, mel) pairs.
    """
    def __call__(self, batch):
        """
        Collates a batch of data.
        """
        # Sort the batch by the length of the text sequence in descending order
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        # Get the length of each mel spectrogram in the batch
        mel_lengths = torch.LongTensor([x[1].shape[1] for x in batch])
        max_mel_len = torch.max(mel_lengths).item()

        # Create padded tensors for text and mels
        text_padded = torch.LongTensor(len(batch), max_input_len)
        mel_padded = torch.FloatTensor(len(batch), config.n_mels, max_mel_len)
        
        # Initialize padded tensors to zeros
        text_padded.zero_()
        mel_padded.zero_()

        # Fill the padded tensors with the data from the sorted batch
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            
            # Pad text
            text_seq = row[0]
            # --- THIS IS THE FIX ---
            text_padded[i, :text_seq.shape[0]] = text_seq
            
            # Pad mels
            mel = row[1]
            mel_padded[i, :, :mel.shape[1]] = mel
            
        return text_padded, input_lengths, mel_padded, mel_lengths