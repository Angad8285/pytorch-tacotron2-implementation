
import torch
from torch.utils.data import Dataset
import pandas as pd
from . import config, text, audio

class TextMelDataset(Dataset):
    """
    A custom PyTorch Dataset for loading (text, mel_spectrogram) pairs.
    """
    def __init__(self, metadata_path):
        self.metadata = pd.read_csv(metadata_path)
        
    def __getitem__(self, index):
        """
        Returns a single processed data point.
        """
        # Get the row from the metadata file
        row = self.metadata.iloc[index]
        
        # 1. Process the text
        text_sequence = text.text_to_sequence(row['text'])
        
        # 2. Process the audio
        mel_spectrogram = audio.get_mel_spectrogram(row['filepath'])
        
        return torch.LongTensor(text_sequence), torch.FloatTensor(mel_spectrogram)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.metadata)


class TextMelCollate:
    """
    A collate function to pad a batch of (text, mel) pairs.
    
    This function takes a list of data points from the Dataset and pads them
    so that all sequences in a batch have the same length.
    """
    def __call__(self, batch):
        """
        Collates a batch of data.
        """
        # Find the max length for both text and mels in the current batch
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        mel_lengths = torch.LongTensor([x[1].size(1) for x in batch])
        max_mel_len = mel_lengths.max()

        # Create padded tensors for text and mels
        text_padded = torch.LongTensor(len(batch), max_input_len)
        mel_padded = torch.FloatTensor(len(batch), config.n_mels, max_mel_len)
        
        # Initialize padded tensors to zeros
        text_padded.zero_()
        mel_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            
            # Pad text
            text_seq = row[0]
            text_padded[i, :text_seq.size(0)] = text_seq
            
            # Pad mels
            mel = row[1]
            mel_padded[i, :, :mel.size(1)] = mel
            
        return text_padded, input_lengths, mel_padded, mel_lengths