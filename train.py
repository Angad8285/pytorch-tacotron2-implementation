# train.py works on google colab

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time

from src import config
from src.model import Tacotron2
from src.data_utils import TextMelDataset, TextMelCollate

# In train.py
class Tacotron2Loss(nn.Module):
    """The loss function for the Tacotron 2 model."""
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def get_mask_from_lengths(self, lengths):
        """Creates a boolean mask from a tensor of sequence lengths."""
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
        mask = (ids < lengths.unsqueeze(1)).bool()
        return ~mask

    def forward(self, model_outputs, targets):
        mel_out_postnet, mel_out, gate_out, _ = model_outputs
        mel_target, gate_target, mel_lengths = targets # Now expects mel_lengths

        mel_out = mel_out.transpose(1, 2)
        mel_out_postnet = mel_out_postnet.transpose(1, 2)
        
        # Create a mask based on the true lengths of the mel spectrograms
        mask = self.get_mask_from_lengths(mel_lengths)
        mask = mask.expand(config.n_mels, mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2).to(mel_target.device)
        
        # Apply the mask to both model outputs and targets
        mel_out.data.masked_fill_(mask, 0.0)
        mel_out_postnet.data.masked_fill_(mask, 0.0)
        mel_target.data.masked_fill_(mask, 0.0)

        # Calculate the loss
        loss_mel = self.mse_loss(mel_out, mel_target) + \
                   self.mse_loss(mel_out_postnet, mel_target)
        loss_gate = self.bce_loss(gate_out, gate_target)
        
        total_loss = loss_mel + loss_gate
        return total_loss


# In train.py

# ... (keep imports and Tacotron2Loss class the same) ...

def train(metadata_path, epochs, batch_size, learning_rate):
    """The main training routine."""
    
    torch.manual_seed(1234)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TextMelDataset(metadata_path)
    collate_fn = TextMelCollate()
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    print(f"Loaded {len(dataset)} training samples.")

    model = Tacotron2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = Tacotron2Loss()
    
    # --- FIX: GradScaler is removed. It is not used for MPS. ---
    
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        print(f"\nEpoch: {epoch + 1}/{epochs}")
        
        for i, batch in enumerate(data_loader):
            text_padded, _, mel_padded, mel_lengths = batch
            text_padded = text_padded.to(device)
            mel_padded = mel_padded.to(device)
            mel_lengths = mel_lengths.to(device)
            
            gate_target = torch.zeros(mel_padded.size(0), mel_padded.size(2), device=device)
            for j, length in enumerate(mel_lengths):
                gate_target[j, length.item()-1:] = 1

            optimizer.zero_grad()
            
            # Use autocast for the forward pass for mixed precision
            with torch.autocast(device_type="mps", dtype=torch.float16):
                model_outputs = model(text_padded, mel_padded)
                loss = criterion(model_outputs, (mel_padded, gate_target, mel_lengths))
            
            # --- FIX: Revert to standard backpropagation without the scaler ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(data_loader)}, Loss: {loss.item():.6f}", end='\r')
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")

    print("\nTraining complete.")

# ... (if __name__ == '__main__': block remains the same) ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', type=str, help='Path to the metadata file.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    
    args = parser.parse_args()
    
    train(
        metadata_path=args.metadata,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )