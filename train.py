# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from src import config
from src.model import Tacotron2
from src.data_utils import TextMelDataset, TextMelCollate

class Tacotron2Loss(nn.Module):
    """The loss function for the Tacotron 2 model."""
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, model_outputs, targets):
        mel_out_postnet, mel_out, gate_out = model_outputs
        mel_target, gate_target = targets
        
        # Calculate the loss
        loss_mel = self.mse_loss(mel_out, mel_target) + self.mse_loss(mel_out_postnet, mel_target)
        loss_gate = self.bce_loss(gate_out, gate_target)
        
        total_loss = loss_mel + loss_gate
        return total_loss

def train(metadata_path, epochs, batch_size, learning_rate):
    """The main training routine."""
    
    # -- Setup --
    torch.manual_seed(1234)
    # Use Mac's M2 GPU (MPS) if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- Data Loading --
    dataset = TextMelDataset(metadata_path)
    collate_fn = TextMelCollate()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    print(f"Loaded {len(dataset)} training samples.")

    # -- Model, Optimizer, Loss --
    model = Tacotron2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = Tacotron2Loss()
    
    model.train() # Set the model to training mode

    # -- The Training Loop --
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}/{epochs}")
        
        for i, batch in enumerate(data_loader):
            # TODO in the next step:
            # 1. Send the batch to the correct device
            # 2. Run the model
            # 3. Calculate the loss
            # 4. Perform backpropagation and update weights
            # 5. Print progress
            
            if i % 10 == 0:
                print(f"  Batch {i}/{len(data_loader)}")

    print("\nTraining complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', type=str, help='Path to the metadata file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    
    args = parser.parse_args()
    
    train(
        metadata_path=args.metadata,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )