# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime  # <-- Added this import

# Add the local NLTK data path
import nltk
nltk.data.path.append('./nltk_data')

from src import config
from src.model import Tacotron2
from src.data_utils import TextMelDataset, TextMelCollate

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def get_mask_from_lengths(self, lengths):
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
        mask = (ids < lengths.unsqueeze(1)).bool()
        return ~mask

    def forward(self, model_outputs, targets):
        mel_out_postnet, mel_out, gate_out, _ = model_outputs
        mel_target, gate_target, mel_lengths = targets

        mel_out = mel_out.transpose(1, 2)
        mel_out_postnet = mel_out_postnet.transpose(1, 2)
        
        mask = self.get_mask_from_lengths(mel_lengths)
        mask = mask.expand(config.n_mels, mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2).to(mel_target.device)
        
        mel_out.data.masked_fill_(mask, 0.0)
        mel_out_postnet.data.masked_fill_(mask, 0.0)
        mel_target.data.masked_fill_(mask, 0.0)

        loss_mel = self.mse_loss(mel_out, mel_target) + \
                   self.mse_loss(mel_out_postnet, mel_target)
        loss_gate = self.bce_loss(gate_out, gate_target)
        
        total_loss = loss_mel + loss_gate
        return total_loss

def save_alignment_plot(alignment, path):
    alignment = alignment[0].detach().cpu().numpy().T
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none', cmap='viridis')
    fig.colorbar(im, ax=ax)
    plt.xlabel("Encoder timestep (Phonemes)")
    plt.ylabel("Decoder timestep")
    plt.title("Attention Alignment")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

######################### DEBUG
def train(metadata_path, checkpoint_dir, epochs, batch_size, learning_rate):
    """
    A modified training routine for a detailed single-batch overfitting test.
    --- THIS VERSION HAS MIXED-PRECISION DISABLED FOR DEBUGGING ---
    """
    # --- TWEAK 1: Increased iterations ---
    iterations = 1200
    
    torch.manual_seed(1234)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = TextMelDataset(metadata_path)
    collate_fn = TextMelCollate()
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    
    model = Tacotron2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = Tacotron2Loss()
    
    print("\n--- DEBUG MODE: Starting single-batch overfitting test (Mixed-Precision DISABLED). ---")
    
    print("Fetching a single batch to overfit...")
    single_batch = next(iter(data_loader))
    
    text_padded, _, mel_padded, mel_lengths = single_batch
    text_padded = text_padded.to(device)
    mel_padded = mel_padded.to(device)
    mel_lengths = mel_lengths.to(device)
    gate_target = torch.zeros(mel_padded.size(0), mel_padded.size(2), device=device)
    for j, length in enumerate(mel_lengths):
        gate_target[j, length.item()-1:] = 1

    model.train()
    
    print(f"--- Training on one batch for {iterations} iterations... ---")
    
    for i in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        
        # Running without autocast and scaler for stability
        model_outputs = model(text_padded, mel_padded)
        loss = criterion(model_outputs, (mel_padded, gate_target, mel_lengths))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{iterations}, Loss: {loss.item():.6f}")

        # --- TWEAK 2: Save plot every 300 iterations ---
        if (i + 1) % 300 == 0:
            print(f"\n--- Saving alignment plot at iteration {i+1}... ---")
            if model_outputs is not None:
                _, _, _, alignments = model_outputs
                alignment_path = os.path.join(checkpoint_dir, f"overfit_alignment_iter_{i+1}.png")
                save_alignment_plot(alignments, alignment_path)
                print(f"Alignment plot saved to {alignment_path}\n")

    print("\n--- Overfitting test complete. ---")

# ######################## ACTUAL
# def train(metadata_path, checkpoint_dir, epochs, batch_size, learning_rate):
#     torch.manual_seed(1234)
    
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")
#     print(f"Using device: {device}")
    
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     dataset = TextMelDataset(metadata_path)
#     collate_fn = TextMelCollate()
#     data_loader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=True,
#         collate_fn=collate_fn, num_workers=0
#     )
#     print(f"Loaded {len(dataset)} training samples.")

#     model = Tacotron2().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = Tacotron2Loss()
    
#     scaler = torch.amp.GradScaler(enabled=(device.type != 'cpu'))
    
#     model.train()
#     model_outputs = None

#     for epoch in range(epochs):
#         start_time = time.time()
#         epoch_loss = 0.0
        
#         # --- TWEAK 1: Added timestamp to the epoch printout ---
#         print(f"\n--- Starting Epoch {epoch + 1}/{epochs} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        
#         for i, batch in enumerate(data_loader):
#             text_padded, _, mel_padded, mel_lengths = batch
            
#             text_padded = text_padded.to(device)
#             mel_padded = mel_padded.to(device)
#             mel_lengths = mel_lengths.to(device)
            
#             gate_target = torch.zeros(mel_padded.size(0), mel_padded.size(2), device=device)
#             for j, length in enumerate(mel_lengths):
#                 gate_target[j, length.item()-1:] = 1

#             optimizer.zero_grad(set_to_none=True)
            
#             with torch.autocast(device_type=device.type, enabled=(device.type != 'cpu')):
#                 model_outputs = model(text_padded, mel_padded)
#                 loss = criterion(model_outputs, (mel_padded, gate_target, mel_lengths))
            
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             scaler.step(optimizer)
#             scaler.update()
            
#             epoch_loss += loss.item()
            
#             # --- TWEAK 2: This block already prints every 10th batch ---
#             if (i + 1) % 10 == 0:
#                 print(f"  Batch {i+1}/{len(data_loader)}, Loss: {loss.item():.6f}")
        
#         avg_epoch_loss = epoch_loss / len(data_loader)
#         epoch_time = time.time() - start_time
#         print(f"\nEpoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")
        
#         checkpoint_path = os.path.join(checkpoint_dir, f"tacotron2_epoch_{epoch+1}.pth")
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': avg_epoch_loss,
#         }, checkpoint_path)
#         print(f"Checkpoint saved to {checkpoint_path}")

#         if model_outputs is not None:
#             _, _, _, alignments = model_outputs
#             alignment_path = os.path.join(checkpoint_dir, f"alignment_epoch_{epoch+1}.png")
#             save_alignment_plot(alignments, alignment_path)
#             print(f"Alignment plot saved to {alignment_path}")

#     print("\nTraining complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', type=str, help='Path to the metadata file.')
    parser.add_argument('checkpoint_dir', type=str, help='Directory to save checkpoints.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    
    args = parser.parse_args()
    
    train(
        metadata_path=args.metadata,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )