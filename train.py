# train.py that works on both CUDA and MPS devices

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import time
import os
import matplotlib.pyplot as plt

from src import config
from src.model import Tacotron2
from src.data_utils import TextMelDataset, TextMelCollate

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss()
        # Attention guidance scheduling params
        self.attn_weight_start = 1.0
        self.min_attn_weight = 0.2
        self.entropy_target = 2.0  # when entropy drops below this, begin decaying weight
        self.current_attention_weight = self.attn_weight_start

        self.global_step = 0  # counts forward() calls (batches/iterations)

    def get_mask_from_lengths(self, lengths):
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
        mask = (ids < lengths.unsqueeze(1)).bool()
        return ~mask

    def create_diagonal_attention_target(self, text_lengths, num_steps, alignments):
        """
        Per-sample diagonal Gaussian targets with annealed σ.
        text_lengths: (B,) tensor of true encoder (text) lengths
        num_steps: decoder time steps (len(alignments))
        """
        batch_size = len(alignments[0])
        max_text_len = int(text_lengths.max().item())

        # σ schedule (same for all samples; could be individualized if desired)
        init_sigma = torch.clamp(
            text_lengths.float() * config.attention_initial_sigma_factor,
            min=3.0, max=config.attention_max_sigma_cap
        )  # (B,)
        progress = min(1.0, self.global_step / float(config.attention_sigma_warmup_steps))
        sigma = init_sigma - (init_sigma - config.attention_min_sigma) * progress  # (B,)
        self.current_sigma = float(sigma.mean().item())

        device = alignments[0].device
        diagonal_target = torch.zeros(batch_size, num_steps, max_text_len, device=device)
        base_positions = torch.arange(max_text_len, device=device, dtype=torch.float)

        for b in range(batch_size):
            Lb = int(text_lengths[b].item())
            pos_slice = base_positions[:Lb]
            sigma_b = sigma[b]
            for t in range(num_steps):
                # Scale expected position to sample's true length
                expected_pos = min(int(t * Lb / num_steps), Lb - 1)
                gaussian = torch.exp(-0.5 * ((pos_slice - expected_pos) / sigma_b) ** 2)
                gaussian = gaussian / (gaussian.sum() + 1e-8)
                diagonal_target[b, t, :Lb] = gaussian
            # Padded tail remains zero => no target mass outside true length
        return diagonal_target

    def forward(self, model_outputs, targets, text_lengths=None):
        mel_out_postnet, mel_out, gate_out, alignments = model_outputs
        mel_target, gate_target, mel_lengths = targets

        # FIX: No transpose needed now - outputs are already (batch, time, n_mels)
        # mel_out and mel_out_postnet are now (batch, time, n_mels)
        # mel_target should be (batch, n_mels, time) - transpose it to match
        mel_target = mel_target.transpose(1, 2)  # (batch, n_mels, time) -> (batch, time, n_mels)
        
        mask = self.get_mask_from_lengths(mel_lengths)
        # Expand mask to cover mel dimensions: (batch, time) -> (batch, time, n_mels)
        mask = mask.unsqueeze(-1).expand(-1, -1, config.n_mels)
        
        # Calculate mel losses with proper masking and normalization
        mel_loss_1 = self.mse_loss(mel_out, mel_target)
        mel_loss_2 = self.mse_loss(mel_out_postnet, mel_target)
        
        # Apply mask and normalize by valid frames
        mel_loss_1.masked_fill_(mask, 0.0)
        mel_loss_2.masked_fill_(mask, 0.0)
        
        # Normalize by number of valid frames and mels
        valid_frames = (~mask).float().sum()
        mel_loss_1 = mel_loss_1.sum() / valid_frames
        mel_loss_2 = mel_loss_2.sum() / valid_frames
        
        loss_mel = mel_loss_1 + mel_loss_2
        loss_gate = self.bce_loss(gate_out, gate_target)

        # --- KL-based attention guidance with entropy-weight schedule ---
        attention_kl = torch.tensor(0.0, device=mel_out.device)
        attn_entropy = torch.tensor(0.0, device=mel_out.device)
        if len(alignments) > 1 and text_lengths is not None:
            try:
                attn_weights = torch.stack(alignments, dim=1)  # (B, T_dec, T_enc_max)
                B, T_dec, T_enc_max = attn_weights.shape
                # Length-aware target
                diagonal_target = self.create_diagonal_attention_target(
                    text_lengths, T_dec, alignments
                )  # (B, T_dec, T_enc_max)
                attn_weights_safe = attn_weights.clamp_min(1e-8)
                log_pred = attn_weights_safe.log()
                attention_kl = F.kl_div(log_pred, diagonal_target, reduction='batchmean')
                attn_entropy = -(attn_weights_safe * log_pred).sum(dim=2).mean()
                # Weight schedule
                if attn_entropy <= self.entropy_target:
                    ratio = (attn_entropy / self.entropy_target).clamp_min(0.0)
                    self.current_attention_weight = max(
                        self.min_attn_weight,
                        self.attn_weight_start * ratio.item()
                    )
                else:
                    self.current_attention_weight = self.attn_weight_start
            except Exception as e:
                print(f"Warning: Attention KL failed: {e}")
                self.current_attention_weight = self.attn_weight_start

        total_loss = loss_mel + loss_gate + self.current_attention_weight * attention_kl
        # Return raw KL (unweighted) so caller can log both raw and weighted contribution
        self.global_step += 1  # increment after each successful forward
        return total_loss, loss_mel, loss_gate, attention_kl

def save_alignment_plot(alignments, path, sample_index: int = 0):
    """
    Saves a plot of the attention alignment to a file.

    alignments: list length T_dec; each element tensor (B, T_enc)
    Produces matrix (T_dec, T_enc) for a chosen sample (default 0).
    """
    # Stack: (T_dec, B, T_enc)
    attn_stack = torch.stack(alignments, dim=0)
    if sample_index >= attn_stack.size(1):
        sample_index = 0
    matrix = attn_stack[:, sample_index, :].detach().cpu().numpy()  # (T_dec, T_enc)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect='auto', origin='lower',
                   interpolation='none', cmap='viridis')
    fig.colorbar(im, ax=ax)
    plt.xlabel("Encoder timestep (Phonemes)")
    plt.ylabel("Decoder timestep")
    plt.title(f"Attention Alignment (sample {sample_index})")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def train(metadata_path, checkpoint_dir, epochs, batch_size, learning_rate, debug_overfit=False):
    """The main training routine."""
    torch.manual_seed(1234)
    
    # --- UNIVERSAL DEVICE SELECTION ---
    # This block checks for an NVIDIA GPU first, then an Apple Silicon GPU,
    # and falls back to the CPU if neither is available.
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
    print(f"Loaded {len(dataset)} training samples.")

    model = Tacotron2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = Tacotron2Loss()
    
    # --- UNIVERSAL MIXED-PRECISION ---
    # GradScaler is enabled only for CUDA, not MPS due to compatibility issues
    use_amp = (device.type == 'cuda')  # Only use AMP on NVIDIA GPUs
    scaler = torch.amp.GradScaler(enabled=use_amp) # type: ignore
    
    model.train()
    
    # === DEBUGGING MODE: OVERFIT ON SINGLE BATCH ===
    if debug_overfit:
        print("🔥 DEBUG MODE: Training on single batch to test overfitting capability")
        # Force batch size 8 for debug mode
        debug_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=TextMelCollate(), drop_last=True)
        debug_batch = next(iter(debug_loader))
        text_padded, input_lengths, mel_padded, mel_lengths = debug_batch  # was '_' before
        # BUG FIX: move text_padded to device (was omitted -> device mismatch on MPS)
        text_padded = text_padded.to(device)
        input_lengths = input_lengths.to(device)
        mel_padded = mel_padded.to(device)
        mel_lengths = mel_lengths.to(device)
        # Create gate targets for this batch
        gate_target = torch.zeros(mel_padded.size(0), mel_padded.size(2), device=device)
        for j, length in enumerate(mel_lengths):
            gate_target[j, length.item()-1:] = 1
        print(f"Debug batch shapes:")
        print(f"  Text: {text_padded.shape}")
        print(f"  Mel: {mel_padded.shape}")
        print(f"  Mel range: [{mel_padded.min():.3f}, {mel_padded.max():.3f}]")
        print(f"  Lengths: {mel_lengths}")
        print("🧪 Testing model forward pass...")
        try:
            with torch.no_grad():
                print("  - Creating model outputs...")
                model_outputs = model(text_padded, mel_padded, input_lengths)  # pass lengths
                print("  - ✅ Forward pass successful!")
                print(f"  - Output shapes: {[x.shape if hasattr(x, 'shape') else len(x) for x in model_outputs]}")
        except Exception as e:
            print(f"  - ❌ Forward pass failed: {e}")
            return
        print("🏋️ Starting training iterations...")
        for iteration in range(epochs * 20):  # Fewer iterations for safety
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                model_outputs = model(text_padded, mel_padded, input_lengths)
                total_loss, mel_loss, gate_loss, attention_kl = criterion(
                    model_outputs, (mel_padded, gate_target, mel_lengths), text_lengths=input_lengths
                )
            # Optional KL cap (uncomment if KL overwhelms):
            # attention_kl = torch.clamp(attention_kl, max=100.0)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if (iteration + 1) % 5 == 0:
                eff_attn = criterion.current_attention_weight * attention_kl.item()
                print(f"Iteration {iteration+1:4d}, Total: {total_loss.item():.6f}")
                print(f"  Mel: {mel_loss.item():.4f} | Gate: {gate_loss.item():.4f} | "
                      f"Attn(KL raw): {attention_kl.item():.4f} | w: {criterion.current_attention_weight:.2f} | "
                      f"w*KL: {eff_attn:.4f} | σ: {getattr(criterion,'current_sigma',float('nan')):.2f}")

                # Add attention diagnostics
                _, _, _, alignments = model_outputs
                if len(alignments) > 0:
                    # Get attention from last decoder step
                    last_attention = alignments[-1][0]  # (encoder_steps,)
                    print(f"  🎯 Attention - max: {last_attention.max().item():.4f}, "
                          f"min: {last_attention.min().item():.4f}, "
                          f"std: {last_attention.std().item():.4f}")
                    print(f"  📍 Attention peak at encoder position: {last_attention.argmax().item()}")
                    
                    # Check attention movement across decoder steps
                    if len(alignments) > 10:
                        first_peak = alignments[0][0].argmax().item()
                        mid_peak = alignments[len(alignments)//2][0].argmax().item() 
                        last_peak = alignments[-1][0].argmax().item()
                        print(f"  🔄 Attention movement: {first_peak} → {mid_peak} → {last_peak} (should increase)")
                        
                        # Check if attention is stuck
                        if first_peak == last_peak:
                            print(f"  ⚠️  WARNING: Attention is stuck at position {first_peak}!")
                    
                    # Check attention sharpness
                    entropy = -(last_attention * torch.log(last_attention + 1e-8)).sum().item()
                    print(f"  📈 Attention entropy: {entropy:.3f} (lower=sharper, target<2.0)")
            if (iteration + 1) % 10 == 0:
                _, _, _, alignments = model_outputs
                alignment_path = os.path.join(checkpoint_dir, f"debug_alignment_iter_{iteration+1}.png")
                save_alignment_plot(alignments, alignment_path)
                print(f"🎯 Alignment saved: {alignment_path}")
            if total_loss.item() < 0.1:
                print(f"🎉 SUCCESS! Loss dropped to {total_loss.item():.6f} - Model can learn!")
                print(f"🎯 Final alignment saved to: debug_alignment_iter_{iteration+1}.png")
                break
        print("🔥 DEBUG MODE COMPLETE")
        return

    # === NORMAL TRAINING MODE ===

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        print(f"\nEpoch: {epoch + 1}/{epochs}")
        
        for i, batch in enumerate(data_loader):
            text_padded, input_lengths, mel_padded, mel_lengths = batch  # capture lengths
            
            text_padded = text_padded.to(device)
            input_lengths = input_lengths.to(device)
            mel_padded = mel_padded.to(device)
            mel_lengths = mel_lengths.to(device)
            
            # FIX: Gate target should match mel sequence length (time dimension)
            # mel_padded is (batch, n_mels, time), we want gate for time dimension
            gate_target = torch.zeros(mel_padded.size(0), mel_padded.size(2), device=device)
            for j, length in enumerate(mel_lengths):
                gate_target[j, length.item()-1:] = 1

            optimizer.zero_grad(set_to_none=True)
            
            # Autocast is also enabled only on GPU
            with torch.autocast(device_type=device.type, enabled=use_amp):
                model_outputs = model(text_padded, mel_padded, input_lengths)
                total_loss, mel_loss, gate_loss, attention_kl = criterion(
                    model_outputs, (mel_padded, gate_target, mel_lengths), text_lengths=input_lengths
                )
            # Optional KL cap (uncomment if needed):
            # attention_kl = torch.clamp(attention_kl, max=100.0)
            # BUG FIX: use total_loss (scalar) for scaler / backward
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()

            eff_attn = criterion.current_attention_weight * attention_kl.item()
            print(f"  Batch {i+1}/{len(data_loader)}, "
                  f"Total: {total_loss.item():.6f} | Mel: {mel_loss.item():.4f} | "
                  f"Gate: {gate_loss.item():.4f} | Attn(KL raw): {attention_kl.item():.4f} | "
                  f"w: {criterion.current_attention_weight:.2f} | w*KL: {eff_attn:.4f} | "
                  f"σ: {getattr(criterion,'current_sigma',float('nan')):.2f}")
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"tacotron2_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save the attention alignment from the last batch of the epoch
        if model_outputs is not None:
            _, _, _, alignments = model_outputs
            alignment_path = os.path.join(checkpoint_dir, f"alignment_epoch_{epoch+1}.png")
            save_alignment_plot(alignments, alignment_path)
            print(f"Alignment plot saved to {alignment_path}")

    print("\nTraining complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', type=str, help='Path to the metadata file.')
    parser.add_argument('checkpoint_dir', type=str, help='Directory to save checkpoints.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--debug', action='store_true', help='Debug mode: overfit on single batch.')
    
    args = parser.parse_args()
    
    train(
        metadata_path=args.metadata,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        debug_overfit=args.debug
    )