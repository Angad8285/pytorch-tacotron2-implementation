# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import numpy as np

from src import config
from src.model import Tacotron2
from src.data_utils import TextMelDataset, TextMelCollate
from src.mel_griffinlim import mel_to_audio as fallback_mel_to_audio

try:
    import soundfile as sf  # for saving wavs
except ImportError:
    sf = None
# Optional audio helper (graceful if missing)
try:
    from src import audio
except ImportError:
    audio = None

# === ADD: helper functions for debug export (were missing) ===
def _ids_to_phoneme_string(id_tensor: torch.Tensor, length: int) -> str:
    """
    Convert token id sequence (with padding) into a readable phoneme string.
    """
    symbols = config.SYMBOLS
    seq = id_tensor[:length].tolist()
    return ' '.join(symbols[i] for i in seq)

def _export_debug_inference(model, batch_tensors, device, checkpoint_dir):
    """
    Run autoregressive inference on the overfit debug batch and save:
      - alignment plot
      - per-sample trimmed mel tensors
      - per-sample phoneme text file
      - optional wav (if audio.mel_to_audio & soundfile available)
      - pairs.csv linking indices to files
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    text_padded = batch_tensors["text"].to(device)
    input_lengths = batch_tensors["input_lengths"].to(device)
    mel_lengths = batch_tensors.get("mel_lengths")  # optional

    # Derive cap from training batch if available
    max_len_cap = None
    if mel_lengths is not None:
        max_len_cap = int(mel_lengths.max().item() * 1.10)  # +10% buffer

    model.eval()
    with torch.no_grad():
        mel_post, mel_coarse, gate, alignments = model.inference(
            text_padded,
            max_len_cap=max_len_cap  # no forced lower gate threshold
        )

    # Alignment plot (sample 0)
    align_path = os.path.join(checkpoint_dir, "debug_infer_alignment.png")
    save_alignment_plot(alignments, align_path, sample_index=0)
    print(f"🔍 Inference alignment saved: {align_path}")

    rows = []
    for b in range(mel_post.size(0)):
        gate_sig = torch.sigmoid(gate[b])
        stops = (gate_sig > 0.5).nonzero(as_tuple=True)[0]
        if len(stops) == 0 and max_len_cap is not None:
            # Fallback: trim to original target length if gate never fired
            end_idx = int(mel_lengths[b].item())
        else:
            end_idx = (stops[0].item() + 1) if len(stops) > 0 else mel_post.size(1)
        mel_b = mel_post[b, :end_idx]  # (T_trim, n_mels)
        mel_file = f"debug_infer_mel_{b}.pt"
        torch.save(mel_b.cpu(), os.path.join(checkpoint_dir, mel_file))

        length_int = int(input_lengths[b].item())
        phoneme_str = _ids_to_phoneme_string(text_padded[b].cpu(), length_int)
        txt_file = f"sample_{b}.txt"
        with open(os.path.join(checkpoint_dir, txt_file), "w", encoding="utf-8") as f:
            f.write(phoneme_str + "\n")

        # WAV export logic updated:
        wav_file = ""
        # Primary path: provided audio.mel_to_audio
        if audio is not None and hasattr(audio, "mel_to_audio") and sf is not None:
            try:
                wav = audio.mel_to_audio(mel_b.transpose(0, 1).cpu())
                wav_file = f"debug_infer_{b}.wav"
                sf.write(os.path.join(checkpoint_dir, wav_file), wav.numpy(), config.SAMPLING_RATE)
            except Exception as e:
                print(f"⚠️  Primary WAV export failed (sample {b}), trying fallback: {e}")
        # Fallback path
        if wav_file == "" and sf is not None:
            try:
                wav = fallback_mel_to_audio(mel_b.transpose(0, 1))  # (n_mels,T)
                wav_file = f"debug_infer_{b}.wav"
                sf.write(os.path.join(checkpoint_dir, wav_file), wav.numpy(), config.SAMPLING_RATE)
            except Exception as e:
                print(f"⚠️  Fallback Griffin-Lim failed (sample {b}): {e}")
        if sf is None and wav_file == "":
            print("⚠️  soundfile not installed; skipping wav export.")

        rows.append({
            "sample_index": b,
            "text_file": txt_file,
            "mel_file": mel_file,
            "wav_file": wav_file
        })

    pairs_path = os.path.join(checkpoint_dir, "pairs.csv")
    with open(pairs_path, "w", newline='', encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["sample_index", "text_file", "mel_file", "wav_file"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"📝 Paired metadata written: {pairs_path}")
    model.train()
# === END ADD ===

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        # Losses
        self.l1_loss = nn.L1Loss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Attention guidance scheduling params
        self.attn_weight_start = 1.0
        self.min_attn_weight = 0.2
        self.entropy_target = 3.5  # earlier decay trigger
        self.current_attention_weight = self.attn_weight_start

        # Step counters / schedules
        self.global_step = 0
        self.sigma_warmup_steps = config.attention_sigma_warmup_steps

    def get_mask_from_lengths(self, lengths):
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
        mask = (ids < lengths.unsqueeze(1)).bool()
        return ~mask  # True where padding

    def create_diagonal_attention_target(self, text_lengths, num_steps, alignments):
        batch_size = len(alignments[0])
        max_text_len = int(text_lengths.max().item())
        init_sigma = torch.clamp(
            text_lengths.float() * config.attention_initial_sigma_factor,
            min=3.0, max=config.attention_max_sigma_cap
        )
        progress = min(1.0, self.global_step / float(self.sigma_warmup_steps))
        sigma = init_sigma - (init_sigma - config.attention_min_sigma) * progress
        self.current_sigma = float(sigma.mean().item())

        device = alignments[0].device
        diagonal_target = torch.zeros(batch_size, num_steps, max_text_len, device=device)
        base_positions = torch.arange(max_text_len, device=device, dtype=torch.float)
        for b in range(batch_size):
            Lb = int(text_lengths[b].item())
            pos_slice = base_positions[:Lb]
            sigma_b = sigma[b]
            for t in range(num_steps):
                expected_pos = min(int(t * Lb / num_steps), Lb - 1)
                gaussian = torch.exp(-0.5 * ((pos_slice - expected_pos) / sigma_b) ** 2)
                gaussian = gaussian / (gaussian.sum() + 1e-8)
                diagonal_target[b, t, :Lb] = gaussian
        return diagonal_target

    def forward(self, model_outputs, targets, text_lengths=None):
        mel_out_postnet, mel_out, gate_out, alignments = model_outputs  # mel: (B, T, n_mels)
        mel_target, gate_target, mel_lengths = targets  # mel_target: (B, n_mels, T)

        # Align target shape to predictions
        mel_target = mel_target.transpose(1, 2)  # -> (B, T, n_mels)

        # Padding mask
        mask = self.get_mask_from_lengths(mel_lengths)  # (B, T)
        mask = mask.unsqueeze(-1).expand(-1, -1, config.n_mels)  # (B, T, n_mels)

        mel_loss_1 = self.l1_loss(mel_out, mel_target)
        mel_loss_2 = self.l1_loss(mel_out_postnet, mel_target)
        mel_loss_1.masked_fill_(mask, 0.0)
        mel_loss_2.masked_fill_(mask, 0.0)
        valid = (~mask).float().sum()
        mel_loss_1 = mel_loss_1.sum() / valid
        mel_loss_2 = mel_loss_2.sum() / valid
        loss_mel = mel_loss_1 + mel_loss_2
        loss_gate = self.bce_loss(gate_out, gate_target)

        attention_kl = torch.tensor(0.0, device=mel_out.device)
        if len(alignments) > 1 and text_lengths is not None:
            try:
                attn_weights = torch.stack(alignments, dim=1)  # (B, T_dec, T_enc_max)
                B, T_dec, _ = attn_weights.shape
                diagonal_target = self.create_diagonal_attention_target(text_lengths, T_dec, alignments)
                attn_safe = attn_weights.clamp_min(1e-8)
                log_pred = attn_safe.log()
                attention_kl = F.kl_div(log_pred, diagonal_target, reduction='batchmean') / T_dec
                attention_kl = torch.clamp(attention_kl, max=150.0)
                attn_entropy = -(attn_safe * log_pred).sum(dim=2).mean()
                if attn_entropy <= self.entropy_target:
                    ratio = (attn_entropy / self.entropy_target).clamp_min(0.0)
                    self.current_attention_weight = max(self.min_attn_weight, self.attn_weight_start * ratio.item())
                else:
                    self.current_attention_weight = self.attn_weight_start
            except Exception as e:
                print(f"Warning: Attention KL failed: {e}")
                self.current_attention_weight = self.attn_weight_start

        total_loss = loss_mel + loss_gate + self.current_attention_weight * attention_kl
        self.global_step += 1
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

def compute_attention_entropy(alignments):
    if len(alignments) == 0:
        return 0.0
    with torch.no_grad():
        attn = torch.stack(alignments, dim=0)  # (T_dec,B,T_enc)
        attn = attn.clamp_min(1e-8)
        ent = -(attn * attn.log()).sum(-1).mean().item()
    return ent

def validate(model, criterion, val_loader, device):
    model.eval()
    total_mel = 0.0
    total_gate = 0.0
    count = 0
    attn_entropy = 0.0
    with torch.no_grad():
        for batch in val_loader:
            text_padded, input_lengths, mel_padded, mel_lengths = batch
            text_padded = text_padded.to(device)
            input_lengths = input_lengths.to(device)
            mel_padded = mel_padded.to(device)
            mel_lengths = mel_lengths.to(device)
            gate_target = torch.zeros(mel_padded.size(0), mel_padded.size(2), device=device)
            for j, l in enumerate(mel_lengths):
                gate_target[j, l.item()-1:] = 1
            outputs = model(text_padded, mel_padded, input_lengths, use_postnet=True)
            _, mel_out, gate_out, alignments = outputs
            # Reuse loss computation sans KL weighting (text_lengths provided)
            _, mel_loss, gate_loss, _ = criterion(outputs, (mel_padded, gate_target, mel_lengths), text_lengths=input_lengths)
            total_mel += mel_loss.item()
            total_gate += gate_loss.item()
            attn_entropy += compute_attention_entropy(alignments)
            count += 1
    model.train()
    return total_mel / count, total_gate / count, attn_entropy / count

def adjust_lr(optimizer, global_step):
    for m in config.lr_decay_milestones:
        if global_step == m:
            for g in optimizer.param_groups:
                g['lr'] *= config.lr_decay_gamma
            print(f"[LR] Decayed learning rate at step {global_step}")
            break

def train(
    metadata_path,
    checkpoint_dir,
    epochs,
    batch_size,
    learning_rate,
    debug_overfit=False,
    val_metadata=None,
    resume=None,
    postnet_freeze_steps_override=None,
    accum_steps=1
):
    """The main training routine."""
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
    
    # OPTIONAL validation loader
    val_loader = None
    if val_metadata:
        val_dataset = TextMelDataset(val_metadata)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=TextMelCollate(),
            num_workers=0
        )
        print(f"Loaded {len(val_dataset)} validation samples.")

    model = Tacotron2().to(device)
    criterion = Tacotron2Loss()
    
    # PostNet freeze steps
    if debug_overfit:
        postnet_freeze_steps = 0  # allow PostNet immediately for clearer audio
    else:
        postnet_freeze_steps = (postnet_freeze_steps_override
                                if postnet_freeze_steps_override is not None
                                else config.postnet_freeze_steps)

    # Optimizer (attention LR scaling in both modes appropriately)
    if debug_overfit:
        attention_params = list(model.decoder.attention.parameters())
        # BUG FIX: avoid 'p not in attention_params' (ambiguous tensor truth value)
        attention_param_ids = {id(p) for p in attention_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in attention_param_ids]
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": learning_rate},
                {"params": attention_params, "lr": learning_rate * 2.0},  # higher LR for attention
            ]
        )
        # Shorter sigma warmup for debug
        criterion.sigma_warmup_steps = 800
    else:
        attention_params = list(model.decoder.attention.parameters())
        attention_param_ids = {id(p) for p in attention_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in attention_param_ids]
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": learning_rate},
                {"params": attention_params, "lr": learning_rate * config.attention_lr_multiplier},
            ]
        )
    
    # Resume checkpoint (before scaler definition)
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        best_val_mel = ckpt.get('best_val_mel', float('inf'))
        print(f"Resumed from {resume} (epoch {start_epoch+1}, step {global_step})")
    else:
        start_epoch = 0
        global_step = 0
        best_val_mel = float('inf')

    # Add log file path
    log_path = os.path.join(checkpoint_dir, "training_log.txt")
    def log_line(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{ts}] {msg}\n")

    # --- UNIVERSAL MIXED-PRECISION ---
    # GradScaler is enabled only for CUDA, not MPS due to compatibility issues
    use_amp = (device.type == 'cuda')  # Only use AMP on NVIDIA GPUs
    scaler = torch.amp.GradScaler(enabled=use_amp) # type: ignore
    
    model.train()
    
    # === DEBUGGING MODE: OVERFIT ON SINGLE BATCH ===
    if debug_overfit:
        print("🔥 DEBUG MODE: Training on single batch to test overfitting capability (L1 loss, log-power mels)")
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
                use_postnet = (global_step >= postnet_freeze_steps)
                model_outputs = model(text_padded, mel_padded, input_lengths, use_postnet=use_postnet)
                total_loss, mel_loss, gate_loss, attention_kl = criterion(
                    model_outputs, (mel_padded, gate_target, mel_lengths), text_lengths=input_lengths
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if (iteration + 1) % 5 == 0:
                eff_attn = criterion.current_attention_weight * attention_kl.item()
                print(f"Iteration {iteration+1:4d}, Total: {total_loss.item():.6f}")
                print(f"  Mel: {mel_loss.item():.4f} | Gate: {gate_loss.item():.4f} | "
                      f"Attn(KL raw): {attention_kl.item():.4f} | w: {criterion.current_attention_weight:.2f} | "
                      f"w*KL: {eff_attn:.4f} | σ: {getattr(criterion,'current_sigma',float('nan')):.2f}")
                _, _, _, alignments = model_outputs
                if len(alignments) > 0:
                    last_attention = alignments[-1][0]
                    entropy = -(last_attention * torch.log(last_attention + 1e-8)).sum().item()
                    print(f"  📈 Attention entropy: {entropy:.3f}")
            if (iteration + 1) % 10 == 0:
                _, _, _, alignments = model_outputs
                alignment_path = os.path.join(checkpoint_dir, f"debug_alignment_iter_{iteration+1}.png")
                save_alignment_plot(alignments, alignment_path)
                print(f"🎯 Alignment saved: {alignment_path}")
            if mel_loss.item() < 1.0:
                print(f"🎉 SUCCESS! Mel L1 dropped to {mel_loss.item():.4f} (threshold 1.0)")
                print(f"🎯 Final alignment saved to: debug_alignment_iter_{iteration+1}.png")
                break
        try:
            export_dir = os.path.join(checkpoint_dir, "debug_export")
            os.makedirs(export_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(export_dir, "overfit_model.pth"))
            torch.save({
                "text": text_padded.cpu(),
                "input_lengths": input_lengths.cpu(),
                "mel": mel_padded.cpu(),
                "mel_lengths": mel_lengths.cpu()
            }, os.path.join(export_dir, "debug_batch.pt"))
            print(f"💾 Saved overfit model + batch to {export_dir}")
            _export_debug_inference(
                model,
                {
                    "text": text_padded.cpu(),
                    "input_lengths": input_lengths.cpu(),
                    "mel_lengths": mel_lengths.cpu()
                },
                device,
                export_dir
            )
        except Exception as e:
            print(f"⚠️  Debug export failed: {e}")
        print("🔥 DEBUG MODE COMPLETE")
        return

    # === NORMAL TRAINING MODE ===
    accum_steps = max(1, accum_steps)
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        epoch_loss = 0.0
        print(f"\nEpoch: {epoch + 1}/{epochs}")
        
        model_outputs = None  # INIT: ensure defined even if no batches
        for i, batch in enumerate(data_loader):
            text_padded, input_lengths, mel_padded, mel_lengths = batch
            
            # NEW: length sort (descending) for better attention gradients
            sort_idx = torch.argsort(input_lengths, descending=True)
            text_padded = text_padded[sort_idx].to(device)
            input_lengths = input_lengths[sort_idx].to(device)
            mel_padded = mel_padded[sort_idx].to(device)
            mel_lengths = mel_lengths[sort_idx].to(device)
            
            gate_target = torch.zeros(mel_padded.size(0), mel_padded.size(2), device=device)
            for j, l in enumerate(mel_lengths):
                gate_target[j, l.item()-1:] = 1

            with torch.autocast(device_type=device.type, enabled=use_amp):
                use_postnet = (global_step >= postnet_freeze_steps)
                model_outputs = model(text_padded, mel_padded, input_lengths, use_postnet=use_postnet)
                total_loss, mel_loss, gate_loss, attention_kl = criterion(
                    model_outputs, (mel_padded, gate_target, mel_lengths), text_lengths=input_lengths
                )
                total_loss = total_loss / accum_steps

            scaler.scale(total_loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += total_loss.item() * accum_steps

            eff_attn = criterion.current_attention_weight * attention_kl.item()
            # Logging
            if (global_step % 200) == 0:
                msg = (f"Step {global_step} | Ep {epoch+1} B {i+1}/{len(data_loader)} "
                       f"Total {epoch_loss/ (i+1):.4f} Mel {mel_loss.item():.4f} Gate {gate_loss.item():.4f} "
                       f"KL {attention_kl.item():.4f} w {criterion.current_attention_weight:.2f} σ {getattr(criterion,'current_sigma',float('nan')):.2f} "
                       f"LR {optimizer.param_groups[0]['lr']:.6f}")
                print(msg)
                log_line(msg)
            # Save step checkpoint
            if config.save_every_steps and (global_step % config.save_every_steps == 0) and global_step > 0:
                step_ckpt = os.path.join(checkpoint_dir, f"step_{global_step}.pth")
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss.item(),
                    "best_val_mel": best_val_mel
                }, step_ckpt)
            # Adjust LR on milestones
            adjust_lr(optimizer, global_step)
            global_step += 1

        avg_epoch_loss = epoch_loss / len(data_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")

        # Validation
        if val_loader is not None:
            val_mel, val_gate, val_attn_ent = validate(model, criterion, val_loader, device)
            val_msg = (f"Validation | Epoch {epoch+1} Mel {val_mel:.4f} Gate {val_gate:.4f} "
                       f"AttnEntropy {val_attn_ent:.3f}")
            print(val_msg)
            log_line(val_msg)
            if val_mel < best_val_mel:
                best_val_mel = val_mel
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mel": val_mel,
                    "best_val_mel": best_val_mel
                }, best_path)
                print(f"Saved best checkpoint: {best_path}")
        # Epoch checkpoint (include global_step)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'best_val_mel': best_val_mel
        }, os.path.join(checkpoint_dir, f"tacotron2_epoch_{epoch+1}.pth"))
        # Alignment save (last batch)
        if model_outputs is not None:
            _, _, _, alignments = model_outputs
            alignment_path = os.path.join(checkpoint_dir, f"alignment_epoch_{epoch+1}.png")
            save_alignment_plot(alignments, alignment_path)
    print("\nTraining complete.")

def _mel_scale_diagnostics(mel: torch.Tensor, tag: str):
    """
    Print statistics to detect if mel looks like (0,1) linear vs log-compressed.
    mel shape expected (B, n_mels, T)
    """
    with torch.no_grad():
        m = mel.float()
        stats = {
            "min": float(m.min().item()),
            "max": float(m.max().item()),
            "mean": float(m.mean().item()),
            "std": float(m.std().item()),
        }
        # Sample percentiles (flattened)
        flat = m.view(-1)
        pct = torch.quantile(flat, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=flat.device)).cpu().numpy()
        # Heuristics
        linear_like = stats["min"] >= -1e-4 and 0.0 <= stats["max"] <= 1.05
        narrow_dyn = (stats["max"] - stats["min"]) < 1.2  # log-mels usually span several dB
        print(f"[MEL DIAG] {tag}: min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} std {stats['std']:.4f}")
        print(f"[MEL DIAG] {tag}: p01 {pct[0]:.4f} p05 {pct[1]:.4f} p50 {pct[2]:.4f} p95 {pct[3]:.4f} p99 {pct[4]:.4f}")
        if linear_like and narrow_dyn:
            print(f"[MEL DIAG] {tag}: Looks like 0–1 linear or min-max normalized (NOT log). HiFi-GAN pretrained expects log-mel (negative values).")
        else:
            print(f"[MEL DIAG] {tag}: Distribution may already be log-compressed (presence of negatives or wider dynamic range).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', type=str, help='Path to the metadata file.')
    parser.add_argument('checkpoint_dir', type=str, help='Directory to save checkpoints.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--debug', action='store_true', help='Debug mode: overfit on single batch.')
    parser.add_argument('--val_metadata', type=str, default=None, help='Optional validation metadata CSV.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume.')
    parser.add_argument('--postnet_freeze_steps', type=int, default=None, help='Override PostNet freeze steps.')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps.')
    
    args = parser.parse_args()
    
    train(
        metadata_path=args.metadata,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        debug_overfit=args.debug,
        val_metadata=args.val_metadata,
        resume=args.resume,
        postnet_freeze_steps_override=args.postnet_freeze_steps,
        accum_steps=args.accum_steps
    )