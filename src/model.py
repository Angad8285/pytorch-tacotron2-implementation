# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class Encoder(nn.Module):
    """
    Encoder module:
    - A phoneme embedding layer
    - A stack of 3 convolutional layers
    - A bidirectional LSTM
    """
    def __init__(self):
        super(Encoder, self).__init__()

        # Phoneme embedding layer
        self.embedding = nn.Embedding(
            len(config.SYMBOLS),
            config.symbols_embedding_dim
        )

        # Stack of 3 convolutional layers
        convs = []
        for _ in range(config.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=config.encoder_embedding_dim,
                    out_channels=config.encoder_embedding_dim,
                    kernel_size=config.encoder_kernel_size,
                    stride=1,
                    padding=int((config.encoder_kernel_size - 1) / 2),
                    dilation=1
                ),
                nn.BatchNorm1d(config.encoder_embedding_dim)
            )
            convs.append(conv_layer)
        self.convolutions = nn.ModuleList(convs)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            config.encoder_embedding_dim,
            int(config.encoder_embedding_dim / 2), # 256 for each direction
            1, # num_layers
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        """
        Runs the forward pass for the encoder.
        Args:
            x: The input phoneme IDs (batch, sequence_length)
        Returns:
            The encoder's output, a sequence of feature vectors.
        """
        # 1. Pass through the embedding layer
        x = self.embedding(x) # (batch, seq_len, embedding_dim)
        
        # 2. Transpose for convolutional layers
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2) # (batch, embedding_dim, seq_len)

        # 3. Pass through the convolutional stack
        for conv in self.convolutions:
            x = F.relu(conv(x))

        # 4. Transpose back for the LSTM layer
        # LSTM expects (batch, seq_len, features)
        x = x.transpose(1, 2) # (batch, seq_len, embedding_dim)

        # 5. Pass through the bidirectional LSTM
        # The outputs are concatenated automatically (256 forward + 256 backward = 512)
        outputs, _ = self.lstm(x)

        return outputs


class LocationSensitiveAttention(nn.Module):
    """
    Standard Location-Sensitive Attention (Tacotron 2):
    energies_t = v^T tanh(W_query * query + W_mem * memory + W_loc * F(attn_prev, attn_cum))
    where F is a 1D conv over concatenated previous and cumulative attention weights.
    """
    def __init__(self):
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = nn.Linear(config.attention_rnn_dim, config.attention_dim, bias=False)
        self.memory_layer = nn.Linear(config.encoder_embedding_dim, config.attention_dim, bias=False)
        # 2-channel input: previous attention + cumulative attention
        self.location_conv = nn.Conv1d(
            in_channels=2,
            out_channels=config.location_n_filters,
            kernel_size=config.location_kernel_size,
            padding=(config.location_kernel_size - 1)//2,
            bias=False
        )
        self.location_dense = nn.Linear(config.location_n_filters, config.attention_dim, bias=False)
        self.v = nn.Linear(config.attention_dim, 1, bias=True)
        self.mask = None
        # NEW: learnable energy scale (temperature inverse)
        self.energy_scale = nn.Parameter(torch.tensor(1.2))  # initialized >1 for mild sharpening

    def init_states(self, memory, mask=None):
        """
        Pre-compute processed memory and reset attention weights.
        memory: (B, T_enc, D_enc)
        mask: optional bool tensor (B, T_enc) where True = PAD to be masked out
        """
        device = memory.device
        B, T_enc, _ = memory.size()
        self.processed_memory = self.memory_layer(memory)            # (B, T_enc, attn_dim)
        self.prev_attn = torch.zeros(B, T_enc, device=device)        # previous step attention
        self.cum_attn = torch.zeros(B, T_enc, device=device)         # cumulative attention
        self.mask = mask

    def get_alignment_energies(self, query):
        """
        query: (B, attn_rnn_dim)
        returns: energies (B, T_enc)
        """
        # (B, T_enc, attn_dim)
        processed_query = self.query_layer(query).unsqueeze(1)       # (B, 1, attn_dim)
        # Location features
        loc_feats_in = torch.stack([self.prev_attn, self.cum_attn], dim=1)  # (B, 2, T_enc)
        loc_feats = self.location_conv(loc_feats_in)                        # (B, F, T_enc)
        loc_feats = loc_feats.transpose(1, 2)                               # (B, T_enc, F)
        loc_feats = self.location_dense(loc_feats)                          # (B, T_enc, attn_dim)
        energies = self.v(torch.tanh(processed_query + self.processed_memory + loc_feats)).squeeze(-1)
        energies = energies * self.energy_scale  # NEW: scale energies
        if self.mask is not None:
            energies = energies.masked_fill(self.mask, -1e9)
        return energies

    def forward(self, query, memory):
        """
        query: attention LSTM hidden state (B, attn_rnn_dim)
        memory: (B, T_enc, D_enc) (unused directly here except for shape integrity)
        returns: context (B, D_enc), attention_weights (B, T_enc)
        """
        energies = self.get_alignment_energies(query)           # (B, T_enc)
        attn_weights = F.softmax(energies, dim=1)
        # Update running stats
        self.prev_attn = attn_weights
        self.cum_attn = self.cum_attn + attn_weights
        # Context
        context = torch.bmm(attn_weights.unsqueeze(1), memory).squeeze(1)  # (B, D_enc)
        return context, attn_weights


# src/model.py
# (Keep all imports and the Encoder and Attention classes above)

class PreNet(nn.Module):
    """
    A pre-net module for the Tacotron 2 decoder.
    This acts as an information bottleneck, as described in the paper.
    """
    def __init__(self, in_dim, sizes):
        super(PreNet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)]
        )

    def forward(self, x):
        # FIX: use self.training so dropout is disabled during model.eval()
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
        return x


class Decoder(nn.Module):
    """
    The Tacotron 2 decoder module.
    Generates a mel spectrogram from the encoder's output.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.n_mels = config.n_mels
        self.max_decoder_steps = config.max_decoder_steps
        self.gate_threshold = config.gate_threshold

        # The Pre-Net processes the input spectrogram frame
        self.prenet = PreNet(
            self.n_mels,
            [config.prenet_dim, config.prenet_dim]
        )

        # The attention mechanism
        self.attention = LocationSensitiveAttention()

        # The main autoregressive LSTMs
        self.attention_lstm = nn.LSTMCell(
            config.prenet_dim + config.encoder_embedding_dim,
            config.decoder_rnn_dim
        )
        # BUG FIX: must accept (decoder_rnn_dim + encoder_embedding_dim) = 1024 + 512 = 1536
        self.decoder_lstm = nn.LSTMCell(
            config.decoder_rnn_dim + config.encoder_embedding_dim,
            config.decoder_rnn_dim
        )
        
        # Linear layers to project the final output
        self.linear_projection = nn.Linear(
            config.decoder_rnn_dim + config.encoder_embedding_dim,
            self.n_mels
        )
        self.gate_layer = nn.Linear(
            config.decoder_rnn_dim + config.encoder_embedding_dim,
            1
        )
        # Initialize gate bias negative so initial stop prob is low
        with torch.no_grad():
            if self.gate_layer.bias is not None:
                self.gate_layer.bias.fill_(-3.0)  # sigmoid ~0.047

    def _initialize_decoder_states(self, memory, mask):
        """Initializes the decoder states for decoding."""
        batch_size = memory.size(0)
        seq_len = memory.size(1)

        # Initialize the attention mechanism
        self.attention.init_states(memory, mask)  # pass mask (can be None)
        # Initialize go frame
        self.decoder_input = torch.zeros(memory.size(0), self.n_mels, device=memory.device)
        # Init LSTM states
        self.attention_hidden = torch.zeros(memory.size(0), config.decoder_rnn_dim, device=memory.device)
        self.attention_cell = torch.zeros(memory.size(0), config.decoder_rnn_dim, device=memory.device)
        self.decoder_hidden = torch.zeros(memory.size(0), config.decoder_rnn_dim, device=memory.device)
        self.decoder_cell = torch.zeros(memory.size(0), config.decoder_rnn_dim, device=memory.device)
        # Zero initial context
        self.context = torch.zeros(memory.size(0), config.encoder_embedding_dim, device=memory.device)

    def _parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """Prepares the decoder outputs for returning."""
        # FIX: Proper stacking and transposing
        # mel_outputs: list of (batch, n_mels) -> (batch, time, n_mels)
        mel_outputs = torch.stack(mel_outputs, dim=1)
        # gate_outputs: list of (batch, 1) -> (batch, time)
        gate_outputs = torch.stack(gate_outputs, dim=1).squeeze(-1)
        
        return mel_outputs, gate_outputs, alignments

    def _decode_step(self, memory, mask):
        """
        One decoding step (teacher-forced or autoregressive).
        Ordering (Tacotron 2):
          1) Prenet(previous mel)
          2) Attention LSTM over [prenet_out, prev_context]
          3) Compute attention with attention_hidden as query
          4) Decoder LSTM over [attention_hidden, context]
          5) Project outputs
        """
        prenet_out = self.prenet(self.decoder_input)  # (B, prenet_dim)
        attn_lstm_in = torch.cat([prenet_out, self.context], dim=-1)
        self.attention_hidden, self.attention_cell = self.attention_lstm(
            attn_lstm_in, (self.attention_hidden, self.attention_cell)
        )
        # Dropout (attention RNN)
        self.attention_hidden = F.dropout(self.attention_hidden, p=config.p_attention_dropout, training=self.training)
        # Attention
        self.context, attn_weights = self.attention(self.attention_hidden, memory)
        # Decoder LSTM
        decoder_lstm_in = torch.cat([self.attention_hidden, self.context], dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_lstm(
            decoder_lstm_in, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(self.decoder_hidden, p=config.p_decoder_dropout, training=self.training)
        # Projections
        proj_in = torch.cat([self.decoder_hidden, self.context], dim=-1)
        mel_output = self.linear_projection(proj_in)
        gate_output = self.gate_layer(proj_in)
        return mel_output, gate_output, attn_weights

    def forward(self, memory, decoder_inputs, mask):
        """
        Training forward (teacher forcing).
        mask: (B, T_enc) bool, True = PAD to be masked out in attention
        """
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = torch.cat(
            (torch.zeros_like(decoder_inputs[:, :1, :]), decoder_inputs[:, :-1, :]),
            dim=1
        )
        self._initialize_decoder_states(memory, mask)
        # Propagate mask into attention (needed each step only via stored state)
        self.attention.mask = mask
        mel_outputs, gate_outputs, alignments = [], [], []
        for i in range(decoder_inputs.size(1)):
            self.decoder_input = decoder_inputs[:, i, :]
            mel_output, gate_output, attention_weights = self._decode_step(memory, mask)
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)
        return self._parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

    def inference(self, memory, max_len_cap: int | None = None, gate_threshold: float | None = None):
        """
        Autoregressive inference.
        max_len_cap: optional hard cap on decoded frames (overrides class max if smaller)
        gate_threshold: optional temporary override of self.gate_threshold
        """
        self._initialize_decoder_states(memory, mask=None)
        eff_gate_threshold = gate_threshold if gate_threshold is not None else self.gate_threshold
        local_max_steps = min(self.max_decoder_steps, max_len_cap) if max_len_cap else self.max_decoder_steps

        # First diagnostic step (unchanged)
        mel_output_debug, gate_output_debug, _ = self._decode_step(memory, mask=None)
        gate_sig = torch.sigmoid(gate_output_debug)
        print("\n--- DEBUGGING FIRST DECODER STEP ---")
        print(f"Initial Stop Token (first sample): {gate_sig[0,0].item():.4f} | mean(batch): {gate_sig.mean().item():.4f}")
        print("Value should be LOW (<0.5). High value ⇒ immediate stop.")
        print("--- END DEBUGGING ---\n")
        self.decoder_input = mel_output_debug.detach()

        mel_outputs, gate_outputs, alignments = [], [], []
        steps = 0
        while True:
            mel_output, gate_output, attention_weights = self._decode_step(memory, mask=None)
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)
            steps += 1
            # Stop if any sample wants to stop AND we produced at least 2 frames
            if steps > 1 and torch.sigmoid(gate_output).max() > eff_gate_threshold:
                break
            if steps >= local_max_steps:
                print(f"Warning! Reached decode cap ({local_max_steps}).")
                break
            self.decoder_input = mel_output
        return self._parse_decoder_outputs(mel_outputs, gate_outputs, alignments)


class PostNet(nn.Module):
    """
    A 5-layer convolutional network to refine the predicted mel spectrogram.
    Predicts a residual to be added to the decoder's output.
    """
    def __init__(self):
        super(PostNet, self).__init__()
        
        self.convolutions = nn.ModuleList()

        # First convolutional layer
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=config.n_mels,
                    out_channels=config.postnet_embedding_dim,
                    kernel_size=config.postnet_kernel_size,
                    stride=1,
                    padding=(config.postnet_kernel_size - 1) // 2,
                    dilation=1
                ),
                nn.BatchNorm1d(config.postnet_embedding_dim)
            )
        )

        # The next 3 convolutional layers
        for _ in range(1, config.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=config.postnet_embedding_dim,
                        out_channels=config.postnet_embedding_dim,
                        kernel_size=config.postnet_kernel_size,
                        stride=1,
                        padding=(config.postnet_kernel_size - 1) // 2,
                        dilation=1
                    ),
                    nn.BatchNorm1d(config.postnet_embedding_dim)
                )
            )

        # The final convolutional layer
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=config.postnet_embedding_dim,
                    out_channels=config.n_mels,
                    kernel_size=config.postnet_kernel_size,
                    stride=1,
                    padding=(config.postnet_kernel_size - 1) // 2,
                    dilation=1
                ),
                nn.BatchNorm1d(config.n_mels)
            )
        )

    def forward(self, x):
        """
        Runs the forward pass for the Post-Net.
        Args:
            x: The mel spectrogram from the decoder (batch, n_mels, time)
        Returns:
            The residual to be added to the input spectrogram.
        """
        # The paper specifies tanh activations on all but the final layer
        for i, conv_seq in enumerate(self.convolutions):
            if i < len(self.convolutions) - 1:
                x = F.dropout(torch.tanh(conv_seq(x)), 0.5, self.training)
            else:
                x = F.dropout(conv_seq(x), 0.5, self.training)
                
        return x


class Tacotron2(nn.Module):
    """
    The full Tacotron 2 model.
    This class brings together the Encoder, Decoder, and PostNet.
    """
    def __init__(self):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = PostNet()
        self.n_mels = config.n_mels
        # Flag to ensure we only compute & apply projection bias once (lazy init)
        self._projection_bias_initialized = False

    def _maybe_init_projection_bias(self, mel_targets: torch.Tensor):
        """Initialize decoder linear_projection bias to per-mel mean from a batch.

        mel_targets shape (B, n_mels, T). Only run once; subsequent calls no-op.
        Helps early convergence & reduces initial gate spikes.
        """
        if self._projection_bias_initialized:
            return
        proj = self.decoder.linear_projection
        if proj.bias is None:
            self._projection_bias_initialized = True
            return
        with torch.no_grad():
            # Mean over batch and time for each mel channel
            channel_means = mel_targets.mean(dim=(0, 2))  # (n_mels,)
            if channel_means.numel() == proj.bias.numel():
                proj.bias.copy_(channel_means)
                self._projection_bias_initialized = True

    def _make_mask(self, lengths, max_len):
        """
        lengths: (B,) int tensor
        returns bool mask (B, max_len) True where PAD
        """
        ids = torch.arange(0, max_len, device=lengths.device)
        mask = ids.unsqueeze(0) >= lengths.unsqueeze(1)
        return mask  # True = pad

    # In the Tacotron2 class in src/model.py

    def forward(self, text_inputs, mel_targets, text_lengths=None, use_postnet=True):
        """
        text_lengths: (B,) tensor of original (unpadded) text lengths
        """
        # Lazy bias init using real data statistics (first batch)
        if not self._projection_bias_initialized:
            self._maybe_init_projection_bias(mel_targets)
        encoder_outputs = self.encoder(text_inputs)  # (B, T_enc, D)
        if text_lengths is None:
            # Fallback assume full length (no padding)
            text_lengths = torch.full(
                (text_inputs.size(0),),
                text_inputs.size(1),
                dtype=torch.long,
                device=text_inputs.device
            )
        enc_mask = self._make_mask(text_lengths, encoder_outputs.size(1))
        mel_outputs_coarse, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_targets, mask=enc_mask
        )
        if use_postnet:
            mel_outputs_coarse_t = mel_outputs_coarse.transpose(1, 2)
            postnet_residual = self.postnet(mel_outputs_coarse_t).transpose(1, 2)
            mel_outputs_postnet = mel_outputs_coarse + postnet_residual
        else:
            mel_outputs_postnet = mel_outputs_coarse  # NEW: bypass PostNet early
        return (mel_outputs_postnet, mel_outputs_coarse, gate_outputs, alignments)

    def inference(self, text_inputs, text_lengths=None, max_len_cap: int | None = None, gate_threshold: float | None = None):
        """
        Autoregressive inference wrapper.
        Adds optional decode length cap and temporary gate threshold override
        (used only in debug export).
        """
        encoder_outputs = self.encoder(text_inputs)
        # Pass new controls to decoder.inference
        mel_outputs_coarse, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs,
            max_len_cap=max_len_cap,
            gate_threshold=gate_threshold
        )
        # Assert at least a minimal number of frames produced
        if mel_outputs_coarse.size(1) < 3:
            print(f"[WARN] Very short mel length ({mel_outputs_coarse.size(1)}) - possible premature stop. Gate threshold={self.decoder.gate_threshold}")
        mel_outputs_coarse_t = mel_outputs_coarse.transpose(1, 2)
        postnet_residual = self.postnet(mel_outputs_coarse_t).transpose(1, 2)
        mel_outputs_postnet = mel_outputs_coarse + postnet_residual
        return (mel_outputs_postnet, mel_outputs_coarse, gate_outputs, alignments)