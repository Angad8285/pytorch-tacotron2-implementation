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
    Location-Sensitive Attention mechanism.
    This implementation is based on the Tacotron 2 paper.
    """
    def __init__(self):
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = nn.Linear(config.attention_rnn_dim, config.attention_dim, bias=False)
        self.memory_layer = nn.Linear(config.encoder_embedding_dim, config.attention_dim, bias=False)
        
        # --- REFINED ---
        # The location convolution now takes 1 channel (the cumulative weights)
        self.location_convolution = nn.Conv1d(
            in_channels=1,
            out_channels=config.location_n_filters,
            kernel_size=config.location_kernel_size,
            stride=1,
            padding=(config.location_kernel_size - 1) // 2,
            bias=False
        )
        self.location_layer = nn.Linear(
            config.location_n_filters,
            config.attention_dim,
            bias=False
        )
        self.v = nn.Linear(config.attention_dim, 1, bias=True)
        self.cumulative_weights = None

    def _get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """Computes the alignment energies."""
        # FIX: query should be (batch, 1, attention_rnn_dim) 
        processed_query = self.query_layer(query)  # (batch, 1, attention_dim)
        
        # Process the location features from cumulative weights
        processed_location = self.location_convolution(attention_weights_cat)
        processed_location = processed_location.transpose(1, 2)  # (batch, time, location_n_filters)
        processed_location = self.location_layer(processed_location)  # (batch, time, attention_dim)
        
        # All should be (batch, time, attention_dim) for broadcasting
        energies = self.v(torch.tanh(
            processed_query + processed_location + processed_memory
        ))  # (batch, time, 1)
        return energies.squeeze(-1)  # (batch, time)

    def forward(self, query, memory, mask):
        """
        Runs the forward pass for the attention mechanism.
        """
        # --- REFINED ---
        # The core logic is simplified here
        
        # Process the cumulative weights to get location features
        attention_weights_cat = self.cumulative_weights.unsqueeze(1) # type: ignore
        
        # Get the alignment scores (energies)
        alignment = self._get_alignment_energies(
            query.unsqueeze(1), self.processed_memory, attention_weights_cat
        )
        
        if mask is not None:
            alignment.data.masked_fill_(mask, -float("inf"))

        # Apply temperature to make attention sharper
        temperature = 0.5 if self.training else 1.0  # Sharper during training
        attention_weights = F.softmax(alignment / temperature, dim=1)
        
        # ANTI-STICKING MECHANISM: Encourage forward movement
        if self.training and hasattr(self, 'step_count'):
            self.step_count += 1
            # More aggressive progression - force attention to move forward
            expected_pos = min(self.step_count * 1.2, alignment.size(1) - 1)  # Faster progression
            
            # Safety check to prevent runaway computation
            if self.step_count > alignment.size(1) * 2:
                # Force attention to move forward if it gets stuck
                expected_pos = min(alignment.size(1) - 1, self.step_count * 0.8)
            
            # Stronger discouragement for staying behind
            discouragement = torch.exp(-0.5 * torch.abs(torch.arange(alignment.size(1), device=alignment.device) - expected_pos))
            discouragement = discouragement.unsqueeze(0).expand_as(alignment)
            alignment = alignment + 2.0 * discouragement.log()  # Much stronger push
            attention_weights = F.softmax(alignment / temperature, dim=1)
            
            # Additional: penalize cumulative weights to prevent sticking
            if hasattr(self, 'cumulative_weights') and self.cumulative_weights is not None:
                cum_penalty = -0.1 * self.cumulative_weights  # Discourage high cumulative attention
                alignment = alignment + cum_penalty
            attention_weights = F.softmax(alignment / temperature, dim=1)
        
        # Update the cumulative weights with regularization to prevent sticking
        self.cumulative_weights = self.cumulative_weights + attention_weights # type: ignore
        
        # Add small noise to prevent attention collapse (reduced amount)
        if self.training:
            noise = torch.randn_like(attention_weights) * 0.005  # Reduced noise
            attention_weights = attention_weights + noise
            attention_weights = F.softmax(attention_weights / temperature, dim=1)  # Re-normalize with temperature
        
        context_vector = torch.bmm(attention_weights.unsqueeze(1), memory)
        context_vector = context_vector.squeeze(1)
        
        return context_vector, attention_weights

    def init_states(self, memory):
        """Initializes the attention states before starting decoding."""
        batch_size = memory.size(0)
        max_time = memory.size(1)
        
        # Ensure we're on the same device as memory
        device = memory.device
        
        # Always reset for new sequence
        self.step_count = 0
        self.cumulative_weights = torch.zeros(batch_size, max_time, device=device)
        # Initialize with small attention bias towards the beginning
        self.cumulative_weights[:, 0] = 0.1
        
        self.processed_memory = self.memory_layer(memory)


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
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
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
        self.decoder_lstm = nn.LSTMCell(
            config.decoder_rnn_dim,
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

    def _initialize_decoder_states(self, memory, mask):
        """Initializes the decoder states for decoding."""
        batch_size = memory.size(0)
        seq_len = memory.size(1)

        # Initialize the attention mechanism
        self.attention.init_states(memory)
        
        # Initialize the first input frame to zeros
        self.decoder_input = torch.zeros(batch_size, self.n_mels, device=memory.device)
        
        # Initialize LSTM hidden states
        self.attention_hidden = torch.zeros(batch_size, config.decoder_rnn_dim, device=memory.device)
        self.attention_cell = torch.zeros(batch_size, config.decoder_rnn_dim, device=memory.device)
        self.decoder_hidden = torch.zeros(batch_size, config.decoder_rnn_dim, device=memory.device)
        self.decoder_cell = torch.zeros(batch_size, config.decoder_rnn_dim, device=memory.device)

    def _parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """Prepares the decoder outputs for returning."""
        # FIX: Proper stacking and transposing
        # mel_outputs: list of (batch, n_mels) -> (batch, time, n_mels)
        mel_outputs = torch.stack(mel_outputs, dim=1)
        # gate_outputs: list of (batch, 1) -> (batch, time)
        gate_outputs = torch.stack(gate_outputs, dim=1).squeeze(-1)
        
        return mel_outputs, gate_outputs, alignments

    def _decode_step(self, memory, mask):
        """Performs a single decoding step."""
        # 1. Pass the previous frame through the Pre-Net
        prenet_output = self.prenet(self.decoder_input)
        
        # 2. Get the context vector from the attention mechanism
        # FIX: Use attention_hidden as query (correct in original Tacotron2)
        context_vector, self.attention_weights = self.attention.forward(
            self.attention_hidden, memory, mask
        )

        # 3. First LSTM: takes prenet output and context vector
        lstm1_input = torch.cat((prenet_output, context_vector), dim=-1)
        self.attention_hidden, self.attention_cell = self.attention_lstm(
            lstm1_input, (self.attention_hidden, self.attention_cell)
        )
        
        # 4. Second LSTM: takes output of the first LSTM
        self.decoder_hidden, self.decoder_cell = self.decoder_lstm(
            self.attention_hidden, (self.decoder_hidden, self.decoder_cell)
        )

        # 5. Project to get the mel frame and stop token
        projection_input = torch.cat((self.decoder_hidden, context_vector), dim=-1)
        
        mel_output = self.linear_projection(projection_input)
        gate_output = self.gate_layer(projection_input)
        
        return mel_output, gate_output, self.attention_weights

    def forward(self, memory, decoder_inputs, mask):
        """
        The forward pass for training (uses teacher forcing).
        """
        # Prepare the ground-truth spectrogram for the decoder
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = torch.cat((torch.zeros_like(decoder_inputs[:, :1, :]), decoder_inputs[:, :-1, :]), dim=1)

        self._initialize_decoder_states(memory, mask)

        mel_outputs, gate_outputs, alignments = [], [], []
        
        # Loop through each frame of the ground-truth spectrogram
        for i in range(decoder_inputs.size(1)):
            self.decoder_input = decoder_inputs[:, i, :]
            mel_output, gate_output, attention_weights = self._decode_step(memory, mask)
            
            # FIX: Don't squeeze - keep proper dimensions
            mel_outputs.append(mel_output)  # (batch, n_mels)
            gate_outputs.append(gate_output)  # (batch, 1)
            alignments.append(attention_weights)
            
        return self._parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

    def inference(self, memory):
        """
        The forward pass for inference (autoregressive).
        """
        self._initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []

        # Loop until the model predicts to stop or we hit the max steps
        while True:
            mel_output, gate_output, attention_weights = self._decode_step(memory, mask=None)
            
            # FIX: Don't squeeze - keep proper dimensions
            mel_outputs.append(mel_output)  # (batch, n_mels)
            gate_outputs.append(gate_output)  # (batch, 1)
            alignments.append(attention_weights)
            
            # Check for stop condition
            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps.")
                break
            
            # Use the predicted frame as the input for the next step
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

    # In the Tacotron2 class in src/model.py

    def forward(self, text_inputs, mel_targets):
        """
        The main forward pass for training.
        """
        encoder_outputs = self.encoder(text_inputs)
        
        mel_outputs_coarse, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_targets, mask=None
        )
        
        # FIX: PostNet expects (batch, n_mels, time), decoder now outputs (batch, time, n_mels)
        mel_outputs_coarse_transposed = mel_outputs_coarse.transpose(1, 2)
        postnet_residual = self.postnet(mel_outputs_coarse_transposed)
        
        # Convert back to (batch, time, n_mels) to match decoder output format
        postnet_residual = postnet_residual.transpose(1, 2)
        
        mel_outputs_postnet = mel_outputs_coarse + postnet_residual
        
        return (mel_outputs_postnet, mel_outputs_coarse, gate_outputs, alignments)

    def inference(self, text_inputs):
        """
        The forward pass for inference (generating new audio).
        """
        encoder_outputs = self.encoder(text_inputs)
        
        mel_outputs_coarse, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
        
        # FIX: Same PostNet handling as training
        mel_outputs_coarse_transposed = mel_outputs_coarse.transpose(1, 2)
        postnet_residual = self.postnet(mel_outputs_coarse_transposed)
        postnet_residual = postnet_residual.transpose(1, 2)
        
        mel_outputs_postnet = mel_outputs_coarse + postnet_residual
        
        # Return in (batch, time, n_mels) format for consistency
        return (mel_outputs_postnet, mel_outputs_coarse, gate_outputs, alignments)