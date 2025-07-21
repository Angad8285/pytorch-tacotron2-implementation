# test_attention.py
import torch
from src import config
from src.model import Encoder, LocationSensitiveAttention

# --- Setup ---
# Create dummy instances of our modules
encoder = Encoder()
attention = LocationSensitiveAttention()

# Create dummy input data
# (batch_size=2, sequence_length=15)
text_input = torch.randint(low=0, high=len(config.SYMBOLS), size=(2, 15), dtype=torch.long)

# Create a dummy decoder hidden state (this would normally come from the Decoder's LSTM)
decoder_hidden_state = torch.randn(2, config.attention_rnn_dim)

# --- The Correct Workflow ---
print("Running the correct workflow...")

# 1. Get the encoder outputs for the sentence
encoder_outputs = encoder(text_input)
print(f"Encoder output shape: {encoder_outputs.shape}")

# 2. IMPORTANT: Initialize the attention states for this sentence
attention.init_states(encoder_outputs)
print("Attention states initialized successfully.")

# 3. Now you can call the forward method (simulating one step of the decoder)
context_vector, attention_weights = attention.forward(decoder_hidden_state, encoder_outputs, mask=None)

print("\nSuccessfully ran one step of attention!")
print(f"Context vector shape: {context_vector.shape}")
print(f"Attention weights shape: {attention_weights.shape}")