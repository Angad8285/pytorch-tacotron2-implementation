# Audio Processing Parameters
SAMPLING_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000


# Text Processing Parameters
# The set of ARPAbet phonemes used by the g2p-en library, plus
# punctuation and a space.
SYMBOLS = [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
    'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
    'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0',
    'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1',
    'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
    'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
    ' ', '.', ','
]

# Model Parameters
symbols_embedding_dim = 512

# Encoder parameters
encoder_n_convolutions = 3
encoder_embedding_dim = 512
encoder_kernel_size = 5

# Decoder parameters
n_mels = 80  # Number of mel-spectrogram channels
decoder_rnn_dim = 1024
prenet_dim = 256
max_decoder_steps = 1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1

# Attention parameters
attention_rnn_dim = 1024
attention_dim = 128

# Location-sensitive attention parameters
location_n_filters = 32
location_kernel_size = 31

# Post-Net parameters
postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5


# --- Model Parameters (Heavily Scaled-Down for Fast Testing) ---
symbols_embedding_dim = 128

# Encoder parameters
encoder_n_convolutions = 3
encoder_embedding_dim = 128
encoder_kernel_size = 5

# Decoder parameters
n_mels = 80
decoder_rnn_dim = 256
prenet_dim = 64
max_decoder_steps = 1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1

# Post-Net parameters
postnet_embedding_dim = 256
postnet_kernel_size = 5
postnet_n_convolutions = 5

# Attention parameters
attention_rnn_dim = 256  # Must match decoder_rnn_dim
attention_dim = 64
location_n_filters = 16
location_kernel_size = 17  # Must be an odd number

# In src/config.py
gate_positive_weight = 10
# --- NEW ---
guided_attention_alpha = 5.0