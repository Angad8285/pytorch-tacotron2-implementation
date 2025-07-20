# test_module.py

# Because this script is in the root, it can see the 'src' package.
from src import config, text, audio

print("✅ Successfully imported our modules!")
print("-" * 20)

# 1. Test the config module
print(f"The sampling rate from config.py is: {config.SAMPLING_RATE}")

# 2. Test the text module
sample_text = "abcdefghijklmnopqrstuvwxyz 1234. Mr. Smith, Dr. Jones, etc."
cleaned_text = text.normalize_text(sample_text)
print(f"Original text: '{sample_text}'")
print(f"Normalized text: '{cleaned_text}'")
print(f"Text to sequence: {text.text_to_sequence(cleaned_text)}")

# 3. Test that the audio module is available
print(f"The audio function is ready to use: {callable(audio.get_mel_spectrogram)}")