# test_module.py

from src import config, text, audio

print("✅ Successfully imported our modules!")
print("-" * 20)

# 1. Test the config module
# The vocabulary is now a list of phoneme symbols
print(f"Number of symbols in vocabulary: {len(config.SYMBOLS)}")


# 2. Test the text module with phoneme conversion
sample_text = "This is a test!"

# Get the sequence of integer IDs from our text module
sequence = text.text_to_sequence(sample_text)

# For display purposes, let's map the IDs back to their symbols
id_to_symbol = {i: s for i, s in enumerate(config.SYMBOLS)}
phonemes = [id_to_symbol[i] for i in sequence]


print(f"\nOriginal text: '{sample_text}'")
print(f"Phonemes: {phonemes}")
print(f"Sequence of IDs: {sequence}")


# 3. Test that the audio module is available
print(f"\nThe audio function is ready to use: {callable(audio.get_mel_spectrogram)}")