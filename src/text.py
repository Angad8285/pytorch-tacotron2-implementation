# src/text.py

from g2p_en import G2p
from . import config

# Create a mapping from symbols to integer IDs
symbol_to_id = {s: i for i, s in enumerate(config.SYMBOLS)}

# Initialize the Grapheme-to-Phoneme converter
g2p = G2p()

def text_to_sequence(text: str) -> list[int]:
    """
    Converts a string of text into a sequence of phoneme IDs.
    
    This function uses a G2P converter to transform the input text into its
    phonetic representation and then maps each phoneme to its corresponding ID.
    """
    # Use the G2P converter to get the phoneme sequence
    phonemes = g2p(text)
    
    sequence = []
    for phoneme in phonemes:
        # Look up the ID for each phoneme
        if phoneme in symbol_to_id:
            sequence.append(symbol_to_id[phoneme])
            
    return sequence