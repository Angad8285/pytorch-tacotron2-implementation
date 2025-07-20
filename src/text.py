# src/text.py

import re
import inflect
from g2p_en import G2p
from . import config

# --- Step 1: Re-introduce our custom text normalizer ---
# Initialize the number-to-words engine
p = inflect.engine()

def _normalize_text(text: str) -> str:
    """
    Performs basic text cleaning and expands numbers to words.
    This runs BEFORE the G2P conversion.
    """
    text = text.lower()

    # Expand numbers to words
    text = re.sub(r"(\d+)", lambda m: p.number_to_words(m.group(0)), text)

    # Replace hyphens and other punctuation with spaces
    text = re.sub(r'[.,-]', ' ', text)
    
    # Remove any remaining non-alphanumeric/non-space characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Step 2: G2P conversion and final sequencing ---
# Initialize the Grapheme-to-Phoneme converter
g2p = G2p()

# Create a mapping from symbols to integer IDs
symbol_to_id = {s: i for i, s in enumerate(config.SYMBOLS)}


def text_to_sequence(text: str) -> list[int]:
    """
    Converts a string of text into a sequence of phoneme IDs.
    This is the main entry point for text processing.
    """
    # First, run our custom normalizer to handle numbers and cleaning
    normalized_text = _normalize_text(text)
    
    # Then, use the G2P converter to get the phoneme sequence
    phonemes = g2p(normalized_text)
    
    sequence = []
    for phoneme in phonemes:
        # Look up the ID for each phoneme
        if phoneme in symbol_to_id:
            sequence.append(symbol_to_id[phoneme])
            
    return sequence