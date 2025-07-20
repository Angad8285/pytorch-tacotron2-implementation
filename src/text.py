# src/text.py

import re
import inflect
from . import config

# Initialize the number-to-words engine
p = inflect.engine()
# Create a mapping from symbols to integer IDs
symbol_to_id = {s: i for i, s in enumerate(config.SYMBOLS)}

def normalize_text(text: str) -> str:
    """
    Normalizes a string of text for TTS processing.
    """
    text = text.lower()
    
    # Expand numbers to words
    text = re.sub(r"(\d+)", lambda m: p.number_to_words(m.group(0)), text)
    
    # Expand common abbreviations
    abbreviations = {
        "mr.": "mister",
        "mrs.": "missis",
        "dr.": "doctor",
        "etc.": "et cetera",
    }
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)
        
    # --- THIS IS THE NEW, FIXED STEP ---
    # Replace punctuation that should be a space (like periods or commas) with a space
    text = re.sub(r'[.,-]', ' ', text)
    
    # Remove any remaining non-alphanumeric/non-space characters (like '$')
    text = re.sub(r'[^\w\s]', '', text)
    
    # Collapse multiple spaces into one and strip leading/trailing space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def text_to_sequence(text: str) -> list[int]:
    """
    Converts a string of cleaned text into a sequence of integer IDs.
    """
    sequence = []
    # First, normalize the text
    cleaned_text = normalize_text(text)
    
    # Then, convert each character to its ID
    for symbol in cleaned_text:
        if symbol in symbol_to_id:
            sequence.append(symbol_to_id[symbol])
    return sequence