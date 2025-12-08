# Expose key functions at the package level for easy import.
# This keeps the package interface clean and user-friendly.
# Example: To expose a function from loader.py, add a new import block for loader.py.

# Functions from preprocess_audio.py
from .preprocess_audio import (
    extract_and_save_mel_features,
    midi_to_targets,
    audio_to_mel_3d,
    midi_to_targets_chunks,
    chunk_mel
)

# Functions from loader.py
from piano_ai.ml_logic.loader import make_datasets

# Functions from binarizer
from .binarizer import binarize_predictions
