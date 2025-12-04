# Expose key functions at the package level for easy import.
#
# IMPORTANT: When you add new functions to any module (e.g. preprocess_audio.py, loader.py)
# that should be available directly from the piano_ai package (e.g. from piano_ai import new_function),
# add them to the relevant import list below.
#
# This keeps the package interface clean and user-friendly.
# Example: To expose a function from loader.py, add a new import block for loader.py.

# Functions from preprocess_audio.py
from .preprocess_audio import (
    extract_and_save_mel_features,
    midi_to_targets,
    audio_to_mel_3d,
    # Add new functions from preprocess_audio.py here
)

# Functions from loader.py
# from .loader import (
#     load_training_data,
#     load_test_data,
#     # Add new functions from loader.py here
# )
