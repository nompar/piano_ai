# =========================
# Global Variables
# =========================

# Library import
import os
from google.cloud import storage


# These variables allow us to optionally set important paths and settings using environment variables.
# This is useful if we want to run the code on different machines or cloud environments without changing the code.
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")      # Path to raw audio/MIDI data (can be set externally)
MAESTRO_DIR = os.environ.get("MAESTRO_DIR")        # Path to MAESTRO dataset (if used)
BUCKET_NAME = os.environ.get("BUCKET_NAME")        # Name of the cloud storage bucket (for cloud workflows)
BUCKET_ID = os.environ.get("BUCKET_ID")            # ID of the cloud storage bucket (for cloud workflows)
ANNEE = os.environ.get("ANNEE")                    # Year or version identifier (if needed)
READ_MODE = os.environ.get("READ_MODE")

# Audio processing parameters:
SR = 22000  # Sample rate: audio will be loaded at 22,050 samples per second
# Used whenever we load/process audio files (e.g., with librosa).

HOP_LENGTH = 220  # Number of samples between successive frames (controls time resolution)
# Used in spectrogram extraction to set the step size between frames.

FPS = SR / HOP_LENGTH   # Frames per second for spectrogram time resolution
# Controls how finely we chop up the audio in time for spectrograms.

N_FFT = 2048    # Number of samples per FFT window (controls frequency resolution)
# Used in spectrogram extraction to set the frequency resolution.

N_MELS = 128    # Number of mel bands (vertical resolution of mel-spectrogram)
# Sets how many frequency bins we use for the mel-spectrogram.

# Piano range variables:
PITCH_MIN = 21    # A0 (lowest piano key)
PITCH_MAX = 108   # C8 (highest piano key)
N_PITCHES = PITCH_MAX - PITCH_MIN + 1
# Used to define the range of notes we care about for MIDI and label processing.

# ----------------------
# Configuration paths
# ----------------------
RAW_AUDIO_DIR = "./raw_data/small_2017"   # Local path to raw audio files
RAW_MIDI_DIR = "./raw_data/small_2017"    # Local path to raw MIDI files
OUT_DIR_MIDI = "./small_y_pred_midi_2017"
MODEL_DIR = f"{BUCKET_ID}/data_08_09_11_18/model"
# Where to save processed data (features/labels)
CHUNK_SIZE = 3000                   # How many frames per chunk when splitting data
# These are used throughout the pipeline to tell the code where to find input data,
# where to save outputs, and how to split up the data for processing and training.

POS_WEIGHT = 50
EPOCHS = 50
BATCH_SIZE = 16
POS_WEIGHT = 50

# --- Data source selection ---
# This block sets the paths for feature and target data depending on where you want to load from
if READ_MODE == 'local':
    # Use local disk paths for training data
    FEATURE_DIR = "./small_2017_npz"   # Local mel-spectrograms
    TARGET_DIR = "./small_2017_target_npz"  # Local MIDI targets
elif READ_MODE == 'gcp':
    # Use GCP bucket paths for training data (as strings, not Blob objects)
    FEATURE_DIR = f"{BUCKET_ID}/data_08_09_11_18/mel_npz"   # GCP mel-spectrograms
    TARGET_DIR  = f"{BUCKET_ID}/data_08_09_11_18/targets_npz"  # GCP MIDI targets
