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
SR = 22050  # Sample rate: audio will be loaded at 22,050 samples per second
# Used whenever we load/process audio files (e.g., with librosa).

FPS = 100   # Frames per second for spectrogram time resolution
# Controls how finely we chop up the audio in time for spectrograms.

HOP_LENGTH = SR // FPS  # Number of samples between successive frames (controls time resolution)
# Used in spectrogram extraction to set the step size between frames.

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
RAW_AUDIO_DIR = "./raw_data/2017"   # Local path to raw audio files
RAW_MIDI_DIR = "./raw_data/2017"    # Local path to raw MIDI files
OUT_DIR = "./2017_npz"              # Where to save processed data (features/labels)
CHUNK_SIZE = 3000                   # How many frames per chunk when splitting data
# These are used throughout the pipeline to tell the code where to find input data,
# where to save outputs, and how to split up the data for processing and training.
EPOCHS = 50
BATCH_SIZE = 4

if READ_MODE == 'local':
    FEATURE_DIR = "/Users/hadriendecaumont/Downloads/all_years_npz_2/mel_npz"
    TARGET_DIR  = "/Users/hadriendecaumont/Downloads/all_years_npz_2/midi_npz"
elif READ_MODE == 'gcp':
    client = storage.Client()
    bucket = client.bucket(BUCKET_ID)
    FEATURE_DIR = bucket.blob('all_years_npz/mel_npz')
    TARGET_DIR = bucket.blob('all_years_npz/midi_npz')
