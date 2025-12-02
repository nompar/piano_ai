
# ==========================================
# FUNCTIONS TO PREPROCESS DATA FOR MODELING
# ==========================================

# Imports for data loading, cleaning, and spectrogram generation functions
import numpy as np  # Numerical operations, arrays
import librosa      # Audio loading and feature extraction

# Data preprocessing imports
import os           # File and directory operations
import glob         # Pattern matching for file names
import numpy as np  # (Redundant import, but harmless)

# =========================
# Global Variables
# =========================

SR = 22050  # Sample rate: audio will be loaded at 22,050 samples per second
FPS = 100   # Frames per second for spectrogram time resolution
HOP_LENGTH = SR // FPS  # Number of samples between successive frames (controls time resolution)
N_FFT = 2048    # Number of samples per FFT window (controls frequency resolution)
N_MELS = 128    # Number of mel bands (vertical resolution of mel-spectrogram)


# ===================================
# Feature Extraction from Audio File
# ===================================

def audio_to_mel_3d(
    audio_path,         # Path to the audio file to process
    sr=SR,              # Desired sample rate for loading audio (default: 22050 Hz)
    hop_length=HOP_LENGTH, # Number of samples between spectrogram frames (controls time resolution)
    n_fft=N_FFT,        # Number of samples per FFT window (controls frequency resolution)
    n_mels=N_MELS,      # Number of mel bands (vertical resolution of mel-spectrogram)
    normalize=True,     # Whether to normalize the spectrogram to [0, 1] range
):
    """
    Loads an audio file and converts it to a normalized mel-spectrogram 3D array.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sample rate for loading audio.
        hop_length (int): Number of samples between frames.
        n_fft (int): FFT window size.
        n_mels (int): Number of mel bands.
        normalize (bool): Whether to normalize the spectrogram to [0, 1].

    Returns:
        np.ndarray: Mel-spectrogram with shape (n_mels, T, 1), where T is the number of time frames.
    """
    # Load audio file as mono at the target sample rate with mono stereo
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Compute mel-spectrogram (power)
    S = librosa.feature.melspectrogram(
        y=y,                # The audio signal (numpy array of samples)
        sr=sr,              # Ensures the spectrogram matches the audio’s sample rate.
        n_fft=n_fft,        # Sets how detailed the frequency analysis is.
        hop_length=hop_length, # Controls how often the spectrogram is sampled in time.
        n_mels=n_mels,      # Sets the number of mel frequency bins (how “tall” the spectrogram is)
        power=2.0,          # Uses the squared amplitude (power) instead of just amplitude, which is standard for audio feature extraction.
    )

    # Convert power spectrogram to decibel scale
    S_db = librosa.power_to_db(S, ref=np.max)  # Transforms the spectrogram values from power (energy) to decibels, making them easier to interpret and more suitable for ML.

    # Optionally normalize to [0, 1]
    if normalize:
        S_min, S_max = S_db.min(), S_db.max()  # Find the minimum and maximum values in the spectrogram.
        S_db = (S_db - S_min) / (S_max - S_min + 1e-8)  # Scale all values to the range [0, 1] for consistency and easier model training.

    # Add a singleton channel dimension for compatibility with ML models
    mel_3d = S_db[..., np.newaxis]  # Converts the 2D spectrogram to 3D (n_mels, T, 1), which is the expected input shape for many deep learning models.

    return mel_3d  # Return the processed mel-spectrogram


# ==============================
# Extract and Save Mel Features
# ==============================

def extract_and_save_mel_features(audio_dir, out_dir):
    """
    Processes all audio files in a directory, extracts mel-spectrogram features,
    and saves them as compressed .npz files for later use in ML models.

    Args:
        audio_dir (str): Directory containing audio files.
        out_dir (str): Directory to save processed feature files.
    """

    # Create the folder output/features if it doesn’t already exist.
    os.makedirs(out_dir, exist_ok=True)

    # Supported audio file extensions
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    audio_files = []

    # Find all audio files in the input directory
    for ext in exts:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    audio_files = sorted(audio_files)

    print(f"Found {len(audio_files)} audio files")
    print(f"sr={SR}, fps={FPS}, hop_length={HOP_LENGTH}, n_mels={N_MELS}")

    # Process each audio file
    for i, audio_path in enumerate(audio_files, 1):
        name = os.path.splitext(os.path.basename(audio_path))[0]  # Get file name without extension
        out_path = os.path.join(out_dir, f"{name}.npz")           # Output file path
        print(f"[{i}/{len(audio_files)}] {audio_path} -> {out_path}")
        mel_3d = audio_to_mel_3d(audio_path)                      # Extract features

        # Save features and metadata as compressed numpy file
        np.savez_compressed(
            out_path,
            mel=mel_3d.astype(np.float32),
            sr=SR,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            n_mels=N_MELS,
        )
    print("Done ✅")
