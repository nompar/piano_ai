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

# =========================
# Feature Extraction
# =========================

def audio_to_mel_3d(
    audio_path,
    sr=SR,
    hop_length=HOP_LENGTH,
    n_fft=N_FFT,
    n_mels=N_MELS,
    normalize=True,
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
    # Load audio file as mono at the target sample rate
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    # Compute mel-spectrogram (power)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    # Convert power spectrogram to decibel scale
    S_db = librosa.power_to_db(S, ref=np.max)
    # Optionally normalize to [0, 1]
    if normalize:
        S_min, S_max = S_db.min(), S_db.max()
        S_db = (S_db - S_min) / (S_max - S_min + 1e-8)
    # Add a singleton channel dimension for compatibility with ML models
    mel_3d = S_db[..., np.newaxis]
    return mel_3d

# =========================
# Batch Processing
# =========================

def batch_preprocess(audio_dir, out_dir):
    """
    Processes all audio files in a directory, extracts mel-spectrogram features,
    and saves them as compressed .npz files for later use in ML models.

    Args:
        audio_dir (str): Directory containing audio files.
        out_dir (str): Directory to save processed feature files.
    """
    # Ensure output directory exists
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
