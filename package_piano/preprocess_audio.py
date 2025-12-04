
# ==========================================
# FUNCTIONS TO PREPROCESS DATA FOR MODELING
# ==========================================

# Import global variables
from params import *

# Imports for data loading, cleaning, and spectrogram generation functions
import numpy as np  # Numerical operations, arrays
import librosa      # Audio loading and feature extraction

# Data preprocessing imports
import os           # File and directory operations
import glob         # Pattern matching for file names
import numpy as np  # (Redundant import, but harmless)

# Import prettymidi to read notes and instruments
import pretty_midi


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


# ==============================
# MIDI to target extraction
# ==============================

def midi_to_targets(midi_path, n_frames, fps=FPS, pitch_min=PITCH_MIN, pitch_max=PITCH_MAX):
    """
    Converts a MIDI file into training target matrices for each time frame:
    - onset: when a note starts
    - offset: when a note ends
    - active: when a note is held down
    - velocity: how hard the note is played (normalized)
    - pedal: sustain pedal state

    Args:
        midi_path (str): Path to the MIDI file.
        n_frames (int): Number of time frames to align with audio features.
        fps (int): Frames per second (time resolution).
        pitch_min (int): Lowest piano key to consider.
        pitch_max (int): Highest piano key to consider.

    Returns:
        onset, offset, active, velocity, pedal: Each is a numpy array of shape (n_frames, n_pitches)
    """
    n_pitches = pitch_max - pitch_min + 1  # Total number of piano notes
    onset    = np.zeros((n_frames, n_pitches), np.float32)  # Matrix for note starts
    offset   = np.zeros((n_frames, n_pitches), np.float32)  # Matrix for note ends
    active   = np.zeros((n_frames, n_pitches), np.float32)  # Matrix for notes being held
    velocity = np.zeros((n_frames, n_pitches), np.float32)  # Matrix for note velocities
    pedal    = np.zeros((n_frames, 1), np.float32)          # Matrix for sustain pedal

    pm = pretty_midi.PrettyMIDI(midi_path)  # Load the MIDI file

    for inst in pm.instruments:             # Go through each instrument
        if inst.is_drum: continue           # Skip drums (not piano)
        for note in inst.notes:             # Go through each note
            if not (pitch_min <= note.pitch <= pitch_max): continue  # Skip notes outside piano range
            p = note.pitch - pitch_min      # Convert MIDI pitch to matrix index
            f_on  = int(round(note.start * fps))  # Frame when note starts
            f_off = int(round(note.end   * fps))  # Frame when note ends
            f_on  = max(0, min(n_frames-1, f_on))  # Make sure frame is in bounds
            f_off = max(0, min(n_frames-1, f_off)) # Make sure frame is in bounds
            onset[f_on, p] = 1.0            # Mark note start
            offset[f_off, p] = 1.0          # Mark note end
            active[f_on:f_off+1, p] = 1.0   # Mark note as active between start and end
            velocity[f_on, p] = note.velocity / 127.0  # Normalize velocity (0-1)

    # Handle sustain pedal (MIDI control change 64)
    cc64 = [(cc.time, cc.value) for inst in pm.instruments for cc in inst.control_changes if cc.number == 64]
    cc64.sort()
    for f in range(n_frames):              # For each time frame
        t = f / fps                        # Convert frame to time in seconds
        state = 0.0                        # Default pedal state is off
        for ct, cv in cc64:                # Go through pedal events
            if ct <= t: state = 1.0 if cv >= 64 else 0.0  # Pedal pressed if value >= 64
            else: break
        pedal[f, 0] = state                # Set pedal state for this frame

    return onset, offset, active, velocity, pedal  # Return all target matrices
