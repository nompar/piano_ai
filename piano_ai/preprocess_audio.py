# ==========================================
# FUNCTIONS TO PREPROCESS DATA FOR MODELING
# ==========================================

from piano_ai.params import *  # Import global parameters for audio/MIDI processing

import numpy as np     # For numerical operations and arrays
import librosa         # For audio loading and feature extraction
import os              # For file and directory operations
import glob            # For finding files by pattern
import pretty_midi     # For reading and processing MIDI files

# ===================================
# Feature Extraction from Audio File
# ===================================

def audio_to_mel_3d(
    audio_path, sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT,
    n_mels=N_MELS, normalize=True
    ):
    """Transform the sound signal into a form
    that a machine learning model can understand and use.
    Load audio and convert to mel-spectrogram 3D array (n_mels, T, 1)"""

    # Load audio file as a waveform
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    # Compute mel-spectrogram from waveform
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels, power=2.0
        )
    # Convert power spectrogram to decibel units
    S_db = librosa.power_to_db(S, ref=np.max)
    # Optionally normalize the spectrogram to [0, 1]
    if normalize:
        S_min, S_max = S_db.min(), S_db.max()
        S_db = (S_db - S_min) / (S_max - S_min + 1e-8)
    # Add a channel dimension for compatibility with models
    return S_db[..., np.newaxis]


def chunk_mel(mel, chunk_size=3000):
    """Chunk mel-spectrogram along time dimension,
    return list of chunks (n_mels, chunk_size, 1)
    This makes it easier for memory efficiency,
    model training and data augmentation"""

    # Get shape of mel-spectrogram
    n_mels, T, c = mel.shape
    # Transpose to put time dimension first
    mel_T_first = np.transpose(mel, (1, 0, 2))  # (T, n_mels, 1)
    # Calculate number of chunks
    n_chunks = T // chunk_size
    # Split into chunks along time axis
    chunks = [
        mel_T_first[i*chunk_size:(i+1)*chunk_size]
        for i in range(n_chunks)
        ]
    # Transpose back to original shape for each chunk
    chunks = [np.transpose(chunk, (1, 0, 2)) for chunk in chunks]
    return chunks


def extract_and_save_mel_features(audio_dir, out_dir, chunk_size=3000):
    """extracts their mel-spectrogram features, splits them into smaller chunks,
    and saves each chunk as a compressed .npz file for later use
    in model training, saving it in out_dir/mel_npz/"""

    # Create output directory for mel features
    mel_dir = os.path.join(out_dir, "mel_npz")
    os.makedirs(mel_dir, exist_ok=True)
    # Supported audio file extensions
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    audio_files = []
    # Find all audio files in the directory
    for ext in exts:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    audio_files = sorted(audio_files)

    print(f"\n=== Extracting and chunking mel-spectrograms ===")
    print(f"Found {len(audio_files)} audio files\n")

    # Process each audio file
    for i, audio_path in enumerate(audio_files, 1):
        name = os.path.splitext(os.path.basename(audio_path))[0]
        # Extract mel-spectrogram
        mel = audio_to_mel_3d(audio_path)
        # Split into chunks
        chunks = chunk_mel(mel, chunk_size=chunk_size)
        if not chunks:
            print(f"⚠️ {name} too short, skipped")
            continue
        # Save each chunk as a compressed .npz file
        for idx, chunk in enumerate(chunks, 1):
            out_file = os.path.join(mel_dir, f"{name}--chunk_{idx}.npz")
            if os.path.exists(out_file):
                print(f"Skipping existing: {out_file}")
                continue
            np.savez_compressed(
                out_file, mel=chunk.astype(np.float32),
                sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT,
                n_mels=N_MELS
                )
            print(f"[{i}/{len(audio_files)}] Saved: {out_file}")
    print("✅ Finished extracting and chunking mel-spectrograms\n")


# ===================================
# MIDI to target extraction
# ===================================

def midi_to_targets(
    midi_path,n_frames, fps=FPS, pitch_min=PITCH_MIN, pitch_max=PITCH_MAX
    ):
    """Convert MIDI to onset/offset/active/velocity/pedal
    arrays aligned with n_frames"""

    # Check for valid number of frames
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")

    # Calculate number of pitches in piano range
    n_pitches = pitch_max - pitch_min + 1
    # Initialize target arrays for each label
    onset = np.zeros((n_frames, n_pitches), np.float32)
    offset = np.zeros((n_frames, n_pitches), np.float32)
    active = np.zeros((n_frames, n_pitches), np.float32)
    velocity = np.zeros((n_frames, n_pitches), np.float32)
    pedal = np.zeros((n_frames, 1), np.float32)

    # Load MIDI file
    pm = pretty_midi.PrettyMIDI(midi_path)
    # Iterate over instruments and notes
    for inst in pm.instruments:
        if inst.is_drum: continue  # Skip drum tracks
        for note in inst.notes:
            if not (pitch_min <= note.pitch <= pitch_max): continue  # Only use piano range
            p = note.pitch - pitch_min
            # Convert note start/end times to frame indices
            f_on = int(round(note.start * fps))
            f_off = int(round(note.end * fps))
            # Clip indices to valid range
            f_on = max(0, min(n_frames - 1, f_on))
            f_off = max(0, min(n_frames - 1, f_off))
            # Set onset, offset, active, and velocity values
            onset[f_on, p] = 1.0
            offset[f_off, p] = 1.0
            active[f_on:f_off+1, p] = 1.0
            velocity[f_on, p] = note.velocity / 127.0  # Normalize velocity

    # Extract pedal (sustain) information from MIDI control changes
    cc64 = [
        (cc.time, cc.value) for inst in pm.instruments
        for cc in inst.control_changes
        if cc.number == 64
        ]
    cc64.sort()
    for f in range(n_frames):
        t = f / fps
        state = 0.0
        for ct, cv in cc64:
            if ct <= t:
                state = 1.0 if cv >= 64 else 0.0
            else:
                break
        pedal[f, 0] = state

    return onset, offset, active, velocity, pedal


def midi_to_targets_chunks(
    midi_dir, mel_chunks_dir, out_dir,
    chunk_size=3000, fps=FPS
    ):
    """Takes a MIDI file and converts it into arrays that represent
    what notes are being played (and how) at each time frame,
    so a machine learning model can learn from it.
    Save in out_dir/midi_npz/"""

    # Create output directory for MIDI targets
    midi_out_dir = os.path.join(out_dir, "midi_npz")
    os.makedirs(midi_out_dir, exist_ok=True)

    # Find all mel chunk files
    mel_files = sorted(glob.glob(os.path.join(mel_chunks_dir, "*.npz")))
    print(f"\n=== Generating chunked MIDI targets ===")
    print(f"Found {len(mel_files)} mel chunks\n")

    # Process each mel chunk
    for mel_path in mel_files:
        mel_name = os.path.basename(mel_path)
        # Match mel chunk to corresponding MIDI file
        base_name = mel_name.split("--chunk_")[0] + ".midi"
        midi_path = os.path.join(midi_dir, base_name)
        if not os.path.exists(midi_path):
            print(f"⚠️ MIDI missing for {mel_name}")
            continue

        # Load mel chunk and get number of frames
        mel = np.load(mel_path)["mel"]
        n_frames = mel.shape[1]
        # Set output path for targets
        out_path = os.path.join(
            midi_out_dir,
            mel_name.replace(".npz", "_targets.npz")
            )
        if os.path.exists(out_path):
            print(f"Skipping existing: {out_path}")
            continue

        # Generate targets for this chunk and save
        onset, offset, active, velocity, pedal = midi_to_targets(
            midi_path,
            n_frames,
            fps=fps
            )
        np.savez_compressed(out_path,
                            onset=onset,
                            offset=offset,
                            active=active,
                            velocity=velocity,
                            pedal=pedal
                            )
        print(f"Saved targets: {out_path}")
