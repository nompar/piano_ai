# ==========================================
# FUNCTIONS TO PREPROCESS DATA FOR MODELING
# ==========================================

from params import *

import numpy as np
import librosa
import os
import glob
import pretty_midi

# ===================================
# Feature Extraction from Audio File
# ===================================

def audio_to_mel_3d(audio_path, sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT, n_mels=N_MELS, normalize=True):
    """Load audio and convert to mel-spectrogram 3D array (n_mels, T, 1)"""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    if normalize:
        S_min, S_max = S_db.min(), S_db.max()
        S_db = (S_db - S_min) / (S_max - S_min + 1e-8)
    return S_db[..., np.newaxis]


def chunk_mel(mel, chunk_size=3000):
    """Chunk mel-spectrogram along time dimension, return list of chunks (n_mels, chunk_size, 1)"""
    n_mels, T, c = mel.shape
    mel_T_first = np.transpose(mel, (1, 0, 2))  # (T, n_mels, 1)
    n_chunks = T // chunk_size
    chunks = [mel_T_first[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
    chunks = [np.transpose(chunk, (1, 0, 2)) for chunk in chunks]
    return chunks


def extract_and_save_mel_features(audio_dir, out_dir, chunk_size=3000):
    """Extract and chunk mel-spectrograms, save in out_dir/mel_npz/"""
    mel_dir = os.path.join(out_dir, "mel_npz")
    os.makedirs(mel_dir, exist_ok=True)
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    audio_files = []
    for ext in exts:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    audio_files = sorted(audio_files)

    print(f"\n=== Extracting and chunking mel-spectrograms ===")
    print(f"Found {len(audio_files)} audio files\n")

    for i, audio_path in enumerate(audio_files, 1):
        name = os.path.splitext(os.path.basename(audio_path))[0]
        mel = audio_to_mel_3d(audio_path)
        chunks = chunk_mel(mel, chunk_size=chunk_size)
        if not chunks:
            print(f"⚠️ {name} too short, skipped")
            continue
        for idx, chunk in enumerate(chunks, 1):
            out_file = os.path.join(mel_dir, f"{name}--chunk_{idx}.npz")
            if os.path.exists(out_file):
                print(f"Skipping existing: {out_file}")
                continue
            np.savez_compressed(out_file, mel=chunk.astype(np.float32), sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT, n_mels=N_MELS)
            print(f"[{i}/{len(audio_files)}] Saved: {out_file}")
    print("✅ Finished extracting and chunking mel-spectrograms\n")


# ===================================
# MIDI to target extraction
# ===================================

def midi_to_targets(midi_path, n_frames, fps=FPS, pitch_min=PITCH_MIN, pitch_max=PITCH_MAX):
    """Convert MIDI to onset/offset/active/velocity/pedal arrays aligned with n_frames"""
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")

    n_pitches = pitch_max - pitch_min + 1
    onset = np.zeros((n_frames, n_pitches), np.float32)
    offset = np.zeros((n_frames, n_pitches), np.float32)
    active = np.zeros((n_frames, n_pitches), np.float32)
    velocity = np.zeros((n_frames, n_pitches), np.float32)
    pedal = np.zeros((n_frames, 1), np.float32)

    pm = pretty_midi.PrettyMIDI(midi_path)
    for inst in pm.instruments:
        if inst.is_drum: continue
        for note in inst.notes:
            if not (pitch_min <= note.pitch <= pitch_max): continue
            p = note.pitch - pitch_min
            f_on = int(round(note.start * fps))
            f_off = int(round(note.end * fps))
            f_on = max(0, min(n_frames - 1, f_on))
            f_off = max(0, min(n_frames - 1, f_off))
            onset[f_on, p] = 1.0
            offset[f_off, p] = 1.0
            active[f_on:f_off+1, p] = 1.0
            velocity[f_on, p] = note.velocity / 127.0

    # Pedal
    cc64 = [(cc.time, cc.value) for inst in pm.instruments for cc in inst.control_changes if cc.number == 64]
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


def midi_to_targets_chunks(midi_dir, mel_chunks_dir, out_dir, chunk_size=3000, fps=FPS):
    """Generate MIDI targets chunked to match mel chunks, save in out_dir/midi_npz/"""
    midi_out_dir = os.path.join(out_dir, "midi_npz")
    os.makedirs(midi_out_dir, exist_ok=True)

    mel_files = sorted(glob.glob(os.path.join(mel_chunks_dir, "*.npz")))
    print(f"\n=== Generating chunked MIDI targets ===")
    print(f"Found {len(mel_files)} mel chunks\n")

    for mel_path in mel_files:
        mel_name = os.path.basename(mel_path)
        base_name = mel_name.split("--chunk_")[0] + ".midi"
        midi_path = os.path.join(midi_dir, base_name)
        if not os.path.exists(midi_path):
            print(f"⚠️ MIDI missing for {mel_name}")
            continue

        mel = np.load(mel_path)["mel"]
        n_frames = mel.shape[1]
        out_path = os.path.join(midi_out_dir, mel_name.replace(".npz", "_targets.npz"))
        if os.path.exists(out_path):
            print(f"Skipping existing: {out_path}")
            continue

        onset, offset, active, velocity, pedal = midi_to_targets(midi_path, n_frames, fps=fps)
        np.savez_compressed(out_path, onset=onset, offset=offset, active=active, velocity=velocity, pedal=pedal)
        print(f"Saved targets: {out_path}")
