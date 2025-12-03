# Entry point: runs the full pipeline (preprocessing, training, evaluation)

# Import functions from preprocess_audio.py
from preprocess_audio import extract_and_save_mel_features
from preprocess_audio import midi_to_targets
import numpy as np
import os

# =======================
# Preprocessing
# =======================

if __name__ == "__main__":

    # --- AUDIO PREPROCESSING ---
    # Set input and output directories for audio
    audio_dir = "./raw_data/2017"      # Folder containing audio files
    out_dir = "./2017_npz"             # Output folder for processed features

    # Extract mel features and save as .npz files
    extract_and_save_mel_features(audio_dir, out_dir)

    # --- MIDI PREPROCESSING ---
    midi_dir = "./raw_data/2017"       # Folder containing MIDI files
    out_dir = "./2017_midi_targets_npz" # Output folder for MIDI targets

    # Ensure output folder exists
    os.makedirs(out_dir, exist_ok=True)

    # Get a list of MIDI files (only files ending with 'midi')
    midi_files = [f for f in sorted(os.listdir(midi_dir)) if f.endswith("midi")]

    print("MIDI files found:", midi_files)

    # Loop through each MIDI file and process
    for midi_path in midi_files:
        print(f"Processing {midi_path}")  # Dummy progress print
        name = os.path.splitext(os.path.basename(midi_path))[0]

        n_frames = 100  # Dummy value: should match your mel features

        # Extract targets from MIDI
        onset, offset, active, velocity, pedal = midi_to_targets(os.path.join(midi_dir, midi_path), n_frames)

        # Save targets as compressed .npz file
        np.savez_compressed(
            os.path.join(out_dir, f"{name}_targets.npz"),
            onset=onset,
            offset=offset,
            active=active,
            velocity=velocity,
            pedal=pedal
        )

# =======================
# Loader : construire le tf.data.Dataset
# =======================

from loader import get_dataset

print("\n=== Construction du dataset TensorFlow ===")

# On réutilise exactement les dossiers de sortie du preprocess
mel_dir = "./2017_npz"
labels_dir = "./2017_midi_targets_npz"

dataset = get_dataset(
    mel_dir=mel_dir,
    labels_dir=labels_dir,
    batch_size=1
)

# Vérification d’un batch
for mel, targets in dataset:
    print("\nBatch chargé ✔️")
    print("mel :", mel.shape)
    print("onset :", targets["onset"].shape)
    print("offset :", targets["offset"].shape)
    print("active :", targets["active"].shape)
    print("velocity :", targets["velocity"].shape)
    print("pedal :", targets["pedal"].shape)
    break
