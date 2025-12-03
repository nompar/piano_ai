# Entry point: runs the full pipeline (preprocessing, training, evaluation)

# Import function and its subfunctions from the preprocess_audio.py
from preprocess_audio import extract_and_save_mel_features
from preprocess_audio import midi_to_targets
import numpy as np
import os


# =======================
# Preprocessing
# =======================

# The following will execute when we call main.py in python
if __name__ == "__main__":

    # Processes audio file to npz

    # Where we get audio from
    audio_dir = "./raw_data/2017"  # Folder with your 2017 audio files

    # Where we save it to
    out_dir = "./2017_npz"  # Output folder for processed features
    extract_and_save_mel_features(audio_dir, out_dir)

    # Process one MIDI file to targets
    midi_dir = "./raw_data/2017"  # Path to your MIDI file
    out_dir = "./2017_midi_targets_npz"
    os.makedirs(out_dir, exist_ok=True)  # Ensure output folder exists

    # Get a list of midi names from the raw data directory
    midi_files = sorted(os.listdir(midi_dir))
    for i, name in enumerate(midi_files):
        if not name.endswith("midi"):
            midi_files.pop(i)

    print(midi_files)

    # Loop through each midi path and pass it to midi_to_targets
    for midi_path in midi_files:
        print(f"Processing {midi_path}")  # <-- Add this line
        name = os.path.splitext(os.path.basename(midi_path))[0]

        n_frames = 100  # Set this to the number of frames you want (should match your mel features)

        # Get the 5 targets
        onset, offset, active, velocity, pedal = midi_to_targets(midi_dir + "/" + midi_path, n_frames)

        # Save the targets for later use
        np.savez_compressed(
            os.path.join(out_dir, f"{name}_targets.npz"),
            onset=onset,
            offset=offset,
            active=active,
            velocity=velocity,
            pedal=pedal
            )
