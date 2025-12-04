from preprocess_audio import extract_and_save_mel_features, midi_to_targets_chunks
from loader import get_dataset
import os

# ----------------------
# Configuration paths
# ----------------------
RAW_AUDIO_DIR = "./raw_data/2017"
RAW_MIDI_DIR = "./raw_data/2017"
OUT_DIR = "./2017_npz"
CHUNK_SIZE = 3000

if __name__ == "__main__":

    # ----------------------
    # Audio preprocessing
    # ----------------------
    extract_and_save_mel_features(RAW_AUDIO_DIR, OUT_DIR, chunk_size=CHUNK_SIZE)

    # ----------------------
    # MIDI preprocessing
    # ----------------------
    mel_chunks_dir = os.path.join(OUT_DIR, "mel_npz")
    midi_to_targets_chunks(RAW_MIDI_DIR, mel_chunks_dir, OUT_DIR, chunk_size=CHUNK_SIZE)

    # ----------------------
    # TensorFlow dataset
    # ----------------------
    from loader import get_dataset

    print("\n=== Constructing TensorFlow dataset ===")
    dataset = get_dataset(
        mel_dir=os.path.join(OUT_DIR, "mel_npz"),
        labels_dir=os.path.join(OUT_DIR, "midi_npz"),
        batch_size=1
    )

    # Vérification d’un batch
    for mel, targets in dataset:
        print("\nBatch loaded ✔️")
        print("mel :", mel.shape)
        print("onset :", targets["onset"].shape)
        print("offset :", targets["offset"].shape)
        print("active :", targets["active"].shape)
        print("velocity :", targets["velocity"].shape)
        print("pedal :", targets["pedal"].shape)
        break
