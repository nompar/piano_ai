from piano_ai import extract_and_save_mel_features, midi_to_targets_chunks
from loader import get_dataset
from params import *
import os

if __name__ == "__main__":

    ##############################
    # Step 1: Audio preprocessing
    #############################

    # This function goes through all raw audio files, extracts mel spectrogram features,
    # splits them into chunks, and saves them for later use in model training.

    extract_and_save_mel_features(RAW_AUDIO_DIR, OUT_DIR, chunk_size=CHUNK_SIZE)

    #############################
    # Step 2: MIDI preprocessing
    #############################

    # This function processes the MIDI files, matches them to the audio chunks,
    # and creates training labels (onset, offset, velocity, etc.) for each chunk.

    mel_chunks_dir = os.path.join(OUT_DIR, "mel_npz")
    midi_to_targets_chunks(RAW_MIDI_DIR, mel_chunks_dir, OUT_DIR, chunk_size=CHUNK_SIZE)

    ##################################
    # Step 3: Build TensorFlow dataset
    ##################################

    # This loads the processed mel features and MIDI labels into a TensorFlow dataset
    # so the model can use them for training and evaluation.

    print("\n=== Constructing TensorFlow dataset ===")
    dataset = get_dataset(
        mel_dir=os.path.join(OUT_DIR, "mel_npz"),
        labels_dir=os.path.join(OUT_DIR, "midi_npz"),
        batch_size=1
    )
    #############################
    # Step 4: Check the dataset
    #############################

    # This loop loads one batch from the dataset and prints the shapes of the data
    # to verify everything is working as expected.

    for mel, targets in dataset:
        print("\nBatch loaded ✔️")
        print("mel :", mel.shape)
        print("onset :", targets["onset"].shape)
        print("offset :", targets["offset"].shape)
        print("active :", targets["active"].shape)
        print("velocity :", targets["velocity"].shape)
        print("pedal :", targets["pedal"].shape)
        break
