
from piano_ai import extract_and_save_mel_features, midi_to_targets_chunks
from piano_ai.loader import make_datasets
from piano_ai.params import *


from piano_ai.ml_logic.model import build_cnn_bilstm_onset_model          # ← supposé dans model.py
from piano_ai.ml_logic.train import train_model          # ← supposé dans train.py
from piano_ai.ml_logic.postprocessing import probs_to_onset_binary, probs_to_midi # ← ton postprocessing.py
from piano_ai.params import *  # RAW_AUDIO_DIR, RAW_MIDI_DIR, OUT_DIR, CHUNK_SIZE, EPOCHS, etc.


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

    print("\n=== Constructing TensorFlow datasets ===")

    train_ds, val_ds, test_ds = make_datasets(
        feature_dir=os.path.join(OUT_DIR, "mel_npz"),
        target_dir=os.path.join(OUT_DIR, "midi_npz"),
        batch_size=1,
        val_ratio=0.1,
        test_ratio=0.1,   # mets 0.0 si tu ne veux pas de test split
        )

    print(f"Train batches: {len(list(train_ds))}")
    print(f"Val batches:   {len(list(val_ds))}")

    if test_ds is not None:
        print(f"Test batches:  {len(list(test_ds))}")
    else:
        print("No test dataset (test_ratio=0)")



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


    #############################
    # Step 5: Build the model
    #############################
    print("\n=== Step 4: Building the model ===")
    # On récupère la shape d'entrée directement depuis un batch
    example_mel, _ = next(iter(train_ds))
    input_shape = example_mel.shape[1:]  # (T, 128), sans la dimension batch
    # build_model est supposée être définie dans model.py
    # ajuste la signature si besoin (par ex. build_model(input_shape, num_pitches=88))
    model = build_cnn_bilstm_onset_model (input_shape=input_shape)
    model.summary()


    ############################
    # Step 5: Train the model
    #############################
    print("\n=== Step 5: Training ===")
    # Deux options :
    #  1) tu as une fonction train_model dans train.py
    #  2) tu entraînes directement ici avec model.fit
    # OPTION 1 : train_model dans train.py
    history = train_model(
        model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=EPOCHS
    )

    #############################
    # Step 6: Postprocessing demo
    #############################
    print("\n=== Step 6: Postprocessing (MIDI generation demo) ===")
    # On prend un batch du set de test si dispo, sinon du val
    if test_ds is not None:
        source_ds = test_ds
        print("Using test_ds for demo prediction.")
    else:
        source_ds = val_ds
        print("No test_ds, using val_ds for demo prediction.")
    mel_batch, onset_batch_true = next(iter(source_ds))

    # Prédiction des probabilités d'onset
    onset_pred = model.predict(mel_batch)[0]  # (T, 88) pour le premier sample

    onset_binary = probs_to_onset_binary(onset_pred,threshold=0.50, min_distance=2)




    # Conversion en MIDI via postprocessing.py
    demo_midi_path = os.path.join(OUT_DIR, "demo_prediction.mid")
    probs_to_midi(
        onset_pred,
        threshold=0.5,
        min_distance=2,
        output_path=demo_midi_path,
    )
    print("Demo MIDI saved at:", demo_midi_path)
