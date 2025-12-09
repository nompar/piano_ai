
import os
import glob
import numpy as np
import tensorflow as tf
from google.cloud import storage
from piano_ai.params import *
import gcsfs

# === CHEMINS PAR DÉFAUT (à adapter si besoin) ================================
#    FEATURE_DIR = "/Users/hadriendecaumont/Downloads/all_years_npz_2/mel_npz"
#    TARGET_DIR  = "/Users/hadriendecaumont/Downloads/all_years_npz_2/midi_npz"
# ============================================================================

def _build_pairs(feature_dir, target_dir):
    """
    Find and pair up mel-spectrogram and MIDI target .npz files in local folders.
    Returns lists of file paths for features and their matching targets.
    """

    feature_paths = sorted(glob.glob(os.path.join(feature_dir, "*.npz")))
    target_paths  = sorted(glob.glob(os.path.join(target_dir, "*.npz")))

    feature_stems = [os.path.splitext(os.path.basename(p))[0] for p in feature_paths]
    target_stems  = [os.path.splitext(os.path.basename(p))[0] for p in target_paths]

    # ex: "XXX_targets" -> base "XXX"
    target_base_to_full = {
        stem.replace("_targets", ""): stem
        for stem in target_stems
    }

    paired_feature_paths = []
    paired_target_paths  = []

    for feat_path, feat_stem in zip(feature_paths, feature_stems):
        if feat_stem in target_base_to_full:
            tgt_stem = target_base_to_full[feat_stem]
            tgt_path = os.path.join(target_dir, tgt_stem + ".npz")
            paired_feature_paths.append(feat_path)
            paired_target_paths.append(tgt_path)
        else:
            print("⚠️ Pas de target pour", feat_stem)

    print("nb paires valides:", len(paired_feature_paths))
    return paired_feature_paths, paired_target_paths


def _load_arrays(paired_feature_paths, paired_target_paths):
    """
    Load all paired .npz files into memory from local disk.
    Returns:
        X: (N, T, 128) - mel-spectrograms
        Y: (N, T, 88)  - onset labels
    """
    mels = []
    onsets = []

    for feat_path, tgt_path in zip(paired_feature_paths, paired_target_paths):
        feat_npz = np.load(feat_path)

        tgt_npz  = np.load(tgt_path)

        mel   = feat_npz["mel"]      # (128, 3000, 1) typiquement
        onset = tgt_npz["onset"]     # (3000, 88)

        # (128, T, 1) -> (T, 128)
        mel = np.squeeze(mel, axis=-1).T

        mels.append(mel.astype("float32"))
        onsets.append(onset.astype("float32"))

    X = np.stack(mels, axis=0)
    Y = np.stack(onsets, axis=0)

    print("X:", X.shape)  # (N, T, 128)
    print("Y:", Y.shape)  # (N, T, 88)
    return X, Y


def _split_train_val_test(X, Y, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Shuffle and split X, Y into train / val / (optional) test sets.
    """
    N = len(X)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)

    test_size = int(N * test_ratio)
    val_size  = int(N * val_ratio)

    test_idx  = indices[:test_size]
    val_idx   = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]

    print("Train :", X_train.shape, Y_train.shape)
    print("Val   :", X_val.shape,   Y_val.shape)
    if test_ratio > 0:
        print("Test  :", X_test.shape,  Y_test.shape)

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def make_datasets(
    feature_dir=FEATURE_DIR,
    target_dir=TARGET_DIR,
    batch_size=4,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    """
    Fonction intelligente qui choisit la bonne méthode (Local vs GCP)
    en fonction du chemin fourni.
    """
    # DÉTECTION AUTOMATIQUE GCP vs LOCAL
    # Si le chemin commence par "gs://", on bascule sur le mode Cloud
    if READ_MODE == 'gcp':
        print(f":nuage: Détection de chemin GCP : {feature_dir}")
        return make_datasets_gcs(
            feature_dir,
            target_dir,
            batch_size,
            val_ratio,
            test_ratio,
            seed
        )
    else:
        # Sinon, on reste sur le comportement local classique
        print(f":ordinateur: Détection de chemin Local : {feature_dir}")
        # --- TON ANCIEN CODE LOCAL ---
        # Pairage

        paired_feature_paths, paired_target_paths = _build_pairs(feature_dir, target_dir)
        # Chargement en X, Y
        X, Y = _load_arrays(paired_feature_paths, paired_target_paths)
        # Split
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = _split_train_val_test(
            X, Y, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        # Création des datasets tf.data
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, Y_train))
            .shuffle(buffer_size=len(X_train), reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, Y_val))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        if len(X_test) > 0:
            test_ds = (
                tf.data.Dataset.from_tensor_slices((X_test, Y_test))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            test_ds = None

        train_ds.save("./train_ds")
        val_ds.save("./val_ds")
        test_ds.save("./test_ds")

        return train_ds, val_ds, test_ds


# -----------------------------------------------------------------------------
# SECTION 2: LOADING DATA FROM GOOGLE CLOUD STORAGE (GCS)
# -----------------------------------------------------------------------------
# These functions are used when your data (.npz files) are stored in a Google Cloud
# Storage bucket. This is useful for large datasets or when running code in the cloud.

def _build_pairs_gcs(feature_dir, target_dir, fs=None):
    """
    Find and pair up mel-spectrogram and MIDI target .npz files in GCS buckets.
    Returns lists of GCS file paths for features and their matching targets.
    """
    if fs is None:
        fs = gcsfs.GCSFileSystem()  # Create a GCS filesystem object if not provided
    feature_paths = sorted(fs.glob(f"{feature_dir}/*.npz"))  # List .npz files in GCS feature_dir
    target_paths  = sorted(fs.glob(f"{target_dir}/*.npz"))   # List .npz files in GCS target_dir
    feature_stems = [os.path.splitext(os.path.basename(p))[0] for p in feature_paths]
    target_stems  = [os.path.splitext(os.path.basename(p))[0] for p in target_paths]
    target_base_to_full = {stem.replace("_targets", ""): stem for stem in target_stems}
    paired_feature_paths, paired_target_paths = [], []
    for feat_path, feat_stem in zip(feature_paths, feature_stems):
        if feat_stem in target_base_to_full:
            tgt_stem = target_base_to_full[feat_stem]
            tgt_path = f"{target_dir}/{tgt_stem}.npz"
            paired_feature_paths.append(feat_path)
            paired_target_paths.append(tgt_path)
        else:
            print("⚠️ No target found for", feat_stem)
    print("Number of valid pairs:", len(paired_feature_paths))
    return paired_feature_paths, paired_target_paths

from tqdm import tqdm

def _load_arrays_gcs(paired_feature_paths, paired_target_paths, fs=None):
    if fs is None:
        fs = gcsfs.GCSFileSystem()
    mels, onsets = [], []

    print(f"⏳ Chargement GCS des {len(paired_feature_paths)} paires...")
    for feat_path, tgt_path in tqdm(zip(paired_feature_paths, paired_target_paths),
                                   total=len(paired_feature_paths),
                                   desc="Lecture NPZ GCS"):
        with fs.open(feat_path, 'rb') as f:
            feat_npz = np.load(f)
            mel = feat_npz["mel"]

        with fs.open(tgt_path, 'rb') as f:
            tgt_npz = np.load(f)
            onset = tgt_npz["onset"]

        mel = np.squeeze(mel, axis=-1).T
        mels.append(mel.astype("float32"))
        onsets.append(onset.astype("float32"))

    X = np.stack(mels, axis=0)
    Y = np.stack(onsets, axis=0)
    print("X:", X.shape)
    print("Y:", Y.shape)
    return X, Y


def make_datasets_gcs(
    feature_dir,
    target_dir,
    batch_size=BATCH_SIZE,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    """
    Complete pipeline for GCS data:
      - Pair mel and target files in GCS
      - Load them into memory from GCS
      - Split into train/val/test
      - Create TensorFlow datasets for training

    Returns:
        train_ds, val_ds, test_ds (or None if test_ratio=0)
    """
    fs = gcsfs.GCSFileSystem()  # Create a GCS filesystem object
    paired_feature_paths, paired_target_paths = _build_pairs_gcs(feature_dir, target_dir, fs)
    X, Y = _load_arrays_gcs(paired_feature_paths, paired_target_paths, fs)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = _split_train_val_test(
        X, Y, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        .shuffle(buffer_size=len(X_train), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    ) if len(X_test) > 0 else None
    with fs.open(os.path.join(DATASET_DIR,"train"),"wb") as f:
        f.write(train_ds)
    with fs.open(os.path.join(DATASET_DIR,"val"),"wb") as f:
        f.write(val_ds)
    with fs.open(os.path.join(DATASET_DIR,"test"),"wb") as f:
        f.write(test_ds)
    return train_ds, val_ds, test_ds




# -----------------------------------------------------------------------------
# SECTION 3: TESTING / DEMO
# -----------------------------------------------------------------------------

if __name__ == "__main__":
     # Example: Test local loading
    train_ds, val_ds, test_ds = make_datasets()

    for mel_batch, onset_batch in train_ds.take(1):
        print("train batch:", mel_batch.shape, onset_batch.shape)

    for mel_batch, onset_batch in val_ds.take(1):
        print("val batch  :", mel_batch.shape, onset_batch.shape)

    for mel_batch, onset_batch in test_ds.take(1):
        print("test batch  :", mel_batch.shape, onset_batch.shape)


    # To test GCS loading, call make_datasets_gcs(...) with your GCS paths
    # Example:
    # train_ds, val_ds, test_ds = make_datasets_gcs("piano_ai/all_years_npz/mel_npz", "piano_ai/all_years_npz/midi_npz")
