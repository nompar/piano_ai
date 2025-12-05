
import os
import glob
import numpy as np
import tensorflow as tf

# === CHEMINS PAR DÉFAUT (à adapter si besoin) ================================
FEATURE_DIR = "/Users/hadriendecaumont/Downloads/all_years_npz_2/mel_npz"
TARGET_DIR  = "/Users/hadriendecaumont/Downloads/all_years_npz_2/midi_npz"

# ============================================================================

def _build_pairs(feature_dir, target_dir):
    """Associe chaque mel à son npz de targets, en gérant le suffixe '_targets'."""
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
    Charge tous les fichiers en mémoire.
    Renvoie:
        X : (N, T, 128)
        Y : (N, T, 88)  (onset uniquement)
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
    Mélange et split X, Y en train / val (/ test optionnel).
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
    Pipeline complet :
      - pairage mel / targets
      - chargement en mémoire
      - split train / val (/ test)
      - création de tf.data.Dataset

    Retourne:
        train_ds, val_ds, test_ds (ou None si test_ratio=0)
    """
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

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Petit test rapide quand tu exécutes `python loader.py`
    train_ds, val_ds, test_ds = make_datasets()

    for mel_batch, onset_batch in train_ds.take(1):
        print("train batch:", mel_batch.shape, onset_batch.shape)

    for mel_batch, onset_batch in val_ds.take(1):
        print("val batch  :", mel_batch.shape, onset_batch.shape)

    for mel_batch, onset_batch in test_ds.take(1):
        print("test batch  :", mel_batch.shape, onset_batch.shape)
