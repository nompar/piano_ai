import os
import glob
import numpy as np
import tensorflow as tf

def get_dataset(mel_dir, labels_dir, batch_size=8):
    """
    Crée un tf.data.Dataset qui charge des paires (X = mel, Y = labels)
    depuis deux dossiers contenant des .npz.

    Args:
        mel_dir (str) : chemin vers le dossier contenant les spectrogrammes MEL (.npz)
        labels_dir (str) : chemin vers le dossier contenant les targets (.npz)
        batch_size (int) : taille du batch. 1 recommandé si les séquences ont des longueurs différentes.

    Returns:
        tf.data.Dataset : dataset prêt à l'emploi
    """

    # -------------------------------
    # 1) Lister tous les fichiers MEL
    # -------------------------------
    mel_files = sorted(glob.glob(os.path.join(mel_dir, "*.npz")))
    pairs = []

    for mel_path in mel_files:
        # Nom de base du fichier (sans extension)
        name = os.path.splitext(os.path.basename(mel_path))[0]

        # Chemin du fichier target correspondant
        # IMPORTANT : on utilise _targets.npz pour correspondre à ton preprocess
        label_path = os.path.join(labels_dir, name + "_targets.npz")

        if os.path.exists(label_path):
            pairs.append((mel_path, label_path))
        else:
            print(f"⚠️ Labels manquants pour : {mel_path}")

    print(f"Nombre de paires trouvées : {len(pairs)}")

    # ---------------------------------
    # 2) Fonction interne pour charger une paire
    # ---------------------------------
    def load_pair(mel_path, labels_path):
        mel_npz = np.load(mel_path)
        mel = mel_npz["mel"].astype(np.float32)  # forme : (128, T, 1)

        labels_npz = np.load(labels_path)
        targets = {
            "onset":    labels_npz["onset"].astype(np.float32),
            "offset":   labels_npz["offset"].astype(np.float32),
            "active":   labels_npz["active"].astype(np.float32),
            "velocity": labels_npz["velocity"].astype(np.float32),
            "pedal":    labels_npz["pedal"].astype(np.float32),
        }
        return mel, targets

    # -------------------------------
    # 3) Générateur Python
    # -------------------------------
    def gen():
        for mel_path, labels_path in pairs:
            mel, targets = load_pair(mel_path, labels_path)
            yield mel, targets

    # -------------------------------
    # 4) Définir la signature de sortie
    # -------------------------------
    output_signature = (
        tf.TensorSpec(shape=(128, None, 1), dtype=tf.float32),
        {
            "onset":    tf.TensorSpec(shape=(None, 88), dtype=tf.float32),
            "offset":   tf.TensorSpec(shape=(None, 88), dtype=tf.float32),
            "active":   tf.TensorSpec(shape=(None, 88), dtype=tf.float32),
            "velocity": tf.TensorSpec(shape=(None, 88), dtype=tf.float32),
            "pedal":    tf.TensorSpec(shape=(None, 1),  dtype=tf.float32),
        },
    )

    # -------------------------------
    # 5) Construction du dataset
    # -------------------------------
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )

    # Batching (batch_size=1 recommandé si T variable)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
