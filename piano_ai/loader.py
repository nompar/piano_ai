import os  # For file and directory operations
import glob  # For finding files by pattern
import numpy as np  # For loading and handling arrays
import tensorflow as tf  # For building and managing datasets

def get_dataset(mel_dir, labels_dir, batch_size=8):
    """
    Creates a TensorFlow dataset that loads pairs (X = mel, Y = labels)
    from two folders containing .npz files.
    """

    # 1) List all mel-spectrogram files in the directory
    mel_files = sorted(glob.glob(os.path.join(mel_dir, "*.npz")))
    pairs = []  # Will hold pairs of (mel file, label file)

    for mel_path in mel_files:
        # Get the base name of the mel file (without extension)
        name = os.path.splitext(os.path.basename(mel_path))[0]

        # Find the corresponding label file (must match naming convention)
        label_path = os.path.join(labels_dir, name + "_targets.npz")

        # Only add pairs if both files exist
        if os.path.exists(label_path):
            pairs.append((mel_path, label_path))
        else:
            print(f"⚠️ Missing labels for: {mel_path}")

    print(f"Number of pairs found: {len(pairs)}")

    # 2) Internal function to load a pair of files
    def load_pair(mel_path, labels_path):
        mel_npz = np.load(mel_path)  # Load mel chunk
        mel = mel_npz["mel"].astype(np.float32)  # Shape: (128, T, 1)

        labels_npz = np.load(labels_path)  # Load label chunk
        targets = {
            "onset":    labels_npz["onset"].astype(np.float32),
            "offset":   labels_npz["offset"].astype(np.float32),
            "active":   labels_npz["active"].astype(np.float32),
            "velocity": labels_npz["velocity"].astype(np.float32),
            "pedal":    labels_npz["pedal"].astype(np.float32),
        }
        return mel, targets

    # 3) Python generator that yields (mel, targets) pairs
    def gen():
        for mel_path, labels_path in pairs:
            mel, targets = load_pair(mel_path, labels_path)
            yield mel, targets

    # 4) Define the output signature (shapes and types) for TensorFlow
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

    # 5) Build the TensorFlow dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )

    # Batch the data (batch_size=1 recommended if sequence length T varies)
    dataset = dataset.batch(batch_size)
    # Prefetch batches for faster training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset  # Return the ready-to-use dataset
