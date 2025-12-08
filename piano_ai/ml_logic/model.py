from tensorflow.keras import layers, models
from piano_ai.params import *

def build_cnn_bilstm_onset_model(
    n_mels=N_MELS,
    n_pitches=N_PITCHES,
):
    # T variable : None
    inputs = layers.Input(shape=(None, n_mels), name="mel_input")  # (T, 128)

    # Ajouter un canal pour Conv2D : (batch, T, 128, 1)
    x = layers.Reshape((-1, N_MELS, 1))(inputs)


    # --- CNN trunk ---
    # On pool uniquement en fréquence pour garder la résolution temporelle T
    filters_list = [32, 64, 96]
    for filters in filters_list:
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            activation="gelu",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(1, 2))(x)  # (time, freq/2)
        x = layers.Dropout(0.3)(x)

    # RESHAPE => Après 3 poolings freq : 128 -> 64 -> 32 -> 16
    freq_red = n_mels // (2 ** len(filters_list))  # 128 / 8 = 16
    channels = filters_list[-1]                    # 96
    feature_dim = freq_red * channels              # 16 * 96 = 1536

    # T est variable, donc Reshape(-1, feature_dim)
    x = layers.Reshape((-1, feature_dim))(x)       # (batch, T, feature_dim)

    # --- BiLSTM sur le temps ---
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True),
        merge_mode="concat"
    )(x)
    x = layers.Dropout(0.3)(x)

    # --- Sortie onsets ---
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(n_pitches, activation="sigmoid")(x)


    model = models.Model(inputs=inputs, outputs=outputs, name="Onset_CNN_BiLSTM_Flex")
    return model
