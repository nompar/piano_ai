import tensorflow as tf
from keras import layers, models, backend as K

N_MELS = 128
N_PITCHES = 88
POS_WEIGHT = 45.0

def build_cnn_bilstm_onset_model(n_mels=N_MELS, n_pitches=N_PITCHES):
    inputs = layers.Input(shape=(None, n_mels), name="mel_input")
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(inputs)

    filters_list = [32, 64, 96]
    for filters in filters_list:
        x = layers.Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="gelu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(1,2))(x)
        x = layers.Dropout(0.3)(x)

    freq_red = n_mels // (2 ** len(filters_list))
    channels = filters_list[-1]
    feature_dim = freq_red * channels
    x = layers.Reshape((-1, feature_dim))(x)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True), merge_mode="concat")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(n_pitches, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Onset_CNN_BiLSTM_Flex")
    return model

def weighted_bce(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    weights = 1.0 + (POS_WEIGHT - 1.0) * y_true
    return K.mean(bce * weights)
