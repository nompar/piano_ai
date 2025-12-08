import tensorflow as tf
from tensorflow.keras import backend as K
from piano_ai.params import *



# =============================================================================
# STEP 1: DEFINE CUSTOM LOSS FUNCTION
# =============================================================================
# This function defines a weighted binary crossentropy loss to help the model learn rare events

def weighted_bce(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    weights = 1.0 + (POS_WEIGHT - 1.0) * y_true
    return K.mean(bce * weights)


# =============================================================================
# STEP 2: DEFINE TRAINING FUNCTION
# =============================================================================
# This function only handles compiling and training the model.
# You pass in the model and datasets; it does not build the model or load data.


def train_model(model, train_ds, val_ds, epochs=EPOCHS):
    """
    Trains the given model using the provided datasets.
    Does NOT build the model or load the datasets.
    """
    # Compile the model with optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=weighted_bce,
        metrics=["binary_accuracy"]
    )
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        min_delta=1e-4,
    )
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop],
    )
    return history
