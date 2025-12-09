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


def train_model(model, train_ds, val_ds, epochs=EPOCHS, save_dir="model"):
    """
    Trains the given model using the provided datasets.
    Does NOT build the model or load the datasets.
    """

    # Crée le dossier de sauvegarde
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)


    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=weighted_bce,
        metrics=["binary_accuracy"]
    )

    # Callbacks
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        # Sauvegarde le meilleur modèle pendant l'entraînement
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # Réduit le learning rate si plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )



    # ===== SAUVEGARDE FINALE =====
    model_path = os.path.join(save_dir, "model.keras")
    model.save(model_path)
    print(f"Model saved: {model_path}")



    # Sauvegarde aussi les poids seuls (backup)
    weights_path = os.path.join(save_dir, "model.weights.h5")
    model.save_weights(weights_path)
    print(f"Weights saved: {weights_path}")

    return history, model_path
