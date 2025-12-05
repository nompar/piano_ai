import tensorflow as tf
from tensorflow.keras import backend as K
from piano_ai.ml_logic.model import build_cnn_bilstm_onset_model
from params import *

# Build the neural network model using the architecture defined in model.py
model = build_cnn_bilstm_onset_model()

def weighted_bce(y_true, y_pred):
    # Calculate binary crossentropy for each element (measures prediction error)
    bce = K.binary_crossentropy(y_true, y_pred)
    # Assign higher weight to positive samples to help the model learn rare events
    weights = 1.0 + (POS_WEIGHT - 1.0) * y_true
    # Apply the weights and average the loss
    return K.mean(bce * weights)

# Compile the model: set optimizer, loss function, and metric for training
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # Adam optimizer with learning rate 1e-4
    loss=weighted_bce,                         # Use custom weighted binary crossentropy
    metrics=["binary_accuracy"]                 # Track binary accuracy during training
)

# Set up early stopping: stop training if validation loss doesn't improve for 15 epochs
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",         # Watch validation loss
    patience=15,                # Wait 15 epochs before stopping
    restore_best_weights=True,  # Restore the best model weights
    min_delta=1e-4,             # Minimum improvement to count as progress
)

# Train the model using the training and validation datasets
history = model.fit(
    train_ds,                   # Training dataset
    validation_data=val_ds,     # Validation dataset
    epochs=EPOCHS,              # Number of epochs
    callbacks=[early_stop],     # Use early stopping
)

# Save the trained model to a file for later use (inference or deployment)
model.save("piano_onset_model.h5")
