import numpy as np  # Lets us work with arrays and numbers

def binarize_predictions(y_pred, thresholds=None):
    """
    Turns model predictions into simple 0/1 values using thresholds.
    """

    # If no thresholds are given, use these default values
    if thresholds is None:
        thresholds = {
            'onset':0.5,
            'offset':0.5,
            'active':0.5,
            'velocity':0,
            'pedal':0.5
            }

    # For each prediction type, check if it's above the threshold (makes it 1), else 0
    onset    = (y_pred[0][0] > thresholds['onset']).astype(np.float32)    # Notes starting
    offset   = (y_pred[1][0] > thresholds['offset']).astype(np.float32)   # Notes ending
    active   = (y_pred[2][0] > thresholds['active']).astype(np.float32)   # Notes being held
    velocity = y_pred[3][0]  # This one stays as it is (how hard notes are played)
    pedal    = (y_pred[4][0] > thresholds['pedal']).astype(np.float32)    # Pedal pressed or not

    # Give back the binarized results in a list
    return [onset, offset, active, velocity, pedal]
