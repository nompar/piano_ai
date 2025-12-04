
import numpy as np

def binarize_predictions(y_pred, thresholds=None):
    """
    Applique des seuils modulables à y_pred.

    y_pred : liste de 5 arrays [onset, offset, active, velocity, pedal]
             shape = (1, T, 88) pour les 4 premières, (1, T, 1) pour pedal
    thresholds : dict avec seuils pour chaque sortie
                 ex : {'onset':0.5, 'offset':0.5, 'active':0.5, 'velocity':0, 'pedal':0.5}
                 velocity reste float
    Retour : y_pred binarisé, même format que l’input
    """
    if thresholds is None:
        thresholds = {'onset':0.5, 'offset':0.5, 'active':0.5, 'velocity':0, 'pedal':0.5}

    onset    = (y_pred[0][0] > thresholds['onset']).astype(np.float32)
    offset   = (y_pred[1][0] > thresholds['offset']).astype(np.float32)
    active   = (y_pred[2][0] > thresholds['active']).astype(np.float32)
    velocity = y_pred[3][0]  # reste linéaire
    pedal    = (y_pred[4][0] > thresholds['pedal']).astype(np.float32)

    return [onset, offset, active, velocity, pedal]
