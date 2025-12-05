"""
Post-processing pour passer des probabilités d'onsets
à un piano roll binaire puis à un fichier MIDI.
"""

import numpy as np
import pretty_midi

# Constantes par défaut (à adapter si besoin)
PITCH_MIN = 21  # A0
N_PITCHES = 88  # 21–108 -> 88 touches du piano
FPS = 100       # frames par seconde (doit matcher le preprocess)


def probs_to_onset_binary(onset_pred, threshold=0.50, min_distance=2):
    """
    Convertit des probabilités d'onsets en matrice binaire (T, 88).

    Paramètres
    ----------
    onset_pred : np.ndarray (T, 88)
        Probabilités d'onset ∈ [0,1] pour chaque frame et pitch.
    threshold : float
        Seuil de probabilité au-dessus duquel on considère un onset.
    min_distance : int
        Nombre de frames minimum entre deux onsets pour un même pitch.

    Retour
    ------
    onset_binary : np.ndarray (T, 88)
        Matrice binaire avec 1.0 aux frames d'onset.
    """
    T, P = onset_pred.shape
    onset_binary = np.zeros_like(onset_pred, dtype=np.float32)

    for p in range(P):
        probs = onset_pred[:, p]
        t = 1
        last_t = -min_distance

        while t < T - 1:
            # Pic local + au-dessus du seuil + dist mini depuis le dernier onset
            if (
                probs[t] >= threshold and
                probs[t] >= probs[t - 1] and
                probs[t] >= probs[t + 1] and
                t - last_t >= min_distance
            ):
                onset_binary[t, p] = 1.0
                last_t = t
                # on saute quelques frames pour éviter les clusters très denses
                t += min_distance
            else:
                t += 1

    return onset_binary


def onset_binary_to_midi(onset_binary,
                         fps=FPS,
                         output_path="output.mid",
                         pitch_min=PITCH_MIN,
                         min_duration=0.05,
                         velocity=80):
    """
    Convertit une matrice binaire d'onsets (T, 88) en fichier MIDI.

    Paramètres
    ----------
    onset_binary : np.ndarray (T, 88)
        Piano roll binaire (0/1) indiquant les onsets.
    fps : int
        Frames par seconde (doit matcher ton preprocess, ex: 100).
    output_path : str
        Chemin du fichier MIDI de sortie.
    pitch_min : int
        Pitch MIDI correspondant à la première colonne (0) -> 21 = A0.
    min_duration : float
        Durée minimale d'une note en secondes.
    velocity : int
        Vélocité MIDI des notes (0–127).
    """
    T, n_pitches = onset_binary.shape
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    for p in range(n_pitches):
        t = 0
        while t < T:
            if onset_binary[t, p] >= 0.5:
                start_frame = t
                # On maintient la note tant que la valeur reste >= 0.5
                while t < T and onset_binary[t, p] >= 0.5:
                    t += 1
                end_frame = t

                start_time = start_frame / fps
                end_time = end_frame / fps
                if end_time - start_time < min_duration:
                    end_time = start_time + min_duration

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch_min + p,
                    start=start_time,
                    end=end_time,
                )
                piano.notes.append(note)
            else:
                t += 1

    pm.instruments.append(piano)
    pm.write(output_path)
    print("MIDI sauvegardé :", output_path)


def probs_to_midi(onset_pred,
                  threshold=0.50,
                  min_distance=2,
                  fps=FPS,
                  output_path="output.mid",
                  pitch_min=PITCH_MIN,
                  min_duration=0.05,
                  velocity=80):
    """
    Helper qui enchaîne :
      proba d'onsets -> binaire -> fichier MIDI.

    onset_pred : (T, 88) proba ∈ [0,1]
    """
    onset_binary = probs_to_onset_binary(
        onset_pred,
        threshold=threshold,
        min_distance=min_distance,
    )

    onset_binary_to_midi(
        onset_binary,
        fps=fps,
        output_path=output_path,
        pitch_min=pitch_min,
        min_duration=min_duration,
        velocity=velocity,
    )

    return onset_binary
