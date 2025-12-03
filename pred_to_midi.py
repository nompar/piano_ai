import pretty_midi
import numpy as np

def transformer(y_binarized, fs=100, output_path='output.mid'):
    """
    Convertit un y_pred déjà binarisé en fichier MIDI.

    y_binarized = [onset, offset, active, velocity, pedal]
    - onset, offset, active, pedal : 0 ou 1
    - velocity : flottant (0-127)
    """
    onset, offset, active, velocity, pedal = y_binarized
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    num_notes = onset.shape[1]
    time_step = 1 / fs

    # Parcours des notes
    for n in range(num_notes):
        note_on = False
        start_time = 0
        vel_list = []

        for t in range(onset.shape[0]):
            if active[t, n] == 1:
                vel_list.append(int(np.clip(velocity[t, n], 0, 127)))
                if not note_on:
                    note_on = True
                    start_time = t * time_step
            else:
                if note_on:
                    end_time = t * time_step
                    avg_vel = int(np.mean(vel_list))
                    note = pretty_midi.Note(avg_vel, n, start_time, end_time)
                    piano.notes.append(note)
                    note_on = False
                    vel_list = []

        # Si la note est encore active à la fin
        if note_on:
            end_time = onset.shape[0] * time_step
            avg_vel = int(np.mean(vel_list))
            note = pretty_midi.Note(avg_vel, n, start_time, end_time)
            piano.notes.append(note)

    # Gestion de la pédale
    pedal_on = False
    for t in range(len(pedal)):
        if pedal[t] == 1 and not pedal_on:
            cc = pretty_midi.ControlChange(number=64, value=127, time=t*time_step)
            piano.control_changes.append(cc)
            pedal_on = True
        elif pedal[t] == 0 and pedal_on:
            cc = pretty_midi.ControlChange(number=64, value=0, time=t*time_step)
            piano.control_changes.append(cc)
            pedal_on = False

    midi.instruments.append(piano)
    midi.write(output_path)
    print(f"MIDI sauvegardé sous {output_path}")
    return midi
