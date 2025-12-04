import streamlit as st
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

st.title("🎹 Visualiseur MIDI type Pianola (API)")

API_URL = "http://127.0.0.1:8000/convert"  

st.write("Récupération du fichier MIDI depuis l'API...")

response = requests.get(API_URL)

if response.status_code != 200:
    st.error("Impossible de récupérer le fichier MIDI depuis l'API.")
    st.stop()

st.success("Fichier MIDI reçu !")

# Charger le MIDI depuis la réponse API
midi_bytes = io.BytesIO(response.content)

try:
    midi_data = pretty_midi.PrettyMIDI(midi_bytes)
except Exception as e:
    st.error("Erreur lors de la lecture du fichier MIDI.")
    st.stop()

st.subheader("🎼 Piano-roll")

fs = 100
piano_roll = midi_data.get_piano_roll(fs=fs)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(piano_roll, aspect='auto', origin='lower', cmap='magma')
ax.set_xlabel("Temps (frames)")
ax.set_ylabel("Notes (pitch)")
ax.set_title("Visualisation Piano-Roll (depuis API)")

st.pyplot(fig)

st.subheader("Informations MIDI")
st.write(f"Durée : {midi_data.get_end_time():.2f} sec")
st.write("Instruments :", [inst.name for inst in midi_data.instruments])
