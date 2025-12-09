
## 🎹 PIANO-AI — Convertisseur Audio → MIDI + Visualizer

**PIANO-AI** est un système complet permettant de convertir un **fichier audio**
(.wav, .mp3, .flac, .ogg, .m4a) en fichier **MIDI**, puis de visualiser le résultat dans une interface web **Streamlit**.



## Le pipeline combine :

1. Un **pré-traitement** audio (mel-spectrogrammes, chunck, frames..),
2. Un **modèle Deep Learning** (CNN + BiLSTM),
3. Du **post-processing** MIDI,
4. Une API **FastAPI** pour l’inférence,TBC
5. Un **visualiseur MIDI** (SynthViz)

## Les Fonctionnalités principales 🚀:

- 🔊 Conversion **Audio** → **MIDI**
- Extraction automatique du **mel-spectrogramme**
- Prédiction des onsets (début de notes) or TBC
- Nettoyage + génération du fichier MIDI ext(.mid)

## Le pipeline **Machine Learning** :

- Pré-traitement audio
- Construction dynamique de datasets (local + GCP=> TBC pour GCP)
- Modèle Deep Learning **CNN-BiLSTM**
- Weighted BCE Loss (adaptée aux onset rares)
- Entraînement avec callbacks (_EarlyStopping, LR decay, checkpoints_)

## 🌐 Interface **Streamlit**:

Interface web permettant :
- De charger un fichier **audio**
- De visualiser le **spectrogramme**
- De lancer la conversion via l’API _TBC_
- D’afficher le piano-roll produit (PrettyMIDI)
- De visualiser les instruments détectés dans le MIDI


## 🌩️ Support Google Cloud Storage (GCS) - _TO be confirmed_

- Chargement transparent des **.npz**
- Gestion automatique local ↔ GCP via READ_MODE
- Utilisation de gcsfs + google.cloud.storage

## 📂 Structure du projet
```
piano_ai/
│
├── ml_logic/
│   ├── inference.py            # Pipeline d’inférence Audio → MIDI
│   ├── model.py                # Modèle CNN-BiLSTM (onset prediction)
│   ├── loader.py               # Chargement datasets local / GCP
│   ├── postprocessing.py       # Nettoyage + conversion proba → MIDI
│   ├── preprocess_audio.py     # Extraction mel, chunking, mapping MIDI → targets
│   ├── train.py                # Entraînement du modèle
│   └── params.py               # Hyperparamètres globaux + chemins
│
├── app.py                      # Interface Streamlit (visualiseur MIDI)
├── binarizer.py                # Binarisation des sorties du modèle
├── main.py                     # Script principal de conversion (TBC)
├── api.py                      # API FastAPI : endpoint /convert
├── README.md                   # Documentation du projet
└── requirements.txt            # Dépendances Python
```
