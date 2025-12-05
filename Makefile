
run_preprocess:
	./.venv/bin/python -c 'from piano_ai.preprocess_audio import preprocess; preprocess()'

# ============================
# Entraînement du modèle
# ============================
# ⚠️ Suppose un fichier piano_ai/ml_logic/train.py avec une fonction train()
run_train:
	./.venv/bin/python -c 'from piano_ai.ml_logic.train import train; train()'

# ============================
# Scripts principaux
# ============================

# Lancer le pipeline global (main.py)
run_main:
	./.venv/bin/python main.py

# Lancer l'API (api.py)
run_api:
	./.venv/bin/python api.py

# Lancer l'app (piano_ai/app.py) – par ex. Streamlit / FastAPI wrapper
run_app:
	./.venv/bin/python piano_ai/app.py
