# =========================
# Global Variables
# =========================

SR = 22050  # Sample rate: audio will be loaded at 22,050 samples per second
FPS = 100   # Frames per second for spectrogram time resolution
HOP_LENGTH = SR // FPS  # Number of samples between successive frames (controls time resolution)
N_FFT = 2048    # Number of samples per FFT window (controls frequency resolution)
N_MELS = 128    # Number of mel bands (vertical resolution of mel-spectrogram)

# Piano range variables
PITCH_MIN = 21    # A0
PITCH_MAX = 108   # C8
N_PITCHES = PITCH_MAX - PITCH_MIN + 1
