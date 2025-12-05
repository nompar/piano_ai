from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import io

app = FastAPI()

@app.post("/convert")
async def convert_audio(file: UploadFile = File(...)):
    """The API receives the uploaded audio file as bytes.
    The function wraps those bytes in an io.BytesIO object.
    Passes this io.BytesIO object.
    The model model processes the audio and returns MIDI bytes.
    The model returns the MIDI bytes back to the user."""

    # Read the uploaded audio file into memory (as bytes)
    audio_bytes = await file.read()

    # Wrap the bytes in a BytesIO object (acts like a file in memory)
    audio_file_like = io.BytesIO(audio_bytes)

    # TODO: Pass audio_file_like (or audio_bytes) to your model and get MIDI bytes back
    # Example: midi_bytes = your_model(audio_file_like)
    midi_bytes = b""  # Replace with actual MIDI bytes from your model

    # Return the MIDI bytes directly as a downloadable file
    return Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={
            "Content-Disposition": "attachment; filename=output.midi"
            }
        ) # type: ignore


# How it works:
