from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import io

app = FastAPI()

@app.post("/convert")
async def convert_audio(file: UploadFile = File(...)):

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

# The API receives the uploaded audio file as bytes.
# You wrap those bytes in an io.BytesIO object (which acts like a file in memory).
# You pass this io.BytesIO object (or the bytes directly, depending on your model) to your model.
# Your model processes the audio and returns MIDI bytes.
# You send the MIDI bytes back to the user.
