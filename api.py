from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

# Create the FastAPI app instance
app = FastAPI()

@app.post("/convert")
# This decorator tells FastAPI: "When someone sends a POST request to /convert, run this function."
# 'async' lets the server handle lots of requests at once, so multiple people can use it at the same time.
# 'file: UploadFile = File(...)' means the user must upload a file for this endpoint to work.
async def convert_audio(file: UploadFile = File(...)):
    # Make a path to save the uploaded file in a temp folder, using its original name
    file_location = f"temp/{file.filename}"
    # Open a new file at that location, ready to write binary data
    with open(file_location, "wb") as f:
        # Read all the data from the uploaded file and write it into the new file
        f.write(await file.read())
    # TODO: Call your model here to process the audio file and save the output MIDI file to midi_path
    # For now, midi_path is just a placeholder for where your model should save the MIDI file
    midi_path = "temp/output.midi"
    # This line sends the processed MIDI file back to the user as a downloadable file
    # FileResponse streams the file directly to the client, so they can download it
    # media_type="audio/midi" tells the browser it's a MIDI file
    # filename="output.midi" is the name the user will see for the download
    return FileResponse(midi_path, media_type="audio/midi", filename="output.midi")

TODO: # Connect midi file as input to pianola (or other app)
TODO: # Find out how to embed pianola in a site (dash? streamlit?)
