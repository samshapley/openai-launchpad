import wandb
from ai import Audio

# Initialize a W&B run
wandb.init(project="audio-transcription")

# Create an instance of the Audio class
audio = Audio()

# Record audio from the microphone
audio_file_path = audio.record_from_microphone()

# Transcribe the recorded audio
transcription = audio.transcribe(audio_file_path)

# Create a W&B Table
audio_table = wandb.Table(columns=["Audio", "Transcription"])

# Add data to the table
audio_table.add_data(wandb.Audio(audio_file_path), transcription)

# Log the table to W&B
wandb.log({"Audio Transcriptions": audio_table})

# Finish the W&B run
wandb.finish()