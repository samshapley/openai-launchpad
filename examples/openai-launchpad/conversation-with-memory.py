# Import capabilities
from ai import Chat, Audio

model = "gpt-3.5-turbo-1106"

audio = Audio()
chat = Chat(model=model, system="Helpful assistant.")

while True:
    transcript = audio.record_and_transcribe()

    if "quit" in transcript:
        break

    print(transcript)
    chat.chat_completion(transcript, speak=True, stream=True, memories=True)