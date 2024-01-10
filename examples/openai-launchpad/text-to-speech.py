from ai import Audio

audio = Audio()

text = "The quick brown fox jumps over the lazy dog."

speech = audio.speak(text=text, voice='shimmer', speed=1, save=True)

print(speech)

print(speech["start_time"])
print(speech["end_time"])
print(speech["duration_ms"])
print(speech["words_per_minute"])