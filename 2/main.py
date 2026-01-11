import librosa
from transformers import pipeline

audio, sr = librosa.load("2/audio.wav", sr=16000)

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    framework="pt"
)

result = asr({"array": audio, "sampling_rate": sr})
print(result["text"])