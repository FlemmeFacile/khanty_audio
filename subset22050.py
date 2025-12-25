import torchaudio
import os

source_dir = r"H:\tts\fin\subset_wav"
target_dir = r"H:\tts\fin\subset_wav_22050"
os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if not filename.endswith(".wav"):
        continue
    path = os.path.join(source_dir, filename)
    waveform, sr = torchaudio.load(path)
    if sr != 22050:
        waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
    torchaudio.save(os.path.join(target_dir, filename), waveform, 22050)

print("Конвертация завершена")
