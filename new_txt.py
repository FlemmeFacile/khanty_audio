import os

source_txt = r"H:\tts\fin\outputs\full_kmeans\finnish_pseudo_for_vits.txt"
subset_txt = r"H:\tts\fin\subset_wav_list.txt"

valid_lines = []

with open(source_txt, "r", encoding="utf-8") as f:
    for line in f:
        wav_path = line.strip().split("|")[0]  # путь до wav
        if os.path.exists(wav_path):
            valid_lines.append(line)

# сохраняем новый .txt с реально существующими файлами
with open(subset_txt, "w", encoding="utf-8") as f:
    f.writelines(valid_lines)

print(f"✅ Поднабор создан: {len(valid_lines)} строк")
