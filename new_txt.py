import os

source_txt = r"H:\tts\fin\filelists\finnish_pseudo_clean.txt"
subset_txt = r"H:\tts\fin\subset_wav_list.txt"

# 🔥 ТВОЯ НОВАЯ ПАПКА С 22050 WAV!
new_wav_dir = r"H:\tts\fin\subset_wav_22050"

valid_lines = []
total = 0
found = 0

print(f"🔍 Ищу в: {new_wav_dir}")

with open(source_txt, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        parts = line.strip().split("|")
        wav_filename = os.path.basename(parts[0])  # Только ИМЯ файла!
        
        # 🔥 Проверяем в НОВОЙ папке!
        new_wav_path = os.path.join(new_wav_dir, wav_filename)
        
        if os.path.exists(new_wav_path):
            valid_lines.append(line)
            found += 1

        if total % 1000 == 0:
            print(f"Проверено: {total}, найдено: {found}")

with open(subset_txt, "w", encoding="utf-8") as f:
    f.writelines(valid_lines)

print(f"✅ ГОТОВО: {found} файлов из {total}")
print(f"📄 Сохранено: {subset_txt}")
