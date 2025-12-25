import os

source_txt = r"H:\tts\fin\filelists\finnish_pseudo_clean.txt"
subset_txt = r"H:\tts\fin\subset_wav_list.txt"

valid_lines = []
total = 0
found = 0

with open(source_txt, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        wav_path = line.strip().split("|")[0]

        if os.path.exists(wav_path):
            valid_lines.append(line)
            found += 1

        # 🔥 лог каждые 10k строк
        if total % 10000 == 0:
            print(f"Проверено: {total}, найдено wav: {found}")

with open(subset_txt, "w", encoding="utf-8") as f:
    f.writelines(valid_lines)

print(f"✅ ГОТОВО: {found} валидных строк из {total}")
print(f"📄 Сохранено в: {subset_txt}")
