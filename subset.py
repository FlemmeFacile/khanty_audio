import os
import random
import shutil

source_dir = r"H:\tts\fin\unlabelled_data_wav\fi"
subset_dir = r"H:\tts\fin\subset_wav"

os.makedirs(subset_dir, exist_ok=True)

# Рекурсивно собираем все wav файлы из всех подпапок
all_files = []
for root, dirs, files in os.walk(source_dir):
    for f in files:
        if f.endswith(".wav"):
            all_files.append(os.path.join(root, f))

# Случайно выбираем 1000 файлов
subset_files = random.sample(all_files, 1000)

# Копируем в новую папку
for f in subset_files:
    shutil.copy(f, os.path.join(subset_dir, os.path.basename(f)))

print(f"Создан поднабор из {len(subset_files)} файлов")
