import os
import random
import shutil

# Папка с исходным финским аудио
source_dir = r"H:\tts\fin\unlabelled_data_wav\fi"
# Папка для поднабора
subset_dir = r"H:\tts\fin\subset_wav"

os.makedirs(subset_dir, exist_ok=True)

# Получаем список всех wav
all_files = [f for f in os.listdir(source_dir) if f.endswith(".wav")]

# Случайно выбираем ~1000 файлов (можно увеличить, чтобы получить ~50-100 часов)
subset_files = random.sample(all_files, 1000)

# Копируем в новую папку
for f in subset_files:
    shutil.copy(os.path.join(source_dir, f), os.path.join(subset_dir, f))

print(f"Создан поднабор из {len(subset_files)} файлов")
