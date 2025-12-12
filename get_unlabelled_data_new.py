# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
ОПТИМИЗИРОВАННАЯ ВЕРСИЯ ДЛЯ HDD ПОСЛЕ ДЕФРАГМЕНТАЦИИ
=================================================
Ключевые улучшения:
1. Продолжение с последнего места (пропуск существующих файлов)
2. Умные паузы для отдыха диска
3. Пакетная обработка
4. Детальная статистика
"""
import os
import argparse
import gzip
import csv
import time
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List
from tqdm import tqdm
from torch.hub import download_url_to_file
import soundfile as sf
import numpy as np
from voxpopuli import LANGUAGES, LANGUAGES_V2, DOWNLOAD_BASE_URL

def _segment(item: Tuple[str, List[Tuple[str, float, float]], str]):
    """Обработка одного аудиофайла с минимальной нагрузкой на HDD"""
    in_path, segments, out_root = item
    _in_path = Path(in_path)
    event_id = _in_path.stem
    lang, year = _in_path.parent.parent.stem, _in_path.parent.stem
    
    # Создаем папку заранее
    year_dir = Path(out_root) / lang / year
    year_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Читаем файл один раз
        waveform, sr = sf.read(in_path, dtype='float32')
    except Exception as e:
        print(f"\n⚠️ ОШИБКА чтения {in_path}: {str(e)}")
        return 0
    
    # Обрабатываем моно-аудио
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    
    segments_written = 0
    for i, start_sec, end_sec in segments:
        start = int(start_sec * sr)
        end = min(int(end_sec * sr), len(waveform))
        
        # Пропускаем некорректные сегменты
        if start >= end or start < 0 or end > len(waveform):
            continue
            
        # Формируем путь для записи
        out_path = year_dir / f'{event_id}_{i}.wav'
        
        # Пропускаем уже существующие файлы
        if out_path.exists():
            segments_written += 1
            continue
            
        try:
            # Пишем сегмент
            sf.write(str(out_path), waveform[start:end], sr)
            segments_written += 1
            
            # Минимальная задержка для HDD
            time.sleep(0.005)  # 5 миллисекунд
            
        except Exception as e:
            print(f"\n⚠️ ОШИБКА записи {out_path}: {str(e)}")
    
    # Задержка после обработки файла
    time.sleep(0.05)  # 50 миллисекунд
    return segments_written

def get_metadata(out_root, subset):
    """Загрузка метаданных с защитой от сбоев"""
    def predicate(id_):
        is_plenary = id_.find("PLENARY") > -1
        if subset in {"10k", "10k_sd"}:
            return is_plenary and 20190101 <= int(id_[:8]) < 20200801
        elif subset in {"100k"}:
            return is_plenary
        elif subset in LANGUAGES:
            return is_plenary and id_.endswith(subset)
        elif subset in LANGUAGES_V2:
            return id_.endswith(subset.split("_")[0])
        return True

    filename = "unlabelled_sd" if subset == "10k_sd" else "unlabelled_v2"
    url = f"{DOWNLOAD_BASE_URL}/annotations/{filename}.tsv.gz"
    tsv_path = out_root / Path(url).name
    
    if not tsv_path.exists():
        print(f"📥 Скачиваем метаданные с {url}...")
        download_url_to_file(url, tsv_path.as_posix())
    
    print("📖 Парсинг метаданных...")
    try:
        with gzip.open(tsv_path, mode="rt") as f:
            if subset == '10k_sd':
                reader = csv.DictReader(f, delimiter="|")
                rows = [
                    (r["session_id"], r["id_"], r["start_time"], r["end_time"])
                    for r in reader if predicate(r["session_id"])
                ]
            else:
                reader = csv.DictReader(f, delimiter="\t")
                rows = [
                    (r["event_id"], r["segment_no"], float(r["start"]), float(r["end"]))
                    for r in reader if predicate(r["event_id"])
                ]
    except Exception as e:
        print(f"❌ ОШИБКА парсинга метаданных: {str(e)}")
        print("Попробуйте удалить файл и перезапустить скрипт:")
        print(f"Удалите: {tsv_path}")
        exit(1)
    
    print(f"✅ Найдено {len(rows):,} сегментов для языка {subset}")
    return rows

def get(args):
    """Основная функция с оптимизацией под HDD после дефрагментации"""
    print("\n" + "="*60)
    print("НАСТРОЙКИ ЗАПУСКА:")
    print(f"  Корневая папка: {args.root}")
    print(f"  Язык: {args.subset}")
    print(f"  Режим: Продолжение с последнего места")
    print(f"  Диск: WD Red (после дефрагментации)")
    print("="*60 + "\n")
    
    audio_root = Path(args.root) / "raw_audios"
    out_root = Path(args.root) / "unlabelled_data_wav"
    out_root.mkdir(exist_ok=True, parents=True)
    
    # Проверка существования исходных файлов
    if not audio_root.exists():
        print(f"❌ ПАПКА С ИСХОДНЫМИ ФАЙЛАМИ НЕ НАЙДЕНА: {audio_root}")
        print("Проверьте путь к данным! Скрипт остановлен.")
        exit(1)
    
    # Загрузка метаданных
    manifest = get_metadata(out_root, args.subset)
    
    # Анализ существующих файлов
    print("\n🔍 Анализ существующих WAV-файлов...")
    items = defaultdict(list)
    existing_count = 0
    total_segments = len(manifest)
    
    for event_id, seg_no, start, end in tqdm(manifest, desc="Проверка сегментов"):
        lang, year = event_id.rsplit("_", 1)[1], event_id[:4]
        out_path = out_root / lang / year / f'{event_id}_{seg_no}.wav'
        
        # Пропускаем существующие файлы
        if out_path.exists():
            existing_count += 1
            continue
            
        # Добавляем в очередь на обработку
        path = audio_root / lang / year / f"{event_id}.ogg"
        if path.exists():
            items[path.as_posix()].append((seg_no, float(start), float(end)))
    
    print(f"\n✅ Пропускаем {existing_count:,} уже обработанных сегментов")
    print(f"🔄 Осталось обработать: {total_segments - existing_count:,} сегментов")
    print(f"📁 Будет обработано файлов: {len(items):,}")
    
    if not items:
        print("\n🎉 ВСЕ СЕГМЕНТЫ УЖЕ ОБРАБОТАНЫ!")
        print(f"📁 Результаты в: {out_root}")
        return
    
    # Подготовка к обработке
    items_list = [(k, v, out_root.as_posix()) for k, v in items.items()]
    total_files = len(items_list)
    batch_size = 100  # Обрабатываем по 100 файлов за раз
    
    print(f"\n⚙️ НАЧИНАЕМ ОБРАБОТКУ")
    print(f"   Всего файлов: {total_files:,}")
    print(f"   Размер пачки: {batch_size} файлов")
    print(f"   Пауза между пачками: 5 секунд")
    
    results = []
    start_time = time.time()
    
    # Обработка пачками
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch = items_list[batch_start:batch_end]
        
        print(f"\n📦 ПАЧКА {batch_start//batch_size + 1}/{(total_files+batch_size-1)//batch_size}")
        print(f"   Файлы: {batch_start+1}-{batch_end} из {total_files}")
        
        # Обработка файлов в пачке
        batch_results = []
        for idx, item in enumerate(tqdm(batch, desc="Обработка файлов", unit="файл")):
            batch_results.append(_segment(item))
        
        results.extend(batch_results)
        batch_segments = sum(batch_results)
        total_done = sum(results)
        
        # Статистика
        elapsed = time.time() - start_time
        remaining = total_files - batch_end
        est_time = (elapsed / (batch_end)) * remaining if batch_end > 0 else 0
        
        print(f"\n✅ Пачка завершена!")
        print(f"   Создано сегментов в пачке: {batch_segments:,}")
        print(f"   Всего создано: {total_done:,} из {total_segments - existing_count:,}")
        print(f"   Скорость: {batch_size / (time.time() - start_time + 1):.1f} файлов/мин")
        print(f"   Прошло времени: {int(elapsed//60)} мин")
        print(f"   Осталось примерно: {int(est_time//60)} мин")
        
        # Пауза для отдыха диска
        if remaining > 0:
            print(f"\n⏸️ ПАУЗА ДЛЯ ОТДЫХА ДИСКА (5 секунд)...")
            for i in range(5, 0, -1):
                print(f"   Продолжение через: {i} сек...", end='\r')
                time.sleep(1)
            print("\n▶️ Продолжаем работу")
    
    # Финальная статистика
    total_segments_created = sum(results)
    total_time = time.time() - start_time
    
    print(f"\n" + "="*60)
    print("🎉 ОБРАБОТКА УСПЕШНО ЗАВЕРШЕНА!")
    print(f"✅ Обработано файлов: {total_files:,}")
    print(f"✅ Создано сегментов: {total_segments_created:,}")
    print(f"⏱️ Общее время: {int(total_time//60)} мин {int(total_time%60)} сек")
    print(f"📁 Результаты сохранены в: {out_root}")
    print("="*60)

def get_args():
    parser = argparse.ArgumentParser(description="Подготовка немаркированных данных в формате WAV")
    parser.add_argument(
        "--root", "-r", type=str, required=True,
        help="Корневой путь к данным (например, H:\\tts\\fin)"
    )
    parser.add_argument(
        "--subset", "-s", type=str, required=True,
        choices=["400k", "100k", "10k", "10k_sd"] + LANGUAGES + LANGUAGES_V2,
        help="Язык или подмножество данных (например, fi для финского)"
    )
    return parser.parse_args()

def main():
    args = get_args()
    
    print("\n" + "="*60)
    print("🚀 СТАРТ ОБРАБОТКИ (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)")
    print(f"💻 Процессор: AMD Ryzen 9 5900X")
    print(f"💾 Диск: WD Red (после дефрагментации)")
    print(f"🎯 Язык: {args.subset}")
    print("="*60)
    
    get(args)

if __name__ == "__main__":
    main()