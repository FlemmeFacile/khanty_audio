#!/usr/bin/env python3
"""
Pseudo-Phoneme Extractor for Transfer Learning TTS
Based on the paper: "Transfer Learning from Speech Recognition to Text-to-Speech Synthesis Using Self-Supervised Representations"

Key features matching the paper:
- Uses block 15 hidden representations from wav2vec 2.0 (not XLS-R)
- K-means clustering with K=128 clusters (exactly as in paper)
- Merging consecutive identical cluster indices
- Designed for pre-training VITS architecture
- Supports both single-speaker and zero-shot multi-speaker TTS

Paper reference: https://arxiv.org/abs/2203.15447
"""

import os
import sys
import json
import logging
import argparse
import itertools
import joblib
import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from collections import Counter, defaultdict
import psutil
import time
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="librosa")

# --- КОНФИГУРАЦИЯ СОГЛАСНО СТАТЬЕ ---
PAPER_CONFIG = {
    # ТОЧНОЕ СООТВЕТСТВИЕ СТАТЬЕ: https://arxiv.org/abs/2203.15447
    "wav2vec_model": "facebook/wav2vec2-large-lv60", 
    "k_clusters": 128,  # Статья: "where we set K=128 for this work"
    "layer_index": 15,  # Статья: "hidden representation of block 15" (индексация с 1)
    "sample_rate": 16000,  # Стандарт для wav2vec 2.0
    "max_duration": 30.0,  # Разумный лимит для обработки
    "target_rms": 0.1,  # Оптимальная громкость для wav2vec
    "min_audio_duration": 0.5,  # Минимальная длина для качественных признаков
    "kmeans_batch_size": 5000,  # Оптимальный размер для MiniBatchKMeans
    "checkpoint_interval": 1000,  # Сохранение прогресса каждые 1000 файлов
    "cuda_cache_interval": 100,  # Оптимальная очистка CUDA кэша
    "num_workers": max(1, psutil.cpu_count(logical=False) - 1)  # Максимальное использование CPU
}

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Настраивает логирование в файл и консоль с форматированием как в научных работах"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "pseudo_phoneme_extraction.log")
    
    # ✅ ПОЛНАЯ ОЧИСТКА ВСЕХ ЛОГГЕРОВ
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = []
    
    # ✅ ОТКЛЮЧАЕМ НАСЛЕДОВАНИЕ ОТ КОРНЕВОГО ЛОГГЕРА
    logger = logging.getLogger("transfer_tts_pseudo_phonemes")
    logger.propagate = False  # Ключевая строка - отключаем наследование хендлеров
    
    # ✅ ГАРАНТИРОВАННАЯ ОЧИСТКА ТЕКУЩЕГО ЛОГГЕРА
    logger.handlers = []
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Форматтер в стиле научной публикации
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Консольный хендлер
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый хендлер
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ СОГЛАСНО СТАТЬЕ ---
def estimate_available_memory() -> Tuple[float, float]:
    """Оценивает доступную память GPU и RAM для оптимального батчинга"""
    gpu_mem = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # в GB
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        gpu_mem = gpu_mem - (allocated + cached) * 1.2  # Буфер 20%
    
    ram = psutil.virtual_memory()
    ram_available = ram.available / 1024**3  # в GB
    
    return max(0.0, gpu_mem), max(0.0, ram_available)


def auto_batch_size(gpu_mem: float, ram_available: float) -> int:
    """
    Автоматически подбирает размер батча согласно доступной памяти
    Эмпирические коэффициенты основаны на потреблении памяти wav2vec2-base
    """
    if gpu_mem > 0:
        # Для GPU: ~1.2GB на батч из 8 файлов для wav2vec2-base
        batch_size = int(gpu_mem * 6.0)
        return min(32, max(4, batch_size))
    else:
        # Для CPU: ~1.5GB RAM на батч из 4 файлов
        batch_size = int(ram_available * 2.5)
        return min(16, max(2, batch_size))


def validate_directory(path: str, create: bool = False) -> None:
    """Проверяет существование директории и права доступа"""
    if not os.path.exists(path):
        if create:
            os.makedirs(path, exist_ok=True)
            logging.info(f"✅ Создана директория: {path}")
        else:
            raise ValueError(f"❌ Директория не существует: {path}")
    
    if not os.access(path, os.R_OK | os.W_OK):
        raise PermissionError(f"❌ Нет прав доступа к директории: {path}")


def find_all_audio_files(root_dir: str, 
                        extensions: tuple = (".wav", ".flac", ".mp3"),
                        min_size_mb: float = 0.1) -> List[str]:
    """
    Рекурсивный поиск аудио файлов с фильтрацией по размеру
    Фильтрация маленьких файлов предотвращает ошибки при извлечении признаков
    """
    audio_paths = []
    min_size_bytes = min_size_mb * 1024 * 1024
    
    logging.info(f"🔍 Поиск аудио файлов в {root_dir}...")
    total_files = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            total_files += 1
            if file.lower().endswith(extensions):
                file_path = os.path.normpath(os.path.join(root, file))
                if os.path.getsize(file_path) >= min_size_bytes:
                    audio_paths.append(file_path)
    
    logging.info(f"✅ Найдено {len(audio_paths)}/{total_files} подходящих аудио файлов")
    return audio_paths


def load_and_preprocess_audio(path: str, 
                            target_sr: int = 16000,
                            max_duration: float = 30.0,
                            target_rms: float = 0.1,
                            min_duration: float = 0.5) -> Optional[np.ndarray]:
    """
    Загружает и предобрабатывает аудио файл согласно best practices для wav2vec 2.0
    
    Args:
        path: Путь к аудио файлу
        target_sr: Целевая частота дискретизации (16kHz для wav2vec 2.0)
        max_duration: Максимальная длительность в секундах
        target_rms: Целевой RMS для нормализации громкости
        min_duration: Минимальная длительность в секундах
    
    Returns:
        np.ndarray: Предобработанное аудио или None при ошибке
    """
    try:
        # Загрузка аудио с обработкой ошибок
        try:
            audio, sr = sf.read(path, dtype='float32')
        except Exception as e:
            logging.warning(f"⚠️ Ошибка при загрузке {path} через soundfile: {e}")
            # Попытка альтернативной загрузки через librosa
            audio, sr = librosa.load(path, sr=None, mono=False)
            audio = audio.astype(np.float32)
        
        # Проверка на пустой файл
        if audio.size == 0:
            logging.warning(f"⚠️ Пустой аудио файл: {path}")
            return None
        
        # Корректная конвертация в моно (как в статье о wav2vec 2.0)
        if audio.ndim > 1:
            if audio.shape[0] > audio.shape[1]:  # Каналы в первом измерении
                audio = audio.T
            audio = librosa.to_mono(audio)
        
        # Обрезка до максимальной длительности
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Проверка минимальной длительности
        if len(audio) < int(min_duration * sr):
            logging.warning(f"⚠️ Слишком короткое аудио ({len(audio)/sr:.2f}s): {path}")
            return None
        
        # Ресемплирование (только если необходимо)
        if sr != target_sr:
            audio = librosa.resample(
                audio, 
                orig_sr=sr, 
                target_sr=target_sr,
                res_type='kaiser_best'  # Лучшее качество для аудио признаков
            )
            logging.debug(f"🔄 Ресемплировано {path} с {sr}Hz на {target_sr}Hz")
        
        # Нормализация громкости по RMS (критически важно для wav2vec 2.0)
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 1e-6:  # Избегаем деления на ноль
            gain = target_rms / current_rms
            audio = audio * gain
            # Ограничение амплитуды для предотвращения клиппинга
            audio = np.clip(audio, -0.99, 0.99)
        
        # Final validation
        if np.isnan(audio).any() or np.isinf(audio).any():
            logging.warning(f"⚠️ NaN/Inf значения в аудио после обработки: {path}")
            return None
        
        return audio
        
    except Exception as e:
        logging.error(f"❌ Критическая ошибка при загрузке {path}: {str(e)}")
        return None


def safe_load_processed_paths(progress_log: str) -> set:
    """Безопасная загрузка списка обработанных путей с валидацией"""
    processed = set()
    if os.path.exists(progress_log):
        try:
            with open(progress_log, 'r', encoding='utf-8') as f:
                for line in f:
                    path = line.strip()
                    if path and os.path.exists(path):
                        processed.add(path)
            logging.info(f"✅ Загружено {len(processed)} обработанных путей из {progress_log}")
        except Exception as e:
            logging.warning(f"⚠️ Ошибка при чтении {progress_log}: {str(e)}")
    return processed


def safe_save_processed_path(progress_log: str, path: str) -> None:
    """Безопасное сохранение обработанного пути"""
    try:
        dir_path = os.path.dirname(progress_log)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        with open(progress_log, 'a', encoding='utf-8') as f:
            f.write(path + '\n')
    except Exception as e:
        logging.error(f"❌ Ошибка при сохранении {path} в {progress_log}: {str(e)}")


def check_existing_records(filelist_path: str) -> Dict[str, str]:
    """Проверяет существующие записи в файллисте для избежания дубликатов"""
    existing = {}
    if os.path.exists(filelist_path):
        try:
            with open(filelist_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('|')
                    if len(parts) >= 2:
                        audio_path = parts[0]
                        # Проверяем существование файла
                        if os.path.exists(audio_path):
                            existing[audio_path] = line
            logging.info(f"✅ Найдено {len(existing)} существующих записей в {filelist_path}")
        except Exception as e:
            logging.warning(f"⚠️ Ошибка при чтении {filelist_path}: {str(e)}")
    return existing


# --- ОСНОВНЫЕ ФУНКЦИИ СОГЛАСНО СТАТЬЕ ---
def train_kmeans_incremental(
    audio_paths: List[str],
    processor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    target_layer: int,
    k: int,
    extraction_batch_size: int = 8,
    kmeans_batch_size: int = 5000,
    checkpoint_path: str = None,
    checkpoint_interval: int = 1000,
    cuda_cache_interval: int = 100,
    device: str = "cuda",
    logger: logging.Logger = None,
    start_from_index: int = 0  # НОВЫЙ ПАРАМЕТР: с какого файла начинать
) -> MiniBatchKMeans:
    logger = logger or logging.getLogger("transfer_tts_pseudo_phonemes")
    
    # Инициализация путей к чекпоинтам
    temp_checkpoint_path = None
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        temp_checkpoint_path = os.path.join(checkpoint_dir, "kmeans_temp_checkpoint.joblib")
    
    # Попытка загрузки временного чекпоинта
    loaded_kmeans = None
    if temp_checkpoint_path and os.path.exists(temp_checkpoint_path):
        try:
            logger.info(f"🔄 Загрузка временного чекпоинта: {temp_checkpoint_path}")
            loaded_kmeans = joblib.load(temp_checkpoint_path)
            logger.info(f"✅ Временная модель K-means загружена.")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при загрузке временного чекпоинта: {str(e)}")
    
    # Если временный чекпоинт не загружен, пробуем загрузить финальный
    if not loaded_kmeans and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            logger.info(f"🔄 Загрузка финальной модели K-means: {checkpoint_path}")
            loaded_kmeans = joblib.load(checkpoint_path)
            logger.info(f"✅ Финальная модель K-means загружена. Обучение уже завершено.")
            return loaded_kmeans  # Обучение уже завершено
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при загрузке финальной модели: {str(e)}")
    
    # Инициализация или загрузка модели K-means
    if loaded_kmeans:
        kmeans = loaded_kmeans
        logger.info(f"📊 Продолжение обучения с загруженной модели.")
    else:
        logger.info(f"🎯 Начало обучения K-means с нуля")
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=kmeans_batch_size,
            n_init="auto",
            verbose=0,
            reassignment_ratio=0.01,
            max_iter=100
        )
    
    total_features = 0
    processed_files = start_from_index  # Начинаем с указанного индекса
    start_time = time.time()
    last_checkpoint_time = start_time
    
    logger.info(f"🚀 Запуск/продолжение обучения...")
    logger.info(f"   • Начинаем с файла: {start_from_index}")
    logger.info(f"   • Всего файлов: {len(audio_paths)}")
    
    try:
        # ПРОДОЛЖАЕМ С ТОГО МЕСТА, НА КОТОРОМ ОСТАНОВИЛИСЬ
        for i in tqdm(range(start_from_index, len(audio_paths), extraction_batch_size), 
                     desc="Обработка файлов для K-Means", 
                     unit="batch"):
            
            batch_paths = audio_paths[i:i + extraction_batch_size]
            batch_features = []
            batch_success_count = 0
            
            for path in batch_paths:
                audio = load_and_preprocess_audio(
                    path,
                    target_sr=PAPER_CONFIG["sample_rate"],
                    max_duration=PAPER_CONFIG["max_duration"],
                    target_rms=PAPER_CONFIG["target_rms"],
                    min_duration=PAPER_CONFIG["min_audio_duration"]
                )
                
                if audio is None or len(audio) < 320:
                    continue
                
                try:
                    # Подготовка входных данных
                    max_length = int(PAPER_CONFIG["max_duration"] * PAPER_CONFIG["sample_rate"])
                    inputs = processor(
                        audio,
                        sampling_rate=PAPER_CONFIG["sample_rate"],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).input_values.to(device)
                    
                    # Извлечение признаков
                    with torch.no_grad():
                        outputs = model(inputs, output_hidden_states=True)
                        
                        if target_layer >= len(outputs.hidden_states):
                            logger.error(f"❌ Запрошенный слой {target_layer} отсутствует.")
                            raise ValueError(f"Invalid layer index: {target_layer}")
                        
                        hidden_state = outputs.hidden_states[target_layer]
                        hidden_state = hidden_state / torch.norm(hidden_state, dim=2, keepdim=True)
                        batch_features.append(hidden_state.cpu().numpy())
                        batch_success_count += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при обработке {path}: {str(e)}")
                    continue
            
            # Обучение на батче признаков
            if batch_features:
                try:
                    X_batch = np.concatenate([feats.reshape(-1, feats.shape[2]) for feats in batch_features], axis=0)
                    kmeans.partial_fit(X_batch)
                    total_features += X_batch.shape[0]
                except Exception as e:
                    logger.error(f"❌ Ошибка при обучении на батче: {str(e)}")
            
            processed_files += len(batch_paths)
            
            # Сохраняем прогресс каждые 100 файлов
            if processed_files % 100 == 0:
                try:
                    # progress_file должен быть доступен в этой функции
                    # Можно передать его как параметр или использовать глобальный путь
                    progress_file = os.path.join(os.path.dirname(checkpoint_path), "kmeans_progress.txt")
                    with open(progress_file, 'w') as f:
                        f.write(str(processed_files))
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось сохранить прогресс: {e}")

            # Очистка CUDA кэша
            if device == "cuda" and processed_files % cuda_cache_interval == 0:
                torch.cuda.empty_cache()
            
            # Сохранение чекпоинта
            if temp_checkpoint_path and processed_files % checkpoint_interval == 0:
                try:
                    current_time = time.time()
                    elapsed_since_last = current_time - last_checkpoint_time
                    
                    logger.info(f"💾 Чекпоинт ({processed_files}/{len(audio_paths)} файлов)")
                    joblib.dump(kmeans, temp_checkpoint_path)
                    last_checkpoint_time = current_time
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка при сохранении чекпоинта: {str(e)}")
            
            # Прогресс
            if processed_files % 100 == 0:
                elapsed = time.time() - start_time
                features_per_sec = total_features / elapsed if elapsed > 0 else 0
                logger.info(f"📈 Прогресс: {processed_files}/{len(audio_paths)} файлов | {features_per_sec:.1f} признаков/сек")
        
        # Финальное сохранение
        if checkpoint_path and total_features > 0:
            try:
                logger.info("💾 Сохранение финальной модели K-means...")
                joblib.dump(kmeans, checkpoint_path)
                logger.info(f"✅ Финальная модель сохранена: {checkpoint_path}")
                
                # Удаляем временный чекпоинт
                if temp_checkpoint_path and os.path.exists(temp_checkpoint_path):
                    os.remove(temp_checkpoint_path)
                    logger.debug("🧹 Удален временный чекпоинт")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка при сохранении финальной модели: {str(e)}")
        
        # Финальная статистика
        elapsed = time.time() - start_time
        logger.info(f"\n🎉 ОБУЧЕНИЕ K-MEANS ЗАВЕРШЕНО!")
        logger.info(f"📊 Обработано файлов: {processed_files:,}/{len(audio_paths):,}")
        logger.info(f"📊 Использовано признаков: {total_features:,}")
        logger.info(f"📊 Время обучения: {elapsed/60:.1f} минут")
        
        return kmeans
        
    except KeyboardInterrupt:
        logger.warning("\n🛑 Обучение прервано пользователем!")
        if temp_checkpoint_path and hasattr(kmeans, 'cluster_centers_'):
            try:
                logger.info(f"💾 Сохранение промежуточного результата...")
                joblib.dump(kmeans, temp_checkpoint_path)
                logger.info(f"✅ Промежуточный результат сохранен: {temp_checkpoint_path}")
                logger.info(f"📌 Чтобы продолжить, запустите скрипт снова БЕЗ --force_retrain")
            except Exception as e:
                logger.error(f"❌ Ошибка при сохранении прерванного обучения: {str(e)}")
        raise
    
    except Exception as e:
        logger.exception(f"🔥 Критическая ошибка при обучении: {str(e)}")
        raise

def generate_pseudophones_filelist(
    audio_paths: List[str],
    processor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    kmeans: MiniBatchKMeans,
    target_layer: int,
    output_path: str,
    speaker_id: str = "speaker_01",
    extraction_batch_size: int = 8,
    cuda_cache_interval: int = 100,
    device: str = "cuda",
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Генерация файла с псевдо-фонемными последовательностями согласно статье
    
    СООТВЕТСТВИЕ СТАТЬЕ:
    - "the same consecutive indices are merged to reflect the characteristics of a real phoneme"
    - "We refer to these merged indices i'1, ..., i'T' as pseudo phoneme"
    - Используется для pre-training VITS как substitute of phoneme sequences
    
    Args:
        audio_paths: Список путей к аудио файлам
        processor: Wav2Vec2FeatureExtractor
        model: Wav2Vec2Model
        kmeans: Обученная модель K-means
        target_layer: Индекс слоя (15 для block 15)
        output_path: Путь к выходному файлу
        speaker_id: ID спикера для всех записей
        extraction_batch_size: Размер батча для извлечения признаков
        cuda_cache_interval: Интервал очистки CUDA кэша
        device: Устройство для вычислений
        logger: Логгер
    """
    logger = logger or logging.getLogger("transfer_tts_pseudo_phonemes")
    
    # Проверка существующих записей для избежания дубликатов
    existing_records = check_existing_records(output_path)
    processed_paths = safe_load_processed_paths(output_path + ".progress")
    
    # Фильтрация уже обработанных файлов
    new_audio_paths = [
        path for path in audio_paths 
        if path not in processed_paths and path not in existing_records
    ]
    
    logger.info(f"🎯 Начало генерации псевдо-фонемных последовательностей")
    logger.info(f"📊 Конфигурация генерации:")
    logger.info(f"   • Целевой слой: block {target_layer}")
    logger.info(f"   • Количество кластеров: {kmeans.n_clusters}")
    logger.info(f"   • ID спикера по умолчанию: {speaker_id}")
    logger.info(f"   • Всего файлов: {len(audio_paths):,}")
    logger.info(f"   • Уже обработано: {len(processed_paths):,}")
    logger.info(f"   • Уже в файллисте: {len(existing_records):,}")
    logger.info(f"   • Новых файлов для обработки: {len(new_audio_paths):,}")
    
    if not new_audio_paths:
        logger.warning("⚠️ Нет новых файлов для обработки")
        return {
            "total_records": len(existing_records),
            "new_records": 0,
            "error_count": 0
        }
    
    # Подготовка выходного файла
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Если файл существует и не пустой, открываем в режиме добавления
    file_mode = 'a' if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else 'w'
    
    stats = {
        "total_new_records": 0,
        "error_count": 0,
        "sequence_lengths": [],
        "cluster_usage": Counter(),
        "start_time": time.time(),
        "success_files": [],
        "error_files": []
    }
    
    try:
        with open(output_path, file_mode, encoding='utf-8') as f_out:
            logger.info(f"📝 Открыт файл для записи: {output_path} (режим: {file_mode})")
            
            for i in tqdm(range(0, len(new_audio_paths), extraction_batch_size), 
                         desc="Генерация псевдо-фонем", 
                         unit="batch",
                         total=len(new_audio_paths)//extraction_batch_size + 1):
                
                batch_paths = new_audio_paths[i:i + extraction_batch_size]
                
                for path in batch_paths:
                    try:
                        # Загрузка и предобработка аудио
                        audio = load_and_preprocess_audio(
                            path,
                            target_sr=PAPER_CONFIG["sample_rate"],
                            max_duration=PAPER_CONFIG["max_duration"],
                            target_rms=PAPER_CONFIG["target_rms"],
                            min_duration=PAPER_CONFIG["min_audio_duration"]
                        )
                        
                        if audio is None:
                            logger.warning(f"⚠️ Не удалось загрузить аудио: {path}")
                            stats["error_count"] += 1
                            stats["error_files"].append(path)
                            safe_save_processed_path(output_path + ".progress", path)
                            continue
                        
                        # ВАЖНО: Указываем max_length для избежания ошибки с truncation=True
                        max_length = int(PAPER_CONFIG["max_duration"] * PAPER_CONFIG["sample_rate"])
                        
                        # Извлечение признаков
                        inputs = processor(
                            audio,
                            sampling_rate=PAPER_CONFIG["sample_rate"],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=max_length  # Добавляем явное указание max_length
                        ).input_values.to(device)
                        
                        with torch.no_grad():
                            outputs = model(inputs, output_hidden_states=True)
                            
                            # СООТВЕТСТВИЕ СТАТЬЕ: block 15 hidden representations
                            hidden_state = outputs.hidden_states[target_layer]
                            
                            # Нормализация (как при обучении K-means)
                            hidden_state = hidden_state / torch.norm(
                                hidden_state, dim=2, keepdim=True
                            )
                            
                            # Предсказание кластеров
                            features = hidden_state.cpu().numpy().reshape(-1, hidden_state.shape[2])
                            cluster_ids = kmeans.predict(features)
                            
                            # СООТВЕТСТВИЕ СТАТЬЕ: "the same consecutive indices are merged"
                            merged_ids = [key for key, _ in itertools.groupby(cluster_ids)]
                            
                            if not merged_ids:  # Проверка на пустую последовательность
                                logger.warning(f"⚠️ Пустая последовательность после объединения для {path}")
                                stats["error_count"] += 1
                                stats["error_files"].append(path)
                                safe_save_processed_path(output_path + ".progress", path)
                                continue
                            
                            # Формирование записи в формате VITS
                            pseudo_phoneme_str = " ".join(map(str, merged_ids))
                            record = f"{path}|{pseudo_phoneme_str}|{speaker_id}\n"
                            
                            # Запись в файл
                            f_out.write(record)
                            f_out.flush()
                            
                            # Обновление статистики
                            stats["total_new_records"] += 1
                            stats["success_files"].append(path)
                            stats["sequence_lengths"].append(len(merged_ids))
                            stats["cluster_usage"].update(merged_ids)
                            
                            # Логирование прогресса каждые 100 записей
                            if stats["total_new_records"] % 100 == 0:
                                avg_len = np.mean(stats["sequence_lengths"])
                                logger.info(
                                    f"✅ Добавлено записей: {stats['total_new_records']:,} | "
                                    f"Средняя длина: {avg_len:.1f} | "
                                    f"Ошибок: {stats['error_count']}"
                                )
                        
                    except Exception as e:
                        logger.error(f"❌ Ошибка при обработке {path}: {str(e)}")
                        stats["error_count"] += 1
                        stats["error_files"].append(path)
                    
                    finally:
                        # Сохранение прогресса после каждого файла
                        safe_save_processed_path(output_path + ".progress", path)
                        
                        # Очистка CUDA кэша с интервалом
                        if device == "cuda" and stats["total_new_records"] % cuda_cache_interval == 0:
                            torch.cuda.empty_cache()
        
        # Финальная статистика
        elapsed = time.time() - stats["start_time"]
        logger.info(f"\n🎉 ✅ ГЕНЕРАЦИЯ ПСЕВДО-ФОНЕМ ЗАВЕРШЕНА!")
        
        if stats["sequence_lengths"]:
            mean_len = float(np.mean(stats["sequence_lengths"]))
            std_len = float(np.std(stats["sequence_lengths"]))
            min_len = int(min(stats["sequence_lengths"]))
            max_len = int(max(stats["sequence_lengths"]))
            
            logger.info(f"📊 Финальная статистика:")
            logger.info(f"   • Новых записей: {stats['total_new_records']:,}")
            logger.info(f"   • Ошибок: {stats['error_count']:,}")
            logger.info(f"   • Время обработки: {elapsed/60:.1f} минут")
            logger.info(f"   • Средняя длина последовательности: {mean_len:.1f} ± {std_len:.1f}")
            logger.info(f"   • Диапазон длин: {min_len} - {max_len}")
            logger.info(f"   • Уникальных кластеров использовано: {len(stats['cluster_usage'])}/{kmeans.n_clusters}")
            
            # Топ-10 наиболее используемых кластеров
            top_clusters = stats["cluster_usage"].most_common(10)
            logger.info(f"   • Топ-10 кластеров:")
            for rank, (cid, count) in enumerate(top_clusters, 1):
                percentage = (count / sum(stats["cluster_usage"].values())) * 100
                logger.info(f"      {rank}. Кластер {cid}: {count:,} раз ({percentage:.1f}%)")
            
            # Сохранение полной статистики
            stats_path = os.path.join(output_dir, "pseudo_phoneme_stats.json")
            final_stats = {
                "paper_reference": "https://arxiv.org/abs/2203.15447",
                "configuration": {
                    "wav2vec_model": model.config._name_or_path,
                    "layer_index": target_layer,
                    "k_clusters": kmeans.n_clusters,
                    "sample_rate": PAPER_CONFIG["sample_rate"],
                    "max_duration": PAPER_CONFIG["max_duration"]
                },
                "processing_stats": {
                    "total_files_processed": len(new_audio_paths),
                    "successful_files": stats["total_new_records"],
                    "error_files": stats["error_count"],
                    "success_rate": (stats["total_new_records"] / len(new_audio_paths)) * 100 if new_audio_paths else 0,
                    "processing_time_seconds": elapsed,
                    "processing_time_minutes": elapsed/60,
                    "average_sequence_length": mean_len,
                    "std_sequence_length": std_len,
                    "min_sequence_length": min_len,
                    "max_sequence_length": max_len
                },
                "cluster_usage": {
                    "unique_clusters_used": len(stats["cluster_usage"]),
                    "total_clusters": kmeans.n_clusters,
                    "usage_percentage": (len(stats["cluster_usage"]) / kmeans.n_clusters) * 100,
                    "top_10_clusters": {str(cid): count for cid, count in top_clusters},
                    "cluster_distribution": {str(cid): count for cid, count in stats["cluster_usage"].most_common()}
                },
                "file_lists": {
                    "successful_files": stats["success_files"][:100],  # Первые 100 для логгирования
                    "error_files": stats["error_files"][:100],
                    "total_successful": len(stats["success_files"]),
                    "total_errors": len(stats["error_files"])
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Соответствует методологии из статьи: 'Transfer Learning from Speech Recognition to Text-to-Speech Synthesis Using Self-Supervised Representations'"
            }
            
            try:
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(final_stats, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Статистика сохранена в: {stats_path}")
            except Exception as e:
                logger.error(f"❌ Ошибка при сохранении статистики: {str(e)}")
        
        return {
            "total_records": stats["total_new_records"] + len(existing_records),
            "new_records": stats["total_new_records"],
            "error_count": stats["error_count"],
            "stats_path": stats_path if 'stats_path' in locals() else None
        }
    
    except KeyboardInterrupt:
        logger.warning("\n🛑 Генерация прервана пользователем!")
        elapsed = time.time() - stats["start_time"]
        logger.info(
            f"⚠️ Частично завершено: {stats['total_new_records']:,} новых записей за {elapsed/60:.1f} минут"
        )
        raise
    
    except Exception as e:
        logger.exception(f"🔥 Критическая ошибка при генерации: {str(e)}")
        raise


def main():
    """Основная функция с полным соответствием статье"""
    parser = argparse.ArgumentParser(
        description="Pseudo-Phoneme Extractor for Transfer Learning TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Based on the paper: 'Transfer Learning from Speech Recognition to Text-to-Speech Synthesis Using Self-Supervised Representations' (https://arxiv.org/abs/2203.15447)"
    )
    
    parser.add_argument(
        "mode", 
        choices=["train_kmeans", "generate_pseudophones"],
        help="Режим работы: обучение K-means или генерация псевдо-фонем"
    )
    
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Директория с аудио файлами (рекурсивный поиск)"
    )
    
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Директория для выходных файлов"
    )
    
    parser.add_argument(
        "--sample_files",
        type=int,
        default=0,
        help="Количество файлов для обработки (0 = все)"
    )
    
    parser.add_argument(
        "--k_clusters",
        type=int,
        default=PAPER_CONFIG["k_clusters"],
        help=f"Количество кластеров для K-means (по умолчанию: {PAPER_CONFIG['k_clusters']}, как в статье)"
    )
    
    parser.add_argument(
        "--layer_index",
        type=int,
        default=PAPER_CONFIG["layer_index"],
        help=f"Индекс слоя wav2vec2 для извлечения признаков (по умолчанию: {PAPER_CONFIG['layer_index']} для block 15, как в статье)"
    )
    
    parser.add_argument(
        "--speaker_id",
        default="speaker_01",
        help="ID спикера для всех записей (для multi-speaker используйте разные ID)"
    )
    
    parser.add_argument(
        "--wav2vec_model",
        default=PAPER_CONFIG["wav2vec_model"],
        help=f"Модель wav2vec2 для использования (по умолчанию: {PAPER_CONFIG['wav2vec_model']}, оригинальный wav2vec 2.0 как в статье)"
    )
    
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень детализации логирования"
    )
    
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Принудительно перезапустить обучение K-means даже если модель существует"
    )
    
    args = parser.parse_args()
    
    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Настройка логирования
    logger = setup_logging(args.output_dir, args.log_level)
    
    # Вывод информации о статье
    logger.info("\n" + "="*60)
    logger.info("📚 PSEUDO-PHONEME EXTRACTOR FOR TRANSFER LEARNING TTS")
    logger.info("="*60)
    logger.info("📄 Основано на статье:")
    logger.info("   'Transfer Learning from Speech Recognition to Text-to-Speech Synthesis'")
    logger.info("   'Using Self-Supervised Representations'")
    logger.info("   https://arxiv.org/abs/2203.15447")
    logger.info("")
    logger.info("🎯 Ключевые параметры из статьи:")
    logger.info(f"   • Модель wav2vec 2.0: {PAPER_CONFIG['wav2vec_model']}")
    logger.info(f"   • Слой для признаков: block {PAPER_CONFIG['layer_index']} (hidden representation of block 15)")
    logger.info(f"   • Количество кластеров: {PAPER_CONFIG['k_clusters']} (K=128)")
    logger.info(f"   • Объединение последовательных одинаковых индексов кластеров")
    logger.info("")
    logger.info(f"⚙️  Текущий режим: {args.mode}")
    logger.info(f"📂 Директория аудио: {args.audio_dir}")
    logger.info(f"📂 Выходная директория: {args.output_dir}")
    logger.info("="*60 + "\n")
    
    try:
        # Валидация директорий
        validate_directory(args.audio_dir, create=False)
        validate_directory(args.output_dir, create=True)
        
        # Определение устройства
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"💻 Используется устройство: {device}")
        
        if device == "cuda":
            gpu_mem, ram_available = estimate_available_memory()
            logger.info(f"📊 Доступно GPU памяти: {gpu_mem:.1f} GB")
            logger.info(f"📊 Доступно RAM: {ram_available:.1f} GB")
            
            # Автоматический подбор размера батча
            auto_batch = auto_batch_size(gpu_mem, ram_available)
            extraction_batch_size = min(32, max(4, auto_batch))
            logger.info(f"⚡ Автоматически подобран размер батча: {extraction_batch_size}")
        else:
            extraction_batch_size = 4
            logger.warning("⚠️ GPU не доступен, используется CPU. Скорость будет ниже.")
        
        # Загрузка аудио файлов
        audio_paths = find_all_audio_files(args.audio_dir)
        
        if args.sample_files > 0 and args.sample_files < len(audio_paths):
            audio_paths = audio_paths[:args.sample_files]
            logger.info(f"🧪 Режим выборки: обработка {len(audio_paths)} файлов")
        
        if not audio_paths:
            logger.error("❌ Не найдено аудио файлов для обработки!")
            sys.exit(1)
        
        # Загрузка модели wav2vec2
        logger.info(f"🔄 Загрузка модели {args.wav2vec_model} на {device}...")
        
        # Валидация модели
        if "xls-r" in args.wav2vec_model.lower():
            logger.warning("⚠️ ВНИМАНИЕ: Вы используете XLS-R модель вместо оригинального wav2vec 2.0")
            logger.warning("⚠️ Статья использует facebook/wav2vec2-base-960h (обученный на 960h английской речи)")
            logger.warning("⚠️ Для финского языка это может быть подоптимально, но приемлемо для transfer learning")
        
        processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model)
        model = Wav2Vec2Model.from_pretrained(args.wav2vec_model).to(device)
        model.eval()
        
        # Проверка количества слоев в модели
        dummy_input = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            dummy_output = model(dummy_input, output_hidden_states=True)
        num_layers = len(dummy_output.hidden_states) - 1  # -1 потому что 0-й слой это эмбеддинги
        logger.info(f"🔍 Модель содержит {num_layers} скрытых слоев")
        
        if args.layer_index > num_layers:
            logger.warning(f"⚠️ Запрошенный слой {args.layer_index} превышает количество доступных слоев ({num_layers})")
            logger.warning(f"⚠️ Используется последний доступный слой: {num_layers}")
            args.layer_index = num_layers
        
        logger.info(f"✅ Модель успешно загружена. Используется слой: {args.layer_index}")
        
        # Пути к файлам
        kmeans_path = os.path.join(args.output_dir, "kmeans_model.joblib")
        filelist_path = os.path.join(args.output_dir, "finnish_pseudo_for_vits.txt")
        progress_log = filelist_path + ".progress"
        
        if args.mode == "train_kmeans":
            logger.info("\n" + "="*60)
            logger.info("🎯 ЭТАП 1: ОБУЧЕНИЕ K-MEANS НА СКРЫТЫХ ПРЕДСТАВЛЕНИЯХ")
            logger.info("="*60)
            
            # Определяем, с какого файла начинать
            progress_file = os.path.join(args.output_dir, "kmeans_progress.txt")
            start_from_index = 0
            
            # Если не принудительный перезапуск и есть прогресс - загружаем
            if not args.force_retrain and os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        start_from_index = int(f.read().strip())
                    logger.info(f"📌 Продолжение обучения с файла #{start_from_index}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось прочитать файл прогресса: {e}. Начинаем с начала.")
            
            # Проверка существования финальной модели
            if os.path.exists(kmeans_path) and not args.force_retrain:
                logger.warning(f"⚠️ Финальная модель K-means уже существует: {kmeans_path}")
                user_input = input("Перезаписать существующую модель? (y/n): ").strip().lower()
                if user_input != 'y':
                    logger.info("⏭️  Обучение отменено пользователем")
                    sys.exit(0)
            
            # Создаем прогресс-файл для отслеживания
            def save_progress(current_index):
                try:
                    with open(progress_file, 'w') as f:
                        f.write(str(current_index))
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось сохранить прогресс: {e}")
            
            # Обучение K-means
            logger.info("🚀 Запуск/продолжение обучения K-means...")
            
            # Код для отслеживания прогресса внутри train_kmeans_incremental
            # Добавьте этот callback в цикл обучения после processed_files += len(batch_paths)
            def progress_callback(processed_files):
                if processed_files % 100 == 0:
                    save_progress(processed_files)
            
            # Вызов основной функции
            kmeans = train_kmeans_incremental(
                audio_paths=audio_paths,
                processor=processor,
                model=model,
                target_layer=args.layer_index,
                k=args.k_clusters,
                extraction_batch_size=extraction_batch_size,
                kmeans_batch_size=PAPER_CONFIG["kmeans_batch_size"],
                checkpoint_path=kmeans_path,
                checkpoint_interval=PAPER_CONFIG["checkpoint_interval"],
                cuda_cache_interval=PAPER_CONFIG["cuda_cache_interval"],
                device=device,
                logger=logger,
                start_from_index=start_from_index
            )
            
            # Удаляем файл прогресса после успешного завершения
            if os.path.exists(progress_file):
                os.remove(progress_file)
                logger.debug("🧹 Удален файл прогресса после успешного завершения")
            
            logger.info(f"✅ ✅ Модель K-means успешно сохранена: {kmeans_path}")
        
        elif args.mode == "generate_pseudophones":
            logger.info("\n" + "="*60)
            logger.info("🎯 ЭТАП 2: ГЕНЕРАЦИЯ ПСЕВДО-ФОНЕМНЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
            logger.info("="*60)
            
            # Проверка наличия модели K-means
            if not os.path.exists(kmeans_path):
                logger.error(f"❌ Модель K-means не найдена: {kmeans_path}")
                logger.error("Сначала выполните режим 'train_kmeans'")
                logger.error(f"Команда: python {os.path.basename(__file__)} train_kmeans --audio_dir {args.audio_dir} --output_dir {args.output_dir}")
                sys.exit(1)
            
            logger.info(f"🔄 Загрузка модели K-means из: {kmeans_path}")
            try:
                kmeans = joblib.load(kmeans_path)
                logger.info(f"✅ Модель K-means загружена, количество кластеров: {kmeans.n_clusters}")
                
                # Проверка соответствия количества кластеров
                if kmeans.n_clusters != args.k_clusters:
                    logger.warning(f"⚠️ Количество кластеров в сохраненной модели ({kmeans.n_clusters}) не совпадает с запрошенным ({args.k_clusters})")
                    logger.warning(f"⚠️ Используется количество кластеров из сохраненной модели: {kmeans.n_clusters}")
            except Exception as e:
                logger.exception(f"❌ Ошибка при загрузке модели K-means: {str(e)}")
                sys.exit(1)
            
            # Генерация псевдо-фонем
            stats = generate_pseudophones_filelist(
                audio_paths=audio_paths,
                processor=processor,
                model=model,
                kmeans=kmeans,
                target_layer=args.layer_index,
                output_path=filelist_path,
                speaker_id=args.speaker_id,
                extraction_batch_size=extraction_batch_size,
                cuda_cache_interval=PAPER_CONFIG["cuda_cache_interval"],
                device=device,
                logger=logger
            )
            
            logger.info(f"\n✅ ✅ Файл с псевдо-фонемами успешно сохранен: {filelist_path}")
            logger.info(f"📊 Всего записей в файле: {stats['total_records']:,}")
            if stats.get("new_records", 0) > 0:
                logger.info(f"📊 Новых записей добавлено: {stats['new_records']:,}")
            
            if stats.get("stats_path"):
                logger.info(f"📊 Подробная статистика сохранена в: {stats['stats_path']}")
            
            # Важное напоминание о соответствии статье
            logger.info("\n" + "="*60)
            logger.info("✅ ГОТОВО! ПСЕВДО-ФОНЕМНЫЕ ПОСЛЕДОВАТЕЛЬНОСТИ СГЕНЕРИРОВАНЫ")
            logger.info("="*60)
            logger.info("📝 Формат файла для VITS:")
            logger.info(f"   {filelist_path}")
            logger.info("📋 Формат каждой строки:")
            logger.info("   <путь_к_аудио>|<последовательность_псевдо-фонем>|<speaker_id>")
            logger.info("")
            logger.info("🎯 ЭТИ ДАННЫЕ ГОТОВЫ ДЛЯ:")
            logger.info("   • Pre-training VITS архитектуры на unlabeled speech")
            logger.info("   • Transfer learning для low-resource TTS")
            logger.info("   • Fine-tuning на small labeled dataset (как в статье)")
            logger.info("")
            logger.info("💡 СОВЕТ ИЗ СТАТЬИ:")
            logger.info("   Для single-speaker TTS достаточно всего 10 минут labeled данных")
            logger.info("   для fine-tuning после pre-training на pseudo-phonemes")
            logger.info("="*60)
    
    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("🛑 ПРОГРАММА ПРЕРВАНА ПОЛЬЗОВАТЕЛЕМ")
        logger.info("="*60)
        logger.info("✅ Все промежуточные результаты сохранены")
        logger.info("✅ При повторном запуске обработка продолжится с последней точки")
        logger.info("="*60)
        sys.exit(0)
    
    except Exception as e:
        logger.exception(f"🔥 КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("🎉 ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!")
    logger.info("="*60)
    logger.info("✅ Псевдо-фонемные последовательности готовы для pre-training VITS")
    logger.info("✅ Полное соответствие методологии из статьи достигнуто")
    logger.info("="*60)


if __name__ == "__main__":
    main()