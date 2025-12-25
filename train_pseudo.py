# train_pseudo.py
import os
import json
import torch
from torch.utils.data import DataLoader
from models import SynthesizerTrn
from data_utils import get_dataset
from torch.nn.utils import clip_grad_norm_

import random
import shutil

# ------------------------------
# 1️⃣ Загрузка конфигурации
# ------------------------------
with open("configs/fi_pseudo_pretrain.json", "r", encoding="utf-8") as f:
    config = json.load(f)

train_config = config["train"]
data_config = config["data"]
model_config = config["model"]

# === Параметры поднабора ===
subset_dir = r"H:\tts\fin\subset_wav_22050"
source_dir = data_config["training_files_dir"]  # обязательно добавь этот ключ в json
num_files = 1000  # количество файлов поднабор

os.makedirs(subset_dir, exist_ok=True)
all_files = [f for f in os.listdir(source_dir) if f.endswith(".wav")]
subset_files = random.sample(all_files, num_files)

for f in subset_files:
    shutil.copy(os.path.join(source_dir, f), os.path.join(subset_dir, f))

print(f"✅ Поднабор для предобучения: {len(subset_files)} файлов")
# меняем путь в конфиге на поднабор
data_config["training_files"] = subset_dir

# ------------------------------
# 2️⃣ Датасет
# ------------------------------
class HParams:
    def __init__(self):
        self.text_cleaners = data_config.get("text_cleaners", [])
        self.max_wav_value = data_config.get("max_wav_value", 32768.0)
        self.sampling_rate = data_config.get("sampling_rate", 22050)
        self.filter_length = data_config.get("filter_length", 1024)
        self.hop_length = data_config.get("hop_length", 256)
        self.win_length = data_config.get("win_length", 1024)
        self.n_mel_channels = data_config.get("n_mel_channels", 80)
        self.mel_fmin = data_config.get("mel_fmin", 0.0)
        self.mel_fmax = data_config.get("mel_fmax", None)
        self.add_blank = data_config.get("add_blank", False)           
        self.min_text_len = data_config.get("min_text_len", 1)       
        self.max_text_len = data_config.get("max_text_len", 1000)    
        self.segment_size = train_config["segment_size"]
        self.n_speakers = 0
        self.distributed_run = False
        self.use_pseudo_text_encoder = True
        self.world_size = 1
        self.rank = 0 

hparams = HParams()
train_dataset, train_loader = get_dataset(
    data_config["training_files"], 
    hparams,           
    num_workers=0, 
    batch_size=train_config["batch_size"]
)
val_dataset, val_loader = get_dataset(
    data_config["validation_files"], 
    hparams, 
    num_workers=0, 
    batch_size=train_config["batch_size"]
)

print(f"✅ Train batches: {len(train_loader)}")
print(f"✅ Val batches: {len(val_loader)}")

# ------------------------------
# 3️⃣ Модель
# ------------------------------
synth = SynthesizerTrn(
    n_vocab=model_config['n_vocab'],
    spec_channels=model_config['spec_channels'],
    segment_size=train_config['segment_size'],
    inter_channels=model_config['inter_channels'],
    hidden_channels=model_config['hidden_channels'],
    filter_channels=model_config['filter_channels'],
    n_heads=model_config['n_heads'],
    n_layers=model_config['n_layers'],
    kernel_size=model_config['kernel_size'],
    p_dropout=model_config['p_dropout'],
    resblock=model_config['resblock'],
    resblock_kernel_sizes=model_config['resblock_kernel_sizes'],
    resblock_dilation_sizes=model_config['resblock_dilation_sizes'],
    upsample_rates=model_config['upsample_rates'],
    upsample_initial_channel=model_config['upsample_initial_channel'],
    upsample_kernel_sizes=model_config['upsample_kernel_sizes'],
    n_speakers=model_config.get('n_speakers', 0),
    gin_channels=model_config.get('gin_channels', 0),
    use_sdp=True,
    use_pseudo_text_encoder=model_config.get('use_pseudo_text_encoder', False),
    pseudo_text_encoder_params=model_config.get('pseudo_text_encoder', {})
)

device = "cuda" if torch.cuda.is_available() else "cpu"
synth.to(device)

# ------------------------------
# 4️⃣ Оптимизатор
# ------------------------------
optimizer = torch.optim.Adam(
    synth.parameters(),
    lr=train_config["learning_rate"],
    betas=train_config["betas"],
    eps=train_config["eps"]
)

# ------------------------------
# 5️⃣ Чекпоинт
# ------------------------------
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"✅ Сохранен чекпоинт: {path}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"🔄 Загрузка чекпоинта: {path}")
    return checkpoint['epoch']

# ------------------------------
# 6️⃣ Функция обучения одной эпохи
# ------------------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch, train=True, checkpoint_dir="checkpoints", checkpoint_interval=1000):
    model.train() if train else model.eval()
    total_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        x, x_lengths, spec, spec_lengths, y, y_lengths, spk_ids = [b.to(device) for b in batch]

        if train:
            optimizer.zero_grad()

        o, l_length, attn, ids_slice, x_mask, y_mask, *_ = model(
            x, x_lengths, spec, spec_lengths, y, y_lengths, spk_ids
        )

        loss = l_length.mean()

        if train:
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        # Сохраняем промежуточный чекпоинт каждые N шагов
        if train and step % checkpoint_interval == 0:
            step_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, step_checkpoint)
            print(f"💾 Сохранили промежуточный чекпоинт: {step_checkpoint}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ------------------------------
# 7️⃣ Основной цикл
# ------------------------------
start_epoch = 1
latest_checkpoint = os.path.join(checkpoint_dir, "latest.pt")
if os.path.exists(latest_checkpoint):
    start_epoch = load_checkpoint(synth, optimizer, latest_checkpoint) + 1

num_epochs = train_config["epochs"]
log_interval = train_config.get("log_interval", 1)
eval_interval = train_config.get("eval_interval", 5)

for epoch in range(start_epoch, num_epochs + 1):
    train_loss = train_one_epoch(synth, train_loader, optimizer, device, epoch, train=True)
    
    if epoch % log_interval == 0:
        print(f"[Epoch {epoch}] train_loss: {train_loss:.6f}")
    
    if epoch % eval_interval == 0:
        val_loss = train_one_epoch(synth, val_loader, optimizer=None, device=device, epoch=epoch, train=False)
        print(f"[Epoch {epoch}] val_loss: {val_loss:.6f}")
    
    # Сохраняем последний чекпоинт
    save_checkpoint(synth, optimizer, epoch, latest_checkpoint)

    # "Вечный" чекпоинт каждые 1000 эпох
    if epoch % 1000 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"synth_epoch_{epoch}.pt")
        save_checkpoint(synth, optimizer, epoch, checkpoint_path)
