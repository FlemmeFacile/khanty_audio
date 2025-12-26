import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import gc
from multiprocessing import freeze_support, set_start_method

# ------------------------------
# 0️⃣ ИМПОРТЫ ДО torch.cuda
# ------------------------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

print("🔥 VITS TRAINING STARTED")

# ------------------------------
# 1️⃣ Конфигурация
# ------------------------------
with open("configs/khanty_train.json", "r", encoding="utf-8") as f:
    config = json.load(f)

train_config = config["train"]
data_config = config["data"]
model_config = config["model"]

print(f"✅ Config loaded: {len(config)} sections")

# ------------------------------
# 2️⃣ Инициализация
# ------------------------------
torch.manual_seed(train_config["seed"])
random.seed(train_config["seed"])
np.random.seed(train_config["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed(train_config["seed"])
    torch.cuda.empty_cache()
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

# ------------------------------
# 3️⃣ Импорты после CUDA init
# ------------------------------
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

hparams = HParams(**{
    "text_cleaners": data_config.get("text_cleaners", []),
    "max_wav_value": data_config.get("max_wav_value", 32768.0),
    "sampling_rate": data_config.get("sampling_rate", 16000),
    "filter_length": data_config.get("filter_length", 1024),
    "hop_length": data_config.get("hop_length", 256),
    "win_length": data_config.get("win_length", 1024),
    "n_mel_channels": 80,
    "mel_fmin": data_config.get("mel_fmin", 0.0),
    "mel_fmax": 8000.0,
    "add_blank": False,
    "min_text_len": 1,
    "max_text_len": 1000,
    "use_pseudo_text_encoder": False,
    "n_speakers": 0
})

# ------------------------------
# 4️⃣ Датасеты
# ------------------------------
batch_size = 2
num_workers = 0

print("🔄 Loading datasets...")
train_dataset = TextAudioLoader(data_config["training_files"], hparams)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=True, collate_fn=TextAudioCollate())

val_dataset = TextAudioLoader(data_config["validation_files"], hparams)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=num_workers, pin_memory=True, collate_fn=TextAudioCollate())

print(f"✅ Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

# ------------------------------
# 5️⃣ Модель для fine-tuning
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

synth = SynthesizerTrn(
    n_vocab=model_config['n_vocab'],   
    spec_channels=80,
    segment_size=2048,
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
    n_speakers=0,
    gin_channels=0,
    use_pseudo_text_encoder=False  # ← ВАЖНО: False!
).to(device)

print(f"✅ Fine-tuning model loaded: {sum(p.numel() for p in synth.parameters()):,} params")

# ------------------------------
# 6️⃣ Загрузка предобученного чекпоинта (transfer learning)
# ------------------------------
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Загружаем ПРЕДОБУЧЕННЫЙ чекпоинт (например, epoch_10.pt)
pretrain_checkpoint_path = "checkpoints/epoch_10.pt"
if os.path.exists(pretrain_checkpoint_path):
    print(f"📂 Loading pre-trained checkpoint: {pretrain_checkpoint_path}")
    pretrain_ckpt = torch.load(pretrain_checkpoint_path, map_location=device)
    
    # Переносим совпадающие веса (кроме pseudo_text_encoder)
    model_dict = synth.state_dict()
    pretrained_dict = {k: v for k, v in pretrain_ckpt['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    synth.load_state_dict(model_dict)
    print("✅ Pre-trained weights loaded (excluding text encoder)")
else:
    print("⚠️ Pre-trained checkpoint not found! Starting from scratch.")

# ------------------------------
# 7️⃣ Заморозка компонентов (по статье)
# ------------------------------
# Замораживаем decoder и posterior_encoder
for param in synth.dec.parameters():
    param.requires_grad = False
for param in synth.posterior_encoder.parameters():
    param.requires_grad = False

# Проверка
trainable_params = sum(p.numel() for p in synth.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in synth.parameters())
print(f"🔒 Frozen: {total_params - trainable_params:,} params")
print(f"🔓 Trainable: {trainable_params:,} params (text_encoder, dp, flow)")



# ------------------------------
# 8️⃣ Создание оптимизатора ТОЛЬКО для обучаемых параметров
# ------------------------------
# Обучаем ТОЛЬКО: text_encoder, duration_predictor (dp), normalizing flow
params_to_optimize = (
    list(synth.text_encoder.parameters()) +
    list(synth.dp.parameters()) +
    list(synth.flow.parameters())
)

optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=train_config.get("learning_rate", 1e-4),  # можно задать в config
    betas=(0.8, 0.99),
    eps=1e-9
)
print("✅ Optimizer created for trainable components only")


# ------------------------------
# 9️⃣ MAIN TRAINING с безопасным прерыванием
# ------------------------------
print(f"\n🚀 START TRAINING from epoch {start_epoch}!")

try:
    for epoch in range(start_epoch, 501):
        synth.train()
        total_loss = 0
        batch_count = 0
        
        print(f"\n🔥 EPOCH {epoch}/500")
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Unpack batch
                x, x_lengths, mel, mel_lengths, y, y_lengths = batch
            
                if batch_idx == 0:
                    print(f"🔥 BATCH 0 SHAPES:")
                    print(f"  mel: {mel.shape}")
                    print(f"  y: {y.shape}") 
            
                # To GPU
                x = x.to(device, non_blocking=True)
                x_lengths = x_lengths.to(device, non_blocking=True)
                mel = mel.to(device, non_blocking=True)
                mel_lengths = mel_lengths.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_lengths = y_lengths.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=True):
                    output = synth(x, x_lengths, mel, mel_lengths, None)
                    y_hat, l_length, _, _, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = output
                    
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * 1.0
                    loss_dur = l_length.mean()
                    loss = loss_kl + loss_dur
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(synth.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}: loss={loss.item():.4f} (kl={loss_kl.item():.4f}, dur={loss_dur.item():.4f})")
                
                del x, mel, y, y_hat, loss
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Skip batch {batch_idx}: {str(e)[:100]}")
                torch.cuda.empty_cache()
                continue
        
        avg_loss = total_loss / max(batch_count, 1)
        print(f"✅ EPOCH {epoch} COMPLETE | Loss: {avg_loss:.4f} | Batches: {batch_count}")
        
        # Сохраняем КАЖДУЮ эпоху (или можно оставить каждые 5 — см. ниже)
        save_checkpoint(epoch)
        
        torch.cuda.empty_cache()
        gc.collect()

    print("🎉 TRAINING FINISHED!")
    save_checkpoint("final")

except KeyboardInterrupt:
    print("\n🛑 Training interrupted by user. Saving emergency checkpoint...")
    save_checkpoint(epoch, tag="_interrupted")
    print("✅ Emergency checkpoint saved. You can resume training next time.")
    raise


# ------------------------------
# 🔟 Loss функция (должна быть определена)
# ------------------------------
def kl_loss(z_p, logs_q, m_p, logs_p, y_mask):
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * y_mask.float())
    denom = torch.sum(y_mask.float())
    return kl / (denom + 1e-8)

# ------------------------------
# 🔟 Утилита сохранения чекпоинта
# ------------------------------
def save_checkpoint(epoch, tag=""):
    filename = f"{checkpoint_dir}/epoch_{epoch}{tag}.pt"
    torch.save({
        'epoch': epoch,
        'model': synth.state_dict(),
        'optimizer': optimizer.state_dict(),
        'hparams': vars(hparams)
    }, filename)
    print(f"💾 Saved checkpoint: {filename}")
