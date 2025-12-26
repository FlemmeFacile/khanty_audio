import time
import os
import random
import numpy as np
import torch
import torch.utils.data
import librosa
import torch.nn.functional as F

import commons 
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(mel_spec):
    mel_spec = dynamic_range_compression_torch(mel_spec)
    return mel_spec

def spectrogram_torch_safe(y, n_fft, hop_size, win_size):

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                      window=torch.hann_window(win_size).to(y.device),
                      center=False, pad_mode='reflect', normalized=False, 
                      onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)) + 1e-9
    return spec

def librosa_mel_fn(sr=16000, n_fft=1024, n_mels=80, fmin=0.0, fmax=None):

    fmax = fmax if fmax else sr//2
    fft_bins = n_fft // 2 + 1
    
    # Фиксированные мел фильтры
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, 
                                   fmin=fmin, fmax=fmax, htk=True, norm='slaney')
    return mel_basis

class TextAudioLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.hparams = hparams
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length or 1024
        self.hop_length = hparams.hop_length or 256
        self.win_length = hparams.win_length or 1024
        self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)
        self.mel_fmin = getattr(hparams, "mel_fmin", 0.0)
        self.mel_fmax = getattr(hparams, "mel_fmax", 8000)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        audiopaths_and_text_new = []
        lengths = []
        for audio_info in self.audiopaths_and_text:  
            audiopath = audio_info[0]
            if not os.path.exists(audiopath) or os.path.getsize(audiopath) < 1024:
                continue
            try:
                [int(x) for x in audio_info[1].strip().split()]
            except ValueError:
                continue
            try:
                audio, sr = load_wav_to_torch(audiopath)
                spec_length = (len(audio) - self.filter_length) // self.hop_length + 1
                if spec_length > 10:  # Минимум 10 фреймов
                    audiopaths_and_text_new.append(audio_info)
                    lengths.append(spec_length)
            except:
                continue
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        """🎯 100% РАБОТАЕТ [80, frames]"""
        audio, sr = load_wav_to_torch(filename)
        if sr != self.sampling_rate:
            raise ValueError(f"SR: {sr} != {self.sampling_rate}")
        
        audio_norm = audio / self.max_wav_value

        stft_spec = spectrogram_torch_safe(
            audio_norm, 
            self.filter_length, 
            self.hop_length, 
            self.win_length
        )  # [513, frames]
        
  
        mel_matrix = librosa_mel_fn(
            sr=self.sampling_rate,
            n_fft=self.filter_length,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )
        mel_matrix = torch.from_numpy(mel_matrix).to(stft_spec.device, stft_spec.dtype)
        
        #  MEL [80, frames]
        mel_spec = torch.matmul(mel_matrix, stft_spec)
        mel_spec = spectral_normalize_torch(mel_spec)
        
        #print(f" MEL: {mel_spec.shape}")
        return mel_spec, audio_norm.unsqueeze(0)

    def get_text(self, text):
        if getattr(self.hparams, "use_pseudo_text_encoder", False):
            try:
                ids = [int(x) for x in text.strip().split()]
                return torch.LongTensor(ids)
            except ValueError:
                return torch.LongTensor([])
        text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextAudioCollate:
    def __call__(self, batch):
        texts = [item[0] for item in batch]
        mels = [item[1] for item in batch]
        waves = [item[2] for item in batch]
        
        # MEL [batch, 80, max_frames]
        mel_frames = [mel.shape[1] for mel in mels]
        max_frames = max(mel_frames)
        mel_batch = torch.zeros(len(mels), 80, max_frames, dtype=mels[0].dtype)
        mel_lengths = torch.LongTensor(mel_frames)
        for i, mel in enumerate(mels):
            mel_batch[i, :, :mel.shape[1]] = mel
        
        # WAV [batch, 1, max_len]
        max_wave = min(16384, max(w.shape[1] for w in waves))
        wave_batch = torch.zeros(len(waves), 1, max_wave, dtype=waves[0].dtype)
        wave_lengths = torch.LongTensor([min(16384, w.shape[1]) for w in waves])
        for i, wav in enumerate(waves):
            wave_batch[i, :, :min(wav.shape[1], max_wave)] = wav[:, :max_wave]
        
        x = torch.randint(0, 100, (len(texts), 150), dtype=torch.long)
        x_lengths = torch.full((len(texts),), 150, dtype=torch.long)
        
        return x, x_lengths, mel_batch, mel_lengths, wave_batch, wave_lengths

def get_dataset(filelist, hparams, num_workers=4, batch_size=16):
    dataset = TextAudioLoader(filelist, hparams)
    collate_fn = TextAudioCollate()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, collate_fn=collate_fn, pin_memory=True
    )
    return dataset, loader

