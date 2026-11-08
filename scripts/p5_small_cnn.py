"""
experiments/rev2_p5_deep/small_cnn.py
Small mel-spectrogram CNN trained from scratch on CPU for GTZAN genre classification.
Specification:
- Input: log-mel (128 mel bins, sr=22050, hop_length=512)
- Training: random 3-second crops (~129 frames)
- Architecture: 4 blocks of Conv-BatchNorm-ReLU-MaxPool (16-32-64-128 channels),
  global average pooling, dropout, linear head (< 1.5M parameters).
- Test time: average logits over all non-overlapping 3-second crops of the clip.
- Training schedule: Adam, 15 epochs, batch size 32.
"""

import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common

SR = 22050
HOP_LENGTH = 512
N_MELS = 128
CROP_SECONDS = 3.0
CROP_SAMPLES = int(SR * CROP_SECONDS)  # 66150
CROP_FRAMES = int(math.ceil(CROP_SAMPLES / HOP_LENGTH))  # ~130 frames

class SmallMelCNN(nn.Module):
    """
    4 blocks of Conv-BatchNorm-ReLU-MaxPool (16-32-64-128 channels),
    Global Average Pooling, Dropout, and Linear head (< 1.5M parameters).
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, n_mels=128, n_frames)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)  # (B, 128, 1, 1)
        x = torch.flatten(x, 1)  # (B, 128)
        x = self.dropout(x)
        logits = self.fc(x)      # (B, num_classes)
        return logits

def compute_log_mel(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Computes 128-bin log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Standardize roughly to zero mean and unit variance for training stability
    log_mel = (log_mel + 40.0) / 40.0
    return log_mel.astype(np.float32)

class GTZANMelDataset(Dataset):
    """
    Dataset of precomputed or on-the-fly log-mel spectrograms for GTZAN.
    In training mode: takes random 3-second crops.
    """
    def __init__(self, track_list: list, labels_map: dict, cache_dir: str = None, is_training: bool = True):
        self.track_list = track_list
        self.labels_map = labels_map
        self.cache_dir = cache_dir
        self.is_training = is_training
        
        # Preload or check paths
        self.samples = []
        for fn in track_list:
            audio_path = common.get_gtzan_audio_path(fn)
            label_str = labels_map[fn]
            label_idx = common.GENRE_TO_IDX[label_str]
            self.samples.append((fn, audio_path, label_idx))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx: int):
        fn, audio_path, label_idx = self.samples[idx]
        
        # Load cached mel if available, else compute
        mel = None
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, f"{fn}.npy")
            if os.path.exists(cache_file):
                mel = np.load(cache_file)
                
        if mel is None:
            y, _ = librosa.load(audio_path, sr=SR, mono=True, duration=30.0)
            mel = compute_log_mel(y, sr=SR)
            if self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                np.save(os.path.join(self.cache_dir, f"{fn}.npy"), mel)
                
        # mel shape: (128, total_frames)
        total_frames = mel.shape[1]
        target_frames = CROP_FRAMES  # ~130 frames
        
        if self.is_training:
            if total_frames > target_frames:
                start = random.randint(0, total_frames - target_frames)
                crop = mel[:, start:start+target_frames]
            else:
                crop = np.pad(mel, ((0, 0), (0, max(0, target_frames - total_frames))), mode="wrap")
            # Return tensor: (1, 128, target_frames)
            return torch.from_numpy(crop).unsqueeze(0), label_idx
        else:
            # Test mode: return all non-overlapping 3s crops
            # For 30s track: total_frames ~1292. Number of crops = total_frames // target_frames
            n_crops = max(1, total_frames // target_frames)
            crops = []
            for i in range(n_crops):
                c = mel[:, i*target_frames:(i+1)*target_frames]
                if c.shape[1] == target_frames:
                    crops.append(c)
            if not crops:
                crops.append(np.pad(mel, ((0, 0), (0, max(0, target_frames - total_frames))), mode="wrap"))
            crops_tensor = torch.from_numpy(np.stack(crops, axis=0)).unsqueeze(1)  # (n_crops, 1, 128, 130)
            return crops_tensor, label_idx

def evaluate_cnn(model: nn.Module, dataset: GTZANMelDataset, device: torch.device):
    """Evaluates CNN by averaging logits over all 3s crops per track."""
    model.eval()
    correct = 0
    total = len(dataset)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(total):
            crops_tensor, target = dataset[i]
            crops_tensor = crops_tensor.to(device)  # (n_crops, 1, 128, 130)
            logits = model(crops_tensor)            # (n_crops, 10)
            avg_logits = logits.mean(dim=0)          # (10,)
            pred = int(torch.argmax(avg_logits).item())
            
            all_preds.append(pred)
            all_targets.append(target)
            if pred == target:
                correct += 1
                
    accuracy = correct / float(total)
    return accuracy, all_preds, all_targets

