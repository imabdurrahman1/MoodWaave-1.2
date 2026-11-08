"""
experiments/rev2_p5_deep/03_train_small_cnn.py
Trains SmallMelCNN from scratch on CPU using the P1 fixed Kereliuk-Sturm split:
- Minimal tuning across 4 configurations (lr {1e-3, 3e-4} x dropout {0.2, 0.5}) on valid_tracks
- Multi-seed evaluation (seeds 42, 43, 44) on test_tracks with 10-crop logit averaging
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common
import small_cnn

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_y)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)
        
    return total_loss / total, correct / total

def run_training(
    train_dataset,
    val_dataset,
    lr: float,
    dropout: float,
    epochs: int = 15,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
):
    set_seed(seed)
    model = small_cnn.SmallMelCNN(num_classes=10, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _, _ = small_cnn.evaluate_cnn(model, val_dataset, device)
        
        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_acc"].append(float(val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    return model, best_state, best_val_acc, history

def main():
    out_dir = common.BASE_DIR
    cache_dir = os.path.join(out_dir, "mel_cache")
    os.makedirs(cache_dir, exist_ok=True)
    device = torch.device("cpu")
    
    with open(common.SPLITS_JSON, "r", encoding="utf-8") as f:
        ks_split = json.load(f)["gtzan"]["kereliuk_sturm_fixed"]
        
    df_labels = pd.read_csv(common.GTZAN_FEATURES_CSV).set_index("filename")
    labels_map = df_labels["label"].to_dict()
    
    tr_ids = ks_split["train_tracks"]
    val_ids = ks_split["valid_tracks"]
    te_ids = ks_split["test_tracks"]
    
    print(f"Creating datasets (Train: {len(tr_ids)}, Val: {len(val_ids)}, Test: {len(te_ids)})...")
    train_ds = small_cnn.GTZANMelDataset(tr_ids, labels_map, cache_dir=cache_dir, is_training=True)
    val_ds = small_cnn.GTZANMelDataset(val_ids, labels_map, cache_dir=cache_dir, is_training=False)
    test_ds = small_cnn.GTZANMelDataset(te_ids, labels_map, cache_dir=cache_dir, is_training=False)
    
    # Pre-warm mel cache to ensure fast epochs
    print("Pre-computing log-mel spectrograms into cache...")
    for ds in [train_ds, val_ds, test_ds]:
        for i in range(len(ds)):
            _ = ds[i]
    print(f"Cache ready at {cache_dir}")
    
    # 1. Tuning Grid (4 configs on valid_tracks)
    grid = [
        {"lr": 1e-3, "dropout": 0.2},
        {"lr": 1e-3, "dropout": 0.5},
        {"lr": 3e-4, "dropout": 0.2},
        {"lr": 3e-4, "dropout": 0.5},
    ]
    
    print("\n=== Minimal Tuning: 4 Configurations on Validation Split (15 Epochs) ===")
    tuning_records = []
    best_config = None
    best_val_score = -1.0
    
    for cfg in grid:
        t0 = time.time()
        _, _, val_acc, hist = run_training(
            train_ds, val_ds, lr=cfg["lr"], dropout=cfg["dropout"], epochs=15, seed=42, device=device
        )
        dt = time.time() - t0
        print(f"  Config lr={cfg['lr']:.0e}, dropout={cfg['dropout']}: Best Val Acc={val_acc:.4f} in {dt:.1f}s")
        tuning_records.append({
            "lr": cfg["lr"],
            "dropout": cfg["dropout"],
            "val_accuracy": float(val_acc),
            "tune_time_sec": round(dt, 1),
        })
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_config = cfg
            
    df_tuning = pd.DataFrame(tuning_records)
    df_tuning.to_csv(os.path.join(out_dir, "cnn_tuning_grid.csv"), index=False)
    print(f"\nBest Config Selected: lr={best_config['lr']}, dropout={best_config['dropout']} (Val Acc: {best_val_score:.4f})")
    
    # 2. Multi-Seed Training (Seeds 42, 43, 44)
    seeds = [42, 43, 44]
    seed_records = []
    all_histories = {}
    
    print("\n=== Multi-Seed Training on Fixed Split (Seeds: 42, 43, 44) ===")
    for seed in seeds:
        t0 = time.time()
        model, best_state, val_acc, hist = run_training(
            train_ds, val_ds, lr=best_config["lr"], dropout=best_config["dropout"], epochs=15, seed=seed, device=device
        )
        train_time = time.time() - t0
        all_histories[f"seed_{seed}"] = hist
        
        # Load best validation checkpoint
        model.load_state_dict(best_state)
        
        # Evaluate on test set (290 tracks, 10-crop logit averaging)
        test_acc, preds, targets = small_cnn.evaluate_cnn(model, test_ds, device)
        test_f1 = f1_score(targets, preds, average="macro")
        
        print(f"  Seed {seed}: Test Accuracy = {test_acc:.4f}, Macro-F1 = {test_f1:.4f} (Train Time: {train_time:.1f}s)")
        seed_records.append({
            "seed": seed,
            "lr": best_config["lr"],
            "dropout": best_config["dropout"],
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_f1),
            "train_time_sec": round(train_time, 1),
        })
        
        # Save model checkpoint
        ckpt_path = os.path.join(out_dir, f"small_cnn_seed_{seed}.pt")
        torch.save(best_state, ckpt_path)
        
    with open(os.path.join(out_dir, "cnn_training_history.json"), "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)
        
    df_seeds = pd.DataFrame(seed_records)
    df_seeds.to_csv(os.path.join(out_dir, "cnn_seeds_results.csv"), index=False)
    
    acc_mean = df_seeds["test_accuracy"].mean()
    acc_std = df_seeds["test_accuracy"].std(ddof=1)
    f1_mean = df_seeds["test_macro_f1"].mean()
    f1_std = df_seeds["test_macro_f1"].std(ddof=1)
    
    print(f"\nFinal SmallMelCNN Results across 3 Seeds on 290 Test Tracks:")
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Macro-F1: {f1_mean:.4f} ± {f1_std:.4f}")

if __name__ == "__main__":
    main()

