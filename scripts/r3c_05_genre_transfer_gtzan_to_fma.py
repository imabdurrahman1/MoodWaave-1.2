import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb

from r3c_common import (
    HERE,
    PROJECT_ROOT,
    SEED,
    LITE_FEATURES,
    FULL_FEATURES,
)

FMA_METADATA_CSV = os.path.join(HERE, "fma_800_metadata.csv")
FMA_FEATURES_CSV = os.path.join(HERE, "fma_800_features.csv")
GTZAN_CSV = os.path.join(PROJECT_ROOT, "data", "gtzan_features.csv")
SPLITS_JSON = os.path.join(PROJECT_ROOT, "experiments", "rev2_p1_splits", "splits.json")
OUT_CSV = os.path.join(HERE, "gtzan_to_fma_transfer.csv")
OUT_CM_CSV = os.path.join(HERE, "gtzan_to_fma_cm.csv")

def get_gtzan_clean_data():
    """Load GTZAN clean 929 tracks without faulty excerpts."""
    df_gtzan = pd.read_csv(GTZAN_CSV)
    with open(SPLITS_JSON, "r") as f:
        s = json.load(f)
    
    # All clean tracks from artist-grouped folds
    clean_tracks = set()
    for fold in s["gtzan"]["fault_removed_artist_grouped"]:
        clean_tracks.update(fold["train_tracks"])
        clean_tracks.update(fold["test_tracks"])
        
    df_clean = df_gtzan[df_gtzan["filename"].isin(clean_tracks)].copy()
    print(f"Loaded GTZAN clean tracks: {len(df_clean)} / {len(df_gtzan)}")
    return df_clean

def run_cross_dataset_transfer():
    print("=== Step 5: Cross-Dataset Genre Transfer (GTZAN -> FMA-small) ===")
    t_start = time.perf_counter()
    
    # 1. Load GTZAN training data
    df_gtzan = get_gtzan_clean_data()
    y_gtzan = df_gtzan["label"].values
    
    # 2. Load FMA data
    df_fma_meta = pd.read_csv(FMA_METADATA_CSV)
    df_fma_feats = pd.read_csv(FMA_FEATURES_CSV).set_index("track_id").loc[df_fma_meta["track_id"]].reset_index()
    
    # 3. Filter FMA tracks to overlapping 3 genres: Hip-Hop, Pop, Rock
    fma_genre_map = {
        "Hip-Hop": "hiphop",
        "Pop": "pop",
        "Rock": "rock"
    }
    
    mask_3genres = df_fma_meta["genre"].isin(fma_genre_map.keys())
    df_fma_meta_3 = df_fma_meta[mask_3genres].copy()
    df_fma_feats_3 = df_fma_feats[mask_3genres].copy()
    
    y_fma_mapped = df_fma_meta_3["genre"].map(fma_genre_map).values
    print(f"Target FMA evaluation tracks (3 genres): {len(y_fma_mapped)}")
    print(pd.Series(y_fma_mapped).value_counts().to_dict())
    
    # Target 3 classes
    target_classes = ["hiphop", "pop", "rock"]
    
    # Model configs to test
    configs = [
        ("MOODWAAVE-Lite-71", LITE_FEATURES),
        ("Full-113 (V0)", FULL_FEATURES),
    ]
    
    lgbm_params = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": 1,
    }
    
    summary_results = []
    
    for cfg_name, feat_cols in configs:
        print(f"\n--- Evaluating {cfg_name} ---")
        X_gtzan = df_gtzan[feat_cols].values
        X_fma = df_fma_feats_3[feat_cols].values
        
        clf = lgb.LGBMClassifier(**lgbm_params)
        clf.fit(X_gtzan, y_gtzan)
        
        # Check classes
        clf_classes = list(clf.classes_)
        class_indices = [clf_classes.index(c) for c in target_classes]
        
        # Predict probabilities on FMA
        probs_all = clf.predict_proba(X_fma) # shape: (N, 10)
        probs_3 = probs_all[:, class_indices] # shape: (N, 3)
        
        # Renormalize across the 3 target classes
        probs_3_norm = probs_3 / (np.sum(probs_3, axis=1, keepdims=True) + 1e-12)
        
        # Predicted labels
        pred_idx = np.argmax(probs_3_norm, axis=1)
        preds = np.array([target_classes[idx] for idx in pred_idx])
        
        acc = accuracy_score(y_fma_mapped, preds)
        macro_f1 = f1_score(y_fma_mapped, preds, average="macro")
        
        # Per class metrics
        f1_per_class = f1_score(y_fma_mapped, preds, average=None, labels=target_classes)
        cm = confusion_matrix(y_fma_mapped, preds, labels=target_classes)
        
        print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%) vs 33.33% chance")
        print(f"Macro-F1: {macro_f1:.4f} vs 0.3333 chance")
        for cls_name, f1_val in zip(target_classes, f1_per_class):
            print(f"  Class '{cls_name}' F1: {f1_val:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=target_classes, columns=target_classes))
        
        summary_results.append({
            "model": cfg_name,
            "n_train_gtzan": len(df_gtzan),
            "n_test_fma": len(y_fma_mapped),
            "accuracy": acc,
            "accuracy_gain_over_chance": acc - (1.0 / 3.0),
            "macro_f1": macro_f1,
            "macro_f1_gain_over_chance": macro_f1 - (1.0 / 3.0),
            "f1_hiphop": f1_per_class[0],
            "f1_pop": f1_per_class[1],
            "f1_rock": f1_per_class[2],
            "cm_json": json.dumps(cm.tolist()),
        })
        
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(OUT_CSV, index=False)
    print(f"\nSaved cross-dataset transfer results to {OUT_CSV}")
    print(df_summary[["model", "accuracy", "accuracy_gain_over_chance", "macro_f1", "f1_hiphop", "f1_pop", "f1_rock"]].to_string(index=False))
    
    print(f"\nStep 5 completed in {(time.perf_counter() - t_start):.2f}s.")
    return df_summary

if __name__ == "__main__":
    run_cross_dataset_transfer()

