"""
experiments/rev2_p4_baselines/03_profile_models.py
Measures serialized model sizes and single-thread CPU inference latencies.
Protocol (Ground Rule 6):
- Single thread: OMP/MKL/OPENBLAS/NUMBA threads = 1
- Evaluated on 200 tracks (100 GTZAN, 100 PMEmo)
- 1 warm-up track excluded
- Median of 3 repeats per track
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common

# Enforce single-thread execution for latency measurement per Ground Rule 6
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

def get_modal_params(df_folds: pd.DataFrame, task_name: str, model_name: str, feat_set: str):
    """Extracts the most frequent hyperparameter configuration from outer folds."""
    sub = df_folds[
        (df_folds["task"] == task_name) &
        (df_folds["model"] == model_name) &
        (df_folds["feature_set"] == feat_set)
    ]
    if len(sub) == 0:
        return {}
    
    # Each row has a JSON string in 'best_params'
    param_counts = Counter(sub["best_params"])
    most_common_json = param_counts.most_common(1)[0][0]
    return json.loads(most_common_json)

def measure_track_latency(model, X_sample: np.ndarray, num_repeats: int = 3):
    """Measures median latency across num_repeats for a single track vector (1, D)."""
    repeats_ms = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        _ = model.predict(X_sample)
        t1 = time.perf_counter()
        repeats_ms.append((t1 - t0) * 1000.0)
    return float(np.median(repeats_ms))

def benchmark_inference(model, X_data: np.ndarray, num_tracks: int = 100):
    """
    Benchmarks single-track inference latency on num_tracks.
    1 warm-up track excluded, median of 3 repeats per track.
    Returns median, mean, and std of track latencies in ms.
    """
    assert len(X_data) >= num_tracks + 1, f"Not enough tracks: {len(X_data)} < {num_tracks + 1}"
    
    # Warm-up track (track 0) - excluded from timing statistics
    X_warmup = X_data[0:1]
    _ = measure_track_latency(model, X_warmup, num_repeats=3)
    
    track_latencies_ms = []
    for i in range(1, num_tracks + 1):
        X_curr = X_data[i:i+1]
        lat = measure_track_latency(model, X_curr, num_repeats=3)
        track_latencies_ms.append(lat)
        
    return {
        "median_ms": float(np.median(track_latencies_ms)),
        "mean_ms": float(np.mean(track_latencies_ms)),
        "std_ms": float(np.std(track_latencies_ms, ddof=1)),
        "min_ms": float(np.min(track_latencies_ms)),
        "max_ms": float(np.max(track_latencies_ms)),
    }

def main():
    out_dir = common.BASE_DIR
    models_dir = os.path.join(out_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    
    gtzan_folds_path = os.path.join(out_dir, "gtzan_baseline_folds.csv")
    pmemo_folds_path = os.path.join(out_dir, "pmemo_baseline_folds.csv")
    
    if not os.path.exists(gtzan_folds_path) or not os.path.exists(pmemo_folds_path):
        print("Fold results not yet complete. Please ensure 01 and 02 have finished.")
        return
        
    df_gtzan_folds = pd.read_csv(gtzan_folds_path)
    df_pmemo_folds = pd.read_csv(pmemo_folds_path)
    
    # Load GTZAN clean dataset
    gtzan_csv = os.path.join(common.PROJECT_ROOT, "data", "gtzan_features.csv")
    rem_csv = os.path.join(common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "removed_tracks.csv")
    df_g = pd.read_csv(gtzan_csv).set_index("filename")
    rem_tracks = set(pd.read_csv(rem_csv)["filename"])
    df_g_clean = df_g.loc[~df_g.index.isin(rem_tracks)]
    y_g = df_g_clean["label"].values
    
    # Load PMEmo dataset
    pmemo_csv = os.path.join(common.PROJECT_ROOT, "data", "pmemo_ready.csv")
    df_p = pd.read_csv(pmemo_csv)
    y_val = df_p["Valence(mean)"].values
    y_aro = df_p["Arousal(mean)"].values
    
    genre_models = ["LightGBM", "Random Forest", "XGBoost", "SVM-RBF", "Logistic Regression"]
    va_models = ["LightGBM", "Random Forest", "XGBoost", "SVR-RBF", "Ridge"]
    feat_sets = {
        "Full-113": common.FULL_113_FEATURES,
        "Lite-71": common.LITE_71_FEATURES,
    }
    
    profiles = []
    
    # 1. Profile Genre Models
    print("=== Profiling Genre Models (GTZAN, 100 tracks latency) ===")
    for feat_set_name, feats in feat_sets.items():
        X_g = df_g_clean[feats].values
        
        for model_name in genre_models:
            modal_params = get_modal_params(df_gtzan_folds, "genre", model_name, feat_set_name)
            print(f"Profiling {model_name} ({feat_set_name}) with params: {modal_params}")
            
            # Instantiate model
            model = common.get_genre_base_model(model_name, seed=42)
            model.set_params(**modal_params)
            
            # Fit on full clean data
            if model_name == "XGBoost":
                le = LabelEncoder()
                y_fit = le.fit_transform(y_g)
            else:
                y_fit = y_g
                
            model.fit(X_g, y_fit)
            
            # Measure serialized size
            model_bytes = pickle.dumps(model)
            size_kb = len(model_bytes) / 1024.0
            size_mb = size_kb / 1024.0
            
            # Save artifact
            save_name = f"genre_{model_name.lower().replace(' ', '_').replace('-', '_')}_{feat_set_name.lower().replace('-', '_')}.pkl"
            save_path = os.path.join(models_dir, save_name)
            with open(save_path, "wb") as f:
                f.write(model_bytes)
                
            # Benchmark latency on 100 tracks
            timing = benchmark_inference(model, X_g, num_tracks=100)
            print(f"  -> Size: {size_kb:.2f} KB | Latency: {timing['median_ms']:.4f} ms/track")
            
            profiles.append({
                "dataset": "GTZAN",
                "task": "genre",
                "model": model_name,
                "feature_set": feat_set_name,
                "serialized_size_kb": round(size_kb, 2),
                "serialized_size_mb": round(size_mb, 4),
                "latency_median_ms": round(timing["median_ms"], 4),
                "latency_mean_ms": round(timing["mean_ms"], 4),
                "latency_std_ms": round(timing["std_ms"], 4),
                "latency_min_ms": round(timing["min_ms"], 4),
                "latency_max_ms": round(timing["max_ms"], 4),
                "modal_params": json.dumps(modal_params),
                "artifact_file": save_name,
            })
            
    # 2. Profile Valence and Arousal Models
    print("\n=== Profiling Valence and Arousal Models (PMEmo, 100 tracks latency) ===")
    for task_name, y_target in [("valence", y_val), ("arousal", y_aro)]:
        for feat_set_name, feats in feat_sets.items():
            X_p = df_p[feats].values
            
            for model_name in va_models:
                modal_params = get_modal_params(df_pmemo_folds, task_name, model_name, feat_set_name)
                print(f"Profiling {task_name.capitalize()} - {model_name} ({feat_set_name}) with params: {modal_params}")
                
                model = common.get_va_base_model(model_name, seed=42)
                model.set_params(**modal_params)
                model.fit(X_p, y_target)
                
                model_bytes = pickle.dumps(model)
                size_kb = len(model_bytes) / 1024.0
                size_mb = size_kb / 1024.0
                
                save_name = f"{task_name}_{model_name.lower().replace(' ', '_').replace('-', '_')}_{feat_set_name.lower().replace('-', '_')}.pkl"
                save_path = os.path.join(models_dir, save_name)
                with open(save_path, "wb") as f:
                    f.write(model_bytes)
                    
                timing = benchmark_inference(model, X_p, num_tracks=100)
                print(f"  -> Size: {size_kb:.2f} KB | Latency: {timing['median_ms']:.4f} ms/track")
                
                profiles.append({
                    "dataset": "PMEmo",
                    "task": task_name,
                    "model": model_name,
                    "feature_set": feat_set_name,
                    "serialized_size_kb": round(size_kb, 2),
                    "serialized_size_mb": round(size_mb, 4),
                    "latency_median_ms": round(timing["median_ms"], 4),
                    "latency_mean_ms": round(timing["mean_ms"], 4),
                    "latency_std_ms": round(timing["std_ms"], 4),
                    "latency_min_ms": round(timing["min_ms"], 4),
                    "latency_max_ms": round(timing["max_ms"], 4),
                    "modal_params": json.dumps(modal_params),
                    "artifact_file": save_name,
                })
                
    out_csv = os.path.join(out_dir, "model_profiles.csv")
    df_prof = pd.DataFrame(profiles)
    df_prof.to_csv(out_csv, index=False)
    print(f"\nModel profiling complete! Saved {len(df_prof)} profiles to {out_csv}")

if __name__ == "__main__":
    main()

