"""
experiments/rev3_a_stats/04_profile_models.py
Measures serialized size (KB) and single-thread CPU inference latency (ms/track)
for each outer fold's selected hyperparameter configuration.
Computes Mean ± SD over outer folds.

Timing protocol (Ground Rule 5 & 6):
- Single thread: OMP/MKL/OPENBLAS/NUMBA threads = 1
- 1 warm-up track excluded, median of 3 repeats per track, 100 tracks per dataset.
"""

import os
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[var] = "1"

import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3a_common as common

OUT_DIR = common.BASE_DIR
SPLITS_JSON = os.path.join(
    common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "splits.json"
)


def measure_track_latency(model, X_sample: np.ndarray, num_repeats: int = 3):
    repeats_ms = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        _ = model.predict(X_sample)
        t1 = time.perf_counter()
        repeats_ms.append((t1 - t0) * 1000.0)
    return float(np.median(repeats_ms))


def benchmark_inference(model, X_data: np.ndarray, num_tracks: int = 100):
    assert len(X_data) >= num_tracks + 1, f"Not enough tracks: {len(X_data)} < {num_tracks + 1}"
    
    # Warm-up track (track 0) - excluded from timing statistics
    X_warmup = X_data[0:1]
    _ = measure_track_latency(model, X_warmup, num_repeats=3)
    
    track_latencies_ms = []
    for i in range(1, num_tracks + 1):
        X_curr = X_data[i:i+1]
        lat = measure_track_latency(model, X_curr, num_repeats=3)
        track_latencies_ms.append(lat)
        
    return float(np.median(track_latencies_ms))


def profile_gtzan():
    print("Profiling GTZAN folds (size & single-thread latency)...", flush=True)
    df_clean, all_folds = common.load_gtzan(SPLITS_JSON)
    gtzan_folds_csv = os.path.join(OUT_DIR, "gtzan_r3a_folds.csv")
    df_folds = pd.read_csv(gtzan_folds_csv)

    models = ["LightGBM", "Random Forest", "XGBoost", "SVM-RBF", "Logistic Regression"]
    eval_df = df_clean.iloc[:101]

    records = []
    for fi in all_folds:
        repeat = fi["repeat"]
        fold_idx = fi["fold"]
        train_df = df_clean.loc[fi["train_tracks"]]
        y_train = train_df["label"].values

        for feat_name, feats in common.FEATURE_SETS.items():
            X_tr = train_df[feats].values
            X_eval = eval_df[feats].values

            for model_name in models:
                row = df_folds[
                    (df_folds["repeat"] == repeat) &
                    (df_folds["fold"] == fold_idx) &
                    (df_folds["model"] == model_name) &
                    (df_folds["feature_set"] == feat_name)
                ].iloc[0]
                best_params = json.loads(row["best_params"])

                model = common.get_genre_base_model(model_name, seed=42)
                model.set_params(**best_params)

                if model_name == "XGBoost":
                    le = LabelEncoder()
                    y_tr_fit = le.fit_transform(y_train)
                else:
                    y_tr_fit = y_train

                model.fit(X_tr, y_tr_fit)

                size_kb = len(pickle.dumps(model)) / 1024.0
                lat_ms = benchmark_inference(model, X_eval, num_tracks=100)

                records.append({
                    "dataset": "GTZAN",
                    "task": "genre",
                    "repeat": repeat,
                    "fold": fold_idx,
                    "model": model_name,
                    "feature_set": feat_name,
                    "size_kb": float(size_kb),
                    "latency_ms": float(lat_ms),
                    "best_params": row["best_params"],
                })

    df_res = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "gtzan_profiles_per_fold.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"Saved GTZAN profiles to {out_csv}", flush=True)
    return df_res


def profile_pmemo():
    print("Profiling PMEmo folds (size & single-thread latency)...", flush=True)
    df_pmemo, all_folds = common.load_pmemo(SPLITS_JSON)
    pmemo_folds_csv = os.path.join(OUT_DIR, "pmemo_r3a_folds.csv")
    df_folds = pd.read_csv(pmemo_folds_csv)

    targets = [("valence", "Valence(mean)"), ("arousal", "Arousal(mean)")]
    models = ["LightGBM", "Random Forest", "XGBoost", "SVR-RBF", "Ridge"]
    eval_df = df_pmemo.iloc[:101]

    records = []
    for fi in all_folds:
        repeat = fi["repeat"]
        fold_idx = fi["fold"]
        train_df = df_pmemo.loc[fi["train_tracks"]]

        for task_name, target_col in targets:
            y_train = train_df[target_col].values

            for feat_name, feats in common.FEATURE_SETS.items():
                X_tr = train_df[feats].values
                X_eval = eval_df[feats].values

                for model_name in models:
                    row = df_folds[
                        (df_folds["task"] == task_name) &
                        (df_folds["repeat"] == repeat) &
                        (df_folds["fold"] == fold_idx) &
                        (df_folds["model"] == model_name) &
                        (df_folds["feature_set"] == feat_name)
                    ].iloc[0]
                    best_params = json.loads(row["best_params"])

                    model = common.get_va_base_model(model_name, seed=42)
                    model.set_params(**best_params)
                    model.fit(X_tr, y_train)

                    size_kb = len(pickle.dumps(model)) / 1024.0
                    lat_ms = benchmark_inference(model, X_eval, num_tracks=100)

                    records.append({
                        "dataset": "PMEmo",
                        "task": task_name,
                        "repeat": repeat,
                        "fold": fold_idx,
                        "model": model_name,
                        "feature_set": feat_name,
                        "size_kb": float(size_kb),
                        "latency_ms": float(lat_ms),
                        "best_params": row["best_params"],
                    })

    df_res = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "pmemo_profiles_per_fold.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"Saved PMEmo profiles to {out_csv}", flush=True)
    return df_res


def main():
    df_g = profile_gtzan()
    df_p = profile_pmemo()

    # Summarize across folds
    summ_records = []
    for (model, fs), grp in df_g.groupby(["model", "feature_set"]):
        summ_records.append({
            "dataset": "GTZAN",
            "task": "genre",
            "model": model,
            "feature_set": fs,
            "size_kb_mean": float(grp["size_kb"].mean()),
            "size_kb_std": float(grp["size_kb"].std(ddof=1)),
            "latency_ms_mean": float(grp["latency_ms"].mean()),
            "latency_ms_std": float(grp["latency_ms"].std(ddof=1)),
        })

    for (task, model, fs), grp in df_p.groupby(["task", "model", "feature_set"]):
        summ_records.append({
            "dataset": "PMEmo",
            "task": task,
            "model": model,
            "feature_set": fs,
            "size_kb_mean": float(grp["size_kb"].mean()),
            "size_kb_std": float(grp["size_kb"].std(ddof=1)),
            "latency_ms_mean": float(grp["latency_ms"].mean()),
            "latency_ms_std": float(grp["latency_ms"].std(ddof=1)),
        })

    df_summary = pd.DataFrame(summ_records)
    out_summary = os.path.join(OUT_DIR, "model_profiles_summary.csv")
    df_summary.to_csv(out_summary, index=False)
    print(f"Saved summary profiles to {out_summary}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()

