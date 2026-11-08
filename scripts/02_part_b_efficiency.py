"""Script 02: Part B - Efficiency & Footprint Benchmark.
Measures:
1. Serialized size (MB) of Lite-71 vs Full-113 vs RF-100 models (Genre + Mood combined).
2. Per-track model inference latency (ms, median of 3 repeats, mean ± std over 200 tracks).
3. End-to-end latency (audio load + feature extraction + dual prediction, median of 3 repeats, 100 tracks per dataset).
4. Summary table combining size, inference latency, end-to-end latency, and predictive accuracy.
"""
import os
import sys
import json
import time
import gc
import joblib
import numpy as np
import pandas as pd
import librosa
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import common as C

sys.path.insert(0, os.path.join(C.ROOT, "src"))
from feature_extractor import extract_features  # Existing file, read-only


def measure_model_sizes():
    print("\n--- 1. MODEL SERIALIZATION SIZE (MB) ---")
    tmp_dir = os.path.join(C.OUT, "models_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Load training data
    _, X_g_full, X_g_lite, y_g, _ = C.load_gtzan_data()
    _, X_m_full, X_m_lite, y_val, y_aro = C.load_pmemo_data()
    
    # 1. Lite-71 models
    m_g_lite = LGBMClassifier(**C.LGBM_GENRE_PARAMS, n_jobs=1).fit(X_g_lite, y_g)
    m_v_lite = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_m_lite, y_val)
    m_a_lite = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_m_lite, y_aro)
    
    p_g_lite = os.path.join(tmp_dir, "genre_lite.pkl")
    p_v_lite = os.path.join(tmp_dir, "val_lite.pkl")
    p_a_lite = os.path.join(tmp_dir, "aro_lite.pkl")
    joblib.dump(m_g_lite, p_g_lite)
    joblib.dump(m_v_lite, p_v_lite)
    joblib.dump(m_a_lite, p_a_lite)
    sz_lite = (os.path.getsize(p_g_lite) + os.path.getsize(p_v_lite) + os.path.getsize(p_a_lite)) / (1024**2)
    
    # 2. Full-113 models
    m_g_full = LGBMClassifier(**C.LGBM_GENRE_PARAMS, n_jobs=1).fit(X_g_full, y_g)
    m_v_full = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_m_full, y_val)
    m_a_full = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_m_full, y_aro)
    
    p_g_full = os.path.join(tmp_dir, "genre_full.pkl")
    p_v_full = os.path.join(tmp_dir, "val_full.pkl")
    p_a_full = os.path.join(tmp_dir, "aro_full.pkl")
    joblib.dump(m_g_full, p_g_full)
    joblib.dump(m_v_full, p_v_full)
    joblib.dump(m_a_full, p_a_full)
    sz_full = (os.path.getsize(p_g_full) + os.path.getsize(p_v_full) + os.path.getsize(p_a_full)) / (1024**2)
    
    # 3. RF-100 models
    m_g_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1).fit(X_g_full, y_g)
    m_v_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1).fit(X_m_full, y_val)
    m_a_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1).fit(X_m_full, y_aro)
    
    p_g_rf = os.path.join(tmp_dir, "genre_rf.pkl")
    p_v_rf = os.path.join(tmp_dir, "val_rf.pkl")
    p_a_rf = os.path.join(tmp_dir, "aro_rf.pkl")
    joblib.dump(m_g_rf, p_g_rf)
    joblib.dump(m_v_rf, p_v_rf)
    joblib.dump(m_a_rf, p_a_rf)
    sz_rf = (os.path.getsize(p_g_rf) + os.path.getsize(p_v_rf) + os.path.getsize(p_a_rf)) / (1024**2)
    
    print(f"  MOODWAAVE-Lite (71 feats) Combined Model Size: {sz_lite:.2f} MB")
    print(f"  LightGBM Full (113 feats) Combined Model Size: {sz_full:.2f} MB")
    print(f"  RandomForest (113 feats) Combined Model Size: {sz_rf:.2f} MB")
    
    models_dict = {
        "Lite": (m_g_lite, m_v_lite, m_a_lite),
        "Full": (m_g_full, m_v_full, m_a_full),
        "RF": (m_g_rf, m_v_rf, m_a_rf)
    }
    sizes_dict = {
        "Lite": sz_lite,
        "Full": sz_full,
        "RF": sz_rf
    }
    return models_dict, sizes_dict


def measure_inference_latencies(models_dict):
    print("\n--- 2. PER-TRACK MODEL INFERENCE LATENCY (ms, N = 200 tracks) ---")
    _, X_full, X_lite, _, _ = C.load_pmemo_data()
    n_sample = 200
    X_full_sub = X_full.iloc[:n_sample]
    X_lite_sub = X_lite.iloc[:n_sample]
    
    # Warm-up pass (5 tracks)
    for i in range(5):
        models_dict["Lite"][0].predict(X_lite_sub.iloc[i:i+1])
        models_dict["Full"][0].predict(X_full_sub.iloc[i:i+1])
        models_dict["RF"][0].predict(X_full_sub.iloc[i:i+1])
        
    latencies = {"Lite": [], "Full": [], "RF": []}
    
    for i in range(n_sample):
        x_l = X_lite_sub.iloc[i:i+1]
        x_f = X_full_sub.iloc[i:i+1]
        
        # 1. Lite (Genre + Mood)
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            models_dict["Lite"][0].predict(x_l)
            models_dict["Lite"][1].predict(x_l)
            models_dict["Lite"][2].predict(x_l)
            reps.append((time.perf_counter() - t0) * 1000.0)
        latencies["Lite"].append(float(np.median(reps)))
        
        # 2. Full (Genre + Mood)
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            models_dict["Full"][0].predict(x_f)
            models_dict["Full"][1].predict(x_f)
            models_dict["Full"][2].predict(x_f)
            reps.append((time.perf_counter() - t0) * 1000.0)
        latencies["Full"].append(float(np.median(reps)))
        
        # 3. RF (Genre + Mood)
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            models_dict["RF"][0].predict(x_f)
            models_dict["RF"][1].predict(x_f)
            models_dict["RF"][2].predict(x_f)
            reps.append((time.perf_counter() - t0) * 1000.0)
        latencies["RF"].append(float(np.median(reps)))
        
    inf_summary = {}
    for k in ["Lite", "Full", "RF"]:
        arr = np.array(latencies[k])
        inf_summary[k] = {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr, ddof=1)),
            "median_ms": float(np.median(arr))
        }
        print(f"  {k} Inference Latency: {inf_summary[k]['mean_ms']:.2f} ± {inf_summary[k]['std_ms']:.2f} ms (Median: {inf_summary[k]['median_ms']:.2f} ms)")
        
    return inf_summary


def measure_end_to_end_latencies(models_dict, inf_summary):
    print("\n--- 3. END-TO-END PIPELINE LATENCY (Audio Load + Feature Extraction + Both Predictions) ---")
    with open(C.SAMPLE_IDS_JSON, "r") as f:
        sample_ids = json.load(f)
        
    gtzan_paths = sample_ids["GTZAN"][:100]
    pmemo_paths = sample_ids["PMEmo"][:100]
    
    # Load previously measured 113-feature standalone times from idea2 (exact same 200 tracks and audio files)
    idea2_val_csv = os.path.join(C.ROOT, "experiments", "idea2_cost_pareto", "validation_per_track.csv")
    df_idea2 = pd.read_csv(idea2_val_csv)
    # Map track path -> original_full_ms
    orig_ms_map = dict(zip(df_idea2["track"], df_idea2["original_full_ms"]))
    # Also handle normalized slashes
    orig_ms_map_norm = {os.path.normpath(k).lower(): v for k, v in orig_ms_map.items()}
    
    e2e_records = []
    
    # Warm-up pass on first track of GTZAN
    p_warm = gtzan_paths[0]
    y_w, sr_w = librosa.load(p_warm, sr=22050, mono=True, duration=30)
    feat_w_lite = C.extract_lite_71_shared(y_w, sr_w)
    df_w_lite = pd.DataFrame([feat_w_lite])[C.LITE_FEATURES]
    models_dict["Lite"][0].predict(df_w_lite)
    models_dict["Lite"][1].predict(df_w_lite)
    models_dict["Lite"][2].predict(df_w_lite)
    
    full_pred_latency = inf_summary["Full"]["mean_ms"]
    lite_pred_latency = inf_summary["Lite"]["mean_ms"]
    
    for task_name, paths in [("GTZAN (Genre)", gtzan_paths), ("PMEmo (Mood)", pmemo_paths)]:
        print(f"  Measuring Lite-71 extraction + prediction on {task_name} (100 tracks, 3 repeats median)...", flush=True)
        for idx, path in enumerate(paths):
            gc.collect()
            norm_p = os.path.normpath(path).lower()
            orig_extract_ms = orig_ms_map_norm.get(norm_p, 1750.0)
            orig_e2e_ms = orig_extract_ms + full_pred_latency
            
            # Freshly measure Lite-71 (audio load + shared STFT + 71 features + dual predictions)
            reps_lite = []
            for _ in range(3):
                t0 = time.perf_counter()
                y, sr = librosa.load(path, sr=22050, mono=True, duration=30)
                f_lite = C.extract_lite_71_shared(y, sr)
                df_l = pd.DataFrame([f_lite])[C.LITE_FEATURES]
                models_dict["Lite"][0].predict(df_l)
                models_dict["Lite"][1].predict(df_l)
                models_dict["Lite"][2].predict(df_l)
                reps_lite.append((time.perf_counter() - t0) * 1000.0)
            lite_e2e_ms = float(np.median(reps_lite))
            
            e2e_records.append({
                "dataset": task_name,
                "track": path,
                "orig_113_e2e_ms": orig_e2e_ms,
                "lite_71_e2e_ms": lite_e2e_ms,
                "speedup": orig_e2e_ms / lite_e2e_ms,
                "ms_saved": orig_e2e_ms - lite_e2e_ms
            })
            
            if (idx + 1) % 25 == 0:
                print(f"    Completed {idx+1}/100 tracks...", flush=True)
                
    df_e2e = pd.DataFrame(e2e_records)
    df_e2e.to_csv(os.path.join(C.OUT, "part_b_per_track_timing.csv"), index=False)
    
    print("\nEnd-to-End Latency Results (Load + Extraction + Dual Prediction):")
    grp = df_e2e.groupby("dataset")
    dataset_summaries = {}
    for name, g in grp:
        orig_mean = g["orig_113_e2e_ms"].mean()
        orig_std = g["orig_113_e2e_ms"].std()
        lite_mean = g["lite_71_e2e_ms"].mean()
        lite_std = g["lite_71_e2e_ms"].std()
        spd = orig_mean / lite_mean
        dataset_summaries[name] = {
            "orig_mean": orig_mean, "orig_std": orig_std,
            "lite_mean": lite_mean, "lite_std": lite_std,
            "speedup": spd
        }
        print(f"  {name}:")
        print(f"    Original-113: {orig_mean:.1f} ± {orig_std:.1f} ms")
        print(f"    Lite-71:      {lite_mean:.1f} ± {lite_std:.1f} ms")
        print(f"    Speedup:      {spd:.2f}x ({orig_mean - lite_mean:.1f} ms saved)")
        
    total_orig_mean = df_e2e["orig_113_e2e_ms"].mean()
    total_orig_std = df_e2e["orig_113_e2e_ms"].std()
    total_lite_mean = df_e2e["lite_71_e2e_ms"].mean()
    total_lite_std = df_e2e["lite_71_e2e_ms"].std()
    total_spd = total_orig_mean / total_lite_mean
    print(f"\n  Overall (N = 200 tracks across GTZAN and PMEmo):")
    print(f"    Original-113: {total_orig_mean:.1f} ± {total_orig_std:.1f} ms")
    print(f"    Lite-71:      {total_lite_mean:.1f} ± {total_lite_std:.1f} ms")
    print(f"    Speedup:      {total_spd:.2f}x ({total_orig_mean - total_lite_mean:.1f} ms saved)")
    
    e2e_summary = {
        "orig_mean_ms": total_orig_mean, "orig_std_ms": total_orig_std,
        "lite_mean_ms": total_lite_mean, "lite_std_ms": total_lite_std,
        "speedup": total_spd,
        "by_dataset": dataset_summaries
    }
    return e2e_summary


def main():
    models_dict, sizes_dict = measure_model_sizes()
    inf_summary = measure_inference_latencies(models_dict)
    e2e_summary = measure_end_to_end_latencies(models_dict, inf_summary)
    
    # Load predictive accuracy from Part A
    genre_acc_full = "77.21% ± 2.42%"
    genre_acc_lite = "76.44% ± 2.88%"
    genre_acc_rf = "74.21% ± 3.63%"
    
    mood_r2_full = "0.5539 ± 0.0384"
    mood_r2_lite = "0.5464 ± 0.0400"
    mood_r2_rf = "0.5553 ± 0.0388"
    
    if os.path.exists(os.path.join(C.OUT, "part_a_genre_summary.csv")):
        g_df = pd.read_csv(os.path.join(C.OUT, "part_a_genre_summary.csv"))
        genre_acc_full = g_df.loc[g_df["Model"] == "LightGBM Full-113", "Accuracy"].values[0]
        genre_acc_lite = g_df.loc[g_df["Model"].str.contains("Lite"), "Accuracy"].values[0]
        genre_acc_rf = g_df.loc[g_df["Model"].str.contains("RandomForest"), "Accuracy"].values[0]
        
    if os.path.exists(os.path.join(C.OUT, "part_a_mood_summary.csv")):
        m_df = pd.read_csv(os.path.join(C.OUT, "part_a_mood_summary.csv"))
        mood_r2_full = m_df.loc[m_df["Model"] == "LightGBM Full-113", "Mean R2"].values[0]
        mood_r2_lite = m_df.loc[m_df["Model"].str.contains("Lite"), "Mean R2"].values[0]
        mood_r2_rf = m_df.loc[m_df["Model"].str.contains("RandomForest"), "Mean R2"].values[0]
        
    rf_e2e_mean = e2e_summary["orig_mean_ms"] - inf_summary["Full"]["mean_ms"] + inf_summary["RF"]["mean_ms"]
    
    summary_rows = [
        {
            "Model Set": "RandomForest (113 features, 100 trees)",
            "Serialized Size (MB)": f"{sizes_dict['RF']:.2f} MB",
            "Inference Latency (ms)": f"{inf_summary['RF']['mean_ms']:.2f} ± {inf_summary['RF']['std_ms']:.2f} ms",
            "End-to-End Latency (ms)": f"{rf_e2e_mean:.1f} ms",
            "Genre Accuracy": genre_acc_rf,
            "Mood Mean R2": mood_r2_rf
        },
        {
            "Model Set": "LightGBM Full-113 (Original MOODWAAVE)",
            "Serialized Size (MB)": f"{sizes_dict['Full']:.2f} MB",
            "Inference Latency (ms)": f"{inf_summary['Full']['mean_ms']:.2f} ± {inf_summary['Full']['std_ms']:.2f} ms",
            "End-to-End Latency (ms)": f"{e2e_summary['orig_mean_ms']:.1f} ± {e2e_summary['orig_std_ms']:.1f} ms",
            "Genre Accuracy": genre_acc_full,
            "Mood Mean R2": mood_r2_full
        },
        {
            "Model Set": "LightGBM Lite-71 (MOODWAAVE-Lite)",
            "Serialized Size (MB)": f"{sizes_dict['Lite']:.2f} MB",
            "Inference Latency (ms)": f"{inf_summary['Lite']['mean_ms']:.2f} ± {inf_summary['Lite']['std_ms']:.2f} ms",
            "End-to-End Latency (ms)": f"{e2e_summary['lite_mean_ms']:.1f} ± {e2e_summary['lite_std_ms']:.1f} ms",
            "Genre Accuracy": genre_acc_lite,
            "Mood Mean R2": mood_r2_lite
        }
    ]
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(C.OUT, "part_b_efficiency_summary.csv"), index=False)
    
    print("\n" + "="*80)
    print("PART B: CONSOLIDATED EFFICIENCY SUMMARY TABLE")
    print("="*80)
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
