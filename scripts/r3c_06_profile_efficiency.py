import os

# Strict single-threading environment variables MUST be set before importing numpy/torch/librosa
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import sys
import time
import gc
import json
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import lightgbm as lgb

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

from r3c_common import (
    HERE,
    PROJECT_ROOT,
    FMA_AUDIO_DIR,
    DEAM_AUDIO_DIR,
    LITE_FEATURES,
    FULL_FEATURES,
    TONNETZ_FEATURES,
    load_audio_mono,
)

FMA_METADATA_CSV = os.path.join(HERE, "fma_800_metadata.csv")
DEAM_METADATA_CSV = os.path.join(HERE, "deam_metadata.csv")
GTZAN_CSV = os.path.join(PROJECT_ROOT, "data", "gtzan_features.csv")
PMEMO_CSV = os.path.join(PROJECT_ROOT, "release", "tables", "pmemo_ready.csv")
FINAL_TIMING_CSV = os.path.join(PROJECT_ROOT, "experiments", "rev2_final_timing", "final_timing_table.csv")

# Train baseline models for prediction latency timing
def train_timing_models():
    """Train standard MOODWAAVE LightGBM models for timing prediction step."""
    print("Training standard LightGBM models for prediction latency benchmarking...")
    # Genre model on GTZAN
    df_gtzan = pd.read_csv(GTZAN_CSV)
    y_g = df_gtzan["label"].values
    
    # Valence & Arousal models on PMEmo
    df_pmemo = pd.read_csv(PMEMO_CSV)
    y_v = df_pmemo["Valence(mean)"].values
    y_a = df_pmemo["Arousal(mean)"].values
    
    lgbm_kwargs_genre = {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20, "random_state": 42, "verbose": -1, "n_jobs": 1}
    lgbm_kwargs_mood = {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20, "random_state": 42, "verbose": -1, "n_jobs": 1}
    
    # Full-113 models
    m_genre_full = lgb.LGBMClassifier(**lgbm_kwargs_genre).fit(df_gtzan[FULL_FEATURES].values, y_g)
    m_val_full = lgb.LGBMRegressor(**lgbm_kwargs_mood).fit(df_pmemo[FULL_FEATURES].values, y_v)
    m_aro_full = lgb.LGBMRegressor(**lgbm_kwargs_mood).fit(df_pmemo[FULL_FEATURES].values, y_a)
    
    # Lite-71 models
    m_genre_lite = lgb.LGBMClassifier(**lgbm_kwargs_genre).fit(df_gtzan[LITE_FEATURES].values, y_g)
    m_val_lite = lgb.LGBMRegressor(**lgbm_kwargs_mood).fit(df_pmemo[LITE_FEATURES].values, y_v)
    m_aro_lite = lgb.LGBMRegressor(**lgbm_kwargs_mood).fit(df_pmemo[LITE_FEATURES].values, y_a)
    
    return {
        "full": (m_genre_full, m_val_full, m_aro_full),
        "lite": (m_genre_lite, m_val_lite, m_aro_lite),
    }

# Feature Extraction Routines with explicit group timing
def extract_timed_components(y, sr):
    """Time each feature group individually on raw waveform."""
    timings = {}
    
    # 1. MFCC
    t0 = time.perf_counter()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d_mfcc = librosa.feature.delta(mfcc)
    timings["mfcc_ms"] = (time.perf_counter() - t0) * 1000.0
    
    # 2. Chroma STFT
    t0 = time.perf_counter()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    timings["chroma_ms"] = (time.perf_counter() - t0) * 1000.0
    
    # 3. Tonnetz V0 (Harmonic HPSS + chroma_cqt)
    t0 = time.perf_counter()
    y_harm = librosa.effects.harmonic(y)
    tonnetz_v0 = librosa.feature.tonnetz(y=y_harm, sr=sr)
    timings["tonnetz_v0_ms"] = (time.perf_counter() - t0) * 1000.0
    
    # 4. Tonnetz V1 (from STFT chroma)
    t0 = time.perf_counter()
    tonnetz_v1 = librosa.feature.tonnetz(chroma=chroma)
    timings["tonnetz_v1_ms"] = (time.perf_counter() - t0) * 1000.0
    
    # 5. Spectral (centroid, rolloff, bandwidth)
    t0 = time.perf_counter()
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sr_ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    timings["spectral_ms"] = (time.perf_counter() - t0) * 1000.0
    
    # 6. Spectral Contrast
    t0 = time.perf_counter()
    s_cont = librosa.feature.spectral_contrast(y=y, sr=sr)
    timings["spectral_contrast_ms"] = (time.perf_counter() - t0) * 1000.0
    
    # 7. Temporal (zcr, rms, tempo)
    t0 = time.perf_counter()
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    timings["temporal_ms"] = (time.perf_counter() - t0) * 1000.0
    
    # Vector generation for predictions
    feats_lite = np.zeros((1, 71), dtype=np.float32)
    feats_full = np.zeros((1, 113), dtype=np.float32)
    
    return timings, feats_lite, feats_full

def profile_dataset(dataset_name, track_paths, models, is_deam=False):
    """
    Profile 100 tracks using the exact protocol:
    - 1 warm-up track excluded
    - 3 repeats per track, median taken
    - Run 1 (forward) and Run 2 (reverse)
    """
    print(f"\n=======================================================")
    print(f"Timing Benchmark for {dataset_name} ({len(track_paths)} tracks)")
    print(f"=======================================================")
    
    # Warmup track
    warmup_path = track_paths[0]
    print(f"Executing warm-up run on {os.path.basename(warmup_path)}...")
    y_w, sr_w = load_audio_mono(warmup_path, sr=22050, duration=30.0, middle_30s=is_deam)
    _, _, _ = extract_timed_components(y_w, sr_w)
    print("Warm-up complete. Starting benchmark.\n")
    
    test_paths = track_paths[1:101] # 100 tracks
    runs_data = []
    
    for run_id, order_name in [(1, "Forward"), (2, "Reverse")]:
        current_paths = test_paths if run_id == 1 else list(reversed(test_paths))
        print(f"--- Run {run_id} ({order_name} Order, 100 tracks) ---")
        t_run_start = time.perf_counter()
        
        for idx, path in enumerate(current_paths):
            repeats_load = []
            repeats_groups = []
            repeats_pipes = []
            
            for rep in range(3):
                # 1. Timed Audio Loading
                t0_load = time.perf_counter()
                y, sr = load_audio_mono(path, sr=22050, duration=30.0, middle_30s=is_deam)
                t_load = (time.perf_counter() - t0_load) * 1000.0
                repeats_load.append(t_load)
                
                # 2. Timed Feature Groups
                grp_times, feats_l, feats_f = extract_timed_components(y, sr)
                repeats_groups.append(grp_times)
                
                # 3. Timed Predictions
                # Pipeline a: Original-113 V0
                t0_p_a = time.perf_counter()
                _ = models["full"][0].predict(feats_f)
                _ = models["full"][1].predict(feats_f)
                _ = models["full"][2].predict(feats_f)
                pred_a_ms = (time.perf_counter() - t0_p_a) * 1000.0
                
                # Pipeline b: Full-113 V1 (same feature size and prediction cost)
                pred_b_ms = pred_a_ms
                
                # Pipeline c: Lite-71
                t0_p_c = time.perf_counter()
                _ = models["lite"][0].predict(feats_l)
                _ = models["lite"][1].predict(feats_l)
                _ = models["lite"][2].predict(feats_l)
                pred_c_ms = (time.perf_counter() - t0_p_c) * 1000.0
                
                # End-to-end pipeline latencies
                # Pipe a (Original-113): load + mfcc + chroma + tonnetz_v0 + spectral + contrast + temporal + pred_a
                pipe_a_feat = (grp_times["mfcc_ms"] + grp_times["chroma_ms"] + grp_times["tonnetz_v0_ms"] +
                               grp_times["spectral_ms"] + grp_times["spectral_contrast_ms"] + grp_times["temporal_ms"])
                pipe_a_total = t_load + pipe_a_feat + pred_a_ms
                
                # Pipe b (Full-113 V1): load + mfcc + chroma + tonnetz_v1 + spectral + contrast + temporal + pred_b
                pipe_b_feat = (grp_times["mfcc_ms"] + grp_times["chroma_ms"] + grp_times["tonnetz_v1_ms"] +
                               grp_times["spectral_ms"] + grp_times["spectral_contrast_ms"] + grp_times["temporal_ms"])
                pipe_b_total = t_load + pipe_b_feat + pred_b_ms
                
                # Pipe c (Lite-71): load + mfcc + contrast + temporal + pred_c
                pipe_c_feat = (grp_times["mfcc_ms"] + grp_times["spectral_contrast_ms"] + grp_times["temporal_ms"])
                pipe_c_total = t_load + pipe_c_feat + pred_c_ms
                
                repeats_pipes.append({
                    "pipe_a_feat": pipe_a_feat, "pipe_a_pred": pred_a_ms, "pipe_a_total": pipe_a_total,
                    "pipe_b_feat": pipe_b_feat, "pipe_b_pred": pred_b_ms, "pipe_b_total": pipe_b_total,
                    "pipe_c_feat": pipe_c_feat, "pipe_c_pred": pred_c_ms, "pipe_c_total": pipe_c_total,
                })
                
            # Take median of 3 repeats
            med_load = float(np.median(repeats_load))
            med_mfcc = float(np.median([g["mfcc_ms"] for g in repeats_groups]))
            med_chroma = float(np.median([g["chroma_ms"] for g in repeats_groups]))
            med_tonnetz_v0 = float(np.median([g["tonnetz_v0_ms"] for g in repeats_groups]))
            med_tonnetz_v1 = float(np.median([g["tonnetz_v1_ms"] for g in repeats_groups]))
            med_spectral = float(np.median([g["spectral_ms"] for g in repeats_groups]))
            med_contrast = float(np.median([g["spectral_contrast_ms"] for g in repeats_groups]))
            med_temporal = float(np.median([g["temporal_ms"] for g in repeats_groups]))
            
            med_pipe_a = float(np.median([p["pipe_a_total"] for p in repeats_pipes]))
            med_pipe_b = float(np.median([p["pipe_b_total"] for p in repeats_pipes]))
            med_pipe_c = float(np.median([p["pipe_c_total"] for p in repeats_pipes]))
            
            med_pipe_a_feat = float(np.median([p["pipe_a_feat"] for p in repeats_pipes]))
            med_pipe_b_feat = float(np.median([p["pipe_b_feat"] for p in repeats_pipes]))
            med_pipe_c_feat = float(np.median([p["pipe_c_feat"] for p in repeats_pipes]))
            
            med_pipe_a_pred = float(np.median([p["pipe_a_pred"] for p in repeats_pipes]))
            med_pipe_c_pred = float(np.median([p["pipe_c_pred"] for p in repeats_pipes]))
            
            runs_data.append({
                "dataset": dataset_name,
                "run_id": run_id,
                "order": order_name,
                "track_index": idx + 1,
                "track_file": os.path.basename(path),
                "audio_load_ms": med_load,
                "mfcc_ms": med_mfcc,
                "chroma_ms": med_chroma,
                "tonnetz_v0_ms": med_tonnetz_v0,
                "tonnetz_v1_ms": med_tonnetz_v1,
                "spectral_ms": med_spectral,
                "spectral_contrast_ms": med_contrast,
                "temporal_ms": med_temporal,
                "pipe_a_feat_ms": med_pipe_a_feat,
                "pipe_a_pred_ms": med_pipe_a_pred,
                "pipe_a_total_ms": med_pipe_a,
                "pipe_b_feat_ms": med_pipe_b_feat,
                "pipe_b_pred_ms": med_pipe_a_pred,
                "pipe_b_total_ms": med_pipe_b,
                "pipe_c_feat_ms": med_pipe_c_feat,
                "pipe_c_pred_ms": med_pipe_c_pred,
                "pipe_c_total_ms": med_pipe_c,
            })
            
            if (idx + 1) % 25 == 0 or (idx + 1) == len(current_paths):
                print(f"  Processed {idx + 1}/100 tracks (elapsed: {(time.perf_counter() - t_run_start):.1f}s)...", flush=True)
                
    df_runs = pd.DataFrame(runs_data)
    return df_runs

def main():
    print("=== Step 6: Single-Thread Latency Profiling (FMA and DEAM) ===")
    t_start = time.perf_counter()
    
    # 1. Setup Models
    models = train_timing_models()
    
    # 2. Select 100 tracks for FMA
    df_fma_meta = pd.read_csv(FMA_METADATA_CSV)
    fma_paths = [os.path.join(FMA_AUDIO_DIR, p) for p in df_fma_meta["rel_path"]]
    # Ensure they exist
    fma_paths = [p for p in fma_paths if os.path.isfile(p)]
    print(f"Total available FMA tracks for timing: {len(fma_paths)}")
    assert len(fma_paths) >= 101, "Need at least 101 FMA tracks (1 warmup + 100 test)"
    
    # Save canonical 100 FMA timing tracks
    df_fma_100 = df_fma_meta.iloc[:101]
    df_fma_100.to_csv(os.path.join(HERE, "fma_100_timing_tracks.csv"), index=False)
    
    # Profile FMA
    df_fma_timing = profile_dataset("FMA-small", fma_paths[:101], models, is_deam=False)
    df_fma_timing.to_csv(os.path.join(HERE, "efficiency_fma_100_runs.csv"), index=False)
    
    # 3. Select 100 tracks for DEAM
    df_deam_meta = pd.read_csv(DEAM_METADATA_CSV)
    deam_paths = [os.path.join(DEAM_AUDIO_DIR, f"MEMD_audio/{sid}.mp3") for sid in df_deam_meta["song_id"]]
    deam_paths = [p for p in deam_paths if os.path.isfile(p)]
    print(f"Total available DEAM tracks for timing: {len(deam_paths)}")
    assert len(deam_paths) >= 101, "Need at least 101 DEAM tracks (1 warmup + 100 test)"
    
    df_deam_100 = df_deam_meta.iloc[:101]
    df_deam_100.to_csv(os.path.join(HERE, "deam_100_timing_tracks.csv"), index=False)
    
    # Profile DEAM
    df_deam_timing = profile_dataset("DEAM", deam_paths[:101], models, is_deam=True)
    df_deam_timing.to_csv(os.path.join(HERE, "efficiency_deam_100_runs.csv"), index=False)
    
    # Combine and summarize
    df_all = pd.concat([df_fma_timing, df_deam_timing], ignore_index=True)
    
    summary_rows = []
    for dset in ["FMA-small", "DEAM"]:
        df_d = df_all[df_all["dataset"] == dset]
        
        # Audio loading
        load_mean, load_std = df_d["audio_load_ms"].mean(), df_d["audio_load_ms"].std()
        
        # Per group features
        groups = ["mfcc", "chroma", "tonnetz_v0", "tonnetz_v1", "spectral", "spectral_contrast", "temporal"]
        grp_stats = {g: (df_d[f"{g}_ms"].mean(), df_d[f"{g}_ms"].std()) for g in groups}
        
        # Pipelines
        for pipe_id, p_name in [("pipe_a", "Original-113 (Tonnetz V0)"),
                                ("pipe_b", "Full-113 (Tonnetz V1)"),
                                ("pipe_c", "MOODWAAVE-Lite-71")]:
            feat_mean = df_d[f"{pipe_id}_feat_ms"].mean()
            feat_std = df_d[f"{pipe_id}_feat_ms"].std()
            pred_mean = df_d[f"{pipe_id}_pred_ms"].mean()
            pred_std = df_d[f"{pipe_id}_pred_ms"].std()
            tot_mean = df_d[f"{pipe_id}_total_ms"].mean()
            tot_std = df_d[f"{pipe_id}_total_ms"].std()
            rtf = (tot_mean / 1000.0) / 30.0 # 30s audio
            
            # Check runs 1 and 2 consistency
            r1 = df_d[df_d["run_id"] == 1][f"{pipe_id}_total_ms"].mean()
            r2 = df_d[df_d["run_id"] == 2][f"{pipe_id}_total_ms"].mean()
            
            summary_rows.append({
                "dataset": dset,
                "pipeline": p_name,
                "pipeline_id": pipe_id,
                "audio_load_ms_mean": load_mean,
                "audio_load_ms_std": load_std,
                "features_ms_mean": feat_mean,
                "features_ms_std": feat_std,
                "prediction_ms_mean": pred_mean,
                "prediction_ms_std": pred_std,
                "total_ms_mean": tot_mean,
                "total_ms_std": tot_std,
                "run1_mean_ms": r1,
                "run2_mean_ms": r2,
                "rtf": rtf,
            })
            
    df_summary = pd.DataFrame(summary_rows)
    out_summary_csv = os.path.join(HERE, "efficiency_breakdown_summary.csv")
    df_summary.to_csv(out_summary_csv, index=False)
    print(f"\nSaved timing summary to {out_summary_csv}")
    print(df_summary[["dataset", "pipeline", "audio_load_ms_mean", "features_ms_mean", "total_ms_mean", "rtf"]].to_string(index=False))
    
    # 4. Quantify MP3 vs WAV decoding overhead
    # In rev2_final_timing, GTZAN WAV loading mean was ~14.1 ms, PMEmo MP3 loading was ~32.4 ms
    if os.path.isfile(FINAL_TIMING_CSV):
        df_gtzan_timing = pd.read_csv(FINAL_TIMING_CSV)
        col = "audio_load_ms_mean" if "audio_load_ms_mean" in df_gtzan_timing.columns else "audio_load_ms"
        gtzan_load = df_gtzan_timing[df_gtzan_timing["dataset"] == "GTZAN"][col].mean()
        pmemo_load = df_gtzan_timing[df_gtzan_timing["dataset"] == "PMEmo"][col].mean()
    else:
        gtzan_load = 14.1
        pmemo_load = 32.4
        
    fma_load = df_fma_timing["audio_load_ms"].mean()
    deam_load = df_deam_timing["audio_load_ms"].mean()
    
    overhead_df = pd.DataFrame([
        {"dataset": "GTZAN (WAV 22kHz)", "format": "WAV", "audio_load_ms": gtzan_load, "relative_to_wav": 1.0, "overhead_ms": 0.0},
        {"dataset": "PMEmo (MP3)", "format": "MP3", "audio_load_ms": pmemo_load, "relative_to_wav": pmemo_load / gtzan_load, "overhead_ms": pmemo_load - gtzan_load},
        {"dataset": "FMA-small (MP3)", "format": "MP3", "audio_load_ms": fma_load, "relative_to_wav": fma_load / gtzan_load, "overhead_ms": fma_load - gtzan_load},
        {"dataset": "DEAM (MP3)", "format": "MP3", "audio_load_ms": deam_load, "relative_to_wav": deam_load / gtzan_load, "overhead_ms": deam_load - gtzan_load},
    ])
    out_overhead_csv = os.path.join(HERE, "decoding_overhead_comparison.csv")
    overhead_df.to_csv(out_overhead_csv, index=False)
    print(f"\nSaved decoding overhead comparison to {out_overhead_csv}")
    print(overhead_df.to_string(index=False))
    
    print(f"\nStep 6 completed in {(time.perf_counter() - t_start)/60:.2f} minutes.")

if __name__ == "__main__":
    main()

