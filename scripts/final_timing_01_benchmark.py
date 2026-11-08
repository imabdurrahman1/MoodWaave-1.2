"""
experiments/rev2_final_timing/01_run_final_timing_benchmark.py
TASK FINAL-TIMING: One session, all pipelines, one master table.
Strict protocol:
- Conditions: idle machine, plugged in (AC), High performance power plan, sleep disabled.
- Single thread for everything (including torch.set_num_threads(1)).
- Same 100 tracks per dataset (GTZAN and PMEmo from sample_ids.json).
- One warm-up track excluded per dataset.
- Median of 3 repeats per track.
- Run twice: Run 1 (forward track order), Run 2 (reverse track order). Both reported.
- Timed region: audio loading + feature/embedding extraction + ALL predictions that system makes.
"""

import os
# Strict single-thread settings before any numeric library imports
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
import joblib
import torch

warnings.filterwarnings("ignore")

import common as C

torch.set_num_threads(1)

REPEATS = 3
NUM_TRACKS = 100

def time_pipeline_on_track(path, pipeline_id, models, sr_target=22050):
    """
    Executes one timed run of pipeline_id on path.
    Returns (audio_load_ms, feat_ms, pred_ms, total_ms).
    """
    if pipeline_id == "a_orig_113":
        m_g, m_v, m_a = models["a"]
        t0 = time.perf_counter()
        y, sr = librosa.load(path, sr=sr_target, mono=True, duration=30.0)
        t1 = time.perf_counter()
        feats = C.extract_original_113_v0(y, sr)
        x_vec = np.array([[feats[c] for c in C.FULL_FEATURES_ORIG]])
        t2 = time.perf_counter()
        _ = m_g.predict(x_vec)
        _ = m_v.predict(x_vec)
        _ = m_a.predict(x_vec)
        t3 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, (t3 - t0) * 1000.0

    elif pipeline_id == "b_full_113_v1":
        m_g, m_v, m_a = models["b"]
        t0 = time.perf_counter()
        y, sr = librosa.load(path, sr=sr_target, mono=True, duration=30.0)
        t1 = time.perf_counter()
        feats = C.extract_full_113_v1(y, sr)
        x_vec = np.array([[feats[c] for c in C.FULL_FEATURES_ORIG]])
        t2 = time.perf_counter()
        _ = m_g.predict(x_vec)
        _ = m_v.predict(x_vec)
        _ = m_a.predict(x_vec)
        t3 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, (t3 - t0) * 1000.0

    elif pipeline_id == "c_lite_71":
        m_g, m_v, m_a = models["c"]
        t0 = time.perf_counter()
        y, sr = librosa.load(path, sr=sr_target, mono=True, duration=30.0)
        t1 = time.perf_counter()
        feats = C.extract_lite_71(y, sr)
        x_vec = np.array([[feats[c] for c in C.LITE_FEATURES]])
        t2 = time.perf_counter()
        _ = m_g.predict(x_vec)
        _ = m_v.predict(x_vec)
        _ = m_a.predict(x_vec)
        t3 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, (t3 - t0) * 1000.0

    elif pipeline_id == "d_lite_83_v1":
        m_g, m_v, m_a = models["d"]
        t0 = time.perf_counter()
        y, sr = librosa.load(path, sr=sr_target, mono=True, duration=30.0)
        t1 = time.perf_counter()
        feats = C.extract_lite_plus_v1_83(y, sr)
        x_vec = np.array([[feats[c] for c in C.LITE_PLUS_V1_FEATURES]])
        t2 = time.perf_counter()
        _ = m_g.predict(x_vec)
        _ = m_v.predict(x_vec)
        _ = m_a.predict(x_vec)
        t3 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, (t3 - t0) * 1000.0

    elif pipeline_id == "e_small_cnn":
        cnn_model = models["e"]
        t0 = time.perf_counter()
        y, sr = librosa.load(path, sr=sr_target, mono=True, duration=30.0)
        t1 = time.perf_counter()
        mel = C.compute_log_mel(y, sr=sr_target)
        total_frames = mel.shape[1]
        target_frames = C.CROP_FRAMES
        n_crops = max(1, total_frames // target_frames)
        crops = [mel[:, i*target_frames:(i+1)*target_frames] for i in range(n_crops)
                 if mel[:, i*target_frames:(i+1)*target_frames].shape[1] == target_frames]
        crops_t = torch.from_numpy(np.stack(crops, axis=0)).unsqueeze(1)
        t2 = time.perf_counter()
        with torch.no_grad():
            logits = cnn_model(crops_t)
            _ = torch.argmax(logits.mean(dim=0)).item()
        t3 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, (t3 - t0) * 1000.0

    elif pipeline_id == "f_vggish":
        vggish, head_g, head_v, head_a = models["f"]
        t0 = time.perf_counter()
        y_16, sr_16 = librosa.load(path, sr=16000, mono=True, duration=30.0)
        t1 = time.perf_counter()
        with torch.no_grad():
            embs = vggish(y_16, sr_16)
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().numpy()
        else:
            embs_np = np.asarray(embs)
        pooled = np.concatenate([np.mean(embs_np, axis=0), np.std(embs_np, axis=0)]).reshape(1, -1)
        t2 = time.perf_counter()
        _ = head_g.predict(pooled)
        _ = head_v.predict(pooled)
        _ = head_a.predict(pooled)
        t3 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, (t3 - t0) * 1000.0

    else:
        raise ValueError(f"Unknown pipeline_id {pipeline_id}")

def benchmark_track_median(path, pipeline_id, models, repeats=REPEATS):
    """Computes median across repeats for load, feat, pred, and total."""
    loads, feats, preds, totals = [], [], [], []
    for _ in range(repeats):
        l_ms, f_ms, p_ms, tot_ms = time_pipeline_on_track(path, pipeline_id, models)
        loads.append(l_ms)
        feats.append(f_ms)
        preds.append(p_ms)
        totals.append(tot_ms)
    return float(np.median(loads)), float(np.median(feats)), float(np.median(preds)), float(np.median(totals))

def run_warmup(warmup_path, models, pipeline_order):
    print(f"  Executing warm-up on {warmup_path} (3 repeats, excluded from statistics)...", flush=True)
    for p_id in pipeline_order:
        _ = benchmark_track_median(warmup_path, p_id, models, repeats=3)
    print("  Warm-up completed successfully.\n", flush=True)

def execute_timing():
    start_time = time.time()
    print("=" * 80)
    print("MOODWAAVE - TASK FINAL-TIMING: Comprehensive Benchmark across All Pipelines")
    print("=" * 80)
    
    # 1. Environment & Logging
    print("[1/5] Logging environment...")
    env = C.log_environment(C.BASE_DIR)
    print(f"OS: {env['os']} | CPU: {env['processor']} | RAM: {env['ram_gb']} GB")
    print(f"Power Plan: {env['power_plan']} | Plugged in: {env['battery_plugged']}")
    print(f"Single-threading: OMP={env['thread_settings']['OMP_NUM_THREADS']}, Torch={env['thread_settings']['torch_threads']}")
    
    # 2. Load Models
    print("\n[2/5] Loading serialized models for all pipelines...")
    MODELS_DIR = os.path.join(C.BASE_DIR, "models")
    
    models = {
        "a": (joblib.load(os.path.join(MODELS_DIR, "pipe_a_genre_orig113.pkl")),
              joblib.load(os.path.join(MODELS_DIR, "pipe_a_val_orig113.pkl")),
              joblib.load(os.path.join(MODELS_DIR, "pipe_a_aro_orig113.pkl"))),
        "b": (joblib.load(os.path.join(MODELS_DIR, "pipe_b_genre_full113_v1.pkl")),
              joblib.load(os.path.join(MODELS_DIR, "pipe_b_val_full113_v1.pkl")),
              joblib.load(os.path.join(MODELS_DIR, "pipe_b_aro_full113_v1.pkl"))),
        "c": (joblib.load(os.path.join(MODELS_DIR, "pipe_c_genre_lite71.pkl")),
              joblib.load(os.path.join(MODELS_DIR, "pipe_c_val_lite71.pkl")),
              joblib.load(os.path.join(MODELS_DIR, "pipe_c_aro_lite71.pkl"))),
        "d": (joblib.load(os.path.join(MODELS_DIR, "pipe_d_genre_lite83_v1.pkl")),
              joblib.load(os.path.join(MODELS_DIR, "pipe_d_val_lite83_v1.pkl")),
              joblib.load(os.path.join(MODELS_DIR, "pipe_d_aro_lite83_v1.pkl"))),
    }
    
    cnn_model = C.SmallMelCNN(num_classes=10, dropout=0.2)
    cnn_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "small_cnn_seed_42.pt"), map_location="cpu"))
    cnn_model.eval()
    models["e"] = cnn_model
    
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish.eval()
    vgg_g = joblib.load(os.path.join(MODELS_DIR, "vggish_genre_head.pkl"))
    vgg_v = joblib.load(os.path.join(MODELS_DIR, "vggish_val_head.pkl"))
    vgg_a = joblib.load(os.path.join(MODELS_DIR, "vggish_aro_head.pkl"))
    models["f"] = (vggish, vgg_g, vgg_v, vgg_a)
    print("All models loaded into memory.")
    
    # 3. Load Sample Tracks
    print("\n[3/5] Loading 100 GTZAN and 100 PMEmo canonical tracks...")
    with open(C.SAMPLE_IDS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)
    gtzan_tracks = samples["GTZAN"][:NUM_TRACKS]
    pmemo_tracks = samples["PMEmo"][:NUM_TRACKS]
    
    warmup_gtzan = os.path.join(C.PROJECT_ROOT, "data", "gtzan", "genres_original", "blues", "blues.00000.wav")
    warmup_pmemo = os.path.join(C.PROJECT_ROOT, "data", "pmemo", "chorus", "1.mp3")
    assert os.path.exists(warmup_gtzan), f"Missing warmup track {warmup_gtzan}"
    assert os.path.exists(warmup_pmemo), f"Missing warmup track {warmup_pmemo}"
    print(f"Loaded {len(gtzan_tracks)} GTZAN tracks and {len(pmemo_tracks)} PMEmo tracks.")
    
    pipeline_configs = [
        ("a_orig_113", "a) Original-113 with original Tonnetz"),
        ("b_full_113_v1", "b) Full-113 with Tonnetz V1"),
        ("c_lite_71", "c) MOODWAAVE-Lite-71"),
        ("d_lite_83_v1", "d) Lite + V1 (83)"),
        ("e_small_cnn", "e) SmallMelCNN (genre only)"),
        ("f_vggish", "f) VGGish + linear heads"),
    ]
    p_ids = [p[0] for p in pipeline_configs]
    
    # Warmup
    print("\nExecuting initial warm-up passes (excluded from statistics)...")
    run_warmup(warmup_gtzan, models, p_ids)
    run_warmup(warmup_pmemo, models, p_ids)
    
    # 4. Benchmark Runs
    print("[4/5] Executing Timing Passes across 2 Runs (Forward and Reverse Order)...")
    
    all_runs_data = []
    
    datasets = [
        ("GTZAN", gtzan_tracks, warmup_gtzan),
        ("PMEmo", pmemo_tracks, warmup_pmemo),
    ]
    
    for run_name, reverse_flag in [("Run_1_Forward", False), ("Run_2_Reverse", True)]:
        run_start = time.time()
        print(f"\n========================================================")
        print(f"STARTING {run_name} (Reverse Order = {reverse_flag})")
        print(f"========================================================")
        
        run_records = []
        
        for ds_name, tracks, w_track in datasets:
            current_tracks = list(reversed(tracks)) if reverse_flag else list(tracks)
            print(f"\n--- [{run_name}] Timing {ds_name} ({len(current_tracks)} tracks) ---")
            
            # Mini-warmup before dataset pass
            run_warmup(w_track, models, p_ids)
            
            for t_idx, track_path in enumerate(current_tracks):
                gc.collect()
                orig_track_idx = (len(tracks) - 1 - t_idx) if reverse_flag else t_idx
                
                for p_id, p_name in pipeline_configs:
                    l_ms, f_ms, p_ms, tot_ms = benchmark_track_median(track_path, p_id, models, repeats=REPEATS)
                    rec = {
                        "run": run_name,
                        "dataset": ds_name,
                        "track_idx": orig_track_idx,
                        "track_order_in_run": t_idx,
                        "track_path": track_path,
                        "pipeline_id": p_id,
                        "pipeline": p_name,
                        "audio_load_ms": l_ms,
                        "feat_ms": f_ms,
                        "pred_ms": p_ms,
                        "total_ms": tot_ms,
                    }
                    run_records.append(rec)
                    all_runs_data.append(rec)
                    
                if (t_idx + 1) % 20 == 0 or (t_idx + 1) == len(current_tracks):
                    print(f"  [{run_name} - {ds_name}] Completed {t_idx+1}/{len(current_tracks)} tracks ({time.time() - run_start:.1f}s elapsed)...", flush=True)
                    
        # Save intermediate run CSV
        df_run = pd.DataFrame(run_records)
        fname = "timing_run1_forward.csv" if not reverse_flag else "timing_run2_reverse.csv"
        df_run.to_csv(os.path.join(C.BASE_DIR, fname), index=False)
        print(f"Successfully saved {fname} ({len(df_run)} records in {time.time() - run_start:.1f}s)")
        
    # 5. Process and Compile Master Tables
    print("\n[5/5] Compiling final master tables...")
    df_all = pd.DataFrame(all_runs_data)
    df_all.to_csv(os.path.join(C.BASE_DIR, "final_timing_raw_all_runs.csv"), index=False)
    
    # Run 1 vs Run 2 Comparison
    comp_records = []
    for (ds, p_name), g in df_all.groupby(["dataset", "pipeline"]):
        g1 = g[g["run"] == "Run_1_Forward"]["total_ms"]
        g2 = g[g["run"] == "Run_2_Reverse"]["total_ms"]
        m1 = g1.mean()
        m2 = g2.mean()
        diff_ms = m2 - m1
        diff_pct = (diff_ms / m1) * 100.0
        comp_records.append({
            "dataset": ds,
            "pipeline": p_name,
            "run1_mean_ms": m1,
            "run1_std_ms": g1.std(ddof=1),
            "run2_mean_ms": m2,
            "run2_std_ms": g2.std(ddof=1),
            "diff_ms (run2 - run1)": diff_ms,
            "diff_pct": diff_pct,
        })
    df_comp = pd.DataFrame(comp_records)
    df_comp.to_csv(os.path.join(C.BASE_DIR, "timing_runs_comparison.csv"), index=False)
    print("Saved timing_runs_comparison.csv")
    
    # Per-track pooled across both runs
    df_track_pooled = df_all.groupby(["dataset", "track_idx", "track_path", "pipeline_id", "pipeline"]).agg(
        audio_load_ms=("audio_load_ms", "mean"),
        feat_ms=("feat_ms", "mean"),
        pred_ms=("pred_ms", "mean"),
        total_ms=("total_ms", "mean"),
    ).reset_index()
    df_track_pooled.to_csv(os.path.join(C.BASE_DIR, "final_timing_per_track.csv"), index=False)
    print("Saved final_timing_per_track.csv")
    
    # Master Table: Aggregation by Dataset and Pipeline
    df_sizes = pd.read_csv(os.path.join(C.BASE_DIR, "system_sizes.csv"))
    size_map = dict(zip(df_sizes["pipeline"], df_sizes["total_serialized_mb"]))
    hp_map = dict(zip(df_sizes["pipeline"], df_sizes["hyperparameters"]))
    
    summary_records = []
    for ds_name in ["GTZAN", "PMEmo", "Overall"]:
        if ds_name == "Overall":
            sub_df = df_track_pooled
        else:
            sub_df = df_track_pooled[df_track_pooled["dataset"] == ds_name]
            
        # Get baseline latency for speedup calculation
        orig_mean = sub_df[sub_df["pipeline_id"] == "a_orig_113"]["total_ms"].mean()
        
        for p_id, p_name in pipeline_configs:
            p_rows = sub_df[sub_df["pipeline_id"] == p_id]
            
            load_m = p_rows["audio_load_ms"].mean()
            load_s = p_rows["audio_load_ms"].std(ddof=1)
            feat_m = p_rows["feat_ms"].mean()
            feat_s = p_rows["feat_ms"].std(ddof=1)
            pred_m = p_rows["pred_ms"].mean()
            pred_s = p_rows["pred_ms"].std(ddof=1)
            tot_m = p_rows["total_ms"].mean()
            tot_s = p_rows["total_ms"].std(ddof=1)
            
            # RTF = compute_seconds / audio_seconds (30s audio)
            rtf = (tot_m / 1000.0) / 30.0
            speedup = orig_mean / tot_m if tot_m > 0 else 1.0
            
            summary_records.append({
                "dataset": ds_name,
                "pipeline": p_name,
                "audio_load_ms_mean": load_m,
                "audio_load_ms_sd": load_s,
                "feat_ms_mean": feat_m,
                "feat_ms_sd": feat_s,
                "pred_ms_mean": pred_m,
                "pred_ms_sd": pred_s,
                "total_ms_mean": tot_m,
                "total_ms_sd": tot_s,
                "rtf": rtf,
                "speedup_vs_orig": speedup,
                "serialized_size_mb": size_map.get(p_name, 0.0),
                "hyperparameters": hp_map.get(p_name, "Default"),
            })
            
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(C.BASE_DIR, "final_timing_table.csv"), index=False)
    print("Saved final_timing_table.csv")
    
    total_elapsed = time.time() - start_time
    print(f"\n================================================================================")
    print(f"FINAL-TIMING BENCHMARK COMPLETED IN {total_elapsed/60.0:.2f} MINUTES ({total_elapsed:.1f}s)")
    print(f"================================================================================")

if __name__ == "__main__":
    execute_timing()

