import os
import sys

# CRITICAL: Enforce single thread before importing any numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import torch
torch.set_num_threads(1)

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio.functional as AF
from pathlib import Path

from r3b_common import (
    BASE_DIR,
    PROJECT_ROOT,
    GTZAN_AUDIO_DIR,
    PMEMO_AUDIO_DIR,
    load_audio_mono_30s,
    MODELS_METADATA,
    log_environment,
)

SAMPLE_IDS_JSON = os.path.join(PROJECT_ROOT, "experiments", "idea2_cost_pareto", "sample_ids.json")

def get_serialized_size_kb(paths: list) -> float:
    tot_bytes = 0
    for p in paths:
        if os.path.isfile(p):
            tot_bytes += os.path.getsize(p)
    return tot_bytes / 1024.0

def main():
    print("=== Step 5: Profile Single-Thread Efficiency (Latency, RTF, Size) ===")
    t_start = time.perf_counter()
    log_environment(BASE_DIR)
    
    assert torch.get_num_threads() == 1, "Torch num threads must be 1"
    print("Environment verified: Single thread enforced for all operations.")
    
    # ---------------------------------------------------------
    # 1. Load Canonical Sample Tracks (100 GTZAN, 100 PMEmo)
    # ---------------------------------------------------------
    with open(SAMPLE_IDS_JSON, "r") as f:
        samples = json.load(f)
    gtzan_100 = samples["GTZAN"][:100]
    pmemo_100 = samples["PMEmo"][:100]
    print(f"Loaded {len(gtzan_100)} GTZAN and {len(pmemo_100)} PMEmo canonical tracks.")
    
    # ---------------------------------------------------------
    # 2. Setup Models
    # ---------------------------------------------------------
    # PANNs
    from panns_inference import AudioTagging
    panns_ckpt = str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth")
    panns_model = AudioTagging(checkpoint_path=panns_ckpt, device="cpu")
    
    # MERT
    from transformers import AutoModel, AutoFeatureExtractor
    mert_processor = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    mert_model.eval()
    
    # Linear Heads (dummy fits on random standardized data to profile prediction latency)
    from sklearn.linear_model import LogisticRegression, Ridge
    head_genre_panns = LogisticRegression().fit(np.random.randn(50, 2048), np.random.randint(0, 10, 50))
    head_v_panns = Ridge().fit(np.random.randn(50, 2048), np.random.randn(50))
    head_a_panns = Ridge().fit(np.random.randn(50, 2048), np.random.randn(50))
    
    head_genre_mert = LogisticRegression().fit(np.random.randn(50, 768), np.random.randint(0, 10, 50))
    head_v_mert = Ridge().fit(np.random.randn(50, 768), np.random.randn(50))
    head_a_mert = Ridge().fit(np.random.randn(50, 768), np.random.randn(50))
    
    # Calculate sizes
    panns_size_kb = os.path.getsize(panns_ckpt) / 1024.0
    mert_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--m-a-p--MERT-v1-95M"
    mert_size_kb = sum(f.stat().st_size for f in mert_cache_dir.rglob("*") if f.is_file()) / 1024.0
    
    results = []
    
    # ---------------------------------------------------------
    # 3. Profile PANNs CNN14 (100 tracks, 1 warmup, median of 3 repeats)
    # ---------------------------------------------------------
    print("\n--- Profiling PANNs CNN14 ---", flush=True)
    for ds_name, tracks, audio_dir, is_gtzan in [("GTZAN", gtzan_100, GTZAN_AUDIO_DIR, True), ("PMEmo", pmemo_100, PMEMO_AUDIO_DIR, False)]:
        cache_file = os.path.join(BASE_DIR, f"cache_panns_{ds_name.lower()}.json")
        if os.path.isfile(cache_file):
            with open(cache_file, "r") as f:
                cached_row = json.load(f)
            results.append(cached_row)
            print(f"Loaded cached PANNs [{ds_name}]: {cached_row['total_ms']:.2f} +/- {cached_row.get('total_sd_ms', 0):.2f} ms/track | RTF={cached_row['rtf']:.4f}", flush=True)
            continue

        track_times = []
        # Warm-up track
        warm_file = tracks[0]
        warm_path = warm_file if os.path.isfile(warm_file) else (os.path.join(audio_dir, warm_file.split(".")[0], warm_file) if is_gtzan else os.path.join(audio_dir, f"{warm_file}.mp3"))
        y_w = load_audio_mono_30s(warm_path, 32000)
        _ = panns_model.inference(y_w[None, :])
        
        for idx, tid in enumerate(tracks[1:]): # Exclude warm-up track (99 timed tracks)
            path = tid if os.path.isfile(tid) else (os.path.join(audio_dir, tid.split(".")[0], tid) if is_gtzan else os.path.join(audio_dir, f"{tid}.mp3"))
            rep_load, rep_emb, rep_pred, rep_tot = [], [], [], []
            for _ in range(3):
                t0 = time.perf_counter()
                y = load_audio_mono_30s(path, 32000)
                t1 = time.perf_counter()
                _, emb = panns_model.inference(y[None, :])
                t2 = time.perf_counter()
                _ = head_genre_panns.predict(emb)
                _ = head_v_panns.predict(emb)
                _ = head_a_panns.predict(emb)
                t3 = time.perf_counter()
                
                rep_load.append((t1 - t0) * 1000.0)
                rep_emb.append((t2 - t1) * 1000.0)
                rep_pred.append((t3 - t2) * 1000.0)
                rep_tot.append((t3 - t0) * 1000.0)
                
            track_times.append({
                "load_ms": float(np.median(rep_load)),
                "emb_ms": float(np.median(rep_emb)),
                "pred_ms": float(np.median(rep_pred)),
                "total_ms": float(np.median(rep_tot)),
            })
            if (idx + 1) % 20 == 0:
                print(f"  PANNs [{ds_name}] processed {idx + 1}/99 tracks...", flush=True)
            
        df_tt = pd.DataFrame(track_times)
        mean_tot = float(df_tt["total_ms"].mean())
        sd_tot = float(df_tt["total_ms"].std())
        rtf = float((mean_tot / 1000.0) / 30.0) # 30s audio
        
        row_res = {
            "pipeline": "PANNs-CNN14",
            "dataset": ds_name,
            "n_tracks": len(track_times),
            "protocol": "median of 3 repeats (1 warmup excluded)",
            "audio_load_ms": float(df_tt["load_ms"].mean()),
            "embedding_ms": float(df_tt["emb_ms"].mean()),
            "prediction_ms": float(df_tt["pred_ms"].mean()),
            "total_ms": mean_tot,
            "total_sd_ms": sd_tot,
            "rtf": rtf,
            "size_kb": panns_size_kb,
            "size_mb": panns_size_kb / 1024.0,
        }
        with open(cache_file, "w") as f:
            json.dump(row_res, f, indent=2)
        results.append(row_res)
        print(f"PANNs [{ds_name}]: {mean_tot:.2f} +/- {sd_tot:.2f} ms/track | RTF={rtf:.4f}", flush=True)
        
    # ---------------------------------------------------------
    # 4. Profile MERT-v1-95M (slower than 3s -> 30 tracks x 1 repeat, 1 warmup)
    # ---------------------------------------------------------
    print("\n--- Profiling MERT-v1-95M (Proxy protocol: 30 tracks x 1 repeat) ---", flush=True)
    for ds_name, tracks, audio_dir, is_gtzan in [("GTZAN", gtzan_100[:31], GTZAN_AUDIO_DIR, True), ("PMEmo", pmemo_100[:31], PMEMO_AUDIO_DIR, False)]:
        cache_file = os.path.join(BASE_DIR, f"cache_mert_{ds_name.lower()}.json")
        if os.path.isfile(cache_file):
            with open(cache_file, "r") as f:
                cached_row = json.load(f)
            results.append(cached_row)
            print(f"Loaded cached MERT [{ds_name}]: {cached_row['total_ms']:.2f} +/- {cached_row.get('total_sd_ms', 0):.2f} ms/track | RTF={cached_row['rtf']:.4f}", flush=True)
            continue

        track_times = []
        # Warm-up track
        warm_file = tracks[0]
        warm_path = warm_file if os.path.isfile(warm_file) else (os.path.join(audio_dir, warm_file.split(".")[0], warm_file) if is_gtzan else os.path.join(audio_dir, f"{warm_file}.mp3"))
        y_w = load_audio_mono_30s(warm_path, 24000)
        in_w = mert_processor(y_w, sampling_rate=24000, return_tensors="pt")
        with torch.no_grad():
            _ = mert_model(**in_w)
            
        for idx, tid in enumerate(tracks[1:]): # 30 timed tracks
            path = tid if os.path.isfile(tid) else (os.path.join(audio_dir, tid.split(".")[0], tid) if is_gtzan else os.path.join(audio_dir, f"{tid}.mp3"))
            t0 = time.perf_counter()
            y = load_audio_mono_30s(path, 24000)
            t1 = time.perf_counter()
            in_t = mert_processor(y, sampling_rate=24000, return_tensors="pt")
            with torch.no_grad():
                out = mert_model(**in_t)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            t2 = time.perf_counter()
            _ = head_genre_mert.predict(emb)
            _ = head_v_mert.predict(emb)
            _ = head_a_mert.predict(emb)
            t3 = time.perf_counter()
            
            track_times.append({
                "load_ms": (t1 - t0) * 1000.0,
                "emb_ms": (t2 - t1) * 1000.0,
                "pred_ms": (t3 - t2) * 1000.0,
                "total_ms": (t3 - t0) * 1000.0,
            })
            if (len(track_times)) % 10 == 0:
                print(f"  MERT [{ds_name}] processed {len(track_times)}/30 tracks...", flush=True)
            
        df_tt = pd.DataFrame(track_times)
        mean_tot = float(df_tt["total_ms"].mean())
        sd_tot = float(df_tt["total_ms"].std())
        rtf = float((mean_tot / 1000.0) / 30.0)
        
        row_res = {
            "pipeline": "MERT-v1-95M",
            "dataset": ds_name,
            "n_tracks": len(track_times),
            "protocol": "proxy: 30 tracks x 1 repeat (slower than 3s/track)",
            "audio_load_ms": float(df_tt["load_ms"].mean()),
            "embedding_ms": float(df_tt["emb_ms"].mean()),
            "prediction_ms": float(df_tt["pred_ms"].mean()),
            "total_ms": mean_tot,
            "total_sd_ms": sd_tot,
            "rtf": rtf,
            "size_kb": mert_size_kb,
            "size_mb": mert_size_kb / 1024.0,
        }
        with open(cache_file, "w") as f:
            json.dump(row_res, f, indent=2)
        results.append(row_res)
        print(f"MERT [{ds_name}]: {mean_tot:.2f} +/- {sd_tot:.2f} ms/track | RTF={rtf:.4f}", flush=True)
        
    # ---------------------------------------------------------
    # 5. Include Prior Baselines from rev2_final_timing
    # ---------------------------------------------------------
    final_timing_csv = os.path.join(PROJECT_ROOT, "experiments", "rev2_final_timing", "final_timing_table.csv")
    if os.path.isfile(final_timing_csv):
        df_ft = pd.read_csv(final_timing_csv)
        for _, row in df_ft.iterrows():
            size_mb = float(row.get("serialized_size_mb", np.nan))
            results.append({
                "pipeline": str(row["pipeline"]),
                "dataset": str(row["dataset"]),
                "n_tracks": 100,
                "protocol": "final_timing canonical (median of 3 repeats)",
                "audio_load_ms": float(row.get("audio_load_ms_mean", np.nan)),
                "embedding_ms": float(row.get("feat_ms_mean", np.nan)),
                "prediction_ms": float(row.get("pred_ms_mean", np.nan)),
                "total_ms": float(row["total_ms_mean"]),
                "total_sd_ms": float(row["total_ms_sd"]),
                "rtf": float(row["rtf"]),
                "size_kb": size_mb * 1024.0 if pd.notna(size_mb) else np.nan,
                "size_mb": size_mb,
            })
            
    df_eff = pd.DataFrame(results)
    eff_out = os.path.join(BASE_DIR, "embeddings_efficiency_profiles.csv")
    df_eff.to_csv(eff_out, index=False)
    print(f"\nEfficiency profiles saved to {eff_out}")
    print(df_eff[["pipeline", "dataset", "total_ms", "rtf", "size_mb"]].to_string(index=False))

if __name__ == "__main__":
    main()

