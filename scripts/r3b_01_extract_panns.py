import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from panns_inference import AudioTagging

from r3b_common import (
    BASE_DIR,
    GTZAN_AUDIO_DIR,
    PMEMO_AUDIO_DIR,
    load_gtzan_metadata,
    load_pmemo_metadata,
    load_audio_mono_30s,
    log_environment,
)

TARGET_SR = 32000

def main():
    print("=== Step 1: Extract PANNs CNN14 Embeddings ===")
    t_start = time.perf_counter()
    log_environment(BASE_DIR)
    
    checkpoint_path = str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth")
    print(f"Loading PANNs model from {checkpoint_path}...")
    at = AudioTagging(checkpoint_path=checkpoint_path, device="cpu")
    print("PANNs model loaded successfully on CPU.")
    
    # ---------------------------------------------------------
    # 1. GTZAN
    # ---------------------------------------------------------
    gtzan_out_csv = os.path.join(BASE_DIR, "panns_gtzan_embeddings.csv")
    df_g, _ = load_gtzan_metadata()
    print(f"\nProcessing GTZAN: {len(df_g)} clean tracks...")
    
    existing_gtzan = {}
    if os.path.isfile(gtzan_out_csv):
        df_exist = pd.read_csv(gtzan_out_csv)
        if "track_id" in df_exist.columns and len(df_exist) > 0:
            existing_gtzan = {row["track_id"]: row.drop("track_id").values for _, row in df_exist.iterrows()}
            print(f"Resuming GTZAN from {len(existing_gtzan)} already extracted tracks.")
            
    gtzan_records = []
    # If resuming, load existing
    for tid, vec in existing_gtzan.items():
        rec = {"track_id": tid}
        for i, val in enumerate(vec):
            rec[f"panns_{i:04d}"] = val
        gtzan_records.append(rec)
        
    extracted_new_g = 0
    t0_g = time.perf_counter()
    for idx, (filename, row) in enumerate(df_g.iterrows()):
        if filename in existing_gtzan:
            continue
        genre = filename.split(".")[0]
        audio_path = os.path.join(GTZAN_AUDIO_DIR, genre, filename)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Missing audio: {audio_path}")
            
        audio = load_audio_mono_30s(audio_path, target_sr=TARGET_SR)
        _, emb = at.inference(audio[None, :])
        emb_vec = emb.flatten()
        
        rec = {"track_id": filename}
        for i, val in enumerate(emb_vec):
            rec[f"panns_{i:04d}"] = float(val)
        gtzan_records.append(rec)
        existing_gtzan[filename] = emb_vec
        extracted_new_g += 1
        
        if (extracted_new_g % 50 == 0) or (idx == len(df_g) - 1):
            pd.DataFrame(gtzan_records).to_csv(gtzan_out_csv, index=False)
            elapsed = time.perf_counter() - t0_g
            rate = elapsed / max(1, extracted_new_g)
            rem = (len(df_g) - len(existing_gtzan)) * rate
            print(f"  GTZAN: {len(existing_gtzan)}/{len(df_g)} tracks done ({rate:.2f} s/trk, ~{rem/60:.1f}m left)")
            
    pd.DataFrame(gtzan_records).to_csv(gtzan_out_csv, index=False)
    print(f"GTZAN PANNs embeddings saved to {gtzan_out_csv} ({len(gtzan_records)} tracks, 2048 dims).")
    
    # ---------------------------------------------------------
    # 2. PMEmo
    # ---------------------------------------------------------
    pmemo_out_csv = os.path.join(BASE_DIR, "panns_pmemo_embeddings.csv")
    df_p, _ = load_pmemo_metadata()
    print(f"\nProcessing PMEmo: {len(df_p)} tracks...")
    
    existing_pmemo = {}
    if os.path.isfile(pmemo_out_csv):
        df_exist = pd.read_csv(pmemo_out_csv)
        if "track_id" in df_exist.columns and len(df_exist) > 0:
            existing_pmemo = {str(row["track_id"]): row.drop("track_id").values for _, row in df_exist.iterrows()}
            print(f"Resuming PMEmo from {len(existing_pmemo)} already extracted tracks.")
            
    pmemo_records = []
    for tid, vec in existing_pmemo.items():
        rec = {"track_id": tid}
        for i, val in enumerate(vec):
            rec[f"panns_{i:04d}"] = val
        pmemo_records.append(rec)
        
    extracted_new_p = 0
    t0_p = time.perf_counter()
    for idx, (music_id, row) in enumerate(df_p.iterrows()):
        mid_str = str(music_id)
        if mid_str in existing_pmemo:
            continue
        audio_path = os.path.join(PMEMO_AUDIO_DIR, f"{mid_str}.mp3")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Missing audio: {audio_path}")
            
        audio = load_audio_mono_30s(audio_path, target_sr=TARGET_SR)
        _, emb = at.inference(audio[None, :])
        emb_vec = emb.flatten()
        
        rec = {"track_id": mid_str}
        for i, val in enumerate(emb_vec):
            rec[f"panns_{i:04d}"] = float(val)
        pmemo_records.append(rec)
        existing_pmemo[mid_str] = emb_vec
        extracted_new_p += 1
        
        if (extracted_new_p % 50 == 0) or (idx == len(df_p) - 1):
            pd.DataFrame(pmemo_records).to_csv(pmemo_out_csv, index=False)
            elapsed = time.perf_counter() - t0_p
            rate = elapsed / max(1, extracted_new_p)
            rem = (len(df_p) - len(existing_pmemo)) * rate
            print(f"  PMEmo: {len(existing_pmemo)}/{len(df_p)} tracks done ({rate:.2f} s/trk, ~{rem/60:.1f}m left)")
            
    pd.DataFrame(pmemo_records).to_csv(pmemo_out_csv, index=False)
    print(f"PMEmo PANNs embeddings saved to {pmemo_out_csv} ({len(pmemo_records)} tracks, 2048 dims).")
    
    total_time = time.perf_counter() - t_start
    print(f"\nAll PANNs CNN14 embeddings extracted successfully in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()

