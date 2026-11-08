import os
import sys
import time
import json
import threading
import concurrent.futures
import pandas as pd
import numpy as np
from remotezip import RemoteZip

from r3c_common import (
    HERE,
    FMA_TRACKS_CSV,
    FMA_AUDIO_DIR,
    SEED,
    log_environment,
    load_audio_mono,
    extract_moodwaave_features,
)

FMA_ZIP_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
METADATA_OUT = os.path.join(HERE, "fma_800_metadata.csv")
FEATURES_OUT = os.path.join(HERE, "fma_800_features.csv")

thread_local = threading.local()

def get_remote_zip():
    if not hasattr(thread_local, "zip"):
        thread_local.zip = RemoteZip(FMA_ZIP_URL)
    return thread_local.zip

def download_track(rel_path: str, dest_dir: str):
    full_path = os.path.join(dest_dir, rel_path)
    if os.path.isfile(full_path) and os.path.getsize(full_path) > 1000:
        return rel_path, True
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    try:
        z = get_remote_zip()
        z.extract(rel_path, dest_dir)
        return rel_path, True
    except Exception as e:
        return rel_path, False

def extract_features_single_track(track_info):
    tid, rel_path = track_info
    full_path = os.path.join(FMA_AUDIO_DIR, rel_path)
    try:
        y, sr = load_audio_mono(full_path, sr=22050, duration=30.0, middle_30s=False)
        feats = extract_moodwaave_features(y, sr)
        feats["track_id"] = tid
        return feats
    except Exception as e:
        print(f"Error extracting features for track {tid} ({rel_path}): {e}")
        return None

def main():
    print("=== Step 1: FMA-small Audio & Feature Extraction (Plan A: 800 tracks) ===")
    t_start = time.perf_counter()
    log_environment(HERE)
    
    # ---------------------------------------------------------
    # 1. Stratified Sample Selection
    # ---------------------------------------------------------
    print("Loading FMA tracks.csv metadata...")
    tracks = pd.read_csv(FMA_TRACKS_CSV, index_col=0, header=[0, 1])
    small = tracks[tracks[("set", "subset")] == "small"]
    
    # 100 tracks per genre, seed 42
    subset = small.groupby(("track", "genre_top"), group_keys=False).apply(
        lambda g: g.sample(n=100, random_state=SEED)
    ).copy()
    
    # Construct clean metadata table
    meta_rows = []
    for tid, row in subset.iterrows():
        tid_int = int(tid)
        folder = f"{tid_int:06d}"[:3]
        file_name = f"{tid_int:06d}.mp3"
        rel_path = f"fma_small/{folder}/{file_name}"
        
        meta_rows.append({
            "track_id": tid_int,
            "genre": row[("track", "genre_top")],
            "artist_id": int(row[("artist", "id")]),
            "artist_name": str(row[("artist", "name")]),
            "official_split": str(row[("set", "split")]),
            "rel_path": rel_path,
        })
        
    df_meta = pd.DataFrame(meta_rows)
    df_meta.to_csv(METADATA_OUT, index=False)
    print(f"Saved FMA metadata to {METADATA_OUT}")
    print(f"Total tracks: {len(df_meta)}")
    print("Genre distribution:", df_meta["genre"].value_counts().to_dict())
    print(f"Unique artists: {df_meta['artist_id'].nunique()}")
    
    # ---------------------------------------------------------
    # 2. Download MP3 Files via Multi-Threaded RemoteZip
    # ---------------------------------------------------------
    os.makedirs(FMA_AUDIO_DIR, exist_ok=True)
    all_rel_paths = df_meta["rel_path"].tolist()
    
    # Check already downloaded
    already_dl = sum(1 for p in all_rel_paths if os.path.isfile(os.path.join(FMA_AUDIO_DIR, p)))
    print(f"Already downloaded: {already_dl} / {len(all_rel_paths)}")
    
    to_dl = [p for p in all_rel_paths if not os.path.isfile(os.path.join(FMA_AUDIO_DIR, p))]
    if to_dl:
        print(f"Downloading {len(to_dl)} tracks using 16 worker threads...")
        t_dl_start = time.perf_counter()
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_path = {executor.submit(download_track, p, FMA_AUDIO_DIR): p for p in to_dl}
            for future in concurrent.futures.as_completed(future_to_path):
                rel_p, success = future.result()
                completed += 1
                if completed % 50 == 0 or completed == len(to_dl):
                    elapsed = time.perf_counter() - t_dl_start
                    rate = completed / elapsed
                    eta_s = (len(to_dl) - completed) / rate if rate > 0 else 0
                    print(f"  Downloaded {completed}/{len(to_dl)} tracks ({rate:.1f} trk/s, ETA: {eta_s/60:.1f} min)...", flush=True)
        print(f"Completed downloading in {(time.perf_counter() - t_dl_start)/60:.2f} minutes.")
    else:
        print("All audio files already downloaded.")
        
    # ---------------------------------------------------------
    # 3. Extract Full-113 Features using All CPU Cores
    # ---------------------------------------------------------
    print(f"\nExtracting Full-113 features for {len(df_meta)} tracks on {os.cpu_count()} CPU cores...")
    t_feat_start = time.perf_counter()
    
    track_pairs = list(zip(df_meta["track_id"], df_meta["rel_path"]))
    feature_rows = []
    
    # Check existing features if script was interrupted
    if os.path.isfile(FEATURES_OUT):
        df_existing = pd.read_csv(FEATURES_OUT)
        existing_tids = set(df_existing["track_id"].tolist())
        print(f"Found {len(existing_tids)} existing feature rows in {FEATURES_OUT}")
        feature_rows = df_existing.to_dict("records")
        track_pairs = [p for p in track_pairs if p[0] not in existing_tids]
        
    if track_pairs:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_tid = {executor.submit(extract_features_single_track, p): p[0] for p in track_pairs}
            extracted_count = 0
            for future in concurrent.futures.as_completed(future_to_tid):
                feats = future.result()
                if feats is not None:
                    feature_rows.append(feats)
                extracted_count += 1
                if extracted_count % 100 == 0 or extracted_count == len(track_pairs):
                    elapsed = time.perf_counter() - t_feat_start
                    rate = extracted_count / elapsed
                    eta_s = (len(track_pairs) - extracted_count) / rate if rate > 0 else 0
                    print(f"  Extracted {extracted_count}/{len(track_pairs)} ({rate:.1f} trk/s, ETA: {eta_s/60:.1f} min)...", flush=True)
                    # Periodic checkpoint
                    pd.DataFrame(feature_rows).to_csv(FEATURES_OUT, index=False)
                    
        df_feats = pd.DataFrame(feature_rows)
        df_feats.to_csv(FEATURES_OUT, index=False)
        print(f"\nSaved {len(df_feats)} feature rows to {FEATURES_OUT}")
        print(f"Feature extraction completed in {(time.perf_counter() - t_feat_start)/60:.2f} minutes.")
    else:
        print("All features already extracted.")
        
    print(f"Step 1 finished in {(time.perf_counter() - t_start)/60:.2f} minutes.")

if __name__ == "__main__":
    main()

