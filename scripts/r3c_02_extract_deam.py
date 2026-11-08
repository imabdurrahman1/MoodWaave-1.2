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
    DEAM_AUDIO_DIR,
    DEAM_ANNOTATIONS_CSV,
    load_audio_mono,
    extract_moodwaave_features,
)

DEAM_ZIP_URL = "https://huggingface.co/datasets/herrjyj/herrjyj-deam-assets/resolve/main/DEAM_audio.zip"
METADATA_OUT = os.path.join(HERE, "deam_metadata.csv")
FEATURES_OUT = os.path.join(HERE, "deam_features.csv")

thread_local = threading.local()

def get_remote_zip():
    if not hasattr(thread_local, "zip"):
        thread_local.zip = RemoteZip(DEAM_ZIP_URL)
    return thread_local.zip

def download_deam_track(member_name: str, dest_dir: str):
    full_path = os.path.join(dest_dir, member_name)
    if os.path.isfile(full_path) and os.path.getsize(full_path) > 1000:
        return member_name, True
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    try:
        z = get_remote_zip()
        z.extract(member_name, dest_dir)
        return member_name, True
    except Exception as e:
        return member_name, False

def extract_features_single_deam_track(track_info):
    song_id, full_path = track_info
    try:
        # Prompt explicitly specifies middle 30s of each clip
        y, sr = load_audio_mono(full_path, sr=22050, duration=30.0, middle_30s=True)
        feats = extract_moodwaave_features(y, sr)
        feats["song_id"] = song_id
        return feats
    except Exception as e:
        print(f"Error extracting features for DEAM song {song_id}: {e}")
        return None

def main():
    print("=== Step 2: DEAM Audio & Feature Extraction (All 1,802 songs) ===")
    t_start = time.perf_counter()
    
    # ---------------------------------------------------------
    # 1. Load Annotations & Normalize Targets
    # ---------------------------------------------------------
    print("Loading DEAM annotations...")
    df_raw = pd.read_csv(DEAM_ANNOTATIONS_CSV)
    
    # Normalise 1-9 scale to [0, 1] using (x - 1) / 8
    df_meta = pd.DataFrame({
        "song_id": df_raw["song_id"].astype(int),
        "valence_raw": df_raw["valence_mean"].astype(float),
        "arousal_raw": df_raw["arousal_mean"].astype(float),
        "valence": ((df_raw["valence_mean"].astype(float) - 1.0) / 8.0).clip(0.0, 1.0),
        "arousal": ((df_raw["arousal_mean"].astype(float) - 1.0) / 8.0).clip(0.0, 1.0),
        "artist": df_raw.get("artist", "Unknown"),
    })
    df_meta.to_csv(METADATA_OUT, index=False)
    print(f"Saved DEAM metadata to {METADATA_OUT}")
    print(f"Total songs: {len(df_meta)}")
    print(f"Valence normalized range: [{df_meta['valence'].min():.4f}, {df_meta['valence'].max():.4f}] (mean: {df_meta['valence'].mean():.4f})")
    print(f"Arousal normalized range: [{df_meta['arousal'].min():.4f}, {df_meta['arousal'].max():.4f}] (mean: {df_meta['arousal'].mean():.4f})")
    print(f"Artist metadata unique values: {df_meta['artist'].nunique()} ({df_meta['artist'].iloc[0]})")
    
    # ---------------------------------------------------------
    # 2. Download MP3 Files via Multi-Threaded RemoteZip
    # ---------------------------------------------------------
    os.makedirs(DEAM_AUDIO_DIR, exist_ok=True)
    
    # Audio members inside DEAM_audio.zip are MEMD_audio/{song_id}.mp3
    members_to_dl = [f"MEMD_audio/{sid}.mp3" for sid in df_meta["song_id"]]
    to_dl = [m for m in members_to_dl if not (os.path.isfile(os.path.join(DEAM_AUDIO_DIR, m)) and os.path.getsize(os.path.join(DEAM_AUDIO_DIR, m)) > 1000)]
    
    print(f"Total DEAM tracks: {len(members_to_dl)}, Already downloaded: {len(members_to_dl) - len(to_dl)}")
    if to_dl:
        print(f"Downloading {len(to_dl)} tracks using 16 worker threads from Hugging Face...")
        t_dl_start = time.perf_counter()
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_m = {executor.submit(download_deam_track, m, DEAM_AUDIO_DIR): m for m in to_dl}
            for future in concurrent.futures.as_completed(future_to_m):
                m_name, success = future.result()
                completed += 1
                if completed % 100 == 0 or completed == len(to_dl):
                    elapsed = time.perf_counter() - t_dl_start
                    rate = completed / elapsed
                    eta_s = (len(to_dl) - completed) / rate if rate > 0 else 0
                    print(f"  Downloaded {completed}/{len(to_dl)} tracks ({rate:.1f} trk/s, ETA: {eta_s/60:.1f} min)...", flush=True)
        print(f"Completed DEAM download in {(time.perf_counter() - t_dl_start)/60:.2f} minutes.")
    else:
        print("All DEAM audio files already downloaded.")
        
    # ---------------------------------------------------------
    # 3. Extract Features (Middle 30s) across All CPU Cores
    # ---------------------------------------------------------
    print(f"\nExtracting Full-113 features for {len(df_meta)} tracks on {os.cpu_count()} CPU cores...")
    t_feat_start = time.perf_counter()
    
    track_pairs = []
    for sid in df_meta["song_id"]:
        path = os.path.join(DEAM_AUDIO_DIR, f"MEMD_audio/{sid}.mp3")
        track_pairs.append((sid, path))
        
    feature_rows = []
    if os.path.isfile(FEATURES_OUT):
        df_existing = pd.read_csv(FEATURES_OUT)
        existing_sids = set(df_existing["song_id"].tolist())
        print(f"Found {len(existing_sids)} existing feature rows in {FEATURES_OUT}")
        feature_rows = df_existing.to_dict("records")
        track_pairs = [p for p in track_pairs if p[0] not in existing_sids]
        
    if track_pairs:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_sid = {executor.submit(extract_features_single_deam_track, p): p[0] for p in track_pairs}
            extracted_count = 0
            for future in concurrent.futures.as_completed(future_to_sid):
                feats = future.result()
                if feats is not None:
                    feature_rows.append(feats)
                extracted_count += 1
                if extracted_count % 150 == 0 or extracted_count == len(track_pairs):
                    elapsed = time.perf_counter() - t_feat_start
                    rate = extracted_count / elapsed
                    eta_s = (len(track_pairs) - extracted_count) / rate if rate > 0 else 0
                    print(f"  Extracted {extracted_count}/{len(track_pairs)} ({rate:.1f} trk/s, ETA: {eta_s/60:.1f} min)...", flush=True)
                    pd.DataFrame(feature_rows).to_csv(FEATURES_OUT, index=False)
                    
        df_feats = pd.DataFrame(feature_rows)
        df_feats.to_csv(FEATURES_OUT, index=False)
        print(f"\nSaved {len(df_feats)} DEAM feature rows to {FEATURES_OUT}")
        print(f"Feature extraction completed in {(time.perf_counter() - t_feat_start)/60:.2f} minutes.")
    else:
        print("All DEAM features already extracted.")
        
    print(f"Step 2 finished in {(time.perf_counter() - t_start)/60:.2f} minutes.")

if __name__ == "__main__":
    main()

