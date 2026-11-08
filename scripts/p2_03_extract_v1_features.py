"""Step 3: Extract Tonnetz V1 features for all GTZAN and PMEmo tracks.
Tonnetz V1 uses the shared STFT chroma:
  S = np.abs(librosa.stft(y))
  chroma = librosa.feature.chroma_stft(S=S**2, sr=sr)
  tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)
Bypasses both HPSS and CQT, taking ~70ms per track.
Outputs:
- gtzan_tonnetz_v1.csv (999 tracks)
- pmemo_tonnetz_v1.csv (767 tracks)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import librosa

import common as C


def extract_tonnetz_v1(path):
    try:
        y, sr = C.load_audio(path)
        S = np.abs(librosa.stft(y))
        chr_mat = librosa.feature.chroma_stft(S=S**2, sr=sr)
        t_v1 = librosa.feature.tonnetz(chroma=chr_mat, sr=sr)
        return C._ms("tonnetz", t_v1)
    except Exception as e:
        print(f"Error extracting {path}: {e}")
        return None


def process_gtzan():
    print("\n--- Extracting Tonnetz V1 for GTZAN (999 tracks) ---")
    df_gtz = pd.read_csv(C.GTZAN_CSV)
    out_rows = []
    t0 = time.time()
    for idx, row in df_gtz.iterrows():
        fn = row["filename"]
        label = row["label"]
        p = os.path.join(C.GTZAN_DIR, label, fn)
        feats = extract_tonnetz_v1(p)
        if feats:
            feats["filename"] = fn
            out_rows.append(feats)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(df_gtz):
            print(f"  GTZAN: {idx+1}/{len(df_gtz)} ({time.time() - t0:.1f}s)", flush=True)

    df_out = pd.DataFrame(out_rows)
    out_path = os.path.join(C.OUT, "gtzan_tonnetz_v1.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(df_out)} tracks.")


def process_pmemo():
    print("\n--- Extracting Tonnetz V1 for PMEmo (767 tracks) ---")
    df_pme = pd.read_csv(C.PMEMO_CSV)
    out_rows = []
    t0 = time.time()
    for idx, row in df_pme.iterrows():
        mid = int(row["musicId"])
        p = os.path.join(C.PMEMO_DIR, f"{mid}.mp3")
        feats = extract_tonnetz_v1(p)
        if feats:
            feats["musicId"] = mid
            out_rows.append(feats)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(df_pme):
            print(f"  PMEmo: {idx+1}/{len(df_pme)} ({time.time() - t0:.1f}s)", flush=True)

    df_out = pd.DataFrame(out_rows)
    out_path = os.path.join(C.OUT, "pmemo_tonnetz_v1.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(df_out)} tracks.")


def main():
    print("=" * 80)
    print("MOODWAAVE - TASK P2.3: FULL DATASET TONNETZ V1 FEATURE EXTRACTION")
    print("=" * 80)
    process_gtzan()
    process_pmemo()
    print("\nCompleted extraction of Tonnetz V1 features for all tracks.")


if __name__ == "__main__":
    main()

