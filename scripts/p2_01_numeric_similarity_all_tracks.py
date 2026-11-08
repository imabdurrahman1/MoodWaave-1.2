"""Step 1 (Fix): Recompute V1 and V2 vs V0 numeric similarity on ALL tracks of both datasets.
Datasets:
- GTZAN: 999 usable tracks
- PMEmo: 767 tracks
Total: 1,766 tracks

V0: Baseline HPSS + CQT Tonnetz (pre-extracted in gtzan_features.csv and pmemo_ready.csv)
V1: Shared STFT chroma Tonnetz (pre-extracted in gtzan_tonnetz_v1.csv and pmemo_tonnetz_v1.csv)
V2: Harmonic STFT chroma Tonnetz (librosa.decompose.hpss(S) -> chroma_stft -> tonnetz)

Uses ProcessPoolExecutor for parallel extraction across available CPU cores.
Outputs:
- numeric_similarity_all_tracks.csv (correlation r, Spearman rho, MAE per feature across all tracks)
- tonnetz_all_tracks_features.csv (raw feature values for all 1,766 tracks)
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import common as C

TONNETZ_COLS = [f"tonnetz_{stat}_{i+1}" for stat in ["mean", "std"] for i in range(6)]
TONNETZ_COLS = sorted(TONNETZ_COLS, key=lambda x: (int(x.split("_")[-1]), x.split("_")[1]))


def extract_v2_single(args):
    dataset, track_id, path = args
    try:
        y, sr = librosa.load(path, sr=22050, duration=30.0)
        S = np.abs(librosa.stft(y))
        H, _ = librosa.decompose.hpss(S)
        chr_harm = librosa.feature.chroma_stft(S=H**2, sr=sr)
        t_v2 = librosa.feature.tonnetz(chroma=chr_harm, sr=sr)
        
        row = {"dataset": dataset, "track_id": track_id}
        for i in range(6):
            row[f"v2_tonnetz_mean_{i+1}"] = float(np.mean(t_v2[i]))
            row[f"v2_tonnetz_std_{i+1}"] = float(np.std(t_v2[i]))
        return row
    except Exception as e:
        print(f"Error on {dataset} {track_id}: {e}", flush=True)
        return None


def main():
    print("=" * 80)
    print("MOODWAAVE - TASK P2-FIX: ALL-TRACKS TONNETZ NUMERIC SIMILARITY (1,766 TRACKS)")
    print("=" * 80)
    t0 = time.time()

    # 1. Load GTZAN V0 and V1
    df_gtz_v0 = pd.read_csv(C.GTZAN_CSV).set_index("filename")
    df_gtz_v1 = pd.read_csv(os.path.join(C.OUT, "gtzan_tonnetz_v1.csv")).set_index("filename")

    # 2. Load PMEmo V0 and V1
    df_pme_v0 = pd.read_csv(C.PMEMO_CSV)
    df_pme_v0["musicId_str"] = df_pme_v0["musicId"].astype(str)
    df_pme_v0.set_index("musicId_str", inplace=True)

    df_pme_v1 = pd.read_csv(os.path.join(C.OUT, "pmemo_tonnetz_v1.csv"))
    df_pme_v1["musicId_str"] = df_pme_v1["musicId"].astype(str)
    df_pme_v1.set_index("musicId_str", inplace=True)

    # Prepare extraction tasks for V2
    tasks = []
    
    # GTZAN tasks
    df_gtz_meta = pd.read_csv(C.GTZAN_CSV)[["filename", "label"]]
    for _, row in df_gtz_meta.iterrows():
        fn = row["filename"]
        label = row["label"]
        p = os.path.join(C.GTZAN_DIR, label, fn)
        tasks.append(("GTZAN", fn, p))

    # PMEmo tasks
    df_pme_meta = pd.read_csv(C.PMEMO_CSV)[["musicId"]]
    for _, row in df_pme_meta.iterrows():
        mid = str(row["musicId"])
        p = os.path.join(C.PMEMO_DIR, f"{mid}.mp3")
        tasks.append(("PMEmo", mid, p))

    print(f"Total tracks to extract for V2: {len(tasks)} (999 GTZAN + 767 PMEmo)")
    print(f"Using ProcessPoolExecutor with 6 parallel workers...", flush=True)

    v2_results = {}
    completed = 0
    t_start_v2 = time.time()

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(extract_v2_single, task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                key = (res["dataset"], res["track_id"])
                v2_results[key] = res
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                elapsed = time.time() - t_start_v2
                rate = completed / elapsed
                rem = (len(tasks) - completed) / rate if rate > 0 else 0
                print(f"  V2 Extraction: {completed}/{len(tasks)} tracks ({elapsed:.1f}s, ~{rem:.0f}s remaining)", flush=True)

    print(f"\nV2 extraction completed in {time.time() - t_start_v2:.1f}s. Assembling full feature dataset...", flush=True)

    # Assemble unified rows
    unified_rows = []
    for dataset, track_id, _ in tasks:
        key = (dataset, track_id)
        if key not in v2_results:
            continue
        v2_data = v2_results[key]

        row = {"dataset": dataset, "track_id": track_id}
        
        # Pull V0 and V1
        if dataset == "GTZAN":
            v0_src = df_gtz_v0.loc[track_id]
            v1_src = df_gtz_v1.loc[track_id]
        else:
            v0_src = df_pme_v0.loc[track_id]
            v1_src = df_pme_v1.loc[track_id]

        for col in TONNETZ_COLS:
            row[f"v0_{col}"] = float(v0_src[col])
            row[f"v1_{col}"] = float(v1_src[col])
            row[f"v2_{col}"] = float(v2_data[f"v2_{col}"])

        unified_rows.append(row)

    df_all = pd.DataFrame(unified_rows)
    raw_path = os.path.join(C.OUT, "tonnetz_all_tracks_features.csv")
    df_all.to_csv(raw_path, index=False)
    print(f"Saved {raw_path} with {len(df_all)} tracks.")

    # Compute correlation metrics across ALL tracks
    sim_records = []
    for col in TONNETZ_COLS:
        v0 = df_all[f"v0_{col}"].values
        v1 = df_all[f"v1_{col}"].values
        v2 = df_all[f"v2_{col}"].values

        # V1 vs V0
        r_v1, _ = stats.pearsonr(v1, v0)
        rho_v1, _ = stats.spearmanr(v1, v0)
        mae_v1 = float(np.mean(np.abs(v1 - v0)))

        # V2 vs V0
        r_v2, _ = stats.pearsonr(v2, v0)
        rho_v2, _ = stats.spearmanr(v2, v0)
        mae_v2 = float(np.mean(np.abs(v2 - v0)))

        # Dataset-specific correlations
        gtz_mask = df_all["dataset"] == "GTZAN"
        pme_mask = df_all["dataset"] == "PMEmo"

        r_v1_gtz, _ = stats.pearsonr(df_all.loc[gtz_mask, f"v1_{col}"], df_all.loc[gtz_mask, f"v0_{col}"])
        r_v1_pme, _ = stats.pearsonr(df_all.loc[pme_mask, f"v1_{col}"], df_all.loc[pme_mask, f"v0_{col}"])

        sim_records.append({
            "feature": col,
            "v0_mean": float(np.mean(v0)),
            "v0_std": float(np.std(v0, ddof=1)),
            "v1_mean": float(np.mean(v1)),
            "v1_std": float(np.std(v1, ddof=1)),
            "v2_mean": float(np.mean(v2)),
            "v2_std": float(np.std(v2, ddof=1)),
            "r_v1_v0": round(r_v1, 4),
            "rho_v1_v0": round(rho_v1, 4),
            "mae_v1_v0": round(mae_v1, 4),
            "r_v2_v0": round(r_v2, 4),
            "rho_v2_v0": round(rho_v2, 4),
            "mae_v2_v0": round(mae_v2, 4),
            "r_v1_v0_gtzan": round(r_v1_gtz, 4),
            "r_v1_v0_pmemo": round(r_v1_pme, 4)
        })

    df_sim = pd.DataFrame(sim_records)
    sim_path = os.path.join(C.OUT, "numeric_similarity_all_tracks.csv")
    df_sim.to_csv(sim_path, index=False)
    # Also overwrite numeric_similarity.csv for seamless integration
    df_sim.to_csv(os.path.join(C.OUT, "numeric_similarity.csv"), index=False)
    print(f"Saved {sim_path} and updated numeric_similarity.csv.")

    print("\n" + "=" * 80)
    print("ALL-TRACKS TONNETZ SIMILARITY RESULTS (N=1,766):")
    print("=" * 80)
    print(df_sim[["feature", "r_v1_v0", "rho_v1_v0", "mae_v1_v0", "r_v2_v0", "rho_v2_v0"]].to_string(index=False))
    print(f"\nMean Pearson r (V1 vs V0): {df_sim['r_v1_v0'].mean():.4f}")
    print(f"Mean Spearman rho (V1 vs V0): {df_sim['rho_v1_v0'].mean():.4f}")
    print(f"Mean Pearson r (V2 vs V0): {df_sim['r_v2_v0'].mean():.4f}")
    print(f"Mean Spearman rho (V2 vs V0): {df_sim['rho_v2_v0'].mean():.4f}")
    print(f"Total time elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

