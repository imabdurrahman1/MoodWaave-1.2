"""
experiments/rev2_p5_deep/01_extract_vggish_embeddings.py
Extracts VGGish frame embeddings (128-dim per 0.96s frame) for all tracks,
applies mean + std pooling across time frames -> 256-dimensional track embedding,
and caches embeddings to CSV files.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common

def extract_embeddings_for_dataset(
    dataset_name: str,
    track_list: list,
    get_audio_path_fn,
    labels_df: pd.DataFrame,
    output_csv: str,
    vggish_model,
):
    """
    Extracts VGGish 256-dim embeddings for track_list, saves to output_csv.
    Resumes incrementally if output_csv already exists.
    """
    existing_records = []
    completed_ids = set()
    if os.path.exists(output_csv):
        df_exist = pd.read_csv(output_csv)
        completed_ids = set(df_exist["track_id"].astype(str))
        existing_records = df_exist.to_dict("records")
        print(f"[{dataset_name}] Found existing cache with {len(completed_ids)} completed tracks.")
        
    remaining_tracks = [t for t in track_list if str(t) not in completed_ids]
    print(f"[{dataset_name}] Processing {len(remaining_tracks)} remaining tracks out of {len(track_list)}...")
    
    records = list(existing_records)
    t0 = time.time()
    
    for i, track_id in enumerate(remaining_tracks):
        audio_path = get_audio_path_fn(str(track_id))
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for track {track_id}: {audio_path}")
            continue
            
        try:
            with torch.no_grad():
                frame_embs = vggish_model(audio_path)  # shape: (n_frames, 128)
                
            if isinstance(frame_embs, torch.Tensor):
                frame_embs_np = frame_embs.cpu().numpy()
            else:
                frame_embs_np = np.asarray(frame_embs)
                
            mean_emb = np.mean(frame_embs_np, axis=0)  # (128,)
            std_emb = np.std(frame_embs_np, axis=0)   # (128,)
            pooled_emb = np.concatenate([mean_emb, std_emb])  # (256,)
            
            row = {"track_id": str(track_id)}
            for dim_idx, val in enumerate(pooled_emb):
                row[f"vggish_{dim_idx:03d}"] = float(val)
                
            records.append(row)
            
            if (i + 1) % 50 == 0 or (i + 1) == len(remaining_tracks):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                rem_sec = (len(remaining_tracks) - (i + 1)) / rate
                print(f"[{dataset_name}] Processed {i+1}/{len(remaining_tracks)} tracks ({rate:.1f} trk/s, est. {rem_sec/60.0:.1f} min left)...")
                pd.DataFrame(records).to_csv(output_csv, index=False)
                
        except Exception as e:
            print(f"Error extracting embedding for {track_id}: {e}")
            
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    print(f"[{dataset_name}] Extraction complete! Saved {len(df_out)} track embeddings to {output_csv}")
    return df_out

def main():
    out_dir = common.BASE_DIR
    common.log_environment(out_dir)
    
    print("Loading pretrained VGGish via PyTorch Hub...")
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish.eval()
    
    # 1. GTZAN Clean Tracks (929 tracks)
    df_g = pd.read_csv(common.GTZAN_FEATURES_CSV).set_index("filename")
    rem_tracks = set(pd.read_csv(common.REMOVED_TRACKS_CSV)["filename"])
    clean_gtzan = df_g.loc[~df_g.index.isin(rem_tracks)]
    gtzan_tracks = list(clean_gtzan.index)
    gtzan_csv = os.path.join(out_dir, "vggish_gtzan_embeddings.csv")
    
    print(f"\n=== Extracting VGGish Embeddings for GTZAN ({len(gtzan_tracks)} clean tracks) ===")
    extract_embeddings_for_dataset(
        dataset_name="GTZAN",
        track_list=gtzan_tracks,
        get_audio_path_fn=common.get_gtzan_audio_path,
        labels_df=clean_gtzan,
        output_csv=gtzan_csv,
        vggish_model=vggish,
    )
    
    # 2. PMEmo Tracks (767 tracks)
    df_p = pd.read_csv(common.PMEMO_READY_CSV)
    pmemo_tracks = [str(mid) for mid in df_p["musicId"]]
    pmemo_csv = os.path.join(out_dir, "vggish_pmemo_embeddings.csv")
    
    print(f"\n=== Extracting VGGish Embeddings for PMEmo ({len(pmemo_tracks)} tracks) ===")
    extract_embeddings_for_dataset(
        dataset_name="PMEmo",
        track_list=pmemo_tracks,
        get_audio_path_fn=common.get_pmemo_audio_path,
        labels_df=df_p,
        output_csv=pmemo_csv,
        vggish_model=vggish,
    )

if __name__ == "__main__":
    main()

