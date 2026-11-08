import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoFeatureExtractor

from r3b_common import (
    BASE_DIR,
    GTZAN_AUDIO_DIR,
    PMEMO_AUDIO_DIR,
    load_gtzan_metadata,
    load_pmemo_metadata,
    load_audio_mono_30s,
)

TARGET_SR = 24000
MODEL_ID = "m-a-p/MERT-v1-95M"

def main():
    print("=== Step 2: Extract MERT-v1-95M Embeddings ===")
    t_start = time.perf_counter()
    
    print(f"Loading MERT model ({MODEL_ID})...")
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    print("MERT model loaded successfully on CPU.")
    
    # Register forward hooks on the 12 encoder layers
    layer_outputs = []
    hooks = []
    for layer in model.encoder.layers:
        def hook_fn(mod, inp, out):
            layer_outputs.append(out[0] if isinstance(out, tuple) else out)
        hooks.append(layer.register_forward_hook(hook_fn))
        
    def extract_mert_both(audio_tensor: torch.Tensor):
        layer_outputs.clear()
        inputs = processor(audio_tensor, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 1. Last hidden state mean-pooled over time:
        last_hidden = outputs.last_hidden_state # [1, T, 768]
        emb_last = last_hidden.mean(dim=1).squeeze(0).cpu().numpy() # [768]
        
        # 2. Layer-averaged representation: mean across 12 layers, then mean over time:
        stacked = torch.stack(layer_outputs) # [12, 1, T, 768]
        emb_layeravg = stacked.mean(dim=0).mean(dim=1).squeeze(0).cpu().numpy() # [768]
        
        return emb_last, emb_layeravg

    # ---------------------------------------------------------
    # 1. GTZAN
    # ---------------------------------------------------------
    gtzan_last_csv = os.path.join(BASE_DIR, "mert_gtzan_last_embeddings.csv")
    gtzan_avg_csv = os.path.join(BASE_DIR, "mert_gtzan_layeravg_embeddings.csv")
    df_g, _ = load_gtzan_metadata()
    print(f"\nProcessing GTZAN: {len(df_g)} clean tracks...")
    
    existing_g_last = {}
    existing_g_avg = {}
    if os.path.isfile(gtzan_last_csv) and os.path.isfile(gtzan_avg_csv):
        df_l = pd.read_csv(gtzan_last_csv)
        df_a = pd.read_csv(gtzan_avg_csv)
        if len(df_l) == len(df_a) and len(df_l) > 0:
            existing_g_last = {row["track_id"]: row.drop("track_id").values for _, row in df_l.iterrows()}
            existing_g_avg = {row["track_id"]: row.drop("track_id").values for _, row in df_a.iterrows()}
            print(f"Resuming GTZAN from {len(existing_g_last)} already extracted tracks.")
            
    records_g_last = []
    records_g_avg = []
    for tid, vec in existing_g_last.items():
        rec_l = {"track_id": tid}
        rec_a = {"track_id": tid}
        vec_a = existing_g_avg[tid]
        for i, val in enumerate(vec):
            rec_l[f"mert_{i:04d}"] = val
        for i, val in enumerate(vec_a):
            rec_a[f"mert_{i:04d}"] = val
        records_g_last.append(rec_l)
        records_g_avg.append(rec_a)
        
    extracted_new_g = 0
    t0_g = time.perf_counter()
    for idx, (filename, row) in enumerate(df_g.iterrows()):
        if filename in existing_g_last:
            continue
        genre = filename.split(".")[0]
        audio_path = os.path.join(GTZAN_AUDIO_DIR, genre, filename)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Missing audio: {audio_path}")
            
        audio = load_audio_mono_30s(audio_path, target_sr=TARGET_SR)
        emb_l, emb_a = extract_mert_both(audio)
        
        rec_l = {"track_id": filename}
        rec_a = {"track_id": filename}
        for i, val in enumerate(emb_l):
            rec_l[f"mert_{i:04d}"] = float(val)
        for i, val in enumerate(emb_a):
            rec_a[f"mert_{i:04d}"] = float(val)
            
        records_g_last.append(rec_l)
        records_g_avg.append(rec_a)
        existing_g_last[filename] = emb_l
        existing_g_avg[filename] = emb_a
        extracted_new_g += 1
        
        if (extracted_new_g % 25 == 0) or (idx == len(df_g) - 1):
            pd.DataFrame(records_g_last).to_csv(gtzan_last_csv, index=False)
            pd.DataFrame(records_g_avg).to_csv(gtzan_avg_csv, index=False)
            elapsed = time.perf_counter() - t0_g
            rate = elapsed / max(1, extracted_new_g)
            rem = (len(df_g) - len(existing_g_last)) * rate
            print(f"  GTZAN MERT: {len(existing_g_last)}/{len(df_g)} tracks done ({rate:.2f} s/trk, ~{rem/60:.1f}m left)", flush=True)
            
    pd.DataFrame(records_g_last).to_csv(gtzan_last_csv, index=False)
    pd.DataFrame(records_g_avg).to_csv(gtzan_avg_csv, index=False)
    print(f"GTZAN MERT embeddings saved ({len(records_g_last)} tracks, 768 dims).")
    
    # ---------------------------------------------------------
    # 2. PMEmo
    # ---------------------------------------------------------
    pmemo_last_csv = os.path.join(BASE_DIR, "mert_pmemo_last_embeddings.csv")
    pmemo_avg_csv = os.path.join(BASE_DIR, "mert_pmemo_layeravg_embeddings.csv")
    df_p, _ = load_pmemo_metadata()
    print(f"\nProcessing PMEmo: {len(df_p)} tracks...")
    
    existing_p_last = {}
    existing_p_avg = {}
    if os.path.isfile(pmemo_last_csv) and os.path.isfile(pmemo_avg_csv):
        df_l = pd.read_csv(pmemo_last_csv)
        df_a = pd.read_csv(pmemo_avg_csv)
        if len(df_l) == len(df_a) and len(df_l) > 0:
            existing_p_last = {str(row["track_id"]): row.drop("track_id").values for _, row in df_l.iterrows()}
            existing_p_avg = {str(row["track_id"]): row.drop("track_id").values for _, row in df_a.iterrows()}
            print(f"Resuming PMEmo from {len(existing_p_last)} already extracted tracks.")
            
    records_p_last = []
    records_p_avg = []
    for tid, vec in existing_p_last.items():
        rec_l = {"track_id": tid}
        rec_a = {"track_id": tid}
        vec_a = existing_p_avg[tid]
        for i, val in enumerate(vec):
            rec_l[f"mert_{i:04d}"] = val
        for i, val in enumerate(vec_a):
            rec_a[f"mert_{i:04d}"] = val
        records_p_last.append(rec_l)
        records_p_avg.append(rec_a)
        
    extracted_new_p = 0
    t0_p = time.perf_counter()
    for idx, (music_id, row) in enumerate(df_p.iterrows()):
        mid_str = str(music_id)
        if mid_str in existing_p_last:
            continue
        audio_path = os.path.join(PMEMO_AUDIO_DIR, f"{mid_str}.mp3")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Missing audio: {audio_path}")
            
        audio = load_audio_mono_30s(audio_path, target_sr=TARGET_SR)
        emb_l, emb_a = extract_mert_both(audio)
        
        rec_l = {"track_id": mid_str}
        rec_a = {"track_id": mid_str}
        for i, val in enumerate(emb_l):
            rec_l[f"mert_{i:04d}"] = float(val)
        for i, val in enumerate(emb_a):
            rec_a[f"mert_{i:04d}"] = float(val)
            
        records_p_last.append(rec_l)
        records_p_avg.append(rec_a)
        existing_p_last[mid_str] = emb_l
        existing_p_avg[mid_str] = emb_a
        extracted_new_p += 1
        
        if (extracted_new_p % 25 == 0) or (idx == len(df_p) - 1):
            pd.DataFrame(records_p_last).to_csv(pmemo_last_csv, index=False)
            pd.DataFrame(records_p_avg).to_csv(pmemo_avg_csv, index=False)
            elapsed = time.perf_counter() - t0_p
            rate = elapsed / max(1, extracted_new_p)
            rem = (len(df_p) - len(existing_p_last)) * rate
            print(f"  PMEmo MERT: {len(existing_p_last)}/{len(df_p)} tracks done ({rate:.2f} s/trk, ~{rem/60:.1f}m left)", flush=True)
            
    pd.DataFrame(records_p_last).to_csv(pmemo_last_csv, index=False)
    pd.DataFrame(records_p_avg).to_csv(pmemo_avg_csv, index=False)
    print(f"PMEmo MERT embeddings saved ({len(records_p_last)} tracks, 768 dims).")
    
    for h in hooks:
        h.remove()
        
    total_time = time.perf_counter() - t_start
    print(f"\nAll MERT-v1-95M embeddings extracted successfully in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()

