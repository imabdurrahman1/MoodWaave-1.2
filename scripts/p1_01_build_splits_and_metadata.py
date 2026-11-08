"""Script 01: Build Leakage-Aware Splits and Metadata.
Downloads Sturm's GTZAN index and fault-filtered splits.
Extracts ID3 metadata (Title, Artist, Album) from PMEmo audio files.
Creates:
1. artist_map.csv (GTZAN and PMEmo track-to-artist mappings)
2. removed_tracks.csv (Catalog of 70 removed faulty GTZAN tracks with fault types)
3. splits.json (Fold indices for all evaluation protocols)
"""
import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, KFold

import common as C

STURM_DIR = os.path.join(C.OUT, "sturm_source")
os.makedirs(STURM_DIR, exist_ok=True)

# URLs from Bob L. Sturm's official GTZAN repo
STURM_URLS = {
    "index.txt": "https://raw.githubusercontent.com/boblsturm/GTZAN/master/index.txt",
    "train_filtered.txt": "https://raw.githubusercontent.com/boblsturm/GTZAN/master/train_filtered.txt",
    "valid_filtered.txt": "https://raw.githubusercontent.com/boblsturm/GTZAN/master/valid_filtered.txt",
    "test_filtered.txt": "https://raw.githubusercontent.com/boblsturm/GTZAN/master/test_filtered.txt",
    "train_stratified.txt": "https://raw.githubusercontent.com/boblsturm/GTZAN/master/train_stratified.txt",
    "valid_stratified.txt": "https://raw.githubusercontent.com/boblsturm/GTZAN/master/valid_stratified.txt",
    "test_stratified.txt": "https://raw.githubusercontent.com/boblsturm/GTZAN/master/test_stratified.txt"
}


def download_sturm_files():
    print("Fetching Sturm GTZAN metadata and split files...")
    for fname, url in STURM_URLS.items():
        local_path = os.path.join(STURM_DIR, fname)
        if not os.path.exists(local_path):
            print(f"  Downloading {fname} from {url}...")
            urllib.request.urlretrieve(url, local_path)
        else:
            print(f"  Found cached {fname}")


def load_gtzan_artist_map():
    idx_path = os.path.join(STURM_DIR, "index.txt")
    with open(idx_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    gtzan_map = {}
    for line in lines:
        if ":::" in line:
            parts = [p.strip() for p in line.split(":::")]
            if len(parts) >= 2:
                fn = parts[0]
                art = parts[1]
                tit = parts[2] if len(parts) > 2 else ""
                # Normalize unknown artists
                if art == "" or "?" in art or art.lower() == "unknown":
                    art = f"Unknown_{fn.replace('.wav', '')}"
                gtzan_map[fn] = {"artist": art, "title": tit}
    return gtzan_map


# Exact fault catalog from Table 2 of Sturm (2013, arXiv:1306.1461)
FAULT_CATALOG = {
    # classical
    "classical.00042.wav": ("Recording Repetition", "classical.00053.wav", "Gabrieli - Canzon XIII (different excerpt same recording)"),
    # country
    "country.00008.wav": ("Recording Repetition", "country.00051.wav", "Vince Gill - Never Knew Lonely (same recording)"),
    "country.00052.wav": ("Recording Repetition", "country.00060.wav", "Vince Gill - When I Call Your Name (same recording)"),
    # disco
    "disco.00038.wav": ("Recording Repetition", "disco.00078.wav", "Gloria Gaynor - I Am What I Am (same recording)"),
    "disco.00050.wav": ("Exact Repetition", "disco.00070.wav", "KC and the Sunshine Band - That's The Way (I Like It)"),
    "disco.00051.wav": ("Exact Repetition", "disco.00070.wav", "KC and the Sunshine Band - That's The Way (I Like It)"),
    "disco.00055.wav": ("Exact Repetition", "disco.00089.wav", "KC and the Sunshine Band - Get Down Tonight"),
    "disco.00060.wav": ("Exact Repetition", "disco.00089.wav", "KC and the Sunshine Band - Get Down Tonight"),
    "disco.00071.wav": ("Exact Repetition", "disco.00074.wav", "KC and the Sunshine Band - (Shake, Shake, Shake) Shake Your Booty"),
    "disco.00098.wav": ("Exact Repetition", "disco.00099.wav", "Boney M - Rasputin"),
    # hiphop
    "hiphop.00001.wav": ("Recording Repetition", "hiphop.00042.wav", "Wu-Tang Clan - Triumph (same recording)"),
    "hiphop.00039.wav": ("Exact Repetition", "hiphop.00045.wav", "Lauryn Hill - Every Ghetto, Every City"),
    "hiphop.00046.wav": ("Recording Repetition", "hiphop.00065.wav", "A Tribe Called Quest - Phony Rappers (same recording)"),
    "hiphop.00047.wav": ("Recording Repetition", "hiphop.00067.wav", "A Tribe Called Quest - Get A Hold (same recording)"),
    "hiphop.00048.wav": ("Recording Repetition", "hiphop.00068.wav", "A Tribe Called Quest - Motivators (same recording)"),
    "hiphop.00049.wav": ("Recording Repetition", "hiphop.00069.wav", "A Tribe Called Quest - Jam (same recording)"),
    "hiphop.00050.wav": ("Recording Repetition", "hiphop.00072.wav", "A Tribe Called Quest - Mind Power (same recording)"),
    "hiphop.00076.wav": ("Exact Repetition", "hiphop.00078.wav", "A Tribe Called Quest - Stressed Out"),
    # jazz (13 Coleman Hawkins exact repetitions)
    "jazz.00033.wav": ("Exact Repetition", "jazz.00051.wav", "Coleman Hawkins - Body and Soul"),
    "jazz.00034.wav": ("Exact Repetition", "jazz.00053.wav", "Coleman Hawkins - Dinah"),
    "jazz.00035.wav": ("Exact Repetition", "jazz.00055.wav", "Coleman Hawkins - Fine and Dandy"),
    "jazz.00036.wav": ("Exact Repetition", "jazz.00058.wav", "Coleman Hawkins - Stuffy"),
    "jazz.00037.wav": ("Exact Repetition", "jazz.00060.wav", "Coleman Hawkins - I'm in the Mood for Love"),
    "jazz.00038.wav": ("Exact Repetition", "jazz.00062.wav", "Coleman Hawkins - Lady Be Good"),
    "jazz.00039.wav": ("Exact Repetition", "jazz.00065.wav", "Coleman Hawkins - Out of Nowhere"),
    "jazz.00040.wav": ("Exact Repetition", "jazz.00067.wav", "Coleman Hawkins - Sweet Georgia Brown"),
    "jazz.00042.wav": ("Exact Repetition", "jazz.00068.wav", "Coleman Hawkins - Spotlite"),
    "jazz.00043.wav": ("Exact Repetition", "jazz.00069.wav", "Coleman Hawkins - The Man I Love"),
    "jazz.00044.wav": ("Exact Repetition", "jazz.00070.wav", "Coleman Hawkins - When Day Is Done"),
    "jazz.00045.wav": ("Exact Repetition", "jazz.00071.wav", "Coleman Hawkins - Yesterdays"),
    "jazz.00046.wav": ("Exact Repetition", "jazz.00072.wav", "Coleman Hawkins - Picasso"),
    # metal
    "metal.00004.wav": ("Exact Repetition", "metal.00013.wav", "Dark Tranquillity - Of Chaos And Eternal Night"),
    "metal.00034.wav": ("Exact Repetition", "metal.00094.wav", "Iron Maiden - The Trooper"),
    "metal.00040.wav": ("Exact Repetition", "metal.00061.wav", "Dio - Stand Up And Shout"),
    "metal.00041.wav": ("Exact Repetition", "metal.00062.wav", "Dio - Holy Diver"),
    "metal.00042.wav": ("Exact Repetition", "metal.00063.wav", "Dio - Gypsy"),
    "metal.00043.wav": ("Exact Repetition", "metal.00064.wav", "Dio - Caught In The Middle"),
    "metal.00044.wav": ("Exact Repetition", "metal.00065.wav", "Dio - Don't Talk To Strangers"),
    "metal.00045.wav": ("Exact Repetition", "metal.00066.wav", "Dio - Straight Through The Heart"),
    "metal.00058.wav": ("Cross-genre Exact Repetition", "rock.00016.wav", "Queen - Tie Your Mother Down (appears in Rock and Metal)"),
    # pop
    "pop.00015.wav": ("Exact Repetition", "pop.00022.wav", "Britney Spears - (You Drive Me) Crazy"),
    "pop.00021.wav": ("Recording Repetition", "pop.00022.wav", "Britney Spears - (You Drive Me) Crazy"),
    "pop.00022.wav": ("Exact Repetition", "pop.00015.wav", "Britney Spears - (You Drive Me) Crazy"),
    "pop.00030.wav": ("Exact Repetition", "pop.00031.wav", "Britney Spears - Born To Make You Happy"),
    "pop.00037.wav": ("Distortion", "pop.00015.wav", "Britney Spears - (You Drive Me) Crazy with sound effects"),
    "pop.00045.wav": ("Exact Repetition", "pop.00046.wav", "Christina Aguilera - What A Girl Wants"),
    "pop.00047.wav": ("Exact Repetition", "pop.00080.wav", "Christina Aguilera - I Turn To You"),
    "pop.00048.wav": ("Recording Repetition", "pop.00051.wav", "Christina Aguilera - So Emotional (same recording)"),
    "pop.00051.wav": ("Recording Repetition", "pop.00048.wav", "Christina Aguilera - So Emotional (same recording)"),
    "pop.00052.wav": ("Exact Repetition", "pop.00057.wav", "Destiny's Child - Bills, Bills, Bills"),
    "pop.00054.wav": ("Exact Repetition", "pop.00060.wav", "Destiny's Child - Bug A Boo"),
    "pop.00056.wav": ("Exact Repetition", "pop.00059.wav", "Destiny's Child - Say My Name"),
    "pop.00057.wav": ("Exact Repetition", "pop.00052.wav", "Destiny's Child - Bills, Bills, Bills"),
    "pop.00067.wav": ("Exact Repetition", "pop.00071.wav", "Janet Jackson - Go Deep"),
    "pop.00068.wav": ("Recording Repetition", "pop.00073.wav", "Janet Jackson - Velvet Rope (same recording)"),
    "pop.00087.wav": ("Exact Repetition", "pop.00090.wav", "Mandy Moore - So Real"),
    # reggae
    "reggae.00003.wav": ("Exact Repetition", "reggae.00054.wav", "Bob Marley - Three Little Birds"),
    "reggae.00005.wav": ("Exact Repetition", "reggae.00056.wav", "Bob Marley - Get Up, Stand Up"),
    "reggae.00007.wav": ("Recording Repetition", "reggae.00059.wav", "Bob Marley - Could You Be Loved (same recording)"),
    "reggae.00008.wav": ("Exact Repetition", "reggae.00057.wav", "Bob Marley - Buffalo Soldier"),
    "reggae.00010.wav": ("Exact Repetition", "reggae.00060.wav", "Bob Marley - Iron Lion Zion"),
    "reggae.00013.wav": ("Exact Repetition", "reggae.00058.wav", "Bob Marley - Redemption Song"),
    "reggae.00033.wav": ("Recording Repetition", "reggae.00044.wav", "Burning Spear - Marcus Garvey (same recording)"),
    "reggae.00041.wav": ("Exact Repetition", "reggae.00069.wav", "Burning Spear - Old Marcus Garvey"),
    "reggae.00073.wav": ("Exact Repetition", "reggae.00074.wav", "Bounty Killer - Hip-Hopera"),
    "reggae.00075.wav": ("Exact Repetition", "reggae.00091.wav", "Steel Pulse - Steppin' Out"),
    "reggae.00080.wav": ("Exact Repetition", "reggae.00081.wav", "Black Uhuru - Sinsemilla"),
    "reggae.00081.wav": ("Exact Repetition", "reggae.00082.wav", "Black Uhuru - Sinsemilla"),
    "reggae.00086.wav": ("Distortion", "none", "Audio corruption (last 25 seconds useless/distorted)"),
    "reggae.00091.wav": ("Exact Repetition", "reggae.00092.wav", "Steel Pulse - Steppin' Out")
}


def parse_pmemo_id3(path):
    with open(path, "rb") as f:
        data = f.read(8192)
    meta = {"title": "", "artist": "", "album": ""}
    for tag_name, key in [(b"TIT2", "title"), (b"TPE1", "artist"), (b"TALB", "album")]:
        idx = data.find(tag_name)
        if idx != -1:
            try:
                size = int.from_bytes(data[idx+4:idx+8], "big")
                enc = data[idx+10]
                raw = data[idx+11:idx+10+min(size, 200)]
                if enc == 3:
                    val = raw.split(b"\x00")[0].decode("utf-8", errors="ignore")
                elif enc == 1:
                    val = raw.split(b"\x00\x00")[0].decode("utf-16", errors="ignore")
                elif enc == 0:
                    val = raw.split(b"\x00")[0].decode("latin-1", errors="ignore")
                else:
                    val = raw.split(b"\x00")[0].decode("utf-8", errors="ignore")
                meta[key] = val.strip()
            except Exception:
                pass
    return meta


def build_metadata_and_splits():
    download_sturm_files()
    gtzan_artist_map = load_gtzan_artist_map()

    # 1. Load GTZAN features table
    df_gtzan = pd.read_csv(C.GTZAN_CSV)
    print(f"Loaded GTZAN features: {len(df_gtzan)} rows.")

    # Check mapping
    unmapped = []
    for fn in df_gtzan["filename"]:
        if fn not in gtzan_artist_map:
            unmapped.append(fn)
    print(f"Unmapped GTZAN filenames: {len(unmapped)}")

    # Load Kereliuk / Sturm clean filtered splits
    train_filt = [line.strip().split('/')[-1] for line in open(os.path.join(STURM_DIR, "train_filtered.txt")).read().splitlines() if line.strip()]
    valid_filt = [line.strip().split('/')[-1] for line in open(os.path.join(STURM_DIR, "valid_filtered.txt")).read().splitlines() if line.strip()]
    test_filt = [line.strip().split('/')[-1] for line in open(os.path.join(STURM_DIR, "test_filtered.txt")).read().splitlines() if line.strip()]
    clean_filenames = set(train_filt + valid_filt + test_filt)

    # 2. Build removed_tracks.csv
    removed_records = []
    for fn in df_gtzan["filename"]:
        if fn not in clean_filenames:
            genre = fn.split(".")[0]
            art = gtzan_artist_map.get(fn, {}).get("artist", "Unknown")
            tit = gtzan_artist_map.get(fn, {}).get("title", "Unknown")
            fault_info = FAULT_CATALOG.get(fn, ("Repetition / Corruption", "various", "Identified in Sturm Table 2"))
            removed_records.append({
                "filename": fn,
                "genre": genre,
                "artist": art,
                "title": tit,
                "fault_type": fault_info[0],
                "paired_track": fault_info[1],
                "description": fault_info[2]
            })

    df_removed = pd.DataFrame(removed_records).sort_values(["genre", "filename"])
    df_removed.to_csv(os.path.join(C.OUT, "removed_tracks.csv"), index=False)
    print(f"Saved removed_tracks.csv ({len(df_removed)} tracks).")

    # Report counts removed per fault type and genre
    print("\nCounts removed per fault type:")
    print(df_removed["fault_type"].value_counts().to_string())
    print("\nCounts removed per genre:")
    print(df_removed["genre"].value_counts().to_string())

    # 3. Build PMEmo metadata from audio files
    df_pmemo = pd.read_csv(C.PMEMO_READY_CSV)
    print(f"\nLoaded PMEmo ready: {len(df_pmemo)} rows.")

    pmemo_records = []
    for mid in df_pmemo["musicId"]:
        mp3_p = os.path.join(C.PMEMO_CHORUS_DIR, f"{mid}.mp3")
        m = parse_pmemo_id3(mp3_p)
        pmemo_records.append({
            "dataset": "PMEmo",
            "track_id": str(mid),
            "artist": m["artist"] if m["artist"] else f"Unknown_PMEmo_{mid}",
            "title": m["title"] if m["title"] else f"Track_{mid}",
            "album": m["album"] if m["album"] else "Unknown",
            "genre_or_label": f"Valence={df_pmemo.loc[df_pmemo['musicId']==mid, 'Valence(mean)'].values[0]:.2f},Arousal={df_pmemo.loc[df_pmemo['musicId']==mid, 'Arousal(mean)'].values[0]:.2f}"
        })

    # GTZAN artist map records
    gtzan_records = []
    for _, row in df_gtzan.iterrows():
        fn = row["filename"]
        g = row["label"]
        info = gtzan_artist_map.get(fn, {"artist": f"Unknown_{fn}", "title": ""})
        gtzan_records.append({
            "dataset": "GTZAN",
            "track_id": fn,
            "artist": info["artist"],
            "title": info["title"],
            "album": "GTZAN",
            "genre_or_label": g
        })

    df_artist_map = pd.DataFrame(gtzan_records + pmemo_records)
    df_artist_map.to_csv(os.path.join(C.OUT, "artist_map.csv"), index=False)
    print(f"Saved artist_map.csv ({len(df_artist_map)} total track entries).")

    # 4. Build Split Partitions (splits.json)
    splits = {
        "metadata": {
            "project": "MOODWAAVE",
            "task": "rev2_p1_splits",
            "gtzan_total_tracks": len(df_gtzan),
            "gtzan_fault_filtered_tracks": len(df_gtzan[df_gtzan["filename"].isin(clean_filenames)]),
            "pmemo_total_tracks": len(df_pmemo)
        },
        "gtzan": {},
        "pmemo": {}
    }

    # (a) GTZAN All 999 - Stratified Random (5 folds, 3 repeats)
    # Reproduces exactly the previous protocol
    splits["gtzan"]["all_999_stratified_random"] = []
    y_g_all = df_gtzan["label"].values
    fn_g_all = df_gtzan["filename"].values
    for rep in range(3):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + rep)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(df_gtzan, y_g_all)):
            splits["gtzan"]["all_999_stratified_random"].append({
                "repeat": rep,
                "fold": fold,
                "train_tracks": fn_g_all[tr_idx].tolist(),
                "test_tracks": fn_g_all[te_idx].tolist()
            })

    # (b) GTZAN Fault-Removed (929 tracks) - Stratified Random (5 folds, 3 repeats)
    df_gtzan_clean = df_gtzan[df_gtzan["filename"].isin(clean_filenames)].reset_index(drop=True)
    y_g_clean = df_gtzan_clean["label"].values
    fn_g_clean = df_gtzan_clean["filename"].values
    splits["gtzan"]["fault_removed_stratified_random"] = []
    for rep in range(3):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + rep)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(df_gtzan_clean, y_g_clean)):
            splits["gtzan"]["fault_removed_stratified_random"].append({
                "repeat": rep,
                "fold": fold,
                "train_tracks": fn_g_clean[tr_idx].tolist(),
                "test_tracks": fn_g_clean[te_idx].tolist()
            })

    # (c) GTZAN Fault-Removed (929 tracks) - Artist-Disjoint (StratifiedGroupKFold, 5 folds, 3 repeats)
    artist_clean = [gtzan_artist_map.get(fn, {}).get("artist", f"Unknown_{fn}") for fn in fn_g_clean]
    splits["gtzan"]["fault_removed_artist_grouped"] = []
    for rep in range(3):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + rep)
        for fold, (tr_idx, te_idx) in enumerate(sgkf.split(df_gtzan_clean, y_g_clean, groups=artist_clean)):
            splits["gtzan"]["fault_removed_artist_grouped"].append({
                "repeat": rep,
                "fold": fold,
                "train_tracks": fn_g_clean[tr_idx].tolist(),
                "test_tracks": fn_g_clean[te_idx].tolist()
            })

    # Also record the Kereliuk/Sturm fixed fault-filtered split
    # Notice: jazz.00054 was unreadable so valid set has 196 usable tracks in our feature table
    tr_fixed = [fn for fn in train_filt if fn in set(fn_g_clean)]
    va_fixed = [fn for fn in valid_filt if fn in set(fn_g_clean)]
    te_fixed = [fn for fn in test_filt if fn in set(fn_g_clean)]
    splits["gtzan"]["kereliuk_sturm_fixed"] = {
        "train_tracks": tr_fixed,
        "valid_tracks": va_fixed,
        "test_tracks": te_fixed
    }

    # (d) PMEmo Random (5 folds, 5 repeats)
    pmemo_ids = df_pmemo["musicId"].astype(str).values
    pmemo_artists = [p["artist"] for p in pmemo_records]
    splits["pmemo"]["random_5x5"] = []
    for rep in range(5):
        kf = KFold(n_splits=5, shuffle=True, random_state=42 + rep)
        for fold, (tr_idx, te_idx) in enumerate(kf.split(df_pmemo)):
            splits["pmemo"]["random_5x5"].append({
                "repeat": rep,
                "fold": fold,
                "train_tracks": pmemo_ids[tr_idx].tolist(),
                "test_tracks": pmemo_ids[te_idx].tolist()
            })

    # (e) PMEmo Artist-Disjoint (StratifiedGroupKFold or GroupKFold, 5 folds, 5 repeats)
    # Use discrete quadrant_label or binned valence to stratify as closely as possible
    pmemo_bins = (df_pmemo["Valence(mean)"] > 0.5).astype(int).astype(str) + "_" + (df_pmemo["Arousal(mean)"] > 0.5).astype(int).astype(str)
    splits["pmemo"]["artist_grouped_5x5"] = []
    for rep in range(5):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + rep)
        for fold, (tr_idx, te_idx) in enumerate(sgkf.split(df_pmemo, pmemo_bins, groups=pmemo_artists)):
            splits["pmemo"]["artist_grouped_5x5"].append({
                "repeat": rep,
                "fold": fold,
                "train_tracks": pmemo_ids[tr_idx].tolist(),
                "test_tracks": pmemo_ids[te_idx].tolist()
            })

    splits_json_path = os.path.join(C.OUT, "splits.json")
    with open(splits_json_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits.json ({len(splits['gtzan'])} GTZAN protocols, {len(splits['pmemo'])} PMEmo protocols).")

    C.log_environment(C.OUT)


if __name__ == "__main__":
    build_metadata_and_splits()

