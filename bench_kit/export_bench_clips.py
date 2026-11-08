#!/usr/bin/env python3
"""
release/bench_kit/export_bench_clips.py

MOODWAAVE Platform Benchmark Kit: Clip Exporter
Exports the canonical 50 GTZAN WAV audio clips to a target directory.
Adheres to the MOODWAAVE timing benchmark protocol.
"""

import os
import sys
import shutil
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# Canonical GTZAN source directories
DEFAULT_SAMPLE_JSON = os.path.join(PROJECT_ROOT, "experiments", "idea2_cost_pareto", "sample_ids.json")
DEFAULT_GTZAN_DIR = os.path.join(PROJECT_ROOT, "data", "gtzan", "genres_original")
DEFAULT_DEST_DIR = os.path.join(HERE, "audio_50")

def get_canonical_clip_paths(sample_json_path, gtzan_dir, count=50):
    """Retrieve the exact list of the first 50 canonical GTZAN clips."""
    clips = []
    
    if os.path.isfile(sample_json_path):
        with open(sample_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "GTZAN" in data:
            raw_list = data["GTZAN"][:count]
            for p in raw_list:
                # Handle path resolution across systems
                if os.path.isfile(p):
                    clips.append(p)
                else:
                    # Try resolving relative to gtzan_dir
                    fname = os.path.basename(p)
                    # Genre is prefix before dot e.g. pop.00094.wav -> pop
                    genre = fname.split(".")[0]
                    alt_path = os.path.join(gtzan_dir, genre, fname)
                    if os.path.isfile(alt_path):
                        clips.append(alt_path)
    
    # Fallback if sample_ids.json not found or missing tracks: deterministic sorted walk
    if len(clips) < count and os.path.isdir(gtzan_dir):
        print("Notice: Locating additional clips via deterministic sorted walk in GTZAN directory...")
        all_wavs = []
        for root, _, files in os.walk(gtzan_dir):
            for file in sorted(files):
                if file.endswith(".wav"):
                    all_wavs.append(os.path.join(root, file))
        all_wavs.sort()
        for p in all_wavs:
            if p not in clips:
                clips.append(p)
            if len(clips) == count:
                break
                
    if len(clips) < count:
        raise FileNotFoundError(f"Could only find {len(clips)} GTZAN audio clips (needed {count}).")
        
    return clips[:count]

def export_clips(dest_dir, count=50, sample_json=DEFAULT_SAMPLE_JSON, gtzan_dir=DEFAULT_GTZAN_DIR):
    """Copy the 50 clips and generate a bench_manifest.json."""
    os.makedirs(dest_dir, exist_ok=True)
    clips = get_canonical_clip_paths(sample_json, gtzan_dir, count=count)
    
    manifest = []
    print(f"Exporting {len(clips)} GTZAN WAV clips to {dest_dir}...")
    
    for idx, src_path in enumerate(clips):
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)
        genre = filename.split(".")[0]
        
        # Copy file if not already existing
        if not os.path.isfile(dest_path) or os.path.getsize(dest_path) != os.path.getsize(src_path):
            shutil.copy2(src_path, dest_path)
            
        manifest.append({
            "index": idx,
            "filename": filename,
            "genre": genre,
            "original_path": src_path,
            "dest_path": dest_path
        })
        
    manifest_path = os.path.join(dest_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Successfully exported {len(clips)} clips.")
    print(f"Manifest written to: {manifest_path}")
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Export 50 GTZAN audio clips for MOODWAAVE benchmark kit.")
    parser.add_argument("--dest-dir", type=str, default=DEFAULT_DEST_DIR, help="Destination directory for audio clips")
    parser.add_argument("--count", type=int, default=50, help="Number of clips to export (default: 50)")
    parser.add_argument("--sample-json", type=str, default=DEFAULT_SAMPLE_JSON, help="Path to sample_ids.json")
    parser.add_argument("--gtzan-dir", type=str, default=DEFAULT_GTZAN_DIR, help="Path to GTZAN genres_original directory")
    args = parser.parse_args()
    
    export_clips(args.dest_dir, count=args.count, sample_json=args.sample_json, gtzan_dir=args.gtzan_dir)

if __name__ == "__main__":
    main()

