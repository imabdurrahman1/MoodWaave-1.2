"""Step 2 (Fix): Interleaved Timing Benchmark Run TWICE (Forward & Reverse Order) on AC Power.
Stability Protocol (MOODWAAVE TASK P2-FIX):
- High Performance power plan active.
- Laptop plugged into AC power (PowerOnline=True, no battery throttling).
- Single-threaded (OMP/MKL/OPENBLAS/NUMBA threads = 1).
- 1 warm-up track excluded from sample.
- Median of 3 repeats per callable using time.perf_counter().
- Exactly 100 tracks per dataset (GTZAN and PMEmo from sample_ids.json).
- Run 1: Forward track order (tracks 0..99).
- Run 2: Reverse track order (tracks 99..0).
- Interleaved per track:
  1. Standalone groups: Audio load, MFCC, Chroma, Tonnetz (V0), Spectral, SpecContrast, Temporal.
  2. Shared-STFT marginal: STFT, MFCC, Chroma, Tonnetz (V1), Tonnetz (V2), Spectral, SpecContrast, Temporal.
  3. End-to-end pipelines: Original 113 (V0), Full-113 (V1), Lite-71, Lite+V1 (83 features).

Outputs:
- timing_run1_forward.csv
- timing_run2_reverse.csv
- timing_runs_comparison.csv (Run 1 vs Run 2 difference)
- timing_per_track.csv (consolidated/pooled across both runs)
- feature_costs_retimed.csv (consolidated per-group standalone and shared-STFT marginal, mean ± SD)
- end_to_end_summary.csv (consolidated E2E latency, sum of parts, gap, Tonnetz share, dual speedups)
"""
import os
import sys
import json
import time
import gc
import numpy as np
import pandas as pd
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

REPEATS = 3
N_TRACKS = 100


def med_time(fn, *args):
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def warmup(warmup_path):
    print(f"Executing warm-up on {warmup_path}...", flush=True)
    y, sr = C.load_audio(warmup_path)
    # Standalone
    C.g_mfcc(y, sr)
    C.g_chroma(y, sr)
    C.g_tonnetz_v0(y, sr)
    C.g_spectral(y, sr)
    C.g_contrast(y, sr)
    C.g_temporal(y, sr)
    # Shared
    S = C.shared_stft(y)
    C.s_mfcc(S, sr)
    chr_mat = librosa.feature.chroma_stft(S=S**2, sr=sr)
    C.s_tonnetz_v1_from_chroma(chr_mat, sr)
    C.s_tonnetz_v2_harmonic(S, sr)
    C.s_spectral(S, sr)
    C.s_contrast(S, sr)
    # End to end
    C.extract_original_113_v0(y, sr)
    C.extract_full_113_v1(y, sr)
    C.extract_lite_71(y, sr)
    C.extract_lite_plus_v1_83(y, sr)
    print("Warm-up complete.\n", flush=True)


def time_dataset_pass(ds_name, track_paths, run_label):
    print(f"\n--- [{run_label}] Timing {len(track_paths)} tracks for {ds_name} ---", flush=True)
    records = []
    t0 = time.time()

    for idx, path in enumerate(track_paths):
        gc.collect()
        rec = {
            "run": run_label,
            "dataset": ds_name,
            "track_order_idx": idx,
            "track": path
        }

        # 1. Audio load
        rec["audio_load_ms"] = med_time(C.load_audio, path)
        y, sr = C.load_audio(path)

        # 2. Standalone groups
        rec["standalone|MFCC"] = med_time(C.g_mfcc, y, sr)
        rec["standalone|Chroma"] = med_time(C.g_chroma, y, sr)
        rec["standalone|Tonnetz_v0"] = med_time(C.g_tonnetz_v0, y, sr)
        rec["standalone|Spectral"] = med_time(C.g_spectral, y, sr)
        rec["standalone|SpecContrast"] = med_time(C.g_contrast, y, sr)
        rec["standalone|Temporal"] = med_time(C.g_temporal, y, sr)

        rec["sum_standalone_v0"] = (
            rec["standalone|MFCC"] + rec["standalone|Chroma"] + rec["standalone|Tonnetz_v0"] +
            rec["standalone|Spectral"] + rec["standalone|SpecContrast"] + rec["standalone|Temporal"]
        )

        # 3. Shared-STFT marginal costs
        rec["shared|STFT"] = med_time(C.shared_stft, y)
        S = C.shared_stft(y)

        rec["shared|MFCC"] = med_time(C.s_mfcc, S, sr)
        rec["shared|Chroma"] = med_time(C.s_chroma, S, sr)

        chr_mat = librosa.feature.chroma_stft(S=S**2, sr=sr)
        rec["shared|Tonnetz_v1_marginal"] = med_time(C.s_tonnetz_v1_from_chroma, chr_mat, sr)
        rec["shared|Tonnetz_v2_harmonic"] = med_time(C.s_tonnetz_v2_harmonic, S, sr)

        rec["shared|Spectral"] = med_time(C.s_spectral, S, sr)
        rec["shared|SpecContrast"] = med_time(C.s_contrast, S, sr)
        rec["shared|Temporal"] = rec["standalone|Temporal"]

        rec["sum_shared_full_v1"] = (
            rec["shared|STFT"] + rec["shared|MFCC"] + rec["shared|Chroma"] +
            rec["shared|Tonnetz_v1_marginal"] + rec["shared|Spectral"] +
            rec["shared|SpecContrast"] + rec["shared|Temporal"]
        )

        rec["sum_shared_lite_71"] = (
            rec["shared|STFT"] + rec["shared|MFCC"] + rec["shared|SpecContrast"] + rec["shared|Temporal"]
        )

        rec["sum_shared_lite_plus_v1_83"] = (
            rec["shared|STFT"] + rec["shared|MFCC"] + rec["shared|SpecContrast"] +
            rec["shared|Chroma"] + rec["shared|Tonnetz_v1_marginal"] + rec["shared|Temporal"]
        )

        # 4. Measured End-to-End Pipelines
        rec["e2e|Original_113_v0"] = med_time(C.extract_original_113_v0, y, sr)
        rec["e2e|Full_113_v1"] = med_time(C.extract_full_113_v1, y, sr)
        rec["e2e|Lite_71"] = med_time(C.extract_lite_71, y, sr)
        rec["e2e|Lite_plus_v1_83"] = med_time(C.extract_lite_plus_v1_83, y, sr)

        records.append(rec)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(track_paths):
            elapsed = time.time() - t0
            print(f"    [{run_label}] {ds_name}: {idx+1}/{len(track_paths)} tracks ({elapsed:.1f}s)", flush=True)

    return records


def build_summary_tables(df_all_runs):
    # 1. Feature costs table (standalone & shared marginal, mean ± SD)
    cost_rows = []
    groups = [
        ("MFCC", "standalone|MFCC", "shared|MFCC"),
        ("Chroma", "standalone|Chroma", "shared|Chroma"),
        ("Tonnetz (V0 Baseline)", "standalone|Tonnetz_v0", None),
        ("Tonnetz (V1 Shared Chroma)", None, "shared|Tonnetz_v1_marginal"),
        ("Tonnetz (V2 Harmonic STFT)", None, "shared|Tonnetz_v2_harmonic"),
        ("Spectral", "standalone|Spectral", "shared|Spectral"),
        ("Spectral Contrast", "standalone|SpecContrast", "shared|SpecContrast"),
        ("Temporal", "standalone|Temporal", "shared|Temporal")
    ]

    for ds, d in df_all_runs.groupby("dataset"):
        cost_rows.append({
            "dataset": ds, "component": "Audio Loading", "mode": "load",
            "mean_ms": round(d["audio_load_ms"].mean(), 2),
            "std_ms": round(d["audio_load_ms"].std(ddof=1), 2)
        })
        cost_rows.append({
            "dataset": ds, "component": "Shared STFT", "mode": "shared_stft",
            "mean_ms": round(d["shared|STFT"].mean(), 2),
            "std_ms": round(d["shared|STFT"].std(ddof=1), 2)
        })
        for name, sa_col, sh_col in groups:
            if sa_col:
                cost_rows.append({
                    "dataset": ds, "component": name, "mode": "standalone",
                    "mean_ms": round(d[sa_col].mean(), 2),
                    "std_ms": round(d[sa_col].std(ddof=1), 2)
                })
            if sh_col:
                cost_rows.append({
                    "dataset": ds, "component": name, "mode": "shared_marginal",
                    "mean_ms": round(d[sh_col].mean(), 2),
                    "std_ms": round(d[sh_col].std(ddof=1), 2)
                })

    df_costs = pd.DataFrame(cost_rows)

    # 2. End-to-end pipeline summary with dual speedups and consistent Tonnetz share
    pipe_rows = []
    for ds, d in df_all_runs.groupby("dataset"):
        e2e_v0_mean = d["e2e|Original_113_v0"].mean()
        e2e_v1_mean = d["e2e|Full_113_v1"].mean()

        configs = [
            ("Original 113 (V0 Baseline)", 113, "e2e|Original_113_v0", "sum_standalone_v0",
             "V0 (HPSS+CQT)", d["standalone|Tonnetz_v0"].mean(), "standalone V0 cost / E2E"),
            ("Full-113 with Tonnetz V1", 113, "e2e|Full_113_v1", "sum_shared_full_v1",
             "V1 (Shared Chroma)", d["shared|Tonnetz_v1_marginal"].mean(), "marginal V1 projection only (chroma in full)"),
            ("MOODWAAVE-Lite (71 features)", 71, "e2e|Lite_71", "sum_shared_lite_71",
             "None (Excluded)", 0.0, "not included"),
            ("MOODWAAVE-Lite + V1 (83 features)", 83, "e2e|Lite_plus_v1_83", "sum_shared_lite_plus_v1_83",
             "V1 (Chroma + Tonnetz)", d["shared|Chroma"].mean() + d["shared|Tonnetz_v1_marginal"].mean(),
             "chroma marginal + V1 projection (both added to Lite)")
        ]

        for p_name, n_feats, e2e_col, sum_col, t_var, t_cost, t_def in configs:
            m_e2e = d[e2e_col].mean()
            s_e2e = d[e2e_col].std(ddof=1)
            m_sum = d[sum_col].mean()
            gap_ms = m_sum - m_e2e
            gap_pct = (gap_ms / m_e2e) * 100.0
            t_share_pct = (t_cost / m_e2e) * 100.0
            speedup_vs_v0 = e2e_v0_mean / m_e2e
            speedup_vs_v1 = e2e_v1_mean / m_e2e

            pipe_rows.append({
                "dataset": ds,
                "pipeline": p_name,
                "features": n_feats,
                "measured_e2e_mean_ms": round(m_e2e, 2),
                "measured_e2e_std_ms": round(s_e2e, 2),
                "latency_display": f"{m_e2e:.1f} ± {s_e2e:.1f} ms",
                "sum_of_components_ms": round(m_sum, 2),
                "overhead_gap_ms": round(gap_ms, 2),
                "overhead_gap_pct": round(gap_pct, 2),
                "tonnetz_variant": t_var,
                "tonnetz_cost_ms": round(t_cost, 2),
                "tonnetz_share_pct": round(t_share_pct, 2),
                "tonnetz_share_definition": t_def,
                "speedup_vs_v0": round(speedup_vs_v0, 2),
                "speedup_vs_full_v1": round(speedup_vs_v1, 2)
            })

    df_pipe = pd.DataFrame(pipe_rows)
    return df_costs, df_pipe


def main():
    print("=" * 80)
    print("MOODWAAVE - TASK P2-FIX: INTERLEAVED TIMING RUN TWICE (FORWARD & REVERSE)")
    print("=" * 80)
    t_global = time.time()
    C.log_environment(C.OUT)

    with open(C.SAMPLE_IDS_JSON, "r") as f:
        sample_ids = json.load(f)

    # Warmup
    gtzan_df = pd.read_csv(C.GTZAN_CSV)
    gtzan_df["path"] = [os.path.join(C.GTZAN_DIR, l, fn) for l, fn in zip(gtzan_df["label"], gtzan_df["filename"])]
    sample_set_g = set(sample_ids["GTZAN"])
    warm_candidates = [p for p in gtzan_df["path"] if p not in sample_set_g and os.path.exists(p)]
    warmup(warm_candidates[0])

    # Run 1: Forward track order
    print("\n================== RUN 1: FORWARD TRACK ORDER ==================", flush=True)
    run1_records = []
    for ds in ["GTZAN", "PMEmo"]:
        paths_fwd = sample_ids[ds][:N_TRACKS]
        recs = time_dataset_pass(ds, paths_fwd, "Run1_Forward")
        run1_records.extend(recs)
    df_run1 = pd.DataFrame(run1_records)
    df_run1.to_csv(os.path.join(C.OUT, "timing_run1_forward.csv"), index=False)
    print(f"Saved timing_run1_forward.csv ({len(df_run1)} tracks).", flush=True)

    # Short cooling pause
    print("\nPausing 10s between runs to stabilize scheduler...", flush=True)
    time.sleep(10)

    # Run 2: Reverse track order
    print("\n================== RUN 2: REVERSE TRACK ORDER ==================", flush=True)
    run2_records = []
    for ds in ["GTZAN", "PMEmo"]:
        paths_rev = list(reversed(sample_ids[ds][:N_TRACKS]))
        recs = time_dataset_pass(ds, paths_rev, "Run2_Reverse")
        run2_records.extend(recs)
    df_run2 = pd.DataFrame(run2_records)
    df_run2.to_csv(os.path.join(C.OUT, "timing_run2_reverse.csv"), index=False)
    print(f"Saved timing_run2_reverse.csv ({len(df_run2)} tracks).", flush=True)

    # Comparison of Run 1 vs Run 2
    comp_rows = []
    e2e_cols = ["e2e|Original_113_v0", "e2e|Full_113_v1", "e2e|Lite_71", "e2e|Lite_plus_v1_83"]
    labels = {
        "e2e|Original_113_v0": "Original 113 (V0)",
        "e2e|Full_113_v1": "Full-113 + V1",
        "e2e|Lite_71": "MOODWAAVE-Lite (71)",
        "e2e|Lite_plus_v1_83": "Lite + V1 (83)"
    }
    for ds in ["GTZAN", "PMEmo"]:
        d1 = df_run1[df_run1["dataset"] == ds]
        d2 = df_run2[df_run2["dataset"] == ds]
        for col in e2e_cols:
            m1 = d1[col].mean()
            m2 = d2[col].mean()
            diff_ms = m2 - m1
            diff_pct = (diff_ms / m1) * 100.0
            comp_rows.append({
                "dataset": ds,
                "pipeline": labels[col],
                "run1_forward_mean_ms": round(m1, 2),
                "run2_reverse_mean_ms": round(m2, 2),
                "diff_ms (run2 - run1)": round(diff_ms, 2),
                "diff_pct": round(diff_pct, 2)
            })

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(os.path.join(C.OUT, "timing_runs_comparison.csv"), index=False)
    print("\nSaved timing_runs_comparison.csv.")
    print("\n--- RUN 1 vs RUN 2 STABILITY COMPARISON ---")
    print(df_comp.to_string(index=False))

    # Consolidated pooled records
    df_pooled = pd.concat([df_run1, df_run2], ignore_index=True)
    df_pooled.to_csv(os.path.join(C.OUT, "timing_per_track.csv"), index=False)

    df_costs, df_pipe = build_summary_tables(df_pooled)
    df_costs.to_csv(os.path.join(C.OUT, "feature_costs_retimed.csv"), index=False)
    df_pipe.to_csv(os.path.join(C.OUT, "end_to_end_summary.csv"), index=False)

    print("\nSaved consolidated feature_costs_retimed.csv and end_to_end_summary.csv.")
    print("\n--- CONSOLIDATED END-TO-END LATENCIES & DUAL SPEEDUPS ---")
    print(df_pipe[["dataset", "pipeline", "latency_display", "sum_of_components_ms", "overhead_gap_pct", "tonnetz_share_pct", "speedup_vs_v0", "speedup_vs_full_v1"]].to_string(index=False))
    print(f"\nAll timing runs completed in {time.time() - t_global:.1f}s.")


if __name__ == "__main__":
    main()

