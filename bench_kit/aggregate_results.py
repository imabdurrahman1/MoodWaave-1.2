#!/usr/bin/env python3
"""
release/bench_kit/aggregate_results.py

MOODWAAVE Platform Benchmark Kit: Result Aggregator
Reads all results/*.json benchmark reports and generates:
1. Consolidated Platform x Library x Pipeline summary table (CSV and JSON).
2. Feature group latency comparison table (CSV).
3. Publication-quality 300-DPI comparison figure.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")
DEFAULT_OUT_DIR = HERE

def parse_all_results(results_dir):
    """Scan and parse all benchmark JSON files."""
    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    valid_reports = []
    
    for jf in json_files:
        # Skip temporary test runs
        if os.path.basename(jf).startswith("test_"):
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "platform" in data and "librosa" in data:
                valid_reports.append((os.path.basename(jf), data))
        except Exception as e:
            print(f"Warning: Failed to parse {jf}: {e}")
            
    return valid_reports

def build_summary_table(reports):
    """Construct unified table across Platform x Library x Pipeline."""
    rows = []
    
    for filename, rep in reports:
        p = rep["platform"]
        plat_label = f"{p['cpu_model']} ({p['os']} {p['os_release']}, {p['architecture']})"
        os_name = p["os"]
        arch = p["architecture"]
        
        # 1. Librosa Pipelines
        lib = rep["librosa"]["summary"]
        load_ms = lib["audio_load_ms"]["mean"]
        
        # Pipeline a: Original-113
        pa = lib["pipe_a_orig113"]
        rows.append({
            "platform": plat_label,
            "os": os_name,
            "arch": arch,
            "library": "librosa",
            "pipeline": "Original-113 (Tonnetz V0)",
            "pipeline_id": "pipe_a",
            "audio_load_ms": load_ms,
            "features_ms": pa["features_ms"]["mean"],
            "prediction_ms": pa["prediction_ms"]["mean"],
            "total_latency_ms": pa["total_ms"]["mean"],
            "total_latency_std": pa["total_ms"]["std"],
            "rtf": pa["rtf"],
            "speedup_vs_orig": 1.0,
            "tonnetz_share_pct": pa["tonnetz_share_pct"],
        })
        
        # Pipeline b: Full-113 V1
        pb = lib["pipe_b_full113_v1"]
        rows.append({
            "platform": plat_label,
            "os": os_name,
            "arch": arch,
            "library": "librosa",
            "pipeline": "Full-113 (Tonnetz V1)",
            "pipeline_id": "pipe_b",
            "audio_load_ms": load_ms,
            "features_ms": pb["features_ms"]["mean"],
            "prediction_ms": pb["prediction_ms"]["mean"],
            "total_latency_ms": pb["total_ms"]["mean"],
            "total_latency_std": pb["total_ms"]["std"],
            "rtf": pb["rtf"],
            "speedup_vs_orig": pb["speedup_vs_orig"],
            "tonnetz_share_pct": (lib["tonnetz_v1_ms"]["mean"] / pb["total_ms"]["mean"]) * 100.0,
        })
        
        # Pipeline c: MOODWAAVE-Lite-71
        pc = lib["pipe_c_lite71"]
        rows.append({
            "platform": plat_label,
            "os": os_name,
            "arch": arch,
            "library": "librosa",
            "pipeline": "MOODWAAVE-Lite-71",
            "pipeline_id": "pipe_c",
            "audio_load_ms": load_ms,
            "features_ms": pc["features_ms"]["mean"],
            "prediction_ms": pc["prediction_ms"]["mean"],
            "total_latency_ms": pc["total_ms"]["mean"],
            "total_latency_std": pc["total_ms"]["std"],
            "rtf": pc["rtf"],
            "speedup_vs_orig": pc["speedup_vs_orig"],
            "tonnetz_share_pct": 0.0,
        })
        
        # 2. Second Library (if available)
        sec = rep.get("second_library")
        if sec and sec.get("summary"):
            s_sum = sec["summary"]
            s_lib_name = s_sum["library"]
            s_load = s_sum["audio_load_ms"]["mean"]
            
            # Feature extraction sum of available analogues
            s_mfcc = s_sum["mfcc_ms"]["mean"]
            s_chroma = s_sum["chroma_ms"]["mean"]
            s_ton = s_sum["tonnetz_ms"]["mean"]
            s_spec = s_sum["spectral_ms"]["mean"] if s_sum["spectral_ms"] else 0.0
            s_temp = s_sum["temporal_ms"]["mean"]
            
            # TorchAudio / nnAudio Equivalent of Full-113 features (analogue)
            s_full_feat = s_mfcc + s_chroma + s_ton + s_spec + s_temp
            s_tot_full = s_load + s_full_feat + pb["prediction_ms"]["mean"]
            
            rows.append({
                "platform": plat_label,
                "os": os_name,
                "arch": arch,
                "library": s_lib_name,
                "pipeline": "Full Analogue (MFCC+Chroma+Ton+Spec+Temp)",
                "pipeline_id": "pipe_second_full",
                "audio_load_ms": s_load,
                "features_ms": s_full_feat,
                "prediction_ms": pb["prediction_ms"]["mean"],
                "total_latency_ms": s_tot_full,
                "total_latency_std": 0.0,
                "rtf": s_tot_full / 30000.0,
                "speedup_vs_orig": pa["total_ms"]["mean"] / s_tot_full,
                "tonnetz_share_pct": (s_ton / s_tot_full) * 100.0,
            })
            
            # TorchAudio / nnAudio Equivalent of Lite features (analogue: MFCC + Temp)
            s_lite_feat = s_mfcc + s_temp
            s_tot_lite = s_load + s_lite_feat + pc["prediction_ms"]["mean"]
            rows.append({
                "platform": plat_label,
                "os": os_name,
                "arch": arch,
                "library": s_lib_name,
                "pipeline": "Lite Analogue (MFCC+Temp)",
                "pipeline_id": "pipe_second_lite",
                "audio_load_ms": s_load,
                "features_ms": s_lite_feat,
                "prediction_ms": pc["prediction_ms"]["mean"],
                "total_latency_ms": s_tot_lite,
                "total_latency_std": 0.0,
                "rtf": s_tot_lite / 30000.0,
                "speedup_vs_orig": pa["total_ms"]["mean"] / s_tot_lite,
                "tonnetz_share_pct": 0.0,
            })
            
    return pd.DataFrame(rows)

def build_group_comparison_table(reports):
    """Build group-level feature extraction comparison between Librosa and Second Library."""
    rows = []
    for filename, rep in reports:
        p = rep["platform"]
        plat_label = f"{p['cpu_model']} ({p['os']})"
        
        lib = rep["librosa"]["summary"]
        sec = rep.get("second_library", {}).get("summary")
        
        rows.append({
            "platform": plat_label,
            "library": "librosa",
            "audio_load_ms": lib["audio_load_ms"]["mean"],
            "mfcc_ms": lib["mfcc_ms"]["mean"],
            "chroma_ms": lib["chroma_ms"]["mean"],
            "tonnetz_v0_ms": lib["tonnetz_v0_ms"]["mean"],
            "tonnetz_v1_ms": lib["tonnetz_v1_ms"]["mean"],
            "spectral_ms": lib["spectral_ms"]["mean"],
            "spectral_contrast_ms": lib["contrast_ms"]["mean"],
            "temporal_ms": lib["temporal_ms"]["mean"],
        })
        
        if sec:
            rows.append({
                "platform": plat_label,
                "library": sec["library"],
                "audio_load_ms": sec["audio_load_ms"]["mean"],
                "mfcc_ms": sec["mfcc_ms"]["mean"],
                "chroma_ms": sec["chroma_ms"]["mean"],
                "tonnetz_v0_ms": None,
                "tonnetz_v1_ms": sec["tonnetz_ms"]["mean"],
                "spectral_ms": sec["spectral_ms"]["mean"] if sec["spectral_ms"] else None,
                "spectral_contrast_ms": None,
                "temporal_ms": sec["temporal_ms"]["mean"],
            })
            
    return pd.DataFrame(rows)

def generate_publication_figure(df_summary, out_png):
    """Generate 300 DPI publication figure."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300)
    
    # Filter librosa standard pipelines
    df_lib = df_summary[df_summary["library"] == "librosa"]
    platforms = df_lib["platform"].unique()
    
    # If single platform or selecting first platform
    target_plat = platforms[0]
    df_plat = df_lib[df_lib["platform"] == target_plat]
    
    # Panel A: Latency Breakdown across Pipelines
    pipelines = ["Original-113\n(Tonnetz V0)", "Full-113\n(Tonnetz V1)", "MOODWAAVE\nLite-71"]
    pipe_ids = ["pipe_a", "pipe_b", "pipe_c"]
    
    sub = df_plat.set_index("pipeline_id").loc[pipe_ids]
    load_vals = sub["audio_load_ms"].values
    feat_vals = sub["features_ms"].values
    pred_vals = sub["prediction_ms"].values
    tot_vals = sub["total_latency_ms"].values
    rtf_vals = sub["rtf"].values
    speedups = sub["speedup_vs_orig"].values
    
    x = np.arange(len(pipelines))
    width = 0.5
    
    p1 = ax1.bar(x, load_vals, width, label="Audio Load (WAV)", color="#3b82f6", alpha=0.9)
    p2 = ax1.bar(x, feat_vals, width, bottom=load_vals, label="Feature Extraction", color="#10b981", alpha=0.9)
    p3 = ax1.bar(x, pred_vals, width, bottom=load_vals + feat_vals, label="Prediction (3 Heads)", color="#f59e0b", alpha=0.9)
    
    ax1.set_ylabel("Single-Thread Latency (ms per 30s track)", fontsize=10, fontweight="bold")
    ax1.set_title("MOODWAAVE Pipeline Latency Breakdown", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pipelines, fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)
    
    for i, (tot, spd) in enumerate(zip(tot_vals, speedups)):
        lbl = f"{tot:.1f} ms\n(1.0×)" if i == 0 else f"{tot:.1f} ms\n({spd:.2f}× speedup)"
        ax1.text(i, tot + 25.0, lbl, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0, max(tot_vals) * 1.18)
    
    # Panel B: Real-Time Factor (RTF) & Tonnetz Bottleneck
    colors = ["#ef4444", "#6366f1", "#10b981"]
    bars2 = ax2.bar(x, rtf_vals, width, color=colors, alpha=0.85)
    ax2.set_ylabel("Real-Time Factor (RTF)", fontsize=10, fontweight="bold")
    ax2.set_title("Real-Time Factor (Lower is Faster)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pipelines, fontsize=9.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Reference line at RTF = 0.01 (100x real-time)
    ax2.axhline(0.01, color="#8b5cf6", linestyle=":", linewidth=1.5, label="100× Faster than Real-Time (RTF=0.01)")
    ax2.legend(loc="upper right", frameon=True)
    
    for i, rtf in enumerate(rtf_vals):
        faster = 1.0 / rtf if rtf > 0 else 0
        ax2.text(i, rtf + 0.002, f"RTF {rtf:.4f}\n({faster:.0f}× real-time)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, max(rtf_vals) * 1.25)
    
    for i, rtf in enumerate(rtf_vals):
        faster = 1.0 / rtf if rtf > 0 else 0
        ax2.text(i, rtf + 0.002, f"RTF {rtf:.4f}\n({faster:.0f}× real-time)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, max(rtf_vals) * 1.25)
    
    plt.suptitle("MOODWAAVE Platform Benchmark Kit: Single-Thread CPU Profiling (GTZAN 50)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication figure to: {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate MOODWAAVE benchmark kit results.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory containing result JSONs")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for table and figure")
    args = parser.parse_args()
    
    reports = parse_all_results(args.results_dir)
    print(f"Found {len(reports)} valid benchmark reports in {args.results_dir}.")
    if not reports:
        print("No benchmark JSON reports found to aggregate. Please run bench_platform.py first.")
        sys.exit(0)
        
    df_summary = build_summary_table(reports)
    df_groups = build_group_comparison_table(reports)
    
    # Output CSV and JSON
    out_csv = os.path.join(args.out_dir, "bench_summary_table.csv")
    out_json = os.path.join(args.out_dir, "bench_summary_table.json")
    out_groups_csv = os.path.join(args.out_dir, "group_comparison_table.csv")
    out_fig = os.path.join(args.out_dir, "bench_platform_latency_and_breakdown.png")
    
    df_summary.to_csv(out_csv, index=False)
    df_summary.to_json(out_json, orient="records", indent=2)
    df_groups.to_csv(out_groups_csv, index=False)
    
    print(f"Saved summary table to: {out_csv} and {out_json}")
    print(f"Saved group comparison table to: {out_groups_csv}\n")
    print(df_summary[["platform", "library", "pipeline", "total_latency_ms", "rtf", "speedup_vs_orig", "tonnetz_share_pct"]].to_string(index=False))
    
    generate_publication_figure(df_summary, out_fig)
    print("\nAggregation complete.")

if __name__ == "__main__":
    main()
