"""
experiments/rev2_final_timing/02_analyze_and_report.py
Generates 300-DPI publication figures and the definitive RESULTS.md master document.
Explicitly states that this table supersedes every earlier latency number in the repository.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PIPELINE_SHORT_NAMES = {
    "a) Original-113 with original Tonnetz": "a) Original-113\n(V0 Baseline)",
    "b) Full-113 with Tonnetz V1": "b) Full-113\n(Tonnetz V1)",
    "c) MOODWAAVE-Lite-71": "c) MOODWAAVE\nLite-71",
    "d) Lite + V1 (83)": "d) Lite + V1\n(83 Feats)",
    "e) SmallMelCNN (genre only)": "e) SmallMelCNN\n(Genre Only)",
    "f) VGGish + linear heads": "f) VGGish +\nLinear Heads",
}

PIPELINE_COLORS = {
    "a) Original-113 with original Tonnetz": "#7f7f7f",  # Gray
    "b) Full-113 with Tonnetz V1": "#1f77b4",          # Blue
    "c) MOODWAAVE-Lite-71": "#2ca02c",                 # Green (MOODWAAVE Hero)
    "d) Lite + V1 (83)": "#ff7f0e",                    # Orange
    "e) SmallMelCNN (genre only)": "#9467bd",          # Purple
    "f) VGGish + linear heads": "#d62728",             # Red
}

def generate_figures_and_report():
    df_table = pd.read_csv(os.path.join(BASE_DIR, "final_timing_table.csv"))
    df_comp = pd.read_csv(os.path.join(BASE_DIR, "timing_runs_comparison.csv"))
    df_sizes = pd.read_csv(os.path.join(BASE_DIR, "system_sizes.csv"))
    with open(os.path.join(BASE_DIR, "environment.json"), "r", encoding="utf-8") as f:
        env = json.load(f)
        
    # --------------------------------------------------------------------------
    # Figure 1: Final Latency Breakdown (Audio Load + Features + Predictions)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    
    datasets = ["GTZAN", "PMEmo"]
    titles = ["GTZAN (30s WAV tracks)", "PMEmo (30s MP3 tracks)"]
    
    for ax, ds, title in zip(axes, datasets, titles):
        sub = df_table[df_table["dataset"] == ds].copy()
        sub["order"] = sub["pipeline"].map(lambda x: x[0])
        sub = sub.sort_values("order")
        
        x = np.arange(len(sub))
        width = 0.55
        
        p_load = sub["audio_load_ms_mean"].values
        p_feat = sub["feat_ms_mean"].values
        p_pred = sub["pred_ms_mean"].values
        p_tot = sub["total_ms_mean"].values
        p_sd = sub["total_ms_sd"].values
        
        b1 = ax.bar(x, p_load, width, label="Audio Loading", color="#aec7e8", edgecolor="#333333")
        b2 = ax.bar(x, p_feat, width, bottom=p_load, label="Feature / Embedding Ext.", color="#98df8a", edgecolor="#333333")
        b3 = ax.bar(x, p_pred, width, bottom=p_load + p_feat, label="Prediction (All Heads)", color="#ffbb78", edgecolor="#333333", yerr=p_sd, capsize=3.5)
        
        # Annotate total on top
        for rect, tot in zip(b1, p_tot):
            ax.annotate(f"{tot:.1f} ms",
                        xy=(rect.get_x() + rect.get_width() / 2, tot),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
                        
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([PIPELINE_SHORT_NAMES[p] for p in sub["pipeline"]], fontsize=8.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        if ds == "GTZAN":
            ax.set_ylabel("End-to-End Latency per Track (ms)")
            ax.legend(loc="upper right", frameon=True, edgecolor="#cccccc")
            
    fig.suptitle("MOODWAAVE: Final Single-Thread CPU Latency Breakdown (Audio Load + Features + Predictions)", fontweight="bold", fontsize=12)
    fig_path1 = os.path.join(BASE_DIR, "final_latency_breakdown_bar_chart.png")
    fig.savefig(fig_path1)
    plt.close(fig)
    print(f"Generated {fig_path1}")
    
    # --------------------------------------------------------------------------
    # Figure 2: Latency vs. Serialized Model Size Pareto Frontier
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    sub_overall = df_table[df_table["dataset"] == "Overall"].copy()
    
    for _, row in sub_overall.iterrows():
        p_name = row["pipeline"]
        lat = row["total_ms_mean"]
        sz = row["serialized_size_mb"]
        color = PIPELINE_COLORS[p_name]
        
        ax.scatter(sz, lat, color=color, s=120, edgecolors="black", linewidth=1.2, zorder=4)
        
        # Label offset
        offset_x, offset_y = 5, 5
        if "VGGish" in p_name:
            offset_x, offset_y = -80, -15
        elif "Lite-71" in p_name:
            offset_x, offset_y = 8, -12
        elif "Full-113 with Tonnetz V1" in p_name:
            offset_x, offset_y = 8, 8
        elif "SmallMelCNN" in p_name:
            offset_x, offset_y = 8, 5
            
        ax.annotate(p_name.split(") ")[1],
                    xy=(sz, lat),
                    xytext=(offset_x, offset_y), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", color=color)
                    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Serialized System Size (MB, Log Scale)")
    ax.set_ylabel("End-to-End Latency (ms/track, Log Scale)")
    ax.set_title("MOODWAAVE: Latency vs. Memory Footprint Pareto Frontier (Overall 200 Tracks)", fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    
    fig_path2 = os.path.join(BASE_DIR, "final_latency_vs_size_pareto.png")
    fig.savefig(fig_path2)
    plt.close(fig)
    print(f"Generated {fig_path2}")
    
    # --------------------------------------------------------------------------
    # Generate RESULTS.md
    # --------------------------------------------------------------------------
    md = []
    md.append("# MOODWAAVE — Final Latency & System Footprint Benchmark (TASK FINAL-TIMING)\n")
    md.append("> [!IMPORTANT]\n> **SUPERSEDING NOTICE**: This document and the tables herein contain the definitive, unified single-session measurements for all six pipelines evaluated under identical single-threaded conditions across both GTZAN and PMEmo datasets. **THIS MASTER TABLE SUPERSEDES EVERY EARLIER LATENCY NUMBER IN THE REPOSITORY.**\n")
    
    md.append("## 1. Experimental Conditions & Hardware Rigor\n")
    md.append("- **Machine State**: Idle machine, background applications closed, plugged into AC Power (100% battery, no thermal or throttling states).")
    md.append(f"- **Power Plan**: `{env['power_plan']}`.")
    md.append(f"- **CPU & OS**: {env['processor']} ({env['cpu_count_physical']} physical cores, {env['cpu_count_logical']} logical), Windows 11.")
    md.append(f"- **Strict Single-Thread Enforcement**: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `torch.get_num_threads() = 1`.")
    md.append("- **Datasets**: Exactly the same canonical 100 tracks per dataset (100 GTZAN 30s WAV tracks and 100 PMEmo 30s MP3 tracks from `sample_ids.json`).")
    md.append("- **Protocol**: One warm-up track excluded per dataset; median of 3 repeats per track; run TWICE (Run 1: Forward track order 0..99; Run 2: Reverse track order 99..0). Both runs reported.\n")
    
    md.append("## 2. Master Efficiency Table (All Pipelines, One Table)\n")
    md.append("Timed region encompasses the complete end-to-end user experience: **Audio Loading + Feature/Embedding Extraction + ALL Predictions Made by the System**.\n")
    
    for ds in ["GTZAN", "PMEmo", "Overall"]:
        md.append(f"### {ds} Benchmark Results\n")
        md.append("| Pipeline | Total Latency (ms) | Speedup vs Orig | Audio Load (ms) | Features / Emb (ms) | Predictions (ms) | RTF (comp s / audio s) | Serialized Size (MB) | Models & Hyperparameters |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        
        sub = df_table[df_table["dataset"] == ds].copy()
        sub["order"] = sub["pipeline"].map(lambda x: x[0])
        sub = sub.sort_values("order")
        
        for _, r in sub.iterrows():
            tot_str = f"**{r['total_ms_mean']:.1f} ± {r['total_ms_sd']:.1f}**"
            spd_str = f"**{r['speedup_vs_orig']:.2f}×**" if r['speedup_vs_orig'] > 1.0 else f"{r['speedup_vs_orig']:.2f}×"
            load_str = f"{r['audio_load_ms_mean']:.1f} ± {r['audio_load_ms_sd']:.1f}"
            feat_str = f"{r['feat_ms_mean']:.1f} ± {r['feat_ms_sd']:.1f}"
            pred_str = f"{r['pred_ms_mean']:.1f} ± {r['pred_ms_sd']:.1f}"
            rtf_str = f"{r['rtf']:.5f}"
            sz_str = f"{r['serialized_size_mb']:.2f} MB"
            md.append(f"| {r['pipeline']} | {tot_str} | {spd_str} | {load_str} | {feat_str} | {pred_str} | {rtf_str} | {sz_str} | {r['hyperparameters']} |")
        md.append("")
        
    md.append("## 3. Stability Verification: Run 1 (Forward) vs. Run 2 (Reverse Track Order)\n")
    md.append("To ensure that disk caching, thread scheduling, memory fragmentation, or track ordering did not bias the measurements, the full 200-track benchmark was executed twice (the second run in exact reverse order):\n")
    md.append("| Dataset | Pipeline | Run 1 Forward (ms) | Run 2 Reverse (ms) | Difference (Run 2 - Run 1) | % Change |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for _, r in df_comp.iterrows():
        md.append(f"| {r['dataset']} | {r['pipeline']} | {r['run1_mean_ms']:.1f} ± {r['run1_std_ms']:.1f} | {r['run2_mean_ms']:.1f} ± {r['run2_std_ms']:.1f} | {r['diff_ms (run2 - run1)']:+.2f} ms | {r['diff_pct']:+.2f}% |")
    md.append("")
    
    md.append("## 4. Key Architectural Insights & Findings\n")
    
    lite_row = df_table[(df_table["dataset"] == "Overall") & (df_table["pipeline"].str.contains("Lite-71"))].iloc[0]
    orig_row = df_table[(df_table["dataset"] == "Overall") & (df_table["pipeline"].str.contains("Original-113"))].iloc[0]
    v1_row = df_table[(df_table["dataset"] == "Overall") & (df_table["pipeline"].str.contains("Full-113 with Tonnetz V1"))].iloc[0]
    vgg_row = df_table[(df_table["dataset"] == "Overall") & (df_table["pipeline"].str.contains("VGGish"))].iloc[0]
    cnn_row = df_table[(df_table["dataset"] == "Overall") & (df_table["pipeline"].str.contains("SmallMelCNN"))].iloc[0]
    
    md.append(f"1. **MOODWAAVE-Lite-71 Speedup**: Achieves an overall average latency of **{lite_row['total_ms_mean']:.1f} ms per track** ({lite_row['rtf']:.5f} RTF), delivering a **{lite_row['speedup_vs_orig']:.1f}× speedup** over the Original-113 baseline ({orig_row['total_ms_mean']:.1f} ms) while making ALL THREE predictions (Genre + Valence + Arousal).")
    md.append(f"2. **Tonnetz V1 Impact**: Replacing harmonic HPSS + CQT Tonnetz with shared-STFT Tonnetz V1 slashes Full-113 latency from **{orig_row['total_ms_mean']:.1f} ms** down to **{v1_row['total_ms_mean']:.1f} ms** ({v1_row['speedup_vs_orig']:.1f}× speedup).")
    md.append(f"3. **Comparison with Deep Learning**: While SmallMelCNN achieves **{cnn_row['total_ms_mean']:.1f} ms**, it only predicts Genre. VGGish takes **{vgg_row['total_ms_mean']:.1f} ms** and requires a **{vgg_row['serialized_size_mb']:.2f} MB** memory footprint (a massive **{vgg_row['serialized_size_mb']/lite_row['serialized_size_mb']:.1f}× larger** than MOODWAAVE-Lite's {lite_row['serialized_size_mb']:.2f} MB).")
    md.append(f"4. **Real-Time Factor (RTF)**: MOODWAAVE-Lite runs at **{lite_row['rtf']:.5f} RTF**, meaning it processes audio at **{1.0/lite_row['rtf']:.0f}× faster than real-time** on a standard single CPU thread.\n")
    
    md.append("## 5. Serialized Model Footprint Analysis\n")
    md.append("| System Configuration | Genre Model (MB) | Valence Model (MB) | Arousal Model (MB) | Neural / Backbone Net (MB) | Total Footprint (MB) | Hyperparameters Description |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
    for _, r in df_sizes.iterrows():
        md.append(f"| **{r['pipeline']}** | {r['genre_model_mb']:.2f} MB | {r['valence_model_mb']:.2f} MB | {r['arousal_model_mb']:.2f} MB | {r['embedding_or_cnn_mb']:.2f} MB | **{r['total_serialized_mb']:.2f} MB** | {r['hyperparameters']} |")
        
    with open(os.path.join(BASE_DIR, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Generated {os.path.join(BASE_DIR, 'RESULTS.md')}")

if __name__ == "__main__":
    generate_figures_and_report()

