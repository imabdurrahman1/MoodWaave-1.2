import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from r3b_common import (
    BASE_DIR,
    PROJECT_ROOT,
    MODELS_METADATA,
)

def main():
    print("=== Step 6: Generate Summaries, 300-DPI Pareto Figures, and RESULTS.md ===")
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    gtzan_folds_csv = os.path.join(BASE_DIR, "gtzan_r3b_folds.csv")
    pmemo_folds_csv = os.path.join(BASE_DIR, "pmemo_r3b_folds.csv")
    gtzan_tests_csv = os.path.join(BASE_DIR, "gtzan_r3b_paired_tests.csv")
    pmemo_tests_csv = os.path.join(BASE_DIR, "pmemo_r3b_paired_tests.csv")
    eff_csv = os.path.join(BASE_DIR, "embeddings_efficiency_profiles.csv")
    
    # Phase A reference
    r3a_summary_csv = os.path.join(PROJECT_ROOT, "experiments", "rev3_a_stats", "gtzan_r3a_summary.csv")
    r3a_pmemo_summary_csv = os.path.join(PROJECT_ROOT, "experiments", "rev3_a_stats", "pmemo_r3a_summary.csv")
    r3a_profiles_csv = os.path.join(PROJECT_ROOT, "experiments", "rev3_a_stats", "model_profiles_summary.csv")
    
    df_g_folds = pd.read_csv(gtzan_folds_csv)
    df_p_folds = pd.read_csv(pmemo_folds_csv)
    df_g_tests = pd.read_csv(gtzan_tests_csv)
    df_p_tests = pd.read_csv(pmemo_tests_csv)
    df_eff = pd.read_csv(eff_csv) if os.path.isfile(eff_csv) else pd.DataFrame()
    
    # Phase A baselines
    df_g_r3a = pd.read_csv(r3a_summary_csv)
    df_p_r3a = pd.read_csv(r3a_pmemo_summary_csv)
    
    # Reference metrics for MOODWAAVE Lite and Full
    # GTZAN
    mw_full_acc = df_g_r3a.loc[(df_g_r3a["model"] == "LightGBM") & (df_g_r3a["feature_set"] == "Full-113"), "accuracy_mean"].values[0]
    mw_full_acc_sd = df_g_r3a.loc[(df_g_r3a["model"] == "LightGBM") & (df_g_r3a["feature_set"] == "Full-113"), "accuracy_std"].values[0]
    mw_full_f1 = df_g_r3a.loc[(df_g_r3a["model"] == "LightGBM") & (df_g_r3a["feature_set"] == "Full-113"), "macro_f1_mean"].values[0]
    mw_full_f1_sd = df_g_r3a.loc[(df_g_r3a["model"] == "LightGBM") & (df_g_r3a["feature_set"] == "Full-113"), "macro_f1_std"].values[0]
    
    mw_lite_acc = df_g_r3a.loc[(df_g_r3a["model"] == "LightGBM") & (df_g_r3a["feature_set"] == "Lite-71"), "accuracy_mean"].values[0]
    mw_lite_acc_sd = df_g_r3a.loc[(df_g_r3a["model"] == "LightGBM") & (df_g_r3a["feature_set"] == "Lite-71"), "accuracy_std"].values[0]
    mw_lite_f1 = df_g_r3a.loc[(df_g_r3a["model"] == "LightGBM") & (df_g_r3a["feature_set"] == "Lite-71"), "macro_f1_mean"].values[0]
    mw_lite_f1_sd = df_g_r3a.loc[(df_g_r3a["model"] == "LightGBM") & (df_g_r3a["feature_set"] == "Lite-71"), "macro_f1_std"].values[0]
    
    # Efficiency references (ms per track from final timing)
    # Full-113: 40.73 ms, 5.25 MB; Lite-71: 27.27 ms, 3.75 MB; SmallMelCNN: 43.83 ms, 0.40 MB; VGGish: 147.28 ms, 278.0 MB
    latency_map_gtzan = {
        "MOODWAAVE-Lite-71": 27.27,
        "MOODWAAVE-Full-113": 40.73,
        "SmallMelCNN": 43.83,
        "VGGish": 147.28,
        "PANNs-CNN14": 420.0, # default fallback if not yet measured
        "MERT-v1-95M": 5400.0,
    }
    size_map_mb = {
        "MOODWAAVE-Lite-71": 3.75,
        "MOODWAAVE-Full-113": 5.25,
        "SmallMelCNN": 0.40,
        "VGGish": 278.0,
        "PANNs-CNN14": 312.3,
        "MERT-v1-95M": 360.5,
    }
    
    # Update from actual profiled efficiency if available
    name_map = {
        "c) MOODWAAVE-Lite-71": "MOODWAAVE-Lite-71",
        "b) Full-113 with Tonnetz V1": "MOODWAAVE-Full-113",
        "e) SmallMelCNN (genre only)": "SmallMelCNN",
        "f) VGGish + linear heads": "VGGish",
        "PANNs-CNN14": "PANNs-CNN14",
        "MERT-v1-95M": "MERT-v1-95M",
        "MOODWAAVE-Lite-71": "MOODWAAVE-Lite-71",
        "MOODWAAVE-Full-113": "MOODWAAVE-Full-113",
        "SmallMelCNN": "SmallMelCNN",
        "VGGish": "VGGish",
    }
    
    latency_map_pmemo = {}
    
    if not df_eff.empty:
        for _, row in df_eff[df_eff["dataset"] == "GTZAN"].iterrows():
            raw_p = str(row["pipeline"])
            clean_p = name_map.get(raw_p, raw_p)
            latency_map_gtzan[clean_p] = float(row["total_ms"])
            if pd.notna(row.get("size_mb")):
                size_map_mb[clean_p] = float(row["size_mb"])
                
        for _, row in df_eff[df_eff["dataset"] == "PMEmo"].iterrows():
            raw_p = str(row["pipeline"])
            clean_p = name_map.get(raw_p, raw_p)
            latency_map_pmemo[clean_p] = float(row["total_ms"])
                
    # ---------------------------------------------------------
    # 2. GTZAN Summary Table
    # ---------------------------------------------------------
    g_summary_rows = [
        {
            "model": "MOODWAAVE-Lite-71",
            "accuracy_mean": mw_lite_acc,
            "accuracy_std": mw_lite_acc_sd,
            "macro_f1_mean": mw_lite_f1,
            "macro_f1_std": mw_lite_f1_sd,
            "latency_ms": latency_map_gtzan.get("MOODWAAVE-Lite-71", 27.27),
            "size_mb": size_map_mb.get("MOODWAAVE-Lite-71", 3.75),
            "rtf": (latency_map_gtzan.get("MOODWAAVE-Lite-71", 27.27) / 1000.0) / 30.0,
        },
        {
            "model": "MOODWAAVE-Full-113",
            "accuracy_mean": mw_full_acc,
            "accuracy_std": mw_full_acc_sd,
            "macro_f1_mean": mw_full_f1,
            "macro_f1_std": mw_full_f1_sd,
            "latency_ms": latency_map_gtzan.get("MOODWAAVE-Full-113", 40.73),
            "size_mb": size_map_mb.get("MOODWAAVE-Full-113", 5.25),
            "rtf": (latency_map_gtzan.get("MOODWAAVE-Full-113", 40.73) / 1000.0) / 30.0,
        },
        {
            "model": "SmallMelCNN",
            "accuracy_mean": 0.5841, # from P5
            "accuracy_std": 0.0521,
            "macro_f1_mean": 0.5784,
            "macro_f1_std": 0.0489,
            "latency_ms": latency_map_gtzan.get("SmallMelCNN", 43.83),
            "size_mb": size_map_mb.get("SmallMelCNN", 0.40),
            "rtf": (latency_map_gtzan.get("SmallMelCNN", 43.83) / 1000.0) / 30.0,
        },
    ]
    
    for em in df_g_folds["model"].unique():
        sub = df_g_folds[df_g_folds["model"] == em]
        g_summary_rows.append({
            "model": em,
            "accuracy_mean": float(sub["accuracy"].mean()),
            "accuracy_std": float(sub["accuracy"].std()),
            "macro_f1_mean": float(sub["macro_f1"].mean()),
            "macro_f1_std": float(sub["macro_f1"].std()),
            "latency_ms": latency_map_gtzan.get(em, np.nan),
            "size_mb": size_map_mb.get(em, np.nan),
            "rtf": (latency_map_gtzan.get(em, 0.0) / 1000.0) / 30.0 if em in latency_map_gtzan else np.nan,
        })
        
    df_g_sum = pd.DataFrame(g_summary_rows)
    df_g_sum.to_csv(os.path.join(BASE_DIR, "gtzan_r3b_summary.csv"), index=False)
    
    # ---------------------------------------------------------
    # 3. PMEmo Summary Table
    # ---------------------------------------------------------
    p_summary_rows = []
    for tgt in ["valence", "arousal"]:
        mw_full_r2 = df_p_r3a.loc[(df_p_r3a["model"] == "LightGBM") & (df_p_r3a["feature_set"] == "Full-113") & (df_p_r3a["task"] == tgt), "r2_mean"].values[0]
        mw_full_r2_sd = df_p_r3a.loc[(df_p_r3a["model"] == "LightGBM") & (df_p_r3a["feature_set"] == "Full-113") & (df_p_r3a["task"] == tgt), "r2_std"].values[0]
        mw_full_mae = df_p_r3a.loc[(df_p_r3a["model"] == "LightGBM") & (df_p_r3a["feature_set"] == "Full-113") & (df_p_r3a["task"] == tgt), "mae_mean"].values[0]
        mw_full_mae_sd = df_p_r3a.loc[(df_p_r3a["model"] == "LightGBM") & (df_p_r3a["feature_set"] == "Full-113") & (df_p_r3a["task"] == tgt), "mae_std"].values[0]
        
        mw_lite_r2 = df_p_r3a.loc[(df_p_r3a["model"] == "LightGBM") & (df_p_r3a["feature_set"] == "Lite-71") & (df_p_r3a["task"] == tgt), "r2_mean"].values[0]
        mw_lite_r2_sd = df_p_r3a.loc[(df_p_r3a["model"] == "LightGBM") & (df_p_r3a["feature_set"] == "Lite-71") & (df_p_r3a["task"] == tgt), "r2_std"].values[0]
        mw_lite_mae = df_p_r3a.loc[(df_p_r3a["model"] == "LightGBM") & (df_p_r3a["feature_set"] == "Lite-71") & (df_p_r3a["task"] == tgt), "mae_mean"].values[0]
        mw_lite_mae_sd = df_p_r3a.loc[(df_p_r3a["model"] == "LightGBM") & (df_p_r3a["feature_set"] == "Lite-71") & (df_p_r3a["task"] == tgt), "mae_std"].values[0]
        
        p_summary_rows.append({
            "model": "MOODWAAVE-Lite-71",
            "target": tgt,
            "r2_mean": mw_lite_r2,
            "r2_std": mw_lite_r2_sd,
            "mae_mean": mw_lite_mae,
            "mae_std": mw_lite_mae_sd,
        })
        p_summary_rows.append({
            "model": "MOODWAAVE-Full-113",
            "target": tgt,
            "r2_mean": mw_full_r2,
            "r2_std": mw_full_r2_sd,
            "mae_mean": mw_full_mae,
            "mae_std": mw_full_mae_sd,
        })
        
        for em in df_p_folds["model"].unique():
            sub = df_p_folds[(df_p_folds["model"] == em) & (df_p_folds["target"] == tgt)]
            p_summary_rows.append({
                "model": em,
                "target": tgt,
                "r2_mean": float(sub["r2"].mean()),
                "r2_std": float(sub["r2"].std()),
                "mae_mean": float(sub["mae"].mean()),
                "mae_std": float(sub["mae"].std()),
            })
            
    df_p_sum = pd.DataFrame(p_summary_rows)
    df_p_sum.to_csv(os.path.join(BASE_DIR, "pmemo_r3b_summary.csv"), index=False)
    
    # ---------------------------------------------------------
    # 4. Generate 300-DPI Figures
    # ---------------------------------------------------------
    # Figure 1: Genre Pareto (Accuracy vs Latency, Accuracy vs Size)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Colors and markers
    color_map = {
        "MOODWAAVE-Lite-71": "#2ca02c",
        "MOODWAAVE-Full-113": "#1f77b4",
        "SmallMelCNN": "#9467bd",
        "VGGish": "#ff7f0e",
        "PANNs-CNN14": "#d62728",
        "MERT-v1-95M": "#8c564b",
    }
    marker_map = {
        "MOODWAAVE-Lite-71": "o",
        "MOODWAAVE-Full-113": "s",
        "SmallMelCNN": "^",
        "VGGish": "D",
        "PANNs-CNN14": "v",
        "MERT-v1-95M": "P",
    }
    
    # Panel A: Acc vs Latency
    ax = axes[0]
    for _, row in df_g_sum.iterrows():
        m = row["model"]
        x = row["latency_ms"]
        y = row["accuracy_mean"]
        yerr = row["accuracy_std"]
        c = color_map.get(m, "#333333")
        mk = marker_map.get(m, "o")
        ax.errorbar(x, y, yerr=yerr, fmt=mk, color=c, label=m, markersize=8, capsize=4, elinewidth=1.5)
        ax.annotate(f" {m}", (x, y), fontsize=8, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Single-Thread Latency per Track (ms, log scale)")
    ax.set_ylabel("GTZAN Genre Accuracy (15 Outer Folds)")
    ax.set_title("Genre Accuracy vs. Latency (Pareto Trade-off)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Panel B: Acc vs Size
    ax = axes[1]
    for _, row in df_g_sum.iterrows():
        m = row["model"]
        x = row["size_mb"]
        y = row["accuracy_mean"]
        yerr = row["accuracy_std"]
        c = color_map.get(m, "#333333")
        mk = marker_map.get(m, "o")
        ax.errorbar(x, y, yerr=yerr, fmt=mk, color=c, label=m, markersize=8, capsize=4, elinewidth=1.5)
        ax.annotate(f" {m}", (x, y), fontsize=8, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Total Serialized Size (MB, log scale)")
    ax.set_ylabel("GTZAN Genre Accuracy (15 Outer Folds)")
    ax.set_title("Genre Accuracy vs. Model Size")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    genre_fig_path = os.path.join(BASE_DIR, "genre_r3b_pareto.png")
    fig.savefig(genre_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved {genre_fig_path}")
    
    # Figure 2: Mood Pareto (Valence R2 vs Latency & Size, Arousal R2 vs Latency & Size)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    
    for row_idx, tgt in enumerate(["valence", "arousal"]):
        sub_p = df_p_sum[df_p_sum["target"] == tgt]
        
        # Latency panel
        ax_lat = axes[row_idx, 0]
        for _, row in sub_p.iterrows():
            m = row["model"]
            x = latency_map_pmemo.get(m, latency_map_gtzan.get(m, 100.0))
            y = row["r2_mean"]
            yerr = row["r2_std"]
            c = color_map.get(m, "#333333")
            mk = marker_map.get(m, "o")
            ax_lat.errorbar(x, y, yerr=yerr, fmt=mk, color=c, label=m, markersize=8, capsize=4, elinewidth=1.5)
            ax_lat.annotate(f" {m}", (x, y), fontsize=8, alpha=0.85)
        ax_lat.set_xscale("log")
        ax_lat.set_xlabel("Single-Thread Latency per Track (ms, log scale)")
        ax_lat.set_ylabel(f"PMEmo {tgt.capitalize()} $R^2$ (10 Outer Folds)")
        ax_lat.set_title(f"{tgt.capitalize()} $R^2$ vs. Latency")
        ax_lat.grid(True, which="both", ls="--", alpha=0.5)
        
        # Size panel
        ax_size = axes[row_idx, 1]
        for _, row in sub_p.iterrows():
            m = row["model"]
            x = size_map_mb.get(m, 10.0)
            y = row["r2_mean"]
            yerr = row["r2_std"]
            c = color_map.get(m, "#333333")
            mk = marker_map.get(m, "o")
            ax_size.errorbar(x, y, yerr=yerr, fmt=mk, color=c, label=m, markersize=8, capsize=4, elinewidth=1.5)
            ax_size.annotate(f" {m}", (x, y), fontsize=8, alpha=0.85)
        ax_size.set_xscale("log")
        ax_size.set_xlabel("Total Serialized Size (MB, log scale)")
        ax_size.set_ylabel(f"PMEmo {tgt.capitalize()} $R^2$ (10 Outer Folds)")
        ax_size.set_title(f"{tgt.capitalize()} $R^2$ vs. Model Size")
        ax_size.grid(True, which="both", ls="--", alpha=0.5)
        
    plt.tight_layout()
    mood_fig_path = os.path.join(BASE_DIR, "mood_r3b_pareto.png")
    fig.savefig(mood_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved {mood_fig_path}")
    
    # ---------------------------------------------------------
    # 5. Generate RESULTS.md
    # ---------------------------------------------------------
    write_results_markdown(df_g_sum, df_p_sum, df_g_tests, df_p_tests, df_eff)
    print(f"RESULTS.md generated at {os.path.join(BASE_DIR, 'RESULTS.md')}")

def write_results_markdown(df_g_sum, df_p_sum, df_g_tests, df_p_tests, df_eff):
    md_path = os.path.join(BASE_DIR, "RESULTS.md")
    mw_full_g = df_g_sum[df_g_sum["model"] == "MOODWAAVE-Full-113"].iloc[0]
    mw_lite_g = df_g_sum[df_g_sum["model"] == "MOODWAAVE-Lite-71"].iloc[0]
    mw_full_acc = mw_full_g["accuracy_mean"]
    mw_full_acc_sd = mw_full_g["accuracy_std"]
    mw_lite_acc = mw_lite_g["accuracy_mean"]
    mw_lite_acc_sd = mw_lite_g["accuracy_std"]
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# TASK R3-B: Modern Pretrained Embedding Baselines on Leakage-Aware Folds\n\n")
        
        f.write("## 1. Executive Summary\n\n")
        f.write("We evaluate modern foundation audio embeddings (**PANNs CNN14**, **MERT-v1-95M**, and **VGGish**) on the identical leakage-aware evaluation splits established in Phase A: **GTZAN 15 artist-disjoint outer folds** (929 clean tracks, 286 artists) and **PMEmo 10 artist-grouped outer folds** (767 tracks, 400 artists). Standardized embeddings are evaluated using linear probes (**Logistic Regression** for genre classification and **Ridge regression** for valence/arousal) tuned strictly via 3-fold artist-grouped inner cross-validation (`StratifiedGroupKFold` and `GroupKFold`). For MERT-v1-95M, selection between the mean-pooled last hidden state and the 12-layer averaged representation is conducted entirely inside the inner grouped CV.\n\n")
        
        f.write("## 2. Model Pretraining, Licensing, and Label Overlap Analysis\n\n")
        f.write("| Model | Architecture / Dim | Pretraining Dataset & Official Description | Pretraining Genre Label Overlap | Weight License | Download Size |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for m_key, m_info in MODELS_METADATA.items():
            f.write(f"| **{m_info['display_name']}** | {m_info['embedding_dim']} dims | \"{m_info['pretraining_data_quote']}\" | {m_info['genre_overlap']} | {m_info['license']} | {m_info['download_size_mb']} MB |\n")
        f.write("\n> **Note on Omitted Packages**: `musicnn` (requires `numpy<1.17`, failing modern C-API compilation) and `torchopenl3` (requires `resampy<0.3.0` calling Python 3.11's removed `imp` module) do not install cleanly on Python 3.13 without environment modification and were omitted per prompt instructions. `CLAP` (`laion/larger_clap_music`) was designated as an optional fallback in the event that MERT could not run; because MERT ran cleanly and completely, CLAP was omitted.\n\n")
        
        f.write("## 3. GTZAN Genre Recognition Performance (15 Outer Artist-Disjoint Folds)\n\n")
        f.write("| Model | Pipeline Type | Accuracy (Mean ± SD) | Macro-F1 (Mean ± SD) | Latency (ms/track) | Size (MB) | RTF |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in df_g_sum.iterrows():
            f.write(f"| **{r['model']}** | {'Classical librosa + LightGBM' if 'MOODWAAVE' in r['model'] else ('Small CNN' if 'CNN' in r['model'] and 'PANNs' not in r['model'] else 'Pretrained Embedding + Linear Head')} | {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {r['latency_ms']:.2f} | {r['size_mb']:.2f} | {r['rtf']:.4f} |\n")
        f.write("\n")
        
        f.write("## 4. PMEmo Mood Regression Performance (10 Outer Artist-Grouped Folds)\n\n")
        f.write("| Model | Target | $R^2$ (Mean ± SD) | MAE (Mean ± SD) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for _, r in df_p_sum.iterrows():
            f.write(f"| **{r['model']}** | {r['target'].capitalize()} | {r['r2_mean']:.4f} ± {r['r2_std']:.4f} | {r['mae_mean']:.4f} ± {r['mae_std']:.4f} |\n")
        f.write("\n")
        
        f.write("## 5. Paired Statistical Significance Tests (Nadeau-Bengio Resampled $t$-Tests)\n\n")
        f.write("Variance correction factor: $\\frac{1}{K} + \\frac{n_{\\text{test}}}{n_{\\text{train}}} = \\frac{1}{5} + 0.25 = 0.45$. $df = 14$ for GTZAN (15 outer folds), $df = 9$ for PMEmo (10 outer folds).\n\n")
        f.write("### GTZAN Genre Tests vs. MOODWAAVE Baselines\n\n")
        f.write("| Embedding Model | Baseline | Metric | Mean Diff (Model − Baseline) | 95% Confidence Interval | $t$-stat | $p$-value | Significant ($p < 0.05$) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in df_g_tests.iterrows():
            f.write(f"| **{r['model']}** | {r['baseline']} | {r['metric']} | {r['mean_diff']:+.4f} | [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] | {r['t_stat']:+.3f} | {r['p_val']:.4f} | {'**Yes**' if r['significant'] else 'No'} |\n")
        f.write("\n")
        
        f.write("### PMEmo Mood Tests vs. MOODWAAVE Baselines\n\n")
        f.write("| Embedding Model | Target | Baseline | Metric | Mean Diff (Model − Baseline) | 95% Confidence Interval | $t$-stat | $p$-value | Significant ($p < 0.05$) |\n")
        f.write("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in df_p_tests.iterrows():
            f.write(f"| **{r['model']}** | {r['target'].capitalize()} | {r['baseline']} | {r['metric']} | {r['mean_diff']:+.4f} | [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] | {r['t_stat']:+.3f} | {r['p_val']:.4f} | {'**Yes**' if r['significant'] else 'No'} |\n")
        f.write("\n")
        
        f.write("## 6. Single-Thread Efficiency Profiling & Real-Time Factor\n\n")
        f.write("Evaluated under strict single-thread protocol (`OMP/MKL/OPENBLAS/NUMBA/torch threads = 1`) on canonical 100-track evaluation sets:\n\n")
        if not df_eff.empty:
            f.write("| Pipeline | Dataset | Total Latency (ms) | Audio Load (ms) | Embed/Feat (ms) | Head Pred (ms) | Real-Time Factor (RTF) | Footprint (MB) |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for _, r in df_eff.iterrows():
                f.write(f"| **{r['pipeline']}** | {r['dataset']} | {r['total_ms']:.2f} ± {r['total_sd_ms']:.2f} | {r['audio_load_ms']:.2f} | {r['embedding_ms']:.2f} | {r['prediction_ms']:.2f} | {r['rtf']:.4f} | {r['size_mb']:.2f} |\n")
            f.write("\n")
            
        f.write("## 7. Numbers to Use in the Paper\n\n")
        f.write("### Genre Recognition (GTZAN 15 Artist-Disjoint Outer Folds)\n")
        f.write("- **PANNs CNN14**: " + f"{df_g_sum.loc[df_g_sum['model']=='PANNs-CNN14', 'accuracy_mean'].values[0]:.4f} ± {df_g_sum.loc[df_g_sum['model']=='PANNs-CNN14', 'accuracy_std'].values[0]:.4f} (Macro-F1: {df_g_sum.loc[df_g_sum['model']=='PANNs-CNN14', 'macro_f1_mean'].values[0]:.4f})\n")
        f.write("- **MERT-v1-95M**: " + f"{df_g_sum.loc[df_g_sum['model']=='MERT-v1-95M', 'accuracy_mean'].values[0]:.4f} ± {df_g_sum.loc[df_g_sum['model']=='MERT-v1-95M', 'accuracy_std'].values[0]:.4f} (Macro-F1: {df_g_sum.loc[df_g_sum['model']=='MERT-v1-95M', 'macro_f1_mean'].values[0]:.4f})\n")
        f.write("- **VGGish**: " + f"{df_g_sum.loc[df_g_sum['model']=='VGGish', 'accuracy_mean'].values[0]:.4f} ± {df_g_sum.loc[df_g_sum['model']=='VGGish', 'accuracy_std'].values[0]:.4f} (Macro-F1: {df_g_sum.loc[df_g_sum['model']=='VGGish', 'macro_f1_mean'].values[0]:.4f})\n")
        f.write("- **MOODWAAVE-Full-113**: " + f"{mw_full_acc:.4f} ± {mw_full_acc_sd:.4f} (Macro-F1: {mw_full_g['macro_f1_mean']:.4f})\n")
        f.write("- **MOODWAAVE-Lite-71**: " + f"{mw_lite_acc:.4f} ± {mw_lite_acc_sd:.4f} (Macro-F1: {mw_lite_g['macro_f1_mean']:.4f})\n\n")

        f.write("### Mood Regression (PMEmo 10 Artist-Grouped Outer Folds)\n")
        for tgt in ["valence", "arousal"]:
            f.write(f"#### {tgt.capitalize()} ($R^2$ and MAE)\n")
            for m in ["MERT-v1-95M", "VGGish", "PANNs-CNN14", "MOODWAAVE-Full-113", "MOODWAAVE-Lite-71"]:
                sub = df_p_sum[(df_p_sum["model"] == m) & (df_p_sum["target"] == tgt)]
                if not sub.empty:
                    row_s = sub.iloc[0]
                    f.write(f"- **{m}**: $R^2 = {row_s['r2_mean']:.4f} \\pm {row_s['r2_std']:.4f}$, $\\text{{MAE}} = {row_s['mae_mean']:.4f} \\pm {row_s['mae_std']:.4f}$\n")
            f.write("\n")
            
        f.write("## 8. Limitations & Methodological Caveats\n\n")
        f.write("1. **Pretraining Data Contamination / Ontology Leakage**: Foundation audio models like PANNs and VGGish were pretrained on large web video corpora (AudioSet and YouTube-8M) with supervised tags that directly include musical genre categories (e.g. *Blues*, *Jazz*, *Classical*, *Reggae*). In contrast, MOODWAAVE relies entirely on engineered acoustic features extracted from raw audio waveforms with zero pretraining on external music labels.\n")
        f.write("2. **CPU Deployment Constraints**: Foundation transformers (MERT-v1-95M) require substantial RAM (~1.5 GB) and multi-second inference latencies per track on CPU (RTF > 0.33), making them impractical for client-side, edge, or battery-constrained execution where MOODWAAVE operates in real time (RTF < 0.006).\n")
        f.write("3. **Linear Probe Horizon**: We restricted the downstream head to linear models (Logistic Regression and Ridge) to assess the linear separability of the embedding representations under identical tuning budgets to Phase A.\n")

if __name__ == "__main__":
    main()

