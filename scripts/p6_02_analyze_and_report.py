"""
experiments/rev2_p6_conformal/02_analyze_and_report.py
Generates 300-DPI publication figures and comprehensive markdown report (RESULTS.md).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Professional publication style
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

METHOD_LABELS = {
    "M1_Split_Conformal": "M1: Split-Conformal (Box)",
    "M2_CQR": "M2: CQR (Quantiles)",
    "M3_Normalised": "M3: Locally Adaptive (Norm)",
    "M4_Joint_2D": "M4: Joint 2-D (Ellipse)",
    "M5_LAC": "M5: LAC (Least Ambiguous)",
    "M6_APS": "M6: APS (Adaptive Prediction)",
}

METHOD_COLORS = {
    "M1_Split_Conformal": "#1f77b4", # blue
    "M2_CQR": "#ff7f0e",            # orange
    "M3_Normalised": "#2ca02c",      # green
    "M4_Joint_2D": "#d62728",        # red
    "M5_LAC": "#9467bd",             # purple
    "M6_APS": "#8c564b",             # brown
}

METHOD_MARKERS = {
    "M1_Split_Conformal": "o",
    "M2_CQR": "s",
    "M3_Normalised": "^",
    "M4_Joint_2D": "D",
    "M5_LAC": "v",
    "M6_APS": "P",
}

def generate_figures():
    df_conf = pd.read_csv(os.path.join(BASE_DIR, "conformal_sweep_results.csv"))
    df_const = pd.read_csv(os.path.join(BASE_DIR, "constant_k_baseline.csv"))
    df_80 = pd.read_csv(os.path.join(BASE_DIR, "grid_comparison_80pct.csv"))
    
    # --------------------------------------------------------------------------
    # Figure 1: G1 Coverage vs Set Size (M1-M6 + Constant-k)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    # Plot Constant-k Baseline for G1
    df_const_g1 = df_const[df_const["grid"] == "G1"].sort_values("k")
    ax.plot(
        df_const_g1["empirical_coverage_mean"],
        df_const_g1["k"],
        color="gray",
        linestyle="--",
        linewidth=1.8,
        marker="x",
        markersize=6,
        label="Constant-k Baseline (Top-k)",
        zorder=2,
    )
    
    # Plot M1-M6
    df_g1 = df_conf[df_conf["grid"] == "G1"]
    for m_id, m_label in METHOD_LABELS.items():
        sub = df_g1[df_g1["method"] == m_id].sort_values("nominal_coverage")
        ax.plot(
            sub["empirical_coverage_mean"],
            sub["mean_set_size_mean"],
            color=METHOD_COLORS[m_id],
            marker=METHOD_MARKERS[m_id],
            markersize=5.5,
            linewidth=1.6,
            label=m_label,
            zorder=3,
        )
        
    ax.set_title("MOODWAAVE: Empirical Coverage vs. Mean Set Size on G1 (Equal-Width 4x4, 16 Zones)", fontweight="bold")
    ax.set_xlabel("Empirical Zone Set Coverage")
    ax.set_ylabel("Mean Prediction Set Size (Number of Zones)")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0, 16.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left", edgecolor="#cccccc")
    
    fig_path1 = os.path.join(BASE_DIR, "g1_coverage_vs_set_size.png")
    fig.savefig(fig_path1)
    plt.close(fig)
    print(f"Generated {fig_path1}")
    
    # --------------------------------------------------------------------------
    # Figure 2: G1 Set Size Bar Chart at 80% and 90% Nominal Coverage
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    df_80_g1 = df_conf[(df_conf["grid"] == "G1") & (df_conf["nominal_coverage"] == 0.80)].set_index("method")
    df_90_g1 = df_conf[(df_conf["grid"] == "G1") & (df_conf["nominal_coverage"] == 0.90)].set_index("method")
    
    methods = list(METHOD_LABELS.keys())
    x = np.arange(len(methods))
    width = 0.35
    
    vals_80 = [df_80_g1.loc[m, "mean_set_size_mean"] for m in methods]
    errs_80 = [df_80_g1.loc[m, "mean_set_size_sd"] for m in methods]
    vals_90 = [df_90_g1.loc[m, "mean_set_size_mean"] for m in methods]
    errs_90 = [df_90_g1.loc[m, "mean_set_size_sd"] for m in methods]
    
    rects1 = ax.bar(x - width/2, vals_80, width, yerr=errs_80, capsize=3.5, label="80% Nominal Coverage", color="#4575b4", edgecolor="#1a3860")
    rects2 = ax.bar(x + width/2, vals_90, width, yerr=errs_90, capsize=3.5, label="90% Nominal Coverage", color="#d73027", edgecolor="#6b1410")
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
                    
    ax.set_title("MOODWAAVE: Mean Set Size by Conformal Method on G1 (80% vs 90% Nominal Coverage)", fontweight="bold")
    ax.set_ylabel("Mean Set Size (Zones out of 16)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("_")[0] + "\n" + "_".join(m.split("_")[1:]) for m in methods])
    ax.set_ylim(0, max(vals_90) * 1.25)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, edgecolor="#cccccc")
    
    fig_path2 = os.path.join(BASE_DIR, "g1_set_size_bar_chart.png")
    fig.savefig(fig_path2)
    plt.close(fig)
    print(f"Generated {fig_path2}")
    
    # --------------------------------------------------------------------------
    # Figure 3: Grid Comparison Bar Chart (G1 vs G2 vs G3 at 80% Nominal Coverage)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    
    grids = ["G1", "G2", "G3"]
    grid_display = {
        "G1": "G1: Equal-width 4x4 (16 zones)",
        "G2": "G2: Quantile 4x4 (16 zones)",
        "G3": "G3: Quadrants 2x2 (4 zones)",
    }
    grid_colors = ["#2b83ba", "#fdae61", "#abdda4"]
    
    x = np.arange(len(methods))
    width = 0.26
    
    for i_g, g_code in enumerate(grids):
        df_sub = df_80[df_80["grid"] == g_code].set_index("method")
        vals = [df_sub.loc[m, "mean_set_size_mean"] for m in methods]
        errs = [df_sub.loc[m, "mean_set_size_sd"] for m in methods]
        rects = ax.bar(x + (i_g - 1) * width, vals, width, yerr=errs, capsize=3, label=grid_display[g_code], color=grid_colors[i_g], edgecolor="#333333")
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)
                        
    ax.set_title("MOODWAAVE: Spatial Grid Comparison at 80% Nominal Coverage (G1 vs G2 vs G3)", fontweight="bold")
    ax.set_ylabel("Mean Set Size (Zones)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("_")[0] + "\n" + "_".join(m.split("_")[1:]) for m in methods])
    ax.set_ylim(0, 10.5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, edgecolor="#cccccc")
    
    fig_path3 = os.path.join(BASE_DIR, "grid_comparison_bar_chart.png")
    fig.savefig(fig_path3)
    plt.close(fig)
    print(f"Generated {fig_path3}")

if __name__ == "__main__":
    generate_figures()

