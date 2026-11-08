"""Script 03: Part C - Uncertainty on the Lite Mood Model.
Executes split-conformal prediction on MOODWAAVE-Lite (71 features) across 20 random seeds (60/20/20).
Sweeps coverage targets: 50%, 70%, 80%, 90%, 95%.
Evaluates empirical coverage, interval width, zone-set size, singleton rate, and zone-set coverage.
Generates:
1. mean_set_size_vs_coverage.png (300 DPI)
2. va_box_scatter_80pct.png (300 DPI)
3. Grid occupancy table of True vs Predicted counts.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

import common as C

# Configure plot aesthetic
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

COVERAGE_TARGETS = [0.50, 0.70, 0.80, 0.90, 0.95]
N_SEEDS = 20
SEEDS = list(range(N_SEEDS))


def run_conformal_lite():
    print("\n" + "="*70)
    print("PART C: UNCERTAINTY ON LITE MOOD MODEL (20 Seeds, 60/20/20)")
    print("="*70)
    df_pmemo, _, X_lite, y_val, y_aro = C.load_pmemo_data()
    n_total = len(df_pmemo)
    print(f"Loaded PMEmo: {n_total} tracks, 71 Lite features.")
    
    sweep_records = []
    hard_acc_records = []
    
    # Track occupancy across all 20 seeds
    true_zone_counts = {z: 0 for z in C.ZONE_NAMES}
    pred_zone_counts = {z: 0 for z in C.ZONE_NAMES}
    total_test_tracks_evaluated = 0
    
    sample_30_tracks_80pct = None
    
    for seed in SEEDS:
        # Split 60% Train, 20% Cal, 20% Test
        X_tr_cal, X_test, y_v_tr_cal, y_v_te, y_a_tr_cal, y_a_te, idx_tr_cal, idx_te = train_test_split(
            X_lite, y_val, y_aro, np.arange(n_total), test_size=0.20, random_state=seed
        )
        X_train, X_cal, y_v_tr, y_v_cal, y_a_tr, y_a_cal = train_test_split(
            X_tr_cal, y_v_tr_cal, y_a_tr_cal, test_size=0.25, random_state=seed
        )
        
        # Fit point models on Train
        params = dict(C.LGBM_MOOD_PARAMS)
        params["random_state"] = seed
        m_val = LGBMRegressor(**params, n_jobs=1).fit(X_train, y_v_tr)
        m_aro = LGBMRegressor(**params, n_jobs=1).fit(X_train, y_a_tr)
        
        # Predictions
        v_pred_cal = m_val.predict(X_cal)
        a_pred_cal = m_aro.predict(X_cal)
        v_pred_te = m_val.predict(X_test)
        a_pred_te = m_aro.predict(X_test)
        
        # Absolute residuals on calibration set
        s_val_cal = np.abs(y_v_cal - v_pred_cal)
        s_aro_cal = np.abs(y_a_cal - a_pred_cal)
        
        # Point predictions accuracy & occupancy
        true_zones = [C.get_zone_name(v, a) for v, a in zip(y_v_te, y_a_te)]
        pred_zones = [C.get_zone_name(v, a) for v, a in zip(v_pred_te, a_pred_te)]
        
        acc_point = float(np.mean([tz == pz for tz, pz in zip(true_zones, pred_zones)]))
        n_distinct_pred = len(set(pred_zones))
        hard_acc_records.append({
            "seed": seed,
            "hard_point_acc": acc_point,
            "distinct_zones_predicted": n_distinct_pred
        })
        
        for tz, pz in zip(true_zones, pred_zones):
            true_zone_counts[tz] += 1
            pred_zone_counts[pz] += 1
            total_test_tracks_evaluated += 1
            
        # Conformal sweep
        for target_cov in COVERAGE_TARGETS:
            alpha = 1.0 - target_cov
            
            # Finite-sample conformal quantiles
            q_val = C.finite_sample_conformal_quantile(s_val_cal, alpha)
            q_aro = C.finite_sample_conformal_quantile(s_aro_cal, alpha)
            
            v_lo = np.clip(v_pred_te - q_val, 0.0, 1.0)
            v_hi = np.clip(v_pred_te + q_val, 0.0, 1.0)
            a_lo = np.clip(a_pred_te - q_aro, 0.0, 1.0)
            a_hi = np.clip(a_pred_te + q_aro, 0.0, 1.0)
            
            cov_v = (y_v_te >= v_lo) & (y_v_te <= v_hi)
            cov_a = (y_a_te >= a_lo) & (y_a_te <= a_hi)
            cov_j = cov_v & cov_a
            
            w_v = v_hi - v_lo
            w_a = a_hi - a_lo
            
            zone_sets = [
                C.get_overlapping_zones(v1, v2, a1, a2)
                for v1, v2, a1, a2 in zip(v_lo, v_hi, a_lo, a_hi)
            ]
            set_sizes = np.array([len(s) for s in zone_sets])
            zone_cov = np.array([tz in s for tz, s in zip(true_zones, zone_sets)])
            
            sweep_records.append({
                "seed": seed,
                "nominal_coverage": target_cov,
                "emp_cov_valence": float(np.mean(cov_v)),
                "emp_cov_arousal": float(np.mean(cov_a)),
                "emp_cov_joint": float(np.mean(cov_j)),
                "mean_width_valence": float(np.mean(w_v)),
                "mean_width_arousal": float(np.mean(w_a)),
                "mean_set_size": float(np.mean(set_sizes)),
                "singleton_rate": float(np.mean(set_sizes == 1)),
                "zone_set_coverage": float(np.mean(zone_cov))
            })
            
            # Save 30 sample tracks at 80% coverage from Seed 0
            if seed == 0 and abs(target_cov - 0.80) < 1e-4:
                sample_indices = np.linspace(0, len(X_test) - 1, 30, dtype=int)
                sample_rows = []
                for idx in sample_indices:
                    sample_rows.append({
                        "track_idx": int(idx_te[idx]),
                        "v_true": float(y_v_te[idx]),
                        "a_true": float(y_a_te[idx]),
                        "v_pred": float(v_pred_te[idx]),
                        "a_pred": float(a_pred_te[idx]),
                        "v_lo": float(v_lo[idx]),
                        "v_hi": float(v_hi[idx]),
                        "a_lo": float(a_lo[idx]),
                        "a_hi": float(a_hi[idx]),
                        "true_zone": true_zones[idx],
                        "pred_zone": pred_zones[idx],
                        "zone_set": ";".join(zone_sets[idx]),
                        "set_size": int(set_sizes[idx])
                    })
                sample_30_tracks_80pct = sample_rows

    df_sweep = pd.DataFrame(sweep_records)
    df_sweep.to_csv(os.path.join(C.OUT, "part_c_conformal_sweep.csv"), index=False)
    
    df_hard = pd.DataFrame(hard_acc_records)
    print("\nPoint Model Baseline (Hard Argmax, N = 20 seeds):")
    print(f"  Overall 16-Zone Accuracy:  {df_hard['hard_point_acc'].mean()*100:.2f}% ± {df_hard['hard_point_acc'].std()*100:.2f}%")
    print(f"  Distinct Zones Predicted: {df_hard['distinct_zones_predicted'].mean():.1f} ± {df_hard['distinct_zones_predicted'].std():.1f} (out of 16)")
    
    # Consolidated sweep table
    grp = df_sweep.groupby("nominal_coverage")
    summary_sweep = []
    for cov_target, g in grp:
        summary_sweep.append({
            "Target Coverage": f"{cov_target*100:.0f}%",
            "Valence Coverage": f"{g['emp_cov_valence'].mean()*100:.2f}% ± {g['emp_cov_valence'].std()*100:.2f}%",
            "Arousal Coverage": f"{g['emp_cov_arousal'].mean()*100:.2f}% ± {g['emp_cov_arousal'].std()*100:.2f}%",
            "Joint Coverage": f"{g['emp_cov_joint'].mean()*100:.2f}% ± {g['emp_cov_joint'].std()*100:.2f}%",
            "Mean Valence Width": f"{g['mean_width_valence'].mean():.3f} ± {g['mean_width_valence'].std():.3f}",
            "Mean Arousal Width": f"{g['mean_width_arousal'].mean():.3f} ± {g['mean_width_arousal'].std():.3f}",
            "Mean Zone-Set Size": f"{g['mean_set_size'].mean():.2f} ± {g['mean_set_size'].std():.2f}",
            "Singleton Rate": f"{g['singleton_rate'].mean()*100:.2f}% ± {g['singleton_rate'].std()*100:.2f}%",
            "Zone-Set Coverage": f"{g['zone_set_coverage'].mean()*100:.2f}% ± {g['zone_set_coverage'].std()*100:.2f}%"
        })
    df_sweep_sum = pd.DataFrame(summary_sweep)
    df_sweep_sum.to_csv(os.path.join(C.OUT, "part_c_conformal_summary.csv"), index=False)
    print("\nConformal Coverage Sweep Summary:")
    print(df_sweep_sum.to_string(index=False))
    
    # 4x4 Grid Occupancy Table
    grid_rows = []
    # Print from top row (Arousal 0.75-1.0) to bottom row (Arousal 0.0-0.25)
    for r in reversed(range(4)):
        a_label = f"Arousal [{C.GRID_BINS[r]:.2f}, {C.GRID_BINS[r+1]:.2f}]"
        row_dict = {"Arousal_Range": a_label}
        for c in range(4):
            v_label = f"Valence [{C.GRID_BINS[c]:.2f}, {C.GRID_BINS[c+1]:.2f}]"
            zname = C.ZONE_MATRIX[r][c]
            tc = true_zone_counts[zname]
            pc = pred_zone_counts[zname]
            row_dict[v_label] = f"{zname}: True={tc} | Pred={pc}"
        grid_rows.append(row_dict)
        
    df_grid = pd.DataFrame(grid_rows)
    df_grid.to_csv(os.path.join(C.OUT, "part_c_zone_occupancy_grid.csv"), index=False)
    print("\nZone Occupancy Grid (Pooled over 20 seeds, N = 3,080 track evaluations):")
    print(df_grid.to_string(index=False))
    
    # Plot 1: Mean Set Size vs Nominal Coverage
    plot_set_size_vs_coverage(df_sweep)
    
    # Plot 2: VA Box Scatter at 80% Level
    if sample_30_tracks_80pct:
        df_samples_80 = pd.DataFrame(sample_30_tracks_80pct)
        df_samples_80.to_csv(os.path.join(C.OUT, "sample_predictions_80pct.csv"), index=False)
        plot_va_box_scatter_80pct(df_samples_80)


def plot_set_size_vs_coverage(df_sweep):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    grp = df_sweep.groupby("nominal_coverage")
    covs = [c * 100 for c in grp.groups.keys()]
    means = [g["mean_set_size"].mean() for _, g in grp]
    stds = [g["mean_set_size"].std() for _, g in grp]
    
    ax.errorbar(covs, means, yerr=stds, fmt="o-", color="#1f77b4", ecolor="#1f77b4",
                capsize=5, elinewidth=1.5, linewidth=2, markersize=7, label="Empirical Zone-Set Size")
    
    # Annotate values
    for x, y, s in zip(covs, means, stds):
        ax.annotate(f"{y:.2f} zones", xy=(x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontweight="bold", fontsize=9.5)
        
    ax.set_title("MOODWAAVE-Lite: Mean Emotion Zone-Set Size vs. Coverage Level\n(20 Random Seeds, 60/20/20 Splits, N = 154 Test Tracks)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Nominal Target Coverage Level (%)", fontweight="bold")
    ax.set_ylabel("Average Number of Emotion Zones in Set $|S(x)|$", fontweight="bold")
    ax.set_xticks(covs)
    ax.set_ylim(1, 16.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Reference lines for context
    ax.axhline(1.0, color="#d62728", linestyle="--", alpha=0.7, label="Single-Zone Certainty ($|S| = 1$)")
    ax.axhline(16.0, color="#888888", linestyle=":", alpha=0.7, label="Complete Uncertainty ($|S| = 16$)")
    ax.legend(framealpha=0.95, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT, "mean_set_size_vs_coverage.png"))
    plt.close()
    print("Saved mean_set_size_vs_coverage.png")


def plot_va_box_scatter_80pct(df_samples):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    
    # Draw 4x4 grid lines
    for b in [0.25, 0.50, 0.75]:
        ax.axvline(b, color="#888888", linestyle="-", linewidth=1.2, alpha=0.6)
        ax.axhline(b, color="#888888", linestyle="-", linewidth=1.2, alpha=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    
    # Major quadrant divider
    ax.axvline(0.50, color="#333333", linestyle="-", linewidth=2.0, alpha=0.8)
    ax.axhline(0.50, color="#333333", linestyle="-", linewidth=2.0, alpha=0.8)
    
    # Annotate the 16 emotion zones in background
    for r in range(4):
        for c in range(4):
            zone_name = C.ZONE_MATRIX[r][c]
            cx = (C.GRID_BINS[c] + C.GRID_BINS[c + 1]) / 2.0
            cy = (C.GRID_BINS[r] + C.GRID_BINS[r + 1]) / 2.0
            ax.text(cx, cy, zone_name.upper(), color="#b8b8b8",
                    fontsize=10.5, fontweight="bold", ha="center", va="center", alpha=0.45)
            
    # Quadrant titles
    ax.text(0.98, 0.98, "Q1: HAPPY", color="#2b5c8f", fontweight="bold", ha="right", va="top", fontsize=11)
    ax.text(0.02, 0.98, "Q2: ANGRY", color="#a33a3a", fontweight="bold", ha="left", va="top", fontsize=11)
    ax.text(0.02, 0.02, "Q3: SAD", color="#6d6d6d", fontweight="bold", ha="left", va="bottom", fontsize=11)
    ax.text(0.98, 0.02, "Q4: CALM", color="#3b7a57", fontweight="bold", ha="right", va="bottom", fontsize=11)
    
    # Plot test tracks and conformal 80% bounding boxes
    for _, row in df_samples.iterrows():
        vt, at = row["v_true"], row["a_true"]
        vp, ap = row["v_pred"], row["a_pred"]
        vlo, vhi = row["v_lo"], row["v_hi"]
        alo, ahi = row["a_lo"], row["a_hi"]
        
        rect = patches.Rectangle(
            (vlo, alo), vhi - vlo, ahi - alo,
            linewidth=1.0, edgecolor="#1f77b4", facecolor="#1f77b4",
            alpha=0.10, zorder=2
        )
        ax.add_patch(rect)
        ax.plot([vp, vt], [ap, at], color="#666666", linestyle=":", linewidth=0.8, alpha=0.6, zorder=3)
        
    ax.scatter(df_samples["v_true"], df_samples["a_true"], color="#1f77b4", s=50,
               edgecolor="black", linewidth=0.7, label="True $(V, A)$ Label", zorder=4)
    ax.scatter(df_samples["v_pred"], df_samples["a_pred"], color="#ff7f0e", marker="^", s=55,
               edgecolor="black", linewidth=0.7, label="MOODWAAVE-Lite Point Prediction", zorder=5)
    
    legend_rect = patches.Patch(facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.25,
                                label="Conformal 80% Joint Region (Split-Conformal)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(legend_rect)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=3, framealpha=0.95, fontsize=10)
    
    ax.set_title("Circumplex Valence-Arousal Space with Conformal 80% Joint Prediction Regions\n(MOODWAAVE-Lite, $N = 30$ Sample Test Tracks)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Valence (0.0 = Negative, 1.0 = Positive)", fontweight="bold")
    ax.set_ylabel("Arousal (0.0 = Low, 1.0 = High)", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT, "va_box_scatter_80pct.png"))
    plt.close()
    print("Saved va_box_scatter_80pct.png")


if __name__ == "__main__":
    run_conformal_lite()
