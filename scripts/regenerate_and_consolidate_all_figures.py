"""Regenerate figures with corrected project name "MOODWAAVE" and consolidate all publication figures into release/figures/.
"""
import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "figure.autolayout": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REL_FIG = os.path.join(ROOT, "release", "figures")
os.makedirs(REL_FIG, exist_ok=True)

# -------------------------------------------------------------
# 1. Regenerate confusion_matrix_genre_lite.png
# -------------------------------------------------------------
print("Regenerating confusion_matrix_genre_lite.png with 'MOODWAAVE-Lite'...")
gtzan_csv = os.path.join(ROOT, "data", "gtzan_features.csv")
df_gtzan = pd.read_csv(gtzan_csv)
if "filename" in df_gtzan.columns:
    df_gtzan = df_gtzan[df_gtzan["filename"] != "jazz.00054.wav"].reset_index(drop=True)
elif "track_id" in df_gtzan.columns:
    df_gtzan = df_gtzan.dropna(subset=["track_id"]).reset_index(drop=True)

y = df_gtzan["genre"] if "genre" in df_gtzan.columns else df_gtzan["label"]
classes = sorted(y.unique())

with open(os.path.join(ROOT, "experiments", "idea2_cost_pareto", "feature_groups.json"), "r") as f:
    fg_data = json.load(f)
groups = fg_data["groups"]
lite_features = []
for g in ["MFCC", "SpecContrast", "Temporal"]:
    lite_features.extend(groups[g])

X_lite = df_gtzan[lite_features]

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
oof_y_true, oof_y_pred_lite = [], []

lgb_params = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 1
}

for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_lite, y)):
    m_lite = lgb.LGBMClassifier(**lgb_params)
    m_lite.fit(X_lite.iloc[train_idx], y.iloc[train_idx])
    p_lite = m_lite.predict(X_lite.iloc[test_idx])
    oof_y_true.extend(y.iloc[test_idx])
    oof_y_pred_lite.extend(p_lite)

cm = confusion_matrix(oof_y_true, oof_y_pred_lite, normalize="true")
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(cm * 100, annot=True, fmt=".1f", cmap="Blues", cbar=True,
            xticklabels=classes, yticklabels=classes, ax=ax,
            annot_kws={"size": 9, "fontweight": "medium"})
ax.set_title("Normalized Confusion Matrix: MOODWAAVE-Lite (71 Features)\n(Pooled 5x3 Repeated Stratified CV, N = 2,997 predictions)",
             fontweight="bold", pad=12)
ax.set_xlabel("Predicted Genre", fontweight="bold")
ax.set_ylabel("True Genre", fontweight="bold")
plt.tight_layout()
cm_out = os.path.join(REL_FIG, "confusion_matrix_genre_lite.png")
plt.savefig(cm_out)
plt.close()
print(f"Saved: {cm_out}")

# -------------------------------------------------------------
# 2. Regenerate mean_set_size_vs_coverage.png
# -------------------------------------------------------------
print("Regenerating mean_set_size_vs_coverage.png with 'MOODWAAVE-Lite'...")
sweep_csv = os.path.join(ROOT, "experiments", "final_consolidation", "part_c_conformal_sweep.csv")
df_sweep = pd.read_csv(sweep_csv)

fig, ax = plt.subplots(figsize=(7, 4.8))
grp = df_sweep.groupby("nominal_coverage")
covs = [c * 100 for c in grp.groups.keys()]
means = [g["mean_set_size"].mean() for _, g in grp]
stds = [g["mean_set_size"].std() for _, g in grp]

ax.errorbar(covs, means, yerr=stds, fmt="o-", color="#1f77b4", ecolor="#1f77b4",
            capsize=5, elinewidth=1.5, linewidth=2, markersize=7, label="Empirical Zone-Set Size")

for x, y_val in zip(covs, means):
    ax.annotate(f"{y_val:.2f} zones", xy=(x, y_val), xytext=(0, 9),
                textcoords="offset points", ha="center", fontweight="bold", fontsize=9.5)

ax.set_title("MOODWAAVE-Lite: Mean Emotion Zone-Set Size vs. Coverage Level\n(20 Random Seeds, 60/20/20 Splits, N = 154 Test Tracks)",
             fontweight="bold", pad=12)
ax.set_xlabel("Nominal Target Coverage Level (%)", fontweight="bold")
ax.set_ylabel("Average Number of Emotion Zones in Set $|S(x)|$", fontweight="bold")
ax.set_xticks(covs)
ax.set_ylim(1, 16.5)
ax.grid(True, linestyle=":", alpha=0.6)
ax.axhline(1.0, color="#d62728", linestyle="--", alpha=0.7, label="Single-Zone Certainty ($|S| = 1$)")
ax.axhline(16.0, color="#7f7f7f", linestyle=":", alpha=0.7, label="Maximum Ambiguity ($|S| = 16$)")
ax.legend(loc="upper left", framealpha=0.95)
plt.tight_layout()
cov_out = os.path.join(REL_FIG, "mean_set_size_vs_coverage.png")
plt.savefig(cov_out)
plt.close()
print(f"Saved: {cov_out}")

# -------------------------------------------------------------
# 3. Regenerate va_box_scatter_80pct.png
# -------------------------------------------------------------
print("Regenerating va_box_scatter_80pct.png with 'MOODWAAVE-Lite'...")
samples_csv = os.path.join(ROOT, "experiments", "final_consolidation", "sample_predictions_80pct.csv")
df_samples = pd.read_csv(samples_csv)

ZONE_MATRIX = [
    ["Depressed", "Sad", "Brooding", "Melancholy"],
    ["Sleepy", "Calm", "Peaceful", "Relaxed"],
    ["Pensive", "Serene", "Pleased", "Content"],
    ["Enraged", "Aggressive", "Excited", "Euphoric"]
]
GRID_BINS = [0.0, 0.25, 0.50, 0.75, 1.0]

fig, ax = plt.subplots(figsize=(8.5, 8.5))

for b in [0.25, 0.50, 0.75]:
    ax.axvline(b, color="#888888", linestyle="-", linewidth=1.2, alpha=0.6)
    ax.axhline(b, color="#888888", linestyle="-", linewidth=1.2, alpha=0.6)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)

ax.axvline(0.50, color="#333333", linestyle="-", linewidth=2.0, alpha=0.8)
ax.axhline(0.50, color="#333333", linestyle="-", linewidth=2.0, alpha=0.8)

for r in range(4):
    for c in range(4):
        zone_name = ZONE_MATRIX[r][c]
        cx = (GRID_BINS[c] + GRID_BINS[c + 1]) / 2.0
        cy = (GRID_BINS[r] + GRID_BINS[r + 1]) / 2.0
        ax.text(cx, cy, zone_name.upper(), color="#b8b8b8",
                fontsize=10.5, fontweight="bold", ha="center", va="center", alpha=0.45)

ax.text(0.98, 0.98, "Q1: HAPPY", color="#2b5c8f", fontweight="bold", ha="right", va="top", fontsize=11)
ax.text(0.02, 0.98, "Q2: ANGRY", color="#a33a3a", fontweight="bold", ha="left", va="top", fontsize=11)
ax.text(0.02, 0.02, "Q3: SAD", color="#6d6d6d", fontweight="bold", ha="left", va="bottom", fontsize=11)
ax.text(0.98, 0.02, "Q4: CALM", color="#3b7a57", fontweight="bold", ha="right", va="bottom", fontsize=11)

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
va_out = os.path.join(REL_FIG, "va_box_scatter_80pct.png")
plt.savefig(va_out)
plt.close()
print(f"Saved: {va_out}")

# -------------------------------------------------------------
# 4. Copy all other publication figures
# -------------------------------------------------------------
copy_list = [
    # Interpretability
    ("experiments/final_consolidation/shap_genre_bar.png", "shap_genre_bar.png"),
    ("experiments/final_consolidation/shap_valence_summary.png", "shap_valence_summary.png"),
    ("experiments/final_consolidation/shap_arousal_summary.png", "shap_arousal_summary.png"),
    # Split leakage
    ("experiments/rev2_p1_splits/gtzan_faults_by_genre.png", "gtzan_faults_by_genre.png"),
    ("experiments/rev2_p1_splits/gtzan_leakage_drop.png", "gtzan_leakage_drop.png"),
    ("experiments/rev2_p1_splits/pmemo_leakage_comparison.png", "pmemo_leakage_comparison.png"),
    # Tonnetz
    ("experiments/rev2_p2_tonnetz/accuracy_latency_pareto_tonnetz.png", "accuracy_latency_pareto_tonnetz.png"),
    ("experiments/rev2_p2_tonnetz/tonnetz_cost_comparison.png", "tonnetz_cost_comparison.png"),
    # Selection stability
    ("experiments/rev2_p3_selection/selection_frequencies.png", "selection_frequencies.png"),
    ("experiments/rev2_p3_selection/tost_equivalence_confidence_intervals.png", "tost_equivalence_confidence_intervals.png"),
    # Baselines
    ("experiments/rev2_p4_baselines/genre_baselines_accuracy.png", "genre_baselines_accuracy.png"),
    ("experiments/rev2_p4_baselines/latency_vs_performance.png", "latency_vs_performance.png"),
    ("experiments/rev2_p4_baselines/mood_baselines_r2.png", "mood_baselines_r2.png"),
    # Deep comparison
    ("experiments/rev2_p5_deep/cnn_training_curves.png", "cnn_training_curves.png"),
    ("experiments/rev2_p5_deep/deep_vs_moodwaave_accuracy.png", "deep_vs_moodwaave_accuracy.png"),
    ("experiments/rev2_p5_deep/deep_vs_moodwaave_efficiency.png", "deep_vs_moodwaave_efficiency.png"),
    # Conformal rev2
    ("experiments/rev2_p6_conformal/g1_coverage_vs_set_size.png", "g1_coverage_vs_set_size.png"),
    ("experiments/rev2_p6_conformal/g1_set_size_bar_chart.png", "g1_set_size_bar_chart.png"),
    ("experiments/rev2_p6_conformal/grid_comparison_bar_chart.png", "grid_comparison_bar_chart.png"),
    # Master timing
    ("experiments/rev2_final_timing/final_latency_breakdown_bar_chart.png", "final_latency_breakdown_bar_chart.png"),
    ("experiments/rev2_final_timing/final_latency_vs_size_pareto.png", "final_latency_vs_size_pareto.png")
]

print("\nCopying existing publication figures...")
for src_rel, dst_name in copy_list:
    src_path = os.path.join(ROOT, src_rel)
    dst_path = os.path.join(REL_FIG, dst_name)
    shutil.copy2(src_path, dst_path)
    print(f"Copied: {src_rel} -> release/figures/{dst_name}")

print("\nAll figures consolidated successfully in release/figures/!")
