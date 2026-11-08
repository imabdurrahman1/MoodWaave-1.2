"""Script 04: Part D - SHAP Interpretability Analysis for MOODWAAVE-Lite.
Calculates SHAP values using shap.TreeExplainer on:
1. Genre Classifier (10 classes, mean |SHAP| across classes)
2. Valence Regressor
3. Arousal Regressor
Generates 300 DPI summary plots and aggregates importance by feature group (MFCC, SpecContrast, Temporal).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMClassifier, LGBMRegressor

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


def run_shap_analysis():
    print("\n" + "="*70)
    print("PART D: SHAP INTERPRETABILITY FOR MOODWAAVE-LITE (71 Features)")
    print("="*70)
    
    # 1. Fit Lite models on full datasets
    _, _, X_g_lite, y_g, genre_classes = C.load_gtzan_data()
    _, _, X_m_lite, y_val, y_aro = C.load_pmemo_data()
    
    print("Training MOODWAAVE-Lite models for SHAP explanation...")
    m_genre = LGBMClassifier(**C.LGBM_GENRE_PARAMS, n_jobs=1).fit(X_g_lite, y_g)
    m_val = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_m_lite, y_val)
    m_aro = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_m_lite, y_aro)
    
    # -------------------------------------------------------------
    # 2. Genre Classifier SHAP
    # -------------------------------------------------------------
    print("Calculating SHAP values for Genre Classifier...")
    explainer_g = shap.TreeExplainer(m_genre)
    shap_vals_g = explainer_g.shap_values(X_g_lite)
    
    # shap_vals_g can be list of 10 arrays (N, 71) or 3D array (N, 71, 10)
    if isinstance(shap_vals_g, list):
        mean_abs_g = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_g], axis=0)
    elif shap_vals_g.ndim == 3:
        mean_abs_g = np.mean(np.abs(shap_vals_g), axis=(0, 2))
    else:
        mean_abs_g = np.abs(shap_vals_g).mean(axis=0)
        
    df_g_imp = pd.DataFrame({
        "feature": C.LITE_FEATURES,
        "group": [C.FEATURE_TO_GROUP[f] for f in C.LITE_FEATURES],
        "mean_abs_shap": mean_abs_g
    }).sort_values("mean_abs_shap", ascending=False)
    
    # Plot Genre Top 15 Bar Plot
    top15_g = df_g_imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_g = ["#1f77b4" if g == "MFCC" else "#ff7f0e" if g == "SpecContrast" else "#2ca02c" for g in top15_g["group"]]
    ax.barh(top15_g["feature"], top15_g["mean_abs_shap"], color=colors_g, edgecolor="black", linewidth=0.7)
    ax.set_title("Top 15 Features: Genre Classification (Mean |SHAP| across 10 Genres)", fontweight="bold", pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Log-Odds)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    
    # Group legend
    import matplotlib.patches as mpatches
    patches_legend = [
        mpatches.Patch(facecolor="#1f77b4", edgecolor="black", label="MFCC"),
        mpatches.Patch(facecolor="#ff7f0e", edgecolor="black", label="Spectral Contrast"),
        mpatches.Patch(facecolor="#2ca02c", edgecolor="black", label="Temporal")
    ]
    ax.legend(handles=patches_legend, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT, "shap_genre_bar.png"))
    plt.close()
    print("Saved shap_genre_bar.png")
    
    # -------------------------------------------------------------
    # 3. Valence Regressor SHAP
    # -------------------------------------------------------------
    print("Calculating SHAP values for Valence Regressor...")
    explainer_v = shap.TreeExplainer(m_val)
    shap_vals_v = explainer_v.shap_values(X_m_lite)
    if isinstance(shap_vals_v, list):
        shap_vals_v = shap_vals_v[0]
        
    mean_abs_v = np.abs(shap_vals_v).mean(axis=0)
    df_v_imp = pd.DataFrame({
        "feature": C.LITE_FEATURES,
        "group": [C.FEATURE_TO_GROUP[f] for f in C.LITE_FEATURES],
        "mean_abs_shap": mean_abs_v
    }).sort_values("mean_abs_shap", ascending=False)
    
    top15_v = df_v_imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_v = ["#1f77b4" if g == "MFCC" else "#ff7f0e" if g == "SpecContrast" else "#2ca02c" for g in top15_v["group"]]
    ax.barh(top15_v["feature"], top15_v["mean_abs_shap"], color=colors_v, edgecolor="black", linewidth=0.7)
    ax.set_title("Top 15 Features: Valence Regression (Mean |SHAP| on PMEmo)", fontweight="bold", pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Valence Prediction)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax.legend(handles=patches_legend, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT, "shap_valence_summary.png"))
    plt.close()
    print("Saved shap_valence_summary.png")
    
    # -------------------------------------------------------------
    # 4. Arousal Regressor SHAP
    # -------------------------------------------------------------
    print("Calculating SHAP values for Arousal Regressor...")
    explainer_a = shap.TreeExplainer(m_aro)
    shap_vals_a = explainer_a.shap_values(X_m_lite)
    if isinstance(shap_vals_a, list):
        shap_vals_a = shap_vals_a[0]
        
    mean_abs_a = np.abs(shap_vals_a).mean(axis=0)
    df_a_imp = pd.DataFrame({
        "feature": C.LITE_FEATURES,
        "group": [C.FEATURE_TO_GROUP[f] for f in C.LITE_FEATURES],
        "mean_abs_shap": mean_abs_a
    }).sort_values("mean_abs_shap", ascending=False)
    
    top15_a = df_a_imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_a = ["#1f77b4" if g == "MFCC" else "#ff7f0e" if g == "SpecContrast" else "#2ca02c" for g in top15_a["group"]]
    ax.barh(top15_a["feature"], top15_a["mean_abs_shap"], color=colors_a, edgecolor="black", linewidth=0.7)
    ax.set_title("Top 15 Features: Arousal Regression (Mean |SHAP| on PMEmo)", fontweight="bold", pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Arousal Prediction)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax.legend(handles=patches_legend, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT, "shap_arousal_summary.png"))
    plt.close()
    print("Saved shap_arousal_summary.png")
    
    # -------------------------------------------------------------
    # 5. Group-Level Aggregation Table
    # -------------------------------------------------------------
    grp_g = df_g_imp.groupby("group")["mean_abs_shap"].sum()
    grp_v = df_v_imp.groupby("group")["mean_abs_shap"].sum()
    grp_a = df_a_imp.groupby("group")["mean_abs_shap"].sum()
    
    group_summary = []
    for g in C.LITE_GROUPS:
        tot_g = grp_g[g]
        pct_g = tot_g / grp_g.sum() * 100
        tot_v = grp_v[g]
        pct_v = tot_v / grp_v.sum() * 100
        tot_a = grp_a[g]
        pct_a = tot_a / grp_a.sum() * 100
        n_feats = len(C.FEATURE_GROUPS[g])
        
        group_summary.append({
            "Feature Group": g,
            "Feature Count": n_feats,
            "Genre SHAP Sum": f"{tot_g:.4f} ({pct_g:.1f}%)",
            "Valence SHAP Sum": f"{tot_v:.4f} ({pct_v:.1f}%)",
            "Arousal SHAP Sum": f"{tot_a:.4f} ({pct_a:.1f}%)",
            "pct_genre": pct_g,
            "pct_valence": pct_v,
            "pct_arousal": pct_a
        })
    df_group_sum = pd.DataFrame(group_summary)
    df_group_sum.to_csv(os.path.join(C.OUT, "part_d_shap_group_importance.csv"), index=False)
    print("\nSHAP Importance Aggregated by Feature Group:")
    print(df_group_sum[["Feature Group", "Feature Count", "Genre SHAP Sum", "Valence SHAP Sum", "Arousal SHAP Sum"]].to_string(index=False))
    
    # Save Top 15 per task
    top15_combined = pd.DataFrame({
        "Rank": range(1, 16),
        "Genre_Feature": top15_g.iloc[::-1]["feature"].values,
        "Genre_Group": top15_g.iloc[::-1]["group"].values,
        "Valence_Feature": top15_v.iloc[::-1]["feature"].values,
        "Valence_Group": top15_v.iloc[::-1]["group"].values,
        "Arousal_Feature": top15_a.iloc[::-1]["feature"].values,
        "Arousal_Group": top15_a.iloc[::-1]["group"].values
    })
    top15_combined.to_csv(os.path.join(C.OUT, "part_d_top15_features.csv"), index=False)


if __name__ == "__main__":
    run_shap_analysis()
