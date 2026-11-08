"""
experiments/rev2_p4_baselines/05_run_default_lightgbm.py
Evaluates default-hyperparameter LightGBM on the exact same outer folds
for Full-113 and Lite-71 on GTZAN and PMEmo, comparing default vs tuned.
Default settings:
- Classifier: n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=20
- Regressors: n_estimators=100, learning_rate=0.1, num_leaves=31, min_child_samples=20
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common

def run_default_gtzan(clean_df: pd.DataFrame, splits: list, feat_sets: dict):
    records = []
    for fold_info in splits:
        repeat = fold_info["repeat"]
        fold_idx = fold_info["fold"]
        train_df = clean_df.loc[fold_info["train_tracks"]]
        test_df = clean_df.loc[fold_info["test_tracks"]]
        y_train = train_df["label"].values
        y_test = test_df["label"].values
        
        for feat_name, feats in feat_sets.items():
            X_tr = train_df[feats].values
            X_te = test_df[feats].values
            
            clf = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            clf.fit(X_tr, y_train)
            preds = clf.predict(X_te)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            
            records.append({
                "dataset": "GTZAN",
                "task": "genre",
                "repeat": repeat,
                "fold": fold_idx,
                "feature_set": feat_name,
                "default_acc": float(acc),
                "default_f1": float(f1),
            })
    return pd.DataFrame(records)

def run_default_pmemo(pmemo_df: pd.DataFrame, splits: list, feat_sets: dict):
    records = []
    targets = [("valence", "Valence(mean)"), ("arousal", "Arousal(mean)")]
    
    for fold_info in splits:
        repeat = fold_info["repeat"]
        fold_idx = fold_info["fold"]
        train_df = pmemo_df.loc[fold_info["train_tracks"]]
        test_df = pmemo_df.loc[fold_info["test_tracks"]]
        
        for task_name, target_col in targets:
            y_train = train_df[target_col].values
            y_test = test_df[target_col].values
            
            for feat_name, feats in feat_sets.items():
                X_tr = train_df[feats].values
                X_te = test_df[feats].values
                
                reg = LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31,
                    min_child_samples=20,
                    random_state=42,
                    verbose=-1,
                    n_jobs=1,
                )
                reg.fit(X_tr, y_train)
                preds = reg.predict(X_te)
                
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                
                records.append({
                    "dataset": "PMEmo",
                    "task": task_name,
                    "repeat": repeat,
                    "fold": fold_idx,
                    "feature_set": feat_name,
                    "default_r2": float(r2),
                    "default_mae": float(mae),
                })
    return pd.DataFrame(records)

def main():
    out_dir = common.BASE_DIR
    
    # 1. Load data
    gtzan_csv = os.path.join(common.PROJECT_ROOT, "data", "gtzan_features.csv")
    rem_csv = os.path.join(common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "removed_tracks.csv")
    splits_json = os.path.join(common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "splits.json")
    pmemo_csv = os.path.join(common.PROJECT_ROOT, "data", "pmemo_ready.csv")
    
    df_g = pd.read_csv(gtzan_csv).set_index("filename")
    rem_tracks = set(pd.read_csv(rem_csv)["filename"])
    df_g_clean = df_g.loc[~df_g.index.isin(rem_tracks)]
    
    df_p = pd.read_csv(pmemo_csv)
    df_p["musicId_str"] = df_p["musicId"].astype(str)
    df_p.set_index("musicId_str", inplace=True)
    
    with open(splits_json, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
        
    g_splits = [f for f in splits_data["gtzan"]["fault_removed_artist_grouped"] if f["repeat"] == 0]
    p_splits = [f for f in splits_data["pmemo"]["artist_grouped_5x5"] if f["repeat"] in (0, 1)]
    
    feat_sets = {
        "Full-113": common.FULL_113_FEATURES,
        "Lite-71": common.LITE_71_FEATURES,
    }
    
    print("Running default LightGBM on GTZAN outer folds...")
    df_g_def = run_default_gtzan(df_g_clean, g_splits, feat_sets)
    
    print("Running default LightGBM on PMEmo outer folds...")
    df_p_def = run_default_pmemo(df_p, p_splits, feat_sets)
    
    # 2. Merge with tuned LightGBM from existing baseline folds
    df_g_tuned = pd.read_csv(os.path.join(out_dir, "gtzan_baseline_folds.csv"))
    df_g_tuned_lgbm = df_g_tuned[df_g_tuned["model"] == "LightGBM"]
    
    df_p_tuned = pd.read_csv(os.path.join(out_dir, "pmemo_baseline_folds.csv"))
    df_p_tuned_lgbm = df_p_tuned[df_p_tuned["model"] == "LightGBM"]
    
    m_g = pd.merge(df_g_def, df_g_tuned_lgbm, on=["repeat", "fold", "feature_set"])
    m_p = pd.merge(df_p_def, df_p_tuned_lgbm, on=["task", "repeat", "fold", "feature_set"])
    
    comparison_records = []
    
    # Compare GTZAN
    for feat_name in ["Full-113", "Lite-71"]:
        sub = m_g[m_g["feature_set"] == feat_name]
        diff_acc = sub["accuracy"].values - sub["default_acc"].values
        diff_f1 = sub["macro_f1"].values - sub["default_f1"].values
        
        stat_acc = common.nadeau_bengio_test(diff_acc, test_train_ratio=0.25)
        stat_f1 = common.nadeau_bengio_test(diff_f1, test_train_ratio=0.25)
        
        comparison_records.append({
            "task": "Genre",
            "feature_set": feat_name,
            "metric": "Accuracy",
            "default_mean": float(np.mean(sub["default_acc"])),
            "default_std": float(np.std(sub["default_acc"], ddof=1)),
            "tuned_mean": float(np.mean(sub["accuracy"])),
            "tuned_std": float(np.std(sub["accuracy"], ddof=1)),
            "mean_diff_tuned_minus_default": stat_acc["mean_diff"],
            "std_diff": stat_acc["std_diff"],
            "se_corrected": stat_acc["se_corrected"],
            "t_stat": stat_acc["t_stat"],
            "p_value": stat_acc["p_value"],
            "ci_lower_95": stat_acc["ci_lower"],
            "ci_upper_95": stat_acc["ci_upper"],
            "stat_sig": "Yes (p < 0.05)" if stat_acc["p_value"] < 0.05 else "No (p >= 0.05)",
        })
        comparison_records.append({
            "task": "Genre",
            "feature_set": feat_name,
            "metric": "Macro-F1",
            "default_mean": float(np.mean(sub["default_f1"])),
            "default_std": float(np.std(sub["default_f1"], ddof=1)),
            "tuned_mean": float(np.mean(sub["macro_f1"])),
            "tuned_std": float(np.std(sub["macro_f1"], ddof=1)),
            "mean_diff_tuned_minus_default": stat_f1["mean_diff"],
            "std_diff": stat_f1["std_diff"],
            "se_corrected": stat_f1["se_corrected"],
            "t_stat": stat_f1["t_stat"],
            "p_value": stat_f1["p_value"],
            "ci_lower_95": stat_f1["ci_lower"],
            "ci_upper_95": stat_f1["ci_upper"],
            "stat_sig": "Yes (p < 0.05)" if stat_f1["p_value"] < 0.05 else "No (p >= 0.05)",
        })
        
    # Compare PMEmo
    for task_name in ["valence", "arousal"]:
        for feat_name in ["Full-113", "Lite-71"]:
            sub = m_p[(m_p["task"] == task_name) & (m_p["feature_set"] == feat_name)]
            diff_r2 = sub["r2"].values - sub["default_r2"].values
            diff_mae = sub["mae"].values - sub["default_mae"].values
            
            stat_r2 = common.nadeau_bengio_test(diff_r2, test_train_ratio=0.25)
            stat_mae = common.nadeau_bengio_test(diff_mae, test_train_ratio=0.25)
            
            comparison_records.append({
                "task": task_name.capitalize(),
                "feature_set": feat_name,
                "metric": "R²",
                "default_mean": float(np.mean(sub["default_r2"])),
                "default_std": float(np.std(sub["default_r2"], ddof=1)),
                "tuned_mean": float(np.mean(sub["r2"])),
                "tuned_std": float(np.std(sub["r2"], ddof=1)),
                "mean_diff_tuned_minus_default": stat_r2["mean_diff"],
                "std_diff": stat_r2["std_diff"],
                "se_corrected": stat_r2["se_corrected"],
                "t_stat": stat_r2["t_stat"],
                "p_value": stat_r2["p_value"],
                "ci_lower_95": stat_r2["ci_lower"],
                "ci_upper_95": stat_r2["ci_upper"],
                "stat_sig": "Yes (p < 0.05)" if stat_r2["p_value"] < 0.05 else "No (p >= 0.05)",
            })
            comparison_records.append({
                "task": task_name.capitalize(),
                "feature_set": feat_name,
                "metric": "MAE",
                "default_mean": float(np.mean(sub["default_mae"])),
                "default_std": float(np.std(sub["default_mae"], ddof=1)),
                "tuned_mean": float(np.mean(sub["mae"])),
                "tuned_std": float(np.std(sub["mae"], ddof=1)),
                "mean_diff_tuned_minus_default": stat_mae["mean_diff"],
                "std_diff": stat_mae["std_diff"],
                "se_corrected": stat_mae["se_corrected"],
                "t_stat": stat_mae["t_stat"],
                "p_value": stat_mae["p_value"],
                "ci_lower_95": stat_mae["ci_lower"],
                "ci_upper_95": stat_mae["ci_upper"],
                "stat_sig": "Yes (p < 0.05)" if stat_mae["p_value"] < 0.05 else "No (p >= 0.05)",
            })
            
    df_comp = pd.DataFrame(comparison_records)
    out_csv = os.path.join(out_dir, "default_vs_tuned_lightgbm.csv")
    df_comp.to_csv(out_csv, index=False)
    print(f"Default vs Tuned comparison saved to {out_csv}")
    print(df_comp[["task", "feature_set", "metric", "default_mean", "tuned_mean", "mean_diff_tuned_minus_default", "p_value", "stat_sig"]].to_string())

if __name__ == "__main__":
    main()

