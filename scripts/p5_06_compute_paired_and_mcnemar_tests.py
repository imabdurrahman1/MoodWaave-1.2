"""
experiments/rev2_p5_deep/06_compute_paired_and_mcnemar_tests.py
Computes:
1. Paired differences and Nadeau-Bengio corrected t-tests on outer folds:
   - GTZAN 5 outer artist-disjoint folds (df = 4): VGGish vs MOODWAAVE-Full, VGGish vs MOODWAAVE-Lite
   - PMEmo 10 outer artist-grouped folds (df = 9): VGGish vs MOODWAAVE-Full, VGGish vs MOODWAAVE-Lite
2. McNemar's exact test on the fixed Kereliuk-Sturm split (290 test tracks):
   - VGGish vs MOODWAAVE-Full
   - VGGish vs MOODWAAVE-Lite
   - SmallMelCNN (seed-averaged logits across seeds 42, 43, 44) vs MOODWAAVE-Lite
   - SmallMelCNN (seed-averaged logits) vs MOODWAAVE-Full
3. Reconciles P1 (train+val 639 tracks, 5 seeds) vs P5 (train-only 442 tracks, single seed) fixed-split numbers.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common
import small_cnn

def nadeau_bengio_test(diffs: np.ndarray, test_train_ratio: float = 0.25):
    """
    Computes Nadeau-Bengio corrected resampled t-test for paired differences across CV folds.
    df = n_folds - 1.
    """
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    std_d = float(np.sqrt(var_d))
    
    # Nadeau & Bengio (2003): sigma^2_corr = (1 / n + n_test / n_train) * S^2
    var_corr = (1.0 / n + test_train_ratio) * var_d
    se_corr = float(np.sqrt(var_corr))
    df = n - 1
    
    if se_corr == 0:
        t_stat = 0.0
        p_val = 1.0
        ci_lower = mean_d
        ci_upper = mean_d
    else:
        t_stat = mean_d / se_corr
        p_val = float(2.0 * stats.t.sf(np.abs(t_stat), df=df))
        t_crit = float(stats.t.ppf(0.975, df=df))
        ci_lower = mean_d - t_crit * se_corr
        ci_upper = mean_d + t_crit * se_corr
        
    return {
        "n_folds": n,
        "df": df,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "se_corrected": se_corr,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
    }

def run_mcnemar_exact(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray):
    """
    Computes McNemar's exact test (two-sided binomial test on discordant pairs).
    Returns n00, n01, n10, n11, and exact p-value.
    """
    corr_a = (preds_a == y_true)
    corr_b = (preds_b == y_true)
    
    n11 = int(np.sum(corr_a & corr_b))
    n10 = int(np.sum(corr_a & (~corr_b)))  # A correct, B incorrect
    n01 = int(np.sum((~corr_a) & corr_b))  # A incorrect, B correct
    n00 = int(np.sum((~corr_a) & (~corr_b)))
    
    n_discordant = n01 + n10
    if n_discordant == 0:
        p_val = 1.0
    else:
        res = stats.binomtest(min(n01, n10), n_discordant, p=0.5)
        p_val = float(res.pvalue)
        
    return {
        "n_total": len(y_true),
        "n11_both_correct": n11,
        "n10_a_correct_b_wrong": n10,
        "n01_a_wrong_b_correct": n01,
        "n00_both_wrong": n00,
        "n_discordant": n_discordant,
        "p_value": p_val,
    }

def main():
    out_dir = common.BASE_DIR
    print("=== Computing Paired Nadeau-Bengio Corrected Tests on Outer Folds ===")
    
    # 1. GTZAN Outer Folds (5 folds, repeat 0)
    vgg_g_csv = os.path.join(out_dir, "vggish_gtzan_folds.csv")
    df_vgg_g = pd.read_csv(vgg_g_csv).sort_values("fold").reset_index(drop=True)
    
    # P4 tuned LightGBM GTZAN folds
    p4_g_csv = os.path.join(common.PROJECT_ROOT, "experiments", "rev2_p4_baselines", "gtzan_baseline_folds.csv")
    df_p4_g = pd.read_csv(p4_g_csv)
    sub_lgbm_g = df_p4_g[(df_p4_g["model"] == "LightGBM") & (df_p4_g["repeat"] == 0)]
    
    full_g_folds = sub_lgbm_g[sub_lgbm_g["feature_set"] == "Full-113"].sort_values("fold").reset_index(drop=True)
    lite_g_folds = sub_lgbm_g[sub_lgbm_g["feature_set"] == "Lite-71"].sort_values("fold").reset_index(drop=True)
    
    # Also default LightGBM GTZAN folds from P1
    p1_g_csv = os.path.join(common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "part_a_gtzan_folds.csv")
    df_p1_g = pd.read_csv(p1_g_csv)
    sub_p1_g = df_p1_g[(df_p1_g["protocol"] == "fault_removed_artist_grouped") & (df_p1_g["repeat"] == 0)].sort_values("fold").reset_index(drop=True)
    
    paired_records = []
    
    # GTZAN VGGish vs Tuned Full
    diff_acc_full_tuned = df_vgg_g["accuracy"].values - full_g_folds["accuracy"].values
    t_acc_full_tuned = nadeau_bengio_test(diff_acc_full_tuned, test_train_ratio=0.25)
    paired_records.append({
        "dataset": "GTZAN",
        "task": "Genre",
        "metric": "Accuracy",
        "comparison": "VGGish minus MOODWAAVE-Full (Tuned)",
        "vggish_mean": float(df_vgg_g["accuracy"].mean()),
        "baseline_mean": float(full_g_folds["accuracy"].mean()),
        **t_acc_full_tuned,
    })
    
    # GTZAN VGGish vs Tuned Lite
    diff_acc_lite_tuned = df_vgg_g["accuracy"].values - lite_g_folds["accuracy"].values
    t_acc_lite_tuned = nadeau_bengio_test(diff_acc_lite_tuned, test_train_ratio=0.25)
    paired_records.append({
        "dataset": "GTZAN",
        "task": "Genre",
        "metric": "Accuracy",
        "comparison": "VGGish minus MOODWAAVE-Lite (Tuned)",
        "vggish_mean": float(df_vgg_g["accuracy"].mean()),
        "baseline_mean": float(lite_g_folds["accuracy"].mean()),
        **t_acc_lite_tuned,
    })
    
    # GTZAN VGGish vs Default Full
    diff_acc_full_def = df_vgg_g["accuracy"].values - sub_p1_g["acc_full"].values
    t_acc_full_def = nadeau_bengio_test(diff_acc_full_def, test_train_ratio=0.25)
    paired_records.append({
        "dataset": "GTZAN",
        "task": "Genre",
        "metric": "Accuracy",
        "comparison": "VGGish minus MOODWAAVE-Full (Default)",
        "vggish_mean": float(df_vgg_g["accuracy"].mean()),
        "baseline_mean": float(sub_p1_g["acc_full"].mean()),
        **t_acc_full_def,
    })
    
    # GTZAN VGGish vs Default Lite
    diff_acc_lite_def = df_vgg_g["accuracy"].values - sub_p1_g["acc_lite"].values
    t_acc_lite_def = nadeau_bengio_test(diff_acc_lite_def, test_train_ratio=0.25)
    paired_records.append({
        "dataset": "GTZAN",
        "task": "Genre",
        "metric": "Accuracy",
        "comparison": "VGGish minus MOODWAAVE-Lite (Default)",
        "vggish_mean": float(df_vgg_g["accuracy"].mean()),
        "baseline_mean": float(sub_p1_g["acc_lite"].mean()),
        **t_acc_lite_def,
    })

    # 2. PMEmo Outer Folds (10 folds: repeats 0, 1)
    vgg_p_csv = os.path.join(out_dir, "vggish_pmemo_folds.csv")
    df_vgg_p = pd.read_csv(vgg_p_csv)
    
    p4_p_csv = os.path.join(common.PROJECT_ROOT, "experiments", "rev2_p4_baselines", "pmemo_baseline_folds.csv")
    df_p4_p = pd.read_csv(p4_p_csv)
    sub_lgbm_p = df_p4_p[(df_p4_p["model"] == "LightGBM") & (df_p4_p["repeat"].isin([0, 1]))]
    
    for task_name, target_col in [("Valence", "Valence(mean)"), ("Arousal", "Arousal(mean)")]:
        sub_vgg = df_vgg_p[df_vgg_p["task"] == task_name.lower()].sort_values(["repeat", "fold"]).reset_index(drop=True)
        sub_full = sub_lgbm_p[(sub_lgbm_p["target"] == target_col) & (sub_lgbm_p["feature_set"] == "Full-113")].sort_values(["repeat", "fold"]).reset_index(drop=True)
        sub_lite = sub_lgbm_p[(sub_lgbm_p["target"] == target_col) & (sub_lgbm_p["feature_set"] == "Lite-71")].sort_values(["repeat", "fold"]).reset_index(drop=True)
        
        diff_full = sub_vgg["r2"].values - sub_full["r2"].values
        diff_lite = sub_vgg["r2"].values - sub_lite["r2"].values
        
        t_full = nadeau_bengio_test(diff_full, test_train_ratio=0.25)
        t_lite = nadeau_bengio_test(diff_lite, test_train_ratio=0.25)
        
        paired_records.append({
            "dataset": "PMEmo",
            "task": task_name,
            "metric": "R²",
            "comparison": f"VGGish minus MOODWAAVE-Full (Tuned)",
            "vggish_mean": float(sub_vgg["r2"].mean()),
            "baseline_mean": float(sub_full["r2"].mean()),
            **t_full,
        })
        paired_records.append({
            "dataset": "PMEmo",
            "task": task_name,
            "metric": "R²",
            "comparison": f"VGGish minus MOODWAAVE-Lite (Tuned)",
            "vggish_mean": float(sub_vgg["r2"].mean()),
            "baseline_mean": float(sub_lite["r2"].mean()),
            **t_lite,
        })
        
    df_paired = pd.DataFrame(paired_records)
    paired_csv = os.path.join(out_dir, "p5_fix_paired_tests.csv")
    df_paired.to_csv(paired_csv, index=False)
    print(f"Saved paired outer folds tests to {paired_csv}")
    print(df_paired[["task", "comparison", "mean_diff", "df", "t_stat", "p_value", "ci_lower_95", "ci_upper_95"]].to_string())

    print("\n=== Computing Fixed Split Test Predictions and McNemar's Tests ===")
    with open(common.SPLITS_JSON, "r", encoding="utf-8") as f:
        ks_split = json.load(f)["gtzan"]["kereliuk_sturm_fixed"]
        
    tr_ids = ks_split["train_tracks"]
    te_ids = ks_split["test_tracks"]
    
    df_gtzan = pd.read_csv(common.GTZAN_FEATURES_CSV).set_index("filename")
    y_test_labels = df_gtzan.loc[te_ids, "label"].values
    
    # 1. MOODWAAVE-Full & Lite predictions on fixed split (train only: 442 tracks, seed 42)
    with open(os.path.join(common.PROJECT_ROOT, "experiments", "idea2_cost_pareto", "feature_groups.json")) as f:
        fg = json.load(f)["groups"]
    full_cols = [c for grp in ["MFCC", "Chroma", "Tonnetz", "Spectral", "SpecContrast", "Temporal"] for c in fg[grp]]
    lite_cols = [c for grp in ["MFCC", "SpecContrast", "Temporal"] for c in fg[grp]]
    
    train_df = df_gtzan.loc[tr_ids]
    test_df = df_gtzan.loc[te_ids]
    y_train_labels = train_df["label"].values
    
    clf_full = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=20, random_state=42, verbose=-1, n_jobs=1)
    clf_full.fit(train_df[full_cols].values, y_train_labels)
    preds_full = clf_full.predict(test_df[full_cols].values)
    
    clf_lite = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=20, random_state=42, verbose=-1, n_jobs=1)
    clf_lite.fit(train_df[lite_cols].values, y_train_labels)
    preds_lite = clf_lite.predict(test_df[lite_cols].values)
    
    # 2. VGGish fixed split predictions
    gtzan_emb_csv = os.path.join(out_dir, "vggish_gtzan_embeddings.csv")
    df_embs = pd.read_csv(gtzan_emb_csv).set_index("track_id")
    vgg_cols = [c for c in df_embs.columns if c.startswith("vggish_")]
    X_tr_vgg = df_embs.loc[tr_ids, vgg_cols].values
    X_te_vgg = df_embs.loc[te_ids, vgg_cols].values
    
    pipe_vgg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    grid_vgg = {"clf__C": [0.003, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0]}
    search_vgg = RandomizedSearchCV(pipe_vgg, grid_vgg, n_iter=8, cv=3, scoring="accuracy", random_state=42, n_jobs=1)
    search_vgg.fit(X_tr_vgg, y_train_labels)
    preds_vgg = search_vgg.best_estimator_.predict(X_te_vgg)
    
    # 3. SmallMelCNN fixed split predictions with SEED-AVERAGED LOGITS
    labels_map = df_gtzan["label"].to_dict()
    test_ds = small_cnn.GTZANMelDataset(te_ids, labels_map, cache_dir=os.path.join(out_dir, "mel_cache"), is_training=False)
    
    seed_logits_list = []
    seeds = [42, 43, 44]
    device = torch.device("cpu")
    
    for s in seeds:
        ckpt_path = os.path.join(out_dir, f"small_cnn_seed_{s}.pt")
        model = small_cnn.SmallMelCNN(num_classes=10, dropout=0.2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        
        track_logits = []
        with torch.no_grad():
            for i in range(len(test_ds)):
                crops_tensor, target = test_ds[i]
                crops_tensor = crops_tensor.to(device)
                logits = model(crops_tensor)  # (n_crops, 10)
                avg_logit = logits.mean(dim=0).cpu().numpy()  # (10,)
                track_logits.append(avg_logit)
        seed_logits_list.append(np.array(track_logits))  # shape: (290, 10)
        
    # Average logits across the 3 seeds
    avg_logits_across_seeds = np.mean(seed_logits_list, axis=0)  # (290, 10)
    pred_indices = np.argmax(avg_logits_across_seeds, axis=1)
    preds_cnn_ensemble = np.array([common.IDX_TO_GENRE[idx] for idx in pred_indices])
    
    acc_full = accuracy_score(y_test_labels, preds_full)
    acc_lite = accuracy_score(y_test_labels, preds_lite)
    acc_vgg = accuracy_score(y_test_labels, preds_vgg)
    acc_cnn = accuracy_score(y_test_labels, preds_cnn_ensemble)
    
    print(f"Fixed Split Accuracies (290 test tracks):")
    print(f"  MOODWAAVE-Full (Train only): {acc_full*100:.2f}%")
    print(f"  MOODWAAVE-Lite (Train only): {acc_lite*100:.2f}%")
    print(f"  VGGish + Logistic Regression: {acc_vgg*100:.2f}%")
    print(f"  SmallMelCNN (Seed-Averaged Logits): {acc_cnn*100:.2f}%")
    
    # McNemar's tests
    mc_tests = []
    
    # VGGish vs MOODWAAVE-Full
    mc_vgg_full = run_mcnemar_exact(y_test_labels, preds_vgg, preds_full)
    mc_tests.append({
        "comparison": "VGGish vs MOODWAAVE-Full",
        "model_a": "VGGish + Logistic Regression",
        "model_b": "MOODWAAVE-Full",
        "acc_a": acc_vgg,
        "acc_b": acc_full,
        **mc_vgg_full,
    })
    
    # VGGish vs MOODWAAVE-Lite
    mc_vgg_lite = run_mcnemar_exact(y_test_labels, preds_vgg, preds_lite)
    mc_tests.append({
        "comparison": "VGGish vs MOODWAAVE-Lite",
        "model_a": "VGGish + Logistic Regression",
        "model_b": "MOODWAAVE-Lite",
        "acc_a": acc_vgg,
        "acc_b": acc_lite,
        **mc_vgg_lite,
    })
    
    # CNN vs MOODWAAVE-Lite
    mc_cnn_lite = run_mcnemar_exact(y_test_labels, preds_cnn_ensemble, preds_lite)
    mc_tests.append({
        "comparison": "SmallMelCNN (Seed-Averaged Logits) vs MOODWAAVE-Lite",
        "model_a": "SmallMelCNN (Seed-Averaged)",
        "model_b": "MOODWAAVE-Lite",
        "acc_a": acc_cnn,
        "acc_b": acc_lite,
        **mc_cnn_lite,
    })
    
    # CNN vs MOODWAAVE-Full
    mc_cnn_full = run_mcnemar_exact(y_test_labels, preds_cnn_ensemble, preds_full)
    mc_tests.append({
        "comparison": "SmallMelCNN (Seed-Averaged Logits) vs MOODWAAVE-Full",
        "model_a": "SmallMelCNN (Seed-Averaged)",
        "model_b": "MOODWAAVE-Full",
        "acc_a": acc_cnn,
        "acc_b": acc_full,
        **mc_cnn_full,
    })
    
    df_mc = pd.DataFrame(mc_tests)
    mc_csv = os.path.join(out_dir, "p5_fix_mcnemar_tests.csv")
    df_mc.to_csv(mc_csv, index=False)
    print(f"\nSaved McNemar tests to {mc_csv}")
    print(df_mc[["comparison", "acc_a", "acc_b", "n10_a_correct_b_wrong", "n01_a_wrong_b_correct", "p_value"]].to_string())

if __name__ == "__main__":
    main()

