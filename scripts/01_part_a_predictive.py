"""Script 01: Part A - Predictive Performance Benchmark.
Evaluates Genre (5x3 repeated stratified K-fold) and Mood (5x5 repeated K-fold).
Generates confusion matrix plot for Lite-71, per-genre F1 table, and paired statistical tests.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, confusion_matrix
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


def run_genre_benchmark():
    print("\n" + "="*70)
    print("PART A.1: GENRE CLASSIFICATION (GTZAN, 999 tracks, 5x3 Repeated Stratified K-Fold)")
    print("="*70)
    df_gtzan, X_full, X_lite, y, classes = C.load_gtzan_data()
    n_tracks = len(df_gtzan)
    print(f"Loaded GTZAN: {n_tracks} tracks, 10 genres: {classes}")
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    genre_models = {
        "Dummy (Prior)": lambda: DummyClassifier(strategy="prior"),
        "RandomForest (100 trees)": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        "LightGBM Full-113": lambda: LGBMClassifier(**C.LGBM_GENRE_PARAMS, n_jobs=1),
        "LightGBM Lite-71": lambda: LGBMClassifier(**C.LGBM_GENRE_PARAMS, n_jobs=1)
    }
    
    fold_records = []
    
    # Store out-of-fold predictions for Lite-71 to compute confusion matrix & per-genre F1
    oof_y_true = []
    oof_y_pred_lite = []
    oof_y_pred_full = []
    
    fold_idx = 0
    for train_idx, test_idx in rskf.split(X_full, y):
        n_tr, n_te = len(train_idx), len(test_idx)
        repeat = fold_idx // 5
        split = fold_idx % 5
        
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # 1. Dummy
        m_dum = genre_models["Dummy (Prior)"]()
        m_dum.fit(X_full.iloc[train_idx], y_tr)
        p_dum = m_dum.predict(X_full.iloc[test_idx])
        acc_dum = accuracy_score(y_te, p_dum)
        f1_dum = f1_score(y_te, p_dum, average="macro", zero_division=0)
        
        # 2. RF (Full 113)
        m_rf = genre_models["RandomForest (100 trees)"]()
        m_rf.fit(X_full.iloc[train_idx], y_tr)
        p_rf = m_rf.predict(X_full.iloc[test_idx])
        acc_rf = accuracy_score(y_te, p_rf)
        f1_rf = f1_score(y_te, p_rf, average="macro", zero_division=0)
        
        # 3. LightGBM Full-113
        m_full = genre_models["LightGBM Full-113"]()
        m_full.fit(X_full.iloc[train_idx], y_tr)
        p_full = m_full.predict(X_full.iloc[test_idx])
        acc_full = accuracy_score(y_te, p_full)
        f1_full = f1_score(y_te, p_full, average="macro", zero_division=0)
        
        # 4. LightGBM Lite-71
        m_lite = genre_models["LightGBM Lite-71"]()
        m_lite.fit(X_lite.iloc[train_idx], y_tr)
        p_lite = m_lite.predict(X_lite.iloc[test_idx])
        acc_lite = accuracy_score(y_te, p_lite)
        f1_lite = f1_score(y_te, p_lite, average="macro", zero_division=0)
        
        oof_y_true.extend(y_te)
        oof_y_pred_lite.extend(p_lite)
        oof_y_pred_full.extend(p_full)
        
        fold_records.append({
            "fold": fold_idx,
            "repeat": repeat,
            "split": split,
            "n_train": n_tr,
            "n_test": n_te,
            "acc_dummy": acc_dum, "f1_dummy": f1_dum,
            "acc_rf": acc_rf, "f1_rf": f1_rf,
            "acc_full": acc_full, "f1_full": f1_full,
            "acc_lite": acc_lite, "f1_lite": f1_lite,
            "diff_acc": acc_lite - acc_full,
            "diff_f1": f1_lite - f1_full
        })
        fold_idx += 1
        
    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv(os.path.join(C.OUT, "part_a_genre_folds.csv"), index=False)
    
    # Statistical tests for Genre Lite vs Full
    stat_acc = C.corrected_resampled_ttest(df_folds["diff_acc"], n_train_ratio=0.80, n_test_ratio=0.20)
    stat_f1 = C.corrected_resampled_ttest(df_folds["diff_f1"], n_train_ratio=0.80, n_test_ratio=0.20)
    
    # Summary Table for Genre
    genre_summary = [
        {"Model": "Dummy (Prior)", "Features": 0,
         "Accuracy": f"{df_folds['acc_dummy'].mean()*100:.2f}% ± {df_folds['acc_dummy'].std()*100:.2f}%",
         "Macro-F1": f"{df_folds['f1_dummy'].mean():.4f} ± {df_folds['f1_dummy'].std():.4f}"},
        {"Model": "RandomForest (100 trees)", "Features": 113,
         "Accuracy": f"{df_folds['acc_rf'].mean()*100:.2f}% ± {df_folds['acc_rf'].std()*100:.2f}%",
         "Macro-F1": f"{df_folds['f1_rf'].mean():.4f} ± {df_folds['f1_rf'].std():.4f}"},
        {"Model": "LightGBM Full-113", "Features": 113,
         "Accuracy": f"{df_folds['acc_full'].mean()*100:.2f}% ± {df_folds['acc_full'].std()*100:.2f}%",
         "Macro-F1": f"{df_folds['f1_full'].mean():.4f} ± {df_folds['f1_full'].std():.4f}"},
        {"Model": "LightGBM Lite-71 (MOODWAAVE-Lite)", "Features": 71,
         "Accuracy": f"{df_folds['acc_lite'].mean()*100:.2f}% ± {df_folds['acc_lite'].std()*100:.2f}%",
         "Macro-F1": f"{df_folds['f1_lite'].mean():.4f} ± {df_folds['f1_lite'].std():.4f}"}
    ]
    df_genre_sum = pd.DataFrame(genre_summary)
    df_genre_sum.to_csv(os.path.join(C.OUT, "part_a_genre_summary.csv"), index=False)
    
    print("\nGenre Summary Performance:")
    print(df_genre_sum.to_string(index=False))
    print(f"\nPaired Test Lite-71 vs Full-113 (N = 15 folds):")
    print(f"  Delta Accuracy: {stat_acc['mean_diff']*100:+.2f}% (t = {stat_acc['t_stat']:.3f}, p = {stat_acc['p_val']:.4f}, 95% CI: [{stat_acc['ci_lower']*100:+.2f}%, {stat_acc['ci_upper']*100:+.2f}%])")
    print(f"  Delta Macro-F1: {stat_f1['mean_diff']:+.4f} (t = {stat_f1['t_stat']:.3f}, p = {stat_f1['p_val']:.4f}, 95% CI: [{stat_f1['ci_lower']:+.4f}, {stat_f1['ci_upper']:+.4f}])")
    
    # Per-genre F1 scores
    f1_per_genre_lite = f1_score(oof_y_true, oof_y_pred_lite, average=None)
    f1_per_genre_full = f1_score(oof_y_true, oof_y_pred_full, average=None)
    
    df_per_genre = pd.DataFrame({
        "Genre": classes,
        "Full_113_F1": f1_per_genre_full,
        "Lite_71_F1": f1_per_genre_lite,
        "Delta_F1": f1_per_genre_lite - f1_per_genre_full
    })
    df_per_genre.to_csv(os.path.join(C.OUT, "part_a_genre_per_class_f1.csv"), index=False)
    print("\nPer-Genre F1 Scores (Pooled Out-of-Fold):")
    print(df_per_genre.to_string(index=False))
    
    # Plot Confusion Matrix for Lite-71
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
    plt.savefig(os.path.join(C.OUT, "confusion_matrix_genre_lite.png"))
    plt.close()
    print("Saved confusion_matrix_genre_lite.png")
    
    return stat_acc, stat_f1


def run_mood_benchmark():
    print("\n" + "="*70)
    print("PART A.2: MOOD REGRESSION (PMEmo, 767 tracks, 5x5 Repeated K-Fold)")
    print("="*70)
    df_pmemo, X_full, X_lite, y_val, y_aro = C.load_pmemo_data()
    n_tracks = len(df_pmemo)
    print(f"Loaded PMEmo: {n_tracks} tracks, Valence range: [{y_val.min():.3f}, {y_val.max():.3f}], Arousal range: [{y_aro.min():.3f}, {y_aro.max():.3f}]")
    
    # 5 repeats of 5-fold CV with seeds 42, 43, 44, 45, 46
    fold_records = []
    
    fold_idx = 0
    for repeat in range(5):
        kf = KFold(n_splits=5, shuffle=True, random_state=42 + repeat)
        for split, (train_idx, test_idx) in enumerate(kf.split(X_full)):
            n_tr, n_te = len(train_idx), len(test_idx)
            
            y_v_tr, y_v_te = y_val[train_idx], y_val[test_idx]
            y_a_tr, y_a_te = y_aro[train_idx], y_aro[test_idx]
            
            # 1. Dummy (Mean)
            d_v = DummyRegressor(strategy="mean").fit(X_full.iloc[train_idx], y_v_tr)
            d_a = DummyRegressor(strategy="mean").fit(X_full.iloc[train_idx], y_a_tr)
            p_v_dum = d_v.predict(X_full.iloc[test_idx])
            p_a_dum = d_a.predict(X_full.iloc[test_idx])
            r2_v_dum = r2_score(y_v_te, p_v_dum)
            r2_a_dum = r2_score(y_a_te, p_a_dum)
            mae_v_dum = mean_absolute_error(y_v_te, p_v_dum)
            mae_a_dum = mean_absolute_error(y_a_te, p_a_dum)
            
            # 2. Ridge (alpha=1.0)
            rg_v = Ridge(alpha=1.0).fit(X_full.iloc[train_idx], y_v_tr)
            rg_a = Ridge(alpha=1.0).fit(X_full.iloc[train_idx], y_a_tr)
            p_v_rg = rg_v.predict(X_full.iloc[test_idx])
            p_a_rg = rg_a.predict(X_full.iloc[test_idx])
            r2_v_rg = r2_score(y_v_te, p_v_rg)
            r2_a_rg = r2_score(y_a_te, p_a_rg)
            mae_v_rg = mean_absolute_error(y_v_te, p_v_rg)
            mae_a_rg = mean_absolute_error(y_a_te, p_a_rg)
            
            # 3. RandomForest Regressor (100 trees)
            rf_v = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1).fit(X_full.iloc[train_idx], y_v_tr)
            rf_a = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1).fit(X_full.iloc[train_idx], y_a_tr)
            p_v_rf = rf_v.predict(X_full.iloc[test_idx])
            p_a_rf = rf_a.predict(X_full.iloc[test_idx])
            r2_v_rf = r2_score(y_v_te, p_v_rf)
            r2_a_rf = r2_score(y_a_te, p_a_rf)
            mae_v_rf = mean_absolute_error(y_v_te, p_v_rf)
            mae_a_rf = mean_absolute_error(y_a_te, p_a_rf)
            
            # 4. LightGBM Full-113
            lgb_v_full = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_full.iloc[train_idx], y_v_tr)
            lgb_a_full = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_full.iloc[train_idx], y_a_tr)
            p_v_full = lgb_v_full.predict(X_full.iloc[test_idx])
            p_a_full = lgb_a_full.predict(X_full.iloc[test_idx])
            r2_v_full = r2_score(y_v_te, p_v_full)
            r2_a_full = r2_score(y_a_te, p_a_full)
            mae_v_full = mean_absolute_error(y_v_te, p_v_full)
            mae_a_full = mean_absolute_error(y_a_te, p_a_full)
            
            # 5. LightGBM Lite-71
            lgb_v_lite = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_lite.iloc[train_idx], y_v_tr)
            lgb_a_lite = LGBMRegressor(**C.LGBM_MOOD_PARAMS, n_jobs=1).fit(X_lite.iloc[train_idx], y_a_tr)
            p_v_lite = lgb_v_lite.predict(X_lite.iloc[test_idx])
            p_a_lite = lgb_a_lite.predict(X_lite.iloc[test_idx])
            r2_v_lite = r2_score(y_v_te, p_v_lite)
            r2_a_lite = r2_score(y_a_te, p_a_lite)
            mae_v_lite = mean_absolute_error(y_v_te, p_v_lite)
            mae_a_lite = mean_absolute_error(y_a_te, p_a_lite)
            
            fold_records.append({
                "fold": fold_idx,
                "repeat": repeat,
                "split": split,
                "n_train": n_tr,
                "n_test": n_te,
                # Dummy
                "r2_v_dum": r2_v_dum, "r2_a_dum": r2_a_dum, "r2_mean_dum": (r2_v_dum + r2_a_dum)/2,
                "mae_v_dum": mae_v_dum, "mae_a_dum": mae_a_dum, "mae_mean_dum": (mae_v_dum + mae_a_dum)/2,
                # Ridge
                "r2_v_rg": r2_v_rg, "r2_a_rg": r2_a_rg, "r2_mean_rg": (r2_v_rg + r2_a_rg)/2,
                "mae_v_rg": mae_v_rg, "mae_a_rg": mae_a_rg, "mae_mean_rg": (mae_v_rg + mae_a_rg)/2,
                # RF
                "r2_v_rf": r2_v_rf, "r2_a_rf": r2_a_rf, "r2_mean_rf": (r2_v_rf + r2_a_rf)/2,
                "mae_v_rf": mae_v_rf, "mae_a_rf": mae_a_rf, "mae_mean_rf": (mae_v_rf + mae_a_rf)/2,
                # Full 113
                "r2_v_full": r2_v_full, "r2_a_full": r2_a_full, "r2_mean_full": (r2_v_full + r2_a_full)/2,
                "mae_v_full": mae_v_full, "mae_a_full": mae_a_full, "mae_mean_full": (mae_v_full + mae_a_full)/2,
                # Lite 71
                "r2_v_lite": r2_v_lite, "r2_a_lite": r2_a_lite, "r2_mean_lite": (r2_v_lite + r2_a_lite)/2,
                "mae_v_lite": mae_v_lite, "mae_a_lite": mae_a_lite, "mae_mean_lite": (mae_v_lite + mae_a_lite)/2,
                # Differences (Lite - Full)
                "diff_r2_mean": (r2_v_lite + r2_a_lite)/2 - (r2_v_full + r2_a_full)/2,
                "diff_r2_val": r2_v_lite - r2_v_full,
                "diff_r2_aro": r2_a_lite - r2_a_full,
                "diff_mae_mean": (mae_v_lite + mae_a_lite)/2 - (mae_v_full + mae_a_full)/2,
            })
            fold_idx += 1
            
    df_mood_folds = pd.DataFrame(fold_records)
    df_mood_folds.to_csv(os.path.join(C.OUT, "part_a_mood_folds.csv"), index=False)
    
    # Statistical tests for Mood Lite vs Full
    stat_r2_mean = C.corrected_resampled_ttest(df_mood_folds["diff_r2_mean"], n_train_ratio=0.80, n_test_ratio=0.20)
    stat_r2_val = C.corrected_resampled_ttest(df_mood_folds["diff_r2_val"], n_train_ratio=0.80, n_test_ratio=0.20)
    stat_r2_aro = C.corrected_resampled_ttest(df_mood_folds["diff_r2_aro"], n_train_ratio=0.80, n_test_ratio=0.20)
    stat_mae_mean = C.corrected_resampled_ttest(df_mood_folds["diff_mae_mean"], n_train_ratio=0.80, n_test_ratio=0.20)
    
    mood_summary = [
        {"Model": "Dummy (Mean)", "Features": 0,
         "Mean R2": f"{df_mood_folds['r2_mean_dum'].mean():.4f} ± {df_mood_folds['r2_mean_dum'].std():.4f}",
         "Valence R2": f"{df_mood_folds['r2_v_dum'].mean():.4f} ± {df_mood_folds['r2_v_dum'].std():.4f}",
         "Arousal R2": f"{df_mood_folds['r2_a_dum'].mean():.4f} ± {df_mood_folds['r2_a_dum'].std():.4f}",
         "Mean MAE": f"{df_mood_folds['mae_mean_dum'].mean():.4f} ± {df_mood_folds['mae_mean_dum'].std():.4f}",
         "Valence MAE": f"{df_mood_folds['mae_v_dum'].mean():.4f} ± {df_mood_folds['mae_v_dum'].std():.4f}",
         "Arousal MAE": f"{df_mood_folds['mae_a_dum'].mean():.4f} ± {df_mood_folds['mae_a_dum'].std():.4f}"},
        {"Model": "Ridge (alpha=1.0)", "Features": 113,
         "Mean R2": f"{df_mood_folds['r2_mean_rg'].mean():.4f} ± {df_mood_folds['r2_mean_rg'].std():.4f}",
         "Valence R2": f"{df_mood_folds['r2_v_rg'].mean():.4f} ± {df_mood_folds['r2_v_rg'].std():.4f}",
         "Arousal R2": f"{df_mood_folds['r2_a_rg'].mean():.4f} ± {df_mood_folds['r2_a_rg'].std():.4f}",
         "Mean MAE": f"{df_mood_folds['mae_mean_rg'].mean():.4f} ± {df_mood_folds['mae_mean_rg'].std():.4f}",
         "Valence MAE": f"{df_mood_folds['mae_v_rg'].mean():.4f} ± {df_mood_folds['mae_v_rg'].std():.4f}",
         "Arousal MAE": f"{df_mood_folds['mae_a_rg'].mean():.4f} ± {df_mood_folds['mae_a_rg'].std():.4f}"},
        {"Model": "RandomForest (100 trees)", "Features": 113,
         "Mean R2": f"{df_mood_folds['r2_mean_rf'].mean():.4f} ± {df_mood_folds['r2_mean_rf'].std():.4f}",
         "Valence R2": f"{df_mood_folds['r2_v_rf'].mean():.4f} ± {df_mood_folds['r2_v_rf'].std():.4f}",
         "Arousal R2": f"{df_mood_folds['r2_a_rf'].mean():.4f} ± {df_mood_folds['r2_a_rf'].std():.4f}",
         "Mean MAE": f"{df_mood_folds['mae_mean_rf'].mean():.4f} ± {df_mood_folds['mae_mean_rf'].std():.4f}",
         "Valence MAE": f"{df_mood_folds['mae_v_rf'].mean():.4f} ± {df_mood_folds['mae_v_rf'].std():.4f}",
         "Arousal MAE": f"{df_mood_folds['mae_a_rf'].mean():.4f} ± {df_mood_folds['mae_a_rf'].std():.4f}"},
        {"Model": "LightGBM Full-113", "Features": 113,
         "Mean R2": f"{df_mood_folds['r2_mean_full'].mean():.4f} ± {df_mood_folds['r2_mean_full'].std():.4f}",
         "Valence R2": f"{df_mood_folds['r2_v_full'].mean():.4f} ± {df_mood_folds['r2_v_full'].std():.4f}",
         "Arousal R2": f"{df_mood_folds['r2_a_full'].mean():.4f} ± {df_mood_folds['r2_a_full'].std():.4f}",
         "Mean MAE": f"{df_mood_folds['mae_mean_full'].mean():.4f} ± {df_mood_folds['mae_mean_full'].std():.4f}",
         "Valence MAE": f"{df_mood_folds['mae_v_full'].mean():.4f} ± {df_mood_folds['mae_v_full'].std():.4f}",
         "Arousal MAE": f"{df_mood_folds['mae_a_full'].mean():.4f} ± {df_mood_folds['mae_a_full'].std():.4f}"},
        {"Model": "LightGBM Lite-71 (MOODWAAVE-Lite)", "Features": 71,
         "Mean R2": f"{df_mood_folds['r2_mean_lite'].mean():.4f} ± {df_mood_folds['r2_mean_lite'].std():.4f}",
         "Valence R2": f"{df_mood_folds['r2_v_lite'].mean():.4f} ± {df_mood_folds['r2_v_lite'].std():.4f}",
         "Arousal R2": f"{df_mood_folds['r2_a_lite'].mean():.4f} ± {df_mood_folds['r2_a_lite'].std():.4f}",
         "Mean MAE": f"{df_mood_folds['mae_mean_lite'].mean():.4f} ± {df_mood_folds['mae_mean_lite'].std():.4f}",
         "Valence MAE": f"{df_mood_folds['mae_v_lite'].mean():.4f} ± {df_mood_folds['mae_v_lite'].std():.4f}",
         "Arousal MAE": f"{df_mood_folds['mae_a_lite'].mean():.4f} ± {df_mood_folds['mae_a_lite'].std():.4f}"}
    ]
    df_mood_sum = pd.DataFrame(mood_summary)
    df_mood_sum.to_csv(os.path.join(C.OUT, "part_a_mood_summary.csv"), index=False)
    
    print("\nMood Summary Performance:")
    print(df_mood_sum.to_string(index=False))
    print(f"\nPaired Test Lite-71 vs Full-113 (N = 25 folds):")
    print(f"  Delta Mean R2:    {stat_r2_mean['mean_diff']:+.4f} (t = {stat_r2_mean['t_stat']:.3f}, p = {stat_r2_mean['p_val']:.4f}, 95% CI: [{stat_r2_mean['ci_lower']:+.4f}, {stat_r2_mean['ci_upper']:+.4f}])")
    print(f"  Delta Valence R2: {stat_r2_val['mean_diff']:+.4f} (t = {stat_r2_val['t_stat']:.3f}, p = {stat_r2_val['p_val']:.4f}, 95% CI: [{stat_r2_val['ci_lower']:+.4f}, {stat_r2_val['ci_upper']:+.4f}])")
    print(f"  Delta Arousal R2: {stat_r2_aro['mean_diff']:+.4f} (t = {stat_r2_aro['t_stat']:.3f}, p = {stat_r2_aro['p_val']:.4f}, 95% CI: [{stat_r2_aro['ci_lower']:+.4f}, {stat_r2_aro['ci_upper']:+.4f}])")
    print(f"  Delta Mean MAE:   {stat_mae_mean['mean_diff']:+.4f} (t = {stat_mae_mean['t_stat']:.3f}, p = {stat_mae_mean['p_val']:.4f}, 95% CI: [{stat_mae_mean['ci_lower']:+.4f}, {stat_mae_mean['ci_upper']:+.4f}])")
    
    return stat_r2_mean, stat_r2_val, stat_r2_aro, stat_mae_mean


def main():
    C.save_env_log()
    stat_acc, stat_f1 = run_genre_benchmark()
    stat_r2_mean, stat_r2_val, stat_r2_aro, stat_mae_mean = run_mood_benchmark()
    
    tests_summary = {
        "genre": {
            "delta_accuracy": {
                "mean_delta": stat_acc["mean_diff"], "se_corrected": stat_acc["se_corrected"],
                "t_stat": stat_acc["t_stat"], "p_val": stat_acc["p_val"], "ci_95": stat_acc["ci_95"]
            },
            "delta_macro_f1": {
                "mean_delta": stat_f1["mean_diff"], "se_corrected": stat_f1["se_corrected"],
                "t_stat": stat_f1["t_stat"], "p_val": stat_f1["p_val"], "ci_95": stat_f1["ci_95"]
            }
        },
        "mood": {
            "delta_r2_mean": {
                "mean_delta": stat_r2_mean["mean_diff"], "se_corrected": stat_r2_mean["se_corrected"],
                "t_stat": stat_r2_mean["t_stat"], "p_val": stat_r2_mean["p_val"], "ci_95": stat_r2_mean["ci_95"]
            },
            "delta_r2_valence": {
                "mean_delta": stat_r2_val["mean_diff"], "se_corrected": stat_r2_val["se_corrected"],
                "t_stat": stat_r2_val["t_stat"], "p_val": stat_r2_val["p_val"], "ci_95": stat_r2_val["ci_95"]
            },
            "delta_r2_arousal": {
                "mean_delta": stat_r2_aro["mean_diff"], "se_corrected": stat_r2_aro["se_corrected"],
                "t_stat": stat_r2_aro["t_stat"], "p_val": stat_r2_aro["p_val"], "ci_95": stat_r2_aro["ci_95"]
            },
            "delta_mae_mean": {
                "mean_delta": stat_mae_mean["mean_diff"], "se_corrected": stat_mae_mean["se_corrected"],
                "t_stat": stat_mae_mean["t_stat"], "p_val": stat_mae_mean["p_val"], "ci_95": stat_mae_mean["ci_95"]
            }
        }
    }
    with open(os.path.join(C.OUT, "part_a_statistical_tests.json"), "w") as f:
        json.dump(tests_summary, f, indent=2)
    print("\nSaved part_a_statistical_tests.json successfully.")


if __name__ == "__main__":
    main()
