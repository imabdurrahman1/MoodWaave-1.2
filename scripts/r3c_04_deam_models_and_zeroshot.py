import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb

from r3c_common import (
    HERE,
    PROJECT_ROOT,
    SEED,
    LITE_FEATURES,
    FULL_FEATURES,
    TONNETZ_FEATURES,
    nadeau_bengio_test,
    tost_equivalence_test,
)

METADATA_CSV = os.path.join(HERE, "deam_metadata.csv")
FEATURES_CSV = os.path.join(HERE, "deam_features.csv")
PMEMO_READY_CSV = os.path.join(PROJECT_ROOT, "release", "tables", "pmemo_ready.csv")

# Hyperparameter search distributions for DEAM regression
LGBM_REG_PARAM_DIST = {
    "reg__num_leaves": [15, 31, 63],
    "reg__max_depth": [-1, 5, 8],
    "reg__learning_rate": [0.03, 0.05, 0.1],
    "reg__min_child_samples": [10, 20, 30],
    "reg__subsample": [0.7, 0.85, 1.0],
    "reg__colsample_bytree": [0.7, 0.85, 1.0],
}

RF_REG_PARAM_DIST = {
    "reg__n_estimators": [100, 200],
    "reg__max_depth": [None, 10, 20],
    "reg__min_samples_split": [2, 5],
    "reg__min_samples_leaf": [1, 2],
    "reg__max_features": ["sqrt", 1.0],
}

RIDGE_REG_PARAM_DIST = {
    "reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
}

MLP_REG_PARAM_DIST = {
    "reg__hidden_layer_sizes": [(64,), (128,), (64, 32)],
    "reg__alpha": [0.0001, 0.001, 0.01],
    "reg__learning_rate_init": [0.001, 0.01],
}

def get_feature_matrix(df_feats: pd.DataFrame, feature_set: str, tonnetz_version: str = "v0"):
    """Extract feature matrix for Lite-71 or Full-113."""
    if feature_set == "Lite-71":
        cols = LITE_FEATURES
        return df_feats[cols].values, cols
    elif feature_set == "Full-113":
        if tonnetz_version == "v0":
            cols = FULL_FEATURES
            return df_feats[cols].values, cols
        else:
            cols = []
            for c in FULL_FEATURES:
                if c.startswith("tonnetz_"):
                    c_v1 = c.replace("tonnetz_", "tonnetz_v1_")
                    cols.append(c_v1)
                else:
                    cols.append(c)
            return df_feats[cols].values, cols
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

def run_deam_model_families(df_meta: pd.DataFrame, df_feats: pd.DataFrame):
    """Part B.2: DEAM Model Families Nested Tuned Comparison on 5x2 Folds (df=9)."""
    print("\n--- Part B.2: DEAM Model Families (Nested Tuned, Random 5x2 CV, df=9) ---")
    results = []
    
    # We evaluate for both targets: valence and arousal
    targets = ["valence", "arousal"]
    
    model_configs = [
        ("LightGBM", "Full-113", "v0", lgb.LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1), LGBM_REG_PARAM_DIST),
        ("LightGBM", "Full-113", "v1", lgb.LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1), LGBM_REG_PARAM_DIST),
        ("LightGBM", "Lite-71", "v0", lgb.LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1), LGBM_REG_PARAM_DIST),
        ("Random Forest", "Full-113", "v0", RandomForestRegressor(random_state=SEED, n_jobs=1), RF_REG_PARAM_DIST),
        ("Ridge", "Full-113", "v0", Ridge(random_state=SEED), RIDGE_REG_PARAM_DIST),
        ("MLP", "Full-113", "v0", MLPRegressor(max_iter=200, early_stopping=True, random_state=SEED), MLP_REG_PARAM_DIST),
    ]
    
    for rep in range(2):
        rep_seed = SEED + rep * 100
        kf_outer = KFold(n_splits=5, shuffle=True, random_state=rep_seed)
        
        for fold, (train_idx, test_idx) in enumerate(kf_outer.split(df_feats)):
            global_fold = rep * 5 + fold
            print(f"\nOuter Fold {global_fold + 1}/10 (Rep {rep+1}, Fold {fold+1}; train: {len(train_idx)}, test: {len(test_idx)})...", flush=True)
            
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=rep_seed + fold)
            
            for target_name in targets:
                y = df_meta[target_name].values
                y_train, y_test = y[train_idx], y[test_idx]
                
                for name, f_set, t_ver, base_reg, p_dist in model_configs:
                    X_all, _ = get_feature_matrix(df_feats, f_set, t_ver)
                    X_train, X_test = X_all[train_idx], X_all[test_idx]
                    
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("reg", base_reg)
                    ])
                    
                    search = RandomizedSearchCV(
                        pipe,
                        p_dist,
                        n_iter=8,
                        cv=inner_cv,
                        scoring="neg_mean_absolute_error",
                        random_state=rep_seed + fold,
                        n_jobs=1,
                    )
                    
                    t0_fit = time.perf_counter()
                    search.fit(X_train, y_train)
                    fit_time_s = time.perf_counter() - t0_fit
                    best_model = search.best_estimator_
                    
                    t0_inf = time.perf_counter()
                    preds = best_model.predict(X_test)
                    infer_time_ms = ((time.perf_counter() - t0_inf) / len(X_test)) * 1000.0
                    
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    pr_val, _ = stats.pearsonr(y_test, preds)
                    sp_val, _ = stats.spearmanr(y_test, preds)
                    
                    label = f"{name}-{f_set}" if t_ver == "v0" else f"{name}-{f_set}-V1"
                    results.append({
                        "target": target_name,
                        "repeat": rep,
                        "fold": fold,
                        "global_fold": global_fold,
                        "model_name": name,
                        "feature_set": f_set,
                        "tonnetz_version": t_ver,
                        "label": label,
                        "r2": r2,
                        "mae": mae,
                        "pearson_r": pr_val,
                        "spearman_rho": sp_val,
                        "fit_time_s": fit_time_s,
                        "infer_time_ms_per_track": infer_time_ms,
                        "best_params": str(search.best_params_),
                    })
                    print(f"  [{target_name.capitalize()}] [{label}] R2={r2:.4f}, MAE={mae:.4f}, r={pr_val:.4f}, rho={sp_val:.4f}, Fit={fit_time_s:.2f}s, Inf={infer_time_ms:.2f}ms/trk", flush=True)
                    
    df_results = pd.DataFrame(results)
    out_folds_csv = os.path.join(HERE, "deam_model_families_folds.csv")
    df_results.to_csv(out_folds_csv, index=False)
    print(f"Saved DEAM model families fold results to {out_folds_csv}")
    
    # Statistical analysis against LightGBM Full-113 & LightGBM Lite-71
    summary_rows = []
    for target_name in targets:
        df_t = df_results[df_results["target"] == target_name]
        ref_full = df_t[df_t["label"] == "LightGBM-Full-113"].sort_values("global_fold")
        ref_lite = df_t[df_t["label"] == "LightGBM-Lite-71"].sort_values("global_fold")
        
        for l in df_t["label"].unique():
            sub = df_t[df_t["label"] == l].sort_values("global_fold")
            
            r2_mean, r2_std = sub["r2"].mean(), sub["r2"].std()
            mae_mean, mae_std = sub["mae"].mean(), sub["mae"].std()
            pr_mean, pr_std = sub["pearson_r"].mean(), sub["pearson_r"].std()
            sp_mean, sp_std = sub["spearman_rho"].mean(), sub["spearman_rho"].std()
            fit_mean, fit_std = sub["fit_time_s"].mean(), sub["fit_time_s"].std()
            inf_mean, inf_std = sub["infer_time_ms_per_track"].mean(), sub["infer_time_ms_per_track"].std()
            
            # Differences for R2
            diffs_full_r2 = sub["r2"].values - ref_full["r2"].values
            diffs_lite_r2 = sub["r2"].values - ref_lite["r2"].values
            
            nb_full = nadeau_bengio_test(diffs_full_r2, test_train_ratio=0.25)
            tost_full = tost_equivalence_test(diffs_full_r2, bound=0.05, test_train_ratio=0.25)
            
            nb_lite = nadeau_bengio_test(diffs_lite_r2, test_train_ratio=0.25)
            tost_lite = tost_equivalence_test(diffs_lite_r2, bound=0.05, test_train_ratio=0.25)
            
            summary_rows.append({
                "target": target_name,
                "model": l,
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "pearson_r_mean": pr_mean,
                "pearson_r_std": pr_std,
                "spearman_rho_mean": sp_mean,
                "spearman_rho_std": sp_std,
                "fit_time_s_mean": fit_mean,
                "fit_time_s_std": fit_std,
                "infer_time_ms_mean": inf_mean,
                "infer_time_ms_std": inf_std,
                "r2_diff_vs_full": nb_full["mean_diff"],
                "r2_pval_vs_full": nb_full["p_val"],
                "r2_sig_vs_full": nb_full["significant"],
                "r2_tost_eq_vs_full": tost_full["equivalent"],
                "r2_tost_pval_vs_full": tost_full["p_tost"],
                "r2_diff_vs_lite": nb_lite["mean_diff"],
                "r2_pval_vs_lite": nb_lite["p_val"],
                "r2_sig_vs_lite": nb_lite["significant"],
                "r2_tost_eq_vs_lite": tost_lite["equivalent"],
                "r2_tost_pval_vs_lite": tost_lite["p_tost"],
            })
            
    df_summary = pd.DataFrame(summary_rows)
    out_sum_csv = os.path.join(HERE, "deam_model_families_summary.csv")
    df_summary.to_csv(out_sum_csv, index=False)
    print(f"Saved DEAM model families summary to {out_sum_csv}")
    print(df_summary[["target", "model", "r2_mean", "mae_mean", "pearson_r_mean", "spearman_rho_mean", "r2_diff_vs_full", "r2_tost_eq_vs_full"]].to_string(index=False))
    return df_summary

def run_zeroshot_pmemo_to_deam(df_meta_deam: pd.DataFrame, df_feats_deam: pd.DataFrame):
    """Part B.3: Zero-shot transfer from PMEmo to DEAM."""
    print("\n--- Part B.3: Zero-Shot Transfer: PMEmo -> DEAM ---")
    
    # 1. Load PMEmo ready data
    print(f"Loading PMEmo training data from {PMEMO_READY_CSV}...")
    df_pmemo = pd.read_csv(PMEMO_READY_CSV)
    print(f"PMEmo tracks: {len(df_pmemo)}")
    
    # Check targets: PMEmo has 'Valence(mean)' and 'Arousal(mean)' in [0, 1]
    y_pmemo_val = df_pmemo["Valence(mean)"].values
    y_pmemo_aro = df_pmemo["Arousal(mean)"].values
    
    # DEAM test targets in [0, 1]
    y_deam_val = df_meta_deam["valence"].values
    y_deam_aro = df_meta_deam["arousal"].values
    
    # PMEmo feature matrices
    X_pmemo_full = df_pmemo[FULL_FEATURES].values
    X_pmemo_lite = df_pmemo[LITE_FEATURES].values
    
    # DEAM feature matrices
    X_deam_full, _ = get_feature_matrix(df_feats_deam, "Full-113", "v0")
    X_deam_lite, _ = get_feature_matrix(df_feats_deam, "Lite-71", "v0")
    
    # Models to train on PMEmo and test on DEAM
    # Project default hyperparameters
    lgbm_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": 1
    }
    
    rf_params = {
        "n_estimators": 100,
        "random_state": SEED,
        "n_jobs": 1
    }
    
    transfer_configs = [
        ("Zero-Shot LightGBM-Full-113", lgb.LGBMRegressor(**lgbm_params), X_pmemo_full, X_deam_full),
        ("Zero-Shot LightGBM-Lite-71", lgb.LGBMRegressor(**lgbm_params), X_pmemo_lite, X_deam_lite),
        ("Zero-Shot Random Forest", RandomForestRegressor(**rf_params), X_pmemo_full, X_deam_full),
    ]
    
    transfer_results = []
    
    # Evaluate models
    for name, model, X_tr, X_te in transfer_configs:
        # Valence
        model.fit(X_tr, y_pmemo_val)
        pred_val = model.predict(X_te)
        r2_v = r2_score(y_deam_val, pred_val)
        mae_v = mean_absolute_error(y_deam_val, pred_val)
        pr_v, _ = stats.pearsonr(y_deam_val, pred_val)
        sp_v, _ = stats.spearmanr(y_deam_val, pred_val)
        
        # Arousal
        model.fit(X_tr, y_pmemo_aro)
        pred_aro = model.predict(X_te)
        r2_a = r2_score(y_deam_aro, pred_aro)
        mae_a = mean_absolute_error(y_deam_aro, pred_aro)
        pr_a, _ = stats.pearsonr(y_deam_aro, pred_aro)
        sp_a, _ = stats.spearmanr(y_deam_aro, pred_aro)
        
        transfer_results.append({
            "model": name,
            "valence_r2": r2_v,
            "valence_mae": mae_v,
            "valence_pearson_r": pr_v,
            "valence_spearman_rho": sp_v,
            "arousal_r2": r2_a,
            "arousal_mae": mae_a,
            "arousal_pearson_r": pr_a,
            "arousal_spearman_rho": sp_a,
            "mean_r2": (r2_v + r2_a) / 2.0,
            "mean_mae": (mae_v + mae_a) / 2.0,
            "mean_pearson_r": (pr_v + pr_a) / 2.0,
            "mean_spearman_rho": (sp_v + sp_a) / 2.0,
        })
        
    # Baseline 1: In-domain DEAM models (from summary if available)
    deam_sum_csv = os.path.join(HERE, "deam_model_families_summary.csv")
    if os.path.isfile(deam_sum_csv):
        df_ds = pd.read_csv(deam_sum_csv)
        for m_name in ["LightGBM-Full-113", "LightGBM-Lite-71"]:
            row_v = df_ds[(df_ds["model"] == m_name) & (df_ds["target"] == "valence")].iloc[0]
            row_a = df_ds[(df_ds["model"] == m_name) & (df_ds["target"] == "arousal")].iloc[0]
            transfer_results.append({
                "model": f"In-Domain DEAM ({m_name})",
                "valence_r2": row_v["r2_mean"],
                "valence_mae": row_v["mae_mean"],
                "valence_pearson_r": row_v["pearson_r_mean"],
                "valence_spearman_rho": row_v["spearman_rho_mean"],
                "arousal_r2": row_a["r2_mean"],
                "arousal_mae": row_a["mae_mean"],
                "arousal_pearson_r": row_a["pearson_r_mean"],
                "arousal_spearman_rho": row_a["spearman_rho_mean"],
                "mean_r2": (row_v["r2_mean"] + row_a["r2_mean"]) / 2.0,
                "mean_mae": (row_v["mae_mean"] + row_a["mae_mean"]) / 2.0,
                "mean_pearson_r": (row_v["pearson_r_mean"] + row_a["pearson_r_mean"]) / 2.0,
                "mean_spearman_rho": (row_v["spearman_rho_mean"] + row_a["spearman_rho_mean"]) / 2.0,
            })
            
    # Baseline 2: Predicting DEAM dataset mean
    deam_mean_v = np.mean(y_deam_val)
    deam_mean_a = np.mean(y_deam_aro)
    pred_base_deam_v = np.full_like(y_deam_val, deam_mean_v)
    pred_base_deam_a = np.full_like(y_deam_aro, deam_mean_a)
    
    r2_bv = r2_score(y_deam_val, pred_base_deam_v)
    mae_bv = mean_absolute_error(y_deam_val, pred_base_deam_v)
    r2_ba = r2_score(y_deam_aro, pred_base_deam_a)
    mae_ba = mean_absolute_error(y_deam_aro, pred_base_deam_a)
    # Correlation of a constant is 0 by definition
    transfer_results.append({
        "model": "Trivial Baseline: DEAM Dataset Mean",
        "valence_r2": r2_bv,
        "valence_mae": mae_bv,
        "valence_pearson_r": 0.0,
        "valence_spearman_rho": 0.0,
        "arousal_r2": r2_ba,
        "arousal_mae": mae_ba,
        "arousal_pearson_r": 0.0,
        "arousal_spearman_rho": 0.0,
        "mean_r2": (r2_bv + r2_ba) / 2.0,
        "mean_mae": (mae_bv + mae_ba) / 2.0,
        "mean_pearson_r": 0.0,
        "mean_spearman_rho": 0.0,
    })
    
    # Baseline 3: Predicting PMEmo training-set mean
    pmemo_mean_v = np.mean(y_pmemo_val)
    pmemo_mean_a = np.mean(y_pmemo_aro)
    pred_base_pmemo_v = np.full_like(y_deam_val, pmemo_mean_v)
    pred_base_pmemo_a = np.full_like(y_deam_aro, pmemo_mean_a)
    
    r2_bpv = r2_score(y_deam_val, pred_base_pmemo_v)
    mae_bpv = mean_absolute_error(y_deam_val, pred_base_pmemo_v)
    r2_bpa = r2_score(y_deam_aro, pred_base_pmemo_a)
    mae_bpa = mean_absolute_error(y_deam_aro, pred_base_pmemo_a)
    transfer_results.append({
        "model": "Trivial Baseline: PMEmo Dataset Mean",
        "valence_r2": r2_bpv,
        "valence_mae": mae_bpv,
        "valence_pearson_r": 0.0,
        "valence_spearman_rho": 0.0,
        "arousal_r2": r2_bpa,
        "arousal_mae": mae_bpa,
        "arousal_pearson_r": 0.0,
        "arousal_spearman_rho": 0.0,
        "mean_r2": (r2_bpv + r2_bpa) / 2.0,
        "mean_mae": (mae_bpv + mae_bpa) / 2.0,
        "mean_pearson_r": 0.0,
        "mean_spearman_rho": 0.0,
    })
    
    df_transfer = pd.DataFrame(transfer_results)
    out_zeroshot_csv = os.path.join(HERE, "pmemo_to_deam_zeroshot.csv")
    df_transfer.to_csv(out_zeroshot_csv, index=False)
    print(f"\nSaved PMEmo -> DEAM zero-shot transfer results to {out_zeroshot_csv}")
    print(df_transfer[["model", "valence_r2", "valence_pearson_r", "arousal_r2", "arousal_pearson_r", "mean_r2", "mean_pearson_r"]].to_string(index=False))
    return df_transfer

def main():
    print("=== Step 4: DEAM Model Families & Zero-Shot Transfer ===")
    t_start = time.perf_counter()
    
    assert os.path.isfile(METADATA_CSV), f"Missing {METADATA_CSV}"
    assert os.path.isfile(FEATURES_CSV), f"Missing {FEATURES_CSV}"
    
    df_meta = pd.read_csv(METADATA_CSV)
    df_feats = pd.read_csv(FEATURES_CSV).set_index("song_id").loc[df_meta["song_id"]].reset_index()
    
    run_deam_model_families(df_meta, df_feats)
    run_zeroshot_pmemo_to_deam(df_meta, df_feats)
    
    print(f"\nStep 4 completed in {(time.perf_counter() - t_start)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
