import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

from r3c_common import (
    HERE,
    SEED,
    LITE_FEATURES,
    FULL_FEATURES,
    TONNETZ_FEATURES,
    nadeau_bengio_test,
    tost_equivalence_test,
)

METADATA_CSV = os.path.join(HERE, "fma_800_metadata.csv")
FEATURES_CSV = os.path.join(HERE, "fma_800_features.csv")

# Hyperparameter Grids matching Phase A
LGBM_PARAM_DIST = {
    "clf__num_leaves": [15, 31, 63],
    "clf__max_depth": [-1, 5, 8],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__min_child_samples": [10, 20, 30],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
}

RF_PARAM_DIST = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features": ["sqrt", "log2"],
}

LOGREG_PARAM_DIST = {
    "clf__C": [0.003, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0]
}

MLP_PARAM_DIST = {
    "clf__hidden_layer_sizes": [(64,), (128,), (64, 32)],
    "clf__alpha": [0.0001, 0.001, 0.01],
    "clf__learning_rate_init": [0.001, 0.01],
}

def get_feature_matrix(df_feats: pd.DataFrame, feature_set: str, tonnetz_version: str = "v0"):
    """Extract feature matrix for Lite-71 or Full-113 (with V0 or V1 Tonnetz)."""
    if feature_set == "Lite-71":
        cols = LITE_FEATURES
        return df_feats[cols].values, cols
    elif feature_set == "Full-113":
        if tonnetz_version == "v0":
            cols = FULL_FEATURES
            return df_feats[cols].values, cols
        else:
            # Tonnetz V1 replaces the 12 Tonnetz columns with tonnetz_v1_mean/std
            cols = []
            for c in FULL_FEATURES:
                if c.startswith("tonnetz_"):
                    # replace e.g. tonnetz_mean_1 with tonnetz_v1_mean_1
                    c_v1 = c.replace("tonnetz_", "tonnetz_v1_")
                    cols.append(c_v1)
                else:
                    cols.append(c)
            return df_feats[cols].values, cols
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

def run_leakage_study(df_meta: pd.DataFrame, df_feats: pd.DataFrame):
    """Protocol (a) Stratified Random vs (b) Artist-Disjoint 5x3 on Default LightGBM & RF."""
    print("\n--- Part A.1: FMA Leakage Drop Study (Random vs. Artist-Disjoint 5x3) ---")
    results = []
    
    y = df_meta["genre"].values
    artists = df_meta["artist_id"].values
    
    configs = [
        ("LightGBM", "Full-113", "v0"),
        ("LightGBM", "Full-113", "v1"),
        ("LightGBM", "Lite-71", "v0"),
        ("Random Forest", "Full-113", "v0"),
        ("Random Forest", "Full-113", "v1"),
        ("Random Forest", "Lite-71", "v0"),
    ]
    
    for split_type in ["Random", "Artist-Disjoint"]:
        print(f"\nEvaluating Split Type: {split_type} (5 folds x 3 repeats)...")
        for rep in range(3):
            rep_seed = 42 + rep * 100
            
            if split_type == "Random":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep_seed)
                split_gen = cv.split(df_feats, y)
            else:
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep_seed)
                split_gen = cv.split(df_feats, y, groups=artists)
                
            for fold, (train_idx, test_idx) in enumerate(split_gen):
                y_train, y_test = y[train_idx], y[test_idx]
                
                for model_name, f_set, t_ver in configs:
                    X_all, _ = get_feature_matrix(df_feats, f_set, t_ver)
                    X_train, X_test = X_all[train_idx], X_all[test_idx]
                    
                    if model_name == "LightGBM":
                        clf = lgb.LGBMClassifier(
                            n_estimators=200,
                            learning_rate=0.1,
                            num_leaves=31,
                            min_child_samples=20,
                            random_state=rep_seed + fold,
                            verbose=-1,
                            n_jobs=1,
                        )
                    else:
                        clf = RandomForestClassifier(
                            n_estimators=100,
                            random_state=rep_seed + fold,
                            n_jobs=1,
                        )
                        
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="macro")
                    
                    label_model = f"{model_name}-{f_set}" if t_ver == "v0" else f"{model_name}-{f_set}-V1"
                    results.append({
                        "split_type": split_type,
                        "repeat": rep,
                        "fold": fold,
                        "model": model_name,
                        "feature_set": f_set,
                        "tonnetz_version": t_ver,
                        "label_model": label_model,
                        "accuracy": acc,
                        "macro_f1": f1,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "train_artists": len(set(artists[train_idx])),
                        "test_artists": len(set(artists[test_idx])),
                    })
                    
    df_leakage = pd.DataFrame(results)
    out_folds_csv = os.path.join(HERE, "fma_leakage_folds.csv")
    df_leakage.to_csv(out_folds_csv, index=False)
    print(f"Saved leakage fold results to {out_folds_csv}")
    
    # Calculate summary & leakage drops
    summary_rows = []
    models_unique = df_leakage["label_model"].unique()
    for m in models_unique:
        sub_rnd = df_leakage[(df_leakage["label_model"] == m) & (df_leakage["split_type"] == "Random")].sort_values(["repeat", "fold"])
        sub_adj = df_leakage[(df_leakage["label_model"] == m) & (df_leakage["split_type"] == "Artist-Disjoint")].sort_values(["repeat", "fold"])
        
        acc_rnd_mean, acc_rnd_std = sub_rnd["accuracy"].mean(), sub_rnd["accuracy"].std()
        acc_adj_mean, acc_adj_std = sub_adj["accuracy"].mean(), sub_adj["accuracy"].std()
        f1_rnd_mean, f1_rnd_std = sub_rnd["macro_f1"].mean(), sub_rnd["macro_f1"].std()
        f1_adj_mean, f1_adj_std = sub_adj["macro_f1"].mean(), sub_adj["macro_f1"].std()
        
        diff_acc = sub_rnd["accuracy"].values - sub_adj["accuracy"].values
        diff_f1 = sub_rnd["macro_f1"].values - sub_adj["macro_f1"].values
        
        nb_acc = nadeau_bengio_test(diff_acc, test_train_ratio=0.25)
        nb_f1 = nadeau_bengio_test(diff_f1, test_train_ratio=0.25)
        
        summary_rows.append({
            "model": m,
            "acc_random": acc_rnd_mean,
            "acc_random_std": acc_rnd_std,
            "acc_artist_disjoint": acc_adj_mean,
            "acc_artist_disjoint_std": acc_adj_std,
            "acc_leakage_drop": nb_acc["mean_diff"],
            "acc_drop_p_val": nb_acc["p_val"],
            "acc_drop_significant": nb_acc["significant"],
            "f1_random": f1_rnd_mean,
            "f1_random_std": f1_rnd_std,
            "f1_artist_disjoint": f1_adj_mean,
            "f1_artist_disjoint_std": f1_adj_std,
            "f1_leakage_drop": nb_f1["mean_diff"],
            "f1_drop_p_val": nb_f1["p_val"],
            "f1_drop_significant": nb_f1["significant"],
        })
        
    df_leakage_sum = pd.DataFrame(summary_rows)
    out_sum_csv = os.path.join(HERE, "fma_leakage_summary.csv")
    df_leakage_sum.to_csv(out_sum_csv, index=False)
    print(f"Saved leakage summary to {out_sum_csv}")
    print(df_leakage_sum[["model", "acc_random", "acc_artist_disjoint", "acc_leakage_drop", "acc_drop_p_val", "acc_drop_significant"]].to_string(index=False))
    return df_leakage_sum

def run_model_families_fma(df_meta: pd.DataFrame, df_feats: pd.DataFrame):
    """Part A.3: Model families nested tuned comparison on Artist-Disjoint 5 folds (df=4)."""
    print("\n--- Part A.3: FMA Model Families (Nested Tuned, Artist-Disjoint 5 Folds, df=4) ---")
    results = []
    
    y = df_meta["genre"].values
    artists = df_meta["artist_id"].values
    
    cv_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    model_configs = [
        ("LightGBM", "Full-113", "v0", lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1), LGBM_PARAM_DIST),
        ("LightGBM", "Full-113", "v1", lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1), LGBM_PARAM_DIST),
        ("LightGBM", "Lite-71", "v0", lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1), LGBM_PARAM_DIST),
        ("Random Forest", "Full-113", "v0", RandomForestClassifier(random_state=SEED, n_jobs=1), RF_PARAM_DIST),
        ("Logistic Regression", "Full-113", "v0", LogisticRegression(max_iter=1000, random_state=SEED), LOGREG_PARAM_DIST),
        ("MLP", "Full-113", "v0", MLPClassifier(max_iter=500, random_state=SEED), MLP_PARAM_DIST),
    ]
    
    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(df_feats, y, groups=artists)):
        print(f"\nOuter Fold {fold + 1}/5 (train: {len(train_idx)}, test: {len(test_idx)})...", flush=True)
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = artists[train_idx]
        
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
        
        for name, f_set, t_ver, base_clf, p_dist in model_configs:
            X_all, _ = get_feature_matrix(df_feats, f_set, t_ver)
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", base_clf)
            ])
            
            search = RandomizedSearchCV(
                pipe,
                p_dist,
                n_iter=8,
                cv=inner_cv,
                scoring="accuracy",
                random_state=SEED + fold,
                n_jobs=1,
            )
            
            search.fit(X_train, y_train, groups=groups_train)
            best_model = search.best_estimator_
            
            preds = best_model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            
            label = f"{name}-{f_set}" if t_ver == "v0" else f"{name}-{f_set}-V1"
            results.append({
                "model_name": name,
                "feature_set": f_set,
                "tonnetz_version": t_ver,
                "label": label,
                "outer_fold": fold,
                "accuracy": acc,
                "macro_f1": f1,
                "best_params": str(search.best_params_),
            })
            print(f"  [{label}] Fold {fold+1}: Acc={acc:.4f}, Macro-F1={f1:.4f}", flush=True)
            
    df_models = pd.DataFrame(results)
    out_mf_folds_csv = os.path.join(HERE, "fma_model_families_folds.csv")
    df_models.to_csv(out_mf_folds_csv, index=False)
    print(f"Saved model families folds to {out_mf_folds_csv}")
    
    # Statistical significance vs LightGBM Full-113 & TOST tests
    ref_full = df_models[df_models["label"] == "LightGBM-Full-113"].sort_values("outer_fold")
    ref_lite = df_models[df_models["label"] == "LightGBM-Lite-71"].sort_values("outer_fold")
    
    sum_rows = []
    for l in df_models["label"].unique():
        sub = df_models[df_models["label"] == l].sort_values("outer_fold")
        acc_mean, acc_std = sub["accuracy"].mean(), sub["accuracy"].std()
        f1_mean, f1_std = sub["macro_f1"].mean(), sub["macro_f1"].std()
        
        diffs_full_acc = sub["accuracy"].values - ref_full["accuracy"].values
        diffs_lite_acc = sub["accuracy"].values - ref_lite["accuracy"].values
        
        nb_full = nadeau_bengio_test(diffs_full_acc, test_train_ratio=0.25)
        tost_full = tost_equivalence_test(diffs_full_acc, bound=0.05, test_train_ratio=0.25)
        
        nb_lite = nadeau_bengio_test(diffs_lite_acc, test_train_ratio=0.25)
        tost_lite = tost_equivalence_test(diffs_lite_acc, bound=0.05, test_train_ratio=0.25)
        
        sum_rows.append({
            "model": l,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "diff_vs_full": nb_full["mean_diff"],
            "p_val_vs_full": nb_full["p_val"],
            "sig_vs_full": nb_full["significant"],
            "tost_eq_vs_full": tost_full["equivalent"],
            "tost_p_vs_full": tost_full["p_tost"],
            "diff_vs_lite": nb_lite["mean_diff"],
            "p_val_vs_lite": nb_lite["p_val"],
            "sig_vs_lite": nb_lite["significant"],
            "tost_eq_vs_lite": tost_lite["equivalent"],
            "tost_p_vs_lite": tost_lite["p_tost"],
        })
        
    df_mf_sum = pd.DataFrame(sum_rows)
    out_mf_sum_csv = os.path.join(HERE, "fma_model_families_summary.csv")
    df_mf_sum.to_csv(out_mf_sum_csv, index=False)
    print(f"Saved model families summary to {out_mf_sum_csv}")
    print(df_mf_sum[["model", "accuracy_mean", "macro_f1_mean", "diff_vs_full", "p_val_vs_full", "tost_eq_vs_full"]].to_string(index=False))
    return df_mf_sum

def main():
    print("=== Step 3: FMA-small Leakage Drop & Model Families ===")
    t_start = time.perf_counter()
    
    assert os.path.isfile(METADATA_CSV), f"Missing {METADATA_CSV}"
    assert os.path.isfile(FEATURES_CSV), f"Missing {FEATURES_CSV}"
    
    df_meta = pd.read_csv(METADATA_CSV)
    df_feats = pd.read_csv(FEATURES_CSV).set_index("track_id").loc[df_meta["track_id"]].reset_index()
    
    run_leakage_study(df_meta, df_feats)
    run_model_families_fma(df_meta, df_feats)
    
    print(f"\nStep 3 completed in {(time.perf_counter() - t_start)/60:.2f} minutes.")

if __name__ == "__main__":
    main()

