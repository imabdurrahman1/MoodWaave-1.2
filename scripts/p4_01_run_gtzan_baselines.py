"""
experiments/rev2_p4_baselines/01_run_gtzan_baselines.py
Nested CV evaluation of classical baselines on GTZAN (Genre classification).
Models: LightGBM, Random Forest, XGBoost, SVM-RBF, Logistic Regression.
Feature sets: Full-113, Lite-71.
Inner tuning: RandomizedSearchCV(n_iter=8, cv=3) on training set only.
Outer CV: P1 artist-disjoint folds from splits.json (clean 929 tracks).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common

def evaluate_fold_config(
    repeat: int,
    fold_idx: int,
    model_name: str,
    feat_set_name: str,
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    features: list,
):
    """
    Runs inner RandomizedSearchCV on outer training fold, then evaluates on outer test fold.
    """
    t0 = time.time()
    
    # Feature slicing
    X_tr = X_train_df[features].values
    X_te = X_test_df[features].values
    
    # Label encoding for XGBoost if needed
    if model_name == "XGBoost":
        le = LabelEncoder()
        y_tr_fit = le.fit_transform(y_train)
        y_te_eval = le.transform(y_test)
    else:
        y_tr_fit = y_train
        y_te_eval = y_test
        
    # Get base estimator and param grid
    base_model = common.get_genre_base_model(model_name, seed=42)
    grid = common.GENRE_PARAM_GRIDS[model_name]
    
    # Inner CV search
    # Fold-specific seed for reproducible yet diverse hyperparameter sampling
    inner_seed = 42 + repeat * 100 + fold_idx
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=grid,
        n_iter=8,
        cv=3,
        scoring="accuracy",
        random_state=inner_seed,
        n_jobs=1,
    )
    
    search.fit(X_tr, y_tr_fit)
    best_estimator = search.best_estimator_
    best_params = search.best_params_
    
    # Predict on test set
    preds = best_estimator.predict(X_te)
    
    acc = accuracy_score(y_te_eval, preds)
    f1 = f1_score(y_te_eval, preds, average="macro")
    fit_time = time.time() - t0
    
    return {
        "dataset": "GTZAN",
        "task": "genre",
        "repeat": repeat,
        "fold": fold_idx,
        "model": model_name,
        "feature_set": feat_set_name,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "best_params": json.dumps(best_params),
        "fit_time_sec": round(fit_time, 2),
    }

def run_single_fold(
    fold_info: dict,
    clean_df: pd.DataFrame,
    models: list,
    feat_sets: dict,
):
    """Runs all models and feature sets for a single outer fold."""
    repeat = fold_info["repeat"]
    fold_idx = fold_info["fold"]
    train_tracks = fold_info["train_tracks"]
    test_tracks = fold_info["test_tracks"]
    
    train_df = clean_df.loc[train_tracks]
    test_df = clean_df.loc[test_tracks]
    
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    
    results = []
    print(f"--> Starting GTZAN Fold (rep={repeat}, fold={fold_idx}) with {len(train_tracks)} train, {len(test_tracks)} test tracks...")
    fold_start = time.time()
    
    for feat_set_name, feats in feat_sets.items():
        for model_name in models:
            res = evaluate_fold_config(
                repeat=repeat,
                fold_idx=fold_idx,
                model_name=model_name,
                feat_set_name=feat_set_name,
                X_train_df=train_df,
                y_train=y_train,
                X_test_df=test_df,
                y_test=y_test,
                features=feats,
            )
            print(f"    [rep={repeat}, fold={fold_idx}] {model_name:18s} ({feat_set_name:8s}): Acc={res['accuracy']:.4f}, F1={res['macro_f1']:.4f} in {res['fit_time_sec']:.1f}s")
            results.append(res)
            
    print(f"<-- Finished GTZAN Fold (rep={repeat}, fold={fold_idx}) in {time.time() - fold_start:.1f}s")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run GTZAN Classical Baselines")
    parser.add_argument("--repeats", type=int, default=1, choices=[1, 3], help="Number of outer repeats (1=5 folds, 3=15 folds)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel worker processes")
    args = parser.parse_args()
    
    out_dir = common.BASE_DIR
    common.log_environment(out_dir)
    
    # 1. Load data
    gtzan_path = os.path.join(common.PROJECT_ROOT, "data", "gtzan_features.csv")
    rem_path = os.path.join(common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "removed_tracks.csv")
    splits_path = os.path.join(common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "splits.json")
    
    df_g = pd.read_csv(gtzan_path).set_index("filename")
    rem_tracks = set(pd.read_csv(rem_path)["filename"])
    clean_df = df_g.loc[~df_g.index.isin(rem_tracks)]
    print(f"Loaded GTZAN clean tracks: {len(clean_df)}")
    
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)["gtzan"]["fault_removed_artist_grouped"]
        
    selected_folds = [f for f in splits if f["repeat"] < args.repeats]
    print(f"Selected {len(selected_folds)} folds across {args.repeats} repeat(s).")
    
    models = ["LightGBM", "Random Forest", "XGBoost", "SVM-RBF", "Logistic Regression"]
    feat_sets = {
        "Full-113": common.FULL_113_FEATURES,
        "Lite-71": common.LITE_71_FEATURES,
    }
    
    results_csv = os.path.join(out_dir, "gtzan_baseline_folds.csv")
    existing_records = []
    if os.path.exists(results_csv):
        df_exist = pd.read_csv(results_csv)
        print(f"Found existing results file with {len(df_exist)} records.")
        # Filter folds that are completely computed
        # A fold is complete if it has len(models) * len(feat_sets) = 10 records
        completed_keys = set()
        for (rep, fld), grp in df_exist.groupby(["repeat", "fold"]):
            if len(grp) == len(models) * len(feat_sets):
                completed_keys.add((rep, fld))
        selected_folds = [f for f in selected_folds if (f["repeat"], f["fold"]) not in completed_keys]
        existing_records = df_exist.to_dict("records")
        print(f"Remaining folds to compute: {len(selected_folds)}")
        
    if not selected_folds:
        print("All requested folds already computed!")
        return
        
    print(f"Launching evaluation on {len(selected_folds)} folds using {args.workers} workers...")
    all_results = list(existing_records)
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_fold, fold_info, clean_df, models, feat_sets): fold_info
            for fold_info in selected_folds
        }
        for fut in as_completed(futures):
            fold_info = futures[fut]
            try:
                res_list = fut.result()
                all_results.extend(res_list)
                # Incremental write
                pd.DataFrame(all_results).to_csv(results_csv, index=False)
                print(f"[PROGRESS] Saved {len(all_results)} total records to {results_csv}")
            except Exception as e:
                print(f"Error processing fold (rep={fold_info['repeat']}, fold={fold_info['fold']}): {e}")
                raise e

    print(f"GTZAN baselines run completed! Total records saved: {len(all_results)}")

if __name__ == "__main__":
    main()

