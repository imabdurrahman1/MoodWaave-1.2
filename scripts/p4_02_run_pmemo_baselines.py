"""
experiments/rev2_p4_baselines/02_run_pmemo_baselines.py
Nested CV evaluation of classical baselines on PMEmo (Valence & Arousal regression).
Models: LightGBM, Random Forest, XGBoost, SVR-RBF, Ridge.
Feature sets: Full-113, Lite-71.
Inner tuning: RandomizedSearchCV(n_iter=8, cv=3, scoring='r2') on training set only.
Outer CV: P1 artist-grouped 5x2 folds from splits.json (10 folds total).
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
from sklearn.metrics import r2_score, mean_absolute_error

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common

def evaluate_fold_config(
    target_name: str,
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
    
    # Base estimator and parameter grid
    base_model = common.get_va_base_model(model_name, seed=42)
    grid = common.VA_PARAM_GRIDS[model_name]
    
    # Inner CV search
    # Target-specific and fold-specific seed for reproducibility
    target_offset = 0 if "valence" in target_name.lower() else 500
    inner_seed = 42 + target_offset + repeat * 100 + fold_idx
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=grid,
        n_iter=8,
        cv=3,
        scoring="r2",
        random_state=inner_seed,
        n_jobs=1,
    )
    
    search.fit(X_tr, y_train)
    best_estimator = search.best_estimator_
    best_params = search.best_params_
    
    # Predict on test set
    preds = best_estimator.predict(X_te)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    fit_time = time.time() - t0
    
    task_label = "valence" if "valence" in target_name.lower() else "arousal"
    
    return {
        "dataset": "PMEmo",
        "task": task_label,
        "target": target_name,
        "repeat": repeat,
        "fold": fold_idx,
        "model": model_name,
        "feature_set": feat_set_name,
        "r2": float(r2),
        "mae": float(mae),
        "best_params": json.dumps(best_params),
        "fit_time_sec": round(fit_time, 2),
    }

def run_single_fold(
    fold_info: dict,
    pmemo_df: pd.DataFrame,
    targets: list,
    models: list,
    feat_sets: dict,
):
    """Runs all targets, models and feature sets for a single outer fold."""
    repeat = fold_info["repeat"]
    fold_idx = fold_info["fold"]
    train_tracks = fold_info["train_tracks"]
    test_tracks = fold_info["test_tracks"]
    
    train_df = pmemo_df.loc[train_tracks]
    test_df = pmemo_df.loc[test_tracks]
    
    results = []
    print(f"--> Starting PMEmo Fold (rep={repeat}, fold={fold_idx}) with {len(train_tracks)} train, {len(test_tracks)} test tracks...")
    fold_start = time.time()
    
    for target_col in targets:
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        task_label = "valence" if "valence" in target_col.lower() else "arousal"
        
        for feat_set_name, feats in feat_sets.items():
            for model_name in models:
                res = evaluate_fold_config(
                    target_name=target_col,
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
                print(f"    [{task_label:7s} rep={repeat}, fold={fold_idx}] {model_name:18s} ({feat_set_name:8s}): R2={res['r2']:.4f}, MAE={res['mae']:.4f} in {res['fit_time_sec']:.1f}s")
                results.append(res)
                
    print(f"<-- Finished PMEmo Fold (rep={repeat}, fold={fold_idx}) in {time.time() - fold_start:.1f}s")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run PMEmo Classical Baselines")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel worker processes")
    args = parser.parse_args()
    
    out_dir = common.BASE_DIR
    common.log_environment(out_dir)
    
    # 1. Load data
    pmemo_path = os.path.join(common.PROJECT_ROOT, "data", "pmemo_ready.csv")
    splits_path = os.path.join(common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "splits.json")
    
    df_p = pd.read_csv(pmemo_path)
    df_p["musicId_str"] = df_p["musicId"].astype(str)
    df_p.set_index("musicId_str", inplace=True)
    print(f"Loaded PMEmo tracks: {len(df_p)}")
    
    with open(splits_path, "r", encoding="utf-8") as f:
        all_splits = json.load(f)["pmemo"]["artist_grouped_5x5"]
        
    # P1 5x2 protocol: repeat in (0, 1) -> 10 folds
    selected_folds = [f for f in all_splits if f["repeat"] in (0, 1)]
    print(f"Selected {len(selected_folds)} folds for P1 5x2 protocol.")
    
    targets = ["Valence(mean)", "Arousal(mean)"]
    models = ["LightGBM", "Random Forest", "XGBoost", "SVR-RBF", "Ridge"]
    feat_sets = {
        "Full-113": common.FULL_113_FEATURES,
        "Lite-71": common.LITE_71_FEATURES,
    }
    
    results_csv = os.path.join(out_dir, "pmemo_baseline_folds.csv")
    existing_records = []
    if os.path.exists(results_csv):
        df_exist = pd.read_csv(results_csv)
        print(f"Found existing results file with {len(df_exist)} records.")
        # Complete fold has: len(targets) * len(feat_sets) * len(models) = 2 * 2 * 5 = 20 records
        completed_keys = set()
        for (rep, fld), grp in df_exist.groupby(["repeat", "fold"]):
            if len(grp) == len(targets) * len(models) * len(feat_sets):
                completed_keys.add((rep, fld))
        selected_folds = [f for f in selected_folds if (f["repeat"], f["fold"]) not in completed_keys]
        existing_records = df_exist.to_dict("records")
        print(f"Remaining folds to compute: {len(selected_folds)}")
        
    if not selected_folds:
        print("All requested PMEmo folds already computed!")
        return
        
    print(f"Launching evaluation on {len(selected_folds)} folds using {args.workers} workers...")
    all_results = list(existing_records)
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_fold, fold_info, df_p, targets, models, feat_sets): fold_info
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

    print(f"PMEmo baselines run completed! Total records saved: {len(all_results)}")

if __name__ == "__main__":
    main()

