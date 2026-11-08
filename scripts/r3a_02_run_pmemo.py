"""
experiments/rev3_a_stats/02_run_pmemo_r3a.py
TASK R3-A: PMEmo nested CV with ARTIST-GROUPED inner tuning.

Outer CV: P1 artist-grouped 5x2 folds = 10 folds (repeats 0, 1).
Inner CV: GroupKFold(n_splits=3) grouped by artist, used in
          RandomizedSearchCV(n_iter=8), scoring='r2'.
Models: LightGBM, XGBoost, Random Forest, SVR-RBF, Ridge.
Feature sets: Full-113, Lite-71.
Targets: Valence, Arousal.

Writes results incrementally to pmemo_r3a_folds.csv after each fold.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

# Suppress harmless sklearn feature name warnings when converting to numpy
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3a_common as common

MODELS = ["LightGBM", "Random Forest", "XGBoost", "SVR-RBF", "Ridge"]
TARGETS = [("valence", "Valence(mean)"), ("arousal", "Arousal(mean)")]
N_INNER_FOLDS = 3
N_ITER = 8
OUT_DIR = common.BASE_DIR
SPLITS_JSON = os.path.join(
    common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "splits.json"
)


def evaluate_one_fold(
    repeat: int,
    fold_idx: int,
    train_tracks: list,
    test_tracks: list,
    df_pmemo: pd.DataFrame,
) -> list:
    """
    Runs all targets x models x feat_sets for one outer PMEmo fold with
    artist-grouped inner CV.
    """
    results = []
    train_df = df_pmemo.loc[train_tracks]
    test_df = df_pmemo.loc[test_tracks]
    groups_train = train_df["artist_id"].values

    inner_cv = common.make_inner_cv_regression(n_splits=N_INNER_FOLDS)

    fold_start = time.time()
    print(
        f"--> PMEmo Fold (rep={repeat}, fold={fold_idx}): "
        f"{len(train_tracks)} train, {len(test_tracks)} test | "
        f"{pd.Series(groups_train).nunique()} training artists",
        flush=True
    )

    for task_label, target_col in TARGETS:
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        target_offset = 0 if task_label == "valence" else 500

        for feat_set_name, feats in common.FEATURE_SETS.items():
            X_tr = train_df[feats].values
            X_te = test_df[feats].values

            for model_name in MODELS:
                t0 = time.time()
                base_model = common.get_va_base_model(model_name, seed=42)
                grid = common.VA_PARAM_GRIDS[model_name]
                inner_seed = 42 + target_offset + repeat * 100 + fold_idx

                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=grid,
                    n_iter=N_ITER,
                    cv=inner_cv,
                    scoring="r2",
                    random_state=inner_seed,
                    n_jobs=1,
                    error_score="raise",
                )
                search.fit(X_tr, y_train, groups=groups_train)

                preds = search.best_estimator_.predict(X_te)
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                elapsed = time.time() - t0

                print(
                    f"    [{task_label:7s} rep={repeat}, fold={fold_idx}] "
                    f"{model_name:20s} ({feat_set_name:8s}): "
                    f"R2={r2:.4f}, MAE={mae:.4f} in {elapsed:.1f}s",
                    flush=True
                )

                results.append({
                    "dataset": "PMEmo",
                    "task": task_label,
                    "repeat": repeat,
                    "fold": fold_idx,
                    "model": model_name,
                    "feature_set": feat_set_name,
                    "r2": float(r2),
                    "mae": float(mae),
                    "best_params": json.dumps(search.best_params_),
                    "fit_time_sec": round(elapsed, 2),
                    "inner_cv": f"GroupKFold(n_splits={N_INNER_FOLDS})",
                    "n_train": len(train_tracks),
                    "n_test": len(test_tracks),
                })

    print(
        f"<-- PMEmo Fold (rep={repeat}, fold={fold_idx}) done in "
        f"{time.time() - fold_start:.1f}s",
        flush=True
    )
    return results


def main():
    common.log_environment(OUT_DIR)

    df_pmemo, all_folds = common.load_pmemo(SPLITS_JSON)
    print(f"Loaded PMEmo: {len(df_pmemo)} tracks")
    print(f"Total outer folds: {len(all_folds)}")

    results_csv = os.path.join(OUT_DIR, "pmemo_r3a_folds.csv")
    all_results = []
    completed_keys = set()

    if os.path.exists(results_csv):
        df_exist = pd.read_csv(results_csv)
        print(f"Resuming: found {len(df_exist)} existing records.")
        n_expected = len(TARGETS) * len(MODELS) * len(common.FEATURE_SETS)
        for (rep, fld), grp in df_exist.groupby(["repeat", "fold"]):
            if len(grp) >= n_expected:
                completed_keys.add((int(rep), int(fld)))
        all_results = df_exist.to_dict("records")
        remaining_folds = [
            f for f in all_folds
            if (f["repeat"], f["fold"]) not in completed_keys
        ]
        print(f"Remaining folds: {len(remaining_folds)}")
    else:
        remaining_folds = all_folds

    if not remaining_folds:
        print("All folds already done.")
        return

    print(
        f"Launching {len(remaining_folds)} folds using ProcessPoolExecutor "
        f"(max_workers=5)...",
        flush=True
    )

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                evaluate_one_fold,
                fi["repeat"],
                fi["fold"],
                fi["train_tracks"],
                fi["test_tracks"],
                df_pmemo,
            ): fi
            for fi in remaining_folds
        }
        for fut in as_completed(futures):
            fi = futures[fut]
            try:
                fold_results = fut.result()
                all_results.extend(fold_results)
                df_out = pd.DataFrame(all_results)
                df_out.to_csv(results_csv, index=False)
                print(
                    f"[SAVED] {len(all_results)} records total "
                    f"(fold rep={fi['repeat']},fold={fi['fold']} done)",
                    flush=True
                )
            except Exception as exc:
                print(
                    f"ERROR: fold rep={fi['repeat']},fold={fi['fold']}: {exc}",
                    flush=True
                )
                raise

    print(f"PMEmo R3-A complete. {len(all_results)} records in {results_csv}")


if __name__ == "__main__":
    main()

