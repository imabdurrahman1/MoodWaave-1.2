"""
experiments/rev3_a_stats/01_run_gtzan_r3a.py
TASK R3-A: GTZAN nested CV with ARTIST-GROUPED inner tuning.

Outer CV: P1 fault-removed artist-disjoint 5-fold x 3 repeats = 15 folds.
Inner CV: StratifiedGroupKFold(n_splits=3) grouped by artist, used in
          RandomizedSearchCV(n_iter=8), scoring='accuracy'.
Models: LightGBM, XGBoost, Random Forest, SVM-RBF, Logistic Regression.
Feature sets: Full-113, Lite-71.

Writes results incrementally to gtzan_r3a_folds.csv after each fold.
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Suppress harmless sklearn feature name warnings when converting to numpy
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3a_common as common

MODELS = [
    "LightGBM",
    "Random Forest",
    "XGBoost",
    "SVM-RBF",
    "Logistic Regression",
]
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
    df_clean: pd.DataFrame,
) -> list:
    """
    Runs all models x feat_sets for one outer GTZAN fold with artist-grouped inner CV.
    """
    results = []
    train_df = df_clean.loc[train_tracks]
    test_df = df_clean.loc[test_tracks]

    y_train = train_df["label"].values
    y_test = test_df["label"].values
    groups_train = train_df["artist_id"].values   # artist groups for inner CV

    inner_cv = common.make_inner_cv_genre(n_splits=N_INNER_FOLDS)

    fold_start = time.time()
    print(
        f"--> GTZAN Fold (rep={repeat}, fold={fold_idx}): "
        f"{len(train_tracks)} train, {len(test_tracks)} test | "
        f"{pd.Series(groups_train).nunique()} training artists",
        flush=True
    )

    for feat_set_name, feats in common.FEATURE_SETS.items():
        X_tr = train_df[feats].values
        X_te = test_df[feats].values

        for model_name in MODELS:
            t0 = time.time()

            # XGBoost needs integer labels
            if model_name == "XGBoost":
                le = LabelEncoder()
                y_tr_fit = le.fit_transform(y_train)
                y_te_eval = le.transform(y_test)
            else:
                y_tr_fit = y_train
                y_te_eval = y_test

            base_model = common.get_genre_base_model(model_name, seed=42)
            grid = common.GENRE_PARAM_GRIDS[model_name]
            inner_seed = 42 + repeat * 100 + fold_idx

            # Artist-grouped inner RandomizedSearchCV
            # groups must match X_tr rows; StratifiedGroupKFold needs y and groups
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=grid,
                n_iter=N_ITER,
                cv=inner_cv,
                scoring="accuracy",
                random_state=inner_seed,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(X_tr, y_tr_fit, groups=groups_train)

            preds = search.best_estimator_.predict(X_te)
            acc = accuracy_score(y_te_eval, preds)
            f1 = f1_score(y_te_eval, preds, average="macro")
            elapsed = time.time() - t0

            print(
                f"    [rep={repeat}, fold={fold_idx}] {model_name:20s} "
                f"({feat_set_name:8s}): Acc={acc:.4f}, F1={f1:.4f} "
                f"in {elapsed:.1f}s | best={json.dumps(search.best_params_)}",
                flush=True
            )

            results.append({
                "dataset": "GTZAN",
                "task": "genre",
                "repeat": repeat,
                "fold": fold_idx,
                "model": model_name,
                "feature_set": feat_set_name,
                "accuracy": float(acc),
                "macro_f1": float(f1),
                "best_params": json.dumps(search.best_params_),
                "fit_time_sec": round(elapsed, 2),
                "inner_cv": f"StratifiedGroupKFold(n_splits={N_INNER_FOLDS})",
                "n_train": len(train_tracks),
                "n_test": len(test_tracks),
            })

    print(
        f"<-- GTZAN Fold (rep={repeat}, fold={fold_idx}) done in "
        f"{time.time() - fold_start:.1f}s",
        flush=True
    )
    return results


def main():
    common.log_environment(OUT_DIR)

    df_clean, all_folds = common.load_gtzan(SPLITS_JSON)
    print(f"Loaded GTZAN clean: {len(df_clean)} tracks")
    print(f"Total outer folds available: {len(all_folds)}")

    # Use all 15 folds (3 repeats x 5 folds)
    selected_folds = all_folds
    print(f"Will run {len(selected_folds)} outer folds.")

    results_csv = os.path.join(OUT_DIR, "gtzan_r3a_folds.csv")
    all_results = []
    completed_keys = set()

    if os.path.exists(results_csv):
        df_exist = pd.read_csv(results_csv)
        print(f"Resuming: found {len(df_exist)} existing records.")
        n_expected = len(MODELS) * len(common.FEATURE_SETS)
        for (rep, fld), grp in df_exist.groupby(["repeat", "fold"]):
            if len(grp) >= n_expected:
                completed_keys.add((int(rep), int(fld)))
        all_results = df_exist.to_dict("records")
        selected_folds = [
            f for f in selected_folds
            if (f["repeat"], f["fold"]) not in completed_keys
        ]
        print(f"Remaining folds: {len(selected_folds)}")

    if not selected_folds:
        print("All folds already done.")
        return

    print(
        f"Launching {len(selected_folds)} folds using ProcessPoolExecutor "
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
                df_clean,
            ): fi
            for fi in selected_folds
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

    print(f"GTZAN R3-A complete. {len(all_results)} records in {results_csv}")


if __name__ == "__main__":
    main()

