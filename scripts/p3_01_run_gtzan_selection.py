"""Script 01: GTZAN Genre Subset Selection Benchmark.
MOODWAAVE: Repeated hold-out (10 outer repeats, 70% selection / 30% held-out, artist-disjoint stratified group split).
Inner 5x2 grouped CV on 95 subsets. 1-SE selection rule based on mean accuracy.
Evaluation on held-out test set for: Chosen subset, Full-113 V0, Full-113 V1, Lite-71.
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from concurrent.futures import ProcessPoolExecutor

import common as C


def load_gtzan_data():
    """Loads clean GTZAN (929 tracks), maps artists, and creates Full-V1 feature table."""
    df_g = pd.read_csv(C.GTZAN_CSV)
    rem = set(pd.read_csv(C.REMOVED_TRACKS_CSV)["filename"])
    df_clean = df_g[~df_g["filename"].isin(rem)].copy().reset_index(drop=True)
    
    df_art = pd.read_csv(C.ARTIST_MAP_CSV)
    gtzan_art = df_art[df_art["dataset"] == "GTZAN"].set_index("track_id")["artist"]
    df_clean["artist"] = df_clean["filename"].map(gtzan_art)
    
    # Load V1 tonnetz
    df_v1 = pd.read_csv(C.GTZAN_V1_CSV)
    df_v1_clean = df_v1[~df_v1["filename"].isin(rem)].copy().reset_index(drop=True)
    
    tonnetz_cols = [c for c in df_v1.columns if c.startswith("tonnetz")]
    df_full_v1 = df_clean.copy()
    for c in tonnetz_cols:
        df_full_v1[c] = df_v1_clean[c]
        
    return df_clean, df_full_v1


def stratified_artist_split_70_30(df, seed):
    """Partitions artists within each genre into ~70% selection and ~30% test sets."""
    rng = np.random.RandomState(seed)
    art_meta = df.groupby("artist").agg(
        genre=("label", lambda s: s.mode()[0]),
        n_tracks=("label", "count")
    ).reset_index()
    
    test_artists = []
    train_artists = []
    
    for g, group in art_meta.groupby("genre"):
        artists = group["artist"].tolist()
        rng.shuffle(artists)
        target_test_tracks = int(round(group["n_tracks"].sum() * 0.30))
        cur_test_tracks = 0
        cur_test_art = []
        cur_train_art = []
        for a in artists:
            cnt = group.loc[group["artist"] == a, "n_tracks"].values[0]
            if cur_test_tracks + cnt <= target_test_tracks or (cur_test_tracks == 0 and cnt > target_test_tracks):
                cur_test_art.append(a)
                cur_test_tracks += cnt
            else:
                cur_train_art.append(a)
        test_artists.extend(cur_test_art)
        train_artists.extend(cur_train_art)
        
    test_idx = df.index[df["artist"].isin(test_artists)].tolist()
    train_idx = df.index[df["artist"].isin(train_artists)].tolist()
    return train_idx, test_idx


def run_gtzan_outer_repeats(n_repeats=10, max_workers=6):
    print("=" * 80)
    print("MOODWAAVE - GTZAN (GENRE) SUBSET SELECTION BENCHMARK")
    print(f"10 Outer Repeats (70% selection / 30% held-out, artist-disjoint stratified group split)")
    print(f"Inner 5x2 CV across 95 candidate subsets using max_workers={max_workers}")
    print("=" * 80)
    
    df_clean, df_full_v1 = load_gtzan_data()
    all_subsets = C.build_candidate_subsets()
    
    lite_71_subset = [s for s in all_subsets if set(s["non_tonnetz_groups"]) == {"MFCC", "SpecContrast", "Temporal"} and s["tonnetz_opt"] == "none"][0]
    full_v0_subset = [s for s in all_subsets if len(s["non_tonnetz_groups"]) == 5 and s["tonnetz_opt"] == "V0"][0]
    full_v1_subset = [s for s in all_subsets if len(s["non_tonnetz_groups"]) == 5 and s["tonnetz_opt"] == "V1"][0]
    
    results_path = os.path.join(C.OUT, "gtzan_outer_results.csv")
    inner_scores_path = os.path.join(C.OUT, "gtzan_inner_cv_scores.csv")
    
    completed_repeats = set()
    if os.path.exists(results_path):
        df_existing = pd.read_csv(results_path)
        completed_repeats = set(df_existing["repeat"].tolist())
        print(f"Found {len(completed_repeats)} existing completed repeats in {results_path}.")
        
    for rep in range(n_repeats):
        seed = 42 + rep
        if rep in completed_repeats:
            print(f"\n>>> Repeat {rep + 1}/{n_repeats} (seed={seed}) already completed. Skipping.")
            continue
            
        print(f"\n>>> Starting Repeat {rep + 1}/{n_repeats} (seed={seed})...")
        t_rep_start = time.time()
        
        # 1. Outer 70/30 artist-disjoint stratified group split
        train_idx, test_idx = stratified_artist_split_70_30(df_clean, seed)
        
        df_sel = df_clean.iloc[train_idx].copy().reset_index(drop=True)
        df_sel_v1 = df_full_v1.iloc[train_idx].copy().reset_index(drop=True)
        
        df_test = df_clean.iloc[test_idx].copy().reset_index(drop=True)
        df_test_v1 = df_full_v1.iloc[test_idx].copy().reset_index(drop=True)
        
        # Verify strict artist disjointness
        overlap = set(df_sel["artist"]) & set(df_test["artist"])
        assert len(overlap) == 0, f"Artist overlap detected in Repeat {rep}: {overlap}"
        
        print(f"  Selection set: {len(df_sel)} tracks ({len(set(df_sel['artist']))} artists)")
        print(f"  Held-out test: {len(df_test)} tracks ({len(set(df_test['artist']))} artists)")
        
        # 2. Inner 5x2 grouped CV on selection set
        sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed * 100 + 1)
        sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed * 100 + 2)
        inner_folds = []
        for tr, val in sgkf1.split(df_sel, df_sel["label"], df_sel["artist"]):
            inner_folds.append((tr, val))
        for tr, val in sgkf2.split(df_sel, df_sel["label"], df_sel["artist"]):
            inner_folds.append((tr, val))
            
        # 3. Build tasks for 95 subsets
        tasks = []
        y_sel = df_sel["label"].values
        
        for s in all_subsets:
            feats = s["features"]
            t_opt = s["tonnetz_opt"]
            source_df = df_sel_v1 if t_opt == "V1" else df_sel
            X = source_df[feats].values
            tasks.append({
                "subset_id": s["subset_id"],
                "name": s["name"],
                "features": feats,
                "tonnetz_opt": t_opt,
                "X": X,
                "y": y_sel,
                "inner_folds": inner_folds
            })
            
        print(f"  Evaluating 95 subsets across 10 inner folds...")
        t_cv_start = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            cv_results = list(executor.map(C.evaluate_gtzan_subset_worker, tasks))
        t_cv_end = time.time()
        print(f"  Inner CV completed in {t_cv_end - t_cv_start:.2f}s.")
        
        # Merge subset metadata with inner CV scores
        subset_records = []
        inner_score_rows = []
        for s, r in zip(all_subsets, cv_results):
            rec = {
                "subset_id": s["subset_id"],
                "name": s["name"],
                "n_features": s["n_features"],
                "tonnetz_opt": s["tonnetz_opt"],
                "cost_ms": s["gtzan_cost_ms"],
                "mean_score": r["mean_score"],
                "sd_score": r["sd_score"],
                "se_score": r["se_score"]
            }
            subset_records.append(rec)
            inner_score_rows.append({
                "repeat": rep,
                "seed": seed,
                "subset_id": s["subset_id"],
                "name": s["name"],
                "n_features": s["n_features"],
                "tonnetz_opt": s["tonnetz_opt"],
                "cost_ms": s["gtzan_cost_ms"],
                "inner_mean_acc": r["mean_score"],
                "inner_se_acc": r["se_score"]
            })
            
        # Append inner scores to CSV
        df_inner_batch = pd.DataFrame(inner_score_rows)
        header = not os.path.exists(inner_scores_path)
        df_inner_batch.to_csv(inner_scores_path, mode="a", header=header, index=False)
        
        # 4. Apply 1-SE Selection Rule
        chosen, best, threshold, eligible = C.select_best_1se(
            subset_records, score_col="mean_score", se_col="se_score", cost_col="cost_ms"
        )
        print(f"  Best subset: {best['name']} (Mean Acc={best['mean_score']:.4f} ± {best['se_score']:.4f}, Cost={best['cost_ms']:.2f}ms)")
        print(f"  1-SE Threshold: {threshold:.4f} ({len(eligible)} eligible subsets)")
        print(f"  Chosen subset: {chosen['name']} (Cost={chosen['cost_ms']:.2f}ms, Mean Acc={chosen['mean_score']:.4f}, Feats={chosen['n_features']})")
        
        # 5. Evaluate on 30% Held-Out Test Set
        models_to_eval = [
            ("chosen", chosen["name"], chosen["n_features"], chosen["tonnetz_opt"], all_subsets[chosen["subset_id"]]["features"]),
            ("full_v0", full_v0_subset["name"], full_v0_subset["n_features"], full_v0_subset["tonnetz_opt"], full_v0_subset["features"]),
            ("full_v1", full_v1_subset["name"], full_v1_subset["n_features"], full_v1_subset["tonnetz_opt"], full_v1_subset["features"]),
            ("lite_71", lite_71_subset["name"], lite_71_subset["n_features"], lite_71_subset["tonnetz_opt"], lite_71_subset["features"])
        ]
        
        eval_metrics = {}
        y_test = df_test["label"].values
        for tag, name, n_f, t_opt, feats in models_to_eval:
            src_train = df_sel_v1 if t_opt == "V1" else df_sel
            src_test = df_test_v1 if t_opt == "V1" else df_test
            
            X_tr = src_train[feats].values
            X_te = src_test[feats].values
            
            clf = LGBMClassifier(**C.LGBM_GENRE_PARAMS)
            clf.fit(X_tr, y_sel)
            preds = clf.predict(X_te)
            
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average="macro"))
            
            eval_metrics[f"test_acc_{tag}"] = acc
            eval_metrics[f"test_f1_{tag}"] = f1
            
        print(f"  Test Held-Out Acc -> Chosen: {eval_metrics['test_acc_chosen']:.4f} | Full-V0: {eval_metrics['test_acc_full_v0']:.4f} | Full-V1: {eval_metrics['test_acc_full_v1']:.4f} | Lite-71: {eval_metrics['test_acc_lite_71']:.4f}")
        
        # 6. Record outer repeat summary
        row = {
            "repeat": rep,
            "seed": seed,
            "n_train": len(df_sel),
            "n_test": len(df_test),
            "chosen_subset_id": chosen["subset_id"],
            "chosen_name": chosen["name"],
            "chosen_n_features": chosen["n_features"],
            "chosen_tonnetz_opt": chosen["tonnetz_opt"],
            "chosen_cost_ms": chosen["cost_ms"],
            "chosen_inner_mean_acc": chosen["mean_score"],
            "chosen_inner_se_acc": chosen["se_score"],
            "best_subset_id": best["subset_id"],
            "best_name": best["name"],
            "best_inner_mean_acc": best["mean_score"],
            "best_inner_se_acc": best["se_score"],
            "diff_acc_chosen_vs_full_v0": eval_metrics["test_acc_chosen"] - eval_metrics["test_acc_full_v0"],
            "diff_acc_chosen_vs_full_v1": eval_metrics["test_acc_chosen"] - eval_metrics["test_acc_full_v1"],
            "diff_acc_lite_vs_full_v0": eval_metrics["test_acc_lite_71"] - eval_metrics["test_acc_full_v0"],
            "diff_acc_lite_vs_full_v1": eval_metrics["test_acc_lite_71"] - eval_metrics["test_acc_full_v1"],
            **eval_metrics
        }
        
        df_row = pd.DataFrame([row])
        header = not os.path.exists(results_path)
        df_row.to_csv(results_path, mode="a", header=header, index=False)
        print(f"  Repeat {rep + 1} completed in {time.time() - t_rep_start:.2f}s and saved to {results_path}.")
        
    print("\nAll 10 GTZAN outer repeats completed successfully!")


if __name__ == "__main__":
    run_gtzan_outer_repeats(n_repeats=10, max_workers=6)

