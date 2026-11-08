"""Script 02: Evaluate GTZAN across Split Protocols (MOODWAAVE).
Evaluates:
(a) All 999 tracks, 5x3 repeated stratified random CV
(b) Fault-removed (929 tracks), 5x3 repeated stratified random CV
(c) Fault-removed + artist-disjoint (929 tracks), 5x3 repeated StratifiedGroupKFold
(d) Kereliuk & Sturm (2015) Fixed Fault-Filtered split (train 443, valid 196, test 290) over 5 seeds

Models:
1. LightGBM Full-113
2. LightGBM Lite-71
3. Random Forest (100 trees)
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import common as C


def evaluate_gtzan_protocols():
    print("=" * 80)
    print("MOODWAAVE - GTZAN SPLIT PROTOCOL BENCHMARK")
    print("=" * 80)

    # Load splits
    splits_path = os.path.join(C.OUT, "splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)["gtzan"]

    # Load features
    df_gtzan = pd.read_csv(C.GTZAN_CSV)
    df_artist = pd.read_csv(os.path.join(C.OUT, "artist_map.csv"))
    gtzan_artist_sub = df_artist[df_artist["dataset"] == "GTZAN"]
    artist_dict = dict(zip(gtzan_artist_sub["track_id"], gtzan_artist_sub["artist"]))

    # Index by filename for fast fold extraction
    df_gtzan.set_index("filename", inplace=True)
    all_genres = sorted(df_gtzan["label"].unique())
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}

    protocols = [
        ("all_999_stratified_random", "All 999 Stratified Random (Baseline)", splits["all_999_stratified_random"]),
        ("fault_removed_stratified_random", "Fault-Removed Stratified Random", splits["fault_removed_stratified_random"]),
        ("fault_removed_artist_grouped", "Fault-Removed + Artist-Disjoint (5x3 Grouped CV)", splits["fault_removed_artist_grouped"])
    ]

    fold_results = []
    summary_results = []

    # Store baseline means for computing drops
    baseline_acc = {}
    baseline_f1 = {}

    for proto_key, proto_name, fold_list in protocols:
        print(f"\n--- Running Protocol: {proto_name} ({len(fold_list)} folds) ---")
        
        # Determine unique tracks and artists in this evaluation set
        sample_tracks = fold_list[0]["train_tracks"] + fold_list[0]["test_tracks"]
        unique_tracks = set(sample_tracks)
        n_tracks = len(unique_tracks)
        n_artists = len(set(artist_dict[t] for t in unique_tracks if t in artist_dict))

        # Model containers
        model_metrics = {
            "LightGBM Full-113": {"acc": [], "f1": []},
            "LightGBM Lite-71 (MOODWAAVE-Lite)": {"acc": [], "f1": []},
            "Random Forest (100 trees)": {"acc": [], "f1": []}
        }

        t0 = time.time()
        for f_idx, fold_data in enumerate(fold_list):
            rep = fold_data["repeat"]
            fold = fold_data["fold"]
            tr_tracks = fold_data["train_tracks"]
            te_tracks = fold_data["test_tracks"]

            df_tr = df_gtzan.loc[tr_tracks]
            df_te = df_gtzan.loc[te_tracks]

            y_tr = df_tr["label"].map(genre_to_idx).values
            y_te = df_te["label"].map(genre_to_idx).values

            # Feature matrices
            X_tr_full = df_tr[C.FULL_FEATURES]
            X_te_full = df_te[C.FULL_FEATURES]
            X_tr_lite = df_tr[C.LITE_FEATURES]
            X_te_lite = df_te[C.LITE_FEATURES]

            # 1. LightGBM Full-113
            m_full = LGBMClassifier(**C.LGBM_GENRE_PARAMS).fit(X_tr_full, y_tr)
            y_pred_full = m_full.predict(X_te_full)
            acc_full = accuracy_score(y_te, y_pred_full)
            f1_full = f1_score(y_te, y_pred_full, average="macro")
            model_metrics["LightGBM Full-113"]["acc"].append(acc_full)
            model_metrics["LightGBM Full-113"]["f1"].append(f1_full)

            # 2. LightGBM Lite-71
            m_lite = LGBMClassifier(**C.LGBM_GENRE_PARAMS).fit(X_tr_lite, y_tr)
            y_pred_lite = m_lite.predict(X_te_lite)
            acc_lite = accuracy_score(y_te, y_pred_lite)
            f1_lite = f1_score(y_te, y_pred_lite, average="macro")
            model_metrics["LightGBM Lite-71 (MOODWAAVE-Lite)"]["acc"].append(acc_lite)
            model_metrics["LightGBM Lite-71 (MOODWAAVE-Lite)"]["f1"].append(f1_lite)

            # 3. Random Forest (100 trees)
            m_rf = RandomForestClassifier(**C.RF_GENRE_PARAMS).fit(X_tr_full, y_tr)
            y_pred_rf = m_rf.predict(X_te_full)
            acc_rf = accuracy_score(y_te, y_pred_rf)
            f1_rf = f1_score(y_te, y_pred_rf, average="macro")
            model_metrics["Random Forest (100 trees)"]["acc"].append(acc_rf)
            model_metrics["Random Forest (100 trees)"]["f1"].append(f1_rf)

            fold_results.append({
                "protocol": proto_key,
                "protocol_name": proto_name,
                "repeat": rep,
                "fold": fold,
                "acc_full": acc_full, "f1_full": f1_full,
                "acc_lite": acc_lite, "f1_lite": f1_lite,
                "acc_rf": acc_rf, "f1_rf": f1_rf
            })

        print(f"  Completed {len(fold_list)} folds in {time.time() - t0:.1f}s")

        for m_name, mets in model_metrics.items():
            mean_acc = float(np.mean(mets["acc"]))
            std_acc = float(np.std(mets["acc"], ddof=1))
            mean_f1 = float(np.mean(mets["f1"]))
            std_f1 = float(np.std(mets["f1"], ddof=1))

            if proto_key == "all_999_stratified_random":
                baseline_acc[m_name] = mean_acc
                baseline_f1[m_name] = mean_f1
                drop_acc = 0.0
                drop_f1 = 0.0
            else:
                drop_acc = mean_acc - baseline_acc[m_name]
                drop_f1 = mean_f1 - baseline_f1[m_name]

            summary_results.append({
                "protocol_key": proto_key,
                "Protocol": proto_name,
                "Model": m_name,
                "Tracks": n_tracks,
                "Artists": n_artists,
                "Accuracy": f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%",
                "Macro-F1": f"{mean_f1:.4f} ± {std_f1:.4f}",
                "raw_acc_mean": mean_acc,
                "raw_acc_std": std_acc,
                "raw_f1_mean": mean_f1,
                "raw_f1_std": std_f1,
                "Acc Drop (pp)": f"{drop_acc*100:+.2f}%" if proto_key != "all_999_stratified_random" else "—",
                "Macro-F1 Drop": f"{drop_f1:+.4f}" if proto_key != "all_999_stratified_random" else "—"
            })
            print(f"  {m_name:35s}: Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%, Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f} (Drop: {drop_acc*100:+.2f}%)")

    # Evaluate Fixed Kereliuk Split over 5 seeds
    fixed_split = splits["kereliuk_sturm_fixed"]
    print("\n--- Running Protocol: Kereliuk & Sturm (2015) Fixed Split (5 Model Seeds) ---")
    tr_tracks = fixed_split["train_tracks"] + fixed_split["valid_tracks"]  # standard train + val
    te_tracks = fixed_split["test_tracks"]

    df_tr = df_gtzan.loc[tr_tracks]
    df_te = df_gtzan.loc[te_tracks]
    y_tr = df_tr["label"].map(genre_to_idx).values
    y_te = df_te["label"].map(genre_to_idx).values
    X_tr_full = df_tr[C.FULL_FEATURES]
    X_te_full = df_te[C.FULL_FEATURES]
    X_tr_lite = df_tr[C.LITE_FEATURES]
    X_te_lite = df_te[C.LITE_FEATURES]

    fixed_metrics = {
        "LightGBM Full-113": {"acc": [], "f1": []},
        "LightGBM Lite-71 (MOODWAAVE-Lite)": {"acc": [], "f1": []},
        "Random Forest (100 trees)": {"acc": [], "f1": []}
    }

    for s_idx, seed in enumerate([42, 101, 202, 303, 404]):
        params_g_full = dict(C.LGBM_GENRE_PARAMS)
        params_g_full["random_state"] = seed
        m_f = LGBMClassifier(**params_g_full).fit(X_tr_full, y_tr)
        yp_f = m_f.predict(X_te_full)
        fixed_metrics["LightGBM Full-113"]["acc"].append(accuracy_score(y_te, yp_f))
        fixed_metrics["LightGBM Full-113"]["f1"].append(f1_score(y_te, yp_f, average="macro"))

        params_g_lite = dict(C.LGBM_GENRE_PARAMS)
        params_g_lite["random_state"] = seed
        m_l = LGBMClassifier(**params_g_lite).fit(X_tr_lite, y_tr)
        yp_l = m_l.predict(X_te_lite)
        fixed_metrics["LightGBM Lite-71 (MOODWAAVE-Lite)"]["acc"].append(accuracy_score(y_te, yp_l))
        fixed_metrics["LightGBM Lite-71 (MOODWAAVE-Lite)"]["f1"].append(f1_score(y_te, yp_l, average="macro"))

        m_rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X_tr_full, y_tr)
        yp_rf = m_rf.predict(X_te_full)
        fixed_metrics["Random Forest (100 trees)"]["acc"].append(accuracy_score(y_te, yp_rf))
        fixed_metrics["Random Forest (100 trees)"]["f1"].append(f1_score(y_te, yp_rf, average="macro"))

    for m_name, mets in fixed_metrics.items():
        mean_acc = float(np.mean(mets["acc"]))
        std_acc = float(np.std(mets["acc"], ddof=1))
        mean_f1 = float(np.mean(mets["f1"]))
        std_f1 = float(np.std(mets["f1"], ddof=1))
        drop_acc = mean_acc - baseline_acc[m_name]
        drop_f1 = mean_f1 - baseline_f1[m_name]
        summary_results.append({
            "protocol_key": "kereliuk_sturm_fixed",
            "Protocol": "Kereliuk & Sturm (2015) Fixed Split (5 Seeds)",
            "Model": m_name,
            "Tracks": len(tr_tracks) + len(te_tracks),
            "Artists": len(set(artist_dict[t] for t in tr_tracks + te_tracks if t in artist_dict)),
            "Accuracy": f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%",
            "Macro-F1": f"{mean_f1:.4f} ± {std_f1:.4f}",
            "raw_acc_mean": mean_acc,
            "raw_acc_std": std_acc,
            "raw_f1_mean": mean_f1,
            "raw_f1_std": std_f1,
            "Acc Drop (pp)": f"{drop_acc*100:+.2f}%",
            "Macro-F1 Drop": f"{drop_f1:+.4f}"
        })
        print(f"  {m_name:35s}: Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%, Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f} (Drop: {drop_acc*100:+.2f}%)")

    # Save summary and fold data
    df_summ = pd.DataFrame(summary_results)
    df_summ.to_csv(os.path.join(C.OUT, "part_a_gtzan_summary.csv"), index=False)
    pd.DataFrame(fold_results).to_csv(os.path.join(C.OUT, "part_a_gtzan_folds.csv"), index=False)
    print("\nSaved part_a_gtzan_summary.csv and part_a_gtzan_folds.csv.")


if __name__ == "__main__":
    evaluate_gtzan_protocols()

