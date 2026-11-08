"""Script 03: Evaluate PMEmo across Split Protocols (MOODWAAVE).
Part B:
1. Metadata inspection: tracks per artist, repeated artists/albums, duplicate titles.
2. Leakage analysis: fraction of test tracks whose artist appears in training fold under random 5x5 CV vs artist-grouped 5x5 CV.
3. Model evaluation: LightGBM Full-113, Lite-71, and RF Regressor on:
   - Protocol 1: Random 5x5 CV (baseline)
   - Protocol 2: Artist-Grouped 5x5 CV (leakage-free)
4. Compute R², MAE (Valence and Arousal), and report performance drops.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

import common as C


def nadeau_bengio_t_test(diffs, n_train, n_test):
    """Corrected resampled t-test for repeated K-fold CV (Nadeau & Bengio, 2003)."""
    k = len(diffs)
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)
    if var_diff == 0:
        return mean_diff, 0.0, 1.0
    correction = (1.0 / k) + (n_test / n_train)
    corrected_var = var_diff * correction
    t_stat = mean_diff / np.sqrt(corrected_var)
    df = k - 1
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
    return float(mean_diff), float(t_stat), float(p_val)


def inspect_pmemo_metadata(df_artist, splits):
    print("=" * 80)
    print("MOODWAAVE - PART B.1: PMEMO METADATA & LEAKAGE INSPECTION")
    print("=" * 80)

    pmemo_artist_df = df_artist[df_artist["dataset"] == "PMEmo"].copy()
    n_tracks = len(pmemo_artist_df)
    
    artist_counts = pmemo_artist_df["artist"].value_counts()
    n_unique_artists = len(artist_counts)
    repeated_artists = artist_counts[artist_counts > 1]
    n_repeated_artists = len(repeated_artists)
    n_tracks_in_repeated = int(repeated_artists.sum())
    pct_tracks_repeated = (n_tracks_in_repeated / n_tracks) * 100

    album_counts = pmemo_artist_df["album"].value_counts()
    n_unique_albums = len(album_counts)
    repeated_albums = album_counts[album_counts > 1]

    title_counts = pmemo_artist_df["title"].value_counts()
    n_unique_titles = len(title_counts)
    duplicate_titles = title_counts[title_counts > 1]

    print(f"Total PMEmo tracks: {n_tracks}")
    print(f"Unique artists: {n_unique_artists}")
    print(f"Artists with multiple tracks: {n_repeated_artists} ({n_repeated_artists/n_unique_artists*100:.1f}% of artists)")
    print(f"Tracks by multi-track artists: {n_tracks_in_repeated} ({pct_tracks_repeated:.1f}% of all tracks)")
    print(f"Max tracks for a single artist: {artist_counts.max()} ({artist_counts.idxmax()})")
    print(f"Unique albums: {n_unique_albums} (Repeated albums: {len(repeated_albums)})")
    print(f"Unique titles: {n_unique_titles} (Duplicate titles: {len(duplicate_titles)})")

    # Top 10 artists
    print("\nTop 10 artists in PMEmo by track count:")
    top10 = artist_counts.head(10).to_dict()
    for art, c in top10.items():
        print(f"  - {art}: {c} tracks")

    # Leakage Analysis under Random 5x5 CV vs Artist-Grouped 5x5 CV
    artist_dict = dict(zip(pmemo_artist_df["track_id"].astype(str), pmemo_artist_df["artist"]))

    random_folds = splits["pmemo"]["random_5x5"]
    grouped_folds = splits["pmemo"]["artist_grouped_5x5"]

    rand_leakage_rates = []
    for f in random_folds:
        tr_arts = set(artist_dict[t] for t in f["train_tracks"])
        te_arts = [artist_dict[t] for t in f["test_tracks"]]
        leaked = sum(1 for a in te_arts if a in tr_arts)
        rand_leakage_rates.append(leaked / len(te_arts))

    mean_rand_leak = float(np.mean(rand_leakage_rates))
    std_rand_leak = float(np.std(rand_leakage_rates, ddof=1))
    print(f"\n[Leakage Analysis] Random 5x5 CV:")
    print(f"  Fraction of test tracks with artist in training fold: {mean_rand_leak*100:.2f}% ± {std_rand_leak*100:.2f}% (range: {min(rand_leakage_rates)*100:.1f}% - {max(rand_leakage_rates)*100:.1f}%)")

    grp_leakage_rates = []
    for f in grouped_folds:
        tr_arts = set(artist_dict[t] for t in f["train_tracks"])
        te_arts = [artist_dict[t] for t in f["test_tracks"]]
        leaked = sum(1 for a in te_arts if a in tr_arts)
        grp_leakage_rates.append(leaked / len(te_arts))

    mean_grp_leak = float(np.mean(grp_leakage_rates))
    print(f"\n[Leakage Analysis] Artist-Grouped 5x5 CV:")
    print(f"  Fraction of test tracks with artist in training fold: {mean_grp_leak*100:.2f}% (zero artist overlap guaranteed)")

    metadata_stats = {
        "dataset": "PMEmo",
        "total_tracks": n_tracks,
        "unique_artists": n_unique_artists,
        "repeated_artists_count": n_repeated_artists,
        "tracks_in_repeated_artists": n_tracks_in_repeated,
        "pct_tracks_in_repeated_artists": round(pct_tracks_repeated, 2),
        "unique_albums": n_unique_albums,
        "unique_titles": n_unique_titles,
        "top_10_artists": top10,
        "random_cv_leakage_rate_pct": {
            "mean": round(mean_rand_leak * 100, 2),
            "std": round(std_rand_leak * 100, 2),
            "min": round(min(rand_leakage_rates) * 100, 2),
            "max": round(max(rand_leakage_rates) * 100, 2)
        },
        "grouped_cv_leakage_rate_pct": round(mean_grp_leak * 100, 2)
    }

    with open(os.path.join(C.OUT, "pmemo_artist_stats.json"), "w") as f:
        json.dump(metadata_stats, f, indent=2)

    return metadata_stats, artist_dict


def evaluate_pmemo_protocols(artist_dict, splits):
    print("\n" + "=" * 80)
    print("MOODWAAVE - PART B.2: PMEMO REGRESSION BENCHMARK (RANDOM VS ARTIST-GROUPED)")
    print("=" * 80)

    df_pmemo = pd.read_csv(C.PMEMO_READY_CSV)
    df_pmemo["musicId_str"] = df_pmemo["musicId"].astype(str)
    df_pmemo.set_index("musicId_str", inplace=True)

    protocols = [
        ("pmemo_random_5x5", "Random 5x5 CV (Standard)", splits["pmemo"]["random_5x5"]),
        ("pmemo_artist_grouped_5x5", "Artist-Grouped 5x5 CV (Leakage-Free)", splits["pmemo"]["artist_grouped_5x5"])
    ]

    fold_rows = []
    summary_rows = []

    # Store fold metrics for paired statistical tests
    fold_perf = {}

    for proto_key, proto_name, fold_list in protocols:
        print(f"\n--- Running Protocol: {proto_name} ({len(fold_list)} folds) ---")
        t0 = time.time()
        
        models_data = {
            "LightGBM Full-113": {
                "r2_v": [], "r2_a": [], "r2_mean": [],
                "mae_v": [], "mae_a": [], "mae_mean": []
            },
            "LightGBM Lite-71 (MOODWAAVE-Lite)": {
                "r2_v": [], "r2_a": [], "r2_mean": [],
                "mae_v": [], "mae_a": [], "mae_mean": []
            },
            "Random Forest (100 trees)": {
                "r2_v": [], "r2_a": [], "r2_mean": [],
                "mae_v": [], "mae_a": [], "mae_mean": []
            }
        }

        for f_idx, fold_data in enumerate(fold_list):
            rep = fold_data["repeat"]
            fold = fold_data["fold"]
            tr_tracks = fold_data["train_tracks"]
            te_tracks = fold_data["test_tracks"]

            df_tr = df_pmemo.loc[tr_tracks]
            df_te = df_pmemo.loc[te_tracks]

            y_tr_v = df_tr["Valence(mean)"].values
            y_tr_a = df_tr["Arousal(mean)"].values
            y_te_v = df_te["Valence(mean)"].values
            y_te_a = df_te["Arousal(mean)"].values

            # Features
            X_tr_full = df_tr[C.FULL_FEATURES]
            X_te_full = df_te[C.FULL_FEATURES]
            X_tr_lite = df_tr[C.LITE_FEATURES]
            X_te_lite = df_te[C.LITE_FEATURES]

            # 1. LightGBM Full-113
            m_v_full = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_tr_full, y_tr_v)
            m_a_full = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_tr_full, y_tr_a)
            pred_v_f = m_v_full.predict(X_te_full)
            pred_a_f = m_a_full.predict(X_te_full)

            r2_v_f = r2_score(y_te_v, pred_v_f)
            r2_a_f = r2_score(y_te_a, pred_a_f)
            r2_m_f = (r2_v_f + r2_a_f) / 2.0
            mae_v_f = mean_absolute_error(y_te_v, pred_v_f)
            mae_a_f = mean_absolute_error(y_te_a, pred_a_f)
            mae_m_f = (mae_v_f + mae_a_f) / 2.0

            models_data["LightGBM Full-113"]["r2_v"].append(r2_v_f)
            models_data["LightGBM Full-113"]["r2_a"].append(r2_a_f)
            models_data["LightGBM Full-113"]["r2_mean"].append(r2_m_f)
            models_data["LightGBM Full-113"]["mae_v"].append(mae_v_f)
            models_data["LightGBM Full-113"]["mae_a"].append(mae_a_f)
            models_data["LightGBM Full-113"]["mae_mean"].append(mae_m_f)

            # 2. LightGBM Lite-71
            m_v_lite = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_tr_lite, y_tr_v)
            m_a_lite = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_tr_lite, y_tr_a)
            pred_v_l = m_v_lite.predict(X_te_lite)
            pred_a_l = m_a_lite.predict(X_te_lite)

            r2_v_l = r2_score(y_te_v, pred_v_l)
            r2_a_l = r2_score(y_te_a, pred_a_l)
            r2_m_l = (r2_v_l + r2_a_l) / 2.0
            mae_v_l = mean_absolute_error(y_te_v, pred_v_l)
            mae_a_l = mean_absolute_error(y_te_a, pred_a_l)
            mae_m_l = (mae_v_l + mae_a_l) / 2.0

            models_data["LightGBM Lite-71 (MOODWAAVE-Lite)"]["r2_v"].append(r2_v_l)
            models_data["LightGBM Lite-71 (MOODWAAVE-Lite)"]["r2_a"].append(r2_a_l)
            models_data["LightGBM Lite-71 (MOODWAAVE-Lite)"]["r2_mean"].append(r2_m_l)
            models_data["LightGBM Lite-71 (MOODWAAVE-Lite)"]["mae_v"].append(mae_v_l)
            models_data["LightGBM Lite-71 (MOODWAAVE-Lite)"]["mae_a"].append(mae_a_l)
            models_data["LightGBM Lite-71 (MOODWAAVE-Lite)"]["mae_mean"].append(mae_m_l)

            # 3. Random Forest (100 trees)
            m_v_rf = RandomForestRegressor(**C.RF_MOOD_PARAMS).fit(X_tr_full, y_tr_v)
            m_a_rf = RandomForestRegressor(**C.RF_MOOD_PARAMS).fit(X_tr_full, y_tr_a)
            pred_v_rf = m_v_rf.predict(X_te_full)
            pred_a_rf = m_a_rf.predict(X_te_full)

            r2_v_rf = r2_score(y_te_v, pred_v_rf)
            r2_a_rf = r2_score(y_te_a, pred_a_rf)
            r2_m_rf = (r2_v_rf + r2_a_rf) / 2.0
            mae_v_rf = mean_absolute_error(y_te_v, pred_v_rf)
            mae_a_rf = mean_absolute_error(y_te_a, pred_a_rf)
            mae_m_rf = (mae_v_rf + mae_a_rf) / 2.0

            models_data["Random Forest (100 trees)"]["r2_v"].append(r2_v_rf)
            models_data["Random Forest (100 trees)"]["r2_a"].append(r2_a_rf)
            models_data["Random Forest (100 trees)"]["r2_mean"].append(r2_m_rf)
            models_data["Random Forest (100 trees)"]["mae_v"].append(mae_v_rf)
            models_data["Random Forest (100 trees)"]["mae_a"].append(mae_a_rf)
            models_data["Random Forest (100 trees)"]["mae_mean"].append(mae_m_rf)

            fold_rows.append({
                "protocol": proto_key,
                "protocol_name": proto_name,
                "repeat": rep,
                "fold": fold,
                # Full-113
                "r2_v_full": r2_v_f, "r2_a_full": r2_a_f, "r2_mean_full": r2_m_f,
                "mae_v_full": mae_v_f, "mae_a_full": mae_a_f, "mae_mean_full": mae_m_f,
                # Lite-71
                "r2_v_lite": r2_v_l, "r2_a_lite": r2_a_l, "r2_mean_lite": r2_m_l,
                "mae_v_lite": mae_v_l, "mae_a_lite": mae_a_l, "mae_mean_lite": mae_m_l,
                # RF
                "r2_v_rf": r2_v_rf, "r2_a_rf": r2_a_rf, "r2_mean_rf": r2_m_rf,
                "mae_v_rf": mae_v_rf, "mae_a_rf": mae_a_rf, "mae_mean_rf": mae_m_rf
            })

        print(f"  Completed {len(fold_list)} folds in {time.time() - t0:.1f}s")
        fold_perf[proto_key] = models_data

        for m_name, mets in models_data.items():
            r2_v_mean, r2_v_std = float(np.mean(mets["r2_v"])), float(np.std(mets["r2_v"], ddof=1))
            r2_a_mean, r2_a_std = float(np.mean(mets["r2_a"])), float(np.std(mets["r2_a"], ddof=1))
            r2_m_mean, r2_m_std = float(np.mean(mets["r2_mean"])), float(np.std(mets["r2_mean"], ddof=1))

            mae_v_mean, mae_v_std = float(np.mean(mets["mae_v"])), float(np.std(mets["mae_v"], ddof=1))
            mae_a_mean, mae_a_std = float(np.mean(mets["mae_a"])), float(np.std(mets["mae_a"], ddof=1))
            mae_m_mean, mae_m_std = float(np.mean(mets["mae_mean"])), float(np.std(mets["mae_mean"], ddof=1))

            summary_rows.append({
                "protocol_key": proto_key,
                "Protocol": proto_name,
                "Model": m_name,
                "Valence R2": f"{r2_v_mean:.4f} ± {r2_v_std:.4f}",
                "Arousal R2": f"{r2_a_mean:.4f} ± {r2_a_std:.4f}",
                "Mean R2": f"{r2_m_mean:.4f} ± {r2_m_std:.4f}",
                "Valence MAE": f"{mae_v_mean:.4f} ± {mae_v_std:.4f}",
                "Arousal MAE": f"{mae_a_mean:.4f} ± {mae_a_std:.4f}",
                "Mean MAE": f"{mae_m_mean:.4f} ± {mae_m_std:.4f}",
                "raw_r2_v_mean": r2_v_mean, "raw_r2_v_std": r2_v_std,
                "raw_r2_a_mean": r2_a_mean, "raw_r2_a_std": r2_a_std,
                "raw_r2_m_mean": r2_m_mean, "raw_r2_m_std": r2_m_std,
                "raw_mae_v_mean": mae_v_mean, "raw_mae_v_std": mae_v_std,
                "raw_mae_a_mean": mae_a_mean, "raw_mae_a_std": mae_a_std,
                "raw_mae_m_mean": mae_m_mean, "raw_mae_m_std": mae_m_std
            })
            print(f"  {m_name:35s}: Mean R² = {r2_m_mean:.4f} ± {r2_m_std:.4f} (V: {r2_v_mean:.4f}, A: {r2_a_mean:.4f}) | Mean MAE = {mae_m_mean:.4f} ± {mae_m_std:.4f}")

    # Compute Paired Differences (Grouped vs Random)
    stat_comparisons = {}
    print("\n--- Paired Performance Differences: Artist-Grouped vs Random 5x5 CV ---")
    for m_name in ["LightGBM Full-113", "LightGBM Lite-71 (MOODWAAVE-Lite)", "Random Forest (100 trees)"]:
        stat_comparisons[m_name] = {}
        for metric_key in ["r2_v", "r2_a", "r2_mean", "mae_v", "mae_a", "mae_mean"]:
            rand_vals = np.array(fold_perf["pmemo_random_5x5"][m_name][metric_key])
            grp_vals = np.array(fold_perf["pmemo_artist_grouped_5x5"][m_name][metric_key])
            diffs = grp_vals - rand_vals  # negative diff in R2 means drop; positive diff in MAE means degradation
            mean_d, t_stat, p_val = nadeau_bengio_t_test(diffs, n_train=int(767 * 0.8), n_test=int(767 * 0.2))
            stat_comparisons[m_name][metric_key] = {
                "mean_diff": round(mean_d, 4),
                "t_stat": round(t_stat, 3),
                "p_val": float(f"{p_val:.4e}")
            }
            if metric_key in ["r2_mean", "r2_v", "r2_a"]:
                print(f"  [{m_name}] {metric_key.upper()} diff: {mean_d:+.4f} (t={t_stat:.2f}, p={p_val:.4f})")

    with open(os.path.join(C.OUT, "pmemo_paired_significance.json"), "w") as f:
        json.dump(stat_comparisons, f, indent=2)

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(os.path.join(C.OUT, "part_b_pmemo_folds.csv"), index=False)
    df_summ = pd.DataFrame(summary_rows)
    df_summ.to_csv(os.path.join(C.OUT, "part_b_pmemo_summary.csv"), index=False)
    print("\nSaved part_b_pmemo_summary.csv, part_b_pmemo_folds.csv, and pmemo_paired_significance.json.")


def main():
    splits_path = os.path.join(C.OUT, "splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    df_artist = pd.read_csv(os.path.join(C.OUT, "artist_map.csv"))
    _, artist_dict = inspect_pmemo_metadata(df_artist, splits)
    evaluate_pmemo_protocols(artist_dict, splits)


if __name__ == "__main__":
    main()

