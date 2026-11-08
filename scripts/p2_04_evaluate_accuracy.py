"""Step 4: Evaluate LightGBM on Full-113(V0), Full-113(V1), Lite-71, and Lite+V1(83).
Uses P1 artist-disjoint evaluation protocols:
- GTZAN: 5x3 repeated StratifiedGroupKFold (fault-removed, 929 tracks, 286 artists)
- PMEmo: 5x5 repeated GroupKFold (767 tracks, 400 artists)
Models:
1. Full-113 (V0 Baseline)
2. Full-113 (V1 Shared Tonnetz)
3. MOODWAAVE-Lite (71 features)
4. MOODWAAVE-Lite + V1 (83 features)
Computes:
- Accuracy, Macro-F1 (GTZAN)
- R², MAE for Valence, Arousal, Mean (PMEmo)
- Paired differences using Nadeau-Bengio corrected resampled t-test on identical folds.
Outputs:
- accuracy_benchmark_summary.csv
- accuracy_benchmark_folds.csv
- statistical_tests.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
from scipy import stats

import common as C


def nadeau_bengio_t_test(diffs, n_train, n_test):
    """Corrected resampled t-test for repeated K-fold CV (Nadeau & Bengio, 2003)."""
    k = len(diffs)
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1))
    if var_diff == 0:
        return mean_diff, 0.0, 1.0
    correction = (1.0 / k) + (n_test / n_train)
    corrected_var = var_diff * correction
    t_stat = mean_diff / np.sqrt(corrected_var)
    df = k - 1
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
    return float(mean_diff), float(t_stat), float(p_val)


def evaluate_gtzan(splits):
    print("\n" + "=" * 80)
    print("GTZAN GENRE EVALUATION (15 Folds, Fault-Removed + Artist-Disjoint 5x3 Grouped CV)")
    print("=" * 80)

    # 1. Base GTZAN features (V0)
    df_gtz_v0 = pd.read_csv(C.GTZAN_CSV).set_index("filename")
    genres = sorted(df_gtz_v0["label"].unique())
    g2idx = {g: i for i, g in enumerate(genres)}

    # 2. V1 Tonnetz features
    v1_csv = os.path.join(C.OUT, "gtzan_tonnetz_v1.csv")
    if not os.path.exists(v1_csv):
        raise FileNotFoundError(f"{v1_csv} not found. Run 03_extract_v1_features.py first.")
    df_v1 = pd.read_csv(v1_csv).set_index("filename")

    # 3. Build Full-113 (V1) by replacing Tonnetz features
    df_gtz_v1 = df_gtz_v0.copy()
    for col in C.TONNETZ_FEATURES:
        df_gtz_v1[col] = df_v1.loc[df_gtz_v0.index, col]

    # Feature subsets
    v1_tonnetz_cols = C.TONNETZ_FEATURES
    lite_plus_v1_cols = C.LITE_FEATURES + v1_tonnetz_cols

    models_config = [
        ("Full-113 (V0 Baseline)", df_gtz_v0, C.FULL_FEATURES),
        ("Full-113 (V1 Shared Tonnetz)", df_gtz_v1, C.FULL_FEATURES),
        ("MOODWAAVE-Lite (71 features)", df_gtz_v0, C.LITE_FEATURES),
        ("MOODWAAVE-Lite + V1 (83 features)", df_gtz_v1, lite_plus_v1_cols)
    ]

    folds = splits["gtzan"]["fault_removed_artist_grouped"]
    print(f"Loaded {len(folds)} folds.")

    fold_metrics = {name: {"acc": [], "f1": []} for name, _, _ in models_config}
    fold_records = []

    t0 = time.time()
    for f_idx, f_data in enumerate(folds):
        rep = f_data["repeat"]
        fold = f_data["fold"]
        tr_tracks = f_data["train_tracks"]
        te_tracks = f_data["test_tracks"]

        y_tr = df_gtz_v0.loc[tr_tracks, "label"].map(g2idx).values
        y_te = df_gtz_v0.loc[te_tracks, "label"].map(g2idx).values

        rec = {"repeat": rep, "fold": fold}

        for m_name, df_source, feat_cols in models_config:
            X_tr = df_source.loc[tr_tracks, feat_cols]
            X_te = df_source.loc[te_tracks, feat_cols]

            clf = LGBMClassifier(**C.LGBM_GENRE_PARAMS).fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred, average="macro")

            fold_metrics[m_name]["acc"].append(acc)
            fold_metrics[m_name]["f1"].append(f1)
            rec[f"{m_name}|acc"] = acc
            rec[f"{m_name}|f1"] = f1

        fold_records.append(rec)
        if (f_idx + 1) % 5 == 0:
            print(f"  Completed fold {f_idx+1}/15 ({time.time() - t0:.1f}s)", flush=True)

    summary_rows = []
    base_acc_vals = np.array(fold_metrics["Full-113 (V0 Baseline)"]["acc"])
    lite_acc_vals = np.array(fold_metrics["MOODWAAVE-Lite (71 features)"]["acc"])

    stat_tests = {"gtzan": {}}

    for m_name, _, feat_cols in models_config:
        acc_s = np.array(fold_metrics[m_name]["acc"])
        f1_s = np.array(fold_metrics[m_name]["f1"])

        mean_acc, std_acc = float(np.mean(acc_s)), float(np.std(acc_s, ddof=1))
        mean_f1, std_f1 = float(np.mean(f1_s)), float(np.std(f1_s, ddof=1))

        # Paired test vs Full-113 V0
        diff_vs_full = acc_s - base_acc_vals
        d_mean, t_stat, p_val = nadeau_bengio_t_test(diff_vs_full, n_train=int(929*0.8), n_test=int(929*0.2))

        # Paired test vs Lite-71
        diff_vs_lite = acc_s - lite_acc_vals
        d_l_mean, t_l_stat, p_l_val = nadeau_bengio_t_test(diff_vs_lite, n_train=int(929*0.8), n_test=int(929*0.2))

        stat_tests["gtzan"][m_name] = {
            "vs_full_v0": {"mean_diff_pp": round(d_mean * 100, 2), "t_stat": round(t_stat, 3), "p_val": float(f"{p_val:.4e}")},
            "vs_lite_71": {"mean_diff_pp": round(d_l_mean * 100, 2), "t_stat": round(t_l_stat, 3), "p_val": float(f"{p_l_val:.4e}")}
        }

        summary_rows.append({
            "dataset": "GTZAN",
            "model": m_name,
            "features_count": len(feat_cols),
            "accuracy": f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%",
            "macro_f1": f"{mean_f1:.4f} ± {std_f1:.4f}",
            "raw_acc_mean": mean_acc,
            "raw_acc_std": std_acc,
            "raw_f1_mean": mean_f1,
            "raw_f1_std": std_f1,
            "diff_vs_full_pp": f"{d_mean*100:+.2f}%",
            "p_val_vs_full": f"{p_val:.4f}",
            "diff_vs_lite_pp": f"{d_l_mean*100:+.2f}%",
            "p_val_vs_lite": f"{p_l_val:.4f}"
        })
        print(f"  {m_name:35s}: Acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%, F1 = {mean_f1:.4f} (Δ vs Full: {d_mean*100:+.2f} pp, p={p_val:.4f})")

    return summary_rows, fold_records, stat_tests, fold_metrics


def evaluate_pmemo(splits, stat_tests):
    print("\n" + "=" * 80)
    print("PMEMO MOOD EVALUATION (25 Folds, Leakage-Free Artist-Grouped 5x5 CV)")
    print("=" * 80)

    # 1. Base PMEmo features (V0)
    df_pme_v0 = pd.read_csv(C.PMEMO_CSV)
    df_pme_v0["musicId_str"] = df_pme_v0["musicId"].astype(str)
    df_pme_v0.set_index("musicId_str", inplace=True)

    # 2. V1 Tonnetz features
    v1_csv = os.path.join(C.OUT, "pmemo_tonnetz_v1.csv")
    if not os.path.exists(v1_csv):
        raise FileNotFoundError(f"{v1_csv} not found. Run 03_extract_v1_features.py first.")
    df_v1 = pd.read_csv(v1_csv)
    df_v1["musicId_str"] = df_v1["musicId"].astype(str)
    df_v1.set_index("musicId_str", inplace=True)

    # 3. Build Full-113 (V1)
    df_pme_v1 = df_pme_v0.copy()
    for col in C.TONNETZ_FEATURES:
        df_pme_v1[col] = df_v1.loc[df_pme_v0.index, col]

    lite_plus_v1_cols = C.LITE_FEATURES + C.TONNETZ_FEATURES

    models_config = [
        ("Full-113 (V0 Baseline)", df_pme_v0, C.FULL_FEATURES),
        ("Full-113 (V1 Shared Tonnetz)", df_pme_v1, C.FULL_FEATURES),
        ("MOODWAAVE-Lite (71 features)", df_pme_v0, C.LITE_FEATURES),
        ("MOODWAAVE-Lite + V1 (83 features)", df_pme_v1, lite_plus_v1_cols)
    ]

    folds = splits["pmemo"]["artist_grouped_5x5"]
    print(f"Loaded {len(folds)} folds.")

    fold_metrics = {name: {"r2_v": [], "r2_a": [], "r2_m": [], "mae_v": [], "mae_a": [], "mae_m": []} for name, _, _ in models_config}
    fold_records = []

    t0 = time.time()
    for f_idx, f_data in enumerate(folds):
        rep = f_data["repeat"]
        fold = f_data["fold"]
        tr_tracks = f_data["train_tracks"]
        te_tracks = f_data["test_tracks"]

        y_tr_v = df_pme_v0.loc[tr_tracks, "Valence(mean)"].values
        y_tr_a = df_pme_v0.loc[tr_tracks, "Arousal(mean)"].values
        y_te_v = df_pme_v0.loc[te_tracks, "Valence(mean)"].values
        y_te_a = df_pme_v0.loc[te_tracks, "Arousal(mean)"].values

        rec = {"repeat": rep, "fold": fold}

        for m_name, df_source, feat_cols in models_config:
            X_tr = df_source.loc[tr_tracks, feat_cols]
            X_te = df_source.loc[te_tracks, feat_cols]

            reg_v = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_tr, y_tr_v)
            reg_a = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_tr, y_tr_a)

            p_v = reg_v.predict(X_te)
            p_a = reg_a.predict(X_te)

            r2_v = r2_score(y_te_v, p_v)
            r2_a = r2_score(y_te_a, p_a)
            r2_m = (r2_v + r2_a) / 2.0

            mae_v = mean_absolute_error(y_te_v, p_v)
            mae_a = mean_absolute_error(y_te_a, p_a)
            mae_m = (mae_v + mae_a) / 2.0

            fold_metrics[m_name]["r2_v"].append(r2_v)
            fold_metrics[m_name]["r2_a"].append(r2_a)
            fold_metrics[m_name]["r2_m"].append(r2_m)
            fold_metrics[m_name]["mae_v"].append(mae_v)
            fold_metrics[m_name]["mae_a"].append(mae_a)
            fold_metrics[m_name]["mae_m"].append(mae_m)

            rec[f"{m_name}|r2_v"] = r2_v
            rec[f"{m_name}|r2_a"] = r2_a
            rec[f"{m_name}|r2_m"] = r2_m

        fold_records.append(rec)
        if (f_idx + 1) % 5 == 0:
            print(f"  Completed fold {f_idx+1}/25 ({time.time() - t0:.1f}s)", flush=True)

    summary_rows = []
    base_r2_m = np.array(fold_metrics["Full-113 (V0 Baseline)"]["r2_m"])
    lite_r2_m = np.array(fold_metrics["MOODWAAVE-Lite (71 features)"]["r2_m"])

    stat_tests["pmemo"] = {}

    for m_name, _, feat_cols in models_config:
        r2_v_s = np.array(fold_metrics[m_name]["r2_v"])
        r2_a_s = np.array(fold_metrics[m_name]["r2_a"])
        r2_m_s = np.array(fold_metrics[m_name]["r2_m"])
        mae_m_s = np.array(fold_metrics[m_name]["mae_m"])

        # Paired test vs Full-113 V0
        d_vs_full, t_stat, p_val = nadeau_bengio_t_test(r2_m_s - base_r2_m, n_train=int(767*0.8), n_test=int(767*0.2))
        d_vs_lite, t_l_stat, p_l_val = nadeau_bengio_t_test(r2_m_s - lite_r2_m, n_train=int(767*0.8), n_test=int(767*0.2))

        stat_tests["pmemo"][m_name] = {
            "vs_full_v0": {"mean_diff_r2": round(d_vs_full, 4), "t_stat": round(t_stat, 3), "p_val": float(f"{p_val:.4e}")},
            "vs_lite_71": {"mean_diff_r2": round(d_vs_lite, 4), "t_stat": round(t_l_stat, 3), "p_val": float(f"{p_l_val:.4e}")}
        }

        summary_rows.append({
            "dataset": "PMEmo",
            "model": m_name,
            "features_count": len(feat_cols),
            "valence_r2": f"{np.mean(r2_v_s):.4f} ± {np.std(r2_v_s, ddof=1):.4f}",
            "arousal_r2": f"{np.mean(r2_a_s):.4f} ± {np.std(r2_a_s, ddof=1):.4f}",
            "mean_r2": f"{np.mean(r2_m_s):.4f} ± {np.std(r2_m_s, ddof=1):.4f}",
            "mean_mae": f"{np.mean(mae_m_s):.4f} ± {np.std(mae_m_s, ddof=1):.4f}",
            "raw_r2_m_mean": float(np.mean(r2_m_s)),
            "raw_r2_m_std": float(np.std(r2_m_s, ddof=1)),
            "raw_mae_m_mean": float(np.mean(mae_m_s)),
            "diff_r2_vs_full": round(d_vs_full, 4),
            "p_val_vs_full": f"{p_val:.4f}",
            "diff_r2_vs_lite": round(d_vs_lite, 4),
            "p_val_vs_lite": f"{p_l_val:.4f}"
        })
        print(f"  {m_name:35s}: Mean R² = {np.mean(r2_m_s):.4f} ± {np.std(r2_m_s, ddof=1):.4f} (Δ vs Full: {d_vs_full:+.4f}, p={p_val:.4f})")

    return summary_rows, fold_records, stat_tests


def main():
    splits_path = C.P1_SPLITS_JSON
    with open(splits_path, "r") as f:
        splits = json.load(f)

    gtz_summary, gtz_folds, stat_tests, _ = evaluate_gtzan(splits)
    pme_summary, pme_folds, stat_tests = evaluate_pmemo(splits, stat_tests)

    # Save summary
    df_summ = pd.DataFrame(gtz_summary + pme_summary)
    df_summ.to_csv(os.path.join(C.OUT, "accuracy_benchmark_summary.csv"), index=False)

    # Save fold data
    pd.DataFrame(gtz_folds).to_csv(os.path.join(C.OUT, "gtzan_accuracy_folds.csv"), index=False)
    pd.DataFrame(pme_folds).to_csv(os.path.join(C.OUT, "pmemo_accuracy_folds.csv"), index=False)

    with open(os.path.join(C.OUT, "statistical_tests.json"), "w") as f:
        json.dump(stat_tests, f, indent=2)

    print("\nSaved accuracy_benchmark_summary.csv, fold files, and statistical_tests.json.")


if __name__ == "__main__":
    main()

