"""
experiments/rev3_a_stats/03_default_vs_tuned_lightgbm.py
TASK R3-A step 4: Default vs. tuned LightGBM on the SAME 15 GTZAN outer folds
and 10 PMEmo outer folds, with paired Nadeau-Bengio tests.

Default LightGBM: n_estimators=100, lr=0.1, num_leaves=31, min_child_samples=20.
Tuned LightGBM: best from R3-A grouped inner tuning (read from folds CSV).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
from lightgbm import LGBMClassifier, LGBMRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3a_common as common

OUT_DIR = common.BASE_DIR
SPLITS_JSON = os.path.join(
    common.PROJECT_ROOT, "experiments", "rev2_p1_splits", "splits.json"
)

LGBM_DEFAULTS = {
    "genre": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
    },
    "mood": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
    },
}


def run_gtzan_default_vs_tuned():
    df_clean, all_folds = common.load_gtzan(SPLITS_JSON)

    tuned_csv = os.path.join(OUT_DIR, "gtzan_r3a_folds.csv")
    if not os.path.exists(tuned_csv):
        raise FileNotFoundError(f"Missing {tuned_csv}. Run 01_run_gtzan_r3a.py first.")
    df_tuned = pd.read_csv(tuned_csv)
    tuned_lgbm = df_tuned[
        (df_tuned["model"] == "LightGBM") & (df_tuned["feature_set"] == "Full-113")
    ].set_index(["repeat", "fold"])

    records = []
    for fi in all_folds:
        repeat, fold_idx = fi["repeat"], fi["fold"]
        train_df = df_clean.loc[fi["train_tracks"]]
        test_df = df_clean.loc[fi["test_tracks"]]
        y_train = train_df["label"].values
        y_test = test_df["label"].values

        feats = common.FULL_113_FEATURES
        X_tr = train_df[feats].values
        X_te = test_df[feats].values

        # Default
        t0 = time.time()
        clf_def = LGBMClassifier(
            random_state=42, verbose=-1, n_jobs=1, subsample_freq=1,
            **LGBM_DEFAULTS["genre"]
        )
        clf_def.fit(X_tr, y_train)
        preds_def = clf_def.predict(X_te)
        acc_def = accuracy_score(y_test, preds_def)
        f1_def = f1_score(y_test, preds_def, average="macro")
        t_def = time.time() - t0

        # Tuned
        t0 = time.time()
        row = tuned_lgbm.loc[(repeat, fold_idx)]
        best_params = json.loads(row["best_params"])
        clf_tuned = LGBMClassifier(
            random_state=42, verbose=-1, n_jobs=1, subsample_freq=1,
            **best_params
        )
        clf_tuned.fit(X_tr, y_train)
        preds_tuned = clf_tuned.predict(X_te)
        acc_tuned = accuracy_score(y_test, preds_tuned)
        f1_tuned = f1_score(y_test, preds_tuned, average="macro")
        t_tuned = time.time() - t0

        records.append({
            "dataset": "GTZAN",
            "task": "genre",
            "repeat": repeat,
            "fold": fold_idx,
            "acc_default": float(acc_def),
            "f1_default": float(f1_def),
            "acc_tuned": float(acc_tuned),
            "f1_tuned": float(f1_tuned),
            "diff_acc": float(acc_tuned - acc_def),
            "diff_f1": float(f1_tuned - f1_def),
            "time_default_s": round(t_def, 2),
            "time_tuned_s": round(t_tuned, 2),
            "tuned_params": row["best_params"],
        })
        print(
            f"GTZAN fold rep={repeat}, fold={fold_idx}: "
            f"default_acc={acc_def:.4f}, tuned_acc={acc_tuned:.4f}, "
            f"delta={acc_tuned - acc_def:+.4f}"
        )

    df_records = pd.DataFrame(records)
    return df_records


def run_pmemo_default_vs_tuned():
    df_pmemo, all_folds = common.load_pmemo(SPLITS_JSON)

    tuned_csv = os.path.join(OUT_DIR, "pmemo_r3a_folds.csv")
    if not os.path.exists(tuned_csv):
        raise FileNotFoundError(f"Missing {tuned_csv}. Run 02_run_pmemo_r3a.py first.")
    df_tuned = pd.read_csv(tuned_csv)
    tuned_lgbm = df_tuned[
        (df_tuned["model"] == "LightGBM") & (df_tuned["feature_set"] == "Full-113")
    ].set_index(["task", "repeat", "fold"])

    records = []
    targets = [("valence", "Valence(mean)"), ("arousal", "Arousal(mean)")]

    for fi in all_folds:
        repeat, fold_idx = fi["repeat"], fi["fold"]
        train_df = df_pmemo.loc[fi["train_tracks"]]
        test_df = df_pmemo.loc[fi["test_tracks"]]
        feats = common.FULL_113_FEATURES
        X_tr = train_df[feats].values
        X_te = test_df[feats].values

        for task_label, target_col in targets:
            y_train = train_df[target_col].values
            y_test = test_df[target_col].values

            # Default
            t0 = time.time()
            reg_def = LGBMRegressor(
                random_state=42, verbose=-1, n_jobs=1, subsample_freq=1,
                **LGBM_DEFAULTS["mood"]
            )
            reg_def.fit(X_tr, y_train)
            preds_def = reg_def.predict(X_te)
            r2_def = r2_score(y_test, preds_def)
            mae_def = mean_absolute_error(y_test, preds_def)
            t_def = time.time() - t0

            # Tuned
            t0 = time.time()
            row = tuned_lgbm.loc[(task_label, repeat, fold_idx)]
            best_params = json.loads(row["best_params"])
            reg_tuned = LGBMRegressor(
                random_state=42, verbose=-1, n_jobs=1, subsample_freq=1,
                **best_params
            )
            reg_tuned.fit(X_tr, y_train)
            preds_tuned = reg_tuned.predict(X_te)
            r2_tuned = r2_score(y_test, preds_tuned)
            mae_tuned = mean_absolute_error(y_test, preds_tuned)
            t_tuned = time.time() - t0

            records.append({
                "dataset": "PMEmo",
                "task": task_label,
                "repeat": repeat,
                "fold": fold_idx,
                "r2_default": float(r2_def),
                "mae_default": float(mae_def),
                "r2_tuned": float(r2_tuned),
                "mae_tuned": float(mae_tuned),
                "diff_r2": float(r2_tuned - r2_def),
                "diff_mae": float(mae_tuned - mae_def),
                "time_default_s": round(t_def, 2),
                "time_tuned_s": round(t_tuned, 2),
                "tuned_params": row["best_params"],
            })
            print(
                f"PMEmo {task_label} fold rep={repeat}, fold={fold_idx}: "
                f"default_r2={r2_def:.4f}, tuned_r2={r2_tuned:.4f}, "
                f"delta={r2_tuned - r2_def:+.4f}"
            )

    return pd.DataFrame(records)


def compute_and_report(df_g: pd.DataFrame, df_p: pd.DataFrame):
    records = []

    # GTZAN accuracy
    diffs_acc = df_g["diff_acc"].values
    nb = common.nadeau_bengio_test(diffs_acc, test_train_ratio=1/4)
    records.append({
        "dataset": "GTZAN", "task": "genre", "metric": "Accuracy",
        "mean_default": float(df_g["acc_default"].mean()),
        "std_default": float(df_g["acc_default"].std(ddof=1)),
        "mean_tuned": float(df_g["acc_tuned"].mean()),
        "std_tuned": float(df_g["acc_tuned"].std(ddof=1)),
        **{f"nb_{k}": v for k, v in nb.items()},
    })

    # GTZAN F1
    diffs_f1 = df_g["diff_f1"].values
    nb_f1 = common.nadeau_bengio_test(diffs_f1, test_train_ratio=1/4)
    records.append({
        "dataset": "GTZAN", "task": "genre", "metric": "Macro-F1",
        "mean_default": float(df_g["f1_default"].mean()),
        "std_default": float(df_g["f1_default"].std(ddof=1)),
        "mean_tuned": float(df_g["f1_tuned"].mean()),
        "std_tuned": float(df_g["f1_tuned"].std(ddof=1)),
        **{f"nb_{k}": v for k, v in nb_f1.items()},
    })

    # PMEmo
    for task_label in ["valence", "arousal"]:
        df_sub = df_p[df_p["task"] == task_label]
        for metric, col in [("R2", "diff_r2"), ("MAE", "diff_mae")]:
            diffs = df_sub[col].values
            nb = common.nadeau_bengio_test(diffs, test_train_ratio=1/4)
            mn_col = f"r2" if metric == "R2" else "mae"
            records.append({
                "dataset": "PMEmo", "task": task_label, "metric": metric,
                "mean_default": float(df_sub[f"{mn_col}_default"].mean()),
                "std_default": float(df_sub[f"{mn_col}_default"].std(ddof=1)),
                "mean_tuned": float(df_sub[f"{mn_col}_tuned"].mean()),
                "std_tuned": float(df_sub[f"{mn_col}_tuned"].std(ddof=1)),
                **{f"nb_{k}": v for k, v in nb.items()},
            })

    df_res = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "default_vs_tuned_lgbm_r3a.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nDefault vs. Tuned LightGBM results saved to {out_csv}")
    print(df_res[["dataset", "task", "metric", "mean_default", "mean_tuned",
                  "nb_mean_diff", "nb_p_value"]].to_string(index=False))
    return df_res


def main():
    print("=== Default vs. Tuned LightGBM (R3-A outer folds) ===\n")

    out_g = os.path.join(OUT_DIR, "default_vs_tuned_gtzan_folds.csv")
    if os.path.exists(out_g):
        print(f"Loading cached GTZAN default-vs-tuned from {out_g}")
        df_g = pd.read_csv(out_g)
    else:
        df_g = run_gtzan_default_vs_tuned()
        df_g.to_csv(out_g, index=False)

    out_p = os.path.join(OUT_DIR, "default_vs_tuned_pmemo_folds.csv")
    if os.path.exists(out_p):
        print(f"Loading cached PMEmo default-vs-tuned from {out_p}")
        df_p = pd.read_csv(out_p)
    else:
        df_p = run_pmemo_default_vs_tuned()
        df_p.to_csv(out_p, index=False)

    compute_and_report(df_g, df_p)
    print("\nDone.")


if __name__ == "__main__":
    main()

