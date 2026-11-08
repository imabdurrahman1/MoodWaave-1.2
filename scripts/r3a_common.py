"""
experiments/rev3_a_stats/common.py
Shared definitions, hyperparameter grids (extended), grouped inner CV helpers,
and statistical tests for TASK R3-A.
"""

import os
import sys
import json
import platform
import psutil
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Directories & Path Resolver
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "tables"))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

def resolve_file(rel_table_path, fallback_project_path):
    p1 = os.path.join(TABLES_DIR, rel_table_path)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(PROJECT_ROOT, fallback_project_path)
    if os.path.exists(p2):
        return p2
    return p1


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
FEATURE_GROUPS_PATH = resolve_file("feature_groups.json", os.path.join("experiments", "idea2_cost_pareto", "feature_groups.json"))
with open(FEATURE_GROUPS_PATH, "r", encoding="utf-8") as _f:
    _fg = json.load(_f)["groups"]

FULL_113_FEATURES = [
    c
    for grp in ["MFCC", "Chroma", "Tonnetz", "Spectral", "SpecContrast", "Temporal"]
    for c in _fg[grp]
]

LITE_71_FEATURES = [
    c for grp in ["MFCC", "SpecContrast", "Temporal"] for c in _fg[grp]
]

FEATURE_SETS = {
    "Full-113": FULL_113_FEATURES,
    "Lite-71": LITE_71_FEATURES,
}

# ---------------------------------------------------------------------------
# R3-A Hyperparameter grids — extended one step beyond P4 boundary hits
# ---------------------------------------------------------------------------
GENRE_PARAM_GRIDS = {
    "LightGBM": {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 6, 9],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    },
    "Random Forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    },
    # SVM-RBF: extended one step beyond boundary hit (C up to 1000, gamma down to 0.001)
    "SVM-RBF": {
        "clf__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "clf__gamma": ["scale", 0.001, 0.003, 0.01],
    },
    # Logistic: extended one step below boundary hit (C down to 0.003)
    "Logistic Regression": {
        "clf__C": [0.003, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0],
    },
}

VA_PARAM_GRIDS = {
    "LightGBM": {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 6, 9],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    },
    "Random Forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    },
    # SVR-RBF: extended one step beyond P4 boundary (C up to 1000, gamma down to 0.001)
    "SVR-RBF": {
        "reg__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "reg__gamma": ["scale", 0.001, 0.003, 0.01],
    },
    # Ridge: extended one step beyond P4 boundary (alpha up to 1e4)
    "Ridge": {
        "reg__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    },
}

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def get_genre_base_model(model_name: str, seed: int = 42):
    """Instantiate base estimator for Genre classification."""
    if model_name == "LightGBM":
        return LGBMClassifier(
            random_state=seed, verbose=-1, n_jobs=1, subsample_freq=1
        )
    elif model_name == "XGBoost":
        return XGBClassifier(
            random_state=seed, eval_metric="mlogloss", n_jobs=1
        )
    elif model_name == "Random Forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    elif model_name == "SVM-RBF":
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=seed))]
        )
    elif model_name == "Logistic Regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
    else:
        raise ValueError(f"Unknown genre model: {model_name}")


def get_va_base_model(model_name: str, seed: int = 42):
    """Instantiate base estimator for Valence/Arousal regression."""
    if model_name == "LightGBM":
        return LGBMRegressor(
            random_state=seed, verbose=-1, n_jobs=1, subsample_freq=1
        )
    elif model_name == "XGBoost":
        return XGBRegressor(
            random_state=seed, eval_metric="rmse", n_jobs=1
        )
    elif model_name == "Random Forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1)
    elif model_name == "SVR-RBF":
        return Pipeline(
            [("scaler", StandardScaler()), ("reg", SVR(kernel="rbf"))]
        )
    elif model_name == "Ridge":
        return Pipeline(
            [("scaler", StandardScaler()), ("reg", Ridge())]
        )
    else:
        raise ValueError(f"Unknown VA model: {model_name}")

# ---------------------------------------------------------------------------
# Grouped inner CV helper
# ---------------------------------------------------------------------------
def make_inner_cv_genre(n_splits: int = 3):
    """Returns StratifiedGroupKFold splitter for genre classification inner tuning."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=False)


def make_inner_cv_regression(n_splits: int = 3):
    """Returns GroupKFold splitter for regression inner tuning."""
    return GroupKFold(n_splits=n_splits)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_gtzan(splits_json_path: str):
    """Load clean GTZAN feature table and P1 splits."""
    gtzan_csv = resolve_file("gtzan_features.csv", os.path.join("data", "gtzan_features.csv"))
    rem_csv = resolve_file("p1_gtzan_removed_tracks.csv", os.path.join("experiments", "rev2_p1_splits", "removed_tracks.csv"))
    artist_map_csv = resolve_file("p1_gtzan_artist_map.csv", os.path.join("experiments", "rev2_p1_splits", "artist_map.csv"))

    df_g = pd.read_csv(gtzan_csv).set_index("filename")
    rem_tracks = set(pd.read_csv(rem_csv)["filename"])
    df_g_clean = df_g.loc[~df_g.index.isin(rem_tracks)].copy()

    # Merge artist info
    df_am = pd.read_csv(artist_map_csv)
    gtzan_am = df_am[df_am["dataset"] == "GTZAN"].set_index("track_id")
    df_g_clean["artist_id"] = gtzan_am.reindex(df_g_clean.index)["artist"].values

    with open(splits_json_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    all_folds = splits_data["gtzan"]["fault_removed_artist_grouped"]
    return df_g_clean, all_folds


def load_pmemo(splits_json_path: str):
    """Load PMEmo feature table and P1 artist-grouped folds."""
    pmemo_csv = resolve_file("pmemo_ready.csv", os.path.join("data", "pmemo_ready.csv"))
    artist_map_csv = resolve_file("p1_gtzan_artist_map.csv", os.path.join("experiments", "rev2_p1_splits", "artist_map.csv"))

    df_p = pd.read_csv(pmemo_csv)
    df_p["musicId_str"] = df_p["musicId"].astype(str)
    df_p.set_index("musicId_str", inplace=True)

    # Merge artist info
    df_am = pd.read_csv(artist_map_csv)
    pmemo_am = df_am[df_am["dataset"] == "PMEmo"].copy()
    pmemo_am["track_id"] = pmemo_am["track_id"].astype(str)
    pmemo_am = pmemo_am.set_index("track_id")
    df_p["artist_id"] = pmemo_am.reindex(df_p.index)["artist"].values

    with open(splits_json_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    # Use repeats 0 and 1 for 5x2 = 10 outer folds (same as P4)
    all_folds = [
        f
        for f in splits_data["pmemo"]["artist_grouped_5x5"]
        if f["repeat"] in (0, 1)
    ]
    return df_p, all_folds

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def nadeau_bengio_test(diffs: np.ndarray, test_train_ratio: float = 0.25):
    """
    Nadeau-Bengio corrected resampled t-test.
    diffs: array of per-fold differences (model - reference).
    test_train_ratio: n_test / n_train (0.25 for 5-fold).
    Returns dict with mean_diff, std_diff, se_corrected, t_stat, p_value, ci_lower, ci_upper.
    """
    n = len(diffs)
    if n <= 1:
        mean_d = float(np.mean(diffs))
        return dict(
            n_folds=n, mean_diff=mean_d, std_diff=0.0, se_corrected=0.0,
            t_stat=np.nan, p_value=np.nan, ci_lower=mean_d, ci_upper=mean_d,
        )

    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    std_d = float(np.sqrt(var_d))

    var_corr = (1.0 / n + test_train_ratio) * var_d
    se_corr = float(np.sqrt(var_corr))

    df = n - 1
    if se_corr == 0:
        t_stat, p_val = 0.0, 1.0
        ci_lower = ci_upper = mean_d
    else:
        t_stat = mean_d / se_corr
        p_val = float(2.0 * scipy_stats.t.sf(np.abs(t_stat), df=df))
        t_crit = float(scipy_stats.t.ppf(0.975, df=df))
        ci_lower = mean_d - t_crit * se_corr
        ci_upper = mean_d + t_crit * se_corr

    return dict(
        n_folds=n,
        mean_diff=mean_d,
        std_diff=std_d,
        se_corrected=se_corr,
        t_stat=t_stat,
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def tost_verdict(diffs: np.ndarray, margin: float, test_train_ratio: float = 0.25):
    """
    Two One-Sided Tests (TOST) equivalence test.
    H0: |mean_diff| >= margin (difference is non-trivial)
    HA: |mean_diff| < margin (equivalent within margin)
    
    Returns (verdict, p_lower, p_upper, p_tost) where verdict is one of:
    'equivalent', 'worse', 'inconclusive'
    p_lower is p-value for diff > -margin (one-sided)
    p_upper is p-value for diff < +margin (one-sided)
    """
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    var_corr = (1.0 / n + test_train_ratio) * var_d
    se_corr = float(np.sqrt(max(var_corr, 1e-15)))
    df = n - 1

    # Test 1: H0: diff <= -margin  (lower bound)
    t_lower = (mean_d + margin) / se_corr
    p_lower = float(scipy_stats.t.sf(t_lower, df=df))  # one-tailed upper

    # Test 2: H0: diff >= +margin  (upper bound)
    t_upper = (mean_d - margin) / se_corr
    p_upper = float(scipy_stats.t.cdf(t_upper, df=df))  # one-tailed lower

    p_tost = max(p_lower, p_upper)

    if p_tost < 0.05:
        verdict = "equivalent"
    elif mean_d < -margin and (mean_d + scipy_stats.t.ppf(0.95, df=df) * se_corr) < 0:
        verdict = "worse"
    else:
        verdict = "inconclusive"

    return verdict, p_lower, p_upper, p_tost

# ---------------------------------------------------------------------------
# Environment logging
# ---------------------------------------------------------------------------
def log_environment(output_dir: str) -> dict:
    """Logs system, hardware, and library versions to output_dir/environment.json."""
    import lightgbm
    import xgboost
    import sklearn

    os.makedirs(output_dir, exist_ok=True)
    env = {
        "project": "MOODWAAVE",
        "task": "TASK R3-A - Grouped Inner Tuning Nested CV",
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": sys.version,
        "libraries": {
            "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    env_path = os.path.join(output_dir, "environment.json")
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(f"Environment logged to {env_path}")
    return env

