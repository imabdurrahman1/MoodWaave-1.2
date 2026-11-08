"""Shared definitions, paths, feature mappings, and utilities for Final Consolidation (MOODWAAVE-Lite).
Ensures single-threading env variables are set BEFORE importing numpy/librosa.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import platform
import numpy as np
import pandas as pd
from scipy import stats
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = HERE

GTZAN_CSV = os.path.join(ROOT, "data", "gtzan_features.csv")
PMEMO_CSV = os.path.join(ROOT, "data", "pmemo_ready.csv")
GTZAN_DIR = os.path.join(ROOT, "data", "gtzan", "genres_original")
PMEMO_DIR = os.path.join(ROOT, "data", "pmemo", "chorus")
FEATURE_GROUPS_JSON = os.path.join(ROOT, "experiments", "idea2_cost_pareto", "feature_groups.json")
SAMPLE_IDS_JSON = os.path.join(ROOT, "experiments", "idea2_cost_pareto", "sample_ids.json")

# Model hyperparameters matching the project baseline
LGBM_GENRE_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1
}

LGBM_MOOD_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1
}

# Load feature group mapping
with open(FEATURE_GROUPS_JSON, "r") as f:
    _FG_DATA = json.load(f)

FEATURE_GROUPS = _FG_DATA["groups"]

# MOODWAAVE-Lite = MFCC (52) + SpecContrast (14) + Temporal (5) = 71 features
LITE_GROUPS = ["MFCC", "SpecContrast", "Temporal"]
EXPENSIVE_GROUPS = ["Chroma", "Tonnetz", "Spectral"]

LITE_FEATURES = []
for g in LITE_GROUPS:
    LITE_FEATURES.extend(FEATURE_GROUPS[g])
assert len(LITE_FEATURES) == 71, f"Expected 71 Lite features, got {len(LITE_FEATURES)}"

FULL_FEATURES = []
for g in ["MFCC", "Chroma", "Tonnetz", "Spectral", "SpecContrast", "Temporal"]:
    FULL_FEATURES.extend(FEATURE_GROUPS[g])
assert len(FULL_FEATURES) == 113, f"Expected 113 Full features, got {len(FULL_FEATURES)}"

FEATURE_TO_GROUP = {}
for g, feats in FEATURE_GROUPS.items():
    for feat in feats:
        FEATURE_TO_GROUP[feat] = g

# 4x4 Grid definition for Valence and Arousal
GRID_BINS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])

# 4x4 Emotion Matrix matching get_nuanced_mood in src/pipeline.py
# Rows index Arousal: 0 = [0, 0.25), 1 = [0.25, 0.50), 2 = [0.50, 0.75), 3 = [0.75, 1.0]
# Cols index Valence: 0 = [0, 0.25), 1 = [0.25, 0.50), 2 = [0.50, 0.75), 3 = [0.75, 1.0]
ZONE_MATRIX = [
    # a in [0.0, 0.25)
    ["Depressed", "Bored", "Calm", "Serene"],
    # a in [0.25, 0.50)
    ["Sad", "Gloomy", "Peaceful", "Relaxed"],
    # a in [0.50, 0.75)
    ["Angry", "Distressed", "Pleased", "Happy"],
    # a in [0.75, 1.0]
    ["Frustrated", "Tense", "Energetic", "Excited"]
]

ZONE_NAMES = [name for row in ZONE_MATRIX for name in row]
ZONE_TO_COORD = {ZONE_MATRIX[r][c]: (r, c) for r in range(4) for c in range(4)}


def val_arousal_to_bin_idx(val):
    val = float(np.clip(val, 0.0, 1.0))
    if val >= 1.0:
        return 3
    return min(3, max(0, int(np.floor(val / 0.25))))


def get_zone_name(v, a):
    c = val_arousal_to_bin_idx(v)
    r = val_arousal_to_bin_idx(a)
    return ZONE_MATRIX[r][c]


def get_overlapping_zones(v_lo, v_hi, a_lo, a_hi):
    v_lo_c = max(0.0, min(1.0, float(v_lo)))
    v_hi_c = max(0.0, min(1.0, float(v_hi)))
    a_lo_c = max(0.0, min(1.0, float(a_lo)))
    a_hi_c = max(0.0, min(1.0, float(a_hi)))
    if v_lo_c > v_hi_c:
        v_lo_c, v_hi_c = v_hi_c, v_lo_c
    if a_lo_c > a_hi_c:
        a_lo_c, a_hi_c = a_hi_c, a_lo_c

    cols = []
    for c in range(4):
        lo, hi = GRID_BINS[c], GRID_BINS[c + 1]
        if (c == 3 and v_lo_c <= hi and v_hi_c >= lo) or (c < 3 and v_lo_c < hi and v_hi_c >= lo):
            cols.append(c)
    if not cols:
        cols = [val_arousal_to_bin_idx((v_lo_c + v_hi_c) / 2.0)]

    rows = []
    for r in range(4):
        lo, hi = GRID_BINS[r], GRID_BINS[r + 1]
        if (r == 3 and a_lo_c <= hi and a_hi_c >= lo) or (r < 3 and a_lo_c < hi and a_hi_c >= lo):
            rows.append(r)
    if not rows:
        rows = [val_arousal_to_bin_idx((a_lo_c + a_hi_c) / 2.0)]

    zones = []
    for r in rows:
        for c in cols:
            zones.append(ZONE_MATRIX[r][c])
    return sorted(list(set(zones)))


def finite_sample_conformal_quantile(scores, alpha):
    n = len(scores)
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    return float(np.quantile(scores, level, method="inverted_cdf" if hasattr(np, "quantile") else "higher"))


def corrected_resampled_ttest(diffs, n_train_ratio=0.80, n_test_ratio=0.20):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    diffs = np.asarray(diffs)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    c_factor = (1.0 / n) + (n_test_ratio / n_train_ratio)
    var_corrected = c_factor * var_d
    se_corrected = float(np.sqrt(max(1e-12, var_corrected)))
    df = n - 1
    t_stat = mean_d / se_corrected if se_corrected > 0 else 0.0
    p_val = float(2.0 * stats.t.sf(np.abs(t_stat), df=df))
    t_crit = float(stats.t.ppf(0.975, df=df))
    ci_lower = mean_d - t_crit * se_corrected
    ci_upper = mean_d + t_crit * se_corrected
    return {
        "mean_diff": mean_d,
        "se_corrected": se_corrected,
        "t_stat": t_stat,
        "p_val": p_val,
        "ci_95": (ci_lower, ci_upper),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


def load_gtzan_data():
    df = pd.read_csv(GTZAN_CSV)
    X_full = df[FULL_FEATURES].copy()
    X_lite = df[LITE_FEATURES].copy()
    y_str = df["label"].values
    classes = sorted(list(np.unique(y_str)))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[s] for s in y_str])
    return df, X_full, X_lite, y, classes


def load_pmemo_data():
    df = pd.read_csv(PMEMO_CSV)
    X_full = df[FULL_FEATURES].copy()
    X_lite = df[LITE_FEATURES].copy()
    y_val = df["Valence(mean)"].values
    y_aro = df["Arousal(mean)"].values
    return df, X_full, X_lite, y_val, y_aro


# Shared STFT extraction functions for Lite-71
def _ms(prefix, M):
    d = {}
    for i in range(M.shape[0]):
        d[f"{prefix}_mean_{i+1}"] = float(np.mean(M[i]))
        d[f"{prefix}_std_{i+1}"] = float(np.std(M[i]))
    return d


def _mfcc_from(mfcc):
    d = _ms("mfcc", mfcc)
    d.update(_ms("delta_mfcc", librosa.feature.delta(mfcc)))
    return d


def _stats(name, M):
    return {f"{name}_mean": float(np.mean(M)), f"{name}_std": float(np.std(M))}


def extract_lite_71_shared(y, sr):
    """Fast shared-STFT extraction of the 71 Lite features."""
    S = np.abs(librosa.stft(y))
    out = {}
    # 1. MFCC (52)
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    out.update(_mfcc_from(mfcc))
    # 2. SpecContrast (14)
    out.update(_ms("spectral_contrast", librosa.feature.spectral_contrast(S=S, sr=sr)))
    # 3. Temporal (5)
    out.update(_stats("zcr", librosa.feature.zero_crossing_rate(y)))
    out.update(_stats("rms", librosa.feature.rms(y=y)))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    out["tempo"] = float(np.asarray(tempo).item())
    return out


def hardware_info():
    import importlib.metadata as md
    info = {
        "os": platform.platform(),
        "python": sys.version,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count()
    }
    try:
        import subprocess
        info["cpu_model"] = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            text=True
        ).strip()
        info["ram_gb"] = round(int(subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
            text=True
        ).strip()) / (1024**3), 2)
    except Exception as e:
        info["cpu_model_error"] = str(e)
    for p in ["numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "shap", "matplotlib", "librosa"]:
        try:
            info[f"pkg_{p}"] = md.version(p)
        except Exception:
            info[f"pkg_{p}"] = None
    info["thread_env"] = {k: os.environ[k] for k in os.environ if k.endswith("_NUM_THREADS")}
    return info


def save_env_log():
    info = hardware_info()
    with open(os.path.join(OUT, "environment.json"), "w") as f:
        json.dump(info, f, indent=2)
    pk = {k[4:]: v for k, v in info.items() if k.startswith("pkg_") and v}
    with open(os.path.join(OUT, "requirements_note.txt"), "w") as f:
        f.write("# Versions used for experiments/final_consolidation\n")
        f.write(f"# Python {sys.version.split()[0]}\n")
        for k, v in pk.items():
            f.write(f"{k}=={v}\n")
    return info
