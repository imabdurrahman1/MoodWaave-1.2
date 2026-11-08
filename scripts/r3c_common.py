import os
import sys
import json
import platform
import psutil
import numpy as np
import pandas as pd
import librosa
from scipy import stats
from pathlib import Path

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

FMA_METADATA_DIR = os.path.join(DATA_DIR, "fma", "fma_metadata")
FMA_TRACKS_CSV = os.path.join(FMA_METADATA_DIR, "tracks.csv")
FMA_AUDIO_DIR = os.path.join(DATA_DIR, "fma", "audio")

DEAM_AUDIO_DIR = os.path.join(DATA_DIR, "deam", "audio")
DEAM_ANNOTATIONS_CSV = os.path.join(DATA_DIR, "deam", "deam_annotations.csv")

GTZAN_CSV = os.path.join(DATA_DIR, "gtzan_features.csv")
PMEMO_READY_CSV = os.path.join(DATA_DIR, "pmemo_ready.csv")

SEED = 42

# Feature Groups definitions from idea2
FEATURE_GROUPS_JSON = os.path.join(PROJECT_ROOT, "experiments", "idea2_cost_pareto", "feature_groups.json")
with open(FEATURE_GROUPS_JSON, "r") as f:
    _FG = json.load(f)["groups"]

LITE_FEATURES = _FG["MFCC"] + _FG["SpecContrast"] + _FG["Temporal"]  # 71 features
FULL_FEATURES = _FG["MFCC"] + _FG["Chroma"] + _FG["Tonnetz"] + _FG["Spectral"] + _FG["SpecContrast"] + _FG["Temporal"]  # 113 features
TONNETZ_FEATURES = _FG["Tonnetz"]  # 12 features

def log_environment(out_dir):
    """Log hardware specifications and exact library versions."""
    import sklearn
    import lightgbm
    import soundfile
    
    cpu_info = platform.processor() or "Unknown CPU"
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    
    env_info = {
        "project": "MOODWAAVE",
        "task": "TASK R3-C: FMA-small and DEAM Generalization Benchmarks",
        "timestamp": pd.Timestamp.now().isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version,
        "cpu": cpu_info,
        "logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "librosa": librosa.__version__,
            "soundfile": soundfile.__version__,
        }
    }
    
    out_path = os.path.join(out_dir, "environment.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
    return env_info

def load_audio_mono(file_path: str, sr: int = 22050, duration: float = 30.0, middle_30s: bool = False):
    """Load audio file as mono at target sr.
    If middle_30s=True, extracts the central 30 seconds of the track (e.g. for DEAM 45s clips).
    Otherwise extracts at most duration seconds from start.
    """
    y, orig_sr = librosa.load(file_path, sr=sr, mono=True)
    total_len = len(y)
    
    if middle_30s:
        target_len = int(duration * sr)
        if total_len > target_len:
            start = (total_len - target_len) // 2
            y = y[start : start + target_len]
        else:
            if total_len < target_len:
                y = np.pad(y, (0, target_len - total_len))
    else:
        target_len = int(duration * sr)
        if total_len > target_len:
            y = y[:target_len]
        elif total_len < target_len:
            y = np.pad(y, (0, target_len - total_len))
            
    return y.astype(np.float32), sr

def extract_moodwaave_features(y: np.ndarray, sr: int = 22050) -> dict:
    """Extract full 113 features (V0 Tonnetz) plus 12 V1 Tonnetz features in a single pass."""
    # 1. MFCCs + Deltas (52 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    
    # 2. Chroma (24 features)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # 3. Spectral Features (6 features)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    
    # 4. Spectral Contrast (14 features)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # 5. Tonnetz V0 (Original HPSS + CQT, 12 features)
    y_harm = librosa.effects.harmonic(y)
    tonnetz_v0 = librosa.feature.tonnetz(y=y_harm, sr=sr)
    
    # 6. Tonnetz V1 (From precomputed STFT chroma, 12 features)
    tonnetz_v1 = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    
    # 7. Temporal Features (5 features)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(np.asarray(tempo).item())
    
    feats = {}
    
    # Pack MFCC & Deltas
    for i in range(13):
        feats[f"mfcc_mean_{i+1}"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_std_{i+1}"] = float(np.std(mfcc[i]))
    for i in range(13):
        feats[f"delta_mfcc_mean_{i+1}"] = float(np.mean(delta_mfcc[i]))
        feats[f"delta_mfcc_std_{i+1}"] = float(np.std(delta_mfcc[i]))
        
    # Pack Chroma
    for i in range(12):
        feats[f"chroma_mean_{i+1}"] = float(np.mean(chroma[i]))
        feats[f"chroma_std_{i+1}"] = float(np.std(chroma[i]))
        
    # Pack Spectral Contrast
    for i in range(7):
        feats[f"spectral_contrast_mean_{i+1}"] = float(np.mean(spectral_contrast[i]))
        feats[f"spectral_contrast_std_{i+1}"] = float(np.std(spectral_contrast[i]))
        
    # Pack Tonnetz V0
    for i in range(6):
        feats[f"tonnetz_mean_{i+1}"] = float(np.mean(tonnetz_v0[i]))
        feats[f"tonnetz_std_{i+1}"] = float(np.std(tonnetz_v0[i]))
        
    # Pack Tonnetz V1 (explicit keys)
    for i in range(6):
        feats[f"tonnetz_v1_mean_{i+1}"] = float(np.mean(tonnetz_v1[i]))
        feats[f"tonnetz_v1_std_{i+1}"] = float(np.std(tonnetz_v1[i]))
        
    # Pack Spectral & Temporal
    feats["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    feats["spectral_centroid_std"] = float(np.std(spectral_centroid))
    feats["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
    feats["spectral_rolloff_std"] = float(np.std(spectral_rolloff))
    feats["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
    feats["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))
    
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))
    feats["tempo"] = tempo_val
    
    return feats

def nadeau_bengio_test(diffs: np.ndarray, test_train_ratio: float = 0.25):
    """Nadeau-Bengio corrected resampled paired t-test."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1)) if n > 1 else 0.0
    
    correction = (1.0 / n) + test_train_ratio
    corrected_var = correction * var_d
    corrected_se = np.sqrt(corrected_var) if corrected_var > 0 else 1e-12
    
    t_stat = mean_d / corrected_se
    df = n - 1
    p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df)))
    
    tcrit = float(stats.t.ppf(0.975, df=df))
    ci_lower = mean_d - tcrit * corrected_se
    ci_upper = mean_d + tcrit * corrected_se
    
    return {
        "mean_diff": mean_d,
        "corrected_se": corrected_se,
        "t_stat": t_stat,
        "df": df,
        "p_val": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_val < 0.05
    }

def tost_equivalence_test(diffs: np.ndarray, bound: float = 0.05, test_train_ratio: float = 0.25):
    """Two One-Sided Tests (TOST) for equivalence within [-bound, +bound]."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1)) if n > 1 else 0.0
    
    correction = (1.0 / n) + test_train_ratio
    corrected_var = correction * var_d
    corrected_se = np.sqrt(corrected_var) if corrected_var > 0 else 1e-12
    df = n - 1
    
    # t1 tests H01: mean_d <= -bound vs H11: mean_d > -bound
    t1 = (mean_d - (-bound)) / corrected_se
    p1 = float(1.0 - stats.t.cdf(t1, df=df))
    
    # t2 tests H02: mean_d >= bound vs H12: mean_d < bound
    t2 = (mean_d - bound) / corrected_se
    p2 = float(stats.t.cdf(t2, df=df))
    
    p_tost = max(p1, p2)
    equivalent = p_tost < 0.05
    
    return {
        "mean_diff": mean_d,
        "bound": bound,
        "t1": t1,
        "p1": p1,
        "t2": t2,
        "p2": p2,
        "p_tost": p_tost,
        "equivalent": equivalent
    }

