import os
import sys
import json
import time
import platform
import psutil
import soundfile as sf
import torch
import torchaudio.functional as AF
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, RandomizedSearchCV

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Directories & Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_JSON = os.path.join(PROJECT_ROOT, "experiments", "rev2_p1_splits", "splits.json")
GTZAN_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "gtzan", "genres_original")
PMEMO_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "pmemo", "chorus")

# ---------------------------------------------------------------------------
# Audio Loader
# ---------------------------------------------------------------------------
def load_audio_mono_30s(file_path: str, target_sr: int) -> np.ndarray:
    """
    Load audio, convert to mono, resample to target_sr, and truncate to at most 30s.
    Returns 1D numpy float32 array.
    """
    data, orig_sr = sf.read(file_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data)
    if orig_sr != target_sr:
        wav = AF.resample(wav, orig_sr, target_sr)
    max_samples = target_sr * 30
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    return wav.numpy()

# ---------------------------------------------------------------------------
# Pretraining Metadata & Model Descriptions
# ---------------------------------------------------------------------------
MODELS_METADATA = {
    "PANNs-CNN14": {
        "display_name": "PANNs CNN14",
        "checkpoint": "Cnn14_mAP=0.431.pth",
        "sample_rate": 32000,
        "embedding_dim": 2048,
        "license": "Apache 2.0 (code) / MIT (weights)",
        "download_size_mb": 312.3,
        "pretraining_data_quote": (
            "PANNs CNN14 was trained on Google AudioSet consisting of 527 audio classes, "
            "1,934,187 audio clips, and 5,234 hours of audio from YouTube. "
            "(Kong et al., 'PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition', IEEE/ACM TASLP 2020)."
        ),
        "genre_overlap": "Direct overlap: AudioSet ontology explicitly includes multiple musical genres (e.g. Pop, Rock, Classical, Jazz, Blues, Reggae, Hip hop, Country).",
    },
    "MERT-v1-95M": {
        "display_name": "MERT-v1-95M",
        "checkpoint": "m-a-p/MERT-v1-95M",
        "sample_rate": 24000,
        "embedding_dim": 768,
        "license": "CC BY-NC-SA 4.0 / Research",
        "download_size_mb": 360.5,
        "pretraining_data_quote": (
            "MERT-v1-95M was pretrained on 160k hours of unlabelled music audio collected from various sources, "
            "including 1000 hours of open-source music (Free Music Archive, Jamendo) and 160,000 hours of web music crawled from YouTube and other platforms. "
            "(Li et al., 'MERT: Acoustic Music Understanding with Large-Scale Self-supervised Training', ICLR 2024)."
        ),
        "genre_overlap": "Potential indirect overlap: Pretrained on 160k hours of music audio without explicit supervised genre labels, but web crawl covers all popular music styles and commercial recordings.",
    },
    "VGGish": {
        "display_name": "VGGish",
        "checkpoint": "vggish_model.ckpt",
        "sample_rate": 16000,
        "embedding_dim": 256, # 128 mean + 128 std
        "license": "Apache 2.0",
        "download_size_mb": 278.0,
        "pretraining_data_quote": (
            "VGGish was pretrained on a YouTube-8M dataset consisting of ~70 million video segments (2.6 billion seconds of audio) "
            "labeled with 30,871 video-level topical tags, including hundreds of musical genre and instrument entities. "
            "(Hershey et al., 'CNN Architectures for Large-Scale Audio Classification', ICASSP 2017)."
        ),
        "genre_overlap": "Direct overlap: YouTube-8M topical tags encompass detailed music genres and categories.",
    },
}

# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
def load_gtzan_metadata():
    """Load clean GTZAN metadata (929 clean tracks, artists, genres)."""
    gtzan_csv = resolve_file("gtzan_features.csv", os.path.join("data", "gtzan_features.csv"))
    rem_csv = resolve_file("p1_gtzan_removed_tracks.csv", os.path.join("experiments", "rev2_p1_splits", "removed_tracks.csv"))
    artist_map_csv = resolve_file("p1_gtzan_artist_map.csv", os.path.join("experiments", "rev2_p1_splits", "artist_map.csv"))
    
    df_g = pd.read_csv(gtzan_csv).set_index("filename")
    rem_tracks = set(pd.read_csv(rem_csv)["filename"])
    df_g_clean = df_g.loc[~df_g.index.isin(rem_tracks)].copy()
    
    df_am = pd.read_csv(artist_map_csv)
    am_dict = dict(zip(df_am["track_id"], df_am["artist"]))
    df_g_clean["artist_id"] = [am_dict.get(fn, "unknown") for fn in df_g_clean.index]
    
    with open(SPLITS_JSON, "r") as f:
        splits_data = json.load(f)
    all_folds = splits_data["gtzan"]["fault_removed_artist_grouped"] # 15 folds
    return df_g_clean, all_folds


def load_pmemo_metadata():
    """Load clean PMEmo metadata (767 tracks, artists, valence, arousal)."""
    pmemo_csv = resolve_file("pmemo_ready.csv", os.path.join("data", "pmemo_ready.csv"))
    artist_map_csv = resolve_file("p1_gtzan_artist_map.csv", os.path.join("experiments", "rev2_p1_splits", "artist_map.csv"))
    
    df_p = pd.read_csv(pmemo_csv)
    df_p["musicId_str"] = df_p["musicId"].astype(str)
    df_p.set_index("musicId_str", inplace=True)
    
    df_am = pd.read_csv(artist_map_csv)
    pmemo_am = df_am[df_am["dataset"] == "PMEmo"].copy()
    pmemo_am["track_id"] = pmemo_am["track_id"].astype(str)
    am_dict = dict(zip(pmemo_am["track_id"], pmemo_am["artist"]))
    df_p["artist_id"] = [am_dict.get(m, "unknown") for m in df_p.index]
    
    with open(SPLITS_JSON, "r") as f:
        splits_data = json.load(f)
    all_folds = [f for f in splits_data["pmemo"]["artist_grouped_5x5"] if f["repeat"] in (0, 1)] # 10 folds
    return df_p, all_folds

# ---------------------------------------------------------------------------
# Statistical Significance: Nadeau-Bengio Corrected Resampled t-test
# ---------------------------------------------------------------------------
def nadeau_bengio_test(diffs: np.ndarray, test_train_ratio: float = 0.25):
    """
    Nadeau-Bengio (2003) corrected resampled t-test for repeated K-fold CV.
    Variance correction factor: 1/K + n_test / n_train = 1/5 + 0.25 = 0.45.
    """
    K = len(diffs)
    mean_diff = float(np.mean(diffs))
    sample_var = float(np.var(diffs, ddof=1)) if K > 1 else 0.0
    
    correction = (1.0 / K) + test_train_ratio
    corrected_var = correction * sample_var
    corrected_se = float(np.sqrt(corrected_var))
    
    if corrected_se < 1e-12:
        t_stat = 0.0
        p_val = 1.0
        ci_lower = mean_diff
        ci_upper = mean_diff
    else:
        t_stat = mean_diff / corrected_se
        df = K - 1
        p_val = float(2.0 * scipy_stats.t.sf(np.abs(t_stat), df=df))
        t_crit = float(scipy_stats.t.ppf(0.975, df=df))
        ci_lower = mean_diff - t_crit * corrected_se
        ci_upper = mean_diff + t_crit * corrected_se
        
    return {
        "mean_diff": mean_diff,
        "sample_sd": float(np.std(diffs, ddof=1)) if K > 1 else 0.0,
        "corrected_se": corrected_se,
        "t_stat": t_stat,
        "df": K - 1,
        "p_val": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

# ---------------------------------------------------------------------------
# Environment Logger
# ---------------------------------------------------------------------------
def log_environment(output_dir: str) -> dict:
    import torch
    import torchaudio
    import transformers
    import lightgbm
    import sklearn
    import librosa
    
    env = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version,
        "cpu": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "packages": {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": transformers.__version__,
            "lightgbm": lightgbm.__version__,
            "sklearn": sklearn.__version__,
            "librosa": librosa.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    env_file = os.path.join(output_dir, "environment.json")
    with open(env_file, "w") as f:
        json.dump(env, f, indent=2)
    return env
