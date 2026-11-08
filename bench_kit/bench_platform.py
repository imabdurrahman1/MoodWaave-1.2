#!/usr/bin/env python3
"""
release/bench_kit/bench_platform.py

MOODWAAVE Platform Benchmark Kit: Comprehensive Single-Thread Profiler
Profiles MOODWAAVE end-to-end efficiency across hardware platforms (x86_64, Windows, Linux, aarch64/ARM)
and DSP libraries (Librosa vs. Essentia / TorchAudio + nnAudio).

Strict Timing Protocol:
- Enforce single thread across all backends:
  OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
  NUMBA_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, torch.set_num_threads(1).
- 1 warm-up clip excluded.
- Median of 3 repeats per clip.
- 50 GTZAN WAV clips (30.0s, 22050 Hz).
- Timed regions: Audio loading, feature extraction (per-group + full pipeline), and LightGBM predictions.
"""

import os
# Strict single-threading enforcement before importing numpy, torch, librosa
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import sys
import time
import json
import argparse
import platform
import warnings
import psutil
import numpy as np
import soundfile as sf
import joblib

warnings.filterwarnings("ignore")

import librosa
import torch
torch.set_num_threads(1)

# Optional second libraries
try:
    import torchaudio
    import torchaudio.transforms as T
    from nnAudio import features as nn_features
    HAS_TORCH_DSP = True
except ImportError:
    HAS_TORCH_DSP = False

try:
    import essentia
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    HAS_ESSENTIA = False

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AUDIO_DIR = os.path.join(HERE, "audio_50")
DEFAULT_MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_OUT_JSON = os.path.join(HERE, "results", f"{platform.system().lower()}_{platform.machine().lower()}.json")

# Harte 6x12 projection matrix for Tonnetz computation
def build_harte_projection():
    dim_map = np.linspace(0, 12, num=12, endpoint=False)
    scale = np.asarray([7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3])
    V = np.multiply.outer(scale, dim_map)
    V[::2] -= 0.5
    R = np.array([1, 1, 1, 1, 0.5, 0.5])
    return R[:, np.newaxis] * np.cos(np.pi * V)

HARTE_PROJECTION_NP = build_harte_projection()
HARTE_PROJECTION_TORCH = torch.tensor(HARTE_PROJECTION_NP, dtype=torch.float32)

def get_cpu_model_name():
    """Detect CPU model string across Windows, Linux, and macOS/ARM."""
    # 1. Windows WMIC
    if platform.system() == "Windows":
        try:
            import subprocess
            out = subprocess.check_output("wmic cpu get name", shell=True, text=True, stderr=subprocess.DEVNULL).strip()
            lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
            if lines:
                return lines[0]
        except Exception:
            pass
            
    # 2. Linux /proc/cpuinfo
    if platform.system() == "Linux" and os.path.isfile("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
            
    # 3. macOS sysctl
    if platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
            if out:
                return out
        except Exception:
            pass
            
    # Fallback
    proc = platform.processor()
    if proc:
        return proc
    return platform.machine()

def collect_platform_info():
    """Collect comprehensive hardware and library metadata."""
    info = {
        "project": "MOODWAAVE",
        "cpu_model": get_cpu_model_name(),
        "architecture": platform.machine(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "single_thread_enforced": True,
        "libraries": {
            "numpy": np.__version__,
            "librosa": librosa.__version__,
            "soundfile": sf.__version__,
            "joblib": joblib.__version__,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__ if HAS_TORCH_DSP else None,
            "nnAudio": "0.3.4" if HAS_TORCH_DSP else None,
            "essentia": essentia.__version__ if HAS_ESSENTIA else None,
        }
    }
    return info

def load_saved_models(models_dir):
    """Load the pre-trained MOODWAAVE LightGBM models for pipelines a, b, c."""
    print(f"Loading MOODWAAVE models from: {models_dir}")
    models = {}
    
    # Pipe a: Original-113
    pa_g = joblib.load(os.path.join(models_dir, "pipe_a_genre_orig113.pkl"))
    pa_v = joblib.load(os.path.join(models_dir, "pipe_a_val_orig113.pkl"))
    pa_a = joblib.load(os.path.join(models_dir, "pipe_a_aro_orig113.pkl"))
    models["pipe_a"] = (pa_g, pa_v, pa_a)
    
    # Pipe b: Full-113 V1
    pb_g = joblib.load(os.path.join(models_dir, "pipe_b_genre_full113_v1.pkl"))
    pb_v = joblib.load(os.path.join(models_dir, "pipe_b_val_full113_v1.pkl"))
    pb_a = joblib.load(os.path.join(models_dir, "pipe_b_aro_full113_v1.pkl"))
    models["pipe_b"] = (pb_g, pb_v, pb_a)
    
    # Pipe c: Lite-71
    pc_g = joblib.load(os.path.join(models_dir, "pipe_c_genre_lite71.pkl"))
    pc_v = joblib.load(os.path.join(models_dir, "pipe_c_val_lite71.pkl"))
    pc_a = joblib.load(os.path.join(models_dir, "pipe_c_aro_lite71.pkl"))
    models["pipe_c"] = (pc_g, pc_v, pc_a)
    
    print("All 9 MOODWAAVE prediction models loaded successfully.")
    return models

# -------------------------------------------------------------------------
# Librosa Feature Extraction & Pipeline Timing
# -------------------------------------------------------------------------
def profile_librosa_on_clip(clip_path, models, repeats=3):
    """Run repeats on one clip with Librosa, returning median timings."""
    repeats_load = []
    repeats_groups = []
    repeats_pipe_a = []
    repeats_pipe_b = []
    repeats_pipe_c = []
    
    feats_113_dummy = np.zeros((1, 113), dtype=np.float32)
    feats_71_dummy = np.zeros((1, 71), dtype=np.float32)
    
    for _ in range(repeats):
        # 1. Timed Audio Loading
        t0 = time.perf_counter()
        y, sr = librosa.load(clip_path, sr=22050, mono=True, duration=30.0)
        t_load = (time.perf_counter() - t0) * 1000.0
        repeats_load.append(t_load)
        
        # 2. Per-Group Feature Extraction
        # MFCC
        t0 = time.perf_counter()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        d_mfcc = librosa.feature.delta(mfcc)
        t_mfcc = (time.perf_counter() - t0) * 1000.0
        
        # Chroma STFT
        t0 = time.perf_counter()
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        t_chroma = (time.perf_counter() - t0) * 1000.0
        
        # Tonnetz V0 (Harmonic HPSS + CQT)
        t0 = time.perf_counter()
        y_harm = librosa.effects.harmonic(y)
        tonnetz_v0 = librosa.feature.tonnetz(y=y_harm, sr=sr)
        t_tonnetz_v0 = (time.perf_counter() - t0) * 1000.0
        
        # Tonnetz V1 (from STFT Chroma)
        t0 = time.perf_counter()
        tonnetz_v1 = librosa.feature.tonnetz(chroma=chroma_stft)
        t_tonnetz_v1 = (time.perf_counter() - t0) * 1000.0
        
        # Spectral (centroid, rolloff, bandwidth)
        t0 = time.perf_counter()
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        t_spectral = (time.perf_counter() - t0) * 1000.0
        
        # Spectral Contrast
        t0 = time.perf_counter()
        scont = librosa.feature.spectral_contrast(y=y, sr=sr)
        t_scont = (time.perf_counter() - t0) * 1000.0
        
        # Temporal (ZCR, RMS, Tempo)
        t0 = time.perf_counter()
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        t_temporal = (time.perf_counter() - t0) * 1000.0
        
        repeats_groups.append({
            "mfcc_ms": t_mfcc, "chroma_ms": t_chroma,
            "tonnetz_v0_ms": t_tonnetz_v0, "tonnetz_v1_ms": t_tonnetz_v1,
            "spectral_ms": t_spectral, "contrast_ms": t_scont,
            "temporal_ms": t_temporal
        })
        
        # 3. Timed Predictions
        # Pipe a (Original-113)
        t0 = time.perf_counter()
        _ = models["pipe_a"][0].predict(feats_113_dummy)
        _ = models["pipe_a"][1].predict(feats_113_dummy)
        _ = models["pipe_a"][2].predict(feats_113_dummy)
        t_pred_a = (time.perf_counter() - t0) * 1000.0
        
        # Pipe b (Full-113 V1)
        t0 = time.perf_counter()
        _ = models["pipe_b"][0].predict(feats_113_dummy)
        _ = models["pipe_b"][1].predict(feats_113_dummy)
        _ = models["pipe_b"][2].predict(feats_113_dummy)
        t_pred_b = (time.perf_counter() - t0) * 1000.0
        
        # Pipe c (Lite-71)
        t0 = time.perf_counter()
        _ = models["pipe_c"][0].predict(feats_71_dummy)
        _ = models["pipe_c"][1].predict(feats_71_dummy)
        _ = models["pipe_c"][2].predict(feats_71_dummy)
        t_pred_c = (time.perf_counter() - t0) * 1000.0
        
        # End-to-end pipelines
        feat_a = t_mfcc + t_chroma + t_tonnetz_v0 + t_spectral + t_scont + t_temporal
        total_a = t_load + feat_a + t_pred_a
        repeats_pipe_a.append({"feat_ms": feat_a, "pred_ms": t_pred_a, "total_ms": total_a})
        
        feat_b = t_mfcc + t_chroma + t_tonnetz_v1 + t_spectral + t_scont + t_temporal
        total_b = t_load + feat_b + t_pred_b
        repeats_pipe_b.append({"feat_ms": feat_b, "pred_ms": t_pred_b, "total_ms": total_b})
        
        feat_c = t_mfcc + t_scont + t_temporal
        total_c = t_load + feat_c + t_pred_c
        repeats_pipe_c.append({"feat_ms": feat_c, "pred_ms": t_pred_c, "total_ms": total_c})
        
    return {
        "audio_load_ms": float(np.median(repeats_load)),
        "mfcc_ms": float(np.median([g["mfcc_ms"] for g in repeats_groups])),
        "chroma_ms": float(np.median([g["chroma_ms"] for g in repeats_groups])),
        "tonnetz_v0_ms": float(np.median([g["tonnetz_v0_ms"] for g in repeats_groups])),
        "tonnetz_v1_ms": float(np.median([g["tonnetz_v1_ms"] for g in repeats_groups])),
        "spectral_ms": float(np.median([g["spectral_ms"] for g in repeats_groups])),
        "contrast_ms": float(np.median([g["contrast_ms"] for g in repeats_groups])),
        "temporal_ms": float(np.median([g["temporal_ms"] for g in repeats_groups])),
        "pipe_a": {
            "feat_ms": float(np.median([p["feat_ms"] for p in repeats_pipe_a])),
            "pred_ms": float(np.median([p["pred_ms"] for p in repeats_pipe_a])),
            "total_ms": float(np.median([p["total_ms"] for p in repeats_pipe_a])),
        },
        "pipe_b": {
            "feat_ms": float(np.median([p["feat_ms"] for p in repeats_pipe_b])),
            "pred_ms": float(np.median([p["pred_ms"] for p in repeats_pipe_b])),
            "total_ms": float(np.median([p["total_ms"] for p in repeats_pipe_b])),
        },
        "pipe_c": {
            "feat_ms": float(np.median([p["feat_ms"] for p in repeats_pipe_c])),
            "pred_ms": float(np.median([p["pred_ms"] for p in repeats_pipe_c])),
            "total_ms": float(np.median([p["total_ms"] for p in repeats_pipe_c])),
        }
    }

# -------------------------------------------------------------------------
# Second Library Feature Extraction (TorchAudio + nnAudio OR Essentia)
# -------------------------------------------------------------------------
def profile_second_library_on_clip(clip_path, repeats=3):
    """
    Benchmarks feature extraction using a second DSP library.
    Where Essentia is available, Essentia algorithms are evaluated.
    Otherwise (e.g. Windows), TorchAudio + nnAudio equivalents are evaluated.
    """
    if HAS_ESSENTIA:
        return profile_essentia_on_clip(clip_path, repeats=repeats)
    elif HAS_TORCH_DSP:
        return profile_torch_dsp_on_clip(clip_path, repeats=repeats)
    else:
        return None

def profile_torch_dsp_on_clip(clip_path, repeats=3):
    """Benchmarks TorchAudio + nnAudio analogue implementations."""
    repeats_load = []
    repeats_mfcc = []
    repeats_chroma = []
    repeats_tonnetz = []
    repeats_spectral = []
    repeats_temporal = []
    
    sr_target = 22050
    mfcc_tf = T.MFCC(sample_rate=sr_target, n_mfcc=13, melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128})
    sc_tf = T.SpectralCentroid(sample_rate=sr_target, n_fft=2048, hop_length=512)
    cqt_layer = nn_features.cqt.CQT2010v2(sr=sr_target, fmin=32.7, n_bins=84, bins_per_octave=12, hop_length=512, verbose=False)
    
    for _ in range(repeats):
        # 1. Timed Audio Load
        t0 = time.perf_counter()
        data, sr = sf.read(clip_path, dtype='float32')
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.mean(dim=-1, keepdim=True).T
        max_samples = 30 * sr_target
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        t_load = (time.perf_counter() - t0) * 1000.0
        repeats_load.append(t_load)
        
        # 2. MFCC via torchaudio
        t0 = time.perf_counter()
        mfcc = mfcc_tf(waveform)
        t_mfcc = (time.perf_counter() - t0) * 1000.0
        repeats_mfcc.append(t_mfcc)
        
        # 3. Spectral Centroid via torchaudio
        t0 = time.perf_counter()
        sc = sc_tf(waveform)
        t_spec = (time.perf_counter() - t0) * 1000.0
        repeats_spectral.append(t_spec)
        
        # 4. Temporal (RMS & ZCR) via PyTorch
        t0 = time.perf_counter()
        rms = torch.sqrt(torch.mean(waveform**2, dim=-1))
        zcr = torch.mean(((waveform[:, :-1] * waveform[:, 1:]) < 0).float())
        t_temp = (time.perf_counter() - t0) * 1000.0
        repeats_temporal.append(t_temp)
        
        # 5. CQT Chroma via nnAudio
        t0 = time.perf_counter()
        cqt = cqt_layer(waveform)
        chroma = cqt.view(1, 7, 12, -1).sum(dim=1).squeeze(0) # 12 x time
        t_chroma = (time.perf_counter() - t0) * 1000.0
        repeats_chroma.append(t_chroma)
        
        # 6. Tonnetz via Harte 6x12 projection from CQT Chroma
        t0 = time.perf_counter()
        chroma_norm = chroma / (torch.sum(chroma, dim=0, keepdim=True) + 1e-8)
        tonnetz = torch.matmul(HARTE_PROJECTION_TORCH, chroma_norm)
        t_tonnetz = (time.perf_counter() - t0) * 1000.0
        repeats_tonnetz.append(t_tonnetz)
        
    return {
        "library": "TorchAudio_nnAudio",
        "audio_load_ms": float(np.median(repeats_load)),
        "mfcc_ms": float(np.median(repeats_mfcc)),
        "chroma_ms": float(np.median(repeats_chroma)),
        "tonnetz_ms": float(np.median(repeats_tonnetz)),
        "spectral_ms": float(np.median(repeats_spectral)),
        "contrast_ms": None,  # Not available natively in torchaudio
        "temporal_ms": float(np.median(repeats_temporal)),
    }

def profile_essentia_on_clip(clip_path, repeats=3):
    """Benchmarks Essentia native algorithms."""
    loader = es.MonoLoader(filename=clip_path, sampleRate=22050)
    repeats_load = []
    repeats_mfcc = []
    repeats_chroma = []
    repeats_tonnetz = []
    repeats_scont = []
    repeats_temp = []
    
    for _ in range(repeats):
        t0 = time.perf_counter()
        audio = loader()
        t_load = (time.perf_counter() - t0) * 1000.0
        repeats_load.append(t_load)
        
        # Essentia MFCC
        t0 = time.perf_counter()
        w = es.Windowing(type='hann')
        spec = es.Spectrum()
        mfcc_algo = es.MFCC(numberCoefficients=13)
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
            _, _ = mfcc_algo(spec(w(frame)))
        t_mfcc = (time.perf_counter() - t0) * 1000.0
        repeats_mfcc.append(t_mfcc)
        
        # Essentia HPCP (Chroma analogue)
        t0 = time.perf_counter()
        hpcp_algo = es.HPCP(size=12)
        hpcp_list = []
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
            hpcp_list.append(hpcp_algo(spec(w(frame))))
        hpcp_mat = np.array(hpcp_list).T # 12 x time
        t_chroma = (time.perf_counter() - t0) * 1000.0
        repeats_chroma.append(t_chroma)
        
        # Tonnetz from HPCP using Harte projection
        t0 = time.perf_counter()
        h_norm = hpcp_mat / (np.sum(hpcp_mat, axis=0, keepdims=True) + 1e-8)
        tonnetz = np.dot(HARTE_PROJECTION_NP, h_norm)
        t_tonnetz = (time.perf_counter() - t0) * 1000.0
        repeats_tonnetz.append(t_tonnetz)
        
        # Essentia Temporal (ZCR, RMS)
        t0 = time.perf_counter()
        _ = es.ZeroCrossingRate()(audio)
        _ = es.RMS()(audio)
        _ = es.RhythmExtractor2013()(audio)
        t_temp = (time.perf_counter() - t0) * 1000.0
        repeats_temp.append(t_temp)
        
    return {
        "library": "Essentia",
        "audio_load_ms": float(np.median(repeats_load)),
        "mfcc_ms": float(np.median(repeats_mfcc)),
        "chroma_ms": float(np.median(repeats_chroma)),
        "tonnetz_ms": float(np.median(repeats_tonnetz)),
        "spectral_ms": None,
        "contrast_ms": None,
        "temporal_ms": float(np.median(repeats_temp)),
    }

# -------------------------------------------------------------------------
# Main Benchmark Execution Routine
# -------------------------------------------------------------------------
def run_benchmark(audio_dir, out_path, models_dir=DEFAULT_MODELS_DIR, count=50, repeats=3):
    print("=================================================================")
    print("MOODWAAVE Platform & Library Benchmark Suite")
    print(f"System: {platform.system()} | Machine: {platform.machine()}")
    print("Strict Protocol: Single-Thread (threads=1), 1 warm-up excluded, median of 3 repeats.")
    print("=================================================================\n")
    
    t_start = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    platform_info = collect_platform_info()
    models = load_saved_models(models_dir)
    
    # Locate WAV files
    wav_files = []
    manifest_path = os.path.join(audio_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
            for item in man:
                fname = item["filename"]
                p = os.path.join(audio_dir, fname)
                if os.path.isfile(p):
                    wav_files.append(p)
    if not wav_files:
        for f in sorted(os.listdir(audio_dir)):
            if f.endswith(".wav"):
                wav_files.append(os.path.join(audio_dir, f))
                
    print(f"Discovered {len(wav_files)} WAV clips in {audio_dir}")
    if len(wav_files) < count:
        raise ValueError(f"Need at least {count} clips, found {len(wav_files)} in {audio_dir}")
        
    benchmark_clips = wav_files[:count]
    warmup_clip = benchmark_clips[0]
    eval_clips = benchmark_clips[1:count] # 49 clips evaluated after warm-up
    print(f"Warm-up clip (excluded from metrics): {os.path.basename(warmup_clip)}")
    print(f"Benchmark clips to evaluate: {len(eval_clips)} (Repeats per clip: {repeats})\n")
    
    # 1. Warm-up run
    print("Executing warm-up run on warm-up clip...")
    _ = profile_librosa_on_clip(warmup_clip, models, repeats=1)
    if HAS_TORCH_DSP or HAS_ESSENTIA:
        _ = profile_second_library_on_clip(warmup_clip, repeats=1)
    print("Warm-up complete. Starting benchmark measurement.\n")
    
    # 2. Benchmark Librosa & Pipelines
    librosa_results = []
    second_lib_results = []
    
    for idx, clip in enumerate(eval_clips):
        cname = os.path.basename(clip)
        res_lib = profile_librosa_on_clip(clip, models, repeats=repeats)
        res_lib["filename"] = cname
        res_lib["index"] = idx + 1
        librosa_results.append(res_lib)
        
        if HAS_TORCH_DSP or HAS_ESSENTIA:
            res_sec = profile_second_library_on_clip(clip, repeats=repeats)
            if res_sec is not None:
                res_sec["filename"] = cname
                res_sec["index"] = idx + 1
                second_lib_results.append(res_sec)
                
        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_clips):
            pa_tot = res_lib["pipe_a"]["total_ms"]
            pc_tot = res_lib["pipe_c"]["total_ms"]
            speedup = pa_tot / pc_tot
            print(f"[{idx + 1}/{len(eval_clips)}] {cname} | Orig-113: {pa_tot:.1f}ms | Lite-71: {pc_tot:.1f}ms (Speedup: {speedup:.2f}x)")
            
    # Compute Summary Aggregations
    def summarize_series(vals):
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        
    librosa_summary = {
        "audio_load_ms": summarize_series([r["audio_load_ms"] for r in librosa_results]),
        "mfcc_ms": summarize_series([r["mfcc_ms"] for r in librosa_results]),
        "chroma_ms": summarize_series([r["chroma_ms"] for r in librosa_results]),
        "tonnetz_v0_ms": summarize_series([r["tonnetz_v0_ms"] for r in librosa_results]),
        "tonnetz_v1_ms": summarize_series([r["tonnetz_v1_ms"] for r in librosa_results]),
        "spectral_ms": summarize_series([r["spectral_ms"] for r in librosa_results]),
        "contrast_ms": summarize_series([r["contrast_ms"] for r in librosa_results]),
        "temporal_ms": summarize_series([r["temporal_ms"] for r in librosa_results]),
        "pipe_a_orig113": {
            "features_ms": summarize_series([r["pipe_a"]["feat_ms"] for r in librosa_results]),
            "prediction_ms": summarize_series([r["pipe_a"]["pred_ms"] for r in librosa_results]),
            "total_ms": summarize_series([r["pipe_a"]["total_ms"] for r in librosa_results]),
            "rtf": float(np.mean([r["pipe_a"]["total_ms"] for r in librosa_results])) / 30000.0,
            "tonnetz_share_pct": float(np.mean([r["tonnetz_v0_ms"] / r["pipe_a"]["total_ms"] * 100.0 for r in librosa_results])),
            "speedup_vs_orig": 1.0,
        },
        "pipe_b_full113_v1": {
            "features_ms": summarize_series([r["pipe_b"]["feat_ms"] for r in librosa_results]),
            "prediction_ms": summarize_series([r["pipe_b"]["pred_ms"] for r in librosa_results]),
            "total_ms": summarize_series([r["pipe_b"]["total_ms"] for r in librosa_results]),
            "rtf": float(np.mean([r["pipe_b"]["total_ms"] for r in librosa_results])) / 30000.0,
            "speedup_vs_orig": float(np.mean([r["pipe_a"]["total_ms"] for r in librosa_results])) / float(np.mean([r["pipe_b"]["total_ms"] for r in librosa_results])),
        },
        "pipe_c_lite71": {
            "features_ms": summarize_series([r["pipe_c"]["feat_ms"] for r in librosa_results]),
            "prediction_ms": summarize_series([r["pipe_c"]["pred_ms"] for r in librosa_results]),
            "total_ms": summarize_series([r["pipe_c"]["total_ms"] for r in librosa_results]),
            "rtf": float(np.mean([r["pipe_c"]["total_ms"] for r in librosa_results])) / 30000.0,
            "speedup_vs_orig": float(np.mean([r["pipe_a"]["total_ms"] for r in librosa_results])) / float(np.mean([r["pipe_c"]["total_ms"] for r in librosa_results])),
        }
    }
    
    second_lib_summary = None
    if second_lib_results:
        lib_name = second_lib_results[0]["library"]
        second_lib_summary = {
            "library": lib_name,
            "status": "analogue_features",
            "covered_groups": ["audio_load", "mfcc", "chroma", "tonnetz", "temporal", "spectral (centroid)"],
            "uncovered_groups": ["spectral_contrast (no native op)", "tempo (no native beat tracker)"],
            "notes": "Features are acoustic analogues for latency comparison, not identical to Librosa.",
            "audio_load_ms": summarize_series([r["audio_load_ms"] for r in second_lib_results]),
            "mfcc_ms": summarize_series([r["mfcc_ms"] for r in second_lib_results]),
            "chroma_ms": summarize_series([r["chroma_ms"] for r in second_lib_results]),
            "tonnetz_ms": summarize_series([r["tonnetz_ms"] for r in second_lib_results]),
            "spectral_ms": summarize_series([r["spectral_ms"] for r in second_lib_results]),
            "temporal_ms": summarize_series([r["temporal_ms"] for r in second_lib_results]),
        }
        
    full_benchmark_output = {
        "platform": platform_info,
        "benchmark_config": {
            "audio_dir": os.path.abspath(audio_dir),
            "num_evaluated_clips": len(eval_clips),
            "warmup_clip": os.path.basename(warmup_clip),
            "repeats": repeats,
            "sample_rate": 22050,
            "clip_duration_s": 30.0,
            "single_thread": True,
        },
        "librosa": {
            "summary": librosa_summary,
            "clips": librosa_results,
        },
        "second_library": {
            "summary": second_lib_summary,
            "clips": second_lib_results,
        },
        "elapsed_benchmark_seconds": round(time.perf_counter() - t_start, 2)
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_benchmark_output, f, indent=2)
        
    print(f"\n=================================================================")
    print(f"Benchmark finished in {full_benchmark_output['elapsed_benchmark_seconds']:.2f} seconds.")
    print(f"Results saved to: {out_path}")
    print(f"Original-113 Latency: {librosa_summary['pipe_a_orig113']['total_ms']['mean']:.1f} ms (RTF: {librosa_summary['pipe_a_orig113']['rtf']:.4f})")
    print(f"Full-113 V1 Latency:  {librosa_summary['pipe_b_full113_v1']['total_ms']['mean']:.1f} ms (Speedup: {librosa_summary['pipe_b_full113_v1']['speedup_vs_orig']:.2f}x)")
    print(f"MOODWAAVE-Lite-71:   {librosa_summary['pipe_c_lite71']['total_ms']['mean']:.1f} ms (Speedup: {librosa_summary['pipe_c_lite71']['speedup_vs_orig']:.2f}x, RTF: {librosa_summary['pipe_c_lite71']['rtf']:.4f})")
    print(f"Tonnetz share of V0:  {librosa_summary['pipe_a_orig113']['tonnetz_share_pct']:.1f}%")
    print("=================================================================")

def main():
    parser = argparse.ArgumentParser(description="MOODWAAVE Platform and Library Benchmark Kit.")
    parser.add_argument("--audio-dir", type=str, default=DEFAULT_AUDIO_DIR, help="Directory containing GTZAN audio clips")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_JSON, help="Path for JSON output")
    parser.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR, help="Directory containing LightGBM models")
    parser.add_argument("--count", type=int, default=50, help="Number of clips to benchmark (default: 50)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per clip (default: 3)")
    args = parser.parse_args()
    
    run_benchmark(audio_dir=args.audio_dir, out_path=args.out, models_dir=args.models_dir, count=args.count, repeats=args.repeats)

if __name__ == "__main__":
    main()

