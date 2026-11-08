# TASK R3-D: MOODWAAVE Platform and Library Benchmark Kit

**System:** MOODWAAVE (Lightweight CPU-Only Music Analysis System)  
**Task:** Turnkey platform and library benchmarking kit with cross-architecture portability and DSP library comparisons.  
**Target Directory:** `release/bench_kit/`  
**Test Host Platform:** 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4 physical cores, 8 logical cores, 7.75 GB RAM, Windows 11 x86_64)  
**Corpus:** 50 canonical GTZAN WAV excerpts (30.0s, 22,050 Hz mono)  
**Measurement Protocol:** Strict single thread (`OMP/MKL/OPENBLAS/NUMBA/torch = 1`), 1 warm-up clip excluded, median of 3 repeats across 49 evaluated clips.

---

## 1. Executive Summary & Deliverables

Task R3-D packages a fully reproducible, self-contained benchmarking suite designed to verify MOODWAAVE's computational efficiency across different hardware platforms (Intel/AMD x86_64, Linux servers, and ARM/aarch64 devices) and DSP libraries (`librosa` vs. `essentia` / `torchaudio` + `nnAudio`). The kit is designed to be portable across Windows, Linux and ARM (aarch64); tested on Windows only.

### Delivered Artifacts
1. **`export_bench_clips.py`**: Automatically extracts the canonical 50 GTZAN WAV clips from repository storage to `audio_50/` and creates a deterministic manifest.
2. **`bench_platform.py`**: Self-contained profiler that records hardware architecture, system metadata, audio decoding, per-feature-group costs, end-to-end pipeline latencies (Original-113, Full-113 V1, Lite-71), and Real-Time Factors (RTF).
3. **Second Library Integration (`torchaudio` + `nnAudio` / `essentia`)**: Evaluates PyTorch and nnAudio implementations of MFCC, Spectral Centroid, RMS, ZCR, and CQT Chroma + Tonnetz via Harte projection.
4. **`aggregate_results.py`**: Parses all `results/*.json` reports to generate consolidated cross-platform tables (`bench_summary_table.csv`, `group_comparison_table.csv`) and a 300-DPI publication figure (`bench_platform_latency_and_breakdown.png`).
5. **`BENCH_README.md`**: Complete step-by-step instructions for running the benchmark on any new machine.

---

## 2. Platform & Hardware Environment

- **Processor:** 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (Family 6 Model 140 Stepping 1, GenuineIntel)
- **Architecture:** x86_64 / AMD64
- **Cores:** 4 Physical, 8 Logical
- **Memory:** 7.75 GB RAM
- **Operating System:** Windows 11 Home (Build 10.0.26200)
- **Python Version:** 3.13.7 (MSC v.1944 64-bit)
- **DSP / ML Libraries:**
  - `numpy`: 2.4.3
  - `librosa`: 0.11.0
  - `soundfile`: 0.13.1
  - `lightgbm`: 4.6.0
  - `torch`: 2.15.0+cpu
  - `torchaudio`: 2.11.1+cpu
  - `nnAudio`: 0.3.4
  - `essentia`: Not available natively on Windows (no PyPI wheels for Python 3.13; WSL2 distribution absent on host). Fallback to `torchaudio` + `nnAudio` executed as specified.

---

## 3. Platform Benchmark Results

### Master Pipeline Latency Table (Single-Thread CPU, 50 GTZAN Clips)

| Platform | Library | Pipeline | Audio Load (ms) | Features (ms) | Predictions (ms) | Total Latency (ms) | Real-Time Factor (RTF) | Speedup vs. Orig | Tonnetz Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Intel Core i5-1135G7 (Win11)** | `librosa` | **Original-113 (Tonnetz V0)** | $3.64 \pm 0.81$ | $2763.81 \pm 312.4$ | $2.63 \pm 0.05$ | **$2770.08 \pm 312.8$** | $0.0923$ | $1.00\times$ | **$78.57\%$** |
| Intel Core i5-1135G7 (Win11) | `librosa` | **Full-113 (Tonnetz V1)** | $3.64 \pm 0.81$ | $557.93 \pm 42.1$ | $2.30 \pm 0.06$ | **$563.87 \pm 42.4$** | $0.0188$ | **$4.91\times$** | $0.07\%$ |
| Intel Core i5-1135G7 (Win11) | `librosa` | **MOODWAAVE-Lite-71** | $3.64 \pm 0.81$ | $227.68 \pm 18.5$ | $2.40 \pm 0.29$ | **$233.73 \pm 18.7$** | **$0.0078$** | **$11.85\times$** | $0.00\%$ |
| Intel Core i5-1135G7 (Win11) | `TorchAudio_nnAudio` | Full Analogue (MFCC+Chroma+Ton+Spec+Temp) | $2.40 \pm 0.15$ | $1462.45 \pm 28.6$ | $2.30 \pm 0.06$ | **$1467.15 \pm 28.7$** | $0.0489$ | $1.89\times$ | $0.01\%$ |
| Intel Core i5-1135G7 (Win11) | `TorchAudio_nnAudio` | Lite Analogue (MFCC+Temp only, no contrast/tempo) | $2.40 \pm 0.15$ | $40.64 \pm 1.8$ | $2.40 \pm 0.29$ | **$45.44 \pm 2.0$** | **$0.0015$** | Non-comparable (lacks 19 features) | $0.00\%$ |

---

## 4. Feature Group Latency Comparison across Libraries

| Feature Group | Librosa Latency (ms) | TorchAudio / nnAudio Latency (ms) | Speedup / Ratio | Operational & Comparability Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Audio Loading (WAV)** | $3.64\text{ ms}$ | $2.40\text{ ms}$ | $1.52\times$ | Uncompressed 30s 22.05 kHz WAV (`soundfile` vs `torchaudio.load`) |
| **MFCC** | $50.54\text{ ms}$ | $36.20\text{ ms}$ | $1.40\times$ | Comparable: 13 coefficients + deltas (TorchAudio `MFCC`) |
| **Tonnetz V1 (Projection)** | **$0.41\text{ ms}$** | **$0.12\text{ ms}$** | $3.42\times$ | Comparable: 6×12 Harte projection from normalized chroma |
| **CQT Chroma Cost** | **$2170.55\text{ ms}$** (Tonnetz V0 path) | **$1391.88\text{ ms}$** (nnAudio CQT) | — | CQT transform bottleneck: Librosa harmonic HPSS + CQT Tonnetz vs nnAudio 84-bin CQT wavelet |
| **Chroma (STFT)** | $91.66\text{ ms}$ (STFT) | *N/A* (CQT only: $1391.88\text{ ms}$) | — | Non-equivalent: Librosa STFT chroma vs nnAudio CQT-based chroma |
| **Spectral** | $233.49\text{ ms}$ | $29.44\text{ ms}$ | — | Non-comparable: Librosa computes 3 operators (centroid + rolloff + bandwidth); Torch computes centroid only |
| **Spectral Contrast** | $52.74\text{ ms}$ | *Not Available* | — | 7-band contrast (no native operator in TorchAudio) |
| **Temporal** | $114.53\text{ ms}$ | $3.95\text{ ms}$ | — | Non-comparable: Librosa includes dynamic-programming beat tracking; Torch computes frame-level ZCR + RMS only |

---

## 5. Key Findings & Insights

1. **Tonnetz Bottleneck Fully Isolated:**
   - In `librosa`, Tonnetz V0 requires **$2,170.55\text{ ms}$** per 30-second track, accounting for **$78.57\%$** of the entire Original-113 pipeline execution time.
   - Re-projecting pre-computed STFT Chroma with Harte's $6 \times 12$ matrix (Tonnetz V1) takes only **$0.41\text{ ms}$**—a **$5,314\times$ reduction** in Tonnetz compute time that drops end-to-end pipeline latency from $2,770.08\text{ ms}$ to $563.87\text{ ms}$ ($4.91\times$ speedup).

2. **MOODWAAVE-Lite-71 Real-Time Factor:**
   - MOODWAAVE-Lite-71 finishes feature extraction and all 3 model predictions in **$233.73\text{ ms}$** on a single thread.
   - This corresponds to a Real-Time Factor (RTF) of **$0.0078$**, executing **$128.4\times$ faster than real time** ($11.85\times$ speedup over Original-113).

3. **DSP Framework Analogue Performance:**
   - PyTorch-native tensor operations (`torchaudio` + PyTorch math) compute MFCCs in $36.20\text{ ms}$ and temporal RMS/ZCR in $3.95\text{ ms}$.
   - For an analogue Lite feature set (MFCC + Temporal), PyTorch executes in just **$45.44\text{ ms}$** ($\text{RTF} = 0.0015$, $>660\times$ faster than real time).
   - However, nnAudio's CPU-based CQT implementation is unoptimized on single-thread CPU ($1,391.88\text{ ms}$), showing that Librosa's STFT Chroma ($91.66\text{ ms}$) is substantially more efficient for CPU-bound feature extraction.

---

## 6. Numbers to Use in the Paper

1. **Tonnetz V0 Bottleneck Share:** **$78.57\%$** of Original-113 latency ($2,170.55\text{ ms}$ out of $2,770.08\text{ ms}$) on a standard modern CPU.
2. **Tonnetz V1 Algorithmic Speedup:** Tonnetz V1 calculation time is **$0.41\text{ ms}$** ($5,314\times$ faster than Tonnetz V0), yielding a **$4.91\times$ end-to-end speedup** ($563.87\text{ ms}$, $\text{RTF} = 0.0188$).
3. **MOODWAAVE-Lite-71 End-to-End Latency:** **$233.73\text{ ms}$** per 30-second audio track, delivering an **$11.85\times$ speedup** and an ultra-fast **$\text{RTF} = 0.0078$** ($>128\times$ faster than real time).
4. **LightGBM Prediction Overhead:** Total inference time for all three models (genre classifier + valence regressor + arousal regressor) is **$2.40\text{ ms}$**, proving that tree inference constitutes $<1.1\%$ of total pipeline latency.
5. **Cross-Platform Kit Portability:** 50 canonical clips, 9 serialized LightGBM models, and self-contained scripts designed to be portable across Windows, Linux, and ARM (tested on Windows).
6. **Authoritative Absolute Latencies:** Note that `experiments/rev2_final_timing` (100 tracks per dataset, tight-loop shared STFT) is declared as the sole canonical source for absolute GTZAN (119.74 ms) and PMEmo (164.57 ms) latencies in the paper. Bench kit numbers reflect diagnostic group-sum extraction.

---

## 7. Limitations & Threats to Validity

- **Essentia Availability on Windows:** Essentia does not provide official PyPI wheels for Python 3.13 on Windows, and compilation from source requires complex MSVC/C++ tooling. The kit includes the Essentia code path for Linux/macOS users while seamlessly providing `torchaudio` / `nnAudio` as the Windows fallback.
- **Feature Analogue Equivalence:** The second library features (`torchaudio` / `nnAudio`) are acoustic analogues for execution time benchmarking, not bit-for-bit mathematical equivalents to Librosa.
- **Thermal Throttling on Mobile Hardware:** Laptops may experience minor CPU clock frequency variations across long runs; the 3-repeat median protocol mitigates transient CPU throttling.
