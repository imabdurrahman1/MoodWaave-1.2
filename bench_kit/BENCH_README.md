# MOODWAAVE Platform & Library Benchmark Kit

A self-contained, cross-platform benchmarking suite to profile **MOODWAAVE** audio feature extraction and inference latency across hardware architectures (Intel/AMD x86_64, ARM aarch64, Apple Silicon) and DSP libraries (`librosa` vs. `essentia` / `torchaudio` + `nnAudio`).

---

## 1. Overview & Benchmark Protocol

The kit benchmarks the three core MOODWAAVE CPU pipelines on 50 canonical GTZAN clips (30.0s, 22,050 Hz mono):
1. **Original-113 (Tonnetz V0):** 113 librosa features with original harmonic HPSS Tonnetz + 3 LightGBM heads (genre, valence, arousal).
2. **Full-113 (Tonnetz V1):** 113 features with Shared-STFT Tonnetz V1 + 3 LightGBM heads.
3. **MOODWAAVE-Lite-71:** 71 lightweight features (MFCC 52 + Spectral Contrast 14 + Temporal 5) + 3 LightGBM heads.

### Strict Measurement Rules
- **Single Thread Enforced:** All numerical backends (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `torch.set_num_threads(1)`).
- **1 Warm-Up Clip Excluded:** Excludes cold-start overhead (JIT compilation, cache loading).
- **Median of 3 Repeats:** Run 3 times per clip; median latency reported.
- **Timed Regions:** Measures audio decoding, per-feature-group costs, end-to-end pipeline latency, and LightGBM predictions.

---

## 2. Directory Structure

```
release/bench_kit/
├── audio_50/                  # 50 canonical GTZAN WAV clips + manifest.json
├── models/                    # Pre-trained MOODWAAVE LightGBM models (9 pkl files)
├── results/                   # JSON benchmark reports per machine
├── bench_platform.py          # Main single-thread profiling script
├── export_bench_clips.py      # Audio exporter script
├── aggregate_results.py       # Reads results/*.json -> tables & 300-DPI figure
├── requirements.txt           # Python dependencies
└── BENCH_README.md            # This instruction manual
```

---

## 3. Step-by-Step Instructions

### Step 1: Clone or Copy the Benchmark Kit
Copy the `release/bench_kit/` folder to the target test machine (laptop, desktop, cloud VM, or ARM single-board computer).

### Step 2: Set Up Python Environment
Ensure Python 3.10, 3.11, 3.12, or 3.13 is installed.

```bash
# Create and activate virtual environment
python -m venv venv

# On Linux / macOS / ARM:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

Install requirements:
```bash
pip install -r requirements.txt
```

*(Optional on Linux/macOS)*: To benchmark native C++ Essentia:
```bash
pip install essentia
```

### Step 3: Populate Audio Clips
If the `audio_50/` folder is empty or not yet generated, run:
```bash
python export_bench_clips.py --dest-dir audio_50
```
This populates 50 standardized GTZAN 30-second WAV files and generates `audio_50/manifest.json`.

### Step 4: Run Platform Benchmark
Ensure the machine is idle (close browsers, background apps), plugged into AC power, and set to High Performance.

Run the benchmark:
```bash
# Format: python bench_platform.py --audio-dir <path> --out results/<name>.json
python bench_platform.py --audio-dir audio_50 --out results/my_device_name.json
```

**Options:**
- `--audio-dir`: Path to folder containing the 50 WAV clips (default: `audio_50`).
- `--out`: Path to output JSON report (default: `results/<os>_<arch>.json`).
- `--repeats`: Number of repeats per clip (default: `3`).
- `--count`: Number of clips to evaluate (default: `50`).

### Step 5: Aggregate Cross-Platform Results
Once one or more benchmark JSON reports exist in `results/`, run:
```bash
python aggregate_results.py
```
This parses all `results/*.json` and generates:
1. `bench_summary_table.csv`: Platform × Library × Pipeline metrics (Latency, RTF, Speedup, Tonnetz share).
2. `bench_summary_table.json`: Machine-readable summary for dashboards.
3. `group_comparison_table.csv`: Group-by-group breakdown across libraries.
4. `bench_platform_latency_and_breakdown.png`: 300-DPI publication figure.

---

## 4. Metrics Explained

| Metric | Definition | Good Value |
| :--- | :--- | :--- |
| **Total Latency (ms)** | Time to load 30s audio + extract features + run 3 model predictions. | Lower is better. |
| **Real-Time Factor (RTF)** | $\text{Compute Time (s)} / \text{Audio Time (s)}$. | $\text{RTF} < 0.01$ means $>100\times$ faster than real time. |
| **Speedup vs. Orig** | $\text{Latency}_{\text{Orig-113}} / \text{Latency}_{\text{Pipeline}}$. | Higher is better ($>10\times$ for Lite-71). |
| **Tonnetz Share (%)** | $\% \text{ of Original-113 latency spent inside Tonnetz V0}$. | Demonstrates the primary CPU bottleneck ($>75\%$). |

---

## 5. Library Comparison & Feature Analogue Notes

- **Librosa:** Reference MOODWAAVE feature extraction implementation.
- **TorchAudio + nnAudio (Fallback):**
  - Evaluates PyTorch-native equivalents: MFCC (`torchaudio.transforms.MFCC`), Spectral Centroid (`torchaudio.transforms.SpectralCentroid`), RMS/ZCR, and CQT Chroma (`nnAudio.features.cqt.CQT2010v2`).
  - Tonnetz is computed from CQT Chroma via Harte's $6 \times 12$ projection matrix.
  - *Note:* Spectral Contrast and Tempo have no native operator in TorchAudio and are excluded from the Torch analogue total. These features are acoustic analogues for execution profiling, not mathematical replicas.
- **Essentia (Linux/macOS):**
  - Evaluates C++ native bindings: `MFCC`, `HPCP`, `ZeroCrossingRate`, `RMS`, `RhythmExtractor2013`. Tonnetz is projected from HPCP using Harte's matrix.

---

## 6. Troubleshooting

- **ModuleNotFoundError: 'nnAudio' or 'torchaudio'**: Run `pip install torchaudio nnAudio`.
- **CPU Throttling**: On laptops, ensure power brick is connected. Thermal throttling will increase standard deviation across the 3 repeats.
- **Multi-threading Leaks**: The benchmark automatically exports single-thread environment variables (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`). Do not modify these.

