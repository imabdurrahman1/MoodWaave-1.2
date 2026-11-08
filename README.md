# MOODWAAVE: Lightweight, High-Efficiency Music Genre and Emotion Analysis on Edge CPU Hardware

[![DOI](https://zenodo.org/badge/1449926159.svg)](https://doi.org/10.5281/zenodo.24047490)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.13](https://img.shields.io/badge/Python-3.13.7-blue.svg)](requirements.txt)
[![Python: 3.11/3.12 (Embeddings)](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blueviolet.svg)](requirements_embeddings.txt)
[![LightGBM: 4.6.0](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](requirements.txt)
[![Librosa: 0.11.0](https://img.shields.io/badge/Librosa-0.11.0-orange.svg)](requirements.txt)
[![PyTorch: CPU](https://img.shields.io/badge/PyTorch-2.15.0%2Bcpu-red.svg)](requirements.txt)

This repository contains the complete open-source reproduction package and cross-platform benchmarking suite for **MOODWAAVE**, an ultra-lightweight, CPU-optimized music information retrieval (MIR) architecture. MOODWAAVE delivers simultaneous 10-class music genre classification (GTZAN) and continuous Valence/Arousal emotion regression (PMEmo) with distribution-free conformal prediction bounds, running orders of magnitude faster than real-time playback on standard consumer laptop CPUs without requiring dedicated GPUs.

---

## Table of Contents

1. [Project Overview & Core Findings](#project-overview--core-findings)
2. [Repository Structure](#repository-structure)
3. [Benchmarking & Reproduction Hardware/Software Environment](#benchmarking--reproduction-hardwaresoftware-environment)
4. [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
   - [Setup & Virtual Environments](#1-setup--virtual-environments)
   - [Execution Order & Paper Table Mapping](#2-execution-order--paper-table-mapping)
5. [Cross-Platform Benchmark Kit](#cross-platform-benchmark-kit)
6. [Figure Regeneration & Verification](#figure-regeneration--verification)
7. [Data Notice, Licenses & Provenance](#data-notice-licenses--provenance)
8. [Citation](#citation)

---

## Project Overview & Core Findings

State-of-the-art deep audio neural networks (e.g., VGGish, PANNs CNN14, MERT-v1-95M) require massive GPU acceleration and hundreds of megabytes of model parameters, rendering them impractical for resource-constrained client devices, low-power edge microcontrollers, or real-time local audio processing.

**MOODWAAVE** demonstrates that carefully curated tabular acoustic features paired with tuned gradient boosted decision trees (**LightGBM**) achieve predictive parity with heavyweight deep neural representations while consuming a fraction of the compute, latency, and memory:

* **Predictive Performance at 37% Lower Feature Dimension**:
  The full baseline pipeline (**Full-113**) extracts 113 acoustic features spanning 6 groups (MFCC, Chroma, Tonnetz, Spectral, Spectral Contrast, and Temporal). By auditing computation costs, we identified that Chroma (24) and Tonnetz (12) consume ~78-86% of the total extraction time. **MOODWAAVE-Lite** (71 features: MFCC 52 + Spectral Contrast 14 + Temporal 5) drops Chroma and Tonnetz entirely:
  - **Genre Accuracy (GTZAN, 15 Outer Folds: 5 Folds x 3 Repeats, Artist-Disjoint)**: MOODWAAVE-Lite achieves **60.72% ± 6.24%** (Macro-F1: **0.5967 ± 0.0601**) vs. Full-113 at **63.31% ± 5.67%** (Macro-F1: **0.6275 ± 0.0498**). The difference is \(\Delta = -2.59\%\) (Nadeau-Bengio corrected resampled \(t = -1.732, p = 0.1053\); 90% CI: \([-5.23\%, +0.04\%]\); Holm-adjusted \(p = 0.7370\)), which is statistically inconclusive at the preset \(\pm 2.0\%\) TOST margin.
  - **Emotion Regression (PMEmo, 10 Outer Folds: 5 Folds x 2 Repeats, Artist-Disjoint)**: 
    - Valence: MOODWAAVE-Lite achieves \(R^2 = 0.4516 \pm 0.0631\) (Full-113: \(0.4595 \pm 0.0577\), \(\Delta = -0.0079, p = 0.5318\); 90% CI: \([-0.0302, +0.0144]\)).
    - Arousal: MOODWAAVE-Lite achieves \(R^2 = 0.6627 \pm 0.0400\) (Full-113: \(0.6665 \pm 0.0427\), \(\Delta = -0.0038, p = 0.6077\); 90% CI: \([-0.0170, +0.0094]\)). Equivalence within the preset \(\pm 0.02\) \(R^2\) margin is rigorously established via Two One-Sided Tests (TOST, \(p < 0.05\)).
* **Dramatic CPU Latency Reduction (Canonical Source: `experiments/rev2_final_timing`)**:
  Benchmarked on an 11th Gen Intel Core i5 laptop under strict single-threaded CPU conditions (OMP/MKL/OPENBLAS/NUMBA/torch = 1, median of 3 repeats across 100 tracks per dataset; timed region = audio loading + feature extraction + inference of all 3 models):
  - **GTZAN (WAV)**:
    - **MOODWAAVE-Lite-71**: **119.74 ms** total latency per 30s track (**Real-Time Factor RTF = 0.00399**, processing audio **250x faster than real-time playback**; feature extraction: 115.69 ms, predictions: 2.33 ms).
    - **Full-113 (Tonnetz V1)**: **240.11 ms** (RTF = 0.00800), yielding a **2.01x end-to-end speedup** for MOODWAAVE-Lite.
    - **Original-113 (Tonnetz V0)**: **1,763.45 ms** (RTF = 0.05878), yielding a **14.73x speedup** for MOODWAAVE-Lite.
  - **PMEmo (MP3)**:
    - **MOODWAAVE-Lite-71**: **164.57 ms** total latency per 30s track (**RTF = 0.00549**, audio loading: 50.79 ms MP3 decode overhead, feature extraction: 111.27 ms, predictions: 2.34 ms).
    - **Original-113 (Tonnetz V0)**: **1,729.02 ms** (10.51x slower).
* **Efficiency vs Foundation Audio Models (Exact Table Ratios)**:
  - **VGGish (128-dim, Apache 2.0)**: **720.78 ms** on GTZAN (**6.02x slower than MOODWAAVE-Lite**) and **732.42 ms** on PMEmo (**4.45x slower**) with 275.24 MB serialized size.
  - **PANNs CNN14 (2048-dim, MIT)**: **1,619.14 ms** on GTZAN (**13.52x slower than MOODWAAVE-Lite**) and **1,589.12 ms** on PMEmo (**9.66x slower**) with 312.30 MB size. Probing mood heads exhibit variance collapse and instability unless zero-variance dimensions are filtered.
  - **MERT-v1-95M (768-dim, CC-BY-NC-4.0)**: **11,186.84 ms** on GTZAN (**93.42x slower than MOODWAAVE-Lite**) and **10,138.50 ms** on PMEmo (**61.61x slower**) with 360.09 MB size.
* **Ultra-Compact Model Footprint**:
  The total serialized size of all 3 MOODWAAVE-Lite models (Genre Classifier + Valence Regressor + Arousal Regressor) is only **3.96 MB** (**69.5x smaller** than VGGish, **78.9x smaller** than PANNs, and **90.9x smaller** than MERT).
* **Cross-Dataset Generalizability**:
  - **FMA-Small (800 tracks balanced subset, 566 unique artists)**: LightGBM Full-113 achieves 42.10% ± 2.54% accuracy (Lite-71: 40.59% ± 4.03%, Random Forest: 44.63% ± 0.95%, MLP under-tuned baseline: 39.83% ± 4.67%).
  - **DEAM Emotion Benchmark (1,802 tracks)**: Strict TOST equivalence confirmed between MOODWAAVE-Lite and Full-113 on DEAM Valence (\(\Delta R^2 = -0.0006\), 90% CI: \([-0.0164, +0.0151]\)) within \(\pm 0.02\) margin.
  - **Leakage Penalty Hypothesis**: Leakage drops when transitioning from random to artist-disjoint splits correspond with tracks-per-artist ratios (GTZAN: 3.25 trk/art, -14.6 pp drop; FMA-800: 1.41 trk/art, -3.1 pp drop; PMEmo: 1.92 trk/art, -0.01 \(R^2\) drop). We formulate this strictly as an observational hypothesis.
* **SmallMelCNN Baseline**:
  Evaluated on the single fixed Kereliuk & Sturm (2015) split (5 seeds, 50 epochs; 58.41% accuracy, 85.86 ms latency, 0.39 MB parameters), not re-trained across 15 CV folds.
* **Calibrated Conformal Uncertainty**:
  Instead of fragile point estimates in Valence-Arousal space, MOODWAAVE integrates split-conformal joint prediction regions mapped to Russell's Circumplex Affect grid. At an 80% nominal confidence level, MOODWAAVE-Lite narrows the prediction down to **2.08 emotion zones** on a 16-zone grid, providing provable finite-sample coverage guarantees.

---

## Repository Structure

```text
release/
├── README.md                           # Master project guide and reproduction manual
├── DATA_NOTICE.md                      # Dataset provenance, download links, and licensing terms
├── LICENSE                             # MIT Open Source License (Code only)
├── CITATION.cff                        # Machine-readable academic citation metadata
├── .zenodo.json                        # Zenodo archival and DOI minting configuration
├── .gitignore                          # Strict git ignore (excludes raw audio binaries)
├── requirements.txt                    # Pinned Python 3.13 dependencies for core MOODWAAVE
├── requirements_embeddings.txt         # Python 3.11/3.12 dependencies for foundation models
├── CHANGES.md                          # Comprehensive release changelog
│
├── bench_kit/                          # Standalone turnkey cross-platform benchmark kit
│   ├── BENCH_README.md                 # Complete benchmark kit instructions
│   ├── RESULTS.md                      # Cross-platform latency logs and empirical findings
│   ├── bench_platform.py               # Single-threaded latency profiling runner
│   ├── aggregate_results.py            # Aggregates multi-platform JSON runs into tables/plots
│   ├── export_bench_clips.py           # Exports 50 canonical GTZAN WAV clips from source
│   ├── requirements.txt                # Lightweight dependencies for benchmark kit
│   ├── bench_summary_table.csv         # Platform latency benchmark summary
│   ├── bench_summary_table.json        # Machine-readable platform latency benchmark summary
│   ├── group_comparison_table.csv      # Feature group computational cost breakdown
│   ├── bench_platform_latency_and_breakdown.png # 300-DPI dual-panel latency visualization
│   └── models/                         # Serialized pre-trained MOODWAAVE models (9 pkl files)
│       ├── pipe_a_genre_orig113.pkl
│       ├── pipe_a_val_orig113.pkl
│       ├── pipe_a_aro_orig113.pkl
│       ├── pipe_b_genre_full113_v1.pkl
│       ├── pipe_b_val_full113_v1.pkl
│       ├── pipe_b_aro_full113_v1.pkl
│       ├── pipe_c_genre_lite71.pkl
│       ├── pipe_c_val_lite71.pkl
│       └── pipe_c_aro_lite71.pkl
│
├── figures/                            # 32 publication-ready figures (300 DPI)
│   ├── accuracy_latency_pareto_tonnetz.png
│   ├── cnn_training_curves.png
│   ├── confusion_matrix_genre_lite.png     # Regenerated with MOODWAAVE-Lite
│   ├── deep_vs_moodwaave_accuracy.png
│   ├── deep_vs_moodwaave_efficiency.png
│   ├── final_latency_breakdown_bar_chart.png
│   ├── final_latency_vs_size_pareto.png
│   ├── g1_coverage_vs_set_size.png
│   ├── g1_set_size_bar_chart.png
│   ├── genre_baselines_accuracy.png
│   ├── grid_comparison_bar_chart.png
│   ├── gtzan_faults_by_genre.png
│   ├── gtzan_leakage_drop.png
│   ├── latency_vs_performance.png
│   ├── mean_set_size_vs_coverage.png       # Regenerated with MOODWAAVE-Lite
│   ├── mood_baselines_r2.png
│   ├── pmemo_leakage_comparison.png
│   ├── selection_frequencies.png
│   ├── shap_arousal_summary.png
│   ├── shap_genre_bar.png
│   ├── shap_valence_summary.png
│   ├── tonnetz_cost_comparison.png
│   ├── tost_equivalence_confidence_intervals.png
│   ├── va_box_scatter_80pct.png            # Regenerated with MOODWAAVE-Lite
│   ├── r3a_genre_accuracy.png              # Rev3-A: 5x5 CV genre accuracy across model families
│   ├── r3a_mood_r2.png                     # Rev3-A: 5x5 CV Valence & Arousal R2 across model families
│   ├── r3b_genre_pareto.png                # Rev3-B: Genre Accuracy vs. CPU Latency Pareto frontier
│   ├── r3b_mood_pareto.png                 # Rev3-B: Valence/Arousal R2 vs. Latency Pareto frontier
│   ├── r3c_leakage_cross_dataset_comparison.png # Rev3-C: Artist leakage across GTZAN & FMA
│   ├── r3c_decoding_overhead_and_latency.png   # Rev3-C: Audio decoding overhead & extraction scaling
│   ├── r3c_deam_model_families_r2_tost.png     # Rev3-C: DEAM emotion regression & TOST intervals
│   └── r3d_bench_platform_latency_and_breakdown.png # Rev3-D: Turnkey benchmark kit results
│
├── tables/                             # 83 CSV/JSON result tables and tabular features
│   ├── gtzan_features.csv                  # 113-dim tabular features for 999 GTZAN tracks
│   ├── pmemo_ready.csv                     # 113-dim tabular features for 767 PMEmo tracks
│   ├── splits.json                         # Exact artist-disjoint CV splits (GTZAN & PMEmo)
│   ├── feature_groups.json                 # Mapping of 113 features to 6 acoustic groups
│   ├── feature_costs.csv                   # Per-feature computational latency profiles
│   ├── final_timing_table.csv              # Master latency benchmark table (supersedes earlier)
│   ├── final_system_sizes.csv              # Disk size comparison across all pipelines
│   ├── final_consolidation_tables.csv      # Consolidated predictive and efficiency summaries
│   ├── p1_part_a_gtzan_summary.csv         # Random vs. Kereliuk-Sturm split benchmarks
│   ├── p1_part_b_pmemo_summary.csv         # Random vs. artist-clean PMEmo benchmarks
│   ├── p2_end_to_end_summary.csv           # Tonnetz V1 algorithmic optimization benchmarks
│   ├── p3_heldout_performance_summary.csv  # 10 repeated 70/30 artist-disjoint hold-out splits (1-SE rule)
│   ├── p3_paired_differences_and_equivalence.csv # TOST statistical equivalence across hold-out splits
│   ├── p3_selection_frequency_genre.csv    # Feature selection frequencies under 1-SE rule (Genre)
│   ├── p3_selection_frequency_mood.csv     # Feature selection frequencies under 1-SE rule (Mood)
│   ├── p4_genre_summary.csv                # Classical genre baselines (LightGBM, RF, ET, XGB, Ridge)
│   ├── p4_pmemo_summary.csv                # Classical emotion regression across baselines
│   ├── p5_fixed_split_comparison.csv       # MOODWAAVE vs. SmallMelCNN and VGGish
│   ├── p6_conformal_sweep_results.csv      # Multi-method (M1-M6) conformal coverage sweeps
│   ├── p6_grid_comparison_80pct.csv        # Circumplex zone set sizes across G1, G2, G3 grids
│   ├── r3a_gtzan_summary.csv               # Rev3-A: 5x5 Repeated Stratified Group CV on GTZAN
│   ├── r3a_gtzan_paired_tests.csv          # Rev3-A: Corrected resampled t-tests & Wilcoxon on GTZAN
│   ├── r3a_pmemo_summary.csv               # Rev3-A: 5x5 Repeated Group CV on PMEmo
│   ├── r3a_pmemo_paired_tests.csv          # Rev3-A: Corrected resampled t-tests & Wilcoxon on PMEmo
│   ├── r3a_default_vs_tuned_lgbm.csv       # Rev3-A: Default vs. Tuned LightGBM across 25 folds
│   ├── r3a_model_profiles_summary.csv      # Rev3-A: Inference latency and memory for classical models
│   ├── r3a_vs_p4_comparison.csv            # Rev3-A: Audit comparing Rev3-A against P4 baselines
│   ├── r3b_gtzan_summary.csv               # Rev3-B: Foundation embeddings on GTZAN (VGGish, PANNs, MERT)
│   ├── r3b_gtzan_paired_tests.csv          # Rev3-B: Paired tests comparing MOODWAAVE vs. foundation models
│   ├── r3b_pmemo_summary.csv               # Rev3-B: Foundation embeddings on PMEmo
│   ├── r3b_pmemo_paired_tests.csv          # Rev3-B: Paired tests for mood regression
│   ├── r3b_embeddings_efficiency_profiles.csv # Rev3-B: Latency, RAM, and disk footprint of foundation models
│   ├── r3c_cross_dataset_master_summary.csv # Rev3-C: Master cross-dataset evaluation summary
│   ├── r3c_fma_leakage_summary.csv         # Rev3-C: Artist leakage analysis on FMA-small
│   ├── r3c_fma_model_families_summary.csv  # Rev3-C: Model families benchmark on FMA-small
│   ├── r3c_deam_model_families_summary.csv # Rev3-C: Model families benchmark on DEAM
│   ├── r3c_gtzan_to_fma_transfer.csv       # Rev3-C: GTZAN to FMA-small genre transfer
│   ├── r3c_pmemo_to_deam_zeroshot.csv      # Rev3-C: PMEmo to DEAM zero-shot emotion transfer
│   ├── r3c_efficiency_breakdown_summary.csv # Rev3-C: Feature extraction latency on long-form audio
│   ├── r3c_decoding_overhead_comparison.csv # Rev3-C: Audio decoding library comparison
│   ├── r3c_fma_800_metadata.csv            # Rev3-C: Metadata for 800 artist-balanced FMA tracks
│   ├── r3c_deam_metadata.csv               # Rev3-C: Metadata for 1,802 DEAM tracks
│   └── benchmarking_environment.json       # Exact hardware, CPU governor, and library versions
│
└── scripts/                            # Self-contained reproduction scripts
    ├── feature_extractor.py                # Core feature extraction logic (113 librosa features)
    ├── pipeline.py                         # End-to-end prediction pipeline wrapper
    ├── common.py                           # Shared constants, paths, and statistical test utilities
    ├── 01_part_a_predictive.py             # Script for Table 4 (Lite-71 vs. Full-113 5x3 CV)
    ├── 02_part_b_efficiency.py             # Script for feature extraction timing breakdowns
    ├── 03_part_c_conformal.py              # Script for conformal prediction calibration
    ├── 04_part_d_shap.py                   # Script for SHAP feature attribution
    ├── 05_compile_final_results.py         # Results compiler for consolidated tables
    ├── p1_01_build_splits_and_metadata.py  # Kereliuk-Sturm and artist split generators
    ├── p1_02_evaluate_gtzan.py             # GTZAN split evaluation runner
    ├── p1_03_evaluate_pmemo.py             # PMEmo split evaluation runner
    ├── p1_04_make_results_and_plots.py     # Split comparison analysis and plotting
    ├── p2_01_numeric_similarity_all_tracks.py # Tonnetz V0 vs. V1 numerical similarity
    ├── p2_02_retime_single_session_twice.py# Tonnetz single-session retiming benchmark
    ├── p2_03_extract_v1_features.py        # Tonnetz V1 feature extractor
    ├── p2_04_evaluate_accuracy.py          # Tonnetz accuracy and numerical similarity test
    ├── p2_05_compile_results.py            # Tonnetz benchmark compiler
    ├── p3_01_run_gtzan_selection.py        # 10 repeated hold-out splits (70/30 artist-disjoint) on GTZAN
    ├── p3_02_run_pmemo_selection.py        # 10 repeated hold-out splits (70/30 artist-disjoint) on PMEmo
    ├── p3_03_analyze_and_report.py         # TOST equivalence analysis across 10 hold-out splits
    ├── p4_01_run_gtzan_baselines.py        # Classical genre baseline benchmark runner
    ├── p4_02_run_pmemo_baselines.py        # Classical mood baseline benchmark runner
    ├── p4_03_profile_models.py             # Inference timing and memory profiling
    ├── p4_04_analyze_and_report.py         # Classical baselines report generator
    ├── p4_05_run_default_lightgbm.py       # Default vs. tuned LightGBM benchmark
    ├── p5_01_extract_vggish_embeddings.py  # VGGish 128-dim embedding extractor
    ├── p5_02_eval_vggish_heads.py          # Linear classification/regression head evaluation
    ├── p5_02b_eval_moodwaave_fixed_split.py# Evaluates MOODWAAVE on identical fixed split
    ├── p5_03_train_small_cnn.py            # Training script for SmallMelCNN baseline
    ├── p5_04_profile_efficiency.py         # Latency profiling for deep models on CPU
    ├── p5_05_analyze_and_report.py         # Deep learning comparison report compiler
    ├── p5_06_compute_paired_and_mcnemar_tests.py # McNemar's tests for deep baselines
    ├── p5_small_cnn.py                     # SmallMelCNN architecture definition
    ├── p6_01_run_conformal_benchmark.py    # Conformal methods (M1-M6) and grid benchmark
    ├── p6_02_analyze_and_report.py         # Conformal analysis and summary tables
    ├── final_timing_00_setup_models.py     # Fits all 6 pipelines on fixed splits
    ├── final_timing_01_benchmark.py        # Master timing session (forward + reverse order)
    ├── final_timing_02_analyze_and_report.py# Generates final timing tables and Pareto figures
    ├── r3a_common.py                       # Rev3-A: Shared configuration & extended grids
    ├── r3a_01_run_gtzan.py                 # Rev3-A: 5x5 Repeated Group CV on GTZAN
    ├── r3a_02_run_pmemo.py                 # Rev3-A: 5x5 Repeated Group CV on PMEmo
    ├── r3a_03_default_vs_tuned_lightgbm.py # Rev3-A: Default vs. Tuned LightGBM evaluation
    ├── r3a_04_profile_models.py            # Rev3-A: Single-threaded CPU latency profiling
    ├── r3a_05_analyze_and_report.py        # Rev3-A: Statistical test compiler & figures
    ├── r3a_audit_p4.py                     # Rev3-A: Audit verifying P4 split methodology
    ├── r3b_common.py                       # Rev3-B: Shared embedding utilities & paths
    ├── r3b_01_extract_panns.py             # Rev3-B: PANNs CNN14 2048-dim feature extraction
    ├── r3b_02_extract_mert.py              # Rev3-B: MERT-v1-95M 768-dim feature extraction
    ├── r3b_03_evaluate_embeddings_heads.py # Rev3-B: Probing heads (Ridge, MLP, LightGBM)
    ├── r3b_04_statistical_tests.py        # Rev3-B: Corrected resampled t-tests & TOST
    ├── r3b_05_profile_efficiency.py        # Rev3-B: CPU latency and memory footprint profiling
    ├── r3b_06_analyze_and_report.py        # Rev3-B: Pareto trade-off tables and figures
    ├── r3c_common.py                       # Rev3-C: Cross-dataset paths & utilities
    ├── r3c_01_extract_fma.py               # Rev3-C: FMA-small feature extractor & cleaner
    ├── r3c_02_extract_deam.py              # Rev3-C: DEAM feature extractor & cleaner
    ├── r3c_03_fma_leakage_and_models.py    # Rev3-C: FMA artist leakage & model baselines
    ├── r3c_04_deam_models_and_zeroshot.py  # Rev3-C: DEAM models & zero-shot transfer
    ├── r3c_05_genre_transfer_gtzan_to_fma.py # Rev3-C: GTZAN to FMA cross-dataset transfer
    ├── r3c_06_profile_efficiency.py        # Rev3-C: Audio decoding & duration scaling
    ├── r3c_07_analyze_and_report.py        # Rev3-C: Master cross-dataset report compiler
    └── regenerate_and_consolidate_all_figures.py # Full figure regeneration and collation
```

---

## Benchmarking & Reproduction Hardware/Software Environment

All latency benchmarks, memory profiles, and accuracy numbers reported in the paper were measured on the following reference system:

* **Processor (CPU)**: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (Tiger Lake, 4 physical cores, 8 logical threads, 8 MB L3 Cache).
* **System RAM**: 8.0 GB DDR4 (7.75 GB usable).
* **Operating System**: Microsoft Windows 11 Enterprise (Build 10.0.26200).
* **Power Configuration**: AC plugged in, Windows High Performance power plan (`Power Scheme GUID: 74a4fc32-ef61-4ed8-b796-ed0035641e51`), background throttling disabled.
* **Deterministic Single-Thread Execution Protocol**:
  To prevent thread contention and ensure strictly reproducible timing benchmarks across platforms, all multi-threading libraries are pinned to 1 thread prior to importing numerical packages:
  ```bash
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMBA_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1
  torch.set_num_threads(1)
  ```
* **Timing Protocol**:
  1 warm-up track excluded; median of 3 repeats per track across identical tracks; executed across two independent runs (Run 1 forward track order, Run 2 reverse track order).
* **Core Library Versions (Primary Environment)**:
  - Python: `3.13.7`
  - LightGBM: `4.6.0`
  - Scikit-Learn: `1.8.0`
  - SciPy: `1.17.1`
  - NumPy: `2.4.3`
  - Pandas: `2.3.3`
  - Librosa: `0.11.0`
  - PyTorch: `2.15.0+cpu`
  - SHAP: `0.52.0`
  - Matplotlib: `3.10.8`

---

## Step-by-Step Reproduction Guide

### 1. Setup & Virtual Environments

#### Primary Environment (Python 3.13 - Core MOODWAAVE & Benchmarks)
```bash
# Navigate to release root
cd release

# Create and activate virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

> **Note on PyTorch (CPU-only)**: If wheels are not resolved automatically on Windows:
> ```bash
> pip install torch==2.15.0+cpu torchaudio==2.11.1+cpu --index-url https://download.pytorch.org/whl/cpu
> ```

#### Dedicated Foundation Model Environment (Python 3.11 / 3.12 - Rev3-B Foundation Models)
For extracting or evaluating deep audio embeddings (MERT-v1-95M, PANNs CNN14, VGGish) which require older C-extension wheels not yet prebuilt for Python 3.13:
```bash
# Create a separate environment with Python 3.11 or 3.12
py -3.11 -m venv venv_embeddings
# Activate and install dependencies
.\venv_embeddings\Scripts\Activate.ps1
pip install -r requirements_embeddings.txt
```

---

### 2. Execution Order & Paper Table Mapping

To reproduce all paper tables and numerical findings from scratch, execute the reproduction scripts in the following order. All intermediate outputs are saved into `release/tables/`:

| Step | Script (`release/scripts/`) | Paper Table / Section | Key Output File (`release/tables/`) | Description |
|:---:|:---|:---|:---|:---|
| **1** | `01_part_a_predictive.py` | **Table 4**: Predictive Parity | `part_a_genre_summary.csv`<br>`part_a_mood_summary.csv`<br>`part_a_statistical_tests.json` | 5x3 Repeated Stratified CV comparing Full-113 vs. Lite-71 on GTZAN and PMEmo with corrected resampled t-tests. |
| **2** | `04_part_d_shap.py` | **Section 4.3**: Model Interpretability | `part_d_top15_features.csv`<br>`part_d_shap_group_importance.csv` | TreeSHAP feature attribution explaining genre and continuous emotion predictions. |
| **3** | `p1_01_build_splits_and_metadata.py`<br>`p1_02_evaluate_gtzan.py`<br>`p1_03_evaluate_pmemo.py`<br>`p1_04_make_results_and_plots.py` | **Table 1**: Dataset Leakage Audit | `p1_part_a_gtzan_summary.csv`<br>`p1_part_b_pmemo_summary.csv`<br>`p1_gtzan_removed_tracks.csv` | Kereliuk-Sturm artist-filtered GTZAN splits and song/artist-clean PMEmo splits. |
| **4** | `p2_01_numeric_similarity_all_tracks.py`<br>`p2_02_retime_single_session_twice.py`<br>`p2_04_evaluate_accuracy.py`<br>`p2_05_compile_results.py` | **Table 2**: Tonnetz Optimization | `p2_end_to_end_summary.csv`<br>`p2_accuracy_benchmark_summary.csv` | Evaluates the numerical parity and latency reduction of Tonnetz V1 (reusing precomputed STFT chroma). |
| **5** | `p3_01_run_gtzan_selection.py`<br>`p3_02_run_pmemo_selection.py`<br>`p3_03_analyze_and_report.py` | **Table 3**: Feature Stability & TOST | `p3_selection_frequency_genre.csv`<br>`p3_heldout_performance_summary.csv`<br>`p3_paired_differences_and_equivalence.csv` | **10 repeated 70/30 artist-disjoint hold-out splits** with inner 5x2 grouped CV on 95 subsets; **1-SE selection rule**; Two One-Sided Tests (TOST) confirming statistical equivalence within \(\pm 1.5\%\). |
| **6** | `p4_01_run_gtzan_baselines.py`<br>`p4_02_run_pmemo_baselines.py`<br>`p4_03_profile_models.py`<br>`p4_04_analyze_and_report.py` | **Table 5**: Classical Model Baselines | `p4_genre_summary.csv`<br>`p4_pmemo_summary.csv`<br>`p4_model_profiles_summary.csv` | 5x3 artist-disjoint CV benchmark comparing LightGBM against Random Forest, Extra Trees, XGBoost, and Ridge. |
| **7** | `p5_01_extract_vggish_embeddings.py`<br>`p5_02_eval_vggish_heads.py`<br>`p5_03_train_small_cnn.py`<br>`p5_04_profile_efficiency.py`<br>`p5_05_analyze_and_report.py` | **Table 6**: Deep Neural Baselines | `p5_fixed_split_comparison.csv`<br>`p5_efficiency_profiles.csv`<br>`p5_deep_system_sizes.csv` | CPU benchmarking of SmallMelCNN (genre) and VGGish 128-dim embeddings with linear classification/regression heads. |
| **8** | `p6_01_run_conformal_benchmark.py`<br>`p6_02_analyze_and_report.py` | **Table 7**: Conformal Benchmarks | `p6_conformal_sweep_results.csv`<br>`p6_grid_comparison_80pct.csv`<br>`p6_point_models_performance.csv` | Rigorous evaluation of 6 conformal prediction methods (M1-M6) across 20 random splits and 3 emotion zone grids (G1, G2, G3). |
| **9** | `final_timing_00_setup_models.py`<br>`final_timing_01_benchmark.py`<br>`final_timing_02_analyze_and_report.py` | **Table 8**: Master Final Timing & Pareto Table | `final_timing_table.csv`<br>`final_system_sizes.csv`<br>`final_timing_runs_comparison.csv` | **Master timing benchmark across all 6 full pipelines** on 200 tracks (Run 1 forward, Run 2 reverse). *Supersedes all earlier timing numbers*. |
| **10** | `r3a_01_run_gtzan.py`<br>`r3a_02_run_pmemo.py`<br>`r3a_03_default_vs_tuned_lightgbm.py`<br>`r3a_04_profile_models.py`<br>`r3a_05_analyze_and_report.py` | **Rev3-A**: Statistical Rigor & Expanded Baselines | `r3a_gtzan_summary.csv`<br>`r3a_gtzan_paired_tests.csv`<br>`r3a_pmemo_summary.csv`<br>`r3a_default_vs_tuned_lgbm.csv` | 5x5 Repeated Stratified Group CV (25 folds) with corrected resampled t-tests, Wilcoxon signed-rank tests, and default vs. tuned LightGBM audits. |
| **11** | `r3b_01_extract_panns.py`<br>`r3b_02_extract_mert.py`<br>`r3b_03_evaluate_embeddings_heads.py`<br>`r3b_04_statistical_tests.py`<br>`r3b_05_profile_efficiency.py`<br>`r3b_06_analyze_and_report.py` | **Rev3-B**: Foundation Audio Models Comparison | `r3b_gtzan_summary.csv`<br>`r3b_pmemo_summary.csv`<br>`r3b_embeddings_efficiency_profiles.csv` | CPU benchmarking of PANNs CNN14 (2048-dim) and MERT-v1-95M (768-dim) across Ridge, MLP, and LightGBM heads with Pareto trade-off curves. |
| **12** | `r3c_01_extract_fma.py`<br>`r3c_02_extract_deam.py`<br>`r3c_03_fma_leakage_and_models.py`<br>`r3c_04_deam_models_and_zeroshot.py`<br>`r3c_05_genre_transfer_gtzan_to_fma.py`<br>`r3c_06_profile_efficiency.py`<br>`r3c_07_analyze_and_report.py` | **Rev3-C**: Cross-Dataset Generalizability & Efficiency | `r3c_cross_dataset_master_summary.csv`<br>`r3c_fma_leakage_summary.csv`<br>`r3c_deam_model_families_summary.csv`<br>`r3c_pmemo_to_deam_zeroshot.csv`<br>`r3c_decoding_overhead_comparison.csv` | Cross-dataset validation on FMA-small (800 tracks) and DEAM (1,802 tracks); GTZAN-to-FMA genre transfer; PMEmo-to-DEAM zero-shot mood transfer; audio decoding benchmarks (`torchaudio` vs. `librosa`). |

---

## Cross-Platform Benchmark Kit

To enable turnkey benchmarking of MOODWAAVE across external hardware architectures (Intel/AMD x86_64, ARM Cortex, Apple Silicon, Raspberry Pi) without downloading large audio archives, the `release/bench_kit/` directory provides a standalone benchmarking suite:

```bash
cd release/bench_kit

# Install lightweight dependencies
pip install -r requirements.txt

# Run single-threaded benchmark on 50 canonical clips
python bench_platform.py --device-label "MyDevice"

# Aggregate results across multiple machines
python aggregate_results.py
```

See [`release/bench_kit/BENCH_README.md`](bench_kit/BENCH_README.md) for complete details.

---

## Figure Regeneration & Verification

All 32 figures used in the manuscript can be regenerated and compiled directly into `release/figures/` using:

```bash
python release/scripts/regenerate_and_consolidate_all_figures.py
```

This script:
1. Re-runs pooled 5x3 repeated stratified cross-validation on GTZAN to regenerate the normalized confusion matrix titled `"MOODWAAVE-Lite"`.
2. Reads conformal calibration records and regenerates `mean_set_size_vs_coverage.png` and `va_box_scatter_80pct.png` with `"MOODWAAVE-Lite"` titles and legends.
3. Consolidates all publication-quality 300-DPI figures across split audits, Tonnetz Pareto curves, hold-out selection frequencies, classical baselines, neural curves, cross-dataset evaluations, and benchmark kit profiles.

---

## Data Notice, Licenses & Provenance

* **Audio Files Policy**: In accordance with dataset redistribution licenses and copyright restrictions, **no raw audio files (`.wav`, `.mp3`, `.au`) are distributed within this repository**.
* **Pre-extracted Features Provided**: Pre-extracted 113-dimensional acoustic feature tables for all 999 usable GTZAN tracks (`release/tables/gtzan_features.csv`) and all 767 PMEmo tracks (`release/tables/pmemo_ready.csv`), along with FMA-small and DEAM metadata, are included directly in `release/tables/` in accordance with upstream academic licenses.
* For full provenance, upstream URLs, and academic licensing guidelines, see [`DATA_NOTICE.md`](DATA_NOTICE.md).

---

## Citation

If you use the MOODWAAVE architecture, feature representations, or conformal prediction benchmarks in your research, please cite:

```bibtex
@article{moodwaave2026,
  author    = {[Author Placeholder]},
  title     = {MOODWAAVE: Lightweight, High-Efficiency Music Genre and Emotion Analysis on Edge CPU Hardware},
  journal   = {[Journal / Conference Venue Placeholder, e.g., IEEE Transactions on Audio, Speech, and Language Processing / ISMIR]},
  year      = {2026},
  month     = {11},
  doi       = {10.5281/zenodo.24047490},
  url       = {https://doi.org/10.5281/zenodo.24047490}
}
```
For machine-readable citation metadata, refer to [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json).
