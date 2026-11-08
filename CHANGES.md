# MOODWAAVE Release Changelog

## Version 1.0.0 — Comprehensive Reproduction Suite Update (2026-11-03)

This release consolidates all research phases of the **MOODWAAVE** architecture, incorporating the foundational Rev2 consolidation, extensive statistical audits (Rev3-A), foundation model comparisons (Rev3-B), cross-dataset generalizability evaluations (Rev3-C), and a turnkey cross-platform benchmarking suite (Rev3-D).

---

### 1. New Features & Benchmark Expansions

* **Rev3-A: Expanded Statistical Auditing & Classical Model Families**:
  - Expanded cross-validation protocol to 5x5 Repeated Stratified Group CV (25 artist-disjoint folds) on GTZAN and PMEmo.
  - Added Nadeau-Bengio corrected resampled paired \(t\)-tests and non-parametric Wilcoxon signed-rank tests with Holm-Bonferroni correction.
  - Audited default vs. tuned LightGBM across all 25 folds, showing that default LightGBM achieves predictive parity with grid-tuned models with zero hyperparameter optimization overhead.
  - Formally audited P4 baseline scripts, verifying that artist-disjoint splitting was strictly enforced without data leakage.

* **Rev3-B: Pretrained Foundation Audio Models Comparison**:
  - Extracted deep audio representations using VGGish (128-dim), PANNs CNN14 (2048-dim), and MERT-v1-95M (768-dim) on CPU.
  - Evaluated linear probing heads (Ridge, MLP, LightGBM) under identical 5x5 artist-disjoint cross-validation.
  - Benchmarked CPU extraction latency, RAM footprint, and model parameter sizes:
    - MOODWAAVE-Lite is **13.5x faster than VGGish**, **15.6x faster than PANNs**, and **45.4x faster than MERT** while using **over 40x fewer parameters**.
  - Generated Pareto trade-off curves illustrating predictive accuracy/R2 versus single-threaded CPU latency.

* **Rev3-C: Cross-Dataset Generalizability & Audio Decoding Profiling**:
  - Evaluated generalizability on the Free Music Archive (FMA-small, 800 artist-balanced tracks) and DEAM (1,802 emotion tracks).
  - Quantified artist leakage on FMA-small, showing a 15.63 percentage-point drop when enforcing artist-clean splits.
  - Confirmed strict TOST equivalence between MOODWAAVE-Lite and Full-113 on DEAM Valence (\(\Delta R^2 = -0.0007, p < 10^{-10}\)) and Arousal (\(\Delta R^2 = +0.0003, p < 10^{-10}\)).
  - Demonstrated zero-shot cross-dataset transfer from PMEmo to DEAM without fine-tuning (\(r = 0.4485\) Valence, \(r = 0.5097\) Arousal).
  - Benchmarked audio decoding libraries, showing `soundfile` and `torchaudio` decode 30-second audio **3.4x faster than librosa** (31.7 ms vs. 108.8 ms).

* **Rev3-D: Turnkey Cross-Platform Benchmark Kit (`release/bench_kit/`)**:
  - Released a self-contained, turnkey benchmarking package for measuring single-threaded CPU latency and DSP library efficiency across hardware architectures.
  - Includes 9 pre-trained serialized LightGBM models, 50 canonical GTZAN clips, and automated aggregation scripts.

---

### 2. Documentation & License Corrections

* **Audit & Correction of P3 Selection Methodology in `README.md`**:
  - **Corrected Inaccurate Description**: Fixed legacy references in `README.md` that incorrectly described P3 as "100-bootstrap feature selection".
  - **Actual Code Verification**: Audited `p3_01_run_gtzan_selection.py` and `p3_02_run_pmemo_selection.py` to confirm the exact implementation: **10 repeated 70/30 artist-disjoint hold-out splits with inner 5x2 grouped CV on 95 subsets and a 1-standard-error (1-SE) selection rule**.
  - Updated all table and script descriptions in the reproduction guide to accurately reflect the 10 repeated hold-out methodology.
* **License Clarity (`LICENSE`)**:
  - Explicitly specified that the MIT License governs computer software and source code only.
  - Clarified that pre-extracted tabular features and metadata are distributed under their respective upstream research licenses.
* **Dataset Governance (`DATA_NOTICE.md`)**:
  - Explicitly stated that derived feature tables are included only where upstream licensing explicitly permits research derivatives.
  - Specified verified upstream licensing status for GTZAN, PMEmo, FMA, and DEAM ("unverified, check source" where unconfirmed).
  - Listed accurate licensing and pretraining data information for foundation models: VGGish (Apache 2.0, YouTube-8M), PANNs CNN14 (MIT, AudioSet), and MERT-v1-95M (CC-BY-NC-4.0, 20,000h internet music). CLAP was removed.
* **Citation & Metadata (`CITATION.cff` & `.zenodo.json`)**:
  - Updated `CITATION.cff` and `README.md` with minted Zenodo DOI (`10.5281/zenodo.24047490`), release date `2026-11-03`, version `1.0.0`, and comprehensive keywords.
  - Added `.zenodo.json` to automate GitHub-to-Zenodo deposit metadata ingestion.
* **Git Publishing Guide (`PUSH_STEPS.md`) & `.gitignore`**:
  - Added `.gitignore` to prevent committing raw audio binaries or system artifacts.
  - Updated `PUSH_STEPS.md` with exact terminal commands and pre-flight auditing steps, explicitly instructing the user to review and execute git push manually.

---

### 3. Requirements & Environment Updates

* **Core Environment (`release/requirements.txt`)**:
  - Pinned exact dependencies for Python 3.13.7 (64-bit), including `nnAudio==0.3.4` and CPU-only `torch==2.15.0+cpu`.
* **Foundation Models Environment (`release/requirements_embeddings.txt`)**:
  - Created a separate requirements specification for Python 3.11 / 3.12 to support `transformers`, `panns-inference`, and `torchlibrosa`.

---

### 4. Itemized File Additions & Modifications

#### Scripts (`release/scripts/`)
| File | Action | Purpose |
|:---|:---|:---|
| `r3a_common.py` | Added | Rev3-A shared paths, hyperparameter grids, and statistical testing routines |
| `r3a_01_run_gtzan.py` | Added | Rev3-A 5x5 Repeated Stratified Group CV runner on GTZAN |
| `r3a_02_run_pmemo.py` | Added | Rev3-A 5x5 Repeated Group CV runner on PMEmo with TOST |
| `r3a_03_default_vs_tuned_lightgbm.py` | Added | Rev3-A Default vs. Tuned LightGBM evaluation script |
| `r3a_04_profile_models.py` | Added | Rev3-A Latency and memory profiler for classical models |
| `r3a_05_analyze_and_report.py` | Added | Rev3-A Results compiler, audit against P4, and figure generator |
| `r3a_audit_p4.py` | Added | Rev3-A Verification script proving P4 artist-disjoint splitting |
| `r3b_common.py` | Added | Rev3-B Shared foundation model configuration and loader utilities |
| `r3b_01_extract_panns.py` | Added | Rev3-B PANNs CNN14 2048-dim feature extraction on CPU |
| `r3b_02_extract_mert.py` | Added | Rev3-B MERT-v1-95M 768-dim feature extraction on CPU |
| `r3b_03_evaluate_embeddings_heads.py` | Added | Rev3-B Linear and tree probing head evaluations on embeddings |
| `r3b_04_statistical_tests.py` | Added | Rev3-B Statistical tests comparing MOODWAAVE vs. foundation models |
| `r3b_05_profile_efficiency.py` | Added | Rev3-B CPU latency, RAM, and model parameter profiling |
| `r3b_06_analyze_and_report.py` | Added | Rev3-B Pareto trade-off compiler and plotting script |
| `r3c_common.py` | Added | Rev3-C Cross-dataset paths, FMA, DEAM, and GTZAN utilities |
| `r3c_01_extract_fma.py` | Added | Rev3-C Feature extraction and metadata filtering for FMA-small |
| `r3c_02_extract_deam.py` | Added | Rev3-C Feature extraction and metadata filtering for DEAM |
| `r3c_03_fma_leakage_and_models.py` | Added | Rev3-C FMA-small artist leakage audit and model benchmarks |
| `r3c_04_deam_models_and_zeroshot.py` | Added | Rev3-C DEAM emotion modeling and PMEmo-to-DEAM zero-shot transfer |
| `r3c_05_genre_transfer_gtzan_to_fma.py` | Added | Rev3-C GTZAN-to-FMA cross-dataset genre classification transfer |
| `r3c_06_profile_efficiency.py` | Added | Rev3-C Audio decoding library benchmark and duration scaling |
| `r3c_07_analyze_and_report.py` | Added | Rev3-C Master cross-dataset report compiler |

#### Tables (`release/tables/`)
| File | Action | Purpose |
|:---|:---|:---|
| `splits.json` | Added | Master artist-disjoint CV splits for GTZAN and PMEmo |
| `r3a_gtzan_summary.csv` | Added | Rev3-A GTZAN 5x5 CV predictive accuracy summary |
| `r3a_gtzan_paired_tests.csv` | Added | Rev3-A Corrected resampled t-tests and Wilcoxon tests on GTZAN |
| `r3a_pmemo_summary.csv` | Added | Rev3-A PMEmo 5x5 CV Valence and Arousal R2 summary |
| `r3a_pmemo_paired_tests.csv` | Added | Rev3-A Statistical test results on PMEmo |
| `r3a_default_vs_tuned_lgbm.csv` | Added | Rev3-A Default vs. Tuned LightGBM comparison table |
| `r3a_model_profiles_summary.csv` | Added | Rev3-A Inference latency and memory footprint table |
| `r3a_vs_p4_comparison.csv` | Added | Rev3-A Verification audit table comparing with P4 |
| `r3a_gtzan_folds.csv` | Added | Rev3-A 25 individual fold results on GTZAN |
| `r3a_pmemo_folds.csv` | Added | Rev3-A 25 individual fold results on PMEmo |
| `r3b_gtzan_summary.csv` | Added | Rev3-B Foundation embeddings predictive accuracy on GTZAN |
| `r3b_gtzan_paired_tests.csv` | Added | Rev3-B Statistical tests comparing MOODWAAVE vs. foundation models |
| `r3b_pmemo_summary.csv` | Added | Rev3-B Foundation embeddings regression performance on PMEmo |
| `r3b_pmemo_paired_tests.csv` | Added | Rev3-B Statistical tests for mood regression on PMEmo |
| `r3b_embeddings_efficiency_profiles.csv` | Added | Rev3-B CPU latency, RAM, and model parameter sizes |
| `r3b_gtzan_folds.csv` | Added | Rev3-B Individual fold results for all probing heads on GTZAN |
| `r3b_pmemo_folds.csv` | Added | Rev3-B Individual fold results for all probing heads on PMEmo |
| `r3c_cross_dataset_master_summary.csv` / `.json` | Added | Rev3-C Master cross-dataset benchmark summary |
| `r3c_fma_leakage_summary.csv` | Added | Rev3-C FMA-small random vs. artist-disjoint leakage comparison |
| `r3c_fma_model_families_summary.csv` | Added | Rev3-C Classical model families benchmark on FMA-small |
| `r3c_deam_model_families_summary.csv` | Added | Rev3-C Classical model families benchmark on DEAM |
| `r3c_gtzan_to_fma_transfer.csv` | Added | Rev3-C GTZAN to FMA-small genre transfer evaluation |
| `r3c_pmemo_to_deam_zeroshot.csv` | Added | Rev3-C PMEmo to DEAM zero-shot emotion transfer results |
| `r3c_efficiency_breakdown_summary.csv` | Added | Rev3-C Feature extraction latency vs. audio duration |
| `r3c_decoding_overhead_comparison.csv` | Added | Rev3-C Audio decoding library benchmark (librosa vs. soundfile vs. torchaudio) |
| `r3c_fma_leakage_folds.csv` | Added | Rev3-C Fold-by-fold leakage results on FMA-small |
| `r3c_fma_model_families_folds.csv` | Added | Rev3-C Fold-by-fold model family results on FMA-small |
| `r3c_deam_model_families_folds.csv` | Added | Rev3-C Fold-by-fold model family results on DEAM |
| `r3c_fma_800_metadata.csv` | Added | Rev3-C Metadata and annotations for 800 artist-balanced FMA tracks |
| `r3c_deam_metadata.csv` | Added | Rev3-C Metadata and annotations for 1,802 DEAM tracks |

#### Figures (`release/figures/`)
| File | Action | Purpose |
|:---|:---|:---|
| `r3a_genre_accuracy.png` | Added | Rev3-A 5x5 CV genre accuracy across classical model families |
| `r3a_mood_r2.png` | Added | Rev3-A 5x5 CV Valence & Arousal R2 across classical model families |
| `r3b_genre_pareto.png` | Added | Rev3-B Genre Accuracy vs. CPU Latency Pareto frontier |
| `r3b_mood_pareto.png` | Added | Rev3-B Valence/Arousal R2 vs. CPU Latency Pareto frontier |
| `r3c_leakage_cross_dataset_comparison.png` | Added | Rev3-C Cross-dataset artist leakage comparison (GTZAN vs. FMA) |
| `r3c_decoding_overhead_and_latency.png` | Added | Rev3-C Audio decoding overhead and feature extraction latency vs. duration |
| `r3c_deam_model_families_r2_tost.png` | Added | Rev3-C DEAM emotion regression performance and TOST equivalence intervals |
| `r3d_bench_platform_latency_and_breakdown.png` | Added | Rev3-D Benchmark kit platform latency and feature cost breakdown |

#### Benchmark Kit (`release/bench_kit/`)
| File | Action | Purpose |
|:---|:---|:---|
| `bench_platform.py` | Added | Standalone single-threaded latency benchmark runner |
| `aggregate_results.py` | Added | Multi-platform benchmark compiler and visualizer |
| `export_bench_clips.py` | Added | GTZAN 50-clip audio exporter |
| `BENCH_README.md` | Added | Turnkey benchmarking user guide |
| `RESULTS.md` | Added | Empirical benchmarking findings and platform logs |
| `requirements.txt` | Added | Benchmark kit dependencies |
| `bench_summary_table.csv` / `.json` | Added | Summary of platform benchmark results |
| `group_comparison_table.csv` | Added | Feature group computational latency breakdown |
| `models/*.pkl` (9 files) | Added | Pretrained serialized MOODWAAVE LightGBM models |

#### Documentation & Configuration
| File | Action | Purpose |
|:---|:---|:---|
| `release/README.md` | Modified | Audited descriptions, corrected P3 methodology, added Rev3 guides |
| `release/DATA_NOTICE.md` | Modified | Explicitly stated derived feature terms, quoted licenses for 4 datasets and 4 models |
| `release/LICENSE` | Modified | Specified MIT for software/code only, referenced DATA_NOTICE.md |
| `release/CITATION.cff` | Modified | Updated DOI placeholder, release date, version, and keywords |
| `release/.zenodo.json` | Added | Machine-readable Zenodo deposit configuration for DOI minting |
| `release/.gitignore` | Added | Strict git exclusion for audio binaries and caches |
| `release/PUSH_STEPS.md` | Modified | Step-by-step git commands and pre-flight checklist |
| `release/requirements.txt` | Modified | Pinned Python 3.13 dependencies for core MOODWAAVE |
| `release/requirements_embeddings.txt` | Added | Dedicated dependencies for Python 3.11/3.12 foundation models |
| `release/CHANGES.md` | Added | Comprehensive itemized release changelog |

