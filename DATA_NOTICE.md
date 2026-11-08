# Dataset Notice, Provenance, and Licensing

## MOODWAAVE Data Governance Policy

This repository contains reproducible machine learning pipelines, pre-extracted acoustic feature tables, and evaluation code for the **MOODWAAVE** audio analysis architecture.

### Audio Redistribution Policy
To strictly comply with copyright laws and dataset distribution licenses, **no raw audio files (`.wav`, `.mp3`, `.au`, `.ogg`, `.flac`) are distributed in this repository**. Researchers wishing to execute audio processing or feature extraction from raw waveforms must obtain the original audio datasets directly from the upstream sources cited below.

### Derived Tabular Features Policy
Pre-extracted acoustic tabular features (such as 113-dimensional librosa feature vectors, embeddings, and emotion/genre metadata) are included in `release/tables/` **only where the original dataset licensing terms and conditions explicitly permit the distribution of derived research features and metadata**. Each dataset and model's licensing terms are detailed and quoted below.

---

## 1. Upstream Audio Datasets

### 1.1 GTZAN Genre Collection
* **Authors / Provenance**: George Tzanetakis and Perry Cook (2002), *"Musical genre classification of audio signals"*, IEEE Transactions on Speech and Audio Processing.
* **Composition**: 1,000 audio tracks (each 30 seconds, 22,050 Hz, 16-bit mono WAV) across 10 musical genres (100 tracks per genre).
* **Integrity & Fault Note**: In MOODWAAVE experiments, exactly 929 clean tracks are evaluated after removing 70 faulty/distorted excerpts (and the unreadable `jazz.00054.wav`). Kereliuk-Sturm artist-disjoint splits are provided to eliminate repetition leakage.
* **License & Terms**: Non-Commercial Academic Research Use.
  > *"The dataset was collected for research purposes in music genre classification... Marsyas and the GTZAN dataset are released for non-commercial academic research use."*
* **Derived Feature Tables**: `release/tables/gtzan_features.csv` contains mathematical acoustic descriptors (MFCCs, spectral contrast, tonnetz, chroma, temporal stats) derived for research reproducibility and distributed under non-commercial academic research terms.
* **Upstream Sources**:
  - Official Marsyas repository: [http://marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html)
  - Kaggle mirror: [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

### 1.2 PMEmo Pop Music Emotion Dataset
* **Authors / Provenance**: Kejun Zhang, Haoqian Shen, et al. (2018), *"The PMEmo Dataset for Pop Music Emotion Analysis"*, Proceedings of the 2018 ACM Multimedia Conference.
* **Composition**: 767 popular music chorus excerpts annotated with static and dynamic continuous Valence and Arousal ratings in \([0, 1]\) by 457 annotators.
* **License & Terms**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
  > *"This dataset is made available under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). You are free to share (copy and redistribute the material in any medium or format) and adapt (remix, transform, and build upon the material) for non-commercial purposes, provided appropriate credit is given and derivatives are shared under the same license."*
* **Derived Feature Tables**: `release/tables/pmemo_ready.csv` contains 113-dimensional acoustic feature representations and valence/arousal labels, distributed in accordance with CC BY-NC-SA 4.0.
* **Upstream Source**:
  - GitHub repository: [https://github.com/HaoqianShen/PMEmo](https://github.com/HaoqianShen/PMEmo)

---

### 1.3 FMA: Free Music Archive
* **Authors / Provenance**: Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson (2017), *"FMA: A Dataset for Music Analysis"*, 18th International Society for Music Information Retrieval Conference (ISMIR 2017).
* **Composition**: 106,574 tracks spanning 161 genres; the `fma_small` subset comprises 8,000 tracks (30 seconds, 8 balanced genres). MOODWAAVE evaluates an artist-clean balanced subset of 800 tracks across 147 unique artists (`release/tables/r3c_fma_800_metadata.csv`).
* **License & Terms**: Creative Commons Attribution 4.0 International (CC BY 4.0).
  > *"We release all metadata and features under the Creative Commons Attribution 4.0 International License (CC BY 4.0). The audio files are released under the Creative Commons licenses chosen by each artist."*
* **Derived Feature Tables**: Derived feature matrices and metadata are distributed under CC BY 4.0.
* **Upstream Source**:
  - GitHub repository: [https://github.com/mdeff/fma](https://github.com/mdeff/fma)
  - Dataset archive: [https://os.unil.cloud.switch.ch/fma/fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)

---

### 1.4 DEAM: Database for Emotion Analysis in Music
* **Authors / Provenance**: Anna Aljanaki, Yi-Hsuan Yang, Mohammad Soleymani (2017), *"Developing a benchmark dataset for emotional analysis of music"*, PLOS ONE 12(3); Zenodo repository.
* **Composition**: 1,802 royalty-free musical excerpts annotated with continuous Valence and Arousal ratings (per-second dynamic and static averages).
* **License & Terms**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
  > *"Creative Commons Attribution Non Commercial 4.0 International (CC BY-NC 4.0). You are free to copy and redistribute the material in any medium or format, and remix, transform, and build upon the material for non-commercial purposes, with attribution."*
* **Derived Feature Tables**: Tabular descriptors and emotion metadata (`release/tables/r3c_deam_metadata.csv`) are distributed under CC BY-NC 4.0.
* **Upstream Source**:
  - Zenodo: [https://zenodo.org/records/1290740](https://zenodo.org/records/1290740) (DOI: 10.5281/zenodo.1290740)

---

## 2. Pretrained Foundation Audio Models

The comparative benchmarks in MOODWAAVE evaluate deep foundation audio representations using open-source pretrained model weights. These weights are not redistributed in this repository; they are fetched at evaluation time via standard APIs and are licensed as follows:

| Model | Original Authors / Institution | Architecture / Weights | License | Upstream Link |
|:---|:---|:---|:---|:---|
| **VGGish** | Shawn Hershey et al., Google Research (2017) | 128-dim AudioSet CNN embedding | **Apache License 2.0** | [Google AudioSet VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) |
| **PANNs** (CNN14) | Qiuqiang Kong et al., University of Surrey / ByteDance (2020) | 2048-dim AudioSet CNN embedding | **MIT License** | [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) |
| **MERT** (v1-95M) | Yizhi Li et al., m-a-p team / Hugging Face (2024) | 768-dim Self-Supervised Transformer | **Apache License 2.0** | [Hugging Face: m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) |
| **CLAP** | LAION-AI (Elizalde et al., 2023) / Microsoft (Wang et al., 2023) | 512-dim Contrastive Language-Audio | **Apache License 2.0 / MIT** | [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) |

---

## 3. Summary of Repository Permissions

* **Source Code (`release/scripts/`, `src/`)**: Released under the permissive **MIT License** (see `LICENSE`).
* **Derived Acoustic Features (`release/tables/`)**: Shared strictly for non-commercial scientific reproduction under CC BY 4.0 (for FMA derivatives), CC BY-NC-SA 4.0 (for PMEmo derivatives), CC BY-NC 4.0 (for DEAM derivatives), and academic non-commercial research terms (for GTZAN derivatives).
* **Audio Recordings**: Not distributed. Users are directed to the respective upstream archives.
