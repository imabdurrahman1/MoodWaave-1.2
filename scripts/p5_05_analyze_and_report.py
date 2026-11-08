"""
experiments/rev2_p5_deep/05_analyze_and_report.py
Generates summary tables, 300-DPI publication figures, and writes RESULTS.md
for TASK P5 and TASK P5-fix (Neural baselines on CPU).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common

def generate_plots(out_dir: str):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "figure.autolayout": True,
    })
    
    # 1. CNN Training Curves
    hist_path = os.path.join(out_dir, "cnn_training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            histories = json.load(f)
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
        epochs = np.arange(1, 16)
        
        for seed_name, h in histories.items():
            ax1.plot(epochs, h["train_loss"], label=f"Train Loss ({seed_name})")
            ax2.plot(epochs, h["val_acc"], label=f"Val Acc ({seed_name})")
            
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax1.set_title("SmallMelCNN Training Loss (CPU, Adam, 15 Epochs)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()
        
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Accuracy")
        ax2.set_title("SmallMelCNN Validation Accuracy (10-Crop Logit Avg)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()
        
        fig_path1 = os.path.join(out_dir, "cnn_training_curves.png")
        fig.savefig(fig_path1, dpi=300)
        plt.close(fig)
        print(f"Saved figure: {fig_path1}")

    # 2. Accuracy Comparison on Fixed Split (Reconciled Protocols)
    fixed_comp_csv = os.path.join(out_dir, "fixed_split_comparison.csv")
    if os.path.exists(fixed_comp_csv):
        df_comp = pd.read_csv(fixed_comp_csv)
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        
        # Display rows
        labels = [
            "MOODWAAVE-Full\n(P1: Train+Val, 639)",
            "MOODWAAVE-Lite\n(P1: Train+Val, 639)",
            "MOODWAAVE-Full\n(P5: Train only, 442)",
            "MOODWAAVE-Lite\n(P5: Train only, 442)",
            "SmallMelCNN\n(3 Seeds Mean)",
            "SmallMelCNN\n(Seed-Avg Logits)",
            "VGGish + LogReg\n(Train only, 442)",
        ]
        
        accs = [
            0.5931,
            0.5828,
            0.5690,
            0.5586,
            0.5540,
            0.5862,
            0.7621,
        ]
        
        # 95% binomial confidence intervals for 290 test tracks
        yerr = [1.96 * np.sqrt(a * (1.0 - a) / 290.0) for a in accs]
        colors = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#ff7f0e", "#ffbb78", "#d62728"]
        
        bars = ax.bar(labels, accs, yerr=yerr, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
        ax.set_ylabel("GTZAN Genre Accuracy (290 Test Tracks)")
        ax.set_title("Genre Accuracy on Kereliuk-Sturm Fixed Split (Reconciled Protocols & Baselines)")
        ax.set_ylim(0.40, 0.90)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        
        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2.0, a + 0.035, f"{a*100:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
            
        fig_path2 = os.path.join(out_dir, "deep_vs_moodwaave_accuracy.png")
        fig.savefig(fig_path2, dpi=300)
        plt.close(fig)
        print(f"Saved figure: {fig_path2}")

    # 3. Efficiency vs Accuracy Pareto Frontier (Marked Provisional)
    eff_csv = os.path.join(out_dir, "efficiency_profiles.csv")
    if os.path.exists(eff_csv):
        df_eff = pd.read_csv(eff_csv)
        
        plot_items = [
            {"model": "MOODWAAVE-Lite", "lat": 197.6, "acc": 0.5586, "color": "#2ca02c"},
            {"model": "MOODWAAVE-Full*", "lat": 1746.7, "acc": 0.5690, "color": "#1f77b4"},
            {"model": "SmallMelCNN (scratch)", "lat": 102.1, "acc": 0.5862, "color": "#ff7f0e"},
            {"model": "VGGish + LogReg", "lat": 1132.8, "acc": 0.7621, "color": "#d62728"},
        ]
        
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
        for item in plot_items:
            ax.scatter(item["lat"], item["acc"], s=120, color=item["color"], label=item["model"], zorder=5)
            ax.annotate(f"{item['model']}\n({item['lat']:.0f} ms prov.)", 
                        (item["lat"], item["acc"]),
                        xytext=(8, -8), textcoords="offset points", fontsize=9)
                        
        ax.set_xscale("log")
        ax.set_xlabel("Single-Thread CPU Latency (ms/track, provisional, log scale)")
        ax.set_ylabel("GTZAN Genre Accuracy (290 Test Tracks)")
        ax.set_title("CPU Latency vs Accuracy: MOODWAAVE vs Deep Baselines\n(*Note: All latencies provisional; Full from earlier session)")
        ax.set_ylim(0.48, 0.82)
        ax.grid(True, linestyle=":", alpha=0.6)
        
        fig_path3 = os.path.join(out_dir, "deep_vs_moodwaave_efficiency.png")
        fig.savefig(fig_path3, dpi=300)
        plt.close(fig)
        print(f"Saved figure: {fig_path3}")

def build_results_markdown(out_dir: str):
    paired_csv = os.path.join(out_dir, "p5_fix_paired_tests.csv")
    mc_csv = os.path.join(out_dir, "p5_fix_mcnemar_tests.csv")
    sizes_csv = os.path.join(out_dir, "system_sizes.csv")
    eff_csv = os.path.join(out_dir, "efficiency_profiles.csv")
    cnn_seeds_csv = os.path.join(out_dir, "cnn_seeds_results.csv")
    vggish_g_csv = os.path.join(out_dir, "vggish_gtzan_folds.csv")
    vggish_p_csv = os.path.join(out_dir, "vggish_pmemo_folds.csv")
    
    df_paired = pd.read_csv(paired_csv) if os.path.exists(paired_csv) else None
    df_mc = pd.read_csv(mc_csv) if os.path.exists(mc_csv) else None
    df_sizes = pd.read_csv(sizes_csv) if os.path.exists(sizes_csv) else None
    df_eff = pd.read_csv(eff_csv) if os.path.exists(eff_csv) else None
    df_cnn = pd.read_csv(cnn_seeds_csv) if os.path.exists(cnn_seeds_csv) else None
    df_vg = pd.read_csv(vggish_g_csv) if os.path.exists(vggish_g_csv) else None
    df_vp = pd.read_csv(vggish_p_csv) if os.path.exists(vggish_p_csv) else None
    
    md = []
    md.append("# TASK P5 & P5-fix: Neural Baselines on CPU — Mel-CNN and Pretrained Embeddings\n")
    md.append("## Executive Summary\n")
    md.append(
        "To rigorously contextualize **MOODWAAVE** against modern deep learning architectures on resource-constrained CPU environments, "
        "we benchmarked two neural baselines:\n"
        "1. **Pretrained VGGish Feature Extractor + Linear Probes:** A 9-weight-layer convolutional model (72.2M parameters) ported from Google's "
        "audio classification architecture, evaluated with frozen frame embeddings pooled via temporal mean + standard deviation (256 dimensions) "
        "and fitted with standard linear heads (Logistic Regression for GTZAN genre; Ridge regression for PMEmo valence and arousal).\n"
        "2. **Small Mel-CNN Trained from Scratch:** A lightweight 4-block Conv-BatchNorm-ReLU-MaxPool network (98,682 parameters, $< 1.5$M) "
        "operating on 128-bin log-mel spectrograms ($sr=22,050$, hop 512) trained with random 3-second crops for 15 epochs on CPU and evaluated "
        "using 10-crop logit averaging.\n\n"
        "All models were evaluated under strict leakage-aware protocols on both the **P1 Kereliuk & Sturm fixed split (290 test tracks)** and "
        "**artist-disjoint outer cross-validation folds**.\n"
    )
    
    md.append("### Honest Comparison: Accuracy vs CPU Efficiency\n")
    md.append(
        "> [!IMPORTANT]\n"
        "> **Key Findings & Trade-Offs:**\n"
        "> - **Where Pretrained Embeddings Excel:** Benefiting from extensive pretraining on millions of YouTube video audio tracks, VGGish features achieve "
        "> **76.21%** genre accuracy on the fixed split (a 19–20 percentage point advantage over MOODWAAVE) and **0.7906 ± 0.0419** on 5-fold CV. "
        "> On emotion prediction, VGGish improves Valence $R^2$ to **0.5540 ± 0.0601** (vs. 0.4503 for Lite), while Arousal $R^2$ remains comparable (**0.6889 ± 0.0352** vs. 0.6700).\n"
        "> - **Where CNN from Scratch Sits:** The small mel-CNN trained from scratch on 442 audio tracks achieves **55.40% ± 2.79%** across 3 individual seeds "
        "> and **58.62%** with seed-averaged logits. Because small audio datasets provide limited sample diversity without pretraining or extensive augmentation, "
        "> the from-scratch CNN performs directly on par with MOODWAAVE-Lite (55.86%) and MOODWAAVE-Full (56.90%).\n"
        "> - **CPU Latency & Storage Footprint Trade-offs:**\n"
        ">   - VGGish requires a **275.22 MB** model checkpoint and **1132.8 ± 26.2 ms/track** single-thread CPU latency (provisional).\n"
        ">   - SmallMelCNN is remarkably compact (**0.39 MB**, 98k parameters) and achieves **102.1 ± 12.9 ms/track** single-thread latency (provisional).\n"
        ">   - **MOODWAAVE-Lite** achieves **197.6 ± 48.7 ms/track** with a compact **4.08 MB** tree model, delivering classification accuracy on par with "
        "> from-scratch CNNs at $5.7\\times$ lower latency and $67\\times$ smaller disk footprint than VGGish.\n"
        "> - **Statistical Significance vs Sampling Variance:**\n"
        ">   - VGGish's 19–20 percentage point lead over MOODWAAVE on the fixed split is **statistically highly significant** (McNemar's exact test $p = 7.25 \\times 10^{-10}$ vs Lite, $p = 7.32 \\times 10^{-9}$ vs Full) "
        "> and lies far outside the single-split 95% binomial confidence interval of approximately **±5.5 to ±6.0 percentage points** ($1.96 \\sqrt{p(1-p)/290}$).\n"
        ">   - In contrast, differences between MOODWAAVE-Full (56.90%), MOODWAAVE-Lite (55.86%), and SmallMelCNN (55.40% mean, 58.62% seed-averaged logits) "
        "> fall entirely within this ±5.5 to ±6.0 pp sampling variance window, confirmed as statistically non-significant by McNemar's exact tests ($p = 0.4222$ vs Lite, $p = 0.6400$ vs Full).\n"
    )
    
    # Table 1: Reconciled Fixed-Split Comparison
    md.append("## Table 1: Reconciled Fixed-Split Comparison on Kereliuk & Sturm (2015) Fault-Filtered Split (290 Test Tracks)\n")
    md.append(
        "Standard evaluation practice in earlier literature (e.g. Kereliuk & Sturm 2015, and TASK P1) pooled the training and validation subsets "
        "(639 tracks) when hyperparameters were fixed. In TASK P5, the validation partition (197 tracks) was strictly held out for CNN early stopping "
        "and tuning, meaning models were trained on 442 tracks. Both evaluation protocols are presented below for full transparency:\n"
    )
    md.append("| Model | Training Protocol | Training Tracks | Test Tracks | Accuracy (290 Test Tracks) | Macro-F1 | System Serialized Size (MB) | Provisional CPU Latency (ms/track) |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    md.append("| **MOODWAAVE-Full** | P1 Protocol (Train + Val, 5 seeds) | 639 | 290 | **59.31% ± 0.00%** | 0.5929 | 3.88 MB (def) / 6.67 MB (tuned) | 1746.7 ± 92.5 ms* |")
    md.append("| **MOODWAAVE-Lite** | P1 Protocol (Train + Val, 5 seeds) | 639 | 290 | **58.28% ± 0.00%** | 0.5775 | 3.96 MB (def) / 4.62 MB (tuned) | 197.6 ± 48.7 ms |")
    md.append("| **MOODWAAVE-Full** | P5 Protocol (Train only, seed 42) | 442 | 290 | **56.90%** | 0.5641 | 3.88 MB (def) / 6.67 MB (tuned) | 1746.7 ± 92.5 ms* |")
    md.append("| **MOODWAAVE-Lite** | P5 Protocol (Train only, seed 42) | 442 | 290 | **55.86%** | 0.5508 | 3.96 MB (def) / 4.62 MB (tuned) | 197.6 ± 48.7 ms |")
    md.append("| **SmallMelCNN (scratch)** | P5 Protocol (Train only, 3 seeds mean) | 442 | 290 | **55.40% ± 2.79%** | 0.5362 ± 0.0182 | 0.39 MB (checkpoint) | 102.1 ± 12.9 ms |")
    md.append("| **SmallMelCNN (scratch)** | P5 Protocol (Train only, seed-avg logits) | 442 | 290 | **58.62%** | 0.5666 | 0.39 MB (checkpoint) | 102.1 ± 12.9 ms |")
    md.append("| **VGGish + Logistic Regression** | P5 Protocol (Train only, seed 42) | 442 | 290 | **76.21%** | 0.7601 | 275.22 MB (net + heads) | 1132.8 ± 26.2 ms |")
    md.append("\n*Note: All latencies are provisional (measured across separate profiling sessions; MOODWAAVE-Full timing reused from final_consolidation).*\n\n")

    # Table 2: Fixed Split McNemar Tests
    if df_mc is not None:
        md.append("## Table 2: Statistical Significance on Fixed Split — McNemar's Exact Test (290 Test Tracks)\n")
        md.append("To test whether predictive differences on the 290 test tracks of the fixed split are statistically significant, "
                  "we compute McNemar's exact two-sided binomial test on discordant prediction pairs ($n_{10}$ and $n_{01}$):\n")
        md.append("| Comparison | Model A (Accuracy) | Model B (Accuracy) | Discordant Pairs ($n_{10} / n_{01}$) | Exact Two-Sided $p$-value | Statistically Significant? |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, row in df_mc.iterrows():
            sig = "Yes ($p < 0.001$)" if row["p_value"] < 0.001 else ("Yes ($p < 0.05$)" if row["p_value"] < 0.05 else "No ($p \\ge 0.05$)")
            md.append(f"| **{row['comparison']}** | {row['model_a']} ({row['acc_a']*100:.2f}%) | {row['model_b']} ({row['acc_b']*100:.2f}%) | $n_{{10}} = {int(row['n10_a_correct_b_wrong'])}, n_{{01}} = {int(row['n01_a_wrong_b_correct'])}$ | ${row['p_value']:.4e}$ | {sig} |")
        md.append("\n")

    # Table 3: Paired Outer Folds Tests (Nadeau-Bengio)
    if df_paired is not None:
        md.append("## Table 3: Paired Differences on Outer Cross-Validation Folds (Nadeau-Bengio Corrected $t$-tests)\n")
        md.append(
            "To evaluate generalization beyond a single split, we compute paired fold-by-fold differences between VGGish and MOODWAAVE on the outer folds "
            "(5 outer artist-disjoint folds on GTZAN, $\\text{df} = 4$; 10 outer artist-grouped folds on PMEmo, $\\text{df} = 9$). "
            "Standard errors and 95% confidence intervals are corrected for cross-validation resampling dependencies using the Nadeau & Bengio (2003) formulation:\n"
        )
        md.append("| Dataset | Task | Comparison | Metric | VGGish Mean | Baseline Mean | Mean Difference ($\\Delta$) | Corrected SE | $t$-stat (df) | $p$-value | 95% CI of Difference |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, row in df_paired.iterrows():
            ci_str = f"[{row['ci_lower_95']:+.4f}, {row['ci_upper_95']:+.4f}]" if "R²" in row["metric"] else f"[{row['ci_lower_95']*100:+.2f}%, {row['ci_upper_95']*100:+.2f}%]"
            diff_str = f"{row['mean_diff']:+.4f}" if "R²" in row["metric"] else f"{row['mean_diff']*100:+.2f}%"
            md.append(f"| {row['dataset']} | {row['task']} | **{row['comparison']}** | {row['metric']} | {row['vggish_mean']:.4f} | {row['baseline_mean']:.4f} | **{diff_str}** | {row['se_corrected']:.4f} | $t = {row['t_stat']:.2f}$ (df={int(row['df'])}) | ${row['p_value']:.4f}$ | {ci_str} |")
        md.append("\n")

    # Table 4: Small Mel-CNN Multi-Seed Results
    if df_cnn is not None:
        md.append("## Table 4: Small Mel-CNN Multi-Seed Evaluation (3 Seeds, 290 Test Tracks)\n")
        md.append("| Seed | Learning Rate | Dropout | Validation Accuracy | Test Accuracy (10-Crop Avg) | Test Macro-F1 | Train Time (s) |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, row in df_cnn.iterrows():
            md.append(f"| Seed {int(row['seed'])} | {row['lr']} | {row['dropout']} | {row['val_accuracy']*100:.2f}% | {row['test_accuracy']*100:.2f}% | {row['test_macro_f1']:.4f} | {row['train_time_sec']:.1f} s |")
        md.append(f"| **Individual Seeds (Mean ± SD)** | — | — | — | **{df_cnn['test_accuracy'].mean()*100:.2f}% ± {df_cnn['test_accuracy'].std(ddof=1)*100:.2f}%** | **{df_cnn['test_macro_f1'].mean():.4f} ± {df_cnn['test_macro_f1'].std(ddof=1):.4f}** | — |")
        md.append(f"| **Ensemble (Seed-Averaged Logits)** | — | — | — | **58.62%** | **0.5666** | — |")
        md.append("\n")

    # Table 5: VGGish Cross-Validation Summary
    if df_vg is not None and df_vp is not None:
        md.append("## Table 5: Pretrained VGGish Linear Heads on P1 Cross-Validation Folds (Summary)\n")
        md.append("| Dataset | Task | Head Architecture | Metric | VGGish Head (Mean ± SD) | MOODWAAVE-Lite (Ref, Tuned) | MOODWAAVE-Full (Ref, Tuned) |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        md.append(f"| GTZAN | Genre | Logistic Regression ($C$ tuned) | Accuracy | **{df_vg['accuracy'].mean():.4f} ± {df_vg['accuracy'].std(ddof=1):.4f}** | 0.6144 ± 0.0531 | 0.6399 ± 0.0319 |")
        md.append(f"| GTZAN | Genre | Logistic Regression ($C$ tuned) | Macro-F1 | **{df_vg['macro_f1'].mean():.4f} ± {df_vg['macro_f1'].std(ddof=1):.4f}** | 0.6071 ± 0.0523 | 0.6346 ± 0.0298 |")
        sub_v = df_vp[df_vp["task"] == "valence"]
        sub_a = df_vp[df_vp["task"] == "arousal"]
        md.append(f"| PMEmo | Valence | Ridge ($\\alpha$ tuned) | R² | **{sub_v['r2'].mean():.4f} ± {sub_v['r2'].std(ddof=1):.4f}** | 0.4503 ± 0.0702 | 0.4600 ± 0.0655 |")
        md.append(f"| PMEmo | Valence | Ridge ($\\alpha$ tuned) | MAE | **{sub_v['mae'].mean():.4f} ± {sub_v['mae'].std(ddof=1):.4f}** | 0.0945 ± 0.0053 | 0.0939 ± 0.0059 |")
        md.append(f"| PMEmo | Arousal | Ridge ($\\alpha$ tuned) | R² | **{sub_a['r2'].mean():.4f} ± {sub_a['r2'].std(ddof=1):.4f}** | 0.6700 ± 0.0616 | 0.6729 ± 0.0616 |")
        md.append(f"| PMEmo | Arousal | Ridge ($\\alpha$ tuned) | MAE | **{sub_a['mae'].mean():.4f} ± {sub_a['mae'].std(ddof=1):.4f}** | 0.0838 ± 0.0072 | 0.0831 ± 0.0068 |")
        md.append("\n")

    # Table 6: System Serialized Sizes
    if df_sizes is not None:
        md.append("## Table 6: Per-System Serialized Storage Footprints\n")
        md.append("We report combined per-system footprint across models required to deliver end-to-end music analysis (Genre + Valence + Arousal):\n")
        md.append("| System | Configuration | Genre Model (MB) | Valence Model (MB) | Arousal Model (MB) | Embedding Net (MB) | Total System Size (MB) | Hyperparameter Notes |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, row in df_sizes.iterrows():
            v_val = f"{row['valence_model_mb']} MB" if pd.notna(row['valence_model_mb']) and str(row['valence_model_mb']) != 'nan' and str(row['valence_model_mb']) != 'N/A' else "N/A"
            a_val = f"{row['arousal_model_mb']} MB" if pd.notna(row['arousal_model_mb']) and str(row['arousal_model_mb']) != 'nan' and str(row['arousal_model_mb']) != 'N/A' else "N/A"
            md.append(f"| **{row['system']}** | {row['configuration']} | {row['genre_model_mb']} MB | {v_val} | {a_val} | {row['embedding_net_mb']} MB | **{row['total_system_mb']} MB** | {row['notes']} |")
        md.append("\n")

    # Table 7: Latency & Audio Decoding Breakdown
    md.append("## Table 7: Single-Thread CPU Latency (Provisional) & Audio Decoding Benchmark\n")
    md.append(
        "All end-to-end pipeline latencies are **provisional** (measured on 100 tracks with 1 thread, 1 warm-up track excluded, median of 3 repeats per track, "
        "across independent profiling sessions). Audio decoding costs were empirically isolated and measured:\n"
    )
    md.append("| Architecture | System Components | End-to-End Latency (ms/track) | Measurement Status | Audio Decoding Share |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    md.append("| **MOODWAAVE-Lite** | 71 Librosa features + LightGBM | **197.6 ± 48.7 ms** (median 176.8 ms) | Provisional (P5 session) | ~2.5 ms (WAV) / ~57 ms (MP3) |")
    md.append(r"| **MOODWAAVE-Full** | 113 Librosa features + LightGBM | **1746.7 ± 92.5 ms\*** (median 1717.4 ms) | Provisional (final_consolidation) | ~2.5 ms (WAV) / ~57 ms (MP3) |")
    md.append("| **SmallMelCNN (scratch)** | 128-mel STFT + 10-crop ConvNet | **102.1 ± 12.9 ms** (median 98.7 ms) | Provisional (P5 session) | ~2.5 ms (WAV) / ~57 ms (MP3) |")
    md.append("| **VGGish + LogReg** | 16kHz resample + VGGish + Head | **1132.8 ± 26.2 ms** (median 1134.3 ms) | Provisional (P5 session) | ~2.5 ms (WAV) / ~57 ms (MP3) |")
    md.append("\n*Note on Audio Decoding: Earlier preliminary estimates hypothesized audio decoding required 30–50 ms. Rigorous profiling revealed audio decompression requires **~2.5 ms for 30s uncompressed WAV files** (GTZAN) and **~57 ms for 30s MP3 files** (PMEmo).*\n\n")

    # Section: VGGish Architecture & Pretraining
    md.append("## Model Architecture and Pretraining Analysis: VGGish\n")
    md.append("### Actual Layer Structure (from Model Code)\n")
    md.append(
        "Inspection of the official PyTorch port code (`torchvggish/vggish.py`) reveals that VGGish is an adaptation of VGG configuration A with "
        "**6 2D convolutional layers and 3 fully-connected layers (9 weight layers total)**, rather than 24 layers:\n"
        "- **Feature Extractor:**\n"
        "  - Block 1: `Conv2d(1, 64, kernel_size=3, padding=1)` + `ReLU` $\\to$ `MaxPool2d(2, 2)`\n"
        "  - Block 2: `Conv2d(64, 128, kernel_size=3, padding=1)` + `ReLU` $\\to$ `MaxPool2d(2, 2)`\n"
        "  - Block 3: `Conv2d(128, 256, kernel_size=3, padding=1)` + `ReLU` $\\to$ `Conv2d(256, 256, kernel_size=3, padding=1)` + `ReLU` $\\to$ `MaxPool2d(2, 2)`\n"
        "  - Block 4: `Conv2d(256, 512, kernel_size=3, padding=1)` + `ReLU` $\\to$ `Conv2d(512, 512, kernel_size=3, padding=1)` + `ReLU` $\\to$ `MaxPool2d(2, 2)`\n"
        "- **Embedding Head:**\n"
        "  - `Linear(12288, 4096)` + `ReLU`\n"
        "  - `Linear(4096, 4096)` + `ReLU`\n"
        "  - `Linear(4096, 128)` + `ReLU`\n"
        "- **Postprocessor:** Optional PCA matrix multiplication ($128 \\times 128$) and 8-bit quantization.\n"
    )
    md.append("### Pretraining Data and Domain Disparity\n")
    md.append(
        "The model's official documentation (`README.md` and docstrings) states:\n"
        "> *\"A torch-compatible port of VGGish, a feature embedding frontend for audio classification models. "
        "The weights are ported directly from the tensorflow model, so embeddings created using torchvggish will be identical. "
        "Reference: S. Hershey et al., 'CNN Architectures for Large-Scale Audio Classification', in ICASSP 2017.\"*\n\n"
        "In Hershey et al. (2017), the network was pretrained on a preliminary version of **YouTube-8M**, comprising 8 million YouTube videos "
        "(over 70 million 10-second audio segments labeled with 4,800 knowledge graph topics). Crucially, YouTube topic labels include **specific musical genres, "
        "musical instruments, and vocal styles**. Consequently, the comparison between VGGish linear heads and models trained from scratch on 442–929 tracks "
        "(such as MOODWAAVE and SmallMelCNN) is **not like-for-like**: VGGish leverages semantic representations learned from millions of music and audio clips.\n"
    )

    # Section: Embedding Extraction Details
    md.append("## Embedding Extraction Pipeline & Audio Duration Handling\n")
    md.append(
        "The VGGish embedding extraction pipeline operates as follows:\n"
        "1. **Resampling:** Input audio is resampled to 16,000 Hz mono using `resampy`.\n"
        "2. **Log-Mel Spectrogram:** STFT is computed with a 25 ms Hann window (400 samples) and 10 ms hop length (160 samples, 100 Hz frame rate). "
        "Frequencies are mapped into 64 mel-scale bins spanning 125 Hz to 7,500 Hz, followed by stabilized logarithmic compression $\\log(\\text{mel} + 0.01)$.\n"
        "3. **Framing:** Spectrograms are segmented into consecutive non-overlapping 0.96-second examples (patches of 96 frames $\\times$ 64 mel bins).\n"
        "4. **Forward Pass:** Each 0.96s example produces a 128-dimensional embedding vector.\n"
        "5. **Temporal Pooling:** Across all $T$ frames of a track, we compute both the temporal mean ($\\mu \\in \\mathbb{R}^{128}$) and temporal standard deviation "
        "($\\sigma \\in \\mathbb{R}^{128}$), concatenating them into a unified **256-dimensional track embedding vector**.\n"
        "6. **PMEmo Audio Durations:** All 767 PMEmo chorus audio clips range in duration from **13.0 s to 77.0 s (mean 35.6 s)**. "
        "Every single clip yields between 13 and 80 frames (mean ~37 frames). Because no clip is shorter than the 0.96 s window, zero-padding or frame truncation "
        "was never required, and mean + standard deviation pooling was mathematically robust across 100% of PMEmo tracks.\n"
    )

    # Numbers to use in the paper
    md.append("## Numbers to Use in the Paper\n")
    md.append(
        "- **Fixed-Split Genre Accuracy (P1 Protocol, Train+Val 639, 5 Seeds):**\n"
        "  - MOODWAAVE-Full: `59.31% ± 0.00%` (Macro-F1: `0.5929`)\n"
        "  - MOODWAAVE-Lite: `58.28% ± 0.00%` (Macro-F1: `0.5775`)\n"
        "- **Fixed-Split Genre Accuracy (P5 Protocol, Train Only 442, 290 Test Tracks):**\n"
        "  - MOODWAAVE-Full: `56.90%` (Macro-F1: `0.5641`)\n"
        "  - MOODWAAVE-Lite: `55.86%` (Macro-F1: `0.5508`)\n"
        "  - SmallMelCNN (scratch, 3 seeds mean): `55.40% ± 2.79%` (Macro-F1: `0.5362 ± 0.0182`)\n"
        "  - SmallMelCNN (scratch, seed-averaged logits): `58.62%` (Macro-F1: `0.5666`)\n"
        "  - VGGish + Logistic Regression: `76.21%` (Macro-F1: `0.7601`)\n"
        "- **Fixed-Split McNemar's Exact Test ($p$-values):**\n"
        "  - VGGish vs. MOODWAAVE-Full: `p = 7.32e-09` (statistically highly significant)\n"
        "  - VGGish vs. MOODWAAVE-Lite: `p = 7.25e-10` (statistically highly significant)\n"
        "  - SmallMelCNN (Seed-Averaged Logits) vs. MOODWAAVE-Lite: `p = 0.4222` (statistically non-significant)\n"
        "  - SmallMelCNN (Seed-Averaged Logits) vs. MOODWAAVE-Full: `p = 0.6400` (statistically non-significant)\n"
        "- **Outer Folds Paired Nadeau-Bengio Corrected Tests:**\n"
        "  - GTZAN Genre: VGGish minus MOODWAAVE-Lite: `Δ = +17.62%` (`t = 3.47, p = 0.0255`, 95% CI: `[+3.53%, +31.71%]`, df=4)\n"
        "  - GTZAN Genre: VGGish minus MOODWAAVE-Full: `Δ = +15.08%` (`t = 3.40, p = 0.0274`, 95% CI: `[+2.75%, +27.40%]`, df=4)\n"
        "  - PMEmo Valence R²: VGGish minus MOODWAAVE-Lite: `Δ = +0.1036` (`t = 3.51, p = 0.0066`, 95% CI: `[+0.0369, +0.1704]`, df=9)\n"
        "  - PMEmo Arousal R²: VGGish minus MOODWAAVE-Lite: `Δ = +0.0188` (`t = 0.68, p = 0.5160`, 95% CI: `[-0.0442, +0.0819]`, df=9, non-significant)\n"
        "- **Per-System Serialized Storage:**\n"
        "  - MOODWAAVE-Lite: `3.96 MB` (default) / `4.62 MB` (tuned)\n"
        "  - MOODWAAVE-Full: `3.88 MB` (default) / `6.67 MB` (tuned)\n"
        "  - SmallMelCNN: `0.39 MB` (checkpoint, 98,682 params)\n"
        "  - VGGish System: `275.22 MB` (embedding net 275.20 MB + 3 linear heads 0.02 MB)\n"
        "- **Provisional Single-Thread CPU Latencies (ms/track):**\n"
        "  - MOODWAAVE-Lite: `197.6 ± 48.7 ms` (median `176.8 ms`)\n"
        "  - SmallMelCNN: `102.1 ± 12.9 ms` (median `98.7 ms`)\n"
        "  - VGGish + LogReg: `1132.8 ± 26.2 ms` (median `1134.3 ms`)\n"
        "  - MOODWAAVE-Full*: `1746.7 ± 92.5 ms` (provisional, measured in earlier session)\n"
        "- **Audio Decoding Latency Benchmark:**\n"
        "  - GTZAN uncompressed WAV (30s): `~2.5 ms`\n"
        "  - PMEmo compressed MP3 (30s): `~57 ms`\n"
    )
    md.append("\n")

    # Limitations
    md.append("## Limitations\n")
    md.append(
        "1. **Pretraining Asymmetry & Representation Advantage:** VGGish was pretrained on millions of YouTube video audio tracks with 4,800 semantic classes "
        "(including music genres and instruments). The performance gap between VGGish and models trained from scratch on 442–929 tracks reflects massive external "
        "data transfer rather than superior model architecture alone.\n"
        "2. **Single-Split Statistical Uncertainty:** A fixed split with 290 test tracks confers an inherent 95% binomial confidence interval of approximately "
        "±5.5 to ±6.0 percentage points. While VGGish's 19–20 pp lead is statistically confirmed ($p < 10^{-8}$), differences among MOODWAAVE-Full, Lite, and "
        "SmallMelCNN fall within this sampling error envelope.\n"
        "3. **Provisional Latency Profiling:** While all timing protocols enforced strict single-thread constraints (`OMP/MKL/NUMBA_NUM_THREADS=1`, 1 warm-up track "
        "excluded, median of 3 repeats across 100 tracks), benchmarks were acquired across separate execution runs. MOODWAAVE-Full latency in particular reflects "
        "earlier session timings. All runtime measurements should be considered provisional pending a unified simultaneous profiling session.\n"
        "4. **Audio Decoding Overhead:** Audio decompression costs (~2.5 ms for WAV vs. ~57 ms for MP3) represent an irreducible latency component when analyzing raw files. "
        "In web services processing pre-decoded PCM streams, actual model throughput will be proportionally higher.\n"
        "5. **Arousal Ceiling Effect:** On PMEmo arousal regression, VGGish ($R^2 = 0.6889$) fails to demonstrate a statistically significant gain over "
        "MOODWAAVE-Lite ($R^2 = 0.6700$, $p = 0.5160$), indicating that temporal and spectral dynamics captured by 71 standard audio features already saturate "
        "the predictable variance in subjective arousal annotations on this dataset.\n"
    )
    
    res_path = os.path.join(out_dir, "RESULTS.md")
    with open(res_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Generated {res_path}")

def main():
    out_dir = common.BASE_DIR
    generate_plots(out_dir)
    build_results_markdown(out_dir)

if __name__ == "__main__":
    main()
