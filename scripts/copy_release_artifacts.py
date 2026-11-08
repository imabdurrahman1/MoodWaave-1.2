"""Copy final scripts and final tables into release/scripts/ and release/tables/.
Strictly excludes raw audio files (.wav, .mp3, .au).
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

REL_SCRIPTS = os.path.join(ROOT, "release", "scripts")
REL_TABLES = os.path.join(ROOT, "release", "tables")

os.makedirs(REL_SCRIPTS, exist_ok=True)
os.makedirs(REL_TABLES, exist_ok=True)

# -------------------------------------------------------------
# Scripts to copy
# -------------------------------------------------------------
scripts_to_copy = [
    # Feature extraction and inference
    ("src/feature_extractor.py", "feature_extractor.py"),
    ("src/pipeline.py", "pipeline.py"),
    
    # Core consolidation scripts
    ("experiments/final_consolidation/common.py", "common.py"),
    ("experiments/final_consolidation/01_part_a_predictive.py", "01_part_a_predictive.py"),
    ("experiments/final_consolidation/02_part_b_efficiency.py", "02_part_b_efficiency.py"),
    ("experiments/final_consolidation/03_part_c_conformal.py", "03_part_c_conformal.py"),
    ("experiments/final_consolidation/04_part_d_shap.py", "04_part_d_shap.py"),
    ("experiments/final_consolidation/05_compile_final_results.py", "05_compile_final_results.py"),
    
    # Rev2 P1 Splits & Leakage
    ("experiments/rev2_p1_splits/01_build_splits_and_metadata.py", "p1_01_build_splits_and_metadata.py"),
    ("experiments/rev2_p1_splits/02_evaluate_gtzan.py", "p1_02_evaluate_gtzan.py"),
    ("experiments/rev2_p1_splits/03_evaluate_pmemo.py", "p1_03_evaluate_pmemo.py"),
    ("experiments/rev2_p1_splits/04_make_results_and_plots.py", "p1_04_make_results_and_plots.py"),
    
    # Rev2 P2 Tonnetz
    ("experiments/rev2_p2_tonnetz/01_numeric_similarity_all_tracks.py", "p2_01_numeric_similarity_all_tracks.py"),
    ("experiments/rev2_p2_tonnetz/02_retime_single_session_twice.py", "p2_02_retime_single_session_twice.py"),
    ("experiments/rev2_p2_tonnetz/03_extract_v1_features.py", "p2_03_extract_v1_features.py"),
    ("experiments/rev2_p2_tonnetz/04_evaluate_accuracy.py", "p2_04_evaluate_accuracy.py"),
    ("experiments/rev2_p2_tonnetz/05_compile_results_fixed.py", "p2_05_compile_results.py"),
    
    # Rev2 P3 Selection
    ("experiments/rev2_p3_selection/01_run_gtzan_selection.py", "p3_01_run_gtzan_selection.py"),
    ("experiments/rev2_p3_selection/02_run_pmemo_selection.py", "p3_02_run_pmemo_selection.py"),
    ("experiments/rev2_p3_selection/03_analyze_and_report.py", "p3_03_analyze_and_report.py"),
    
    # Rev2 P4 Baselines
    ("experiments/rev2_p4_baselines/01_run_gtzan_baselines.py", "p4_01_run_gtzan_baselines.py"),
    ("experiments/rev2_p4_baselines/02_run_pmemo_baselines.py", "p4_02_run_pmemo_baselines.py"),
    ("experiments/rev2_p4_baselines/03_profile_models.py", "p4_03_profile_models.py"),
    ("experiments/rev2_p4_baselines/04_analyze_and_report.py", "p4_04_analyze_and_report.py"),
    ("experiments/rev2_p4_baselines/05_run_default_lightgbm.py", "p4_05_run_default_lightgbm.py"),
    
    # Rev2 P5 Deep Learning
    ("experiments/rev2_p5_deep/01_extract_vggish_embeddings.py", "p5_01_extract_vggish_embeddings.py"),
    ("experiments/rev2_p5_deep/02_eval_vggish_heads.py", "p5_02_eval_vggish_heads.py"),
    ("experiments/rev2_p5_deep/02b_eval_moodwaave_fixed_split.py", "p5_02b_eval_moodwaave_fixed_split.py"),
    ("experiments/rev2_p5_deep/03_train_small_cnn.py", "p5_03_train_small_cnn.py"),
    ("experiments/rev2_p5_deep/04_profile_efficiency.py", "p5_04_profile_efficiency.py"),
    ("experiments/rev2_p5_deep/05_analyze_and_report.py", "p5_05_analyze_and_report.py"),
    ("experiments/rev2_p5_deep/06_compute_paired_and_mcnemar_tests.py", "p5_06_compute_paired_and_mcnemar_tests.py"),
    ("experiments/rev2_p5_deep/small_cnn.py", "p5_small_cnn.py"),
    
    # Rev2 P6 Conformal
    ("experiments/rev2_p6_conformal/01_run_conformal_benchmark.py", "p6_01_run_conformal_benchmark.py"),
    ("experiments/rev2_p6_conformal/02_analyze_and_report.py", "p6_02_analyze_and_report.py"),
    
    # Rev2 Final Master Timing
    ("experiments/rev2_final_timing/00_setup_models.py", "final_timing_00_setup_models.py"),
    ("experiments/rev2_final_timing/01_run_final_timing_benchmark.py", "final_timing_01_benchmark.py"),
    ("experiments/rev2_final_timing/02_analyze_and_report.py", "final_timing_02_analyze_and_report.py"),
    ("experiments/rev2_final_timing/test_all_pipelines_one_track.py", "final_timing_test_one_track.py")
]

print("Copying reproduction scripts...")
for src_rel, dst_name in scripts_to_copy:
    src_path = os.path.join(ROOT, src_rel)
    dst_path = os.path.join(REL_SCRIPTS, dst_name)
    shutil.copy2(src_path, dst_path)
    print(f"Copied script: {src_rel} -> release/scripts/{dst_name}")

# -------------------------------------------------------------
# Tables to copy
# -------------------------------------------------------------
tables_to_copy = [
    # Feature group and cost definitions
    ("experiments/idea2_cost_pareto/feature_groups.json", "feature_groups.json"),
    ("experiments/idea2_cost_pareto/feature_costs.csv", "feature_costs.csv"),
    
    # Pre-extracted tabular features (code & tabular features only, NEVER raw audio)
    ("data/gtzan_features.csv", "gtzan_features.csv"),
    ("data/pmemo_ready.csv", "pmemo_ready.csv"),
    
    # Consolidation results
    ("experiments/final_consolidation/final_tables.csv", "final_consolidation_tables.csv"),
    ("experiments/final_consolidation/final_tables.json", "final_consolidation_tables.json"),
    ("experiments/final_consolidation/part_a_genre_summary.csv", "part_a_genre_summary.csv"),
    ("experiments/final_consolidation/part_a_genre_per_class_f1.csv", "part_a_genre_per_class_f1.csv"),
    ("experiments/final_consolidation/part_a_mood_summary.csv", "part_a_mood_summary.csv"),
    ("experiments/final_consolidation/part_a_statistical_tests.json", "part_a_statistical_tests.json"),
    ("experiments/final_consolidation/part_b_efficiency_summary.csv", "part_b_efficiency_summary.csv"),
    ("experiments/final_consolidation/part_c_conformal_summary.csv", "part_c_conformal_summary.csv"),
    ("experiments/final_consolidation/part_c_conformal_sweep.csv", "part_c_conformal_sweep.csv"),
    ("experiments/final_consolidation/part_c_zone_occupancy_grid.csv", "part_c_zone_occupancy_grid.csv"),
    ("experiments/final_consolidation/part_d_shap_group_importance.csv", "part_d_shap_group_importance.csv"),
    ("experiments/final_consolidation/part_d_top15_features.csv", "part_d_top15_features.csv"),
    
    # Rev2 P1 Splits
    ("experiments/rev2_p1_splits/part_a_gtzan_summary.csv", "p1_part_a_gtzan_summary.csv"),
    ("experiments/rev2_p1_splits/part_b_pmemo_summary.csv", "p1_part_b_pmemo_summary.csv"),
    ("experiments/rev2_p1_splits/removed_tracks.csv", "p1_gtzan_removed_tracks.csv"),
    ("experiments/rev2_p1_splits/artist_map.csv", "p1_gtzan_artist_map.csv"),
    ("experiments/rev2_p1_splits/pmemo_artist_stats.json", "p1_pmemo_artist_stats.json"),
    
    # Rev2 P2 Tonnetz
    ("experiments/rev2_p2_tonnetz/accuracy_benchmark_summary.csv", "p2_accuracy_benchmark_summary.csv"),
    ("experiments/rev2_p2_tonnetz/end_to_end_summary.csv", "p2_end_to_end_summary.csv"),
    ("experiments/rev2_p2_tonnetz/feature_costs_retimed.csv", "p2_feature_costs_retimed.csv"),
    ("experiments/rev2_p2_tonnetz/timing_runs_comparison.csv", "p2_timing_runs_comparison.csv"),
    ("experiments/rev2_p2_tonnetz/numeric_similarity.csv", "p2_numeric_similarity.csv"),
    
    # Rev2 P3 Selection
    ("experiments/rev2_p3_selection/selection_frequency_genre.csv", "p3_selection_frequency_genre.csv"),
    ("experiments/rev2_p3_selection/selection_frequency_mood.csv", "p3_selection_frequency_mood.csv"),
    ("experiments/rev2_p3_selection/heldout_performance_summary.csv", "p3_heldout_performance_summary.csv"),
    ("experiments/rev2_p3_selection/paired_differences_and_equivalence.csv", "p3_paired_differences_and_equivalence.csv"),
    ("experiments/rev2_p3_selection/speedup_and_cost_summary.csv", "p3_speedup_and_cost_summary.csv"),
    ("experiments/rev2_p3_selection/tonnetz_option_frequency.csv", "p3_tonnetz_option_frequency.csv"),
    
    # Rev2 P4 Baselines
    ("experiments/rev2_p4_baselines/genre_summary.csv", "p4_genre_summary.csv"),
    ("experiments/rev2_p4_baselines/pmemo_summary.csv", "p4_pmemo_summary.csv"),
    ("experiments/rev2_p4_baselines/model_profiles_summary.csv", "p4_model_profiles_summary.csv"),
    ("experiments/rev2_p4_baselines/paired_differences_vs_lightgbm_full.csv", "p4_paired_differences.csv"),
    ("experiments/rev2_p4_baselines/selected_hyperparameters.csv", "p4_selected_hyperparameters.csv"),
    ("experiments/rev2_p4_baselines/default_vs_tuned_lightgbm.csv", "p4_default_vs_tuned_lightgbm.csv"),
    
    # Rev2 P5 Deep Learning
    ("experiments/rev2_p5_deep/fixed_split_comparison.csv", "p5_fixed_split_comparison.csv"),
    ("experiments/rev2_p5_deep/efficiency_profiles.csv", "p5_efficiency_profiles.csv"),
    ("experiments/rev2_p5_deep/system_sizes.csv", "p5_deep_system_sizes.csv"),
    ("experiments/rev2_p5_deep/cnn_seeds_results.csv", "p5_cnn_seeds_results.csv"),
    ("experiments/rev2_p5_deep/p5_fix_paired_tests.csv", "p5_paired_tests.csv"),
    ("experiments/rev2_p5_deep/p5_fix_mcnemar_tests.csv", "p5_mcnemar_tests.csv"),
    
    # Rev2 P6 Conformal
    ("experiments/rev2_p6_conformal/conformal_sweep_results.csv", "p6_conformal_sweep_results.csv"),
    ("experiments/rev2_p6_conformal/grid_comparison_80pct.csv", "p6_grid_comparison_80pct.csv"),
    ("experiments/rev2_p6_conformal/point_models_performance.csv", "p6_point_models_performance.csv"),
    ("experiments/rev2_p6_conformal/constant_k_baseline.csv", "p6_constant_k_baseline.csv"),
    
    # Rev2 Final Master Timing (Supersedes all earlier numbers)
    ("experiments/rev2_final_timing/final_timing_table.csv", "final_timing_table.csv"),
    ("experiments/rev2_final_timing/timing_run1_forward.csv", "final_timing_run1_forward.csv"),
    ("experiments/rev2_final_timing/timing_run2_reverse.csv", "final_timing_run2_reverse.csv"),
    ("experiments/rev2_final_timing/timing_runs_comparison.csv", "final_timing_runs_comparison.csv"),
    ("experiments/rev2_final_timing/system_sizes.csv", "final_system_sizes.csv"),
    ("experiments/rev2_final_timing/environment.json", "benchmarking_environment.json")
]

print("\nCopying reproduction tables...")
for src_rel, dst_name in tables_to_copy:
    src_path = os.path.join(ROOT, src_rel)
    dst_path = os.path.join(REL_TABLES, dst_name)
    shutil.copy2(src_path, dst_path)
    print(f"Copied table: {src_rel} -> release/tables/{dst_name}")

print("\nAll scripts and tables copied successfully!")
