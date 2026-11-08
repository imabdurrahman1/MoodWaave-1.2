import os
import sys
import numpy as np
import pandas as pd

from r3b_common import (
    BASE_DIR,
    PROJECT_ROOT,
    nadeau_bengio_test,
)

def main():
    print("=== Step 4: Paired Statistical Significance Tests (Nadeau-Bengio) ===")
    
    # ---------------------------------------------------------
    # 1. GTZAN
    # ---------------------------------------------------------
    r3b_gtzan_csv = os.path.join(BASE_DIR, "gtzan_r3b_folds.csv")
    r3a_gtzan_csv = os.path.join(PROJECT_ROOT, "experiments", "rev3_a_stats", "gtzan_r3a_folds.csv")
    
    assert os.path.isfile(r3b_gtzan_csv), f"Missing {r3b_gtzan_csv}"
    assert os.path.isfile(r3a_gtzan_csv), f"Missing {r3a_gtzan_csv}"
    
    df_r3b_g = pd.read_csv(r3b_gtzan_csv)
    df_r3a_g = pd.read_csv(r3a_gtzan_csv)
    
    # Filter Phase A reference models (tuned LightGBM)
    lgbm_full_g = df_r3a_g[(df_r3a_g["model"] == "LightGBM") & (df_r3a_g["feature_set"] == "Full-113")].sort_values(["repeat", "fold"])
    lgbm_lite_g = df_r3a_g[(df_r3a_g["model"] == "LightGBM") & (df_r3a_g["feature_set"] == "Lite-71")].sort_values(["repeat", "fold"])
    
    gtzan_tests = []
    embed_models = df_r3b_g["model"].unique()
    
    for em in embed_models:
        df_em = df_r3b_g[df_r3b_g["model"] == em].sort_values(["repeat", "fold"])
        
        for baseline_name, df_base in [("MOODWAAVE-Full-113", lgbm_full_g), ("MOODWAAVE-Lite-71", lgbm_lite_g)]:
            diffs_acc = df_em["accuracy"].values - df_base["accuracy"].values
            diffs_f1 = df_em["macro_f1"].values - df_base["macro_f1"].values
            
            res_acc = nadeau_bengio_test(diffs_acc, test_train_ratio=0.25)
            res_f1 = nadeau_bengio_test(diffs_f1, test_train_ratio=0.25)
            
            gtzan_tests.append({
                "model": em,
                "baseline": baseline_name,
                "metric": "Accuracy",
                "model_mean": float(df_em["accuracy"].mean()),
                "model_sd": float(df_em["accuracy"].std()),
                "baseline_mean": float(df_base["accuracy"].mean()),
                "baseline_sd": float(df_base["accuracy"].std()),
                "mean_diff": res_acc["mean_diff"],
                "ci_lower": res_acc["ci_lower"],
                "ci_upper": res_acc["ci_upper"],
                "t_stat": res_acc["t_stat"],
                "df": res_acc["df"],
                "p_val": res_acc["p_val"],
                "significant": res_acc["p_val"] < 0.05,
            })
            
            gtzan_tests.append({
                "model": em,
                "baseline": baseline_name,
                "metric": "Macro-F1",
                "model_mean": float(df_em["macro_f1"].mean()),
                "model_sd": float(df_em["macro_f1"].std()),
                "baseline_mean": float(df_base["macro_f1"].mean()),
                "baseline_sd": float(df_base["macro_f1"].std()),
                "mean_diff": res_f1["mean_diff"],
                "ci_lower": res_f1["ci_lower"],
                "ci_upper": res_f1["ci_upper"],
                "t_stat": res_f1["t_stat"],
                "df": res_f1["df"],
                "p_val": res_f1["p_val"],
                "significant": res_f1["p_val"] < 0.05,
            })
            
    df_gtzan_tests = pd.DataFrame(gtzan_tests)
    gtzan_tests_out = os.path.join(BASE_DIR, "gtzan_r3b_paired_tests.csv")
    df_gtzan_tests.to_csv(gtzan_tests_out, index=False)
    print(f"GTZAN paired tests saved to {gtzan_tests_out}")
    print(df_gtzan_tests[["model", "baseline", "metric", "mean_diff", "p_val", "significant"]].to_string(index=False))
    
    # ---------------------------------------------------------
    # 2. PMEmo
    # ---------------------------------------------------------
    r3b_pmemo_csv = os.path.join(BASE_DIR, "pmemo_r3b_folds.csv")
    r3a_pmemo_csv = os.path.join(PROJECT_ROOT, "experiments", "rev3_a_stats", "pmemo_r3a_folds.csv")
    
    assert os.path.isfile(r3b_pmemo_csv), f"Missing {r3b_pmemo_csv}"
    assert os.path.isfile(r3a_pmemo_csv), f"Missing {r3a_pmemo_csv}"
    
    df_r3b_p = pd.read_csv(r3b_pmemo_csv)
    df_r3a_p = pd.read_csv(r3a_pmemo_csv)
    
    pmemo_tests = []
    for tgt in ["valence", "arousal"]:
        lgbm_full_p = df_r3a_p[(df_r3a_p["model"] == "LightGBM") & (df_r3a_p["feature_set"] == "Full-113") & (df_r3a_p["task"] == tgt)].sort_values(["repeat", "fold"])
        lgbm_lite_p = df_r3a_p[(df_r3a_p["model"] == "LightGBM") & (df_r3a_p["feature_set"] == "Lite-71") & (df_r3a_p["task"] == tgt)].sort_values(["repeat", "fold"])
        
        for em in embed_models:
            df_em = df_r3b_p[(df_r3b_p["model"] == em) & (df_r3b_p["target"] == tgt)].sort_values(["repeat", "fold"])
            
            for baseline_name, df_base in [("MOODWAAVE-Full-113", lgbm_full_p), ("MOODWAAVE-Lite-71", lgbm_lite_p)]:
                diffs_r2 = df_em["r2"].values - df_base["r2"].values
                diffs_mae = df_em["mae"].values - df_base["mae"].values
                
                res_r2 = nadeau_bengio_test(diffs_r2, test_train_ratio=0.25)
                res_mae = nadeau_bengio_test(diffs_mae, test_train_ratio=0.25)
                
                pmemo_tests.append({
                    "model": em,
                    "target": tgt,
                    "baseline": baseline_name,
                    "metric": "R2",
                    "model_mean": float(df_em["r2"].mean()),
                    "model_sd": float(df_em["r2"].std()),
                    "baseline_mean": float(df_base["r2"].mean()),
                    "baseline_sd": float(df_base["r2"].std()),
                    "mean_diff": res_r2["mean_diff"],
                    "ci_lower": res_r2["ci_lower"],
                    "ci_upper": res_r2["ci_upper"],
                    "t_stat": res_r2["t_stat"],
                    "df": res_r2["df"],
                    "p_val": res_r2["p_val"],
                    "significant": res_r2["p_val"] < 0.05,
                })
                
                pmemo_tests.append({
                    "model": em,
                    "target": tgt,
                    "baseline": baseline_name,
                    "metric": "MAE",
                    "model_mean": float(df_em["mae"].mean()),
                    "model_sd": float(df_em["mae"].std()),
                    "baseline_mean": float(df_base["mae"].mean()),
                    "baseline_sd": float(df_base["mae"].std()),
                    "mean_diff": res_mae["mean_diff"],
                    "ci_lower": res_mae["ci_lower"],
                    "ci_upper": res_mae["ci_upper"],
                    "t_stat": res_mae["t_stat"],
                    "df": res_mae["df"],
                    "p_val": res_mae["p_val"],
                    "significant": res_mae["p_val"] < 0.05,
                })
                
    df_pmemo_tests = pd.DataFrame(pmemo_tests)
    pmemo_tests_out = os.path.join(BASE_DIR, "pmemo_r3b_paired_tests.csv")
    df_pmemo_tests.to_csv(pmemo_tests_out, index=False)
    print(f"\nPMEmo paired tests saved to {pmemo_tests_out}")
    print(df_pmemo_tests[["model", "target", "baseline", "metric", "mean_diff", "p_val", "significant"]].to_string(index=False))

if __name__ == "__main__":
    main()

